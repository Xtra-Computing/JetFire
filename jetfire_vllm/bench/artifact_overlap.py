"""Measure the complete eager JetFire correction/value timeline (paper Table 9).

From the source root::

    python -m jetfire_vllm.bench.artifact_overlap \
        --model /path/to/Llama-3.1-8B-Instruct --gpu 0 \
        --out-dir results/overlap --dry-run

Remove --dry-run to collect separate unprofiled and CPU+CUDA-profiled requests.
The same warmed server runs serial and overlapped CPU correction, with legacy
K3V2/G64/k256 kernels, 32 CPU threads, B1, and activation partitions/graphs OFF.
Raw Chrome traces, worker wall ranges, per-layer components, and paired timings
are retained. Profiling overhead and uncovered intervals remain visible.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request

try:
    from .artifact_optimizations import ROOT, HERE, environment, harness_args, _load_harness, save
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jetfire_vllm.bench.artifact_optimizations import (
        ROOT, HERE, environment, harness_args, _load_harness, save)

STAGES = {
    "V2_01_attn": "score", "V2_02_topk_d2h_fork": "transfer",
    "V2_01b_softmax_v": "value", "V2_05_post_corr": "combine",
    "V2_04c_worker_h2d": "return_copy",
}
LIMITATIONS = [
    "Profiled requests are diagnostic and separate from unprofiled latency trials; "
    "profiler, Python labels, and timestamp collection add overhead.",
    "CPU correction ranges are wall-clock calls including production gather and "
    "compute, not instruction-active CPU hardware counters or gather-only timing.",
    "Transfer GPU intervals are actual device copy activity. The enclosing host "
    "D2H scope also waits for queued GPU work; its wall time is reported separately.",
    "Mapped host output has no explicit correction-result H2D copy. Its PCIe "
    "reads occur inside the blend kernel and cannot be isolated from that kernel.",
    "CPU/value overlap is an interval intersection on one measured timeline; "
    "it does not by itself establish an equal reduction in request latency.",
    "Effective saving is the separate serial-minus-overlap TPOT difference, "
    "normalized by model layers. Dispatch and scheduling effects remain included.",
    "Legacy eager kernels reconstruct the documented Fig. 9/Table 9 controls. "
    "This is not an authentication of the original historical build or logs.",
]


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative timeline interval")
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def length(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(a, b):
    return sum(max(0., min(ae, be) - max(aa, bb))
               for aa, ae in merged(a) for bb, be in merged(b))


def write_csv(path, rows):
    if not rows:
        return
    import csv
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_trace(trace_path: Path, *, expected_layers: int, decode_intervals: int):
    """Correlate real CUDA runtime calls to activities inside labelled stages.

    Chrome timestamps are microseconds relative to baseTimeNanoseconds. Worker
    monotonic ranges use a measured wall/monotonic anchor on that same clock.
    Missing correlation data is a failed coverage check, never estimated time.
    """
    trace = json.loads(trace_path.read_text())
    sidecar = json.loads(trace_path.with_suffix(".cpu.json").read_text())
    base = trace.get("baseTimeNanoseconds")
    if not isinstance(base, (int, float)):
        raise ValueError("Chrome trace lacks baseTimeNanoseconds; CPU alignment unavailable")
    clock, end_clock = sidecar["start_clock"], sidecar["end_clock"]
    drift_ns = ((end_clock["wall_ns"] - clock["wall_ns"])
                - (end_clock["perf_ns"] - clock["perf_ns"]))
    clock_bound_us = (abs(drift_ns) + clock["uncertainty_ns"]
                      + end_clock["uncertainty_ns"]) / 1000.
    def aligned(perf):
        return (clock["wall_ns"] - base + perf - clock["perf_ns"]) / 1000.

    runtimes = defaultdict(list)
    activities = defaultdict(list)
    scopes = []
    for index, event in enumerate(trace.get("traceEvents", [])):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = str(event.get("cat", ""))
        if category in ("cuda_runtime", "cuda_driver"):
            runtimes[event.get("tid")].append(event)
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                activities[correlation].append((index, event))
        if str(event.get("name", "")).startswith("JF|stage="):
            scopes.append(event)
    starts = {}
    for tid, calls in runtimes.items():
        calls.sort(key=lambda r: r["ts"])
        starts[tid] = [r["ts"] for r in calls]
    groups = defaultdict(lambda: defaultdict(list))
    for row in sidecar["cpu_wall_ranges"]:
        if row["trace"] < 0:
            continue
        key = (row["layer"], row["trace"])
        groups[key]["host_" + row["stage"]].append(
            (aligned(row["start_perf_ns"]), aligned(row["end_perf_ns"])))
    mapped = set()
    for scope in scopes:
        tags = dict(part.split("=", 1) for part in scope["name"].split("|")[1:])
        stage = STAGES.get(tags["stage"])
        trace_id = int(tags.get("trace", -1))
        if stage is None or trace_id < 0:
            continue
        key = (tags["layer"], trace_id)
        start, end = scope["ts"], scope["ts"] + scope["dur"]
        calls = runtimes.get(scope.get("tid"), [])
        ticks = starts.get(scope.get("tid"), [])
        for call in calls[bisect.bisect_left(ticks, start):bisect.bisect_right(ticks, end)]:
            if call["ts"] + call["dur"] > end + .01:
                continue
            for index, activity in activities.get(call.get("args", {}).get("correlation"), []):
                is_copy = activity["cat"] == "gpu_memcpy"
                if (stage in ("transfer", "return_copy") and not is_copy) or (
                        stage not in ("transfer", "return_copy") and activity["cat"] != "kernel"):
                    continue
                marker = (key, stage, index)
                if marker in mapped:
                    continue
                mapped.add(marker)
                groups[key][stage].append((activity["ts"], activity["ts"] + activity["dur"]))

    rows = []
    incomplete = []
    for (layer, trace_id), stages in sorted(groups.items(), key=lambda p: p[0][1]):
        cpu = stages["host_V2_04b_worker_cpu_attn"]
        gpu = sum((stages[stage] for stage in ("score", "transfer", "value", "combine", "return_copy")), [])
        required = ("score", "transfer", "value", "combine", "host_V2_04b_worker_cpu_attn")
        missing = [stage for stage in required if not stages[stage]]
        if missing:
            incomplete.append(dict(layer=layer, trace=trace_id, missing=missing))
        overlap = intersection(cpu, stages["value"])
        # Shifting one merged CPU interval by <=u changes its intersection
        # with any GPU interval union by at most 2u. This lower bound holds
        # across every such shift, including offsets between the endpoints.
        overlap_low = max(0., overlap - 2 * clock_bound_us * len(merged(cpu)))
        envelope = ((max(e for s, e in gpu) - min(s for s, e in gpu)) if gpu else None)
        occupied = length(gpu + cpu)
        rows.append(dict(layer=layer, trace=trace_id, coverage_complete=not missing,
            score_gpu_us=length(stages["score"]), transfer_gpu_us=length(stages["transfer"]),
            cpu_correction_wall_us=length(cpu), value_gpu_us=length(stages["value"]),
            combine_gpu_us=length(stages["combine"]), return_copy_gpu_us=length(stages["return_copy"]),
            d2h_host_scope_us=length(stages["host_V2_02_topk_d2h_fork"]),
            submit_host_scope_us=length(stages["host_V2_02c_tpe_submit"]),
            join_host_wait_us=length(stages["host_V2_04d_main_wait_worker"]),
            cpu_commsync_wall_us=length(stages["host_V2_04a_worker_commsync"]),
            cpu_value_overlap_us=overlap,
            cpu_value_overlap_clock_shift_lower_bound_us=overlap_low,
            attention_gpu_envelope_us=envelope,
            occupied_cpu_gpu_union_us=occupied,
            measured_unassigned_gap_us=(envelope - occupied if envelope is not None
                and all(s >= min(x for x, y in gpu) and e <= max(y for x, y in gpu)
                        for s, e in cpu) else None)))
    expected = expected_layers * decode_intervals
    complete = len(rows) == expected and not incomplete
    numeric = [key for key in rows[0] if key.endswith("_us")] if rows else []
    means = {key: statistics.mean(row[key] for row in rows)
             for key in numeric if all(row[key] is not None for row in rows)}
    result = dict(status="complete" if complete else "incomplete_trace_coverage",
        mode=sidecar["mode"], expected_layer_steps=expected, observed_layer_steps=len(rows),
        missing=incomplete, mean_per_layer=means,
        correction_cpu_ms_per_token=(sum(row["cpu_correction_wall_us"] for row in rows)
                                     / decode_intervals / 1000.),
        clock_alignment=dict(base_time_nanoseconds=base, end_to_start_drift_ns=drift_ns,
                             conservative_clock_offset_us=clock_bound_us),
        gpu_mapping="CUDA runtime/driver correlation IDs inside labelled host launch scopes",
        return_path=("configured mapped host output; no separate H2D copy observed"
            if complete and sidecar.get("runtime", {}).get("environment", {}).get("JETFIRE_MAPPED_O_NEW", "1") == "1"
                and all(row["return_copy_gpu_us"] == 0 for row in rows)
            else "explicit H2D copies observed" if any(row["return_copy_gpu_us"] > 0 for row in rows)
            else "return-copy coverage unavailable"),
        limitations=LIMITATIONS)
    write_csv(trace_path.with_suffix(".layers.csv"), rows)
    save(trace_path.with_suffix(".analysis.json"), result)
    return result


def rpc(url, **kwargs):
    request = urllib.request.Request(url + "/collective_rpc", data=json.dumps({
        "method": "jetfire_artifact_overlap_control", "kwargs": kwargs}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    workers = result.get("results", [])
    if len(workers) != 1:
        raise RuntimeError(f"Expected one profiling worker: {result}")
    return workers[0]


async def request_once(url, sample, model_name, destination):
    """Use the pinned vLLM official streaming request function, without CLI probes."""
    import aiohttp
    from vllm.benchmarks.lib import endpoint_request_func as endpoint
    usage = []
    class Observer:
        def __getattr__(self, name):
            return getattr(json, name)
        def loads(self, data, *args, **kwargs):
            value = json.loads(data, *args, **kwargs)
            if isinstance(value, dict) and value.get("usage"):
                usage.append(value["usage"])
            return value
    original = endpoint.json
    endpoint.json = Observer()
    try:
        request = endpoint.RequestFuncInput(prompt=sample.prompt,
            api_url=url + "/v1/completions", prompt_len=sample.prompt_len,
            output_len=16, model=model_name, ignore_eos=True,
            extra_body={"temperature": 0, "seed": 42})
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            output = await endpoint.async_request_openai_completions(request, session)
    finally:
        endpoint.json = original
    raw = asdict(output)
    raw["server_usage"] = usage
    save(destination, raw)
    if not output.success or output.output_tokens != 16 or not usage or usage[-1]["completion_tokens"] != 16:
        raise RuntimeError(f"Incomplete profiling request; see {destination}")
    return dict(ttft_s=output.ttft, tpot_ms=sum(output.itl) / 15 * 1000,
                e2e_s=output.latency, ttft_plus_itl_s=output.ttft + sum(output.itl),
                output_sha256=hashlib.sha256(output.generated_text.encode()).hexdigest(),
                server_usage=usage[-1], raw_result=str(destination))


def run_context(options, context, destination):
    harness = _load_harness()
    original_env = harness._base_env
    controls = environment("o1")
    controls["JETFIRE_ARTIFACT_TRACE_ROOT"] = str(destination / "traces")
    for key in list(os.environ):
        if key.startswith("JETFIRE_"):
            del os.environ[key]
    os.environ.update(controls)

    def base_env(parsed, backend, concurrency):
        env = original_env(parsed, backend, concurrency)
        env.update(controls)
        env["PYTHONPATH"] = str(HERE / "overlap_hooks") + os.pathsep + env["PYTHONPATH"]
        return env
    harness._base_env = base_env

    def execute(parsed, spec):
        env = base_env(parsed, "jetfire", 1)
        port = harness._free_port("127.0.0.1")
        url = f"http://127.0.0.1:{port}"
        command = harness._server_cmd(parsed, "jetfire", 1, port)
        save(destination / "server_command.json", dict(command=command, environment={
            k: v for k, v in env.items() if k.startswith(("JETFIRE_", "VLLM_"))
            or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
        from transformers import AutoTokenizer
        from vllm.benchmarks.datasets import RandomDataset
        tokenizer = AutoTokenizer.from_pretrained(options.model)
        sample = RandomDataset(random_seed=42).sample(tokenizer, num_requests=1,
            input_len=context, output_len=16, range_ratio=0)[0]
        save(destination / "prompt.json", dict(prompt=sample.prompt,
            client_prompt_len=sample.prompt_len, nominal_context=context,
            sha256=hashlib.sha256(sample.prompt.encode()).hexdigest(), seed=42))
        trials, profiles = [], []
        with (destination / "server.log").open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                harness._wait_ready(url, proc, options.server_timeout_s)
                observed = rpc(url)
                save(destination / "runtime_before.json", observed)
                if (observed["format"] != [3, 2, 64, 256]
                        or observed["model_shape"] != [32, 32, 8, 128] or observed["torch_cpu_threads"] != 32
                        or observed["cudagraph_mode"].split(".")[-1] != "NONE"
                        or any(observed["activation_partitions"].values())):
                    raise RuntimeError("Table 9 profiling requires Llama8B/K3V2/G64/k256/32 threads, eager, partitions off")
                for mode in ("serial", "overlap"):
                    rpc(url, action="set_mode", mode=mode)
                    row = asyncio.run(request_once(url, sample, parsed.served_model_name,
                        destination / f"warmup_{mode}.json"))
                    if row["server_usage"]["prompt_tokens"] != context:
                        raise RuntimeError(f"Actual prompt length differs from requested {context}: {row}")
                for pair in range(options.runs):
                    order = ("serial", "overlap") if pair % 2 == 0 else ("overlap", "serial")
                    for mode in order:
                        rpc(url, action="set_mode", mode=mode)
                        row = asyncio.run(request_once(url, sample, parsed.served_model_name,
                            destination / f"unprofiled_pair{pair}_{mode}.json"))
                        trials.append(dict(pair=pair, mode=mode, **row))
                        save(destination / "unprofiled_trials.json", trials)
                for repeat in range(options.profile_runs):
                    for mode in ("serial", "overlap"):
                        rpc(url, action="set_mode", mode=mode)
                        name = f"{mode}_{repeat}.json"
                        rpc(url, action="start", trace_name=name)
                        try:
                            row = asyncio.run(request_once(url, sample, parsed.served_model_name,
                                destination / f"profiled_{mode}_{repeat}.json"))
                        finally:
                            snapshot = rpc(url, action="stop")
                            save(destination / f"profiled_{mode}_{repeat}.runtime.json", snapshot)
                        analysis = analyze_trace(Path(snapshot["trace_path"]), expected_layers=32, decode_intervals=15)
                        profiles.append(dict(mode=mode, repetition=repeat, request=row,
                                             trace=snapshot["trace_path"], analysis=analysis))
                        save(destination / "profiles.json", profiles)
                save(destination / "runtime_after.json", rpc(url))
            finally:
                harness._terminate(proc)
        pairs = []
        for pair in range(options.runs):
            a = next(row for row in trials if row["pair"] == pair and row["mode"] == "serial")
            b = next(row for row in trials if row["pair"] == pair and row["mode"] == "overlap")
            pairs.append(dict(pair=pair, tpot_saving_ms=a["tpot_ms"] - b["tpot_ms"],
                effective_saving_us_per_layer=(a["tpot_ms"] - b["tpot_ms"]) * 1000 / 32,
                same_output=a["output_sha256"] == b["output_sha256"]))
        result = dict(context=context,
            status="complete" if all(p["analysis"]["status"] == "complete" for p in profiles)
                and all(p["same_output"] for p in pairs) else "incomplete_or_different_outputs",
            unprofiled_trials=trials, pairs=pairs, profiles=profiles,
            actual_prompt_tokens=sorted({row["server_usage"]["prompt_tokens"] for row in trials}),
            median_effective_saving_us_per_layer=statistics.median(p["effective_saving_us_per_layer"] for p in pairs),
            limitations=LIMITATIONS)
        save(destination / "result.json", result)
        # The harness supplies parsing/launch policy and writes its own launch
        # summary; detailed measured results remain in the file above.
        return dict(status="ok" if result["status"] == "complete" else "profile_incomplete",
                    backend="jetfire", max_concurrency=1, num_prompts=1,
                    completed=1 if result["status"] == "complete" else 0,
                    failed=0 if result["status"] == "complete" else 1,
                    total_output_tokens=16 if result["status"] == "complete" else 0,
                    output_len=16)

    harness._run_one = execute
    launch_options = argparse.Namespace(**vars(options))
    launch_options.runs = 1
    argv = harness_args(launch_options, "o1", context, destination / "harness")
    sys.argv = [str(HERE / "official_serve_benchmark.py")] + argv
    harness.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536],
                        help="Table 9 uses 32768/65536; smaller values are smoke tests")
    parser.add_argument("--runs", type=int, default=3, help="Unprofiled serial/overlap pairs")
    parser.add_argument("--profile-runs", type=int, default=1, help="Separate captures per mode")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--bench-timeout-s", type=float, default=1800)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-trace", type=Path, help="Reanalyze an existing trace and CPU sidecar; no GPU")
    args = parser.parse_args()
    if args.analyze_trace:
        result = analyze_trace(args.analyze_trace, expected_layers=32, decode_intervals=15)
        print(json.dumps(result, indent=2))
        return
    if (min(args.contexts) < 256 or args.runs < 1 or args.profile_runs < 1
            or len(set(args.contexts)) != len(args.contexts) or "," in args.gpu):
        parser.error("Use unique contexts>=256, positive repetitions, and one GPU")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0,1]")
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.out_dir}")
    save(args.out_dir / "plan.json", dict(paper="Table 9", model=args.model, contexts=args.contexts,
        format="K3V2/G64/k256", cpu_threads=32, batch=1, output_tokens=16,
        graphs=False, activation_partitioning=False,
        unprofiled_pairs=args.runs, profiled_repetitions_per_mode=args.profile_runs,
        profiler="torch.profiler CPU+CUDA; record_shapes/with_stack/profile_memory disabled",
        synchronization="Existing production waits retained; no added per-layer synchronization",
        environment=environment("o1"), only_between_request_control="JETFIRE_SYNC_CPU_ATTN=1/0",
        temperature=0, seed=42, dry_run=args.dry_run,
        limitations=LIMITATIONS))
    print(f"[artifact-overlap] plan: {args.out_dir / 'plan.json'}", flush=True)
    if args.dry_run:
        return
    for context in args.contexts:
        destination = args.out_dir / f"c{context}"
        destination.mkdir()
        run_context(args, context, destination)
    results = [json.loads((args.out_dir / f"c{c}" / "result.json").read_text()) for c in args.contexts]
    save(args.out_dir / "summary.json", dict(results=results, limitations=LIMITATIONS))
    if any(result["status"] != "complete" for result in results):
        raise SystemExit("Incomplete trace coverage or differing paired outputs; inspect summary.json")


if __name__ == "__main__":
    main()
