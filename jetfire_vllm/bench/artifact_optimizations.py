"""Reproduce the JetFire-only cumulative optimization experiment (paper Fig. 9).

Example, from the repository root::

    python -m jetfire_vllm.bench.artifact_optimizations \
        --model /path/to/Llama-3.1-8B-Instruct --gpu 0 \
        --out-dir results/optimizations --dry-run

This reconstructs the documented historical controls on the supplied runtime.
It does not authenticate the original paper's unavailable build or raw logs.
Every case uses BF16 weights, K3V2/G64/k256, B1, 16 output tokens, and no
activation partitioning. The ``baseline`` case is JetFire with optimizations
disabled; no competing system is imported. Timing uses the existing official
vLLM serving benchmark, with a fresh server and one warmup for each repetition.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODES = ("baseline", "o1", "o12", "o123")
PARTITIONS = (
    "JETFIRE_EMBED_CHUNK_TOKENS", "JETFIRE_MLP_CHUNK_TOKENS",
    "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS", "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS",
    "JETFIRE_PREFILL_QKV_CHUNK_TOKENS",
)
CAVEATS = [
    "Documented historical controls reconstructed on this runtime; the original "
    "paper build and raw logs are not authenticated by this runner.",
    "O1 changes fused quantization/writes, prefill mirroring, and write-worker "
    "count together; it also changes decode KV writes.",
    "Before O2, eager CPU correction finishes before the GPU value phase is "
    "enqueued. O2 submits correction to an executor before enqueuing that phase.",
    "O3 uses legacy kernels and graph-safe async CPU submit plus host join. "
    "It changes dispatch as well as CUDA launch overhead; strict release "
    "kernel policy is deliberately disabled for this labelled experiment.",
    "Official client startup/readiness behavior may differ from historical "
    "vLLM endpoint-client measurements. Preserve versions and raw records.",
    "No per-layer overlap timing is inferred here. Table 9 needs a separate "
    "timeline measurement; a successful request does not prove full overlap.",
]


def environment(mode: str) -> dict[str, str]:
    """Return all experiment controls, independent of inherited JetFire flags."""
    writes = mode != "baseline"
    graph = mode == "o123"
    env = {
        "JETFIRE_PREFILL_PIPELINE": str(int(writes)),
        "JETFIRE_FUSED_QUANT_WRITE": str(int(writes)),
        "JETFIRE_FUSED_QUANT": str(int(writes)),
        "JETFIRE_WRITE_WORKERS": "4" if writes else "1",
        "JETFIRE_SYNC_CPU_ATTN": "1" if mode in ("baseline", "o1") else "0",
        "JETFIRE_DECODE_GRAPH": str(int(graph)),
        "JETFIRE_ENABLE_VLLM_CUDAGRAPH": str(int(graph)),
        "JETFIRE_RELEASE_GRAPH_POLICY": "0",
        "JETFIRE_RELEASE_REQUIRE_READY": "0",
        "JETFIRE_USE_INT8_TC": "0", "JETFIRE_USE_INT8_TC_SV": "0",
        "JETFIRE_WVS_VARIANT": "legacy",
        "JETFIRE_SHADOW": "1", "JETFIRE_SHADOW_DEVICE": "cpu",
        "JETFIRE_SHADOW_DTYPE": "bf16", "JETFIRE_CPU_CORRECTION": "1",
        "JETFIRE_DECODE_CPU_ATTN": "1", "JETFIRE_SKIP_NATIVE_KV": "1",
        "JETFIRE_COMPRESSED_KV_SPEC": "1", "JETFIRE_INFLATE_BLOCKS": "0",
        "JETFIRE_ZERO_INIT_CACHE": "0", "JETFIRE_PROFILE_DECODE": "0",
        "JETFIRE_USE_PERSISTENT_WORKER": "0", "JETFIRE_FUSED_JOIN_BLEND": "0",
        "JETFIRE_CPU_DEBUG": "0", "JETFIRE_STAGE_TRACE": "0",
        "JETFIRE_TRACE_DECODE_STEPS": "0", "JETFIRE_CPU_ATTN_THREADS": "32",
        "OMP_NUM_THREADS": "32", "MKL_NUM_THREADS": "32",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    env.update({key: "0" for key in PARTITIONS})
    if graph:
        env.update({
            "JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK": "1",
            "JETFIRE_GRAPH_SIDE_JOIN_CPU_TOPK": "0",
            "JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK": "0",
            "JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK": "0",
        })
    return env


def harness_args(args: argparse.Namespace, mode: str, context: int,
                 destination: Path) -> list[str]:
    return [
        "--mode", "fixed-concurrency", "--backend", "jetfire",
        "--model", args.model, "--gpu", args.gpu,
        "--ctx", str(context), "--output-len", "16",
        "--concurrency", "1", "--num-prompts", "1",
        "--runs", str(args.runs), "--warmups", "1", "--seed", "42",
        "--client", "official", "--tensor-parallel-size", "1",
        "--jetfire-mode", "SymInt3K_SymInt2V", "--jetfire-group-size", "64",
        "--top-k", "256", "--cpu-attn-threads", "32",
        "--jetfire-profile", "graph" if mode == "o123" else "eager",
        "--jetfire-graph-scope", "decode-only",
        "--jetfire-cudagraph-capture-sizes", "1",
        "--jetfire-release-require-ready", "off", "--jetfire-int8tc", "off",
        "--jetfire-prefill-chunk-tokens", "off",
        "--jetfire-prefill-sub-batch-size", "off", "--chunked-prefill", "off",
        "--max-model-len", str(context + 32),
        "--max-num-batched-tokens", str(context + 32),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--server-timeout-s", str(args.server_timeout_s),
        "--bench-timeout-s", str(args.bench_timeout_s),
        "--out-dir", str(destination),
    ]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "jetfire_artifact_official_harness", HERE / "official_serve_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_case(args: argparse.Namespace) -> None:
    """Adapt only the harness's experiment controls, never production kernels."""
    mode, context = args._case, args.contexts[0]
    target = args.out_dir
    selected = environment(mode)
    # Remove stale experimental settings before importing any runtime modules.
    for key in list(os.environ):
        if key.startswith("JETFIRE_"):
            del os.environ[key]
    os.environ.update(selected)
    harness = _load_harness()
    original_env, original_wait = harness._base_env, harness._wait_ready
    original_bench = harness._bench_cmd
    launch_number = 0

    def controlled_env(parsed, backend, concurrency):
        env = original_env(parsed, backend, concurrency)
        env.update(selected)  # The stock harness otherwise enables release policy.
        env["PYTHONPATH"] = str(HERE / "optimization_hooks") + os.pathsep + env["PYTHONPATH"]
        save(target / "resolved_environment.json", {
            k: v for k, v in env.items()
            if k.startswith(("JETFIRE_", "VLLM_")) or k in
            ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        })
        return env

    def historical_preflight(parsed, specs):
        # This is a narrow opt-in historical experiment, not a release-ready
        # performance result. Keep the production release-policy guard intact.
        if not (parsed.backend == "jetfire" and parsed.jetfire_mode == "SymInt3K_SymInt2V"
                and parsed.jetfire_group_size == 64 and parsed.top_k == 256
                and parsed.tensor_parallel_size == 1
                and all(s.backend == "jetfire" and s.concurrency == 1 for s in specs)):
            raise RuntimeError("Historical Fig. 9 controls require K3V2/G64/k256, TP1, B1")
        print("[artifact] historical optimization controls; release policy disabled", flush=True)

    def wait_and_audit(url, proc, timeout_s):
        nonlocal launch_number
        original_wait(url, proc, timeout_s)
        req = urllib.request.Request(
            url + "/collective_rpc", data=json.dumps({
                "method": "jetfire_artifact_optimization_snapshot"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            snapshot = json.load(response)
        save(target / f"runtime_launch_{launch_number}.json", snapshot)
        launch_number += 1
        workers = snapshot.get("results", [])
        if len(workers) != 1:
            raise RuntimeError(f"Expected one tensor-parallel worker: {snapshot}")
        observed = workers[0]
        expected_graph = "FULL_DECODE_ONLY" if mode == "o123" else "NONE"
        valid = (observed["format"] == [3, 2, 64, 256]
                 and observed["torch_cpu_threads"] == 32
                 and observed["model_shape"] == [32, 32, 8, 128]
                 and observed["cudagraph_mode"].split(".")[-1] == expected_graph
                 and all(v == 0 for v in observed["activation_partitions"].values())
                 and all(observed["environment"].get(k) == v for k, v in selected.items()
                         if k.startswith("JETFIRE_")))
        if not valid:
            raise RuntimeError(f"Runtime does not match Fig. 9 controls; see {target}")

    def fixed_seed_bench(*positional, **keywords):
        command = original_bench(*positional, **keywords)
        command[command.index("--seed") + 1] = "42"
        command.extend(["--temperature", "0", "--extra-body", '{"seed":42}'])
        return command

    harness._base_env = controlled_env
    harness._preflight_release_policy = historical_preflight
    harness._wait_ready = wait_and_audit
    harness._bench_cmd = fixed_seed_bench
    # Do not attach a canonical release kernel plan to this historical run.
    harness._release_plan_row = lambda *_: {
        "jetfire_release_stage_a": "legacy", "jetfire_release_stage_b": "legacy",
        "jetfire_release_ready": False,
        "jetfire_release_promotion_state": "historical_optimization_experiment",
        "jetfire_release_notes": "; ".join(CAVEATS),
    }
    sys.argv = [str(HERE / "official_serve_benchmark.py")] + harness_args(args, mode, context, target)
    harness.main()


def summary(cases: list[dict], runs: int) -> dict:
    rows = []
    for case in cases:
        path = Path(case["out_dir"]) / "summary.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        trials = raw.get("rows", [])
        valid = [r for r in trials if r.get("status") == "ok"
                 and r.get("completed") == 1 and r.get("failed") == 0
                 and r.get("total_output_tokens") == 16]
        row = dict(context=case["context"], optimization=case["optimization"],
                   status="ok" if len(valid) == runs else "incomplete",
                   completed_trials=len(valid), requested_trials=runs,
                   raw_summary=str(path))
        for label, key, scale in (("ttft_s", "batch_ttft_ms", .001),
                                   ("tpot_ms", "mean_tpot_ms", 1.),
                                   ("e2e_s", "batch_e2el_ms", .001)):
            values = [float(r[key]) * scale for r in valid if r.get(key) is not None]
            row[label] = statistics.median(values) if len(values) == runs else None
        rows.append(row)
    return {"paper": "Figure 9", "caveats": CAVEATS, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Llama-3.1-8B-Instruct checkpoint")
    parser.add_argument("--gpu", default="0", help="One CUDA device identifier")
    parser.add_argument("--contexts", type=int, nargs="+", choices=(32768, 65536),
                        default=[32768, 65536])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--bench-timeout-s", type=float, default=1800)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_case", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.runs < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--runs must be positive and GPU memory utilization in (0, 1]")
    if "," in args.gpu:
        parser.error("Fig. 9 requires one GPU")
    if len(args.contexts) != len(set(args.contexts)) or len(args.modes) != len(set(args.modes)):
        parser.error("Duplicate contexts/modes would overwrite experiment records")
    if args._case:
        run_case(args)
        return
    if (args.out_dir / "plan.json").exists():
        raise SystemExit(f"Refusing to overwrite an existing experiment: {args.out_dir}")
    cases = []
    for context in args.contexts:
        for mode in args.modes:
            destination = args.out_dir / f"c{context}" / mode
            command = [sys.executable, str(Path(__file__).resolve()),
                "--_case", mode, "--model", args.model, "--gpu", args.gpu,
                "--contexts", str(context), "--runs", str(args.runs),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--server-timeout-s", str(args.server_timeout_s),
                "--bench-timeout-s", str(args.bench_timeout_s),
                "--out-dir", str(destination)]
            cases.append(dict(context=context, optimization=mode, out_dir=str(destination),
                              command=command, harness_arguments=harness_args(args, mode, context, destination),
                              environment=environment(mode)))
    save(args.out_dir / "plan.json", dict(paper="Figure 9", model=args.model,
        format="K3V2/G64/k256", batch=1, output_tokens=16, decode_intervals=15,
        activation_partitioning=False, warmups_per_server=1,
        server_lifecycle="fresh server per repetition", repetitions=args.runs,
        prompt_seed=42, generation_seed=42, temperature=0,
        caveats=CAVEATS, dry_run=args.dry_run, cases=cases))
    print(f"[artifact] planned {len(cases)} cases: {args.out_dir / 'plan.json'}", flush=True)
    if args.dry_run:
        return
    for case in cases:
        destination = Path(case["out_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("JETFIRE_") and k != "PYTHONPATH"}
        with (destination / "controller.log").open("w") as log:
            completed = subprocess.run(case["command"], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        save(destination / "controller_status.json", {"returncode": completed.returncode})
        result = summary(cases, args.runs)
        save(args.out_dir / "summary.json", result)
        with (args.out_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        print(f"[artifact] {case['optimization']} c{case['context']}: "
              f"exit {completed.returncode}; {destination / 'controller.log'}", flush=True)
    if any(row["status"] != "ok" for row in summary(cases, args.runs)["rows"]):
        raise SystemExit("Some optimization cases failed or are incomplete; inspect summary.json")


if __name__ == "__main__":
    main()
