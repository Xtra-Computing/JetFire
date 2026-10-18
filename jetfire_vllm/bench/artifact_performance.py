"""Reproducible JetFire-only end-to-end performance and generation campaign.

Example (checkpoint locations may instead use the named JETFIRE_* variables)::

    python -m jetfire_vllm.bench.artifact_performance \
      --model-path llama31_8b=/models/Llama-3.1-8B-Instruct \
      --model-path qwen3_30b_a3b=/models/Qwen3-30B-A3B-Instruct-2507 \
      --model-path llama33_70b=/models/Llama-3.3-70B-Instruct \
      --single-gpus 1 --tp-gpus 0,1 --out-dir results/performance

The default is 24 cells: three models, two quantization modes, input lengths
4096/32768, batches 1/4, and 128 generated tokens. Each cell runs three trials,
each in a fresh server with one untimed arithmetic check and one warmup. All
serving measurements come from the existing official vLLM benchmark harness.
Resume skips only complete, matching evidence; every failed attempt is kept.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import string
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jetfire_vllm.artifact import MODEL_ENV, _save, _source_hashes

ALIASES = {"qwen3_30b": "qwen3_30b_a3b"}
MODES = {"k3v2": ("SymInt3K_SymInt2V", 64), "k4v4": ("SymInt4K_SymInt4V", 128)}
MODEL_SHAPES = {
    "llama31_8b": ("llama", 4096, 32),
    "qwen3_30b_a3b": ("qwen3_moe", 2048, 48),
    "llama33_70b": ("llama", 8192, 80),
}
METRICS = ("batch_e2el_ms", "batch_ttft_ms", "median_tpot_ms", "output_throughput",
           "total_token_throughput", "request_throughput", "gpu_peak_memory_gib")
INPUT_TOKEN_TOLERANCE_PER_REQUEST = 1
SEMANTIC_PROMPT = "What is 2 + 2? Answer with only the number."


@dataclass(frozen=True)
class Case:
    model: str
    model_path: str
    tp: int
    gpus: str
    mode: str
    ctx: int
    batch: int
    output_len: int
    runs: int

    @property
    def key(self) -> str:
        return f"{self.model}/{self.mode}/b{self.batch}_ctx{self.ctx}_out{self.output_len}"


def _csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected a nonempty list without duplicates: {raw!r}")
    return values


def _positive_csv(raw: str) -> list[int]:
    values = [int(value) for value in _csv(raw)]
    if min(values) <= 0:
        raise ValueError("contexts and batches must be positive")
    if len(values) != len(set(values)):
        raise ValueError("contexts and batches must not repeat the same numeric value")
    return values


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def _model_input(path: str) -> dict[str, Any]:
    directory = Path(path)
    config = _load(directory / "config.json")
    files = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                 "special_tokens_map.json", "model.safetensors.index.json"):
        candidate = directory / name
        if candidate.is_file():
            files[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    # Weight shards are inventoried, not hashed: reading hundreds of GB would
    # unnecessarily warm the filesystem cache before model-load measurements.
    shards = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in sorted(directory.glob("*.safetensors"))}
    return {"path": str(directory.resolve()), "config": config, "input_sha256": files,
            "weight_shards": shards, "weights_hashed": False}


def _cases(args: argparse.Namespace) -> tuple[list[Case], dict[str, Any]]:
    models = [ALIASES.get(key, key) for key in _csv(args.models)]
    if len(set(models)) != len(models):
        raise ValueError("duplicate model aliases")
    unknown = set(models) - set(MODEL_ENV)
    modes = _csv(args.modes)
    if unknown or set(modes) - set(MODES):
        raise ValueError(f"unknown models/modes: {sorted(unknown | (set(modes) - set(MODES)))}")
    contexts, batches = _positive_csv(args.contexts), _positive_csv(args.batches)
    overrides = {}
    for item in args.model_path:
        key, separator, value = item.partition("=")
        key = ALIASES.get(key, key)
        if not separator or key not in MODEL_ENV or not value or key in overrides:
            raise ValueError("--model-path requires a unique known MODEL=PATH")
        overrides[key] = value
    result, inputs = [], {}
    for key in models:
        path = overrides.get(key) or os.environ.get(MODEL_ENV[key], "")
        if not path and args.dry_run:
            path = "/path/to/" + key
        if not path:
            raise ValueError(f"set --model-path {key}=PATH or {MODEL_ENV[key]}")
        path = str(Path(path).expanduser().resolve())
        tp = 2 if key == "llama33_70b" else 1
        gpus = args.tp_gpus if tp > 1 else args.single_gpus
        gpu_ids = _csv(gpus)
        if len(gpu_ids) != tp or any(not part.isdigit() for part in gpu_ids):
            raise ValueError(f"{key} needs exactly {tp} distinct numeric GPU IDs; received {gpus!r}")
        if args.dry_run and path.startswith("/path/to/"):
            inputs[key] = {"path": path, "status": "checkpoint_not_configured"}
        else:
            inputs[key] = _model_input(path)
            config = inputs[key]["config"]
            actual = (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers"))
            if actual != MODEL_SHAPES[key]:
                raise ValueError(f"checkpoint shape does not match {key}: {actual}")
            if config.get("num_key_value_heads", 0) % tp:
                raise ValueError(f"KV heads must divide tensor parallel size for {key}")
            limit = int(config["max_position_embeddings"])
            if max(contexts) + args.output_len > limit:
                raise ValueError(f"context + output tokens exceeds {key} model limit {limit}")
        for mode in modes:
            for ctx in contexts:
                for batch in batches:
                    result.append(Case(key, path, tp, gpus, mode, ctx, batch, args.output_len, args.runs))
    return result, inputs


def _environment(args: argparse.Namespace) -> dict[str, str]:
    # Avoid inherited development/runtime switches changing the evaluated path.
    env = {key: value for key, value in os.environ.items() if not key.startswith("JETFIRE_")}
    env.update(PYTHONPATH=str(ROOT), PYTHON_BIN=sys.executable, VLLM_DO_NOT_TRACK="1",
               VLLM_NO_USAGE_STATS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS=str(args.cpu_threads), JETFIRE_CPU_ATTN_THREADS=str(args.cpu_threads))
    env["TORCH_EXTENSIONS_DIR"] = str(Path(args.torch_extensions_dir).expanduser().resolve())
    return env


def _command(args: argparse.Namespace, case: Case, attempt: Path) -> list[str]:
    mode, group = MODES[case.mode]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--harness-worker",
           str(attempt / "semantic_probes.jsonl") if args.semantic_probe else "off", "--",
           "--mode", "fixed-concurrency", "--backend", "jetfire", "--model", case.model_path,
           "--gpu", case.gpus, "--tensor-parallel-size", str(case.tp), "--ctx", str(case.ctx),
           "--output-len", str(case.output_len), "--concurrency", str(case.batch),
           "--num-prompts", str(case.batch), "--runs", str(case.runs), "--warmups", str(args.warmups),
           "--seed", str(args.seed), "--jetfire-profile", args.profile,
           "--jetfire-graph-scope", "decode-only", "--jetfire-mode", mode,
           "--jetfire-group-size", str(group), "--top-k", str(args.top_k),
           "--jetfire-int8tc", "auto", "--chunked-prefill", "off",
           "--jetfire-kv-cache-budget", "auto", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--cpu-attn-threads", str(args.cpu_threads), "--jetfire-prefill-chunk-tokens", args.prefill_chunk,
           "--torch-extensions-dir", args.torch_extensions_dir,
           "--server-timeout-s", str(args.server_timeout_s), "--bench-timeout-s", str(args.bench_timeout_s),
           "--memory-sample-interval-s", "0.25", "--client", "official", "--out-dir", str(attempt)]
    return cmd


def _harness_worker(argv: list[str]) -> int:
    """Attach an untimed correctness probe to the existing harness, without forks."""
    from jetfire_vllm.bench import official_serve_benchmark as harness
    probe_path, separator, *harness_args = argv
    if separator != "--":
        raise ValueError("internal worker requires -- before official harness arguments")
    if probe_path != "off":
        original = harness._wait_ready

        def checked_ready(base_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
            original(base_url, proc, timeout_s)
            record: dict[str, Any] = {"prompt": SEMANTIC_PROMPT, "started_utc": _utc()}
            try:
                payload = {"model": "jetfire-bench-model", "messages": [{"role": "user", "content": SEMANTIC_PROMPT}],
                           "temperature": 0, "max_tokens": 32, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                request = urllib.request.Request(base_url + "/v1/chat/completions",
                                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=min(timeout_s, 120)) as response:
                    result = json.load(response)
                text = result["choices"][0]["message"]["content"]
                record.update(text=text, usage=result.get("usage"), status="passed" if isinstance(text, str) and text.strip(string.whitespace + string.punctuation) == "4" else "failed")
                if record["status"] != "passed":
                    raise RuntimeError(f"arithmetic generation probe expected only the answer 4: {text!r}")
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                with Path(probe_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        harness._wait_ready = checked_ready
    sys.argv = ["official_serve_benchmark", *harness_args]
    harness.main()
    return 0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate(case: Case, attempt: Path, returncode: int, semantic_probe: bool) -> dict[str, Any]:
    errors, rows, probes = [], [], []
    try:
        summary = _load(attempt / "summary.json")
        rows = summary["rows"]
        if not isinstance(rows, list):
            raise ValueError("rows is not a list")
        if summary.get("status") != "passed":
            errors.append(f"official harness status: {summary.get('status', 'missing')}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rows = []
        errors.append(f"cannot read official summary: {exc}")
    if returncode:
        errors.append(f"official harness exited {returncode}")
    if len(rows) != case.runs:
        errors.append(f"measured runs {len(rows)}/{case.runs}")
    valid = []
    input_observations = []
    run_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"run {index}: invalid record {row!r}")
            continue
        reasons = []
        expected = {"status": "ok", "num_prompts": case.batch, "completed": case.batch, "failed": 0,
                    "total_output_tokens": case.batch * case.output_len,
                    "ctx": case.ctx, "output_len": case.output_len, "max_concurrency": case.batch,
                    "jetfire_mode": MODES[case.mode][0], "jetfire_group_size": MODES[case.mode][1]}
        for key, value in expected.items():
            if row.get(key) != value:
                reasons.append(f"{key}={row.get(key)!r}, expected {value!r}")
        try:
            reported_input = int(row["total_input_tokens"])
            delta = reported_input - case.batch * case.ctx
            input_observations.append({"run": row.get("run"), "nominal_total_input_tokens": case.batch * case.ctx,
                                       "client_reported_total_input_tokens": reported_input, "delta_from_nominal": delta,
                                       "client_reported_input_tokens_per_request_mean": reported_input / case.batch})
            if reported_input < 0 or abs(delta) > case.batch * INPUT_TOKEN_TOLERANCE_PER_REQUEST:
                reasons.append(f"reported input tokens {reported_input} differ from nominal {case.batch * case.ctx} by more than {INPUT_TOKEN_TOLERANCE_PER_REQUEST} token/request")
        except (KeyError, ValueError, TypeError):
            reasons.append("missing/invalid total_input_tokens")
        for metric in METRICS[:-1]:
            # One generated token has no inter-token interval. The official
            # client may report TPOT as either zero or unavailable in this case.
            single_token_tpot = metric == "median_tpot_ms" and case.output_len == 1
            if single_token_tpot and row.get(metric) is None:
                continue
            try:
                value = float(row[metric])
                if not math.isfinite(value) or value < 0 or (value == 0 and not single_token_tpot):
                    raise ValueError()
            except (KeyError, ValueError, TypeError):
                reasons.append(f"missing/nonpositive metric {metric}")
        run_ids.append(row.get("run"))
        if reasons:
            errors.append(f"run {index}: " + "; ".join(reasons) + (f"; {row['error']}" if row.get("error") else ""))
        else:
            valid.append(row)
    if run_ids != list(range(case.runs)):
        errors.append(f"unexpected trial IDs: {run_ids!r}")
    if semantic_probe:
        try:
            probes = [json.loads(line) for line in (attempt / "semantic_probes.jsonl").read_text().splitlines() if line]
            if len(probes) != case.runs or any(probe.get("status") != "passed" for probe in probes):
                errors.append(f"semantic probes passed {sum(p.get('status') == 'passed' for p in probes)}/{case.runs}")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read semantic probes: {exc}")
    out: dict[str, Any] = {**asdict(case), "status": "failed" if errors else "passed",
                          "runs_complete": len(valid), "semantic_probes_passed": sum(p.get("status") == "passed" for p in probes),
                          "attempt_dir": str(attempt), "returncode": returncode, "errors": errors, "wall_seconds": None,
                          "input_token_tolerance_per_request": INPUT_TOKEN_TOLERANCE_PER_REQUEST,
                          "input_token_observations": input_observations,
                          "input_token_notes": ["vLLM random-dataset input accounting excludes tokenizer-added special tokens; decode/re-encode can also change length. Reported client token counts differ from nominal; see input_token_observations."] if any(item["delta_from_nominal"] for item in input_observations) else []}
    # Aggregate complete campaigns only; partial measurements remain in raw rows.
    for metric in METRICS:
        values = [float(row[metric]) for row in valid if row.get(metric) not in (None, "")]
        out[metric + "_median"] = statistics.median(values) if not errors and values else None
    capacity = [float(row["gpu_kv_cache_tokens"]) / (case.ctx + case.output_len)
                for row in valid if row.get("gpu_kv_cache_tokens") not in (None, "")]
    out["resident_capacity_min"] = min(capacity) if capacity else None
    out["all_requests_resident"] = min(capacity) >= case.batch if capacity else None
    out["release_notes"] = sorted({row.get("jetfire_release_notes", "") for row in rows if isinstance(row, dict)} - {""})
    return out


def _write_summary(out_dir: Path, rows: list[dict[str, Any]], planned: int, expected_runs: int, interrupted: bool = False) -> dict[str, Any]:
    passed = sum(row["status"] == "passed" for row in rows)
    summary = {"status": "interrupted" if interrupted else "passed" if passed == planned else "failed",
               "updated_utc": _utc(), "cases_expected": planned, "cases_finished": len(rows), "cases_passed": passed,
               "measured_runs_expected": expected_runs,
               "measured_runs_complete": sum(row["runs_complete"] for row in rows), "rows": rows}
    _save(out_dir / "performance_summary.json", summary)
    if rows:
        with (out_dir / "performance_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    lines = ["# JetFire performance campaign", "", f"Status: **{summary['status']}**. Passed cells: {passed}/{planned}.", "",
             "Each measured trial uses a fresh server, an untimed generation check (if enabled), and configured warmup requests.",
             "Reported latency and throughput are medians across complete trials. TTFT/E2E use the slowest request in each batch; TPOT is per-request median.",
             "Input lengths in the table are nominal token counts. Exact client-reported totals and their differences from nominal are retained per trial in JSON/CSV.",
             "The official random client excludes tokenizer-added special tokens from input accounting; up to one token/request difference is accepted and recorded explicitly.",
             "Output tok/s excludes input tokens; total tok/s includes client-reported input tokens.",
             "Resident capacity is KV tokens divided by input + output length, and is distinct from submitted batch size.", "",
             "| Model | Mode | Prefill tokens | Decode tokens | Batch | Trials | Status | E2E ms | TTFT ms | TPOT ms | Output tok/s | Resident capacity |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(value: Any) -> str:
        return "" if value is None else f"{value:.2f}"
    for row in rows:
        lines.append(f"| {row['model']} | {row['mode']} | {row['ctx']} | {row['output_len']} | {row['batch']} | {row['runs_complete']}/{row['runs']} | {row['status']} | "
                     + " | ".join(fmt(row.get(key)) for key in ("batch_e2el_ms_median", "batch_ttft_ms_median", "median_tpot_ms_median", "output_throughput_median", "resident_capacity_min")) + " |")
    for row in rows:
        if row["errors"]:
            lines.extend(["", f"## {row['model']} {row['mode']} prefill={row['ctx']} decode={row['output_len']} batch={row['batch']}", "", *["- " + error for error in row["errors"]]])
    input_notes = sorted({note for row in rows for note in row["input_token_notes"]})
    if input_notes:
        lines.extend(["", "## Input token accounting", "", *["- " + note for note in input_notes]])
    notes = sorted({note for row in rows for note in row["release_notes"]})
    if notes:
        lines.extend(["", "## Runtime release notes", "", *["- " + note for note in notes]])
    (out_dir / "performance_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def _provenance(inputs: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    interesting = {key: value for key, value in env.items() if key.startswith(("JETFIRE_", "VLLM_", "OMP_", "MKL_", "OPENBLAS_", "KMP_"))
                   or key in {"PYTHONPATH", "PYTHON_BIN", "TORCH_EXTENSIONS_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}}
    return {"python": sys.executable, "python_version": sys.version, "platform": platform.platform(),
            "packages": dict(sorted(packages.items())), "source_sha256": _source_hashes(),
            "model_inputs": inputs, "environment": interesting}


def run_campaign(args: argparse.Namespace) -> int:
    cases, inputs = _cases(args)
    out_dir = args.out_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"status": "planned", "cells": len(cases), "measured_runs": len(cases) * args.runs,
                          "cases": [{**asdict(case), "command": _command(args, case, out_dir / case.key / "attempt_0001")} for case in cases]}, indent=2))
        return 0
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"evidence exists at {out_dir}; use --resume or choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _environment(args)
    provenance = _provenance(inputs, env)
    invocation = {"started_utc": _utc(), "arguments": vars(args), **provenance, "cases": [asdict(case) for case in cases]}
    if shutil.which("nvidia-smi"):
        invocation["gpu_inventory"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], capture_output=True, text=True, check=False).stdout
    invocation_dir = out_dir / "invocations"
    invocation_dir.mkdir(exist_ok=True)
    _save(invocation_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json"), invocation)
    rows = []
    try:
        for case in cases:
            case_dir = out_dir / case.key
            canonical = {"case": asdict(case), "command": _command(args, case, Path("<ATTEMPT>")),
                         "provenance": {**provenance, "model_inputs": inputs[case.model]}}
            signature = _fingerprint(canonical)
            marker = case_dir / "case.json"
            if marker.exists():
                prior = _load(marker)
                if prior["signature"] != signature:
                    raise ValueError(f"inputs, source, environment, or options changed for {case.key}; choose a new output directory")
                if prior.get("status") == "passed":
                    old = _evaluate(case, Path(prior["attempt_dir"]), int(prior["returncode"]), args.semantic_probe)
                    if old["status"] == "passed":
                        rows.append(old)
                        print(f"[campaign] resume {case.key}", flush=True)
                        _write_summary(out_dir, rows, len(cases), len(cases) * args.runs)
                        continue
            case_dir.mkdir(parents=True, exist_ok=True)
            index = 1
            while (case_dir / f"attempt_{index:04d}").exists():
                index += 1
            attempt = case_dir / f"attempt_{index:04d}"
            attempt.mkdir()
            cmd = _command(args, case, attempt)
            record = {"signature": signature, "case": asdict(case), "command": cmd,
                      "attempt_dir": str(attempt), "status": "running", "started_utc": _utc()}
            _save(marker, record)
            _save(attempt / "invocation.json", record)
            print(f"[campaign] {case.key} runs={case.runs} GPUs={case.gpus} evidence={attempt}", flush=True)
            started = time.monotonic()
            with (attempt / "campaign.log").open("w") as handle:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
            result = _evaluate(case, attempt, completed.returncode, args.semantic_probe)
            result["wall_seconds"] = time.monotonic() - started
            record.update(status=result["status"], returncode=completed.returncode, finished_utc=_utc())
            _save(attempt / "invocation.json", record)
            _save(marker, record)
            rows.append(result)
            _write_summary(out_dir, rows, len(cases), len(cases) * args.runs)
            print(f"[campaign] {result['status']} runs={result['runs_complete']}/{case.runs} E2E_ms={result['batch_e2el_ms_median']}", flush=True)
            if result["errors"]:
                print("[campaign] " + "\n[campaign] ".join(result["errors"]), flush=True)
                if args.fail_fast:
                    break
    except KeyboardInterrupt:
        _write_summary(out_dir, rows, len(cases), len(cases) * args.runs, interrupted=True)
        return 130
    except Exception as exc:
        summary = _write_summary(out_dir, rows, len(cases), len(cases) * args.runs)
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save(out_dir / "performance_summary.json", summary)
        raise
    summary = _write_summary(out_dir, rows, len(cases), len(cases) * args.runs)
    return 0 if summary["status"] == "passed" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="llama31_8b,qwen3_30b_a3b,llama33_70b")
    parser.add_argument("--modes", default="k3v2,k4v4")
    parser.add_argument("--model-path", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--contexts", default="4096,32768")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--single-gpus", default="1")
    parser.add_argument("--tp-gpus", default="0,1")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--profile", choices=("graph", "eager"), default="graph")
    parser.add_argument("--prefill-chunk", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--server-timeout-s", type=int, default=1200)
    parser.add_argument("--bench-timeout-s", type=float, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--semantic-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-extensions-dir", default=os.environ.get("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache/torch_extensions")))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/performance")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("output_len", "runs", "top_k", "cpu_threads", "server_timeout_s", "bench_timeout_s"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmups < 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("warmups must be nonnegative and GPU memory utilization must be in (0, 1]")
    try:
        _cases(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--harness-worker":
        return _harness_worker(argv[1:])
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
