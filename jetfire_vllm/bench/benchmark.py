"""Plan or run JetFire benchmarks with user-selected token lengths.

For example, after configuring a local checkpoint::

    python -m jetfire_vllm.bench.benchmark --models llama31_8b \
      --prefill-len 8192 --decode-len 256 --batches 4 --runs 3 --execute

Planning is the default and inspects neither checkpoints nor GPUs. Each timed
trial uses a fresh server and the requested prefill and decode token counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Iterator

from jetfire_vllm.bench import artifact_performance as campaign
from jetfire_vllm.bench import official_serve_benchmark as harness


MODELS = ("llama31_8b", "qwen3_30b_a3b")

POLICY = {
    "version": 1,
    "kv_policy": "bounded",
    "scheduler_policy": "full-fit",
    "gpu_memory_utilization": 0.95,
    "scratch_reserve_gib": 5.0,
    "kv_budget_safety": 1.02,
    "kv_cache_block_size": 16,
    "max_model_len_padding": 16,
    "prefill_sub_batch_policy": "auto",
    "prefill_runtime_reserve_gib": 25.0,
    "prefill_activation_safety": 1.15,
    "prefill_auto_max_sub_batch_size": 4,
    "queued_scheduler_headroom_tokens": 4096,
    "top_k": 256,
    "profile": "graph",
    "graph_scope": "decode-only",
    "prefill_chunk": "auto",
    "chunked_prefill": "off",
}


class CapacityPreflightError(ValueError):
    """The configured device cannot admit one complete request."""


def _models(raw: str) -> list[str]:
    values = [campaign.ALIASES.get(value, value) for value in campaign._csv(raw)]
    if len(values) != len(set(values)) or set(values) - set(MODELS):
        raise ValueError("--models must contain unique llama31_8b and/or qwen3_30b (qwen3_30b_a3b)")
    return values


def _overrides(args: argparse.Namespace) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in args.model_path:
        key, separator, value = item.partition("=")
        key = campaign.ALIASES.get(key, key)
        if not separator or key not in MODELS or not value or key in result:
            raise ValueError("--model-path requires a unique known MODEL=PATH")
        result[key] = value
    if set(result) - set(_models(args.models)):
        raise ValueError("--model-path names a model not selected by --models")
    return result


def _requested_cases(args: argparse.Namespace) -> tuple[list[campaign.Case], dict[str, Any]]:
    """Use exactly the requested lengths, validating local inputs at execution."""
    overrides = _overrides(args)
    if args.dry_run:
        cases, inputs = [], {}
        for model in _models(args.models):
            configured = overrides.get(model) or os.environ.get(campaign.MODEL_ENV[model])
            path = str(Path(configured).expanduser().resolve()) if configured else "/path/to/" + model
            inputs[model] = {"path": path, "status": "not_inspected_in_dry_run"}
            cases.extend(campaign.Case(model, path, 1, args.single_gpus, mode,
                                       args.prefill_len, batch, args.decode_len, args.runs)
                         for mode in campaign._csv(args.modes)
                         for batch in campaign._positive_csv(args.batches))
        return cases, inputs
    cases, inputs = campaign._cases(args)
    for model, inventory in inputs.items():
        if not inventory["weight_shards"]:
            raise ValueError(f"{model}: expected locally downloaded BF16 .safetensors weight shards")
        if inventory["config"].get("quantization_config"):
            raise ValueError(f"{model}: use the original BF16 checkpoint, not quantized model weights")
    return cases, inputs


def budget_namespace(case: campaign.Case) -> SimpleNamespace:
    """Inputs to the same KV rounding and automatic prefill policy as serving."""
    return SimpleNamespace(
        model=case.model_path, gpu=case.gpus, tensor_parallel_size=case.tp,
        ctx=case.ctx, output_len=case.output_len, concurrency=case.batch,
        max_model_len=0, max_model_len_padding=POLICY["max_model_len_padding"],
        jetfire_mode=campaign.MODES[case.mode][0], jetfire_group_size=campaign.MODES[case.mode][1],
        kv_cache_memory_bytes=None, jetfire_kv_cache_budget="auto",
        jetfire_kv_budget_safety=POLICY["kv_budget_safety"],
        kv_cache_block_size=POLICY["kv_cache_block_size"],
        gpu_memory_utilization=POLICY["gpu_memory_utilization"],
        jetfire_prefill_sub_batch_size=POLICY["prefill_sub_batch_policy"],
        jetfire_prefill_runtime_reserve_gib=POLICY["prefill_runtime_reserve_gib"],
        jetfire_prefill_activation_safety=POLICY["prefill_activation_safety"],
        jetfire_prefill_auto_max_sub_batch_size=POLICY["prefill_auto_max_sub_batch_size"],
    )


def bounded_allocation(ns: SimpleNamespace, submitted: int, gpu_total_gib: float,
                       weights_gib: float, scratch_gib: float) -> dict[str, Any]:
    """Choose the largest integral resident workload below an explicit HBM cap."""
    if submitted <= 0 or not all(math.isfinite(value) and value >= 0
                                 for value in (gpu_total_gib, weights_gib, scratch_gib)):
        raise CapacityPreflightError("invalid bounded-KV allocation inputs")
    limit_bytes = math.floor((POLICY["gpu_memory_utilization"] * gpu_total_gib
                              - weights_gib - scratch_gib) * 1024**3)
    workload = harness._kv_budget(ns, "jetfire", submitted)
    for resident in range(submitted, 0, -1):
        candidate = harness._kv_budget(ns, "jetfire", resident)
        if candidate.bytes is not None and candidate.bytes <= limit_bytes:
            return {
                "resident_requests_target": resident,
                "manual_kv_budget_bytes": candidate.bytes,
                "bounded_kv_limit_bytes": limit_bytes,
                "scratch_reserve_gib": scratch_gib,
                "all_submitted_workload_budget_bytes": workload.bytes,
                "kv_budget_safety": candidate.safety,
                "kv_budget_block_size": candidate.block_size,
                "kv_budget_base_blocks": candidate.base_blocks,
                "kv_budget_inflate_ratio": candidate.inflate_ratio,
            }
    raise CapacityPreflightError(
        f"No full request fits bounded KV cap {limit_bytes} bytes; "
        f"GPU={gpu_total_gib:.3f} GiB weights={weights_gib:.3f} GiB scratch={scratch_gib:.3f} GiB")


def resolved_budget(case: campaign.Case) -> dict[str, Any]:
    ns = budget_namespace(case)
    model_len = harness._max_model_len(ns)
    gpu_total = harness._gpu_total_gib(case.gpus)
    weights = harness._model_weights_gib(ns)
    bounded = bounded_allocation(ns, case.batch, gpu_total, weights, POLICY["scratch_reserve_gib"])
    # Automatic prefill sizing must see the cap actually sent to vLLM.
    ns.kv_cache_memory_bytes = bounded["manual_kv_budget_bytes"]
    sub = harness._resolved_prefill_sub_batch_size(ns, "jetfire", case.batch)
    admitted_sub = min(sub or case.batch, bounded["resident_requests_target"])
    tokens = (admitted_sub * model_len + POLICY["queued_scheduler_headroom_tokens"]
              if admitted_sub < case.batch else case.batch * model_len)
    return {
        "scheduler_policy": POLICY["scheduler_policy"],
        "max_num_batched_tokens_explicit": tokens,
        "automatic_prefill_sub_batch": sub,
        "scheduler_prefill_sub_batch": admitted_sub,
        "max_model_len": model_len,
        "kv_policy": POLICY["kv_policy"],
        "weights_gib": weights,
        "gpu_total_gib": gpu_total,
        "prefill_sub_batch_policy": POLICY["prefill_sub_batch_policy"],
        **bounded,
    }


def _query_gpu_total_gib(gpus: str) -> float:
    """Require real capacity at execution; do not silently assume an 80 GiB GPU."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={gpus}", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        value = float(output.strip()) / 1024.0
        if not math.isfinite(value) or value <= 0:
            raise ValueError("nonpositive/nonfinite device capacity")
        return value
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Cannot query GPU {gpus} total HBM with nvidia-smi") from exc


def _affinity() -> list[int] | None:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def _command_with_budget(original: Any, args: argparse.Namespace, case: campaign.Case,
                         attempt: Path, budget: dict[str, Any]) -> list[str]:
    return original(args, case, attempt) + [
        "--kv-cache-memory-bytes", str(budget["manual_kv_budget_bytes"]),
        "--max-num-batched-tokens", str(budget["max_num_batched_tokens_explicit"]),
        "--jetfire-prefill-sub-batch-size", "auto",
        "--jetfire-kv-budget-safety", str(POLICY["kv_budget_safety"]),
        "--kv-cache-block-size", str(POLICY["kv_cache_block_size"]),
        "--max-model-len-padding", str(POLICY["max_model_len_padding"]),
        "--jetfire-prefill-runtime-reserve-gib", str(POLICY["prefill_runtime_reserve_gib"]),
        "--jetfire-prefill-activation-safety", str(POLICY["prefill_activation_safety"]),
        "--jetfire-prefill-auto-max-sub-batch-size", str(POLICY["prefill_auto_max_sub_batch_size"]),
    ]


@contextmanager
def configured_campaign(args: argparse.Namespace, cases: list[campaign.Case],
                        inputs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Scope policy adaptation to this invocation; campaign fingerprints it all."""
    original_cases, original_command = campaign._cases, campaign._command
    original_provenance, original_gpu_total = campaign._provenance, harness._gpu_total_gib
    gpu_total = _query_gpu_total_gib(args.single_gpus)
    affinity = _affinity()
    try:
        harness._gpu_total_gib = lambda gpu: gpu_total if gpu == args.single_gpus else original_gpu_total(gpu)
        budgets = {case.key: resolved_budget(case) for case in cases}
        campaign._cases = lambda _: (cases, inputs)

        def command(options: argparse.Namespace, case: campaign.Case, attempt: Path) -> list[str]:
            if _affinity() != affinity:
                raise RuntimeError("CPU affinity changed after matrix preflight")
            return _command_with_budget(original_command, options, case, attempt, budgets[case.key])

        def provenance(model_inputs: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
            return {**original_provenance(model_inputs, env), "benchmark_policy": dict(POLICY),
                    "gpu_total_gib": gpu_total, "cpu_affinity": affinity}

        campaign._command, campaign._provenance = command, provenance
        yield budgets
    finally:
        campaign._cases, campaign._command = original_cases, original_command
        campaign._provenance, harness._gpu_total_gib = original_provenance, original_gpu_total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="llama31_8b", help="comma-separated llama31_8b and/or qwen3_30b_a3b")
    parser.add_argument("--model-path", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--prefill-len", type=int, required=True, help="prompt length in tokens")
    parser.add_argument("--decode-len", type=int, required=True, help="generated output length in tokens")
    parser.add_argument("--modes", default="k3v2,k4v4", help="comma-separated k3v2 and/or k4v4")
    parser.add_argument("--batches", default="1", help="comma-separated positive request batch sizes")
    parser.add_argument("--runs", type=int, default=3, help="fresh-server timed trials per configuration")
    parser.add_argument("--warmups", type=int, default=1, help="untimed warmup requests per trial")
    parser.add_argument("--semantic-probe", action=argparse.BooleanOptionalAction, default=False,
                        help="opt into an untimed arithmetic check; its prompt needs sufficient model-window space")
    parser.add_argument("--single-gpus", default="0", help="one numeric GPU index; both supported models use TP=1")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--server-timeout-s", type=int, default=1200)
    parser.add_argument("--bench-timeout-s", type=float, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-extensions-dir", default=os.environ.get(
        "TORCH_EXTENSIONS_DIR", str(campaign.ROOT / ".cache/torch_extensions")))
    parser.add_argument("--out-dir", type=Path, default=Path("results/performance"))
    parser.add_argument("--resume", action="store_true", help="reuse only complete records with matching controls and provenance")
    parser.add_argument("--fail-fast", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="start GPU benchmarks; omitted means plan only")
    execution.add_argument("--dry-run", action="store_true", help="print the plan without model/GPU access or creating files (default)")
    args = parser.parse_args(argv)
    args.dry_run = not args.execute
    args.contexts, args.output_len = str(args.prefill_len), args.decode_len
    args.tp_gpus = "0,1"  # Shared campaign field; this frontend selects TP=1 models.
    args.top_k = POLICY["top_k"]
    args.profile, args.prefill_chunk = POLICY["profile"], POLICY["prefill_chunk"]
    args.gpu_memory_utilization = POLICY["gpu_memory_utilization"]
    try:
        _models(args.models)
        _overrides(args)
        if min(args.prefill_len, args.decode_len, args.runs) <= 0:
            raise ValueError("prefill length, decode length and runs must be positive")
        if args.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if set(campaign._csv(args.modes)) - set(campaign.MODES):
            raise ValueError("--modes must select k3v2 and/or k4v4")
        campaign._positive_csv(args.batches)
        if not args.single_gpus.isdigit():
            raise ValueError("--single-gpus requires one numeric GPU index")
        if min(args.cpu_threads, args.server_timeout_s, args.bench_timeout_s) <= 0 or not math.isfinite(args.bench_timeout_s):
            raise ValueError("CPU threads and timeouts must be positive and finite")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases, inputs = _requested_cases(args)
    if args.dry_run:
        placeholder = {"manual_kv_budget_bytes": "<resolved from checkpoint and GPU capacity>",
                       "max_num_batched_tokens_explicit": "<resolved from resident and prefill capacity>"}
        plan = {
            "status": "planned", "execute_required": True, "cells": len(cases),
            "measured_runs": len(cases) * args.runs,
            "measured_requests": sum(case.batch * case.runs for case in cases),
            "generated_output_tokens": sum(case.batch * case.runs * case.output_len for case in cases),
            "policy": dict(POLICY),
            "budget_note": "Budgets use total HBM and local BF16 checkpoint bytes at execution; no GPU or checkpoint is inspected while planning.",
            "cases": [{**asdict(case), "command_template": _command_with_budget(
                campaign._command, args, case, args.out_dir.expanduser().resolve() / case.key / "attempt_0001", placeholder)}
                      for case in cases],
        }
        print(json.dumps(plan, indent=2))
        return 0
    with configured_campaign(args, cases, inputs):
        return campaign.run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
