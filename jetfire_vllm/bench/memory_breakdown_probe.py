#!/usr/bin/env python3
"""JetFire memory footprint breakdown probe.

This script has two modes:

* computed-only: write shape-derived accounting rows without starting vLLM.
* run: start one vLLM/JetFire point, run one prompt, and append measured rows.

The CSV schema is intentionally long-form so plots and paper tables can pivot it
without re-running experiments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GIB = 1 << 30


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: str
    layers: int
    kv_heads: int
    head_dim: int
    tp: int
    max_window: int


@dataclass(frozen=True)
class ConfigSpec:
    key: str
    label: str
    backend: str
    mode: str
    group_size: int
    k_bits: float
    v_bits: float
    scale_bits_avg: float

    @property
    def avg_bits(self) -> float:
        return (self.k_bits + self.v_bits) / 2.0 + self.scale_bits_avg


MODELS: dict[str, ModelSpec] = {
    "llama31_8b": ModelSpec(
        key="llama31_8b",
        label="Llama-3.1-8B-Instruct",
        path=os.environ.get("JETFIRE_LLAMA31_8B", "/path/to/Llama-3.1-8B-Instruct"),
        layers=32,
        kv_heads=8,
        head_dim=128,
        tp=1,
        max_window=131072,
    ),
    "qwen3_30b_a3b": ModelSpec(
        key="qwen3_30b_a3b",
        label="Qwen3-30B-A3B",
        path=os.environ.get("JETFIRE_QWEN3_30B", "/path/to/Qwen3-30B-A3B-Instruct"),
        layers=48,
        kv_heads=4,
        head_dim=128,
        tp=1,
        max_window=256000,
    ),
    "llama33_70b": ModelSpec(
        key="llama33_70b",
        label="Llama-3.3-70B-Instruct",
        path=os.environ.get("JETFIRE_LLAMA33_70B", "/path/to/Llama-3.3-70B-Instruct"),
        layers=80,
        kv_heads=8,
        head_dim=128,
        tp=2,
        max_window=131072,
    ),
}


CONFIGS: dict[str, ConfigSpec] = {
    "k4v4": ConfigSpec(
        key="k4v4",
        label="JetFire K4V4 g128",
        backend="jetfire",
        mode="SymInt4K_SymInt4V",
        group_size=128,
        k_bits=4.0,
        v_bits=4.0,
        scale_bits_avg=0.125,
    ),
    "k3v2": ConfigSpec(
        key="k3v2",
        label="JetFire K3V2 g64",
        backend="jetfire",
        mode="SymInt3K_SymInt2V",
        group_size=64,
        k_bits=3.0,
        v_bits=2.0,
        scale_bits_avg=0.25,
    ),
}


FIELDNAMES = [
    "model",
    "model_label",
    "config",
    "config_label",
    "ctx",
    "tp",
    "rank",
    "side",
    "stage",
    "component",
    "computed_gib",
    "measured_gib",
    "source",
    "notes",
]


def _gib(x: float | int | None) -> str:
    if x is None:
        return ""
    return f"{float(x) / GIB:.6f}"


def _bytes_per_token(model: ModelSpec, bits: float) -> float:
    return model.layers * 2 * model.kv_heads * model.head_dim * bits / 8.0


def _component_bytes(model: ModelSpec, config: ConfigSpec, ctx: int) -> dict[str, float]:
    elems = model.layers * model.kv_heads * model.head_dim * ctx
    return {
        "KV replica - packed K": elems * config.k_bits / 8.0,
        "KV replica - packed V": elems * config.v_bits / 8.0,
        "KV replica - quant scales/zeros": elems * 2.0 * config.scale_bits_avg / 8.0,
    }


def computed_rows(model: ModelSpec, config: ConfigSpec, ctx: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gpu_components = _component_bytes(model, config, ctx)
    for rank, divisor, note in [(-1, 1, "aggregate"), *[(r, model.tp, "per-GPU estimate") for r in range(model.tp)]]:
        for component, nbytes in gpu_components.items():
            rows.append(_row(model, config, ctx, "gpu", "computed", component, nbytes / divisor, None, "shape", note, rank))
        gpu_total = sum(gpu_components.values()) / divisor
        rows.append(_row(model, config, ctx, "gpu", "computed", "KV replica - total", gpu_total, None, "shape", note, rank))
    gpu_total = sum(gpu_components.values())

    host_replica = _bytes_per_token(model, 16.0) * ctx
    if config.backend == "jetfire":
        for rank, divisor, note in [(-1, 1, "aggregate"), *[(r, model.tp, "per-GPU estimate") for r in range(model.tp)]]:
            rows.append(_row(model, config, ctx, "host", "computed", "full-precision host replica", host_replica / divisor, None, "shape", note, rank))
            rows.append(_row(model, config, ctx, "host", "computed", "host/GPU replica ratio", None, host_replica / max(gpu_total, 1), "shape", f"{note}; ratio, not GiB", rank))
    return rows


def _row(
    model: ModelSpec,
    config: ConfigSpec,
    ctx: int,
    side: str,
    stage: str,
    component: str,
    computed_bytes: float | int | None,
    measured_bytes_or_value: float | int | None,
    source: str,
    notes: str = "",
    rank: int = -1,
) -> dict[str, Any]:
    measured = ""
    if measured_bytes_or_value is not None:
        if "not GiB" in notes or "ratio" in component:
            measured = f"{float(measured_bytes_or_value):.6f}"
        else:
            measured = _gib(measured_bytes_or_value)
    return {
        "model": model.key,
        "model_label": model.label,
        "config": config.key,
        "config_label": config.label,
        "ctx": int(ctx),
        "tp": int(model.tp),
        "rank": int(rank),
        "side": side,
        "stage": stage,
        "component": component,
        "computed_gib": _gib(computed_bytes),
        "measured_gib": measured,
        "source": source,
        "notes": notes,
    }


def _nvidia_smi(gpus: str) -> dict[int, dict[str, int]]:
    wanted = {int(x) for x in str(gpus).split(",") if x.strip()}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    rows: dict[int, dict[str, int]] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, used, total = map(int, parts)
        if not wanted or idx in wanted:
            rows[idx] = {"used_mib": used, "total_mib": total}
    return rows


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _children_recursive(pid: int) -> list[int]:
    children: list[int] = []
    proc = Path("/proc")
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            ppid = int(stat.split()[3])
        except Exception:
            continue
        if ppid == pid:
            cpid = int(p.name)
            children.append(cpid)
            children.extend(_children_recursive(cpid))
    return children


def _process_tree_rss() -> int:
    pids = [os.getpid(), *_children_recursive(os.getpid())]
    return sum(_rss_bytes(pid) for pid in pids)


def _make_prompt_tokens(model_path: str, ctx: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = dict  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_id = tok.encode(" hello", add_special_tokens=False)[0]
    ids = [int(token_id)] * int(ctx)
    try:
        return TokensPrompt(prompt_token_ids=ids)
    except TypeError:
        return {"prompt_token_ids": ids}


def _configure_env(model: ModelSpec, config: ConfigSpec, args: argparse.Namespace) -> None:
    os.environ["JETFIRE_USE_CUSTOM_BACKEND"] = "1"
    os.environ["JETFIRE_MODEL_PATH"] = model.path
    os.environ["JETFIRE_MODE"] = config.mode
    os.environ["JETFIRE_GROUP_SIZE"] = str(config.group_size)
    os.environ["JETFIRE_TOPK"] = str(args.top_k)
    os.environ["JETFIRE_MAX_BATCH_SIZE"] = "1"
    os.environ["JETFIRE_CPU_ATTN_THREADS"] = str(args.cpu_attn_threads)
    os.environ.setdefault("JETFIRE_CPU_CORRECTION", "1")
    os.environ.setdefault("JETFIRE_DECODE_CPU_ATTN", "1")
    os.environ.setdefault("JETFIRE_SKIP_NATIVE_KV", "1")
    os.environ.setdefault("JETFIRE_PRELOAD_DECODE_EXTENSIONS", "1")
    chunk_env_keys = (
        "JETFIRE_EMBED_CHUNK_TOKENS",
        "JETFIRE_MLP_CHUNK_TOKENS",
        "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS",
        "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS",
        "JETFIRE_PREFILL_QKV_CHUNK_TOKENS",
    )
    if int(args.jetfire_prefill_chunk_tokens) > 0:
        chunk_s = str(int(args.jetfire_prefill_chunk_tokens))
        os.environ.setdefault("JETFIRE_EMBED_EMPTY_CACHE", "1")
        for key in chunk_env_keys:
            os.environ[key] = chunk_s
    else:
        for key in chunk_env_keys:
            os.environ.pop(key, None)
    os.environ["JETFIRE_DECODE_GRAPH"] = "1" if args.jetfire_graph else "0"
    os.environ["JETFIRE_ENABLE_VLLM_CUDAGRAPH"] = "1" if args.jetfire_graph else "0"
    os.environ["JETFIRE_RELEASE_GRAPH_POLICY"] = "1" if args.jetfire_graph else "0"
    os.environ["JETFIRE_RELEASE_REQUIRE_READY"] = "1" if args.jetfire_graph else "0"
    os.environ["JETFIRE_SYNC_CPU_ATTN"] = "0" if args.jetfire_graph else "1"
    os.environ["JETFIRE_DISABLE_ASYNC_SCHEDULING"] = "1"
    if args.jetfire_graph:
        os.environ.setdefault("JETFIRE_ZERO_INIT_CACHE", "0")
    if args.jetfire_graph:
        from jetfire_vllm.adapter.config import apply_graph_cpu_topk_release_preset
        apply_graph_cpu_topk_release_preset()


def _llm_kwargs(model: ModelSpec, config: ConfigSpec, ctx: int, args: argparse.Namespace) -> dict[str, Any]:
    required_len = int(ctx if args.skip_generate else ctx + args.output_len)
    kwargs: dict[str, Any] = {
        "tensor_parallel_size": int(args.tensor_parallel_size or model.tp),
        "max_model_len": required_len,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
    }
    if int(kwargs["tensor_parallel_size"]) > 1:
        kwargs["disable_custom_all_reduce"] = True
    if config.backend == "jetfire":
        kwargs["enable_chunked_prefill"] = False
        # vLLM may need one extra paged block at the prefill→decode boundary
        # before the request is finalized.  A strict ceil(required_len/block)
        # budget can leave the scheduler with no progress at ctx exactly on a
        # block boundary (for example 4096 prompt + 2 generated tokens with
        # block_size=16).  Keep the measurement "just enough" while adding a
        # tiny fixed safety margin; this changes 128K-scale footprint by <0.1%.
        kwargs["num_gpu_blocks_override"] = max(
            1,
            math.ceil(required_len / args.kv_cache_block_size)
            + int(args.kv_block_safety),
        )
        if args.jetfire_graph:
            kwargs["enforce_eager"] = False
            kwargs["compilation_config"] = {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
                "max_cudagraph_capture_size": 1,
            }
        else:
            kwargs["enforce_eager"] = True
    return kwargs


def run_point(model: ModelSpec, config: ConfigSpec, ctx: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = computed_rows(model, config, ctx)
    _configure_env(model, config, args)

    rss_before = _process_tree_rss()
    smi_before = _nvidia_smi(args.gpus)
    stage = "after_model_and_kv_init"
    llm = None
    try:
        if config.backend == "jetfire":
            print("[memory-breakdown] building JetFire LLM", flush=True)
            from jetfire_vllm.adapter.engine import (
                build_llm,
                collect_jetfire_worker_memory,
                collect_jetfire_worker_stats,
                reset_jetfire_worker_peak_memory,
            )
            llm = build_llm(
                model=model.path,
                use_jetfire_backend=True,
                llm_kwargs=_llm_kwargs(model, config, ctx, args),
            )
            print("[memory-breakdown] JetFire LLM ready", flush=True)
            if args.collect_init_worker_rpc:
                print("[memory-breakdown] collecting init worker RPC memory/stats", flush=True)
                stats = collect_jetfire_worker_stats(getattr(llm, "_llm", llm))
                mem = collect_jetfire_worker_memory(getattr(llm, "_llm", llm))
                for rank, m in enumerate(mem):
                    rows.extend(_jetfire_memory_rows(model, config, ctx, stage, m, rank))
                for rank, s in enumerate(stats):
                    cap = _capacity_tokens(s)
                    if cap:
                        rows.append(_row(model, config, ctx, "gpu", stage, "resident capacity tokens", None, cap, "jetfire_stats", "tokens, not GiB", rank))
            smi = _nvidia_smi(args.gpus)
            for gpu, row in smi.items():
                before = smi_before.get(gpu, {"used_mib": 0})["used_mib"]
                delta = max(0, row["used_mib"] - before) * (1 << 20)
                rows.append(_row(model, config, ctx, "gpu", stage, "nvidia-smi used delta", None, delta, "nvidia-smi", rank=gpu))
            rows.append(_row(model, config, ctx, "host", stage, "process RSS delta", None, _process_tree_rss() - rss_before, "rss"))
            if args.collect_init_worker_rpc:
                reset_jetfire_worker_peak_memory(getattr(llm, "_llm", llm))

        if args.skip_generate:
            print("[memory-breakdown] skip generate requested", flush=True)
            return rows

        from vllm import SamplingParams
        prompt = _make_prompt_tokens(model.path, ctx)
        params = SamplingParams(max_tokens=int(args.output_len), temperature=0.0, ignore_eos=True)
        if config.backend == "jetfire" and args.reset_peak_before_generate:
            reset_jetfire_worker_peak_memory(getattr(llm, "_llm", llm))
        print("[memory-breakdown] starting generate", flush=True)
        t0 = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        print(f"[memory-breakdown] generate done elapsed={elapsed:.3f}s", flush=True)
        rows.append(_row(model, config, ctx, "gpu", "end_to_end", "elapsed seconds", None, elapsed, "timer", "seconds, not GiB"))
        smi = _nvidia_smi(args.gpus)
        for gpu, row in smi.items():
            before = smi_before.get(gpu, {"used_mib": 0})["used_mib"]
            delta = max(0, row["used_mib"] - before) * (1 << 20)
            rows.append(_row(model, config, ctx, "gpu", "after_generate", "nvidia-smi used delta", None, delta, "nvidia-smi", rank=gpu))
        rows.append(_row(model, config, ctx, "host", "after_generate", "process RSS delta", None, _process_tree_rss() - rss_before, "rss"))
        if config.backend == "jetfire":
            if not args.skip_post_worker_rpc:
                from jetfire_vllm.adapter.engine import collect_jetfire_worker_memory
                print("[memory-breakdown] collecting post-generate worker RPC memory", flush=True)
                mem = collect_jetfire_worker_memory(getattr(llm, "_llm", llm))
                for rank, m in enumerate(mem):
                    rows.extend(_jetfire_memory_rows(model, config, ctx, "after_generate", m, rank))
    finally:
        del llm
        gc.collect()
    return rows


def _capacity_tokens(stats: dict[str, Any]) -> int | None:
    caps: list[int] = []
    layouts = stats.get("layer_layouts")
    if isinstance(layouts, dict):
        for layout in layouts.values():
            if isinstance(layout, dict) and "num_blocks" in layout and "block_size" in layout:
                caps.append(int(layout["num_blocks"]) * int(layout["block_size"]))
    return min(caps) if caps else None


def _jetfire_memory_rows(model: ModelSpec, config: ConfigSpec, ctx: int, stage: str, m: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    rows = [
        _row(model, config, ctx, "gpu", stage, "torch.cuda.memory_allocated", None, int(m.get("cuda_memory_allocated", 0)), "torch", rank=rank),
        _row(model, config, ctx, "gpu", stage, "torch.cuda.memory_reserved", None, int(m.get("cuda_memory_reserved", 0)), "torch", rank=rank),
        _row(model, config, ctx, "gpu", stage, "torch.cuda.max_memory_allocated", None, int(m.get("cuda_max_memory_allocated", 0)), "torch", rank=rank),
        _row(model, config, ctx, "gpu", stage, "JetFire GPU KV allocated", None, int(m.get("jetfire_int4_hbm_bytes", 0)), "jetfire_worker", rank=rank),
        _row(model, config, ctx, "gpu", stage, "JetFire GPU KV used", None, int(m.get("jetfire_int4_hbm_used_bytes", 0)), "jetfire_worker", rank=rank),
        _row(model, config, ctx, "host", stage, "full-precision host replica allocated", None, int(m.get("jetfire_cpu_pinned_bytes", 0)), "jetfire_worker", rank=rank),
        _row(model, config, ctx, "host", stage, "full-precision host replica used", None, int(m.get("jetfire_cpu_pinned_used_bytes", 0)), "jetfire_worker", rank=rank),
        _row(model, config, ctx, "gpu", stage, "native vLLM KV bytes", None, int(m.get("vllm_kv_bytes", 0)), "jetfire_worker", rank=rank),
    ]
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--mode", choices=["computed-only", "run"], default="computed-only")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--kv-cache-block-size", type=int, default=16)
    parser.add_argument("--kv-block-safety", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--jetfire-prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--cpu-attn-threads", type=int, default=32)
    parser.add_argument("--jetfire-graph", action="store_true")
    parser.add_argument("--collect-init-worker-rpc", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-post-worker-rpc", action="store_true")
    parser.add_argument("--reset-peak-before-generate", action="store_true")
    parser.add_argument("--allow-over-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MODELS[args.model]
    from jetfire_vllm.bench.model_paths import resolve_model_path

    model = replace(model, path=resolve_model_path(args.model, args.model_path, model.path), tp=args.tensor_parallel_size or model.tp)
    if args.ctx <= 0 or model.tp <= 0 or args.kv_cache_block_size <= 0:
        raise ValueError("Context length, tensor parallel size, and cache block size must be positive")
    if args.mode == "run" and (model.path.startswith("/path/to/") or not Path(model.path).expanduser().is_dir()):
        raise FileNotFoundError(f"Local model directory unavailable: {model.path}. Set --model-path or the model JETFIRE_* environment variable.")
    config = CONFIGS[args.config]
    if args.ctx > model.max_window and not args.allow_over_window:
        print(
            f"[memory-breakdown] skip model={model.key} ctx={args.ctx} "
            f"above model window={model.max_window}",
            flush=True,
        )
        return
    rows = computed_rows(model, config, args.ctx)
    if args.mode == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        rows = run_point(model, config, args.ctx, args)
    write_rows(args.out_csv, rows)
    print(f"[memory-breakdown] wrote {len(rows)} rows to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
