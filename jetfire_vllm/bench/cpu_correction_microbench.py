#!/usr/bin/env python3
"""Measure JetFire's production CPU top-k correction kernel in isolation.

The benchmark uses the same C++/AVX-512 BF16 extension as release decode.
It deliberately excludes GPU hand-off and graph scheduling so the reported
latency isolates CPU correction scaling with top-k, batch, and thread count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import platform
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from jetfire_vllm.kernels.cpu_topk_attn import load_cpu_topk_attn_ext


MODEL_SHAPES = {
    "llama31_8b": {"num_query_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    "qwen3_30b": {"num_query_heads": 32, "num_kv_heads": 4, "head_dim": 128},
    # Per-rank shape for Llama-3.3-70B with tensor parallelism 2.
    "llama33_70b_tp2": {
        "num_query_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
}


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _make_inputs(
    *,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    top_k: int,
    num_slots: int,
    k_shadow: torch.Tensor,
    v_shadow: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    query = torch.empty(
        (batch, num_query_heads, head_dim), dtype=torch.bfloat16
    ).uniform_(-0.25, 0.25)
    gqa_ratio = num_query_heads // num_kv_heads
    per_head_map = (
        torch.arange(num_query_heads, dtype=torch.int64) // gqa_ratio
    )
    head_map = per_head_map.view(1, num_query_heads, 1).expand(
        batch, num_query_heads, top_k
    ).contiguous()
    slots = torch.randint(
        0,
        num_slots,
        (batch, num_query_heads, top_k),
        dtype=torch.int64,
    )
    output = torch.empty(
        (batch, num_query_heads, head_dim), dtype=torch.bfloat16
    )
    return query, k_shadow, v_shadow, head_map, slots, output


def run(args: argparse.Namespace) -> None:
    shape = MODEL_SHAPES[args.model]
    batches = _csv_ints(args.batches)
    topks = _csv_ints(args.topks)
    thread_counts = _csv_ints(args.threads)
    if not batches or not topks or not thread_counts:
        raise ValueError("batches, topks, and threads must be non-empty")
    if shape["num_query_heads"] % shape["num_kv_heads"]:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(args.seed)
    num_slots = int(args.context)
    shadow_shape = (
        shape["num_kv_heads"],
        num_slots,
        shape["head_dim"],
    )
    print(
        f"[cpu-correction] allocating two BF16 shadows with shape={shadow_shape}",
        flush=True,
    )
    k_shadow = torch.empty(shadow_shape, dtype=torch.bfloat16).uniform_(-0.25, 0.25)
    v_shadow = torch.empty(shadow_shape, dtype=torch.bfloat16).uniform_(-0.25, 0.25)

    ext = load_cpu_topk_attn_ext()
    rows: list[dict[str, Any]] = []
    for batch in batches:
        for top_k in topks:
            inputs = _make_inputs(
                batch=batch,
                top_k=top_k,
                num_slots=num_slots,
                k_shadow=k_shadow,
                v_shadow=v_shadow,
                **shape,
            )
            scale = 1.0 / math.sqrt(shape["head_dim"])
            slots = inputs[4]
            logical_read_bytes = (
                batch
                * shape["num_query_heads"]
                * top_k
                * shape["head_dim"]
                * 2  # BF16 bytes
                * 2  # K and V
            )
            for threads in thread_counts:
                ext.set_cpu_attn_threads(threads)
                for _ in range(args.warmups):
                    slots.random_(0, num_slots)
                    ext.cpu_topk_attn(*inputs, scale)
                samples_us: list[float] = []
                for _ in range(args.repeats):
                    # Production decode receives a new selected-token set on
                    # every step.  Reusing one set here makes the sparse K/V
                    # working set resident in LLC and badly overstates DRAM
                    # throughput, especially at batch 1.
                    slots.random_(0, num_slots)
                    start_ns = time.perf_counter_ns()
                    ext.cpu_topk_attn(*inputs, scale)
                    samples_us.append((time.perf_counter_ns() - start_ns) / 1_000.0)
                median_us = statistics.median(samples_us)
                row = {
                    "model": args.model,
                    "context_slots": num_slots,
                    "batch": batch,
                    "num_query_heads": shape["num_query_heads"],
                    "num_kv_heads": shape["num_kv_heads"],
                    "head_dim": shape["head_dim"],
                    "top_k": top_k,
                    "threads": threads,
                    "median_us": median_us,
                    "mean_us": statistics.fmean(samples_us),
                    "p95_us": _percentile(samples_us, 0.95),
                    "min_us": min(samples_us),
                    "logical_kv_read_mib": logical_read_bytes / (1024**2),
                    "logical_read_gib_s_at_median": (
                        logical_read_bytes / (1024**3) / (median_us / 1_000_000.0)
                    ),
                    "repeats": args.repeats,
                }
                rows.append(row)
                print(
                    "[cpu-correction] "
                    f"B={batch} k={top_k} threads={threads} "
                    f"median={median_us:.1f}us p95={row['p95_us']:.1f}us",
                    flush=True,
                )

    ext.set_cpu_attn_threads(0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "model_shape": shape,
        "context_slots": num_slots,
        "shadow_bytes": (k_shadow.nbytes + v_shadow.nbytes),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "randomized_slots_per_call": True,
        "note": (
            "Production CPU correction kernel only; excludes GPU hand-off, "
            "CUDA graph scheduling, and Stage-B overlap."
        ),
    }
    with (out_dir / "cpu_correction_rows.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)
    with (out_dir / "cpu_correction_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), default="llama31_8b")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--topks", default="32,64,128,256,512")
    parser.add_argument("--threads", default="8,16,32,64")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
