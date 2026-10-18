#!/usr/bin/env python3
"""Measure pinned host/device transfer latency and bandwidth."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _measure_copy(
    *,
    destination: torch.Tensor,
    source: torch.Tensor,
    repeats: int,
    warmups: int,
) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            destination.copy_(source, non_blocking=True)
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(repeats):
            destination.copy_(source, non_blocking=True)
        end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1_000.0 / repeats


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sizes = _csv_ints(args.sizes_bytes)
    rows: list[dict[str, Any]] = []
    for size_bytes in sizes:
        repeats = max(
            args.min_repeats,
            min(args.max_repeats, args.target_bytes // size_bytes),
        )
        host = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        direction_samples: dict[str, list[float]] = {"h2d": [], "d2h": []}
        for _ in range(args.trials):
            direction_samples["h2d"].append(
                _measure_copy(
                    destination=device,
                    source=host,
                    repeats=repeats,
                    warmups=args.warmups,
                )
            )
            direction_samples["d2h"].append(
                _measure_copy(
                    destination=host,
                    source=device,
                    repeats=repeats,
                    warmups=args.warmups,
                )
            )
        for direction, samples_us in direction_samples.items():
            median_us = statistics.median(samples_us)
            rows.append(
                {
                    "direction": direction,
                    "size_bytes": size_bytes,
                    "repeats_per_trial": repeats,
                    "trials": args.trials,
                    "median_us": median_us,
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "bandwidth_gb_s": size_bytes / (median_us / 1_000_000.0) / 1e9,
                }
            )
            print(
                f"[pcie] {direction} bytes={size_bytes} "
                f"median={median_us:.2f}us "
                f"bandwidth={rows[-1]['bandwidth_gb_s']:.2f}GB/s",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "method": (
                "Repeated non-blocking copies between CUDA device memory and "
                "torch pinned host memory, timed with CUDA events."
            ),
        },
        "rows": rows,
    }
    with (out_dir / "pcie_transfer_rows.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (out_dir / "pcie_transfer_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-bytes",
        default="4096,16384,65536,262144,1048576,4194304,16777216,67108864",
    )
    parser.add_argument("--target-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--min-repeats", type=int, default=20)
    parser.add_argument("--max-repeats", type=int, default=5000)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
