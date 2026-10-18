#!/usr/bin/env python3
"""Single-process JetFire extension preload for benchmark launchers.

TP workers can otherwise race in ``torch.utils.cpp_extension.load`` and wait
on stale ``lock`` files.  This script compiles/loads the JetFire decode
extensions once per requested mode before vLLM worker processes are spawned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _same_path(path: str, target: Path) -> bool:
    try:
        return Path(path).resolve() == target.resolve()
    except Exception:
        return path == str(target)


def _pythonpath() -> str:
    existing = []
    if os.environ.get("JETFIRE_ALLOW_INHERITED_PYTHONPATH", "0") == "1":
        blocked = [
            ROOT.parent,
            ROOT.parent / "jetfire_vllm" / "adapter",
        ]
        existing = [
            p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if p and not any(_same_path(p, blocked_path) for blocked_path in blocked)
        ]
    paths = [str(ROOT / "jetfire_vllm" / "adapter"), str(ROOT), *existing]
    merged: list[str] = []
    for path in paths:
        if path not in merged:
            merged.append(path)
    return os.pathsep.join(merged)


def _run_child(
    mode: str,
    top_k: int,
    group_size: int,
    cpu_attn_threads: int,
    int8tc: str,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["JETFIRE_EXPECTED_ROOT"] = str(ROOT)
    env["JETFIRE_USE_CUSTOM_BACKEND"] = "1"
    env["JETFIRE_MODE"] = mode
    env["JETFIRE_TOPK"] = str(top_k)
    env["JETFIRE_GROUP_SIZE"] = str(group_size)
    env["JETFIRE_CPU_ATTN_THREADS"] = str(cpu_attn_threads)
    env.setdefault("OMP_NUM_THREADS", str(cpu_attn_threads))
    env.setdefault("JETFIRE_CPU_CORRECTION", "1")
    env.setdefault("JETFIRE_DECODE_CPU_ATTN", "1")
    env.setdefault("JETFIRE_SKIP_NATIVE_KV", "1")
    env.setdefault("JETFIRE_PRELOAD_DECODE_EXTENSIONS", "1")
    if int8tc == "on":
        env["JETFIRE_USE_INT8_TC"] = "1"
        env.pop("JETFIRE_USE_INT8_TC_SV", None)
    elif int8tc == "off":
        env["JETFIRE_USE_INT8_TC"] = "0"
        env["JETFIRE_USE_INT8_TC_SV"] = "0"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode",
        mode,
        "--top-k",
        str(top_k),
        "--group-size",
        str(group_size),
        "--cpu-attn-threads",
        str(cpu_attn_threads),
        "--int8tc",
        int8tc,
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def _child_main(
    mode: str,
    top_k: int,
    group_size: int,
    cpu_attn_threads: int,
    int8tc: str,
) -> int:
    os.environ["JETFIRE_USE_CUSTOM_BACKEND"] = "1"
    os.environ["JETFIRE_MODE"] = mode
    os.environ["JETFIRE_TOPK"] = str(top_k)
    os.environ["JETFIRE_GROUP_SIZE"] = str(group_size)
    os.environ["JETFIRE_CPU_ATTN_THREADS"] = str(cpu_attn_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_attn_threads))
    os.environ.setdefault("JETFIRE_CPU_CORRECTION", "1")
    os.environ.setdefault("JETFIRE_DECODE_CPU_ATTN", "1")
    os.environ.setdefault("JETFIRE_SKIP_NATIVE_KV", "1")
    if int8tc == "on":
        os.environ["JETFIRE_USE_INT8_TC"] = "1"
        os.environ.pop("JETFIRE_USE_INT8_TC_SV", None)
    elif int8tc == "off":
        os.environ["JETFIRE_USE_INT8_TC"] = "0"
        os.environ["JETFIRE_USE_INT8_TC_SV"] = "0"

    import jetfire_vllm
    from jetfire_vllm.adapter.config import reset_jetfire_config
    from jetfire_vllm.adapter.decode_path import preload_decode_extensions

    pkg = Path(jetfire_vllm.__file__).resolve()
    if ROOT not in pkg.parents:
        raise RuntimeError(f"wrong jetfire_vllm package: {pkg}; expected {ROOT}")

    reset_jetfire_config()
    print(
        f"[jetfire-precompile] package={pkg} mode={mode} "
        f"top_k={top_k} group_size={group_size} int8tc={int8tc}",
        flush=True,
    )
    print(f"[jetfire-precompile] preload start mode={mode}", flush=True)
    preload_decode_extensions()
    print(f"[jetfire-precompile] preload done mode={mode}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompile JetFire JIT extensions.")
    parser.add_argument(
        "--modes",
        default=(
            "SymInt4K_SymInt4V,"
            "SymInt4K_SymInt3V,"
            "SymInt4K_SymInt2V,"
            "SymInt3K_SymInt4V,"
            "SymInt3K_SymInt3V,"
            "SymInt3K_SymInt2V,"
            "SymInt2K_SymInt4V,"
            "SymInt2K_SymInt3V,"
            "SymInt2K_SymInt2V"
        ),
        help="Comma-separated JETFIRE_MODE values to preload.",
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--cpu-attn-threads", type=int, default=32)
    parser.add_argument(
        "--int8tc",
        choices=("auto", "on", "off"),
        default="auto",
        help="Precompile INT8 tensor-core path: auto keeps defaults.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        if not args.mode:
            raise SystemExit("--child requires --mode")
        return _child_main(
            str(args.mode),
            int(args.top_k),
            int(args.group_size),
            int(args.cpu_attn_threads),
            str(args.int8tc),
        )

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    for mode in modes:
        _run_child(
            mode,
            int(args.top_k),
            int(args.group_size),
            int(args.cpu_attn_threads),
            str(args.int8tc),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
