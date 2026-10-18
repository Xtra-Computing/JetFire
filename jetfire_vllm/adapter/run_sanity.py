#!/usr/bin/env python3
"""Fast release sanity checks for the JetFire release snapshot.

This script intentionally avoids launching vLLM engines or compiling CUDA
extensions by default.  It verifies that the snapshot is importable, the release
configuration resolves, and the documented dtype matrix has dispatch entries.
Use ``python -m jetfire_vllm.artifact kernels`` for CUDA kernel validation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


TARGET_MODES = (
    "SymInt4K_SymInt4V",
    "SymInt4K_SymInt3V",
    "SymInt4K_SymInt2V",
    "SymInt3K_SymInt4V",
    "SymInt3K_SymInt3V",
    "SymInt3K_SymInt2V",
    "SymInt2K_SymInt4V",
    "SymInt2K_SymInt3V",
    "SymInt2K_SymInt2V",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    return str(getattr(mod, "__version__", "unknown"))


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-vllm",
        action="store_true",
        help="Fail if vLLM cannot be imported.",
    )
    args = parser.parse_args()

    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    print(f"[sanity] root={root}")
    print(f"[sanity] python={sys.executable}")
    print(f"[sanity] platform={platform.platform()}")
    print(f"[sanity] torch={_version('torch')}")
    print(f"[sanity] triton={_version('triton')}")
    vllm_version = _version("vllm")
    print(f"[sanity] vllm={vllm_version}")
    if not _module_available("torch"):
        print("[sanity][error] torch is required; install requirements.txt")
        return 2
    if not _module_available("triton"):
        print("[sanity][error] triton is required; install requirements.txt")
        return 2
    if args.require_vllm and vllm_version.startswith("unavailable"):
        return 2

    from jetfire_vllm.adapter.config import (
        QUANT_SCHEMES,
        get_jetfire_config,
        reset_jetfire_config,
    )
    from jetfire_vllm.adapter.release_policy import plan_release_graph
    from jetfire_vllm.kernels._dispatch import get_kernel_set

    missing = [mode for mode in TARGET_MODES if mode not in QUANT_SCHEMES]
    if missing:
        print(f"[sanity][error] target modes missing from config: {missing}")
        return 1

    original_mode = os.environ.get("JETFIRE_MODE")
    try:
        os.environ["JETFIRE_MODE"] = "SymInt3K_SymInt2V"
        os.environ.setdefault("JETFIRE_GROUP_SIZE", "64")
        os.environ.setdefault("JETFIRE_TOPK", "256")
        reset_jetfire_config()
        cfg = get_jetfire_config()
        print(
            "[sanity] release_config="
            f"mode={cfg.scheme.name} topk={cfg.topk} "
            f"cpu_correction={cfg.cpu_correction} "
            f"decode_cpu_attn={cfg.decode_cpu_attn}"
        )

        for mode in TARGET_MODES:
            ks = get_kernel_set(mode)
            print(
                "[sanity] mode="
                f"{mode} K={ks.nbits_k} V={ks.nbits_v} "
                f"topk={ks.supported_top_k}"
            )
        plan = plan_release_graph(
            mode="SymInt3K_SymInt2V",
            group_size=64,
            top_k=256,
            max_graph_batch=1,
        )
        print(
            "[sanity] release_policy="
            f"mode={plan.mode} stage_a={plan.stage_a} "
            f"stage_b={plan.stage_b} state={plan.promotion_state} "
            f"ready={plan.release_ready}"
        )
        for blocker in plan.blockers:
            print(f"[sanity][policy-blocker] {blocker}")
    finally:
        if original_mode is None:
            os.environ.pop("JETFIRE_MODE", None)
        else:
            os.environ["JETFIRE_MODE"] = original_mode
        reset_jetfire_config()

    print("[sanity] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
