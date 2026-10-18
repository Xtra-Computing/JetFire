"""JIT loader for the Pattern A v1 (WMMA + cp.async) fused kernel.

Same outputs as v0 (correctness foundation in
``int2_fused_attn_topk.cu``) but uses WMMA bf16 mma_sync for Q·K and
P·V, cp.async double-buffered SMEM for K/V loads, and a per-warp
register-distributed top-K heap.

Constraints (validated at host side):
  D == 128
  block_size == 16
  top_k ∈ {32, 64, 96, 128}
  num_groups (= H_q / H_kv) ∈ {1, 2, 4}
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401  — required for the cpp_extension JIT loader

_EXT = None
_LOCK = threading.Lock()


def load_int2_fused_attn_topk_v1_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int2_fused_attn_topk_v1.cu")
        _EXT = load(
            name="jetfire_int2_fused_attn_topk_v1",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_FUSED_ATTN_TOPK_VERBOSE", "0") == "1",
        )
        return _EXT
