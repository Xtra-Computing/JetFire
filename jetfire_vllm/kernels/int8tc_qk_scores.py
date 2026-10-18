"""JIT loader for the JetFire Int8 tensor-core Q·K^T kernel.

This is an alternative Stage A to int4_qk_scores that runs the mma on
H100 INT8 tensor cores (1979 TFLOPS peak, vs 989 TFLOPS BF16).

The on-disk KV layout is unchanged (INT4 packed bytes, BF16 k_scales).
The kernel quantizes Q on the fly to Int8 with per-row scalar scale.

Exposes:
    load_int8tc_qk_scores_ext() -> module with:
        int8tc_qk_scores(q, k_data, k_scales, block_table, seq_lens,
                          max_S, H_q, H_kv, block_size, attn_scale)
            -> scores: (B, H_q, max_S) float32
        int8tc_qk_scores_topk(..., top_k)
            -> [scores, topk_idx, topk_slots]
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401

_EXT = None
_LOCK = threading.Lock()


def load_int8tc_qk_scores_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int8tc_qk_scores.cu")
        _EXT = load(
            name="jetfire_int8tc_qk_scores",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                # Force sm_90 so the m16n8k32.s8 PTX is accepted.
                "-gencode=arch=compute_90,code=sm_90",
            ],
            verbose=os.environ.get("JETFIRE_SPLIT_VERBOSE", "0") == "1",
        )
        return _EXT
