"""JIT loader for the Pattern A fused INT2 attention + topK kernel.

Single-call replacement for ``int2_qk_scores_topk`` + ``int2_softmax_v_with_pre_corr``:

    o_hat, topk_slots, o_old_unnorm, W = ext.int2_fused_attn_topk(
        q, k_data, k_scales, k_zp,
        v_data, v_scales,
        block_table, seq_lens,
        max_S, H_q, H_kv, block_size, top_k, attn_scale,
    )

Streams K and V once each per (b, h_q); no (B, H_q, max_S) scores or
weights tensor materialised in HBM.  See
``jetfire_vllm/csrc/int2_fused_attn_topk.cu`` for the algorithm.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401  — required for the cpp_extension JIT loader

_EXT = None
_LOCK = threading.Lock()


def load_int2_fused_attn_topk_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int2_fused_attn_topk.cu")
        _EXT = load(
            name="jetfire_int2_fused_attn_topk",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_FUSED_ATTN_TOPK_VERBOSE", "0") == "1",
        )
        return _EXT
