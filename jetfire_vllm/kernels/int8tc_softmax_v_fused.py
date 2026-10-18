"""JIT loader for the fused INT8 tensor-core Stage B kernel.

Single-launch replacement for the
    softmax_prep + weighted_v_split_* + 3 zero-fills
pipeline that the INT4 Stage B currently runs through.

Provides:
    int8tc_softmax_v_fused(scores, seq_lens, v_data, v_scales, block_table,
                           max_S, H_q, H_kv, block_size) -> o_hat
    int8tc_softmax_v_fused_inplace(...)

    # pre_corr drop-in for int4_softmax_v_with_pre_corr_wvs2 (returns
    # [o_hat, o_old_unnorm, W]):
    int8tc_softmax_v_with_pre_corr(
        scores, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k,
    ) -> [o_hat, o_old_unnorm, W]
    int8tc_softmax_v_with_pre_corr_inplace(...)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401


_EXT = None
_LOCK = threading.Lock()


def load_ext():
    """JIT-compile / cache the int8tc_softmax_v_fused extension."""
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int8tc_softmax_v_fused.cu")
        # SM 9.0 (H100): -arch=sm_90a enables s8/s4 MMA + cp.async.bulk.
        _EXT = load(
            name="jetfire_int8tc_softmax_v_fused",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--use_fast_math",
            ],
            verbose=os.environ.get("JETFIRE_INT8TC_VERBOSE", "0") == "1",
        )
        return _EXT
