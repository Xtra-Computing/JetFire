"""JIT loader for the JetFire INT2 attention split kernels.

Two kernels:
  * int2_qk_scores  — Q·K^T → scores                    (Stage A)
  * int2_softmax_v  — softmax(scores) · V_int → o_hat   (Stage B)

Together they're equivalent to ``fused_decode_wmma_v5_int2_with_scores``.
The split lets Stage B run in parallel with CPU's ``cpu_topk_attn``.

Validated by ``unit_test_split/test_split_e2e.py`` (50/50 PASS).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401

from jetfire_vllm.adapter._gpu_ext import _remove_stale_lock_if_prebuilt

_EXT_QK = None
_EXT_SV = None
_LOCK_QK = threading.Lock()
_LOCK_SV = threading.Lock()


def load_qk_scores_ext():
    global _EXT_QK
    if _EXT_QK is not None:
        return _EXT_QK
    with _LOCK_QK:
        if _EXT_QK is not None:
            return _EXT_QK
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int2_qk_scores.cu")
        _remove_stale_lock_if_prebuilt("jetfire_int2_qk_scores")
        _EXT_QK = load(
            name="jetfire_int2_qk_scores",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_SPLIT_VERBOSE", "0") == "1",
        )
        return _EXT_QK


def load_softmax_v_ext():
    global _EXT_SV
    if _EXT_SV is not None:
        return _EXT_SV
    with _LOCK_SV:
        if _EXT_SV is not None:
            return _EXT_SV
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "int2_softmax_v.cu")
        _remove_stale_lock_if_prebuilt("jetfire_int2_softmax_v")
        _EXT_SV = load(
            name="jetfire_int2_softmax_v",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_SPLIT_VERBOSE", "0") == "1",
        )
        return _EXT_SV
