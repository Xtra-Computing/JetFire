"""JIT loader for the fused_post_correction CUDA extension."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401

_EXT = None
_LOCK = threading.Lock()


def load_fused_correction_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "fused_correction.cu")
        _EXT = load(
            name="jetfire_fused_correction",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
            ],
            verbose=os.environ.get("JETFIRE_FUSED_CORRECTION_VERBOSE", "0") == "1",
        )
        return _EXT
