"""JIT loader for the JetFire Phase 0 filler kernel.

The filler kernel occupies the current CUDA stream for a configurable
number of microseconds.  Used to quantify how much extra GPU work we can
inject into the decode pipeline's CPU-overlap window before throughput
starts to drop.

Used by ``decode_path.py`` when ``JETFIRE_FILLER_US > 0``.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401

_EXT = None
_LOCK = threading.Lock()


def load_filler_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load

        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "filler_kernel.cu")
        _EXT = load(
            name="jetfire_filler",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_FILLER_VERBOSE", "0") == "1",
        )
        return _EXT
