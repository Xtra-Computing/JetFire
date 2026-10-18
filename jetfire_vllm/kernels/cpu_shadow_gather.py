"""JIT loader for the C++ CPU shadow gather extension.

The extension exposes `gather_kv_topk(k_shadow, v_shadow, head_map, ts_buf,
k_out, v_out)` and releases the GIL during the parallel memcpy so a worker
thread can actually overlap with main-thread CUDA work.  See ../csrc/
cpu_shadow_gather.cpp for the full design notes.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401  — required for the cpp_extension JIT loader

_EXT = None
_LOCK = threading.Lock()


def load_cpu_shadow_gather_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "cpu_shadow_gather.cpp")
        _EXT = load(
            name="jetfire_cpu_shadow_gather",
            sources=[src],
            extra_cflags=[
                "-O3", "-std=c++17",
                "-fopenmp",
                "-march=native",     # AVX-512 / AVX2 memcpy where available
            ],
            extra_ldflags=["-fopenmp"],
            verbose=os.environ.get("JETFIRE_GATHER_VERBOSE", "0") == "1",
        )
        return _EXT
