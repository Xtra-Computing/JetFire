"""JIT loader for the C++ CPU top-K attention extension.

The extension exposes ``cpu_topk_attn(q, k_shadow, v_shadow, head_map,
ts_buf, o_new, scale)`` which gathers top-K K/V from the shadow, runs
``softmax(Q·K^T / sqrt(D))·V``, and writes the result into ``o_new``.
It releases the GIL so a worker thread can run concurrently with main-
thread CUDA op issue.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401  — required for the cpp_extension JIT loader

_EXT = None
_LOCK = threading.Lock()


def load_cpu_topk_attn_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load
        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "cpu_topk_attn.cpp")
        # Need CUDA headers for cudaHostAlloc / cudaHostGetDevicePointer used
        # by the alloc_mapped_pinned_pair helper.  Link against cudart for
        # the runtime symbols.
        from torch.utils.cpp_extension import _find_cuda_home
        cuda_home = _find_cuda_home()
        cuda_include = f"{cuda_home}/include" if cuda_home else "/usr/local/cuda/include"
        cuda_lib     = f"{cuda_home}/lib64"   if cuda_home else "/usr/local/cuda/lib64"
        _EXT = load(
            name="jetfire_cpu_topk_attn",
            sources=[src],
            extra_cflags=[
                "-O3", "-std=c++17",
                "-fopenmp",
                "-march=native",     # AVX-512 / AVX-512-BF16 on EPYC 9654
                "-ffast-math",
                f"-I{cuda_include}",
            ],
            extra_ldflags=["-fopenmp", f"-L{cuda_lib}", "-lcudart", "-lcuda"],
            verbose=os.environ.get("JETFIRE_TOPK_ATTN_VERBOSE", "0") == "1",
        )
        return _EXT
