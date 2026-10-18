"""JIT loader for the JetFire blend kernel.

Replaces the V2_05_post_corr Python op chain
    o_new_f = o_new_gpu.to(fp32)
    o_corr  = o_hat + W.unsqueeze(-1) * (o_new_f - o_old)
    output[:n_seqs].copy_(o_corr.to(bf16))
which compiles to ~5–7 separate kernel launches, with a single fused
kernel:
    blend_into(o_hat, o_old, W, o_new, out)

Validated by ``unit_test/test_blend_kernel.py`` (10/10 PASS, max rel err
< 5e-3 within bf16 precision).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch  # noqa: F401

_EXT = None
_LOCK = threading.Lock()


def load_blend_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load

        here = Path(__file__).resolve().parent.parent
        src = str(here / "csrc" / "blend.cu")
        if os.environ.get("JETFIRE_BLEND_TRACE", "0") == "1":
            print(
                f"[jetfire][blend] torch_load_enter name=jetfire_blend src={src}",
                flush=True,
            )
        _EXT = load(
            name="jetfire_blend",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=os.environ.get("JETFIRE_BLEND_VERBOSE", "0") == "1",
        )
        if os.environ.get("JETFIRE_BLEND_TRACE", "0") == "1":
            print("[jetfire][blend] torch_load_done", flush=True)
        return _EXT
