"""JIT-build + load the JetFire vLLM CUDA kernels.

Cached & thread-safe.  Returns None on build failure so callers can fall
back to the Triton path (`fused_qk` + `F.softmax` + `fused_vscan` in
`decode_path.py`).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

_wmma_lock = threading.Lock()
_wmma_mod = None
_wmma_loaded = False


def _csrc_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "csrc"))


def _cuda_cflags() -> list[str]:
    return [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
    ]


def _extension_build_active(name: str, build_dir: str) -> bool:
    """Return whether a local process appears to be building ``name``."""

    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return False
    normalized_build_dir = os.path.realpath(build_dir)
    for entry in os.scandir(proc_root):
        if not entry.name.isdigit():
            continue
        proc_dir = entry.path
        try:
            if (
                os.path.realpath(os.readlink(os.path.join(proc_dir, "cwd")))
                == normalized_build_dir
            ):
                return True
        except OSError:
            pass
        try:
            with open(os.path.join(proc_dir, "cmdline"), "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
            if name in cmdline or normalized_build_dir in cmdline:
                return True
        except OSError:
            pass
    return False


def _remove_stale_lock_if_prebuilt(name: str, verbose: bool = False) -> None:
    """Remove a stale torch-extension lock when a built .so already exists.

    torch.utils.cpp_extension uses a bare ``lock`` file as its inter-process
    build baton.  If a benchmark process is killed while holding it, later
    runtime workers can wait forever even though the extension is already
    compiled.  Only remove old locks when a matching shared object exists.
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory

        build_dir = _get_build_directory(name, verbose=False)
        lock_path = os.path.join(build_dir, "lock")
        so_path = os.path.join(build_dir, f"{name}.so")
        if not (os.path.exists(lock_path) and os.path.exists(so_path)):
            return
        max_age_s = float(os.environ.get("JETFIRE_STALE_EXT_LOCK_S", "30"))
        age_s = time.time() - os.path.getmtime(lock_path)
        if age_s < max_age_s:
            return
        if _extension_build_active(name, build_dir):
            return
        os.remove(lock_path)
        if verbose or os.environ.get("JETFIRE_EXT_LOCK_VERBOSE", "0") == "1":
            print(
                f"[jetfire_vllm] removed stale extension lock: {lock_path}",
                flush=True,
            )
    except Exception as exc:
        if verbose or os.environ.get("JETFIRE_EXT_LOCK_VERBOSE", "0") == "1":
            print(f"[jetfire_vllm] stale-lock cleanup skipped: {exc}", flush=True)


def load_wmma_ext(verbose: bool = False) -> Optional[object]:
    """Build/load the WMMA decode extension.

    Exposes `fused_qk_wmma_bf16(q, k_bf16, block_table, seq_lens, max_S,
    H_q, H_kv, block_size, attn_scale)` → scores (B, H_q, max_S) fp32.
    """
    global _wmma_mod, _wmma_loaded
    if _wmma_loaded:
        return _wmma_mod
    with _wmma_lock:
        if _wmma_loaded:
            return _wmma_mod
        _wmma_loaded = True
        try:
            from torch.utils.cpp_extension import load
        except ImportError:
            return None

        src = os.path.join(_csrc_dir(), "fused_decode_wmma.cu")
        if not os.path.exists(src):
            return None

        try:
            _remove_stale_lock_if_prebuilt("jetfire_vllm_wmma_ext", verbose=verbose)
            _wmma_mod = load(
                name="jetfire_vllm_wmma_ext",
                sources=[src],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=_cuda_cflags(),
                verbose=verbose,
            )
        except Exception as e:
            if verbose or os.environ.get("JETFIRE_WMMA_EXT_VERBOSE", "0") == "1":
                print(f"[jetfire_vllm] wmma ext load failed: {e}")
            _wmma_mod = None
        return _wmma_mod
