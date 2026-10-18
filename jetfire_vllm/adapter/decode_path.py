"""JetFire vLLM decode/prefill path.

Prefill: unchanged flash-attn path; shadow KV is mirrored via hooks in
        ``cache_store``.

Decode (split correction blend):
        split Stage A QK+topK                              → scores, topk slots
        async D2H(q, topk_slots) on comm stream            ─┐
        worker thread (GIL-released C++):                   │   overlap with main
            CPU shadow gather → pinned bf16                  │   stream's Stage B
            H2D copy on comm stream                         ─┘
        split Stage B softmax(V)+pre-correction            → o_hat, o_old, W
        join worker; wait comm                             (sync point)
        correction blend                                   → o_corrected

The kernel_set is dispatched from ``JETFIRE_MODE`` (see
:mod:`jetfire_vllm.adapter.config`).  The split path covers the release dtype
matrix: INT4/INT3/INT2 K and V combinations, including mixed K/V bit-widths.

Decode falls back to native flash-attn when:
  * ``cpu_correction=False`` (mirror-only ablation)
  * profiling forward (no attn_metadata)
  * ``max_query_len > 1`` (chunked prefill / cascaded request)
"""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from concurrent.futures import TimeoutError, ThreadPoolExecutor

import torch

from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func

from .config import apply_graph_cpu_topk_release_preset, get_jetfire_config
from .kv_pool import get_global_kv_pool_manager
from .metadata_trace import record_runtime_metadata
from .release_policy import (
    ReleaseKernelPlan,
    plan_release_graph,
    release_policy_enabled,
    release_require_ready,
    validate_release_environment,
)


# ── Profiling ──────────────────────────────────────────────────────────
# When JetFireConfig.profile_decode is True, each named stage is wrapped
# with cuda.synchronize + perf_counter.  Adds ~10× cuda.sync per layer per
# token, so the wall-clock total is pessimistic — use only for stage
# attribution, not end-to-end TPOT.
_PROFILE = get_jetfire_config().profile_decode
_PROFILE_ACCUM: dict[str, list[float]] = {}
_TRACE_DECODE_ID = 0
_STEP_TRACE_FH = None


@contextmanager
def _timer(name: str):
    # Always emit NVTX range so nsys can see stage boundaries on the
    # GPU/CPU timeline (even when JETFIRE_PROFILE_DECODE=0 and we don't
    # add cuda.synchronize overhead).  Negligible cost when nsys isn't
    # capturing.
    torch.cuda.nvtx.range_push(name)
    try:
        if not _PROFILE:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            _PROFILE_ACCUM.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
    finally:
        torch.cuda.nvtx.range_pop()


def jetfire_get_profile() -> dict[str, list[float]]:
    out = {k: list(v) for k, v in _PROFILE_ACCUM.items()}
    _PROFILE_ACCUM.clear()
    return out


def _next_trace_decode_id() -> int:
    global _TRACE_DECODE_ID
    _TRACE_DECODE_ID += 1
    return _TRACE_DECODE_ID


def _stage_trace(
    trace_id: int,
    stage: str,
    *,
    layer_name: str,
    n_seqs: int | None = None,
    max_S: int | None = None,
    top_k: int | None = None,
    extra: str = "",
) -> None:
    if os.environ.get("JETFIRE_STAGE_TRACE", "0") != "1":
        return
    every = max(1, int(os.environ.get("JETFIRE_STAGE_TRACE_EVERY", "1")))
    if trace_id % every != 0:
        return
    print(
        "[jetfire][stage] "
        f"id={trace_id} stage={stage} layer={layer_name} "
        f"B={n_seqs if n_seqs is not None else '-'} "
        f"S={max_S if max_S is not None else '-'} "
        f"K={top_k if top_k is not None else '-'} "
        f"t={time.time():.6f}"
        + (f" {extra}" if extra else ""),
        flush=True,
    )


def _get_dist_rank() -> str:
    rank = os.environ.get("RANK")
    if rank is not None:
        return rank
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return "?"


def _step_trace(
    trace_id: int,
    stage: str,
    *,
    layer_name: str,
    n_seqs: int | None = None,
    max_S: int | None = None,
    top_k: int | None = None,
    extra: str = "",
) -> None:
    """Low-volume decode heartbeat for timing-sensitive TP hangs."""
    if os.environ.get("JETFIRE_STEP_TRACE", "0") != "1":
        return
    every = max(1, int(os.environ.get("JETFIRE_STEP_TRACE_EVERY", "1")))
    if trace_id % every != 0:
        return
    stages = os.environ.get("JETFIRE_STEP_TRACE_STAGES", "")
    if stages:
        allowed = {s.strip() for s in stages.split(",") if s.strip()}
        if stage not in allowed:
            return

    try:
        cuda_dev = torch.cuda.current_device()
    except Exception:
        cuda_dev = "?"
    line = (
        "[jetfire][step] "
        f"id={trace_id} rank={_get_dist_rank()} pid={os.getpid()} "
        f"cuda={cuda_dev} stage={stage} layer={layer_name} "
        f"B={n_seqs if n_seqs is not None else '-'} "
        f"S={max_S if max_S is not None else '-'} "
        f"K={top_k if top_k is not None else '-'} "
        f"t={time.time():.6f}"
        + (f" {extra}" if extra else "")
    )
    path = os.environ.get("JETFIRE_STEP_TRACE_FILE", "")
    if not path:
        print(line, flush=True)
        return

    global _STEP_TRACE_FH
    if _STEP_TRACE_FH is None:
        trace_path = f"{path}.pid{os.getpid()}"
        _STEP_TRACE_FH = open(trace_path, "a", buffering=1)
    _STEP_TRACE_FH.write(line + "\n")


# ── Module-level resources ─────────────────────────────────────────────
# All caches keyed by shape; built lazily on first use, then reused.

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_HEAD_MAP_CPU: dict[tuple[int, int, int, int], torch.Tensor] = {}
_PINNED_BUFS:  dict[tuple[tuple, torch.dtype, str], torch.Tensor] = {}
_DEVICE_BUFS:  dict[tuple[tuple, torch.dtype, str, str], torch.Tensor] = {}
_COMM_STREAM: torch.cuda.Stream | None = None
_GATHER_EXEC: ThreadPoolExecutor | None = None

# ── Graph-mode CPU submit machinery ────────────────────────────────────
# When ``JETFIRE_DECODE_GRAPH=1`` the decode path uses cudaLaunchHostFunc
# plus stream memory ops so the CPU/GPU handshake can be captured into a
# CUDA graph.  The release completion path is async submit + side-stream event
# join: the submit host node dispatches CPU topK work and returns before Stage
# B, and the later event wait happens only at the correction boundary.  The
# device-signal, host join, and mapped-host wait remain available for explicit
# A/B work, but are not release defaults because long sweeps have exposed TP
# rank stalls in those paths.
#
# The submit-ctx encapsulates (q_buf, ts_buf, head_map, k_shadow, v_shadow,
# o_new_pin, scale, flag_host) for one (layer, top_k, n_seqs) combo and is
# cached for the process lifetime.  The flag is a shared mapped-pinned int64
# pair (one per process); the release side-join path keeps it only as a
# host-visible diagnostic flag, while legacy A/B modes still reset/wait it
# through stream ops.
_GRAPH_FLAG_HOST: torch.Tensor | None = None
_GRAPH_FLAG_DEV:  torch.Tensor | None = None
# key: device string -> CUDA int64 doorbell tensor.
_GRAPH_SIGNAL_DEV: dict[str, torch.Tensor] = {}
# key: (store object, top_k, n_seqs, buf_B, mapped_o_new, signal mode) → submit_ctx int
# handle (opaque).  Include buf_B/mapped_o_new so a later run in the same
# process cannot accidentally reuse a context that points at differently-shaped
# or differently-owned o_new storage.
_GRAPH_SUBMIT_CTX: dict[tuple, int] = {}

# Resolved on first decode hit, cached thereafter.
_KERNEL_SET = None       # KernelSet for the active scheme
_EXT_GATHER = None       # CPU shadow gather (C++, GIL-released; scheme-independent)

# Tracks warmup_decode_buffers by exact buffer shape.  A single bool is not
# enough because release top-k sweeps can warm a smaller K before CUDA graph
# capture later needs a larger K in the same worker process.
_BUFFERS_WARMED_UP = False
_BUFFER_WARMUP_KEYS: set[tuple[int, int, int, int, int, str]] = set()
_WARMUP_OVERFLOW_LOGGED = False


def _debug_graph_init(message: str) -> None:
    if os.environ.get("JETFIRE_DEBUG_GRAPH_INIT", "0") != "1":
        return
    rank = os.environ.get("RANK")
    if rank is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = str(dist.get_rank())
        except Exception:
            rank = "?"
    print(
        f"[jetfire][graph-init][rank={rank if rank is not None else '?'}] "
        f"{message} t={time.time():.6f}",
        flush=True,
    )


def _load_kernel_set():
    """Resolve and load the scheme's KernelSet on first use."""
    global _KERNEL_SET
    if _KERNEL_SET is None:
        from jetfire_vllm.kernels._dispatch import get_kernel_set
        ks = get_kernel_set(get_jetfire_config().scheme.name)
        ks.load()
        _KERNEL_SET = ks
    return _KERNEL_SET


def _load_gather_ext():
    global _EXT_GATHER
    if _EXT_GATHER is None:
        from jetfire_vllm.kernels.cpu_shadow_gather import load_cpu_shadow_gather_ext
        _EXT_GATHER = load_cpu_shadow_gather_ext()
    return _EXT_GATHER


_EXT_CPU_TOPK_ATTN = None
_EXT_FILLER = None
_EXT_BLEND = None
_EXT_QK_SCORES = None
_EXT_SOFTMAX_V = None
_EXT_FUSED_ATTN_TOPK    = None
_EXT_FUSED_ATTN_TOPK_V1 = None


def _load_fused_attn_topk_ext():
    """Lazy-load the v0 (correctness-first, scalar) fused kernel.

    v0 stays as the reference implementation; v1 is the WMMA + cp.async
    optimised version that should replace it in production once
    end-to-end perf is confirmed.
    """
    global _EXT_FUSED_ATTN_TOPK
    if _EXT_FUSED_ATTN_TOPK is None:
        from jetfire_vllm.kernels.int2_fused_attn_topk import (
            load_int2_fused_attn_topk_ext,
        )
        _EXT_FUSED_ATTN_TOPK = load_int2_fused_attn_topk_ext()
    return _EXT_FUSED_ATTN_TOPK


def _load_fused_attn_topk_v1_ext():
    """Lazy-load the v1 (WMMA + cp.async) fused kernel.

    Same outputs as v0 but uses WMMA bf16 mma_sync for Q·K and P·V,
    cp.async double-buffered SMEM, per-warp register-distributed top-K
    heap.  Constraints: D=128, block_size=16, top_k % 32 == 0,
    num_groups (= H_q / H_kv) in {1, 2, 4}.  Default is v1 when
    ``JETFIRE_USE_FUSED_ATTN_TOPK=1``; set ``JETFIRE_FUSED_KERNEL=v0``
    to fall back to the scalar reference.
    """
    global _EXT_FUSED_ATTN_TOPK_V1
    if _EXT_FUSED_ATTN_TOPK_V1 is None:
        from jetfire_vllm.kernels.int2_fused_attn_topk_v1 import (
            load_int2_fused_attn_topk_v1_ext,
        )
        _EXT_FUSED_ATTN_TOPK_V1 = load_int2_fused_attn_topk_v1_ext()
    return _EXT_FUSED_ATTN_TOPK_V1


def _load_qk_scores_ext():
    """Lazy-load the Stage A (Q·K → scores) split kernel.

    Dispatches by scheme:
      INT2 → ``int2_split``, INT3 → ``int3_split``, INT4 → ``int4_split``.
    """
    global _EXT_QK_SCORES
    if _EXT_QK_SCORES is None:
        nbits = get_jetfire_config().scheme.k_nbits
        if nbits == 4:
            from jetfire_vllm.kernels.int4_split import load_qk_scores_ext
        elif nbits == 3:
            from jetfire_vllm.kernels.int3_split import load_qk_scores_ext
        else:
            from jetfire_vllm.kernels.int2_split import load_qk_scores_ext
        _EXT_QK_SCORES = load_qk_scores_ext()
    return _EXT_QK_SCORES


def _load_softmax_v_ext():
    """Lazy-load the Stage B (softmax(scores)·V → o_hat) split kernel.

    Dispatches by scheme:
      INT2 → ``int2_split``, INT3 → ``int3_split``, INT4 → ``int4_split``.
    """
    global _EXT_SOFTMAX_V
    if _EXT_SOFTMAX_V is None:
        nbits = get_jetfire_config().scheme.v_nbits
        if nbits == 4:
            from jetfire_vllm.kernels.int4_split import load_softmax_v_ext
        elif nbits == 3:
            from jetfire_vllm.kernels.int3_split import load_softmax_v_ext
        else:
            from jetfire_vllm.kernels.int2_split import load_softmax_v_ext
        _EXT_SOFTMAX_V = load_softmax_v_ext()
    return _EXT_SOFTMAX_V


def _load_filler_ext():
    """Lazy-load the Phase 0 filler kernel."""
    global _EXT_FILLER
    if _EXT_FILLER is None:
        from jetfire_vllm.kernels.filler import load_filler_ext
        _EXT_FILLER = load_filler_ext()
    return _EXT_FILLER


def _load_blend_ext():
    """Lazy-load the fused blend kernel.  Replaces the V2_05_post_corr
    torch op chain with a single CUDA launch."""
    global _EXT_BLEND
    if _EXT_BLEND is None:
        if os.environ.get("JETFIRE_BLEND_TRACE", "0") == "1":
            print("[jetfire][blend] load_enter", flush=True)
        from jetfire_vllm.kernels.blend import load_blend_ext
        _EXT_BLEND = load_blend_ext()
        if os.environ.get("JETFIRE_BLEND_TRACE", "0") == "1":
            print("[jetfire][blend] load_done", flush=True)
    return _EXT_BLEND


def _load_cpu_topk_attn_ext():
    global _EXT_CPU_TOPK_ATTN
    if _EXT_CPU_TOPK_ATTN is None:
        import os as _os
        from jetfire_vllm.kernels.cpu_topk_attn import load_cpu_topk_attn_ext
        _EXT_CPU_TOPK_ATTN = load_cpu_topk_attn_ext()
        # vLLM's engine subprocess often starts with OMP_NUM_THREADS=1, which
        # makes at::parallel_for serial.  The C++ side now uses an explicit
        # `omp parallel for num_threads(...)` driven by JETFIRE_CPU_ATTN_THREADS
        # (default = min(num_procs, 32)).  Surface the resolved values once so
        # we can confirm the worker is actually multi-core.
        target = _os.environ.get("JETFIRE_CPU_ATTN_THREADS", "<auto>")
        omp = _os.environ.get("OMP_NUM_THREADS", "<unset>")
        print(
            f"[jetfire] cpu_topk_attn ext loaded; "
            f"torch.get_num_threads()={int(torch.get_num_threads())} "
            f"OMP_NUM_THREADS={omp} "
            f"JETFIRE_CPU_ATTN_THREADS={target} "
            f"os.cpu_count={_os.cpu_count()}",
            flush=True,
        )
    return _EXT_CPU_TOPK_ATTN


def _get_head_map_cpu(B: int, H_q: int, num_groups: int, K: int) -> torch.Tensor:
    """(B, H_q, K) int64 head_map for CPU gather."""
    key = (B, H_q, num_groups, K)
    out = _HEAD_MAP_CPU.get(key)
    if out is None:
        hm = (torch.arange(H_q, dtype=torch.int64) // num_groups).unsqueeze(1).expand(H_q, K)
        out = hm.unsqueeze(0).expand(B, H_q, K).contiguous()
        _HEAD_MAP_CPU[key] = out
    return out


def _get_pinned(shape: tuple, dtype: torch.dtype, tag: str) -> torch.Tensor:
    key = (shape, dtype, tag)
    out = _PINNED_BUFS.get(key)
    if out is None:
        out = torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=True)
        _PINNED_BUFS[key] = out
    return out


def _get_device(
    shape: tuple,
    dtype: torch.dtype,
    tag: str,
    device: torch.device | str,
) -> torch.Tensor:
    dev = torch.device(device)
    key = (shape, dtype, tag, str(dev))
    out = _DEVICE_BUFS.get(key)
    if out is None:
        out = torch.empty(*shape, dtype=dtype, device=dev)
        _DEVICE_BUFS[key] = out
    return out


def _decode_buffer_warmup_key(
    *,
    max_batch_size: int,
    H_q: int,
    H_kv: int,
    D: int,
    top_k: int,
    device: torch.device | str | None,
) -> tuple[int, int, int, int, int, str]:
    dev = "" if device is None else str(torch.device(device))
    return (int(max_batch_size), int(H_q), int(H_kv), int(D), int(top_k), dev)


# Cache for cudaHostAllocMapped buffers: each entry is (host, device) sharing
# the same physical memory.  CPU writes via host_tensor are visible to GPU
# through device_tensor over PCIe — no explicit H2D needed.
_MAPPED_BUFS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_mapped_pair(shape: tuple, dtype: torch.dtype, tag: str):
    """Return cached (host_tensor, device_tensor) pair from cudaHostAllocMapped.

    The two tensors share the same physical buffer.  Caller writes via
    host_tensor and reads from device_tensor; no copy needed.  Memory is
    write-combined, so the writer must issue an sfence (cpu_topk_attn
    already does this) before the GPU reads are issued.
    """
    key = (shape, dtype, tag)
    pair = _MAPPED_BUFS.get(key)
    if pair is None:
        from jetfire_vllm.kernels.cpu_topk_attn import load_cpu_topk_attn_ext
        ext = load_cpu_topk_attn_ext()
        host, dev = ext.alloc_mapped_pinned_pair(list(shape), dtype)
        pair = (host, dev)
        _MAPPED_BUFS[key] = pair
    return pair


def _graph_store_tag(store, base: str) -> str:
    """Per-store graph buffer tag.

    CUDA graph CPU-submit jobs can overlap with nearby captured work.  Keep the
    host/device staging buffers distinct per layer store so one layer cannot
    overwrite another layer's CPU correction inputs or outputs before its graph
    wait has consumed them.
    """
    return f"{base}_graph_store_{id(store)}"


def preload_decode_extensions() -> None:
    """Eagerly JIT-compile all decode-path C++ extensions.

    Call this once before the first timed benchmark run to avoid confusing
    nvcc/ninja compile time with actual hang.  Safe to call multiple times.
    """
    cfg = get_jetfire_config()
    _debug_graph_init("preload: load kernel set start")
    _load_kernel_set()
    _debug_graph_init("preload: load kernel set done")
    _debug_graph_init("preload: load gather ext start")
    _load_gather_ext()
    _debug_graph_init("preload: load gather ext done")
    _debug_graph_init("preload: load cpu_topk_attn ext start")
    _load_cpu_topk_attn_ext()
    _debug_graph_init("preload: load cpu_topk_attn ext done")
    _debug_graph_init("preload: load qk_scores ext start")
    _load_qk_scores_ext()
    _debug_graph_init("preload: load qk_scores ext done")
    _debug_graph_init("preload: load softmax_v ext start")
    _load_softmax_v_ext()
    _debug_graph_init("preload: load softmax_v ext done")
    if os.environ.get("JETFIRE_USE_BLEND_KERNEL", "0") == "1":
        _debug_graph_init("preload: load blend ext start")
        _load_blend_ext()
        _debug_graph_init("preload: load blend ext done")
    if os.environ.get("JETFIRE_USE_FUSED_ATTN_TOPK", "0") == "1":
        _kernel_pref = os.environ.get("JETFIRE_FUSED_KERNEL", "v1")
        if _kernel_pref == "v1":
            _debug_graph_init("preload: load fused_attn_topk_v1 ext start")
            _load_fused_attn_topk_v1_ext()
            _debug_graph_init("preload: load fused_attn_topk_v1 ext done")
        else:
            _debug_graph_init("preload: load fused_attn_topk ext start")
            _load_fused_attn_topk_ext()
            _debug_graph_init("preload: load fused_attn_topk ext done")
    if (
        cfg.scheme.k_nbits == 4
        and not cfg.scheme.k_asymmetric
        and os.environ.get("JETFIRE_USE_INT8_TC", "1") == "1"
    ):
        from jetfire_vllm.kernels.int8tc_qk_scores import load_int8tc_qk_scores_ext
        _debug_graph_init("preload: load int8tc_qk_scores ext start")
        load_int8tc_qk_scores_ext()
        _debug_graph_init("preload: load int8tc_qk_scores ext done")
    if (
        cfg.scheme.v_nbits == 4
        and not cfg.scheme.v_asymmetric
        and os.environ.get("JETFIRE_USE_INT8_TC_SV", "auto") not in ("0", "off", "false", "False")
    ):
        from jetfire_vllm.kernels.int8tc_softmax_v_fused import (
            load_ext as load_int8tc_softmax_v_ext,
        )
        _debug_graph_init("preload: load int8tc_softmax_v ext start")
        load_int8tc_softmax_v_ext()
        _debug_graph_init("preload: load int8tc_softmax_v ext done")


def warmup_decode_buffers(
    *,
    max_batch_size: int,
    H_q: int,
    H_kv: int,
    D: int,
    top_k: int,
    device: torch.device | str | None = None,
) -> None:
    """Pre-allocate every decode-path host buffer at ``max_batch_size`` shape.

    After this runs, the hot path uses ``[:n_seqs]`` slices of the same
    physical memory across all forwards — no pinned/mapped allocation, no
    dict-key miss, and pointers stay stable (a prerequisite for CUDA Graph
    capture later on).

    Idempotent and safe to call multiple times; only the first call does
    work.  Tagged buffers covered:

      * ``q``, ``topk_slots`` — pinned host (D2H staging)
      * ``o_new`` — mapped pinned pair (CPU writes, GPU reads via PCIe)
      * ``cpu_flag`` — pinned int64 doorbell for the fused-join blend path
      * ``head_map`` — CPU int64 head→kv-head map
    """
    global _BUFFERS_WARMED_UP
    if max_batch_size <= 0:
        return
    warmup_key = _decode_buffer_warmup_key(
        max_batch_size=max_batch_size,
        H_q=H_q,
        H_kv=H_kv,
        D=D,
        top_k=top_k,
        device=device,
    )
    if warmup_key in _BUFFER_WARMUP_KEYS:
        return
    num_groups = H_q // H_kv
    _get_pinned((max_batch_size, H_q, D),     torch.bfloat16, "q")
    _get_pinned((max_batch_size, H_q, top_k), torch.int64,    "topk_slots")
    _get_pinned((1,),                         torch.int64,    "cpu_flag")
    _get_mapped_pair((max_batch_size, H_q, D), torch.bfloat16, "o_new")
    if device is not None:
        _get_device(
            (max_batch_size, H_q, D),
            torch.bfloat16,
            "o_new_graph_h2d",
            device,
        )
    _get_head_map_cpu(max_batch_size, H_q, num_groups, top_k)
    # Eagerly spawn the all-C++ async CPU executor so the first decode does
    # not pay thread setup latency.  Keep the warmup gate aligned with the
    # actual decode-path gate below: default OFF unless explicitly A/B testing
    # the async CPU path.
    if (
        os.environ.get("JETFIRE_USE_PERSISTENT_WORKER", "0") == "1"
        or _graph_mode_enabled()
    ):
        try:
            ext = _load_cpu_topk_attn_ext()
            if _graph_mode_enabled():
                ext.graph_pool_warmup()
            else:
                ext.persistent_worker_warmup()
        except Exception:
            pass
    _BUFFER_WARMUP_KEYS.add(warmup_key)
    _BUFFERS_WARMED_UP = True


def _resolve_buf_B(n_seqs: int, cfg_max_B: int) -> int:
    """Pick the buffer batch dim: max-prealloc shape if ≥ n_seqs, else n_seqs.

    When the configured max_batch_size is exceeded (rare), we fall back to a
    one-off buffer at ``n_seqs`` and warn once.  Steady state hits the cache.
    """
    if cfg_max_B <= 0:
        return n_seqs
    if n_seqs <= cfg_max_B:
        return cfg_max_B
    global _WARMUP_OVERFLOW_LOGGED
    if not _WARMUP_OVERFLOW_LOGGED:
        _WARMUP_OVERFLOW_LOGGED = True
        print(
            f"[jetfire] decode batch n_seqs={n_seqs} exceeds "
            f"JETFIRE_MAX_BATCH_SIZE={cfg_max_B}; allocating a one-off "
            f"buffer.  Increase JETFIRE_MAX_BATCH_SIZE to keep prealloc.",
            flush=True,
        )
    return n_seqs


def _graph_mode_enabled() -> bool:
    """Whether to use stream-captured CPU/GPU handshake (cudaLaunchHostFunc
    + cuStreamWaitValue64) instead of the host-side wait_for_flag/submit
    path.

    This remains opt-in outside the release preset.  Release initialization
    preloads graph resources and installs stores before real capture; a miss
    during capture raises instead of baking native flash-attn into the graph.
    """
    if os.environ.get("JETFIRE_USE_PERSISTENT_WORKER", "0") == "1":
        return False
    return os.environ.get("JETFIRE_DECODE_GRAPH", "0") == "1"


def _graph_use_mapped_o_new(n_seqs: int) -> bool:
    """Select graph-mode o_new transport.

    Default to a graph-captured H2D copy from pinned host memory.  Direct GPU
    reads from mapped host memory are faster, but Qasper A/B showed they can
    change greedy decode outputs even in enforce_eager+graph-submit mode.  Keep
    the mapped path as an explicit perf experiment only.
    """
    mode = os.environ.get("JETFIRE_GRAPH_MAPPED_O_NEW", "auto").lower()
    if mode in ("1", "true", "yes", "on", "mapped"):
        return True
    if mode in ("0", "false", "no", "off", "h2d", "pinned"):
        return False
    return False


def _graph_capture_active() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _graph_context_batch_sizes(max_batch_size: int) -> list[int]:
    """Batch sizes whose graph submit contexts must exist before capture.

    By default we preserve the historical all-sizes behavior. Benchmarks and
    production launchers can set JETFIRE_GRAPH_CAPTURE_SIZES to the exact vLLM
    CUDA graph capture sizes, avoiding unnecessary layer x n_seqs contexts.
    """
    if max_batch_size <= 0:
        return []
    raw = os.environ.get("JETFIRE_GRAPH_CAPTURE_SIZES", "").strip()
    if not raw:
        return list(range(1, max_batch_size + 1))
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise RuntimeError(
                "JETFIRE_GRAPH_CAPTURE_SIZES must be a comma-separated list "
                f"of integers; got {raw!r}"
            ) from None
        if 1 <= value <= max_batch_size:
            out.add(value)
    out.add(max_batch_size)
    return sorted(out)


def _raise_graph_decode_miss(reason: str, *, layer_name: str) -> None:
    raise RuntimeError(
        "JetFire CUDA-graph decode miss would fall back to native flash-attn "
        "and poison the captured graph. "
        f"reason={reason} layer={layer_name}. "
        "Pre-create JetFire stores and graph submit contexts before vLLM "
        "captures decode, or disable JETFIRE_DECODE_GRAPH."
    )


def _try_graph_profile_placeholder(*, pool, output) -> bool:
    """Handle vLLM's CUDA graph memory profiling dummy run.

    That graph is only for memory estimation.  Returning a placeholder avoids
    native fallback while keeping real decode graph capture strict after the
    profiling phase.  vLLM may install a minimal 1-block KV cache first, so do
    not rely on "no layer layouts" as the only profiling signal.
    """
    if not _graph_mode_enabled():
        return False
    in_vllm_graph_profile = (
        os.environ.get("JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE", "0") == "1"
    )
    if pool.layer_layouts and not in_vllm_graph_profile:
        return False
    if output is None:
        return False
    # During vLLM's pre-KV-init CUDA graph memory profiling there are no
    # JetFire stores yet.  Do not launch even a placeholder CUDA memset here:
    # for larger capture sizes this can be the only JetFire op inside a
    # partially captured graph and has triggered stream-capture invalidation.
    # The values are irrelevant for memory profiling; the tensor only needs
    # to flow through the dummy forward.
    if not _graph_capture_active():
        output.zero_()
    pool.record_decode_fallback("graph_profile_placeholder")
    return True


def _ensure_graph_flag() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (flag_host, flag_dev) mapped pinned int64 pair, allocated once.

    Both halves point to the same physical memory; CPU writes via flag_host
    are visible to GPU through flag_dev (PCIe) and vice-versa.  This is kept
    for the legacy mapped-host wait path and for a stable host-side done flag.
    """
    global _GRAPH_FLAG_HOST, _GRAPH_FLAG_DEV
    if _GRAPH_FLAG_HOST is None:
        ext = _load_cpu_topk_attn_ext()
        _GRAPH_FLAG_HOST, _GRAPH_FLAG_DEV = ext.alloc_mapped_pinned_pair(
            [1], torch.int64
        )
        _GRAPH_FLAG_HOST.fill_(1)  # initial state: "no pending CPU work"
    return _GRAPH_FLAG_HOST, _GRAPH_FLAG_DEV


def _ensure_graph_signal_dev(device: torch.device | str) -> torch.Tensor:
    """Return a CUDA int64 doorbell for the release device-signal path.

    The main stream resets this device flag to 0 before submit; the last CPU
    worker writes 1 to it from a side CUDA stream.  This preserves CPU/GPU
    overlap without captured host-join or side-join replay state.
    """
    dev = torch.device(device)
    key = str(dev)
    out = _GRAPH_SIGNAL_DEV.get(key)
    if out is None:
        out = torch.empty(1, dtype=torch.int64, device=dev)
        out.fill_(1)
        _GRAPH_SIGNAL_DEV[key] = out
    return out


def _get_graph_submit_ctx(
    *,
    store,
    n_seqs: int,
    buf_B: int,
    H_q: int,
    num_groups: int,
    top_k: int,
    D: int,
    scale: float,
    mapped_o_new: bool,
    signal_device: torch.Tensor | None,
    device: torch.device | str | None = None,
) -> int:
    """Lazily build a CPU-topk-attn submit context for graph-mode decode.

    Cached by (store identity, top_k, n_seqs) — shadow K/V are per-layer
    (each cache_store owns its own pair) so the cache key includes the
    store.  The buffers (q_buf, ts_buf, o_new_pin, head_map) are global
    and shared across layers; their pointers stay stable across graph
    replays.  Returns an opaque int handle for stream_submit_cpu_topk_attn.
    """
    # Keep the store object itself in the key rather than id(store).  vLLM can
    # rebuild JetFire stores during startup/profiling; Python may then reuse an
    # old object id while the C++ submit ctx still holds tensor references to
    # the previous store.  Identity-keying by object plus explicit graph-context
    # reset prevents stale shadow pointers from being replayed.
    signal_key = str(signal_device.device) if signal_device is not None else "mapped_host"
    key = (store, top_k, n_seqs, buf_B, bool(mapped_o_new), signal_key)
    h = _GRAPH_SUBMIT_CTX.get(key)
    if h is not None:
        return h
    if _graph_capture_active():
        raise RuntimeError(
            "JetFire graph submit context was not pre-created before CUDA "
            f"graph capture (store_id={id(store)} top_k={top_k} n_seqs={n_seqs}). "
            "Call warmup_graph_decode_submit_contexts before vLLM captures decode."
        )
    ext = _load_cpu_topk_attn_ext()
    q_tag = _graph_store_tag(store, "q")
    ts_tag = _graph_store_tag(store, "topk_slots")
    o_tag = _graph_store_tag(
        store, "o_new" if mapped_o_new else "o_new_graph_h2d"
    )
    q_buf_full  = _get_pinned((buf_B, H_q, D),     torch.bfloat16, q_tag)
    ts_buf_full = _get_pinned((buf_B, H_q, top_k), torch.int64,    ts_tag)
    if mapped_o_new:
        o_pin_full, _ = _get_mapped_pair(
            (buf_B, H_q, D), torch.bfloat16, o_tag
        )
    else:
        o_pin_full = _get_pinned(
            (buf_B, H_q, D), torch.bfloat16, o_tag
        )
        if device is not None:
            _get_device((buf_B, H_q, D), torch.bfloat16, o_tag, device)
    head_map_full = _get_head_map_cpu(buf_B, H_q, num_groups, top_k)
    flag_host, _ = _ensure_graph_flag()
    # Slice to n_seqs so the CPU worker only processes the relevant rows.
    # The slices share storage with the full buffers (same data_ptr origin
    # plus offset 0 for first-N slicing → unchanged), so pointers stay
    # stable across graph replays.
    args = (
        q_buf_full[:n_seqs].contiguous(),
        store.k_shadow,
        store.v_shadow,
        head_map_full[:n_seqs].contiguous(),
        ts_buf_full[:n_seqs].contiguous(),
        o_pin_full[:n_seqs].contiguous(),
        float(scale),
        flag_host,
    )
    if signal_device is not None:
        h = ext.create_cpu_topk_submit_ctx_device_signal(*args, signal_device)
    else:
        h = ext.create_cpu_topk_submit_ctx(*args)
    _GRAPH_SUBMIT_CTX[key] = h
    return h


def warmup_graph_decode_submit_contexts(
    *,
    max_batch_size: int,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Pre-create graph-mode CPU submit contexts for every mirrored layer.

    This must run before vLLM captures decode graphs.  The captured graph
    stores raw pointers to q/topk/o_new buffers, shadow KV, and the mapped
    doorbell; creating any of those lazily inside capture either fails or
    records the wrong path.  By default this preserves the historical
    all-sizes behavior; production launchers should set
    JETFIRE_GRAPH_CAPTURE_SIZES to the exact vLLM capture sizes.
    """
    if max_batch_size <= 0:
        raise RuntimeError(
            "JETFIRE_DECODE_GRAPH=1 requires JETFIRE_MAX_BATCH_SIZE>0 so "
            "decode host buffers and submit contexts have stable pointers "
            "before CUDA graph capture."
        )

    cfg = get_jetfire_config()
    if not (cfg.cpu_correction and cfg.decode_cpu_attn):
        raise RuntimeError(
            "JETFIRE_DECODE_GRAPH=1 currently requires "
            "JETFIRE_CPU_CORRECTION=1 and JETFIRE_DECODE_CPU_ATTN=1."
        )

    pool = get_global_kv_pool_manager()
    if not pool.layer_stores:
        raise RuntimeError(
            "JETFIRE_DECODE_GRAPH=1 cannot warm submit contexts before "
            "JetFire layer stores exist."
        )

    kernel_set = _load_kernel_set()
    top_k = _select_top_k(cfg.topk, cfg.topk, kernel_set.supported_top_k)
    if top_k is None:
        raise RuntimeError(
            f"JETFIRE_TOPK={cfg.topk} is not supported by the active kernel set."
        )
    # vLLM's decode capture dummy can have max_S smaller than configured
    # JETFIRE_TOPK.  The hot path then clamps top_k to max_S, so graph warmup
    # must pre-create every supported top_k shape up to the configured value.
    warmup_top_ks = tuple(
        int(k)
        for k in kernel_set.supported_top_k
        if int(k) <= int(top_k)
    )
    if not warmup_top_ks:
        warmup_top_ks = (int(top_k),)

    _debug_graph_init("warmup submit contexts: preload extensions start")
    preload_decode_extensions()
    _debug_graph_init("warmup submit contexts: preload extensions done")
    _debug_graph_init("warmup submit contexts: load gather ext start")
    _load_gather_ext()
    _debug_graph_init("warmup submit contexts: load gather ext done")
    _debug_graph_init("warmup submit contexts: load cpu_topk_attn ext start")
    ext = _load_cpu_topk_attn_ext()
    requested_cpu_threads = int(ext.get_cpu_attn_threads())
    graph_cpu_threads = requested_cpu_threads
    if os.environ.get("JETFIRE_GRAPH_CAP_CPU_ATTN_THREADS", "1") == "1":
        max_hq = max(
            int(cfg.num_query_heads)
            for cfg in pool.layer_configs.values()
        )
        cap = max(1, int(max_batch_size) * max_hq)
        graph_cpu_threads = min(requested_cpu_threads, cap)
        _debug_graph_init(
            "warmup submit contexts: CPU attn thread plan "
            f"requested={requested_cpu_threads} selected={graph_cpu_threads} "
            f"max_batch_size={max_batch_size} max_H_q={max_hq}"
        )
    _debug_graph_init("warmup submit contexts: graph_pool_warmup start")
    if hasattr(ext, "graph_pool_warmup_threads"):
        graph_cpu_threads = int(ext.graph_pool_warmup_threads(graph_cpu_threads))
    else:
        ext.set_cpu_attn_threads(graph_cpu_threads)
        ext.graph_pool_warmup()
    _debug_graph_init("warmup submit contexts: graph_pool_warmup done")
    _debug_graph_init("warmup submit contexts: ensure graph flag start")
    _ensure_graph_flag()
    _debug_graph_init("warmup submit contexts: ensure graph flag done")
    use_device_signal = (
        os.environ.get("JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK", "0") == "1"
    )
    signal_device = None
    if use_device_signal:
        signal_device = _ensure_graph_signal_dev(
            device if device is not None else torch.device("cuda", torch.cuda.current_device())
        )

    made = 0
    context_batch_sizes = _graph_context_batch_sizes(max_batch_size)
    _debug_graph_init(
        "warmup submit contexts: layer loop start "
        f"layers={len(pool.layer_stores)} batches={context_batch_sizes} "
        f"top_ks={warmup_top_ks}"
    )
    for layer_name, store in list(pool.layer_stores.items()):
        layer_cfg = pool.get_layer_config(layer_name)
        if layer_cfg is None:
            continue
        if store.k_shadow is None or store.v_shadow is None:
            raise RuntimeError(
                "JETFIRE_DECODE_GRAPH=1 needs CPU shadow KV for correction; "
                f"layer {layer_name} has no shadow store."
            )
        H_q = layer_cfg.num_query_heads
        H_kv = layer_cfg.num_kv_heads
        D = layer_cfg.head_dim
        if H_q % H_kv != 0:
            raise RuntimeError(
                f"Invalid GQA layout for {layer_name}: H_q={H_q}, H_kv={H_kv}"
            )
        num_groups = H_q // H_kv
        _debug_graph_init(f"warmup layer {layer_name}: decode buffers start")
        for warmup_top_k in warmup_top_ks:
            warmup_decode_buffers(
                max_batch_size=max_batch_size,
                H_q=H_q,
                H_kv=H_kv,
                D=D,
                top_k=warmup_top_k,
                device=device,
            )
        _debug_graph_init(f"warmup layer {layer_name}: graph append start")
        store.warmup_graph_append(max_batch_size=max_batch_size)
        _debug_graph_init(f"warmup layer {layer_name}: submit ctx start")
        scale = 1.0 / math.sqrt(D)
        for warmup_top_k in warmup_top_ks:
            for n_seqs in context_batch_sizes:
                before = len(_GRAPH_SUBMIT_CTX)
                _get_graph_submit_ctx(
                    store=store,
                    n_seqs=n_seqs,
                    buf_B=max_batch_size,
                    H_q=H_q,
                    num_groups=num_groups,
                    top_k=warmup_top_k,
                    D=D,
                    scale=scale,
                    mapped_o_new=_graph_use_mapped_o_new(n_seqs),
                    signal_device=signal_device,
                    device=device,
                )
                made += int(len(_GRAPH_SUBMIT_CTX) > before)
        _debug_graph_init(f"warmup layer {layer_name}: done")

    _debug_graph_init(
        "warmup submit contexts: done "
        f"created={made} total={len(_GRAPH_SUBMIT_CTX)}"
    )
    return {
        "layers": len(pool.layer_stores),
        "max_batch_size": int(max_batch_size),
        "context_batch_sizes": [int(x) for x in context_batch_sizes],
        "top_k": int(top_k),
        "cpu_attn_threads_requested": int(requested_cpu_threads),
        "cpu_attn_threads_selected": int(graph_cpu_threads),
        "submit_contexts_created": int(made),
        "submit_contexts_total": int(len(_GRAPH_SUBMIT_CTX)),
    }


def jetfire_get_graph_stats() -> dict[str, int]:
    try:
        ext = _load_cpu_topk_attn_ext()
        out = {
            "graph_submit_count": int(ext.get_graph_submit_count()),
            "graph_shadow_append_count": int(ext.get_shadow_append_count()),
            "cpu_topk_sanitize_count": int(
                ext.get_cpu_topk_sanitize_count()
            ),
            "graph_submit_contexts": int(len(_GRAPH_SUBMIT_CTX)),
        }
        if hasattr(ext, "get_graph_timing_stats"):
            for key, value in ext.get_graph_timing_stats().items():
                out[f"graph_timing_{key}"] = (
                    float(value) if isinstance(value, float) else int(value)
                )
        return out
    except Exception:
        return {
            "graph_submit_count": 0,
            "graph_submit_contexts": int(len(_GRAPH_SUBMIT_CTX)),
        }


def reset_jetfire_graph_stats() -> None:
    try:
        _load_cpu_topk_attn_ext().reset_graph_submit_count()
    except Exception:
        pass


def reset_jetfire_graph_contexts() -> None:
    """Drop graph submit/append contexts tied to the current KV store epoch."""
    _GRAPH_SUBMIT_CTX.clear()
    try:
        _load_cpu_topk_attn_ext().clear_graph_contexts()
    except Exception:
        pass


def reset_jetfire_graph_runtime_state() -> None:
    """Reset graph replay CPU-barrier state without invalidating handles.

    CUDA graphs capture the integer submit-context handles, so graph mode
    cannot call ``reset_jetfire_graph_contexts`` between independent requests.
    This helper keeps the contexts and their tensor pointers alive while
    restoring the per-request CPU topK completion barriers/counters.
    """
    try:
        ext = _load_cpu_topk_attn_ext()
        if hasattr(ext, "reset_graph_runtime_state"):
            ext.reset_graph_runtime_state()
        else:
            ext.reset_graph_submit_count()
    except Exception:
        pass


def reset_jetfire_decode_runtime_state() -> None:
    """Clear per-request decode runtime state after a serial LongBench sample.

    The kernel/extensions are kept loaded, but host/device staging buffers and
    CPU worker state are rebuilt for the next request.  This avoids reusing
    mapped-pinned buffers, stream-ordered submit contexts, or a stuck
    ThreadPoolExecutor across independent LongBench samples.
    """
    global _COMM_STREAM, _GATHER_EXEC, _GRAPH_FLAG_HOST, _GRAPH_FLAG_DEV
    global _BUFFERS_WARMED_UP, _WARMUP_OVERFLOW_LOGGED

    reset_jetfire_graph_contexts()
    _HEAD_MAP_CPU.clear()
    _PINNED_BUFS.clear()
    _DEVICE_BUFS.clear()
    _MAPPED_BUFS.clear()
    _BUFFER_WARMUP_KEYS.clear()
    _GRAPH_SIGNAL_DEV.clear()
    _GRAPH_FLAG_HOST = None
    _GRAPH_FLAG_DEV = None
    _COMM_STREAM = None
    _BUFFERS_WARMED_UP = False
    _WARMUP_OVERFLOW_LOGGED = False

    if _GATHER_EXEC is not None:
        _GATHER_EXEC.shutdown(wait=True, cancel_futures=True)
        _GATHER_EXEC = None


def _get_comm_stream() -> torch.cuda.Stream:
    global _COMM_STREAM
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream()
    return _COMM_STREAM


def _get_gather_exec() -> ThreadPoolExecutor:
    global _GATHER_EXEC
    if _GATHER_EXEC is None:
        _GATHER_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jetfire-gather")
    return _GATHER_EXEC


def _cpu_worker_timeout_debug(cpu_future, *, layer_name: str, trace_id: int) -> str:
    parts = [f"layer={layer_name}", f"trace_id={trace_id}"]
    if cpu_future is not None:
        try:
            parts.append(f"future_running={int(cpu_future.running())}")
            parts.append(f"future_done={int(cpu_future.done())}")
        except Exception:
            pass
    try:
        parts.append(f"graph_stats={jetfire_get_graph_stats()}")
    except Exception as exc:
        parts.append(f"graph_stats_error={type(exc).__name__}: {exc}")
    return "; ".join(parts)


def _sanitize_decode_output_if_needed(output: torch.Tensor, n_seqs: int) -> None:
    """Keep graph replay from propagating nonfinite correction rows.

    The normal finite path is unchanged.  When a quantized decode correction
    produces NaN/Inf, letting it enter the next layer poisons Q/K/V and can
    turn the CPU correction worker into a long-running nonfinite cleanup path.
    ``nan_to_num`` is capturable and the tensor is tiny in decode
    (B * local heads * head_dim), so the guard is cheap.
    """
    if os.environ.get("JETFIRE_SANITIZE_DECODE_OUTPUT", "1") == "0":
        return
    if not _graph_mode_enabled():
        return
    if n_seqs <= 0:
        return
    view = output[:n_seqs]
    torch.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0, out=view)


# ── Top-K selection ────────────────────────────────────────────────────
_TOPK_CLAMP_LOGGED: set[tuple[int, int, tuple[int, ...]]] = set()


def _select_top_k(
    config_top_k: int, seq_len: int, supported: tuple[int, ...]
) -> int | None:
    """Select a supported correction width, padding short sequences with -1.

    Kernels mask unused top-k lanes, so a real context shorter than the
    minimum supported width can still run through the complete correction
    path. Native attention is unavailable when its cache has been released.
    Return ``None`` only for empty sequences or unsupported configurations.
    """
    requested, length = int(config_top_k), int(seq_len)
    configured = [k for k in supported if 0 < k <= requested]
    if not configured or length <= 0:
        return None
    candidates = [k for k in configured if k <= length]
    selected = max(candidates) if candidates else min(configured)
    if selected != requested:
        key = (requested, selected, tuple(supported))
        if key not in _TOPK_CLAMP_LOGGED:
            _TOPK_CLAMP_LOGGED.add(key)
            print(
                "[jetfire] requested top_k="
                f"{requested} clamped to kernel-supported top_k={selected}; "
                f"supported={tuple(supported)}.",
                flush=True,
            )
    return selected


# ── Prefill ────────────────────────────────────────────────────────────
def try_jetfire_prefill(*, impl, layer, query, key, value, attn_metadata, output):
    """Handle prefill (and the V1 prefill+decode mixed batch).

    Routing rules:
      * pure prefill (one-shot, no cache): flash-attn varlen on the full
        batch, KV mirroring done via ``do_kv_cache_update`` before forward.
      * pure decode batch (max_query_len == 1): return ``None`` so the
        caller falls through to ``try_jetfire_decode``.
      * mixed batch (some seqs query_len==1, others >1): split the rows
        and route each half — flash-attn varlen for the prefill seqs,
        ``try_jetfire_decode`` (with sub-selected metadata) for the decode
        seqs.  V1's scheduler can pack a decode token together with new
        prefills in the same step even when ``enable_chunked_prefill=False``,
        so this path is needed whenever batch>1.

    Returns ``None`` for cases the caller should handle:
      • skip_native_kv off, DCP>1, non-decoder, cascade
      • num_actual_tokens <= 1 (single decode token)
      • pure decode batch (let ``try_jetfire_decode`` handle it)
    """
    cfg = get_jetfire_config()
    if not cfg.skip_native_kv:
        return None
    if attn_metadata is None:
        return None
    if getattr(impl, "dcp_world_size", 1) > 1:
        return None
    if impl.attn_type.name != "DECODER":
        return None
    if attn_metadata.use_cascade:
        return None
    n_actual = int(attn_metadata.num_actual_tokens)
    if n_actual <= 1:
        return None

    max_q = int(attn_metadata.max_query_len)
    # Pure decode batch (B>1): hand off to try_jetfire_decode.
    if max_q == 1:
        return None

    max_s = int(attn_metadata.max_seq_len)
    num_decodes_meta = getattr(attn_metadata, "num_decodes", None)
    num_prefills_meta = getattr(attn_metadata, "num_prefills", None)
    if num_decodes_meta is not None and num_prefills_meta is not None:
        # JetFire's metadata builder carries vLLM's CPU-side reordered split:
        # decode requests are a contiguous prefix, prefill requests are a
        # contiguous tail.  Use it instead of doing per-layer D2H/tolist.
        prefill_has_context = bool(
            getattr(attn_metadata, "jetfire_prefill_has_context", False)
        )
        if int(num_prefills_meta) > 0 and prefill_has_context:
            decode_equiv = _try_decode_equiv_extend_prefill(
                impl=impl,
                layer=layer,
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                output=output,
            )
            if decode_equiv is not None:
                return decode_equiv
            record_runtime_metadata(
                event="unsupported_extend_prefill",
                attn_metadata=attn_metadata,
                layer_name=getattr(layer, "layer_name", "unknown_layer"),
                query_shape=tuple(query.shape),
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
            )
            raise RuntimeError(
                "JetFire skip-native KV does not support chunked/extend "
                "prefill yet: prefill requests have cached context "
                f"(max context={getattr(attn_metadata, 'jetfire_max_prefill_context_len', 0)}). "
                "This can be caused by vLLM scheduler token-budget splitting, "
                "chunked/extend prefill, or prefix-cache/block eviction. "
                "The qlen=1 cached-context shape is routed through JetFire's "
                "decode-equivalent path; this error means at least one row needs "
                "true multi-token extend-prefill attention.  Lower batch/context, "
                "raise max_num_batched_tokens, increase JetFire KV capacity, or "
                "keep native KV enabled until an INT4 extend-prefill attention "
                "path exists."
            )
        one_shot = int(num_decodes_meta) == 0
    elif query.is_cuda and torch.cuda.is_current_stream_capturing():
        # CUDA-graph capture: `seq_lens.sum().item()` would issue a D2H sync
        # and poison the capture stream (cudaErrorStreamCaptureInvalidated).
        # Under jetfire's required config (no chunked prefill, no prefix
        # cache) every prefill seq has seq_len == query_len, so the one_shot
        # fast path is the only legitimate prefill shape under capture; the
        # mixed-batch branch is host-decided and cannot be captured anyway.
        one_shot = True
    else:
        sum_s = int(attn_metadata.seq_lens.sum().item())
        one_shot = (max_q == max_s and sum_s == n_actual)
    if not one_shot:
        # Mixed batch: at least one seq is in decode (qlen==1) and at
        # least one is fresh prefill (qlen>1).  Split per-seq.
        return _jetfire_mixed_batch_attn(
            impl=impl, layer=layer, query=query, key=key, value=value,
            attn_metadata=attn_metadata, output=output,
        )

    sliding = list(impl.sliding_window) if impl.sliding_window is not None else None
    descale_shape = (
        int(attn_metadata.query_start_loc.shape[0]) - 1,
        impl.num_kv_heads,
    )
    flash_attn_varlen_func(
        q=query[:n_actual], k=key[:n_actual], v=value[:n_actual],
        out=output[:n_actual],
        cu_seqlens_q=attn_metadata.query_start_loc,
        cu_seqlens_k=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        max_seqlen_k=attn_metadata.max_query_len,
        softmax_scale=impl.scale,
        causal=attn_metadata.causal,
        alibi_slopes=impl.alibi_slopes,
        window_size=sliding,
        softcap=impl.logits_soft_cap,
        fa_version=impl.vllm_flash_attn_version,
        q_descale=layer._q_scale.expand(descale_shape),
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        num_splits=attn_metadata.max_num_splits,
        s_aux=impl.sinks,
    )
    _flush_deferred_kv_mirror(layer)
    return output


# ── Mixed batch (V1 prefill+decode in one step) ────────────────────────
_MIXED_LOG_DONE = False
_EXTEND_DECODE_EQUIV_LOG_DONE = False


def _flush_deferred_kv_mirror(layer) -> None:
    pending_mirror = getattr(layer, "_jetfire_deferred_kv_mirror", None)
    if pending_mirror is None:
        return
    delattr(layer, "_jetfire_deferred_kv_mirror")
    mirror_layer_name, mirror_key, mirror_value, mirror_slots = pending_mirror
    get_global_kv_pool_manager().mirror_kv(
        layer_name=mirror_layer_name,
        key=mirror_key,
        value=mirror_value,
        slot_mapping=mirror_slots,
    )


def _try_decode_equiv_extend_prefill(
    *,
    impl,
    layer,
    query,
    key,
    value,
    attn_metadata,
    output,
):
    """Handle cached-context prefill rows that are actually one-token decode.

    vLLM can classify a row as prefill when it has cached context but only one
    newly scheduled query token.  JetFire does not yet support true multi-token
    extend-prefill, but this narrow shape is semantically equivalent to decode:
    do KV update first, then attend the single query token to the full sequence.
    """
    if os.environ.get("JETFIRE_EXTEND_PREFILL_DECODE_EQUIV", "1") != "1":
        return None
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        return None

    num_prefills = int(getattr(attn_metadata, "num_prefills", 0) or 0)
    if num_prefills <= 0:
        return None

    qstart = attn_metadata.query_start_loc.to("cpu", dtype=torch.int64).tolist()
    seq_lens_cpu = attn_metadata.seq_lens.to("cpu", dtype=torch.int64).tolist()
    qlens = [qstart[i + 1] - qstart[i] for i in range(len(qstart) - 1)]
    if len(qlens) != len(seq_lens_cpu):
        return None

    has_decode_equiv = False
    unsupported: list[tuple[int, int, int]] = []
    for i, (qlen, seq_len) in enumerate(zip(qlens, seq_lens_cpu)):
        context = int(seq_len) - int(qlen)
        if context <= 0:
            continue
        if int(qlen) == 1:
            has_decode_equiv = True
        else:
            unsupported.append((i, int(qlen), int(context)))
    if not has_decode_equiv or unsupported:
        return None

    global _EXTEND_DECODE_EQUIV_LOG_DONE
    if not _EXTEND_DECODE_EQUIV_LOG_DONE:
        _EXTEND_DECODE_EQUIV_LOG_DONE = True
        print(
            "[jetfire] routing qlen=1 cached-context prefill rows through "
            f"decode-equivalent mixed path: qlens[:8]={qlens[:8]} "
            f"seq_lens[:8]={seq_lens_cpu[:8]}",
            flush=True,
        )

    return _jetfire_mixed_batch_attn(
        impl=impl,
        layer=layer,
        query=query,
        key=key,
        value=value,
        attn_metadata=attn_metadata,
        output=output,
    )


def _jetfire_mixed_batch_attn(*, impl, layer, query, key, value,
                                attn_metadata, output):
    """Split a V1 mixed prefill+decode batch and route each half.

    With ``enable_chunked_prefill=False``, every prefill seq in this step
    has ``seq_len == query_len`` (no partial cache) — verified below.
    Decode seqs have ``query_len == 1`` and an existing INT mirror entry
    populated by ``do_kv_cache_update`` *before* this forward call.
    """
    from types import SimpleNamespace

    num_decodes_meta = getattr(attn_metadata, "num_decodes", None)
    num_prefills_meta = getattr(attn_metadata, "num_prefills", None)
    num_decode_tokens_meta = getattr(attn_metadata, "num_decode_tokens", None)
    num_prefill_tokens_meta = getattr(attn_metadata, "num_prefill_tokens", None)
    if (
        num_decodes_meta is not None
        and num_prefills_meta is not None
        and num_decode_tokens_meta is not None
        and num_prefill_tokens_meta is not None
    ):
        num_decodes = int(num_decodes_meta)
        num_prefills = int(num_prefills_meta)
        num_decode_tokens = int(num_decode_tokens_meta)
        num_prefill_tokens = int(num_prefill_tokens_meta)
        if num_decodes > 0 and num_prefills > 0 and num_decode_tokens == num_decodes:
            return _jetfire_mixed_batch_attn_ordered(
                impl=impl,
                layer=layer,
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                output=output,
                num_decodes=num_decodes,
                num_prefills=num_prefills,
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )

    # Legacy host-decided dispatch (D2H tolist + dynamic index_select) is not
    # capture-safe.  The ordered path above is the only mixed route allowed
    # under PIECEWISE capture.
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        return None

    qstart = attn_metadata.query_start_loc.to("cpu", dtype=torch.int64).tolist()
    seq_lens_cpu = attn_metadata.seq_lens.to("cpu", dtype=torch.int64).tolist()
    n_seqs = len(qstart) - 1
    qlens = [qstart[i + 1] - qstart[i] for i in range(n_seqs)]

    decode_seqs = [i for i in range(n_seqs) if qlens[i] == 1]
    prefill_seqs = [i for i in range(n_seqs) if qlens[i] > 1]
    pool = get_global_kv_pool_manager()
    pool.record_mixed_batch(
        prefill_seqs=len(prefill_seqs),
        decode_seqs=len(decode_seqs),
        prefill_tokens=sum(qlens[i] for i in prefill_seqs),
        decode_tokens=len(decode_seqs),
    )

    for i in prefill_seqs:
        if seq_lens_cpu[i] != qlens[i]:
            raise RuntimeError(
                "JetFire mixed-batch handler expects each prefill seq to "
                "have seq_len == query_len (i.e. no chunked prefill). "
                f"Got seq[{i}] seq_len={seq_lens_cpu[i]} qlen={qlens[i]}. "
                "Pass enable_chunked_prefill=False to vLLM."
            )

    global _MIXED_LOG_DONE
    if not _MIXED_LOG_DONE:
        _MIXED_LOG_DONE = True
        print(
            f"[jetfire] mixed-batch dispatch: "
            f"prefill_seqs={len(prefill_seqs)} decode_seqs={len(decode_seqs)} "
            f"qlens[:8]={qlens[:8]} seq_lens[:8]={seq_lens_cpu[:8]}",
            flush=True,
        )

    device = query.device

    # ── 1. Prefill portion: flash-attn varlen on the prefill rows ──
    if prefill_seqs:
        prefill_rows: list[int] = []
        for i in prefill_seqs:
            prefill_rows.extend(range(qstart[i], qstart[i + 1]))
        rows_t = torch.tensor(prefill_rows, device=device, dtype=torch.long)
        q_pre = query.index_select(0, rows_t).contiguous()
        k_pre = key.index_select(0, rows_t).contiguous()
        v_pre = value.index_select(0, rows_t).contiguous()
        out_pre = torch.empty_like(q_pre)

        sub_cu = [0]
        for i in prefill_seqs:
            sub_cu.append(sub_cu[-1] + qlens[i])
        sub_cu_t = torch.tensor(sub_cu, device=device, dtype=torch.int32)
        max_q_sub = max(qlens[i] for i in prefill_seqs)
        sliding = list(impl.sliding_window) if impl.sliding_window is not None else None
        descale_shape = (len(prefill_seqs), impl.num_kv_heads)
        flash_attn_varlen_func(
            q=q_pre, k=k_pre, v=v_pre, out=out_pre,
            cu_seqlens_q=sub_cu_t, cu_seqlens_k=sub_cu_t,
            max_seqlen_q=max_q_sub, max_seqlen_k=max_q_sub,
            softmax_scale=impl.scale,
            causal=attn_metadata.causal,
            alibi_slopes=impl.alibi_slopes,
            window_size=sliding,
            softcap=impl.logits_soft_cap,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=impl.sinks,
        )
        output.index_copy_(0, rows_t, out_pre)

    # ── 2. Decode portion: sub-selected attn_metadata → try_jetfire_decode
    if decode_seqs:
        decode_q_rows = [qstart[i] for i in decode_seqs]
        rows_t = torch.tensor(decode_q_rows, device=device, dtype=torch.long)
        q_dec = query.index_select(0, rows_t).contiguous()

        decode_idx_t = torch.tensor(
            decode_seqs, device=attn_metadata.seq_lens.device, dtype=torch.long,
        )
        sub_seq_lens = attn_metadata.seq_lens.index_select(0, decode_idx_t)
        sub_block_table = attn_metadata.block_table.index_select(0, decode_idx_t)
        b_dec = len(decode_seqs)
        sub_meta = SimpleNamespace(
            seq_lens=sub_seq_lens,
            max_seq_len=int(sub_seq_lens.max().item()),
            max_query_len=1,
            num_actual_tokens=b_dec,
            block_table=sub_block_table,
            use_cascade=False,
            query_start_loc=torch.arange(
                b_dec + 1, device=device, dtype=torch.int32,
            ),
            causal=getattr(attn_metadata, "causal", True),
            max_num_splits=getattr(attn_metadata, "max_num_splits", 1),
        )
        out_dec = torch.empty_like(q_dec)
        result = try_jetfire_decode(
            impl=impl, layer=layer, query=q_dec,
            attn_metadata=sub_meta, output=out_dec,
        )
        if result is None:
            raise RuntimeError(
                "JetFire mixed-batch decode sub-call returned None. "
                f"Recent decode_fallbacks: {pool.decode_fallbacks}"
            )
        output.index_copy_(0, rows_t, out_dec)

    _flush_deferred_kv_mirror(layer)
    return output


def _jetfire_mixed_batch_attn_ordered(
    *,
    impl,
    layer,
    query,
    key,
    value,
    attn_metadata,
    output,
    num_decodes: int,
    num_prefills: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
):
    """Fast mixed path for vLLM's reordered decode-prefix/prefill-tail batch."""
    from types import SimpleNamespace

    pool = get_global_kv_pool_manager()
    pool.record_mixed_batch(
        prefill_seqs=num_prefills,
        decode_seqs=num_decodes,
        prefill_tokens=num_prefill_tokens,
        decode_tokens=num_decode_tokens,
    )

    global _MIXED_LOG_DONE
    if not _MIXED_LOG_DONE:
        _MIXED_LOG_DONE = True
        print(
            f"[jetfire] mixed-batch ordered dispatch: "
            f"prefill_seqs={num_prefills} decode_seqs={num_decodes} "
            f"decode_tokens={num_decode_tokens} prefill_tokens={num_prefill_tokens}",
            flush=True,
        )

    # Prefill requests are a contiguous tail.  Avoid per-layer index_select
    # over a 32K-token span; slicing preserves the existing contiguous storage.
    if num_prefills > 0:
        pre_start = num_decode_tokens
        pre_end = pre_start + num_prefill_tokens
        q_pre = query[pre_start:pre_end]
        k_pre = key[pre_start:pre_end]
        v_pre = value[pre_start:pre_end]
        out_pre = output[pre_start:pre_end]
        q_start_pre = (
            attn_metadata.query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - pre_start
        )
        sliding = list(impl.sliding_window) if impl.sliding_window is not None else None
        descale_shape = (num_prefills, impl.num_kv_heads)
        flash_attn_varlen_func(
            q=q_pre,
            k=k_pre,
            v=v_pre,
            out=out_pre,
            cu_seqlens_q=q_start_pre,
            cu_seqlens_k=q_start_pre,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=impl.scale,
            causal=attn_metadata.causal,
            alibi_slopes=impl.alibi_slopes,
            window_size=sliding,
            softcap=impl.logits_soft_cap,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=impl.sinks,
        )

    # Decode requests are a contiguous prefix with q_len=1 in the current
    # non-speculative path.  Keep all tensors as views so the decode graph path
    # can use pre-created contexts for the actual decode batch size.
    if num_decodes > 0:
        sub_meta = SimpleNamespace(
            seq_lens=attn_metadata.seq_lens[:num_decodes],
            max_seq_len=int(attn_metadata.max_seq_len),
            max_query_len=1,
            num_actual_tokens=num_decode_tokens,
            block_table=attn_metadata.block_table[:num_decodes],
            use_cascade=False,
            query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
            causal=getattr(attn_metadata, "causal", True),
            max_num_splits=getattr(attn_metadata, "max_num_splits", 1),
        )
        out_dec = output[:num_decode_tokens]
        result = try_jetfire_decode(
            impl=impl,
            layer=layer,
            query=query[:num_decode_tokens],
            attn_metadata=sub_meta,
            output=out_dec,
        )
        if result is None:
            raise RuntimeError(
                "JetFire ordered mixed-batch decode sub-call returned None. "
                f"Recent decode_fallbacks: {pool.decode_fallbacks}"
            )

    _flush_deferred_kv_mirror(layer)
    return output


# ── Decode gating ──────────────────────────────────────────────────────
def _decode_unsupported(*, impl, layer, query, attn_metadata) -> str | None:
    """One-shot dispatcher gate.  Returns reason string on miss, None on hit."""
    pool = get_global_kv_pool_manager()
    if not pool.config.enabled:               return "pool_disabled"
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        # Graph mode opt-in: when JETFIRE_DECODE_GRAPH=1 the decode body uses
        # stream-level CPU/GPU handshake (cudaLaunchHostFunc + cuStream
        # WaitValue64), so it CAN be captured into a CUDA graph.  Without
        # this env, fall back to flash-attn to preserve the existing
        # capture-time behavior.
        if not _graph_mode_enabled():
            return "cuda_graph_capture"
    if attn_metadata is None:                  return "profiling_run"
    if impl.attn_type.name != "DECODER":      return "attn_type"
    if attn_metadata.use_cascade:              return "cascade"
    if int(attn_metadata.max_query_len) != 1: return "prefill_or_multi_query"
    n_seqs = int(attn_metadata.seq_lens.numel())
    if int(attn_metadata.num_actual_tokens) != n_seqs:
        return "batch_mismatch"
    if int(query.shape[0]) < n_seqs:           return "empty_query"
    layer_name = getattr(layer, "layer_name", None)
    if layer_name is None:                     return "layer_name_missing"
    layer_cfg = pool.get_layer_config(layer_name)
    if layer_cfg is None:                      return "layer_config_missing"
    if int(query.shape[-1]) != layer_cfg.head_dim:
        return "head_dim_mismatch"
    if (layer_cfg.head_dim % pool.config.group_size) != 0:
        return "group_size_mismatch"
    return None


# ── Decode (batched correction) ───────────────────────────────────────
def try_jetfire_decode(*, impl, layer, query, attn_metadata, output,
                        key=None, value=None):
    """Production decode entry.

    Returns ``output`` on success, or ``None`` if any precondition fails
    (caller falls back to native flash-attn).  Works at any B ≥ 1; B=1 is
    just the degenerate case of the batched kernels.
    """
    with _timer("D_total"):
        return _decode_batched(impl=impl, layer=layer, query=query,
                                output=output, attn_metadata=attn_metadata)


def _decode_batched(*, impl, layer, query, attn_metadata, output):
    pool = get_global_kv_pool_manager()
    pool.record_decode_attempt()
    layer_name = getattr(layer, "layer_name", "unknown_layer")
    trace_id = _next_trace_decode_id()
    _stage_trace(trace_id, "enter", layer_name=layer_name)
    _step_trace(trace_id, "enter", layer_name=layer_name)
    pool.ensure_layer_config_from_layer(layer_name, layer)

    reason = _decode_unsupported(impl=impl, layer=layer, query=query,
                                  attn_metadata=attn_metadata)
    if reason is not None:
        pool.record_decode_fallback(reason)
        if _try_graph_profile_placeholder(pool=pool, output=output):
            return output
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss(reason, layer_name=layer_name)
        return None

    if not get_jetfire_config().cpu_correction:
        pool.record_decode_fallback("cpu_correction_disabled")
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss("cpu_correction_disabled", layer_name=layer_name)
        return None

    layer_cfg = pool.get_layer_config(layer_name)
    store = pool.get_layer_store(layer_name)
    if store is None:
        pool.record_decode_fallback("store_missing")
        if _try_graph_profile_placeholder(pool=pool, output=output):
            return output
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss("store_missing", layer_name=layer_name)
        return None

    # Pipelined prefill may have INT writes / shadow D2H still in flight.
    _stage_trace(trace_id, "flush_pending_enter", layer_name=layer_name)
    store.flush_pending_writes()
    _stage_trace(trace_id, "flush_pending_done", layer_name=layer_name)

    n_seqs = int(attn_metadata.seq_lens.numel())
    # Graph mode: skip the D2H+`.item()` sanity check (it would block stream
    # capture).  Caller is responsible for ensuring shadow KV is populated
    # before invoking decode in graph mode.
    if not _graph_mode_enabled():
        _stage_trace(trace_id, "seq_check_enter", layer_name=layer_name)
        seq_lens_cpu = attn_metadata.seq_lens.to("cpu", dtype=torch.long)
        _stage_trace(trace_id, "seq_lens_cpu_done", layer_name=layer_name)
        if os.environ.get("JETFIRE_FAST_STORE_COMPLETE_CHECK", "0") == "1":
            store_complete = store.fast_has_sequence(int(seq_lens_cpu.max().item()))
        else:
            store_complete = True
            for seq_idx, seq_len in enumerate(seq_lens_cpu.tolist()):
                _stage_trace(
                    trace_id,
                    "has_sequence_enter",
                    layer_name=layer_name,
                    n_seqs=n_seqs,
                    max_S=int(seq_len),
                )
                if not store.has_sequence(attn_metadata.block_table[seq_idx], int(seq_len)):
                    store_complete = False
                    break
                _stage_trace(
                    trace_id,
                    "has_sequence_done",
                    layer_name=layer_name,
                    n_seqs=n_seqs,
                    max_S=int(seq_len),
                )
        if not store_complete:
            pool.record_decode_fallback("store_incomplete")
            return None
        _stage_trace(trace_id, "seq_check_done", layer_name=layer_name)

    try:
        _stage_trace(trace_id, "load_kernel_set_enter", layer_name=layer_name)
        kernel_set = _load_kernel_set()
        _stage_trace(trace_id, "load_kernel_set_done", layer_name=layer_name)
    except Exception:
        pool.record_decode_fallback("ext_load_failed")
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss("ext_load_failed", layer_name=layer_name)
        return None
    _stage_trace(trace_id, "load_gather_enter", layer_name=layer_name)
    _load_gather_ext()
    _stage_trace(trace_id, "load_gather_done", layer_name=layer_name)

    device  = query.device
    H_q, H_kv, D = layer_cfg.num_query_heads, layer_cfg.num_kv_heads, layer_cfg.head_dim
    num_groups = H_q // H_kv
    scale   = getattr(impl, "scale", 1.0 / math.sqrt(D))
    max_S   = int(attn_metadata.max_seq_len)
    top_k   = _select_top_k(pool.config.topk, max_S, kernel_set.supported_top_k)
    _stage_trace(
        trace_id, "shape_ready", layer_name=layer_name,
        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
    )
    _step_trace(
        trace_id, "shape_ready", layer_name=layer_name,
        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        extra=f"Hq={H_q} Hkv={H_kv} D={D}",
    )
    if top_k is None:
        pool.record_decode_fallback("seq_len_too_short")
        graph_profile_short = (
            _graph_mode_enabled()
            and (
                _graph_capture_active()
                or int(getattr(store, "written_count", 0)) == 0
                or max_S <= max(2, int(get_jetfire_config().max_batch_size))
            )
        )
        if graph_profile_short:
            pool.record_decode_fallback("graph_profile_placeholder")
            if not _graph_capture_active():
                output.zero_()
            return output
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss("seq_len_too_short", layer_name=layer_name)
        return None
    bs      = int(store.block_size)

    with _timer("V2_00_prep"):
        _stage_trace(
            trace_id, "prep_enter", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )
        q_batch        = query[:n_seqs].contiguous().view(n_seqs, H_q, D)
        bt_batch       = attn_metadata.block_table[:n_seqs].to(torch.int64)
        seq_lens_batch = attn_metadata.seq_lens.to(torch.int32)
        _stage_trace(
            trace_id, "prep_done", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )

    # Resolve the host-buffer batch dim: when JETFIRE_MAX_BATCH_SIZE is set,
    # all pinned/mapped buffers below are sized to that constant and we use
    # ``[:n_seqs]`` slices.  Trip the warmup on first decode if it hasn't run
    # yet — by this point H_q, H_kv, D, top_k are all final, so the warmup
    # has the shape it needs.
    cfg = get_jetfire_config()
    cfg_max_B = int(cfg.max_batch_size)
    buf_B = _resolve_buf_B(n_seqs, cfg_max_B)
    release_plan: ReleaseKernelPlan | None = None
    if release_policy_enabled():
        validate_release_environment()
        release_plan = plan_release_graph(
            mode=cfg.scheme.name,
            group_size=pool.config.group_size,
            top_k=top_k,
            head_dim=D,
            num_query_heads=H_q,
            num_kv_heads=H_kv,
            max_graph_batch=cfg_max_B if cfg_max_B > 0 else n_seqs,
            require_ready=release_require_ready(),
        )
        _stage_trace(
            trace_id, "release_policy_ready", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=(
                f"A={release_plan.stage_a} B={release_plan.stage_b} "
                f"ready={int(release_plan.release_ready)}"
            ),
        )
    warmup_key = _decode_buffer_warmup_key(
        max_batch_size=cfg_max_B,
        H_q=H_q,
        H_kv=H_kv,
        D=D,
        top_k=top_k,
        device=device if _graph_mode_enabled() else None,
    )
    if cfg_max_B > 0 and warmup_key not in _BUFFER_WARMUP_KEYS and buf_B == cfg_max_B:
        if _graph_mode_enabled() and _graph_capture_active():
            _raise_graph_decode_miss(
                f"decode_buffers_not_warmed_topk_{top_k}",
                layer_name=layer_name,
            )
        warmup_decode_buffers(
            max_batch_size=cfg_max_B,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            top_k=top_k,
            device=device if _graph_mode_enabled() else None,
        )

    # 1. Batched INT attention with scores output.
    # Split mode: only run Stage A (Q·K → scores) here; Stage B
    # (softmax·V → o_hat) is queued AFTER V2_02 so it overlaps with the
    # CPU worker's cpu_topk_attn instead of running before it.
    #
    # Default ON since v3 (split-S + SMEM staging in Stage B): Stage B
    # microbench at B=4, max_S=4096 dropped from 1072 µs (v1) → 59 µs (v3),
    # putting the split path within ~5% of the fused path while delivering
    # the architecturally-intended CPU/GPU V-cache overlap.  Set
    # JETFIRE_USE_SPLIT_INT2=0 to restore the legacy fused path.
    #
    # Despite the historical env name, this now covers all split-capable
    # schemes: AsymInt2K/SymInt2V plus symmetric INT3/INT4 and mixed K/V
    # bit-widths.  The split Stage-A topK launcher is radix-select only in the
    # release path; older small-K register-heap variants are not dispatchable.
    scheme = get_jetfire_config().scheme
    gqa_ratio = H_q // H_kv if H_kv > 0 and H_q % H_kv == 0 else 0
    if release_plan is not None:
        use_split_decode = True
    else:
        use_split_decode = (
            os.environ.get("JETFIRE_USE_SPLIT_INT2", "1") == "1"
            and D == 128
            and bs == 16
            and gqa_ratio > 0
            and gqa_ratio <= 16
            and scheme.k_nbits in (2, 3, 4)
            and scheme.v_nbits in (2, 3, 4)
        )
    # Pattern A v0: single fused kernel replaces Stage A + Stage B.  No
    # (B, H_q, max_S) scores/weights tensors in HBM.  v0 is scalar-fp32
    # inside the kernel, so on small ctx the WMMA split path is faster;
    # at long ctx (where the HBM-write of scores dominates) v0 should win.
    # The optimised WMMA + cp.async port is the next step; for now the
    # flag is OFF by default and exists for unit-test + bring-up.
    use_fused_attn_topk = (
        release_plan is None
        and os.environ.get("JETFIRE_USE_FUSED_ATTN_TOPK", "0") == "1"
        and H_kv == 8 and D == 128 and bs == 16
        # v0 currently sized for top_k ≤ K_MAX=128 in the kernel
    )
    softmax_m_gpu = None
    softmax_inv_z_gpu = None
    topk_idx_gpu = None
    topk_slots_gpu = None
    with _timer("V2_01_attn"):
        _stage_trace(
            trace_id, "stageA_enter", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=(
                f"split={int(use_split_decode)} "
                f"scheme={scheme.name} Hq={H_q} Hkv={H_kv} "
                f"D={D} bs={bs} gqa={gqa_ratio}"
            ),
        )
        if use_fused_attn_topk:
            # Kernel choice: v1 (WMMA + cp.async) by default; v0 (scalar
            # reference) when JETFIRE_FUSED_KERNEL=v0 or when v1's stricter
            # constraints aren't met (top_k % 32 != 0, num_groups not in
            # {1,2,4}).
            _kernel_pref = os.environ.get("JETFIRE_FUSED_KERNEL", "v1")
            _v1_ok = (
                _kernel_pref == "v1"
                and (top_k % 32 == 0) and 32 <= top_k <= 128
                and num_groups in (1, 2, 4)
            )
            if _v1_ok:
                (
                    o_hat_full,
                    topk_slots_gpu,
                    o_old_unnorm_split,
                    W_split,
                ) = _load_fused_attn_topk_v1_ext().int2_fused_attn_topk_v1(
                    q_batch, store.k_data, store.k_scales, store.k_zp,
                    store.v_data, store.v_scales,
                    bt_batch, seq_lens_batch,
                    max_S, H_q, H_kv, bs, top_k, float(scale),
                )
            else:
                (
                    o_hat_full,
                    topk_slots_gpu,
                    o_old_unnorm_split,
                    W_split,
                ) = _load_fused_attn_topk_ext().int2_fused_attn_topk(
                    q_batch, store.k_data, store.k_scales, store.k_zp,
                    store.v_data, store.v_scales,
                    bt_batch, seq_lens_batch,
                    max_S, H_q, H_kv, bs, top_k, float(scale),
                )
            # Match the variable layout the rest of _decode_batched expects:
            # o_hat is fp32 here (will go straight into the blend kernel),
            # topk_idx_gpu is unused (we already have translated slots), so
            # set to None — no consumer downstream when use_fused_attn_topk.
            o_hat = o_hat_full
            scores = None
            topk_idx_gpu = None
        elif use_split_decode:
            # Stage A v4: Q·K → scores AND in-kernel mask + top-K + slot
            # translate.  Replaces the 9-op Python chain that V2_02 used to
            # do (mask, topk, //bs, %bs, gather, mul, add).
            _ext_qk = _load_qk_scores_ext()
            _k_nbits = get_jetfire_config().scheme.k_nbits
            # Current main Stage A path for the best measured decode setup:
            # int8tc QK + external softmax stats. Experimental fused-topk
            # variants remain explicit opt-in via env.
            _int8tc_supported_topk = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
            if release_plan is not None:
                _use_qk_stats = release_plan.stage_a == "qk_int8tc_stats"
                _use_int8tc = release_plan.stage_a == "qk_int8tc_stats"
            else:
                _use_qk_stats = os.environ.get("JETFIRE_INT4_QK_STATS", "1") == "1"
                _use_int8tc = (
                    _k_nbits == 4
                    and not get_jetfire_config().scheme.k_asymmetric
                    # The INT8 tensor-core QK kernel stores one K scale per
                    # slot.  group_size < head_dim produces multiple K scale
                    # groups per slot, so route those configs through the native
                    # INT4 QK kernel which supports num_groups={1,2}.
                    and int(store.k_scales.shape[-1]) == 1
                    and os.environ.get("JETFIRE_USE_INT8_TC", "1") == "1"
                    and top_k in _int8tc_supported_topk
                )
            if _use_int8tc:
                from jetfire_vllm.kernels.int8tc_qk_scores import (
                    load_int8tc_qk_scores_ext,
                )
                _ext_int8tc = load_int8tc_qk_scores_ext()
                if _use_qk_stats:
                    _want_group_fused_topk = (
                        os.environ.get("JETFIRE_INT8TC_QK_GROUP_FUSED_TOPK", "0") == "1"
                    )
                    _want_fused_topk = (
                        os.environ.get("JETFIRE_INT8TC_QK_FUSED_TOPK", "0") == "1"
                    )
                    _allow_experimental_topk = (
                        os.environ.get("JETFIRE_ALLOW_EXPERIMENTAL_QK_TOPK", "0") == "1"
                    )
                    if (
                        (_want_group_fused_topk or _want_fused_topk)
                        and not _allow_experimental_topk
                    ):
                        raise RuntimeError(
                            "JETFIRE_INT8TC_QK_FUSED_TOPK and "
                            "JETFIRE_INT8TC_QK_GROUP_FUSED_TOPK are experimental "
                            "Stage-A topK paths. They are not release-safe because "
                            "microbench validation found topK-set drift and/or slower "
                            "latency. Unset these env vars for the CUDA graph release "
                            "path, or set JETFIRE_ALLOW_EXPERIMENTAL_QK_TOPK=1 only "
                            "for isolated kernel debugging."
                        )
                    if (
                        _want_group_fused_topk
                        and (H_q // H_kv) <= 4
                    ):
                        (
                            scores,
                            topk_idx_gpu,
                            topk_slots_gpu,
                            softmax_m_gpu,
                            softmax_inv_z_gpu,
                        ) = _ext_int8tc.int8tc_qk_scores_topk_stats_group_fused(
                                q_batch, store.k_data, store.k_scales,
                                bt_batch, seq_lens_batch,
                                max_S, H_q, H_kv, bs, float(scale), top_k,
                            )
                    elif (
                        _want_fused_topk
                        and (H_q // H_kv) <= 4
                    ):
                        (
                            scores,
                            topk_idx_gpu,
                            topk_slots_gpu,
                            softmax_m_gpu,
                            softmax_inv_z_gpu,
                        ) = _ext_int8tc.int8tc_qk_scores_topk_stats_fused(
                                q_batch, store.k_data, store.k_scales,
                                bt_batch, seq_lens_batch,
                                max_S, H_q, H_kv, bs, float(scale), top_k,
                            )
                    else:
                        (
                            scores,
                            topk_idx_gpu,
                            topk_slots_gpu,
                            softmax_m_gpu,
                            softmax_inv_z_gpu,
                        ) = _ext_int8tc.int8tc_qk_scores_topk_stats(
                                q_batch, store.k_data, store.k_scales,
                                bt_batch, seq_lens_batch,
                                max_S, H_q, H_kv, bs, float(scale), top_k,
                            )
                else:
                    scores, topk_idx_gpu, topk_slots_gpu = (
                        _ext_int8tc.int8tc_qk_scores_topk(
                            q_batch, store.k_data, store.k_scales,
                            bt_batch, seq_lens_batch,
                            max_S, H_q, H_kv, bs, float(scale), top_k,
                        )
                    )
            elif _k_nbits == 4:
                # INT4 sym K: no k_zp argument.
                if _use_qk_stats:
                    (
                        scores,
                        topk_idx_gpu,
                        topk_slots_gpu,
                        softmax_m_gpu,
                        softmax_inv_z_gpu,
                    ) = _ext_qk.int4_qk_scores_topk_stats(
                            q_batch, store.k_data, store.k_scales,
                            bt_batch, seq_lens_batch,
                            max_S, H_q, H_kv, bs, float(scale), top_k,
                        )
                else:
                    scores, topk_idx_gpu, topk_slots_gpu = (
                        _ext_qk.int4_qk_scores_topk(
                            q_batch, store.k_data, store.k_scales,
                            bt_batch, seq_lens_batch,
                            max_S, H_q, H_kv, bs, float(scale), top_k,
                        )
                    )
            elif _k_nbits == 3:
                # INT3 sym K: no k_zp argument.
                scores, topk_idx_gpu, topk_slots_gpu = (
                    _ext_qk.int3_qk_scores_topk(
                        q_batch, store.k_data, store.k_scales,
                        bt_batch, seq_lens_batch,
                        max_S, H_q, H_kv, bs, float(scale), top_k,
                    )
                )
            else:
                k_zp_for_qk = store.k_zp
                if k_zp_for_qk is None:
                    # Symmetric INT2 K uses the same offset-packed INT2
                    # representation as the asymmetric kernel but has no
                    # additive zero-point.  Pass a zero-length sentinel so the
                    # C++ wrapper routes the CUDA kernel through zp=0 without
                    # materializing a full k_zp tensor in graph replay.
                    k_zp_for_qk = _get_device(
                        (0,),
                        torch.bfloat16,
                        _graph_store_tag(store, "int2_sym_k_zp_empty"),
                        device,
                    )
                scores, topk_idx_gpu, topk_slots_gpu = (
                    _ext_qk.int2_qk_scores_topk(
                        q_batch, store.k_data, store.k_scales, k_zp_for_qk,
                        bt_batch, seq_lens_batch,
                        max_S, H_q, H_kv, bs, float(scale), top_k,
                    )
                )
            o_hat = None   # placeholder; computed by Stage B after V2_02
        else:
            o_hat, scores = kernel_set.attn_with_scores(
                q_batch, store.k_data, store.k_scales, store.k_zp,
                store.v_data, store.v_scales, bt_batch, seq_lens_batch,
                max_S, H_q, H_kv, bs, float(scale),
            )
            # o_hat: (B, H_q, D) fp32 ; scores: (B, H_q, max_S) fp32
        _stage_trace(
            trace_id, "stageA_done", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=(
                f"release_stage_a={release_plan.stage_a}"
                if release_plan is not None else ""
            ),
        )
        _step_trace(
            trace_id, "stageA_done", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=(
                f"split={int(use_split_decode)} scheme={scheme.name} "
                + (
                    f"release_stage_a={release_plan.stage_a}"
                    if release_plan is not None else "release_stage_a="
                )
            ),
        )

    # Accuracy-only k=0 ablation.  It executes the complete production graph,
    # including the CPU join and correction blend, then overwrites the visible
    # output with Stage B's o_hat.  Keeping every graph dependency intact makes
    # this a controlled output-fidelity ablation; its latency is meaningless.
    debug_approx_only = (
        os.environ.get("JETFIRE_DEBUG_APPROX_ONLY", "0") == "1"
    )

    # 2. Mask + top-K + slot translate.  In split mode (and the new fused
    # mode) Stage A already produced topk_idx_gpu / topk_slots_gpu, so this
    # is a no-op there.  Legacy WMMA path still needs the Python chain.
    with _timer("V2_02_topk_d2h_fork"):
        _stage_trace(
            trace_id, "d2h_enter", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )
        if (not use_fused_attn_topk) and (
            (not use_split_decode) or topk_slots_gpu is None
        ):
            pos = torch.arange(max_S, device=device, dtype=torch.int32).unsqueeze(0)
            mask = pos < seq_lens_batch.unsqueeze(1)              # (B, max_S)
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))
            _, topk_idx_gpu = scores.topk(top_k, dim=-1)          # (B, H_q, K)
            block_idx = topk_idx_gpu // bs
            in_block  = topk_idx_gpu %  bs
            bt_exp    = bt_batch.unsqueeze(1).expand(-1, H_q, -1)
            topk_slots_gpu = torch.gather(bt_exp, -1, block_idx) * bs + in_block

        # Compute whether downstream paths need the comm stream.  The
        # production default (cpu_attn_mode + use_mapped_o_new) does not
        # — the CPU worker reads pinned q/ts after main has landed the
        # D2H, then writes mapped o_new which the GPU reads via PCIe.
        # Two legacy variants still need comm:
        #   * use_mapped_o_new=False: worker enqueues H2D of o_new on comm.
        #   * cpu_attn_mode=False (GPU-only correction): worker H2Ds the
        #     full K_topk / V_topk on comm.
        #
        # JF_USE_COMM_STREAM=1 force-enables the legacy two-stream path
        # for benchmarking at small ctx.  Don't set this at ctx≥32k —
        # the per-layer cross-stream event chain saturates the CUDA
        # driver queue and deadlocks.
        _cpu_attn_mode_local = get_jetfire_config().decode_cpu_attn
        _mapped_local = (
            _graph_use_mapped_o_new(n_seqs)
            if _graph_mode_enabled()
            else os.environ.get("JETFIRE_MAPPED_O_NEW", "1") != "0"
        )
        _need_comm = (
            os.environ.get("JF_USE_COMM_STREAM", "0") == "1"
            or (not _cpu_attn_mode_local)
            or ((not _mapped_local) and (not _graph_mode_enabled()))
        )

        # D2H of q + topk_slots.  When prealloc is on, the underlying tensors
        # are sized at ``buf_B`` and we slice; the slice shares storage so
        # ``copy_(non_blocking=True)`` lands at a stable host address.
        q_tag = _graph_store_tag(store, "q") if _graph_mode_enabled() else "q"
        ts_tag = (
            _graph_store_tag(store, "topk_slots")
            if _graph_mode_enabled()
            else "topk_slots"
        )
        q_buf_full  = _get_pinned((buf_B, H_q, D),     torch.bfloat16, q_tag)
        ts_buf_full = _get_pinned((buf_B, H_q, top_k), torch.int64,    ts_tag)
        q_buf  = q_buf_full[:n_seqs]
        ts_buf = ts_buf_full[:n_seqs]
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print(f"[jetfire][cpu_async] py_d2h_start need_comm={_need_comm} cpu_attn={_cpu_attn_mode_local}", flush=True)
        if _need_comm:
            comm = _get_comm_stream()
            comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm):
                q_buf.copy_(q_batch, non_blocking=True)
                ts_buf.copy_(topk_slots_gpu, non_blocking=True)
        else:
            q_buf.copy_(q_batch, non_blocking=True)
            ts_buf.copy_(topk_slots_gpu, non_blocking=True)
            # Graph mode: skip host synchronize.  The cudaLaunchHostFunc
            # callback that submits CPU work is stream-ordered AFTER the
            # D2H copies above, so the CPU side sees fully-landed q/ts_buf
            # when it begins compute.  In eager mode we still synchronize
            # so the legacy submit_cpu_topk_attn_async path observes the
            # same invariant (Python ThreadPoolExecutor doesn't honor CUDA
            # stream order).
            if not _graph_mode_enabled():
                torch.cuda.current_stream().synchronize()
            comm = None
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_d2h_done", flush=True)
        _stage_trace(
            trace_id, "d2h_done", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=f"need_comm={_need_comm}",
        )
        _step_trace(
            trace_id, "d2h_done", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            extra=f"need_comm={int(_need_comm)}",
        )

    # 3. Worker thread: split path based on JETFIRE_DECODE_CPU_ATTN.
    head_map_cpu_full = _get_head_map_cpu(buf_B, H_q, num_groups, top_k)
    head_map_cpu = head_map_cpu_full[:n_seqs]
    cpu_attn_mode = get_jetfire_config().decode_cpu_attn

    dev_idx   = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    k_shadow  = store.k_shadow
    v_shadow  = store.v_shadow

    # Default values for branches below — both legacy GPU-correction path
    # (cpu_attn_mode=False) and persistent-worker path skip mapped-o_new.
    use_mapped_o_new = False
    use_persistent_worker = False
    use_fused_join_blend = False
    use_graph_submit = False
    use_sync_cpu_attn = False
    cpu_done_inline = False
    cpu_flag = None
    _flag_dev = None

    if cpu_attn_mode:
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_cpu_path", flush=True)
        # CPU compute path: CPU does gather + Q·K^T·softmax·V → o_new.
        # Two storage modes:
        #   - default (mapped):       cudaHostAllocMapped (zero-copy, +3-4% e2e)
        #   - JETFIRE_MAPPED_O_NEW=0: legacy pinned host buf + explicit H2D
        # The mapped variant skips V2_04c_worker_h2d entirely (~82 µs/layer):
        # GPU reads o_new directly from the mapped pinned buffer over PCIe
        # in the blend kernel.  cpu_topk_attn issues _mm_sfence() before
        # returning so the write-combined CPU stores are flushed.
        use_mapped_o_new = (
            _graph_use_mapped_o_new(n_seqs)
            if _graph_mode_enabled()
            else os.environ.get("JETFIRE_MAPPED_O_NEW", "1") != "0"
        )
        # Fused V2_04+V2_05 via blend ext.  Worker writes a 1-element atomic
        # flag after cpu_topk_attn; main thread spin-waits in C++ instead of
        # going through Future.result()'s threading.Event (cv) path.  Saves
        # ~22 µs/layer cv-wakeup at B=4 ctx=4096 (+2.2% TPOT measured).
        #
        # **Default OFF.**  At long ctx + larger B (verified: B=16 ctx=32k),
        # the busy-spin in C++ deadlocks against the ThreadPoolExecutor
        # worker — main thread holds 100% of a core while the worker can't
        # progress through `comm.synchronize()` / `cpu_topk_attn`, so
        # `cpu_flag` never transitions to 1 and main spins forever.  The
        # cv-wait path (Future.result()) yields the core and avoids the
        # interaction.  Set JETFIRE_FUSED_JOIN_BLEND=1 to opt back in for
        # ctx≤4k workloads where the +2% holds and there is no hang.
        use_fused_join_blend = (
            use_split_decode
            and use_mapped_o_new
            and output.dtype == torch.bfloat16
            and os.environ.get("JETFIRE_USE_BLEND_KERNEL", "0") == "1"
            and os.environ.get("JETFIRE_FUSED_JOIN_BLEND", "0") == "1"
        )

        if use_mapped_o_new:
            o_tag = (
                _graph_store_tag(store, "o_new")
                if _graph_mode_enabled()
                else "o_new"
            )
            o_new_pin_full, o_new_gpu_full = _get_mapped_pair(
                (buf_B, H_q, D), torch.bfloat16, o_tag
            )
            o_new_pin = o_new_pin_full[:n_seqs]
            o_new_gpu = o_new_gpu_full[:n_seqs]
        else:
            o_tag = (
                _graph_store_tag(store, "o_new_graph_h2d")
                if _graph_mode_enabled()
                else "o_new"
            )
            o_new_pin_full = _get_pinned(
                (buf_B, H_q, D),
                torch.bfloat16,
                o_tag,
            )
            o_new_pin = o_new_pin_full[:n_seqs]
            if _graph_mode_enabled():
                o_new_gpu_full = _get_device(
                    (buf_B, H_q, D),
                    torch.bfloat16,
                    o_tag,
                    device,
                )
                o_new_gpu = o_new_gpu_full[:n_seqs]
            else:
                o_new_gpu = torch.empty(
                    (n_seqs, H_q, D), dtype=torch.bfloat16, device=device
                )

        # Cached completion flag (1-elem int64 pinned).  Reset to 0 before
        # each submit; worker stores 1 with release ordering inside
        # cpu_topk_attn after _mm_sfence().  Always allocated when the
        # persistent-worker path is on; only when ``use_fused_join_blend``
        # otherwise (the original codepath).
        #
        # Default OFF: at production B=32/ctx=32k the combination of the
        # persistent C++ dispatcher and the custom CPU pool is not stable and
        # can hang in decode after TTFT.  The legacy ThreadPoolExecutor path
        # stays on the successful TPOT envelope while avoiding that deadlock.
        # Set ``JETFIRE_USE_PERSISTENT_WORKER=1`` only for focused A/B runs.
        use_persistent_worker = (
            use_mapped_o_new
            and (not _need_comm)
            and os.environ.get("JETFIRE_USE_PERSISTENT_WORKER", "0") == "1"
        )
        # Graph-mode submission supersedes the persistent-worker path: it
        # uses cudaLaunchHostFunc plus a capturable stream-side completion wait
        # so the CPU submit is captureable.  o_new can be either mapped
        # zero-copy or regular pinned host memory followed by a captured H2D
        # copy.
        use_graph_submit = (
            (not _need_comm)
            and _graph_mode_enabled()
        )
        # Stability override for long-context serving.  The release default
        # keeps the async ThreadPoolExecutor path so CPU correction can overlap
        # Stage B.  The join below is guarded by a timeout, so an executor stall
        # becomes a clear EngineCore error instead of a silent hang.  Set
        # JETFIRE_SYNC_CPU_ATTN=1 to force the slower inline path when debugging
        # CPU-worker issues.
        use_sync_cpu_attn = (
            (not use_graph_submit)
            and (not use_persistent_worker)
            and os.environ.get("JETFIRE_SYNC_CPU_ATTN", "0") == "1"
        )
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print(
                f"[jetfire][cpu_async] py_mode mapped={use_mapped_o_new} "
                f"need_comm={_need_comm} persistent={use_persistent_worker} "
                f"graph={use_graph_submit} sync={use_sync_cpu_attn}",
                flush=True,
            )
        if use_graph_submit:
            # Release graph mode uses async submit + side-stream event join.  CPU
            # topK still overlaps Stage B: submit dispatches the C++ pool job
            # and returns before Stage B; the wait runs only at the correction
            # boundary.  Device-signal, host-join, and mapped-host waits are
            # explicit A/B paths gated by JETFIRE_ALLOW_EXPERIMENTAL_GRAPH_CPU_TOPK.
            apply_graph_cpu_topk_release_preset()
            cpu_flag, _mapped_flag_dev = _ensure_graph_flag()
            _use_device_signal = (
                os.environ.get("JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK", "0") == "1"
            )
            _use_side_join = (
                os.environ.get("JETFIRE_GRAPH_SIDE_JOIN_CPU_TOPK", "0") == "1"
                and os.environ.get("JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK", "0") != "1"
                and os.environ.get("JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK", "0") != "1"
                and not _use_device_signal
            )
            if _use_device_signal:
                _flag_dev = _ensure_graph_signal_dev(
                    torch.device("cuda", torch.cuda.current_device())
                )
            else:
                _flag_dev = _mapped_flag_dev
            # Don't reset the flag here — the reset is a stream op queued
            # right before stream_submit_cpu_topk_attn (see V2_02c block).
        elif use_fused_join_blend or use_persistent_worker:
            cpu_flag = _get_pinned((1,), torch.int64, "cpu_flag")
            cpu_flag.zero_()
        else:
            cpu_flag = None

        def _worker_cpu_attn() -> None:
            torch.cuda.set_device(dev_idx)
            with torch.inference_mode():
                if comm is not None:
                    with _timer("V2_04a_worker_commsync"):
                        _stage_trace(
                            trace_id, "cpu_comm_sync_enter",
                            layer_name=layer_name, n_seqs=n_seqs,
                            max_S=max_S, top_k=top_k,
                        )
                        comm.synchronize()    # wait q + topk_slots D2H
                        _stage_trace(
                            trace_id, "cpu_comm_sync_done",
                            layer_name=layer_name, n_seqs=n_seqs,
                            max_S=max_S, top_k=top_k,
                        )
                with _timer("V2_04b_worker_cpu_attn"):
                    _stage_trace(
                        trace_id, "cpu_topk_enter", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                    )
                    _step_trace(
                        trace_id, "cpu_topk_enter", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                    )
                    _load_cpu_topk_attn_ext().cpu_topk_attn(
                        q_buf, k_shadow, v_shadow, head_map_cpu, ts_buf,
                        o_new_pin, float(scale),
                        cpu_flag,
                    )
                    _stage_trace(
                        trace_id, "cpu_topk_done", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                    )
                    _step_trace(
                        trace_id, "cpu_topk_done", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                    )
                if not use_mapped_o_new:
                    with _timer("V2_04c_worker_h2d"):
                        with torch.cuda.stream(comm):
                            o_new_gpu.copy_(o_new_pin, non_blocking=True)
                # Mapped path: o_new_gpu IS o_new_pin (same physical bytes,
                # different virtual address) — GPU sees the writes via PCIe
                # immediately after the cpu_topk_attn _mm_sfence.

        with _timer("V2_02c_tpe_submit"):
            _stage_trace(
                trace_id, "submit_enter", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            )
            if use_graph_submit:
                # Graph-mode submission: cuStreamWriteValue64 resets the
                # mapped-pinned doorbell, then cudaLaunchHostFunc queues
                # the CPU correction job to jetfire_pool.  Both ops are
                # stream-ordered after the D2H of q/ts_buf above, so the
                # CPU side observes fully-landed data.  Each layer creates
                # its submit ctx lazily and reuses across replays.
                ext = _load_cpu_topk_attn_ext()
                submit_ctx_handle = _get_graph_submit_ctx(
                    store=store,
                    n_seqs=n_seqs,
                    buf_B=buf_B,
                    H_q=H_q,
                    num_groups=num_groups,
                    top_k=top_k,
                    D=D,
                    scale=float(scale),
                    mapped_o_new=use_mapped_o_new,
                    signal_device=_flag_dev if _use_device_signal else None,
                    device=device,
                )
                stream_id = torch.cuda.current_stream().cuda_stream
                if _use_side_join:
                    ext.stream_submit_cpu_topk_attn_side_join(
                        stream_id, submit_ctx_handle
                    )
                else:
                    ext.stream_write_value64(stream_id, _flag_dev, 0)
                    ext.stream_submit_cpu_topk_attn(stream_id, submit_ctx_handle)
                cpu_future = None
            elif use_persistent_worker:
                if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                    print("[jetfire][cpu_async] py_submit", flush=True)
                # All-C++ async CPU executor: main publishes the correction
                # job directly to a fixed worker set and later waits only on
                # ``cpu_flag``.  No Python ThreadPoolExecutor and no nested
                # C++ dispatcher that blocks on a second pool.
                _load_cpu_topk_attn_ext().submit_cpu_topk_attn_async(
                    q_buf, k_shadow, v_shadow, head_map_cpu, ts_buf,
                    o_new_pin, float(scale), cpu_flag,
                )
                cpu_future = None    # marker: all-C++ async path, not Executor
            elif use_sync_cpu_attn:
                if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                    print("[jetfire][cpu_async] py_sync_cpu_attn", flush=True)
                _worker_cpu_attn()
                cpu_done_inline = True
                cpu_future = None
            else:
                cpu_future = _get_gather_exec().submit(_worker_cpu_attn)
            _stage_trace(
                trace_id, "submit_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=(
                    f"sync={use_sync_cpu_attn} graph={use_graph_submit} "
                    f"persistent={use_persistent_worker}"
                ),
            )
            _step_trace(
                trace_id, "submit_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=(
                    f"sync={int(use_sync_cpu_attn)} "
                    f"graph={int(use_graph_submit)} "
                    f"persistent={int(use_persistent_worker)}"
                ),
            )
    else:
        # Legacy path: gather K_topk/V_topk on CPU, H2D the MB-scale K/V,
        # then GPU computes Q·K_topk·softmax·V_topk inside post_correction.
        k_topk_pin   = _get_pinned((n_seqs, H_q, top_k, D), torch.bfloat16, "k_topk")
        v_topk_pin   = _get_pinned((n_seqs, H_q, top_k, D), torch.bfloat16, "v_topk")
        k_topk_gpu   = torch.empty((n_seqs, H_q, top_k, D), dtype=torch.float32, device=device)
        v_topk_gpu   = torch.empty((n_seqs, H_q, top_k, D), dtype=torch.float32, device=device)

        def _worker_gpu_attn() -> None:
            torch.cuda.set_device(dev_idx)
            with torch.inference_mode():
                comm.synchronize()    # wait q + topk_slots D2H
                _load_gather_ext().gather_kv_topk_batched(
                    k_shadow, v_shadow, head_map_cpu, ts_buf, k_topk_pin, v_topk_pin,
                )
                with torch.cuda.stream(comm):
                    k_topk_gpu.copy_(k_topk_pin, non_blocking=True)
                    v_topk_gpu.copy_(v_topk_pin, non_blocking=True)

        cpu_future = _get_gather_exec().submit(_worker_gpu_attn)

    debug_cpu_topk_only = (
        cpu_attn_mode
        and os.environ.get("JETFIRE_DEBUG_CPU_TOPK_ONLY", "0") == "1"
    )
    _debug_residual_scale_env = os.environ.get(
        "JETFIRE_RELEASE_RESIDUAL_SCALE",
        os.environ.get("JETFIRE_DEBUG_RESIDUAL_SCALE"),
    )
    debug_residual_scale = (
        float(_debug_residual_scale_env)
        if _debug_residual_scale_env is not None
        else 1.0
    )
    debug_residual_scaled = (
        cpu_attn_mode
        and (not debug_cpu_topk_only)
        and debug_residual_scale != 1.0
    )
    debug_corr_mode = os.environ.get("JETFIRE_DEBUG_CORRECTION_MODE", "").strip()
    if debug_corr_mode and debug_corr_mode not in {
        "full",
        "hat_plus_topk",
        "topk_minus_old",
        "residual_only",
        "topk_only",
    }:
        raise RuntimeError(
            "Unsupported JETFIRE_DEBUG_CORRECTION_MODE="
            f"{debug_corr_mode!r}. Expected one of: full, hat_plus_topk, "
            "topk_minus_old, residual_only, topk_only."
        )
    debug_corr_mode_active = bool(debug_corr_mode and debug_corr_mode != "full")

    # Split-mode Stage B v4: now that the CPU worker is dispatched, queue
    # the FUSED softmax·V kernel on the compute stream.  This single call
    # produces o_hat (full attention output) AND o_old_unnorm + W (the
    # top-K-only branch that pre_corr used to compute separately).  No
    # standalone V2_03 pre_corr launch in this path.
    # Fused mode: o_hat / o_old_unnorm / W were all produced by the single
    # ``int2_fused_attn_topk`` call above; no Stage B needed.
    if debug_cpu_topk_only:
        o_old_unnorm_split = None
        W_split = None
    elif use_fused_attn_topk:
        # Already populated; re-use the names so the rest of the function
        # finds them.  (o_hat is the variable from the V2_01_attn block.)
        pass
    elif use_split_decode:
        o_old_unnorm_split = None
        W_split = None
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_stageb_enter", flush=True)
        with _timer("V2_01b_softmax_v"):
            _stage_trace(
                trace_id, "stageB_enter", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=(
                    f"release_stage_b={release_plan.stage_b}"
                    if release_plan is not None else ""
                ),
            )
            # Stage B kernel selection:
            #   JETFIRE_FUSED_SOFTMAX_V=v2 → softmax_norm + weighted_v_split_v2
            #     (eliminates the weights tensor; experimental for INT2 V)
            #   JETFIRE_WVS_VARIANT=legacy → softmax_prep + weighted_v_split (legacy)
            #   default                    → softmax_prep + weighted_v_split_wvs2
            #     for INT3/INT4.  Non-release INT2 keeps the conservative
            #     selector unless explicitly enabled; strict release policy
            #     directly selects the validated INT2 WVS2 plan.
            #
            # INT4-only dispatch:
            #   JETFIRE_INT4_SVS_VARIANT=auto/default → wvs2 score-weight path.
            #       wvs2 itself has a B=1 split-S specialization in the CUDA
            #       launcher, so we no longer select kernels via a magic
            #       n_seqs threshold here.
            #   JETFIRE_INT4_SVS_VARIANT=v2 → force WMMA v2/v3 debug path.
            _ext = _load_softmax_v_ext()
            if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                print("[jetfire][cpu_async] py_stageb_ext_loaded", flush=True)
            if release_plan is not None:
                _fused = release_plan.stage_b == "sv_int2_v2"
                _wvs_env = ""
                _wvs = release_plan.stage_b in (
                    "sv_int2_wvs2",
                    "sv_int3_wvs2",
                    "sv_int4_wvs2",
                    "sv_int4_wvs2_scores",
                )
                _int4_svs = "auto"
                _int4_weight_path = (
                    "scores" if release_plan.stage_b == "sv_int4_wvs2_scores"
                    else "weights"
                )
            else:
                _fused = os.environ.get("JETFIRE_FUSED_SOFTMAX_V", "legacy") == "v2"
                _wvs_env = os.environ.get("JETFIRE_WVS_VARIANT", "")
                _wvs   = (_wvs_env != "legacy")  # default → wvs2; "legacy" opts out
                _int4_svs = os.environ.get("JETFIRE_INT4_SVS_VARIANT", "auto")
                _int4_weight_path = os.environ.get(
                    "JETFIRE_INT4_WEIGHT_PATH", "scores"
                )
            _use_score_weights = (_int4_weight_path == "scores")
            _stageb_kernel_name = ""
            # Pick the int{2,3,4}_softmax_v_with_pre_corr* family by scheme.
            # Only INT2 ships the _v2 (weights-tensor-elimination) variant;
            # INT3 and INT4 ship legacy + wvs2 only.  INT4 additionally ships
            # wmma_v2 (Phase 1 optimisation).
            _v_nbits = get_jetfire_config().scheme.v_nbits
            if _v_nbits == 4:
                _prefix = "int4_softmax_v"
            elif _v_nbits == 3:
                _prefix = "int3_softmax_v"
            else:
                _prefix = "int2_softmax_v"
            _is_int_high = (_v_nbits != 2)
            _allow_unstable_int2_stageb = (
                (
                    release_plan is not None
                    and release_plan.stage_b in ("sv_int2_v2", "sv_int2_wvs2")
                )
                or (
                    release_plan is None
                    and os.environ.get("JETFIRE_ALLOW_UNSTABLE_INT2_STAGEB", "0") == "1"
                )
            )
            if _v_nbits == 2 and not _allow_unstable_int2_stageb:
                _fused = False
                _wvs = False
            # Legacy INT4 Stage B selector.  Keep wvs2 as the default and WMMA
            # as an explicit debug path instead of selecting on a magic n_seqs
            # threshold.
            _use_int4_wmma = (_v_nbits == 4 and _int4_svs == "v2")
            if release_plan is None and num_groups != 1:
                # Legacy non-release selector.  This variable is the GQA ratio
                # H_q/H_kv, not the quant-scale group count in v_scales.  Keep
                # the old conservative behavior only for experimental non-policy
                # runs; release_policy is authoritative and the WVS2 kernels
                # validate their real quant-group constraints internally.
                _wvs = False
                _use_int4_wmma = False
            # INT8 TC fused Stage B.  B=1 stays on the legacy wvs2 default
            # because Qasper/ctx32k accuracy and TPOT have been validated
            # there.  Batched decode is different: wvs2_scores scales almost
            # linearly with B, while reg_stats is materially faster at
            # top_k=256.  Choose by the configured max batch, not the current
            # replay batch size, so all capture sizes in one vLLM graph set use
            # one Stage-B kernel family.  Mixing B=1 wvs2 with B>1 reg_stats in
            # the same graph set has shown replay hangs.  Large top_k now stays
            # on reg_stats too: Stage B switches to a chunk-parallel correction
            # kernel for top_k>=1024 instead of scanning topK serially.
            # Values:
            #   unset (default)  — max_B=1 legacy; max_B>1/supported top_k auto.
            #   "0"             — force legacy wvs2/wmma selector.
            #   "auto"           — use reg_stats when Stage A produced softmax
            #                      stats, otherwise use pc.
            #   "1"   / "fused"  — use int8tc_softmax_v_with_pre_corr (legacy fused)
            #   "reg"            — register-resident INT4 → INT8 expansion
            #                      (skips v_unpack SMEM round-trip; +3% vs svv3)
            #   "pc"             — producer/consumer warp-specialized with
            #                      mbarrier async sync (target: higher occupancy
            #                      via 8-warp CTA + barrier-stall elimination)
            if release_plan is not None:
                _int8tc_sv_env = (
                    "reg" if release_plan.stage_b == "sv_int8tc_reg_stats" else "0"
                )
            else:
                _int8tc_sv_env = os.environ.get("JETFIRE_USE_INT8_TC_SV")
            _int8tc_sv_disabled = _int8tc_sv_env in (
                "0", "off", "false", "False"
            )
            _int8tc_sv_auto = _int8tc_sv_env is None or _int8tc_sv_env in (
                "", "auto"
            )
            _batched_auto_int8tc_sv = (
                _int8tc_sv_env is None
                and (cfg_max_B if cfg_max_B > 0 else n_seqs) > 1
                and top_k in _int8tc_supported_topk
                and softmax_m_gpu is not None
            )
            _use_int8tc_sv = (
                _v_nbits == 4
                and not get_jetfire_config().scheme.v_asymmetric
                and not _int8tc_sv_disabled
                and (
                    _int8tc_sv_env is not None
                    or _batched_auto_int8tc_sv
                )
            )
            if _use_int8tc_sv:
                from jetfire_vllm.kernels.int8tc_softmax_v_fused import (
                    load_ext as _load_int8tc_sv_ext,
                )
                _ext_int8tc_sv = _load_int8tc_sv_ext()
                _int8tc_sv_mode = (
                    "reg" if (_int8tc_sv_auto and softmax_m_gpu is not None)
                    else "pc" if _int8tc_sv_auto
                    else _int8tc_sv_env
                )
                if (
                    _int8tc_sv_mode == "reg"
                    and softmax_m_gpu is not None
                ):
                    _stageb_kernel_name = "int8tc_softmax_v_with_pre_corr_reg_stats"
                    _fn_sv = _ext_int8tc_sv.int8tc_softmax_v_with_pre_corr_reg_stats
                elif _int8tc_sv_mode == "pc":
                    _stageb_kernel_name = "int8tc_softmax_v_with_pre_corr_pc"
                    _fn_sv = _ext_int8tc_sv.int8tc_softmax_v_with_pre_corr_pc
                elif _int8tc_sv_mode == "reg":
                    _stageb_kernel_name = "int8tc_softmax_v_with_pre_corr_reg"
                    _fn_sv = _ext_int8tc_sv.int8tc_softmax_v_with_pre_corr_reg
                else:
                    _stageb_kernel_name = "int8tc_softmax_v_with_pre_corr"
                    _fn_sv = _ext_int8tc_sv.int8tc_softmax_v_with_pre_corr
                if (
                    _int8tc_sv_mode == "reg"
                    and softmax_m_gpu is not None
                ):
                    o_hat, o_old_unnorm_split, W_split = _fn_sv(
                        scores, softmax_m_gpu, softmax_inv_z_gpu,
                        store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
                else:
                    o_hat, o_old_unnorm_split, W_split = _fn_sv(
                        scores, store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
            elif _fused and not _is_int_high:
                _stageb_kernel_name = "int2_softmax_v_with_pre_corr_v2"
                o_hat, o_old_unnorm_split, W_split = (
                    _ext.int2_softmax_v_with_pre_corr_v2(
                        scores, store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
                )
            elif _use_int4_wmma:
                # Current main Stage B path for the best measured decode
                # setup is wmma_v3 + score-weight path.
                _use_v3 = os.environ.get("JETFIRE_INT4_SVS", "v3") == "v3"
                if _use_v3:
                    if _use_score_weights and softmax_m_gpu is not None:
                        _stageb_kernel_name = (
                            "int4_softmax_v_with_pre_corr_wmma_v3_scores_stats"
                        )
                        o_hat, o_old_unnorm_split, W_split = (
                            _ext.int4_softmax_v_with_pre_corr_wmma_v3_scores_stats(
                                scores, softmax_m_gpu, softmax_inv_z_gpu,
                                store.v_data, store.v_scales,
                                bt_batch, seq_lens_batch, topk_idx_gpu,
                                max_S, H_q, H_kv, bs, top_k,
                            )
                        )
                        _fn_v = None
                    else:
                        _fn_v = (
                            _ext.int4_softmax_v_with_pre_corr_wmma_v3_scores
                            if _use_score_weights else
                            _ext.int4_softmax_v_with_pre_corr_wmma_v3
                        )
                        _stageb_kernel_name = (
                            "int4_softmax_v_with_pre_corr_wmma_v3_scores"
                            if _use_score_weights else
                            "int4_softmax_v_with_pre_corr_wmma_v3"
                        )
                else:
                    _fn_v = (
                        _ext.int4_softmax_v_with_pre_corr_wmma_v2_scores
                        if _use_score_weights else
                        _ext.int4_softmax_v_with_pre_corr_wmma_v2
                    )
                    _stageb_kernel_name = (
                        "int4_softmax_v_with_pre_corr_wmma_v2_scores"
                        if _use_score_weights else
                        "int4_softmax_v_with_pre_corr_wmma_v2"
                    )
                if _fn_v is not None:
                    o_hat, o_old_unnorm_split, W_split = _fn_v(
                        scores, store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
            elif _wvs:
                _fn_inplace = getattr(
                    _ext, f"{_prefix}_with_pre_corr_wvs2_inplace", None
                )
                if (
                    _graph_mode_enabled()
                    and os.environ.get("JETFIRE_GRAPH_STAGEB_INPLACE", "0") == "1"
                    and _fn_inplace is not None
                ):
                    _stageb_kernel_name = f"{_prefix}_with_pre_corr_wvs2_inplace"
                    tag_base = _graph_store_tag(store, _stageb_kernel_name)
                    weights = _get_device(
                        (n_seqs, H_q, max_S),
                        torch.float32,
                        f"{tag_base}:weights",
                        device,
                    )
                    o_hat = _get_device(
                        (n_seqs, H_q, D),
                        torch.float32,
                        f"{tag_base}:o_hat",
                        device,
                    )
                    o_old_unnorm_split = _get_device(
                        (n_seqs, H_q, D),
                        torch.float32,
                        f"{tag_base}:o_old",
                        device,
                    )
                    W_split = _get_device(
                        (n_seqs, H_q),
                        torch.float32,
                        f"{tag_base}:W",
                        device,
                    )
                    v_scales_arg = store.v_scales
                    if v_scales_arg.dim() == 2:
                        v_scales_arg = v_scales_arg.unsqueeze(-1)
                    if not v_scales_arg.is_contiguous():
                        raise RuntimeError(
                            "JetFire graph Stage-B inplace path requires "
                            "contiguous v_scales."
                        )
                    _fn_inplace(
                        scores, store.v_data, v_scales_arg,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        weights, o_hat, o_old_unnorm_split, W_split,
                        max_S, H_q, H_kv, bs, top_k,
                    )
                elif _v_nbits == 4 and _use_score_weights:
                    _fn = _ext.int4_softmax_v_with_pre_corr_wvs2_scores
                    _stageb_kernel_name = "int4_softmax_v_with_pre_corr_wvs2_scores"
                    o_hat, o_old_unnorm_split, W_split = _fn(
                        scores, store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
                else:
                    _stageb_kernel_name = f"{_prefix}_with_pre_corr_wvs2"
                    _fn = getattr(_ext, _stageb_kernel_name)
                    o_hat, o_old_unnorm_split, W_split = _fn(
                        scores, store.v_data, store.v_scales,
                        bt_batch, seq_lens_batch, topk_idx_gpu,
                        max_S, H_q, H_kv, bs, top_k,
                    )
            else:
                _stageb_kernel_name = f"{_prefix}_with_pre_corr"
                _fn = getattr(_ext, _stageb_kernel_name)
                o_hat, o_old_unnorm_split, W_split = _fn(
                    scores, store.v_data, store.v_scales,
                    bt_batch, seq_lens_batch, topk_idx_gpu,
                    max_S, H_q, H_kv, bs, top_k,
                )
            _stage_trace(
                trace_id, "stageB_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=(
                    f"kernel={_stageb_kernel_name} "
                    + (
                        f"release_stage_b={release_plan.stage_b}"
                        if release_plan is not None else "release_stage_b="
                    )
                ),
            )
            _step_trace(
                trace_id, "stageB_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=(
                    f"kernel={_stageb_kernel_name} "
                    + (
                        f"release_stage_b={release_plan.stage_b}"
                        if release_plan is not None else "release_stage_b="
                    )
                ),
            )
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_stageb_done", flush=True)
    else:
        o_old_unnorm_split = None
        W_split = None

    # Phase 0 experiment: inject configurable GPU "filler" work AFTER the
    # CPU worker is dispatched and BEFORE pre_correction.  This sits in
    # the same CPU-overlap window that Path A's Stage B would occupy.  If
    # throughput stays flat as filler grows, that window has headroom.
    _filler_us = int(os.environ.get("JETFIRE_FILLER_US", "0"))
    if _filler_us > 0:
        with _timer("V2_02b_filler"):
            _load_filler_ext().launch_filler(_filler_us)

    # Control experiment: pure host-thread sleep (no GPU work).  Used to
    # separate "GPU pipelining benefit" from "host-thread scheduling side
    # effect" in the filler experiment.  If throughput improves with this
    # set but no JETFIRE_FILLER_US, the +8% peak is mostly from host-side
    # effects (less spin contention with CPU worker), not real GPU pipelining.
    _host_sleep_us = int(os.environ.get("JETFIRE_HOST_SLEEP_US", "0"))
    if _host_sleep_us > 0:
        time.sleep(_host_sleep_us / 1e6)

    # 4. Pre-correction.  In split mode and the new fused mode this is
    # already done inside the Stage B / fused kernel (which produces
    # o_old_unnorm + W).  Legacy WMMA path needs the standalone kernel.
    if debug_cpu_topk_only:
        o_old = None
        W_2d = None
    elif use_split_decode or use_fused_attn_topk:
        # Stage B v4 / fused-attn-topK produced o_old_unnorm and W;
        # downstream blend uses the v4 formula:
        #   o_corr = o_hat + W·o_new − o_old_unnorm
        o_old = o_old_unnorm_split  # NOTE: this is UNNORMALIZED (W·o_old_norm)
        W_2d  = W_split
    else:
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_precorr_enter", flush=True)
        with _timer("V2_03_pre_corr"):
            o_old, W_2d = kernel_set.fused_pre_correction_batched(
                scores, topk_idx_gpu, bt_batch, seq_lens_batch,
                store.v_data, store.v_scales,
                bs, H_kv, num_groups,
            )
        if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
            print("[jetfire][cpu_async] py_precorr_done", flush=True)

    # 5+6 fused: spin-wait on cpu_flag + blend kernel, all in one C++ call
    # with GIL released.  Replaces Python Future.result() (cv wakeup ~25 µs)
    # + blend_into (2 separate Python→C++ round-trips).  Mapped o_new mode
    # means o_new lives in shared host/device memory, so no cross-stream
    # wait is needed — the cpu_flag release-store + _mm_sfence already
    # orders the worker's WC stores before the GPU read.
    if (
        cpu_attn_mode
        and use_fused_join_blend
        and not debug_cpu_topk_only
        and not debug_residual_scaled
        and not debug_corr_mode_active
    ):
        with _timer("V2_04_05_fused"):
            _load_blend_ext().wait_and_blend_unnorm_into(
                cpu_flag,
                o_hat, o_old, W_2d, o_new_gpu, output[:n_seqs],
            )
        if debug_approx_only:
            output[:n_seqs].copy_(o_hat.to(output.dtype))
        pool.record_decode_hit()
        _sanitize_decode_output_if_needed(output, n_seqs)
        _step_trace(
            trace_id, "fused_join_blend_return", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )
        return output

    # 5. Sync points: worker done, then (non-mapped only) comm done so the
    # worker's H2D of o_new has landed.  In mapped mode the worker writes
    # directly to host/device-shared memory and there is no H2D — once
    # `cpu_future.result()` returns, o_new_gpu is readable.
    with _timer("V2_04_join"):
        with _timer("V2_04d_main_wait_worker"):
            _stage_trace(
                trace_id, "join_enter", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                extra=f"future_none={cpu_future is None}",
            )
            if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                print(f"[jetfire][cpu_async] py_join_enter future_none={cpu_future is None}", flush=True)
            if use_graph_submit:
                # Release graph path: captured side-stream event wait observes
                # the async CPU job after Stage B.  Other waits are retained
                # only as explicit A/B modes.
                ext = _load_cpu_topk_attn_ext()
                stream_id = torch.cuda.current_stream().cuda_stream
                if os.environ.get("JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK", "0") == "1":
                    ext.stream_wait_value64(stream_id, _flag_dev, 1)
                elif _use_device_signal:
                    ext.stream_wait_value64(stream_id, _flag_dev, 1)
                elif _use_side_join:
                    ext.stream_wait_cpu_topk_attn_done_event(
                        stream_id, submit_ctx_handle
                    )
                else:
                    ext.stream_wait_cpu_topk_attn(stream_id, submit_ctx_handle)
            elif cpu_done_inline:
                pass
            elif cpu_future is None:
                if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                    print("[jetfire][cpu_async] py_wait", flush=True)
                # Persistent-worker path: spin-then-yield on the atomic
                # flag inside C++ (GIL released).  No Future, no cv wakeup.
                _load_cpu_topk_attn_ext().wait_for_flag(cpu_flag)
                if os.environ.get("JETFIRE_CPU_DEBUG", "0") == "1":
                    print("[jetfire][cpu_async] py_wait_done", flush=True)
            else:
                _timeout_s = float(os.environ.get("JETFIRE_CPU_FUTURE_TIMEOUT_S", "30"))
                try:
                    if _timeout_s > 0:
                        cpu_future.result(timeout=_timeout_s)
                    else:
                        cpu_future.result()
                except TimeoutError as exc:
                    debug = _cpu_worker_timeout_debug(
                        cpu_future, layer_name=layer_name, trace_id=trace_id
                    )
                    raise RuntimeError(
                        "JetFire CPU correction worker did not finish within "
                        f"{_timeout_s:.1f}s. This is a guarded failure instead "
                        "of a silent EngineCore hang; use "
                        "JETFIRE_SYNC_CPU_ATTN=1 for the stable inline path. "
                        f"Debug: {debug}"
                    ) from exc
            _stage_trace(
                trace_id, "join_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            )
            _step_trace(
                trace_id, "join_done", layer_name=layer_name,
                n_seqs=n_seqs, max_S=max_S, top_k=top_k,
            )
        if (not use_mapped_o_new) and use_graph_submit:
            # Graph-safe non-mapped path: CPU writes regular pinned host
            # memory, the stream waits on the CPU doorbell, then the H2D copy
            # is captured as a normal graph memcpy node before blend reads it.
            o_new_gpu.copy_(o_new_pin, non_blocking=True)
        elif not use_mapped_o_new:
            with _timer("V2_04e_main_wait_stream"):
                torch.cuda.current_stream().wait_stream(comm)

    if debug_cpu_topk_only:
        output[:n_seqs].copy_(o_new_gpu.to(output.dtype))
        pool.record_decode_hit()
        _sanitize_decode_output_if_needed(output, n_seqs)
        _step_trace(
            trace_id, "debug_cpu_topk_only_return", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )
        return output

    # 6. Combine: blend kernel writes directly into output[:n_seqs].
    # Split mode uses the v4 formula `o_hat + W·o_new − o_old_unnorm`
    # (Stage B v4 produced o_old UNNORMALIZED).  Legacy mode uses
    # the original `o_hat + W·(o_new − o_old)` formula.
    with _timer("V2_05_post_corr"):
        _stage_trace(
            trace_id, "blend_enter", layer_name=layer_name,
            n_seqs=n_seqs, max_S=max_S, top_k=top_k,
        )
        if cpu_attn_mode:
            unnorm_path = use_split_decode or use_fused_attn_topk
            if (output.dtype == torch.bfloat16
                    and os.environ.get("JETFIRE_USE_BLEND_KERNEL", "0") == "1"
                    and not debug_residual_scaled):
                _stage_trace(
                    trace_id, "blend_load_enter", layer_name=layer_name,
                    n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                    extra=(
                        f"unnorm={unnorm_path} "
                        f"o_hat={tuple(o_hat.shape)} o_old={tuple(o_old.shape)} "
                        f"W={tuple(W_2d.shape)} o_new={tuple(o_new_gpu.shape)} "
                        f"out={tuple(output[:n_seqs].shape)}"
                    ),
                )
                blend_ext = _load_blend_ext()
                _stage_trace(
                    trace_id, "blend_load_done", layer_name=layer_name,
                    n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                )
                if unnorm_path:
                    _stage_trace(
                        trace_id, "blend_call_enter", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                        extra="fn=blend_unnorm_into",
                    )
                    blend_ext.blend_unnorm_into(
                        o_hat, o_old, W_2d, o_new_gpu, output[:n_seqs],
                    )
                else:
                    _stage_trace(
                        trace_id, "blend_call_enter", layer_name=layer_name,
                        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                        extra="fn=blend_into",
                    )
                    blend_ext.blend_into(
                        o_hat, o_old, W_2d, o_new_gpu, output[:n_seqs],
                    )
                if debug_approx_only:
                    output[:n_seqs].copy_(o_hat.to(output.dtype))
                _stage_trace(
                    trace_id, "blend_call_done", layer_name=layer_name,
                    n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                )
                pool.record_decode_hit()
                _sanitize_decode_output_if_needed(output, n_seqs)
                _stage_trace(
                    trace_id, "blend_done_return", layer_name=layer_name,
                    n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                )
                _step_trace(
                    trace_id, "blend_done_return", layer_name=layer_name,
                    n_seqs=n_seqs, max_S=max_S, top_k=top_k,
                )
                return output
            # Fallback fp32 op chain — release default while the fused blend
            # kernel remains an opt-in micro-optimization.
            o_new_f = o_new_gpu.to(torch.float32)
            if unnorm_path:
                # o_old is UNNORMALIZED here (= W · o_old_norm).
                if debug_corr_mode_active:
                    w_topk = W_2d.unsqueeze(-1) * o_new_f
                    if debug_corr_mode == "hat_plus_topk":
                        o_corr = o_hat + w_topk
                    elif debug_corr_mode == "topk_minus_old":
                        o_corr = w_topk - o_old
                    elif debug_corr_mode == "residual_only":
                        o_corr = o_hat - o_old
                    elif debug_corr_mode == "topk_only":
                        o_corr = w_topk
                    else:
                        o_corr = o_hat + w_topk - o_old
                elif debug_residual_scaled:
                    o_corr = (
                        W_2d.unsqueeze(-1) * o_new_f
                        + debug_residual_scale * (o_hat - o_old)
                    )
                else:
                    o_corr = o_hat + W_2d.unsqueeze(-1) * o_new_f - o_old
            else:
                o_corr = o_hat + W_2d.unsqueeze(-1) * (o_new_f - o_old)
        else:
            # GPU-only correction path (cpu_attn_mode=False).
            o_corr = kernel_set.fused_post_correction(
                q_batch, k_topk_gpu, v_topk_gpu, o_hat, o_old, W_2d.unsqueeze(-1),
                float(scale),
            )

    if debug_approx_only:
        o_corr = o_hat

    output[:n_seqs].copy_(o_corr.to(output.dtype))
    pool.record_decode_hit()
    _sanitize_decode_output_if_needed(output, n_seqs)
    _stage_trace(
        trace_id, "done_return", layer_name=layer_name,
        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
    )
    _step_trace(
        trace_id, "done_return", layer_name=layer_name,
        n_seqs=n_seqs, max_S=max_S, top_k=top_k,
    )
    return output
