"""Centralized JetFire runtime configuration.

Single source of truth that replaces the dozen+ scattered ``os.environ.get``
calls.  Build a :class:`JetFireConfig` once via :func:`get_jetfire_config` and
read the resolved fields anywhere they are needed.

Public API
----------
* ``JETFIRE_MODE``: selects a quantization scheme from :data:`QUANT_SCHEMES`.
  Defaults to the release/paper target ``SymInt3K_SymInt2V``. The
  high-accuracy baseline is ``SymInt4K_SymInt4V``; K4V3/K4V2/K3V4/K3V3 and
  K2V* modes are sensitivity configurations. ``AsymInt2K_SymInt2V`` remains
  accepted as a compatibility alias for ``AsymInt2K4Level_SymInt2V``.
* ``JETFIRE_USE_CUSTOM_BACKEND``: master on/off vs vLLM's native FA3.
* ``JETFIRE_CPU_CORRECTION``: blend top-K bf16 KV from a CPU shadow into
  the INT decode output.  Off ⇒ INT mirror is built but decode falls back
  to native FA3 (mirror-only HBM ablation).
* ``JETFIRE_SKIP_NATIVE_KV``: tell vLLM not to allocate its own KV cache.
* Other knobs (``JETFIRE_TOPK``, ``JETFIRE_GROUP_SIZE``,
  ``JETFIRE_MIRROR_LAYER_STRIDE``, ``JETFIRE_SHADOW``, etc.) are ablation
  parameters; defaults match the paper-target configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ── Quantization schemes ────────────────────────────────────────────────


@dataclass(frozen=True)
class QuantScheme:
    """Specification of a low-precision KV-cache quantization scheme.

    ``kernel_module`` is the dotted ``module:attr`` path of a
    :class:`KernelSet` instance providing the fast-path GPU kernels.
    ``None`` means the scheme has no fast-path implementation: it can only
    run with ``cpu_correction=False`` (mirror-only HBM ablation).
    """

    name: str
    k_nbits: int
    v_nbits: int
    k_asymmetric: bool
    v_asymmetric: bool
    kernel_module: Optional[str] = None


# Registered schemes.  Add a new entry here when implementing a new
# bit-width or sym/asym combination.  Sym is treated as Asym(zp=0): the
# storage layout omits the zero-point tensor when ``*_asymmetric`` is False.
ASYM_INT2K_4LEVEL_SYM_INT2V = QuantScheme(
    name="AsymInt2K4Level_SymInt2V",
    k_nbits=2, v_nbits=2,
    k_asymmetric=True, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.int2_asym_k_sym_v:KERNELS",
)
ASYM_INT2K_4LEVEL_SYM_INT3V = QuantScheme(
    name="AsymInt2K4Level_SymInt3V",
    k_nbits=2, v_nbits=3,
    k_asymmetric=True, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.int2_asym_k_sym_int3_v:KERNELS",
)
ASYM_INT2K_4LEVEL_SYM_INT4V = QuantScheme(
    name="AsymInt2K4Level_SymInt4V",
    k_nbits=2, v_nbits=4,
    k_asymmetric=True, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.int2_asym_k_sym_int4_v:KERNELS",
)
SYM_INT2K_SYM_INT2V = QuantScheme(
    name="SymInt2K_SymInt2V",
    k_nbits=2, v_nbits=2,
    k_asymmetric=False, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.sym_int2_k_sym_int2_v:KERNELS",
)
SYM_INT2K_SYM_INT3V = QuantScheme(
    name="SymInt2K_SymInt3V",
    k_nbits=2, v_nbits=3,
    k_asymmetric=False, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.sym_int2_k_sym_int3_v:KERNELS",
)
SYM_INT2K_SYM_INT4V = QuantScheme(
    name="SymInt2K_SymInt4V",
    k_nbits=2, v_nbits=4,
    k_asymmetric=False, v_asymmetric=False,
    kernel_module="jetfire_vllm.kernels._dispatch.sym_int2_k_sym_int4_v:KERNELS",
)

QUANT_SCHEMES: dict[str, QuantScheme] = {
    # ---- Extreme compression INT2 scheme ------------------------------
    # K uses the full INT2 codebook: stored code [0,3], signed q=code-2.
    "AsymInt2K4Level_SymInt2V": ASYM_INT2K_4LEVEL_SYM_INT2V,
    "AsymInt2K4Level_SymInt3V": ASYM_INT2K_4LEVEL_SYM_INT3V,
    "AsymInt2K4Level_SymInt4V": ASYM_INT2K_4LEVEL_SYM_INT4V,
    # Historical name kept as an alias so existing scripts/env vars still run.
    "AsymInt2K_SymInt2V": ASYM_INT2K_4LEVEL_SYM_INT2V,
    "AsymInt2K_SymInt3V": ASYM_INT2K_4LEVEL_SYM_INT3V,
    "AsymInt2K_SymInt4V": ASYM_INT2K_4LEVEL_SYM_INT4V,
    # ---- Asymmetric placeholders (kernels not yet implemented) --------
    # Selecting these with cpu_correction=True raises at config time.
    "AsymInt3K_SymInt3V": QuantScheme(
        name="AsymInt3K_SymInt3V",
        k_nbits=3, v_nbits=3,
        k_asymmetric=True, v_asymmetric=False,
        kernel_module=None,
    ),
    "AsymInt4K_SymInt4V": QuantScheme(
        name="AsymInt4K_SymInt4V",
        k_nbits=4, v_nbits=4,
        k_asymmetric=True, v_asymmetric=False,
        kernel_module=None,
    ),
    # ---- Symmetric fast paths -----------------------------------------
    "SymInt2K_SymInt2V": QuantScheme(
        name=SYM_INT2K_SYM_INT2V.name,
        k_nbits=SYM_INT2K_SYM_INT2V.k_nbits, v_nbits=SYM_INT2K_SYM_INT2V.v_nbits,
        k_asymmetric=SYM_INT2K_SYM_INT2V.k_asymmetric,
        v_asymmetric=SYM_INT2K_SYM_INT2V.v_asymmetric,
        kernel_module=SYM_INT2K_SYM_INT2V.kernel_module,
    ),
    "SymInt2K_SymInt3V": SYM_INT2K_SYM_INT3V,
    "SymInt2K_SymInt4V": SYM_INT2K_SYM_INT4V,
    "SymInt3K_SymInt3V": QuantScheme(
        name="SymInt3K_SymInt3V",
        k_nbits=3, v_nbits=3,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int3_k_sym_int3_v:KERNELS",
    ),
    "SymInt4K_SymInt4V": QuantScheme(
        name="SymInt4K_SymInt4V",
        k_nbits=4, v_nbits=4,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int4_k_sym_int4_v:KERNELS",
    ),
    # ---- Mixed-precision sym schemes ---------------------------------
    # Independent K and V bit-widths.  Both directions sym (no zp).
    "SymInt4K_SymInt2V": QuantScheme(
        name="SymInt4K_SymInt2V",
        k_nbits=4, v_nbits=2,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int4_k_sym_int2_v:KERNELS",
    ),
    "SymInt3K_SymInt2V": QuantScheme(
        name="SymInt3K_SymInt2V",
        k_nbits=3, v_nbits=2,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int3_k_sym_int2_v:KERNELS",
    ),
    "SymInt3K_SymInt4V": QuantScheme(
        name="SymInt3K_SymInt4V",
        k_nbits=3, v_nbits=4,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int3_k_sym_int4_v:KERNELS",
    ),
    "SymInt4K_SymInt3V": QuantScheme(
        name="SymInt4K_SymInt3V",
        k_nbits=4, v_nbits=3,
        k_asymmetric=False, v_asymmetric=False,
        kernel_module="jetfire_vllm.kernels._dispatch.sym_int4_k_sym_int3_v:KERNELS",
    ),
}


DEFAULT_MODE = "SymInt3K_SymInt2V"


_GRAPH_DEFAULT_ENV = {
    "JETFIRE_CPU_CORRECTION": "1",
    "JETFIRE_DECODE_CPU_ATTN": "1",
    "JETFIRE_INFLATE_BLOCKS": "1",
    "JETFIRE_USE_INT8_TC": "1",
    "JETFIRE_INT4_QK_STATS": "1",
    "JETFIRE_INT4_WEIGHT_PATH": "scores",
    "JETFIRE_DECODE_GRAPH": "1",
    "JETFIRE_SKIP_NATIVE_KV": "1",
    "JETFIRE_USE_PERSISTENT_WORKER": "0",
    "JETFIRE_CPU_ATTN_THREADS": "32",
    "OMP_NUM_THREADS": "32",
}


def apply_mode_defaults(mode_name: str | None = None) -> None:
    """Apply stable mode-specific defaults before resolving config.

    Any scheme with a fast-path kernel is a release-capable JetFire decode
    mode.  Callers should only need to set ``JETFIRE_MODE`` and optionally
    ``JETFIRE_TOPK``.  User-provided env values still win for ordinary knobs;
    graph CPU-topK synchronization is sanitized separately because inherited
    A/B env vars can otherwise select a known-unsafe path.
    """
    resolved_mode = mode_name or os.environ.get("JETFIRE_MODE", DEFAULT_MODE)
    scheme = QUANT_SCHEMES.get(resolved_mode)
    if scheme is None or scheme.kernel_module is None:
        return
    if os.environ.get("JETFIRE_USE_CUSTOM_BACKEND", "1") == "0":
        return
    for name, value in _GRAPH_DEFAULT_ENV.items():
        os.environ.setdefault(name, value)


def apply_graph_cpu_topk_release_preset() -> None:
    """Force the release-safe graph CPU-topK synchronization mode.

    JetFire has several CPU-correction graph synchronization experiments
    (device-signal, host-join, GPU mapped-host wait).  They are useful for
    kernel A/B work, but stale env vars from those experiments have repeatedly
    caused long accuracy/benchmark sweeps to hang or leave one TP rank waiting
    while the other rank keeps executing.

    Release graph decode therefore uses async CPU submit followed by a
    stream-ordered host join at the correction boundary.  CPU topK still
    overlaps Stage-B because submit happens before Stage-B; the join only
    blocks when correction needs o_new.  The older side-stream event join is
    kept for A/B work because long Qwen/Qasper sweeps exposed event-replay
    stalls with the per-context reused CUDA events.  Experimental graph
    synchronization paths must be run outside strict release-policy validation.
    """
    if os.environ.get("JETFIRE_USE_CUSTOM_BACKEND", "1") == "0":
        return
    if os.environ.get("JETFIRE_DECODE_GRAPH", "0") != "1":
        return
    if os.environ.get("JETFIRE_ALLOW_EXPERIMENTAL_GRAPH_CPU_TOPK", "0") == "1":
        return

    os.environ["JETFIRE_GRAPH_SIDE_JOIN_CPU_TOPK"] = "0"
    os.environ["JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK"] = "0"
    os.environ["JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK"] = "0"
    os.environ["JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK"] = "1"
    os.environ.pop("JETFIRE_GRAPH_INLINE_CPU_TOPK", None)
    os.environ.pop("JETFIRE_GRAPH_SYNC_CPU_TOPK", None)


# ── Resolved runtime config ─────────────────────────────────────────────


@dataclass(frozen=True)
class JetFireConfig:
    """Fully-resolved JetFire runtime configuration."""

    scheme: QuantScheme
    use_custom_backend: bool = True
    cpu_correction: bool = True
    skip_native_kv: bool = True
    install_hooks: bool = False
    # Inflate vLLM's planned num_blocks by ``BF16/INT bytes-per-block``.  Off
    # by default — turning it on without also turning shadow off (or moving
    # to the CPU-compute decode path) inflates the pinned shadow allocation
    # 7–8× and stalls engine init.  Used for prefill-only / long-context
    # capacity demos.
    inflate_blocks: bool = False
    # Run the v2 correction's ``softmax(Q·K_topk^T)·V_topk → o_new`` kernel
    # on CPU (multi-threaded bf16 via the ``cpu_topk_attn`` extension) and
    # H2D only the small ``o_new`` (~tens of KB) instead of multi-MB
    # K_topk/V_topk.  Cuts decode H2D traffic ~125× and lets shadow be
    # unpinned (so inflation no longer balloons shadow allocation).
    decode_cpu_attn: bool = False

    group_size: int = 64
    topk: int = 256
    layer_stride: int = 1
    # Pre-allocate decode buffers (q, topk_slots, o_new mapped, etc.) at this
    # batch size during engine init, so steady-state decode never allocates
    # pinned/mapped memory and the dict-key cache always hits.  ``0`` (default)
    # disables prealloc and falls back to the legacy lazy-on-first-use path.
    # When set, ``n_seqs`` may be ≤ this value (we slice the buffer to
    # ``[:n_seqs]``); if a step exceeds it, we silently allocate a one-off
    # buffer for that step and warn once.
    max_batch_size: int = 0

    shadow_enabled: bool = True
    shadow_device: str = "cpu"   # "cpu" | "gpu"
    shadow_dtype: str = "bf16"   # "bf16" | "fp16"

    profile_decode: bool = False
    debug_log: bool = False


# ── Env parsing ─────────────────────────────────────────────────────────


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() == "1"


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# ── Resolver ────────────────────────────────────────────────────────────


_CACHED_CONFIG: Optional[JetFireConfig] = None


def get_jetfire_config() -> JetFireConfig:
    """Resolve and cache JetFire configuration from environment variables."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    mode_name = os.environ.get("JETFIRE_MODE", DEFAULT_MODE)
    apply_mode_defaults(mode_name)
    apply_graph_cpu_topk_release_preset()
    if mode_name not in QUANT_SCHEMES:
        raise ValueError(
            f"JETFIRE_MODE={mode_name!r} is not a known quant scheme. "
            f"Available: {sorted(QUANT_SCHEMES)}"
        )
    scheme = QUANT_SCHEMES[mode_name]

    use_custom = _bool_env("JETFIRE_USE_CUSTOM_BACKEND", True)
    cpu_corr = _bool_env("JETFIRE_CPU_CORRECTION", True)
    skip_native = _bool_env("JETFIRE_SKIP_NATIVE_KV", True)
    install_hooks = _bool_env("JETFIRE_INSTALL_HOOKS", False)
    layer_stride = _int_env("JETFIRE_MIRROR_LAYER_STRIDE", 1)
    shadow_enabled = _bool_env("JETFIRE_SHADOW", True)
    inflate_blocks = _bool_env("JETFIRE_INFLATE_BLOCKS", False)
    decode_cpu_attn = _bool_env("JETFIRE_DECODE_CPU_ATTN", False)

    if use_custom:
        # ── Invariants (only enforced when JetFire is active) ──────────
        # cpu_correction requires a fast-path kernel for this scheme.
        if cpu_corr and scheme.kernel_module is None:
            raise ValueError(
                f"JETFIRE_MODE={mode_name!r} has no fast-path kernel "
                f"implemented yet, so JETFIRE_CPU_CORRECTION=1 is invalid. "
                f"Pick a supported scheme with a kernel "
                f"({', '.join(k for k, v in QUANT_SCHEMES.items() if v.kernel_module)}) "
                f"or set JETFIRE_CPU_CORRECTION=0 for a mirror-only ablation."
            )
        # cpu_correction off + skip_native_kv on leaves decode with no
        # implementation — fine for prefill-only workloads (max_tokens=1
        # samples from prefill logits, decode forward never runs), but
        # decoding any token would crash.  Allow it iff shadow is also
        # off (signals "I'm not using JetFire correction infrastructure
        # at all").
        if not cpu_corr and skip_native and shadow_enabled:
            raise ValueError(
                "JETFIRE_CPU_CORRECTION=0 + JETFIRE_SKIP_NATIVE_KV=1 with "
                "shadow on is unsupported (decode would have nowhere to go). "
                "Either enable CPU correction (default) or also set "
                "JETFIRE_SHADOW=0 for a prefill-only configuration."
            )
        # layer_stride > 1 leaves some layers without a JetFire store —
        # they fall back to native FA3, which needs the native KV cache.
        if layer_stride > 1 and skip_native:
            raise ValueError(
                f"JETFIRE_MIRROR_LAYER_STRIDE={layer_stride} skips some "
                f"layers, which fall back to native FA3, but "
                f"JETFIRE_SKIP_NATIVE_KV=1 has freed the native KV cache. "
                f"Set JETFIRE_SKIP_NATIVE_KV=0 for this ablation."
            )
        # Inflation interacts with shadow allocation: a pinned shadow
        # sized to inflated num_blocks is huge (e.g. 195GB pinned at
        # 7.31×) and stalls engine init.  Two ways to use inflation:
        #   • Disable shadow entirely (prefill-only / capacity demos)
        #   • Use the CPU-compute decode path with unpinned shadow
        if inflate_blocks and shadow_enabled and not decode_cpu_attn:
            raise ValueError(
                "JETFIRE_INFLATE_BLOCKS=1 with JETFIRE_SHADOW=1 needs "
                "JETFIRE_DECODE_CPU_ATTN=1 (so shadow can live in "
                "unpinned host memory).  Otherwise set JETFIRE_SHADOW=0 "
                "for a prefill-only configuration."
            )

    cfg = JetFireConfig(
        scheme=scheme,
        use_custom_backend=use_custom,
        cpu_correction=cpu_corr,
        skip_native_kv=skip_native,
        install_hooks=install_hooks,
        inflate_blocks=inflate_blocks,
        decode_cpu_attn=decode_cpu_attn,
        group_size=_int_env("JETFIRE_GROUP_SIZE", 64),
        topk=_int_env("JETFIRE_TOPK", 256),
        layer_stride=layer_stride,
        max_batch_size=_int_env("JETFIRE_MAX_BATCH_SIZE", 0),
        shadow_enabled=shadow_enabled,
        shadow_device=os.environ.get("JETFIRE_SHADOW_DEVICE", "cpu").lower(),
        shadow_dtype=os.environ.get("JETFIRE_SHADOW_DTYPE", "bf16").lower(),
        profile_decode=_bool_env("JETFIRE_PROFILE_DECODE", False),
        debug_log=_bool_env("JETFIRE_DEBUG_LOG", False),
    )
    _CACHED_CONFIG = cfg
    return cfg


def reset_jetfire_config() -> None:
    """For tests: re-read env on next ``get_jetfire_config()`` call."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None


# ── Block-count inflation ───────────────────────────────────────────────
#
# vLLM's KV-cache planner sizes ``num_blocks`` assuming BF16 KV storage.
# When skip_native_kv is on, JetFire stores INT-N KV instead, which costs
# a fraction of the BF16 footprint per block.  We therefore inflate
# vLLM's num_blocks by the equivalent bytes ratio so the scheduler admits
# more concurrent sequences and JetFire's INT mirror actually uses the
# whole HBM budget instead of leaving 7/8 of it idle.


def compute_inflate_ratio(scheme: QuantScheme, group_size: int = 128) -> float:
    """Effective ``BF16_bytes / INT_bytes`` per token for this scheme.

    Per element (K and V each contribute):
      * data: ``nbits`` bits
      * scale: ``16 / group_size`` bits (one bf16 per group)
      * zero-point: same as scale, only when asymmetric
    """

    def per_elem_bits(nbits: int, asym: bool) -> float:
        return nbits + 16.0 / group_size + (16.0 / group_size if asym else 0.0)

    int_total_bits = (
        per_elem_bits(scheme.k_nbits, scheme.k_asymmetric)
        + per_elem_bits(scheme.v_nbits, scheme.v_asymmetric)
    )
    return 32.0 / int_total_bits  # 32 = 2 × 16 bits (BF16 K + BF16 V)
