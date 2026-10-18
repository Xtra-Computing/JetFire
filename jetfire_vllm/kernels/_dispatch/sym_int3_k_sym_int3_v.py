"""KernelSet for the SymInt3K_SymInt3V scheme.

INT3 path is split-mode only. No fused WMMA decode kernel and no GPU
correction kernels for INT3 — requires:
    JETFIRE_USE_SPLIT_INT2=1     (Stage A + Stage B split)
    JETFIRE_DECODE_CPU_ATTN=1    (CPU correction path)

Wraps:
  * ``int3_qk_scores.cu`` / ``int3_softmax_v.cu`` — split-mode INT3 attn
  * ``fused_quant_write_int3_sym``                 — store-time quant for K and V

``attn_with_scores`` and the fused correction methods raise
NotImplementedError; decode_path uses the split-mode path directly via
``int3_split.py`` when ``scheme.nbits_k == 3``.
"""
from __future__ import annotations

import torch

from jetfire_vllm.kernels.fused_quant_write import fused_quant_write_int3_sym


class _SymInt3KSymInt3VKernels:
    nbits_k = 3
    nbits_v = 3
    asymmetric_k = False
    asymmetric_v = False
    head_dim = 128
    group_size = 128
    supported_top_k: tuple[int, ...] = (
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    )

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        # Split-mode kernels are loaded lazily by decode_path via
        # ``int3_split.load_qk_scores_ext`` / ``load_softmax_v_ext``.
        self._loaded = True

    # ── Decode ──────────────────────────────────────────────────────────

    def attn_with_scores(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt3K_SymInt3V has no fused WMMA decode kernel — "
            "set JETFIRE_USE_SPLIT_INT2=1 to use the split-mode path."
        )

    def fused_pre_correction_batched(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt3K_SymInt3V has no GPU pre-correction kernel — "
            "set JETFIRE_DECODE_CPU_ATTN=1 to use the CPU correction path."
        )

    def fused_post_correction(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt3K_SymInt3V has no GPU post-correction kernel — "
            "set JETFIRE_DECODE_CPU_ATTN=1 to use the CPU correction path."
        )

    # ── Store ───────────────────────────────────────────────────────────

    def quant_write_k(
        self, key_tok, slot_idx, k_data, k_scales, k_zp, group_size,
    ):
        if k_zp is not None:
            raise ValueError(
                "SymInt3K does not use a k_zp tensor; got non-None."
            )
        fused_quant_write_int3_sym(
            key_tok, slot_idx, k_data, k_scales, int(group_size),
        )

    def quant_write_v(
        self, value_tok, slot_idx, v_data, v_scales, v_zp, group_size,
    ):
        if v_zp is not None:
            raise ValueError(
                "SymInt3V does not use a v_zp tensor; got non-None."
            )
        fused_quant_write_int3_sym(
            value_tok, slot_idx, v_data, v_scales, int(group_size),
        )


KERNELS = _SymInt3KSymInt3VKernels()
