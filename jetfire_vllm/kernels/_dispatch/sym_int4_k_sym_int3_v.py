"""KernelSet for SymInt4K_SymInt3V — INT4 sym K + INT3 sym V (mixed precision).

Reuses the existing INT4 K kernel (Stage A) and INT3 V kernel (Stage B);
the K and V dispatch are independent in decode_path.  Split-mode +
cpu-correction only.
"""
from __future__ import annotations

from jetfire_vllm.kernels.fused_quant_write import (
    fused_quant_write_int3_sym,
    fused_quant_write_int4_sym,
)


class _SymInt4KSymInt3VKernels:
    nbits_k = 4
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
        self._loaded = True

    def attn_with_scores(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt4K_SymInt3V has no fused WMMA decode kernel — "
            "set JETFIRE_USE_SPLIT_INT2=1."
        )

    def fused_pre_correction_batched(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt4K_SymInt3V has no GPU pre-correction kernel — "
            "set JETFIRE_DECODE_CPU_ATTN=1."
        )

    def fused_post_correction(self, *args, **kwargs):
        raise NotImplementedError(
            "SymInt4K_SymInt3V has no GPU post-correction kernel — "
            "set JETFIRE_DECODE_CPU_ATTN=1."
        )

    def quant_write_k(self, key_tok, slot_idx, k_data, k_scales, k_zp, group_size):
        if k_zp is not None:
            raise ValueError("SymInt4K does not use a k_zp tensor.")
        fused_quant_write_int4_sym(key_tok, slot_idx, k_data, k_scales, int(group_size))

    def quant_write_v(self, value_tok, slot_idx, v_data, v_scales, v_zp, group_size):
        if v_zp is not None:
            raise ValueError("SymInt3V does not use a v_zp tensor.")
        fused_quant_write_int3_sym(value_tok, slot_idx, v_data, v_scales, int(group_size))


KERNELS = _SymInt4KSymInt3VKernels()
