"""KernelSet for the AsymInt2K4Level_SymInt2V scheme.

Wraps three CUDA extensions:
  * ``fused_decode_wmma.cu`` — WMMA-v5 INT2 attention with raw-scores output
  * ``fused_correction.cu``  — fused pre/post-correction kernels
  * ``fused_quant_write``    — INT2 asym K + sym V store-time quant kernels

All three are JIT-built on first ``load()`` call and cached.
"""
from __future__ import annotations

import torch

from jetfire_vllm.adapter._gpu_ext import load_wmma_ext
from jetfire_vllm.kernels.fused_correction import load_fused_correction_ext
from jetfire_vllm.kernels.fused_quant_write import (
    fused_quant_write_int2_asym,
    fused_quant_write_int2_sym,
)


class _Int2AsymKSymVKernels:
    nbits_k = 2
    nbits_v = 2
    asymmetric_k = True
    asymmetric_v = False
    head_dim = 128
    group_size = 128
    # Release split Stage A/B uses radix-select topK.
    supported_top_k: tuple[int, ...] = (
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    )

    def __init__(self) -> None:
        self._wmma = None
        self._fc = None

    def load(self) -> None:
        if self._wmma is None:
            self._wmma = load_wmma_ext()
        if self._fc is None:
            self._fc = load_fused_correction_ext()
        if self._wmma is None or self._fc is None:
            raise RuntimeError(
                "AsymInt2K4Level_SymInt2V kernel build failed (wmma=%s, fc=%s)"
                % (self._wmma, self._fc)
            )

    # ── Decode ──────────────────────────────────────────────────────────

    def attn_with_scores(
        self,
        q, k_data, k_scales, k_zp, v_data, v_scales,
        block_table, seq_lens, max_S, H_q, H_kv, block_size, scale,
    ):
        if self._wmma is None:
            self.load()
        return self._wmma.fused_decode_wmma_v5_int2_with_scores(
            q, k_data, k_scales, k_zp,
            v_data, v_scales,
            block_table, seq_lens,
            int(max_S), int(H_q), int(H_kv), int(block_size), 0,
            float(scale),
        )

    def fused_pre_correction_batched(
        self,
        scores, topk_idx, block_table, seq_lens,
        v_data, v_scales, block_size, H_kv, num_groups,
    ):
        if self._fc is None:
            self.load()
        return self._fc.fused_pre_correction_batched(
            scores, topk_idx, block_table, seq_lens,
            v_data, v_scales,
            int(block_size), int(H_kv), int(num_groups),
        )

    def fused_post_correction(
        self, q, k_topk, v_topk, o_hat, o_old, W, scale,
    ):
        if self._fc is None:
            self.load()
        return self._fc.fused_post_correction(
            q, k_topk, v_topk, o_hat, o_old, W, float(scale),
        )

    # ── Store ───────────────────────────────────────────────────────────

    def quant_write_k(
        self, key_tok, slot_idx, k_data, k_scales, k_zp, group_size,
    ):
        if k_zp is None:
            raise ValueError(
                "AsymInt2K4Level requires a k_zp tensor; got None."
            )
        fused_quant_write_int2_asym(
            key_tok, slot_idx, k_data, k_scales, k_zp, int(group_size),
        )

    def quant_write_v(
        self, value_tok, slot_idx, v_data, v_scales, v_zp, group_size,
    ):
        if v_zp is not None:
            raise ValueError(
                "SymInt2V does not support a v_zp tensor; got non-None."
            )
        fused_quant_write_int2_sym(
            value_tok, slot_idx, v_data, v_scales, int(group_size),
        )


KERNELS = _Int2AsymKSymVKernels()
