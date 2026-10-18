"""Fused quantize+pack+cache-scatter in one Triton launch.

Takes bf16 input (T, H_kv, D) plus per-token slot indices, writes directly
into the INT2-packed HBM cache tensors.  Replaces the chain:

    k_pack, k_scale, k_zp = quantize_and_pack_int2_asym(key_tok)      # Triton
    k_pack = k_pack.permute(1, 0, 2).contiguous()                     # kernel
    k_scale = k_scale.permute(1, 0, 2).contiguous()                   # kernel
    k_zp = k_zp.permute(1, 0, 2).contiguous()                         # kernel
    self.k_data[:, slots, :] = k_pack.to(self.device)                 # kernel
    self.k_scales[:, slots, :] = k_scale.to(..., dtype=bf16)          # kernel
    self.k_zp[:, slots, :] = k_zp.to(self.device)                     # kernel

→ one launch.  Saves ~6 kernel launches × 32 layers = 192 launches per
decode token.

Kernel grid: (T, H_kv) — one program per (token, kv_head).
Each program:
  1. Load D bf16 values from input[t, h, :]
  2. Compute per-group scale (+ zero-point if asym)
  3. Quantize + pack into D/4 uint8 bytes
  4. Look up slot = slots[t]
  5. Store packed bytes → out_data[h, slot, :]
  6. Store scale / zp   → out_scale[h, slot, :] / out_zp[h, slot, :]
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_quant_write_int2_asym_kernel(
    X_ptr,            # (T, H_kv, D) bf16
    Slots_ptr,        # (T,) int64
    OutData_ptr,      # (H_kv, num_slots, D/4) uint8
    OutScale_ptr,     # (H_kv, num_slots, NG) bf16
    OutZp_ptr,        # (H_kv, num_slots, NG) bf16
    T,                # num tokens (runtime)
    H_kv: tl.constexpr,
    D: tl.constexpr,
    GS: tl.constexpr,
    NG: tl.constexpr,
    D_QUARTER: tl.constexpr,
    x_stride_t,       # stride in bf16 elements (T axis)
    x_stride_h,       # stride in bf16 elements (H_kv axis)
    out_data_stride_h,      # stride in bytes (H_kv axis)
    out_data_stride_slot,   # = D_QUARTER
    out_scale_stride_h,     # stride in bf16 elements (H_kv axis)
    out_scale_stride_slot,  # = NG
    out_zp_stride_h,
    out_zp_stride_slot,
    QMAX: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    """One program per (token, kv_head).  Grid = (T, H_kv)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return

    # Load slot index for this token.
    slot = tl.load(Slots_ptr + pid_t)

    # Mask out rows whose slot is < 0 (shouldn't happen with our write_slots
    # filter but kept for safety).
    if slot < 0:
        return

    # Input row pointer: X[pid_t, pid_h, :]
    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h

    # Output row pointers.
    out_data_row = OutData_ptr + pid_h * out_data_stride_h + slot * out_data_stride_slot
    out_scale_row = OutScale_ptr + pid_h * out_scale_stride_h + slot * out_scale_stride_slot
    out_zp_row = OutZp_ptr + pid_h * out_zp_stride_h + slot * out_zp_stride_slot

    # Process each group independently (one group per NG loop iteration;
    # for the paper config group_size == D so NG == 1 — loop trips once).
    for g in range(NG):
        col_start = g * GS
        col_off = col_start + tl.arange(0, BLOCK_GS)
        col_mask = tl.arange(0, BLOCK_GS) < GS
        x = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)

        # Full-codebook INT2 asym:
        #   stored code ∈ [0, 3], signed q = code - 2 ∈ [-2, 1]
        #   scale=(max-min)/3, zp=min+2*scale.
        x_for_max = tl.where(col_mask, x, -float("inf"))
        x_for_min = tl.where(col_mask, x,  float("inf"))
        xmax = tl.max(x_for_max, axis=0)
        xmin = tl.min(x_for_min, axis=0)
        scale = tl.maximum((xmax - xmin) / (float(OFFSET) + float(QMAX)), 1e-8)
        zp = xmin + float(OFFSET) * scale
        scale_bf16 = scale.to(tl.bfloat16)
        zp_bf16 = zp.to(tl.bfloat16)
        tl.store(out_scale_row + g, scale_bf16)
        tl.store(out_zp_row + g, zp_bf16)
        scale_f = scale_bf16.to(tl.float32)
        zp_f = zp_bf16.to(tl.float32)

        # Pack 4 int2 values per byte.
        byte_off_in_group = tl.arange(0, BLOCK_GS // 4)
        byte_mask = byte_off_in_group < (GS // 4)
        stride4 = tl.arange(0, BLOCK_GS // 4) * 4

        x0 = tl.load(x_row + col_start + stride4 + 0, mask=byte_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_row + col_start + stride4 + 1, mask=byte_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_row + col_start + stride4 + 2, mask=byte_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_row + col_start + stride4 + 3, mask=byte_mask, other=0.0).to(tl.float32)

        qf0 = libdevice.rint((x0 - zp_f) / scale_f)
        qf0 = tl.minimum(tl.maximum(qf0, -float(OFFSET)), float(QMAX))
        v0 = qf0.to(tl.int32) + OFFSET
        qf1 = libdevice.rint((x1 - zp_f) / scale_f)
        qf1 = tl.minimum(tl.maximum(qf1, -float(OFFSET)), float(QMAX))
        v1 = qf1.to(tl.int32) + OFFSET
        qf2 = libdevice.rint((x2 - zp_f) / scale_f)
        qf2 = tl.minimum(tl.maximum(qf2, -float(OFFSET)), float(QMAX))
        v2 = qf2.to(tl.int32) + OFFSET
        qf3 = libdevice.rint((x3 - zp_f) / scale_f)
        qf3 = tl.minimum(tl.maximum(qf3, -float(OFFSET)), float(QMAX))
        v3 = qf3.to(tl.int32) + OFFSET

        packed = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
        byte_dst = col_start // 4 + byte_off_in_group
        tl.store(out_data_row + byte_dst, packed.to(tl.uint8), mask=byte_mask)


@triton.jit
def _fused_quant_write_int2_sym_kernel(
    X_ptr, Slots_ptr, OutData_ptr, OutScale_ptr,
    T, H_kv: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
    D_QUARTER: tl.constexpr,
    x_stride_t, x_stride_h,
    out_data_stride_h, out_data_stride_slot,
    out_scale_stride_h, out_scale_stride_slot,
    QMAX: tl.constexpr, OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return
    slot = tl.load(Slots_ptr + pid_t)
    if slot < 0:
        return

    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    out_data_row = OutData_ptr + pid_h * out_data_stride_h + slot * out_data_stride_slot
    out_scale_row = OutScale_ptr + pid_h * out_scale_stride_h + slot * out_scale_stride_slot

    for g in range(NG):
        col_start = g * GS
        col_off = col_start + tl.arange(0, BLOCK_GS)
        col_mask = tl.arange(0, BLOCK_GS) < GS
        x = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax / QMAX, 1e-8)
        scale_bf16 = scale.to(tl.bfloat16)
        tl.store(out_scale_row + g, scale_bf16)
        scale_f = scale_bf16.to(tl.float32)

        byte_off_in_group = tl.arange(0, BLOCK_GS // 4)
        byte_mask = byte_off_in_group < (GS // 4)
        stride4 = tl.arange(0, BLOCK_GS // 4) * 4

        x0 = tl.load(x_row + col_start + stride4 + 0, mask=byte_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_row + col_start + stride4 + 1, mask=byte_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_row + col_start + stride4 + 2, mask=byte_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_row + col_start + stride4 + 3, mask=byte_mask, other=0.0).to(tl.float32)

        qf0 = libdevice.rint(x0 / scale_f)
        qf0 = tl.minimum(tl.maximum(qf0, -float(QMAX)), float(QMAX))
        v0 = qf0.to(tl.int32) + OFFSET
        qf1 = libdevice.rint(x1 / scale_f)
        qf1 = tl.minimum(tl.maximum(qf1, -float(QMAX)), float(QMAX))
        v1 = qf1.to(tl.int32) + OFFSET
        qf2 = libdevice.rint(x2 / scale_f)
        qf2 = tl.minimum(tl.maximum(qf2, -float(QMAX)), float(QMAX))
        v2 = qf2.to(tl.int32) + OFFSET
        qf3 = libdevice.rint(x3 / scale_f)
        qf3 = tl.minimum(tl.maximum(qf3, -float(QMAX)), float(QMAX))
        v3 = qf3.to(tl.int32) + OFFSET

        packed = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
        byte_dst = col_start // 4 + byte_off_in_group
        tl.store(out_data_row + byte_dst, packed.to(tl.uint8), mask=byte_mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_quant_write_int2_asym(
    key_tok: torch.Tensor,   # (T, H_kv, D) bf16
    slots: torch.Tensor,     # (T,) int64, already on key_tok.device
    out_data: torch.Tensor,  # (H_kv, num_slots, D/4) uint8
    out_scale: torch.Tensor, # (H_kv, num_slots, NG) bf16
    out_zp: torch.Tensor,    # (H_kv, num_slots, NG) bf16
    group_size: int,
) -> None:
    """In-place fused INT2-asymmetric quantize + pack + scatter to cache."""
    assert key_tok.is_cuda and key_tok.dtype == torch.bfloat16
    T, H_kv, D = key_tok.shape
    ng = D // group_size
    d_quarter = D // 4
    assert D % group_size == 0 and D % 4 == 0
    BLOCK_GS = _next_pow2(group_size)

    # Make out_scale / out_zp indexable as (H_kv, num_slots, NG) even if
    # caller passed (H_kv, num_slots) — squeeze/reshape semantically.
    if out_scale.dim() == 2:
        out_scale = out_scale.unsqueeze(-1)
    if out_zp.dim() == 2:
        out_zp = out_zp.unsqueeze(-1)

    grid = (T, H_kv)
    _fused_quant_write_int2_asym_kernel[grid](
        key_tok, slots, out_data, out_scale, out_zp,
        T, H_kv=H_kv, D=D, GS=group_size, NG=ng, D_QUARTER=d_quarter,
        x_stride_t=key_tok.stride(0), x_stride_h=key_tok.stride(1),
        out_data_stride_h=out_data.stride(0),
        out_data_stride_slot=out_data.stride(1),
        out_scale_stride_h=out_scale.stride(0),
        out_scale_stride_slot=out_scale.stride(1),
        out_zp_stride_h=out_zp.stride(0),
        out_zp_stride_slot=out_zp.stride(1),
        QMAX=1, OFFSET=2, BLOCK_GS=BLOCK_GS,
    )


def fused_quant_write_int2_sym(
    val_tok: torch.Tensor,
    slots: torch.Tensor,
    out_data: torch.Tensor,
    out_scale: torch.Tensor,
    group_size: int,
) -> None:
    """In-place fused INT2-symmetric quantize + pack + scatter to cache."""
    assert val_tok.is_cuda and val_tok.dtype == torch.bfloat16
    T, H_kv, D = val_tok.shape
    ng = D // group_size
    d_quarter = D // 4
    assert D % group_size == 0 and D % 4 == 0
    BLOCK_GS = _next_pow2(group_size)

    if out_scale.dim() == 2:
        out_scale = out_scale.unsqueeze(-1)

    grid = (T, H_kv)
    _fused_quant_write_int2_sym_kernel[grid](
        val_tok, slots, out_data, out_scale,
        T, H_kv=H_kv, D=D, GS=group_size, NG=ng, D_QUARTER=d_quarter,
        x_stride_t=val_tok.stride(0), x_stride_h=val_tok.stride(1),
        out_data_stride_h=out_data.stride(0),
        out_data_stride_slot=out_data.stride(1),
        out_scale_stride_h=out_scale.stride(0),
        out_scale_stride_slot=out_scale.stride(1),
        QMAX=1, OFFSET=2, BLOCK_GS=BLOCK_GS,
    )


# ── INT4 symmetric ─────────────────────────────────────────────────────────
# Pack format (mirrors intN_pack._pack_int4):
#   2 values per byte, D_PACKED = D // 2
#   byte = ((q_even + 8) << 4) | (q_odd + 8)
#   where q_even = xq[..., 0::2], q_odd = xq[..., 1::2]
# Quant range: q ∈ [-7, +7] (qmax=7), offset=8.

@triton.jit
def _fused_quant_write_int4_sym_kernel(
    X_ptr, Slots_ptr, OutData_ptr, OutScale_ptr,
    T, H_kv: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
    D_HALF: tl.constexpr,
    x_stride_t, x_stride_h,
    out_data_stride_h, out_data_stride_slot,
    out_scale_stride_h, out_scale_stride_slot,
    QMAX: tl.constexpr, OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    """One program per (token, kv_head).  Grid = (T, H_kv)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return
    slot = tl.load(Slots_ptr + pid_t)
    if slot < 0:
        return

    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    out_data_row = OutData_ptr + pid_h * out_data_stride_h + slot * out_data_stride_slot
    out_scale_row = OutScale_ptr + pid_h * out_scale_stride_h + slot * out_scale_stride_slot

    for g in range(NG):
        col_start = g * GS
        col_off = col_start + tl.arange(0, BLOCK_GS)
        col_mask = tl.arange(0, BLOCK_GS) < GS
        x = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)

        # Symmetric: scale = amax / qmax
        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax / QMAX, 1e-8)
        scale_bf16 = scale.to(tl.bfloat16)
        tl.store(out_scale_row + g, scale_bf16)
        scale_f = scale_bf16.to(tl.float32)

        # Pack 2 int4 values per byte: byte = ((even+8) << 4) | (odd+8)
        byte_off_in_group = tl.arange(0, BLOCK_GS // 2)
        byte_mask = byte_off_in_group < (GS // 2)
        stride2 = tl.arange(0, BLOCK_GS // 2) * 2

        x_even = tl.load(x_row + col_start + stride2 + 0, mask=byte_mask, other=0.0).to(tl.float32)
        x_odd  = tl.load(x_row + col_start + stride2 + 1, mask=byte_mask, other=0.0).to(tl.float32)

        qfe = libdevice.rint(x_even / scale_f)
        qfe = tl.minimum(tl.maximum(qfe, -float(QMAX)), float(QMAX))
        ve = qfe.to(tl.int32) + OFFSET
        qfo = libdevice.rint(x_odd / scale_f)
        qfo = tl.minimum(tl.maximum(qfo, -float(QMAX)), float(QMAX))
        vo = qfo.to(tl.int32) + OFFSET

        packed = (ve << 4) | vo
        byte_dst = col_start // 2 + byte_off_in_group
        tl.store(out_data_row + byte_dst, packed.to(tl.uint8), mask=byte_mask)


def fused_quant_write_int4_sym(
    value_tok: torch.Tensor,   # (T, H_kv, D) bf16
    slot_idx: torch.Tensor,    # (T,) int64, on value_tok.device
    v_data: torch.Tensor,      # (H_kv, num_slots, D/2) uint8
    v_scales: torch.Tensor,    # (H_kv, num_slots, NG) bf16
    group_size: int,
) -> None:
    """In-place fused INT4-symmetric quantize + pack + scatter to cache.

    Output bytes match `intN_pack._pack_int4`:
        byte = ((q_even + 8) << 4) | (q_odd + 8)
    """
    assert value_tok.is_cuda and value_tok.dtype == torch.bfloat16
    T, H_kv, D = value_tok.shape
    ng = D // group_size
    d_half = D // 2
    assert D % group_size == 0 and D % 2 == 0
    BLOCK_GS = _next_pow2(group_size)

    if v_scales.dim() == 2:
        v_scales = v_scales.unsqueeze(-1)

    grid = (T, H_kv)
    _fused_quant_write_int4_sym_kernel[grid](
        value_tok, slot_idx, v_data, v_scales,
        T, H_kv=H_kv, D=D, GS=group_size, NG=ng, D_HALF=d_half,
        x_stride_t=value_tok.stride(0), x_stride_h=value_tok.stride(1),
        out_data_stride_h=v_data.stride(0),
        out_data_stride_slot=v_data.stride(1),
        out_scale_stride_h=v_scales.stride(0),
        out_scale_stride_slot=v_scales.stride(1),
        QMAX=7, OFFSET=8, BLOCK_GS=BLOCK_GS,
    )


@triton.jit
def _fused_quant_write_int4_sym_kv_kernel(
    K_ptr, V_ptr, Slots_ptr,
    KData_ptr, KScale_ptr, VData_ptr, VScale_ptr,
    T, H_kv: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
    D_HALF: tl.constexpr,
    k_stride_t, k_stride_h,
    v_stride_t, v_stride_h,
    k_data_stride_h, k_data_stride_slot,
    k_scale_stride_h, k_scale_stride_slot,
    v_data_stride_h, v_data_stride_slot,
    v_scale_stride_h, v_scale_stride_slot,
    QMAX: tl.constexpr, OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    """One program per (token, kv_head), quantizing K and V in one launch."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return
    slot = tl.load(Slots_ptr + pid_t)
    if slot < 0:
        return

    k_row = K_ptr + pid_t * k_stride_t + pid_h * k_stride_h
    v_row = V_ptr + pid_t * v_stride_t + pid_h * v_stride_h
    k_data_row = KData_ptr + pid_h * k_data_stride_h + slot * k_data_stride_slot
    k_scale_row = KScale_ptr + pid_h * k_scale_stride_h + slot * k_scale_stride_slot
    v_data_row = VData_ptr + pid_h * v_data_stride_h + slot * v_data_stride_slot
    v_scale_row = VScale_ptr + pid_h * v_scale_stride_h + slot * v_scale_stride_slot

    for g in range(NG):
        col_start = g * GS
        col_off = col_start + tl.arange(0, BLOCK_GS)
        col_mask = tl.arange(0, BLOCK_GS) < GS

        k = tl.load(k_row + col_off, mask=col_mask, other=0.0).to(tl.float32)
        k_amax = tl.max(tl.abs(k), axis=0)
        k_scale = tl.maximum(k_amax / QMAX, 1e-8)
        k_scale_bf16 = k_scale.to(tl.bfloat16)
        tl.store(k_scale_row + g, k_scale_bf16)
        k_scale_f = k_scale_bf16.to(tl.float32)

        v = tl.load(v_row + col_off, mask=col_mask, other=0.0).to(tl.float32)
        v_amax = tl.max(tl.abs(v), axis=0)
        v_scale = tl.maximum(v_amax / QMAX, 1e-8)
        v_scale_bf16 = v_scale.to(tl.bfloat16)
        tl.store(v_scale_row + g, v_scale_bf16)
        v_scale_f = v_scale_bf16.to(tl.float32)

        byte_off_in_group = tl.arange(0, BLOCK_GS // 2)
        byte_mask = byte_off_in_group < (GS // 2)
        stride2 = tl.arange(0, BLOCK_GS // 2) * 2

        k_even = tl.load(k_row + col_start + stride2 + 0, mask=byte_mask, other=0.0).to(tl.float32)
        k_odd = tl.load(k_row + col_start + stride2 + 1, mask=byte_mask, other=0.0).to(tl.float32)
        k_qe = libdevice.rint(k_even / k_scale_f)
        k_qe = tl.minimum(tl.maximum(k_qe, -float(QMAX)), float(QMAX))
        k_ve = k_qe.to(tl.int32) + OFFSET
        k_qo = libdevice.rint(k_odd / k_scale_f)
        k_qo = tl.minimum(tl.maximum(k_qo, -float(QMAX)), float(QMAX))
        k_vo = k_qo.to(tl.int32) + OFFSET
        k_packed = (k_ve << 4) | k_vo

        v_even = tl.load(v_row + col_start + stride2 + 0, mask=byte_mask, other=0.0).to(tl.float32)
        v_odd = tl.load(v_row + col_start + stride2 + 1, mask=byte_mask, other=0.0).to(tl.float32)
        v_qe = libdevice.rint(v_even / v_scale_f)
        v_qe = tl.minimum(tl.maximum(v_qe, -float(QMAX)), float(QMAX))
        v_ve = v_qe.to(tl.int32) + OFFSET
        v_qo = libdevice.rint(v_odd / v_scale_f)
        v_qo = tl.minimum(tl.maximum(v_qo, -float(QMAX)), float(QMAX))
        v_vo = v_qo.to(tl.int32) + OFFSET
        v_packed = (v_ve << 4) | v_vo

        byte_dst = col_start // 2 + byte_off_in_group
        tl.store(k_data_row + byte_dst, k_packed.to(tl.uint8), mask=byte_mask)
        tl.store(v_data_row + byte_dst, v_packed.to(tl.uint8), mask=byte_mask)


@triton.jit
def _fused_quant_write_int4_sym_kv_bt_kernel(
    K_ptr, V_ptr, Slots_ptr,
    KData_ptr, KScale_ptr, VData_ptr, VScale_ptr,
    T, H_kv: tl.constexpr, D: tl.constexpr, D_HALF: tl.constexpr,
    k_stride_t, k_stride_h,
    v_stride_t, v_stride_h,
    k_data_stride_h, k_data_stride_slot,
    k_scale_stride_h, k_scale_stride_slot,
    v_data_stride_h, v_data_stride_slot,
    v_scale_stride_h, v_scale_stride_slot,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    QMAX: tl.constexpr, OFFSET: tl.constexpr,
):
    """One program per (token block, kv_head) for D==group_size INT4 K+V."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_b * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    slots = tl.load(Slots_ptr + offs_t, mask=mask_t, other=-1)
    valid_t = mask_t & (slots >= 0)

    k_row = K_ptr + offs_t[:, None] * k_stride_t + pid_h * k_stride_h + offs_d[None, :]
    v_row = V_ptr + offs_t[:, None] * v_stride_t + pid_h * v_stride_h + offs_d[None, :]
    elem_mask = valid_t[:, None] & (offs_d[None, :] < D)

    k = tl.load(k_row, mask=elem_mask, other=0.0).to(tl.float32)
    v = tl.load(v_row, mask=elem_mask, other=0.0).to(tl.float32)
    k_amax = tl.max(tl.abs(k), axis=1)
    v_amax = tl.max(tl.abs(v), axis=1)
    k_scale = tl.maximum(k_amax / QMAX, 1e-8)
    v_scale = tl.maximum(v_amax / QMAX, 1e-8)
    k_scale_bf16 = k_scale.to(tl.bfloat16)
    v_scale_bf16 = v_scale.to(tl.bfloat16)

    tl.store(
        KScale_ptr + pid_h * k_scale_stride_h + slots * k_scale_stride_slot,
        k_scale_bf16,
        mask=valid_t,
    )
    tl.store(
        VScale_ptr + pid_h * v_scale_stride_h + slots * v_scale_stride_slot,
        v_scale_bf16,
        mask=valid_t,
    )

    offs_b = tl.arange(0, BLOCK_D // 2)
    byte_mask = valid_t[:, None] & (offs_b[None, :] < D_HALF)
    stride2 = offs_b * 2
    k_even = tl.load(
        K_ptr + offs_t[:, None] * k_stride_t + pid_h * k_stride_h + stride2[None, :],
        mask=byte_mask,
        other=0.0,
    ).to(tl.float32)
    k_odd = tl.load(
        K_ptr + offs_t[:, None] * k_stride_t + pid_h * k_stride_h + stride2[None, :] + 1,
        mask=byte_mask,
        other=0.0,
    ).to(tl.float32)
    v_even = tl.load(
        V_ptr + offs_t[:, None] * v_stride_t + pid_h * v_stride_h + stride2[None, :],
        mask=byte_mask,
        other=0.0,
    ).to(tl.float32)
    v_odd = tl.load(
        V_ptr + offs_t[:, None] * v_stride_t + pid_h * v_stride_h + stride2[None, :] + 1,
        mask=byte_mask,
        other=0.0,
    ).to(tl.float32)

    k_scale_f = k_scale_bf16.to(tl.float32)[:, None]
    v_scale_f = v_scale_bf16.to(tl.float32)[:, None]

    k_qe = libdevice.rint(k_even / k_scale_f)
    k_qe = tl.minimum(tl.maximum(k_qe, -float(QMAX)), float(QMAX))
    k_qo = libdevice.rint(k_odd / k_scale_f)
    k_qo = tl.minimum(tl.maximum(k_qo, -float(QMAX)), float(QMAX))
    v_qe = libdevice.rint(v_even / v_scale_f)
    v_qe = tl.minimum(tl.maximum(v_qe, -float(QMAX)), float(QMAX))
    v_qo = libdevice.rint(v_odd / v_scale_f)
    v_qo = tl.minimum(tl.maximum(v_qo, -float(QMAX)), float(QMAX))

    k_packed = ((k_qe.to(tl.int32) + OFFSET) << 4) | (k_qo.to(tl.int32) + OFFSET)
    v_packed = ((v_qe.to(tl.int32) + OFFSET) << 4) | (v_qo.to(tl.int32) + OFFSET)
    k_dst = (
        KData_ptr
        + pid_h * k_data_stride_h
        + slots[:, None] * k_data_stride_slot
        + offs_b[None, :]
    )
    v_dst = (
        VData_ptr
        + pid_h * v_data_stride_h
        + slots[:, None] * v_data_stride_slot
        + offs_b[None, :]
    )
    tl.store(k_dst, k_packed.to(tl.uint8), mask=byte_mask)
    tl.store(v_dst, v_packed.to(tl.uint8), mask=byte_mask)


def fused_quant_write_int4_sym_kv(
    key_tok: torch.Tensor,
    value_tok: torch.Tensor,
    slot_idx: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    group_size: int,
) -> None:
    """Fused INT4-symmetric K+V quantize + pack + scatter in one launch."""
    assert key_tok.is_cuda and value_tok.is_cuda
    assert key_tok.dtype == torch.bfloat16 and value_tok.dtype == torch.bfloat16
    T, H_kv, D = key_tok.shape
    assert value_tok.shape == key_tok.shape
    ng = D // group_size
    d_half = D // 2
    assert D % group_size == 0 and D % 2 == 0
    block_t = int(os.environ.get("JETFIRE_INT4_KV_BLOCK_T", "8"))
    if block_t > 1 and group_size == D:
        if k_scales.dim() == 2:
            k_scales = k_scales.unsqueeze(-1)
        if v_scales.dim() == 2:
            v_scales = v_scales.unsqueeze(-1)
        grid = (triton.cdiv(T, block_t), H_kv)
        _fused_quant_write_int4_sym_kv_bt_kernel[grid](
            key_tok, value_tok, slot_idx,
            k_data, k_scales, v_data, v_scales,
            T, H_kv=H_kv, D=D, D_HALF=d_half,
            k_stride_t=key_tok.stride(0), k_stride_h=key_tok.stride(1),
            v_stride_t=value_tok.stride(0), v_stride_h=value_tok.stride(1),
            k_data_stride_h=k_data.stride(0),
            k_data_stride_slot=k_data.stride(1),
            k_scale_stride_h=k_scales.stride(0),
            k_scale_stride_slot=k_scales.stride(1),
            v_data_stride_h=v_data.stride(0),
            v_data_stride_slot=v_data.stride(1),
            v_scale_stride_h=v_scales.stride(0),
            v_scale_stride_slot=v_scales.stride(1),
            BLOCK_T=block_t, BLOCK_D=_next_pow2(D), QMAX=7, OFFSET=8,
        )
        return
    BLOCK_GS = _next_pow2(group_size)

    if k_scales.dim() == 2:
        k_scales = k_scales.unsqueeze(-1)
    if v_scales.dim() == 2:
        v_scales = v_scales.unsqueeze(-1)

    grid = (T, H_kv)
    _fused_quant_write_int4_sym_kv_kernel[grid](
        key_tok, value_tok, slot_idx,
        k_data, k_scales, v_data, v_scales,
        T, H_kv=H_kv, D=D, GS=group_size, NG=ng, D_HALF=d_half,
        k_stride_t=key_tok.stride(0), k_stride_h=key_tok.stride(1),
        v_stride_t=value_tok.stride(0), v_stride_h=value_tok.stride(1),
        k_data_stride_h=k_data.stride(0),
        k_data_stride_slot=k_data.stride(1),
        k_scale_stride_h=k_scales.stride(0),
        k_scale_stride_slot=k_scales.stride(1),
        v_data_stride_h=v_data.stride(0),
        v_data_stride_slot=v_data.stride(1),
        v_scale_stride_h=v_scales.stride(0),
        v_scale_stride_slot=v_scales.stride(1),
        QMAX=7, OFFSET=8, BLOCK_GS=BLOCK_GS,
    )


# ── INT3 symmetric ─────────────────────────────────────────────────────────
# Pack format (mirrors intN_pack._pack_int3_planar):
#   8 values → 3 bytes, planar layout, D_PACKED = D * 3 // 8.
#   For each group of 8 values:
#     combined = (q[0]+4)<<21 | (q[1]+4)<<18 | (q[2]+4)<<15 | (q[3]+4)<<12
#              | (q[4]+4)<<9  | (q[5]+4)<<6  | (q[6]+4)<<3  | (q[7]+4)
#     byte0 = (combined >> 16) & 0xFF
#     byte1 = (combined >> 8)  & 0xFF
#     byte2 =  combined        & 0xFF
#   Per-(h, slot) layout: [plane0: n_groups bytes | plane1 | plane2],
#   where n_groups = D // 8.
# Quant range: q ∈ [-3, +3] (qmax=3), offset=4 → unsigned [1, 7].

@triton.jit
def _fused_quant_write_int3_sym_kernel(
    X_ptr, Slots_ptr, OutData_ptr, OutScale_ptr,
    T, H_kv: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
    N_GROUPS_PER_HEAD: tl.constexpr,  # D // 8 (planar plane size in bytes)
    x_stride_t, x_stride_h,
    out_data_stride_h, out_data_stride_slot,
    out_scale_stride_h, out_scale_stride_slot,
    QMAX: tl.constexpr, OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    """One program per (token, kv_head).  Grid = (T, H_kv).

    Produces planar INT3 output identical to `_pack_int3_planar`.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return
    slot = tl.load(Slots_ptr + pid_t)
    if slot < 0:
        return

    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    out_data_row = OutData_ptr + pid_h * out_data_stride_h + slot * out_data_stride_slot
    out_scale_row = OutScale_ptr + pid_h * out_scale_stride_h + slot * out_scale_stride_slot

    # plane offsets within the (h, slot) byte row.
    plane0_off = 0
    plane1_off = N_GROUPS_PER_HEAD
    plane2_off = 2 * N_GROUPS_PER_HEAD

    for g in range(NG):
        col_start = g * GS
        col_off = col_start + tl.arange(0, BLOCK_GS)
        col_mask = tl.arange(0, BLOCK_GS) < GS
        x = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)

        # Symmetric: scale = amax / qmax
        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax / QMAX, 1e-8)
        scale_bf16 = scale.to(tl.bfloat16)
        tl.store(out_scale_row + g, scale_bf16)
        scale_f = scale_bf16.to(tl.float32)

        # Each group of GS elements contains GS // 8 sub-groups of 8 values
        # → GS // 8 packed (byte0, byte1, byte2) triples.
        sub_off = tl.arange(0, BLOCK_GS // 8)
        sub_mask = sub_off < (GS // 8)
        stride8 = sub_off * 8

        # Load 8 values per sub-group.
        x0 = tl.load(x_row + col_start + stride8 + 0, mask=sub_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_row + col_start + stride8 + 1, mask=sub_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_row + col_start + stride8 + 2, mask=sub_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_row + col_start + stride8 + 3, mask=sub_mask, other=0.0).to(tl.float32)
        x4 = tl.load(x_row + col_start + stride8 + 4, mask=sub_mask, other=0.0).to(tl.float32)
        x5 = tl.load(x_row + col_start + stride8 + 5, mask=sub_mask, other=0.0).to(tl.float32)
        x6 = tl.load(x_row + col_start + stride8 + 6, mask=sub_mask, other=0.0).to(tl.float32)
        x7 = tl.load(x_row + col_start + stride8 + 7, mask=sub_mask, other=0.0).to(tl.float32)

        # Quantize each lane: q = round(x / scale), clamp to [-qmax, +qmax],
        # then add OFFSET so unsigned q ∈ [1, 7] (3 bits, low bit may be unused
        # at saturation — matches _pack_int3_planar).
        qf0 = libdevice.rint(x0 / scale_f)
        qf0 = tl.minimum(tl.maximum(qf0, -float(QMAX)), float(QMAX))
        v0 = qf0.to(tl.int32) + OFFSET
        qf1 = libdevice.rint(x1 / scale_f)
        qf1 = tl.minimum(tl.maximum(qf1, -float(QMAX)), float(QMAX))
        v1 = qf1.to(tl.int32) + OFFSET
        qf2 = libdevice.rint(x2 / scale_f)
        qf2 = tl.minimum(tl.maximum(qf2, -float(QMAX)), float(QMAX))
        v2 = qf2.to(tl.int32) + OFFSET
        qf3 = libdevice.rint(x3 / scale_f)
        qf3 = tl.minimum(tl.maximum(qf3, -float(QMAX)), float(QMAX))
        v3 = qf3.to(tl.int32) + OFFSET
        qf4 = libdevice.rint(x4 / scale_f)
        qf4 = tl.minimum(tl.maximum(qf4, -float(QMAX)), float(QMAX))
        v4 = qf4.to(tl.int32) + OFFSET
        qf5 = libdevice.rint(x5 / scale_f)
        qf5 = tl.minimum(tl.maximum(qf5, -float(QMAX)), float(QMAX))
        v5 = qf5.to(tl.int32) + OFFSET
        qf6 = libdevice.rint(x6 / scale_f)
        qf6 = tl.minimum(tl.maximum(qf6, -float(QMAX)), float(QMAX))
        v6 = qf6.to(tl.int32) + OFFSET
        qf7 = libdevice.rint(x7 / scale_f)
        qf7 = tl.minimum(tl.maximum(qf7, -float(QMAX)), float(QMAX))
        v7 = qf7.to(tl.int32) + OFFSET

        # Combine 8 × 3-bit values into 24-bit MSB-first integer.
        combined = (
            (v0 << 21)
            | (v1 << 18)
            | (v2 << 15)
            | (v3 << 12)
            | (v4 << 9)
            | (v5 << 6)
            | (v6 << 3)
            | v7
        )

        byte0 = ((combined >> 16) & 0xFF).to(tl.uint8)
        byte1 = ((combined >> 8) & 0xFF).to(tl.uint8)
        byte2 = (combined & 0xFF).to(tl.uint8)

        # Sub-group index inside this (h, slot) plane.
        plane_idx = g * (GS // 8) + sub_off

        tl.store(out_data_row + plane0_off + plane_idx, byte0, mask=sub_mask)
        tl.store(out_data_row + plane1_off + plane_idx, byte1, mask=sub_mask)
        tl.store(out_data_row + plane2_off + plane_idx, byte2, mask=sub_mask)


def fused_quant_write_int3_sym(
    value_tok: torch.Tensor,   # (T, H_kv, D) bf16
    slot_idx: torch.Tensor,    # (T,) int64, on value_tok.device
    v_data: torch.Tensor,      # (H_kv, num_slots, D*3//8) uint8
    v_scales: torch.Tensor,    # (H_kv, num_slots, NG) bf16
    group_size: int,
) -> None:
    """In-place fused INT3-symmetric quantize + pack + scatter to cache.

    Output bytes match `intN_pack._pack_int3_planar`: per (h, slot), the
    `D*3//8` bytes are organised as three contiguous planes of `D//8` bytes
    each (byte0 / byte1 / byte2 of the MSB-first 24-bit packing of every
    8-value subgroup).
    """
    assert value_tok.is_cuda and value_tok.dtype == torch.bfloat16
    T, H_kv, D = value_tok.shape
    ng = D // group_size
    n_groups_per_head = D // 8
    assert D % group_size == 0 and D % 8 == 0
    BLOCK_GS = _next_pow2(group_size)

    if v_scales.dim() == 2:
        v_scales = v_scales.unsqueeze(-1)

    grid = (T, H_kv)
    _fused_quant_write_int3_sym_kernel[grid](
        value_tok, slot_idx, v_data, v_scales,
        T, H_kv=H_kv, D=D, GS=group_size, NG=ng,
        N_GROUPS_PER_HEAD=n_groups_per_head,
        x_stride_t=value_tok.stride(0), x_stride_h=value_tok.stride(1),
        out_data_stride_h=v_data.stride(0),
        out_data_stride_slot=v_data.stride(1),
        out_scale_stride_h=v_scales.stride(0),
        out_scale_stride_slot=v_scales.stride(1),
        QMAX=3, OFFSET=4, BLOCK_GS=BLOCK_GS,
    )
