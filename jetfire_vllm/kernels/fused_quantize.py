"""Fused Triton kernels for KV-cache quantization + packing.

Replaces the torch-op chain in ``intN_pack.quantize_and_pack(_asym)`` — a
15-to-30 op pipeline (abs → amax → scale → divide → round → clamp → cast →
slice-and-shift) with a single kernel launch per tensor.  This drops
per-layer Python dispatch time on main thread from ~2 ms to ~0.3 ms,
which is the dominant un-overlappable cost in prefill pipelining.

Supported today:
    INT2 symmetric   (for V cache in the paper config)
    INT2 asymmetric  (for K cache in the paper config)

Each kernel writes output in the SAME shape as the torch reference:
    data   (..., D // 4)   uint8   (INT2: 4 values/byte, big-endian)
    scales (..., D // gs)  bfloat16
    zps    (..., D // gs)  bfloat16   (asym only)

so the caller's downstream permute + fancy-index-put logic is unchanged.
Leading dims are flattened internally; D and group_size are compile-time.

Numerical parity with the torch reference: exact match when group_size
== D (single group per row) because both paths reduce to the same
arithmetic in fp32.  Multi-group parity confirmed on random inputs via
the unit test in /tmp/test_fused_quant.py.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ── INT2 symmetric ───────────────────────────────────────────────────────────
@triton.jit
def _fused_quant_int2_sym_kernel(
    X_ptr,                 # (N_ROWS, D) bf16  — flattened leading dims
    Out_data_ptr,          # (N_ROWS, D_QUARTER) uint8
    Out_scale_ptr,         # (N_ROWS, NG)        bf16
    row_stride_x,
    row_stride_out_data,
    row_stride_out_scale,
    N_ROWS,
    D: tl.constexpr,
    GS: tl.constexpr,              # group size
    NG: tl.constexpr,              # D // GS
    D_QUARTER: tl.constexpr,       # D // 4 (INT2 pack factor)
    QMAX: tl.constexpr,            # INT2 qmax = 1
    OFFSET: tl.constexpr,          # INT2 offset = 2
    BLOCK_GS: tl.constexpr,        # power-of-two >= GS
):
    """One program per (row, group).  Grid = (N_ROWS, NG).

    Layout strategy for the pack:  tile `q` is shape (GS,) but we need it
    as (GS/4, 4) to combine 4 consecutive int2 values into a byte.  Triton
    lets us do this with 4 masked loads at offsets 0..3 — we must ensure
    each of those loads uses the SAME scale (the group's scale), so there
    is a single division site in registers rather than 4 separate ones.
    """
    pid_row = tl.program_id(0)
    pid_grp = tl.program_id(1)
    if pid_row >= N_ROWS:
        return

    x_row = X_ptr + pid_row * row_stride_x
    out_data_row = Out_data_ptr + pid_row * row_stride_out_data
    out_scale_row = Out_scale_ptr + pid_row * row_stride_out_scale

    # Phase 1: compute and store scale from the full group.
    col_start = pid_grp * GS
    col_off = col_start + tl.arange(0, BLOCK_GS)
    col_mask = tl.arange(0, BLOCK_GS) < GS
    x_group = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x_group), axis=0)
    scale = tl.maximum(amax / QMAX, 1e-8)
    scale_bf16 = scale.to(tl.bfloat16)
    scale_f = scale_bf16.to(tl.float32)
    tl.store(out_scale_row + pid_grp, scale_bf16)

    # Phase 2: quantize all GS elements (same scale), then pack 4-at-a-time.
    # Reshape `x_group` into (BLOCK_GS/4, 4) via strided reloads of the
    # same bf16 memory — this guarantees bit-identical values to Phase 1
    # (torch reference computes quant the same way: single xg / scale).
    n_bytes = GS // 4
    byte_off = tl.arange(0, BLOCK_GS // 4)
    byte_mask = byte_off < n_bytes
    byte_start = col_start // 4

    stride4 = tl.arange(0, BLOCK_GS // 4) * 4
    x0 = tl.load(x_row + col_start + stride4 + 0, mask=byte_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_row + col_start + stride4 + 1, mask=byte_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + col_start + stride4 + 2, mask=byte_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_row + col_start + stride4 + 3, mask=byte_mask, other=0.0).to(tl.float32)

    # Inline quant op (Triton can't hold nested `def` under @jit).
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
    tl.store(out_data_row + byte_start + byte_off, packed.to(tl.uint8),
             mask=byte_mask)


# ── INT2 asymmetric ──────────────────────────────────────────────────────────
@triton.jit
def _fused_quant_int2_asym_kernel(
    X_ptr,
    Out_data_ptr,
    Out_scale_ptr,
    Out_zp_ptr,
    row_stride_x,
    row_stride_out_data,
    row_stride_out_scale,
    row_stride_out_zp,
    N_ROWS,
    D: tl.constexpr,
    GS: tl.constexpr,
    NG: tl.constexpr,
    D_QUARTER: tl.constexpr,
    QMAX: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_grp = tl.program_id(1)
    if pid_row >= N_ROWS:
        return

    x_row = X_ptr + pid_row * row_stride_x
    out_data_row = Out_data_ptr + pid_row * row_stride_out_data
    out_scale_row = Out_scale_ptr + pid_row * row_stride_out_scale
    out_zp_row = Out_zp_ptr + pid_row * row_stride_out_zp

    col_start = pid_grp * GS
    col_off = col_start + tl.arange(0, BLOCK_GS)
    col_mask = tl.arange(0, BLOCK_GS) < GS
    x = tl.load(x_row + col_off, mask=col_mask, other=0.0).to(tl.float32)

    # Full-codebook INT2 asym:
    #   stored code ∈ [0, 3], signed q = code - 2 ∈ [-2, 1]
    #   scale=(max-min)/3, zp=min+2*scale so code 0 maps to min and
    #   code 3 maps to max.  This keeps the existing offset-encoded layout
    #   while using all four INT2 levels.
    x_for_max = tl.where(col_mask, x, -float("inf"))
    x_for_min = tl.where(col_mask, x,  float("inf"))
    xmax = tl.max(x_for_max, axis=0)
    xmin = tl.min(x_for_min, axis=0)
    scale = tl.maximum((xmax - xmin) / (float(OFFSET) + float(QMAX)), 1e-8)
    zp = xmin + float(OFFSET) * scale

    scale_bf16 = scale.to(tl.bfloat16)
    zp_bf16 = zp.to(tl.bfloat16)
    tl.store(out_scale_row + pid_grp, scale_bf16)
    tl.store(out_zp_row + pid_grp, zp_bf16)
    # Quant math uses bf16-round-tripped scale/zp for pack/unpack parity.
    scale_f = scale_bf16.to(tl.float32)
    zp_f = zp_bf16.to(tl.float32)

    byte_start = col_start // 4
    n_bytes = GS // 4
    byte_off = tl.arange(0, BLOCK_GS // 4)
    byte_mask = byte_off < n_bytes
    b0 = tl.arange(0, BLOCK_GS // 4) * 4 + 0
    b1 = tl.arange(0, BLOCK_GS // 4) * 4 + 1
    b2 = tl.arange(0, BLOCK_GS // 4) * 4 + 2
    b3 = tl.arange(0, BLOCK_GS // 4) * 4 + 3
    x0 = tl.load(x_row + col_start + b0, mask=byte_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_row + col_start + b1, mask=byte_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + col_start + b2, mask=byte_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_row + col_start + b3, mask=byte_mask, other=0.0).to(tl.float32)

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
    tl.store(out_data_row + byte_start + byte_off, packed.to(tl.uint8),
             mask=byte_mask)


# ── Python entry points ──────────────────────────────────────────────────────

def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    return flat, leading


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_quantize_and_pack_int2(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT2 quantize + pack, fused into one Triton launch.

    Drop-in replacement for ``intN_pack.quantize_and_pack(x, group_size,
    nbits=2)`` — returns ``(data uint8 [..., D//4], scales bf16 [..., NG])``.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    D = x.shape[-1]
    assert D % group_size == 0
    ng = D // group_size
    d_quarter = D // 4
    assert D % 4 == 0

    flat, leading = _flatten_leading(x)
    flat = flat.contiguous()
    n_rows = flat.shape[0]

    out_data = torch.empty((n_rows, d_quarter), dtype=torch.uint8, device=x.device)
    out_scale = torch.empty((n_rows, ng), dtype=torch.bfloat16, device=x.device)

    BLOCK_GS = _next_pow2(group_size)
    grid = (n_rows, ng)
    _fused_quant_int2_sym_kernel[grid](
        flat, out_data, out_scale,
        flat.stride(0), out_data.stride(0), out_scale.stride(0),
        n_rows,
        D=D, GS=group_size, NG=ng, D_QUARTER=d_quarter,
        QMAX=1, OFFSET=2, BLOCK_GS=BLOCK_GS,
    )
    return (out_data.reshape(*leading, d_quarter),
            out_scale.reshape(*leading, ng))


def fused_quantize_and_pack_int2_asym(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric INT2 quantize + pack (single Triton launch).

    Returns ``(data uint8 [..., D//4], scales bf16 [..., NG], zps bf16 [..., NG])``.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    D = x.shape[-1]
    assert D % group_size == 0
    ng = D // group_size
    d_quarter = D // 4
    assert D % 4 == 0

    flat, leading = _flatten_leading(x)
    flat = flat.contiguous()
    n_rows = flat.shape[0]

    out_data = torch.empty((n_rows, d_quarter), dtype=torch.uint8, device=x.device)
    out_scale = torch.empty((n_rows, ng), dtype=torch.bfloat16, device=x.device)
    out_zp = torch.empty((n_rows, ng), dtype=torch.bfloat16, device=x.device)

    BLOCK_GS = _next_pow2(group_size)
    grid = (n_rows, ng)
    _fused_quant_int2_asym_kernel[grid](
        flat, out_data, out_scale, out_zp,
        flat.stride(0), out_data.stride(0), out_scale.stride(0), out_zp.stride(0),
        n_rows,
        D=D, GS=group_size, NG=ng, D_QUARTER=d_quarter,
        QMAX=1, OFFSET=2, BLOCK_GS=BLOCK_GS,
    )
    return (out_data.reshape(*leading, d_quarter),
            out_scale.reshape(*leading, ng),
            out_zp.reshape(*leading, ng))
