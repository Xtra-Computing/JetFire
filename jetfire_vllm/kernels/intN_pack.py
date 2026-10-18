"""
Multi-bit-width INT quantization: pack / unpack / quantize-and-pack.

Supports nbits in {2, 3, 4} with per-group symmetric quantization.

Packing formats:
  INT4: 2 values per uint8  → D_PACKED = D // 2
        byte = ((q_even + 8) << 4) | (q_odd + 8)

  INT2: 4 values per uint8  → D_PACKED = D // 4
        byte = ((q0 + 2) << 6) | ((q1 + 2) << 4) | ((q2 + 2) << 2) | (q3 + 2)

  INT3: 8 values into 3 bytes, **planar** layout → D_PACKED = D * 3 // 8
        Every 8 consecutive values are packed into 24 bits (3 bytes) MSB-first.
        Stored in 3 contiguous planes of D//8 bytes each:
          data[..., 0:D//8]         = byte0 (bits[23:16] of each group)
          data[..., D//8:2*D//8]    = byte1 (bits[15:8])
          data[..., 2*D//8:3*D//8]  = byte2 (bits[7:0])
        This planar layout enables efficient contiguous tl.load in Triton.

Symmetric quantization ranges (unsigned encoding, subtract offset to get signed):
  INT2: qmax=1, offset=2, unsigned [1,3] → signed [-1,+1]
  INT3: qmax=3, offset=4, unsigned [1,7] → signed [-3,+3]
  INT4: qmax=7, offset=8, unsigned [1,15] → signed [-7,+7]

Asymmetric quantization uses the full unsigned code range:
  INT2: unsigned [0,3]  → signed [-2,+1]
  INT3: unsigned [0,7]  → signed [-4,+3]
  INT4: unsigned [0,15] → signed [-8,+7]
"""

import torch


def packed_dim(head_dim: int, nbits: int) -> int:
    """Return the packed dimension (bytes) for a given head_dim and bit-width."""
    if nbits == 2:
        return head_dim // 4
    elif nbits == 3:
        return head_dim * 3 // 8
    elif nbits == 4:
        return head_dim // 2
    else:
        raise ValueError(f"Unsupported nbits={nbits}, must be 2, 3, or 4")


def _qmax(nbits: int) -> int:
    return (1 << (nbits - 1)) - 1


def _offset(nbits: int) -> int:
    return 1 << (nbits - 1)


def _asym_qrange(nbits: int) -> tuple[int, int]:
    offset = _offset(nbits)
    return -offset, offset - 1


def quantize_and_pack(
    x: torch.Tensor,
    group_size: int = 128,
    nbits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize float tensor to packed INT{nbits} + per-group scales (SYMMETRIC).

    Args:
        x: (..., D) float tensor. D must be divisible by group_size and by
           the appropriate packing factor (2 for INT4, 4 for INT2, 8 for INT3).
        group_size: quantization group size (default 128).
        nbits: bit width (2, 3, or 4).

    Returns:
        data:   (..., D_PACKED) uint8 — packed values
        scales: (..., D // group_size) bfloat16 — per-group scale factors
    """
    D = x.shape[-1]
    leading = x.shape[:-1]
    ng = D // group_size
    qmax = _qmax(nbits)
    offset = _offset(nbits)

    assert D % group_size == 0, f"D={D} not divisible by gs={group_size}"

    # Reshape to groups: (..., ng, gs)
    xg = x.float().reshape(*leading, ng, group_size)

    # Per-group scales, cast to bf16 for storage, then recast to fp32 for
    # quantization math so pack/unpack use the same numerical scale.
    amax = xg.abs().amax(dim=-1)  # (..., ng)
    scales = (amax / qmax).clamp(min=1e-8)  # (..., ng) float32
    scales_bf16 = scales.to(torch.bfloat16)
    scales_f = scales_bf16.float().unsqueeze(-1)

    # Quantize: round and clamp
    xq = (xg / scales_f).round().clamp(-qmax, qmax)  # (..., ng, gs)
    xq = xq.reshape(*leading, D).to(torch.int8)  # (..., D) int8

    if nbits == 4:
        return _pack_int4(xq, offset), scales_bf16
    elif nbits == 2:
        return _pack_int2(xq, offset), scales_bf16
    elif nbits == 3:
        return _pack_int3_planar(xq, offset), scales_bf16
    else:
        raise ValueError(f"Unsupported nbits={nbits}")


def quantize_and_pack_asym(
    x: torch.Tensor,
    group_size: int = 128,
    nbits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize float tensor to packed INT{nbits} + per-group ASYMMETRIC (scale, zp).

    Signed quantization with zero-point over the full unsigned codebook:
        x ≈ q * scale + zp,   q ∈ [-offset, offset - 1]
        scale = (xmax - xmin) / (2^nbits - 1)
        zp    = xmin - qmin * scale

    Packing format is UNCHANGED from symmetric — q is still stored in
    signed offset-encoded form, so dequant pipelines that
    read `q` the same way still work.

    scale and zp are stored as bfloat16 (matches kernel load dtype).

    Returns:
        data:   (..., D_PACKED)       uint8
        scales: (..., D // group_size) bfloat16
        zps:    (..., D // group_size) bfloat16
    """
    D = x.shape[-1]
    leading = x.shape[:-1]
    ng = D // group_size
    offset = _offset(nbits)
    qmin, qmax = _asym_qrange(nbits)

    assert D % group_size == 0, f"D={D} not divisible by gs={group_size}"

    xg = x.float().reshape(*leading, ng, group_size)
    xmax = xg.amax(dim=-1)  # (..., ng)
    xmin = xg.amin(dim=-1)

    scale = ((xmax - xmin) / (qmax - qmin)).clamp(min=1e-8)  # (..., ng) f32
    zp = xmin - qmin * scale                                  # (..., ng) f32

    # Cast to bf16 storage dtype, then back to fp32 for quantization math.
    # This ensures the quantization uses the same numerical scale/zp that
    # kernels will load at dequant time — no asymmetry between pack and unpack.
    scale_bf16 = scale.to(torch.bfloat16)
    zp_bf16 = zp.to(torch.bfloat16)
    scale_f = scale_bf16.float().unsqueeze(-1)
    zp_f = zp_bf16.float().unsqueeze(-1)

    xq = ((xg - zp_f) / scale_f).round().clamp(qmin, qmax)  # (..., ng, gs)
    xq = xq.reshape(*leading, D).to(torch.int8)

    if nbits == 4:
        data = _pack_int4(xq, offset)
    elif nbits == 2:
        data = _pack_int2(xq, offset)
    elif nbits == 3:
        data = _pack_int3_planar(xq, offset)
    else:
        raise ValueError(f"Unsupported nbits={nbits}")

    return data, scale_bf16, zp_bf16


def _pack_int4(xq: torch.Tensor, offset: int = 8) -> torch.Tensor:
    """Pack INT4: 2 values per byte. (..., D) → (..., D//2) uint8."""
    even = xq[..., 0::2]  # (..., D//2)
    odd = xq[..., 1::2]
    packed = ((even.to(torch.int16) + offset) << 4) | (odd.to(torch.int16) + offset)
    return packed.to(torch.uint8)


def _pack_int2(xq: torch.Tensor, offset: int = 2) -> torch.Tensor:
    """Pack INT2: 4 values per byte. (..., D) → (..., D//4) uint8."""
    D = xq.shape[-1]
    assert D % 4 == 0
    v0 = xq[..., 0::4]  # (..., D//4)
    v1 = xq[..., 1::4]
    v2 = xq[..., 2::4]
    v3 = xq[..., 3::4]
    packed = (
        ((v0.to(torch.int16) + offset) << 6) |
        ((v1.to(torch.int16) + offset) << 4) |
        ((v2.to(torch.int16) + offset) << 2) |
        (v3.to(torch.int16) + offset)
    )
    return packed.to(torch.uint8)


def _pack_int3_planar(xq: torch.Tensor, offset: int = 4) -> torch.Tensor:
    """Pack INT3: 8 values → 3 bytes, planar layout. (..., D) → (..., D*3//8) uint8.

    Planar layout: [plane0: D//8 bytes | plane1: D//8 bytes | plane2: D//8 bytes]
    Each group of 8 values → 24 bits stored MSB-first across the 3 planes.
    """
    D = xq.shape[-1]
    assert D % 8 == 0
    leading = xq.shape[:-1]
    n_groups = D // 8  # number of 8-value groups

    # Reshape to (..., n_groups, 8) and shift to unsigned
    vals = (xq.reshape(*leading, n_groups, 8).to(torch.int32) + offset)  # [1, 7] range

    # Pack 8 × 3-bit values into 24-bit integer MSB-first
    combined = torch.zeros(*leading, n_groups, dtype=torch.int32, device=xq.device)
    for i in range(8):
        combined = combined | (vals[..., i] << (21 - 3 * i))

    # Split into 3 bytes
    byte0 = ((combined >> 16) & 0xFF).to(torch.uint8)  # (..., n_groups)
    byte1 = ((combined >> 8) & 0xFF).to(torch.uint8)
    byte2 = (combined & 0xFF).to(torch.uint8)

    # Planar layout: concatenate planes
    return torch.cat([byte0, byte1, byte2], dim=-1)  # (..., D*3//8)


# ── Unpack functions ──────────────────────────────────────────────────────

def unpack(
    data: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    nbits: int = 4,
    zps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack packed INT{nbits} tensor back to dequantized float.

    If zps is provided (asymmetric): x = q * scale + zp.
    Otherwise (symmetric): x = q * scale.
    """
    if nbits == 4:
        out = _unpack_int4(data, scales, group_size)
    elif nbits == 2:
        out = _unpack_int2(data, scales, group_size)
    elif nbits == 3:
        out = _unpack_int3_planar(data, scales, group_size)
    else:
        raise ValueError(f"Unsupported nbits={nbits}")

    if zps is not None:
        D = out.shape[-1]
        ng = D // group_size
        zp_expand = zps.float().unsqueeze(-1).expand(*zps.shape, group_size).reshape(
            *out.shape[:-1], D
        )
        out = (out.float() + zp_expand).to(out.dtype)
    return out


def _unpack_int4(data: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    leading = data.shape[:-1]
    D_half = data.shape[-1]
    D = D_half * 2
    ng = D // group_size

    even = ((data.to(torch.int16) >> 4) & 0xF) - 8
    odd = (data.to(torch.int16) & 0xF) - 8
    xq = torch.stack([even, odd], dim=-1).reshape(*leading, D)

    xq_grouped = xq.float().reshape(*leading, ng, group_size)
    return (xq_grouped * scales.float().unsqueeze(-1)).reshape(*leading, D).to(scales.dtype)


def _unpack_int2(data: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    leading = data.shape[:-1]
    D_quarter = data.shape[-1]
    D = D_quarter * 4
    ng = D // group_size

    v0 = ((data.to(torch.int16) >> 6) & 0x3) - 2
    v1 = ((data.to(torch.int16) >> 4) & 0x3) - 2
    v2 = ((data.to(torch.int16) >> 2) & 0x3) - 2
    v3 = (data.to(torch.int16) & 0x3) - 2
    xq = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*leading, D)

    xq_grouped = xq.float().reshape(*leading, ng, group_size)
    return (xq_grouped * scales.float().unsqueeze(-1)).reshape(*leading, D).to(scales.dtype)


def _unpack_int3_planar(data: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    leading = data.shape[:-1]
    D_packed = data.shape[-1]
    n_groups = D_packed // 3  # D // 8
    D = n_groups * 8
    ng = D // group_size

    # Split planar layout
    byte0 = data[..., :n_groups].to(torch.int32)
    byte1 = data[..., n_groups:2 * n_groups].to(torch.int32)
    byte2 = data[..., 2 * n_groups:3 * n_groups].to(torch.int32)

    # Combine into 24-bit integer
    combined = (byte0 << 16) | (byte1 << 8) | byte2  # (..., n_groups)

    # Extract 8 values per group
    vals = []
    for i in range(8):
        v = ((combined >> (21 - 3 * i)) & 0x7) - 4  # signed
        vals.append(v)
    xq = torch.stack(vals, dim=-1).reshape(*leading, D)  # (..., D)

    xq_grouped = xq.float().reshape(*leading, ng, group_size)
    return (xq_grouped * scales.float().unsqueeze(-1)).reshape(*leading, D).to(scales.dtype)


# ── Reference quantization (for correctness checks) ──────────────────────

def quantize_ref(t: torch.Tensor, nbits: int, gs: int) -> torch.Tensor:
    """Reference dequantized quantization (simulates quantization error without packing)."""
    if nbits >= 16:
        return t.clone()
    shape = t.shape
    *leading, D = shape
    flat = t.reshape(-1, D)
    BH = flat.shape[0]
    ng = D // gs
    qmax = _qmax(nbits)
    r = flat.float().view(BH, ng, gs)
    mx = r.abs().amax(dim=2)
    s = (mx / qmax).clamp(min=1e-8)
    q = (r / s.unsqueeze(2)).round().clamp(-qmax, qmax)
    return (q * s.unsqueeze(2)).view(shape).to(t.dtype)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    D, gs = 128, 128
    shapes = [(8, 1024, D), (1, D)]
    passed = 0
    total = 0

    for nbits in [2, 3, 4]:
        for shape in shapes:
            total += 1
            x = torch.randn(*shape, dtype=torch.bfloat16)
            data, scales = quantize_and_pack(x, group_size=gs, nbits=nbits)

            # Check packed dimension
            expected_packed = packed_dim(D, nbits)
            assert data.shape[-1] == expected_packed, \
                f"INT{nbits} packed dim: got {data.shape[-1]}, expected {expected_packed}"

            # Roundtrip
            x_rt = unpack(data, scales, group_size=gs, nbits=nbits)

            # Reference
            x_ref = quantize_ref(x, nbits, gs)

            # Compare roundtrip vs reference
            err = (x_rt.float() - x_ref.float()).abs().max().item()
            # Tolerance: scale precision differs (float32 pack vs bfloat16 ref)
            # Relative to typical value magnitude
            rmag = x_ref.float().abs().mean().item()
            ok = err < 0.02 * max(rmag, 0.01)
            status = "PASS" if ok else "FAIL"
            print(f"  INT{nbits} shape={shape} D_PACKED={data.shape[-1]:>3} "
                  f"max_err={err:.6f}  {status}")
            if ok:
                passed += 1

    print(f"\n{passed}/{total} symmetric tests passed")

    # ── Asymmetric round-trip ─────────────────────────────────────────────
    print("\nAsymmetric (bf16 scale + bf16 zp):")
    asym_passed = 0
    asym_total = 0
    for nbits in [2, 3, 4]:
        for shape in shapes:
            asym_total += 1
            x = torch.randn(*shape, dtype=torch.bfloat16) * 2.0 + 0.5  # non-zero mean
            data, scale, zp = quantize_and_pack_asym(x, group_size=gs, nbits=nbits)
            assert scale.dtype == torch.bfloat16, f"scale dtype {scale.dtype}"
            assert zp.dtype == torch.bfloat16, f"zp dtype {zp.dtype}"
            x_rt = unpack(data, scale, group_size=gs, nbits=nbits, zps=zp)

            # Compare to symmetric-quant MSE for sanity
            x_sym_pack, x_sym_scale = quantize_and_pack(x, group_size=gs, nbits=nbits)
            x_sym = unpack(x_sym_pack, x_sym_scale, group_size=gs, nbits=nbits)

            asym_mse = (x_rt.float() - x.float()).pow(2).mean().item()
            sym_mse = (x_sym.float() - x.float()).pow(2).mean().item()
            improve = (1 - asym_mse / sym_mse) * 100

            # Asym should be <= sym (5% tolerance for small random inputs where
            # zp happens to round to 0 and both paths produce identical grids)
            ok = asym_mse <= sym_mse * 1.05
            status = "PASS" if ok else "FAIL"
            print(f"  INT{nbits} shape={shape}  sym_mse={sym_mse:.4e}  "
                  f"asym_mse={asym_mse:.4e}  improve={improve:+.1f}%  {status}")
            if ok:
                asym_passed += 1
    print(f"\n{asym_passed}/{asym_total} asymmetric tests passed")

    sys.exit(0 if (passed == total and asym_passed == asym_total) else 1)
