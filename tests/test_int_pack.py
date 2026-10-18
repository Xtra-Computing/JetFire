"""Wire-format fixtures independent of the CUDA/Triton implementations."""
from __future__ import annotations

import pytest
import torch

from jetfire_vllm.kernels.intN_pack import quantize_and_pack, unpack


@pytest.mark.parametrize("group_size", (64, 128))
@pytest.mark.parametrize(
    "bits,values,planes",
    (
        (2, (-1, 0, 1, 0), (0x6E,)),
        (3, (-3, -2, -1, 0, 1, 2, 3, 0), (0x29, 0xCB, 0xBC)),
        (4, (-7, 7), (0x1F,)),
    ),
)
def test_known_wire_bytes_round_trip(bits, values, planes, group_size):
    x = torch.tensor(values, dtype=torch.bfloat16).repeat(128 // len(values)).view(1, 128)
    data, scales = quantize_and_pack(x, group_size=group_size, nbits=bits)
    assert torch.equal(scales, torch.ones_like(scales))
    # INT3 stores each byte position in its own contiguous plane.
    repeats = 128 // (8 if bits == 3 else 8 // bits)
    expected = torch.tensor([byte for byte in planes for _ in range(repeats)], dtype=torch.uint8)
    assert torch.equal(data[0], expected)
    assert torch.equal(unpack(data, scales, group_size=group_size, nbits=bits), x)


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_zero_groups_remain_finite(bits):
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    data, scales = quantize_and_pack(x, group_size=64, nbits=bits)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert torch.equal(unpack(data, scales, group_size=64, nbits=bits), x)
