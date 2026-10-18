"""Per-quant-scheme kernel dispatch for JetFire.

A :class:`KernelSet` is the bag of GPU kernels and metadata that decode_path
needs to run a given quantization scheme.  One :class:`KernelSet` instance
exists per scheme; :func:`get_kernel_set` resolves it via the
``QuantScheme.kernel_module`` pointer in :mod:`jetfire_vllm.adapter.config`.

To add a new scheme (e.g. INT3, INT4):

    1. Implement a module under ``jetfire_vllm.kernels._dispatch.<name>``
       exposing a ``KERNELS`` instance that satisfies the protocol below.
    2. Register the scheme in :data:`jetfire_vllm.adapter.config.QUANT_SCHEMES`
       with ``kernel_module="jetfire_vllm.kernels._dispatch.<name>:KERNELS"``.
    3. Done — no other code changes needed.
"""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class KernelSet(Protocol):
    """GPU kernel set for one quantization scheme."""

    nbits_k: int
    nbits_v: int
    asymmetric_k: bool
    asymmetric_v: bool
    head_dim: int
    group_size: int
    supported_top_k: tuple[int, ...]

    def load(self) -> None:
        """Build/load all CUDA extensions this scheme depends on.

        Must be idempotent.  Raises on build failure.
        """
        ...

    # Decode-time kernels ------------------------------------------------

    def attn_with_scores(
        self,
        q: torch.Tensor,
        k_data: torch.Tensor, k_scales: torch.Tensor, k_zp: torch.Tensor | None,
        v_data: torch.Tensor, v_scales: torch.Tensor,
        block_table: torch.Tensor, seq_lens: torch.Tensor,
        max_S: int, H_q: int, H_kv: int, block_size: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(o_hat, scores)`` for the full (paged) INT attention pass."""
        ...

    def fused_pre_correction_batched(
        self,
        scores: torch.Tensor, topk_idx: torch.Tensor,
        block_table: torch.Tensor, seq_lens: torch.Tensor,
        v_data: torch.Tensor, v_scales: torch.Tensor,
        block_size: int, H_kv: int, num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(o_old, W_2d)`` — softmax + topk_anorm + V_int dequant + bmm."""
        ...

    def fused_post_correction(
        self,
        q: torch.Tensor,
        k_topk: torch.Tensor, v_topk: torch.Tensor,
        o_hat: torch.Tensor, o_old: torch.Tensor, W: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """``o_corrected`` — Q·K_topk + softmax + ·V_topk + blend."""
        ...

    # Store-time kernels -------------------------------------------------

    def quant_write_k(
        self,
        key_tok: torch.Tensor, slot_idx: torch.Tensor,
        k_data: torch.Tensor, k_scales: torch.Tensor,
        k_zp: torch.Tensor | None, group_size: int,
    ) -> None:
        """In-place quant + scatter into the K cache."""
        ...

    def quant_write_v(
        self,
        value_tok: torch.Tensor, slot_idx: torch.Tensor,
        v_data: torch.Tensor, v_scales: torch.Tensor,
        v_zp: torch.Tensor | None, group_size: int,
    ) -> None:
        """In-place quant + scatter into the V cache."""
        ...


def get_kernel_set(scheme_name: str) -> KernelSet:
    """Resolve the :class:`KernelSet` for the named quant scheme.

    Raises :class:`NotImplementedError` if the scheme has no kernel module
    registered (e.g. INT3/INT4 placeholders).
    """
    from jetfire_vllm.adapter.config import QUANT_SCHEMES

    scheme = QUANT_SCHEMES.get(scheme_name)
    if scheme is None:
        raise ValueError(f"Unknown quant scheme {scheme_name!r}")
    if scheme.kernel_module is None:
        raise NotImplementedError(
            f"Quant scheme {scheme_name!r} has no fast-path kernel "
            f"implementation yet"
        )

    module_path, _, attr = scheme.kernel_module.partition(":")
    if not attr:
        raise ValueError(
            f"Bad kernel_module spec for {scheme_name!r}: "
            f"{scheme.kernel_module!r} (expected 'module:attr')"
        )
    mod = importlib.import_module(module_path)
    ks = getattr(mod, attr)
    if not isinstance(ks, KernelSet):  # type: ignore[misc]
        raise TypeError(
            f"{scheme.kernel_module} did not yield a KernelSet instance"
        )
    return ks
