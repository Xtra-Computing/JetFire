"""JetFire attention backend shim for vLLM."""
from __future__ import annotations

import os
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)

from .config import get_jetfire_config
from .decode_path import try_jetfire_decode, try_jetfire_prefill
from .kv_pool import get_global_kv_pool_manager
from .metadata_trace import record_builder_metadata
from .registry import _maybe_patch_worker_methods

_maybe_patch_worker_methods()

_CFG = get_jetfire_config()


def _should_defer_prefill_kv_mirror(layer_name: str, key) -> bool:
    mode = os.environ.get("JETFIRE_DEFER_PREFILL_KV_MIRROR", "0")
    if mode not in ("1", "force"):
        return False
    if key is None or not getattr(key, "is_cuda", False):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    min_tokens = int(os.environ.get("JETFIRE_DEFER_PREFILL_KV_MIRROR_MIN_TOKENS", "64"))
    if int(key.shape[0]) < min_tokens:
        return False
    if mode == "force":
        return True

    try:
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(layer_name)
        if attn_metadata is None:
            return False
        if int(getattr(attn_metadata, "max_query_len", 0)) <= 1:
            return False
        num_decodes = getattr(attn_metadata, "num_decodes", None)
        if num_decodes is not None and int(num_decodes) != 0:
            return False
        if bool(getattr(attn_metadata, "jetfire_prefill_has_context", False)):
            return False
    except Exception:
        return False
    return True


class JetFireAttentionMetadataBuilder(
    FlashAttentionMetadataBuilder,
):
    """FlashAttention metadata plus vLLM's decode/prefill split counts.

    vLLM already computes request order on CPU before attention metadata is
    built.  Carry those counts into JetFire so the mixed prefill+decode path
    can avoid per-layer GPU->CPU syncs and dynamic Python row discovery.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
        )
        metadata.num_decodes = int(num_decodes)
        metadata.num_prefills = int(num_prefills)
        metadata.num_decode_tokens = int(num_decode_tokens)
        metadata.num_prefill_tokens = int(num_prefill_tokens)
        query_lens_cpu = None
        seq_lens_cpu = None
        computed_tokens_cpu = None
        if num_prefills > 0:
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            if seq_lens_cpu is None:
                seq_lens_cpu = common_attn_metadata.seq_lens.to("cpu")
            computed_tokens_cpu = (
                seq_lens_cpu.to(dtype=query_lens_cpu.dtype) - query_lens_cpu
            )
            prefill_computed = computed_tokens_cpu[
                int(num_decodes) : int(num_decodes + num_prefills)
            ]
            metadata.jetfire_prefill_has_context = bool(
                (prefill_computed > 0).any().item()
            )
            metadata.jetfire_max_prefill_context_len = int(
                prefill_computed.max().item()
            )
        else:
            metadata.jetfire_prefill_has_context = False
            metadata.jetfire_max_prefill_context_len = 0
        record_builder_metadata(
            metadata=metadata,
            query_lens_cpu=query_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            computed_tokens_cpu=computed_tokens_cpu,
        )
        return metadata


class JetFireAttentionImpl(FlashAttentionImpl):
    """Drop-in impl with JetFire low-precision KV + CPU-correction decode."""

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if output is not None:
            jf_prefill = try_jetfire_prefill(
                impl=self, layer=layer, query=query, key=key, value=value,
                attn_metadata=attn_metadata, output=output,
            )
            if jf_prefill is not None:
                return jf_prefill

            jf_decode = try_jetfire_decode(
                impl=self, layer=layer, query=query, key=key, value=value,
                attn_metadata=attn_metadata, output=output,
            )
            if jf_decode is not None:
                return jf_decode

        # Fallback path.  When skip_native_kv=1 the native KV cache is
        # gone; calling super().forward() would silently produce garbage.
        # Permit the profiling forward (attn_metadata is None) since the
        # output is discarded.
        if (
            _CFG.skip_native_kv
            and attn_metadata is not None
            and self.attn_type.name == "DECODER"
        ):
            pool = get_global_kv_pool_manager()
            raise RuntimeError(
                "JetFire decode fallback hit under skip_native_kv=1, but "
                "the native KV cache has been freed.\n"
                f"Recent decode_fallbacks: {pool.decode_fallbacks}\n"
                "Common causes:\n"
                "  • seq_len_too_short: empty sequence or unsupported "
                "JETFIRE_TOPK. Use a positive supported correction width.\n"
                "  • cpu_correction_disabled: set "
                "JETFIRE_CPU_CORRECTION=1 (default) or "
                "JETFIRE_SKIP_NATIVE_KV=0.\n"
                "  • store_missing: layer not mirrored.  Check "
                "JETFIRE_MIRROR_LAYER_STRIDE (must be 1 when "
                "skip_native_kv=1)."
            )

        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output=output, output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        pool = get_global_kv_pool_manager()
        layer_name = getattr(layer, "layer_name", "unknown_layer")
        pool.ensure_layer_config_from_layer(layer_name, layer)
        if _should_defer_prefill_kv_mirror(layer_name, key):
            layer._jetfire_deferred_kv_mirror = (
                layer_name, key, value, slot_mapping
            )
        else:
            pool.mirror_kv(layer_name=layer_name, key=key, value=value,
                            slot_mapping=slot_mapping)
        if _CFG.skip_native_kv:
            return
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)


class JetFireAttentionBackend(FlashAttentionBackend):
    """Custom backend entrypoint reserved for JetFire integration."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "float16", "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"  # vLLM enum key for the custom backend slot

    @staticmethod
    def get_impl_cls() -> type[JetFireAttentionImpl]:
        return JetFireAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[JetFireAttentionMetadataBuilder]:
        return JetFireAttentionMetadataBuilder
