"""JetFire low-precision KV pool config and staging manager for vLLM.

Schemes (INT2/3/4, sym/asym) live in :mod:`jetfire_vllm.adapter.config`;
this module only stores the resolved bit-widths/asymmetry flags and routes
prefill mirror writes into per-layer :class:`JetFireInt4CacheStore` shards.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass

import torch

from .cache_store import JetFireInt4CacheStore
from .config import get_jetfire_config


@dataclass(frozen=True)
class JetFireKVPoolConfig:
    """Pool-side view of the JetFire config — bit-widths + sweep knobs."""

    enabled: bool
    group_size: int
    topk: int
    layer_stride: int
    k_nbits: int
    v_nbits: int
    k_asymmetric: bool
    v_asymmetric: bool


@dataclass(frozen=True)
class JetFireLayerConfig:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class JetFireLayerLayout:
    num_blocks: int
    block_size: int


class JetFireKVPoolManager:
    """Manager for JetFire INT4 sidecar cache and decode metrics."""

    def __init__(self, config: JetFireKVPoolConfig):
        self.config = config
        self.token_counters: dict[str, int] = {}
        self.shadow_int4: dict[str, object] = {}
        self.slot_tables: dict[str, dict[int, object]] = {}
        self.layer_stores: dict[str, JetFireInt4CacheStore] = {}
        self.layer_configs: dict[str, JetFireLayerConfig] = {}
        self.layer_layouts: dict[str, JetFireLayerLayout] = {}
        self.skipped_prefill = 0
        self.skipped_layers = 0
        self.written_slots = 0
        self.decode_attempts = 0
        self.decode_hits = 0
        self.decode_fallbacks: dict[str, int] = {}
        self.mixed_batch_count = 0
        self.mixed_batch_prefill_seqs = 0
        self.mixed_batch_decode_seqs = 0
        self.mixed_batch_prefill_tokens = 0
        self.mixed_batch_decode_tokens = 0
        self.prefill_tail_flushes = 0
        self._debug_enabled = get_jetfire_config().debug_log
        self._debug_seen: set[str] = set()

    def estimate_int4_bytes(self, num_tokens: int, num_kv_heads: int, head_dim: int) -> int:
        # K and V both quantized to int4 -> 1 byte stores two values.
        values = num_tokens * num_kv_heads * head_dim * 2
        return (values + 1) // 2

    def mirror_kv(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Quantize latest K/V chunk into a GPU-resident INT4 sidecar cache."""
        if not self.config.enabled:
            return
        if os.environ.get("JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE", "0") == "1":
            # vLLM captures a dummy forward while estimating CUDA graph memory.
            # That pass is not a real prefill and CUDA forbids pinned-memory
            # allocation during capture, so do not mirror KV there.
            self.skipped_prefill += 1
            return
        if os.environ.get("JETFIRE_DISABLE_KV_MIRROR", "0") == "1":
            self.skipped_prefill += 1
            return
        if key.numel() == 0 or value.numel() == 0:
            return
        if not self._should_mirror_layer(layer_name):
            self.skipped_layers += 1
            return

        config = self.layer_configs.get(layer_name)
        if config is None:
            # Layer config not yet installed.  This happens during vLLM's
            # startup profiling forward pass — install_jetfire_worker_hooks
            # hasn't run yet.  Skip silently; the first real forward will
            # have the config registered.
            self.skipped_layers += 1
            return
        if int(key.shape[1]) != config.num_kv_heads or int(key.shape[2]) != config.head_dim:
            raise RuntimeError(
                "JetFire layer config mismatch for "
                f"{layer_name}: expected ({config.num_kv_heads}, {config.head_dim}), "
                f"got ({int(key.shape[1])}, {int(key.shape[2])})"
            )

        if self.layer_layouts.get(layer_name) is None:
            # Same reasoning as the layer-config skip above — layouts come
            # from `_jetfire_install_hooks`, which runs post-LLM-init.
            self.skipped_layers += 1
            return

        store = self._get_or_create_store(
            layer_name=layer_name,
            device=key.device,
        )
        if key.is_cuda and torch.cuda.is_current_stream_capturing():
            written = store.write_slots_graph(
                key=key,
                value=value,
                slot_mapping=slot_mapping,
                max_batch_size=get_jetfire_config().max_batch_size,
            )
        else:
            written = store.write_slots(
                key=key,
                value=value,
                slot_mapping=slot_mapping,
            )
        if written <= 0:
            return

        self.token_counters[layer_name] = self.token_counters.get(layer_name, 0) + written
        self.written_slots += written
        self.shadow_int4[layer_name] = {"slots": written}
        self._debug_once(
            f"mirror:{layer_name}",
            f"JETFIRE_DEBUG mirror layer={layer_name} written_slots={written}",
        )
        self._maybe_flush_graph_prefill_tail(
            layer_name=layer_name,
            key=key,
            written=written,
        )

    def _get_or_create_store(
        self,
        *,
        layer_name: str,
        device: torch.device,
    ) -> JetFireInt4CacheStore:
        config = self.layer_configs.get(layer_name)
        layout = self.layer_layouts.get(layer_name)
        if config is None:
            raise RuntimeError(f"JetFire layer config missing for {layer_name}")
        if layout is None:
            raise RuntimeError(f"JetFire layer layout missing for {layer_name}")
        store = self.layer_stores.get(layer_name)
        if (
            store is None
            or store.num_blocks != layout.num_blocks
            or store.block_size != layout.block_size
            or store.num_kv_heads != config.num_kv_heads
            or store.head_dim != config.head_dim
        ):
            store = JetFireInt4CacheStore(
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                group_size=self.config.group_size,
                num_blocks=layout.num_blocks,
                block_size=layout.block_size,
                device=device,
                k_nbits=self.config.k_nbits,
                v_nbits=self.config.v_nbits,
                k_asymmetric=self.config.k_asymmetric,
                v_asymmetric=self.config.v_asymmetric,
                shadow_enabled=get_jetfire_config().shadow_enabled,
            )
            self.layer_stores[layer_name] = store
        return store

    def ensure_layer_store(
        self,
        layer_name: str,
        *,
        device: torch.device,
    ) -> JetFireInt4CacheStore:
        """Create the per-layer JetFire store before the first decode.

        Historically the store was created lazily by ``mirror_kv`` during
        prefill.  That is unsafe for CUDA graph decode: vLLM may capture a
        decode warmup before any real prefill has mirrored data, and a missing
        store silently falls back to native flash-attn when native KV is alive.
        Pre-creating the store fixes the pointer topology while prefill keeps
        writing the same store contents later.
        """
        return self._get_or_create_store(layer_name=layer_name, device=device)

    def ensure_all_layer_stores(self, *, device: torch.device) -> int:
        created = 0
        for layer_name in list(self.layer_layouts):
            if layer_name not in self.layer_configs:
                continue
            before = self.layer_stores.get(layer_name)
            self.ensure_layer_store(layer_name, device=device)
            if before is None:
                created += 1
        return created

    def prewarm_prefill_staging(
        self,
        *,
        num_tokens: int,
        device: torch.device,
    ) -> dict[str, object]:
        created = self.ensure_all_layer_stores(device=device)
        warmed = []
        estimated_bytes = 0
        for layer_name, store in self.layer_stores.items():
            if not self._should_mirror_layer(layer_name):
                continue
            info = store.prewarm_prefill_staging(num_tokens)
            if info.get("enabled"):
                estimated_bytes = max(
                    estimated_bytes,
                    int(info.get("estimated_pinned_bytes", 0)),
                )
            warmed.append({"layer": layer_name, **info})
        return {
            "created_stores": int(created),
            "layers": len(warmed),
            "num_tokens": int(num_tokens),
            "estimated_pinned_bytes_per_shape": int(estimated_bytes),
            "layer_results": warmed,
        }

    def register_layer_config(
        self,
        layer_name: str,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.layer_configs[layer_name] = JetFireLayerConfig(
            num_query_heads=int(num_query_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        )

    def register_layer_layout(
        self,
        layer_name: str,
        *,
        num_blocks: int,
        block_size: int,
    ) -> None:
        self.layer_layouts[layer_name] = JetFireLayerLayout(
            num_blocks=int(num_blocks),
            block_size=int(block_size),
        )

    def get_layer_store(self, layer_name: str) -> JetFireInt4CacheStore | None:
        return self.layer_stores.get(layer_name)

    def get_layer_config(self, layer_name: str) -> JetFireLayerConfig | None:
        return self.layer_configs.get(layer_name)

    def ensure_layer_config_from_layer(self, layer_name: str, layer: object) -> JetFireLayerConfig | None:
        config = self.layer_configs.get(layer_name)
        if config is not None:
            return config

        num_heads = getattr(layer, "num_heads", None)
        num_kv_heads = getattr(layer, "num_kv_heads", None)
        head_size = getattr(layer, "head_size", None)
        if num_heads is None and hasattr(layer, "impl"):
            impl = getattr(layer, "impl")
            num_heads = getattr(impl, "num_heads", None)
            num_kv_heads = getattr(impl, "num_kv_heads", None)
            head_size = getattr(impl, "head_size", None)
        if num_heads is None or num_kv_heads is None or head_size is None:
            return None

        self.register_layer_config(
            layer_name,
            num_query_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_size),
        )
        return self.layer_configs[layer_name]

    def record_decode_attempt(self) -> None:
        self.decode_attempts += 1

    def record_decode_hit(self) -> None:
        self.decode_hits += 1
        self._debug_once(
            "decode_hit:first",
            "JETFIRE_DEBUG decode_hit",
        )

    def record_decode_fallback(self, reason: str) -> None:
        self.decode_fallbacks[reason] = self.decode_fallbacks.get(reason, 0) + 1
        self._debug_once(
            f"decode_fallback:{reason}",
            f"JETFIRE_DEBUG decode_fallback reason={reason}",
        )

    def record_mixed_batch(
        self,
        *,
        prefill_seqs: int,
        decode_seqs: int,
        prefill_tokens: int = 0,
        decode_tokens: int = 0,
    ) -> None:
        self.mixed_batch_count += 1
        self.mixed_batch_prefill_seqs += int(prefill_seqs)
        self.mixed_batch_decode_seqs += int(decode_seqs)
        self.mixed_batch_prefill_tokens += int(prefill_tokens)
        self.mixed_batch_decode_tokens += int(decode_tokens)

    def _last_mirrored_layer_name(self) -> str | None:
        candidates = [
            name for name in self.layer_configs
            if name in self.layer_layouts and self._should_mirror_layer(name)
        ]
        if not candidates:
            return None

        def _layer_sort_key(name: str) -> tuple[int, int]:
            match = re.search(r"\.(\d+)\.", name)
            if match is None:
                return (0, candidates.index(name))
            return (1, int(match.group(1)))

        return max(candidates, key=_layer_sort_key)

    def _maybe_flush_graph_prefill_tail(
        self,
        *,
        layer_name: str,
        key: torch.Tensor,
        written: int,
    ) -> None:
        """Drain async prefill writes before decode CUDA graph replay.

        Graph replay bypasses Python, so the normal decode-side
        ``flush_pending_writes()`` is only executed during graph capture, not
        at the real request's prefill→decode boundary.  If prefill uses the
        pipelined side-stream path, the captured decode graph can otherwise
        read incomplete INT/shadow KV.  Flush once at the tail mirrored layer:
        this preserves inter-layer prefill overlap and establishes a concrete
        boundary before vLLM schedules the first decode replay.
        """
        if os.environ.get("JETFIRE_DECODE_GRAPH", "0") != "1":
            return
        if os.environ.get("JETFIRE_GRAPH_FLUSH_PREFILL_TAIL", "1") != "1":
            return
        if key.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        max_decode_B = max(1, int(get_jetfire_config().max_batch_size))
        if int(written) <= max_decode_B:
            return
        if layer_name != self._last_mirrored_layer_name():
            return
        self.flush_pending_writes()
        self.prefill_tail_flushes += 1
        self._debug_once(
            "prefill_tail_flush:first",
            "JETFIRE_DEBUG graph prefill_tail_flush",
        )

    def _debug_once(self, key: str, message: str) -> None:
        if not self._debug_enabled or key in self._debug_seen:
            return
        self._debug_seen.add(key)
        print(message, flush=True)

    def flush_pending_writes(self) -> None:
        """Block until every layer's pipelined prefill work (INT HBM write +
        bf16 shadow D2H + CPU shadow index_copy_) has landed.  Called by
        decode before it reads the cache.  Cheap when nothing is pending.
        """
        for store in self.layer_stores.values():
            store.flush_pending_writes()

    def reset_request_state_for_graph_reuse(self) -> None:
        """Clear per-request state without invalidating captured graph pointers.

        Decode CUDA graphs capture JetFire submit contexts and their backing
        layer-store pointers.  Independent accuracy samples still need request
        bookkeeping cleared, but dropping stores/graph contexts would leave the
        captured graph with stale handles.
        """
        self.token_counters.clear()
        self.shadow_int4.clear()
        self.slot_tables.clear()
        self.skipped_prefill = 0
        self.skipped_layers = 0
        self.written_slots = 0
        self.decode_attempts = 0
        self.decode_hits = 0
        self.decode_fallbacks.clear()
        self.mixed_batch_count = 0
        self.mixed_batch_prefill_seqs = 0
        self.mixed_batch_decode_seqs = 0
        self.mixed_batch_prefill_tokens = 0
        self.mixed_batch_decode_tokens = 0
        self.prefill_tail_flushes = 0
        for store in self.layer_stores.values():
            store.written_count = 0
            if hasattr(store, "slot_filled"):
                store.slot_filled.fill_(False)
        self._debug_seen.clear()

    def reset_runtime_state(self, *, preserve_layouts: bool = True) -> None:
        if os.environ.get("JETFIRE_DECODE_GRAPH", "0") == "1":
            try:
                from .decode_path import reset_jetfire_graph_contexts
                reset_jetfire_graph_contexts()
            except Exception:
                pass
        self.token_counters.clear()
        self.shadow_int4.clear()
        self.slot_tables.clear()
        self.layer_stores.clear()
        self.skipped_prefill = 0
        self.skipped_layers = 0
        self.written_slots = 0
        self.decode_attempts = 0
        self.decode_hits = 0
        self.decode_fallbacks.clear()
        self.mixed_batch_count = 0
        self.mixed_batch_prefill_seqs = 0
        self.mixed_batch_decode_seqs = 0
        self.mixed_batch_prefill_tokens = 0
        self.mixed_batch_decode_tokens = 0
        self.prefill_tail_flushes = 0
        if not preserve_layouts:
            self.layer_layouts.clear()
            self.layer_configs.clear()
        self._debug_seen.clear()

    def get_stats(self) -> dict[str, object]:
        return {
            "token_counters": dict(self.token_counters),
            "layer_configs": {k: vars(v) for k, v in self.layer_configs.items()},
            "layer_layouts": {k: vars(v) for k, v in self.layer_layouts.items()},
            "written_slots": self.written_slots,
            "decode_attempts": self.decode_attempts,
            "decode_hits": self.decode_hits,
            "decode_fallbacks": dict(self.decode_fallbacks),
            "mixed_batch_count": self.mixed_batch_count,
            "mixed_batch_prefill_seqs": self.mixed_batch_prefill_seqs,
            "mixed_batch_decode_seqs": self.mixed_batch_decode_seqs,
            "mixed_batch_prefill_tokens": self.mixed_batch_prefill_tokens,
            "mixed_batch_decode_tokens": self.mixed_batch_decode_tokens,
            "prefill_tail_flushes": self.prefill_tail_flushes,
            "shadow_layers": len(self.shadow_int4),
            "skipped_prefill": self.skipped_prefill,
            "skipped_layers": self.skipped_layers,
        }

    def _should_mirror_layer(self, layer_name: str) -> bool:
        stride = max(1, int(self.config.layer_stride))
        if stride <= 1:
            return True
        m = re.search(r"\.(\d+)\.", layer_name)
        if m is None:
            return True
        layer_idx = int(m.group(1))
        return (layer_idx % stride) == 0


_GLOBAL_MANAGER: JetFireKVPoolManager | None = None


def get_global_kv_pool_manager() -> JetFireKVPoolManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        runtime_cfg = get_jetfire_config()
        scheme = runtime_cfg.scheme
        pool_cfg = JetFireKVPoolConfig(
            enabled=runtime_cfg.use_custom_backend,
            group_size=runtime_cfg.group_size,
            topk=runtime_cfg.topk,
            layer_stride=runtime_cfg.layer_stride,
            k_nbits=scheme.k_nbits,
            v_nbits=scheme.v_nbits,
            k_asymmetric=scheme.k_asymmetric,
            v_asymmetric=scheme.v_asymmetric,
        )
        _GLOBAL_MANAGER = JetFireKVPoolManager(pool_cfg)
    return _GLOBAL_MANAGER
