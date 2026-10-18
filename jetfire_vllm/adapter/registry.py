"""Runtime registration for JetFire vLLM backend hooks."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import replace

import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.worker.gpu_model_runner import bind_kv_cache
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from .config import get_jetfire_config
from .kv_pool import get_global_kv_pool_manager

JETFIRE_BACKEND_PATH = "jetfire_vllm.adapter.jetfire_backend.JetFireAttentionBackend"
_PATCHED_WORKER_METHODS = False
_PATCHED_PLANNER = False
_PATCHED_LLAMA_MLP_CHUNKING = False
_PATCHED_QWEN2_MLP_CHUNKING = False
_PATCHED_QWEN3_MOE_CHUNKING = False
_PATCHED_LLAMA_ATTN_O_PROJ_CHUNKING = False
_PATCHED_QWEN2_ATTN_O_PROJ_CHUNKING = False
_PATCHED_QWEN3_MOE_ATTN_O_PROJ_CHUNKING = False
_PATCHED_VOCAB_EMBED_CHUNKING = False
_PATCHED_VLLM_ATTENTION_SHAPE = False
_PATCHED_PREFILL_FIRST_SCHEDULER = False
_MLP_CHUNK_RUNTIME_LOGGED = False
_ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED = False
_PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = False
_EMBED_CHUNK_RUNTIME_LOGGED = False


def ensure_jetfire_registered() -> None:
    """Register JetFire as the CUSTOM attention backend.

    Early-returns when ``JETFIRE_USE_CUSTOM_BACKEND=0``: leaves vLLM's
    Worker methods unpatched and CUSTOM backend slot empty, so the
    process behaves as pure vLLM.  This is what makes it safe for
    baseline subprocesses to share ``PYTHONPATH`` with JetFire ones —
    sitecustomize.py blindly calls this on every interpreter startup.
    """
    cfg = get_jetfire_config()
    if not cfg.use_custom_backend:
        return
    if cfg.inflate_blocks and not _compressed_kv_spec_enabled():
        _maybe_patch_planner()
    _maybe_patch_vllm_attention_shape()
    _maybe_patch_vocab_embedding_chunking()
    _maybe_patch_llama_mlp_chunking()
    _maybe_patch_qwen2_mlp_chunking()
    _maybe_patch_qwen3_moe_chunking()
    _maybe_patch_llama_attention_o_proj_chunking()
    _maybe_patch_qwen2_attention_o_proj_chunking()
    _maybe_patch_qwen3_moe_attention_o_proj_chunking()
    _maybe_patch_prefill_first_scheduler()
    _maybe_patch_worker_methods()
    register_backend(AttentionBackendEnum.CUSTOM, JETFIRE_BACKEND_PATH)


def _maybe_patch_prefill_first_scheduler() -> None:
    """Prefer admissible full prefills without starving active requests."""
    global _PATCHED_PREFILL_FIRST_SCHEDULER
    if _PATCHED_PREFILL_FIRST_SCHEDULER:
        return
    if os.environ.get("JETFIRE_PREFILL_FIRST_SCHEDULING", "0") != "1":
        return

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    original_schedule = Scheduler.schedule

    def _jetfire_prefill_first_schedule(self):
        waiting = bool(getattr(self, "waiting", None)) or bool(
            getattr(self, "skipped_waiting", None)
        )
        running = getattr(self, "running", None)
        if not waiting or not running:
            return original_schedule(self)

        saved_running = list(running)
        saved_limit = self.max_num_running_reqs
        if len(saved_running) >= saved_limit:
            return original_schedule(self)

        # Check admission before deferring decode.  An unsuccessful native
        # scheduling attempt consumes notifications even when it schedules no
        # tokens, so retrying schedule() would lose worker cleanup information.
        queue = self._select_waiting_queue_for_scheduling()
        head = queue.peek_request() if queue else None
        if (
            head is None
            or head.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            or not self.kv_cache_manager.can_fit_full_sequence(head)
        ):
            return original_schedule(self)

        self.running = []
        self.max_num_running_reqs = saved_limit - len(saved_running)
        try:
            output = original_schedule(self)
            new_running = list(self.running)
        finally:
            self.running = saved_running + list(self.running)
            self.max_num_running_reqs = saved_limit
        if os.environ.get("JETFIRE_DEBUG_SCHEDULER", "0") == "1":
            print(
                "[jetfire] prefill-first scheduler step: "
                f"deferred_running={len(saved_running)} "
                f"scheduled_prefill_running={len(new_running)}",
                flush=True,
            )
        return output

    Scheduler.schedule = _jetfire_prefill_first_schedule
    Scheduler._jetfire_original_schedule = original_schedule
    _PATCHED_PREFILL_FIRST_SCHEDULER = True

    if os.environ.get("JETFIRE_DEBUG_SCHEDULER", "0") == "1":
        print("[jetfire] installed prefill-first scheduler patch", flush=True)


def _compressed_kv_spec_enabled() -> bool:
    raw = os.environ.get("JETFIRE_COMPRESSED_KV_SPEC")
    if raw is not None:
        return raw.strip() == "1"
    cfg = get_jetfire_config()
    return cfg.use_custom_backend and cfg.skip_native_kv


def _mlp_chunk_tokens() -> int:
    raw = os.environ.get("JETFIRE_MLP_CHUNK_TOKENS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            "JETFIRE_MLP_CHUNK_TOKENS must be an integer token count."
        ) from None


def _attn_o_proj_chunk_tokens() -> int:
    raw = os.environ.get("JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS must be an integer token count."
        ) from None


def _prefill_attn_chunk_tokens() -> int:
    raw = os.environ.get("JETFIRE_PREFILL_ATTN_CHUNK_TOKENS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS must be an integer token count."
        ) from None


def _prefill_qkv_chunk_tokens() -> int:
    raw = os.environ.get("JETFIRE_PREFILL_QKV_CHUNK_TOKENS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            "JETFIRE_PREFILL_QKV_CHUNK_TOKENS must be an integer token count."
        ) from None


def _embed_chunk_tokens() -> int:
    raw = os.environ.get("JETFIRE_EMBED_CHUNK_TOKENS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            "JETFIRE_EMBED_CHUNK_TOKENS must be an integer token count."
        ) from None


def _maybe_patch_vocab_embedding_chunking() -> None:
    """Chunk TP embedding all-reduce for very long prefills."""
    global _PATCHED_VOCAB_EMBED_CHUNKING
    if _PATCHED_VOCAB_EMBED_CHUNKING:
        return
    if _embed_chunk_tokens() <= 0:
        return

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers import vocab_parallel_embedding as emb_mod

    VocabParallelEmbedding = emb_mod.VocabParallelEmbedding
    original_forward_native = VocabParallelEmbedding.forward_native

    def _chunked_forward_native(self, input_):
        global _EMBED_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _embed_chunk_tokens()
        if (
            chunk_tokens <= 0
            or input_.dim() != 1
            or input_.shape[0] <= chunk_tokens
            or not input_.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward_native(self, input_)

        if (
            os.environ.get("JETFIRE_DEBUG_EMBED_CHUNK", "0") == "1"
            and not _EMBED_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking VocabParallelEmbedding forward: "
                f"tokens={int(input_.shape[0])} hidden={int(self.embedding_dim)} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _EMBED_CHUNK_RUNTIME_LOGGED = True

        output = torch.empty(
            (input_.shape[0], self.embedding_dim),
            device=input_.device,
            dtype=self.weight.dtype,
        )
        for start in range(0, input_.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, input_.shape[0])
            masked_input, input_mask = emb_mod.get_masked_input_and_mask(
                input_[start:end],
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            output_parallel = self.quant_method.embedding(self, masked_input.long())
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            if getattr(self, "tp_size", 1) > 1:
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
            output[start:end].copy_(output_parallel)
        if os.environ.get("JETFIRE_EMBED_EMPTY_CACHE", "1") == "1":
            torch.cuda.empty_cache()
        return output

    VocabParallelEmbedding.forward_native = _chunked_forward_native
    VocabParallelEmbedding.forward_cuda = _chunked_forward_native
    VocabParallelEmbedding._jetfire_original_forward_native = original_forward_native
    _PATCHED_VOCAB_EMBED_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_EMBED_CHUNK", "0") == "1":
        print(
            "[jetfire] installed VocabParallelEmbedding token chunking: "
            f"JETFIRE_EMBED_CHUNK_TOKENS={_embed_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_llama_mlp_chunking() -> None:
    """Chunk LLaMA MLP over tokens to lower long-prefill activation peak.

    This intentionally leaves attention unchanged.  It only splits the FFN
    sequence:
      gate_up_proj -> silu_and_mul -> down_proj
    across the token dimension, then writes each chunk back into the now-dead
    input buffer.  This avoids allocating another full hidden-state tensor for
    very long prefills.
    """
    global _PATCHED_LLAMA_MLP_CHUNKING
    if _PATCHED_LLAMA_MLP_CHUNKING:
        return
    if _mlp_chunk_tokens() <= 0:
        return

    from vllm.model_executor.models.llama import LlamaMLP

    original_forward = LlamaMLP.forward

    def _chunked_forward(self, x):
        global _MLP_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _mlp_chunk_tokens()
        if (
            chunk_tokens <= 0
            or x.dim() != 2
            or x.shape[0] <= chunk_tokens
            or not x.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward(self, x)

        if (
            os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1"
            and not _MLP_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking LlamaMLP forward: "
                f"tokens={int(x.shape[0])} hidden={int(x.shape[1])} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _MLP_CHUNK_RUNTIME_LOGGED = True

        out = x
        for start in range(0, x.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, x.shape[0])
            y, _ = self.gate_up_proj(x[start:end])
            y = self.act_fn(y)
            y, _ = self.down_proj(y)
            out[start:end].copy_(y)
        return out

    LlamaMLP.forward = _chunked_forward
    LlamaMLP._jetfire_original_forward = original_forward
    _PATCHED_LLAMA_MLP_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1":
        print(
            "[jetfire] installed LlamaMLP token chunking: "
            f"JETFIRE_MLP_CHUNK_TOKENS={_mlp_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_qwen2_mlp_chunking() -> None:
    """Chunk Qwen2 MLP over tokens to lower long-prefill activation peak."""
    global _PATCHED_QWEN2_MLP_CHUNKING
    if _PATCHED_QWEN2_MLP_CHUNKING:
        return
    if _mlp_chunk_tokens() <= 0:
        return

    try:
        from vllm.model_executor.models.qwen2 import Qwen2MLP
    except Exception:
        return

    original_forward = Qwen2MLP.forward

    def _chunked_forward(self, x):
        global _MLP_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _mlp_chunk_tokens()
        if (
            chunk_tokens <= 0
            or x.dim() != 2
            or x.shape[0] <= chunk_tokens
            or not x.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward(self, x)

        if (
            os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1"
            and not _MLP_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking Qwen2MLP forward: "
                f"tokens={int(x.shape[0])} hidden={int(x.shape[1])} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _MLP_CHUNK_RUNTIME_LOGGED = True

        out = x
        for start in range(0, x.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, x.shape[0])
            y, _ = self.gate_up_proj(x[start:end])
            y = self.act_fn(y)
            y, _ = self.down_proj(y)
            out[start:end].copy_(y)
        return out

    Qwen2MLP.forward = _chunked_forward
    Qwen2MLP._jetfire_original_forward = original_forward
    _PATCHED_QWEN2_MLP_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1":
        print(
            "[jetfire] installed Qwen2MLP token chunking: "
            f"JETFIRE_MLP_CHUNK_TOKENS={_mlp_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_qwen3_moe_chunking() -> None:
    """Chunk Qwen3-MoE MLP/MoE over tokens for one-shot long prefills."""
    global _PATCHED_QWEN3_MOE_CHUNKING
    if _PATCHED_QWEN3_MOE_CHUNKING:
        return
    if _mlp_chunk_tokens() <= 0:
        return

    try:
        from vllm.model_executor.models import qwen3_moe as qwen3_moe_mod
    except Exception:
        return

    Qwen3MoeMLP = getattr(qwen3_moe_mod, "Qwen3MoeMLP", None)
    Qwen3MoeSparseMoeBlock = getattr(
        qwen3_moe_mod, "Qwen3MoeSparseMoeBlock", None
    )
    sequence_parallel_chunk = getattr(
        qwen3_moe_mod, "sequence_parallel_chunk", None
    )
    tensor_model_parallel_all_gather = getattr(
        qwen3_moe_mod, "tensor_model_parallel_all_gather", None
    )
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        get_forward_context = None
        is_forward_context_available = None
    if Qwen3MoeMLP is None and Qwen3MoeSparseMoeBlock is None:
        return

    if Qwen3MoeMLP is not None:
        original_mlp_forward = Qwen3MoeMLP.forward

        def _chunked_mlp_forward(self, x):
            global _MLP_CHUNK_RUNTIME_LOGGED
            chunk_tokens = _mlp_chunk_tokens()
            if (
                chunk_tokens <= 0
                or x.dim() != 2
                or x.shape[0] <= chunk_tokens
                or not x.is_cuda
                or torch.compiler.is_compiling()
            ):
                return original_mlp_forward(self, x)

            if (
                os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1"
                and not _MLP_CHUNK_RUNTIME_LOGGED
            ):
                print(
                    "[jetfire] chunking Qwen3MoeMLP forward: "
                    f"tokens={int(x.shape[0])} hidden={int(x.shape[1])} "
                    f"chunk_tokens={chunk_tokens}",
                    flush=True,
                )
                _MLP_CHUNK_RUNTIME_LOGGED = True

            out = x
            for start in range(0, x.shape[0], chunk_tokens):
                end = min(start + chunk_tokens, x.shape[0])
                x_chunk = x[start:end]
                y, _ = self.gate_up_proj(x_chunk)
                y = self.act_fn(y)
                y, _ = self.down_proj(y)
                if self.expert_gate is not None:
                    y = torch.nn.functional.sigmoid(self.expert_gate(x_chunk)[0]) * y
                out[start:end].copy_(y)
            return out

        Qwen3MoeMLP.forward = _chunked_mlp_forward
        Qwen3MoeMLP._jetfire_original_forward = original_mlp_forward

    if Qwen3MoeSparseMoeBlock is not None:
        original_moe_forward = Qwen3MoeSparseMoeBlock.forward

        def _chunked_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            global _MLP_CHUNK_RUNTIME_LOGGED
            chunk_tokens = _mlp_chunk_tokens()
            if (
                chunk_tokens <= 0
                or hidden_states.dim() > 2
                or hidden_states.shape[0] <= chunk_tokens
                or not hidden_states.is_cuda
                or torch.compiler.is_compiling()
            ):
                return original_moe_forward(self, hidden_states)

            is_input_1d = hidden_states.dim() == 1
            if is_input_1d:
                return original_moe_forward(self, hidden_states)
            num_tokens, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            if getattr(self, "is_sequence_parallel", False):
                if sequence_parallel_chunk is None:
                    return original_moe_forward(self, hidden_states)
                hidden_states = sequence_parallel_chunk(hidden_states)

            if hidden_states.shape[0] <= chunk_tokens:
                return original_moe_forward(self, hidden_states)

            if (
                os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1"
                and not _MLP_CHUNK_RUNTIME_LOGGED
            ):
                print(
                    "[jetfire] chunking Qwen3MoeSparseMoeBlock forward: "
                    f"tokens={int(hidden_states.shape[0])} "
                    f"hidden={int(hidden_states.shape[1])} "
                    f"chunk_tokens={chunk_tokens}",
                    flush=True,
                )
                _MLP_CHUNK_RUNTIME_LOGGED = True

            out = hidden_states
            forward_context = None
            old_all_moe_layers = None
            old_moe_layer_index = None
            uses_static_moe_layers = False
            if (
                get_forward_context is not None
                and is_forward_context_available is not None
                and is_forward_context_available()
            ):
                forward_context = get_forward_context()
                old_all_moe_layers = getattr(
                    forward_context, "all_moe_layers", None
                )
                old_moe_layer_index = getattr(
                    forward_context, "moe_layer_index", None
                )
                uses_static_moe_layers = old_all_moe_layers is not None
                if uses_static_moe_layers:
                    forward_context.all_moe_layers = None

            try:
                for start in range(0, hidden_states.shape[0], chunk_tokens):
                    end = min(start + chunk_tokens, hidden_states.shape[0])
                    hidden_chunk = hidden_states[start:end]
                    router_logits, _ = self.gate(hidden_chunk)
                    shared_out, fused_out = self.experts(
                        hidden_states=hidden_chunk,
                        router_logits=router_logits,
                    )
                    final_hidden_states = (
                        shared_out + fused_out
                        if shared_out is not None
                        else fused_out
                    )
                    if (
                        not getattr(self, "is_sequence_parallel", False)
                        and getattr(self, "tp_size", 1) > 1
                    ):
                        final_hidden_states = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                final_hidden_states
                            )
                        )
                    if final_hidden_states.shape != out[start:end].shape:
                        return original_moe_forward(self, hidden_states)
                    out[start:end].copy_(final_hidden_states)
            finally:
                if uses_static_moe_layers and forward_context is not None:
                    forward_context.all_moe_layers = old_all_moe_layers
                    if old_moe_layer_index is not None:
                        forward_context.moe_layer_index = old_moe_layer_index + 1

            if getattr(self, "is_sequence_parallel", False):
                if tensor_model_parallel_all_gather is None:
                    return original_moe_forward(self, hidden_states)
                out = tensor_model_parallel_all_gather(out, 0)
                out = out[:num_tokens]
            return out

        Qwen3MoeSparseMoeBlock.forward = _chunked_moe_forward
        Qwen3MoeSparseMoeBlock._jetfire_original_forward = original_moe_forward

    _PATCHED_QWEN3_MOE_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_MLP_CHUNK", "0") == "1":
        print(
            "[jetfire] installed Qwen3-MoE token chunking: "
            f"JETFIRE_MLP_CHUNK_TOKENS={_mlp_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_llama_attention_o_proj_chunking() -> None:
    """Chunk LLaMA attention output/projection over tokens.

    Long one-shot prefill keeps ``attn_output`` for the full sequence and then
    ``o_proj`` allocates another full-size output tensor.  At 70B/TP=2 this is
    about 1.8 GiB for 118k tokens.  Chunking ``o_proj`` and copying each chunk
    back into the now-dead full-hidden ``hidden_states`` buffer removes the
    projection output peak.

    For the longest single-request prefills, ``JETFIRE_PREFILL_ATTN_CHUNK_TOKENS``
    goes one step earlier: it computes flash-attention itself in query chunks
    and immediately applies ``o_proj``.  This avoids allocating the full
    ``attn_output`` tensor.  It is deliberately restricted to JetFire's
    one-shot, no-cached-context prefill shape; decode, mixed batches, and
    CUDA-graph capture continue through vLLM's normal attention wrapper.
    """
    global _PATCHED_LLAMA_ATTN_O_PROJ_CHUNKING
    if _PATCHED_LLAMA_ATTN_O_PROJ_CHUNKING:
        return
    if _attn_o_proj_chunk_tokens() <= 0:
        return

    from vllm.model_executor.models.llama import LlamaAttention

    original_forward = LlamaAttention.forward

    def _chunked_forward(self, positions, hidden_states):
        global _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        global _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _attn_o_proj_chunk_tokens()
        attn_chunk_tokens = _prefill_attn_chunk_tokens()
        qkv_chunk_tokens = _prefill_qkv_chunk_tokens()
        if (
            max(chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens) <= 0
            or hidden_states.dim() != 2
            or hidden_states.shape[0] <= max(
                chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens
            )
            or not hidden_states.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward(self, positions, hidden_states)

        if qkv_chunk_tokens > 0:
            streamed_qkv = _try_stream_llama_prefill_qkv_attention(
                llama_attn=self,
                positions=positions,
                hidden_states=hidden_states,
                chunk_tokens=qkv_chunk_tokens,
            )
            if streamed_qkv is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking LlamaAttention prefill QKV+attention: "
                        f"tokens={int(hidden_states.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={qkv_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed_qkv

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if attn_chunk_tokens > 0:
            streamed = _try_stream_llama_prefill_attention(
                llama_attn=self,
                hidden_states=hidden_states,
                q=q,
                k=k,
                v=v,
                chunk_tokens=attn_chunk_tokens,
            )
            if streamed is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking LlamaAttention prefill attention: "
                        f"tokens={int(q.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={attn_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed

        if chunk_tokens <= 0 or hidden_states.shape[0] <= chunk_tokens:
            return original_forward(self, positions, hidden_states)

        attn_output = self.attn(q, k, v)
        if attn_output.dim() != 2 or attn_output.shape[0] <= chunk_tokens:
            output, _ = self.o_proj(attn_output)
            return output

        if (
            os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1"
            and not _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking LlamaAttention o_proj: "
                f"tokens={int(attn_output.shape[0])} "
                f"hidden={int(attn_output.shape[1])} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED = True

        out = hidden_states
        if out.shape[0] != attn_output.shape[0]:
            output, _ = self.o_proj(attn_output)
            return output

        for start in range(0, attn_output.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, attn_output.shape[0])
            y, _ = self.o_proj(attn_output[start:end])
            if out.shape[1] != y.shape[1]:
                output, _ = self.o_proj(attn_output)
                return output
            out[start:end].copy_(y)
        return out

    LlamaAttention.forward = _chunked_forward
    LlamaAttention._jetfire_original_forward = original_forward
    _PATCHED_LLAMA_ATTN_O_PROJ_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1":
        print(
            "[jetfire] installed LlamaAttention o_proj token chunking: "
            "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS="
            f"{_attn_o_proj_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_qwen2_attention_o_proj_chunking() -> None:
    """Chunk Qwen2 attention/output projection over tokens."""
    global _PATCHED_QWEN2_ATTN_O_PROJ_CHUNKING
    if _PATCHED_QWEN2_ATTN_O_PROJ_CHUNKING:
        return
    if _attn_o_proj_chunk_tokens() <= 0:
        return

    try:
        from vllm.model_executor.models.qwen2 import Qwen2Attention
    except Exception:
        return

    original_forward = Qwen2Attention.forward

    def _chunked_forward(self, positions, hidden_states):
        global _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        global _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _attn_o_proj_chunk_tokens()
        attn_chunk_tokens = _prefill_attn_chunk_tokens()
        qkv_chunk_tokens = _prefill_qkv_chunk_tokens()
        if (
            max(chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens) <= 0
            or hidden_states.dim() != 2
            or hidden_states.shape[0] <= max(
                chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens
            )
            or not hidden_states.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward(self, positions, hidden_states)

        if qkv_chunk_tokens > 0:
            streamed_qkv = _try_stream_llama_prefill_qkv_attention(
                llama_attn=self,
                positions=positions,
                hidden_states=hidden_states,
                chunk_tokens=qkv_chunk_tokens,
            )
            if streamed_qkv is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking Qwen2Attention prefill QKV+attention: "
                        f"tokens={int(hidden_states.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={qkv_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed_qkv

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if attn_chunk_tokens > 0:
            streamed = _try_stream_llama_prefill_attention(
                llama_attn=self,
                hidden_states=hidden_states,
                q=q,
                k=k,
                v=v,
                chunk_tokens=attn_chunk_tokens,
            )
            if streamed is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking Qwen2Attention prefill attention: "
                        f"tokens={int(q.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={attn_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed

        if chunk_tokens <= 0 or hidden_states.shape[0] <= chunk_tokens:
            return original_forward(self, positions, hidden_states)

        attn_output = self.attn(q, k, v)
        if attn_output.dim() != 2 or attn_output.shape[0] <= chunk_tokens:
            output, _ = self.o_proj(attn_output)
            return output

        if (
            os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1"
            and not _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking Qwen2Attention o_proj: "
                f"tokens={int(attn_output.shape[0])} "
                f"hidden={int(attn_output.shape[1])} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED = True

        out = hidden_states
        if out.shape[0] != attn_output.shape[0]:
            output, _ = self.o_proj(attn_output)
            return output

        for start in range(0, attn_output.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, attn_output.shape[0])
            y, _ = self.o_proj(attn_output[start:end])
            if out.shape[1] != y.shape[1]:
                output, _ = self.o_proj(attn_output)
                return output
            out[start:end].copy_(y)
        return out

    Qwen2Attention.forward = _chunked_forward
    Qwen2Attention._jetfire_original_forward = original_forward
    _PATCHED_QWEN2_ATTN_O_PROJ_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1":
        print(
            "[jetfire] installed Qwen2Attention o_proj token chunking: "
            "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS="
            f"{_attn_o_proj_chunk_tokens()}",
            flush=True,
        )


def _maybe_patch_qwen3_moe_attention_o_proj_chunking() -> None:
    """Chunk Qwen3-MoE attention/output projection over tokens."""
    global _PATCHED_QWEN3_MOE_ATTN_O_PROJ_CHUNKING
    if _PATCHED_QWEN3_MOE_ATTN_O_PROJ_CHUNKING:
        return
    if _attn_o_proj_chunk_tokens() <= 0:
        return

    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
    except Exception:
        return

    original_forward = Qwen3MoeAttention.forward

    def _chunked_forward(self, positions, hidden_states):
        global _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        global _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
        chunk_tokens = _attn_o_proj_chunk_tokens()
        attn_chunk_tokens = _prefill_attn_chunk_tokens()
        qkv_chunk_tokens = _prefill_qkv_chunk_tokens()
        if (
            max(chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens) <= 0
            or hidden_states.dim() != 2
            or hidden_states.shape[0] <= max(
                chunk_tokens, attn_chunk_tokens, qkv_chunk_tokens
            )
            or not hidden_states.is_cuda
            or torch.compiler.is_compiling()
        ):
            return original_forward(self, positions, hidden_states)

        if qkv_chunk_tokens > 0:
            streamed_qkv = _try_stream_llama_prefill_qkv_attention(
                llama_attn=self,
                positions=positions,
                hidden_states=hidden_states,
                chunk_tokens=qkv_chunk_tokens,
            )
            if streamed_qkv is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking Qwen3MoeAttention prefill "
                        "QKV+attention: "
                        f"tokens={int(hidden_states.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={qkv_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed_qkv

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)

        if attn_chunk_tokens > 0:
            streamed = _try_stream_llama_prefill_attention(
                llama_attn=self,
                hidden_states=hidden_states,
                q=q,
                k=k,
                v=v,
                chunk_tokens=attn_chunk_tokens,
            )
            if streamed is not None:
                if (
                    os.environ.get("JETFIRE_DEBUG_PREFILL_ATTN_CHUNK", "0") == "1"
                    and not _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED
                ):
                    print(
                        "[jetfire] chunking Qwen3MoeAttention prefill attention: "
                        f"tokens={int(q.shape[0])} "
                        f"hidden={int(hidden_states.shape[1])} "
                        f"chunk_tokens={attn_chunk_tokens}",
                        flush=True,
                    )
                    _PREFILL_ATTN_CHUNK_RUNTIME_LOGGED = True
                return streamed

        if chunk_tokens <= 0 or hidden_states.shape[0] <= chunk_tokens:
            return original_forward(self, positions, hidden_states)

        attn_output = self.attn(q, k, v)
        if attn_output.dim() != 2 or attn_output.shape[0] <= chunk_tokens:
            output, _ = self.o_proj(attn_output)
            return output

        if (
            os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1"
            and not _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED
        ):
            print(
                "[jetfire] chunking Qwen3MoeAttention o_proj: "
                f"tokens={int(attn_output.shape[0])} "
                f"hidden={int(attn_output.shape[1])} "
                f"chunk_tokens={chunk_tokens}",
                flush=True,
            )
            _ATTN_O_PROJ_CHUNK_RUNTIME_LOGGED = True

        out = hidden_states
        if out.shape[0] != attn_output.shape[0]:
            output, _ = self.o_proj(attn_output)
            return output

        for start in range(0, attn_output.shape[0], chunk_tokens):
            end = min(start + chunk_tokens, attn_output.shape[0])
            y, _ = self.o_proj(attn_output[start:end])
            if out.shape[1] != y.shape[1]:
                output, _ = self.o_proj(attn_output)
                return output
            out[start:end].copy_(y)
        return out

    Qwen3MoeAttention.forward = _chunked_forward
    Qwen3MoeAttention._jetfire_original_forward = original_forward
    _PATCHED_QWEN3_MOE_ATTN_O_PROJ_CHUNKING = True

    if os.environ.get("JETFIRE_DEBUG_ATTN_O_PROJ_CHUNK", "0") == "1":
        print(
            "[jetfire] installed Qwen3MoeAttention o_proj token chunking: "
            "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS="
            f"{_attn_o_proj_chunk_tokens()}",
            flush=True,
        )


def _try_stream_llama_prefill_attention(
    *,
    llama_attn,
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor | None:
    """Run exact one-shot prefill attention in chunks, then chunk ``o_proj``.

    This is a memory optimization for JetFire long context only.  It does not
    create scheduler-level chunked prefill: every layer still receives the full
    sequence QKV, and each chunk attends to the high-precision K/V already
    materialized for this layer.
    """
    cfg = get_jetfire_config()
    if not (cfg.use_custom_backend and cfg.skip_native_kv):
        return None
    if chunk_tokens <= 0 or q.dim() != 2 or q.shape[0] <= chunk_tokens:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    try:
        from vllm.model_executor.layers.attention import attention as attention_mod
        from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
    except Exception:
        return None

    layer_name = getattr(llama_attn.attn, "layer_name", None)
    if not layer_name:
        return None
    try:
        attn_metadata, attn_layer, kv_cache, layer_slot_mapping = (
            attention_mod.get_attention_context(layer_name)
        )
    except Exception:
        return None
    if attn_metadata is None or getattr(attn_layer, "impl", None) is None:
        return None

    impl = attn_layer.impl
    if getattr(impl, "attn_type", None) is None or impl.attn_type.name != "DECODER":
        return None
    if getattr(impl, "dcp_world_size", 1) > 1:
        return None
    if bool(getattr(attn_metadata, "use_cascade", False)):
        return None
    if int(getattr(attn_metadata, "num_actual_tokens", 0)) != int(q.shape[0]):
        return None
    if int(getattr(attn_metadata, "max_query_len", 0)) <= 1:
        return None

    num_decodes = getattr(attn_metadata, "num_decodes", None)
    num_prefills = getattr(attn_metadata, "num_prefills", None)
    if num_decodes is None or num_prefills is None:
        return None
    if int(num_decodes) != 0 or int(num_prefills) != 1:
        return None
    if bool(getattr(attn_metadata, "jetfire_prefill_has_context", False)):
        return None
    if int(attn_metadata.query_start_loc.shape[0]) != 2:
        return None

    n_actual = int(attn_metadata.num_actual_tokens)
    head_size_v = int(getattr(impl, "head_size_v", impl.head_size))
    q3 = q[:n_actual].view(-1, impl.num_heads, impl.head_size)
    k3 = k[:n_actual].view(-1, impl.num_kv_heads, impl.head_size)
    v3 = v[:n_actual].view(-1, impl.num_kv_heads, head_size_v)

    if layer_slot_mapping is not None:
        impl.do_kv_cache_update(attn_layer, k3, v3, kv_cache, layer_slot_mapping)

    out = hidden_states
    if out.shape[0] != n_actual:
        return None

    sliding = list(impl.sliding_window) if impl.sliding_window is not None else None
    descale_shape = (1, impl.num_kv_heads)
    cu_q = attn_metadata.query_start_loc.new_empty((2,))
    cu_k = attn_metadata.query_start_loc.new_empty((2,))
    cu_q[0] = 0
    cu_k[0] = 0

    for start in range(0, n_actual, chunk_tokens):
        end = min(start + chunk_tokens, n_actual)
        q_len = end - start
        cu_q[1] = q_len
        cu_k[1] = end
        attn_chunk = torch.empty_like(q3[start:end])
        flash_attn_varlen_func(
            q=q3[start:end],
            k=k3[:end],
            v=v3[:end],
            out=attn_chunk,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=end,
            softmax_scale=impl.scale,
            causal=attn_metadata.causal,
            alibi_slopes=impl.alibi_slopes,
            window_size=sliding,
            softcap=impl.logits_soft_cap,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=attn_layer._q_scale.expand(descale_shape),
            k_descale=attn_layer._k_scale.expand(descale_shape),
            v_descale=attn_layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=impl.sinks,
        )
        projected, _ = llama_attn.o_proj(attn_chunk.reshape(q_len, -1))
        if projected.shape[1] != out.shape[1]:
            return None
        out[start:end].copy_(projected)
    return out


def _jetfire_one_shot_prefill_context(llama_attn, n_tokens: int):
    cfg = get_jetfire_config()
    if not (cfg.use_custom_backend and cfg.skip_native_kv):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    try:
        from vllm.model_executor.layers.attention import attention as attention_mod
    except Exception:
        return None

    layer_name = getattr(llama_attn.attn, "layer_name", None)
    if not layer_name:
        return None
    try:
        attn_metadata, attn_layer, kv_cache, layer_slot_mapping = (
            attention_mod.get_attention_context(layer_name)
        )
    except Exception:
        return None
    if attn_metadata is None or getattr(attn_layer, "impl", None) is None:
        return None

    impl = attn_layer.impl
    if getattr(impl, "attn_type", None) is None or impl.attn_type.name != "DECODER":
        return None
    if getattr(impl, "dcp_world_size", 1) > 1:
        return None
    if bool(getattr(attn_metadata, "use_cascade", False)):
        return None
    if int(getattr(attn_metadata, "num_actual_tokens", 0)) != int(n_tokens):
        return None
    if int(getattr(attn_metadata, "max_query_len", 0)) <= 1:
        return None

    num_decodes = getattr(attn_metadata, "num_decodes", None)
    num_prefills = getattr(attn_metadata, "num_prefills", None)
    if num_decodes is None or num_prefills is None:
        return None
    if int(num_decodes) != 0 or int(num_prefills) != 1:
        return None
    if bool(getattr(attn_metadata, "jetfire_prefill_has_context", False)):
        return None
    if int(attn_metadata.query_start_loc.shape[0]) != 2:
        return None
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping, impl


def _try_stream_llama_prefill_qkv_attention(
    *,
    llama_attn,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor | None:
    """Stream QKV projection, exact causal prefill attention, and ``o_proj``."""
    if chunk_tokens <= 0 or hidden_states.dim() != 2:
        return None
    n_actual = int(hidden_states.shape[0])
    if n_actual <= chunk_tokens:
        return None
    context = _jetfire_one_shot_prefill_context(llama_attn, n_actual)
    if context is None:
        return None

    try:
        from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
    except Exception:
        return None

    attn_metadata, attn_layer, kv_cache, layer_slot_mapping, impl = context
    head_size_v = int(getattr(impl, "head_size_v", impl.head_size))
    device = hidden_states.device
    dtype = hidden_states.dtype

    k_all = torch.empty((n_actual, llama_attn.kv_size), device=device, dtype=dtype)
    v_all = torch.empty((n_actual, llama_attn.kv_size), device=device, dtype=dtype)
    sliding = list(impl.sliding_window) if impl.sliding_window is not None else None
    descale_shape = (1, impl.num_kv_heads)
    cu_q = attn_metadata.query_start_loc.new_empty((2,))
    cu_k = attn_metadata.query_start_loc.new_empty((2,))
    cu_q[0] = 0
    cu_k[0] = 0

    out = hidden_states
    for start in range(0, n_actual, chunk_tokens):
        end = min(start + chunk_tokens, n_actual)
        q_len = end - start
        qkv, _ = llama_attn.qkv_proj(hidden_states[start:end])
        q, k, v = qkv.split(
            [llama_attn.q_size, llama_attn.kv_size, llama_attn.kv_size],
            dim=-1,
        )
        if hasattr(llama_attn, "q_norm"):
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // llama_attn.head_dim, llama_attn.head_dim
            )
            q = llama_attn.q_norm(q_by_head).view(q.shape)
        if hasattr(llama_attn, "k_norm"):
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // llama_attn.head_dim, llama_attn.head_dim
            )
            k = llama_attn.k_norm(k_by_head).view(k.shape)
        q, k = llama_attn.rotary_emb(positions[start:end], q, k)
        k_all[start:end].copy_(k)
        v_all[start:end].copy_(v)

        q3 = q.view(-1, impl.num_heads, impl.head_size)
        k3 = k.view(-1, impl.num_kv_heads, impl.head_size)
        v3 = v.view(-1, impl.num_kv_heads, head_size_v)
        if layer_slot_mapping is not None:
            impl.do_kv_cache_update(
                attn_layer,
                k3,
                v3,
                kv_cache,
                layer_slot_mapping[start:end],
            )

        cu_q[1] = q_len
        cu_k[1] = end
        attn_chunk = torch.empty_like(q3)
        flash_attn_varlen_func(
            q=q3,
            k=k_all[:end].view(-1, impl.num_kv_heads, impl.head_size),
            v=v_all[:end].view(-1, impl.num_kv_heads, head_size_v),
            out=attn_chunk,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=end,
            softmax_scale=impl.scale,
            causal=attn_metadata.causal,
            alibi_slopes=impl.alibi_slopes,
            window_size=sliding,
            softcap=impl.logits_soft_cap,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=attn_layer._q_scale.expand(descale_shape),
            k_descale=attn_layer._k_scale.expand(descale_shape),
            v_descale=attn_layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=impl.sinks,
        )
        projected, _ = llama_attn.o_proj(attn_chunk.reshape(q_len, -1))
        if projected.shape[1] != out.shape[1]:
            return None
        out[start:end].copy_(projected)
    return out


def _maybe_patch_vllm_attention_shape() -> None:
    """Avoid a PyTorch AOT/SymInt failure in vLLM's attention wrapper.

    On vLLM 0.19 + torch nightly, 70B TP=2 with inductor can fail while
    compiling Attention.forward at ``num_tokens = query.shape[0]`` followed by
    ``torch.Size((num_tokens, hidden))``.  For the LLaMA path used here, the
    attention output has the same 2D shape as the local query projection, so
    ``empty_like(query)`` is equivalent and keeps the shape expression out of
    AOT guard propagation.
    """
    global _PATCHED_VLLM_ATTENTION_SHAPE
    if _PATCHED_VLLM_ATTENTION_SHAPE:
        return
    if os.environ.get("JETFIRE_PATCH_VLLM_ATTENTION_SHAPE", "1") != "1":
        return

    from vllm.model_executor.layers.attention import attention as attention_mod

    Attention = attention_mod.Attention

    def _jetfire_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
        output_dtype = query.dtype
        if self.query_quant is not None:
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            if output_shape is None:
                output = torch.empty_like(query)
                hidden_size = self.num_heads * self.head_size_v
            else:
                output = torch.empty(
                    output_shape, dtype=output_dtype, device=query.device
                )
                hidden_size = output_shape[-1]
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size_v)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size_v)
            kv_cache_dummy_dep = None
            if self.use_direct_call:
                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = attention_mod.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                attention_mod.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            else:
                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                torch.ops.vllm.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            return output.view(-1, hidden_size)

        assert self.attn_backend.forward_includes_kv_cache_update, (
            "Split KV cache update not supported when output tensor not provided."
        )
        if self.use_direct_call:
            return attention_mod.unified_attention(query, key, value, self.layer_name)
        return torch.ops.vllm.unified_attention(query, key, value, self.layer_name)

    Attention.forward = _jetfire_attention_forward
    Attention._jetfire_shape_patch = True
    _PATCHED_VLLM_ATTENTION_SHAPE = True


def _maybe_warmup_decode_buffers() -> None:
    """Trigger decode-buffer prealloc when ``JETFIRE_MAX_BATCH_SIZE`` is set.

    Pulls H_q/H_kv/D from any registered layer (uniform across LLaMA-style
    models) and ``top_k`` from the resolved config.  Idempotent — safe to
    call from multiple worker hooks.  No-op when prealloc is disabled or no
    layer config has been registered yet.
    """
    cfg = get_jetfire_config()
    if cfg.max_batch_size <= 0:
        return
    pool = get_global_kv_pool_manager()
    if not pool.layer_configs:
        return
    layer_cfg = next(iter(pool.layer_configs.values()))
    from .decode_path import preload_decode_extensions, warmup_decode_buffers
    if os.environ.get("JETFIRE_PRELOAD_DECODE_EXTENSIONS", "1") == "1":
        preload_decode_extensions()
    warmup_decode_buffers(
        max_batch_size=cfg.max_batch_size,
        H_q=layer_cfg.num_query_heads,
        H_kv=layer_cfg.num_kv_heads,
        D=layer_cfg.head_dim,
        top_k=cfg.topk,
    )


def _decode_graph_enabled() -> bool:
    if os.environ.get("JETFIRE_USE_PERSISTENT_WORKER", "0") == "1":
        return False
    return os.environ.get("JETFIRE_DECODE_GRAPH", "0") == "1"


def _maybe_prepare_decode_graph(device: torch.device | str) -> dict[str, object] | None:
    """Allocate decode graph state before vLLM captures decode."""
    if not _decode_graph_enabled():
        return None

    cfg = get_jetfire_config()
    if cfg.max_batch_size <= 0:
        raise RuntimeError(
            "JETFIRE_DECODE_GRAPH=1 requires JETFIRE_MAX_BATCH_SIZE>0. "
            "Set it to the largest decode batch size vLLM may capture."
        )

    pool = get_global_kv_pool_manager()
    created = pool.ensure_all_layer_stores(device=torch.device(device))

    from .decode_path import (
        reset_jetfire_graph_stats,
        warmup_graph_decode_submit_contexts,
    )

    graph_info = warmup_graph_decode_submit_contexts(
        max_batch_size=cfg.max_batch_size,
        device=device,
    )
    reset_jetfire_graph_stats()
    graph_info["stores_created"] = int(created)
    return graph_info


def _maybe_patch_planner() -> None:
    """Inflate vLLM's planned ``num_blocks`` by the BF16/INT cost ratio.

    vLLM sizes ``num_blocks`` assuming BF16 KV layout.  When skip_native_kv
    is on, our actual KV is INT2 (or whatever the scheme is) — costing a
    fraction of the BF16 bytes per block.  Without this patch, JetFire's
    INT mirror only fills ~1/N of the HBM budget and the scheduler still
    admits the same concurrency as BF16, so the HBM saving is invisible.

    Patching at ``get_num_blocks`` is the single source of truth: there's
    exactly one call site (``_get_kv_cache_config_uniform_*``) and the
    returned value flows into ``KVCacheConfig.num_blocks``, which the
    scheduler then divides by ``num_groups`` to admit sequences.
    """
    global _PATCHED_PLANNER
    if _PATCHED_PLANNER:
        return

    from vllm.v1.core import kv_cache_utils

    from .config import compute_inflate_ratio

    original_get_num_blocks = kv_cache_utils.get_num_blocks
    original_get_kv_cache_configs = kv_cache_utils.get_kv_cache_configs
    original_get_kv_cache_config_from_groups = (
        kv_cache_utils.get_kv_cache_config_from_groups
    )
    def _debug_planner_enabled() -> bool:
        return os.environ.get("JETFIRE_DEBUG_PLANNER", "0") == "1"

    if _debug_planner_enabled():
        print("[jetfire] installing KV planner inflation patch", flush=True)

    def _inflate_num_blocks(
        *,
        vllm_config,
        num_blocks: int,
        available_memory: int,
        per_block_bf16: int,
        ratio: float,
    ) -> int:
        inflated_naive = max(int(num_blocks * ratio), int(num_blocks))

        try:
            import torch as _torch
            total_hbm = _torch.cuda.get_device_properties(0).total_memory
            gmu = float(vllm_config.cache_config.gpu_memory_utilization)
            warmup_peak_est = max(
                0, int(total_hbm * gmu) - int(available_memory)
            )
        except Exception:
            warmup_peak_est = 0

        allocator_factor = float(os.environ.get(
            "JETFIRE_ALLOCATOR_RESIDUAL_FACTOR", "0.0"
        ))
        jf_fixed_bytes = int(float(os.environ.get(
            "JETFIRE_FIXED_OVERHEAD_GB", "0.5"
        )) * (1 << 30))

        deduction = int(allocator_factor * warmup_peak_est) + jf_fixed_bytes
        safe_available = max(0, int(available_memory) - deduction)
        n_safe_baseline = (
            safe_available // int(per_block_bf16)
            if int(per_block_bf16) > 0
            else int(num_blocks)
        )
        n_safe_inflated = max(int(n_safe_baseline * ratio), int(num_blocks))
        final = min(inflated_naive, n_safe_inflated)

        if final != int(num_blocks) and _debug_planner_enabled():
            print(
                f"[jetfire] num_blocks {int(num_blocks)} -> {final} "
                f"(naive_inflate {inflated_naive}, safety_cap {n_safe_inflated}, "
                f"ratio {ratio:.2f}, peak_est {warmup_peak_est/(1<<30):.1f}GiB, "
                f"deduction {deduction/(1<<30):.2f}GiB)",
                flush=True,
            )
        return int(final)

    def _jetfire_get_num_blocks(vllm_config, num_layers, available_memory, page_size):
        # Keep this as a transparent wrapper.  vLLM has multiple planner
        # branches: some call get_num_blocks(), while LLaMA-style uniform KV
        # specs bypass it entirely.  We therefore inflate the final
        # KVCacheConfig below, after all branches have converged.
        return original_get_num_blocks(
            vllm_config, num_layers, available_memory, page_size
        )

    kv_cache_utils.get_num_blocks = _jetfire_get_num_blocks

    def _jetfire_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        config = original_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )
        cfg = get_jetfire_config()
        if not (cfg.use_custom_backend and cfg.skip_native_kv):
            return config
        if not kv_cache_groups:
            return config
        uniform_type = getattr(kv_cache_utils, "UniformTypeKVCacheSpecs", None)
        if uniform_type is None:
            return config
        if not (
            len(kv_cache_groups) == 1
            and isinstance(kv_cache_groups[0].kv_cache_spec, uniform_type)
        ):
            # The general path already goes through patched get_num_blocks().
            return config
        if _debug_planner_enabled():
            print("[jetfire] applying uniform KV planner inflation", flush=True)
        old = int(config.num_blocks)
        ratio = compute_inflate_ratio(cfg.scheme, cfg.group_size)
        per_block_bf16 = int(kv_cache_groups[0].kv_cache_spec.page_size_bytes)
        new = _inflate_num_blocks(
            vllm_config=vllm_config,
            num_blocks=old,
            available_memory=available_memory,
            per_block_bf16=per_block_bf16,
            ratio=ratio,
        )
        if new <= old:
            return config
        config.num_blocks = new
        for tensor in config.kv_cache_tensors:
            if tensor.size % old != 0:
                raise RuntimeError(
                    "JetFire cannot inflate uniform KV tensor size: "
                    f"size={tensor.size} num_blocks={old}"
                )
            tensor.size = tensor.size // old * new
        setattr(config, "_jetfire_inflated_num_blocks", True)
        return config

    kv_cache_utils.get_kv_cache_config_from_groups = (
        _jetfire_get_kv_cache_config_from_groups
    )

    def _jetfire_get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        configs = original_get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )
        cfg = get_jetfire_config()
        if not (cfg.use_custom_backend and cfg.skip_native_kv):
            return configs

        candidates: list[int] = []
        old_blocks: list[int] = []
        ratio = compute_inflate_ratio(cfg.scheme, cfg.group_size)
        for config, avail_mem in zip(configs, available_memory):
            old = int(config.num_blocks)
            old_blocks.append(old)
            if getattr(config, "_jetfire_inflated_num_blocks", False):
                candidates.append(old)
                continue
            if old <= 0 or not config.kv_cache_tensors:
                candidates.append(old)
                continue
            total_size = sum(int(tensor.size) for tensor in config.kv_cache_tensors)
            if total_size <= 0 or total_size % old != 0:
                raise RuntimeError(
                    "JetFire cannot infer per-block KV size for planner inflation: "
                    f"total_size={total_size} num_blocks={old}"
                )
            per_block_bf16 = total_size // old
            candidates.append(
                _inflate_num_blocks(
                    vllm_config=vllm_config,
                    num_blocks=old,
                    available_memory=avail_mem,
                    per_block_bf16=per_block_bf16,
                    ratio=ratio,
                )
            )

        if not candidates:
            return configs
        target = min(int(n) for n in candidates)
        for config, old in zip(configs, old_blocks):
            if target <= old or getattr(config, "_jetfire_inflated_num_blocks", False):
                continue
            if _debug_planner_enabled():
                print(
                    f"[jetfire] postprocess KV planner inflation {old} -> {target}",
                    flush=True,
                )
            config.num_blocks = target
            for tensor in config.kv_cache_tensors:
                if tensor.size % old != 0:
                    raise RuntimeError(
                        "JetFire cannot inflate KV tensor size: "
                        f"size={tensor.size} num_blocks={old}"
                    )
                tensor.size = tensor.size // old * target
            setattr(config, "_jetfire_inflated_num_blocks", True)
        return configs

    kv_cache_utils.get_kv_cache_configs = _jetfire_get_kv_cache_configs
    try:
        import vllm.v1.engine.core as engine_core
        engine_core.get_kv_cache_configs = _jetfire_get_kv_cache_configs
        engine_core.EngineCore.__init__.__globals__["get_kv_cache_configs"] = (
            _jetfire_get_kv_cache_configs
        )
    except Exception:
        pass

    # ── Patch admission check ───────────────────────────────────────────
    # vLLM's `_check_enough_kv_cache_memory` rejects engine init when
    #   max_model_len * BF16_page_size_bytes > available_memory.
    # That formula assumes BF16 KV; with INT2 storage we actually use
    # ~1/ratio of those bytes, so the check is too conservative and blocks
    # otherwise-feasible setups (e.g. low gpu_memory_utilization for HBM-
    # constrained throughput demos).  Wrap `get_needed_memory` to scale by
    # 1/ratio when JetFire is the active backend.
    original_check_enough = kv_cache_utils._check_enough_kv_cache_memory

    def _jetfire_check_enough_kv_cache_memory(
        available_memory, get_needed_memory, max_model_len, estimate_max_model_len
    ):
        cfg = get_jetfire_config()
        if not (cfg.use_custom_backend and cfg.skip_native_kv):
            return original_check_enough(
                available_memory, get_needed_memory, max_model_len,
                estimate_max_model_len,
            )
        ratio = compute_inflate_ratio(cfg.scheme, cfg.group_size)
        if os.environ.get("JETFIRE_DEBUG_PLANNER", "0") == "1":
            try:
                needed_dbg = get_needed_memory()
            except Exception:
                needed_dbg = -1
            print(
                f"[jetfire] check KV memory avail={int(available_memory)/(1<<30):.2f}GiB "
                f"needed_bf16={int(needed_dbg)/(1<<30):.2f}GiB "
                f"needed_int={int(needed_dbg / ratio)/(1<<30):.2f}GiB "
                f"ratio={ratio:.2f}",
                flush=True,
            )

        def scaled_needed():
            return int(get_needed_memory() / ratio)

        def scaled_estimate(avail):
            # Estimator inverts the BF16 formula; multiply available memory
            # by ratio so the inverse arithmetic stays self-consistent.
            return estimate_max_model_len(int(avail * ratio))

        return original_check_enough(
            available_memory, scaled_needed, max_model_len, scaled_estimate,
        )

    kv_cache_utils._check_enough_kv_cache_memory = (
        _jetfire_check_enough_kv_cache_memory
    )
    _PATCHED_PLANNER = True


def _maybe_patch_worker_methods() -> None:
    global _PATCHED_WORKER_METHODS
    if _PATCHED_WORKER_METHODS:
        return

    original_initialize_kv_cache_tensors = GPUModelRunner.initialize_kv_cache_tensors
    original_profile_run = GPUModelRunner.profile_run
    original_profile_cudagraph_memory = getattr(
        GPUModelRunner, "profile_cudagraph_memory", None
    )
    original_worker_load_model = Worker.load_model
    original_worker_get_kv_cache_spec = Worker.get_kv_cache_spec
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker_module
        original_kernel_warmup = gpu_worker_module.kernel_warmup
    except Exception:
        gpu_worker_module = None
        original_kernel_warmup = None

    def _startup_dummy_kv_enabled() -> bool:
        return get_jetfire_config().skip_native_kv

    def _skip_profile_sampler_enabled() -> bool:
        raw = os.environ.get("JETFIRE_SKIP_PROFILE_SAMPLER")
        if raw is not None:
            return raw.strip() == "1"
        cfg = get_jetfire_config()
        return cfg.use_custom_backend and cfg.skip_native_kv

    def _skip_kernel_warmup_enabled() -> bool:
        raw = os.environ.get("JETFIRE_SKIP_KERNEL_WARMUP")
        if raw is not None:
            return raw.strip() == "1"
        cfg = get_jetfire_config()
        return cfg.use_custom_backend and cfg.skip_native_kv

    def _debug_graph_init(message: str) -> None:
        if os.environ.get("JETFIRE_DEBUG_GRAPH_INIT", "0") == "1":
            print(f"[jetfire] {message}", flush=True)

    def _maybe_prepare_decode_graph_for_runner(model_runner):
        if getattr(model_runner, "_jetfire_in_cudagraph_memory_profile", False):
            model_runner._jetfire_decode_graph_deferred = True
            model_runner._jetfire_decode_graph_info = None
            _debug_graph_init(
                "defer decode graph prepare during vLLM CUDA graph memory profile"
            )
            return None

        info = _maybe_prepare_decode_graph(model_runner.device)
        if info is not None:
            model_runner._jetfire_decode_graph_deferred = False
            _debug_graph_init(
                "prepared decode graph contexts "
                f"stores_created={info.get('stores_created', 0)}"
            )
        return info

    def _install_layer_configs(model_runner) -> int:
        pool = get_global_kv_pool_manager()
        installed = 0
        for layer_name, layer in model_runner.compilation_config.static_forward_context.items():
            num_heads = getattr(layer, "num_heads", None)
            num_kv_heads = getattr(layer, "num_kv_heads", None)
            head_size = getattr(layer, "head_size", None)
            if num_heads is None or num_kv_heads is None or head_size is None:
                continue
            pool.register_layer_config(
                layer_name,
                num_query_heads=int(num_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_size),
            )
            installed += 1
        return installed

    def _jetfire_worker_load_model(self, *, load_dummy_weights: bool = False) -> None:
        original_worker_load_model(self, load_dummy_weights=load_dummy_weights)
        if not _startup_dummy_kv_enabled():
            return
        _install_layer_configs(self.model_runner)

    def _jetfire_worker_get_kv_cache_spec(self):
        kv_cache_spec = original_worker_get_kv_cache_spec(self)
        cfg = get_jetfire_config()
        if _startup_dummy_kv_enabled():
            _install_layer_configs(self.model_runner)
        if not (
            cfg.use_custom_backend
            and cfg.skip_native_kv
            and _compressed_kv_spec_enabled()
        ):
            return kv_cache_spec

        from .config import compute_inflate_ratio

        ratio = compute_inflate_ratio(cfg.scheme, cfg.group_size)
        compressed: dict[str, object] = {}
        for layer_name, spec in kv_cache_spec.items():
            num_kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
            head_size = int(getattr(spec, "head_size", 0) or 0)
            dtype = getattr(spec, "dtype", None)
            if num_kv_heads <= 0 or head_size <= 0 or dtype is None:
                compressed[layer_name] = spec
                continue

            # vLLM's scheduler only needs the number of KV blocks.  JetFire
            # does not use native KV tensors when skip_native_kv=1, so expose
            # a placeholder KV spec whose page size matches JetFire's INT-N
            # bytes per block.  Keep dtype bf16 and reduce the logical
            # num_kv_heads*head_size product so vLLM's normal page-size math
            # and allocator stay self-consistent.
            native_product = num_kv_heads * head_size
            compressed_product = max(1, int(round(native_product / ratio)))
            kwargs = {
                "num_kv_heads": 1,
                "head_size": compressed_product,
            }
            if hasattr(spec, "head_size_v"):
                kwargs["head_size_v"] = compressed_product
            compressed[layer_name] = replace(spec, **kwargs)

        if os.environ.get("JETFIRE_DEBUG_PLANNER", "0") == "1":
            first = next(iter(compressed.values()), None)
            if first is not None:
                print(
                    "[jetfire] compressed KV spec enabled: "
                    f"mode={cfg.scheme.name} ratio={ratio:.2f} "
                    f"page_size={getattr(first, 'page_size_bytes', '?')} bytes",
                    flush=True,
                )
        return compressed

    def _jetfire_profile_run(self):
        if not _skip_profile_sampler_enabled():
            return original_profile_run(self)

        original_dummy_sampler_run = self._dummy_sampler_run

        def _jetfire_profile_dummy_sampler_run(hidden_states):
            # vLLM's memory profile can OOM in dummy logits all-gather at very
            # long context after the large dummy forward.  This profiles
            # sampler scratch, not JetFire KV capacity.  Real generate still
            # uses the original sampler path.
            return torch.empty(0, device=self.device)

        self._dummy_sampler_run = _jetfire_profile_dummy_sampler_run
        try:
            return original_profile_run(self)
        finally:
            self._dummy_sampler_run = original_dummy_sampler_run

    def _jetfire_profile_cudagraph_memory(self):
        if original_profile_cudagraph_memory is None:
            return 0
        self._jetfire_in_cudagraph_memory_profile = True
        old_profile_env = os.environ.get("JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE")
        os.environ["JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE"] = "1"
        try:
            return original_profile_cudagraph_memory(self)
        finally:
            self._jetfire_in_cudagraph_memory_profile = False
            if old_profile_env is None:
                os.environ.pop("JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE", None)
            else:
                os.environ["JETFIRE_IN_VLLM_CUDAGRAPH_PROFILE"] = old_profile_env

    def _jetfire_kernel_warmup(worker):
        if _skip_kernel_warmup_enabled():
            return None
        if original_kernel_warmup is None:
            return None
        return original_kernel_warmup(worker)

    def _jetfire_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        pool = get_global_kv_pool_manager()
        # Always register layer configs + layouts — they're what mirror_kv /
        # _decode_batched look up to gate the shadow store creation, and they
        # must be in place regardless of whether we free vLLM's native KV
        # cache.  (Historically this was inside the skip_native_kv branch,
        # which silently disabled the INT4 path when skip_native_kv=0 — for
        # example under graph mode where we keep native KV alive for the
        # CUDA-graph capture warmup fallback.)
        _install_layer_configs(self)
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in pool.layer_configs:
                    pool.register_layer_layout(
                        layer_name,
                        num_blocks=int(kv_cache_config.num_blocks),
                        block_size=int(group.kv_cache_spec.block_size),
                    )

        if not _startup_dummy_kv_enabled():
            kv_caches = original_initialize_kv_cache_tensors(
                self, kv_cache_config, kernel_block_sizes
            )
            self._jetfire_decode_graph_info = (
                _maybe_prepare_decode_graph_for_runner(self)
            )
            return kv_caches

        pool.reset_runtime_state(preserve_layouts=True)
        _install_layer_configs(self)

        # NOTE: we follow vLLM's planned num_blocks for our shadow / INT3 cache
        # sizing.  vLLM's planner sizes num_blocks by (available GPU memory) /
        # (per-block real KV size) — for single-sequence workloads this can
        # over-allocate 5–6× vs what max_model_len actually requires.
        # Capping it here breaks vLLM's scheduler (it assigns block_ids in
        # [0, planned_num_blocks)).  Instead, the true per-token cost is
        # reported as `written_count × per_token_bytes` in the paper.
        layer_specs = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                layer_specs[layer_name] = group.kv_cache_spec
                if layer_name in pool.layer_configs:
                    pool.register_layer_layout(
                        layer_name,
                        num_blocks=int(kv_cache_config.num_blocks),
                        block_size=int(group.kv_cache_spec.block_size),
                    )

        self._jetfire_decode_graph_info = _maybe_prepare_decode_graph_for_runner(self)

        kv_caches: dict[str, torch.Tensor] = {}
        for tensor_cfg in kv_cache_config.kv_cache_tensors:
            if not tensor_cfg.shared_by:
                continue
            first_layer = tensor_cfg.shared_by[0]
            spec = layer_specs[first_layer]
            block_size = getattr(spec, "block_size", self.cache_config.block_size)
            dtype = getattr(spec, "dtype", torch.float16)
            dummy_kv = torch.empty(
                (2, int(kv_cache_config.num_blocks), int(block_size), 1, 1),
                dtype=dtype,
                device=self.device,
            )
            for layer_name in tensor_cfg.shared_by:
                kv_caches[layer_name] = dummy_kv

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        self.cross_layers_kv_cache = None
        self.cross_layers_attn_backend = None

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def _jetfire_install_hooks(self):
        pool = get_global_kv_pool_manager()
        model_runner = self.model_runner
        if _startup_dummy_kv_enabled():
            if not pool.layer_configs:
                _install_layer_configs(model_runner)
            pool.reset_runtime_state(preserve_layouts=True)
            model_runner._jetfire_decode_graph_info = (
                _maybe_prepare_decode_graph_for_runner(model_runner)
            )
            _maybe_warmup_decode_buffers()
            gc.collect()
            torch.accelerator.empty_cache()
            return {
                "installed": True,
                "layers": len(pool.layer_configs),
                "released_bytes": 0,
                "mode": "load_model_install",
            }

        released_ptrs: set[int] = set()
        released_bytes = 0
        dummy_kv_caches: dict[str, torch.Tensor] = {}

        for layer_name, layer in model_runner.compilation_config.static_forward_context.items():
            kv_cache = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim < 5:
                continue

            _, num_blocks, block_size, num_kv_heads, head_dim = kv_cache.shape
            if layer_name not in pool.layer_configs:
                num_heads = getattr(layer, "num_heads", None)
                if num_heads is not None:
                    pool.register_layer_config(
                        layer_name,
                        num_query_heads=int(num_heads),
                        num_kv_heads=int(num_kv_heads),
                        head_dim=int(head_dim),
                    )
            pool.register_layer_layout(
                layer_name,
                num_blocks=int(num_blocks),
                block_size=int(block_size),
            )

            ptr = kv_cache.untyped_storage().data_ptr()
            if ptr not in released_ptrs:
                released_ptrs.add(ptr)
                released_bytes += kv_cache.untyped_storage().nbytes()

            dummy_kv_caches[layer_name] = torch.empty(
                (2, 1, 16, max(1, int(num_kv_heads)), max(1, int(head_dim))),
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            )

        torch.accelerator.synchronize()
        if hasattr(model_runner, "kv_caches") and model_runner.kv_caches:
            for i in range(len(model_runner.kv_caches)):
                model_runner.kv_caches[i] = None  # type: ignore
            model_runner.kv_caches.clear()
        model_runner.cross_layers_kv_cache = None
        model_runner.cross_layers_attn_backend = None

        num_attn_module = (
            2 if model_runner.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            dummy_kv_caches,
            model_runner.compilation_config.static_forward_context,
            model_runner.kv_caches,
            num_attn_module,
        )
        pool.reset_runtime_state(preserve_layouts=True)
        model_runner._jetfire_decode_graph_info = (
            _maybe_prepare_decode_graph_for_runner(model_runner)
        )
        _maybe_warmup_decode_buffers()
        gc.collect()
        torch.accelerator.empty_cache()
        return {
            "installed": True,
            "layers": len(dummy_kv_caches),
            "released_bytes": released_bytes,
        }

    def _jetfire_get_stats(self):
        stats = get_global_kv_pool_manager().get_stats()
        try:
            import torch.distributed as dist

            rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else -1
            world_size = int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1
        except Exception:
            rank = -1
            world_size = 1
        try:
            cuda_device = int(torch.cuda.current_device())
        except Exception:
            cuda_device = -1
        stats.update({
            "worker_pid": int(os.getpid()),
            "worker_rank": int(rank),
            "worker_world_size": int(world_size),
            "worker_device": str(self.device),
            "worker_cuda_device": int(cuda_device),
            "decode_graph_enabled": os.environ.get("JETFIRE_DECODE_GRAPH", "0"),
            "graph_cpu_topk_device_signal": os.environ.get("JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK", "0"),
            "graph_cpu_topk_host_join": os.environ.get("JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK", "0"),
            "graph_cpu_topk_side_join": os.environ.get("JETFIRE_GRAPH_SIDE_JOIN_CPU_TOPK", "0"),
            "graph_cpu_topk_gpu_wait": os.environ.get("JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK", "0"),
        })
        try:
            from .decode_path import jetfire_get_graph_stats
            stats.update(jetfire_get_graph_stats())
        except Exception:
            pass
        graph_info = getattr(self.model_runner, "_jetfire_decode_graph_info", None)
        if graph_info is not None:
            stats["decode_graph_info"] = dict(graph_info)
        return stats

    def _jetfire_get_memory(self):
        pool = get_global_kv_pool_manager()
        device = torch.device(self.device)
        torch.cuda.synchronize(device)

        # Two views of "JetFire KV memory":
        #   *_alloc_bytes  → what we actually allocated (vLLM's planner sizing)
        #   *_used_bytes   → what we'd need given the actually-written tokens
        #                    (per-token-bytes × max_written_count); this is the
        #                    paper-relevant number, since vLLM over-plans
        #                    num_blocks 5–6× for single-seq workloads.
        jetfire_int4_bytes = 0
        jetfire_int4_used_bytes = 0
        max_written = 0
        for store in pool.layer_stores.values():
            jetfire_int4_bytes += store.k_data.numel() * store.k_data.element_size()
            jetfire_int4_bytes += store.k_scales.numel() * store.k_scales.element_size()
            jetfire_int4_bytes += store.v_data.numel() * store.v_data.element_size()
            jetfire_int4_bytes += store.v_scales.numel() * store.v_scales.element_size()
            jetfire_int4_bytes += store.slot_filled.numel() * store.slot_filled.element_size()
            if store.k_asymmetric and store.k_zp is not None:
                jetfire_int4_bytes += store.k_zp.numel() * store.k_zp.element_size()
            if store.v_asymmetric and store.v_zp is not None:
                jetfire_int4_bytes += store.v_zp.numel() * store.v_zp.element_size()
            max_written = max(max_written, int(store.written_count))

        # Per-token cost: (k_data + k_scales + k_zp(asym)) × num_kv_heads + same for V
        if pool.layer_stores:
            s = next(iter(pool.layer_stores.values()))
            from jetfire_vllm.kernels.intN_pack import packed_dim
            kp = packed_dim(s.head_dim, s.k_nbits)
            vp = packed_dim(s.head_dim, s.v_nbits)
            ng = s.head_dim // s.group_size
            k_scale_b = 2   # bf16 always
            v_scale_b = 2
            k_zp_b = 2 if s.k_asymmetric else 0
            v_zp_b = 2 if s.v_asymmetric else 0
            per_token_int = (
                (kp + ng * (k_scale_b + k_zp_b))
                + (vp + ng * (v_scale_b + v_zp_b))
            ) * s.num_kv_heads
            jetfire_int4_used_bytes = per_token_int * max_written * len(pool.layer_stores)
            if s.k_shadow is not None:
                per_token_shadow = (
                    s.head_dim * s.k_shadow.element_size() * 2 * s.num_kv_heads
                )
                jetfire_cpu_used_bytes = (
                    per_token_shadow * max_written * len(pool.layer_stores)
                )
            else:
                jetfire_cpu_used_bytes = 0
        else:
            jetfire_cpu_used_bytes = 0

        jetfire_cpu_pinned_bytes = 0
        for store in pool.layer_stores.values():
            if store.k_full_cpu is not None:
                jetfire_cpu_pinned_bytes += (
                    store.k_full_cpu.numel() * store.k_full_cpu.element_size()
                )
            if store.v_full_cpu is not None:
                jetfire_cpu_pinned_bytes += (
                    store.v_full_cpu.numel() * store.v_full_cpu.element_size()
                )

        # vLLM-side KV cache (dummy in skip-native mode, real otherwise).
        vllm_kv_bytes = 0
        seen_ptrs: set[int] = set()
        model_runner = self.model_runner
        for layer in model_runner.compilation_config.static_forward_context.values():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None:
                continue
            tensors = kv_cache if isinstance(kv_cache, list) else [kv_cache]
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    continue
                ptr = t.untyped_storage().data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                vllm_kv_bytes += t.untyped_storage().nbytes()

        return {
            "device": str(device),
            "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)),
            "cuda_memory_reserved": int(torch.cuda.memory_reserved(device)),
            "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "jetfire_int4_hbm_used_bytes": int(jetfire_int4_used_bytes),
            "jetfire_cpu_pinned_used_bytes": int(jetfire_cpu_used_bytes),
            "max_written_tokens": int(max_written),
            "jetfire_int4_hbm_bytes": int(jetfire_int4_bytes),
            "jetfire_cpu_pinned_bytes": int(jetfire_cpu_pinned_bytes),
            "vllm_kv_bytes": int(vllm_kv_bytes),
            "num_layer_stores": len(pool.layer_stores),
        }

    def _jetfire_reset_peak_memory(self):
        device = torch.device(self.device)
        torch.cuda.reset_peak_memory_stats(device)
        return True

    def _jetfire_reset_written_count(self):
        """Reset per-store written_count so next prefill measures 1 context
        worth of data (no accumulation across runs)."""
        pool = get_global_kv_pool_manager()
        with torch.inference_mode():
            for store in pool.layer_stores.values():
                store.written_count = 0
                if hasattr(store, "slot_filled"):
                    store.slot_filled.fill_(False)
        return True

    def _jetfire_get_profile(self):
        from .decode_path import jetfire_get_profile
        return jetfire_get_profile()

    def _jetfire_reset_graph_stats(self):
        from .decode_path import reset_jetfire_graph_stats
        reset_jetfire_graph_stats()
        return True

    def _jetfire_reset_runtime_state(self):
        pool = get_global_kv_pool_manager()
        torch.cuda.synchronize(torch.device(self.device))
        pool.flush_pending_writes()
        if os.environ.get("JETFIRE_DECODE_GRAPH", "0") == "1":
            pool.reset_request_state_for_graph_reuse()
            from .decode_path import reset_jetfire_graph_runtime_state
            reset_jetfire_graph_runtime_state()
            torch.accelerator.empty_cache()
            return True
        pool.reset_runtime_state(preserve_layouts=True)
        from .decode_path import reset_jetfire_decode_runtime_state
        reset_jetfire_decode_runtime_state()
        torch.accelerator.empty_cache()
        return True

    def _jetfire_prewarm_prefill_staging(self, num_tokens: int):
        pool = get_global_kv_pool_manager()
        device = torch.device(self.device)
        if not pool.layer_configs:
            _install_layer_configs(self.model_runner)
        t0 = time.perf_counter()
        info = pool.prewarm_prefill_staging(
            num_tokens=int(num_tokens),
            device=device,
        )
        torch.cuda.synchronize(device)
        info["wall_s"] = float(time.perf_counter() - t0)
        return info

    GPUModelRunner.initialize_kv_cache_tensors = _jetfire_initialize_kv_cache_tensors
    GPUModelRunner.profile_run = _jetfire_profile_run
    if original_profile_cudagraph_memory is not None:
        GPUModelRunner.profile_cudagraph_memory = _jetfire_profile_cudagraph_memory
    if gpu_worker_module is not None:
        gpu_worker_module.kernel_warmup = _jetfire_kernel_warmup
    Worker.load_model = _jetfire_worker_load_model
    Worker.get_kv_cache_spec = _jetfire_worker_get_kv_cache_spec
    Worker.jetfire_install_hooks = _jetfire_install_hooks
    Worker.jetfire_get_stats = _jetfire_get_stats
    Worker.jetfire_get_memory = _jetfire_get_memory
    Worker.jetfire_reset_peak_memory = _jetfire_reset_peak_memory
    Worker.jetfire_reset_written_count = _jetfire_reset_written_count
    Worker.jetfire_get_profile = _jetfire_get_profile
    Worker.jetfire_reset_graph_stats = _jetfire_reset_graph_stats
    Worker.jetfire_reset_runtime_state = _jetfire_reset_runtime_state
    Worker.jetfire_prewarm_prefill_staging = _jetfire_prewarm_prefill_staging
    _PATCHED_WORKER_METHODS = True
