"""JetFire-oriented vLLM engine construction helpers."""

from __future__ import annotations

import os
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Avoid the PyTorch caching-allocator fragmentation OOM at long ctx
# (ctx ≥ 32k).  Without this, at high ``gpu_memory_utilization`` vLLM's
# profile/warmup forward pass leaves ~22 % of in-use bytes as
# reserved-but-unallocated, which pushes total HBM use past the physical
# limit during init.  ``expandable_segments`` lets the allocator grow
# segments dynamically rather than locking power-of-2 chunks, eliminating
# the fragmentation.  Hardware/model agnostic.
#
# Set BEFORE importing torch (any vLLM/torch import does this implicitly).
# ``setdefault`` so user can override via the env var.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def _prefer_vllm_bundled_triton_kernels() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return
    path = Path(next(iter(spec.submodule_search_locations))) / "third_party"
    if not (path / "triton_kernels").is_dir():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if path_str not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *parts])


_prefer_vllm_bundled_triton_kernels()


# ── Orphan-prevention via PR_SET_PDEATHSIG ─────────────────────────────
# vLLM spawns its EngineCore as a separate ``multiprocessing.spawn``
# subprocess.  When the parent dies abnormally (SIGTERM from ``timeout``,
# OOM-killer, kill -9, etc.), the EngineCore is reparented to init and
# continues holding 70+ GiB of GPU memory until manually killed.  That
# breaks every subsequent run on the same GPU.
#
# Fix: the EngineCore process tells the kernel to send it ``SIGKILL`` the
# instant its parent exits.  ``prctl(PR_SET_PDEATHSIG)`` is Linux-only
# but JetFire only targets Linux + CUDA, so this is fine.
#
# We install this via a ``multiprocessing.util.Finalize``-style hook by
# monkey-patching ``multiprocessing.spawn._main`` so every spawned child
# (vLLM uses spawn, not fork) sets the death signal at the very top of
# its ``__main__`` execution before any vLLM code runs.
def _install_pdeathsig_in_spawned_children() -> None:
    """Make every multiprocessing.spawn child auto-die when its parent dies.

    Effect: when this process (the bench / API server) is killed, the
    kernel immediately sends SIGKILL to every descendant, including
    vLLM's EngineCore.  No more orphan EngineCores holding GPU memory.
    """
    try:
        import ctypes
        import multiprocessing.spawn as _spawn

        if getattr(_spawn, "_jetfire_pdeathsig_installed", False):
            return

        _PR_SET_PDEATHSIG = 1
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _orig_main = _spawn._main

        def _patched_main(*args, **kwargs):
            try:
                # SIGKILL = 9.  Cannot be caught/ignored; cleanest exit
                # for a stuck CUDA process.
                _libc.prctl(_PR_SET_PDEATHSIG, 9, 0, 0, 0)
            except Exception:
                pass
            return _orig_main(*args, **kwargs)

        _spawn._main = _patched_main
        _spawn._jetfire_pdeathsig_installed = True
    except Exception:
        # Non-Linux or libc not available: silently skip.  The bench-side
        # process-group cleanup is a sufficient fallback.
        pass


_install_pdeathsig_in_spawned_children()

from vllm import LLM

from .config import apply_graph_cpu_topk_release_preset, get_jetfire_config
from .release_policy import ReleasePolicyError, release_policy_enabled
from .registry import ensure_jetfire_registered


_ADAPTER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADAPTER_DIR.parents[1]


def _prompt_count(prompts: Any) -> int:
    if isinstance(prompts, (str, bytes)):
        return 1
    if isinstance(prompts, Mapping):
        return 1
    if isinstance(prompts, Sequence):
        return len(prompts)
    return 1


def _jetfire_capacity_tokens_from_stats(stats: list[dict[str, object]]) -> int | None:
    capacities: list[int] = []
    for row in stats:
        layouts = row.get("layer_layouts")
        if not isinstance(layouts, dict):
            continue
        for layout in layouts.values():
            if not isinstance(layout, dict):
                continue
            num_blocks = layout.get("num_blocks")
            block_size = layout.get("block_size")
            if num_blocks is None or block_size is None:
                continue
            capacities.append(int(num_blocks) * int(block_size))
    return min(capacities) if capacities else None


def _clone_sampling_params_with_prefix_bypass(sampling_params: Any) -> Any:
    if sampling_params is None:
        return None
    if isinstance(sampling_params, Sequence) and not isinstance(
        sampling_params, (str, bytes)
    ):
        cloned = []
        for params in sampling_params:
            item = params.clone() if hasattr(params, "clone") else params
            if hasattr(item, "skip_reading_prefix_cache"):
                item.skip_reading_prefix_cache = True
            cloned.append(item)
        return cloned
    cloned = sampling_params.clone() if hasattr(sampling_params, "clone") else sampling_params
    if hasattr(cloned, "skip_reading_prefix_cache"):
        cloned.skip_reading_prefix_cache = True
    return cloned


def _slice_optional_sequence(value: Any, start: int, end: int) -> Any:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[start:end]
    return value


def _target_microbatch_size(n_prompts: int) -> int:
    configured = os.environ.get("JETFIRE_GENERATE_MICROBATCH_SIZE")
    if configured:
        return max(1, int(configured))
    # Default to one-shot admission.  Capacity and scheduler-budget checks
    # below will still split unsafe requests, but we should not silently turn
    # a requested B64 experiment into four smaller B16 runs.
    target_chunks = max(1, int(os.environ.get("JETFIRE_GENERATE_TARGET_CHUNKS", "1")))
    return max(1, (int(n_prompts) + target_chunks - 1) // target_chunks)


def _maybe_expand_scheduler_budget(kwargs: dict[str, Any]) -> None:
    """Let JetFire use a performance-oriented one-shot prefill microbatch.

    vLLM's default long-context benchmark budget is often ``ctx + O(1k)``.
    That is fine for native KV + chunked prefill, but under JetFire
    ``skip_native_kv`` it forces B32/ctx32k into cached-context
    extend-prefill, which JetFire intentionally rejects.  Instead of
    degrading to B1, raise the scheduler budget so each JetFire microbatch
    can be prefetched in one scheduler step.
    """
    if os.environ.get("JETFIRE_AUTO_SCHEDULER_BUDGET", "1") != "1":
        return
    cfg = get_jetfire_config()
    if not (cfg.use_custom_backend and cfg.skip_native_kv):
        return
    max_num_seqs = int(kwargs.get("max_num_seqs") or 0)
    max_model_len = int(kwargs.get("max_model_len") or 0)
    if max_num_seqs <= 1 or max_model_len <= 0:
        return
    target_batch = min(max_num_seqs, _target_microbatch_size(max_num_seqs))
    desired = target_batch * max_model_len
    # Safety valve for vLLM startup/profile activation memory.  This is not a
    # JetFire KV-capacity limit: vLLM still runs a dummy/profile prefill whose
    # activation/compile peak scales with max_num_batched_tokens.  Exposing
    # B*ctx for very large batches can OOM before JetFire decode starts.
    #
    # The default is the previously validated single-GPU budget, scaled by TP
    # size.  Set JETFIRE_AUTO_MAX_BATCHED_TOKENS_CAP=0 to force one-shot
    # exposure, or set an explicit positive global cap for a machine/model.
    cap_raw = os.environ.get("JETFIRE_AUTO_MAX_BATCHED_TOKENS_CAP")
    if cap_raw is None:
        tp_size = max(1, int(kwargs.get("tensor_parallel_size") or 1))
        cap = 266240 * tp_size
    else:
        cap = int(cap_raw or 0)
    if cap > 0:
        desired = min(desired, cap)
    budget_batch = max(1, desired // max_model_len)
    current = int(kwargs.get("max_num_batched_tokens") or 0)
    if desired <= current:
        return
    kwargs["max_num_batched_tokens"] = desired
    print(
        "[jetfire] expanded vLLM scheduler token budget for one-shot "
        f"JetFire microbatch prefill: {current} -> {desired} "
        f"(target_microbatch={target_batch}, budget_microbatch={budget_batch}, "
        f"max_model_len={max_model_len}).",
        flush=True,
    )


class JetFireLLM:
    """Thin JetFire wrapper around vLLM's LLM object.

    The wrapper keeps benchmark/API call sites unchanged while allowing
    JetFire-specific admission safeguards that vLLM does not know about.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        max_model_len: int | None,
        max_num_batched_tokens: int | None,
        prefix_caching_enabled: bool,
    ) -> None:
        self._llm = llm
        self._max_model_len = int(max_model_len or 0)
        self._max_num_batched_tokens = int(max_num_batched_tokens or 0)
        self._prefix_caching_enabled = bool(prefix_caching_enabled)
        self._cached_capacity_tokens: int | None = None
        self._prefix_bypass_logged = False
        self._microbatch_logged = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def _get_capacity_tokens(self) -> int | None:
        if self._cached_capacity_tokens is not None:
            return self._cached_capacity_tokens
        try:
            self._cached_capacity_tokens = _jetfire_capacity_tokens_from_stats(
                collect_jetfire_worker_stats(self._llm)
            )
        except Exception:
            self._cached_capacity_tokens = None
        return self._cached_capacity_tokens

    def _oversized_capacity_request(self, prompts: Any) -> tuple[bool, int, int]:
        if os.environ.get("JETFIRE_PREFIX_CACHE_CAPACITY_BYPASS", "1") != "1":
            return False, 0, 0
        cfg = get_jetfire_config()
        if not (cfg.use_custom_backend and cfg.skip_native_kv):
            return False, 0, 0
        if not self._prefix_caching_enabled or self._max_model_len <= 0:
            return False, 0, 0
        capacity = self._get_capacity_tokens()
        if capacity is None or capacity <= 0:
            return False, 0, 0
        n_prompts = _prompt_count(prompts)
        required = n_prompts * self._max_model_len
        margin = float(os.environ.get("JETFIRE_PREFIX_CACHE_BYPASS_MARGIN", "1.0"))
        if required <= int(capacity * margin):
            return False, int(required), int(capacity)
        return True, int(required), int(capacity)

    def _should_bypass_prefix_cache(self, prompts: Any) -> bool:
        oversized, required, capacity = self._oversized_capacity_request(prompts)
        if not oversized:
            return False
        if not self._prefix_bypass_logged:
            self._prefix_bypass_logged = True
            n_prompts = _prompt_count(prompts)
            print(
                "[jetfire] prefix-cache read bypass enabled for this oversized "
                f"batch: prompts={n_prompts} max_model_len={self._max_model_len} "
                f"required≈{required} capacity={capacity}. "
                "This avoids partial-prefix extend-prefill under skip-native KV; "
                "prefill will be recomputed and the run may be slower.",
                flush=True,
            )
        return True

    def _microbatch_size(self, prompts: Any) -> int | None:
        if os.environ.get("JETFIRE_GENERATE_MICROBATCH", "1") != "1":
            return None
        if isinstance(prompts, (str, bytes, Mapping)):
            return None
        if not isinstance(prompts, Sequence):
            return None
        n_prompts = len(prompts)
        if n_prompts <= 1:
            return None
        reasons: list[str] = []
        limits: list[int] = [n_prompts]
        cfg = get_jetfire_config()

        # Debug-only escape hatch: force sequential sub-generates to avoid
        # vLLM's mixed prefill+decode admission pattern.  This is not the
        # release default because it hides JetFire's capacity advantage by
        # turning a valid B>1 long-context request into B1 serial work.
        if (
            cfg.use_custom_backend
            and cfg.skip_native_kv
            and os.environ.get("JETFIRE_AVOID_MIXED_PREFILL_DECODE", "0") == "1"
        ):
            limits.append(1)
            reasons.append("avoid vLLM mixed prefill+decode under skip-native KV")

        oversized, required, capacity = self._oversized_capacity_request(prompts)
        if oversized:
            limits.append(max(1, capacity // max(1, self._max_model_len)))
            reasons.append(f"capacity required≈{required} capacity={capacity}")

        # With skip-native KV, JetFire cannot currently serve vLLM's
        # cached-context extend-prefill path.  If one generate() request is too
        # large for the scheduler token budget, vLLM will eventually mix
        # decode tokens from active requests with an extend-prefill for another
        # request.  Keep each JetFire sub-generate within the scheduler budget
        # so prefill is one-shot and decode remains the only cached-context path.
        if (
            cfg.use_custom_backend
            and cfg.skip_native_kv
            and self._max_model_len > 0
            and self._max_num_batched_tokens > 0
        ):
            budget_safe = max(1, self._max_num_batched_tokens // self._max_model_len)
            if n_prompts > budget_safe:
                limits.append(budget_safe)
                reasons.append(
                    "scheduler_budget "
                    f"max_num_batched_tokens={self._max_num_batched_tokens} "
                    f"max_model_len={self._max_model_len}"
                )

        if not reasons:
            return None
        configured = _target_microbatch_size(n_prompts)
        chunk = max(1, min(configured, *limits))
        if chunk >= n_prompts:
            return None
        if not self._microbatch_logged:
            self._microbatch_logged = True
            print(
                "[jetfire] oversized generate will run as sequential "
                f"micro-batches: prompts={n_prompts} chunk={chunk}; "
                + "; ".join(reasons),
                flush=True,
            )
        return chunk

    def generate(
        self,
        prompts: Any,
        sampling_params: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        chunk = self._microbatch_size(prompts)
        if chunk is not None:
            results = []
            n_prompts = len(prompts)
            for start in range(0, n_prompts, chunk):
                end = min(start + chunk, n_prompts)
                chunk_prompts = prompts[start:end]
                chunk_sampling_params = _slice_optional_sequence(
                    sampling_params, start, end
                )
                if chunk_sampling_params is None:
                    chunk_sampling_params = self._llm.get_default_sampling_params()
                chunk_sampling_params = _clone_sampling_params_with_prefix_bypass(
                    chunk_sampling_params
                )
                chunk_kwargs = dict(kwargs)
                for name in ("lora_request", "priority"):
                    if name in chunk_kwargs:
                        chunk_kwargs[name] = _slice_optional_sequence(
                            chunk_kwargs[name], start, end
                        )
                results.extend(
                    self._llm.generate(
                        chunk_prompts,
                        chunk_sampling_params,
                        *args,
                        **chunk_kwargs,
                    )
                )
            return results

        if self._should_bypass_prefix_cache(prompts):
            if sampling_params is None:
                sampling_params = self._llm.get_default_sampling_params()
            sampling_params = _clone_sampling_params_with_prefix_bypass(
                sampling_params
            )
        return self._llm.generate(prompts, sampling_params, *args, **kwargs)


def _ensure_pythonpath_for_subprocess() -> None:
    """vLLM spawns the EngineCore in a fresh Python subprocess.  That
    subprocess relies on ``sitecustomize.py`` (next to this file) being
    auto-imported to register the JetFire backend.  Add the adapter dir +
    repo root to ``PYTHONPATH`` so the spawned interpreter finds it.
    """
    pp = os.environ.get("PYTHONPATH", "")
    parts = pp.split(":") if pp else []
    needed = [str(_ADAPTER_DIR), str(_REPO_ROOT)]
    if all(n in parts for n in needed):
        return
    new_pp = ":".join(needed + [p for p in parts if p and p not in needed])
    os.environ["PYTHONPATH"] = new_pp


def _apply_llm_env_overrides(kwargs: dict[str, Any]) -> None:
    tp_size = os.environ.get("JETFIRE_TENSOR_PARALLEL_SIZE")
    if tp_size:
        kwargs["tensor_parallel_size"] = max(1, int(tp_size))


def _maybe_disable_vllm_graph(kwargs: dict[str, Any]) -> None:
    """Keep JetFire out of unsafe vLLM graph-capture regimes.

    The stable production path is still the full graph path for the batch
    sizes we have validated.  Very large one-shot batches can fail before
    any JetFire decode code runs, during vLLM's model graph profiling/capture.
    In development mode, use an explicit eager fallback rather than failing
    EngineCore startup.  Strict release policy instead raises: a result that
    silently changed execution paths must not be reported as release-graph
    data.  Users can raise the threshold only after validating the new graph
    size.
    """
    if os.environ.get("JETFIRE_FORCE_VLLM_CUDAGRAPH", "0") == "1":
        return

    reason = None
    if os.environ.get("JETFIRE_ENABLE_VLLM_CUDAGRAPH") == "0":
        reason = "JETFIRE_ENABLE_VLLM_CUDAGRAPH=0"

    if reason is None and bool(kwargs.get("enable_prefix_caching", False)):
        reason = "prefix-cache/decode-cache path is not graph-safe yet"

    max_graph_batch = int(os.environ.get("JETFIRE_VLLM_GRAPH_MAX_BATCH", "32") or 0)
    max_num_seqs = int(kwargs.get("max_num_seqs") or 0)
    if reason is None and max_graph_batch > 0 and max_num_seqs > max_graph_batch:
        reason = f"max_num_seqs={max_num_seqs} > JETFIRE_VLLM_GRAPH_MAX_BATCH={max_graph_batch}"

    max_model_len = int(kwargs.get("max_model_len") or 0)
    max_num_batched_tokens = int(kwargs.get("max_num_batched_tokens") or 0)
    if (
        reason is None
        and max_num_seqs > 1
        and max_model_len > 0
        and max_num_batched_tokens > 0
    ):
        budget_batch = max(1, max_num_batched_tokens // max_model_len)
        if max_num_seqs > budget_batch:
            reason = (
                "scheduler budget requires sequential microbatch "
                f"(max_num_seqs={max_num_seqs}, budget_batch={budget_batch})"
            )

    if reason is None:
        return

    if release_policy_enabled():
        raise ReleasePolicyError(
            "strict release graph policy cannot switch to eager decode: "
            f"{reason}. Adjust the graph/scheduler configuration or validate "
            "and raise JETFIRE_VLLM_GRAPH_MAX_BATCH; do not publish this run "
            "as a release-graph result."
        )

    had_graph = (not bool(kwargs.get("enforce_eager", False))) or (
        "compilation_config" in kwargs
    )
    kwargs["enforce_eager"] = True
    kwargs.pop("compilation_config", None)
    # JetFire's graph-mode CPU/GPU handshake is only needed when vLLM captures
    # decode.  Turn it off with vLLM graphs to avoid preparing stale capture
    # state for an eager execution.
    os.environ["JETFIRE_DECODE_GRAPH"] = "0"
    os.environ["JETFIRE_ENABLE_VLLM_CUDAGRAPH"] = "0"
    if had_graph:
        print(
            "[jetfire] disabled vLLM CUDA graph for this run "
            f"({reason}); using eager decode. Set "
            "JETFIRE_FORCE_VLLM_CUDAGRAPH=1 to override.",
            flush=True,
        )


def _maybe_disable_async_scheduling(kwargs: dict[str, Any]) -> None:
    """Keep skip-native JetFire out of vLLM's async mixed prefill/decode path."""
    if os.environ.get("JETFIRE_FORCE_ASYNC_SCHEDULING", "0") == "1":
        return
    if os.environ.get("JETFIRE_DISABLE_ASYNC_SCHEDULING", "1") != "1":
        return

    cfg = get_jetfire_config()
    if not (cfg.use_custom_backend and cfg.skip_native_kv):
        return

    previous = kwargs.get("async_scheduling", None)
    kwargs["async_scheduling"] = False
    if previous is not False:
        print(
            "[jetfire] disabled vLLM async scheduling for skip-native KV. "
            "This prevents async mixed prefill+decode steps that JetFire "
            "cannot serve efficiently yet. Set "
            "JETFIRE_FORCE_ASYNC_SCHEDULING=1 to override.",
            flush=True,
        )


def _maybe_expand_prefill_concurrency(kwargs: dict[str, Any]) -> None:
    """Allow a JetFire microbatch's full-prefill requests into one step."""
    # vLLM 0.19 rejects max_num_partial_prefills != 1 in EngineArgs, so this
    # remains an explicit experiment until the scheduler supports concurrent
    # partial prefill in this path.
    if os.environ.get("JETFIRE_AUTO_PREFILL_CONCURRENCY", "0") != "1":
        return

    cfg = get_jetfire_config()
    if not (cfg.use_custom_backend and cfg.skip_native_kv):
        return

    max_num_seqs = int(kwargs.get("max_num_seqs") or 0)
    if max_num_seqs <= 1:
        return

    target = min(max_num_seqs, _target_microbatch_size(max_num_seqs))
    cap_raw = os.environ.get("JETFIRE_MAX_PREFILL_CONCURRENCY")
    if cap_raw:
        target = min(target, max(1, int(cap_raw)))

    changed = False
    for key in ("max_num_partial_prefills", "max_long_partial_prefills"):
        current = int(kwargs.get(key) or 0)
        if current < target:
            kwargs[key] = target
            changed = True

    if int(kwargs.get("long_prefill_token_threshold") or 0) != 0:
        kwargs["long_prefill_token_threshold"] = 0
        changed = True

    if changed:
        print(
            "[jetfire] expanded vLLM prefill concurrency for skip-native KV: "
            f"max_num_partial_prefills={kwargs.get('max_num_partial_prefills')} "
            f"max_long_partial_prefills={kwargs.get('max_long_partial_prefills')} "
            "long_prefill_token_threshold=0.",
            flush=True,
        )


def build_llm(
    *,
    model: str,
    use_jetfire_backend: bool = True,
    llm_kwargs: dict[str, Any] | None = None,
) -> LLM:
    kwargs = dict(llm_kwargs or {})
    _apply_llm_env_overrides(kwargs)
    if use_jetfire_backend:
        max_num_seqs = kwargs.get("max_num_seqs")
        if max_num_seqs is not None:
            os.environ.setdefault("JETFIRE_MAX_BATCH_SIZE", str(int(max_num_seqs)))
        kwargs.setdefault("attention_backend", "CUSTOM")
        _maybe_expand_scheduler_budget(kwargs)
        _maybe_expand_prefill_concurrency(kwargs)
        _maybe_disable_async_scheduling(kwargs)
        _maybe_disable_vllm_graph(kwargs)
        apply_graph_cpu_topk_release_preset()
        _ensure_pythonpath_for_subprocess()
        ensure_jetfire_registered()
    llm = LLM(model=model, **kwargs)
    if use_jetfire_backend and get_jetfire_config().install_hooks:
        llm._jetfire_hook_results = install_jetfire_worker_hooks(llm)
    if use_jetfire_backend:
        return JetFireLLM(
            llm,
            max_model_len=kwargs.get("max_model_len"),
            max_num_batched_tokens=kwargs.get("max_num_batched_tokens"),
            prefix_caching_enabled=bool(kwargs.get("enable_prefix_caching", False)),
        )
    return llm


def install_jetfire_worker_hooks(llm: LLM) -> list[dict[str, object]]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_install_hooks")


def collect_jetfire_worker_stats(llm: LLM) -> list[dict[str, object]]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_get_stats")


def collect_jetfire_worker_memory(llm: LLM) -> list[dict[str, object]]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_get_memory")


def collect_jetfire_worker_profile(llm: LLM) -> list[dict[str, list[float]]]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_get_profile")


def reset_jetfire_worker_peak_memory(llm: LLM) -> list[bool]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_reset_peak_memory")


def reset_jetfire_worker_graph_stats(llm: LLM) -> list[bool]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_reset_graph_stats")


def reset_jetfire_worker_runtime_state(llm: LLM) -> list[bool]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc("jetfire_reset_runtime_state")


def prewarm_jetfire_worker_prefill_staging(
    llm: LLM,
    num_tokens: int,
) -> list[dict[str, object]]:
    engine_core = llm.llm_engine.engine_core
    if not hasattr(engine_core, "collective_rpc"):
        raise RuntimeError("Current vLLM engine core does not support collective_rpc")
    return engine_core.collective_rpc(
        "jetfire_prewarm_prefill_staging",
        args=(int(num_tokens),),
    )


# NOTE: a calibrated ``recommend_gpu_memory_utilization`` helper used to
# live here.  It was working around a PyTorch caching-allocator
# fragmentation OOM at long ctx (the allocator was retaining ~22 % of
# in-use bytes as reserved-but-unallocated, pushing past the physical
# HBM limit).  ``PYTORCH_ALLOC_CONF=expandable_segments:True`` (set at
# module import above) eliminates the fragmentation entirely — vLLM's
# default ``gpu_memory_utilization=0.85`` then works at every ctx tested
# (4k, 8k, 32k, 64k, 128k) with no calibration.
