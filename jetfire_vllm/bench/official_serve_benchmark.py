"""Official vLLM serving benchmarks for JetFire release claims.

This is the only release benchmark entrypoint for TTFT/TPOT numbers.  It
drives vLLM's own ``bench serve`` command and preserves the same per-request
metrics vLLM prints: TTFT, TPOT, ITL, E2E latency, request throughput, output
throughput, and total token throughput.

For JetFire release tables, ``batch_ttft_ms`` and ``batch_e2el_ms`` are the
primary latency numbers. They are computed from vLLM's detailed per-request
metrics as ``max(ttfts)`` and ``max(ttft + sum(itls))``: the time until every
request in the benchmark batch has produced its first token, and the time
until every request has completed. This avoids comparing JetFire's non-chunked
prefill against vLLM chunked prefill using only median per-request TTFT, which
can report early first-token arrivals before the whole batch has finished.

The benchmark is organized around JetFire's actual claim:

    lower KV-cache footprint -> higher long-context concurrency ->
    better end-to-end serving throughput/makespan at larger workloads.

Modes:

* ``fixed-concurrency``: compare latency/throughput at one concurrency.
* ``capacity-sweep``: sweep concurrency and record the highest passing point.
* ``fixed-workload``: run a fixed number of requests at a chosen concurrency.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXT_DIR = "/tmp/jetfire_release_torch_extensions"
DEFAULT_PREFILL_CHUNK_TOKENS = 2048
PREFILL_PARTITION_REF_HIDDEN = 8192
PREFILL_PARTITION_REF_INTERMEDIATE = 28672
PREFILL_PARTITION_REF_TP = 2
PREFILL_PARTITION_REF_LOCAL_INTERMEDIATE = (
    PREFILL_PARTITION_REF_INTERMEDIATE // PREFILL_PARTITION_REF_TP
)
PREFILL_PARTITION_REF_ATTN_ELEMS = PREFILL_PARTITION_REF_HIDDEN
PREFILL_PARTITION_PROFILE_ENV = "JETFIRE_PREFILL_PARTITION_PROFILE"
PREFILL_CHUNK_ENV_KEYS = (
    "JETFIRE_EMBED_CHUNK_TOKENS",
    "JETFIRE_MLP_CHUNK_TOKENS",
    "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS",
    "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS",
    "JETFIRE_PREFILL_QKV_CHUNK_TOKENS",
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class RunSpec:
    backend: str
    concurrency: int
    num_prompts: int
    run_id: int


@dataclass(frozen=True)
class KVBudget:
    bytes: int | None
    source: str
    target_tokens: int
    block_size: int
    target_blocks: int
    base_blocks: int
    bf16_bytes_per_block: int
    inflate_ratio: float
    safety: float

    def as_row(self) -> dict[str, Any]:
        if self.source == "vllm":
            return {
                "kv_cache_memory_bytes": "",
                "kv_cache_budget_source": "vllm",
                "kv_cache_target_tokens": "",
                "kv_cache_budget_safety": "",
                "kv_cache_bf16_bytes_per_block": "",
                "kv_cache_inflate_ratio": "",
            }
        return {
            "kv_cache_memory_bytes": self.bytes,
            "kv_cache_budget_source": self.source,
            "kv_cache_target_tokens": self.target_tokens,
            "kv_cache_budget_safety": self.safety,
            "kv_cache_bf16_bytes_per_block": self.bf16_bytes_per_block,
            "kv_cache_inflate_ratio": self.inflate_ratio,
        }


@dataclass(frozen=True)
class PrefillPartitionProfile:
    name: str
    min_scratch_gib: float
    embed: int
    mlp: int
    attn_o_proj: int
    prefill_attn: int
    prefill_qkv: int

    def env_values(self) -> dict[str, int]:
        return {
            "JETFIRE_EMBED_CHUNK_TOKENS": self.embed,
            "JETFIRE_MLP_CHUNK_TOKENS": self.mlp,
            "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS": self.attn_o_proj,
            "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS": self.prefill_attn,
            "JETFIRE_PREFILL_QKV_CHUNK_TOKENS": self.prefill_qkv,
        }


PREFILL_PARTITION_PROFILES: tuple[PrefillPartitionProfile, ...] = (
    PrefillPartitionProfile("perf_12g", 12.0, 4096, 4096, 2048, 2048, 2048),
    PrefillPartitionProfile("relaxed_10g5", 10.5, 4096, 3072, 1536, 1536, 1536),
    PrefillPartitionProfile("balanced_9g", 9.0, 2048, 2048, 1024, 1024, 1024),
    PrefillPartitionProfile("balanced_7g5", 7.5, 2048, 1536, 768, 768, 768),
    PrefillPartitionProfile("tight_6g25", 6.25, 2048, 1024, 512, 512, 512),
    PrefillPartitionProfile("tight_5g5", 5.5, 1024, 512, 480, 480, 480),
    PrefillPartitionProfile("tight_5g25", 5.25, 1024, 512, 384, 384, 384),
    PrefillPartitionProfile("floor_0g", 0.0, 1024, 384, 256, 256, 256),
)


class MemoryMonitor:
    """Sample nvidia-smi while one benchmark run is active."""

    def __init__(self, gpus: str, interval_s: float) -> None:
        self.gpu_ids = {int(x.strip()) for x in str(gpus).split(",") if x.strip()}
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_mib = 0
        self.total_mib = 0
        self.samples = 0

    def start(self) -> None:
        if self.interval_s <= 0 or not self.gpu_ids:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self.samples <= 0:
            return {}
        return {
            "gpu_peak_memory_mib": int(self.peak_used_mib),
            "gpu_total_memory_mib": int(self.total_mib),
            "gpu_memory_samples": int(self.samples),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            return
        if completed.returncode != 0:
            return

        selected_used = []
        selected_total = []
        for raw_line in completed.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])
            except ValueError:
                continue
            if idx in self.gpu_ids:
                selected_used.append(used)
                selected_total.append(total)
        if not selected_used:
            return
        self.peak_used_mib = max(self.peak_used_mib, max(selected_used))
        self.total_mib = max(self.total_mib, max(selected_total))
        self.samples += 1


def _parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(x <= 0 for x in values):
        raise ValueError(f"all values must be positive: {raw}")
    return values


def _vllm_third_party_path() -> str | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "third_party"
    return str(path) if (path / "triton_kernels").is_dir() else None


def _same_path(path: str, target: Path) -> bool:
    try:
        return Path(path).resolve() == target.resolve()
    except Exception:
        return path == str(target)


def _pythonpath(
    *paths: str,
    include_existing: bool = False,
    drop_release_tree: bool = False,
) -> str:
    existing = []
    if include_existing:
        existing = [
            p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
        ]

    # Drop checkout-level paths that can expose a sibling development package
    # named ``jetfire_vllm`` instead of the release tree.
    blocked = [
        ROOT.parent,
        ROOT.parent / "jetfire_vllm" / "adapter",
    ]
    if drop_release_tree:
        blocked.extend([ROOT, ROOT / "jetfire_vllm" / "adapter"])
    existing = [
        p for p in existing
        if not any(_same_path(p, blocked_path) for blocked_path in blocked)
    ]

    merged: list[str] = []
    for path in [*paths, *existing]:
        if path and path not in merged:
            merged.append(path)
    return os.pathsep.join(merged)


def _assert_release_import_path() -> None:
    import jetfire_vllm

    pkg = Path(jetfire_vllm.__file__).resolve()
    if ROOT not in pkg.parents:
        raise RuntimeError(
            "Imported the wrong jetfire_vllm package: "
            f"{pkg}; expected it under {ROOT}."
        )


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + int(timeout_s)
    url = f"{base_url}/v1/models"
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _terminate(proc: subprocess.Popen, timeout_s: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=timeout_s)


def _base_env(args: argparse.Namespace, backend: str, concurrency: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["VLLM_DO_NOT_TRACK"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    third_party_paths = [_vllm_third_party_path()] if _vllm_third_party_path() else []
    include_existing_pythonpath = (
        os.environ.get("JETFIRE_ALLOW_INHERITED_PYTHONPATH", "0") == "1"
    )
    if backend != "jetfire":
        raise ValueError("This artifact supports only the JetFire backend")
    env["PYTHONPATH"] = _pythonpath(
        str(ROOT / "jetfire_vllm" / "adapter"), str(ROOT),
        *third_party_paths, include_existing=include_existing_pythonpath,
    )
    env["JETFIRE_EXPECTED_ROOT"] = str(ROOT)
    env.setdefault("JETFIRE_PRINT_IMPORT_PATH", "1")
    if args.torch_extensions_dir:
        env["TORCH_EXTENSIONS_DIR"] = args.torch_extensions_dir

    graph_profile = args.jetfire_profile in ("graph", "graph-mixed")
    mixed_profile = args.jetfire_profile in ("mixed", "graph-mixed")
    env["JETFIRE_USE_CUSTOM_BACKEND"] = "1"
    env["JETFIRE_MODEL_PATH"] = args.model
    env["JETFIRE_MODE"] = args.jetfire_mode
    env["JETFIRE_GROUP_SIZE"] = str(args.jetfire_group_size)
    env["JETFIRE_TOPK"] = str(args.top_k)
    env["JETFIRE_MAX_BATCH_SIZE"] = str(concurrency)
    env["JETFIRE_CPU_ATTN_THREADS"] = str(args.cpu_attn_threads)
    env["OMP_NUM_THREADS"] = str(args.cpu_attn_threads)
    env.setdefault("JETFIRE_CPU_CORRECTION", "1")
    env.setdefault("JETFIRE_DECODE_CPU_ATTN", "1")
    env.setdefault("JETFIRE_SKIP_NATIVE_KV", "1")
    env["JETFIRE_DECODE_GRAPH"] = "1" if graph_profile else "0"
    env["JETFIRE_ENABLE_VLLM_CUDAGRAPH"] = "1" if graph_profile else "0"
    env["JETFIRE_RELEASE_GRAPH_POLICY"] = "1" if graph_profile else "0"
    env["JETFIRE_RELEASE_REQUIRE_READY"] = (
        "1"
        if graph_profile and args.jetfire_release_require_ready == "on"
        else "0"
    )
    env["JETFIRE_DISABLE_ASYNC_SCHEDULING"] = "0" if mixed_profile else "1"
    env.setdefault("JETFIRE_AVOID_MIXED_PREFILL_DECODE", "0")
    prefill_sub_batch = _resolved_prefill_sub_batch_size(
        args, backend, concurrency
    )
    if 0 < prefill_sub_batch < concurrency:
        env["JETFIRE_PREFILL_FIRST_SCHEDULING"] = "1"
    env.setdefault("JETFIRE_SYNC_CPU_ATTN", "0")
    env.setdefault("JETFIRE_USE_PERSISTENT_WORKER", "0")
    env.setdefault("JETFIRE_FUSED_JOIN_BLEND", "0")
    env.setdefault("JETFIRE_CPU_FUTURE_TIMEOUT_S", "120")
    env.setdefault("JETFIRE_FAST_STORE_COMPLETE_CHECK", "1")
    if args.jetfire_int8tc == "on":
        env["JETFIRE_USE_INT8_TC"] = "1"
        # Keep Stage-B/SV on its internal policy.  For B=1 the validated
        # legacy INT4 Stage B is faster; the INT8TC SV path is only useful
        # as an explicit experiment or for batched auto-selection.
        env.pop("JETFIRE_USE_INT8_TC_SV", None)
    elif args.jetfire_int8tc == "off":
        env["JETFIRE_USE_INT8_TC"] = "0"
        env["JETFIRE_USE_INT8_TC_SV"] = "0"
    if args.jetfire_zero_init_cache == "on":
        env["JETFIRE_ZERO_INIT_CACHE"] = "1"
    elif args.jetfire_zero_init_cache == "off":
        env["JETFIRE_ZERO_INIT_CACHE"] = "0"
    elif graph_profile:
        # Graph capture needs stable buffers, not a full cache memset.  The
        # long-context memset is startup-only work and can look like a hang.
        env.setdefault("JETFIRE_ZERO_INIT_CACHE", "0")
    if graph_profile:
        env.setdefault("JETFIRE_PRELOAD_DECODE_EXTENSIONS", "1")
        env.setdefault(
            "JETFIRE_GRAPH_CAPTURE_SIZES",
            ",".join(
                str(i) for i in _jetfire_cudagraph_capture_sizes(args, concurrency)
            ),
        )
    if _should_enable_prefill_partition(args, concurrency):
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        partition_profile = _resolved_prefill_partition_profile(args, concurrency)
        if partition_profile.prefill_qkv > 0:
            env.setdefault("JETFIRE_EMBED_EMPTY_CACHE", "1")
            env[PREFILL_PARTITION_PROFILE_ENV] = partition_profile.name
            partition_shape = _prefill_partition_shape(args)
            for key, value in partition_shape.items():
                env[f"JETFIRE_PREFILL_PARTITION_{key.upper()}"] = str(value)
            for key, value in partition_profile.env_values().items():
                env[key] = str(value)
        elif args.jetfire_prefill_chunk_tokens is not None:
            env.pop(PREFILL_PARTITION_PROFILE_ENV, None)
            for key in _prefill_partition_shape(args):
                env.pop(f"JETFIRE_PREFILL_PARTITION_{key.upper()}", None)
            for key in PREFILL_CHUNK_ENV_KEYS:
                env.pop(key, None)
    return env


def _prefill_scratch_budget_gib(
    args: argparse.Namespace,
    concurrency: int | None = None,
) -> float | None:
    concurrency = max(1, int(concurrency or getattr(args, "concurrency", 1)))
    kv_budget = _kv_budget(args, "jetfire", concurrency)
    if kv_budget.bytes is None:
        return None
    total_gib = _gpu_total_gib(args.gpu) * float(args.gpu_memory_utilization)
    return total_gib - _model_weights_gib(args) - _gib(int(kv_budget.bytes))


def _prefill_partition_shape(args: argparse.Namespace) -> dict[str, int]:
    cfg = _load_model_config(args.model)
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    intermediate = int(cfg.get("intermediate_size") or 0)
    num_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads") or num_heads)
    head_dim = int(cfg.get("head_dim") or (hidden // num_heads if num_heads else 0))
    tp = max(1, int(args.tensor_parallel_size))
    local_hidden = max(1, ceil(hidden / tp)) if hidden > 0 else 1
    local_intermediate = (
        max(1, ceil(intermediate / tp)) if intermediate > 0 else local_hidden
    )
    local_kv = max(1, _ceil_div(num_kv_heads, tp) * head_dim) if head_dim > 0 else 1
    qkv_elems = local_hidden + 2 * local_kv
    return {
        "hidden": max(1, hidden),
        "intermediate": max(1, intermediate),
        "num_heads": max(1, num_heads),
        "num_kv_heads": max(1, num_kv_heads),
        "head_dim": max(1, head_dim),
        "tp": tp,
        "local_hidden": local_hidden,
        "local_intermediate": local_intermediate,
        "qkv_elems": qkv_elems,
        "attn_elems": max(local_hidden, qkv_elems, hidden),
    }


def _round_partition_tokens(value: float) -> int:
    if value <= 0:
        return 0
    rounded = int(value // 16) * 16
    return max(16, rounded)


def _parse_prefill_chunk_tokens(raw: object) -> int | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in ("", "auto"):
        return None
    if value in ("0", "off", "none", "false"):
        return 0
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--jetfire-prefill-chunk-tokens must be auto, off, or >= 0")
    return parsed


def _prefill_partition_model_scale(args: argparse.Namespace) -> float:
    shape = _prefill_partition_shape(args)
    hidden_scale = shape["hidden"] / float(PREFILL_PARTITION_REF_HIDDEN)
    intermediate_scale = shape["intermediate"] / float(
        PREFILL_PARTITION_REF_INTERMEDIATE
    )
    return max(0.25, hidden_scale, intermediate_scale)


def _scale_prefill_partition_profile(
    args: argparse.Namespace,
    profile: PrefillPartitionProfile,
) -> PrefillPartitionProfile:
    shape = _prefill_partition_shape(args)
    embed = _round_partition_tokens(
        profile.embed * PREFILL_PARTITION_REF_HIDDEN / shape["hidden"]
    )
    mlp = _round_partition_tokens(
        profile.mlp
        * PREFILL_PARTITION_REF_LOCAL_INTERMEDIATE
        / shape["local_intermediate"]
    )
    attn = _round_partition_tokens(
        profile.attn_o_proj * PREFILL_PARTITION_REF_ATTN_ELEMS / shape["attn_elems"]
    )
    prefill_attn = _round_partition_tokens(
        profile.prefill_attn
        * PREFILL_PARTITION_REF_ATTN_ELEMS
        / shape["attn_elems"]
    )
    prefill_qkv = _round_partition_tokens(
        profile.prefill_qkv
        * PREFILL_PARTITION_REF_ATTN_ELEMS
        / shape["attn_elems"]
    )
    return PrefillPartitionProfile(
        profile.name,
        profile.min_scratch_gib,
        embed,
        mlp,
        attn,
        prefill_attn,
        prefill_qkv,
    )


def _resolved_prefill_partition_profile(
    args: argparse.Namespace,
    concurrency: int | None = None,
) -> PrefillPartitionProfile:
    manual_chunk_tokens = _parse_prefill_chunk_tokens(
        getattr(args, "jetfire_prefill_chunk_tokens", None)
    )
    if manual_chunk_tokens is not None:
        value = int(manual_chunk_tokens)
        return PrefillPartitionProfile(
            "user_uniform",
            0.0,
            value,
            value,
            value,
            value,
            value,
        )

    concurrency = max(1, int(concurrency or getattr(args, "concurrency", 1)))
    activation_tokens = int(concurrency * _max_model_len(args))
    if activation_tokens < 120000:
        return PrefillPartitionProfile(
            "default",
            0.0,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
        )

    scratch_gib = _prefill_scratch_budget_gib(args, concurrency)
    if scratch_gib is None:
        return PrefillPartitionProfile(
            "default",
            0.0,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
            DEFAULT_PREFILL_CHUNK_TOKENS,
        )

    normalized_scratch_gib = scratch_gib / _prefill_partition_model_scale(args)
    for profile in PREFILL_PARTITION_PROFILES:
        if normalized_scratch_gib >= profile.min_scratch_gib:
            return _scale_prefill_partition_profile(args, profile)
    return _scale_prefill_partition_profile(args, PREFILL_PARTITION_PROFILES[-1])


def _resolved_prefill_chunk_tokens(
    args: argparse.Namespace,
    concurrency: int | None = None,
) -> int:
    return _resolved_prefill_partition_profile(args, concurrency).prefill_qkv


def _prefill_activation_tokens(args: argparse.Namespace, concurrency: int) -> int:
    """Estimate the full-prefill token count seen by vLLM profile/dummy runs.

    The long-context activation guard used to key only off single-request
    context length.  That misses cases like B=16, ctx=32k: each request is
    below 64k, but the non-chunked full-prefill profile run can materialize
    hundreds of thousands of tokens through MLP gate/up projection.
    """
    token_budget = _server_token_budget(args, "jetfire", concurrency)
    if token_budget is not None:
        return int(token_budget)
    return int(concurrency * _max_model_len(args))


def _jetfire_cudagraph_capture_sizes(
    args: argparse.Namespace, concurrency: int
) -> list[int]:
    raw = str(getattr(args, "jetfire_cudagraph_capture_sizes", "") or "").strip()
    if not raw:
        return list(range(1, int(concurrency) + 1))
    sizes: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--jetfire-cudagraph-capture-sizes entries must be > 0")
        sizes.append(value)
    if not sizes:
        raise ValueError("--jetfire-cudagraph-capture-sizes did not contain sizes")
    return sorted(set(sizes))


def _requested_max_model_len(args: argparse.Namespace) -> int:
    if args.max_model_len:
        return int(args.max_model_len)
    return int(args.ctx + args.output_len + args.max_model_len_padding)


def _model_position_limit(args: argparse.Namespace) -> int | None:
    cfg = _load_model_config(args.model)
    for key in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
        "model_max_length",
    ):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            limit = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < limit < 10**9:
            return limit
    return None


def _max_model_len(args: argparse.Namespace) -> int:
    requested = _requested_max_model_len(args)
    limit = _model_position_limit(args)
    if limit is None:
        return requested
    return min(requested, limit)


def _server_token_budget(
    args: argparse.Namespace,
    backend: str,
    concurrency: int | None = None,
) -> int | None:
    # The direct client can enforce JetFire prefill micro-batches by launching
    # requests in waves.  Do not use vLLM's max_num_batched_tokens for that
    # path: it enables chunked/extend prefill, while JetFire skip-native KV
    # only supports full one-shot prefill plus decode.
    if backend == "jetfire" and getattr(args, "client", "official") == "direct":
        return None
    if args.max_num_batched_tokens is None:
        if backend != "jetfire":
            return None
        effective_concurrency = int(concurrency if concurrency is not None else args.concurrency)
        sub_batch = _resolved_prefill_sub_batch_size(
            args, "jetfire", effective_concurrency
        )
        if sub_batch <= 0 or sub_batch >= effective_concurrency:
            return None
        return int(sub_batch * _max_model_len(args) + 4096)
    return int(args.max_num_batched_tokens)


def _load_model_config(model: str) -> dict[str, Any]:
    config_path = Path(model) / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, local_files_only=True)
        return dict(cfg.to_dict())
    except Exception as exc:
        raise RuntimeError(
            "Cannot resolve model config for automatic JetFire KV budget. "
            "Pass --kv-cache-memory-bytes explicitly or set "
            "--jetfire-kv-cache-budget vllm."
        ) from exc


def _gib(num_bytes: float) -> float:
    return float(num_bytes) / float(1024**3)


def _gpu_total_gib(gpu: str) -> float:
    first_gpu = str(gpu).split(",", 1)[0].strip()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={first_gpu}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        # Conservative fallback for H100/A100-class release machines.
        return 80.0


def _model_weights_gib(args: argparse.Namespace) -> float:
    model_path = Path(args.model)
    if model_path.exists():
        files = list(model_path.glob("*.safetensors"))
        if not files:
            files = list(model_path.glob("pytorch_model*.bin"))
        if files:
            return sum(p.stat().st_size for p in files) / float(1024**3) / max(
                1, int(args.tensor_parallel_size)
            )

    cfg = _load_model_config(args.model)
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    intermediate = int(cfg.get("intermediate_size") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    if min(hidden, layers, intermediate, vocab) <= 0:
        return 0.0
    # Rough dense decoder estimate: qkv/o + gate/up/down + two embeddings.
    params = layers * (4 * hidden * hidden + 3 * hidden * intermediate)
    params += 2 * vocab * hidden
    return _gib(params * 2) / max(1, int(args.tensor_parallel_size))


def _prefill_activation_gib_per_request(args: argparse.Namespace) -> float:
    cfg = _load_model_config(args.model)
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    num_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads") or num_heads)
    head_dim = int(cfg.get("head_dim") or (hidden // num_heads if num_heads else 0))
    intermediate = int(cfg.get("intermediate_size") or 0)
    tp = max(1, int(args.tensor_parallel_size))
    if min(hidden, num_heads, num_kv_heads, head_dim) <= 0:
        return 0.0

    local_hidden = ceil(hidden / tp)
    local_kv = _ceil_div(num_kv_heads, tp) * head_dim
    local_intermediate = ceil(intermediate / tp) if intermediate > 0 else local_hidden
    qkv_elems = local_hidden + 2 * local_kv
    attn_out_elems = local_hidden
    mlp_act_elems = local_intermediate
    elems_per_token = max(qkv_elems, attn_out_elems, mlp_act_elems)
    return _gib(_max_model_len(args) * elems_per_token * 2)


def _should_enable_prefill_partition(
    args: argparse.Namespace,
    concurrency: int,
) -> bool:
    if _parse_prefill_chunk_tokens(
        getattr(args, "jetfire_prefill_chunk_tokens", None)
    ) is not None:
        return True

    activation_tokens = _prefill_activation_tokens(args, concurrency)
    if int(args.ctx) >= 65536:
        return True
    if activation_tokens >= int(args.jetfire_prefill_chunk_threshold_tokens):
        return True

    activation_gib = _prefill_activation_gib_per_request(args) * max(
        1, int(concurrency)
    )
    threshold_gib = float(args.jetfire_prefill_chunk_activation_threshold_gib)
    scratch_gib = _prefill_scratch_budget_gib(args, concurrency)
    if scratch_gib is not None and scratch_gib > 0:
        threshold_gib = max(
            threshold_gib,
            scratch_gib * float(args.jetfire_prefill_chunk_scratch_fraction),
        )
    return activation_gib >= threshold_gib


def _explicit_prefill_sub_batch_size(args: argparse.Namespace) -> int | None:
    raw = str(getattr(args, "jetfire_prefill_sub_batch_size", "auto")).strip().lower()
    if raw in ("", "auto"):
        return None
    if raw in ("0", "off", "none", "false"):
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError("--jetfire-prefill-sub-batch-size must be auto, off, or >= 1")
    return value


def _resolved_prefill_sub_batch_size(
    args: argparse.Namespace,
    backend: str,
    concurrency: int,
) -> int:
    if backend != "jetfire":
        return 0
    explicit = _explicit_prefill_sub_batch_size(args)
    if explicit is not None:
        return min(int(explicit), int(concurrency))

    concurrency = int(concurrency)
    if concurrency <= 1:
        return 0

    kv_budget = _kv_budget(args, "jetfire", concurrency)
    kv_gib = _gib(kv_budget.bytes or 0)
    weights_gib = _model_weights_gib(args)
    total_gib = _gpu_total_gib(args.gpu)
    activation_per_req = _prefill_activation_gib_per_request(args)
    if activation_per_req <= 0:
        return 0

    activation_budget = (
        total_gib
        - weights_gib
        - kv_gib
        - float(args.jetfire_prefill_runtime_reserve_gib)
    )
    max_fit = int(activation_budget // (activation_per_req * args.jetfire_prefill_activation_safety))
    max_fit = max(1, min(concurrency, max_fit))
    if max_fit >= concurrency:
        selected = concurrency
        if os.environ.get("JETFIRE_DEBUG_SUB_BATCH", "0") == "1":
            print(
                "[jetfire-bench] auto prefill sub-batch: "
                f"selected={selected} concurrency={concurrency} full_batch=1 "
                f"total_gib={total_gib:.2f} weights_gib={weights_gib:.2f} "
                f"kv_gib={kv_gib:.2f} reserve_gib={args.jetfire_prefill_runtime_reserve_gib:.2f} "
                f"act_per_req_gib={activation_per_req:.2f} max_fit={max_fit}",
                flush=True,
            )
        return selected

    # Prefer stable power-of-two-like buckets. They map well to graph capture
    # sizes and avoid overfitting to a few hundred MiB of transient allocator
    # noise near the memory limit.
    candidates = [1, 2, 4, 8, 16, 32, 64]
    auto_cap = max(1, int(args.jetfire_prefill_auto_max_sub_batch_size))
    max_fit = min(max_fit, auto_cap)
    selected = max(c for c in candidates if c <= max_fit)
    selected = min(selected, concurrency)

    if os.environ.get("JETFIRE_DEBUG_SUB_BATCH", "0") == "1":
        print(
            "[jetfire-bench] auto prefill sub-batch: "
            f"selected={selected} concurrency={concurrency} "
            f"total_gib={total_gib:.2f} weights_gib={weights_gib:.2f} "
            f"kv_gib={kv_gib:.2f} reserve_gib={args.jetfire_prefill_runtime_reserve_gib:.2f} "
            f"act_per_req_gib={activation_per_req:.2f} max_fit={max_fit} "
            f"auto_cap={auto_cap}",
            flush=True,
        )
    return selected


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _jetfire_inflate_ratio(mode: str, group_size: int) -> float:
    from jetfire_vllm.adapter.config import QUANT_SCHEMES, compute_inflate_ratio

    scheme = QUANT_SCHEMES.get(mode)
    if scheme is None:
        raise ValueError(f"Unknown JetFire mode for KV budget: {mode!r}")
    return float(compute_inflate_ratio(scheme, int(group_size)))


def _bf16_kv_bytes_per_block(args: argparse.Namespace) -> int:
    cfg = _load_model_config(args.model)
    num_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    num_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads") or num_heads)
    hidden_size = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    head_dim = int(cfg.get("head_dim") or (hidden_size // num_heads if num_heads else 0))
    if min(num_layers, num_heads, num_kv_heads, head_dim) <= 0:
        raise RuntimeError(
            "Cannot infer BF16 KV page size from model config. "
            "Need num_hidden_layers, num_attention_heads, num_key_value_heads, "
            "and head_dim/hidden_size."
        )

    # vLLM's uniform KV planner first computes BF16 blocks from
    #   available_memory // per_layer_page_size // num_layers.
    # Therefore the budget passed to --kv-cache-memory-bytes is the BF16
    # per-GPU bytes for all layers at the *pre-inflation* block count.
    local_kv_heads = max(1, _ceil_div(num_kv_heads, int(args.tensor_parallel_size)))
    bf16_bytes = 2
    kv_tensors = 2  # K and V
    return (
        int(args.kv_cache_block_size)
        * num_layers
        * kv_tensors
        * local_kv_heads
        * head_dim
        * bf16_bytes
    )


def _kv_budget(args: argparse.Namespace, backend: str, concurrency: int) -> KVBudget:
    if args.kv_cache_memory_bytes is not None:
        return KVBudget(
            bytes=int(args.kv_cache_memory_bytes),
            source="manual",
            target_tokens=int(concurrency * _max_model_len(args)),
            block_size=int(args.kv_cache_block_size),
            target_blocks=0,
            base_blocks=0,
            bf16_bytes_per_block=0,
            inflate_ratio=0.0,
            safety=0.0,
        )
    if backend != "jetfire" or args.jetfire_kv_cache_budget == "vllm":
        return KVBudget(
            bytes=None,
            source="vllm",
            target_tokens=0,
            block_size=int(args.kv_cache_block_size),
            target_blocks=0,
            base_blocks=0,
            bf16_bytes_per_block=0,
            inflate_ratio=0.0,
            safety=0.0,
        )

    block_size = int(args.kv_cache_block_size)
    ratio = _jetfire_inflate_ratio(args.jetfire_mode, args.jetfire_group_size)
    bf16_bytes_per_block = _bf16_kv_bytes_per_block(args)
    target_tokens = int(
        ceil(concurrency * _max_model_len(args) * args.jetfire_kv_budget_safety)
    )
    # vLLM validates max_model_len after integer block rounding.  Keep one
    # compressed block of headroom so short-smoke runs do not fail exactly at
    # the planner boundary (for example 4120 requested vs 4096 admitted).
    target_blocks = _ceil_div(target_tokens, block_size) + 1
    min_base_blocks = max(1, int(ceil(target_blocks / ratio)))

    if args.jetfire_kv_cache_budget == "reserve":
        usable_gib = (
            _gpu_total_gib(args.gpu) * float(args.gpu_memory_utilization)
            - _model_weights_gib(args)
            - float(args.jetfire_runtime_scratch_reserve_gib)
        )
        usable_bytes = max(0, int(usable_gib * 1024**3))
        reserve_base_blocks = max(1, usable_bytes // bf16_bytes_per_block)
        base_blocks = max(min_base_blocks, reserve_base_blocks)
        budget_tokens = int(base_blocks * ratio * block_size)
        return KVBudget(
            bytes=int(base_blocks * bf16_bytes_per_block),
            source="reserve",
            target_tokens=budget_tokens,
            block_size=block_size,
            target_blocks=target_blocks,
            base_blocks=base_blocks,
            bf16_bytes_per_block=bf16_bytes_per_block,
            inflate_ratio=ratio,
            safety=float(args.jetfire_kv_budget_safety),
        )

    base_blocks = min_base_blocks
    return KVBudget(
        bytes=int(base_blocks * bf16_bytes_per_block),
        source="auto",
        target_tokens=target_tokens,
        block_size=block_size,
        target_blocks=target_blocks,
        base_blocks=base_blocks,
        bf16_bytes_per_block=bf16_bytes_per_block,
        inflate_ratio=ratio,
        safety=float(args.jetfire_kv_budget_safety),
    )


def _server_cmd(
    args: argparse.Namespace,
    backend: str,
    concurrency: int,
    port: int,
) -> list[str]:
    if backend != "jetfire":
        raise ValueError("This artifact supports only the JetFire backend")
    max_model_len = _max_model_len(args)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-num-seqs",
        str(concurrency),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--disable-log-stats",
        "--no-enable-prefix-caching",
    ]
    if args.hf_overrides:
        cmd.extend(["--hf-overrides", args.hf_overrides])
    if args.hf_config_path:
        cmd.extend(["--hf-config-path", args.hf_config_path])
    token_budget = _server_token_budget(args, backend, concurrency)
    if token_budget is not None:
        cmd.extend(["--max-num-batched-tokens", str(token_budget)])
    kv_budget = _kv_budget(args, backend, concurrency)
    if kv_budget.bytes is not None:
        cmd.extend(["--kv-cache-memory-bytes", str(kv_budget.bytes)])
    if backend == "jetfire":
        cmd.extend(["--attention-backend", "CUSTOM"])
        graph_profile = args.jetfire_profile in ("graph", "graph-mixed")
        if graph_profile:
            cmd.append("--no-enforce-eager")
            capture_sizes = _jetfire_cudagraph_capture_sizes(args, concurrency)
            graph_scope = args.jetfire_graph_scope
            if graph_scope == "auto":
                # Very long one-shot prefills can OOM inside inductor autotune
                # for the large prefill compile range.  Keep the release
                # performance path focused on decode CUDA graphs there.
                graph_scope = "decode-only" if int(args.ctx) >= 65536 else "full"
            if graph_scope == "decode-only":
                compilation_config = {
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": capture_sizes,
                    "max_cudagraph_capture_size": max(capture_sizes) if capture_sizes else 0,
                }
                if int(args.tensor_parallel_size) > 1:
                    cmd.append("--disable-custom-all-reduce")
                cmd.extend([
                    "--compilation-config",
                    json.dumps(compilation_config, separators=(",", ":")),
                ])
            else:
                cmd.extend([
                    "--max-cudagraph-capture-size",
                    str(max(capture_sizes) if capture_sizes else 0),
                ])
                cmd.append("--cudagraph-capture-sizes")
                cmd.extend(str(i) for i in capture_sizes)
        else:
            cmd.append("--enforce-eager")
        if args.jetfire_profile in ("mixed", "graph-mixed"):
            cmd.append("--async-scheduling")
        else:
            cmd.append("--no-async-scheduling")

    if backend == "jetfire" and args.chunked_prefill == "auto":
        # JetFire skip-native KV supports full one-shot prefill plus decode.
        # vLLM 0.19 may enable chunked prefill by default, which creates
        # cached-context extend-prefill rows that JetFire intentionally rejects.
        cmd.append("--no-enable-chunked-prefill")
    elif args.chunked_prefill == "on":
        cmd.append("--enable-chunked-prefill")
    elif args.chunked_prefill == "off":
        cmd.append("--no-enable-chunked-prefill")
    return cmd


def _bench_cmd(
    args: argparse.Namespace,
    base_url: str,
    spec: RunSpec,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "jetfire_vllm.bench.official_client",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.ctx),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(spec.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(spec.concurrency),
        "--ready-check-timeout-sec",
        str(min(int(args.server_timeout_s), 60)),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--seed",
        str(args.seed + spec.run_id),
    ]
    # Older official vLLM bench clients do not expose --num-warmups.
    # Keep warmup-free compatibility runs on those versions usable.
    if int(args.warmups) > 0:
        cmd[cmd.index("--request-rate"):cmd.index("--request-rate")] = [
            "--num-warmups",
            str(args.warmups),
        ]
    return cmd


def _direct_prompt(args: argparse.Namespace) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    candidate_ids = tokenizer(" the", add_special_tokens=False).input_ids
    token_id = int(candidate_ids[0] if candidate_ids else 100)
    # Do not decode to text here: tokenizer round-trips can change the exact
    # token count at model-window boundaries and turn a valid 128K test into an
    # artificial 400 Bad Request.
    return [token_id] * int(args.ctx)


def _run_direct_completion(
    *,
    base_url: str,
    model: str,
    prompt: list[int],
    output_len: int,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(output_len),
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=int(timeout_s)) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_line = line.split("data:", 1)[1].strip()
            if data_line == "[DONE]":
                break
            try:
                event = json.loads(data_line)
            except json.JSONDecodeError:
                continue
            if "error" in event:
                raise RuntimeError(f"stream error: {event['error']}")
            choices = event.get("choices") or []
            text = ""
            if choices:
                text = str(choices[0].get("text") or "")
            if text:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    if first is None:
        first = end
    return {
        "ttft": float(first - start),
        "e2el": float(end - start),
        "output_chunks": int(chunks),
    }


def _run_direct_benchmark(
    args: argparse.Namespace,
    base_url: str,
    spec: RunSpec,
    result_path: Path,
) -> dict[str, Any]:
    prompt = _direct_prompt(args)
    client_concurrency = int(spec.concurrency)
    if spec.backend == "jetfire":
        sub_batch = _resolved_prefill_sub_batch_size(
            args, spec.backend, spec.concurrency
        )
        if 0 < sub_batch < client_concurrency:
            client_concurrency = int(sub_batch)
    # Warmups are intentionally serial. They exercise graph capture / first
    # decode without perturbing measured request concurrency.
    for _ in range(max(0, int(args.warmups))):
        _run_direct_completion(
            base_url=base_url,
            model=args.served_model_name,
            prompt=prompt,
            output_len=args.output_len,
            timeout_s=args.direct_request_timeout_s,
        )

    results: list[dict[str, Any] | None] = [None] * int(spec.num_prompts)
    errors: list[str] = []

    def worker(index: int) -> None:
        try:
            results[index] = _run_direct_completion(
                base_url=base_url,
                model=args.served_model_name,
                prompt=prompt,
                output_len=args.output_len,
                timeout_s=args.direct_request_timeout_s,
            )
        except Exception as exc:
            errors.append(f"request {index}: {exc}")

    start = time.perf_counter()
    deadline = start + float(args.bench_timeout_s)
    active: list[threading.Thread] = []
    next_index = 0
    timed_out = False
    while next_index < int(spec.num_prompts) or active:
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        while (
            next_index < int(spec.num_prompts)
            and len(active) < client_concurrency
            and time.perf_counter() < deadline
        ):
            thread = threading.Thread(target=worker, args=(next_index,), daemon=True)
            thread.start()
            active.append(thread)
            next_index += 1
        active = [thread for thread in active if thread.is_alive()]
        time.sleep(0.01)
    duration = time.perf_counter() - start
    if timed_out:
        errors.append(
            f"direct client timed out after {float(args.bench_timeout_s):.1f}s "
            f"with {sum(1 for r in results if r is not None)}/{spec.num_prompts} completed"
        )

    completed_results = [r for r in results if r is not None]
    ttfts = [float(r["ttft"]) for r in completed_results]
    e2els = [float(r["e2el"]) for r in completed_results]
    output_chunks = [int(r["output_chunks"]) for r in completed_results]
    per_request_tpots = [
        max(0.0, (e2el - ttft) / max(1, chunks - 1))
        for ttft, e2el, chunks in zip(ttfts, e2els, output_chunks)
        if chunks > 1
    ]
    flattened_itls = [
        tpot
        for tpot, chunks in zip(per_request_tpots, output_chunks)
        for _ in range(max(0, chunks - 1))
    ]
    total_output_tokens = sum(output_chunks)
    total_input_tokens = int(args.ctx) * len(completed_results)
    result = {
        "date": time.strftime("%Y%m%d-%H%M%S"),
        "backend": "direct",
        "num_prompts": int(spec.num_prompts),
        "max_concurrency": int(spec.concurrency),
        "duration": duration,
        "completed": len(completed_results),
        "failed": len(errors),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": len(completed_results) / duration if duration else 0.0,
        "output_throughput": total_output_tokens / duration if duration else 0.0,
        "total_token_throughput": (
            (total_input_tokens + total_output_tokens) / duration if duration else 0.0
        ),
        "ttfts": ttfts,
        "itls": [
            [tpot] * max(0, chunks - 1)
            for tpot, chunks in zip(per_request_tpots, output_chunks)
        ],
        "mean_ttft_ms": (sum(ttfts) / len(ttfts) * 1000.0) if ttfts else None,
        "median_ttft_ms": sorted(ttfts)[len(ttfts) // 2] * 1000.0 if ttfts else None,
        "p90_ttft_ms": max(ttfts) * 1000.0 if ttfts else None,
        "p99_ttft_ms": max(ttfts) * 1000.0 if ttfts else None,
        "mean_tpot_ms": (
            sum(flattened_itls) / len(flattened_itls) * 1000.0
            if flattened_itls else None
        ),
        "median_tpot_ms": (
            sorted(flattened_itls)[len(flattened_itls) // 2] * 1000.0
            if flattened_itls else None
        ),
        "p90_tpot_ms": max(flattened_itls) * 1000.0 if flattened_itls else None,
        "p99_tpot_ms": max(flattened_itls) * 1000.0 if flattened_itls else None,
        "mean_e2el_ms": (sum(e2els) / len(e2els) * 1000.0) if e2els else None,
        "median_e2el_ms": sorted(e2els)[len(e2els) // 2] * 1000.0 if e2els else None,
        "errors": errors,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _parse_gib_from_line(line: str, marker: str) -> float | None:
    if marker not in line:
        return None
    tail = line.split(marker, 1)[1].strip()
    parts = tail.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _server_facts(server_log: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    if not server_log.exists():
        return facts
    for line in server_log.read_text(errors="ignore").splitlines():
        if "GPU KV cache size:" in line:
            value = line.split("GPU KV cache size:", 1)[1].split("tokens", 1)[0]
            facts["gpu_kv_cache_tokens"] = int(value.strip().replace(",", ""))
        elif "Maximum concurrency for" in line:
            facts["vllm_reported_max_concurrency"] = line.rsplit(":", 1)[-1].strip()
        elif "Available KV cache memory:" in line:
            facts["available_kv_cache_memory"] = line.rsplit(":", 1)[-1].strip()
        elif "Model loading took" in line and "GiB memory" in line:
            value = _parse_gib_from_line(line, "Model loading took")
            if value is not None:
                facts["model_weights_gib"] = value
        elif "reserved" in line and "GiB memory for KV Cache" in line:
            value = _parse_gib_from_line(line, "reserved")
            if value is not None:
                facts["kv_cache_reserved_gib"] = value
    return facts


def _add_memory_breakdown(row: dict[str, Any], memory_stats: dict[str, Any]) -> None:
    row.update(memory_stats)
    peak_mib = row.get("gpu_peak_memory_mib")
    total_mib = row.get("gpu_total_memory_mib")
    if peak_mib:
        row["gpu_peak_memory_gib"] = float(peak_mib) / 1024.0
    if total_mib:
        row["gpu_total_memory_gib"] = float(total_mib) / 1024.0

    peak_gib = row.get("gpu_peak_memory_gib")
    weights_gib = row.get("model_weights_gib")
    kv_gib = row.get("kv_cache_reserved_gib")
    if peak_gib is None or weights_gib is None or kv_gib is None:
        return

    residual = max(0.0, float(peak_gib) - float(weights_gib) - float(kv_gib))
    row["residual_peak_gib"] = residual
    if peak_gib:
        row["model_weights_peak_pct"] = float(weights_gib) / float(peak_gib) * 100.0
        row["kv_cache_peak_pct"] = float(kv_gib) / float(peak_gib) * 100.0
        row["residual_peak_pct"] = residual / float(peak_gib) * 100.0


def _compact_result(
    *,
    args: argparse.Namespace,
    spec: RunSpec,
    status: str,
    result: dict[str, Any] | None,
    result_path: Path | None,
    server_log: Path,
    bench_log: Path,
    memory_stats: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "backend": spec.backend,
        "mode": args.mode,
        "jetfire_profile": args.jetfire_profile if spec.backend == "jetfire" else "",
        "jetfire_mode": args.jetfire_mode if spec.backend == "jetfire" else "",
        "jetfire_group_size": (
            int(args.jetfire_group_size) if spec.backend == "jetfire" else ""
        ),
        "top_k": int(args.top_k) if spec.backend == "jetfire" else "",
        "ctx": int(args.ctx),
        "output_len": int(args.output_len),
        "num_prompts": int(spec.num_prompts),
        "max_concurrency": int(spec.concurrency),
        "run": int(spec.run_id),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "requested_max_model_len": _requested_max_model_len(args),
        "max_model_len": _max_model_len(args),
        "model_position_limit": _model_position_limit(args) or "",
        "max_num_batched_tokens": _server_token_budget(
            args, spec.backend, spec.concurrency
        ) or "",
        "jetfire_prefill_sub_batch_size": (
            _resolved_prefill_sub_batch_size(args, spec.backend, spec.concurrency)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_sub_batch_policy": (
            str(args.jetfire_prefill_sub_batch_size)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_runtime_reserve_gib": (
            float(args.jetfire_prefill_runtime_reserve_gib)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_activation_safety": (
            float(args.jetfire_prefill_activation_safety)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_auto_max_sub_batch_size": (
            int(args.jetfire_prefill_auto_max_sub_batch_size)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_partition_profile": (
            _resolved_prefill_partition_profile(args, spec.concurrency).name
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_scratch_budget_gib": (
            _prefill_scratch_budget_gib(args, spec.concurrency)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_normalized_scratch_gib": (
            (
                _prefill_scratch_budget_gib(args, spec.concurrency)
                / _prefill_partition_model_scale(args)
            )
            if spec.backend == "jetfire"
            and _prefill_scratch_budget_gib(args, spec.concurrency) is not None
            else ""
        ),
        "jetfire_prefill_partition_model_scale": (
            _prefill_partition_model_scale(args) if spec.backend == "jetfire" else ""
        ),
        "jetfire_prefill_partition_hidden": (
            _prefill_partition_shape(args)["hidden"] if spec.backend == "jetfire" else ""
        ),
        "jetfire_prefill_partition_intermediate": (
            _prefill_partition_shape(args)["intermediate"]
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_chunk_tokens": (
            _resolved_prefill_chunk_tokens(args, spec.concurrency)
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_embed_chunk_tokens": (
            _resolved_prefill_partition_profile(args, spec.concurrency).embed
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_mlp_chunk_tokens": (
            _resolved_prefill_partition_profile(args, spec.concurrency).mlp
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_attn_o_proj_chunk_tokens": (
            _resolved_prefill_partition_profile(args, spec.concurrency).attn_o_proj
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_attn_chunk_tokens": (
            _resolved_prefill_partition_profile(args, spec.concurrency).prefill_attn
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_qkv_chunk_tokens": (
            _resolved_prefill_partition_profile(args, spec.concurrency).prefill_qkv
            if spec.backend == "jetfire"
            else ""
        ),
        "jetfire_prefill_chunk_policy": (
            "auto" if args.jetfire_prefill_chunk_tokens is None else "user"
        )
        if spec.backend == "jetfire"
        else "",
        "jetfire_prefill_chunk_threshold_tokens": (
            int(args.jetfire_prefill_chunk_threshold_tokens)
            if spec.backend == "jetfire"
            else ""
        ),
        "server_log": str(server_log),
        "bench_log": str(bench_log),
        "result_path": str(result_path) if result_path else "",
        "error": error,
    }
    row.update(_release_plan_row(args, spec))
    row.update(_kv_budget(args, spec.backend, spec.concurrency).as_row())
    row.update(_server_facts(server_log))
    _add_memory_breakdown(row, memory_stats or {})
    if result:
        ttfts = result.get("ttfts")
        if isinstance(ttfts, list) and ttfts:
            # vLLM stores per-request TTFT values in seconds.  Release TTFT
            # is batch-level: all benchmark requests have reached first token.
            row["batch_ttft_ms"] = max(float(x) for x in ttfts) * 1000.0
        itls = result.get("itls")
        if isinstance(ttfts, list) and isinstance(itls, list) and len(ttfts) == len(itls):
            e2els = []
            for ttft, req_itls in zip(ttfts, itls):
                if isinstance(req_itls, list):
                    e2els.append(float(ttft) + sum(float(x) for x in req_itls))
                else:
                    e2els.append(float(ttft))
            if e2els:
                # Batch-level E2E latency: every benchmark request completed.
                if result.get("backend") == "direct" and result.get("duration") is not None:
                    # The direct client may intentionally launch JetFire
                    # requests in prefill micro-batch waves.  Per-request E2E
                    # is measured from each wave's own start; the batch
                    # makespan is the full client wall-clock duration.
                    row["batch_e2el_ms"] = float(result["duration"]) * 1000.0
                else:
                    row["batch_e2el_ms"] = max(e2els) * 1000.0
        for key in [
            "completed",
            "failed",
            "total_input_tokens",
            "total_output_tokens",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p90_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p90_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "mean_e2el_ms",
            "median_e2el_ms",
        ]:
            row[key] = result.get(key)
    return row


def _release_plan_row(args: argparse.Namespace, spec: RunSpec) -> dict[str, Any]:
    if (
        spec.backend != "jetfire"
        or args.jetfire_profile not in ("graph", "graph-mixed")
    ):
        return {
            "jetfire_release_stage_a": "",
            "jetfire_release_stage_b": "",
            "jetfire_release_ready": "",
            "jetfire_release_promotion_state": "",
            "jetfire_release_notes": "",
        }
    try:
        from jetfire_vllm.adapter.release_policy import plan_release_graph

        plan = plan_release_graph(
            mode=args.jetfire_mode,
            group_size=int(args.jetfire_group_size),
            top_k=int(args.top_k),
            max_graph_batch=int(spec.concurrency),
            require_ready=False,
        )
    except Exception as exc:
        return {
            "jetfire_release_stage_a": "",
            "jetfire_release_stage_b": "",
            "jetfire_release_ready": "",
            "jetfire_release_promotion_state": "",
            "jetfire_release_notes": f"plan_error={exc}",
        }
    return {
        "jetfire_release_stage_a": plan.stage_a,
        "jetfire_release_stage_b": plan.stage_b,
        "jetfire_release_ready": bool(plan.release_ready),
        "jetfire_release_promotion_state": plan.promotion_state,
        "jetfire_release_notes": "; ".join(plan.notes),
    }


def _tail_text(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    return text[-max_chars:]


def _looks_like_cuda_oom(text: str) -> bool:
    lower = text.lower()
    return (
        "cuda out of memory" in lower
        or "torch.outofmemoryerror" in lower
        or "out of memory" in lower and "cuda" in lower
    )


def _next_prefill_chunk_suggestion(current: int) -> int:
    candidates = [
        2048,
        1024,
        512,
        496,
        480,
        448,
        384,
        256,
        192,
        160,
        128,
        96,
        80,
        64,
    ]
    for candidate in candidates:
        if candidate < current:
            return candidate
    return max(16, current // 2)


def _augment_failure_error(
    args: argparse.Namespace,
    spec: RunSpec,
    status: str,
    error: str,
    server_log: Path,
    bench_log: Path,
) -> str:
    if status == "ok" or spec.backend != "jetfire":
        return error
    text = "\n".join([error, _tail_text(server_log), _tail_text(bench_log)])
    if not _looks_like_cuda_oom(text):
        return error

    chunk_tokens = _resolved_prefill_chunk_tokens(args)
    if chunk_tokens <= 0:
        return (
            f"{error}; detected CUDA OOM during JetFire prefill. "
            "Activation chunking is disabled; retry with "
            f"--jetfire-prefill-chunk-tokens {DEFAULT_PREFILL_CHUNK_TOKENS}, "
            "then lower it if OOM persists."
        )
    suggested = _next_prefill_chunk_suggestion(chunk_tokens)
    return (
        f"{error}; detected CUDA OOM during JetFire prefill. "
        f"Current --jetfire-prefill-chunk-tokens={chunk_tokens}. "
        f"Retry with --jetfire-prefill-chunk-tokens {suggested}; smaller "
        "chunks reduce activation peak but slow TTFT. For long-context "
        "capacity runs, sweep 496,480,448,384,256,192,128,96,64 and keep "
        "the largest value that does not OOM."
    )


def _run_one(args: argparse.Namespace, spec: RunSpec) -> dict[str, Any]:
    run_dir = args.out_dir / spec.backend / f"conc_{spec.concurrency}" / f"run_{spec.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = int(args.port or _free_port(args.host))
    base_url = f"http://{args.host}:{port}"
    env = _base_env(args, spec.backend, spec.concurrency)
    server_log = run_dir / "server.log"
    bench_log = run_dir / "bench.log"
    result_path = run_dir / "official_result.json"
    server_cmd = _server_cmd(args, spec.backend, spec.concurrency, port)
    monitor = MemoryMonitor(args.gpu, args.memory_sample_interval_s)

    def finish(**kwargs) -> dict[str, Any]:
        kwargs["error"] = _augment_failure_error(
            args=args,
            spec=spec,
            status=str(kwargs.get("status", "")),
            error=str(kwargs.get("error", "")),
            server_log=server_log,
            bench_log=bench_log,
        )
        return _compact_result(memory_stats=monitor.stop(), **kwargs)

    print(
        f"[bench] start backend={spec.backend} concurrency={spec.concurrency} "
        f"prompts={spec.num_prompts} run={spec.run_id}",
        flush=True,
    )
    with server_log.open("w") as sf:
        proc = subprocess.Popen(
            server_cmd,
            cwd=str(ROOT),
            env=env,
            stdout=sf,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
    monitor.start()

    try:
        try:
            _wait_ready(base_url, proc, args.server_timeout_s)
            if args.client == "official":
                cmd = _bench_cmd(args, base_url, spec, run_dir, result_path.name)
                with bench_log.open("w") as bf:
                    bf.write(f"{' '.join(cmd)}\n")
                    try:
                        completed = subprocess.run(
                            cmd,
                            cwd=str(run_dir),
                            env=env,
                            stdout=bf,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                            timeout=float(args.bench_timeout_s),
                        )
                    except subprocess.TimeoutExpired:
                        return finish(
                            args=args,
                            spec=spec,
                            status="bench_timeout",
                            result=None,
                            result_path=result_path if result_path.exists() else None,
                            server_log=server_log,
                            bench_log=bench_log,
                            error=(
                                "vllm bench serve timed out after "
                                f"{float(args.bench_timeout_s):.1f}s"
                            ),
                        )
                if completed.returncode != 0:
                    return finish(
                        args=args,
                        spec=spec,
                        status="bench_failed",
                        result=None,
                        result_path=result_path if result_path.exists() else None,
                        server_log=server_log,
                        bench_log=bench_log,
                        error=f"vllm bench serve exited {completed.returncode}",
                    )
                result = _load_json(result_path)
            else:
                with bench_log.open("w") as bf:
                    bf.write(
                        "direct streaming client "
                        f"num_prompts={spec.num_prompts} "
                        f"concurrency={spec.concurrency}\n"
                    )
                result = _run_direct_benchmark(args, base_url, spec, result_path)
            failed = int(result.get("failed", 0) or 0)
            completed_reqs = int(result.get("completed", 0) or 0)
            total_output_tokens = int(result.get("total_output_tokens", 0) or 0)
            expected_output_tokens = int(spec.num_prompts) * int(args.output_len)
            output_complete = (
                total_output_tokens >= expected_output_tokens
                if args.client == "official"
                else (
                    completed_reqs == spec.num_prompts
                    and total_output_tokens >= expected_output_tokens
                )
            )
            status = (
                "ok"
                if completed_reqs == spec.num_prompts and failed == 0 and output_complete
                else "request_failed"
            )
            error = ""
            if status != "ok":
                error = (
                    f"completed={completed_reqs} failed={failed} "
                    f"output_tokens={total_output_tokens}/{expected_output_tokens}"
                )
            return finish(
                args=args,
                spec=spec,
                status=status,
                result=result,
                result_path=result_path,
                server_log=server_log,
                bench_log=bench_log,
                error=error,
            )
        except Exception as exc:
            return finish(
                args=args,
                spec=spec,
                status="server_failed",
                result=None,
                result_path=None,
                server_log=server_log,
                bench_log=bench_log,
                error=str(exc),
            )
    finally:
        _terminate(proc)


def _build_specs(args: argparse.Namespace) -> list[RunSpec]:
    backends = [args.backend]
    specs: list[RunSpec] = []
    if args.mode == "fixed-concurrency":
        for backend in backends:
            for run in range(args.runs):
                specs.append(RunSpec(backend, args.concurrency, args.concurrency, run))
    elif args.mode == "fixed-workload":
        for backend in backends:
            for run in range(args.runs):
                specs.append(RunSpec(backend, args.concurrency, args.num_prompts, run))
    elif args.mode == "capacity-sweep":
        for backend in backends:
            for concurrency in _parse_int_list(args.concurrency_list):
                for run in range(args.runs):
                    specs.append(RunSpec(backend, concurrency, concurrency, run))
    else:
        raise ValueError(f"unknown mode: {args.mode}")
    return specs


def _preflight_model_length(args: argparse.Namespace) -> None:
    limit = _model_position_limit(args)
    if limit is None:
        return

    request_len = int(args.ctx) + int(args.output_len)
    if request_len > limit:
        raise SystemExit(
            "[model-length][error] ctx + output_len exceeds model position limit: "
            f"{request_len} > {limit}. Reduce --ctx or --output-len."
        )

    requested = _requested_max_model_len(args)
    if requested > limit:
        print(
            "[model-length] capping max_model_len to model position limit: "
            f"requested={requested} limit={limit}",
            file=sys.stderr,
            flush=True,
        )


def _preflight_release_policy(args: argparse.Namespace, specs: list[RunSpec]) -> None:
    if args.backend != "jetfire":
        return
    if args.jetfire_profile not in ("graph", "graph-mixed"):
        return
    if args.jetfire_int8tc != "auto":
        raise SystemExit(
            "[release-policy][error] --jetfire-int8tc must be auto for graph "
            "release runs; Stage-A kernel selection is owned by release_policy.py"
        )

    from jetfire_vllm.adapter.release_policy import (
        ReleasePolicyError,
        plan_release_graph,
    )

    for spec in specs:
        if spec.backend != "jetfire":
            continue
        try:
            plan = plan_release_graph(
                mode=args.jetfire_mode,
                group_size=int(args.jetfire_group_size),
                top_k=int(args.top_k),
                max_graph_batch=int(spec.concurrency),
                require_ready=args.jetfire_release_require_ready == "on",
            )
        except ReleasePolicyError as exc:
            raise SystemExit(f"[release-policy][error] {exc}") from exc
        print(
            "[release-policy] "
            f"mode={plan.mode} g={plan.group_size} top_k={plan.top_k} "
            f"A={plan.stage_a} B={plan.stage_b} state={plan.promotion_state} "
            f"ready={plan.release_ready} sync={plan.correction_sync}",
            flush=True,
        )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "status",
        "backend",
        "mode",
        "jetfire_profile",
        "jetfire_mode",
        "ctx",
        "output_len",
        "num_prompts",
        "max_concurrency",
        "run",
        "requested_max_model_len",
        "max_model_len",
        "model_position_limit",
        "completed",
        "failed",
        "available_kv_cache_memory",
        "gpu_kv_cache_tokens",
        "vllm_reported_max_concurrency",
        "model_weights_gib",
        "kv_cache_reserved_gib",
        "gpu_peak_memory_gib",
        "gpu_total_memory_gib",
        "residual_peak_gib",
        "model_weights_peak_pct",
        "kv_cache_peak_pct",
        "residual_peak_pct",
        "kv_cache_memory_bytes",
        "kv_cache_budget_source",
        "kv_cache_target_tokens",
        "kv_cache_budget_safety",
        "kv_cache_bf16_bytes_per_block",
        "kv_cache_inflate_ratio",
        "max_num_batched_tokens",
        "jetfire_prefill_partition_profile",
        "jetfire_prefill_scratch_budget_gib",
        "jetfire_prefill_normalized_scratch_gib",
        "jetfire_prefill_partition_model_scale",
        "jetfire_prefill_partition_hidden",
        "jetfire_prefill_partition_intermediate",
        "jetfire_prefill_chunk_tokens",
        "jetfire_embed_chunk_tokens",
        "jetfire_mlp_chunk_tokens",
        "jetfire_attn_o_proj_chunk_tokens",
        "jetfire_prefill_attn_chunk_tokens",
        "jetfire_prefill_qkv_chunk_tokens",
        "jetfire_prefill_chunk_policy",
        "jetfire_prefill_chunk_threshold_tokens",
        "jetfire_release_stage_a",
        "jetfire_release_stage_b",
        "jetfire_release_ready",
        "jetfire_release_promotion_state",
        "jetfire_release_notes",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "batch_ttft_ms",
        "batch_e2el_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "median_itl_ms",
        "median_e2el_ms",
        "server_log",
        "bench_log",
        "result_path",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _best_capacity(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        backend = str(row["backend"])
        out[backend] = max(out.get(backend, 0), int(row["max_concurrency"]))
    return out


def _write_report(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> None:
    best = _best_capacity(rows)
    partition_profile = _resolved_prefill_partition_profile(args, args.concurrency)
    lines = [
        "# JetFire Official Serving Benchmark",
        "",
        f"- mode: `{args.mode}`",
        f"- ctx: `{args.ctx}`",
        f"- output_len: `{args.output_len}`",
        f"- request_rate: `{args.request_rate}`",
        f"- prefix_cache: `off`",
        f"- chunked_prefill: `{args.chunked_prefill}`",
        f"- JetFire prefill partition profile: `{partition_profile.name}` "
        f"({'auto' if args.jetfire_prefill_chunk_tokens is None else 'user'}); "
        f"embed={partition_profile.embed}, mlp={partition_profile.mlp}, "
        f"attn_o_proj={partition_profile.attn_o_proj}, "
        f"prefill_attn={partition_profile.prefill_attn}, "
        f"prefill_qkv={partition_profile.prefill_qkv}.",
        f"- JetFire profile: `{args.jetfire_profile}`",
        f"- JetFire release policy: `{'on' if args.jetfire_profile in ('graph', 'graph-mixed') else 'off'}`",
        f"- JetFire KV budget: `{args.jetfire_kv_cache_budget}` "
        f"(safety `{args.jetfire_kv_budget_safety}`); explicit "
        "`--kv-cache-memory-bytes` overrides this.",
        "- `auto` is the release default: it computes just-enough KV cache "
        "from max_model_len, mode, group size, TP, and block size, leaving "
        "the rest of HBM for prefill/runtime/CUDA graph scratch. `vllm` is "
        "only for capacity/profiling sweeps that intentionally let vLLM use "
        "available HBM for KV.",
        "- Latency definition: `batch_ttft_ms = max(ttfts)` and "
        "`batch_e2el_ms = max(ttft + sum(itls))`; vLLM percentile metrics are "
        "kept as per-request references.",
        "",
        "## Results",
        "",
        "| backend | jetfire mode | status | submitted concurrency | max resident concurrency | prompts | peak GiB | weights % | KV % | residual % | KV tokens | batch TTFT ms | batch E2E ms | median E2E ms | median TTFT ms | p99 TTFT ms | req/s | tok/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {jetfire_mode} | {status} | {conc} | {resident_conc} | {prompts} | {peak} | {weights_pct} | {kv_pct} | {resid_pct} | {kv_tokens} | {batch_ttft} | {batch_e2e} | {e2e} | {ttft} | {p99_ttft} | {rps} | {tps} |".format(
                backend=row.get("backend", ""),
                jetfire_mode=row.get("jetfire_mode", ""),
                status=row.get("status", ""),
                conc=row.get("max_concurrency", ""),
                resident_conc=row.get("vllm_reported_max_concurrency", ""),
                prompts=row.get("num_prompts", ""),
                peak=_fmt(row.get("gpu_peak_memory_gib")),
                weights_pct=_fmt(row.get("model_weights_peak_pct")),
                kv_pct=_fmt(row.get("kv_cache_peak_pct")),
                resid_pct=_fmt(row.get("residual_peak_pct")),
                kv_tokens=row.get("gpu_kv_cache_tokens", ""),
                batch_ttft=_fmt(row.get("batch_ttft_ms")),
                batch_e2e=_fmt(row.get("batch_e2el_ms")),
                ttft=_fmt(row.get("median_ttft_ms")),
                p99_ttft=_fmt(row.get("p99_ttft_ms")),
                e2e=_fmt(row.get("median_e2el_ms")),
                rps=_fmt(row.get("request_throughput")),
                tps=_fmt(row.get("total_token_throughput")),
            )
        )
    if best:
        lines.extend(["", "## Capacity Summary", ""])
        for backend, concurrency in sorted(best.items()):
            lines.append(f"- `{backend}` max successful concurrency: `{concurrency}`")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _all_runs_complete(rows: list[dict[str, Any]], output_len: int) -> bool:
    return bool(rows) and all(
        row.get("status") == "ok"
        and int(row.get("completed") or 0) == int(row.get("num_prompts") or 0)
        and int(row.get("failed") or 0) == 0
        and int(row.get("total_output_tokens") or 0) >= int(row.get("num_prompts") or 0) * output_len
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixed-concurrency", "fixed-workload", "capacity-sweep"),
        required=True,
    )
    parser.add_argument("--backend", choices=("jetfire",), default="jetfire")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--served-model-name", default="jetfire-bench-model")
    parser.add_argument(
        "--hf-overrides",
        default="",
        help=(
            "JSON/dict string forwarded to vLLM --hf-overrides. Useful for "
            "model-config compatibility overrides such as disabling "
            "dual_chunk_attention_config."
        ),
    )
    parser.add_argument(
        "--hf-config-path",
        default="",
        help="Optional path forwarded to vLLM --hf-config-path.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--concurrency-list", default="1,2,4,8,16")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--client",
        choices=("official", "direct"),
        default="official",
        help=(
            "official runs vLLM's benchmark client. direct is a debug fallback "
            "that posts streaming /v1/completions requests itself to separate "
            "engine hangs from vLLM benchmark-client hangs."
        ),
    )
    parser.add_argument(
        "--direct-request-timeout-s",
        type=int,
        default=1800,
        help="Per-request timeout for --client direct.",
    )
    parser.add_argument(
        "--bench-timeout-s",
        type=float,
        default=1800.0,
        help="Wall-clock timeout for benchmark client execution.",
    )
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument(
        "--max-model-len-padding",
        type=int,
        default=16,
        help=(
            "Extra sequence-length headroom added when --max-model-len is not "
            "set. Keep this small for long-context capacity tests; the request "
            "itself already reserves ctx + output_len tokens."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Optional vLLM scheduler token budget. By default this is omitted "
            "so vLLM keeps max_num_seqs and max_num_batched_tokens independent."
        ),
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help=(
            "Explicit vLLM KV cache memory budget per GPU. Overrides JetFire's "
            "automatic just-enough KV budget."
        ),
    )
    parser.add_argument(
        "--jetfire-kv-cache-budget",
        choices=("auto", "vllm", "reserve"),
        default="auto",
        help=(
            "For JetFire runs without --kv-cache-memory-bytes. auto is the "
            "release default: it computes a just-enough KV budget from "
            "submitted concurrency, max_model_len, model KV shape, TP, block "
            "size, and quantization ratio, leaving remaining HBM for runtime "
            "scratch. vllm is for capacity/profiling sweeps where vLLM should "
            "profile available HBM and allocate KV itself. reserve keeps a "
            "runtime scratch reserve and gives the remaining budget to KV "
            "cache."
        ),
    )
    parser.add_argument(
        "--jetfire-runtime-scratch-reserve-gib",
        type=float,
        default=1.0,
        help=(
            "Per-GPU HBM reserve used by --jetfire-kv-cache-budget reserve. "
            "The reserve is kept for prefill/runtime/CUDA graph scratch; the "
            "remaining budget is assigned to KV cache."
        ),
    )
    parser.add_argument(
        "--jetfire-kv-budget-safety",
        type=float,
        default=1.02,
        help="Safety multiplier on concurrency * max_model_len for auto KV budget.",
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=16,
        help="vLLM KV block size used by the auto KV-budget derivation.",
    )
    parser.add_argument(
        "--memory-sample-interval-s",
        type=float,
        default=0.5,
        help="nvidia-smi sampling interval for per-run peak memory; <=0 disables.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument(
        "--chunked-prefill",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto disables vLLM chunked prefill for JetFire one-shot prefill.",
    )
    parser.add_argument(
        "--jetfire-profile",
        choices=("eager", "graph", "mixed", "graph-mixed"),
        default="eager",
        help=(
            "JetFire execution profile. eager runs synchronous decode; graph "
            "enables vLLM CUDA graph plus JetFire graph-safe CPU/GPU submit; "
            "mixed enables vLLM async mixed prefill/decode scheduling; "
            "graph-mixed enables both."
        ),
    )
    parser.add_argument(
        "--jetfire-graph-scope",
        choices=("auto", "full", "decode-only"),
        default="auto",
        help=(
            "CUDA graph scope for JetFire graph profiles. auto uses full "
            "vLLM graph capture below 64K context and decode-only graph at "
            "long context to avoid prefill inductor autotune OOM."
        ),
    )
    parser.add_argument(
        "--jetfire-release-require-ready",
        choices=("on", "off"),
        default="on",
        help=(
            "Require release_policy.py to mark each graph policy cell ready. "
            "Keep on for release numbers. Use off only for candidate "
            "specialization validation before promotion."
        ),
    )
    parser.add_argument(
        "--jetfire-cudagraph-capture-sizes",
        default="",
        help=(
            "Comma-separated vLLM CUDA graph capture sizes for JetFire graph "
            "runs. Defaults to 1..concurrency. Use a smaller set for "
            "long-context latency runs that need more activation/runtime headroom."
        ),
    )
    parser.add_argument("--jetfire-mode", default="SymInt3K_SymInt2V")
    parser.add_argument("--jetfire-group-size", type=int, default=64)
    parser.add_argument(
        "--jetfire-int8tc",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "INT8 tensor-core decode path control for INT4 modes. "
            "auto keeps the release policy; on/off are debug overrides and "
            "are rejected by release graph policy."
        ),
    )
    parser.add_argument(
        "--jetfire-zero-init-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control startup cache memset. auto disables it for JetFire graph "
            "profiles to avoid long-context startup stalls."
        ),
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--cpu-attn-threads", type=int, default=32)
    parser.add_argument(
        "--jetfire-prefill-sub-batch-size",
        default="auto",
        help=(
            "JetFire-only prefill sub-batch size. auto estimates the largest "
            "stable full-request prefill batch from GPU memory, weight size, "
            "JetFire KV budget, and activation peak. Use off/0 to disable, or "
            "an integer to override."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-runtime-reserve-gib",
        type=float,
        default=25.0,
        help=(
            "Per-GPU reserve kept out of JetFire auto prefill sub-batch "
            "selection for residual/runtime/CUDA graph/allocator peaks."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-activation-safety",
        type=float,
        default=1.15,
        help=(
            "Safety multiplier applied to the estimated per-request prefill "
            "activation peak when choosing JetFire auto sub-batch."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-auto-max-sub-batch-size",
        type=int,
        default=4,
        help=(
            "Upper bucket used when the memory model requires JetFire to split "
            "a prefill batch. A full batch that fits the modeled budget bypasses "
            "this cap."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-chunk-threshold-tokens",
        type=int,
        default=262144,
        help=(
            "Enable JetFire's layer-local prefill activation chunking when "
            "the estimated profile/full-prefill token budget reaches this "
            "many tokens. This covers batched long-context cases whose "
            "single-request ctx is below the old 64k guard."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-chunk-activation-threshold-gib",
        type=float,
        default=1.0,
        help=(
            "Enable JetFire prefill activation partitioning when the estimated "
            "largest local BF16 activation tensor reaches this many GiB."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-chunk-scratch-fraction",
        type=float,
        default=0.20,
        help=(
            "Also require the estimated activation tensor to exceed this "
            "fraction of the remaining scratch budget before auto-enabling "
            "partitioning. This keeps small models on the faster unpartitioned "
            "prefill path while protecting large-model memory cliffs."
        ),
    )
    parser.add_argument(
        "--jetfire-prefill-chunk-tokens",
        default=None,
        help=(
            "Debug/manual uniform JetFire prefill activation chunk size. "
            "When omitted, release auto selects a module-specific partition "
            "profile from the estimated scratch-memory budget and sets "
            "EMBED/MLP/ATTN_O_PROJ/PREFILL_ATTN/PREFILL_QKV separately. "
            "Set an explicit value such as 512,384,256,128,96,64 to override "
            "all modules uniformly; use 0/off to disable this chunking."
        ),
    )
    parser.add_argument(
        "--torch-extensions-dir",
        default=os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_EXT_DIR),
    )
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp") / f"jetfire_official_serve_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    for key in ("tensor_parallel_size", "ctx", "output_len", "concurrency", "num_prompts", "runs", "cpu_attn_threads", "kv_cache_block_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmups < 0 or args.top_k <= 0:
        parser.error("--warmups must be nonnegative and --top-k must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    gpu_ids = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if len(set(gpu_ids)) < args.tensor_parallel_size:
        parser.error("--gpu must provide at least --tensor-parallel-size distinct devices")
    if args.model.startswith("/path/to/") or (args.model.startswith(("/", "./", "../", "~")) and not Path(args.model).expanduser().is_dir()):
        parser.error(f"Model directory unavailable: {args.model}. Set --model to a local checkpoint.")
    try:
        args.jetfire_prefill_chunk_tokens = _parse_prefill_chunk_tokens(
            args.jetfire_prefill_chunk_tokens
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.jetfire_prefill_chunk_threshold_tokens < 0:
        raise SystemExit("--jetfire-prefill-chunk-threshold-tokens must be >= 0")
    if args.jetfire_prefill_chunk_activation_threshold_gib < 0:
        raise SystemExit(
            "--jetfire-prefill-chunk-activation-threshold-gib must be >= 0"
        )
    if args.jetfire_prefill_chunk_scratch_fraction < 0:
        raise SystemExit("--jetfire-prefill-chunk-scratch-fraction must be >= 0")
    if args.backend == "jetfire":
        _assert_release_import_path()

    _preflight_model_length(args)
    specs = _build_specs(args)
    _preflight_release_policy(args, specs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = _run_one(args, spec)
        rows.append(row)
        print(
            "[bench] {backend} conc={conc} prompts={prompts} status={status} "
            "TTFT={ttft}ms TPOT={tpot}ms".format(
                backend=row.get("backend"),
                conc=row.get("max_concurrency"),
                prompts=row.get("num_prompts"),
                status=row.get("status"),
                ttft=_fmt(row.get("median_ttft_ms")),
                tpot=_fmt(row.get("median_tpot_ms")),
            ),
            flush=True,
        )

    summary_json = args.out_dir / "summary.json"
    summary_csv = args.out_dir / "summary.csv"
    report_md = args.out_dir / "report.md"
    summary = {
        "mode": args.mode,
        "model": args.model,
        "ctx": int(args.ctx),
        "output_len": int(args.output_len),
        "rows": rows,
        "max_successful_concurrency": _best_capacity(rows),
        "status": "passed" if _all_runs_complete(rows, int(args.output_len)) else "failed",
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    _write_csv(rows, summary_csv)
    _write_report(args, rows, report_md)
    print(f"[bench] wrote {summary_json}", flush=True)
    print(f"[bench] wrote {summary_csv}", flush=True)
    print(f"[bench] wrote {report_md}", flush=True)
    if not _all_runs_complete(rows, int(args.output_len)):
        raise SystemExit("Serving evaluation failed or incomplete; see summary.json and server/client logs.")


if __name__ == "__main__":
    main()
