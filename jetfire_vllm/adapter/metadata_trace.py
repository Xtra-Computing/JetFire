"""Small JSONL trace helpers for JetFire scheduler metadata.

The trace is intentionally CPU-side and opt-in.  It records enough shape and
request-split information to reproduce vLLM extend-prefill scheduler states
without rerunning a full LongBench/serving benchmark.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_COUNT = 0


def trace_enabled() -> bool:
    return os.environ.get("JETFIRE_TRACE_EXTEND_PREFILL", "0") == "1"


def trace_path() -> Path:
    configured = os.environ.get("JETFIRE_TRACE_EXTEND_PREFILL_PATH")
    if configured:
        return Path(configured)
    return Path(f"/tmp/jetfire_extend_prefill_trace_{os.getpid()}.jsonl")


def _limit() -> int:
    return max(1, int(os.environ.get("JETFIRE_TRACE_EXTEND_PREFILL_LIMIT", "64")))


def _max_rows() -> int:
    return max(1, int(os.environ.get("JETFIRE_TRACE_EXTEND_PREFILL_MAX_ROWS", "16")))


def _max_cols() -> int:
    return max(1, int(os.environ.get("JETFIRE_TRACE_EXTEND_PREFILL_MAX_COLS", "16")))


def _tensor_list(value: Any, *, max_rows: int | None = None) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    rows = _max_rows() if max_rows is None else max_rows
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.to(dtype=torch.int64)
    if tensor.ndim == 0:
        return [int(tensor.item())]
    if tensor.ndim == 1:
        return [int(x) for x in tensor[:rows].tolist()]
    cols = _max_cols()
    out = []
    for row in tensor[:rows]:
        out.append([int(x) for x in row[:cols].tolist()])
    return out


def _tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.shape]
    return None


def _write(record: dict[str, Any]) -> None:
    global _COUNT
    if not trace_enabled():
        return
    with _LOCK:
        if _COUNT >= _limit():
            return
        _COUNT += 1
        record.setdefault("pid", os.getpid())
        record.setdefault("time_s", time.time())
        record.setdefault("ordinal", _COUNT)
        path = trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def record_builder_metadata(
    *,
    metadata: Any,
    query_lens_cpu: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor | None,
    computed_tokens_cpu: torch.Tensor | None,
) -> None:
    """Record one attention-metadata build when it matters for JetFire."""
    if not trace_enabled():
        return
    prefill_has_context = bool(
        getattr(metadata, "jetfire_prefill_has_context", False)
    )
    trace_all = os.environ.get("JETFIRE_TRACE_ALL_METADATA", "0") == "1"
    if not (prefill_has_context or trace_all):
        return
    record = {
        "event": "metadata_build",
        "num_decodes": int(getattr(metadata, "num_decodes", 0)),
        "num_prefills": int(getattr(metadata, "num_prefills", 0)),
        "num_decode_tokens": int(getattr(metadata, "num_decode_tokens", 0)),
        "num_prefill_tokens": int(getattr(metadata, "num_prefill_tokens", 0)),
        "num_actual_tokens": int(getattr(metadata, "num_actual_tokens", 0)),
        "max_query_len": int(getattr(metadata, "max_query_len", 0)),
        "max_seq_len": int(getattr(metadata, "max_seq_len", 0)),
        "prefill_has_context": prefill_has_context,
        "max_prefill_context_len": int(
            getattr(metadata, "jetfire_max_prefill_context_len", 0)
        ),
        "query_lens": _tensor_list(query_lens_cpu),
        "seq_lens": _tensor_list(seq_lens_cpu),
        "computed_tokens": _tensor_list(computed_tokens_cpu),
        "query_start_loc": _tensor_list(getattr(metadata, "query_start_loc", None)),
        "block_table_shape": _tensor_shape(getattr(metadata, "block_table", None)),
        "block_table_head": _tensor_list(getattr(metadata, "block_table", None)),
    }
    _write(record)


def record_runtime_metadata(
    *,
    event: str,
    attn_metadata: Any,
    layer_name: str,
    query_shape: tuple[int, ...] | list[int],
    key_shape: tuple[int, ...] | list[int],
    value_shape: tuple[int, ...] | list[int],
) -> None:
    if not trace_enabled():
        return
    record = {
        "event": event,
        "layer_name": layer_name,
        "query_shape": [int(x) for x in query_shape],
        "key_shape": [int(x) for x in key_shape],
        "value_shape": [int(x) for x in value_shape],
        "num_decodes": int(getattr(attn_metadata, "num_decodes", 0)),
        "num_prefills": int(getattr(attn_metadata, "num_prefills", 0)),
        "num_decode_tokens": int(getattr(attn_metadata, "num_decode_tokens", 0)),
        "num_prefill_tokens": int(getattr(attn_metadata, "num_prefill_tokens", 0)),
        "num_actual_tokens": int(getattr(attn_metadata, "num_actual_tokens", 0)),
        "max_query_len": int(getattr(attn_metadata, "max_query_len", 0)),
        "max_seq_len": int(getattr(attn_metadata, "max_seq_len", 0)),
        "prefill_has_context": bool(
            getattr(attn_metadata, "jetfire_prefill_has_context", False)
        ),
        "max_prefill_context_len": int(
            getattr(attn_metadata, "jetfire_max_prefill_context_len", 0)
        ),
        "seq_lens": _tensor_list(getattr(attn_metadata, "seq_lens", None)),
        "query_start_loc": _tensor_list(getattr(attn_metadata, "query_start_loc", None)),
        "block_table_shape": _tensor_shape(getattr(attn_metadata, "block_table", None)),
        "block_table_head": _tensor_list(getattr(attn_metadata, "block_table", None)),
    }
    _write(record)
