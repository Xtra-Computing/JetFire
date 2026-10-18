from __future__ import annotations

import os

import pytest

from jetfire_vllm.adapter.engine import _maybe_disable_vllm_graph
from jetfire_vllm.adapter.release_policy import ReleasePolicyError


def _unsafe_large_batch_kwargs() -> dict[str, object]:
    return {
        "enforce_eager": False,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "max_num_seqs": 64,
    }


def test_strict_release_rejects_implicit_eager_fallback(monkeypatch) -> None:
    monkeypatch.setenv("JETFIRE_RELEASE_GRAPH_POLICY", "1")

    kwargs = _unsafe_large_batch_kwargs()
    with pytest.raises(ReleasePolicyError, match="cannot switch to eager"):
        _maybe_disable_vllm_graph(kwargs)

    assert kwargs["enforce_eager"] is False
    assert "compilation_config" in kwargs


def test_nonrelease_mode_keeps_explicit_eager_fallback(monkeypatch) -> None:
    monkeypatch.delenv("JETFIRE_RELEASE_GRAPH_POLICY", raising=False)
    monkeypatch.delenv("JETFIRE_FORCE_VLLM_CUDAGRAPH", raising=False)
    monkeypatch.setenv("JETFIRE_DECODE_GRAPH", "1")
    monkeypatch.setenv("JETFIRE_ENABLE_VLLM_CUDAGRAPH", "1")

    kwargs = _unsafe_large_batch_kwargs()
    _maybe_disable_vllm_graph(kwargs)

    assert kwargs["enforce_eager"] is True
    assert "compilation_config" not in kwargs
    assert os.environ["JETFIRE_DECODE_GRAPH"] == "0"
    assert os.environ["JETFIRE_ENABLE_VLLM_CUDAGRAPH"] == "0"
