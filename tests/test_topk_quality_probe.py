from __future__ import annotations

from pathlib import Path

import pytest

from jetfire_vllm.bench.topk_quality_probe import _load_filler_text


def test_filler_loader_rejects_missing_longbench_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no LongBench JSONL contexts"):
        _load_filler_text(tmp_path, ("qasper",))


def test_synthetic_filler_requires_explicit_opt_in(tmp_path: Path) -> None:
    text = _load_filler_text(tmp_path, ("qasper",), allow_synthetic=True)
    assert "long technical report" in text


def test_filler_loader_uses_matching_jsonl_context(tmp_path: Path) -> None:
    (tmp_path / "qasper.jsonl").write_text(
        '{"context": "real long-context evidence"}\n',
        encoding="utf-8",
    )
    assert _load_filler_text(tmp_path, ("qasper",)) == "real long-context evidence"
