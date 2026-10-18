"""Regression checks for artifact provenance, selection, and memory accounting."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jetfire_vllm.bench import release_accuracy_eval as accuracy
from jetfire_vllm.bench.memory_breakdown_probe import CONFIGS, MODELS, computed_rows


def test_accuracy_marks_skipped_samples_incomplete(tmp_path: Path) -> None:
    results = {
        "qasper": {"score": 100.0, "num_samples": 2, "too_long": 1, "partial": False},
        "trec": {"score": 50.0, "num_samples": 2, "too_long": 0, "partial": False},
    }
    path = tmp_path / "scores.json"
    accuracy._write_overall(results, path)
    saved = json.loads(path.read_text())
    assert saved["overall"] == 75.0
    assert saved["status"] == "incomplete"
    assert not path.with_suffix(".json.tmp").exists()


def test_accuracy_does_not_report_empty_dataset_as_complete(tmp_path: Path) -> None:
    results = {}
    accuracy._write_overall(results, tmp_path / "scores.json")
    assert results["status"] == "incomplete"


def test_full_precision_host_shadow_matches_quantized_memory_accounting() -> None:
    for model in MODELS.values():
        for config in CONFIGS.values():
            rows = computed_rows(model, config, 8192)
            aggregate = [row for row in rows if row["rank"] == -1]
            gpu = next(float(row["computed_gib"]) for row in aggregate if row["component"] == "KV replica - total")
            host = next(float(row["computed_gib"]) for row in aggregate if row["component"] == "full-precision host replica")
            assert host / gpu == pytest.approx(16 / config.avg_bits, rel=1e-4)
            per_gpu = [float(row["computed_gib"]) for row in rows if row["rank"] >= 0 and row["component"] == "KV replica - total"]
            assert sum(per_gpu) == pytest.approx(gpu, abs=2e-6)


@pytest.mark.parametrize("flag,value", [("--niah-repeats", "0"), ("--niah-depths", "101"), ("--niah-contexts", "-4")])
def test_invalid_niah_configuration_fails_before_model_loading(flag: str, value: str, tmp_path: Path) -> None:
    argv = ["accuracy", "--model-key", "llama31_8b", "--backend", "jetfire_k3v2", "--suite", "niah_32k", "--out-dir", str(tmp_path), flag, value]
    with patch("sys.argv", argv), pytest.raises(SystemExit, match="2"):
        accuracy.parse_args()


def test_niah_sample_spec_is_reproducible_and_varies_by_repeat() -> None:
    assert accuracy._niah_sample_spec(0, 42) == accuracy._niah_sample_spec(0, 42)
    assert accuracy._niah_sample_spec(0, 42) != accuracy._niah_sample_spec(1, 42)


def test_local_longbench_data_and_provenance(tmp_path: Path) -> None:
    from types import SimpleNamespace
    path = tmp_path / "qasper.jsonl"
    path.write_text('{"context":"Evidence", "answers":["yes"]}\n', encoding="utf-8")
    args = SimpleNamespace(suite="longbench_v1", tasks="qasper", longbench_v1_data_dir=str(tmp_path))
    assert accuracy._load_longbench_v1_dataset("qasper", args)[0]["context"] == "Evidence"
    first = accuracy._dataset_provenance(args)["dataset_files"][str(path)]["sha256"]
    path.write_text('{"context":"Changed evidence", "answers":["no"]}\n', encoding="utf-8")
    second = accuracy._dataset_provenance(args)["dataset_files"][str(path)]["sha256"]
    assert first != second
    with pytest.raises(FileNotFoundError, match="LongBench task file"):
        accuracy._load_longbench_v1_dataset("trec", args)


def test_offline_longbench_pipeline_reports_complete_score(tmp_path: Path) -> None:
    from types import SimpleNamespace
    (tmp_path / "trec.jsonl").write_text(
        '{"context":"Examples", "input":"Who invented this?", "answers":["Individual"], "all_classes":["Individual", "Date"]}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        tasks="trec", longbench_v1_data_dir=str(tmp_path), resume=False,
        sample_start=0, sample_end=0, max_samples=0, suite="longbench_v1",
        model_key="llama31_8b", max_model_len=2048, generation_token_margin=32,
        only_over_model_len=0, save_predictions=True, backend="jetfire_k3v2",
        jetfire_reset_runtime_between_samples=False,
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))
    with patch.object(accuracy, "_generate_one", return_value="Individual"):
        scores = accuracy.run_longbench_v1(object(), tokenizer, tmp_path / "scores.json", args)
    assert scores["status"] == "complete"
    assert scores["overall"] == 100.0
    assert scores["trec"]["num_samples"] == 1


def test_checkpoint_precedence_respects_named_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetfire_vllm.bench.model_paths import resolve_model_path

    monkeypatch.setenv("JETFIRE_MODEL_PATH", "/models/llama8b")
    monkeypatch.setenv("JETFIRE_QWEN3_30B", "/models/qwen30b")
    assert resolve_model_path("qwen3_30b_a3b") == "/models/qwen30b"
    assert resolve_model_path("qwen3_30b_a3b", "/models/explicit") == "/models/explicit"
    monkeypatch.delenv("JETFIRE_QWEN3_30B")
    assert resolve_model_path("qwen3_30b_a3b") == "/models/llama8b"


def test_serving_completion_requires_all_requests_and_tokens() -> None:
    from jetfire_vllm.bench.official_serve_benchmark import _all_runs_complete

    row = {"status": "ok", "completed": 2, "num_prompts": 2, "failed": 0, "total_output_tokens": 16}
    assert _all_runs_complete([row], 8)
    assert not _all_runs_complete([{**row, "total_output_tokens": 15}], 8)
    assert not _all_runs_complete([{**row, "completed": 1}], 8)
    assert not _all_runs_complete([{**row, "status": "server_failed"}], 8)
    assert not _all_runs_complete([], 8)


def test_multi_model_selection_cannot_reuse_generic_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetfire_vllm.bench.model_paths import validate_model_matrix

    monkeypatch.setenv("JETFIRE_MODEL_PATH", "/models/llama8b")
    monkeypatch.setenv("JETFIRE_LLAMA31_8B", "/models/llama8b")
    monkeypatch.delenv("JETFIRE_QWEN3_30B", raising=False)
    with pytest.raises(ValueError, match="JETFIRE_QWEN3_30B"):
        validate_model_matrix(["llama31_8b", "qwen3_30b"])
    monkeypatch.setenv("JETFIRE_QWEN3_30B", "/models/qwen30b")
    validate_model_matrix(["llama31_8b", "qwen3_30b"])
