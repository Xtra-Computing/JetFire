"""Failure propagation and evidence reuse for the end-to-end campaign."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jetfire_vllm.bench import artifact_performance as campaign


def _case() -> campaign.Case:
    return campaign.Case("llama31_8b", "/models/llama", 1, "1", "k3v2", 4096, 4, 128, 3)


def _rows(case: campaign.Case) -> list[dict]:
    return [{"run": run, "status": "ok", "num_prompts": case.batch, "completed": case.batch, "failed": 0,
             "total_input_tokens": case.ctx * case.batch, "total_output_tokens": case.output_len * case.batch,
             "ctx": case.ctx, "output_len": case.output_len, "max_concurrency": case.batch,
             "jetfire_mode": campaign.MODES[case.mode][0], "jetfire_group_size": campaign.MODES[case.mode][1],
             "gpu_kv_cache_tokens": 18000, **{metric: float(run + 1) for metric in campaign.METRICS}}
            for run in range(case.runs)]


def _evidence(tmp_path: Path, rows: list[dict], probes: int = 3) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary.json").write_text(json.dumps({"status": "passed", "rows": rows}))
    (tmp_path / "semantic_probes.jsonl").write_text('{"status":"passed","text":"4"}\n' * probes)
    return tmp_path


def _checkpoint(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4096,
                                                 "num_hidden_layers": 32, "num_key_value_heads": 8,
                                                 "max_position_embeddings": 131072}))
    return model


def test_default_dry_run_covers_three_models_without_creating_evidence(tmp_path: Path, monkeypatch, capsys) -> None:
    for name in campaign.MODEL_ENV.values():
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "results"
    assert campaign.main(["--dry-run", "--out-dir", str(out)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cells"] == 24
    assert plan["measured_runs"] == 72
    assert {case["model"] for case in plan["cases"]} == set(campaign.MODEL_ENV)
    tp_cases = [case for case in plan["cases"] if case["tp"] == 2]
    assert all(case["gpus"] == "0,1" for case in tp_cases)
    assert not out.exists()


@pytest.mark.parametrize("change", [{"total_output_tokens": 511}, {"completed": 3}, {"status": "bench_timeout"}, {"total_input_tokens": 16379}, {"median_tpot_ms": float("nan")}])
def test_truncated_or_invalid_trial_fails_entire_cell(tmp_path: Path, change: dict) -> None:
    case = _case()
    rows = _rows(case)
    rows[1].update(change)
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "failed"
    assert result["batch_e2el_ms_median"] is None
    assert result["errors"]


def test_aggregation_uses_trial_median_and_reports_actual_capacity(tmp_path: Path) -> None:
    case = _case()
    rows = _rows(case)
    rows[-1]["batch_e2el_ms"] = 10000
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "passed"
    assert result["batch_e2el_ms_median"] == 2.0
    assert result["resident_capacity_min"] == pytest.approx(18000 / (4096 + 128))
    assert result["all_requests_resident"] is True
    assert result["semantic_probes_passed"] == 3


@pytest.mark.parametrize("tpot", [None, 0, 0.25])
def test_single_token_decode_accepts_missing_or_nonnegative_tpot(tmp_path: Path, tpot) -> None:
    # One output token has no inter-token interval; E2E and completion remain required.
    case = replace(_case(), output_len=1)
    rows = _rows(case)
    for row in rows:
        row["median_tpot_ms"] = tpot
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "passed"
    assert result["median_tpot_ms_median"] == tpot
    assert result["batch_e2el_ms_median"] == 2.0


@pytest.mark.parametrize("tpot", [-0.1, float("nan"), float("inf")])
def test_single_token_decode_rejects_invalid_tpot(tmp_path: Path, tpot) -> None:
    case = replace(_case(), output_len=1)
    rows = _rows(case)
    rows[0]["median_tpot_ms"] = tpot
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "failed"
    assert result["batch_e2el_ms_median"] is None


@pytest.mark.parametrize("tpot", [None, 0])
def test_multi_token_decode_still_requires_positive_tpot(tmp_path: Path, tpot) -> None:
    case = replace(_case(), output_len=64)
    rows = _rows(case)
    rows[0]["median_tpot_ms"] = tpot
    assert campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)["status"] == "failed"


def test_single_token_decode_still_requires_every_output(tmp_path: Path) -> None:
    case = replace(_case(), output_len=1)
    rows = _rows(case)
    rows[0].update(median_tpot_ms=None, total_output_tokens=case.batch - 1)
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "failed"
    assert result["batch_e2el_ms_median"] is None


def test_nonzero_exit_and_missing_trials_cannot_reuse_success_summary(tmp_path: Path) -> None:
    case = _case()
    attempt = _evidence(tmp_path, _rows(case))
    assert campaign._evaluate(case, attempt, 7, True)["status"] == "failed"
    assert campaign._evaluate(replace(case, runs=4), attempt, 0, True)["status"] == "failed"
    assert campaign._evaluate(case, _evidence(tmp_path, _rows(case), probes=2), 0, True)["status"] == "failed"


@pytest.mark.parametrize("args", [["--contexts", "-4"], ["--runs", "0"], ["--tp-gpus", "1"], ["--models", "other"], ["--batches", "1,1"]])
def test_invalid_campaign_fails_before_model_loading(args: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        campaign.parse_args(["--dry-run", *args])


def test_checkpoint_label_validation_rejects_wrong_architecture(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        campaign.parse_args(["--models", "llama33_70b", "--model-path", f"llama33_70b={path}"])


def test_resume_preserves_failures_and_checks_returncode_again(tmp_path: Path, monkeypatch) -> None:
    model = _checkpoint(tmp_path)
    out = tmp_path / "results"
    argv = ["--models", "llama31_8b", "--modes", "k3v2", "--contexts", "4096", "--batches", "4",
            "--model-path", f"llama31_8b={model}", "--out-dir", str(out)]
    monkeypatch.setattr(campaign, "_provenance", lambda *_: {"test_provenance": "stable"})
    monkeypatch.setattr(campaign.shutil, "which", lambda _: None)
    calls = []

    def execute(cmd: list[str], **kwargs) -> SimpleNamespace:
        attempt = Path(cmd[cmd.index("--out-dir") + 1])
        calls.append(attempt)
        rows = _rows(_case())
        if len(calls) == 1:
            rows[0]["status"] = "bench_timeout"
        _evidence(attempt, rows)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(campaign.subprocess, "run", execute)
    assert campaign.main(argv) == 1
    assert campaign.main([*argv, "--resume"]) == 0
    assert len(calls) == 2
    assert calls[0].exists() and calls[1].exists()
    assert campaign._load(calls[0] / "invocation.json")["status"] == "failed"
    assert campaign.main([*argv, "--resume"]) == 0
    assert len(calls) == 2
    # Recheck evidence before skipping it, rather than trusting the success marker.
    summary = campaign._load(calls[1] / "summary.json")
    summary["rows"][0]["total_output_tokens"] = 1
    (calls[1] / "summary.json").write_text(json.dumps(summary))
    assert campaign.main([*argv, "--resume"]) == 0
    assert len(calls) == 3
    assert campaign._load(out / "performance_summary.json")["cases_passed"] == 1


@pytest.mark.parametrize("answer", ["5", "not 4, 5", "2+2=4", "4 or 5"])
def test_semantic_probe_is_untimed_and_rejects_wrong_answer(tmp_path: Path, monkeypatch, answer: str) -> None:
    from jetfire_vllm.bench import official_serve_benchmark as harness
    probe = tmp_path / "semantic.jsonl"
    ready = []
    monkeypatch.setattr(harness, "_wait_ready", lambda *args: ready.append(args))
    monkeypatch.setattr(harness, "main", lambda: harness._wait_ready("http://localhost", object(), 5))

    class Response:
        def __enter__(self):
            from io import StringIO
            return StringIO(json.dumps({"choices": [{"message": {"content": answer}}], "usage": {"completion_tokens": 1}}))
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(campaign.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "argv", ["test"])
    with pytest.raises(RuntimeError, match="expected only the answer 4"):
        campaign._harness_worker([str(probe), "--"])
    assert len(ready) == 1
    saved = json.loads(probe.read_text())
    assert saved["status"] == "failed"
    assert saved["text"] == answer


def test_official_client_special_token_accounting_is_recorded_explicitly(tmp_path: Path) -> None:
    case = _case()
    rows = _rows(case)
    for row in rows:
        row["total_input_tokens"] -= case.batch
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == "passed"
    assert result["input_token_notes"]
    assert result["input_token_tolerance_per_request"] == 1
    for observation in result["input_token_observations"]:
        assert observation["client_reported_input_tokens_per_request_mean"] == 4095
        assert observation["delta_from_nominal"] == -4


@pytest.mark.parametrize("ctx,reported_input,expected_status", [(1, 0, "passed"), (2, 0, "failed"), (1, -1, "failed")])
def test_one_token_prefill_accepts_only_nonnegative_special_token_accounting(
        tmp_path: Path, ctx: int, reported_input: int, expected_status: str) -> None:
    case = replace(_case(), ctx=ctx)
    rows = _rows(case)
    for row in rows:
        row["total_input_tokens"] = reported_input
    result = campaign._evaluate(case, _evidence(tmp_path, rows), 0, True)
    assert result["status"] == expected_status
    assert result["input_token_observations"][0]["client_reported_total_input_tokens"] == reported_input
