import json
import sys

import pytest

from jetfire_vllm import artifact


def test_dry_run_is_a_plan_without_outputs(tmp_path, monkeypatch, capsys):
    out = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["jetfire-artifact", "serve", "--mode", "k4v4", "--dry-run", "--out-dir", str(out)])
    assert artifact.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned"
    command = plan["commands"][0]
    assert command[command.index("--jetfire-group-size") + 1] == "128"
    assert not out.exists()


def test_failed_step_records_failure_and_stops(tmp_path, monkeypatch):
    out = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["jetfire-artifact", "validate", "--out-dir", str(out)])
    monkeypatch.setattr(artifact, "_versions", lambda: {})
    monkeypatch.setattr(artifact, "_source_hashes", lambda: {})
    monkeypatch.setattr(artifact.shutil, "which", lambda _: None)
    monkeypatch.setattr(artifact, "_plan", lambda *_: [[sys.executable, "-c", "raise SystemExit(7)"], ["this-must-not-run"]])
    assert artifact.main() == 7
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert len(manifest["steps"]) == 1
    assert manifest["steps"][0]["returncode"] == 7


def test_existing_evidence_is_preserved(tmp_path, monkeypatch):
    existing = tmp_path / "manifest.json"
    existing.write_text("keep existing evidence")
    monkeypatch.setattr(sys, "argv", ["jetfire-artifact", "doctor", "--out-dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        artifact.main()
    assert existing.read_text() == "keep existing evidence"


def test_accuracy_requires_explicit_inputs(monkeypatch):
    for key in artifact.MODEL_ENV.values():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("JETFIRE_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["jetfire-artifact", "accuracy"])
    with pytest.raises(SystemExit) as exc:
        artifact.main()
    assert exc.value.code == 2


def test_wheel_kernel_validation_requires_source_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(artifact, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["jetfire-artifact", "kernels"])
    with pytest.raises(SystemExit) as exc:
        artifact.main()
    assert exc.value.code == 2
    assert "requires the full JetFire source checkout" in capsys.readouterr().err
