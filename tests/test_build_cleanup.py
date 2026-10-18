"""Build cleanup preserves live compiler work and removes abandoned locks."""
from __future__ import annotations

import os
from pathlib import Path
import time

from jetfire_vllm.bench import clean_jetfire_runtime_artifacts as cleanup


def _build_with_old_lock(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "extensions"
    build = root / "jetfire_test_extension"
    build.mkdir(parents=True)
    lock = build / "lock"
    lock.touch()
    (build / "jetfire_test_extension.so").write_bytes(b"prebuilt")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    return root, build


def _use_fake_proc(tmp_path: Path, monkeypatch) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    # Redirect /proc only; ordinary filesystem operations use actual Path.
    monkeypatch.setattr(cleanup, "Path", lambda value: proc if value == "/proc" else Path(value))
    return proc


def test_multidigit_compiler_pid_preserves_lock_and_extension(tmp_path, monkeypatch):
    root, build = _build_with_old_lock(tmp_path)
    proc = _use_fake_proc(tmp_path, monkeypatch)
    compiler = proc / "12345"
    compiler.mkdir()
    (compiler / "cwd").symlink_to(build, target_is_directory=True)
    (compiler / "cmdline").write_bytes(b"g++\0-c\0kernel.cpp\0")

    assert cleanup._dir_has_active_build_process(build)
    assert cleanup.clean_stale_locks([root], min_age_s=60, dry_run=False) == 0
    assert cleanup.purge_extensions([root], dry_run=False) == 0
    assert (build / "lock").is_file()
    assert (build / "jetfire_test_extension.so").is_file()


def test_multidigit_compiler_command_preserves_lock_without_build_cwd(tmp_path, monkeypatch):
    root, build = _build_with_old_lock(tmp_path)
    proc = _use_fake_proc(tmp_path, monkeypatch)
    compiler = proc / "23456"
    compiler.mkdir()
    (compiler / "cwd").symlink_to(tmp_path, target_is_directory=True)
    (compiler / "cmdline").write_bytes(b"nvcc\0-o\0" + str(build / "kernel.o").encode() + b"\0")

    assert cleanup.clean_stale_locks([root], min_age_s=60, dry_run=False) == 0
    assert (build / "lock").is_file()


def test_inactive_prebuilt_extension_loses_only_stale_lock(tmp_path, monkeypatch):
    root, build = _build_with_old_lock(tmp_path)
    _use_fake_proc(tmp_path, monkeypatch)

    assert cleanup.clean_stale_locks([root], min_age_s=60, dry_run=False) == 1
    assert not (build / "lock").exists()
    assert (build / "jetfire_test_extension.so").read_bytes() == b"prebuilt"
