from __future__ import annotations

import os
import time

from jetfire_vllm.adapter import _gpu_ext


def _make_old_lock(tmp_path, name: str) -> None:
    (tmp_path / "lock").write_text("", encoding="utf-8")
    (tmp_path / f"{name}.so").write_text("built", encoding="utf-8")
    old = time.time() - 120
    os.utime(tmp_path / "lock", (old, old))


def test_active_extension_build_keeps_lock(tmp_path, monkeypatch) -> None:
    name = "test_active_extension"
    _make_old_lock(tmp_path, name)
    monkeypatch.setattr(
        "torch.utils.cpp_extension._get_build_directory",
        lambda _name, verbose=False: str(tmp_path),
    )
    monkeypatch.setattr(_gpu_ext, "_extension_build_active", lambda *_: True)

    _gpu_ext._remove_stale_lock_if_prebuilt(name)

    assert (tmp_path / "lock").exists()


def test_inactive_stale_extension_lock_is_removed(tmp_path, monkeypatch) -> None:
    name = "test_stale_extension"
    _make_old_lock(tmp_path, name)
    monkeypatch.setattr(
        "torch.utils.cpp_extension._get_build_directory",
        lambda _name, verbose=False: str(tmp_path),
    )
    monkeypatch.setattr(_gpu_ext, "_extension_build_active", lambda *_: False)

    _gpu_ext._remove_stale_lock_if_prebuilt(name)

    assert not (tmp_path / "lock").exists()
