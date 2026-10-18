#!/usr/bin/env python3
"""Clean stale JetFire runtime/build artifacts before benchmark launches.

Default mode is intentionally conservative: remove only old PyTorch extension
``lock`` files under JetFire build directories when a shared object already
exists and no live process has the lock open.  Use ``--purge-extensions`` only
when you want a clean rebuild of JetFire JIT extensions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def _default_torch_extension_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        roots.append(Path(os.environ["TORCH_EXTENSIONS_DIR"]).expanduser())
        return roots

    cache_root = Path.home() / ".cache" / "torch_extensions"
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    try:
        import torch

        if torch.version.cuda:
            cuda_tag = "cu" + torch.version.cuda.replace(".", "")
            roots.append(cache_root / f"{py_tag}_{cuda_tag}")
    except Exception:
        pass

    if cache_root.exists():
        roots.extend(sorted(cache_root.glob(f"{py_tag}_*")))
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            seen.add(resolved)
            out.append(root)
    return out


def _path_has_open_fd(path: Path) -> bool:
    try:
        target = path.resolve()
        for fd_root in Path("/proc").glob("[0-9]*/fd"):
            try:
                for fd in fd_root.iterdir():
                    try:
                        if fd.resolve() == target:
                            return True
                    except OSError:
                        continue
            except (OSError, PermissionError):
                continue
    except Exception:
        return True
    return False


def _dir_has_open_fd(path: Path) -> bool:
    try:
        target = path.resolve()
        for fd_root in Path("/proc").glob("[0-9]*/fd"):
            try:
                for fd in fd_root.iterdir():
                    try:
                        fd_target = fd.resolve()
                    except OSError:
                        continue
                    if fd_target == target or target in fd_target.parents:
                        return True
            except (OSError, PermissionError):
                continue
    except Exception:
        return True
    return False


def _dir_has_active_build_process(path: Path) -> bool:
    try:
        target = path.resolve()
    except OSError:
        target = path
    build_name = target.name
    for proc_root in Path("/proc").glob("[0-9]*"):
        try:
            cwd = (proc_root / "cwd").resolve()
            if cwd == target or target in cwd.parents:
                return True
        except OSError:
            pass
        try:
            cmdline = (proc_root / "cmdline").read_bytes().replace(b"\0", b" ")
        except (OSError, PermissionError):
            continue
        if not cmdline:
            continue
        text = cmdline.decode(errors="ignore")
        if str(target) in text:
            return True
        if build_name in text and any(
            name in text for name in ("ninja", "nvcc", "gcc", "g++", "cc1plus", "ptxas")
        ):
            return True
    return False


def _jetfire_build_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if root.exists():
            dirs.extend(sorted(p for p in root.glob("jetfire*") if p.is_dir()))
    return dirs


def clean_stale_locks(roots: list[Path], *, min_age_s: float, dry_run: bool) -> int:
    removed = 0
    now = time.time()
    for build_dir in _jetfire_build_dirs(roots):
        lock_path = build_dir / "lock"
        if not lock_path.exists():
            continue
        if not any(build_dir.glob("*.so")):
            continue
        try:
            age_s = now - lock_path.stat().st_mtime
        except OSError:
            continue
        if age_s < min_age_s:
            continue
        if _dir_has_active_build_process(build_dir):
            print(f"[jetfire-clean] keep active build lock: {lock_path}", flush=True)
            continue
        if _path_has_open_fd(lock_path):
            print(f"[jetfire-clean] keep live lock: {lock_path}", flush=True)
            continue
        print(f"[jetfire-clean] remove stale lock: {lock_path}", flush=True)
        if not dry_run:
            try:
                lock_path.unlink()
                removed += 1
            except OSError as exc:
                print(f"[jetfire-clean] failed to remove {lock_path}: {exc}", flush=True)
        else:
            removed += 1
    return removed


def purge_extensions(roots: list[Path], *, dry_run: bool) -> int:
    removed = 0
    for build_dir in _jetfire_build_dirs(roots):
        if _dir_has_active_build_process(build_dir) or _dir_has_open_fd(build_dir):
            print(f"[jetfire-clean] keep live extension dir: {build_dir}", flush=True)
            continue
        print(f"[jetfire-clean] purge extension dir: {build_dir}", flush=True)
        if not dry_run:
            shutil.rmtree(build_dir, ignore_errors=False)
        removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean stale JetFire torch-extension artifacts."
    )
    parser.add_argument(
        "--torch-extensions-dir",
        action="append",
        dest="roots",
        help="Torch extension root to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-lock-age-s",
        type=float,
        default=float(os.environ.get("JETFIRE_STALE_EXTENSION_LOCK_AGE_S", "600")),
        help="Only remove stale locks older than this many seconds.",
    )
    parser.add_argument(
        "--purge-extensions",
        action="store_true",
        help="Delete JetFire extension build dirs for a clean rebuild.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [Path(p).expanduser() for p in args.roots] if args.roots else _default_torch_extension_roots()
    print(
        "[jetfire-clean] roots="
        + ",".join(str(root) for root in roots)
        + f" dry_run={int(args.dry_run)} purge_extensions={int(args.purge_extensions)}",
        flush=True,
    )

    removed_locks = clean_stale_locks(
        roots, min_age_s=float(args.min_lock_age_s), dry_run=bool(args.dry_run)
    )
    removed_dirs = 0
    if args.purge_extensions:
        removed_dirs = purge_extensions(roots, dry_run=bool(args.dry_run))

    print(
        f"[jetfire-clean] done removed_locks={removed_locks} "
        f"removed_extension_dirs={removed_dirs}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
