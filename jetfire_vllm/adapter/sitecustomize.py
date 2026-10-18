"""Auto-register JetFire backend in spawned vLLM worker processes.

Also installs ``PR_SET_PDEATHSIG = SIGKILL`` for *every* Python
interpreter that imports this module (i.e. every process whose
``PYTHONPATH`` includes ``adapter/``).  When the parent dies (timeout
SIGTERM, OOM-killer, kill -9), the kernel immediately sends SIGKILL to
this process — no orphan EngineCore left holding GPU memory.

Why here and not in ``engine.py``: vLLM's EngineCore is started via
``multiprocessing.spawn``, which boots a *fresh Python interpreter* —
it does NOT inherit monkey-patches from the parent.  ``sitecustomize``
runs at every interpreter startup, so the death-signal hook applies to
both the bench-launched child and the spawn-launched grandchild.
"""

import os
import sys
from pathlib import Path


def _install_pdeathsig_self() -> None:
    """Make this very process die when its parent dies.

    Linux-only.  Idempotent — calling twice is harmless.  Failure is
    silent (non-Linux, missing libc, etc.) since the bench-side
    process-group cleanup is a sufficient fallback.
    """
    try:
        import ctypes
        _PR_SET_PDEATHSIG = 1
        _SIGKILL = 9
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.prctl(_PR_SET_PDEATHSIG, _SIGKILL, 0, 0, 0)
    except Exception:
        pass


_install_pdeathsig_self()


def _install_faulthandler() -> None:
    if os.environ.get("JETFIRE_FAULTHANDLER", "0") != "1":
        return
    try:
        import faulthandler
        import signal

        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass


_install_faulthandler()


_ROOT_DIR = Path(__file__).resolve().parents[2]
_expected_root = os.environ.get("JETFIRE_EXPECTED_ROOT")
if _expected_root and Path(_expected_root).resolve() != _ROOT_DIR:
    raise RuntimeError(
        "JetFire import root mismatch: "
        f"sitecustomize root={_ROOT_DIR} expected={Path(_expected_root).resolve()}"
    )

# Keep the release tree ahead of the process cwd and any stale PYTHONPATH
# entry.  This prevents a sibling checkout-level ``jetfire_vllm`` package from
# shadowing the release tree.
_root_str = str(_ROOT_DIR)
sys.path = [_root_str] + [p for p in sys.path if p != _root_str]

from jetfire_vllm.adapter.registry import ensure_jetfire_registered

ensure_jetfire_registered()

if os.environ.get("JETFIRE_PRINT_IMPORT_PATH", "0") == "1":
    import jetfire_vllm

    print(f"[jetfire] package={jetfire_vllm.__file__}", flush=True)
