"""Consistent checkpoint selection across single- and multi-model evaluations."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_PATH_ENV = {
    "llama31_8b": "JETFIRE_LLAMA31_8B",
    "llama33_70b": "JETFIRE_LLAMA33_70B",
    "qwen3_30b": "JETFIRE_QWEN3_30B",
    "qwen3_30b_a3b": "JETFIRE_QWEN3_30B",
}


def resolve_model_path(model_key: str, explicit_path: str = "", default: str = "") -> str:
    """An explicit path wins; a named model setting wins over generic fallback."""
    path = (
        explicit_path
        or os.environ.get(MODEL_PATH_ENV[model_key], "")
        or os.environ.get("JETFIRE_MODEL_PATH", "")
        or default
    )
    return str(Path(path).expanduser()) if path.startswith("~") else path


def validate_model_matrix(model_keys: list[str]) -> None:
    """Prevent one generic checkpoint from acquiring several model labels."""
    if len(set(model_keys)) > 1 and os.environ.get("JETFIRE_MODEL_PATH"):
        missing = sorted({MODEL_PATH_ENV[key] for key in model_keys if not os.environ.get(MODEL_PATH_ENV[key])})
        if missing:
            raise ValueError("Multi-model evaluation with JETFIRE_MODEL_PATH requires named paths: " + ", ".join(missing))
