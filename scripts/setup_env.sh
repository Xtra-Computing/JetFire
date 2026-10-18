#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [ENV_NAME]

Create a fresh Conda environment (default: jetfire-ae) with Python 3.11.17
and install JetFire with its pinned CUDA 12.8 dependency constraints.
Requires Conda and the system CUDA compiler described in the README.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

ENV_NAME="${1-jetfire-ae}"
if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$ENV_NAME" == "base" || "$ENV_NAME" == "root" ]]; then
  echo "Choose an environment name using letters, digits, dots, underscores or hyphens; base and root are reserved." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-$(type -P conda || true)}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Conda was not found. Install Conda and add it to PATH, or set CONDA_EXE to its executable." >&2
  exit 2
fi

# Conda create --yes can replace an existing prefix. Check every named-env
# directory first, including directories that are not registered environments.
CONDA_BASE="$("$CONDA_BIN" info --base)"
CONDA_INFO="$("$CONDA_BIN" info --json)"
"$CONDA_BASE/bin/python" -c '
import json
from pathlib import Path
import sys
name = sys.argv[1]
info = json.load(sys.stdin)
directories = info["envs_dirs"]
if not directories:
    raise SystemExit("Conda did not report any environment directories.")
if any((Path(directory) / name).exists() for directory in directories):
    raise SystemExit(f"Environment or directory {name!r} already exists. Choose a different environment name.")
' "$ENV_NAME" <<< "$CONDA_INFO"

"$CONDA_BIN" create --yes --name "$ENV_NAME" --no-default-packages python=3.11.17 pip
"$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --constraint "$ROOT/env/constraints-py311-cu128.txt" --editable "$ROOT[plots]"
"$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" python -m pip check

printf '\nEnvironment ready. Activate it with:\n  conda activate %s\n' "$ENV_NAME"
printf 'Then configure the checkpoint/data paths and run the benchmarks in README.md.\n'
