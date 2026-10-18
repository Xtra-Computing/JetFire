#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PRESET="release"
MODES="SymInt4K_SymInt4V,SymInt4K_SymInt3V,SymInt4K_SymInt2V,SymInt3K_SymInt4V,SymInt3K_SymInt3V,SymInt3K_SymInt2V,SymInt2K_SymInt4V,SymInt2K_SymInt3V,SymInt2K_SymInt2V"
TOP_K="256"
GROUP_SIZE="64"
CPU_ATTN_THREADS="32"
PURGE=0

usage() {
  cat <<'EOF'
Usage: compile_jetfire_release.sh [options]

Options:
  --preset release|stable|fast|all
                               release=policy-owned selectors; stable/fast are
                               debug-only overrides; all precompiles both overrides
  --modes LIST                 comma-separated JETFIRE_MODE list
                               default covers release + sensitivity modes
  --top-k N                    top-k used by decode path
  --group-size N               JetFire KV quantization group size
  --cpu-attn-threads N         CPU correction worker threads
  --purge-extensions           remove JetFire torch extension dirs before build
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --cpu-attn-threads) CPU_ATTN_THREADS="$2"; shift 2 ;;
    --purge-extensions) PURGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
if [[ "${JETFIRE_ALLOW_INHERITED_PYTHONPATH:-0}" == "1" ]]; then
  export PYTHONPATH="${ROOT}/jetfire_vllm/adapter:${ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}/jetfire_vllm/adapter:${ROOT}"
fi
export JETFIRE_EXPECTED_ROOT="${ROOT}"
export JETFIRE_CPU_ATTN_THREADS="${CPU_ATTN_THREADS}"
export OMP_NUM_THREADS="${CPU_ATTN_THREADS}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/jetfire_release_torch_extensions}"

CLEAN_ARGS=(--min-lock-age-s 0)
if [[ "${PURGE}" == "1" ]]; then
  CLEAN_ARGS+=(--purge-extensions)
fi

echo "[jetfire-release-compile] root=${ROOT}"
echo "[jetfire-release-compile] preset=${PRESET} modes=${MODES} top_k=${TOP_K} group_size=${GROUP_SIZE} cpu_threads=${CPU_ATTN_THREADS}"
echo "[jetfire-release-compile] torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/clean_jetfire_runtime_artifacts.py" "${CLEAN_ARGS[@]}"

run_precompile() {
  local int8tc="$1"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/precompile_jetfire_extensions.py" \
    --modes "${MODES}" \
    --top-k "${TOP_K}" \
    --group-size "${GROUP_SIZE}" \
    --cpu-attn-threads "${CPU_ATTN_THREADS}" \
    --int8tc "${int8tc}"
}

case "${PRESET}" in
  release)
    run_precompile auto
    ;;
  stable)
    run_precompile off
    ;;
  fast)
    run_precompile on
    ;;
  all)
    run_precompile off
    run_precompile on
    ;;
  *)
    echo "invalid --preset ${PRESET}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" "${SCRIPT_DIR}/clean_jetfire_runtime_artifacts.py" --min-lock-age-s 0
echo "[jetfire-release-compile] done"
