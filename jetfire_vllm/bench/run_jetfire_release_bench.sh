#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PRESET="release"
MODEL="${JETFIRE_MODEL_PATH:-${JETFIRE_LLAMA31_8B:-/path/to/Llama-3.1-8B-Instruct}}"
GPU="0"
CTX="64000"
OUTPUT_LEN="32"
CONCURRENCY="1"
TOP_K="256"
CPU_ATTN_THREADS="32"
TP="1"
GPU_MEMORY_UTILIZATION="0.96"
JETFIRE_MODE="SymInt3K_SymInt2V"
JETFIRE_GROUP_SIZE="64"
JETFIRE_KV_CACHE_BUDGET="auto"
JETFIRE_PREFILL_CHUNK_TOKENS=""
OUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_jetfire_release_bench.sh [options] [-- extra official_serve_benchmark.py args]

Options:
  --preset release|stable|fast release=policy-owned selectors; stable/fast are debug only
  --model PATH
  --gpu LIST
  --ctx N
  --output-len N
  --concurrency N
  --tensor-parallel-size N
  --gpu-memory-utilization FLOAT  default: 0.96
  --jetfire-mode MODE
  --jetfire-group-size N
  --jetfire-kv-cache-budget auto|vllm|reserve
  --jetfire-prefill-chunk-tokens auto|N  default: auto; use 0/off to disable
  --top-k N
  --cpu-attn-threads N
  --out-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu|--gpus) GPU="$2"; shift 2 ;;
    --ctx|--ctx-len|--context-length) CTX="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --concurrency|--batch|--batch-size) CONCURRENCY="$2"; shift 2 ;;
    --tensor-parallel-size|--tp) TP="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --jetfire-mode) JETFIRE_MODE="$2"; shift 2 ;;
    --jetfire-group-size) JETFIRE_GROUP_SIZE="$2"; shift 2 ;;
    --jetfire-kv-cache-budget) JETFIRE_KV_CACHE_BUDGET="$2"; shift 2 ;;
    --jetfire-prefill-chunk-tokens) JETFIRE_PREFILL_CHUNK_TOKENS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --cpu-attn-threads) CPU_ATTN_THREADS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "${PRESET}" in
  release) INT8TC="auto"; PROFILE="graph" ;;
  stable) INT8TC="off"; PROFILE="eager" ;;
  fast) INT8TC="on"; PROFILE="eager" ;;
  *) echo "invalid --preset ${PRESET}" >&2; exit 2 ;;
esac

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="/tmp/jetfire_release_${PRESET}_ctx${CTX}_b${CONCURRENCY}_o${OUTPUT_LEN}_$(date +%Y%m%d_%H%M%S)"
fi

cd "${ROOT}"
if [[ "${JETFIRE_ALLOW_INHERITED_PYTHONPATH:-0}" == "1" ]]; then
  export PYTHONPATH="${ROOT}/jetfire_vllm/adapter:${ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}/jetfire_vllm/adapter:${ROOT}"
fi
export JETFIRE_EXPECTED_ROOT="${ROOT}"
export JETFIRE_CPU_ATTN_THREADS="${CPU_ATTN_THREADS}"
export OMP_NUM_THREADS="${CPU_ATTN_THREADS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/jetfire_release_torch_extensions}"

echo "[jetfire-release-bench] root=${ROOT}"
echo "[jetfire-release-bench] preset=${PRESET} int8tc=${INT8TC} gpu=${GPU} model=${MODEL}"
echo "[jetfire-release-bench] ctx=${CTX} output_len=${OUTPUT_LEN} concurrency=${CONCURRENCY} tp=${TP} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} top_k=${TOP_K} mode=${JETFIRE_MODE} group_size=${JETFIRE_GROUP_SIZE} kv_budget=${JETFIRE_KV_CACHE_BUDGET} prefill_chunk_tokens=${JETFIRE_PREFILL_CHUNK_TOKENS:-auto}"
echo "[jetfire-release-bench] torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/official_serve_benchmark.py" \
  --mode fixed-concurrency \
  --backend jetfire \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --tensor-parallel-size "${TP}" \
  --ctx "${CTX}" \
  --output-len "${OUTPUT_LEN}" \
  --concurrency "${CONCURRENCY}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --num-prompts "${CONCURRENCY}" \
  --runs 1 \
  --warmups 1 \
  --chunked-prefill off \
  --jetfire-profile "${PROFILE}" \
  --jetfire-graph-scope decode-only \
  --jetfire-mode "${JETFIRE_MODE}" \
  --jetfire-group-size "${JETFIRE_GROUP_SIZE}" \
  --jetfire-kv-cache-budget "${JETFIRE_KV_CACHE_BUDGET}" \
  --jetfire-int8tc "${INT8TC}" \
  --jetfire-zero-init-cache off \
  --top-k "${TOP_K}" \
  --cpu-attn-threads "${CPU_ATTN_THREADS}" \
  --out-dir "${OUT_DIR}"
)
if [[ -n "${JETFIRE_PREFILL_CHUNK_TOKENS}" ]]; then
  CMD+=(--jetfire-prefill-chunk-tokens "${JETFIRE_PREFILL_CHUNK_TOKENS}")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"

echo "[jetfire-release-bench] summary=${OUT_DIR}/summary.json"
