#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/jetfire_memory_$(date +%Y%m%d_%H%M%S)}"
OUT_CSV="${OUT_DIR}/memory_breakdown.csv"
MODE="${MODE:-computed-only}"
GPUS="${GPUS:-0,1}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_POST_WORKER_RPC="${SKIP_POST_WORKER_RPC:-0}"
if [[ "${MODE}" == "run" ]]; then
  COLLECT_INIT_WORKER_RPC="${COLLECT_INIT_WORKER_RPC:-1}"
  RESET_PEAK_BEFORE_GENERATE="${RESET_PEAK_BEFORE_GENERATE:-1}"
else
  COLLECT_INIT_WORKER_RPC="${COLLECT_INIT_WORKER_RPC:-0}"
  RESET_PEAK_BEFORE_GENERATE="${RESET_PEAK_BEFORE_GENERATE:-0}"
fi
ALLOW_OVER_WINDOW="${ALLOW_OVER_WINDOW:-0}"
JETFIRE_GRAPH="${JETFIRE_GRAPH:-0}"
JETFIRE_PREFILL_CHUNK_TOKENS="${JETFIRE_PREFILL_CHUNK_TOKENS:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"

mkdir -p "${OUT_DIR}"
if [[ -s "${OUT_CSV}" ]]; then
  echo "Refusing to append duplicate measurements to ${OUT_CSV}; choose a new OUT_DIR." >&2
  exit 2
fi

MODELS="${MODELS:-llama31_8b,qwen3_30b_a3b,llama33_70b}"
CONFIGS="${CONFIGS:-k4v4,k3v2}"
CTX_LIST="${CTX_LIST:-4096,8192,16384,32768,65536,131072,196608,256000}"

IFS=',' read -ra MODEL_ARR <<< "${MODELS}"
if [[ "${#MODEL_ARR[@]}" -gt 1 && -n "${JETFIRE_MODEL_PATH:-}" ]]; then
  for model_key in "${MODEL_ARR[@]}"; do
    case "${model_key}" in
      llama31_8b) model_var="JETFIRE_LLAMA31_8B" ;;
      llama33_70b) model_var="JETFIRE_LLAMA33_70B" ;;
      qwen3_30b_a3b) model_var="JETFIRE_QWEN3_30B" ;;
      *) echo "Unknown model: ${model_key}" >&2; exit 2 ;;
    esac
    if [[ -z "${!model_var:-}" ]]; then
      echo "Multi-model evaluation with JETFIRE_MODEL_PATH requires ${model_var}." >&2
      exit 2
    fi
  done
fi
IFS=',' read -ra CONFIG_ARR <<< "${CONFIGS}"
IFS=',' read -ra CTX_ARR <<< "${CTX_LIST}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
if [[ "${MODE}" == "run" ]]; then
  export PYTHONPATH="${ROOT}:${ROOT}/jetfire_vllm/adapter:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
fi
export JETFIRE_EXPECTED_ROOT="${ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export JETFIRE_CPU_ATTN_THREADS="${JETFIRE_CPU_ATTN_THREADS:-32}"
if [[ "${ALLOW_OVER_WINDOW}" == "1" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
EXTRA_ARGS=()
if [[ "${SKIP_GENERATE}" == "1" ]]; then EXTRA_ARGS+=(--skip-generate); fi
if [[ "${SKIP_POST_WORKER_RPC}" == "1" ]]; then EXTRA_ARGS+=(--skip-post-worker-rpc); fi
if [[ "${COLLECT_INIT_WORKER_RPC}" == "1" ]]; then EXTRA_ARGS+=(--collect-init-worker-rpc); fi
if [[ "${RESET_PEAK_BEFORE_GENERATE}" == "1" ]]; then EXTRA_ARGS+=(--reset-peak-before-generate); fi
if [[ "${ALLOW_OVER_WINDOW}" == "1" ]]; then EXTRA_ARGS+=(--allow-over-window); fi
if [[ "${JETFIRE_GRAPH}" == "1" ]]; then EXTRA_ARGS+=(--jetfire-graph); fi
EXTRA_ARGS+=(--jetfire-prefill-chunk-tokens "${JETFIRE_PREFILL_CHUNK_TOKENS}")

for model in "${MODEL_ARR[@]}"; do
  for config in "${CONFIG_ARR[@]}"; do
    for ctx in "${CTX_ARR[@]}"; do
      echo "[memory-breakdown] model=${model} config=${config} ctx=${ctx} mode=${MODE}"
      "${PYTHON_BIN}" "${ROOT}/./jetfire_vllm/bench/memory_breakdown_probe.py" \
        --model "${model}" \
        --config "${config}" \
        --ctx "${ctx}" \
        --mode "${MODE}" \
        --out-csv "${OUT_CSV}" \
        --gpus "${GPUS}" \
        --output-len "${OUTPUT_LEN}" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

"${PYTHON_BIN}" "${ROOT}/./jetfire_vllm/bench/summarize_memory_breakdown.py" \
  --csv "${OUT_CSV}" \
  --out-dir "${OUT_DIR}"

echo "[memory-breakdown] done: ${OUT_DIR}"
