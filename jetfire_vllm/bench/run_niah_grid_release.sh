#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUITE="${SCRIPT_DIR}/run_release_accuracy_suite.sh"

OUT_DIR="${OUT_DIR:-/tmp/jetfire_niah_grid_$(date +%Y%m%d_%H%M%S)}"
PHASE="all"
GPUS_LLAMA31="0"
GPUS_QWEN3="0"
GPUS_LLAMA33="0,1"
REPEATS="10"
DEPTHS="0,10,20,30,40,50,60,70,80,90,100"
EDGE_MARGIN_PCT="5"
TOP_K="256"
CPU_ATTN_THREADS="32"
JETFIRE_GRAPH="1"

usage() {
  cat <<'EOF'
Usage:
  bash ./jetfire_vllm/bench/run_niah_grid_release.sh [options]

Options:
  --phase all|llama31|qwen3|llama33
  --out-dir DIR
  --gpus-llama31 GPU_LIST, default 0
  --gpus-qwen3 GPU_LIST, default 0
  --gpus-llama33 GPU_LIST, default 0,1
  --repeats N, default 10
  --edge-margin-pct N, default 5
  --jetfire-eager, disable JetFire decode-only graph for debugging

Experiment definition:
  depths: requested 0,10,...,100 and mapped into [5,95] by default.
  target tokens for Llama: 16000,32000,64000,125000.
  target tokens for Qwen: 16000,32000,64000,125000,250000.
  These decimal targets leave generation headroom; they are not exact
  128 Ki-token / 256 Ki-token prompts. Each result records actual prompt_tokens.
  Main K4V4 uses group size 128; other configurations use group size 64.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --gpus-llama31) GPUS_LLAMA31="$2"; shift 2 ;;
    --gpus-qwen3) GPUS_QWEN3="$2"; shift 2 ;;
    --gpus-llama33) GPUS_LLAMA33="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --edge-margin-pct) EDGE_MARGIN_PCT="$2"; shift 2 ;;
    --jetfire-graph) JETFIRE_GRAPH="1"; shift ;;
    --jetfire-eager) JETFIRE_GRAPH="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${PHASE}" == "all" && -n "${JETFIRE_MODEL_PATH:-}" ]]; then
  if [[ -z "${JETFIRE_LLAMA31_8B:-}" || -z "${JETFIRE_LLAMA33_70B:-}" || -z "${JETFIRE_QWEN3_30B:-}" ]]; then
    echo "For --phase all, configure all three per-model paths or unset JETFIRE_MODEL_PATH." >&2
    exit 2
  fi
fi

mkdir -p "${OUT_DIR}/logs"

run_one() {
  local model="$1"
  local backend="$2"
  local suite="$3"
  local contexts="$4"
  local gpus="$5"
  local tp="$6"
  local log="${OUT_DIR}/logs/${model}_${backend}_${suite}.log"
  local group_size="64"
  if [[ "${backend}" == "jetfire_k4v4" ]]; then
    group_size="128"
  fi
  local graph_flag="--jetfire-eager"
  if [[ "${JETFIRE_GRAPH}" == "1" ]]; then
    graph_flag="--jetfire-graph"
  fi

  echo "[niah-grid] model=${model} backend=${backend} suite=${suite} contexts=${contexts} gpus=${gpus} tp=${tp} group_size=${group_size}"
  bash "${RUN_SUITE}" \
    --model "${model}" \
    --backend "${backend}" \
    --suite "${suite}" \
    --gpus "${gpus}" \
    --tensor-parallel-size "${tp}" \
    --top-k "${TOP_K}" \
    --jetfire-group-size "${group_size}" \
    --cpu-attn-threads "${CPU_ATTN_THREADS}" \
    --out-dir "${OUT_DIR}" \
    --niah-contexts "${contexts}" \
    --niah-depths "${DEPTHS}" \
    --niah-repeats "${REPEATS}" \
    --niah-edge-margin-pct "${EDGE_MARGIN_PCT}" \
    --save-predictions \
    "${graph_flag}" \
    2>&1 | tee "${log}"
}

run_llama31() {
  local contexts="16000,32000,64000,125000"
  local backends=(jetfire_k4v4 jetfire_k4v2 jetfire_k3v2 jetfire_k2v2)
  for backend in "${backends[@]}"; do
    run_one llama31_8b "${backend}" niah_128k "${contexts}" "${GPUS_LLAMA31}" 1
  done
}

run_qwen3() {
  local contexts_256="16000,32000,64000,125000,250000"
  local backends=(jetfire_k4v4 jetfire_k4v2 jetfire_k3v2 jetfire_k2v2)
  for backend in "${backends[@]}"; do
    run_one qwen3_30b_a3b "${backend}" niah_256k "${contexts_256}" "${GPUS_QWEN3}" 1
  done
}

run_llama33() {
  local contexts="16000,32000,64000,125000"
  local backends=(jetfire_k4v4 jetfire_k4v2 jetfire_k3v2 jetfire_k2v2)
  for backend in "${backends[@]}"; do
    run_one llama33_70b "${backend}" niah_128k "${contexts}" "${GPUS_LLAMA33}" 2
  done
}

case "${PHASE}" in
  all)
    run_llama31
    run_qwen3
    run_llama33
    ;;
  llama31) run_llama31 ;;
  qwen3) run_qwen3 ;;
  llama33) run_llama33 ;;
  *) echo "Unknown phase: ${PHASE}" >&2; usage >&2; exit 2 ;;
esac

echo "[niah-grid] completed out_dir=${OUT_DIR}"
