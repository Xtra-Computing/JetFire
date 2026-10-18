#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUITE="${SCRIPT_DIR}/run_release_accuracy_suite.sh"

TOP_KS=(0 32 64 128 256 512 1024 2048)
OUT_DIR="/tmp/jetfire_topk_sensitivity_$(date +%Y%m%d_%H%M%S)"
MODEL="qwen3_30b_a3b"
BACKEND="jetfire_k3v2"
GROUP_SIZE=""
SUITE="longbench_v1"
GPUS="0,1"
MAX_SAMPLES="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash ./jetfire_vllm/bench/run_release_topk_sensitivity.sh [options]

Runs top-k sensitivity on a JetFire configuration.

Fixed knobs:
  --jetfire-graph

Swept top_k:
  0 (dense low-bit output), 32, 64, 128, 256, 512, 1024, 2048

Options:
  --model MODEL              default: qwen3_30b_a3b
  --backend BACKEND          default: jetfire_k3v2; e.g. jetfire_k2v2
  --jetfire-group-size N     default: 128 for K4V4, otherwise 64
  --suite SUITE              default: longbench_v1
  --gpus GPUS                default: 0,1
  --out-dir DIR              default: /tmp/jetfire_topk_sensitivity_<timestamp>
  --max-samples N            default: 0 full suite
  --tasks CSV                optional LongBench v1 task filter
  Any other options are forwarded to run_release_accuracy_suite.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --jetfire-group-size) GROUP_SIZE="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${GROUP_SIZE}" ]]; then
  GROUP_SIZE="64"
  if [[ "${BACKEND}" == "jetfire_k4v4" || "${BACKEND}" == "jetfire_int4" ]]; then GROUP_SIZE="128"; fi
fi

mkdir -p "${OUT_DIR}"
for top_k in "${TOP_KS[@]}"; do
  runtime_top_k="${top_k}"
  ablation_args=()
  if [[ "${top_k}" == "0" ]]; then
    runtime_top_k="256"
    ablation_args+=(--jetfire-approx-only)
  fi
  bash "${RUN_SUITE}" \
    --model "${MODEL}" \
    --backend "${BACKEND}" \
    --suite "${SUITE}" \
    --gpus "${GPUS}" \
    --out-dir "${OUT_DIR}/topk_${top_k}" \
    --max-samples "${MAX_SAMPLES}" \
    --jetfire-group-size "${GROUP_SIZE}" \
    --top-k "${runtime_top_k}" \
    --jetfire-graph \
    "${ablation_args[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "[topk-sensitivity] completed. Results: ${OUT_DIR}"
