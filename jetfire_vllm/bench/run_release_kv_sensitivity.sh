#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUITE="${SCRIPT_DIR}/run_release_accuracy_suite.sh"

BACKENDS=(
  jetfire_k4v4
  jetfire_k4v3
  jetfire_k4v2
  jetfire_k3v4
  jetfire_k3v3
  jetfire_k3v2
  jetfire_k2v4
  jetfire_k2v3
  jetfire_k2v2
)

OUT_DIR="/tmp/jetfire_kv_sensitivity_$(date +%Y%m%d_%H%M%S)"
MODEL="qwen3_30b_a3b"
SUITE="longbench_v1"
GPUS="0,1"
MAX_SAMPLES="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash ./jetfire_vllm/bench/run_release_kv_sensitivity.sh [options]

Runs the release KV-bit sensitivity matrix with one path:
  K4V4, K4V3, K4V2, K3V4, K3V3, K3V2, K2V4, K2V3, K2V2

Fixed knobs:
  --jetfire-group-size 64
  --top-k 256
  --jetfire-graph

Options:
  --model MODEL              default: qwen3_30b_a3b
  --suite SUITE              default: longbench_v1
  --gpus GPUS                default: 0,1
  --out-dir DIR              default: /tmp/jetfire_kv_sensitivity_<timestamp>
  --max-samples N            default: 0 full suite
  --tasks CSV                optional LongBench v1 task filter
  Any other options are forwarded to run_release_accuracy_suite.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "${OUT_DIR}"
for backend in "${BACKENDS[@]}"; do
  bash "${RUN_SUITE}" \
    --model "${MODEL}" \
    --backend "${backend}" \
    --suite "${SUITE}" \
    --gpus "${GPUS}" \
    --out-dir "${OUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --jetfire-group-size 64 \
    --top-k 256 \
    --jetfire-graph \
    "${EXTRA_ARGS[@]}"
done

echo "[kv-sensitivity] completed. Results: ${OUT_DIR}"
