#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_PY="${SCRIPT_DIR}/release_accuracy_eval.py"
CLEAN_PY="${SCRIPT_DIR}/clean_jetfire_runtime_artifacts.py"
PRECOMPILE_PY="${SCRIPT_DIR}/precompile_jetfire_extensions.py"

MODEL="all"
MODEL_PATH=""
BACKEND="all"
SUITE="all"
GPUS="0,1"
OUT_DIR="/tmp/jetfire_release_accuracy_$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES="0"
SAMPLE_START="0"
SAMPLE_END="0"
ONLY_OVER_MODEL_LEN="0"
MAX_MODEL_LEN="0"
SAVE_PREDICTIONS="0"
RESUME="1"
GPU_MEMORY_UTILIZATION="0.96"
TOP_K="256"
JETFIRE_GROUP_SIZE="${JETFIRE_GROUP_SIZE:-64}"
JETFIRE_PREFILL_CHUNK_TOKENS="${JETFIRE_PREFILL_CHUNK_TOKENS:-auto}"
CPU_ATTN_THREADS="32"
TENSOR_PARALLEL_SIZE="0"
KV_CACHE_BLOCK_SIZE="16"
JETFIRE_KV_BUDGET_SAFETY="1.0"
JETFIRE_NUM_GPU_BLOCKS_OVERRIDE="0"
NIAH_DEPTHS="0,10,20,30,40,50,60,70,80,90,100"
NIAH_REPEATS="10"
NIAH_CONTEXTS=""
NIAH_EDGE_MARGIN_PCT="5"
NIAH_FILLER_DATA_DIR="${JETFIRE_LONGBENCH_DATA_DIR:-/path/to/longbench_v1/data}"
NIAH_SAMPLE_SEED="42"
NIAH_CHAT_OVERHEAD_TOKENS="120"
TASKS=""
LONGBENCH_V1_DATA_DIR="${JETFIRE_LONGBENCH_DATA_DIR:-}"
LONGBENCH_V2_DATA_FILE="${JETFIRE_LONGBENCH_V2_DATA_FILE:-}"
JETFIRE_GRAPH="1"
JETFIRE_ASYNC_CPU_CORRECTION="0"
JETFIRE_APPROX_ONLY="0"
JETFIRE_RESET_RUNTIME_BETWEEN_SAMPLES="0"
CLEAN_RUNTIME_ARTIFACTS="1"
CLEAN_PURGE_EXTENSIONS="0"
PRECOMPILE_JETFIRE_EXTENSIONS="1"

usage() {
  cat <<'EOF'
Usage:
  bash ./jetfire_vllm/bench/run_release_accuracy_suite.sh [options]

Options:
  --model llama31_8b|llama33_70b|qwen3_30b_a3b|all
  --model-path PATH, overrides model path for a single --model
  --backend jetfire_k4v4|jetfire_k4v3|jetfire_k4v2|jetfire_k3v4|jetfire_k3v3|jetfire_k3v2|jetfire_k2v4|jetfire_k2v3|jetfire_k2v2|all|sensitivity_kv
  --suite longbench_v1|longbench_v2_128k|longbench_v2_256k|niah_32k|niah_128k|niah_256k|all
  --gpus CUDA_VISIBLE_DEVICES value, default: 0,1
  --out-dir DIR, default: /tmp/jetfire_release_accuracy_<timestamp>
  --max-samples N, default: 0 means full suite
  --sample-start N, zero-based sample start for debugging, default: 0
  --sample-end N, exclusive sample end for debugging, default: full
  --only-over-model-len N, LongBench v1 repair mode for old shorter windows
  --max-model-len N, optional override for vLLM max_model_len
  --top-k N, default: 256 for JetFire CPU correction
  --jetfire-group-size N, default: 64
  --jetfire-prefill-chunk-tokens auto|N, default: auto; use 0 to disable
  --cpu-attn-threads N, default: 32
  --tensor-parallel-size N, default: model-specific (1, or 2 for 70B)
  --gpu-memory-utilization FLOAT, default: 0.96
  --kv-cache-block-size N, default: 16
  --jetfire-kv-budget-safety FLOAT, default: 1.0
  --jetfire-num-gpu-blocks-override N, default: 0 auto-caps JetFire
      accuracy KV blocks to max_model_len / block_size; positive is explicit;
      negative leaves vLLM's planner uncapped.
  --niah-depths CSV, default: 0,10,...,100
  --niah-repeats N, default: 10
  --niah-contexts CSV, optional explicit NIAH context lengths
  --niah-edge-margin-pct N, map depths into [N,100-N], default: 5
  --niah-filler-data-dir DIR, LongBench v1 jsonl filler directory
  --niah-sample-seed N, deterministic NIAH sample seed, default: 42
  --niah-chat-overhead-tokens N, reserved chat-template overhead, default: 120
  --tasks CSV, optional LongBench v1 task filter, e.g. qasper
  --longbench-v1-data-dir DIR, official task.jsonl directory (or JETFIRE_LONGBENCH_DATA_DIR)
  --longbench-v2-data-file FILE, official JSON/JSONL data (or JETFIRE_LONGBENCH_V2_DATA_FILE)
  --jetfire-graph, use JetFire decode-only CUDA graph for accuracy runs (default)
  --jetfire-approx-only, k=0 accuracy ablation using dense low-bit output
  --jetfire-eager, disable JetFire decode-only CUDA graph
  --jetfire-async-cpu-correction, opt into async CPU correction in eager mode
  --jetfire-reset-runtime-between-samples, debug-only destructive JetFire
      runtime reset between samples (default off)
  --no-clean-runtime-artifacts, skip pre-launch JetFire stale lock cleanup
  --purge-jetfire-extensions, clean-rebuild JetFire torch extensions before run
  --no-precompile-jetfire-extensions, skip single-process JetFire extension preload
  --save-predictions
  --no-resume

Release matrix encoded by default:
  llama31_8b:
    LongBench v1, LongBench-v2 whole-subtask 128K, NIAH-128K
  llama33_70b:
    LongBench v1, NIAH-32K
  qwen3_30b_a3b:
    LongBench v1, LongBench-v2 whole-subtask 128K/256K, NIAH-128K/256K
  backends:
    JetFire K4V4, JetFire K3V2, JetFire K2V2.
    K2V* uses symmetric INT2 K.

Examples:
  # Quick smoke test, one sample per suite/backend/model:
  bash ./jetfire_vllm/bench/run_release_accuracy_suite.sh --max-samples 1

  # Full Qwen3 JetFire INT4 256K NIAH:
  bash ./jetfire_vllm/bench/run_release_accuracy_suite.sh \
    --model qwen3_30b_a3b --backend jetfire_int4 --suite niah_256k --gpus 0

  # Full release matrix:
  bash ./jetfire_vllm/bench/run_release_accuracy_suite.sh --gpus 0,1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --sample-start) SAMPLE_START="$2"; shift 2 ;;
    --sample-end) SAMPLE_END="$2"; shift 2 ;;
    --only-over-model-len) ONLY_OVER_MODEL_LEN="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --jetfire-group-size) JETFIRE_GROUP_SIZE="$2"; shift 2 ;;
    --jetfire-prefill-chunk-tokens) JETFIRE_PREFILL_CHUNK_TOKENS="$2"; shift 2 ;;
    --cpu-attn-threads) CPU_ATTN_THREADS="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --kv-cache-block-size) KV_CACHE_BLOCK_SIZE="$2"; shift 2 ;;
    --jetfire-kv-budget-safety) JETFIRE_KV_BUDGET_SAFETY="$2"; shift 2 ;;
    --jetfire-num-gpu-blocks-override) JETFIRE_NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
    --niah-depths) NIAH_DEPTHS="$2"; shift 2 ;;
    --niah-repeats) NIAH_REPEATS="$2"; shift 2 ;;
    --niah-contexts) NIAH_CONTEXTS="$2"; shift 2 ;;
    --niah-edge-margin-pct) NIAH_EDGE_MARGIN_PCT="$2"; shift 2 ;;
    --niah-filler-data-dir) NIAH_FILLER_DATA_DIR="$2"; shift 2 ;;
    --niah-sample-seed) NIAH_SAMPLE_SEED="$2"; shift 2 ;;
    --niah-chat-overhead-tokens) NIAH_CHAT_OVERHEAD_TOKENS="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --longbench-v1-data-dir) LONGBENCH_V1_DATA_DIR="$2"; shift 2 ;;
    --longbench-v2-data-file) LONGBENCH_V2_DATA_FILE="$2"; shift 2 ;;
    --jetfire-graph) JETFIRE_GRAPH="1"; shift ;;
    --jetfire-approx-only) JETFIRE_APPROX_ONLY="1"; shift ;;
    --jetfire-eager) JETFIRE_GRAPH="0"; shift ;;
    --jetfire-async-cpu-correction) JETFIRE_ASYNC_CPU_CORRECTION="1"; shift ;;
    --jetfire-reset-runtime-between-samples) JETFIRE_RESET_RUNTIME_BETWEEN_SAMPLES="1"; shift ;;
    --no-clean-runtime-artifacts) CLEAN_RUNTIME_ARTIFACTS="0"; shift ;;
    --purge-jetfire-extensions) CLEAN_PURGE_EXTENSIONS="1"; shift ;;
    --no-precompile-jetfire-extensions) PRECOMPILE_JETFIRE_EXTENSIONS="0"; shift ;;
    --save-predictions) SAVE_PREDICTIONS="1"; shift ;;
    --no-resume) RESUME="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${MODEL}" == "all" && -n "${MODEL_PATH}" ]]; then
  echo "--model-path requires a single --model." >&2
  exit 2
fi
if [[ "${MODEL}" == "all" && -n "${JETFIRE_MODEL_PATH:-}" ]]; then
  if [[ -z "${JETFIRE_LLAMA31_8B:-}" || -z "${JETFIRE_LLAMA33_70B:-}" || -z "${JETFIRE_QWEN3_30B:-}" ]]; then
    echo "For --model all, configure all three per-model paths or unset JETFIRE_MODEL_PATH to avoid labeling one checkpoint as multiple models." >&2
    exit 2
  fi
fi

if [[ "${MODEL}" == "all" ]]; then
  MODELS=(llama31_8b llama33_70b qwen3_30b_a3b)
else
  MODELS=("${MODEL}")
fi

if [[ "${BACKEND}" == "all" ]]; then
  BACKENDS=(jetfire_k4v4 jetfire_k3v2 jetfire_k2v2)
elif [[ "${BACKEND}" == "sensitivity_kv" ]]; then
  BACKENDS=(
    jetfire_k4v4
    jetfire_k4v3
    jetfire_k4v2
    jetfire_k3v4
    jetfire_k3v3
    jetfire_k3v2
    jetfire_k2v3
    jetfire_k2v4
    jetfire_k2v2
  )
else
  BACKENDS=("${BACKEND}")
fi

backend_mode() {
  case "$1" in
    jetfire_k4v4|jetfire_int4)
      echo "SymInt4K_SymInt4V"
      ;;
    jetfire_k4v3)
      echo "SymInt4K_SymInt3V"
      ;;
    jetfire_k4v2)
      echo "SymInt4K_SymInt2V"
      ;;
    jetfire_k3v3)
      echo "SymInt3K_SymInt3V"
      ;;
    jetfire_k3v4)
      echo "SymInt3K_SymInt4V"
      ;;
    jetfire_k3v2)
      echo "SymInt3K_SymInt2V"
      ;;
    jetfire_k2v2|jetfire_int2)
      echo "SymInt2K_SymInt2V"
      ;;
    jetfire_k2v3)
      echo "SymInt2K_SymInt3V"
      ;;
    jetfire_k2v4)
      echo "SymInt2K_SymInt4V"
      ;;
    jetfire_k2v2_4level)
      echo "AsymInt2K4Level_SymInt2V"
      ;;
    jetfire_sym_k2v2)
      echo "SymInt2K_SymInt2V"
      ;;
    jetfire_sym_k2v3)
      echo "SymInt2K_SymInt3V"
      ;;
    jetfire_sym_k2v4)
      echo "SymInt2K_SymInt4V"
      ;;
    *)
      echo "Unknown backend: $1" >&2
      exit 2
      ;;
  esac
}

model_suites() {
  local model="$1"
  if [[ "${SUITE}" != "all" ]]; then
    echo "${SUITE}"
    return
  fi
  if [[ "${model}" == "llama31_8b" ]]; then
    echo "longbench_v1 longbench_v2_128k niah_128k"
  elif [[ "${model}" == "llama33_70b" ]]; then
    echo "longbench_v1 niah_32k"
  elif [[ "${model}" == "qwen3_30b_a3b" ]]; then
    echo "longbench_v1 longbench_v2_128k longbench_v2_256k niah_128k niah_256k"
  else
    echo "Unknown model: ${model}" >&2
    exit 2
  fi
}

mkdir -p "${OUT_DIR}"
echo "[release-accuracy] out_dir=${OUT_DIR}"
echo "[release-accuracy] gpus=${GPUS}"
echo "[release-accuracy] tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
echo "[release-accuracy] jetfire_graph=${JETFIRE_GRAPH}"
echo "[release-accuracy] jetfire_group_size=${JETFIRE_GROUP_SIZE}"
echo "[release-accuracy] jetfire_prefill_chunk_tokens=${JETFIRE_PREFILL_CHUNK_TOKENS}"
echo "[release-accuracy] models=${MODELS[*]}"
echo "[release-accuracy] backends=${BACKENDS[*]}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export JETFIRE_EXPECTED_ROOT="${ROOT}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/jetfire_release_torch_extensions}"
if [[ "${JETFIRE_ALLOW_INHERITED_PYTHONPATH:-0}" == "1" ]]; then
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}"
fi

if [[ "${CLEAN_RUNTIME_ARTIFACTS}" == "1" ]]; then
  clean_cmd=("${PYTHON_BIN}" "${CLEAN_PY}" --min-lock-age-s 0)
  if [[ "${CLEAN_PURGE_EXTENSIONS}" == "1" ]]; then
    clean_cmd+=(--purge-extensions)
  fi
  "${clean_cmd[@]}"
fi

needs_jetfire="0"
jetfire_modes=()
for backend_name in "${BACKENDS[@]}"; do
  mode_name="$(backend_mode "${backend_name}")"
  if [[ -n "${mode_name}" ]]; then
    needs_jetfire="1"
    jetfire_modes+=("${mode_name}")
  fi
done
if [[ "${needs_jetfire}" == "1" && "${PRECOMPILE_JETFIRE_EXTENSIONS}" == "1" ]]; then
  modes_csv="$(printf '%s\n' "${jetfire_modes[@]}" | awk '!seen[$0]++' | paste -sd, -)"
  "${PYTHON_BIN}" "${PRECOMPILE_PY}" \
    --modes "${modes_csv}" \
    --top-k "${TOP_K}" \
    --group-size "${JETFIRE_GROUP_SIZE}" \
    --cpu-attn-threads "${CPU_ATTN_THREADS}"
fi

for model in "${MODELS[@]}"; do
  read -r -a suites <<< "$(model_suites "${model}")"
  for suite_name in "${suites[@]}"; do
    for backend_name in "${BACKENDS[@]}"; do
      echo
      echo "======================================================================"
      echo "[release-accuracy] model=${model} backend=${backend_name} suite=${suite_name}"
      echo "======================================================================"
      cmd=(
        "${PYTHON_BIN}" "${EVAL_PY}"
        --model-key "${model}"
        --backend "${backend_name}"
        --suite "${suite_name}"
        --out-dir "${OUT_DIR}"
        --max-samples "${MAX_SAMPLES}"
        --sample-start "${SAMPLE_START}"
        --sample-end "${SAMPLE_END}"
        --only-over-model-len "${ONLY_OVER_MODEL_LEN}"
        --max-model-len "${MAX_MODEL_LEN}"
        --top-k "${TOP_K}"
        --jetfire-group-size "${JETFIRE_GROUP_SIZE}"
        --jetfire-prefill-chunk-tokens "${JETFIRE_PREFILL_CHUNK_TOKENS}"
        --cpu-attn-threads "${CPU_ATTN_THREADS}"
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --kv-cache-block-size "${KV_CACHE_BLOCK_SIZE}"
        --jetfire-kv-budget-safety "${JETFIRE_KV_BUDGET_SAFETY}"
        --jetfire-num-gpu-blocks-override "${JETFIRE_NUM_GPU_BLOCKS_OVERRIDE}"
        --niah-depths "${NIAH_DEPTHS}"
        --niah-repeats "${NIAH_REPEATS}"
        --niah-edge-margin-pct "${NIAH_EDGE_MARGIN_PCT}"
        --niah-filler-data-dir "${NIAH_FILLER_DATA_DIR}"
        --niah-sample-seed "${NIAH_SAMPLE_SEED}"
        --niah-chat-overhead-tokens "${NIAH_CHAT_OVERHEAD_TOKENS}"
      )
      if [[ -n "${MODEL_PATH}" ]]; then
        cmd+=(--model-path "${MODEL_PATH}")
      fi
      if [[ -n "${NIAH_CONTEXTS}" ]]; then
        cmd+=(--niah-contexts "${NIAH_CONTEXTS}")
      fi
      if [[ -n "${LONGBENCH_V1_DATA_DIR}" ]]; then
        cmd+=(--longbench-v1-data-dir "${LONGBENCH_V1_DATA_DIR}")
      fi
      if [[ -n "${LONGBENCH_V2_DATA_FILE}" ]]; then
        cmd+=(--longbench-v2-data-file "${LONGBENCH_V2_DATA_FILE}")
      fi
      if [[ -n "${TASKS}" ]]; then
        cmd+=(--tasks "${TASKS}")
      fi
      if [[ "${RESUME}" == "1" ]]; then
        cmd+=(--resume)
      fi
      if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
        cmd+=(--save-predictions)
      fi
      if [[ "${JETFIRE_GRAPH}" == "1" ]]; then
        cmd+=(--jetfire-graph)
      fi
      if [[ "${JETFIRE_APPROX_ONLY}" == "1" ]]; then
        cmd+=(--jetfire-approx-only)
      fi
      if [[ "${JETFIRE_ASYNC_CPU_CORRECTION}" == "1" ]]; then
        cmd+=(--jetfire-async-cpu-correction)
      fi
      if [[ "${JETFIRE_RESET_RUNTIME_BETWEEN_SAMPLES}" == "1" ]]; then
        cmd+=(--jetfire-reset-runtime-between-samples)
      fi
      "${cmd[@]}"
    done
  done
done

echo
echo "[release-accuracy] completed. Results: ${OUT_DIR}"
