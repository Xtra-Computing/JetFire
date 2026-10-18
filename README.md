# JetFire

JetFire combines low-bit GPU attention with high-precision CPU top-k correction.

[Paper](https://drive.google.com/file/d/1AOmO2idWrcDd-OW__JfwXeb1o92EPzEq/view)
· [Detailed instructions](docs/ARTIFACT_EVALUATION.md)
· [Model and dataset downloads](docs/MODELS_AND_DATA.md)

## Setup

Use Linux x86-64, Conda, CUDA 12.8 with `nvcc`, and a C++17 compiler.
Llama 8B and Qwen need one H100 80 GB GPU; Llama 70B uses two GPUs (TP=2).
See the [hardware guide](docs/ARTIFACT_EVALUATION.md#preparation-and-execution-record)
for CPU and RAM requirements.

Run from the repository root, using a new environment name:

```bash
bash scripts/setup_env.sh jetfire-ae
conda activate jetfire-ae
export PYTHON_BIN=python
export JETFIRE_WORK_DIR=/absolute/path/to/jetfire-work
export TORCH_EXTENSIONS_DIR="$JETFIRE_WORK_DIR/torch_extensions"
export MAX_JOBS=8
mkdir -p "$JETFIRE_WORK_DIR"
```

The setup script installs the pinned dependencies and checks the environment.

## Models and data

Download checkpoints separately:

| Model | Download | Path variable |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | [Meta](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `JETFIRE_LLAMA31_8B` |
| Qwen3-30B-A3B-Instruct-2507 | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `JETFIRE_QWEN3_30B` |
| Llama-3.3-70B-Instruct | [Meta](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | `JETFIRE_LLAMA33_70B` |

Follow the [download guide](docs/MODELS_AND_DATA.md) for model access, pinned
revisions and LongBench v1 data, then set your local paths:

```bash
export JETFIRE_LLAMA31_8B=/absolute/path/to/Llama-3.1-8B-Instruct
export JETFIRE_QWEN3_30B=/absolute/path/to/Qwen3-30B-A3B-Instruct-2507
export JETFIRE_LONGBENCH_DATA_DIR=/absolute/path/to/longbench_v1/data
```

## Performance

Choose the prefill length, decode length and batch size:

```bash
PREFILL_LEN=32768
DECODE_LEN=128
BATCH_SIZE=4

python -m jetfire_vllm.bench.benchmark \
  --models llama31_8b,qwen3_30b_a3b --single-gpus 0 --execute \
  --prefill-len "$PREFILL_LEN" --decode-len "$DECODE_LEN" \
  --batches "$BATCH_SIZE" --runs 3 \
  --out-dir "$JETFIRE_WORK_DIR/performance_p${PREFILL_LEN}_d${DECODE_LEN}_b${BATCH_SIZE}"
```

Both lengths are inputs in tokens. The command tests K3V2 and K4V4 and reports
three-run medians. Use a fresh output directory for each independent run.
See the [benchmark guide](docs/ARTIFACT_EVALUATION.md#configurable-performance-benchmark)
for options and results, or the [Llama 70B commands](docs/ARTIFACT_EVALUATION.md#serving-and-memory).

## Accuracy

Run LongBench and NIAH after configuring the dataset path:

```bash
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k3v2 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2"

python -m jetfire_vllm.artifact niah --model-key llama31_8b \
  --mode k3v2 --gpus 0 --context 32768 --niah-repeats 10 \
  --out-dir "$JETFIRE_WORK_DIR/niah_llama8_32k"
```

Change `--model-key` and `--mode` to select other configurations.
See the [accuracy guide](docs/ARTIFACT_EVALUATION.md#accuracy-and-correction)
for full task and context coverage.

## Related work

| System | Implementation |
| --- | --- |
| TurboQuant | [Community implementation](https://github.com/0xSero/turboquant); [vLLM](https://github.com/vllm-project/vllm) |
| BitDecoding | [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding) |
| SnapKV | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) |
| Quest | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) |
| MagicPIG | [Infini-AI-Lab/MagicPIG](https://github.com/Infini-AI-Lab/MagicPIG) |
| NEO | [NEO-MLSys25/NEO](https://github.com/NEO-MLSys25/NEO) |
| InfiniGen | [snu-comparch/InfiniGen](https://github.com/snu-comparch/InfiniGen) |
| A²ATS | [junhuihe-hjh/A2ATS](https://github.com/junhuihe-hjh/A2ATS) |
| TailorKV | [ydyhello/TailorKV](https://github.com/ydyhello/TailorKV) |

JetFire is licensed under [MIT](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md)
for evaluation utility attribution.
