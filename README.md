# JetFire

JetFire accelerates long-context LLM inference using a hybrid GPU–CPU attention
pipeline. It combines a low-bit KV cache on the GPU with high-precision CPU
correction for top-k tokens, reducing GPU memory usage while maintaining high
accuracy.

This repository provides the implementation of “JetFire: Approximate-then-Correct
Attention on Hybrid Memory for Long-Context LLM Inference,” accepted at ATC 2026.

[Paper](https://drive.google.com/file/d/1AOmO2idWrcDd-OW__JfwXeb1o92EPzEq/view)
· [Detailed instructions](docs/ARTIFACT_EVALUATION.md)
· [Model and dataset downloads](docs/MODELS_AND_DATA.md)

## Getting Started

The reference environment uses Linux x86-64, Conda, CUDA 12.8 with `nvcc`, and
a C++17 compiler. Evaluation uses one H100 80 GB GPU for Llama 8B and Qwen,
and two for Llama 70B with tensor parallelism (TP=2). Please see the
[hardware guide](docs/ARTIFACT_EVALUATION.md#preparation-and-execution-record)
for CPU and RAM requirements.

From the repository root, create and activate a new Conda environment:

```bash
bash scripts/setup_env.sh jetfire-ae
conda activate jetfire-ae
export PYTHON_BIN=python
export JETFIRE_WORK_DIR=/absolute/path/to/jetfire-work
export TORCH_EXTENSIONS_DIR="$JETFIRE_WORK_DIR/torch_extensions"
export MAX_JOBS=8
mkdir -p "$JETFIRE_WORK_DIR"
```

The setup script installs the pinned Python dependencies and checks their
consistency with `pip check`.

## Models and Data

Model checkpoints are available from the following repositories:

| Model | Download | Path variable |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | [Meta](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `JETFIRE_LLAMA31_8B` |
| Qwen3-30B-A3B-Instruct-2507 | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `JETFIRE_QWEN3_30B` |
| Llama-3.3-70B-Instruct | [Meta](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | `JETFIRE_LLAMA33_70B` |

Please follow the [download guide](docs/MODELS_AND_DATA.md) for model access,
pinned revisions, and LongBench v1 data. Set the paths to your local downloads:

```bash
export JETFIRE_LLAMA31_8B=/absolute/path/to/Llama-3.1-8B-Instruct
export JETFIRE_QWEN3_30B=/absolute/path/to/Qwen3-30B-A3B-Instruct-2507
export JETFIRE_LLAMA33_70B=/absolute/path/to/Llama-3.3-70B-Instruct
export JETFIRE_LONGBENCH_DATA_DIR=/absolute/path/to/longbench_v1/data
```

## Performance Evaluation

The following example measures end-to-end serving time for Llama 8B and Qwen.
Set the prefill and decode lengths in tokens, and choose a batch size:

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

The benchmark evaluates K3V2 (3-bit keys, 2-bit values) and K4V4 (4-bit keys
and values), reporting median latency and throughput over three timed runs
per configuration.
Please use a new output directory for each invocation. See the
[benchmark guide](docs/ARTIFACT_EVALUATION.md#configurable-performance-benchmark)
for configuration options and the
[Llama 70B example](docs/ARTIFACT_EVALUATION.md#serving-and-memory) for TP=2 evaluation.

## Accuracy Evaluation

After configuring the dataset path, the following examples evaluate Llama 8B
on LongBench v1 and Needle-in-a-Haystack (NIAH):

```bash
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k3v2 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2"

python -m jetfire_vllm.artifact niah --model-key llama31_8b \
  --mode k3v2 --gpus 0 --context 32768 --niah-repeats 10 \
  --out-dir "$JETFIRE_WORK_DIR/niah_llama8_32k"
```

For other models, quantization formats, context lengths, and scoring details,
please refer to the [accuracy guide](docs/ARTIFACT_EVALUATION.md#accuracy-and-correction).

## Related Work

The following repositories provide implementations of related systems.

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
