# JetFire

JetFire combines a low-bit dense KV-cache scan on the GPU with high-precision
top-k attention correction on the CPU. The source includes the vLLM adapter,
INT2/INT3/INT4 CUDA kernels, BF16 host KV mirror, CPU correction workers, output
blending, decode CUDA graphs, tests, and experiment drivers.

[Paper](https://drive.google.com/file/d/1AOmO2idWrcDd-OW__JfwXeb1o92EPzEq/view)
· [Evaluation instructions](docs/ARTIFACT_EVALUATION.md)
· [Model and dataset downloads](docs/MODELS_AND_DATA.md)

This is a source-only artifact. Reviewers download checkpoints and data from
upstream, then run the supplied commands to generate their own results.
Model weights, datasets, compiled extensions, author run logs, predictions, and
measured-result records are excluded. External baseline implementations are
linked in [Related work](#related-work).

## Requirements and installation

The tested platform is Linux x86-64, Python 3.11.17, CUDA 12.8, PyTorch 2.10.0,
Triton 3.6.0, vLLM 0.19.0, and Transformers 4.57.6. Building the JIT extensions
requires `nvcc`, a C++17 compiler, and Ninja. Use a CUDA-compatible NVIDIA
driver; the Python packages do not install the system CUDA compiler. The
adapter depends on the pinned vLLM internal APIs.

Llama-3.1-8B and Qwen3-30B use one H100 PCIe 80 GB GPU with BF16 model weights.
Llama-3.3-70B uses two 80 GB GPUs with TP=2. CPU correction defaults to 32
workers per TP rank. For Llama 8B and Qwen TP=1 benchmarks, reserve 64 physical CPU
cores where possible. For 70B TP=2, retain the full host CPU affinity available
to the job and avoid
inheriting a restricted TP=1 `taskset` mask. Report worker count per rank and
CPU affinity with timing results. Avoid competing GPU and memory-intensive
jobs; CPU topology and memory bandwidth affect performance substantially.

For batch-16 benchmarks at long contexts, plan for **512 GiB host RAM and 150 GB free disk**
for the two checkpoints, environment and working space. These are conservative
planning estimates, not measured minimum requirements. See the guide for the
KV mirror formula and the additional storage needed for 70B.

Clone or download this repository, then run these commands from its root:

```bash
bash scripts/setup_env.sh jetfire-ae
conda activate jetfire-ae
export PYTHON_BIN=python
export JETFIRE_WORK_DIR=/absolute/path/to/jetfire-work
mkdir -p "$JETFIRE_WORK_DIR"
export TORCH_EXTENSIONS_DIR="$JETFIRE_WORK_DIR/torch_extensions"
export MAX_JOBS=8
```

The [setup script](scripts/setup_env.sh) creates Python 3.11.17, installs the
constrained editable package with plotting dependencies, and runs `pip check`.
Use an unused environment name; existing environments are preserved. Conda
must be available on `PATH` or through `CONDA_EXE`.

`requirements.txt` pins the runtime dependencies; the constraints file records
the tested Linux/Python 3.11/CUDA 12.8 dependency resolution. Keep the full source
checkout: build and evaluation tools use the supplied source files. No sibling
development directory is required. Internet access is needed for installation
and downloads; runs can use the downloaded local paths.

## Download and configure inputs

| Model | Official download | Local path variable |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | [Meta checkpoint](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `JETFIRE_LLAMA31_8B` |
| Qwen3-30B-A3B-Instruct-2507 | [Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `JETFIRE_QWEN3_30B` |
| Llama-3.3-70B-Instruct, optional | [Meta checkpoint](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | `JETFIRE_LLAMA33_70B` |

Meta checkpoints require upstream access approval. The
[download guide](docs/MODELS_AND_DATA.md) covers licenses, immutable revisions,
download commands, and original LongBench v1 data. Use the named Qwen **2507
Instruct** checkpoint, rather than a different Qwen3 variant.

```bash
export JETFIRE_LLAMA31_8B=/absolute/path/to/Llama-3.1-8B-Instruct
export JETFIRE_QWEN3_30B=/absolute/path/to/Qwen3-30B-A3B-Instruct-2507
export JETFIRE_LONGBENCH_DATA_DIR=/absolute/path/to/longbench_v1/data
```

## Reproduce experiments

| Benchmark | Entry point |
| --- | --- |
| Performance | [Configurable benchmark](jetfire_vllm/bench/benchmark.py) |
| LongBench accuracy | [Accuracy runner](jetfire_vllm/bench/run_release_accuracy_suite.sh), also available as `python -m jetfire_vllm.artifact accuracy` |
| NIAH accuracy | [Context grid](jetfire_vllm/bench/run_niah_grid_release.sh), or `python -m jetfire_vllm.artifact niah` for one context up to 128K |

Set the prefill length, generated length, and batch size for the performance run:

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

Both lengths are required inputs in tokens; change them for each experiment.
Prefill plus generation must fit the checkpoint's context window. The driver
benchmarks K3V2/group 64 and K4V4/group 128 with k=256 by default; `--modes`
selects either format, and `--batches 1,4,16` sweeps batch sizes. It recalculates
KV and admission budgets for the supplied lengths. `--runs` controls the number
of fresh-server trials per configuration, with one untimed warmup by default
(`--warmups`). Optional `--semantic-probe` adds an untimed response check.

The driver generates per-trial records and `performance_summary.json`, `.csv`
and `.md`, with expected coverage derived from the requested settings. Latency
includes prefill, the requested generated tokens, and request queueing; model
loading and server startup are outside the timed request interval. Report
submitted and resident concurrency together. Use a fresh output directory;
`--resume` continues matching incomplete runs. Replace `--execute` with
`--dry-run` to inspect the plan without launching GPU jobs. The
[evaluation guide](docs/ARTIFACT_EVALUATION.md) also gives the TP=2 command for
Llama-3.3-70B, scheduler policy, CPU affinity, metrics and troubleshooting.

Optional accuracy evaluation:

```bash
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k3v2 --gpus 0 \
  --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2"
python -m jetfire_vllm.artifact niah --model-key llama31_8b \
  --mode k3v2 --gpus 0 --context 32768 --niah-repeats 10 \
  --out-dir "$JETFIRE_WORK_DIR/niah_llama8_32k"
```

LongBench and NIAH runners are optional evaluation utilities. The guide gives
full task coverage and paper experiment commands for memory, sensitivity,
correction ablation, optimization controls and component timing.

## Runtime scope and limitations

Prefill and the first generated token use BF16 attention. Decode combines the
full-sequence low-bit output with selected-token CPU correction, retaining the
low-bit estimate of selected attention mass. Correction completes before its
output is consumed; a slow worker adds latency. Decode graphs and layer-local
prefill activation partitioning are implemented.

Scheduler chunked prefill and rebuilding evicted GPU KV from the CPU mirror
are outside the validated serving path. Submitted batches can run as queued
microbatches when the resident KV budget is insufficient. Hardware, checkpoint
revision, prompt protocol, admission and client protocol must match for a fair
comparison; identical paper numbers are not guaranteed on another machine.

## Related work

| System | Implementation |
| --- | --- |
| TurboQuant | [0xSero/turboquant](https://github.com/0xSero/turboquant) (community implementation); [vLLM](https://github.com/vllm-project/vllm) |
| BitDecoding | [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding) |
| SnapKV | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) |
| Quest | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) |
| MagicPIG | [Infini-AI-Lab/MagicPIG](https://github.com/Infini-AI-Lab/MagicPIG) |
| NEO | [NEO-MLSys25/NEO](https://github.com/NEO-MLSys25/NEO) |
| InfiniGen | [snu-comparch/InfiniGen](https://github.com/snu-comparch/InfiniGen) |
| A²ATS | [junhuihe-hjh/A2ATS](https://github.com/junhuihe-hjh/A2ATS) |
| TailorKV | [ydyhello/TailorKV](https://github.com/ydyhello/TailorKV) |

The TurboQuant link identifies a community implementation. This artifact does
not install these systems. JetFire source is under [MIT](LICENSE); evaluation
utility attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
