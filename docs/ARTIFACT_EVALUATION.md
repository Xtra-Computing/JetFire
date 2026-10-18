# Artifact Evaluation

This guide describes how to reproduce the performance, accuracy, and component
experiments in the [JetFire paper](https://drive.google.com/file/d/1AOmO2idWrcDd-OW__JfwXeb1o92EPzEq/view).
Download the inputs separately and use the commands below to generate logs,
predictions, and results. Implementations of related systems are linked in the
[README](../README.md#related-work).

The examples use Llama-3.1-8B and Qwen3-30B-A3B-Instruct-2507. Llama-3.3-70B
uses two GPUs with tensor parallelism (TP=2).

## Preparation and Execution Record

From the repository root, create and activate a new Conda environment:

```bash
bash scripts/setup_env.sh jetfire-ae
conda activate jetfire-ae
```

The script installs the pinned Python dependencies and runs `pip check`.
Please choose a different environment name if `jetfire-ae` already exists.
The [README](../README.md#getting-started) lists the system and hardware requirements.
Follow the [download guide](MODELS_AND_DATA.md) to configure
`JETFIRE_LLAMA31_8B`, `JETFIRE_QWEN3_30B`, and `JETFIRE_LONGBENCH_DATA_DIR`;
set `JETFIRE_LLAMA33_70B` when evaluating 70B. Accuracy evaluation uses ten
original LongBench v1 English task files. NIAH uses the same text as filler;
serving measurements use seeded random prompts and require only the checkpoint.

Evaluation uses BF16 weights and a host KV mirror. The mirror requires
approximately `2 * layers * KV_heads * head_dim * resident_tokens * 2` bytes
in total, divided across TP ranks. Here, `resident_tokens` counts tokens across
all resident requests. For Llama 8B at 128K context, the mirror occupies about
16 GiB per request, or 256 GiB for sixteen requests. Allow additional RAM for
model loading, staging buffers, workers, and allocator overhead. A practical
planning estimate for long-context batch-16 experiments is 512 GiB of host RAM;
the actual requirement depends on the configuration.

For timing comparisons, keep the hardware, NUMA placement, CPU thread count,
and affinity consistent. The reference processor is AMD EPYC 9654, and CPU
correction defaults to 32 workers per TP rank. For Llama 8B/Qwen TP=1
benchmarks, reserve 64 physical cores where possible. Inspect
`lscpu -e=CPU,CORE,SOCKET,NODE` and, if needed, prefix the benchmark command with
`taskset -c <64 physical-core IDs>`. Select IDs from your machine, using one
logical CPU per physical core. The benchmark records the inherited affinity.

For Llama 70B TP=2, launch from a shell with the full host CPU affinity available
to the job. Each rank uses 32 correction workers, with additional CPU capacity
needed for engine work. Please record the worker count and inherited affinity
with timing results, and use a machine without competing GPU or CPU/DRAM-intensive jobs.

```bash
export PYTHON_BIN=python
export JETFIRE_WORK_DIR=/absolute/path/to/jetfire-work
mkdir -p "$JETFIRE_WORK_DIR"
export TORCH_EXTENSIONS_DIR="$JETFIRE_WORK_DIR/torch_extensions"
export MAX_JOBS=8
```

The artifact entry point saves a manifest and step logs under `--out-dir`;
the default is a timestamped directory under `results/` inside the checkout.
An external working directory keeps generated files separate from the source.
Use a new output directory for independent runs; the artifact wrapper rejects
an existing directory. Before interpreting results, check `status` and each
step's return code, and retain the logs and predictions from failed runs.
Benchmark resumption is described in [Reading Generated Results](#reading-generated-results).

## Paper Coverage

The following table maps paper experiments to the supplied entry points.

| Paper item | JetFire entry point | Configuration |
| --- | --- | --- |
| Main accuracy table | `artifact accuracy` | Three models, K4V4/G128 and K3V2/G64, k256; ten tasks |
| NIAH | `artifact niah`, `run_niah_grid_release.sh` | 11 depths, 10 samples per depth/context; grid targets listed below |
| Table 4 serving latency | `benchmark` (Llama 8B/Qwen), `artifact serve` (single configuration, including 70B TP=2) | Configurable prefill/decode lengths; paper settings: batches 1/4/16, 16K–128K (Qwen also 256K), 128 output tokens, median of three runs |
| Fig. 5 memory | `memory_breakdown_probe.py --mode run` | Llama 8B, 64K, 32 decode tokens; serving harness also reports residency |
| Fig. 6 K/V allocation | `run_release_kv_sensitivity.sh` | Nine K/V bit-width pairs with symmetric quantization, Qwen 2Wiki, k32, G64 |
| Fig. 7 correction | `release_accuracy_eval.py --jetfire-approx-only` | k0 output ablation vs k256, 8B/Qwen, both formats, ten tasks |
| Fig. 8 k sensitivity | `run_release_topk_sensitivity.sh` | k32/64/128/256/512/1024; K3V2 and K2V2 |
| Table 7 memory breakdown | `run_memory_breakdown_suite.sh` | Qwen K4V4, 8K–256K; separate calculated and measured rows |
| Table 8 selection quality | `topk_quality_probe.py` | Index recall, selected/oracle mass, low-bit/corrected output error |
| Fig. 9 optimization ablation | `artifact_optimizations.py` | JetFire / +O1 / +O1+O2 / +O1+O2+O3; partitioning off |
| Table 9 overlap components | `artifact_overlap.py` | Paired eager timing and separate CPU/CUDA timeline captures |

For Fig. 5 at batch 16, collect a concurrent serving measurement in addition
to the single-request memory probe. The optimization runner reconstructs the
documented historical controls on the supplied runtime.

## Configurable Performance Benchmark

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

`--prefill-len` and `--decode-len` are required positive token counts. Each
invocation benchmarks one length pair; prefill plus generation must fit the
checkpoint's context window. Optional scheduling guard space uses only the
remaining window and preserves the requested lengths.

| Option | Behavior |
| --- | --- |
| `--models` | Select Llama 8B, Qwen, or both, as in the example. |
| `--batches` | Positive comma-separated batch sizes, such as `1,4,16`. |
| `--modes` | `k3v2`, `k4v4`, or both; defaults to K3V2/G64 and K4V4/G128, with k256 and BF16 weights. |
| `--runs` | Fresh-server timed trials per configuration; default three. |
| `--warmups` | Untimed warmups per server; default one. |
| `--semantic-probe` | Optional meaningful-response check, separate from random-prompt throughput; disabled by default. |
| `--dry-run` | Replace `--execute` to inspect the plan without launching GPU jobs. |

The plan includes lengths, modes, coverage, budget policy, and command templates.
Execution resolves numerical byte/token budgets from the local checkpoint and GPU.

For each length pair, configuration count is models × modes × batch sizes.
Expected trials are configuration count × `--runs`. Expected requests are
`--runs` × the sum of batch sizes across configurations; multiply by
`--decode-len` for expected generated tokens. The plan and summaries record
these totals.

For these TP=1 models, the driver computes a bounded KV allocation from physical
GPU memory, BF16 weights, and a 5 GiB scratch reservation. The cap is
`floor((0.95 * total_GPU_GiB - BF16_weights_GiB - 5) * 2**30)` bytes;
compressed KV estimates use blocks of 16 tokens and a safety factor of 1.02.
Resident capacity is calculated from these estimates. Requests run together
when residency and prefill capacity permit; additional requests queue behind
resident microbatches. Execution records byte and scheduler-token budgets,
including 4,096 scheduler headroom tokens for queued cases. Budgets are
recalculated for the requested lengths. The single-configuration `artifact serve`
command uses the serving harness's automatic capacity policy.

Total execution time includes repeated server loading and compilation.
Initial extension compilation can take roughly 10–30 minutes; large performance
or accuracy sweeps can take hours to days, depending on context and coverage.

## Accuracy and Correction

The following examples evaluate Llama 8B on LongBench v1 in both main formats:

```bash
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k3v2 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2"
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k4v4 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k4v4"
```

For Qwen, replace `llama31_8b` with `qwen3_30b_a3b`. For Llama 70B, use
`--model-key llama33_70b --gpus 0,1 --tp 2`. Full LongBench coverage comprises
1,950 examples per configuration: 150 MultiFieldQA examples and 200 for each
other task. The reported aggregate is the arithmetic mean of the ten task scores.
Generation is greedy with the official task-specific limits. TREC and SAMSum
use their raw official templates and the first output line for scoring.
Comparable paper results require complete coverage and the same prompt protocol;
changes to that protocol invalidate existing resume rows.

The k0 experiment returns the dense approximation while keeping correction
machinery active. Use it as an **accuracy-only output ablation**; its timings
include correction work:

```bash
bash jetfire_vllm/bench/run_release_accuracy_suite.sh \
  --model llama31_8b --backend jetfire_k3v2 --suite longbench_v1 \
  --gpus 0 --tensor-parallel-size 1 --jetfire-group-size 64 \
  --top-k 256 --jetfire-approx-only --save-predictions \
  --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2_k0"
```

For Needle-in-a-Haystack (NIAH), choose a context length that leaves room for
generation and prompt overhead within the checkpoint's window. By default,
eleven requested depths from 0–100% map into effective depths of 5–95%, using
an edge margin of 5%. Ten repeats give 110 trials per context. Results retain
both requested and effective depths.

```bash
python -m jetfire_vllm.artifact niah --model-key llama31_8b \
  --mode k3v2 --gpus 0 --context 32768 --niah-repeats 10 \
  --out-dir "$JETFIRE_WORK_DIR/niah_llama8_32k"
```

For the full grid, use `run_niah_grid_release.sh` with `--phase llama31`,
`--phase qwen3`, or `--phase llama33`. It evaluates the main formats plus
K4V2 and K2V2. Llama target lengths are 16,000, 32,000, 64,000, and 125,000
tokens; Qwen adds 250,000 tokens. These decimal targets leave generation
headroom, and each result records the actual `prompt_tokens`. Use the Qwen
grid for the 256K suite; single-context `artifact niah` uses a 128K suite window.

```bash
bash jetfire_vllm/bench/run_niah_grid_release.sh --phase qwen3 \
  --gpus-qwen3 0 --out-dir "$JETFIRE_WORK_DIR/niah_qwen_grid"
```

NIAH compares the entire output with the target after removing only punctuation
and whitespace. Extra explanations, letters, or numbers count as mismatches.
Results also retain exact match on the stripped raw strings. Please report
the scoring protocol with the score and retain the raw outputs.

## Official Benchmark Client Framing

The serving harness runs the installed official vLLM `bench serve` CLI through
`jetfire_vllm.bench.official_client`. Its `strict-sse-frames-v1` adapter delivers
complete LF-delimited SSE events to the unchanged official request coroutine,
preserving UTF-8, split `data: ` prefixes, and JSON string whitespace across
HTTP chunks. Strict UTF-8 validation includes the final bytes at EOF; invalid
or truncated UTF-8 produces a failed request. The protocol targets the pinned
vLLM emitter's LF stream; general CRLF handling and strict validation of
arbitrary malformed JSON/SSE trailers are outside its scope.

Each trial records adapter and upstream source identities in `client_decoder.json`.
CLI arguments, request payloads, the server, and upstream timing/metric
statements are unchanged. Delivering complete events can defer timestamp
observation until the event separator arrives and can add overhead. Please
use the same client protocol when comparing trials.

## Serving and Memory

For a single configuration, including Llama-3.3-70B on two GPUs, set the prefill
length, decode length, and batch size:

```bash
PREFILL_LEN=32768
DECODE_LEN=128
BATCH_SIZE=4

python -m jetfire_vllm.artifact serve --model-key llama33_70b \
  --mode k3v2 --gpus 0,1 --tp 2 \
  --prefill-len "$PREFILL_LEN" --decode-len "$DECODE_LEN" \
  --batch "$BATCH_SIZE" --runs 3 \
  --out-dir "$JETFIRE_WORK_DIR/serve_llama70_k3v2_p${PREFILL_LEN}_d${DECODE_LEN}_b${BATCH_SIZE}"
```

Set `JETFIRE_LLAMA33_70B` to the downloaded checkpoint before running this command.
For Llama 8B or Qwen, use `--model-key llama31_8b` or
`--model-key qwen3_30b_a3b` with `--gpus 0 --tp 1`. The aliases `--context`
and `--output-len` correspond to `--prefill-len` and `--decode-len`.
For TP=1 batch or format sweeps with bounded KV allocation, please use the
[configurable benchmark](#configurable-performance-benchmark).

The harness writes official vLLM client JSON, server/client logs, summary
JSON/CSV, and a report. Report end-to-end time, time to first token (TTFT),
time per output token (TPOT), actual output tokens, successful/failed requests,
prefill activation chunks, and resident capacity alongside submitted concurrency.
The first token comes from BF16 prefill. End-to-end time includes prefill;
generation lengths should match when comparing latency. Diagnose OOM from
allocation errors, since a timeout alone does not identify its cause.

```bash
MODE=run MODELS=qwen3_30b_a3b CONFIGS=k4v4 \
  CTX_LIST=8192,16384,32768,65536,131072,256000 GPUS=0 \
  OUT_DIR="$JETFIRE_WORK_DIR/memory_qwen_k4v4" \
  bash jetfire_vllm/bench/run_memory_breakdown_suite.sh
```

The memory wrapper defaults to **computed-only**; set `MODE=run` for measurement.
It separates GPU KV, CPU BF16 mirrors, metadata/staging, and runtime allocation.
Calculated sizes describe storage requirements. Capacity comparisons also need
measured GPU residency and peak allocation at the chosen concurrency.

## Sensitivity and Optimization Controls

The kernels support nine K/V bit-width pairs drawn from {2, 3, 4}, using signed
symmetric quantization and group sizes 64/128. Symmetric INT2 has three signed
levels; the asymmetric four-level INT2 compatibility mode is a separate format.
The paper's main configurations are K3V2/G64 and K4V4/G128.

```bash
bash jetfire_vllm/bench/run_release_kv_sensitivity.sh \
  --model qwen3_30b_a3b --tasks 2wikimqa --top-k 32 \
  --gpus 0 --tensor-parallel-size 1 --out-dir "$JETFIRE_WORK_DIR/kv_bits"
bash jetfire_vllm/bench/run_release_topk_sensitivity.sh \
  --model qwen3_30b_a3b --tasks 2wikimqa --gpus 0 \
  --tensor-parallel-size 1 --out-dir "$JETFIRE_WORK_DIR/topk_k3v2"
bash jetfire_vllm/bench/run_release_topk_sensitivity.sh \
  --model qwen3_30b_a3b --backend jetfire_k2v2 --tasks 2wikimqa --gpus 0 \
  --tensor-parallel-size 1 --out-dir "$JETFIRE_WORK_DIR/topk_k2v2"
python -m jetfire_vllm.bench.topk_quality_probe \
  --model "$JETFIRE_LLAMA31_8B" --contexts 16384  \
  --filler-data-dir "$JETFIRE_LONGBENCH_DATA_DIR" --out-dir "$JETFIRE_WORK_DIR/topk_probe"
python -m jetfire_vllm.bench.artifact_optimizations \
  --model "$JETFIRE_LLAMA31_8B" --gpu 0 \
  --out-dir "$JETFIRE_WORK_DIR/optimizations"
```

The optimization command executes the cases; add `--dry-run` to inspect its
plan. O1 combines asynchronous prefill mirroring and fused KV writes; O2
overlaps CPU correction with the GPU value phase; O3 adds decode graphs.
The runner fixes K3V2/G64/k256, batch 1, 16 output tokens, 32 CPU threads,
and 32K/64K contexts, with activation partitioning off. Its plan records all
controls and qualifications for the reconstructed experiment.

Selection-quality probes use real model traces to measure offline mechanism
behavior. Serving TPOT and full task accuracy require the corresponding
evaluations above. Synthetic filler is available through explicit opt-in and
uses a different data protocol from the paper.

## Component Timing and Profiling

These commands isolate production CPU correction and pinned-memory copies:

```bash
python -m jetfire_vllm.bench.cpu_correction_microbench \
  --out-dir "$JETFIRE_WORK_DIR/cpu_correction"
python -m jetfire_vllm.bench.pcie_transfer_microbench \
  --out-dir "$JETFIRE_WORK_DIR/pcie"
```

For the complete correction/value timeline, run:

```bash
python -m jetfire_vllm.bench.artifact_overlap \
  --model "$JETFIRE_LLAMA31_8B" --gpu 0 --contexts 32768 65536 \
  --runs 3 --profile-runs 1 --out-dir "$JETFIRE_WORK_DIR/overlap"
```

Add `--dry-run` to inspect the profiling plan. The runner fixes K3V2/G64/k256,
batch 1, 16 output tokens, and 32 CPU threads, with graphs and activation
partitioning off. It alternates serial and overlapped correction on the same
warmed server, saving unprofiled request timings separately from CPU/CUDA captures.
Outputs include Chrome traces, CPU timestamp sidecars, per-layer CSVs, runtime
settings, and coverage reports. Reports identify D2H copies, CPU gather/compute,
value kernels, join wait, blend, clock uncertainty, and uncovered gaps.
Mapped host reads occur during blend and are not a separate H2D copy.

A valid evaluation requires complete timeline mapping and matching paired
outputs. Please report profiling overhead and use unprofiled timings for latency
comparisons. End-to-end speedup depends on the full request path; interval
overlap alone does not establish a speedup, and short contexts may show a slowdown.

## Reading Generated Results

For `artifact` stages, start with `manifest.json`. A status of `passed` means
the stage's commands completed; model accuracy is assessed from coverage and
scored predictions. Check each step's exit status and local log.

Serving outputs are `summary.json`, `summary.csv`, and `report.md`, with
per-trial official client output and server/client logs. The configurable
benchmark writes `performance_summary.json`, `performance_summary.csv`, and
`performance_summary.md`. A complete benchmark has status `passed`,
`cases_passed == cases_expected`, and
`measured_runs_complete == measured_runs_expected`. Expected counts follow the
requested models, modes, batch sizes, and run count.

To resume an interrupted configurable benchmark, use `--execute --resume` with
the same output directory. Source, model input, environment, and policy
fingerprints must match; changed settings require a new directory.

Please report individual trials and their median, successful/failed requests,
actual output token counts, submitted batch size, and resident capacity.
End-to-end batch latency includes prefill, generation, queueing, and client
transport; server startup and weight loading are excluded. TTFT and TPOT are
separate metrics. Comparisons with Table 4 require the same units and aggregation;
per-request latency and batch completion time measure different quantities.

Accuracy output contains per-task counts/scores and predictions when requested.
Check coverage, including skipped and overlength samples, before computing the
ten-task macro average. NIAH output includes requested/effective depths,
targets, and both normalized and raw match. Keep missing or failed cells
separate from measured zero scores, and classify execution failures from their
recorded errors.

For comparisons with the paper, retain environment/input records and report
differences in software versions, random prompts, scheduler budgets, and timing
boundaries. These settings and the hardware can affect numerical results;
functional correctness is assessed from completed evaluations and their outputs.
Speedup comparisons also require measurements of the separately linked systems.

## Troubleshooting and Reruns

| Symptom | Check or action |
| --- | --- |
| Checkpoint download returns 401/403 | Confirm the model ID and upstream access approval, then authenticate locally with `hf auth login`. Keep access tokens out of commands and reports. |
| `nvcc` missing or extension compilation fails | Check CUDA 12.8, the C++17 compiler, and `TORCH_EXTENSIONS_DIR`. Lower `MAX_JOBS` if compilation exhausts host memory; retain the compiler error for diagnosis. |
| vLLM import/API failure | Recreate the pinned Python environment with the setup script, then run `python -m pip check`. |
| GPU OOM or host allocation failure | Check competing processes, free GPU/host memory, checkpoint precision, and resident capacity. Smaller batches/contexts can help diagnosis; record the changed settings. |
| Missing LongBench task or NIAH filler | Point `JETFIRE_LONGBENCH_DATA_DIR` at the directory directly containing the ten original `task.jsonl` files. |
| An output directory already exists | Choose a new directory for an independent run. The configurable benchmark and lower-level accuracy tools support explicit resume with matching inputs and protocol. |
| Slow runs or timing variation | Check GPU ownership, CPU affinity/NUMA, worker count, throttling and competing DRAM traffic. Compare unprofiled trials with the same admission and client policy. |
| Stream decoding/request failure | Inspect client/server errors and the recorded decoder identity. Retain the failed trial and resolve the failure before reporting a complete benchmark. |

Please use `--help` on each entry point for available options. Rebuild
extensions after changing CUDA, compiler, or runtime versions, and keep
generated records in the chosen working directory.

