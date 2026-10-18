# Artifact evaluation

This guide maps the [JetFire paper](https://drive.google.com/file/d/1AOmO2idWrcDd-OW__JfwXeb1o92EPzEq/view)
to executable source. The distribution contains source, input recipes and
instructions. Reviewers generate all logs, predictions and results themselves;
no author run records are needed by the commands below. External attention
systems appear only as upstream implementation links in the README.

The main release workflow targets Llama-3.1-8B and Qwen3-30B-A3B-Instruct-2507.
The 70B commands use two GPUs with TP=2. The coverage table maps paper
experiments to the supplied entry points and configurations.

## Preparation and execution record

Create the environment from the source root, then activate it:

```bash
bash scripts/setup_env.sh jetfire-ae
conda activate jetfire-ae
```

The script installs the constrained dependencies and runs `pip check`.
Choose a new name if `jetfire-ae` already exists. The [README](../README.md)
lists the system CUDA/compiler and hardware requirements.
Follow [model and data preparation](MODELS_AND_DATA.md), then set
`JETFIRE_LLAMA31_8B`, `JETFIRE_QWEN3_30B` and
`JETFIRE_LONGBENCH_DATA_DIR`. Set `JETFIRE_LLAMA33_70B` only when evaluating
70B. The accuracy evaluator uses ten original LongBench v1 English task JSONL
files, and NIAH uses that same text as filler. Serving measurements use seeded random
prompts and do not require the dataset.

Use BF16 weights and reserve CPU RAM for the host KV mirror. The approximate
mirror requirement is `2 * layers * KV_heads * head_dim * resident_tokens * 2`
bytes, divided across TP ranks. It scales with resident requests as well as
context. For example, an 8B 128K context needs approximately 16 GiB of host
KV mirror per resident request, or 256 GiB for sixteen. Model loading, staging buffers,
worker memory and allocator overhead require additional RAM. Plan for 512 GiB
host RAM for long-context batch-16 benchmarks; smaller experiments can require less.
This is a sizing estimate, not a demonstrated minimum.

Keep the machine, NUMA placement, CPU thread count and affinity consistent for
comparisons. The tested processor family was AMD EPYC 9654. CPU correction
defaults to 32 workers per TP rank. For Llama 8B/Qwen TP=1 benchmarks,
reserve 64 physical cores where possible. Inspect
`lscpu -e=CPU,CORE,SOCKET,NODE` and, if needed, prefix the benchmark command with
`taskset -c <64 physical-core IDs>`. Choose one logical CPU per physical core;
do not copy another machine's CPU IDs blindly. The benchmark records inherited
affinity.

For Llama 70B TP=2, retain the full host CPU affinity available to the job.
Launch from a shell that has not inherited the TP=1 `taskset` restriction:
two TP ranks each use 32 CPU correction workers and need CPU headroom for the
remaining engine work. Report worker count per rank and the inherited CPU
affinity alongside timing results. Avoid competing GPU jobs and
CPU/DRAM-intensive work.

```bash
export PYTHON_BIN=python
export JETFIRE_WORK_DIR=/absolute/path/to/jetfire-work
mkdir -p "$JETFIRE_WORK_DIR"
export TORCH_EXTENSIONS_DIR="$JETFIRE_WORK_DIR/torch_extensions"
export MAX_JOBS=8
```

The artifact entry point saves a manifest and step logs under `--out-dir`
(default: a new timestamped `results/` directory inside the checkout). Use an
external working directory to keep the source tree clean.
Use a new output directory for each invocation; an existing directory is
rejected. Inspect `status` and every step return code. Keep raw logs and
prediction files, including failed runs. A nonzero exit is a failure, not an accuracy zero
or evidence of OOM.

## Paper coverage

| Paper item | JetFire entry point | Configuration |
| --- | --- | --- |
| Main accuracy table | `artifact accuracy` | Three models, K4V4/G128 and K3V2/G64, k256; ten tasks |
| NIAH | `artifact niah`, `run_niah_grid_release.sh` | 11 depths, 10 samples/depth/context; 16K–128K, Qwen up to 256K |
| Table 4 serving latency | `benchmark` (Llama 8B/Qwen), `artifact serve` (single configuration, including 70B TP=2) | Supply prefill and decode lengths; the paper uses batches 1/4/16, 16K–128K (Qwen also 256K), 128 outputs and median 3 |
| Fig. 5 memory | `memory_breakdown_probe.py --mode run` | Llama 8B, 64K, 32 decode tokens; serving harness also reports residency |
| Fig. 6 K/V allocation | `run_release_kv_sensitivity.sh` | Nine symmetric K/V combinations, Qwen 2Wiki, k32, G64 |
| Fig. 7 correction | `release_accuracy_eval.py --jetfire-approx-only` | k0 output ablation vs k256, 8B/Qwen, both formats, ten tasks |
| Fig. 8 k sensitivity | `run_release_topk_sensitivity.sh` | k32/64/128/256/512/1024; K3V2 and K2V2 |
| Table 7 memory breakdown | `run_memory_breakdown_suite.sh` | Qwen K4V4, 8K–256K; separate calculated and measured rows |
| Table 8 selection quality | `topk_quality_probe.py` | Index recall, selected/oracle mass, low-bit/corrected output error |
| Fig. 9 optimization ablation | `artifact_optimizations.py` | JetFire / +O1 / +O1+O2 / +O1+O2+O3; partitioning off |
| Table 9 overlap components | `artifact_overlap.py` | Paired eager timing and separate CPU/CUDA timeline captures |

Fig. 5 B16 requires a concurrent serving measurement in addition to the
single-request memory probe. The optimization runner uses the documented
historical controls on this runtime.

## Configurable performance benchmark

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
invocation uses that length pair; change the variables and output directory for
another experiment. `--models` selects Llama 8B, Qwen, or both. `--batches`
accepts positive comma-separated sizes such as `1,4,16`; `--modes` selects
`k3v2`, `k4v4`, or both. The default mode pair is K3V2/G64 and K4V4/G128,
with k256 and BF16 weights. Prefill plus generation must fit the checkpoint's
context window. The driver bounds optional scheduling guard space by the
remaining window; it does not replace the requested prefill or decode length.

`--runs` sets the number of fresh-server timed trials per configuration
(default three). `--warmups` sets untimed warmups per server (default one).
The optional `--semantic-probe` checks meaningful model responses separately
from random-prompt throughput and is disabled by default. Replace `--execute`
with `--dry-run` to inspect the requested lengths, modes, coverage, budget
policy and command templates without creating GPU jobs. Numerical byte/token
budgets are resolved from the local checkpoint and GPU during execution.

For one length pair, the configuration count is the number of selected models
times selected modes times batch sizes. Expected measured trials are that count
times `--runs`. Expected measured requests are `--runs` times the sum of batch
sizes across configurations; multiply requests by `--decode-len` for expected
generated tokens. The generated plan and summaries record these totals.

For these TP=1 models, the driver fixes a bounded KV allocation from physical
GPU memory, BF16 weights and a 5 GiB scratch reservation. It rounds compressed
sequence storage to blocks with a safety allowance, then computes resident
request capacity. The cap is
`floor((0.95 * total_GPU_GiB - BF16_weights_GiB - 5) * 2**30)` bytes;
compressed KV estimates use block size 16 and safety factor 1.02.
The scheduler budget covers all submitted requests when residency and
automatic prefill capacity permit; otherwise, requests queue behind resident microbatches.
Execution records the resulting byte and scheduler-token budgets, including
4,096 scheduler headroom tokens for queued cases. KV allocation and scheduler
budgets are recalculated for the requested input and output lengths. This
TP=1 policy differs from the serving
harness's default automatic budgets used by `artifact serve` below.

Execution time depends on the selected models, lengths, batches and run count,
including repeated server loading and compilation. Initial extension
compilation can take roughly 10–30 minutes. Large performance, LongBench or
NIAH sweeps can take hours to days depending on context and coverage.

## Accuracy and correction

```bash
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k3v2 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2"
python -m jetfire_vllm.artifact accuracy --model-key llama31_8b \
  --mode k4v4 --gpus 0 --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k4v4"
```

Replace `llama31_8b` with `qwen3_30b_a3b` for Qwen. For optional 70B, use
`--model-key llama33_70b --gpus 0,1 --tp 2`. Full LongBench coverage
is 1,950 examples per configuration: 150 MultiFieldQA examples and 200 for
each other task. Compare the arithmetic mean of the ten task scores, rather
than a sample-weighted mean. Generation is greedy with task-specific official
limits; TREC and SAMSum use their raw official templates and first output
line for scoring. Changed prompt protocols invalidate old resume rows.
Partial or skipped coverage cannot serve as a full paper result.

The k0 experiment keeps correction machinery active and returns the dense
approximation. It is an **accuracy-only output ablation**, not a timing
measurement of a runtime without correction:

```bash
bash jetfire_vllm/bench/run_release_accuracy_suite.sh \
  --model llama31_8b --backend jetfire_k3v2 --suite longbench_v1 \
  --gpus 0 --tensor-parallel-size 1 --jetfire-group-size 64 \
  --top-k 256 --jetfire-approx-only --save-predictions \
  --out-dir "$JETFIRE_WORK_DIR/accuracy_llama8_k3v2_k0"
```

For NIAH, select each context explicitly and preserve requested and effective
depths. Default 0–100% requested depths map into 5–95% with edge margin 5.
Ten repeats at eleven depths give 110 trials per context. Choose a context
that leaves room for generation and prompt overhead inside the checkpoint's
window; do not force an out-of-window result and describe it as supported.

```bash
python -m jetfire_vllm.artifact niah --model-key llama31_8b \
  --mode k3v2 --gpus 0 --context 32768 --niah-repeats 10 \
  --out-dir "$JETFIRE_WORK_DIR/niah_llama8_32k"
```

For the full context grid, use `run_niah_grid_release.sh` with `--phase llama31`,
`--phase qwen3` or `--phase llama33`. The grid includes the main formats and
additional JetFire sensitivity formats. Use the Qwen grid for its 256K context;
the single-context `artifact niah` command uses a 128K suite window.

```bash
bash jetfire_vllm/bench/run_niah_grid_release.sh --phase qwen3 \
  --gpus-qwen3 0 --out-dir "$JETFIRE_WORK_DIR/niah_qwen_grid"
```

The NIAH score compares the entire output and target after removing punctuation
and whitespace only. Explanations, extra letters, and additional numbers fail.
Results also retain raw stripped-string exact match. Keep the scoring protocol
and raw outputs with the score; normalized match differs from raw exact match.

## Official benchmark client framing

The serving harness runs the installed official vLLM `bench serve` CLI through
`jetfire_vllm.bench.official_client`. The subprocess adapter presents complete
LF-delimited SSE events to the unchanged official request coroutine, preserving
UTF-8 code points, split `data: ` prefixes, and JSON string whitespace across
arbitrary HTTP transport chunks. Invalid or truncated UTF-8 remains a failed
request. Each timed trial records the adapter and upstream source identity in
`client_decoder.json`; the official CLI argument tail, request payload, server,
and upstream clock/metric statements are unchanged.

The adapter identifies its framing protocol as `strict-sse-frames-v1`.
Complete-event delivery can defer timestamp observation until
the event separator arrives, so compare trials using the same client protocol;
zero timing overhead is not claimed. Its delimiter scope is the pinned vLLM
emitter's LF SSE stream. General CRLF SSE support and strict validation of
arbitrary malformed JSON/SSE trailers are not claimed. Strict UTF-8 validation
also checks the final bytes at EOF.

## Serving and memory

For a single configuration, including Llama-3.3-70B on two GPUs, set the
lengths and batch explicitly:

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

Set `JETFIRE_LLAMA33_70B` to the downloaded checkpoint first. For Llama 8B or
Qwen, use `--model-key llama31_8b` or `--model-key qwen3_30b_a3b` with
`--gpus 0 --tp 1`. `--context` and `--output-len` remain aliases for
`--prefill-len` and `--decode-len`. The single-configuration command uses the
serving harness's automatic capacity policy. Use the configurable benchmark
above for TP=1 batch or format sweeps with its explicit bounded KV policy.

The serving harness writes the official vLLM client JSON, server/client logs,
summary JSON/CSV, and a report. Report end-to-end time, TTFT, TPOT, actual
output tokens, successful/failed requests, chosen prefill activation chunks,
and resident admission alongside submitted concurrency. The first token comes
from BF16 prefill. Do not call end-to-end time decode-only latency, extrapolate
16-token measurements to 128 tokens, or infer OOM from a timeout.

```bash
MODE=run MODELS=qwen3_30b_a3b CONFIGS=k4v4 \
  CTX_LIST=8192,16384,32768,65536,131072,256000 GPUS=0 \
  OUT_DIR="$JETFIRE_WORK_DIR/memory_qwen_k4v4" \
  bash jetfire_vllm/bench/run_memory_breakdown_suite.sh
```

The memory wrapper defaults to **computed-only** unless `MODE=run` is set.
Calculated KV sizes do not establish measured capacity. The breakdown
distinguishes GPU KV, CPU BF16 shadows, metadata/staging and runtime allocation;
query GPU residency and peak allocation separately when comparing concurrency.

## Sensitivity and optimization controls

The kernels support all nine symmetric K/V bit combinations from {2,3,4},
with group sizes 64/128. Symmetric INT2 has three signed levels; the separately
named asymmetric four-level INT2 compatibility mode is a different format.
The paper's primary recipes use K3V2/G64 and K4V4/G128.

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

The command executes the optimization cases; add `--dry-run` for an optional
plan. O1 bundles async prefill
mirroring and fused KV writes; O2 overlaps CPU correction with the GPU value
phase; O3 adds decode graphs. The runner fixes K3V2/G64/k256, batch 1,
16 outputs, 32 CPU threads and 32K/64K contexts. Activation partitioning is
off. Its plan records all toggles and qualifications. Selection-quality probes
use real model traces but are offline mechanism probes; they do not establish
serving TPOT or replace full task accuracy. Synthetic filler requires explicit
opt-in and cannot be reported as the paper's data protocol.

## Component timing and profiling

```bash
python -m jetfire_vllm.bench.cpu_correction_microbench \
  --out-dir "$JETFIRE_WORK_DIR/cpu_correction"
python -m jetfire_vllm.bench.pcie_transfer_microbench \
  --out-dir "$JETFIRE_WORK_DIR/pcie"
```

These isolate production CPU correction and pinned-memory copies respectively.
For the complete correction/value timeline, run:

```bash
python -m jetfire_vllm.bench.artifact_overlap \
  --model "$JETFIRE_LLAMA31_8B" --gpu 0 --contexts 32768 65536 \
  --runs 3 --profile-runs 1 --out-dir "$JETFIRE_WORK_DIR/overlap"
```

The command executes the profiling experiment; add `--dry-run` for an optional
plan. The runner fixes K3V2/G64/k256, batch 1,
16 outputs, 32 CPU threads, graphs off, and activation partitioning off. It
alternates serial and overlapped correction on the same warmed server, saving
unprofiled request timings separately from diagnostic CPU/CUDA captures. Raw
Chrome traces, CPU timestamp sidecars, per-layer CSVs, runtime settings and
coverage reports are retained. Reports expose D2H copies, CPU gather/compute,
value kernels, join wait, blend, clock uncertainty, and uncovered gaps. Mapped
host reads during blend cannot be separated into an independent H2D copy.

Incomplete timeline mapping or different paired outputs fails the evaluation.
Profiling adds overhead; report that overhead and use the separate unprofiled
timings for latency claims. Interval overlap by itself does not establish a
corresponding end-to-end speedup, and short-context results may show a slowdown.

## Reading generated results

For `artifact` stages, start with `manifest.json`: `passed` means that stage's
commands completed. Check every step's exit status and its local log. Model
accuracy must be assessed from the completed coverage and scored predictions.

The serving harness writes `summary.json`, `summary.csv` and `report.md`
beneath its output directory, plus per-trial official client output and
server/client logs. The configurable benchmark aggregates its generated results in
`performance_summary.json`, `performance_summary.csv` and
`performance_summary.md`. A complete run must have overall status `passed`,
`cases_passed == cases_expected`, and
`measured_runs_complete == measured_runs_expected`, with expected counts
derived from the requested models, modes, batch sizes and run count. Use
`--execute --resume` with the same output directory after an interrupted run;
source, model input, environment and policy fingerprints must match. Changed
settings require a new directory.

Report every trial value and its aggregate median, successful/failed requests,
actual output token counts, submitted batch and resident capacity. End-to-end
batch latency includes prefill, generation, queueing and client transport;
server startup and weight loading are excluded. TTFT and TPOT are separate
metrics. Use the same units and aggregation when comparing with Table 4;
per-request latency is not automatically equivalent to batch completion time.

Accuracy output contains per-task counts/scores and predictions when requested.
Inspect partial/skipped/too-long samples before computing the ten-task macro
average. NIAH output includes requested/effective depths, targets and both
normalized and raw match. A missing or failed cell remains missing; neither a
zero score nor an OOM label should be substituted for it. The guide and paper
supply the comparison protocol; the source archive contains no saved measured
numbers to import into a new result.

The paper does not specify every software version, random prompt, scheduler
budget, or timing boundary. Preserve the generated environment/input records
and report those differences. Exact numerical equality across hardware and
software revisions is not an acceptance criterion for functional correctness.
Full baseline speedup claims require the separately linked baseline systems.

## Troubleshooting and reruns

| Symptom | Check or action |
| --- | --- |
| Checkpoint download returns 401/403 | Confirm the exact model ID, upstream access approval and local `hf auth login`; do not put tokens in commands or artifact reports. |
| `nvcc` missing or extension compilation fails | Check CUDA 12.8, C++17 compiler and `TORCH_EXTENSIONS_DIR`; lower `MAX_JOBS` if compilation exhausts host memory. Keep the compiler error from the failed stage. |
| vLLM import/API failure | Recreate the pinned Python 3.11 environment, apply the supplied constraints, and run `python -m pip check`. |
| GPU OOM or host allocation failure | Check other processes, free GPU/host memory, checkpoint precision, and computed resident capacity. Smaller batches/contexts are useful diagnostics but change the experiment. |
| Missing LongBench task or NIAH filler | Point `JETFIRE_LONGBENCH_DATA_DIR` at the directory directly containing the ten original `task.jsonl` files. |
| An output directory already exists | Choose a fresh directory for an independent run. The artifact wrapper rejects reuse; lower-level accuracy tools support explicit protocol-checked resume. |
| Slow runs or timing variation | Check GPU ownership, CPU affinity/NUMA, worker count, throttling and competing DRAM traffic. Compare unprofiled trials with the same admission and client policy. |
| Stream decoding/request failure | Inspect the actual client/server failure and recorded decoder identity; retain it as a failed trial rather than dropping it from a median. |

Use `--help` on each entry point for its supported options. Rebuild extensions
when changing CUDA/compiler/runtime versions. Keep each run's generated records
in the chosen working area; they are reviewer outputs and are not part of the source release.

