# Model and dataset downloads

Download inputs separately from the source artifact. These upstream model cards
identify the checkpoints and their access terms:

| Model ID | Official files and model card | License/access |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | [Meta on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE); gated access |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/LICENSE) |
| `meta-llama/Llama-3.3-70B-Instruct` | [Meta on Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [Llama 3.3 Community License](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE); gated access; TP=2 |

Use the Hugging Face/Transformers safetensors checkpoints with BF16 weights,
their configuration files and tokenizer files. GGUF, AWQ, GPTQ and other weight
quantizations change the experiment. Qwen3-30B-A3B-Instruct-2507 is the specific
30B MoE checkpoint; its active parameter count does not reduce the storage of
all model weights.

## Download checkpoints

The installed environment provides the `hf` CLI. For Meta models, request
access and accept the upstream terms on the model page, then authenticate
locally:

```bash
hf auth login
```

Use the following immutable revisions for the download recipe. Retain these
IDs with experiment notes.

| Input | Immutable upstream revision |
| --- | --- |
| Llama 3.1 8B | [0e9e39f249a16976918f6564b8830bc894c89659](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) |
| Qwen 2507 | [0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe) |
| Llama 3.3 70B, optional | [6f6073b423013f6a7d4d9f39144961bfbfbc386b](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/tree/6f6073b423013f6a7d4d9f39144961bfbfbc386b) |
| LongBench v1 | [5e628be450b7e67fb7ae6e201bd6d8f7056f7672](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) |

```bash
export JETFIRE_LLAMA31_REVISION=0e9e39f249a16976918f6564b8830bc894c89659
export JETFIRE_QWEN3_REVISION=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
export JETFIRE_LLAMA31_8B=/absolute/path/to/models/Llama-3.1-8B-Instruct
export JETFIRE_QWEN3_30B=/absolute/path/to/models/Qwen3-30B-A3B-Instruct-2507
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision "$JETFIRE_LLAMA31_REVISION" --local-dir "$JETFIRE_LLAMA31_8B" \
  --exclude 'original/*'
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --revision "$JETFIRE_QWEN3_REVISION" --local-dir "$JETFIRE_QWEN3_30B"
```

The `original/` Llama files are a separate checkpoint format and are not needed
for this artifact. For optional 70B:

```bash
export JETFIRE_LLAMA33_70B=/absolute/path/to/models/Llama-3.3-70B-Instruct
hf download meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --local-dir "$JETFIRE_LLAMA33_70B" --exclude 'original/*'
```

Approximate BF16 weight storage is 16 GB for 8B, 61 GB
for Qwen 30B, and 141 GB for 70B, excluding tokenizer/configuration files and
download caches. These are planning estimates. Avoid keeping redundant formats
if disk space is limited.

The serving/evaluation tools record configuration/tokenizer hashes and model
file inventories where available; these do not constitute full weight-content
checksums. Record upstream revision IDs and verify weight checksums from the
upstream file metadata if full input authentication is required. See the
[Hugging Face CLI reference](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli)
for authentication, download filtering and revision selection.

## Original LongBench v1

Use the [original LongBench v1 instructions](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)
and its [official dataset repository](https://huggingface.co/datasets/zai-org/LongBench).
The earlier `THUDM/LongBench` dataset URL redirects to `zai-org/LongBench`.
Download the repository's `data.zip`; do not substitute LongBench v2 or the
`*_e.jsonl` LongBench-E files. Check the upstream dataset and constituent task
terms before use.

```bash
export JETFIRE_LONGBENCH_REVISION=5e628be450b7e67fb7ae6e201bd6d8f7056f7672
export JETFIRE_DATA_ROOT=/absolute/path/to/datasets/longbench_v1
mkdir -p "$JETFIRE_DATA_ROOT"
hf download zai-org/LongBench data.zip --repo-type dataset \
  --revision "$JETFIRE_LONGBENCH_REVISION" --local-dir "$JETFIRE_DATA_ROOT"
python -m zipfile -e "$JETFIRE_DATA_ROOT/data.zip" "$JETFIRE_DATA_ROOT"
export JETFIRE_LONGBENCH_DATA_DIR="$JETFIRE_DATA_ROOT/data"
```

Check that the configured directory directly contains these files:

```text
narrativeqa.jsonl     qasper.jsonl       multifieldqa_en.jsonl
hotpotqa.jsonl        2wikimqa.jsonl     musique.jsonl
gov_report.jsonl     qmsum.jsonl        trec.jsonl      samsum.jsonl
```

The ten-task full evaluation contains 1,950 examples: 150 MultiFieldQA examples
and 200 examples for each other task. NIAH uses this same local text as filler
and generates its needles deterministically; no separate NIAH download is
required. Dataset JSONL hashes are captured by the artifact entry point.

Models and data can be kept outside the checkout and reused across invocations.
Once all dependencies and inputs are present, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` if you require the run to avoid Hub access.
