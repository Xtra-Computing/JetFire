# Model and Dataset Downloads

Please download model checkpoints and evaluation data separately. The following
model repositories provide the checkpoints, model cards, and access terms:

| Model ID | Official files and model card | License/access |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | [Meta on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE); gated access |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/LICENSE) |
| `meta-llama/Llama-3.3-70B-Instruct` | [Meta on Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [Llama 3.3 Community License](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE); gated access; TP=2 |

The reference configuration uses Hugging Face/Transformers safetensors
checkpoints with BF16 weights, together with their configuration and tokenizer
files. GGUF, AWQ, GPTQ, and other weight quantizations change the experimental
configuration. Qwen3-30B-A3B-Instruct-2507 is a mixture-of-experts (MoE) model;
storage requirements reflect all model weights, including inactive experts.

## Download Checkpoints

The JetFire environment includes the `hf` command-line tool. For Meta models,
please request access and accept the terms on the model page, then sign in
locally:

```bash
hf auth login
```

The following immutable revisions identify the evaluation inputs. Please retain
these IDs with your experiment records.

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

The Llama download excludes `original/`, which contains a separate checkpoint
format. To include Llama 70B, use:

```bash
export JETFIRE_LLAMA33_70B=/absolute/path/to/models/Llama-3.3-70B-Instruct
hf download meta-llama/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --local-dir "$JETFIRE_LLAMA33_70B" --exclude 'original/*'
```

Allow approximately 16 GB for Llama 8B, 61 GB for Qwen 30B, and 141 GB for
Llama 70B BF16 weights. These estimates exclude configuration files, tokenizers,
and download caches. Keeping only the required checkpoint format saves disk
space.

The evaluation tools record configuration and tokenizer hashes, along with model
file inventories where available. To verify the weight contents, please retain
the upstream revision IDs and check the weight checksums against the upstream
file metadata; the recorded inventories do not include full weight checksums.
See the
[Hugging Face CLI reference](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli)
for authentication, download filtering, and revision selection.

## Original LongBench v1

Evaluation uses the [original LongBench v1](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)
from the [official dataset repository](https://huggingface.co/datasets/zai-org/LongBench).
The earlier `THUDM/LongBench` dataset URL redirects to `zai-org/LongBench`.
Please review the dataset and constituent task terms before downloading
`data.zip`. This evaluation uses the original task files; LongBench v2 and
the `*_e.jsonl` LongBench-E files define different evaluations.

```bash
export JETFIRE_LONGBENCH_REVISION=5e628be450b7e67fb7ae6e201bd6d8f7056f7672
export JETFIRE_DATA_ROOT=/absolute/path/to/datasets/longbench_v1
mkdir -p "$JETFIRE_DATA_ROOT"
hf download zai-org/LongBench data.zip --repo-type dataset \
  --revision "$JETFIRE_LONGBENCH_REVISION" --local-dir "$JETFIRE_DATA_ROOT"
python -m zipfile -e "$JETFIRE_DATA_ROOT/data.zip" "$JETFIRE_DATA_ROOT"
export JETFIRE_LONGBENCH_DATA_DIR="$JETFIRE_DATA_ROOT/data"
```

The configured directory should directly contain the following files:

```text
narrativeqa.jsonl     qasper.jsonl       multifieldqa_en.jsonl
hotpotqa.jsonl        2wikimqa.jsonl     musique.jsonl
gov_report.jsonl     qmsum.jsonl        trec.jsonl      samsum.jsonl
```

The full ten-task evaluation contains 1,950 examples: 150 for MultiFieldQA and
200 for each other task. Needle-in-a-Haystack (NIAH) uses the same local text
as filler and generates needles deterministically, so it requires no separate
download. The artifact entry point records the dataset JSONL hashes.

Models and data may be stored outside the checkout and reused across runs.
For offline evaluation, download all dependencies and inputs first, then set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to disable Hub access.
