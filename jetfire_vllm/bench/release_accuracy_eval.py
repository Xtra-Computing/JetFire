"""Release accuracy harness for LongBench v1/v2 and NIAH.

This script runs one model/backend/suite combination per process.  The
companion shell script ``run_release_accuracy_suite.sh`` launches the release
matrix requested for JetFire accuracy tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import string
import unicodedata
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCORING_PROTOCOL_VERSION = "jetfire_artifact_paper_v2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


MODEL_CONFIGS = {
    "llama31_8b": {
        "path": os.environ.get("JETFIRE_LLAMA31_8B", "/path/to/Llama-3.1-8B-Instruct"),
        "tp": 1,
        "max_context": 131072,
    },
    "llama33_70b": {
        "path": os.environ.get("JETFIRE_LLAMA33_70B", "/path/to/Llama-3.3-70B-Instruct"),
        "tp": 2,
        "max_context": 131072,
    },
    "qwen3_30b_a3b": {
        "path": os.environ.get("JETFIRE_QWEN3_30B", "/path/to/Qwen3-30B-A3B-Instruct"),
        "tp": 1,
        "max_context": 256080,
    },
}


BACKEND_CONFIGS = {
    "jetfire_k4v4": {
        "label": "jetfire_sym_int4k_int4v",
        "jetfire": True,
        "mode": "SymInt4K_SymInt4V",
    },
    "jetfire_int4": {
        "label": "jetfire_sym_int4",
        "jetfire": True,
        "mode": "SymInt4K_SymInt4V",
    },
    "jetfire_k4v3": {
        "label": "jetfire_sym_int4k_int3v",
        "jetfire": True,
        "mode": "SymInt4K_SymInt3V",
    },
    "jetfire_k4v2": {
        "label": "jetfire_sym_int4k_int2v",
        "jetfire": True,
        "mode": "SymInt4K_SymInt2V",
    },
    "jetfire_k3v3": {
        "label": "jetfire_sym_int3k_int3v",
        "jetfire": True,
        "mode": "SymInt3K_SymInt3V",
    },
    "jetfire_k3v4": {
        "label": "jetfire_sym_int3k_int4v",
        "jetfire": True,
        "mode": "SymInt3K_SymInt4V",
    },
    "jetfire_k3v2": {
        "label": "jetfire_sym_int3k_int2v",
        "jetfire": True,
        "mode": "SymInt3K_SymInt2V",
    },
    "jetfire_k2v4": {
        "label": "jetfire_sym_int2k_int4v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt4V",
    },
    "jetfire_k2v3": {
        "label": "jetfire_sym_int2k_int3v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt3V",
    },
    "jetfire_k2v2": {
        "label": "jetfire_sym_int2k_int2v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt2V",
    },
    "jetfire_sym_k2v4": {
        "label": "jetfire_sym_int2k_int4v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt4V",
    },
    "jetfire_sym_k2v3": {
        "label": "jetfire_sym_int2k_int3v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt3V",
    },
    "jetfire_sym_k2v2": {
        "label": "jetfire_sym_int2k_int2v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt2V",
    },
    "jetfire_int2": {
        "label": "jetfire_sym_int2k_int2v",
        "jetfire": True,
        "mode": "SymInt2K_SymInt2V",
    },
    "jetfire_k2v2_4level": {
        "label": "jetfire_asym_int2k4level_int2v",
        "jetfire": True,
        "mode": "AsymInt2K4Level_SymInt2V",
    },
}


LONGBENCH_V1_TASKS = [
    "qasper",
    "multifieldqa_en",
    "2wikimqa",
    "gov_report",
    "hotpotqa",
    "musique",
    "narrativeqa",
    "qmsum",
    "samsum",
    "trec",
]

LONGBENCH_V1_PROMPTS = {
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "narrativeqa": 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "samsum": 'Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}',
    "trec": 'Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}',
}

LONGBENCH_V1_MAX_NEW_TOKENS = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "2wikimqa": 32,
    "gov_report": 512,
    "hotpotqa": 32,
    "musique": 32,
    "narrativeqa": 128,
    "qmsum": 512,
    "samsum": 128,
    "trec": 64,
}

NO_CHAT_TEMPLATE_V1 = {"trec", "samsum"}
FIRST_LINE_V1 = {"trec", "samsum"}

# Measured with the release prompt/chat template and model tokenizer. LongBench
# v1 is not a 128K dataset; 64K leaves too little decode headroom for the
# longest NarrativeQA/GovReport samples after vLLM's internal prompt accounting.
# Use 96K as the default full-suite window to avoid unnecessary profile/KV
# pressure while still covering the full LongBench v1 workload.
LONGBENCH_V1_FULL_WINDOW = 98304
LONGBENCH_V1_FULL_WINDOW_BY_MODEL = {
    "llama33_70b": LONGBENCH_V1_FULL_WINDOW,
}

LONGBENCH_V2_SUBTASKS = {
    "128k": [
        ("Long-dialogue History Understanding", "Agent history QA"),
        ("Long-dialogue History Understanding", "Dialogue history QA"),
    ],
    "256k": [
        ("Long In-context Learning", "Many-shot learning"),
        ("Long-dialogue History Understanding", "Agent history QA"),
        ("Long-dialogue History Understanding", "Dialogue history QA"),
        ("Multi-Document QA", "Legal"),
        ("Multi-Document QA", "Multi-news"),
        ("Single-Document QA", "Detective"),
        ("Single-Document QA", "Financial"),
    ],
}

NIAH_FILLER_TASKS = [
    "gov_report",
    "multi_news",
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
]
NIAH_TOPICS = [
    "the Golden Gate Bridge",
    "Mount Kilimanjaro",
    "the Mariana Trench",
    "the Sahara Desert",
    "the Great Barrier Reef",
]
NIAH_DEFAULT_LB_DATA = Path(
    os.environ.get("JETFIRE_LONGBENCH_DATA_DIR", "/path/to/longbench_v1/data")
)
_NIAH_FILLER_TOKEN_CACHE: dict[tuple[int, str, int], list[int]] = {}


def _load_longbench_v1_metrics():
    from jetfire_vllm.bench.longbench_metrics import (
        classification_score,
        qa_f1_score,
        rouge_score,
    )

    return {
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "gov_report": rouge_score,
        "hotpotqa": qa_f1_score,
        "musique": qa_f1_score,
        "narrativeqa": qa_f1_score,
        "qmsum": rouge_score,
        "samsum": rouge_score,
        "trec": classification_score,
    }


def _fmt_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _suite_window(suite: str, model_key: str = "") -> int:
    if suite == "niah_32k":
        return 32768
    if suite == "longbench_v1" and model_key in LONGBENCH_V1_FULL_WINDOW_BY_MODEL:
        return LONGBENCH_V1_FULL_WINDOW_BY_MODEL[model_key]
    if suite == "longbench_v1":
        return LONGBENCH_V1_FULL_WINDOW
    if suite in ("longbench_v2_128k", "niah_128k"):
        return 131072
    if suite in ("longbench_v2_256k", "niah_256k"):
        return 256080
    raise ValueError(f"unknown suite: {suite}")


def _manual_prefill_chunk_tokens(args: argparse.Namespace) -> int | None:
    from jetfire_vllm.bench.official_serve_benchmark import (
        _parse_prefill_chunk_tokens,
    )

    return _parse_prefill_chunk_tokens(
        getattr(args, "jetfire_prefill_chunk_tokens", "auto")
    )


def _accuracy_prefill_partition_args(
    *,
    model_path: str,
    backend: str,
    max_model_len: int,
    tensor_parallel_size: int,
    args: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        model=model_path,
        tensor_parallel_size=int(tensor_parallel_size),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        ctx=int(max_model_len),
        output_len=0,
        max_model_len=int(max_model_len),
        max_model_len_padding=0,
        concurrency=1,
        kv_cache_memory_bytes=None,
        kv_cache_block_size=int(args.kv_cache_block_size),
        jetfire_kv_cache_budget="auto",
        jetfire_kv_budget_safety=float(args.jetfire_kv_budget_safety),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        jetfire_prefill_chunk_tokens=_manual_prefill_chunk_tokens(args),
        jetfire_mode=BACKEND_CONFIGS[backend]["mode"],
        jetfire_group_size=int(args.jetfire_group_size),
    )


def _accuracy_prefill_partition_metadata(
    *,
    model_path: str,
    backend: str,
    max_model_len: int,
    tensor_parallel_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not BACKEND_CONFIGS[backend]["jetfire"]:
        return {}
    from jetfire_vllm.bench.official_serve_benchmark import (
        _prefill_partition_model_scale,
        _prefill_partition_shape,
        _prefill_scratch_budget_gib,
        _resolved_prefill_partition_profile,
    )

    partition_args = _accuracy_prefill_partition_args(
        model_path=model_path,
        backend=backend,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        args=args,
    )
    profile = _resolved_prefill_partition_profile(partition_args, 1)
    shape = _prefill_partition_shape(partition_args)
    scratch = _prefill_scratch_budget_gib(partition_args, 1)
    scale = _prefill_partition_model_scale(partition_args)
    return {
        "jetfire_prefill_partition_profile": profile.name,
        "jetfire_prefill_scratch_budget_gib": scratch,
        "jetfire_prefill_normalized_scratch_gib": (
            scratch / scale if scratch is not None else None
        ),
        "jetfire_prefill_partition_model_scale": scale,
        "jetfire_prefill_partition_hidden": shape["hidden"],
        "jetfire_prefill_partition_intermediate": shape["intermediate"],
        "jetfire_embed_chunk_tokens": profile.embed,
        "jetfire_mlp_chunk_tokens": profile.mlp,
        "jetfire_attn_o_proj_chunk_tokens": profile.attn_o_proj,
        "jetfire_prefill_attn_chunk_tokens": profile.prefill_attn,
        "jetfire_prefill_qkv_chunk_tokens": profile.prefill_qkv,
    }


def _configure_backend(
    model_path: str,
    backend: str,
    args: argparse.Namespace,
    *,
    max_model_len: int,
    tensor_parallel_size: int,
) -> bool:
    cfg = BACKEND_CONFIGS[backend]
    use_jetfire = bool(cfg["jetfire"])
    if not use_jetfire:
        os.environ["JETFIRE_USE_CUSTOM_BACKEND"] = "0"
        return False

    os.environ["JETFIRE_USE_CUSTOM_BACKEND"] = "1"
    os.environ["JETFIRE_MODEL_PATH"] = model_path
    os.environ["JETFIRE_MODE"] = str(cfg["mode"])
    os.environ["JETFIRE_GROUP_SIZE"] = str(args.jetfire_group_size)
    os.environ["JETFIRE_TOPK"] = str(args.top_k)
    os.environ["JETFIRE_MAX_BATCH_SIZE"] = str(args.max_num_seqs)
    os.environ["JETFIRE_CPU_ATTN_THREADS"] = str(args.cpu_attn_threads)
    os.environ.setdefault("JETFIRE_CPU_CORRECTION", "1")
    os.environ.setdefault("JETFIRE_DECODE_CPU_ATTN", "1")
    os.environ.setdefault("JETFIRE_SKIP_NATIVE_KV", "1")
    if args.jetfire_approx_only:
        os.environ["JETFIRE_DEBUG_APPROX_ONLY"] = "1"
        os.environ["JETFIRE_RELEASE_ACCURACY_ABLATION"] = "1"
    else:
        os.environ.pop("JETFIRE_DEBUG_APPROX_ONLY", None)
        os.environ.pop("JETFIRE_RELEASE_ACCURACY_ABLATION", None)
    sync_cpu_attn = not (args.jetfire_graph or args.jetfire_async_cpu_correction)
    os.environ["JETFIRE_SYNC_CPU_ATTN"] = "1" if sync_cpu_attn else "0"
    if sync_cpu_attn:
        # Eager smoke/debug runs use inline CPU correction.  Keep that path on
        # the OMP fallback by default: the custom pool is optimized for the
        # release graph/async path but can rank-stall under TP=2 sync sweeps.
        os.environ.setdefault("JETFIRE_CUSTOM_POOL", "0")
    os.environ.setdefault("JETFIRE_USE_PERSISTENT_WORKER", "0")
    os.environ.setdefault("JETFIRE_FUSED_JOIN_BLEND", "0")
    os.environ.setdefault("JETFIRE_CPU_FUTURE_TIMEOUT_S", "120")
    os.environ.setdefault("JETFIRE_FAST_STORE_COMPLETE_CHECK", "1")
    os.environ.setdefault("JETFIRE_PRELOAD_DECODE_EXTENSIONS", "1")
    from jetfire_vllm.bench.official_serve_benchmark import (
        PREFILL_CHUNK_ENV_KEYS,
        PREFILL_PARTITION_PROFILE_ENV,
        _prefill_partition_shape,
        _resolved_prefill_partition_profile,
    )

    partition_args = _accuracy_prefill_partition_args(
        model_path=model_path,
        backend=backend,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        args=args,
    )
    profile = _resolved_prefill_partition_profile(partition_args, 1)
    if profile.prefill_qkv > 0:
        os.environ.setdefault("JETFIRE_EMBED_EMPTY_CACHE", "1")
        os.environ[PREFILL_PARTITION_PROFILE_ENV] = profile.name
        partition_shape = _prefill_partition_shape(partition_args)
        for key, value in partition_shape.items():
            os.environ[f"JETFIRE_PREFILL_PARTITION_{key.upper()}"] = str(value)
        for key, value in profile.env_values().items():
            os.environ[key] = str(value)
    else:
        os.environ.pop(PREFILL_PARTITION_PROFILE_ENV, None)
        for key in _prefill_partition_shape(partition_args):
            os.environ.pop(f"JETFIRE_PREFILL_PARTITION_{key.upper()}", None)
        for key in PREFILL_CHUNK_ENV_KEYS:
            os.environ.pop(key, None)
    os.environ["JETFIRE_DECODE_GRAPH"] = "1" if args.jetfire_graph else "0"
    os.environ["JETFIRE_ENABLE_VLLM_CUDAGRAPH"] = "1" if args.jetfire_graph else "0"
    strict_release_graph = args.jetfire_graph
    os.environ["JETFIRE_RELEASE_GRAPH_POLICY"] = "1" if strict_release_graph else "0"
    os.environ["JETFIRE_RELEASE_REQUIRE_READY"] = "1" if strict_release_graph else "0"
    if args.jetfire_graph:
        os.environ.setdefault("JETFIRE_ZERO_INIT_CACHE", "0")
    if strict_release_graph:
        from jetfire_vllm.adapter.config import apply_graph_cpu_topk_release_preset

        apply_graph_cpu_topk_release_preset()
    os.environ["JETFIRE_DISABLE_ASYNC_SCHEDULING"] = "1"
    return True


def _jetfire_accuracy_num_gpu_blocks(max_model_len: int, args: argparse.Namespace) -> int:
    if args.jetfire_num_gpu_blocks_override > 0:
        return int(args.jetfire_num_gpu_blocks_override)
    if args.jetfire_num_gpu_blocks_override < 0:
        return 0
    block_size = max(1, int(args.kv_cache_block_size))
    safety = max(1.0, float(args.jetfire_kv_budget_safety))
    return max(1, int(math.ceil(max_model_len * safety / block_size)))


def _load_llm(model_key: str, backend: str, suite: str, args: argparse.Namespace):
    model_cfg = MODEL_CONFIGS[model_key]
    model_path = str(model_cfg["path"])
    max_model_len = int(args.max_model_len or _suite_window(suite, model_key))
    tensor_parallel_size = int(args.tensor_parallel_size or model_cfg["tp"])
    use_jetfire = _configure_backend(
        model_path,
        backend,
        args,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    llm_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "enable_prefix_caching": False,
        "trust_remote_code": bool(args.trust_remote_code),
        "seed": int(args.seed),
    }
    if args.moe_backend:
        llm_kwargs["moe_backend"] = str(args.moe_backend)
    if int(llm_kwargs["tensor_parallel_size"]) > 1:
        llm_kwargs["disable_custom_all_reduce"] = True
    if use_jetfire:
        llm_kwargs["enable_chunked_prefill"] = False
        num_blocks_override = _jetfire_accuracy_num_gpu_blocks(max_model_len, args)
        if num_blocks_override > 0:
            llm_kwargs["num_gpu_blocks_override"] = num_blocks_override
        if args.jetfire_graph:
            llm_kwargs["enforce_eager"] = False
            llm_kwargs["compilation_config"] = {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
                "max_cudagraph_capture_size": 1,
            }
        else:
            llm_kwargs["enforce_eager"] = True
        from jetfire_vllm.adapter.engine import build_llm

        return build_llm(
            model=model_path,
            use_jetfire_backend=True,
            llm_kwargs=llm_kwargs,
        )



def _load_tokenizer(model_key: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_CONFIGS[model_key]["path"]),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _sample_window(items: list[Any], args: argparse.Namespace) -> list[tuple[int, Any]]:
    start = max(0, int(getattr(args, "sample_start", 0)))
    end_arg = int(getattr(args, "sample_end", 0))
    end = end_arg if end_arg > 0 else len(items)
    out = list(enumerate(items))[start:end]
    if args.max_samples > 0:
        return out[:args.max_samples]
    return out


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_len(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def _capacity_budget_tokens(
    prompt_tokens: int,
    max_new_tokens: int,
    args: argparse.Namespace,
) -> int:
    # vLLM may account for a few internal/template tokens that are not visible
    # in the harness-side encode(add_special_tokens=False) length.  Reserve a
    # small margin so near-boundary samples are skipped instead of replaying a
    # decode graph into the final KV slot and stalling the engine.
    return (
        int(prompt_tokens)
        + int(max_new_tokens)
        + int(args.generation_token_margin)
    )


def _generate_one(llm: Any, prompt: str, max_tokens: int, stop: list[str] | None):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=int(max_tokens),
        stop=stop or [],
    )
    return llm.generate([prompt], params, use_tqdm=False)[0].outputs[0].text.strip()


def _maybe_reset_jetfire_runtime_state(llm: Any, args: argparse.Namespace) -> None:
    if not BACKEND_CONFIGS[args.backend]["jetfire"]:
        return
    if not bool(getattr(args, "jetfire_reset_runtime_between_samples", True)):
        return
    try:
        from jetfire_vllm.adapter.engine import reset_jetfire_worker_runtime_state

        reset_jetfire_worker_runtime_state(getattr(llm, "_llm", llm))
    except Exception as exc:
        print(
            "[jetfire][warn] failed to reset runtime state after sample: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _score_longbench_v1_prediction(
    task: str,
    prediction: str,
    sample: dict[str, Any],
    metric_fn: Any,
) -> float:
    if task in FIRST_LINE_V1:
        prediction = prediction.lstrip("\n").split("\n")[0]
    answers = sample["answers"] if isinstance(sample["answers"], list) else [sample["answers"]]
    all_classes = sample.get("all_classes", None)
    return max(float(metric_fn(prediction, gt, all_classes=all_classes)) for gt in answers)


def _load_longbench_v1_dataset(task: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    local_dir = str(getattr(args, "longbench_v1_data_dir", "") or "")
    if local_dir:
        path = Path(local_dir).expanduser() / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"LongBench task file unavailable: {path}. Supply the official LongBench v1 data directory.")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    from datasets import load_dataset

    return list(load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True))


def run_longbench_v1(
    llm: Any,
    tokenizer: Any,
    out_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = _load_longbench_v1_metrics()
    tasks = args.tasks.split(",") if args.tasks else LONGBENCH_V1_TASKS
    results: dict[str, Any] = _load_existing(out_file, args.resume)

    for task in tasks:
        task = task.strip()
        if not task:
            continue
        if (
            args.resume
            and task in results
            and not results[task].get("partial")
            and results[task].get("scoring_protocol") == SCORING_PROTOCOL_VERSION
        ):
            print(f"[skip] {task} already complete", flush=True)
            continue
        dataset = _load_longbench_v1_dataset(task, args)
        total_samples = len(dataset)
        dataset = _sample_window(dataset, args)
        metric_fn = metrics[task]
        max_new_tokens = LONGBENCH_V1_MAX_NEW_TOKENS[task]
        scores = []
        examples = []
        too_long = 0
        completed_indices: set[int] = set()
        existing = results.get(task, {}) if args.resume else {}
        existing_examples = existing.get("examples", [])
        existing_indices = [
            int(example["idx"])
            for example in existing_examples
            if "idx" in example
        ]
        current_indices = {idx for idx, _sample in dataset}
        can_resume_partial = (
            bool(existing.get("partial"))
            and existing.get("scoring_protocol") == SCORING_PROTOCOL_VERSION
            and int(existing.get("too_long", 0)) == 0
            and len(existing_examples) == int(existing.get("num_samples", 0))
            and all("idx" in example and "score" in example
                    for example in existing_examples)
            and len(set(existing_indices)) == len(existing_indices)
            and set(existing_indices).issubset(current_indices)
        )
        if can_resume_partial:
            examples = list(existing_examples)
            scores = [float(example["score"]) for example in examples]
            completed_indices = set(existing_indices)
            print(
                f"[resume] {task} continuing after "
                f"{len(completed_indices)} completed samples",
                flush=True,
            )
        elif existing.get("partial"):
            print(
                f"[resume] {task} partial state is not safely resumable; "
                "restarting the task",
                flush=True,
            )
        for idx, sample in dataset:
            if idx in completed_indices:
                continue
            prompt = LONGBENCH_V1_PROMPTS[task].format(**sample)
            if task not in NO_CHAT_TEMPLATE_V1:
                prompt = _apply_chat_template(tokenizer, prompt)
            prompt_tokens = _token_len(tokenizer, prompt)
            total_budget_tokens = _capacity_budget_tokens(
                prompt_tokens, max_new_tokens, args)
            if (
                int(getattr(args, "only_over_model_len", 0)) > 0
                and total_budget_tokens <= int(args.only_over_model_len)
            ):
                continue
            max_model_len = int(
                args.max_model_len or _suite_window(args.suite, args.model_key))
            if total_budget_tokens > max_model_len:
                too_long += 1
                continue
            print(
                f"[{task}] sample {idx + 1}/{total_samples} "
                f"tokens={prompt_tokens}",
                flush=True,
            )
            pred = _generate_one(
                llm,
                prompt,
                max_new_tokens,
                ["<|eot_id|>", "<|end_of_text|>"],
            )
            score = _score_longbench_v1_prediction(task, pred, sample, metric_fn)
            scores.append(score)
            if args.save_predictions:
                examples.append({
                    "idx": idx,
                    "prompt_tokens": prompt_tokens,
                    "prediction": pred,
                    "answers": sample.get("answers"),
                    "all_classes": sample.get("all_classes"),
                    "score": score,
                })
            _write_partial(results, out_file, task, scores, too_long, examples)
            _maybe_reset_jetfire_runtime_state(llm, args)
        _write_partial(results, out_file, task, scores, too_long, examples, partial=False)

    _write_overall(results, out_file)
    return results


def _build_longbench_v2_prompt(row: dict[str, Any]) -> str:
    choices = (
        f"A. {row['choice_A']}\n"
        f"B. {row['choice_B']}\n"
        f"C. {row['choice_C']}\n"
        f"D. {row['choice_D']}"
    )
    return (
        "Please read the following text and answer the questions below.\n\n"
        "<text>\n"
        f"{row['context']}\n"
        "</text>\n\n"
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' "
        "(without quotes) where [LETTER] is one of A, B, C, or D. "
        "Think step by step before answering.\n\n"
        f"{row['question']}\n\n"
        f"{choices}"
    )


def _parse_mcq_answer(text: str) -> str:
    patterns = [
        r"ANSWER\s*:\s*([ABCD])",
        r"Answer\s*:\s*([ABCD])",
        r"\b([ABCD])\b\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _selected_longbench_v2_subtasks(suite: str) -> list[tuple[str, str]]:
    if suite == "longbench_v2_128k":
        return LONGBENCH_V2_SUBTASKS["128k"]
    if suite == "longbench_v2_256k":
        return LONGBENCH_V2_SUBTASKS["256k"]
    raise ValueError(f"not a LongBench v2 suite: {suite}")


def run_longbench_v2(
    llm: Any,
    tokenizer: Any,
    out_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from datasets import load_dataset

    if args.longbench_v2_data_file:
        path = Path(args.longbench_v2_data_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LongBench v2 data file unavailable: {path}")
        with path.open(encoding="utf-8") as handle:
            dataset = [json.loads(line) for line in handle if line.strip()] if path.suffix == ".jsonl" else json.load(handle)
        if not isinstance(dataset, list):
            raise ValueError("LongBench v2 local data must be a JSON array or JSONL rows")
    else:
        dataset = list(load_dataset("THUDM/LongBench-v2", split="train"))
    selected = set(_selected_longbench_v2_subtasks(args.suite))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        key = (row["domain"], row["sub_domain"])
        if key in selected:
            by_task[f"{key[0]} / {key[1]}"].append(row)

    results: dict[str, Any] = _load_existing(out_file, args.resume)
    max_input = int(args.max_model_len or _suite_window(args.suite, args.model_key))
    for task_name in sorted(by_task):
        if (
            args.resume
            and task_name in results
            and not results[task_name].get("partial")
            and results[task_name].get("scoring_protocol")
            == SCORING_PROTOCOL_VERSION
        ):
            print(f"[skip] {task_name} already complete", flush=True)
            continue
        rows = _sample_window(by_task[task_name], args)
        total_samples = len(by_task[task_name])
        scores = []
        examples = []
        too_long = 0
        for idx, row in rows:
            prompt = _build_longbench_v2_prompt(row)
            prompt_tokens = _token_len(tokenizer, prompt)
            if _capacity_budget_tokens(
                    prompt_tokens,
                    args.longbench_v2_max_new_tokens,
                    args) > max_input:
                too_long += 1
                print(
                    f"[too-long] {task_name} sample={idx} tokens={prompt_tokens}",
                    flush=True,
                )
                continue
            print(
                f"[{task_name}] sample {idx + 1}/{total_samples} "
                f"tokens={prompt_tokens}",
                flush=True,
            )
            pred = _generate_one(
                llm,
                prompt,
                args.longbench_v2_max_new_tokens,
                ["<|eot_id|>", "<|end_of_text|>"],
            )
            parsed = _parse_mcq_answer(pred)
            score = 1.0 if parsed == str(row["answer"]).strip().upper() else 0.0
            scores.append(score)
            if args.save_predictions:
                examples.append({
                    "idx": idx,
                    "_id": row.get("_id"),
                    "prompt_tokens": prompt_tokens,
                    "prediction": pred,
                    "parsed": parsed,
                    "answer": row.get("answer"),
                    "score": score,
                })
            _write_partial(results, out_file, task_name, scores, too_long, examples)
            _maybe_reset_jetfire_runtime_state(llm, args)
        _write_partial(results, out_file, task_name, scores, too_long, examples, partial=False)

    _write_overall(results, out_file)
    return results


def _load_niah_filler_text(args: argparse.Namespace) -> str:
    data_dir_arg = str(getattr(args, "niah_filler_data_dir", "") or "")
    data_dir = Path(data_dir_arg) if data_dir_arg else NIAH_DEFAULT_LB_DATA
    parts: list[str] = []
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.jsonl")):
            if not any(task in path.name for task in NIAH_FILLER_TASKS):
                continue
            with path.open() as f:
                for line in f:
                    try:
                        context = json.loads(line).get("context", "") or ""
                    except json.JSONDecodeError:
                        continue
                    if context:
                        parts.append(context)
    if parts:
        return "\n\n".join(parts)

    from datasets import load_dataset

    for task in NIAH_FILLER_TASKS:
        dataset = load_dataset(
            "THUDM/LongBench",
            task,
            split="test",
            trust_remote_code=True,
        )
        for row in dataset:
            context = row.get("context", "") or ""
            if context:
                parts.append(context)
    if not parts:
        raise RuntimeError("NIAH filler text is empty")
    return "\n\n".join(parts)


def _niah_filler_token_budget(args: argparse.Namespace) -> int:
    if args.niah_contexts:
        max_context = max(
            int(value) for value in args.niah_contexts.split(",") if value.strip()
        )
    else:
        max_context = int(
            args.max_model_len or _suite_window(args.suite, args.model_key)
        )
    repeat_span = max(1, int(args.niah_repeats)) * 9973
    return max_context + repeat_span + 4096


def _get_niah_filler_ids(tokenizer: Any, args: argparse.Namespace) -> list[int]:
    source = str(getattr(args, "niah_filler_data_dir", "") or NIAH_DEFAULT_LB_DATA)
    token_budget = _niah_filler_token_budget(args)
    key = (id(tokenizer), source, token_budget)
    if key not in _NIAH_FILLER_TOKEN_CACHE:
        text = _load_niah_filler_text(args)
        # Repeats use deterministic offsets of 9,973 tokens.  Encoding the
        # entire LongBench filler corpus can create more than 10M Python ints,
        # although a 128K/10-repeat grid only reads the first ~220K.  Encode an
        # exact text prefix with ample look-ahead, growing it only if needed.
        char_limit = min(len(text), max(4096, token_budget * 5))
        while True:
            filler_ids = tokenizer(
                text[:char_limit],
                add_special_tokens=False,
            ).input_ids
            if len(filler_ids) >= token_budget or char_limit >= len(text):
                break
            char_limit = min(len(text), char_limit * 2)
        if not filler_ids:
            raise RuntimeError("NIAH filler text produced no tokens; supply nonempty LongBench contexts")
        _NIAH_FILLER_TOKEN_CACHE[key] = filler_ids[:token_budget]
    return _NIAH_FILLER_TOKEN_CACHE[key]


def _niah_sample_spec(sample_idx: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed * 1000 + sample_idx)
    topic = NIAH_TOPICS[sample_idx % len(NIAH_TOPICS)]
    number = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return topic, number


def _niah_effective_depth(depth_pct: int, edge_margin_pct: int) -> float:
    margin = max(0.0, min(49.0, float(edge_margin_pct)))
    return margin + (float(depth_pct) / 100.0) * (100.0 - 2.0 * margin)


def _normalize_niah_output(text: str) -> str:
    """Paper protocol: remove punctuation and whitespace from the whole output."""
    return "".join(
        char for char in (text or "")
        if not char.isspace() and char not in string.punctuation
        and not unicodedata.category(char).startswith("P")
    )


def _score_niah(prediction: str, magic_number: str) -> bool:
    return _normalize_niah_output(prediction) == _normalize_niah_output(magic_number)


def _make_niah_prompt(
    tokenizer: Any,
    target_tokens: int,
    depth_pct: int,
    sample_idx: int,
    args: argparse.Namespace,
) -> tuple[str, str, str, float]:
    topic, number = _niah_sample_spec(sample_idx, int(args.niah_sample_seed))
    needle = f" The special magic number for {topic} is: {number}. "
    question = (
        f"\n\nWhat is the special magic number for {topic}? "
        "Answer with the number only, nothing else."
    )
    preamble = (
        "Below is a long document. Read it carefully; somewhere inside it is "
        "one special magic number. You will be asked to recall it.\n\n"
    )

    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    preamble_ids = tokenizer(preamble, add_special_tokens=False).input_ids
    hay_budget = (
        int(target_tokens)
        - len(needle_ids)
        - len(question_ids)
        - len(preamble_ids)
        - int(args.niah_chat_overhead_tokens)
    )
    hay_budget = max(100, hay_budget)

    filler = _get_niah_filler_ids(tokenizer, args)
    if len(filler) < hay_budget + 1:
        repeat = math.ceil((hay_budget + 1) / max(1, len(filler)))
        filler = (filler * repeat)[: hay_budget + 1]
    start_mod = max(1, len(filler) - hay_budget - 1)
    start = (sample_idx * 9973) % start_mod
    hay = filler[start:start + hay_budget]

    effective_depth = _niah_effective_depth(
        depth_pct,
        int(args.niah_edge_margin_pct),
    )
    pos = int(len(hay) * effective_depth / 100.0)
    body_ids = preamble_ids + hay[:pos] + needle_ids + hay[pos:] + question_ids
    body_text = tokenizer.decode(body_ids, skip_special_tokens=True)
    prompt = _apply_chat_template(tokenizer, body_text)
    return prompt, number, topic, effective_depth


def run_niah(
    llm: Any,
    tokenizer: Any,
    out_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.suite == "niah_32k":
        max_ctx = 32768
    elif args.suite == "niah_128k":
        max_ctx = 131072
    else:
        max_ctx = 256000
    if args.niah_contexts:
        contexts = [int(x) for x in args.niah_contexts.split(",") if x.strip()]
    else:
        if args.suite == "niah_32k":
            contexts = [32768]
        else:
            contexts = [8192, 16384, 32768, 65536, max_ctx]
            contexts = sorted(set(ctx for ctx in contexts if ctx <= max_ctx))
            if max_ctx == 256000:
                contexts.insert(-1, 131072)
    depths = [int(x) for x in args.niah_depths.split(",") if x.strip()]
    results: dict[str, Any] = _load_existing(out_file, args.resume)
    task_name = args.suite
    if (
        args.resume
        and task_name in results
        and not results[task_name].get("partial")
        and results[task_name].get("scoring_protocol")
        == SCORING_PROTOCOL_VERSION
    ):
        return results

    rows = []
    scores = []
    raw_scores = []
    max_cases = args.max_samples if args.max_samples > 0 else None
    case_idx = 0
    for ctx in contexts:
        for depth in depths:
            for rep in range(args.niah_repeats):
                if max_cases is not None and case_idx >= max_cases:
                    break
                prompt, value, topic, effective_depth = _make_niah_prompt(
                    tokenizer, ctx, depth, rep, args)
                prompt_tokens = _token_len(tokenizer, prompt)
                max_model_len = int(args.max_model_len or _suite_window(args.suite, args.model_key))
                if _capacity_budget_tokens(prompt_tokens, 32, args) > max_model_len:
                    raise ValueError(
                        f"NIAH prompt ({prompt_tokens} tokens) plus generation and margin "
                        f"exceeds max_model_len={max_model_len}; lower --niah-contexts "
                        "or increase --max-model-len."
                    )
                print(
                    f"[{task_name}] case={case_idx} ctx={ctx} depth={depth} "
                    f"effective_depth={effective_depth} tokens={prompt_tokens}",
                    flush=True,
                )
                pred = _generate_one(llm, prompt, 32, ["<|eot_id|>", "<|end_of_text|>"])
                ok = _score_niah(pred, value)
                raw_ok = pred.strip() == value.strip()
                raw_scores.append(float(raw_ok))
                scores.append(1.0 if ok else 0.0)
                rows.append({
                    "case": case_idx,
                    "target_context": ctx,
                    "depth_pct": depth,
                    "effective_depth_pct": effective_depth,
                    "prompt_tokens": prompt_tokens,
                    "topic": topic,
                    "magic_number": value,
                    "prediction": pred if args.save_predictions else "",
                    "raw_strip_exact_match": raw_ok,
                    "normalized_exact_match": ok,
                    "ok": ok,
                })
                results[task_name] = {
                    "score": sum(scores) / len(scores) * 100.0,
                    "normalized_exact_match": sum(scores) / len(scores) * 100.0,
                    "raw_strip_exact_match": sum(raw_scores) / len(raw_scores) * 100.0,
                    "normalization": "remove_punctuation_and_whitespace_from_entire_output",
                    "num_samples": len(scores),
                    "partial": True,
                    "scoring_protocol": SCORING_PROTOCOL_VERSION,
                    "cases": rows,
                }
                _save_json(out_file, results)
                _maybe_reset_jetfire_runtime_state(llm, args)
                case_idx += 1
            if max_cases is not None and case_idx >= max_cases:
                break
        if max_cases is not None and case_idx >= max_cases:
            break

    results[task_name]["partial"] = False
    results["status"] = "complete"
    results["overall"] = results[task_name]["score"]
    _save_json(out_file, results)
    return results


def _load_existing(path: Path, resume: bool) -> dict[str, Any]:
    if resume and path.exists():
        return json.loads(path.read_text())
    return {}


def _write_partial(
    results: dict[str, Any],
    out_file: Path,
    task: str,
    scores: list[float],
    too_long: int,
    examples: list[dict[str, Any]],
    partial: bool = True,
) -> None:
    score = sum(scores) / len(scores) * 100.0 if scores else 0.0
    row: dict[str, Any] = {
        "score": score,
        "num_samples": len(scores),
        "too_long": too_long,
        "partial": partial,
        "scoring_protocol": SCORING_PROTOCOL_VERSION,
    }
    if examples:
        row["examples"] = examples
    results[task] = row
    _save_json(out_file, results)


def _write_overall(results: dict[str, Any], out_file: Path) -> None:
    task_scores = [
        float(v["score"])
        for k, v in results.items()
        if k != "overall"
        and isinstance(v, dict)
        and not v.get("partial")
        and "error" not in v
        and int(v.get("num_samples", 0)) > 0
    ]
    results["overall"] = sum(task_scores) / len(task_scores) if task_scores else 0.0
    task_rows = [v for v in results.values() if isinstance(v, dict) and "score" in v]
    incomplete = not task_rows or any(
        row.get("partial") or row.get("error") or int(row.get("too_long", 0)) > 0
        or int(row.get("num_samples", 0)) == 0 for row in task_rows
    )
    results["status"] = "incomplete" if incomplete else "complete"
    _save_json(out_file, results)


def _maybe_print_jetfire_worker_stats(llm: Any, args: argparse.Namespace) -> None:
    if os.environ.get("JETFIRE_PRINT_WORKER_STATS", "0") != "1":
        return
    if not BACKEND_CONFIGS[args.backend]["jetfire"]:
        return
    try:
        from jetfire_vllm.adapter.engine import collect_jetfire_worker_stats

        rows = collect_jetfire_worker_stats(getattr(llm, "_llm", llm))
    except Exception as exc:
        print(
            "[jetfire][worker-stats][warn] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    summary = []
    for row in rows:
        summary.append({
            "rank": row.get("worker_rank"),
            "world_size": row.get("worker_world_size"),
            "device": row.get("worker_device"),
            "cuda": row.get("worker_cuda_device"),
            "decode_graph": row.get("decode_graph_enabled"),
            "device_signal": row.get("graph_cpu_topk_device_signal"),
            "host_join": row.get("graph_cpu_topk_host_join"),
            "side_join": row.get("graph_cpu_topk_side_join"),
            "gpu_wait": row.get("graph_cpu_topk_gpu_wait"),
            "decode_attempts": row.get("decode_attempts"),
            "decode_hits": row.get("decode_hits"),
            "decode_fallbacks": row.get("decode_fallbacks"),
            "graph_submit_count": row.get("graph_submit_count"),
        })
    print(
        "[jetfire][worker-stats] "
        + json.dumps(summary, sort_keys=True, ensure_ascii=False),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-path", default="",
                        help="Local model directory or Hugging Face model ID; overrides the model-key path.")
    parser.add_argument("--backend", choices=sorted(BACKEND_CONFIGS), required=True)
    parser.add_argument(
        "--suite",
        choices=(
            "longbench_v1",
            "longbench_v2_128k",
            "longbench_v2_256k",
            "niah_32k",
            "niah_128k",
            "niah_256k",
        ),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--longbench-v1-data-dir", default=os.environ.get("JETFIRE_LONGBENCH_DATA_DIR", ""),
                        help="Directory of official task.jsonl files; omit to use THUDM/LongBench from the datasets cache/hub.")
    parser.add_argument("--longbench-v2-data-file", default=os.environ.get("JETFIRE_LONGBENCH_V2_DATA_FILE", ""),
                        help="Optional official LongBench v2 JSON/JSONL file.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=0)
    parser.add_argument(
        "--only-over-model-len",
        type=int,
        default=0,
        help=(
            "LongBench v1 repair mode: only run samples whose prompt plus "
            "generation budget exceeds this length. Default 0 disables it."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument(
        "--generation-token-margin",
        type=int,
        default=32,
        help=(
            "Extra capacity guard tokens reserved beyond prompt+max_new_tokens. "
            "This avoids near-boundary decode graph stalls when vLLM accounts "
            "for internal prompt tokens not counted by the harness tokenizer."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--moe-backend",
        default="",
        help="Optional vLLM MoE backend override, recorded in run metadata.",
    )
    parser.add_argument("--kv-cache-block-size", type=int, default=16)
    parser.add_argument("--jetfire-kv-budget-safety", type=float, default=1.0)
    parser.add_argument(
        "--jetfire-num-gpu-blocks-override",
        type=int,
        default=0,
        help=(
            "JetFire accuracy KV block budget. 0 means derive from "
            "max_model_len / kv_cache_block_size; positive means explicit "
            "num_gpu_blocks_override; negative means leave vLLM's planner "
            "uncapped."
        ),
    )
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--jetfire-prefill-chunk-tokens",
        default="auto",
        help=(
            "Debug/manual uniform JetFire prefill activation chunk size. "
            "Default auto uses the release scratch-budget and model-shape "
            "partition profile. Use 0/off to disable chunking."
        ),
    )
    parser.add_argument(
        "--jetfire-group-size",
        type=int,
        default=int(os.environ.get("JETFIRE_GROUP_SIZE", "64")),
        help=(
            "JetFire per-head-dim quantization group size. This is written "
            "explicitly into the worker environment."
        ),
    )
    parser.add_argument("--cpu-attn-threads", type=int, default=32)
    parser.add_argument("--jetfire-graph", action="store_true", default=False)
    parser.add_argument("--jetfire-eager", dest="jetfire_graph", action="store_false")
    parser.add_argument(
        "--jetfire-approx-only",
        action="store_true",
        help=(
            "Accuracy-only k=0 ablation: return the dense low-precision GPU "
            "attention output without blending the CPU BF16 correction. "
            "With --jetfire-graph this keeps the same strict release kernel "
            "policy and graph as corrected ATC, then exposes Stage-B o_hat. "
            "Its latency is not reported because the unused correction runs."
        ),
    )
    parser.add_argument(
        "--jetfire-async-cpu-correction",
        action="store_true",
        help=(
            "Use JetFire's Python async CPU correction path in eager mode. "
            "Graph accuracy runs use the release async side-join path."
        ),
    )
    parser.add_argument(
        "--jetfire-reset-runtime-between-samples",
        dest="jetfire_reset_runtime_between_samples",
        action="store_true",
        help=(
            "Debug-only: destructively reset JetFire worker runtime/shadow "
            "state between independent accuracy samples. The release harness "
            "leaves this off because vLLM block tables already isolate serial "
            "requests, and resetting the JetFire cache store between samples "
            "can perturb subsequent decode outputs."
        ),
    )
    parser.add_argument(
        "--no-jetfire-reset-runtime-between-samples",
        dest="jetfire_reset_runtime_between_samples",
        action="store_false",
        help=(
            "Do not clear JetFire worker runtime/shadow state between "
            "independent accuracy samples."
        ),
    )
    parser.set_defaults(jetfire_reset_runtime_between_samples=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--longbench-v2-max-new-tokens", type=int, default=128)
    parser.add_argument("--niah-contexts", default="")
    parser.add_argument("--niah-depths", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--niah-repeats", type=int, default=10)
    parser.add_argument(
        "--niah-edge-margin-pct",
        type=int,
        default=5,
        help=(
            "Map requested NIAH depths into [margin, 100-margin]. For example, "
            "5 maps requested depth 0/50/100 to actual insertion depth 5/50/95."
        ),
    )
    parser.add_argument(
        "--niah-filler-data-dir",
        default=str(NIAH_DEFAULT_LB_DATA),
        help=(
            "Directory containing LongBench v1 jsonl files used as NIAH filler. "
            "If unavailable, the harness loads THUDM/LongBench filler tasks."
        ),
    )
    parser.add_argument("--niah-sample-seed", type=int, default=42)
    parser.add_argument("--niah-chat-overhead-tokens", type=int, default=120)
    args = parser.parse_args()
    for key in ("max_samples", "sample_start", "sample_end", "max_model_len", "top_k", "generation_token_margin"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    for key in ("max_num_seqs", "cpu_attn_threads", "kv_cache_block_size", "niah_repeats", "longbench_v2_max_new_tokens"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.sample_end and args.sample_end <= args.sample_start:
        parser.error("--sample-end must be larger than --sample-start")
    if args.top_k == 0:
        parser.error("Use --jetfire-approx-only for k=0; --top-k must stay positive for the release graph")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 0 <= args.niah_edge_margin_pct < 50:
        parser.error("--niah-edge-margin-pct must be in [0, 50)")
    try:
        depths = [int(x.strip()) for x in args.niah_depths.split(",") if x.strip()]
        contexts = [int(x.strip()) for x in args.niah_contexts.split(",") if x.strip()]
    except ValueError:
        parser.error("NIAH depths and contexts must be comma-separated integers")
    if not depths or any(x < 0 or x > 100 for x in depths):
        parser.error("--niah-depths requires at least one depth, all in [0, 100]")
    if any(x <= 0 for x in contexts):
        parser.error("--niah-contexts values must be positive")
    unknown_tasks = set(filter(None, (x.strip() for x in args.tasks.split(",")))) - set(LONGBENCH_V1_TASKS)
    if unknown_tasks:
        parser.error(f"unsupported LongBench v1 tasks: {sorted(unknown_tasks)}")
    return args


def _dataset_provenance(args: argparse.Namespace) -> dict[str, Any]:
    paths: list[Path] = []
    if args.suite == "longbench_v1" and args.longbench_v1_data_dir:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()] if args.tasks else LONGBENCH_V1_TASKS
        paths = [Path(args.longbench_v1_data_dir).expanduser() / f"{task}.jsonl" for task in tasks]
    elif args.suite.startswith("longbench_v2") and args.longbench_v2_data_file:
        paths = [Path(args.longbench_v2_data_file).expanduser()]
    elif args.suite.startswith("niah"):
        directory = Path(args.niah_filler_data_dir).expanduser()
        if directory.is_dir():
            paths = sorted(path for path in directory.glob("*.jsonl") if any(task in path.name for task in NIAH_FILLER_TASKS))
    files = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation data file unavailable: {path}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files[str(path.resolve())] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return {"dataset_files": files}


def main() -> None:
    args = parse_args()
    from jetfire_vllm.bench.model_paths import resolve_model_path

    model_path = resolve_model_path(args.model_key, args.model_path, str(MODEL_CONFIGS[args.model_key]["path"]))
    MODEL_CONFIGS[args.model_key]["path"] = model_path
    if model_path.startswith("/path/to/") or (model_path.startswith(("/", "./", "../", "~")) and not Path(model_path).expanduser().is_dir()):
        raise FileNotFoundError(
            f"Model directory is unavailable: {model_path}. Set --model-path, "
            "JETFIRE_MODEL_PATH, or the per-model JETFIRE_LLAMA31_8B / "
            "JETFIRE_LLAMA33_70B / JETFIRE_QWEN3_30B environment variable."
        )
    random.seed(args.seed)
    dataset_provenance = _dataset_provenance(args)
    if args.jetfire_approx_only and not BACKEND_CONFIGS[args.backend]["jetfire"]:
        raise ValueError("--jetfire-approx-only requires a JetFire backend")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend_run_name = (
        f"{args.backend}_k0" if args.jetfire_approx_only else args.backend
    )
    out_file = args.out_dir / f"{args.model_key}_{backend_run_name}_{args.suite}.json"
    meta_file = (
        args.out_dir / f"{args.model_key}_{backend_run_name}_{args.suite}.meta.json"
    )
    max_model_len = int(
        args.max_model_len or _suite_window(args.suite, args.model_key)
    )
    tensor_parallel_size = int(
        args.tensor_parallel_size or MODEL_CONFIGS[args.model_key]["tp"]
    )
    partition_meta = _accuracy_prefill_partition_metadata(
        model_path=str(MODEL_CONFIGS[args.model_key]["path"]),
        backend=args.backend,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        args=args,
    )
    meta = {
        "software_versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "vllm", "transformers")
        },
        "python_version": sys.version.split()[0],
        "scoring_protocol": SCORING_PROTOCOL_VERSION,
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "sample_start": int(args.sample_start),
        "sample_end": int(args.sample_end),
        "tasks": args.tasks,
        "longbench_v1_data_dir": args.longbench_v1_data_dir,
        "longbench_v2_data_file": args.longbench_v2_data_file,
        "generation_token_margin": int(args.generation_token_margin),
        "only_over_model_len": int(args.only_over_model_len),
        "longbench_v2_max_new_tokens": int(args.longbench_v2_max_new_tokens),
        "model_key": args.model_key,
        "model_path": MODEL_CONFIGS[args.model_key]["path"],
        "backend": args.backend,
        "backend_label": BACKEND_CONFIGS[args.backend]["label"],
        "kv_cache_dtype": BACKEND_CONFIGS[args.backend].get("kv_cache_dtype"),
        "jetfire_mode": BACKEND_CONFIGS[args.backend]["mode"],
        "jetfire_approx_only": bool(args.jetfire_approx_only),
        "effective_correction_top_k": (
            0 if args.jetfire_approx_only
            else int(args.top_k) if BACKEND_CONFIGS[args.backend]["jetfire"]
            else None
        ),
        "suite": args.suite,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "moe_backend": args.moe_backend or None,
        "kv_cache_block_size": int(args.kv_cache_block_size),
        "top_k": int(args.top_k),
        "jetfire_group_size": int(args.jetfire_group_size),
        "jetfire_prefill_chunk_tokens": str(args.jetfire_prefill_chunk_tokens),
        "jetfire_graph": bool(args.jetfire_graph),
        "jetfire_async_cpu_correction": bool(args.jetfire_async_cpu_correction),
        "jetfire_kv_budget_safety": float(args.jetfire_kv_budget_safety),
        "jetfire_num_gpu_blocks_override": int(args.jetfire_num_gpu_blocks_override),
        "jetfire_reset_runtime_between_samples": bool(
            args.jetfire_reset_runtime_between_samples
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
        "niah_scoring": "whole_output_exact_match_after_removing_punctuation_and_whitespace",
        "niah_secondary_scoring": "raw_strip_exact_match",
        "niah_contexts": args.niah_contexts,
        "niah_depths": args.niah_depths,
        "niah_repeats": int(args.niah_repeats),
        "niah_edge_margin_pct": int(args.niah_edge_margin_pct),
        "niah_filler_data_dir": args.niah_filler_data_dir,
        "niah_sample_seed": int(args.niah_sample_seed),
        "niah_chat_overhead_tokens": int(args.niah_chat_overhead_tokens),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": _fmt_path(out_file),
    }
    meta.update(partition_meta)
    meta.update(dataset_provenance)
    if args.resume and out_file.exists():
        if not meta_file.exists():
            raise ValueError(f"Cannot safely resume {out_file}: metadata is missing. Use a new output directory or disable --resume.")
        previous = json.loads(meta_file.read_text())
        ignored = {"started_at", "software_versions", "python_version", "cuda_visible_devices", "torch_extensions_dir", "output"}
        changed = [key for key in meta if key not in ignored and previous.get(key) != meta[key]]
        if changed:
            raise ValueError(f"Resume settings changed ({', '.join(changed)}). Use a new output directory or disable --resume.")
    _save_json(meta_file, meta)

    tokenizer = _load_tokenizer(args.model_key, args.trust_remote_code)
    llm = _load_llm(args.model_key, args.backend, args.suite, args)

    if args.suite == "longbench_v1":
        results = run_longbench_v1(llm, tokenizer, out_file, args)
    elif args.suite in ("longbench_v2_128k", "longbench_v2_256k"):
        results = run_longbench_v2(llm, tokenizer, out_file, args)
    elif args.suite in ("niah_32k", "niah_128k", "niah_256k"):
        results = run_niah(llm, tokenizer, out_file, args)
    else:
        raise ValueError(args.suite)

    _maybe_print_jetfire_worker_stats(llm, args)
    if results.get("status") == "incomplete":
        raise SystemExit(f"Evaluation incomplete: empty tasks or over-capacity samples. See {out_file} for per-task counts; increase --max-model-len if needed.")
    print(json.dumps({"output": _fmt_path(out_file), "overall": results.get("overall")}, indent=2))


if __name__ == "__main__":
    main()
