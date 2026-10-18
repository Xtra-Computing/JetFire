#!/usr/bin/env python3
"""JetFire approximate top-k quality probe.

This is an offline mechanism-level evaluator.  It does not exercise the vLLM
serving path.  Instead, it runs a real HF model on real long-context text,
extracts per-layer Q/K/V for selected layers, applies JetFire's quantization
schemes to K/V, and compares approximate low-bit attention and ATC against
BF16 attention.

Metrics are aggregated over query heads for the last prompt token:
  - top-k index recall vs BF16 top-k
  - BF16 attention mass covered by approximate top-k
  - oracle BF16 top-k mass
  - dense low-precision (k=0) output error
  - ATC output error after correcting the approximate top-k set with BF16 K/V
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from jetfire_vllm.bench.model_paths import resolve_model_path
from jetfire_vllm.adapter.config import QUANT_SCHEMES
from jetfire_vllm.kernels.intN_pack import quantize_and_pack, quantize_and_pack_asym, unpack


DEFAULT_MODEL_PATH = resolve_model_path("llama31_8b", default="/path/to/Llama-3.1-8B-Instruct")

DEFAULT_FILLER_DIR = Path(
    os.environ.get("JETFIRE_LONGBENCH_DATA_DIR", "/path/to/longbench_v1/data")
)
DEFAULT_FILLER_TASKS = ("gov_report", "multi_news", "narrativeqa", "qasper", "multifieldqa_en")


@dataclass(frozen=True)
class ProbeConfig:
    label: str
    mode: str
    group_size: int


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _load_filler_text(
    data_dir: Path,
    tasks: tuple[str, ...],
    *,
    allow_synthetic: bool = False,
) -> str:
    parts: list[str] = []
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.jsonl")):
            if not any(task in path.name for task in tasks):
                continue
            with path.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    context = obj.get("context") or obj.get("input") or obj.get("prompt") or ""
                    if context:
                        parts.append(context)
                    if len(parts) >= 256:
                        break
            if len(parts) >= 256:
                break
    if not parts and allow_synthetic:
        fallback = (
            "This is a long technical report about airport infrastructure, "
            "legal hearings, financial disclosures, multi-hop question answering, "
            "and public meeting transcripts. The document contains many sections, "
            "tables, named entities, dates, budgets, procedural details, and "
            "cross-references. A reader must connect evidence from distant parts "
            "of the document to answer retrieval questions accurately. "
        )
        parts = [fallback] * 4096
    if not parts:
        raise FileNotFoundError(
            f"no LongBench JSONL contexts found under {data_dir}; pass a valid "
            "--filler-data-dir or explicitly use --allow-synthetic-filler"
        )
    return "\n\n".join(parts)


def _make_input_ids(tokenizer, ctx_len: int, filler_text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(filler_text, add_special_tokens=False, truncation=True, max_length=max(ctx_len + 16, 1024)).input_ids
    if not ids:
        raise RuntimeError("empty tokenizer output")
    while len(ids) < ctx_len + 16:
        ids.extend(ids)
    ids = ids[:ctx_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _dequantize_like_jetfire(x: torch.Tensor, *, nbits: int, group_size: int, asymmetric: bool) -> torch.Tensor:
    if asymmetric:
        data, scales, zps = quantize_and_pack_asym(x, group_size=group_size, nbits=nbits)
        return unpack(data, scales, group_size=group_size, nbits=nbits, zps=zps)
    data, scales = quantize_and_pack(x, group_size=group_size, nbits=nbits)
    return unpack(data, scales, group_size=group_size, nbits=nbits)


def _summ(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p05": float("nan"), "min": float("nan")}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "p05": float(torch.quantile(t, 0.05).item()),
        "min": float(t.min().item()),
    }


def _record_layer_metrics(
    *,
    rows: list[dict[str, Any]],
    model_name: str,
    ctx_len: int,
    layer_idx: int,
    q: torch.Tensor,  # [Hq, D]
    k: torch.Tensor,  # [Hkv, S, D]
    v: torch.Tensor,  # [Hkv, S, D]
    configs: list[ProbeConfig],
    topks: list[int],
) -> None:
    device = q.device
    hq, head_dim = q.shape
    hkv = k.shape[0]
    seq_len = k.shape[1]
    max_topk = min(max(topks), seq_len)
    kv_group = hq // hkv
    kv_for_q = torch.arange(hq, device=device) // kv_group

    k_for_q = k[kv_for_q].float()
    v_for_q = v[kv_for_q].float()
    qf = q.float()
    true_scores = torch.einsum("hd,hsd->hs", qf, k_for_q) / math.sqrt(head_dim)
    true_prob = torch.softmax(true_scores, dim=-1)
    true_top_max = torch.topk(true_scores, k=max_topk, dim=-1).indices
    true_out = torch.einsum("hs,hsd->hd", true_prob, v_for_q)

    for cfg in configs:
        scheme = QUANT_SCHEMES[cfg.mode]
        k_deq = _dequantize_like_jetfire(
            k, nbits=scheme.k_nbits, group_size=cfg.group_size, asymmetric=scheme.k_asymmetric
        )
        v_deq = _dequantize_like_jetfire(
            v, nbits=scheme.v_nbits, group_size=cfg.group_size, asymmetric=False
        )
        v_deq_for_q = v_deq[kv_for_q].float()
        approx_scores = torch.einsum("hd,hsd->hs", qf, k_deq[kv_for_q].float()) / math.sqrt(head_dim)
        approx_prob = torch.softmax(approx_scores, dim=-1)
        approx_out = torch.einsum("hs,hsd->hd", approx_prob, v_deq_for_q)
        approx_top_max = torch.topk(approx_scores, k=max_topk, dim=-1).indices

        for topk in topks:
            kk = min(topk, seq_len)
            approx_idx = approx_top_max[:, :kk]
            true_idx = true_top_max[:, :kk]

            recalls: list[float] = []
            masses: list[float] = []
            oracle_masses: list[float] = []
            corrected_errs: list[float] = []
            oracle_errs: list[float] = []
            approx_output_errs: list[float] = []
            atc_output_errs: list[float] = []
            atc_error_reductions: list[float] = []
            for h in range(hq):
                a = approx_idx[h]
                t = true_idx[h]
                # torch.isin is fine here: Hq is small and this is offline.
                recalls.append(float(torch.isin(a, t).float().mean().item()))
                masses.append(float(true_prob[h, a].sum().item()))
                oracle_masses.append(float(true_prob[h, t].sum().item()))

                sel_scores = true_scores[h, a]
                sel_prob = torch.softmax(sel_scores, dim=-1)
                sel_out = torch.einsum("s,sd->d", sel_prob, v_for_q[h, a])
                denom = true_out[h].norm().clamp(min=1e-6)
                corrected_errs.append(float(((sel_out - true_out[h]).norm() / denom).item()))

                oracle_scores = true_scores[h, t]
                oracle_prob = torch.softmax(oracle_scores, dim=-1)
                oracle_out = torch.einsum("s,sd->d", oracle_prob, v_for_q[h, t])
                oracle_errs.append(float(((oracle_out - true_out[h]).norm() / denom).item()))

                approx_err = (approx_out[h] - true_out[h]).norm() / denom
                selected_mass = approx_prob[h, a].sum()
                old_selected = torch.einsum(
                    "s,sd->d", approx_prob[h, a], v_deq_for_q[h, a]
                )
                atc_out = approx_out[h] + selected_mass * sel_out - old_selected
                atc_err = (atc_out - true_out[h]).norm() / denom
                approx_err_value = float(approx_err.item())
                atc_err_value = float(atc_err.item())
                approx_output_errs.append(approx_err_value)
                atc_output_errs.append(atc_err_value)
                atc_error_reductions.append(
                    1.0 - atc_err_value / max(approx_err_value, 1e-12)
                )

            recall_s = _summ(recalls)
            mass_s = _summ(masses)
            oracle_s = _summ(oracle_masses)
            corr_s = _summ(corrected_errs)
            oracle_err_s = _summ(oracle_errs)
            approx_output_err_s = _summ(approx_output_errs)
            atc_output_err_s = _summ(atc_output_errs)
            atc_error_reduction_s = _summ(atc_error_reductions)
            rows.append({
                "model": model_name,
                "ctx_len": ctx_len,
                "layer": layer_idx,
                "config": cfg.label,
                "mode": cfg.mode,
                "group_size": cfg.group_size,
                "top_k": kk,
                "index_recall_mean": recall_s["mean"],
                "index_recall_p05": recall_s["p05"],
                "index_recall_min": recall_s["min"],
                "captured_mass_mean": mass_s["mean"],
                "captured_mass_p05": mass_s["p05"],
                "captured_mass_min": mass_s["min"],
                "ideal_topk_mass_mean": oracle_s["mean"],
                "mass_recall_ratio_mean": mass_s["mean"] / max(oracle_s["mean"], 1e-12),
                # Backward-compatible field names used by the 2026-06-10 Llama report.
                "topk_recall_mean": recall_s["mean"],
                "topk_recall_p05": recall_s["p05"],
                "topk_recall_min": recall_s["min"],
                "mass_recall_mean": mass_s["mean"],
                "mass_recall_p05": mass_s["p05"],
                "mass_recall_min": mass_s["min"],
                "oracle_mass_mean": oracle_s["mean"],
                "mass_gap_mean": oracle_s["mean"] - mass_s["mean"],
                "selected_bf16_output_relerr_mean": corr_s["mean"],
                "oracle_topk_output_relerr_mean": oracle_err_s["mean"],
                "approx_output_relerr_mean": approx_output_err_s["mean"],
                "atc_output_relerr_mean": atc_output_err_s["mean"],
                "atc_error_reduction_fraction_mean": atc_error_reduction_s["mean"],
                "num_query_heads": hq,
                "seq_len": seq_len,
            })


def _config_list(raw: str) -> list[ProbeConfig]:
    out: list[ProbeConfig] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        # label:mode:group_size
        parts = item.split(":")
        if len(parts) == 1:
            label = parts[0]
            mode = parts[0]
            group_size = 128
        elif len(parts) == 2:
            label = parts[0]
            mode = parts[0]
            group_size = int(parts[1])
        elif len(parts) == 3:
            label, mode, gs = parts
            group_size = int(gs)
        else:
            raise ValueError(f"bad config spec: {item}")
        if mode not in QUANT_SCHEMES:
            raise ValueError(f"unknown JetFire mode {mode}")
        out.append(ProbeConfig(label=label, mode=mode, group_size=group_size))
    return out


def run_probe(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model-level top-k probe")
    layer_ids = _parse_csv_ints(args.layers)
    topks = _parse_csv_ints(args.topks)
    ctx_lens = _parse_csv_ints(args.contexts)
    configs = _config_list(args.configs)
    filler_text = _load_filler_text(
        Path(args.filler_data_dir),
        DEFAULT_FILLER_TASKS,
        allow_synthetic=bool(args.allow_synthetic_filler),
    )


    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()


    rows: list[dict[str, Any]] = []
    handles = []
    active_ctx = {"ctx_len": 0}

    def make_hook(layer_idx: int):
        def _hook(module, hook_args, hook_kwargs=None):
            with torch.no_grad():
                hook_kwargs = hook_kwargs or {}
                hidden_states = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
                if len(hook_args) > 1:
                    position_embeddings = hook_args[1]
                else:
                    position_embeddings = hook_kwargs["position_embeddings"]
                cos, sin = position_embeddings
                bsz, seq_len, _ = hidden_states.shape
                if bsz != 1:
                    raise RuntimeError("probe expects batch size 1")
                num_heads = module.config.num_attention_heads
                num_kv_heads = module.config.num_key_value_heads
                head_dim = module.head_dim
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                v = module.v_proj(hidden_states)
                q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                # Qwen3-MoE applies per-head RMSNorm to Q/K before RoPE. Llama
                # modules do not have these attributes, so this remains a no-op
                # for the original Llama probe.
                if hasattr(module, "q_norm"):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm"):
                    k = module.k_norm(k)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                _record_layer_metrics(
                    rows=rows,
                    model_name=Path(args.model).name,
                    ctx_len=int(active_ctx["ctx_len"]),
                    layer_idx=layer_idx,
                    q=q[0, :, -1, :].detach(),
                    k=k[0].detach(),
                    v=v[0].detach(),
                    configs=configs,
                    topks=topks,
                )
        return _hook

    for layer_idx in layer_ids:
        handles.append(model.model.layers[layer_idx].self_attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    try:
        for ctx_len in ctx_lens:
            active_ctx["ctx_len"] = int(ctx_len)
            input_ids = _make_input_ids(tokenizer, int(ctx_len), filler_text, device)
            with torch.inference_mode():
                _ = model(input_ids=input_ids, use_cache=False)
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    csv_path = out_dir / "topk_quality_rows.csv"
    json_path = out_dir / "topk_quality_rows.json"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    # Compact aggregate useful for reports.
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["ctx_len"], row["config"], row["group_size"], row["top_k"])].append(row)
    summary = []
    for (ctx_len, config, group_size, top_k), group_rows in sorted(grouped.items()):
        def avg(key: str) -> float:
            return sum(float(r[key]) for r in group_rows) / len(group_rows)
        summary.append({
            "ctx_len": ctx_len,
            "config": config,
            "group_size": group_size,
            "top_k": top_k,
            "layers": len(group_rows),
            "index_recall_mean": avg("index_recall_mean"),
            "captured_mass_mean": avg("captured_mass_mean"),
            "ideal_topk_mass_mean": avg("ideal_topk_mass_mean"),
            "mass_recall_ratio_mean": avg("mass_recall_ratio_mean"),
            "topk_recall_mean": avg("topk_recall_mean"),
            "mass_recall_mean": avg("mass_recall_mean"),
            "mass_recall_p05_mean": avg("mass_recall_p05"),
            "oracle_mass_mean": avg("oracle_mass_mean"),
            "mass_gap_mean": avg("mass_gap_mean"),
            "selected_bf16_output_relerr_mean": avg("selected_bf16_output_relerr_mean"),
            "oracle_topk_output_relerr_mean": avg("oracle_topk_output_relerr_mean"),
            "approx_output_relerr_mean": avg("approx_output_relerr_mean"),
            "atc_output_relerr_mean": avg("atc_output_relerr_mean"),
            "atc_error_reduction_fraction_mean": avg("atc_error_reduction_fraction_mean"),
        })
    with (out_dir / "topk_quality_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[topk-quality] wrote {csv_path}")
    print(f"[topk-quality] wrote {out_dir / 'topk_quality_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--contexts", default="4096,8192,16384")
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--topks", default="32,64,128,256,512,1024")
    parser.add_argument(
        "--configs",
        default=(
            "K4V4-g128:SymInt4K_SymInt4V:128,"
            "K4V4-g64:SymInt4K_SymInt4V:64,"
            "K3V3-g64:SymInt3K_SymInt3V:64,"
            "K3V2-g64:SymInt3K_SymInt2V:64,"
            "K2V2-g64:SymInt2K_SymInt2V:64"
        ),
    )
    parser.add_argument("--filler-data-dir", default=str(DEFAULT_FILLER_DIR))
    parser.add_argument(
        "--allow-synthetic-filler",
        action="store_true",
        help="Use repeated synthetic text when LongBench data is unavailable.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
