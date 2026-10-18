#!/usr/bin/env python3
"""Check production JetFire kernels against independent PyTorch references.

No model checkpoint or vLLM engine is needed. This is a correctness gate,
not a latency benchmark or a substitute for model-level accuracy evaluation.
The first run JIT-compiles CUDA/C++ extensions. Run on an H100 with CUDA.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def decode_codes(data, bits):
    """Independently decode unsigned wire codes, including INT3 planes."""
    import torch
    d = data.to(torch.int32)
    if bits == 3:
        planes = d.reshape(*d.shape[:-1], 3, -1)
        words = (planes[..., 0, :] << 16) | (planes[..., 1, :] << 8) | planes[..., 2, :]
        return torch.stack([(words >> (21 - 3 * i)) & 7 for i in range(8)], -1).flatten(-2)
    lanes = 8 // bits
    return torch.stack([(d >> (8 - bits * (i + 1))) & ((1 << bits) - 1)
                        for i in range(lanes)], -1).flatten(-2)


def dequant(data, scales, bits, group_size, zp=None, *, bf16=True):
    import torch
    codes = decode_codes(data, bits).float() - (1 << (bits - 1))
    shape = codes.shape
    out = codes.reshape(*shape[:-1], -1, group_size) * scales.float().unsqueeze(-1)
    if zp is not None:
        out = out + zp.float().unsqueeze(-1)
    out = out.reshape(shape)
    return out.to(torch.bfloat16).float() if bf16 else out


def quantize_query_reference(query):
    """Model INT8TC's FP32 division and reciprocal before rounding Q codes.

    PyTorch CUDA division by a Python scalar multiplies its reciprocal.
    A tensor divisor keeps the correctly rounded FP32 division used in CUDA
    C++, including half-code ties where that one-ULP scale difference matters.
    """
    import torch
    groups = query.float().reshape(*query.shape[:-1], -1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    scale = (maximum / torch.full_like(maximum, 127)).clamp_min(1e-12)
    codes = (groups * scale.reciprocal()).round().clamp(-128, 127)
    return (codes * scale).reshape_as(query)


def write_cache(x, bits, group_size, *, asymmetric=False):
    """Exercise fused cache scatter with permuted slots and a masked row."""
    import torch
    from jetfire_vllm.kernels import fused_quant_write as writers
    from jetfire_vllm.kernels.intN_pack import quantize_and_pack, quantize_and_pack_asym
    tokens, heads, dim = x.shape
    slots = torch.randperm(tokens, device=x.device)
    data = torch.full((heads, tokens + 1, dim * bits // 8), 0xA5,
                      device=x.device, dtype=torch.uint8)
    scales = torch.zeros(heads, tokens + 1, dim // group_size, device=x.device,
                         dtype=torch.bfloat16)
    zp = torch.zeros_like(scales) if asymmetric else None
    padded_x = torch.cat((x, x[:1]), 0)
    masked_slots = torch.cat((slots, slots.new_tensor([-1])))
    if asymmetric:
        writers.fused_quant_write_int2_asym(padded_x, masked_slots, data, scales, zp, group_size)
        ref_data, ref_scale, ref_zp = quantize_and_pack_asym(x, group_size, bits)
        torch.testing.assert_close(zp[:, slots].transpose(0, 1), ref_zp, rtol=0, atol=0)
    else:
        getattr(writers, f"fused_quant_write_int{bits}_sym")(
            padded_x, masked_slots, data, scales, group_size)
        ref_data, ref_scale = quantize_and_pack(x, group_size, bits)
    actual = data[:, slots].transpose(0, 1)
    actual_scale = scales[:, slots].transpose(0, 1)
    torch.testing.assert_close(actual_scale, ref_scale, rtol=0, atol=0)
    codes, ref_codes = decode_codes(actual, bits), decode_codes(ref_data, bits)
    delta = (codes - ref_codes).abs()
    # Triton's fast division may choose the adjacent code at a rounding tie.
    # Bound this explicitly per lane; arbitrary byte corruption cannot pass.
    assert int(delta.max()) <= 1, "fused writer produced an invalid code"
    agreement = float((delta == 0).float().mean())
    assert agreement >= 0.995, f"quantization code agreement is only {agreement:.6f}"
    assert bool((data[:, -1] == 0xA5).all()), "negative slot wrote to the cache"
    # A shuffled scatter must preserve every physical slot, not just row 0.
    dense_x = torch.empty_like(x)
    dense_x[slots] = x
    return data[:, :tokens].contiguous(), scales[:, :tokens].contiguous(), (
        zp[:, :tokens].contiguous() if zp is not None else None
    ), dense_x.transpose(0, 1).contiguous(), agreement


def run_case(mode, group_size, batch, context, top_k):
    import torch
    from jetfire_vllm.adapter.config import QUANT_SCHEMES
    from jetfire_vllm.adapter.release_policy import plan_release_graph
    from jetfire_vllm.kernels.cpu_topk_attn import load_cpu_topk_attn_ext
    from jetfire_vllm.kernels.blend import load_blend_ext
    scheme = QUANT_SCHEMES[mode]
    hq, hkv, dim, block = 8, 2, 128, 16
    torch.manual_seed(7000 + group_size + batch)
    tokens = batch * context
    xk = torch.randn(tokens, hkv, dim, device="cuda", dtype=torch.bfloat16)
    xv = torch.randn_like(xk)
    kd, ks, kz, kfull, k_agree = write_cache(xk, scheme.k_nbits, group_size,
                                           asymmetric=scheme.k_asymmetric)
    vd, vs, _, vfull, v_agree = write_cache(xv, scheme.v_nbits, group_size)
    q = torch.randn(batch, hq, dim, device="cuda", dtype=torch.bfloat16)
    bt = torch.randperm(tokens // block, device="cuda").reshape(batch, context // block)
    lens = torch.tensor([context - 13 * b for b in range(batch)], device="cuda", dtype=torch.int32)
    positions = torch.arange(context, device="cuda")
    slots = bt[:, positions // block] * block + positions % block
    plan = plan_release_graph(mode=mode, group_size=group_size, top_k=top_k,
                              max_graph_batch=batch, num_query_heads=hq, num_kv_heads=hkv,
                              require_ready=True)
    scale = dim ** -0.5
    stats = None
    if plan.stage_a == "qk_int8tc_stats":
        from jetfire_vllm.kernels.int8tc_qk_scores import load_int8tc_qk_scores_ext
        scores, selected, selected_slots, m, inv_z = load_int8tc_qk_scores_ext().int8tc_qk_scores_topk_stats(
            q, kd, ks, bt, lens, context, hq, hkv, block, scale, top_k)
        stats = (m, inv_z)
        qref = quantize_query_reference(q)
        k0 = dequant(kd, ks, scheme.k_nbits, group_size, bf16=False)
    else:
        ext = importlib.import_module(f"jetfire_vllm.kernels.int{scheme.k_nbits}_split").load_qk_scores_ext()
        args = [q, kd, ks]
        if scheme.k_nbits == 2:
            args.append(kz if kz is not None else torch.empty(0, device="cuda", dtype=torch.bfloat16))
        scores, selected, selected_slots = getattr(ext, f"int{scheme.k_nbits}_qk_scores_topk")(
            *args, bt, lens, context, hq, hkv, block, scale, top_k)
        qref = q.float()
        k0 = dequant(kd, ks, scheme.k_nbits, group_size, kz)
    head = torch.arange(hq, device="cuda") // (hq // hkv)
    logical_k = k0[head[:, None], slots[:, None, :]]
    score_ref = torch.einsum("bhd,bhsd->bhs", qref, logical_k) * scale
    mask = positions[None, None, :] < lens[:, None, None]
    torch.testing.assert_close(scores[mask.expand_as(scores)], score_ref[mask.expand_as(scores)],
                               atol=2e-5, rtol=2e-5)
    assert bool(torch.isneginf(scores.masked_select(~mask.expand_as(scores))).all())
    torch.testing.assert_close(scores.gather(-1, selected).sort(-1).values,
                               scores.topk(top_k, dim=-1).values.sort(-1).values, atol=0, rtol=0)
    assert bool((selected < lens[:, None, None]).all()), "top-k includes padding"
    assert torch.equal(selected_slots, slots[:, None, :].expand(batch, hq, context).gather(-1, selected))
    sv_ext = importlib.import_module(f"jetfire_vllm.kernels.int{scheme.v_nbits}_split").load_softmax_v_ext()
    if plan.stage_b == "sv_int8tc_reg_stats":
        assert stats is not None, "policy requires Stage-A softmax stats"
        from jetfire_vllm.kernels.int8tc_softmax_v_fused import load_ext
        ohat, old, weight = load_ext().int8tc_softmax_v_with_pre_corr_reg_stats(
            scores, *stats, vd, vs, bt, lens, selected, context, hq, hkv, block, top_k)
    else:
        suffix = "wvs2_scores" if plan.stage_b == "sv_int4_wvs2_scores" else "wvs2"
        ohat, old, weight = getattr(sv_ext, f"int{scheme.v_nbits}_softmax_v_with_pre_corr_{suffix}")(
            scores, vd, vs, bt, lens, selected, context, hq, hkv, block, top_k)
    probs = torch.softmax(scores, -1)
    # Stage B multiplies integer V codes by scales in fp32, without BF16 rounding.
    v0 = dequant(vd, vs, scheme.v_nbits, group_size, bf16=False)
    logical_v = v0[head[:, None], slots[:, None, :]]
    ohat_ref = torch.einsum("bhs,bhsd->bhd", probs, logical_v)
    selected_prob = probs.gather(-1, selected)
    selected_v = logical_v.gather(2, selected[..., None].expand(batch, hq, top_k, dim))
    old_ref = (selected_prob[..., None] * selected_v).sum(2)
    weight_ref = selected_prob.sum(-1)
    # INT8TC Stage B quantizes probabilities; split/wvs2 uses fp32 weights.
    tol = 0.003 if plan.stage_b == "sv_int8tc_reg_stats" else 3e-6
    torch.testing.assert_close(ohat, ohat_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(old, old_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(weight, weight_ref, atol=3e-6, rtol=3e-6)
    kcpu, vcpu, qcpu = kfull.cpu(), vfull.cpu(), q.cpu()
    selected_cpu = selected_slots.cpu()
    head_cpu = head.cpu().view(1, hq, 1).expand(batch, hq, top_k).contiguous()
    out_cpu = torch.empty_like(qcpu)
    cpu_ext = load_cpu_topk_attn_ext()
    cpu_ext.set_cpu_attn_threads(4)
    done = torch.zeros(1, dtype=torch.int64)
    cpu_ext.cpu_topk_attn(qcpu, kcpu, vcpu, head_cpu, selected_cpu, out_cpu, scale, done)
    assert int(done[0]) == 1, "CPU correction completion flag was not set"
    knew = kfull[head[:, None], selected_slots]
    vnew = vfull[head[:, None], selected_slots]
    new_probs = torch.softmax(torch.einsum("bhd,bhkd->bhk", q.float(), knew.float()) * scale, -1)
    new_ref = torch.einsum("bhk,bhkd->bhd", new_probs, vnew.float())
    torch.testing.assert_close(out_cpu.float(), new_ref.cpu(), atol=0.002, rtol=0.006)
    corrected = torch.empty_like(q)
    load_blend_ext().blend_unnorm_into(ohat, old, weight, out_cpu.to("cuda"), corrected)
    expected = ohat_ref + weight_ref[..., None] * new_ref - old_ref
    torch.testing.assert_close(corrected.float(), expected, atol=0.006, rtol=0.015)
    torch.cuda.synchronize()
    return {"mode": mode, "group_size": group_size, "batch": batch,
            "context": context, "top_k": top_k, "stage_a": plan.stage_a,
            "stage_b": plan.stage_b, "pack_code_agreement": min(k_agree, v_agree),
            "stage_a_max_abs": float((scores - score_ref).masked_select(mask.expand_as(scores)).abs().max()),
            "stage_b_max_abs": float((ohat - ohat_ref).abs().max()),
            "cpu_max_abs": float((out_cpu.float() - new_ref.cpu()).abs().max()),
            "corrected_max_abs": float((corrected.float() - expected).abs().max()), "ok": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-modes", action="store_true", help="All nine symmetric K/V bit combinations.")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--groups", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--check-filler", action="store_true", help="Also build and launch optional filler kernel.")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required; this validation never silently skips GPU checks")
    if args.context % 16 or args.context - 13 * (max(args.batches) - 1) < args.top_k:
        parser.error("context must be divisible by 16 and every sequence must be at least top-k long")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    modes = args.modes or ([f"SymInt{k}K_SymInt{v}V" for k in (2, 3, 4) for v in (2, 3, 4)]
                           if args.all_modes else ["SymInt3K_SymInt2V", "SymInt4K_SymInt4V"])
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "extension_dir": os.environ.get("TORCH_EXTENSIONS_DIR"), "cases": [], "ok": False}
    started = time.monotonic()
    try:
        for mode in modes:
            for group in args.groups:
                for batch in args.batches:
                    print(f"[runtime] checking {mode} g={group} batch={batch}", flush=True)
                    row = run_case(mode, group, batch, args.context, args.top_k)
                    result["cases"].append(row)
                    print(json.dumps(row), flush=True)
        if args.check_filler:
            from jetfire_vllm.kernels.filler import load_filler_ext
            measured = load_filler_ext().calibrate_filler(20)
            assert 0 < measured < 100000, f"filler launch did not complete correctly: {measured} us"
            result["filler_measured_us"] = measured
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[runtime] PASS {len(result['cases'])} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
