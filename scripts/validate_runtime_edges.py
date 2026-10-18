#!/usr/bin/env python3
"""Validate short/mixed decode lengths and graph CPU correction against Torch.

Unlike validate_runtime.py, this gate deliberately uses sequences shorter than
the fixed kernel top-k, including one-token contexts and TP2 Llama-70B local
head geometry. No checkpoints or vLLM server are required. CUDA is mandatory.
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
sys.path.insert(0, str(ROOT / "scripts"))
from validate_runtime import dequant, quantize_query_reference, write_cache


def run_case(mode, group_size, context, lengths, top_k, hq, hkv, replays):
    import torch
    from jetfire_vllm.adapter.config import QUANT_SCHEMES
    from jetfire_vllm.adapter.release_policy import plan_release_graph
    from jetfire_vllm.kernels.blend import load_blend_ext
    from jetfire_vllm.kernels.cpu_topk_attn import load_cpu_topk_attn_ext

    batch, dim, block = len(lengths), 128, 16
    torch.manual_seed(9341 + context + group_size)
    scheme = QUANT_SCHEMES[mode]
    physical = batch * ((context + block - 1) // block) * block
    xk = torch.randn(physical, hkv, dim, device="cuda", dtype=torch.bfloat16)
    xv = torch.randn_like(xk)
    kd, ks, kz, kfull, _ = write_cache(xk, scheme.k_nbits, group_size,
                                      asymmetric=scheme.k_asymmetric)
    vd, vs, _, vfull, _ = write_cache(xv, scheme.v_nbits, group_size)
    q = torch.randn(batch, hq, dim, device="cuda", dtype=torch.bfloat16)
    bt = torch.randperm(physical // block, device="cuda").reshape(batch, -1)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    pos = torch.arange(context, device="cuda")
    slots = bt[:, pos // block] * block + pos % block
    head = torch.arange(hq, device="cuda") // (hq // hkv)
    plan = plan_release_graph(mode=mode, group_size=group_size, top_k=top_k,
                              max_graph_batch=batch, num_query_heads=hq,
                              num_kv_heads=hkv, require_ready=True)
    scale = dim ** -0.5
    if plan.stage_a == "qk_int8tc_stats":
        from jetfire_vllm.kernels.int8tc_qk_scores import load_int8tc_qk_scores_ext
        qk_ext = load_int8tc_qk_scores_ext()
    else:
        qk_ext = importlib.import_module(
            f"jetfire_vllm.kernels.int{scheme.k_nbits}_split").load_qk_scores_ext()
    sv_ext = importlib.import_module(
        f"jetfire_vllm.kernels.int{scheme.v_nbits}_split").load_softmax_v_ext()
    stats_ext = None
    if plan.stage_b == "sv_int8tc_reg_stats":
        from jetfire_vllm.kernels.int8tc_softmax_v_fused import load_ext
        stats_ext = load_ext()
    cpu_ext, blend = load_cpu_topk_attn_ext(), load_blend_ext()
    cpu_ext.set_cpu_attn_threads(4)
    cpu_ext.graph_pool_warmup_threads(4)
    kcpu, vcpu = kfull.cpu(), vfull.cpu()
    hm = head.cpu().view(1, hq, 1).expand(batch, hq, top_k).contiguous()
    qp = torch.empty(batch, hq, dim, dtype=torch.bfloat16, pin_memory=True)
    sp = torch.empty(batch, hq, top_k, dtype=torch.int64, pin_memory=True)
    op, od = cpu_ext.alloc_mapped_pinned_pair([batch, hq, dim], torch.bfloat16)
    flag, _ = cpu_ext.alloc_mapped_pinned_pair([1], torch.int64)
    flag.zero_()
    ctx = cpu_ext.create_cpu_topk_submit_ctx(qp, kcpu, vcpu, hm, sp, op, scale, flag)

    def pipeline():
        stats = None
        if plan.stage_a == "qk_int8tc_stats":
            scores, selected, selected_slots, m, iz = qk_ext.int8tc_qk_scores_topk_stats(
                q, kd, ks, bt, lens, context, hq, hkv, block, scale, top_k)
            stats = (m, iz)
        else:
            args = [q, kd, ks]
            if scheme.k_nbits == 2:
                args.append(kz if kz is not None else torch.empty(
                    0, device="cuda", dtype=torch.bfloat16))
            scores, selected, selected_slots = getattr(
                qk_ext, f"int{scheme.k_nbits}_qk_scores_topk")(
                    *args, bt, lens, context, hq, hkv, block, scale, top_k)
        qp.copy_(q, non_blocking=True)
        sp.copy_(selected_slots, non_blocking=True)
        stream_id = torch.cuda.current_stream().cuda_stream
        cpu_ext.stream_submit_cpu_topk_attn(stream_id, ctx)
        if stats_ext is not None:
            ohat, old, weight = stats_ext.int8tc_softmax_v_with_pre_corr_reg_stats(
                scores, *stats, vd, vs, bt, lens, selected,
                context, hq, hkv, block, top_k)
        else:
            suffix = "wvs2_scores" if plan.stage_b == "sv_int4_wvs2_scores" else "wvs2"
            ohat, old, weight = getattr(
                sv_ext, f"int{scheme.v_nbits}_softmax_v_with_pre_corr_{suffix}")(
                    scores, vd, vs, bt, lens, selected,
                    context, hq, hkv, block, top_k)
        cpu_ext.stream_wait_cpu_topk_attn(stream_id, ctx)
        output = torch.empty_like(q)
        blend.blend_unnorm_into(ohat, old, weight, od, output)
        return scores, selected, selected_slots, ohat, old, weight, output

    k0 = dequant(kd, ks, scheme.k_nbits, group_size, kz,
                 bf16=plan.stage_a != "qk_int8tc_stats")
    v0 = dequant(vd, vs, scheme.v_nbits, group_size, bf16=False)
    logical_k = k0[head[:, None], slots[:, None, :]]
    logical_v = v0[head[:, None], slots[:, None, :]]
    errors, dense_errors = [], []

    def check(values):
        scores, selected, selected_slots, ohat, old, weight, output = values
        torch.cuda.synchronize()
        qref = q.float()
        if plan.stage_a == "qk_int8tc_stats":
            qref = quantize_query_reference(q)
        reference_scores = torch.einsum("bhd,bhsd->bhs", qref, logical_k) * scale
        valid = pos[None, None, :] < lens[:, None, None]
        reference_scores = reference_scores.masked_fill(~valid, float("-inf"))
        torch.testing.assert_close(scores, reference_scores, atol=2e-5, rtol=2e-5)
        # Every real position appears exactly once when S<K; only -1 may pad.
        for b, length in enumerate(lens.tolist()):
            for h in range(hq):
                actual = selected[b, h]
                real = actual[actual >= 0]
                expected_count = min(top_k, length)
                assert real.numel() == expected_count, ("incorrect top-k valid count", b, h,
                                                        length, real.numel(), expected_count)
                assert real.unique().numel() == expected_count, "top-k contains duplicate positions"
                assert bool((real < length).all()), "top-k contains sequence padding"
                assert bool((actual[actual < 0] == -1).all()), "invalid index must be -1"
                torch.testing.assert_close(scores[b, h, real].sort().values,
                                           scores[b, h, :length].topk(expected_count).values.sort().values,
                                           atol=0, rtol=0)
        selected_valid = selected >= 0
        safe = selected.clamp_min(0)
        expected_slots = slots[:, None, :].expand(batch, hq, context).gather(-1, safe)
        expected_slots = expected_slots.masked_fill(~selected_valid, -1)
        assert torch.equal(selected_slots, expected_slots), "slot translation/padding mismatch"
        # vLLM pads captured decode batches with zero-length inactive rows.
        probs = reference_scores.softmax(-1).nan_to_num(0)
        hat_ref = torch.einsum("bhs,bhsd->bhd", probs, logical_v)
        if stats_ext is not None:
            # INT8TC multiplies signed INT4 V codes by INT8-quantized P*Vscale,
            # using one FP32 scale per 32-token MMA tile. Short peaked rows
            # can differ visibly from dense FP32 PV, so model this arithmetic
            # explicitly instead of weakening the numerical tolerance.
            # These fixed edge cases have eight 64-token splits, each aligned
            # to the 32-token tiles used by the production kernel.
            assert context % 256 == 0
            logical_scale = vs[head[:, None], slots[:, None, :], 0]
            weighted = (probs * logical_scale).reshape(batch, hq, -1, 32)
            maximum = weighted.abs().amax(-1, keepdim=True)
            tile_scale = maximum / torch.full_like(maximum, 127)
            inverse = torch.where(tile_scale > 0, tile_scale.reciprocal(), 0)
            quantized_weight = ((weighted * inverse).round().clamp(-127, 127)
                                * tile_scale).reshape(batch, hq, context)
            value_codes = logical_v / logical_scale[..., None]
            quantized_hat = torch.einsum("bhs,bhsd->bhd", quantized_weight, value_codes)
            dense_errors.append(float((quantized_hat - hat_ref).abs().max()))
            hat_ref = quantized_hat
        psel = probs.gather(-1, safe).masked_fill(~selected_valid, 0)
        vsel = logical_v.gather(2, safe[..., None].expand(batch, hq, top_k, dim))
        old_ref = (psel[..., None] * vsel).sum(2)
        w_ref = psel.sum(-1)
        torch.testing.assert_close(ohat, hat_ref, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(old, old_ref, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(weight, w_ref, atol=3e-6, rtol=3e-6)
        safe_slots = selected_slots.clamp_min(0)
        full_k, full_v = kfull[head[:, None], safe_slots], vfull[head[:, None], safe_slots]
        new_scores = torch.einsum("bhd,bhkd->bhk", q.float(), full_k.float()) * scale
        new_probs = new_scores.masked_fill(~selected_valid, float("-inf")).softmax(-1).nan_to_num(0)
        new_ref = torch.einsum("bhk,bhkd->bhd", new_probs, full_v.float())
        torch.testing.assert_close(op.float(), new_ref.cpu(), atol=.002, rtol=.006)
        expected = hat_ref + w_ref[..., None] * new_ref - old_ref
        torch.testing.assert_close(output.float(), expected, atol=.006, rtol=.015)
        errors.append(float((output.float() - expected).abs().max()))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        values = pipeline()
    torch.cuda.current_stream().wait_stream(stream)
    check(values)
    if replays:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            values = pipeline()
        # CUDA capture does not execute host callbacks. Each replay must
        # submit CPU work and consume newly copied Q/sequence lengths.
        count_before = int(cpu_ext.get_graph_submit_count())
        for replay in range(replays):
            q.copy_(torch.randn_like(q))
            lens.copy_(torch.tensor(lengths if replay % 2 == 0 else lengths[::-1],
                                    dtype=torch.int32, device="cuda"))
            graph.replay()
            check(values)
        assert int(cpu_ext.get_graph_submit_count()) - count_before == replays
        del graph
    torch.cuda.synchronize()
    return {"mode": mode, "group_size": group_size, "context": context,
            "lengths": lengths, "top_k": top_k, "query_heads": hq,
            "kv_heads": hkv, "stage_a": plan.stage_a, "stage_b": plan.stage_b,
            "graph_replays": replays, "corrected_max_abs": max(errors),
            "stage_b_dense_approx_max_abs": max(dense_errors, default=0), "ok": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=[
        f"SymInt{k}K_SymInt{v}V" for k in (2, 3, 4) for v in (2, 3, 4)])
    parser.add_argument("--groups", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required; GPU checks are never silently skipped")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    result = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "extension_dir": os.environ.get("TORCH_EXTENSIONS_DIR"), "cases": [], "ok": False}
    # Actual production short contexts; arbitrary prompt lengths; TP2 70B
    # local shape. Both supported group sizes and the full K/V dtype matrix.
    cases = [(1, [1], 32, 8, 2),
             (17, [0, 1, 9, 17], 32, 8, 2),
             (512, [17, 257, 512], 256, 32, 4),
             (512, [17, 257, 512], 256, 32, 2)]
    try:
        for mode in args.modes:
            for group in args.groups:
                for context, lengths, top_k, hq, hkv in cases:
                    print(f"[runtime-edge] {mode} g={group} S={lengths} K={top_k}", flush=True)
                    row = run_case(mode, group, context, lengths, top_k, hq, hkv, args.replays)
                    result["cases"].append(row)
                    print(json.dumps(row), flush=True)
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[runtime-edge] PASS {len(result['cases'])} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
