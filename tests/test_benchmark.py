"""CPU checks for variable-length workloads, admission and safe resume."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jetfire_vllm.bench import benchmark as matrix


def _args(argv=()):
    return matrix.parse_args(["--prefill-len", "16384", "--decode-len", "128", *argv])


def _case(mode="k3v2", ctx=255856, batch=16, output_len=128):
    return matrix.campaign.Case("qwen3_30b_a3b", "/models/qwen", 1, "0", mode, ctx, batch, output_len, 3)


def test_plan_uses_requested_lengths_without_models_gpus_or_files(tmp_path, monkeypatch, capsys):
    for name in matrix.campaign.MODEL_ENV.values():
        # Even an inherited unavailable checkpoint does not make planning read files.
        monkeypatch.setenv(name, str(tmp_path / "unavailable"))
    monkeypatch.setattr(matrix.campaign, "_model_input", lambda _: pytest.fail("planning read a checkpoint"))
    monkeypatch.setattr(matrix, "_query_gpu_total_gib", lambda _: pytest.fail("planning inspected a GPU"))
    out = tmp_path / "results"
    assert matrix.main(["--prefill-len", "1009", "--decode-len", "257", "--out-dir", str(out)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["cells"], plan["measured_runs"], plan["measured_requests"], plan["generated_output_tokens"]) == (2, 6, 6, 1542)
    assert {case["model"] for case in plan["cases"]} == {"llama31_8b"}
    assert all(case["tp"] == 1 and case["gpus"] == "0" for case in plan["cases"])
    assert {case["ctx"] for case in plan["cases"]} == {1009}
    assert all(case["output_len"] == 257 and case["runs"] == 3 for case in plan["cases"])
    assert not out.exists()


def test_alias_plans_only_requested_workload(capsys):
    assert matrix.main(["--models", "qwen3_30b", "--prefill-len", "999", "--decode-len", "64",
                        "--modes", "k4v4", "--batches", "7", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cells"] == 1
    assert plan["cases"][0]["model"] == "qwen3_30b_a3b"
    assert plan["cases"][0]["ctx"] == 999
    assert plan["cases"][0]["output_len"] == 64
    assert plan["cases"][0]["batch"] == 7


@pytest.mark.parametrize("decode_len", [1, 64, 257])
def test_lengths_and_arbitrary_batches_apply_to_every_selected_model(decode_len, capsys):
    assert matrix.main(["--models", "llama31_8b,qwen3_30b", "--prefill-len", "4097",
                        "--decode-len", str(decode_len), "--batches", "2,7", "--runs", "2",
                        "--warmups", "0"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["cells"], plan["measured_runs"], plan["measured_requests"]) == (8, 16, 72)
    assert plan["generated_output_tokens"] == 72 * decode_len
    assert {case["model"] for case in plan["cases"]} == set(matrix.MODELS)
    assert all(case["ctx"] == 4097 and case["output_len"] == decode_len and case["runs"] == 2
               for case in plan["cases"])


@pytest.mark.parametrize("argv", [
    ["--models", "llama33_70b"], ["--models", "qwen3_30b,qwen3_30b_a3b"],
    ["--modes", "k2v2"], ["--contexts", "999"], ["--batches", "0"],
    ["--prefill-len", "0"], ["--prefill-len", "-1"], ["--prefill-len", "1.5"],
    ["--decode-len", "0"], ["--decode-len", "-1"], ["--decode-len", "64,257"],
    ["--batches", ""], ["--batches", "1,1"], ["--batches", "1,01"], ["--batches", "-1"],
    ["--runs", "0"], ["--warmups", "-1"],
    ["--single-gpus", "0,1"], ["--bench-timeout-s", "nan"], ["--bench-timeout-s", "inf"], ["--cpu-threads", "0"],
    ["--execute", "--dry-run"], ["--model-path", "llama31_8b=/one", "--model-path", "llama31_8b=/two"],
])
def test_invalid_benchmark_options_are_rejected(argv):
    with pytest.raises(SystemExit, match="2"):
        _args(argv)


@pytest.mark.parametrize("argv", [[], ["--prefill-len", "1009"], ["--decode-len", "64"]])
def test_both_lengths_are_required(argv):
    with pytest.raises(SystemExit, match="2"):
        matrix.parse_args(argv)


def test_semantic_probe_is_optional_and_disabled_by_default():
    assert _args().semantic_probe is False
    assert _args(["--semantic-probe"]).semantic_probe is True
    assert _args(["--no-semantic-probe"]).semantic_probe is False


def test_largest_resident_count_obeys_cap_and_rounding(monkeypatch):
    mib = 1024**2
    monkeypatch.setattr(matrix.harness, "_kv_budget", lambda ns, backend, count: SimpleNamespace(
        bytes=(count * 100 + 7) * mib, safety=1.02, block_size=16, base_blocks=count * 3, inflate_ratio=5.8))
    result = matrix.bounded_allocation(SimpleNamespace(), 16, 10, 8, 1)
    assert result["resident_requests_target"] == 5
    assert result["manual_kv_budget_bytes"] == 507 * mib
    assert result["manual_kv_budget_bytes"] <= result["bounded_kv_limit_bytes"] < 607 * mib
    assert result["kv_budget_safety"] == 1.02 and result["kv_budget_block_size"] == 16


@pytest.mark.parametrize("gpu,weights,scratch", [(80, 79, 5), (80, 76, 5), (float("nan"), 10, 5)])
def test_no_complete_request_fits_fails_preflight(monkeypatch, gpu, weights, scratch):
    monkeypatch.setattr(matrix.harness, "_kv_budget", lambda *args: SimpleNamespace(bytes=1024**3))
    with pytest.raises(matrix.CapacityPreflightError):
        matrix.bounded_allocation(SimpleNamespace(), 16, gpu, weights, scratch)


def _hardware_and_config(monkeypatch):
    # Model metadata only: tests never need real checkpoint weights or a GPU.
    config = {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
              "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 768,
              "max_position_embeddings": 262144}
    monkeypatch.setattr(matrix.harness, "_load_model_config", lambda _: config)
    monkeypatch.setattr(matrix.harness, "_model_weights_gib", lambda _: 56.928)
    monkeypatch.setattr(matrix.harness, "_gpu_total_gib", lambda _: 81559 / 1024)


@pytest.mark.parametrize("mode,resident", [("k3v2", 3), ("k4v4", 2)])
def test_real_kv_rounding_at_256k_selects_expected_residency(monkeypatch, mode, resident):
    _hardware_and_config(monkeypatch)
    result = matrix.resolved_budget(_case(mode=mode))
    assert result["resident_requests_target"] == resident
    assert result["manual_kv_budget_bytes"] == 13237223424
    assert result["max_num_batched_tokens_explicit"] == 260096
    assert result["scheduler_prefill_sub_batch"] <= resident
    assert result["manual_kv_budget_bytes"] < result["all_submitted_workload_budget_bytes"]


def test_prefill_resolution_sees_manual_budget_and_scheduler_limits_residency(monkeypatch):
    _hardware_and_config(monkeypatch)
    observed = []

    def prefill(ns, backend, submitted):
        observed.append(ns.kv_cache_memory_bytes)
        return submitted

    monkeypatch.setattr(matrix.harness, "_resolved_prefill_sub_batch_size", prefill)
    result = matrix.resolved_budget(_case())
    assert observed == [13237223424]
    assert result["scheduler_prefill_sub_batch"] == 3
    assert result["max_num_batched_tokens_explicit"] == 3 * 256000 + 4096


def test_full_fit_uses_whole_batch_without_extra_scheduler_headroom(monkeypatch):
    _hardware_and_config(monkeypatch)
    monkeypatch.setattr(matrix.harness, "_resolved_prefill_sub_batch_size", lambda ns, backend, batch: batch)
    result = matrix.resolved_budget(_case(ctx=16384))
    assert result["resident_requests_target"] == 16
    assert result["max_num_batched_tokens_explicit"] == 16 * (16384 + 128 + 16)


@pytest.mark.parametrize("mode", ["k3v2", "k4v4"])
@pytest.mark.parametrize("decode_len", [64, 257])
def test_budget_and_scheduler_use_actual_lengths(monkeypatch, mode, decode_len):
    _hardware_and_config(monkeypatch)
    monkeypatch.setattr(matrix.harness, "_resolved_prefill_sub_batch_size", lambda ns, backend, batch: batch)
    case = _case(mode=mode, ctx=1009, batch=7, output_len=decode_len)
    namespace = matrix.budget_namespace(case)
    expected_model_len = 1009 + decode_len + 16
    assert matrix.harness._max_model_len(namespace) == expected_model_len
    result = matrix.resolved_budget(case)
    expected_budget = matrix.harness._kv_budget(namespace, "jetfire", 7)
    assert result["max_model_len"] == expected_model_len
    assert result["manual_kv_budget_bytes"] == expected_budget.bytes
    assert result["max_num_batched_tokens_explicit"] == 7 * expected_model_len


@pytest.mark.parametrize("decode_len", [64, 257])
def test_queued_scheduler_uses_actual_lengths_and_full_request_headroom(monkeypatch, decode_len):
    _hardware_and_config(monkeypatch)
    monkeypatch.setattr(matrix.harness, "_resolved_prefill_sub_batch_size", lambda ns, backend, batch: 2)
    result = matrix.resolved_budget(_case(ctx=1009, batch=7, output_len=decode_len))
    assert result["scheduler_prefill_sub_batch"] == 2
    assert result["max_num_batched_tokens_explicit"] == 2 * (1009 + decode_len + 16) + 4096


def test_model_limit_allows_full_request_and_caps_only_padding(monkeypatch):
    config = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
              "num_attention_heads": 32, "num_key_value_heads": 8,
              "max_position_embeddings": 4096}
    monkeypatch.setattr(matrix.campaign, "_model_input", lambda path: {
        "path": path, "config": config, "weight_shards": {"model.safetensors": {"bytes": 1}}})
    monkeypatch.setattr(matrix.harness, "_load_model_config", lambda _: config)
    args = _args(["--execute", "--model-path", "llama31_8b=/models/llama",
                  "--prefill-len", "4032", "--decode-len", "64"])
    cases, _ = matrix._requested_cases(args)
    assert all(matrix.harness._max_model_len(matrix.budget_namespace(case)) == 4096 for case in cases)
    args.contexts = "4033"
    args.prefill_len = 4033
    with pytest.raises(ValueError, match="exceeds.*model limit"):
        matrix._requested_cases(args)


def test_scoped_policy_is_in_commands_and_provenance_and_restores_on_failure(tmp_path, monkeypatch):
    args = _args(["--models", "qwen3_30b", "--out-dir", str(tmp_path)])
    case = _case()
    monkeypatch.setattr(matrix, "_query_gpu_total_gib", lambda _: 81559 / 1024)
    monkeypatch.setattr(matrix, "_affinity", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(matrix, "resolved_budget", lambda _: {"manual_kv_budget_bytes": 123, "max_num_batched_tokens_explicit": 456})
    monkeypatch.setattr(matrix.campaign, "_provenance", lambda *_: {"original": True})
    originals = (matrix.campaign._cases, matrix.campaign._command, matrix.campaign._provenance, matrix.harness._gpu_total_gib)
    with pytest.raises(RuntimeError, match="test exception"):
        with matrix.configured_campaign(args, [case], {case.model: {"path": case.model_path}}):
            command = matrix.campaign._command(args, case, tmp_path)
            assert command[command.index("--kv-cache-memory-bytes") + 1] == "123"
            assert command[command.index("--max-num-batched-tokens") + 1] == "456"
            assert command[command.index("--warmups") + 1] == "1"
            assert command[command.index("--runs") + 1] == "3"
            assert command[command.index("--output-len") + 1] == "128"
            provenance = matrix.campaign._provenance({}, {})
            assert provenance["original"]
            assert provenance["cpu_affinity"] == [0, 1, 2, 3]
            assert provenance["benchmark_policy"] == matrix.POLICY
            raise RuntimeError("test exception")
    assert originals == (matrix.campaign._cases, matrix.campaign._command, matrix.campaign._provenance, matrix.harness._gpu_total_gib)


def test_preflight_failure_restores_gpu_query_helper(monkeypatch):
    args = _args()
    original = matrix.harness._gpu_total_gib
    monkeypatch.setattr(matrix, "_query_gpu_total_gib", lambda _: 80)
    monkeypatch.setattr(matrix, "resolved_budget", lambda _: (_ for _ in ()).throw(matrix.CapacityPreflightError("no fit")))
    with pytest.raises(matrix.CapacityPreflightError):
        with matrix.configured_campaign(args, [_case()], {}):
            pytest.fail("preflight failure entered campaign")
    assert matrix.harness._gpu_total_gib is original


def test_gpu_query_failure_never_assumes_capacity(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(matrix.subprocess, "check_output", unavailable)
    with pytest.raises(RuntimeError, match="Cannot query GPU"):
        matrix._query_gpu_total_gib("0")


def test_resume_reuses_matching_policy_and_rejects_changed_manual_budget(tmp_path, monkeypatch):
    args = _args(["--execute", "--models", "qwen3_30b", "--modes", "k3v2",
                  "--prefill-len", "255856", "--batches", "16", "--semantic-probe",
                  "--out-dir", str(tmp_path / "results")])
    case = _case()
    budget = {"manual_kv_budget_bytes": 123, "max_num_batched_tokens_explicit": 456}
    inputs = {case.model: {"path": case.model_path}}
    monkeypatch.setattr(matrix, "_query_gpu_total_gib", lambda _: 80)
    monkeypatch.setattr(matrix, "resolved_budget", lambda _: dict(budget))
    monkeypatch.setattr(matrix.campaign, "_provenance", lambda *_: {"test_provenance": "stable"})
    monkeypatch.setattr(matrix.campaign.shutil, "which", lambda _: None)
    calls = []

    def execute(command, **kwargs):
        attempt = Path(command[command.index("--out-dir") + 1])
        calls.append(attempt)
        rows = [{"run": run, "status": "ok", "num_prompts": case.batch,
                 "completed": case.batch, "failed": 0,
                 "total_input_tokens": case.ctx * case.batch,
                 "total_output_tokens": case.output_len * case.batch,
                 "ctx": case.ctx, "output_len": case.output_len, "max_concurrency": case.batch,
                 "jetfire_mode": matrix.campaign.MODES[case.mode][0],
                 "jetfire_group_size": matrix.campaign.MODES[case.mode][1],
                 "gpu_kv_cache_tokens": (case.ctx + case.output_len) * case.batch,
                 **{metric: float(run + 1) for metric in matrix.campaign.METRICS}}
                for run in range(case.runs)]
        (attempt / "summary.json").write_text(json.dumps({"status": "passed", "rows": rows}))
        (attempt / "semantic_probes.jsonl").write_text('{"status":"passed","text":"4"}\n' * case.runs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.campaign.subprocess, "run", execute)
    with matrix.configured_campaign(args, [case], inputs):
        assert matrix.campaign.run_campaign(args) == 0
    args.resume = True
    with matrix.configured_campaign(args, [case], inputs):
        assert matrix.campaign.run_campaign(args) == 0
    assert len(calls) == 1
    budget["manual_kv_budget_bytes"] += 1
    with matrix.configured_campaign(args, [case], inputs), pytest.raises(ValueError, match="options changed"):
        matrix.campaign.run_campaign(args)
    assert len(calls) == 1


def test_resume_does_not_reuse_evidence_for_changed_prefill_or_decode_length(tmp_path, monkeypatch):
    out = tmp_path / "results"
    monkeypatch.setattr(matrix, "_query_gpu_total_gib", lambda _: 80)
    monkeypatch.setattr(matrix, "resolved_budget", lambda _: {
        "manual_kv_budget_bytes": 123, "max_num_batched_tokens_explicit": 456})
    monkeypatch.setattr(matrix.campaign, "_provenance", lambda *_: {"test_provenance": "stable"})
    monkeypatch.setattr(matrix.campaign.shutil, "which", lambda _: None)
    calls = []

    def execute(command, **kwargs):
        ctx = int(command[command.index("--ctx") + 1])
        decode_len = int(command[command.index("--output-len") + 1])
        calls.append((ctx, decode_len))
        attempt = Path(command[command.index("--out-dir") + 1])
        rows = [{"run": run, "status": "ok", "num_prompts": 2, "completed": 2, "failed": 0,
                 "total_input_tokens": ctx * 2, "total_output_tokens": decode_len * 2,
                 "ctx": ctx, "output_len": decode_len, "max_concurrency": 2,
                 "jetfire_mode": matrix.campaign.MODES["k3v2"][0],
                 "jetfire_group_size": matrix.campaign.MODES["k3v2"][1],
                 **{metric: float(run + 1) for metric in matrix.campaign.METRICS}}
                for run in range(3)]
        (attempt / "summary.json").write_text(json.dumps({"status": "passed", "rows": rows}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.campaign.subprocess, "run", execute)
    combinations = [(1009, 64), (1010, 64), (1009, 257), (1009, 64)]
    for index, (ctx, decode_len) in enumerate(combinations):
        args = _args(["--execute", "--models", "qwen3_30b", "--modes", "k3v2", "--batches", "2",
                      "--prefill-len", str(ctx), "--decode-len", str(decode_len), "--out-dir", str(out),
                      *(["--resume"] if index else [])])
        case = _case(ctx=ctx, batch=2, output_len=decode_len)
        with matrix.configured_campaign(args, [case], {case.model: {"path": case.model_path}}):
            assert matrix.campaign.run_campaign(args) == 0
    assert calls == combinations[:3]
    assert len(list(out.glob("qwen3_30b_a3b/k3v2/*/case.json"))) == 3


@pytest.mark.parametrize("decode_len", [64, 257])
def test_benchmark_command_reaches_real_harness_and_client_without_starting_gpu(tmp_path, monkeypatch, decode_len):
    args = _args(["--models", "qwen3_30b", "--prefill-len", "1009", "--decode-len", str(decode_len)])
    case = matrix.campaign.Case("qwen3_30b_a3b", str(tmp_path), 1, "0", "k3v2", 1009, 7, decode_len, 3)
    command = matrix._command_with_budget(matrix.campaign._command, args, case, tmp_path / "attempt", {
        "manual_kv_budget_bytes": 13237223424, "max_num_batched_tokens_explicit": 260096})
    observed = []

    class Parsed(Exception):
        pass

    def stop_after_parse(parsed):
        observed.append(parsed)
        raise Parsed()

    monkeypatch.setattr(matrix.harness, "_preflight_model_length", stop_after_parse)
    monkeypatch.setattr(sys, "argv", ["official_serve_benchmark", *command[command.index("--") + 1:]])
    with pytest.raises(Parsed):
        matrix.harness.main()
    parsed = observed[0]
    assert parsed.ctx == 1009
    assert parsed.output_len == decode_len
    assert parsed.kv_cache_memory_bytes == 13237223424
    assert parsed.max_num_batched_tokens == 260096
    assert parsed.jetfire_prefill_sub_batch_size == "auto"
    assert parsed.max_model_len_padding == 16
    assert parsed.jetfire_kv_budget_safety == 1.02
    assert parsed.jetfire_prefill_runtime_reserve_gib == 25
    specs = matrix.harness._build_specs(parsed)
    assert [(spec.run_id, spec.concurrency) for spec in specs] == [(0, 7), (1, 7), (2, 7)]
    client = matrix.harness._bench_cmd(parsed, "http://127.0.0.1:8000", specs[0], tmp_path, "result.json")
    assert client[client.index("--random-input-len") + 1] == "1009"
    assert client[client.index("--random-output-len") + 1] == str(decode_len)
    assert client[client.index("--random-range-ratio") + 1] == "0"
    assert "--ignore-eos" in client
