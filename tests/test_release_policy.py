from __future__ import annotations

import pytest

from jetfire_vllm.adapter.release_policy import (
    ReleasePolicyError,
    plan_release_graph,
    validate_release_environment,
)


@pytest.mark.parametrize("k_bits", (2, 3, 4))
@pytest.mark.parametrize("top_k", (32, 256))
def test_batched_int4_v_policy_has_required_stage_a_statistics(k_bits, top_k):
    plan = plan_release_graph(
        mode=f"SymInt{k_bits}K_SymInt4V", group_size=128,
        top_k=top_k, max_graph_batch=2, stageb_candidate="", require_ready=True,
    )
    if plan.stage_b == "sv_int8tc_reg_stats":
        assert plan.stage_a == "qk_int8tc_stats"
    else:
        assert plan.stage_b == "sv_int4_wvs2_scores"


def _enable_release_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JETFIRE_RELEASE_GRAPH_POLICY", "1")


def test_release_policy_rejects_unmarked_approx_only(monkeypatch) -> None:
    _enable_release_policy(monkeypatch)
    monkeypatch.setenv("JETFIRE_DEBUG_APPROX_ONLY", "1")

    with pytest.raises(ReleasePolicyError, match="JETFIRE_DEBUG_APPROX_ONLY"):
        validate_release_environment()


def test_release_policy_accepts_accuracy_ablation(monkeypatch) -> None:
    _enable_release_policy(monkeypatch)
    monkeypatch.setenv("JETFIRE_DEBUG_APPROX_ONLY", "1")
    monkeypatch.setenv("JETFIRE_RELEASE_ACCURACY_ABLATION", "1")

    validate_release_environment()


def test_accuracy_ablation_does_not_allow_other_debug_knobs(monkeypatch) -> None:
    _enable_release_policy(monkeypatch)
    monkeypatch.setenv("JETFIRE_DEBUG_APPROX_ONLY", "1")
    monkeypatch.setenv("JETFIRE_RELEASE_ACCURACY_ABLATION", "1")
    monkeypatch.setenv("JETFIRE_DEBUG_CORRECTION_MODE", "gather")

    with pytest.raises(ReleasePolicyError, match="JETFIRE_DEBUG_CORRECTION_MODE"):
        validate_release_environment()
