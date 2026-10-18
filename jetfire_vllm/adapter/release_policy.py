"""Explicit release graph kernel policy.

This module is the boundary between "one release graph path" and the many
kernel specializations needed underneath it.  The policy must be deterministic:
given a quantization mode, group size, top-K, head shape, and graph batch, it
returns one plan or raises.  No caller should silently fall through to another
semantic path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from jetfire_vllm.adapter.config import QUANT_SCHEMES, QuantScheme


RELEASE_MODES: tuple[str, ...] = (
    "SymInt3K_SymInt2V",
    "SymInt4K_SymInt4V",
    "SymInt4K_SymInt3V",
    "SymInt4K_SymInt2V",
    "SymInt3K_SymInt4V",
    "SymInt3K_SymInt3V",
    "SymInt2K_SymInt4V",
    "SymInt2K_SymInt3V",
    "SymInt2K_SymInt2V",
)

RELEASE_GROUP_SIZES: tuple[int, ...] = (64, 128)
RELEASE_TOP_K: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
RELEASE_HEAD_DIM = 128
RELEASE_BLOCK_SIZE = 16
RELEASE_POLICY_ENV = "JETFIRE_RELEASE_GRAPH_POLICY"
RELEASE_REQUIRE_READY_ENV = "JETFIRE_RELEASE_REQUIRE_READY"
RELEASE_STAGEB_CANDIDATE_ENV = "JETFIRE_RELEASE_STAGEB_CANDIDATE"
RELEASE_STAGEB_CANDIDATES: tuple[str, ...] = ("", "int2-v2", "int2-wvs2")
RELEASE_ACCURACY_ABLATION_ENV = "JETFIRE_RELEASE_ACCURACY_ABLATION"

EXPERIMENTAL_RELEASE_ENV: tuple[str, ...] = (
    "JETFIRE_USE_FUSED_ATTN_TOPK",
    "JETFIRE_FUSED_KERNEL",
    "JETFIRE_INT8TC_QK_FUSED_TOPK",
    "JETFIRE_INT8TC_QK_GROUP_FUSED_TOPK",
    "JETFIRE_USE_INT8_TC",
    "JETFIRE_ALLOW_EXPERIMENTAL_QK_TOPK",
    "JETFIRE_FUSED_SOFTMAX_V",
    "JETFIRE_WVS_VARIANT",
    "JETFIRE_INT4_SVS_VARIANT",
    "JETFIRE_INT4_SVS",
    "JETFIRE_ALLOW_UNSTABLE_INT2_STAGEB",
    "JETFIRE_USE_INT8_TC_SV",
    "JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK",
    "JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK",
    "JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK",
    "JETFIRE_GRAPH_INLINE_CPU_TOPK",
    "JETFIRE_GRAPH_SYNC_CPU_TOPK",
    "JETFIRE_ALLOW_EXPERIMENTAL_GRAPH_CPU_TOPK",
    "JETFIRE_GRAPH_STAGEB_INPLACE",
    "JETFIRE_USE_PERSISTENT_WORKER",
    "JETFIRE_FUSED_JOIN_BLEND",
    "JETFIRE_USE_BLEND_KERNEL",
    "JETFIRE_DEBUG_CPU_TOPK_ONLY",
    "JETFIRE_DEBUG_APPROX_ONLY",
    "JETFIRE_DEBUG_CORRECTION_MODE",
    "JETFIRE_RELEASE_RESIDUAL_SCALE",
    "JETFIRE_DEBUG_RESIDUAL_SCALE",
    "JETFIRE_USE_CUSTOM_BACKEND",
    "JETFIRE_CPU_CORRECTION",
    "JETFIRE_DECODE_CPU_ATTN",
    "JETFIRE_SKIP_NATIVE_KV",
    "JETFIRE_SYNC_CPU_ATTN",
    "JETFIRE_DISABLE_ASYNC_SCHEDULING",
    "JETFIRE_FUSED_QUANT",
    "JETFIRE_FUSED_QUANT_WRITE",
    "JETFIRE_FUSED_KV_QUANT_WRITE",
    "JETFIRE_PREFILL_PIPELINE",
    "JETFIRE_PREFILL_STREAM_TOKENS",
    "JETFIRE_PIPELINE_MIN_TOKENS",
    "JETFIRE_SHADOW",
    "JETFIRE_SHADOW_DEVICE",
    "JETFIRE_SHADOW_DTYPE",
    "JETFIRE_ZERO_INIT_CACHE",
    "JETFIRE_SHADOW_DIRECT_D2H",
)

RELEASE_PREFILL_CHUNK_ENV: tuple[str, ...] = (
    "JETFIRE_EMBED_CHUNK_TOKENS",
    "JETFIRE_MLP_CHUNK_TOKENS",
    "JETFIRE_ATTN_O_PROJ_CHUNK_TOKENS",
    "JETFIRE_PREFILL_ATTN_CHUNK_TOKENS",
    "JETFIRE_PREFILL_QKV_CHUNK_TOKENS",
)
PREFILL_PARTITION_PROFILE_ENV = "JETFIRE_PREFILL_PARTITION_PROFILE"
RELEASE_PREFILL_PARTITION_SHAPE_ENV: tuple[str, ...] = (
    "JETFIRE_PREFILL_PARTITION_HIDDEN",
    "JETFIRE_PREFILL_PARTITION_LOCAL_INTERMEDIATE",
    "JETFIRE_PREFILL_PARTITION_ATTN_ELEMS",
)
RELEASE_PREFILL_PARTITION_PROFILES: dict[str, tuple[int, int, int, int, int]] = {
    "default": (2048, 2048, 2048, 2048, 2048),
    "perf_12g": (4096, 4096, 2048, 2048, 2048),
    "relaxed_10g5": (4096, 3072, 1536, 1536, 1536),
    "balanced_9g": (2048, 2048, 1024, 1024, 1024),
    "balanced_7g5": (2048, 1536, 768, 768, 768),
    "tight_6g25": (2048, 1024, 512, 512, 512),
    "tight_5g5": (1024, 512, 480, 480, 480),
    "tight_5g25": (1024, 512, 384, 384, 384),
    "floor_0g": (1024, 384, 256, 256, 256),
}
PREFILL_PARTITION_REF_HIDDEN = 8192
PREFILL_PARTITION_REF_INTERMEDIATE = 28672
PREFILL_PARTITION_REF_TP = 2
PREFILL_PARTITION_REF_LOCAL_INTERMEDIATE = (
    PREFILL_PARTITION_REF_INTERMEDIATE // PREFILL_PARTITION_REF_TP
)
PREFILL_PARTITION_REF_ATTN_ELEMS = PREFILL_PARTITION_REF_HIDDEN

CANONICAL_RELEASE_ENV: dict[str, tuple[str, ...]] = {
    "JETFIRE_USE_PERSISTENT_WORKER": ("0",),
    "JETFIRE_FUSED_JOIN_BLEND": ("0",),
    "JETFIRE_USE_INT8_TC": ("1",),
    "JETFIRE_USE_CUSTOM_BACKEND": ("1",),
    "JETFIRE_CPU_CORRECTION": ("1",),
    "JETFIRE_DECODE_CPU_ATTN": ("1",),
    "JETFIRE_SKIP_NATIVE_KV": ("1",),
    "JETFIRE_SYNC_CPU_ATTN": ("0",),
    "JETFIRE_DISABLE_ASYNC_SCHEDULING": ("1",),
    "JETFIRE_USE_INT8_TC_SV": ("0",),
    "JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK": ("0",),
    "JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK": ("0",),
    "JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK": ("1",),
    "JETFIRE_FUSED_QUANT": ("1",),
    "JETFIRE_FUSED_QUANT_WRITE": ("1",),
    "JETFIRE_FUSED_KV_QUANT_WRITE": ("1",),
    "JETFIRE_PREFILL_PIPELINE": ("1",),
    "JETFIRE_PREFILL_STREAM_TOKENS": ("0",),
    "JETFIRE_PIPELINE_MIN_TOKENS": ("64",),
    "JETFIRE_SHADOW": ("1",),
    "JETFIRE_SHADOW_DEVICE": ("cpu",),
    "JETFIRE_SHADOW_DTYPE": ("bf16",),
    "JETFIRE_ZERO_INIT_CACHE": ("0",),
    "JETFIRE_SHADOW_DIRECT_D2H": ("0",),
}


class ReleasePolicyError(ValueError):
    """The requested config is outside the release graph contract."""


@dataclass(frozen=True)
class ReleaseKernelPlan:
    mode: str
    k_nbits: int
    v_nbits: int
    group_size: int
    top_k: int
    head_dim: int
    num_quant_groups: int
    stage_a: str
    stage_b: str
    correction_sync: str
    graph_scope: str
    promotion_state: str
    release_ready: bool
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def assert_ready(self) -> None:
        if not self.release_ready:
            details = "; ".join(self.blockers) if self.blockers else "unknown blocker"
            raise ReleasePolicyError(
                f"release graph plan is not ready for {self.mode} "
                f"g{self.group_size} top_k={self.top_k}: {details}"
            )


def _require_in(name: str, value: int | str, allowed: Iterable[int | str]) -> None:
    allowed_tuple = tuple(allowed)
    if value not in allowed_tuple:
        raise ReleasePolicyError(
            f"{name}={value!r} is not in release matrix {allowed_tuple!r}"
        )


def _scheme(mode: str) -> QuantScheme:
    _require_in("mode", mode, RELEASE_MODES)
    scheme = QUANT_SCHEMES.get(mode)
    if scheme is None or scheme.kernel_module is None:
        raise ReleasePolicyError(f"mode={mode!r} has no release kernel module")
    return scheme


def release_policy_enabled() -> bool:
    return os.environ.get(RELEASE_POLICY_ENV, "0") == "1"


def release_require_ready() -> bool:
    return os.environ.get(RELEASE_REQUIRE_READY_ENV, "0") == "1"


def release_stageb_candidate() -> str:
    return os.environ.get(RELEASE_STAGEB_CANDIDATE_ENV, "").strip()


def _release_prefill_partition_shape() -> dict[str, int] | None:
    keys = {
        "hidden": "JETFIRE_PREFILL_PARTITION_HIDDEN",
        "local_intermediate": "JETFIRE_PREFILL_PARTITION_LOCAL_INTERMEDIATE",
        "attn_elems": "JETFIRE_PREFILL_PARTITION_ATTN_ELEMS",
    }
    values: dict[str, int] = {}
    for field, env_name in keys.items():
        raw = os.environ.get(env_name)
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        values[field] = value
    return values


def _round_partition_tokens(value: float) -> int:
    if value <= 0:
        return 0
    rounded = int(value // 16) * 16
    return max(16, rounded)


def _expected_prefill_partition_values(
    partition_profile: str,
) -> dict[str, int] | None:
    base = RELEASE_PREFILL_PARTITION_PROFILES.get(partition_profile)
    if base is None:
        return None
    if partition_profile == "default":
        return dict(zip(RELEASE_PREFILL_CHUNK_ENV, base))
    shape = _release_prefill_partition_shape()
    if shape is None:
        return None
    embed, mlp, attn_o_proj, prefill_attn, prefill_qkv = base
    expected = (
        _round_partition_tokens(
            embed * PREFILL_PARTITION_REF_HIDDEN / shape["hidden"]
        ),
        _round_partition_tokens(
            mlp
            * PREFILL_PARTITION_REF_LOCAL_INTERMEDIATE
            / shape["local_intermediate"]
        ),
        _round_partition_tokens(
            attn_o_proj
            * PREFILL_PARTITION_REF_ATTN_ELEMS
            / shape["attn_elems"]
        ),
        _round_partition_tokens(
            prefill_attn
            * PREFILL_PARTITION_REF_ATTN_ELEMS
            / shape["attn_elems"]
        ),
        _round_partition_tokens(
            prefill_qkv
            * PREFILL_PARTITION_REF_ATTN_ELEMS
            / shape["attn_elems"]
        ),
    )
    return dict(zip(RELEASE_PREFILL_CHUNK_ENV, expected))


def validate_release_environment() -> None:
    """Reject hidden branch-changing env knobs during release policy runs."""

    if not release_policy_enabled():
        return
    allow_graph_sync_ab = (
        os.environ.get("JETFIRE_ALLOW_EXPERIMENTAL_GRAPH_CPU_TOPK", "0") == "1"
    )
    allow_accuracy_ablation = (
        os.environ.get(RELEASE_ACCURACY_ABLATION_ENV, "0") == "1"
    )
    offenders: list[str] = []
    for name in EXPERIMENTAL_RELEASE_ENV:
        raw = os.environ.get(name)
        if raw is None:
            continue
        if allow_graph_sync_ab and name in {
            "JETFIRE_GRAPH_DEVICE_SIGNAL_CPU_TOPK",
            "JETFIRE_GRAPH_GPU_WAIT_CPU_TOPK",
            "JETFIRE_GRAPH_HOST_JOIN_CPU_TOPK",
            "JETFIRE_GRAPH_INLINE_CPU_TOPK",
            "JETFIRE_GRAPH_SYNC_CPU_TOPK",
            "JETFIRE_USE_BLEND_KERNEL",
            "JETFIRE_FUSED_JOIN_BLEND",
        }:
            continue
        if (
            allow_accuracy_ablation
            and name == "JETFIRE_DEBUG_APPROX_ONLY"
            and raw == "1"
        ):
            continue
        canonical = CANONICAL_RELEASE_ENV.get(name)
        if canonical is not None and raw.lower() in canonical:
            continue
        offenders.append(f"{name}={raw!r}")
    partition_profile = os.environ.get(PREFILL_PARTITION_PROFILE_ENV)
    if partition_profile is not None and partition_profile not in {
        *RELEASE_PREFILL_PARTITION_PROFILES.keys(),
        "user_uniform",
    }:
        offenders.append(f"{PREFILL_PARTITION_PROFILE_ENV}={partition_profile!r}")
    if (
        partition_profile in RELEASE_PREFILL_PARTITION_PROFILES
        and partition_profile != "default"
        and _release_prefill_partition_shape() is None
    ):
        offenders.append(
            "invalid release prefill partition shape metadata: "
            + ", ".join(
                f"{name}={os.environ.get(name)!r}"
                for name in RELEASE_PREFILL_PARTITION_SHAPE_ENV
            )
        )

    chunk_values: dict[str, int] = {}
    for name in RELEASE_PREFILL_CHUNK_ENV:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            offenders.append(f"{name}={raw!r}")
            continue
        if value < 0:
            offenders.append(f"{name}={raw!r}")
            continue
        chunk_values[name] = value
    if partition_profile is not None and not chunk_values:
        offenders.append(f"{PREFILL_PARTITION_PROFILE_ENV}={partition_profile!r}")
    if chunk_values:
        values = set(chunk_values.values())
        if partition_profile in RELEASE_PREFILL_PARTITION_PROFILES:
            expected = _expected_prefill_partition_values(partition_profile)
            if chunk_values != expected:
                offenders.append(
                    f"{PREFILL_PARTITION_PROFILE_ENV}={partition_profile!r} "
                    "does not match release prefill partition profile: "
                    + ", ".join(
                        f"{k}={chunk_values.get(k)!r}"
                        for k in RELEASE_PREFILL_CHUNK_ENV
                    )
                )
        elif len(values) != 1:
            offenders.append(
                "inconsistent prefill chunk envs: "
                + ", ".join(f"{k}={v!r}" for k, v in sorted(chunk_values.items()))
            )
    if offenders:
        raise ReleasePolicyError(
            "release graph policy rejects experimental env knobs: "
            + ", ".join(sorted(offenders))
        )
    candidate = release_stageb_candidate()
    if candidate not in RELEASE_STAGEB_CANDIDATES:
        raise ReleasePolicyError(
            f"{RELEASE_STAGEB_CANDIDATE_ENV}={candidate!r} is not one of "
            f"{RELEASE_STAGEB_CANDIDATES!r}"
        )


def _stage_a_family(
    *,
    scheme: QuantScheme,
    group_size: int,
    head_dim: int,
    top_k: int,
) -> tuple[str, tuple[str, ...]]:
    notes: list[str] = []
    if (
        scheme.k_nbits == 4
        and not scheme.k_asymmetric
        and group_size == head_dim
        and top_k in (64, 128, 256, 512, 1024, 2048, 4096, 8192)
    ):
        return "qk_int8tc_stats", tuple(notes)

    if scheme.k_nbits in (2, 3, 4):
        if scheme.k_nbits == 4 and group_size != head_dim:
            notes.append(
                "K4 g64 uses split QK because current INT8TC QK expects one "
                "scale group per head."
            )
        return f"qk_int{scheme.k_nbits}_split_topk", tuple(notes)

    raise ReleasePolicyError(f"unsupported K nbits: {scheme.k_nbits}")


def _stage_b_family(
    *,
    scheme: QuantScheme,
    group_size: int,
    head_dim: int,
    max_graph_batch: int,
    stage_a: str,
    stageb_candidate: str = "",
) -> tuple[str, str, bool, tuple[str, ...], tuple[str, ...]]:
    blockers: list[str] = []
    notes: list[str] = []
    num_quant_groups = head_dim // group_size

    if stageb_candidate:
        if stageb_candidate not in RELEASE_STAGEB_CANDIDATES:
            raise ReleasePolicyError(
                f"{RELEASE_STAGEB_CANDIDATE_ENV}={stageb_candidate!r} is not "
                f"one of {RELEASE_STAGEB_CANDIDATES!r}"
            )
        if scheme.v_nbits != 2:
            raise ReleasePolicyError(
                f"{RELEASE_STAGEB_CANDIDATE_ENV}={stageb_candidate!r} only "
                "applies to INT2 V modes"
            )
        if stageb_candidate == "int2-wvs2" and num_quant_groups != 1:
            raise ReleasePolicyError(
                "INT2 wvs2 Stage-B requires one V scale group per head "
                f"(group_size={group_size}, head_dim={head_dim})"
            )
        blockers.append(
            f"candidate Stage-B override {stageb_candidate!r} requires "
            "graph-level promotion evidence before release."
        )
        stage_b = (
            "sv_int2_v2" if stageb_candidate == "int2-v2"
            else "sv_int2_wvs2"
        )
        return stage_b, "candidate", False, tuple(blockers), tuple(notes)

    if (
        scheme.v_nbits == 4
        and stage_a == "qk_int8tc_stats"
        and group_size == head_dim
        and max_graph_batch > 1
    ):
        # Only INT4 Stage A emits the softmax statistics consumed here.
        # Other K formats use the score-based Stage B below.
        return "sv_int8tc_reg_stats", "ready", True, tuple(blockers), tuple(notes)

    if scheme.v_nbits == 4 and group_size == head_dim:
        return "sv_int4_wvs2_scores", "ready", True, tuple(blockers), tuple(notes)

    if scheme.v_nbits == 3 and group_size == head_dim:
        return "sv_int3_wvs2", "ready", True, tuple(blockers), tuple(notes)

    if scheme.v_nbits == 2 and num_quant_groups == 1:
        notes.append(
            "INT2 V wvs2 is promoted for the single-GPU release graph path; "
            "TP=2 long-context performance remains validation-only until the "
            "full graph-level chunk-size sweep is complete."
        )
        return "sv_int2_wvs2", "ready", True, tuple(blockers), tuple(notes)

    if scheme.v_nbits in (2, 3, 4) and num_quant_groups == 2:
        stage_b = f"sv_int{scheme.v_nbits}_wvs2"
        notes.append("multi-group wvs2 Stage-B is promoted for group_size=64.")
        if scheme.v_nbits == 2:
            notes.append(
                "INT2 V wvs2 is promoted for the single-GPU release graph path; "
                "TP=2 long-context performance remains validation-only until the "
                "full graph-level chunk-size sweep is complete."
            )
        return stage_b, "ready", True, tuple(blockers), tuple(notes)

    # Current code can execute this split family for unsupported V/group
    # combinations. Keep it explicit and blocked instead of silently falling
    # through.
    if scheme.v_nbits in (2, 3, 4):
        stage_b = f"sv_int{scheme.v_nbits}_split_multigroup"
        if scheme.v_nbits == 2:
            blockers.append(
                "INT2 V Stage-B for this shape has no promoted release kernel."
            )
        if num_quant_groups != 1:
            blockers.append(
                "multi-group Stage-B for group_size < head_dim currently uses "
                "split implementation instead of wvs2/WMMA release kernel."
            )
        return stage_b, "candidate", False, tuple(blockers), tuple(notes)

    raise ReleasePolicyError(f"unsupported V nbits: {scheme.v_nbits}")


def plan_release_graph(
    *,
    mode: str,
    group_size: int,
    top_k: int,
    head_dim: int = RELEASE_HEAD_DIM,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    max_graph_batch: int = 1,
    stageb_candidate: str | None = None,
    require_ready: bool = False,
) -> ReleaseKernelPlan:
    """Return the explicit release graph plan for one matrix cell.

    `release_ready=False` means the current copied code can likely execute the
    cell, but the cell still relies on a path that must be promoted or replaced
    before it can be called a release specialization.
    """

    del num_query_heads, num_kv_heads  # Reserved for GQA-specific policy.
    scheme = _scheme(mode)
    _require_in("group_size", int(group_size), RELEASE_GROUP_SIZES)
    _require_in("top_k", int(top_k), RELEASE_TOP_K)
    if int(head_dim) != RELEASE_HEAD_DIM:
        raise ReleasePolicyError(f"head_dim={head_dim} is not release head_dim=128")
    if head_dim % group_size != 0:
        raise ReleasePolicyError(
            f"group_size={group_size} does not divide head_dim={head_dim}"
        )
    if int(max_graph_batch) < 1:
        raise ReleasePolicyError("max_graph_batch must be >= 1")
    candidate = release_stageb_candidate() if stageb_candidate is None else stageb_candidate
    if candidate and require_ready:
        raise ReleasePolicyError(
            f"{RELEASE_STAGEB_CANDIDATE_ENV}={candidate!r} is only allowed "
            "when require_ready=False"
        )

    stage_a, notes_a = _stage_a_family(
        scheme=scheme,
        group_size=int(group_size),
        head_dim=int(head_dim),
        top_k=int(top_k),
    )
    stage_b, promotion_state, ready_b, blockers_b, notes_b = _stage_b_family(
        scheme=scheme,
        group_size=int(group_size),
        head_dim=int(head_dim),
        max_graph_batch=int(max_graph_batch),
        stage_a=stage_a,
        stageb_candidate=candidate,
    )
    plan = ReleaseKernelPlan(
        mode=mode,
        k_nbits=scheme.k_nbits,
        v_nbits=scheme.v_nbits,
        group_size=int(group_size),
        top_k=int(top_k),
        head_dim=int(head_dim),
        num_quant_groups=int(head_dim) // int(group_size),
        stage_a=stage_a,
        stage_b=stage_b,
        correction_sync="cpu_topk_host_join",
        graph_scope="decode_only",
        promotion_state=promotion_state,
        release_ready=ready_b,
        blockers=blockers_b,
        notes=notes_a + notes_b,
    )
    if require_ready:
        plan.assert_ready()
    return plan


def iter_release_matrix(
    *,
    modes: Iterable[str] = RELEASE_MODES,
    group_sizes: Iterable[int] = RELEASE_GROUP_SIZES,
    top_ks: Iterable[int] = RELEASE_TOP_K,
    max_graph_batch: int = 1,
) -> list[ReleaseKernelPlan]:
    plans: list[ReleaseKernelPlan] = []
    for mode in modes:
        for group_size in group_sizes:
            for top_k in top_ks:
                plans.append(
                    plan_release_graph(
                        mode=mode,
                        group_size=group_size,
                        top_k=top_k,
                        max_graph_batch=max_graph_batch,
                    )
                )
    return plans
