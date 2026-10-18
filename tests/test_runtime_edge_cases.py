"""CPU-side regression checks for decode shape routing.

Numerical and CUDA-graph coverage is provided by validate_runtime_edges.py.
"""
from types import SimpleNamespace

import pytest

from jetfire_vllm.adapter.decode_path import _get_head_map_cpu, _select_top_k
from jetfire_vllm.adapter.kv_pool import JetFireKVPoolConfig, JetFireKVPoolManager


@pytest.mark.parametrize("length,expected", [(1, 32), (17, 32), (31, 32),
                                            (32, 32), (63, 32), (64, 64),
                                            (255, 128), (256, 256), (4096, 256)])
def test_positive_short_prompts_keep_cpu_correction(length, expected):
    assert _select_top_k(256, length, (32, 64, 128, 256)) == expected


@pytest.mark.parametrize("requested,length,supported", [(0, 128, (32, 64)),
                                                       (16, 128, (32, 64)),
                                                       (32, 0, (32, 64)),
                                                       (32, -1, (32, 64)),
                                                       (256, 128, ())])
def test_empty_or_unsupported_decode_width_is_rejected(requested, length, supported):
    assert _select_top_k(requested, length, supported) is None


def test_tp2_local_gqa_map_keeps_heads_inside_local_shadow():
    # Llama-3.3-70B TP2 has 32 query heads and 4 KV heads per worker.
    mapping = _get_head_map_cpu(2, 32, 8, 256)
    assert mapping.shape == (2, 32, 256)
    assert mapping.is_contiguous()
    for local_query_head in range(32):
        assert bool((mapping[:, local_query_head] == local_query_head // 8).all())


def test_tp2_layer_config_uses_local_attention_heads(monkeypatch):
    config = JetFireKVPoolConfig(enabled=True, group_size=128, topk=256,
                                layer_stride=1, k_nbits=3, v_nbits=2,
                                k_asymmetric=False, v_asymmetric=False)
    monkeypatch.setenv("JETFIRE_DEBUG_LOG", "0")
    pool = JetFireKVPoolManager(config)
    layer = SimpleNamespace(num_heads=32, num_kv_heads=4, head_size=128)
    actual = pool.ensure_layer_config_from_layer("model.layers.0.self_attn.attn", layer)
    assert actual.num_query_heads == 32
    assert actual.num_kv_heads == 4
    assert actual.head_dim == 128
