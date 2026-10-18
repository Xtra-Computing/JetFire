"""Read-only, between-request audit RPC for the Fig. 9 experiment.

Uses the repository's ordinary startup hook and adds no hot-path instrumentation.
Only artifact_optimizations.py places this directory on a server's PYTHONPATH.
"""

import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "jetfire_vllm" / "adapter" / "sitecustomize.py"))

import torch
from vllm.v1.worker.gpu_worker import Worker


def snapshot(worker):
    from jetfire_vllm.adapter.config import get_jetfire_config
    from jetfire_vllm.adapter import registry
    import jetfire_vllm

    cfg = get_jetfire_config()
    model = worker.vllm_config.model_config.hf_config
    hidden = int(model.hidden_size)
    heads = int(model.num_attention_heads)
    return {
        "package": jetfire_vllm.__file__,
        "format": [cfg.scheme.k_nbits, cfg.scheme.v_nbits, cfg.group_size, cfg.topk],
        "model_shape": [int(model.num_hidden_layers), heads,
                        int(model.num_key_value_heads),
                        int(getattr(model, "head_dim", None) or hidden // heads)],
        "torch_cpu_threads": torch.get_num_threads(),
        "cudagraph_mode": str(worker.vllm_config.compilation_config.cudagraph_mode),
        "activation_partitions": {
            name: getattr(registry, "_" + name + "_chunk_tokens")()
            for name in ("embed", "mlp", "attn_o_proj", "prefill_attn", "prefill_qkv")
        },
        "environment": {k: v for k, v in os.environ.items() if k.startswith("JETFIRE_")},
        "timeout_gpu_fallback_implemented": False,
        "preemption_reconstruction_implemented": False,
    }


Worker.jetfire_artifact_optimization_snapshot = snapshot
