"""Experiment-only PyTorch timeline collection for eager JetFire decode.

No kernel changes or added per-layer CUDA synchronizations. CPU worker ranges
are also saved with a monotonic clock because Kineto may omit annotations from
existing executor threads. Profiler start/stop synchronize only between requests.
"""

import contextlib
import inspect
import os
from pathlib import Path
import runpy
import threading
import time
import json

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "jetfire_vllm" / "adapter" / "sitecustomize.py"))

import torch
from vllm.v1.worker.gpu_worker import Worker
from jetfire_vllm.adapter import decode_path as decode
from jetfire_vllm.adapter import registry
from jetfire_vllm.adapter.config import get_jetfire_config
from jetfire_vllm.adapter.kv_pool import get_global_kv_pool_manager

ORIGINAL_TIMER = decode._timer
CAPTURE = False
PROFILER = None
ROWS = []
CLOCK = None
DESTINATION = None
MODE = "serial"


def clock_pair():
    before = time.perf_counter_ns()
    wall = time.time_ns()
    after = time.perf_counter_ns()
    return dict(perf_ns=(before + after) // 2, wall_ns=wall,
                uncertainty_ns=(after - before) // 2)


def timer(name):
    if not CAPTURE:
        return ORIGINAL_TIMER(name)
    caller = inspect.currentframe().f_back
    local = caller.f_locals
    layer = local.get("layer_name")
    if layer is None and local.get("layer") is not None:
        layer = getattr(local["layer"], "layer_name", None)
    trace = int(local.get("trace_id", -1))
    del caller
    label = f"JF|stage={name}|layer={layer}|trace={trace}"

    @contextlib.contextmanager
    def scope():
        with torch.profiler.record_function(label):
            start = time.perf_counter_ns()
            try:
                with ORIGINAL_TIMER(name):
                    yield
            finally:
                ROWS.append(dict(stage=name, layer=str(layer), trace=trace,
                    start_perf_ns=start, end_perf_ns=time.perf_counter_ns(),
                    tid=threading.get_native_id()))
    return scope()


decode._timer = timer
decode._PROFILE = False


def snapshot(worker):
    cfg = get_jetfire_config()
    pool = get_global_kv_pool_manager()
    model = worker.vllm_config.model_config.hf_config
    heads = int(model.num_attention_heads)
    return dict(mode=MODE, capture=CAPTURE,
        format=[cfg.scheme.k_nbits, cfg.scheme.v_nbits, cfg.group_size, cfg.topk],
        model_layers=int(model.num_hidden_layers), torch_cpu_threads=torch.get_num_threads(),
        model_shape=[int(model.num_hidden_layers), heads, int(model.num_key_value_heads),
                     int(getattr(model, "head_dim", None) or int(model.hidden_size) // heads)],
        cudagraph_mode=str(worker.vllm_config.compilation_config.cudagraph_mode),
        activation_partitions={name: getattr(registry, "_" + name + "_chunk_tokens")()
            for name in ("embed", "mlp", "attn_o_proj", "prefill_attn", "prefill_qkv")},
        environment={k: v for k, v in os.environ.items() if k.startswith("JETFIRE_")},
        decode_attempts=pool.decode_attempts, decode_hits=pool.decode_hits,
        native_attention_fallback_reasons=dict(pool.decode_fallbacks),
        timeout_gpu_fallback_implemented=False)


def control(worker, action="snapshot", mode=None, trace_name=None):
    global CAPTURE, PROFILER, CLOCK, DESTINATION, MODE
    if action not in ("snapshot", "set_mode", "start", "stop"):
        raise ValueError(action)
    # These RPCs run only after the preceding HTTP request has completed.
    torch.cuda.synchronize()
    if os.environ.get("JETFIRE_DECODE_GRAPH") != "0":
        raise RuntimeError("Overlap profiling requires eager decode, graphs OFF")
    if mode is not None:
        if mode not in ("serial", "overlap") or CAPTURE:
            raise ValueError("Change serial/overlap only between captures")
        MODE = mode
        os.environ["JETFIRE_SYNC_CPU_ATTN"] = "1" if mode == "serial" else "0"
    if action == "start":
        if PROFILER is not None:
            raise RuntimeError("A profiler capture is already active")
        if not trace_name or Path(trace_name).name != trace_name:
            raise ValueError("trace_name must be a simple filename")
        directory = Path(os.environ["JETFIRE_ARTIFACT_TRACE_ROOT"]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        DESTINATION = directory / trace_name
        ROWS.clear()
        pool = get_global_kv_pool_manager()
        pool.decode_attempts = 0
        pool.decode_hits = 0
        pool.decode_fallbacks.clear()
        CLOCK = clock_pair()
        PROFILER = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False, profile_memory=False,
        )
        PROFILER.start()
        CAPTURE = True
    elif action == "stop":
        if PROFILER is None:
            raise RuntimeError("No profiler capture is active")
        CAPTURE = False
        end_clock = clock_pair()
        PROFILER.stop()
        PROFILER.export_chrome_trace(str(DESTINATION))
        PROFILER = None
        sidecar = DESTINATION.with_suffix(".cpu.json")
        sidecar.write_text(json.dumps(dict(mode=MODE, start_clock=CLOCK,
            end_clock=end_clock, cpu_wall_ranges=ROWS, runtime=snapshot(worker)), indent=2))
        result = snapshot(worker)
        result.update(trace_path=str(DESTINATION), cpu_sidecar=str(sidecar),
                      cpu_ranges=len(ROWS))
        return result
    return snapshot(worker)


Worker.jetfire_artifact_overlap_control = control
