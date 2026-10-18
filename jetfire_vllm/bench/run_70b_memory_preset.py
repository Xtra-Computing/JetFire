"""Run optional TP=2 memory settings using the release launcher.

Example (activate the jetfire environment first):
  python jetfire_vllm/bench/run_70b_memory_preset.py \
    --model /path/to/Llama-3.3-70B-Instruct --gpu 0,1 --out-dir /path/to/results
Add --dry-run to inspect without starting a GPU process. This is not a generic
70B auto-tuner. The optional B4/64K candidate is admission-limited, not full-batch
resident and not an authenticated historical configuration. Environment overrides
of JetFire settings are deliberately removed to preserve the preset contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setting", choices=("b16_c16384", "b4_c65536_admission_candidate"),
                        default="b16_c16384")
    args = parser.parse_args()
    gpu_ids = args.gpu.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or any(not v.strip() for v in gpu_ids):
        parser.error("This TP2 preset requires two distinct GPU IDs")
    bench = Path(__file__).resolve().parent
    preset = json.loads((bench / f"presets/llama33_70b_k4v4_{args.setting}.json").read_text())
    config = args.model / "config.json"
    if not config.is_file() or hashlib.sha256(config.read_bytes()).hexdigest() != preset["model_config_sha256"]:
        parser.error("Model config does not match the preset Llama-3.3-70B-Instruct config")
    cmd = ["bash", str(bench / "run_jetfire_release_bench.sh"),
           "--model", str(args.model.resolve()), "--gpu", args.gpu,
           "--out-dir", str(args.out_dir.resolve()), *preset["benchmark_args"]]
    print(shlex.join(cmd), flush=True)
    print(preset["scope"], flush=True)
    if args.dry_run:
        return 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("JETFIRE_")}
    env["PYTHON_BIN"] = sys.executable
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("PYTORCH_ALLOC_CONF", None)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
