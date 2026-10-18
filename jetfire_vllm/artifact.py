"""Auditable, JetFire-only artifact evaluation entry point.

Every executed command has a log and return code. A failed command stops the
stage; a dry run produces a plan without claiming validation results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV = {
    "llama31_8b": "JETFIRE_LLAMA31_8B",
    "llama33_70b": "JETFIRE_LLAMA33_70B",
    "qwen3_30b_a3b": "JETFIRE_QWEN3_30B",
}


def _save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "triton", "vllm", "transformers", "datasets", "ninja", "rouge", "pytest"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _source_hashes() -> dict[str, str]:
    result = {}
    paths = [ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    for directory in ("jetfire_vllm", "scripts", "tests"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".cu", ".cpp", ".h", ".sh", ".json"})
    for path in sorted(paths):
        if path.is_file():
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _plan(args: argparse.Namespace, out: Path) -> list[list[str]]:
    py = sys.executable
    bench = "jetfire_vllm.bench."
    group = args.group_size or (128 if args.mode == "k4v4" else 64)
    runtime_mode = "SymInt4K_SymInt4V" if args.mode == "k4v4" else "SymInt3K_SymInt2V"
    if args.stage == "validate":
        return [[py, "-m", "pytest", "-q"], [py, str(ROOT / "scripts/validate_release_policy.py")],
                [py, "-m", "jetfire_vllm.adapter.run_sanity", "--require-vllm"]]
    if args.stage == "kernels":
        return [
            [py, str(ROOT / "scripts/validate_runtime.py"), "--all-modes", "--check-filler",
             "--json-out", str(out / "runtime.json")],
            [py, str(ROOT / "scripts/validate_runtime.py"), "--all-modes", "--top-k", "32",
             "--json-out", str(out / "runtime_k32.json")],
            [py, str(ROOT / "scripts/validate_runtime_edges.py"),
             "--json-out", str(out / "runtime_edges.json")],
        ]
    if args.stage == "compile":
        return [["bash", str(ROOT / "jetfire_vllm/bench/compile_jetfire_release.sh"), "--preset", "release",
                 "--group-size", str(group), "--top-k", str(args.top_k), "--cpu-attn-threads", str(args.cpu_threads)]]
    if args.stage == "serve":
        return [[py, "-m", bench + "official_serve_benchmark", "--mode", "fixed-concurrency", "--backend", "jetfire",
                 "--model", args.model_path, "--gpu", args.gpus, "--tensor-parallel-size", str(args.tp),
                 "--ctx", str(args.context), "--output-len", str(args.output_len), "--concurrency", str(args.batch),
                 "--num-prompts", str(args.batch), "--runs", str(args.runs), "--warmups", "1",
                 "--jetfire-profile", "graph", "--jetfire-graph-scope", "decode-only", "--chunked-prefill", "off",
                 "--jetfire-mode", runtime_mode, "--jetfire-group-size", str(group), "--top-k", str(args.top_k),
                 "--jetfire-prefill-chunk-tokens", str(args.prefill_chunk), "--cpu-attn-threads", str(args.cpu_threads),
                 "--out-dir", str(out / "serving")]]
    if args.stage in {"accuracy", "niah"}:
        cmd = ["bash", str(ROOT / "jetfire_vllm/bench/run_release_accuracy_suite.sh"), "--model", args.model_key,
               "--model-path", args.model_path,
               "--backend", "jetfire_" + args.mode, "--suite", "longbench_v1" if args.stage == "accuracy" else "niah_128k",
               "--gpus", args.gpus, "--tensor-parallel-size", str(args.tp), "--jetfire-group-size", str(group),
               "--top-k", str(args.top_k), "--max-samples", str(args.max_samples), "--save-predictions",
               "--jetfire-graph", "--cpu-attn-threads", str(args.cpu_threads), "--out-dir", str(out / args.stage)]
        if args.stage == "accuracy" and args.tasks:
            cmd.extend(["--tasks", args.tasks])
        if args.stage == "niah":
            cmd.extend(["--niah-contexts", str(args.context), "--niah-repeats", str(args.niah_repeats)])
        return [cmd]
    raise ValueError(args.stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["doctor", "validate", "kernels", "compile", "serve", "accuracy", "niah"])
    parser.add_argument("--model-key", choices=MODEL_ENV, default="llama31_8b")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-dir", default=os.environ.get("JETFIRE_LONGBENCH_DATA_DIR", ""))
    parser.add_argument("--mode", choices=["k3v2", "k4v4"], default="k3v2")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--context", "--prefill-len", dest="context", type=int, default=32768)
    parser.add_argument("--output-len", "--decode-len", dest="output_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--group-size", type=int, choices=[64, 128], default=0)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--prefill-chunk", default="auto")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates all samples; positive values are smoke tests")
    parser.add_argument("--niah-repeats", type=int, default=10)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_inputs = {
        "validate": [ROOT / "tests", ROOT / "scripts/validate_release_policy.py"],
        "kernels": [ROOT / "scripts/validate_runtime.py", ROOT / "scripts/validate_runtime_edges.py"],
        "compile": [ROOT / "jetfire_vllm/bench" / name for name in (
            "compile_jetfire_release.sh", "precompile_jetfire_extensions.py",
            "clean_jetfire_runtime_artifacts.py",
        )],
    }
    missing = [str(path) for path in source_inputs.get(args.stage, []) if not path.exists()]
    if missing and not args.dry_run:
        parser.error(
            f"{args.stage} requires the full JetFire source checkout. "
            "Run `python -m pip install -e '.[dev]'` from that checkout, then "
            "run this command there. Missing: " + ", ".join(missing)
        )
    for name in ("tp", "context", "output_len", "batch", "runs", "cpu_threads", "niah_repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples < 0 or args.top_k <= 0:
        parser.error("--max-samples must be nonnegative and --top-k must be positive")
    args.model_path = args.model_path or os.environ.get(MODEL_ENV[args.model_key], "") or os.environ.get("JETFIRE_MODEL_PATH", "")
    needs_model = args.stage in {"serve", "accuracy", "niah"}
    if needs_model and not args.model_path:
        if args.dry_run:
            args.model_path = "/path/to/" + args.model_key
        else:
            parser.error(f"set --model-path or {MODEL_ENV[args.model_key]}")
    if args.stage in {"accuracy", "niah"} and not args.data_dir and not args.dry_run:
        parser.error("set --data-dir or JETFIRE_LONGBENCH_DATA_DIR")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out_dir or ROOT / "results" / (args.stage + "_" + stamp)).resolve()
    commands = [] if args.stage == "doctor" else _plan(args, out)
    if args.dry_run:
        print(json.dumps({"status": "planned", "stage": args.stage, "commands": commands,
                          "environment": {MODEL_ENV[args.model_key]: args.model_path, "JETFIRE_LONGBENCH_DATA_DIR": args.data_dir}}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith("JETFIRE_")}
    # Keep a sibling checkout from satisfying an import accidentally.
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHON_BIN"] = sys.executable
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["TORCH_EXTENSIONS_DIR"] = env.get("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache/torch_extensions"))
    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
    env["JETFIRE_CPU_ATTN_THREADS"] = str(args.cpu_threads)
    env["JETFIRE_MODEL_PATH"] = args.model_path
    env[MODEL_ENV[args.model_key]] = args.model_path
    env["JETFIRE_LONGBENCH_DATA_DIR"] = args.data_dir
    input_hashes = {}
    if args.model_path:
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            candidate = Path(args.model_path) / name
            if candidate.is_file():
                input_hashes[str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if args.data_dir:
        for candidate in sorted(Path(args.data_dir).glob("*.jsonl")):
            with candidate.open("rb") as handle:
                input_hashes[str(candidate)] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"stage": args.stage, "status": "running", "started_utc": stamp,
                "arguments": vars(args), "python": sys.executable, "platform": platform.platform(),
                "versions": _versions(), "source_sha256": _source_hashes(), "input_sha256": input_hashes, "steps": [],
                "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "TORCH_EXTENSIONS_DIR", "OMP_NUM_THREADS")}}
    for name, cmd in (("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
                      ("cuda_compiler", ["nvcc", "--version"]), ("cpu", ["lscpu"])):
        manifest[name] = subprocess.run(cmd, capture_output=True, text=True).stdout if shutil.which(cmd[0]) else "unavailable"
    path = out / "manifest.json"
    _save(path, manifest)
    print(f"Artifact evidence: {out}", flush=True)
    try:
        for index, cmd in enumerate(commands):
            print("+ " + shlex.join(cmd), flush=True)
            log = out / f"step_{index:02d}.log"
            with log.open("w") as handle:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            manifest["steps"].append({"command": cmd, "returncode": proc.returncode, "log": log.name})
            _save(path, manifest)
            print(log.read_text()[-16000:], flush=True)
            if proc.returncode:
                manifest["status"] = "failed"
                _save(path, manifest)
                return proc.returncode if proc.returncode > 0 else 1
        manifest["status"] = "passed" if commands else "environment_recorded"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        _save(path, manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _save(path, manifest)
        raise
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _save(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
