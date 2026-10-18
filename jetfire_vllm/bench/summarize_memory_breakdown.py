#!/usr/bin/env python3
"""Summarize JetFire memory breakdown CSV into report and figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "qwen3_30b_a3b": "Qwen3-30B-A3B",
    "llama33_70b": "Llama-3.3-70B-Instruct",
}

CONFIG_LABELS = {
    "k4v4": "JetFire K4V4 g128",
    "k3v2": "JetFire K3V2 g64",
}

AVG_BITS = {"k4v4": 4.125, "k3v2": 2.75}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _computed_kv(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = defaultdict(float)
    for r in rows:
        if r.get("rank") != "-1":
            continue
        if r["side"] == "gpu" and r["stage"] == "computed" and r["component"].startswith("KV replica"):
            if r["component"] == "KV replica - total":
                continue
            out[(r["model"], r["config"], int(r["ctx"]))] += _float(r, "computed_gib")
    return dict(out)


def _host_replica(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for r in rows:
        if r.get("rank") != "-1":
            continue
        if r["side"] == "host" and r["stage"] == "computed" and r["component"] == "full-precision host replica":
            out[(r["model"], r["config"], int(r["ctx"]))] = _float(r, "computed_gib")
    return out


def _ratio_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for r in rows:
        if r.get("rank") != "-1":
            continue
        if r["component"] == "host/GPU replica ratio":
            out[(r["model"], r["config"], int(r["ctx"]))] = _float(r, "measured_gib")
    return out


def _write_report(rows: list[dict[str, str]], out_dir: Path) -> None:
    kv = _computed_kv(rows)
    host = _host_replica(rows)
    ratio = _ratio_rows(rows)
    max_ctx_by_model = {}
    for model, _config, ctx in kv:
        max_ctx_by_model[model] = max(max_ctx_by_model.get(model, 0), ctx)
    lines = [
        "# JetFire Memory Footprint Breakdown",
        "",
        "## Goal",
        "",
        "Measure GPU HBM and host DRAM footprint for JetFire K4V4 and JetFire K3V2, and make the KV-capacity argument visible instead of relying only on arithmetic in the paper text.",
        "",
        "## Setup",
        "",
        "- Configs: JetFire K4V4 g=128, JetFire K3V2 g=64.",
        "- Context grid: 4K, 8K, 16K, 32K, 64K, 128K; Qwen also includes 196K and 256K.",
        "- Rows with `rank=-1` are aggregate model-level accounting. Rows with `rank>=0` are per-GPU estimates for TP runs.",
        "- The current generated CSV is shape-derived unless `MODE=run` is used. Runtime measured rows can be appended with the same schema.",
        "",
        "Reproduce the computed matrix:",
        "",
        "```bash",
        "OUT_DIR=/tmp/jetfire_memory MODE=computed-only \\",
        "  bash ./jetfire_vllm/bench/run_memory_breakdown_suite.sh",
        "python /tmp/jetfire_memory/plot_memory_breakdown.py",
        "```",
        "",
        "## Law 1: linear-in-S KV, flat otherwise",
        "",
        "The KV replica rows scale linearly with context length. We keep weights/runtime/workspace separate in measured mode so non-KV overhead can be checked against this line.",
        "",
        "| Model | Config | ctx | GPU KV replica GiB |",
        "|---|---|---:|---:|",
    ]
    for (model, config, ctx), val in sorted(kv.items()):
        lines.append(f"| {model} | {config} | {ctx} | {val:.3f} |")
    lines.extend([
        "",
        "## Max-context anchor rows",
        "",
        "| Model | ctx | Config | Avg bits | GPU KV GiB | Host replica GiB | Host/GPU ratio |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for model in sorted(max_ctx_by_model):
        ctx = max_ctx_by_model[model]
        for config in ["k4v4", "k3v2"]:
            key = (model, config, ctx)
            if key not in kv:
                continue
            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {ctx} | {CONFIG_LABELS.get(config, config)} | "
                f"{AVG_BITS[config]:.3f} | {kv[key]:.3f} | {host.get(key, 0.0):.3f} | "
                f"{ratio.get(key, 0.0):.3f} |"
            )
    lines.extend([
        "",
        "## Law 2: slope proportional to average bits",
        "",
        "Expected average bits: BF16=16, K4V4=4.125, K3V2=2.75. The quant scale/zero row is kept separate in the CSV, so the small gap between pure packed bits and measured/computed KV is visible.",
        "",
        "| Config | Avg bits | Expected BF16/GPU ratio |",
        "|---|---:|---:|",
        "| K4V4 g128 | 4.125 | 3.879 |",
        "| K3V2 g64 | 2.750 | 5.818 |",
        "",
        "## Law 3: conservation across PCIe",
        "",
        "For JetFire configs, the HBM saved by low-bit KV reappears as the host full-precision replica. The aggregate host/GPU ratios are 3.879x for K4V4 and 5.818x for K3V2, matching 16/avg_bits.",
        "",
        "## Generated artifacts",
        "",
        "- `memory_breakdown.csv`: long-form source of truth for tables and plots.",
        "- `jetfire_memory_breakdown.pdf`: three-panel figure.",
        "- `plot_memory_breakdown.py`: matplotlib script used to regenerate the PDF.",
        "",
        "## Limitations",
        "",
        "- Computed rows are estimates from model shape. Runtime measured rows are present only when MODE=run was used.",
        "- JetFire detailed pool split uses worker RPC; the probe defaults avoid pre-generate worker RPC unless --collect-init-worker-rpc is explicitly requested.",
    ])
    (out_dir / "jetfire_memory_breakdown.md").write_text("\n".join(lines) + "\n")


def _write_plot_script(out_dir: Path) -> None:
    script = r'''#!/usr/bin/env python3
from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "memory_breakdown.csv"
rows = list(csv.DictReader(CSV.open()))

def f(x):
    try: return float(x or 0)
    except Exception: return 0.0

kv = defaultdict(float)
parts = defaultdict(lambda: defaultdict(float))
host = defaultdict(float)
for r in rows:
    if r.get("rank") != "-1":
        continue
    key = (r["model"], r["config"], int(r["ctx"]))
    if r["side"] == "gpu" and r["stage"] == "computed" and r["component"].startswith("KV replica") and r["component"] != "KV replica - total":
        kv[key] += f(r["computed_gib"])
        parts[key][r["component"]] += f(r["computed_gib"])
    if r["side"] == "host" and r["stage"] == "computed" and r["component"] == "full-precision host replica":
        host[key] += f(r["computed_gib"])

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# Panel A: Llama-70B K3V2 stack vs S.
model = "llama33_70b"; config = "k3v2"
ctxs = sorted({ctx for (m,c,ctx) in parts if m == model and c == config})
bottom = [0.0] * len(ctxs)
colors = {
    "KV replica - packed K": "#4C78A8",
    "KV replica - packed V": "#F58518",
    "KV replica - quant scales/zeros": "#54A24B",
}
for comp in ["KV replica - packed K", "KV replica - packed V", "KV replica - quant scales/zeros"]:
    vals = [parts[(model, config, ctx)].get(comp, 0.0) for ctx in ctxs]
    axes[0].bar([str(c//1024)+"K" for c in ctxs], vals, bottom=bottom, label=comp.replace("KV replica - ",""), color=colors[comp])
    bottom = [b+v for b,v in zip(bottom, vals)]
axes[0].axhline(160, color="black", linestyle="--", linewidth=1, label="160 GiB HBM")
axes[0].set_title("A. 70B K3V2 GPU KV budget")
axes[0].set_ylabel("GiB")
axes[0].legend(fontsize=8)

# Panel B: GPU KV line vs S.
for model, style in [("llama31_8b","-"), ("qwen3_30b_a3b","--"), ("llama33_70b",":")]:
    for config, color in [("k4v4","#4C78A8"), ("k3v2","#E45756")]:
        xs = sorted({ctx for (m,c,ctx) in kv if m == model and c == config})
        ys = [kv[(model, config, x)] for x in xs]
        if xs:
            axes[1].plot(xs, ys, linestyle=style, color=color, marker="o", alpha=0.8, label=f"{model}-{config}")
axes[1].set_xscale("log", base=2)
axes[1].set_title("B. GPU KV slope follows avg bits")
axes[1].set_xlabel("Context length")
axes[1].set_ylabel("GPU KV GiB")
axes[1].legend(fontsize=6, ncol=2)

# Panel C: Host vs GPU for K3V2.
model = "llama33_70b"; config = "k3v2"
ctxs = sorted({ctx for (m,c,ctx) in kv if m == model and c == config})
w = 0.35
labels = [str(c//1024)+"K" for c in ctxs]
x = range(len(ctxs))
axes[2].bar([i-w/2 for i in x], [kv[(model,config,c)] for c in ctxs], width=w, label="GPU low-bit replica", color="#E45756")
axes[2].bar([i+w/2 for i in x], [host[(model,config,c)] for c in ctxs], width=w, label="Host BF16 replica", color="#72B7B2")
axes[2].set_xticks(list(x), labels)
axes[2].set_title("C. HBM saved reappears in host DRAM")
axes[2].set_ylabel("GiB")
axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(ROOT / "jetfire_memory_breakdown.pdf")
'''
    path = out_dir / "plot_memory_breakdown.py"
    path.write_text(script)
    path.chmod(0o755)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read(args.csv)
    _write_report(rows, args.out_dir)
    _write_plot_script(args.out_dir)
    print(f"[memory-breakdown] wrote report and plot script to {args.out_dir}")


if __name__ == "__main__":
    main()
