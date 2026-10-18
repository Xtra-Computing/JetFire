#!/usr/bin/env python3
"""Validate the explicit release graph policy without launching vLLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    from jetfire_vllm.adapter.release_policy import iter_release_matrix

    rows = []
    failed = False
    for plan in iter_release_matrix():
        row = {
            "mode": plan.mode,
            "group_size": plan.group_size,
            "top_k": plan.top_k,
            "stage_a": plan.stage_a,
            "stage_b": plan.stage_b,
            "correction_sync": plan.correction_sync,
            "graph_scope": plan.graph_scope,
            "promotion_state": plan.promotion_state,
            "release_ready": plan.release_ready,
            "blockers": list(plan.blockers),
            "notes": list(plan.notes),
        }
        rows.append(row)
        status = "READY" if plan.release_ready else plan.promotion_state.upper()
        if args.require_ready and not plan.release_ready:
            failed = True
        print(
            f"{status:7s} {plan.mode:30s} "
            f"g={plan.group_size:<3d} top_k={plan.top_k:<5d} "
            f"A={plan.stage_a:<22s} B={plan.stage_b}"
        )
        for blocker in plan.blockers:
            print(f"        blocker: {blocker}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
