#!/usr/bin/env python
"""Convert a standalone LIBERO eval console log into ood_eval JSONL.

Training runs write per-episode OOD results themselves, but a standalone
``only_eval`` job (evaluations/libero/*, run via eval_pi0_spatial_plus.sh) only
emits ``[libero eval] task_id=..., trial_id=..., success=...`` lines. This turns
those into the same schema, so vla_augm/utils/analyze_ood_eval.py works on both and a
standalone full-suite sweep can be compared against a training curve.

Usage:
    python vla_augm/utils/parse_libero_eval_log.py vla_augm/console_logs/pi0-spatial-plus-eval_24815250.log \\
        --out logs/pi0_plus_sft/ood_eval --suite libero_spatial --step 500
    python vla_augm/utils/analyze_ood_eval.py logs/pi0_plus_sft/ood_eval

Unlike the training-time eval, this labels *all seven* LIBERO-Plus axes, so it
answers which axes have dynamic range for a given policy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# vla_augm/utils/<file> -> repo root is three levels up.
REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)
os.environ.setdefault("REPO_PATH", REPO)

from rlinf.envs.libero.libero_plus_axes import (  # noqa: E402
    AXIS_CATEGORIES,
    resolve_classification_path,
)

LINE = re.compile(
    r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)"
)
# Strip SLURM/Ray ANSI colour codes before matching.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_axis_map(suite: str, path: str | None) -> dict[int, str]:
    """task_id (0-based) -> short axis name, covering all seven axes."""
    resolved = resolve_classification_path(path)
    with open(resolved) as f:
        data = json.load(f)
    if suite not in data:
        raise SystemExit(f"suite {suite!r} not in {resolved}; have {sorted(data)}")
    to_short = {v: k for k, v in AXIS_CATEGORIES.items()}
    return {
        int(t["id"]) - 1: to_short.get(str(t["category"]), "unknown")
        for t in data[suite]
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="console log from a standalone eval job")
    ap.add_argument("--out", required=True, help="output ood_eval directory")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--step", type=int, default=0, help="step label for the JSONL")
    ap.add_argument("--task-classification-path", default=None)
    args = ap.parse_args()

    axis_of = load_axis_map(args.suite, args.task_classification_path)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"step_{args.step}.jsonl")

    written = 0
    unknown = 0
    with open(args.log, errors="replace") as src, open(out_path, "w") as dst:
        for raw in src:
            m = LINE.search(ANSI.sub("", raw))
            if not m:
                continue
            task_id, trial_id, success = int(m[1]), int(m[2]), m[3] == "True"
            axis = axis_of.get(task_id)
            if axis is None:
                unknown += 1
                continue
            dst.write(
                json.dumps(
                    {
                        "step": args.step,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "axis": axis,
                        "success_once": 1.0 if success else 0.0,
                    }
                )
                + "\n"
            )
            written += 1

    print(f"wrote {written} episodes to {out_path}")
    if unknown:
        print(f"  {unknown} episodes had task ids outside the {args.suite} map")
    if written == 0:
        print("  no '[libero eval]' lines found - is this a standalone eval log?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
