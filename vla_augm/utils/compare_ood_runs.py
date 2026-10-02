"""Measure eval noise: compare two sweeps of the SAME checkpoint.

The OOD numbers are compared across arms, so a claim like "shift beats none"
only means something if it is larger than the spread between two runs of one
checkpoint. Evaluation is not deterministic despite ``use_fixed_reset_state_ids``
-- GPU nondeterminism, env subprocess scheduling and the policy's own sampling
all move individual episodes.

This prints the aggregate delta, the per-axis deltas, and the fraction of tasks
whose success flips, which together give the threshold below which an arm-vs-arm
difference should not be believed.

Usage:
    python vla_augm/utils/compare_ood_runs.py <run_a.jsonl> <run_b.jsonl>
"""

import argparse
import json
import pathlib
from collections import defaultdict

from rlinf.envs.libero.libero_plus_axes import AXIS_CATEGORIES, get_axis_task_ids

SUITE = "libero_spatial"


def load(path: pathlib.Path) -> dict[int, float]:
    """success_once per task id."""
    out: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[int(row["task_id"])] = float(row["success_once"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=pathlib.Path)
    parser.add_argument("run_b", type=pathlib.Path)
    args = parser.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no task ids in common")

    per_axis = get_axis_task_ids(SUITE, sorted(AXIS_CATEGORIES), None, 1.0)
    axis_of = {t: ax for ax, ids in per_axis.items() for t in ids}

    mean_a = sum(a[t] for t in shared) / len(shared)
    mean_b = sum(b[t] for t in shared) / len(shared)
    flips = sum(1 for t in shared if a[t] != b[t])

    print(f"tasks compared: {len(shared)}")
    print(f"overall: {mean_a:.4f} vs {mean_b:.4f}   delta {abs(mean_a - mean_b):.4f}")
    print(f"flipped: {flips} ({flips / len(shared):.1%} of tasks)")
    print("\nper axis:")
    worst = 0.0
    for axis in sorted(AXIS_CATEGORIES):
        ids = [t for t in shared if axis_of.get(t) == axis]
        if not ids:
            continue
        ma = sum(a[t] for t in ids) / len(ids)
        mb = sum(b[t] for t in ids) / len(ids)
        worst = max(worst, abs(ma - mb))
        print(f"  {axis:<12} n={len(ids):<4} {ma:.3f} vs {mb:.3f}   delta {abs(ma - mb):.3f}")

    # The headline number the paper needs: two runs of one checkpoint should
    # never differ by more than this, so an arm-vs-arm gap at or below it is
    # not evidence of anything.
    print(
        f"\nnoise floor: aggregate {abs(mean_a - mean_b):.3f}, "
        f"worst axis {worst:.3f} -- treat differences at or below these as noise"
    )


if __name__ == "__main__":
    main()
