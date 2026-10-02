#!/usr/bin/env python
"""Pick the LIBERO-Plus tasks to record videos for, ranked by robustness.

The first version picked tasks from a single ep300 sweep and 34% of the clips
came out with the opposite outcome (job 25834083): a one-env reel draws
different reset-state indices than a 32-env sweep, so the episodes start from
different object placements. A single sweep says nothing about whether a task is
reliably solved or reliably missed.

So rank tasks by how many of the arm's five late sweeps (ep200..ep300) solved
them. 5/5 is a robust success, 0/5 a robust failure, and those survive an init
change far better. Then over-record -- more candidates than the figure needs --
because the outcome is only knowable after running. The clips are named
<axis>_base<b>_task<id>_<ok|fail>.mp4, so post-selecting the figure's two of
each per axis is a matter of reading filenames.

One axis cannot be balanced at all: the arm solves EVERY texture task in at
least 2 of 5 sweeps (0 tasks at 0/5, 1 at 2/5, 190 at 5/5). Its failure
candidates are therefore "sometimes fails", not "always fails", and the script
says so rather than pretending otherwise.

Usage:
    python vla_augm/analyze/pick_video_tasks.py
    python vla_augm/analyze/pick_video_tasks.py --n-ok 6 --n-fail 8
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rlinf.envs.libero.libero_plus_axes import (  # noqa: E402
    AXIS_CATEGORIES,
    get_axis_task_ids,
)

FIVE_AXES = ["textures", "lighting", "layout", "camera", "noise"]
# Late sweeps only: early checkpoints are a different policy, not a repeat.
SWEEP_EPOCHS = (200, 225, 250, 275, 300)
SWEEPS = REPO / "vla_augm/results/ood_full"
ARM = "critic_only_ovl_alpha05"
BDDL = REPO / ".venv-pi/libero/libero/libero/bddl_files/libero_spatial"
CLASSIFICATION = REPO / "vla_augm/utils/task_classification.json"

# LIBERO-Spatial is one manipulation x ten spatial relations, so the only
# diversity on offer is where the bowl starts. Group by that anchor so the reel
# does not shoot the same scene five times.
ANCHOR_GROUP = {
    0: "plate/ramekin", 4: "plate/ramekin", 5: "plate/ramekin", 7: "plate/ramekin",
    3: "cookie box", 6: "cookie box",
    2: "wooden cabinet", 9: "wooden cabinet",
    8: "stove",
    1: "table center",
}
GROUP_BONUS = 1.0


def base_task_index() -> tuple[list[str], dict[int, int]]:
    """(base task names, LIBERO-Plus task id -> base task index)."""
    base = sorted(p.stem for p in BDDL.glob("*.bddl"))
    names = {
        r["id"] - 1: r["name"]
        for r in json.loads(CLASSIFICATION.read_text())["libero_spatial"]
    }
    mapping = {}
    for tid, name in names.items():
        hits = [b for b in base if name.startswith(b)]
        if hits:
            mapping[tid] = base.index(max(hits, key=len))
    return base, mapping


def success_counts() -> dict[int, tuple[int, int]]:
    """task id -> (sweeps solved, sweeps run)."""
    counts: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for epoch in SWEEP_EPOCHS:
        path = SWEEPS / f"{ARM}_ep{epoch}.jsonl"
        if not path.is_file():
            continue
        for line in open(path):
            row = json.loads(line)
            tid = int(row["task_id"])
            counts[tid][0] += float(row["success_once"]) > 0.5
            counts[tid][1] += 1
    return {t: tuple(v) for t, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ok", type=int, default=6, help="success candidates/axis")
    parser.add_argument("--n-fail", type=int, default=10, help="failure candidates/axis")
    args = parser.parse_args()

    base_names, base_of = base_task_index()
    counts = success_counts()
    # Trials per task, not sweeps: ep300 is the full 7-axis file and repeats 30
    # of its 2402 tasks, so a task can have 6 trials across 5 sweeps. Requiring
    # == the max silently emptied every bucket.
    n_sweeps = len([e for e in SWEEP_EPOCHS if (SWEEPS / f"{ARM}_ep{e}.jsonl").is_file()])
    per_axis = get_axis_task_ids("libero_spatial", sorted(AXIS_CATEGORIES), None, 1.0)

    def candidates(axis: str, base: int | None) -> tuple[list[int], list[int]]:
        """(most reliable successes, most reliable failures); base None = whole axis."""
        ids = [
            t for t in per_axis[axis]
            if (base is None or base_of.get(t) == base)
            and counts.get(t, (0, 0))[1] >= n_sweeps
            and t in base_of
        ]
        # Sort by how often it succeeded; ties by task id so the list is
        # reproducible from the repo rather than from a shuffle seed.
        ok = sorted(ids, key=lambda t: (-counts[t][0], t))
        fail = sorted(ids, key=lambda t: (counts[t][0], t))
        return ok, fail

    # Score a base by how many RELIABLE candidates it can supply, capped at what
    # the figure needs: a base with 40 certain successes and no failures is
    # worth less than one with 4 of each.
    def score(axis: str, base: int) -> float:
        ok, fail = candidates(axis, base)
        n_ok = sum(1 for t in ok if counts[t][0] == counts[t][1])
        # "Mostly fails", not "never succeeds". No base has a never-solved
        # lighting or layout task, so a strict count scored every base equally
        # and the scene bonus alone chose -- landing on bases whose weakest
        # candidate still succeeded 4 times in 5. Tasks solved at most a fifth
        # of the time are the real failure candidates on those axes.
        n_fail = sum(1 for t in fail if counts[t][0] <= 0.2 * counts[t][1])
        return min(n_ok, args.n_ok) + min(n_fail, args.n_fail)

    # No base pinning any more. Pinning one base per axis was worth it while the
    # goal was scenery, but the binding constraint is now "at least two clips of
    # each outcome per axis", and reliable failures simply do not exist on every
    # base -- pinning left textures with candidates that still succeed 80% of
    # the time. Drawing from the whole axis puts the genuinely weak tasks in
    # reach; the bases they belong to come along for free and are still varied,
    # and every base that appears gets its own IND twin.
    meta = []
    print(f"{'axis':<9} {'ok cand':<8} {'fail cand':<10} {'bases':<22} weakest fail cand")
    for axis in FIVE_AXES:
        ok, fail = candidates(axis, None)
        take_ok = ok[: args.n_ok]
        take_fail = [t for t in fail if t not in take_ok][: args.n_fail]
        # Report the weakest candidate as a rate: trials per task vary (ep300
        # repeats 30 of its tasks), so "6/5" is not a thing.
        weakest = max((counts[t][0] / counts[t][1] for t in take_fail), default=None)
        note = ""
        if weakest:
            note = f"  (weakest fail candidate still solved {weakest:.0%} of sweeps)"
        bases = sorted({base_of[t] for t in take_ok + take_fail})
        print(
            f"{axis:<9} {len(take_ok):<8} {len(take_fail):<10} "
            f"{str(bases):<22} {note.strip() or '-'}"
        )
        for t in take_ok + take_fail:
            meta.append({
                "task_id": t,
                "axis": axis,
                "base_task": base_of[t],
                "base_task_name": base_names[base_of[t]],
                "sweeps_solved": counts[t][0],
                "sweeps_run": counts[t][1],
                "expect": "ok" if t in take_ok else "fail",
            })

    chosen = sorted(m["task_id"] for m in meta)
    out = REPO / "vla_augm/results/video_task_selection.json"
    out.write_text(json.dumps(sorted(meta, key=lambda m: m["task_id"]), indent=2) + "\n")
    print(f"\ntotal candidates: {len(chosen)}  (recorded, then post-selected)")
    print("\n    task_id_filter:", json.dumps(chosen))
    print("    IND twins:", json.dumps(sorted({m['base_task'] for m in meta})))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
