#!/usr/bin/env python
"""Summarise periodic LIBERO-Plus OOD eval results.

Reads the per-episode JSONL that ``EmbodiedRunner`` writes for every OOD eval
point (``<log_path>/ood_eval/step_<N>.jsonl``) and prints per-axis success
rates with binomial standard errors, either as a training curve or as a
single-step table.

Usage:
    python vla_augm/utils/analyze_ood_eval.py logs/<run>/ood_eval
    python vla_augm/utils/analyze_ood_eval.py logs/<run>/ood_eval --metric success_at_end
    python vla_augm/utils/analyze_ood_eval.py logs/<run>/ood_eval --step 40 --per-task
    python vla_augm/utils/analyze_ood_eval.py runA/ood_eval runB/ood_eval --labels SFT,RL

Standard error is sqrt(p(1-p)/n); a 10-point difference between two arms needs
SE around 3.5 points each, i.e. roughly 200+ episodes per axis.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load_dir(path: str) -> dict[int, list[dict]]:
    """Return {step: [episode records]} for one ood_eval directory."""
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(
            glob.glob(os.path.join(path, "step_*.jsonl")),
            key=lambda p: int(os.path.basename(p)[5:-6]),
        )
    if not files:
        raise SystemExit(f"No step_*.jsonl found under {path}")

    by_step: dict[int, list[dict]] = defaultdict(list)
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                by_step[int(record.get("step", -1))].append(record)
    return dict(by_step)


def rate(records: list[dict], metric: str) -> tuple[float, float, int]:
    """Mean, standard error, and count for one metric over episodes.

    At p=0 or p=1 the naive binomial SE is exactly zero, which would claim
    certainty from a floor/ceiling axis. Fall back to the rule of three there:
    the 95% bound is 3/n, so treat 3/n/1.96 as the equivalent SE.
    """
    values = [float(r[metric]) for r in records if metric in r]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(values) / n
    if not 0.0 <= mean <= 1.0:
        return mean, 0.0, n
    se = math.sqrt(mean * (1.0 - mean) / n)
    if se == 0.0:
        se = (3.0 / n) / 1.96
    return mean, se, n


def axes_of(records: list[dict]) -> list[str]:
    seen = {r.get("axis", "unknown") for r in records}
    ordered = [a for a in ("textures", "lighting", "layout") if a in seen]
    return ordered + sorted(seen - set(ordered))


def print_curve(by_step: dict[int, list[dict]], metric: str, label: str) -> None:
    steps = sorted(by_step)
    axes = axes_of(by_step[steps[0]])
    header = f"{'step':>6} " + " ".join(f"{a:>16}" for a in axes) + f" {'all':>16}"
    print(f"\n=== {label}  metric={metric}")
    print(header)
    print("-" * len(header))
    for step in steps:
        records = by_step[step]
        cells = []
        for axis in axes:
            mean, se, n = rate([r for r in records if r.get("axis") == axis], metric)
            cells.append(f"{mean * 100:6.1f} ±{se * 100:4.1f}" if n else f"{'-':>12}")
        known = [r for r in records if r.get("axis", "unknown") != "unknown"]
        mean, se, n = rate(known, metric)
        cells.append(f"{mean * 100:6.1f} ±{se * 100:4.1f}" if n else f"{'-':>12}")
        print(f"{step:>6} " + " ".join(f"{c:>16}" for c in cells))
    counts = {a: sum(1 for r in by_step[steps[-1]] if r.get("axis") == a) for a in axes}
    print(f"episodes per axis (last step): {counts}")


def print_per_task(records: list[dict], metric: str, top: int) -> None:
    per_task: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        if metric in r:
            per_task[(r.get("axis", "unknown"), int(r["task_id"]))].append(
                float(r[metric])
            )
    rows = [
        (axis, task_id, sum(v) / len(v), len(v))
        for (axis, task_id), v in per_task.items()
    ]
    rows.sort(key=lambda x: x[2])
    print(f"\n--- hardest {top} tasks by {metric}")
    for axis, task_id, mean, n in rows[:top]:
        print(f"  {axis:<10} task {task_id:<6} {mean * 100:5.1f}%  (n={n})")
    print(f"--- easiest {top}")
    for axis, task_id, mean, n in rows[-top:][::-1]:
        print(f"  {axis:<10} task {task_id:<6} {mean * 100:5.1f}%  (n={n})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="one or more ood_eval directories")
    ap.add_argument("--metric", default="success_once")
    ap.add_argument("--labels", default=None, help="comma-separated names for dirs")
    ap.add_argument("--step", type=int, default=None, help="only this step")
    ap.add_argument("--per-task", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else args.dirs
    if len(labels) != len(args.dirs):
        raise SystemExit("--labels count must match number of directories")

    loaded = []
    for path, label in zip(args.dirs, labels):
        by_step = load_dir(path)
        if args.step is not None:
            if args.step not in by_step:
                raise SystemExit(f"step {args.step} not in {path}: {sorted(by_step)}")
            by_step = {args.step: by_step[args.step]}
        print_curve(by_step, args.metric, label)
        loaded.append((label, by_step))
        if args.per_task:
            last = by_step[max(by_step)]
            print_per_task(last, args.metric, args.top)

    if len(loaded) == 2:
        (label_a, a), (label_b, b) = loaded
        step_a, step_b = max(a), max(b)
        print(f"\n=== {label_b} minus {label_a}  (steps {step_b} vs {step_a})")
        for axis in axes_of(a[step_a]):
            ma, sa, na = rate([r for r in a[step_a] if r.get("axis") == axis], args.metric)
            mb, sb, nb = rate([r for r in b[step_b] if r.get("axis") == axis], args.metric)
            if not na or not nb:
                continue
            diff = (mb - ma) * 100
            se = math.sqrt(sa**2 + sb**2) * 100
            sigma = f"{abs(diff) / se:.1f} sigma" if se > 0 else "n/a"
            floor = " [both at floor]" if ma == 0.0 and mb == 0.0 else ""
            print(f"  {axis:<10} {diff:+6.1f} ± {se:4.1f} pts  ({sigma}){floor}")


if __name__ == "__main__":
    main()
