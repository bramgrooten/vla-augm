#!/usr/bin/env python
"""Recompute the paper's OOD numbers from the per-episode eval JSONL.

The final-checkpoint evals (``vla_augm/scripts/cluster_a/eval/submit_ood_evals.sh``) write
one JSON line per episode with a ``task_id`` but *no* ``axis`` field -- unlike the
periodic evals written by ``EmbodiedRunner._dump_ood_task_results``, which
``vla_augm/utils/analyze_ood_eval.py`` consumes. This module maps ``task_id`` to its
LIBERO-Plus perturbation axis via ``task_classification.json`` and aggregates.

Aggregation matters. The two conventions differ by 1-2 points on the same file:

    micro   pool every episode, then take the mean
    macro   take each axis's success rate, then average those rates unweighted

They disagree because the axes have very unequal sizes (258 tasks for background
textures against 390 for language instructions on libero_spatial) and the largest
axes are the hardest. The axis sizes are an artifact of how LIBERO-Plus was built,
not a statement about which shifts matter, so the paper reports **macro** -- but
note that wandb's ``eval_ood/all/success_once`` is **micro**
(``embodied_runner.py``: ``merged[key][mask].mean()`` over episodes), so the two
are not interchangeable. Always say which one a number is.

The paper's headline OOD metric is the macro average over the five *visual* axes.
Robot initial states and language instructions are excluded: no image augmentation
can address either, so including them only dilutes the effect being measured.

Usage:
    # one file, per-axis breakdown
    python vla_augm/analyze/ood_table.py vla_augm/results/ood_full/none_ep300.jsonl

    # compare arms against a baseline (first file is the baseline)
    python vla_augm/analyze/ood_table.py vla_augm/results/ood_full/{none,critic_only_ovl_alpha05}_ep300.jsonl

    # eval noise floor: two sweeps of the SAME checkpoint
    python vla_augm/analyze/ood_table.py --floor \
        vla_augm/results/ood_gr00t/none_ep300.jsonl \
        vla_augm/results/ood_gr00t/noise_floor/none_ep300_run2.jsonl

    # emit the LaTeX body of the placement table
    python vla_augm/analyze/ood_table.py --latex ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict

# Short name -> the category string used in task_classification.json. Mirrors
# rlinf/envs/libero/libero_plus_axes.AXIS_CATEGORIES; duplicated rather than
# imported so this script runs on a laptop without the rlinf/LIBERO deps.
AXIS_CATEGORIES = {
    "textures": "Background Textures",
    "lighting": "Light Conditions",
    "layout": "Objects Layout",
    "camera": "Camera Viewpoints",
    "noise": "Sensor Noise",
    "robot_init": "Robot Initial States",
    "language": "Language Instructions",
}

# The five axes an image augmentation could plausibly affect.
VISUAL_AXES = ["textures", "lighting", "layout", "camera", "noise"]

_DEFAULT_CLASSIFICATION = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "utils",
    "task_classification.json",
)


def load_axis_map(suite: str, path: str | None = None) -> dict[int, str]:
    """Return ``{task_id: short axis name}`` for one suite.

    The JSON's ``id`` is 1-based; ``task_id`` in the eval records is the 0-based
    position in the suite's task list, hence ``id - 1`` (same convention as
    ``rlinf/envs/libero/libero_plus_axes._load_classification``).
    """
    path = path or _DEFAULT_CLASSIFICATION
    with open(path) as f:
        data = json.load(f)
    if suite not in data:
        raise SystemExit(f"suite {suite!r} not in {path}; have {sorted(data)}")
    short_of = {v: k for k, v in AXIS_CATEGORIES.items()}
    out = {}
    for task in data[suite]:
        category = str(task["category"])
        if category not in short_of:
            raise SystemExit(f"unknown category {category!r} in {path}")
        out[int(task["id"]) - 1] = short_of[category]
    return out


def load_episodes(path: str, axis_map: dict[int, str], metric: str) -> list[dict]:
    """Read one eval JSONL, attaching each episode's axis.

    Records carrying their own ``axis`` (the periodic training-time dumps) keep
    it; the rest are labelled from ``task_id``. An episode whose task is not in
    the classification is dropped, matching what the runner does when it builds
    the ``all`` mask -- but it is reported, because silently dropping episodes
    is exactly how two "the same" numbers end up differing.
    """
    episodes, unknown = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if metric not in record:
                continue
            axis = record.get("axis") or axis_map.get(int(record["task_id"]))
            if axis is None or axis == "unknown":
                unknown += 1
                continue
            episodes.append({"axis": axis, "value": float(record[metric])})
    if unknown:
        print(f"  note: {unknown} episodes in {path} have no known axis; dropped")
    if not episodes:
        raise SystemExit(f"no usable episodes in {path}")
    return episodes


# Axes re-evaluated after the LIBERO-Plus asset fix, and the directory holding
# each rerun. The original sweeps ran against an environment where these axes did
# not render as intended, so where a rerun exists it supersedes the original --
# but only for its own axes; camera and noise were unaffected and are still read
# from the original file.
RERUN_AXES = {
    "ood_textures_lighting_rerun": ("textures", "lighting"),
    "ood_layout_rerun": ("layout",),
}

# Original eval directory -> the model prefix the rerun files use.
RERUN_MODELS = {"ood_full": "pi05", "ood_gr00t": "gr00t"}


def rerun_stem(relpath: str) -> str | None:
    """Rerun filename stem for an original eval path, or None if it has none.

    ``ood_full/none_ep300.jsonl`` -> ``pi05_none_ep300``. The SFT checkpoints are
    the one irregular pair: they are ``base.jsonl`` originally but carry their
    epoch in the rerun, as ``<model>_base_ep0``.
    """
    directory, name = os.path.split(relpath)
    model = RERUN_MODELS.get(directory)
    if model is None or not name.endswith(".jsonl"):
        return None
    stem = name[: -len(".jsonl")]
    if stem == "base":
        stem = "base_ep0"
    return f"{model}_{stem}"


def load_episodes_with_reruns(
    relpath: str, axis_map: dict[int, str], metric: str, root: str
) -> tuple[list[dict], set[str]]:
    """Episodes for one arm with re-evaluated axes swapped in.

    Returns ``(episodes, replaced)``. ``replaced`` is empty when no rerun file
    exists, which is the honest outcome for the two GR00T arms that collapsed and
    were never re-evaluated -- their rows keep the original numbers rather than
    silently mixing two evaluation environments within a single axis.
    """
    episodes = load_episodes(os.path.join(root, relpath), axis_map, metric)
    stem = rerun_stem(relpath)
    if stem is None:
        return episodes, set()

    replacement, replaced = [], set()
    for directory, axes in RERUN_AXES.items():
        path = os.path.join(root, directory, stem + ".jsonl")
        if not os.path.isfile(path):
            continue
        fresh = [e for e in load_episodes(path, axis_map, metric) if e["axis"] in axes]
        if not fresh:
            continue
        replacement += fresh
        replaced.update({e["axis"] for e in fresh})

    kept = [e for e in episodes if e["axis"] not in replaced]
    return kept + replacement, replaced


def _mean_and_se(values: list[float]) -> tuple[float, float, int]:
    """Mean, binomial standard error, and count.

    At p=0 or p=1 the naive SE is exactly zero, which would claim certainty from
    a floor/ceiling axis; fall back to the rule of three (95% bound 3/n), as
    vla_augm/utils/analyze_ood_eval.py does.
    """
    n = len(values)
    mean = sum(values) / n
    se = math.sqrt(mean * (1.0 - mean) / n) if 0.0 <= mean <= 1.0 else 0.0
    if se == 0.0 and 0.0 <= mean <= 1.0:
        se = (3.0 / n) / 1.96
    return mean, se, n


def summarise(episodes: list[dict], axes: list[str]) -> dict:
    """Per-axis rates plus the micro and macro aggregates over ``axes``."""
    by_axis = defaultdict(list)
    for episode in episodes:
        by_axis[episode["axis"]].append(episode["value"])

    per_axis = {}
    for axis in axes:
        if by_axis.get(axis):
            per_axis[axis] = _mean_and_se(by_axis[axis])

    pooled = [v for axis in axes for v in by_axis.get(axis, [])]
    micro, micro_se, n = _mean_and_se(pooled)
    macro = statistics.mean(m for m, _, _ in per_axis.values())
    # Each axis contributes 1/k of the macro average, so its SE does too.
    macro_se = math.sqrt(sum(se**2 for _, se, _ in per_axis.values())) / len(per_axis)
    return {
        "per_axis": per_axis,
        "micro": micro,
        "micro_se": micro_se,
        "macro": macro,
        "macro_se": macro_se,
        "n": n,
    }


def print_summary(label: str, s: dict, axes: list[str]) -> None:
    cells = " ".join(
        f"{axis}={s['per_axis'][axis][0] * 100:5.1f}" if axis in s["per_axis"] else f"{axis}=  -- "
        for axis in axes
    )
    print(
        f"{label:<40s} macro={s['macro'] * 100:5.2f}  micro={s['micro'] * 100:5.2f}"
        f"  (n={s['n']:5d})   {cells}"
    )


def print_delta(label_a: str, sa: dict, label_b: str, sb: dict, axes: list[str]) -> None:
    """Per-axis and aggregate difference, in points, with a combined SE."""
    print(f"\n  {label_b} minus {label_a}:")
    for axis in axes:
        if axis not in sa["per_axis"] or axis not in sb["per_axis"]:
            continue
        (ma, sea, _), (mb, seb, _) = sa["per_axis"][axis], sb["per_axis"][axis]
        diff, se = (mb - ma) * 100, math.hypot(sea, seb) * 100
        sigma = f"{abs(diff) / se:4.1f} sigma" if se > 0 else "     n/a"
        print(f"    {axis:<10s} {diff:+6.1f} +- {se:4.1f} pts  ({sigma})")
    diff = (sb["macro"] - sa["macro"]) * 100
    se = math.hypot(sa["macro_se"], sb["macro_se"]) * 100
    print(f"    {'MACRO':<10s} {diff:+6.1f} +- {se:4.1f} pts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="eval JSONL files; the first is the baseline")
    ap.add_argument("--metric", default="success_once")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--classification", default=None, help=f"default: {_DEFAULT_CLASSIFICATION}")
    ap.add_argument("--axes", default=",".join(VISUAL_AXES), help="comma-separated, or 'all'")
    ap.add_argument("--labels", default=None, help="comma-separated display names")
    ap.add_argument(
        "--floor",
        action="store_true",
        help="treat the files as repeat evals of ONE checkpoint and report the spread "
        "as the evaluation noise floor",
    )
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    args = ap.parse_args()

    axes = list(AXIS_CATEGORIES) if args.axes == "all" else args.axes.split(",")
    for axis in axes:
        if axis not in AXIS_CATEGORIES:
            raise SystemExit(f"unknown axis {axis!r}; have {sorted(AXIS_CATEGORIES)}")

    labels = args.labels.split(",") if args.labels else [
        os.path.basename(f).replace(".jsonl", "") for f in args.files
    ]
    if len(labels) != len(args.files):
        raise SystemExit("--labels count must match the number of files")

    axis_map = load_axis_map(args.suite, args.classification)
    print(f"suite={args.suite}  metric={args.metric}  axes={','.join(axes)}\n")

    summaries = []
    for path, label in zip(args.files, labels):
        s = summarise(load_episodes(path, axis_map, args.metric), axes)
        print_summary(label, s, axes)
        summaries.append((label, s))

    if args.floor:
        if len(summaries) < 2:
            raise SystemExit("--floor needs at least two evals of the same checkpoint")
        macros = [s["macro"] for _, s in summaries]
        spread = (max(macros) - min(macros)) * 100
        print(
            f"\nevaluation noise floor (macro, {len(macros)} sweeps of one checkpoint): "
            f"{spread:.2f} pts"
        )
        print("An arm-vs-arm difference smaller than this should not be believed.")
        return

    if len(summaries) > 1:
        (label_a, sa) = summaries[0]
        for label_b, sb in summaries[1:]:
            print_delta(label_a, sa, label_b, sb, axes)

    if args.latex:
        print("\n% LaTeX rows (OOD column only; Train/IND come from wandb)")
        for label, s in summaries:
            print(f"{label} & & & & {s['macro'] * 100:.1f} \\\\")


if __name__ == "__main__":
    main()
