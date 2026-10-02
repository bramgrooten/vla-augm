#!/usr/bin/env python
"""Regenerate Table 1 of the paper (Section 4.1, augmentation placement).

Prints the table as text and, with --latex, as the LaTeX body that lives in
``paper/vla_augm/main.tex``. Run it after any re-evaluation so the paper's
numbers are never hand-copied.

    python vla_augm/analyze/placement_table.py
    python vla_augm/analyze/placement_table.py --latex

OOD is recomputed here from the per-episode JSONL (macro average over the five
visual axes; see ood_table.py for why macro and not micro). IND is recomputed the
same way for any arm that has an IND JSONL under vla_augm/results/ind/; the rest are
transcribed from wandb, which is the only place the in-distribution eval curve is
stored for a training run. Give an arm a path instead of a number and it moves
onto the recompute path -- prefer that, and update the JSONL rather than the
number when an arm is re-run.
"""

from __future__ import annotations

import argparse
import json
import os

from ood_table import (VISUAL_AXES, load_axis_map, load_episodes,
                       load_episodes_with_reruns, summarise)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# placement label, OOD results file, VLM prefix passes per training step, IND, the
# wandb run id, footnote. IND is either a float read off wandb at the arm's final
# epoch, or a path under vla_augm/results/ind/ that is recomputed here; None prints
# "--". "dagger" arms collapsed and were stopped early, so their row reports the
# epoch-100 checkpoint. IND is 500 episodes = 10 LIBERO-Spatial tasks x 50 fixed
# initial states, at 240 max steps, success_once -- the same protocol whether the
# number came from wandb or from a JSONL.
ARMS = {
    "pi05": [
        ("SFT checkpoint (no RL)", "ood_full/base.jsonl", None, "ind/pi05_sft_ep0.jsonl", "-", ""),
        ("no augm. (baseline)", "ood_full/none_ep300.jsonl", 1, 93.2, "i2h16jg0", ""),
        ("actor + critic", "ood_full/uniform_ovl_alpha05_ep300.jsonl", 1, 0.8, "g8c75o3g", ""),
        ("actor-only", "ood_full/actor_only_ovl_alpha05_ep100.jsonl", 2, 11.0, "1a9ftkl1", "dagger"),
        ("critic-only", "ood_full/critic_only_ovl_alpha05_ep300.jsonl", 2, 98.0, "np9cfrv3", ""),
    ],
    "gr00t": [
        ("SFT checkpoint (no RL)", "ood_gr00t/base.jsonl", None, "ind/gr00t_sft_ep0.jsonl", "-", ""),
        ("no augm. (baseline)", "ood_gr00t/none_ep300.jsonl", 1, 95.4, "5v06nwqx", ""),
        ("actor + critic", "ood_gr00t/uniform_ovl_alpha05_ep100.jsonl", 1, 0.2, "9racd1gs", "dagger"),
        ("actor-only", "ood_gr00t/actor_only_ovl_alpha05_ep100.jsonl", 2, 0.0, "bidtbb4q", "dagger"),
        ("critic-only", "ood_gr00t/critic_only_ovl_alpha05_ep300.jsonl", 2, 97.4, "zmkh2lop", ""),
    ],
}

# N1.5, not N1: model_type "gr00t" dispatches to gr00t_n1d5, and the Cluster B
# scripts that produced these runs all say N1.5 (launch_gr00t_augm_grid.sh:2).
MODEL_NAMES = {"pi05": r"$\pi_{0.5}$", "gr00t": "GR00T N1.5"}

# Cite keys from paper/vla_augm/citations.bib, placed after the \emph{} so the
# citation itself is not italicised.
MODEL_CITES = {"pi05": "black2025pi05", "gr00t": "bjorck2025groot"}

# Repeat evals of one checkpoint, for the evaluation noise floor.
FLOOR_PAIRS = {
    "pi05": ("ood_full/base.jsonl", "ood_full/base_bigmem.jsonl"),
    "gr00t": ("ood_gr00t/none_ep300.jsonl", "ood_gr00t/noise_floor/none_ep300_run2.jsonl"),
}


def axis_task_counts(axis_map: dict[int, str]) -> dict[str, int]:
    """Tasks per axis in the suite. Each is evaluated once, so this is also the
    per-axis episode count of a full-suite sweep."""
    counts = dict.fromkeys(VISUAL_AXES, 0)
    for axis in axis_map.values():
        if axis in counts:
            counts[axis] += 1
    return counts


RESULTS = os.path.join(REPO, "vla_augm", "results")


def stats_of(relpath: str, axis_map: dict[int, str]) -> dict:
    path = os.path.join(RESULTS, relpath)
    return summarise(load_episodes(path, axis_map, "success_once"), VISUAL_AXES)


def stats_pair(relpath: str, axis_map: dict[int, str]) -> tuple[dict, dict, set[str]]:
    """``(original, corrected, replaced axes)`` for one arm.

    Both are computed so the table can show what the re-evaluation changed. When
    no rerun exists the two are the same object and ``replaced`` is empty.
    """
    original = stats_of(relpath, axis_map)
    episodes, replaced = load_episodes_with_reruns(
        relpath, axis_map, "success_once", RESULTS
    )
    if not replaced:
        return original, original, replaced
    return original, summarise(episodes, VISUAL_AXES), replaced


def ind_success(relpath: str) -> float:
    """Success rate over an IND eval JSONL, as a percentage.

    IND has no axis structure -- it is one flat pool of 500 episodes (10 tasks x
    50 fixed initial states) -- so this is a plain mean of success_once rather
    than the macro average OOD needs.
    """
    path = os.path.join(REPO, "vla_augm", "results", relpath)
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) != IND_EPISODES:
        raise ValueError(
            f"{relpath}: expected {IND_EPISODES} IND episodes, found {len(rows)}. "
            "The fixed-reset pool is 10 tasks x 50 states; a different count means "
            "the eval did not sweep it exactly once."
        )
    return sum(r["success_once"] for r in rows) / len(rows) * 100


def resolve_ind(value: float | str | None) -> float | None:
    """A float passes through; a JSONL path is recomputed; None stays None."""
    return ind_success(value) if isinstance(value, str) else value


def fmt(value: float | int | None) -> str:
    if value is None:
        return "--"
    return str(value) if isinstance(value, int) else f"{value:.1f}"


def tex_num(value: float | None, best: bool = False) -> str:
    """One LaTeX number cell.

    ``\\phantom{0}`` keeps single-digit rates aligned under two-digit ones (the
    collapsed arms are all below 10), and ``\\textbf`` marks the best value in the
    column. Emitting both here is what keeps the generated body identical to the
    one in the paper -- hand-adding them to main.tex is how the two drifted apart
    the first time.
    """
    if value is None:
        return "--"
    text = f"{value:.1f}"
    if best:
        text = rf"\textbf{{{text}}}"
    return (r"\phantom{0}" + text) if value < 10 else text


# Column header -> axis key, in table order.
AXIS_HEADS = [("Tex.", "textures"), ("Light", "lighting"), ("Layout", "layout"),
              ("Cam.", "camera"), ("Noise", "noise")]

# The count row reports *tasks*, so it means the same thing in every column: the
# OOD columns evaluate one episode per task, and IND is 10 LIBERO-Spatial tasks
# (x 50 fixed initial states = 500 episodes per arm, which the caption states).
IND_TASKS = 10
IND_EPISODES = 500  # 10 x 50; ind_success() checks a JSONL sweeps exactly this


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    axis_map = load_axis_map("libero_spatial")
    counts = axis_task_counts(axis_map)

    rows = {}
    for model, arms in ARMS.items():
        rows[model] = [
            (label, passes, resolve_ind(ind)) + stats_pair(path, axis_map) + (mark,)
            for label, path, passes, ind, _run, mark in arms
        ]

    never_rerun = [(model, label)
                   for model, table in rows.items()
                   for label, _p, _i, _o, _n, replaced, _m in table if not replaced]

    floors = {}
    for model, (a, b) in FLOOR_PAIRS.items():
        floors[model] = abs(stats_of(a, axis_map)["macro"] - stats_of(b, axis_map)["macro"]) * 100

    if not args.latex:
        heads = "".join(f"{h:>7s}" for h, _ in AXIS_HEADS)
        for model, table in rows.items():
            print(f"\n=== {model} ===")
            print(f"{'placement':<26s} {'passes':>6s} {'IND':>7s}{heads}{'AVG':>8s}")
            print(f"{'# tasks':<26s} {'':>6s} {IND_TASKS:>7d}"
                  + "".join(f"{counts[k]:>7d}" for _, k in AXIS_HEADS)
                  + f"{sum(counts.values()):>8d}")
            for label, passes, ind, old_s, s, _replaced, mark in table:
                cells = "".join(f"{s['per_axis'][k][0] * 100:7.1f}" for _, k in AXIS_HEADS)
                star = "  (dagger: epoch 100)" if mark else ""
                print(f"{label:<26s} {fmt(passes):>6s} {fmt(ind):>7s}{cells}"
                      f"{s['macro'] * 100:8.1f}{star}")
                if old_s is not s:
                    was = "".join(f"{old_s['per_axis'][k][0] * 100:7.1f}"
                                  for _, k in AXIS_HEADS)
                    print(f"{'  (superseded)':<26s} {'':>6s} {'':>7s}{was}"
                          f"{old_s['macro'] * 100:8.1f}")
            print(f"evaluation noise floor: {floors[model]:.2f} pts")
        if never_rerun:
            print("\nNOT re-evaluated (original numbers stand):")
            for model, label in never_rerun:
                print(f"  {model:6s} {label}")
        return

    print(r"\emph{\# tasks} & & " + str(IND_TASKS) + " & "
          + " & ".join(str(counts[k]) for _, k in AXIS_HEADS)
          + " & " + str(sum(counts.values())) + r" \\")
    print(r"\midrule")
    for block, (model, table) in enumerate(rows.items()):
        # IND, the five axes, then Avg -- the numeric columns, in table order.
        def numbers(stats, ind):
            return ([ind] + [stats["per_axis"][k][0] * 100 for _, k in AXIS_HEADS]
                    + [stats["macro"] * 100])

        series = [numbers(s, ind) for _l, _p, ind, _o, s, _r, _m in table]
        # Best per column, compared on the displayed (1-decimal) value so two rows
        # that print the same number are either both bold or neither. The SFT row
        # competes but never wins.
        best = [max((round(v, 1) for v in col if v is not None), default=None)
                for col in zip(*series)]

        print(f"\\multicolumn{{9}}{{l}}{{\\emph{{{MODEL_NAMES[model]}}}"
              f"~\\citep{{{MODEL_CITES[model]}}}}} \\\\")
        for (label, passes, _ind, _o, _s, _r, mark), values in zip(table, series):
            name = label + ("$^\\dagger$" if mark else "")
            cells = " & ".join(
                tex_num(v, v is not None and b is not None and round(v, 1) == b)
                for v, b in zip(values, best)
            )
            print(f"{name:<34s} & {fmt(passes)} & {cells} \\\\")
        if block < len(rows) - 1:
            print(r"\midrule")


if __name__ == "__main__":
    main()
