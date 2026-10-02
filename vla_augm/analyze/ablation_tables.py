#!/usr/bin/env python
"""Regenerate Tables 2 and 3 (Sections 4.2 and 4.3: augmentation type and strength).

Same contract as placement_table.py: OOD is recomputed here from the per-episode
JSONLs (macro over the five visual axes), IND is transcribed from wandb via
vla_augm/results/experiment_overview.tsv, and --latex emits the table body so the
paper never carries a hand-typed number.

    python vla_augm/analyze/ablation_tables.py            # both tables, as text
    python vla_augm/analyze/ablation_tables.py --latex    # LaTeX bodies
    python vla_augm/analyze/ablation_tables.py --check    # recomputed vs the TSV

Every row holds placement at critic-only and epoch at 300, so these tables vary
exactly one thing each: which augmentation, and how strong. The `none` baseline
is repeated at the top of both so each table reads on its own.

IND is not recomputed. Only the two SFT checkpoints have IND eval JSONLs on disk;
for a trained arm the in-distribution number exists only in wandb, so it is read
from the overview TSV -- which is also what --check compares OOD against, since
the TSV records an independently entered OOD value per arm.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from ood_table import (VISUAL_AXES, load_axis_map, load_episodes,
                       load_episodes_with_reruns, summarise)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AXIS_HEADS = [("Tex.", "textures"), ("Light", "lighting"), ("Layout", "layout"),
              ("Cam.", "camera"), ("Noise", "noise")]
MODEL_NAMES = {"pi05": r"$\pi_{0.5}$", "gr00t": "GR00T N1.5"}
MODEL_CITES = {"pi05": "black2025pi05", "gr00t": "bjorck2025groot"}
DIRS = {"pi05": "ood_full", "gr00t": "ood_gr00t"}

# label, results file (relative to the model's dir), TSV run_name.
# None means the run does not exist for that model -- GR00T's strength sweep
# covers three alphas, not five, so those rows print as dashes rather than
# silently shrinking the table and making the two models look comparable.
TYPE_TABLE = {
    "pi05": [
        ("no augmentation", "none_ep300.jsonl", "augm_none-seed52"),
        ("overlay ($\\alpha = 0.5$)", "critic_only_ovl_alpha05_ep300.jsonl", "augm_critic_only_ovl_alpha05-seed52"),
        ("shift ($10$ px)", "critic_only_shift_ep300.jsonl", "augm_critic_only_shift-seed52"),
        ("overlay + shift", "critic_only_ovlshift_alpha05_ep300.jsonl", "augm_critic_only_ovlshift_alpha05-seed52"),
    ],
    "gr00t": [
        ("no augmentation", "none_ep300.jsonl", "augm_none-seed52"),
        ("overlay ($\\alpha = 0.5$)", "critic_only_ovl_alpha05_ep300.jsonl", "augm_critic_only_ovl_alpha05-seed52"),
        ("shift ($10$ px)", "critic_only_shift_ep300.jsonl", "augm_critic_only_shift-seed52"),
        ("overlay + shift", "critic_only_ovlshift_alpha05_ep300.jsonl", "augm_critic_only_ovlshift_alpha05-seed52"),
    ],
}

STRENGTH_TABLE = {
    "pi05": [
        ("no augmentation", "none_ep300.jsonl", "augm_none-seed52"),
        ("$\\alpha = 0.10$", "critic_only_ovl_alpha01_ep300.jsonl", "augm_critic_only_ovl_alpha01-seed52"),
        ("$\\alpha = 0.25$", "critic_only_ovl_alpha025_ep300.jsonl", "augm_critic_only_ovl_alpha025-seed52"),
        ("$\\alpha = 0.50$", "critic_only_ovl_alpha05_ep300.jsonl", "augm_critic_only_ovl_alpha05-seed52"),
        ("$\\alpha = 0.75$", "critic_only_ovl_alpha075_ep300.jsonl", "augm_critic_only_ovl_alpha075-seed52"),
        ("$\\alpha = 0.90$", "critic_only_ovl_alpha09_ep300.jsonl", "augm_critic_only_ovl_alpha09-seed52"),
    ],
    "gr00t": [
        ("no augmentation", "none_ep300.jsonl", "augm_none-seed52"),
        ("$\\alpha = 0.10$", None, None),
        ("$\\alpha = 0.25$", "critic_only_ovl_alpha025_ep300.jsonl", "augm_critic_only_ovl_alpha025-seed52"),
        ("$\\alpha = 0.50$", "critic_only_ovl_alpha05_ep300.jsonl", "augm_critic_only_ovl_alpha05-seed52"),
        ("$\\alpha = 0.75$", "critic_only_ovl_alpha075_ep300.jsonl", "augm_critic_only_ovl_alpha075-seed52"),
        ("$\\alpha = 0.90$", None, None),
    ],
}

MODEL_KEY = {"pi0.5": "pi05", "GR00T": "gr00t"}


def load_tsv() -> dict[tuple[str, str], dict]:
    """(model, run_name) -> row of vla_augm/results/experiment_overview.tsv."""
    path = os.path.join(REPO, "vla_augm", "results", "experiment_overview.tsv")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["run_name"] and row["model"] in MODEL_KEY:
                out[(MODEL_KEY[row["model"]], row["run_name"])] = row
    return out


def pct(value: str) -> float | None:
    """A TSV cell as a percentage; 'n/a' and blanks become None."""
    value = (value or "").strip()
    if not value or value == "n/a":
        return None
    return float(value) * 100


RESULTS = os.path.join(REPO, "vla_augm", "results")


def stats_of(model: str, fname: str, axis_map) -> dict:
    path = os.path.join(RESULTS, DIRS[model], fname)
    return summarise(load_episodes(path, axis_map, "success_once"), VISUAL_AXES)


def stats_pair(model: str, fname: str, axis_map) -> tuple[dict, dict]:
    """``(original, corrected)`` for one run; the same object where no rerun exists."""
    original = stats_of(model, fname, axis_map)
    episodes, replaced = load_episodes_with_reruns(
        os.path.join(DIRS[model], fname), axis_map, "success_once", RESULTS
    )
    return (original, original) if not replaced else (
        original, summarise(episodes, VISUAL_AXES))


def build(spec, axis_map, tsv):
    """-> {model: [(label, ind, axes, macro, was_axes, was_macro), ...]}.

    ``was_*`` carry the pre-rerun values so the table can show what the
    re-evaluation changed; they equal the current ones where nothing moved.
    None appears wherever a run is missing.
    """
    out = {}
    for model, rows in spec.items():
        built = []
        for label, fname, run in rows:
            if fname is None:
                built.append((label, None, [None] * len(AXIS_HEADS), None,
                              [None] * len(AXIS_HEADS), None))
                continue
            old_s, s = stats_pair(model, fname, axis_map)
            ind = pct(tsv[(model, run)]["IND eval"]) if (model, run) in tsv else None
            axes = [s["per_axis"][k][0] * 100 for _, k in AXIS_HEADS]
            was = [old_s["per_axis"][k][0] * 100 for _, k in AXIS_HEADS]
            built.append((label, ind, axes, s["macro"] * 100,
                          was, old_s["macro"] * 100))
        out[model] = built
    return out


def tex_num(value, best=False) -> str:
    if value is None:
        return "--"
    text = f"{value:.1f}"
    if best:
        text = rf"\textbf{{{text}}}"
    return (r"\phantom{0}" + text) if value < 10 else text


def emit_text(title, table):
    heads = "".join(f"{h:>7s}" for h, _ in AXIS_HEADS)
    print(f"\n=== {title} ===")
    for model, rows in table.items():
        print(f"\n{model}")
        print(f"{'':<26}{'IND':>7}{heads}{'AVG':>8}")
        for label, ind, axes, macro, was, was_macro in rows:
            plain = label.replace("$", "").replace("\\alpha", "alpha").replace("\\", "")
            cells = "".join(f"{v:7.1f}" if v is not None else f"{'--':>7}" for v in axes)
            ind_s = "--" if ind is None else f"{ind:.1f}"
            mac_s = "--" if macro is None else f"{macro:.1f}"
            print(f"{plain:<26}{ind_s:>7}{cells}{mac_s:>8}")
            if macro is not None and round(macro, 1) != round(was_macro, 1):
                old_cells = "".join(f"{v:7.1f}" for v in was)
                print(f"{'  (superseded)':<26}{'':>7}{old_cells}{was_macro:8.1f}")


def emit_latex(table):
    for block, (model, rows) in enumerate(table.items()):
        series = [[ind] + axes + [macro] for _l, ind, axes, macro, _w, _wm in rows]
        best = [max((round(v, 1) for v in col if v is not None), default=None)
                for col in zip(*series)]
        print(f"\\multicolumn{{8}}{{l}}{{\\emph{{{MODEL_NAMES[model]}}}"
              f"~\\citep{{{MODEL_CITES[model]}}}}} \\\\")
        for (label, *_), values in zip(rows, series):
            cells = " & ".join(
                tex_num(v, v is not None and b is not None and round(v, 1) == b)
                for v, b in zip(values, best))
            print(f"{label:<26s} & {cells} \\\\")
        if block < len(table) - 1:
            print(r"\midrule")


def check(spec, axis_map, tsv) -> int:
    """Recomputed macro OOD against the value hand-entered in the TSV."""
    bad = 0
    for model, rows in spec.items():
        for _label, fname, run in rows:
            if fname is None:
                continue
            macro = stats_of(model, fname, axis_map)["macro"] * 100
            recorded = pct(tsv[(model, run)]["OOD 5axes @300"])
            if recorded is None:
                print(f"  {model:<7}{run:<45}recomputed {macro:5.1f}  TSV n/a")
                continue
            flag = "" if abs(macro - recorded) < 0.05 else "   <-- MISMATCH"
            bad += bool(flag)
            print(f"  {model:<7}{run:<45}recomputed {macro:5.1f}  TSV {recorded:5.1f}{flag}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    axis_map = load_axis_map("libero_spatial")
    tsv = load_tsv()

    if args.check:
        print("Table 2 (type):")
        bad = check(TYPE_TABLE, axis_map, tsv)
        print("Table 3 (strength):")
        bad += check(STRENGTH_TABLE, axis_map, tsv)
        raise SystemExit(1 if bad else 0)

    type_t = build(TYPE_TABLE, axis_map, tsv)
    strength_t = build(STRENGTH_TABLE, axis_map, tsv)

    # Runs the re-evaluation has not covered. Their rows still carry the original
    # numbers, so within a block they are not comparable to the corrected rows --
    # worth stating outright rather than leaving to be noticed.
    missing = []
    for name, spec in (("type", TYPE_TABLE), ("strength", STRENGTH_TABLE)):
        for model, rows in spec.items():
            for label, fname, _run in rows:
                if fname is None:
                    continue
                _o, _n = stats_pair(model, fname, axis_map)
                if _o is _n:
                    missing.append(f"{name:9s} {model:6s} {label}")
    if missing:
        # stderr, so `--latex > table.tex` stays a pastable table.
        print("\nNOT re-evaluated (original numbers stand; not comparable to the "
              "corrected rows in the same block):", file=sys.stderr)
        for line in sorted(set(missing)):
            print(f"  {line}", file=sys.stderr)
    if args.latex:
        print("% ---- Table 2: augmentation type ----")
        emit_latex(type_t)
        print("\n% ---- Table 3: augmentation strength ----")
        emit_latex(strength_t)
    else:
        emit_text("Table 2: which augmentation type", type_t)
        emit_text("Table 3: how strong", strength_t)


if __name__ == "__main__":
    main()
