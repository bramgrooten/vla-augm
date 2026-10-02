#!/usr/bin/env python3
"""Training-dynamics and trade-off figures, as pgfplots bodies for \\input.

Complements `make_augmentation_figures.py axiscurves`, which plots OOD success
per perturbation axis for three runs. Everything here uses data the paper does
not otherwise show:

  dynamics   How training itself behaves under each placement -- reward, IND
             success, policy loss, value loss, entropy and approximate KL. This
             is what turns "augmenting the actor collapses training" from an
             endpoint number into a visible failure.
  oodcurves  Macro OOD success against epoch for every pi0.5 configuration,
             split into augmentation type and overlay strength.
  tradeoff   IND against OOD for every configuration of both models, which is
             the direct evidence that robustness is not bought with
             in-distribution success.

    python vla_augm/analyze/make_training_figures.py all
    python vla_augm/analyze/make_training_figures.py dynamics

Per-epoch OOD comes from the eval JSONL under vla_augm/results/; everything else
comes from the wandb cache written by `fetch_wandb_curves.py`. Run that first.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood_table import (VISUAL_AXES, load_axis_map,  # noqa: E402
                       load_episodes_with_reruns, summarise)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "vla_augm", "results")
CURVES_DIR = os.path.join(RESULTS, "wandb_curves")
DEFAULT_OUT = os.path.join(REPO, "..", "paper", "vla_augm", "figures")

EPOCHS = list(range(25, 301, 25))

# One colour per placement, shared across every panel so a reader learns the
# mapping once. Matches the palette already defined in main.tex.
PLACEMENTS = [
    ("no augm.", "pi05_augm_none-seed52", "ood_full/none_ep%d.jsonl", "black!45", "square*", "dashed"),
    ("actor + critic", "pi05_augm_uniform_ovl_alpha05-seed52", "ood_full/uniform_ovl_alpha05_ep%d.jsonl", "ACMOrange", "diamond*", "solid"),
    ("actor-only", "pi05_augm_actor_only_ovl_alpha05-seed52", "ood_full/actor_only_ovl_alpha05_ep%d.jsonl", "ACMRed", "triangle*", "solid"),
    ("critic-only", "pi05_augm_critic_only_ovl_alpha05-seed52", "ood_full/critic_only_ovl_alpha05_ep%d.jsonl", "ACMDarkBlue", "*", "solid"),
]

TYPES = [
    ("no augm.", "ood_full/none_ep%d.jsonl", "black!45", "square*", "dashed"),
    ("overlay", "ood_full/critic_only_ovl_alpha05_ep%d.jsonl", "ACMDarkBlue", "*", "solid"),
    ("shift", "ood_full/critic_only_shift_ep%d.jsonl", "ACMGreen", "triangle*", "solid"),
    ("overlay + shift", "ood_full/critic_only_ovlshift_alpha05_ep%d.jsonl", "ACMPurple", "diamond*", "solid"),
]

STRENGTHS = [
    ("no augm.", "ood_full/none_ep%d.jsonl", "black!45", "square*", "dashed"),
    (r"$\alpha=0.10$", "ood_full/critic_only_ovl_alpha01_ep%d.jsonl", "ACMLightBlue", "*", "solid"),
    (r"$\alpha=0.25$", "ood_full/critic_only_ovl_alpha025_ep%d.jsonl", "ACMGreen", "*", "solid"),
    (r"$\alpha=0.50$", "ood_full/critic_only_ovl_alpha05_ep%d.jsonl", "ACMDarkBlue", "*", "solid"),
    (r"$\alpha=0.75$", "ood_full/critic_only_ovl_alpha075_ep%d.jsonl", "ACMOrange", "*", "solid"),
    (r"$\alpha=0.90$", "ood_full/critic_only_ovl_alpha09_ep%d.jsonl", "ACMRed", "*", "solid"),
]


def load_curve(tag: str) -> dict[str, list[tuple[int, float]]]:
    """{metric: [(epoch, value)]} from one cached wandb TSV."""
    path = os.path.join(CURVES_DIR, tag + ".tsv")
    if not os.path.isfile(path):
        sys.exit(f"missing {path}\nRun: python vla_augm/analyze/fetch_wandb_curves.py")
    out: dict[str, list[tuple[int, float]]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            epoch = int(row["epoch"])
            for key, value in row.items():
                if key != "epoch" and value:
                    out.setdefault(key, []).append((epoch, float(value)))
    return out


def ood_series(pattern: str, axis_map) -> list[tuple[int, float]]:
    """Macro OOD success against epoch, using the corrected axes where available."""
    points = []
    for epoch in EPOCHS:
        rel = pattern % epoch
        if not os.path.isfile(os.path.join(RESULTS, rel)):
            continue
        episodes, _ = load_episodes_with_reruns(rel, axis_map, "success_once", RESULTS)
        points.append((epoch, summarise(episodes, VISUAL_AXES)["macro"] * 100))
    return points


def smooth(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    """Centred moving average. Per-epoch training metrics are noisy enough that
    four raw traces on one axis read as a smear; the trend is the point here."""
    if window <= 1 or len(points) < window:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    half = window // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append((xs[i], sum(ys[lo:hi]) / (hi - lo)))
    return out


def plot(coords, color, mark, style, width="1.0pt", marks=True, opacity=None):
    body = " ".join(f"({x},{y:.4g})" for x, y in coords)
    mark_opt = f"mark={mark}" if marks else "mark=none"
    extra = f", opacity={opacity}" if opacity else ""
    return (rf"\addplot[color={color}, {mark_opt}, {style}, line width={width}"
            rf"{extra}] coordinates {{{body}}};")


def legend(entries, per_row=4, xstep=3.3):
    """A hand-drawn legend. pgfplots' legend-to-name does not survive \\input."""
    out = [r"\begin{tikzpicture}[font=\small]"]
    for j, (label, color, mark, style) in enumerate(entries):
        col, row = j % per_row, j // per_row
        x, y = col * xstep, -row * 0.55
        out.append(rf"\draw[color={color}, {style}, line width=1.0pt, mark={mark}, "
                   rf"mark size=1.4pt] plot coordinates {{({x},{y}) ({x + 0.55},{y})}};")
        out.append(rf"\node[anchor=west, inner sep=2pt] at ({x + 0.72},{y}) {{{label}}};")
    out.append(r"\end{tikzpicture}")
    return out


def write(out_dir: str, name: str, lines: list[str]) -> None:
    path = os.path.join(os.path.abspath(out_dir), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------- dynamics
def build_dynamics(out_dir: str, window: int) -> None:
    """Six panels of training internals, one line per placement."""
    # The entropy term is identically zero in these runs (no entropy bonus in the
    # PPO config), so it is not plotted; the clip fraction carries the same kind
    # of information about how hard the update is pushing.
    panels = [
        ("env/reward", "training reward", None, ""),
        ("eval/success_once", r"IND success (\%)", 100.0, "ymin=-5, ymax=104, ytick={0,25,50,75,100}"),
        ("train/actor/policy_loss", "actor policy loss", None, ""),
        ("train/critic/value_loss", "critic value loss", None, "ymode=log"),
        ("train/actor/clip_fraction", "actor clip fraction", None, ""),
        ("train/actor/approx_kl", "approx.\\ KL", None, "ymode=log"),
    ]
    data = {label: load_curve(tag) for label, tag, _p, _c, _m, _s in PLACEMENTS}

    out = ["% Generated by vla_augm/analyze/make_training_figures.py dynamics.",
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[",
           # xlabels at=edge bottom prints "training epoch" only under the
           # bottom row. It suppresses the label without touching `vertical sep`,
           # which is the separation between the axis boxes, so the rows stay
           # exactly where they were and the gap between them is unchanged.
           r"  group style={group size=3 by 2, horizontal sep=1.05cm, vertical sep=1.5cm,",
           r"               xlabels at=edge bottom},",
           r"  width=0.33\linewidth, height=3.3cm,",
           r"  xlabel={training epoch},",
           r"  xlabel style={font=\scriptsize}, ylabel style={font=\scriptsize},",
           r"  tick label style={font=\scriptsize},",
           r"  title style={font=\small, yshift=-2pt},",
           r"  xmin=0, xmax=310, xtick={0,100,200,300},",
           r"  axis lines=left, tick align=outside,",
           r"  scaled y ticks=false,",
           r"  every axis plot/.append style={mark size=0.9pt},",
           r"]"]
    for metric, title, scale, opts in panels:
        head = f"title={{{title}}}"
        if opts:
            head += ", " + opts
        out.append(rf"\nextgroupplot[{head}]")
        for label, _tag, _pat, color, mark, style in PLACEMENTS:
            series = data[label].get(metric, [])
            if not series:
                continue
            if scale:
                series = [(e, v * scale) for e, v in series]
            dense = metric.startswith(("env/", "train/"))
            if dense:
                series = smooth(series, window)
            out.append(plot(series, color, mark, style,
                            width="0.9pt", marks=not dense))
    out += [r"\end{groupplot}", r"\end{tikzpicture}", "", r"\vspace{6pt}", ""]
    out += legend([(l, c, m, s) for l, _t, _p, c, m, s in PLACEMENTS])
    write(out_dir, "training_dynamics.tex", out)


# --------------------------------------------------------------- ood curves
def build_oodcurves(out_dir: str) -> None:
    """Macro OOD against epoch: augmentation type, then overlay strength."""
    axis_map = load_axis_map("libero_spatial")
    out = ["% Generated by vla_augm/analyze/make_training_figures.py oodcurves.",
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[",
           r"  group style={group size=2 by 1, horizontal sep=1.35cm},",
           r"  width=0.47\linewidth, height=4.4cm,",
           r"  xlabel={training epoch}, ylabel={OOD success (\%)},",
           r"  xlabel style={font=\scriptsize}, ylabel style={font=\scriptsize},",
           r"  tick label style={font=\scriptsize},",
           r"  title style={font=\small, yshift=-2pt},",
           r"  xmin=0, xmax=310, ymin=48, ymax=88,",
           r"  xtick={0,100,200,300}, ytick={50,60,70,80},",
           r"  axis lines=left, tick align=outside,",
           r"  every axis plot/.append style={mark size=1.0pt},",
           r"]"]
    for title, spec in [("augmentation type", TYPES), ("overlay strength", STRENGTHS)]:
        out.append(rf"\nextgroupplot[title={{{title}}}]")
        for label, pattern, color, mark, style in spec:
            series = ood_series(pattern, axis_map)
            if series:
                out.append(plot(series, color, mark, style, width="0.9pt"))
                print(f"  {title:20s} {label:16s} {series[-1][1]:.1f} at epoch {series[-1][0]}")
    out += [r"\end{groupplot}", r"\end{tikzpicture}", "", r"\vspace{6pt}", ""]
    out += legend([(l, c, m, s) for l, _p, c, m, s in TYPES], per_row=4, xstep=3.0)
    out += ["", r"\vspace{5pt}", ""]
    out += legend([(l, c, m, s) for l, _p, c, m, s in STRENGTHS], per_row=3, xstep=3.0)
    write(out_dir, "ood_curves.tex", out)


# ----------------------------------------------------------------- tradeoff
TRADEOFF = {
    "pi05": [
        ("no augm.", "ood_full/none_ep300.jsonl", 93.2, "black!55", "square*"),
        ("actor + critic", "ood_full/uniform_ovl_alpha05_ep300.jsonl", 0.8, "ACMOrange", "diamond*"),
        ("actor-only", "ood_full/actor_only_ovl_alpha05_ep100.jsonl", 11.0, "ACMRed", "triangle*"),
        ("critic-only", "ood_full/critic_only_ovl_alpha05_ep300.jsonl", 98.0, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_shift_ep300.jsonl", 95.8, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_ovlshift_alpha05_ep300.jsonl", 93.6, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_ovl_alpha01_ep300.jsonl", 96.0, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_ovl_alpha025_ep300.jsonl", 93.8, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_ovl_alpha075_ep300.jsonl", 94.8, "ACMDarkBlue", "*"),
        ("critic-only", "ood_full/critic_only_ovl_alpha09_ep300.jsonl", 93.2, "ACMDarkBlue", "*"),
    ],
    "gr00t": [
        ("no augm.", "ood_gr00t/none_ep300.jsonl", 95.4, "black!55", "square*"),
        ("actor + critic", "ood_gr00t/uniform_ovl_alpha05_ep100.jsonl", 0.2, "ACMOrange", "diamond*"),
        ("actor-only", "ood_gr00t/actor_only_ovl_alpha05_ep100.jsonl", 0.0, "ACMRed", "triangle*"),
        ("critic-only", "ood_gr00t/critic_only_ovl_alpha05_ep300.jsonl", 97.4, "ACMDarkBlue", "*"),
        ("critic-only", "ood_gr00t/critic_only_shift_ep300.jsonl", 96.4, "ACMDarkBlue", "*"),
        ("critic-only", "ood_gr00t/critic_only_ovlshift_alpha05_ep300.jsonl", 95.4, "ACMDarkBlue", "*"),
        # GR00T alpha 0.25 and 0.75 are deliberately absent: they were never
        # re-evaluated after the LIBERO-Plus asset fix, so their OOD values come
        # from a different evaluation environment than every other point here and
        # would sit ~10 points too low. The collapsed runs above are kept because
        # a run that fails on every axis cannot be moved by that correction.
    ],
}
MODEL_TITLES = {"pi05": r"$\pi_{0.5}$", "gr00t": "GR00T N1.5"}


def build_tradeoff(out_dir: str) -> None:
    """IND against OOD, one marker per configuration.

    The axes are cropped to the configurations that trained successfully. Plotted
    over the full 0-100 range the collapsed runs sit at the origin and squeeze
    everything else into one corner, where the comparison the figure exists to
    make -- critic-only against the baseline, a few points in each direction --
    is not resolvable. The collapsed runs are named in the caption instead, and
    \\Cref{fig:dynamics} already shows them failing.

    A dashed crosshair marks the unaugmented baseline, so "above and to the
    right" can be read off directly rather than inferred from the axis ticks.
    """
    axis_map = load_axis_map("libero_spatial")
    out = ["% Generated by vla_augm/analyze/make_training_figures.py tradeoff.",
           r"\begin{tikzpicture}",
           r"\begin{groupplot}[",
           r"  group style={group size=2 by 1, horizontal sep=1.35cm},",
           r"  width=0.47\linewidth, height=4.8cm,",
           r"  xlabel={in-distribution success (\%)}, ylabel={OOD success (\%)},",
           r"  xlabel style={font=\scriptsize}, ylabel style={font=\scriptsize},",
           r"  tick label style={font=\scriptsize},",
           r"  title style={font=\small, yshift=-2pt},",
           r"  xmin=91.5, xmax=99.5, ymin=62, ymax=85,",
           r"  xtick={92,94,96,98}, ytick={65,70,75,80},",
           r"  axis lines=left, tick align=outside,",
           r"  every axis plot/.append style={only marks, mark size=2.4pt},",
           r"]"]
    for model, rows in TRADEOFF.items():
        out.append(rf"\nextgroupplot[title={{{MODEL_TITLES[model]}}}]")
        points, baseline = [], None
        for label, rel, ind, color, mark in rows:
            if label != "critic-only" and label != "no augm.":
                continue  # collapsed runs are off-scale by design
            if not os.path.isfile(os.path.join(RESULTS, rel)):
                print(f"  skip missing {rel}")
                continue
            eps, _ = load_episodes_with_reruns(rel, axis_map, "success_once", RESULTS)
            ood = summarise(eps, VISUAL_AXES)["macro"] * 100
            if label == "no augm.":
                baseline = (ind, ood)
            else:
                points.append((ind, ood))
        if baseline:
            # Crosshair first, so the markers sit on top of it.
            out.append(rf"\draw[black!35, dashed, line width=0.6pt] "
                       rf"(axis cs:91.5,{baseline[1]:.4g}) -- (axis cs:99.5,{baseline[1]:.4g});")
            out.append(rf"\draw[black!35, dashed, line width=0.6pt] "
                       rf"(axis cs:{baseline[0]:.4g},62) -- (axis cs:{baseline[0]:.4g},85);")
        body = " ".join(f"({x:.4g},{y:.4g})" for x, y in points)
        out.append(rf"\addplot[color=ACMDarkBlue, mark=*, "
                   rf"mark options={{fill=ACMDarkBlue}}] coordinates {{{body}}};")
        if baseline:
            out.append(rf"\addplot[color=black, mark=square*, mark size=2.6pt, "
                       rf"mark options={{fill=black!55}}] coordinates "
                       rf"{{({baseline[0]:.4g},{baseline[1]:.4g})}};")
        print(f"  {model:6s} baseline {baseline}  critic-only points {len(points)}")
    out += [r"\end{groupplot}", r"\end{tikzpicture}", "", r"\vspace{6pt}", ""]
    out += legend([("no augm. (dashed crosshair)", "black!55", "square*", "solid"),
                   ("critic-only, every type and strength", "ACMDarkBlue", "*", "solid")],
                  per_row=2, xstep=5.4)
    write(out_dir, "tradeoff.tex", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["dynamics", "oodcurves", "tradeoff", "all"])
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smooth", type=int, default=9,
                    help="moving-average window for per-epoch training metrics")
    args = ap.parse_args()
    if args.what in ("dynamics", "all"):
        build_dynamics(args.out_dir, args.smooth)
    if args.what in ("oodcurves", "all"):
        build_oodcurves(args.out_dir)
    if args.what in ("tradeoff", "all"):
        build_tradeoff(args.out_dir)


if __name__ == "__main__":
    main()
