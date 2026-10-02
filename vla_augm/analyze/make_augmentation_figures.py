#!/usr/bin/env python3
"""Build the two figures made from the example augmentation clips.

    python vla_augm/analyze/make_augmentation_figures.py teaser   # Figure 1
    python vla_augm/analyze/make_augmentation_figures.py ladder   # appendix
    python vla_augm/analyze/make_augmentation_figures.py survey   # pick frames

teaser  One clean observation, the same observation at alpha 0.25 and 0.5, and a
        curve of OOD success against training epoch for no augmentation, naive
        (actor + critic) augmentation and critic-only. Four panels on one row.
ladder  One row of clean observations over the same frames at rising alpha.
survey  Contact sheet of candidate frames, so overlays can be chosen by eye.

The curve is emitted as a pgfplots fragment rather than a raster: it keeps the
paper's fonts, scales cleanly, and means this script needs nothing beyond the
standard library and ffmpeg -- matplotlib and numpy are not dependencies of
anything else in vla_augm/analyze either.

CHOOSING FRAMES IS A JUDGEMENT CALL, NOT AN INDEX. The overlay bank is unfiltered
ImageNet, so a draw can blend in legible text, a recognizable person, or an animal
close-up that is simply unpleasant to meet in a paper. None of that is detectable
from the frame number, so the defaults below were picked by eye from `survey` and
are deliberately restricted to fabric, stone and foliage. Re-run `survey` and
re-pick if the source clip ever changes.

DERIVING ALPHAS. Only 0.25/0.5/0.75 clips exist. The blend is exactly linear,
o~ = (1-a) o + a m, so from the clean frame A and a blended frame B rendered at a0,
o~(a) = A + (a/a0)(B - A) reproduces any strength from that same overlay draw. We
derive from the 0.75 clip so every factor stays <= 1.2 and mp4 noise is barely
amplified: rebuilding the real 0.5 and 0.25 clips this way scores 34.2 and 35.0 dB
PSNR, against 29.5 dB when extrapolating up from the 0.25 clip. The alpha clips
share one overlay sequence, which is what makes this valid.

GEOMETRY. Clips are 512x256, main|wrist at 256x256, so x=0..255 is the main
camera. The recorder burns a HUD onto the render -- reward and termination on top,
the instruction along the bottom, the latter wrapping to two lines when a
perturbation suffix is appended -- which is not part of the observation. The
default crop clears both bands.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood_table import VISUAL_AXES, load_axis_map, load_episodes, summarise  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEOS = os.path.join(REPO, "vla_augm", "example_videos")
FIGURES = os.path.join(REPO, "..", "paper", "vla_augm", "figures")
CROP = "crop=256:190:0:40"
# The HUD-free band is y=40..229, so 190 is the tallest crop with no burnt-in text.
# Removing the text properly would mean re-rendering with the simulator.
# The teaser trims 16 px from each side as well: the overlay is randomly resized
# and cropped before blending, which can leave its own edge running down the frame,
# and those seams are distracting at figure size. The ladder keeps full width.
CROP_TEASER = "crop=204:190:26:40"
LADDER_SRC, LADDER_ALPHA = "epoch0_step0_alpha075_augm.mp4", 0.75

# Fabric, stone and foliage only -- see the note above.
TEASER_FRAME = 120            # flowering plant; no creature, no text
# water, grass, flowers, rock, pebbles. The bank also holds an octopus, a shark,
# a scuba diver, boxers and cell microscopy, none of which belong in a figure.
LADDER_FRAMES = [84, 180, 120, 132, 108]
LADDER_ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]

EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# The teaser plots the sensor-noise axis rather than the five-axis mean: it is where
# critic-only helps most (40.7 -> 67.8 at epoch 300, against 74.0 -> 82.3 averaged),
# so the figure shows the effect at full size instead of diluted across axes that
# are already near ceiling. Set to VISUAL_AXES for the macro average.
# The y-label says only "OOD success"; which axis it is comes from the caption,
# so keep the caption in sync if CURVE_AXES changes.
CURVE_AXES = ["noise"]
CURVE_YLABEL = r"OOD success (\%)"
CURVES = [
    ("no augm.", "none_ep%d.jsonl"),
    ("naive", "uniform_ovl_alpha05_ep%d.jsonl"),
    ("critic-only", "critic_only_ovl_alpha05_ep%d.jsonl"),
]


def ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, check=True)


def frame(video: str, n: int, out: str, crop: str = CROP) -> None:
    ff(["-i", os.path.join(VIDEOS, video), "-vf", f"select='eq(n\\,{n})',{crop}",
        "-frames:v", "1", "-vsync", "0", out])
    if not os.path.exists(out):
        sys.exit(f"no frame {n} in {video}")


def blend(clean: str, src: str, alpha: float, out: str) -> None:
    """One frame at an arbitrary alpha, from the clean frame and the 0.75 render."""
    k = alpha / LADDER_ALPHA
    ff(["-i", clean, "-i", src, "-filter_complex",
        f"blend=all_expr='clip(A+{k}*(B-A),0,255)'", out])


def hstack(paths: list[str], out: str, gap: int = 5) -> None:
    inputs, filters, labels = [], "", ""
    for i, p in enumerate(paths):
        inputs += ["-i", p]
        if i < len(paths) - 1:
            filters += f"[{i}]pad=iw+{gap}:ih:0:0:white[p{i}];"
            labels += f"[p{i}]"
        else:
            labels += f"[{i}]"
    ff(inputs + ["-filter_complex", f"{filters}{labels}hstack=inputs={len(paths)}[o]",
                 "-map", "[o]", out])


def ood_curves() -> dict[str, list[tuple[int, float]]]:
    """Macro OOD over the five visual axes, per checkpoint, for the three arms."""
    axis_map = load_axis_map("libero_spatial")
    out = {}
    for label, pattern in CURVES:
        points = []
        for e in EPOCHS:
            path = os.path.join(REPO, "vla_augm", "results", "ood_full", pattern % e)
            if not os.path.exists(path):
                continue
            s = summarise(load_episodes(path, axis_map, "success_once"), CURVE_AXES)
            points.append((e, s["macro"] * 100))
        if not points:
            sys.exit(f"no checkpoints found for {label}")
        out[label] = points
    return out


def write_curve_tex(curves, out: str) -> None:
    # critic-only is the result, so it gets the solid heavy line; the baseline is
    # dashed so the two stay distinguishable where they overlap, which is most of
    # the first 200 epochs.
    marks = {"no augm.": ("black!45", "square*", "dashed", "0.8pt"),
             "naive": ("ACMRed", "triangle*", "solid", "0.9pt"),
             "critic-only": ("ACMDarkBlue", "*", "solid", "1.4pt")}
    lines = [
        "% Generated by vla_augm/analyze/make_augmentation_figures.py -- do not hand-edit.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=\linewidth, height=3.1cm,",
        rf"  xlabel={{training epoch}}, ylabel={{{CURVE_YLABEL}}},",
        r"  xlabel style={font=\footnotesize, yshift=2pt},",
        r"  ylabel style={font=\footnotesize, yshift=-4pt},",
        r"  tick label style={font=\scriptsize},",
        # On the sensor-noise axis the curves span 0-68 and cross each other, so no band
# inside the data is free. The legend therefore sits just above the axis --
# anchor=south, so the box grows upward; anchor=north at the same coordinate hangs
# it back down over the plot, which is the easy mistake here.
        r"  xmin=0, xmax=315, ymin=-4, ymax=74,",
        r"  xtick={0,100,200,300}, ytick={0,25,50,75},",
        r"  legend cell align=left, legend columns=2,",
        r"  legend style={font=\tiny, at={(0.5,1.03)}, anchor=south,",
        r"                draw=none, fill=white,",
        r"                row sep=-1pt, inner sep=2pt},",
        r"  axis lines=left, tick align=outside,",
        r"  every axis plot/.append style={mark size=1.2pt},",
        r"]",
    ]
    for label, points in curves.items():
        color, mark, style, width = marks[label]
        coords = " ".join(f"({e},{v:.2f})" for e, v in points)
        lines.append(rf"\addplot[color={color}, mark={mark}, {style}, "
                     rf"line width={width}] coordinates {{{coords}}};")
        lines.append(rf"\addlegendentry{{{label}}}")
    lines += [r"\end{axis}", r"\end{tikzpicture}", ""]
    with open(out, "w") as fh:
        fh.write("\n".join(lines))


def build_teaser(figdir: str) -> None:
    tmp = os.path.join(figdir, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    clean = os.path.join(figdir, "teaser_clean.png")
    src = os.path.join(tmp, "src075.png")
    frame("clean.mp4", TEASER_FRAME, clean, CROP_TEASER)
    frame(LADDER_SRC, TEASER_FRAME, src, CROP_TEASER)
    for alpha, name in ((0.25, "teaser_a025.png"), (0.50, "teaser_a05.png")):
        blend(clean, src, alpha, os.path.join(figdir, name))
    # Recover and keep the ImageNet image itself. The blend is linear, so
    # m = A + (B - A)/a0 inverts it exactly (up to mp4 noise). TEASER_FRAME already
    # pins the choice, but the clips could be re-recorded with a different overlay
    # sequence; this file is the draw we actually picked.
    ff(["-i", clean, "-i", src, "-filter_complex",
        f"blend=all_expr='clip(A+{1 / LADDER_ALPHA:.6f}*(B-A),0,255)'",
        os.path.join(figdir, "teaser_overlay_source.png")])
    curves = ood_curves()
    write_curve_tex(curves, os.path.join(figdir, "teaser_curve.tex"))
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"frame {TEASER_FRAME}: teaser_clean.png, teaser_a025.png, teaser_a05.png")
    print("teaser_curve.tex:")
    for label, pts in curves.items():
        print(f"  {label:<24}{pts[0][1]:5.1f} at ep{pts[0][0]} -> {pts[-1][1]:5.1f} at ep{pts[-1][0]}")


def build_ladder(figdir: str) -> None:
    if len(LADDER_FRAMES) != len(LADDER_ALPHAS):
        sys.exit("LADDER_FRAMES and LADDER_ALPHAS must be the same length")
    tmp = os.path.join(figdir, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    cleans, augs = [], []
    for i, (n, alpha) in enumerate(zip(LADDER_FRAMES, LADDER_ALPHAS)):
        c = os.path.join(tmp, f"c{i}.png")
        b = os.path.join(tmp, f"b{i}.png")
        a = os.path.join(tmp, f"a{i}.png")
        frame("clean.mp4", n, c)
        frame(LADDER_SRC, n, b)
        blend(c, b, alpha, a)
        cleans.append(c)
        augs.append(a)
    hstack(cleans, os.path.join(figdir, "overlay_clean.png"))
    hstack(augs, os.path.join(figdir, "overlay_aug.png"))
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("overlay_clean.png, overlay_aug.png")
    print(f"frames {LADDER_FRAMES} at alphas {LADDER_ALPHAS}")


AXIS_TITLES = [("textures", "background textures"), ("lighting", "light conditions"),
               ("layout", "objects layout"), ("camera", "camera viewpoints"),
               ("noise", "sensor noise")]


# The zoomed variant drops the collapsed arm. With "naive" in the panel the y-axis
# has to span 0-100 and the two surviving curves are squashed into the top band;
# without it each panel can be scaled to the pair actually being compared.
ZOOM_CURVES = [c for c in CURVES if c[0] != "naive"]


def axis_limits(values: list[float], pad: float = 2.0, step: float = 5.0):
    """Rounded (ymin, ymax, tick step) enclosing `values` with a little air."""
    lo, hi = min(values) - pad, max(values) + pad
    lo = max(0.0, math.floor(lo / step) * step)
    hi = min(100.0, math.ceil(hi / step) * step)
    span = hi - lo
    tick = 5.0 if span <= 25 else (10.0 if span <= 60 else 25.0)
    return lo, hi, tick


def build_axis_curves(figdir: str, curves=None, zoom: bool = False,
                      name: str = "axis_curves.tex") -> None:
    """The teaser's curve, once per visual axis, as a 3x2 groupplot.

    Figure 1 shows only sensor noise. This is the same three runs scored on each
    axis separately, so a reader can check that the headline is not one lucky
    axis -- and see where critic-only does little, which the five-axis mean hides
    in both directions.

    The legend goes in the empty sixth cell rather than above the block: five
    panels leave one free, and a legend there needs no space of its own.
    """
    curves = curves or CURVES
    axis_map = load_axis_map("libero_spatial")
    marks = {"no augm.": ("black!45", "square*", "dashed", "0.8pt"),
             "naive": ("ACMRed", "triangle*", "solid", "0.9pt"),
             "critic-only": ("ACMDarkBlue", "*", "solid", "1.3pt")}

    data = {}
    for label, pattern in curves:
        for e in EPOCHS:
            path = os.path.join(REPO, "vla_augm", "results", "ood_full", pattern % e)
            if not os.path.exists(path):
                continue
            eps = load_episodes(path, axis_map, "success_once")
            s = summarise(eps, [a for a, _ in AXIS_TITLES])
            for axis, _t in AXIS_TITLES:
                data.setdefault(axis, {}).setdefault(label, []).append(
                    (e, s["per_axis"][axis][0] * 100))

    # Shared limits keep the five panels comparable; per-panel limits make a
    # small gap legible. The zoomed variant wants the second, because without the
    # collapsed arm the pair being compared occupies a narrow band that differs
    # per axis -- textures sits near 95, sensor noise spans 13 to 68.
    shared = ("  ymin=-5, ymax=104,\n  ytick={0,25,50,75,100},"
              if not zoom else "")

    def panel_head(axis, title):
        head = f"title={{{title}}}"
        if zoom:
            vals = [v for label, _p in curves for _e, v in data[axis][label]]
            lo, hi, tick = axis_limits(vals)
            ticks = ",".join(f"{t:g}" for t in
                             [lo + k * tick for k in range(int((hi - lo) / tick) + 1)])
            head += f", ymin={lo:g}, ymax={hi:g}, ytick={{{ticks}}}"
        return head

    def block(panels, columns, rows):
        """One tikzpicture holding a `columns` x `rows` grid of panels."""
        lines = [
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            rf"  group style={{group size={columns} by {rows}, horizontal sep=1.15cm, vertical sep=1.85cm}},",
            r"  width=0.365\linewidth, height=3.5cm,",
            r"  xlabel={training epoch}, ylabel={success (\%)},",
            r"  xlabel style={font=\scriptsize}, ylabel style={font=\scriptsize},",
            r"  tick label style={font=\scriptsize},",
            r"  title style={font=\small, yshift=-2pt},",
            r"  xmin=0, xmax=315,",
        ]
        if shared:
            lines.append(shared)
        lines += [
            r"  xtick={0,100,200,300},",
            r"  axis lines=left, tick align=outside,",
            r"  every axis plot/.append style={mark size=1.1pt},",
            r"]",
        ]
        for axis, title in panels:
            lines.append(rf"\nextgroupplot[{panel_head(axis, title)}]")
            for label, _pat in curves:
                color, mark, style, width = marks[label]
                coords = " ".join(f"({e},{v:.2f})" for e, v in data[axis][label])
                lines.append(rf"\addplot[color={color}, mark={mark}, {style}, "
                             rf"line width={width}] coordinates {{{coords}}};")
        lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
        return lines

    out = [f"% Generated by vla_augm/analyze/make_augmentation_figures.py "
           f"{'axiscurveszoom' if zoom else 'axiscurves'}."]
    if zoom:
        # Five panels in a 3x2 grid leave a hole in the bottom right. Emitting the
        # rows as two tikzpictures instead lets the second one hold just two
        # panels, which the surrounding \centering then centres -- groupplots
        # cannot offset a row by half a cell to achieve the same thing.
        out += block(AXIS_TITLES[:3], 3, 1)
        out += ["", r"\vspace{6pt}", ""]
        out += block(AXIS_TITLES[3:], 2, 1)
    else:
        out += block(AXIS_TITLES, 3, 2)
    out += ["", r"\vspace{4pt}", ""]
    # An explicit legend, drawn with tikz. pgfplots' legend-to-name + \ref does
    # not survive \input into the document, and a legend-only \nextgroupplot
    # fails outright ("Dimension too large") because the panel has no coordinates
    # to fit a scale to. Three line segments cost less than either workaround.
    out += [r"\begin{tikzpicture}[font=\small]"]
    for j, (label, _pat) in enumerate(curves):
        color, mark, style, width = marks[label]
        x = j * 3.3
        out.append(rf"\draw[color={color}, {style}, line width={width}, mark={mark}, "
                   rf"mark size=1.4pt] plot coordinates {{({x},0) ({x + 0.55},0)}};")
        out.append(rf"\node[anchor=west, inner sep=2pt] at ({x + 0.72},0) {{{label}}};")
    out += [r"\end{tikzpicture}", ""]

    path = os.path.join(figdir, name)
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {path}")
    for axis, _t in AXIS_TITLES:
        cells = "  ".join(f"{lbl}: {data[axis][lbl][-1][1]:.1f}" for lbl, _ in curves)
        print(f"  {axis:<10}at epoch 300 -> {cells}")


def build_policyview(figdir: str) -> None:
    """The observation at the size and shape the policy actually receives.

    Both cameras, uncropped and scaled to 224x224 -- the resolution the model
    resizes to (DEFAULT_IMAGE_SIZE in rlinf/algorithms/augmentation.py). The
    recorder's HUD is part of these pixels and cannot be removed without also
    cropping the view, so it is left in and called out in the caption: the point
    of this figure is the framing and aspect ratio, not a clean picture.
    """
    for cam, x in (("main", 0), ("wrist", 256)):
        out = os.path.join(figdir, f"policyview_{cam}.png")
        ff(["-i", os.path.join(VIDEOS, "clean.mp4"), "-vf",
            f"select='eq(n\\,{TEASER_FRAME})',crop=256:256:{x}:0,scale=224:224",
            "-frames:v", "1", "-vsync", "0", out])
    print("policyview_main.png, policyview_wrist.png (224x224, HUD included)")


def build_survey(figdir: str, step: int) -> None:
    out = os.path.join(figdir, "survey.png")
    ff(["-i", os.path.join(VIDEOS, LADDER_SRC), "-vf",
        f"{CROP},select='not(mod(n\\,{step}))',scale=150:98,"
        f"tile=7x3:padding=3:color=white", "-frames:v", "1", "-vsync", "0", out])
    print(f"{out}\ncolumns are frames 0,{step},{2 * step},... left to right, top to bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["teaser", "ladder", "survey", "policyview",
                                     "axiscurves", "axiscurveszoom"])
    ap.add_argument("--out-dir", default=FIGURES)
    ap.add_argument("--survey-step", type=int, default=12)
    args = ap.parse_args()
    figdir = os.path.abspath(args.out_dir)
    os.makedirs(figdir, exist_ok=True)
    {"axiscurves": lambda: build_axis_curves(figdir),
     "axiscurveszoom": lambda: build_axis_curves(
         figdir, curves=ZOOM_CURVES, zoom=True, name="axis_curves_zoom.tex"),
     "policyview": lambda: build_policyview(figdir),
     "teaser": lambda: build_teaser(figdir),
     "ladder": lambda: build_ladder(figdir),
     "survey": lambda: build_survey(figdir, args.survey_step)}[args.what]()


if __name__ == "__main__":
    main()
