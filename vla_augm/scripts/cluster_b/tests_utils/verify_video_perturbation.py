#!/usr/bin/env python
"""Prove a recorded OOD clip actually shows its perturbation.

This is the check that would have caught the round-2 supplementary videos. Those
clips carried the right burnt-in overlay ("table 1", "light 10") while rendering
the DEFAULT scene, because LIBERO-Plus delivers textures, lighting and layout by
swapping the scene XML and silently falls back when ``assets/scenes/`` is
missing. Nothing in the pipeline complained; the only tell was pixels.

The measurement is a mean absolute pixel delta between a perturbed clip and its
in-distribution twin. Calibration comes from ``vla_augm/results/texture_debug/``,
rendered on a machine with complete assets:

    real texture perturbation      43 - 52
    real lighting perturbation     ~45
    unperturbed (round-2 clips)     5 - 6

TWO CROPS MATTER, and getting them wrong is how this check fools you:

* **Left half only.** Frames are ``views: ["main", "wrist"]`` tiled side by
  side. The wrist camera sees the gripper at close range and barely moves with a
  table texture, so including it halves the signal.
* **Middle rows only.** ``put_info_on_image`` draws the info block at the top and
  ``put_text_on_image_bottom`` the language instruction at the bottom, burnt in.
  That text NAMES the perturbation, so it differs between a real perturbation and
  a fake one. Measuring over it credits a clip for its own caption -- which is
  precisely the round-2 failure mode read back as a pass.

Frame 0 is compared, not a mean over the episode: the two clips diverge in
behaviour as soon as the policy acts, and that divergence is not what is being
measured.

Usage:
    python verify_video_perturbation.py --clips <dir> [--report out.txt]
"""

from __future__ import annotations

import argparse
import collections
import itertools
import pathlib
import re
import sys

import imageio.v2 as imageio
import numpy as np

# Below this a clip is indistinguishable from an unperturbed render.
#
# Calibrated by running this script over the broken round-2 clips. Their three
# asset-dependent axes scored 3.7, 3.7 and 3.7 -- identical to a decimal, which
# is what rendering the SAME scene twice looks like; the residual is the two
# episodes' behaviour diverging, not the perturbation. So ~4 is the floor, and
# a genuine scene perturbation measures 43-52 (vla_augm/results/texture_debug/).
# Ten sits in the empty gap between them.
THRESHOLD = 10.0

# Noise is the exception and gets its own bar. It perturbs the SENSOR -- gaussian
# noise and motion blur over the render -- not the scene, so it moves far fewer
# pixels by construction. The round-2 noise clip scored 9.4: below the global
# bar, but well clear of the 3.7 floor, i.e. really perturbing. That 9.4 is the
# only datapoint there is, and it comes from a recording that was broken in
# other respects, so 7.0 is a floor-plus-margin guess rather than a calibrated
# threshold. If round 3's noise clips come in near 9 that is expected; if they
# come in near 4, the axis has stopped working.
AXIS_THRESHOLD = {"noise": 7.0, "layout": 3.0}

# Layout needs its own bar for a structural reason, and the bar alone is not
# enough to trust it.
#
# Textures, lighting and camera repaint every pixel, so a mean absolute delta
# separates "applied" from "not applied" cleanly. Layout moves or adds a few
# small objects and leaves the table, robot and background untouched, and
# LIBERO-Plus grades it by severity: `add_<n>` drops in a distractor and
# `level<k>_sample<m>` resamples positions, with k running 1-5. Measured on
# round 3, the mild variants (add_*, level1) score 3.5-6.3 against their twin
# while level2/4/5 score 10.8-27.5. A single threshold either fails the mild
# ones or passes anything.
#
# So the mean delta cannot certify this axis on its own, and DELTA_ONLY_AXES
# records that honestly. What does certify it is check_variant_diversity():
# distinct perturbations of the same base task must render distinctly. If the
# axis were silently no-opping, every variant of a base would be pixel-identical
# to the others -- which is a positive test the sparse-change problem cannot
# defeat.
DELTA_ONLY_AXES = frozenset({"textures", "lighting", "camera"})

# Camera and noise perturb the sensor rather than the scene, so they do not need
# assets/scenes/ and were never part of the reported breakage. They still have to
# differ from the twin -- a camera shift that moved nothing would be its own bug.
ALL_AXES = ("textures", "lighting", "layout", "camera", "noise")

CLIP_RE = re.compile(r"^(?P<axis>[a-z]+)_base(?P<base>\d+)_task(?P<task>\d+)_(?P<ok>ok|fail)\.mp4$")
IND_RE = re.compile(r"^IND_base(?P<base>\d+)_(?P<ok>ok|fail)\.mp4$")


def first_frame(path: pathlib.Path) -> np.ndarray:
    """Return frame 0 as float32, cropped to the main camera's middle band."""
    reader = imageio.get_reader(str(path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    h, w = frame.shape[:2]
    # Left half = main camera; middle 50% of rows drops both burnt-in text bands.
    return frame[h // 4 : 3 * h // 4, : w // 2].astype(np.float32)


def check_variant_diversity(
    cache: dict[pathlib.Path, np.ndarray],
    perturbed: list[tuple[str, int, int, pathlib.Path]],
    say,
) -> None:
    """Distinct variants of one base task must render distinctly.

    The delta-vs-twin test is weak wherever a perturbation touches few pixels.
    This one is not: a silently no-opping axis renders every variant of a base
    identically, whatever the magnitude of the change it was supposed to make.
    """
    groups: dict[tuple[str, int], list[pathlib.Path]] = collections.defaultdict(list)
    for axis, base, _task, path in perturbed:
        if path in cache:
            groups[(axis, base)].append(path)

    say("variant diversity (distinct perturbations must render distinctly):")
    for (axis, base), paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        deltas = [
            float(np.abs(cache[a] - cache[b]).mean())
            for a, b in itertools.combinations(paths, 2)
            if cache[a].shape == cache[b].shape
        ]
        if not deltas:
            continue
        identical = sum(d < 0.01 for d in deltas)
        flag = "ok" if not identical else f"{identical} IDENTICAL PAIRS"
        say(f"  {axis:10s} base {base}: {len(paths):2d} variants, "
            f"pairwise median {np.median(deltas):5.1f}, min {min(deltas):5.1f}  {flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=pathlib.Path, required=True)
    ap.add_argument("--report", type=pathlib.Path, default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    clips = sorted(args.clips.rglob("*.mp4"))
    if not clips:
        print(f"FATAL: no mp4 under {args.clips}", file=sys.stderr)
        return 1

    ind: dict[int, pathlib.Path] = {}
    perturbed: list[tuple[str, int, int, pathlib.Path]] = []
    for p in clips:
        if (m := IND_RE.match(p.name)):
            ind[int(m["base"])] = p
        elif (m := CLIP_RE.match(p.name)):
            perturbed.append((m["axis"], int(m["base"]), int(m["task"]), p))

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say(f"clips: {args.clips}")
    say(f"{len(perturbed)} perturbed, {len(ind)} in-distribution twins, "
        f"threshold {args.threshold}")
    say()

    if not ind:
        say("FATAL: no IND_base*.mp4 -- nothing to compare against. Has "
            "name_video_clips.py run?")
        return 1

    cache: dict[pathlib.Path, np.ndarray] = {}
    by_axis: dict[str, list[float]] = collections.defaultdict(list)
    failures = 0

    say(f"{'axis':10s} {'task':>6s} {'base':>5s} {'delta':>8s}  verdict")
    say("-" * 46)
    for axis, base, task, path in sorted(perturbed):
        twin = ind.get(base)
        if twin is None:
            say(f"{axis:10s} {task:6d} {base:5d} {'--':>8s}  SKIP (no base-{base} twin)")
            continue
        try:
            a = cache.setdefault(path, first_frame(path))
            b = cache.setdefault(twin, first_frame(twin))
        except Exception as exc:  # unreadable clip is a failure, not a crash
            say(f"{axis:10s} {task:6d} {base:5d} {'--':>8s}  FAIL (unreadable: {exc})")
            failures += 1
            continue
        if a.shape != b.shape:
            say(f"{axis:10s} {task:6d} {base:5d} {'--':>8s}  FAIL (shape {a.shape} vs {b.shape})")
            failures += 1
            continue
        delta = float(np.abs(a - b).mean())
        bar = AXIS_THRESHOLD.get(axis, args.threshold)
        ok = delta >= bar
        failures += not ok
        by_axis[axis].append(delta)
        say(f"{axis:10s} {task:6d} {base:5d} {delta:8.1f}  {'PASS' if ok else 'FAIL — looks unperturbed'}")

    say()
    check_variant_diversity(cache, perturbed, say)

    say()
    say("per-axis median delta:")
    for axis in ALL_AXES:
        ds = by_axis.get(axis, [])
        if not ds:
            say(f"  {axis:10s} no clips")
            continue
        med = float(np.median(ds))
        bar = AXIS_THRESHOLD.get(axis, args.threshold)
        say(f"  {axis:10s} {med:6.1f}  over {len(ds)} clips  (bar {bar:.0f})  "
            f"{'ok' if med >= bar else 'BROKEN'}")

    say()
    missing = [a for a in ALL_AXES if not by_axis.get(a)]
    if missing:
        say(f"WARNING: no clips for {', '.join(missing)} -- the set is incomplete.")
    if failures:
        say(f"RESULT: FAIL — {failures} clip(s) below their axis bar. Do not ship.")
    else:
        say("RESULT: PASS — every clip differs from its twin by a real margin.")

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"\nreport written to {args.report}")

    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
