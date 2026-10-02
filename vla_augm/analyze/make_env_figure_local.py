#!/usr/bin/env python3
"""Rebuild the five-perturbation-axes figure by rendering locally on a Mac.

Same figure as `make_env_figure.py`, built a shorter way. That script works from
cluster clips: it has to find a video per axis, pick a frame out of it, and crop
off the HUD the recorder burned in. Here the simulator is right there, so the
figure is rendered straight to PNG -- no video, no frame search, and **no crop**,
because nothing is overlaid on the image in the first place.

    python vla_augm/analyze/make_env_figure_local.py
    python vla_augm/analyze/make_env_figure_local.py --list
    python vla_augm/analyze/make_env_figure_local.py --pick textures=table_37

Needs the Mac venv from vla_augm/install.md ("Rendering LIBERO-Plus locally on Mac"):

    export MAGICK_HOME="$(brew --prefix imagemagick)"

Writes one PNG per panel, `env_<tag>_<ind|ood>_<axis>.png`, leaving
`make_env_figure.py`'s `env_row_<axis>.png` untouched so the two can sit side by
side in the paper. Panels are written separately rather than stitched into rows
because the figure lays them out as a grid, and the gaps between them have to
come from LaTeX.

WHY THE PAIRS LINE UP HERE AND NOT ON THE CLUSTER. Both panels of a row are the
same base task, same seed, same actions, same step -- and none of the five
perturbations touches the dynamics, so the arm reaches the same pose in both
(whether it holds still or rolls out). The only thing that differs is the
perturbation. Cluster clips cannot do this: they come from a
*trained* policy that acts differently in the two scenes, so by the time you pick
a frame the arm has moved somewhere else and the pair differs for two reasons at
once.

Zero actions also mean the base task choice is free. Every axis covers all ten
LIBERO-Spatial base tasks, so axis i simply takes base i -- no backtracking
search like `make_env_figure.py` needs, where an axis may have clips for only a
few bases.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood_table import VISUAL_AXES  # noqa: E402
from record_libero_videos import CAM_H, CAM_W, actions, bddl_path, load_suite  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, "..", "paper", "vla_augm", "figures")

# Steps to hold a zero action before grabbing the frame. The scene starts with
# objects dropped slightly above their resting pose, so frame 0 catches them
# mid-fall; a few steps of settling looks like a normal observation. The arm does
# not move under a zero action, so this does not desynchronise the pair.
SETTLE = 8

# The perturbed variant to use per axis, where the first one on the axis is a poor
# illustration. Chosen with --survey; override with --pick.
#
#   lighting: `light_1` and `light_10` render the scene essentially black (mean
#     brightness 0.0 and 1.6 against the unperturbed 119.9). They score a huge
#     pixel difference precisely because nothing is visible, which is the opposite
#     of what the figure needs. `light_11` is clearly dimmer and still readable.
#   layout: this axis adds distractor objects, so it moves few pixels by nature --
#     every variant scores between 1.7 and 7.0. `add_19` is the largest, and puts
#     several distractors in open view rather than at the frame edge.
DEFAULT_PICKS = {"lighting": "light_11", "layout": "add_19"}


def render_obs(task: dict, settle: int, seed: int, rollout: int = 0) -> dict:
    """The full observation dict, with every camera oriented as the policy gets it.

    `[::-1, ::-1]` is a 180-degree rotation, not just the bottom-up flip MuJoCo
    needs: `rlinf/envs/libero/utils.py:90` applies exactly this to both cameras
    before the observation reaches the model ("rotate 180 degrees to match train
    preprocessing"). Doing only the vertical half leaves the scene mirrored --
    the cabinet lands on the wrong side, and the figure no longer shows what the
    policy sees.

    `rollout` steps of the same random policy the recorder uses run after the
    settle. Both halves of a pair get the same seed and therefore the same action
    sequence, so the arm ends up in the same pose in both -- none of the five
    perturbations changes the dynamics, only what the cameras see.
    """
    from liberoplus.liberoplus.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=bddl_path(task["folder"], task["bddl"]),
                             camera_heights=CAM_H, camera_widths=CAM_W)
    try:
        env.seed(seed)
        obs = env.reset()
        for _ in range(settle):
            obs, _r, _d, _i = env.step(np.zeros(7))
        if rollout:
            for a in actions("smooth", rollout, np.random.default_rng(seed), 1.0):
                obs, _r, _d, _i = env.step(a)
        return {k: (v[::-1, ::-1] if k.endswith("_image") else v)
                for k, v in obs.items()}
    finally:
        env.close()


def render(task: dict, settle: int, seed: int, rollout: int = 0) -> np.ndarray:
    """Just the base camera, which is all this figure shows."""
    return render_obs(task, settle, seed, rollout)["agentview_image"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--settle", type=int, default=SETTLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick", action="append", default=[], metavar="AXIS=SUBSTRING",
                    help="force the perturbed variant for one axis, e.g. lighting=light_23")
    ap.add_argument("--list", action="store_true",
                    help="print the variants that would be used, render nothing")
    ap.add_argument("--survey", metavar="AXIS",
                    help="render the first --survey-n variants of one axis and report how "
                         "different and how bright each is, to choose one for --pick")
    ap.add_argument("--survey-n", type=int, default=10)
    ap.add_argument("--rollout", type=int, default=0,
                    help="steps of the random policy to run after settling, so the arm is "
                         "somewhere other than its home pose (0 = hold still)")
    ap.add_argument("--tag", help="filename tag; defaults to 'still' or 'roll<N>'")
    args = ap.parse_args()

    if not os.environ.get("MAGICK_HOME"):
        sys.exit('MAGICK_HOME is unset; run: export MAGICK_HOME="$(brew --prefix imagemagick)"')

    forced = {**DEFAULT_PICKS, **dict(p.split("=", 1) for p in args.pick)}
    _bench, tasks = load_suite(args.suite)

    if args.survey:
        if args.survey not in VISUAL_AXES:
            sys.exit(f"unknown axis {args.survey!r}; have {', '.join(VISUAL_AXES)}")
        base_id = VISUAL_AXES.index(args.survey)
        pool = [t for t in tasks if t["axis"] == args.survey and t["base_id"] == base_id]
        ref = render({**pool[0], "bddl": pool[0]["base"] + ".bddl"}, args.settle, args.seed)
        print(f"{args.survey} on base{base_id}; IND brightness {ref.mean():.1f}\n")
        print(f"{'variant':46s} {'diff':>7s} {'bright':>7s}")
        for t in pool[: args.survey_n]:
            img = render(t, args.settle, args.seed)
            diff = np.abs(ref.astype(float) - img.astype(float)).mean()
            print(f"{t['bddl'][-46:]:46s} {diff:7.2f} {img.mean():7.1f}")
        print("\nWant a large diff at a brightness near IND's: a very dark variant "
              "scores a huge diff by going black, which shows nothing.")
        return

    chosen = {}
    for i, axis in enumerate(VISUAL_AXES):
        pool = [t for t in tasks if t["axis"] == axis and t["base_id"] == i]
        if axis in forced:
            pool = [t for t in pool if forced[axis] in t["bddl"]]
            if not pool:
                sys.exit(f"--pick {axis}={forced[axis]}: no task on axis {axis} "
                         f"with base {i} matches. Try --list.")
        if not pool:
            sys.exit(f"axis {axis} has no task on base {i}")
        chosen[axis] = pool[0]

    if args.list:
        for axis, t in chosen.items():
            print(f"{axis:9s} base{t['base_id']} task{t['id']:04d}  {t['bddl'][-46:]}")
        return

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    import imageio.v2 as imageio

    # One PNG per panel rather than a pre-stitched row: the figure lays the ten
    # panels out as a 5-wide grid, and gaps between them have to come from LaTeX.
    # Baking them into the image would mean guessing the column width.
    tag = args.tag or (f"roll{args.rollout}" if args.rollout else "still")
    for axis, t in chosen.items():
        ind = render({**t, "bddl": t["base"] + ".bddl"}, args.settle, args.seed, args.rollout)
        ood = render(t, args.settle, args.seed, args.rollout)
        for kind, img in (("ind", ind), ("ood", ood)):
            imageio.imwrite(os.path.join(out_dir, f"env_{tag}_{kind}_{axis}.png"), img)
        diff = np.abs(ind.astype(float) - ood.astype(float)).mean()
        print(f"{axis:9s} base{t['base_id']} task{t['id']:04d}  "
              f"mean|IND-OOD| {diff:6.2f}  {t['bddl'][-42:]}")

    print(f"\nwrote env_{tag}_{{ind,ood}}_*.png to {out_dir}")
    print("A near-zero mean|IND-OOD| means the perturbation did not render.")


if __name__ == "__main__":
    main()
