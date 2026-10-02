#!/usr/bin/env python3
"""Record LIBERO-Plus episodes locally on a Mac, driven by a random policy.

Produces the same 512x256 (main | wrist) clips that the cluster eval writes, so
the output is a drop-in replacement for `paper/videos/<run>/seed_0` and feeds
`make_env_figure.py` unchanged. The point is not the behaviour -- a random policy
never solves anything -- but the *scene*: what a perturbation actually does to
the observation, rendered without needing a GPU or a VLA checkpoint.

    python vla_augm/analyze/record_libero_videos.py --list
    python vla_augm/analyze/record_libero_videos.py --axis noise --n 3
    python vla_augm/analyze/record_libero_videos.py --figure-set     # all 5 axes + IND

Needs the Mac venv from vla_augm/install.md ("Rendering LIBERO-Plus locally"). The
script sets MUJOCO_GL, LIBERO_TYPE, LIBERO_SUFFIX and MAGICK_THREAD_LIMIT itself;
the one thing it cannot guess is where Homebrew put ImageMagick:

    export MAGICK_HOME="$(brew --prefix imagemagick)"

NO HUD. The cluster recorder burns reward, termination and the instruction onto
the render; that band is not part of the observation, which is why
`make_env_figure.py` has to crop it away. This script writes the raw camera
images, so clips recorded here need no crop at all -- pass `--crop` an empty
string there if you rebuild the figure from these.

THE PERTURBATION LIVES IN THE FILENAME. LIBERO-Plus does not take a perturbation
argument: `ControlEnv.__init__` parses the *bddl path string* and splits off
`_view_<h>_<v>_<scale>_<rot>_<vert>_initstate_<N>[_noise_<M>]` before checking
that the remaining `<base>.bddl` exists (env_wrapper.py:204). So the pseudo-path
from the benchmark must be passed through verbatim -- resolving it to a real file
first, or normalising it, silently drops the perturbation and renders a clean
scene that looks fine and is wrong.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood_table import VISUAL_AXES, load_axis_map  # noqa: E402

# Set before anything imports liberoplus -- python-wand loads libMagickWand at
# import time, and ImageMagick reads its thread limit once, at library init.
#
# MAGICK_THREAD_LIMIT is not optional on macOS. Homebrew's ImageMagick 7 is built
# against libomp, torch and numba bring their own OpenMP runtime, and two OpenMP
# runtimes in one process segfault -- but only once a MuJoCo GL context is live,
# which is why motion_blur is fine in isolation and dies inside env.reset(). It
# reproduced 3/3 without this and 0/3 with it, on the sensor-noise axis (the only
# axis that calls ImageMagick). OMP_NUM_THREADS=1 fixes it equally well; this one
# is narrower, since it leaves torch's own threading alone.
for _var, _val in (("MUJOCO_GL", "cgl"), ("LIBERO_TYPE", "plus"),
                   ("LIBERO_SUFFIX", "all"), ("MAGICK_THREAD_LIMIT", "1")):
    os.environ.setdefault(_var, _val)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, "..", "paper", "videos", "random_policy_local", "seed_0")

# Matches the cluster eval config (env/libero_plus_ood.yaml init_params) so the
# frames are the same size as the clips already in paper/videos/.
CAM_H = CAM_W = 256
CAMERAS = ["agentview", "robot0_eye_in_hand"]

# The camera/robot_init/noise/language axes always carry a full
# `_view_<...>_initstate_<N>` tail, so chopping at `_view_` recovers their base
# task exactly. That is the only suffix worth hardcoding.
VIEW_RE = re.compile(r"_view_.*$")
# `_language_<N>` sits *between* the base name and `_view_`, so chopping at
# `_view_` alone would leave 390 phantom "base tasks" -- one per reworded prompt.
LANG_RE = re.compile(r"_language_\d+$")


def canonical_bases(names: list[str]) -> list[str]:
    """The unperturbed task names of the suite, longest first.

    Derived rather than enumerated. The texture and layout axes use a zoo of
    suffixes (`_table_N`, `_tb_N`, `_add_N`, and more) that is not documented
    anywhere and grew between releases, so stripping suffixes only works for the
    `_view_` axes -- and those already cover all ten base tasks. Everything else
    is then matched by prefix against this set, which cannot silently miss a
    suffix the way a hardcoded list does.
    """
    bases = {LANG_RE.sub("", VIEW_RE.sub("", n[:-5] if n.endswith(".bddl") else n))
             for n in names if "_view_" in n}
    # longest first so `..._on_the_plate_table_1` cannot match a shorter base that
    # happens to be a prefix of the real one
    return sorted(bases, key=len, reverse=True)


def base_stem(name: str, bases: list[str]) -> str:
    stem = name[:-5] if name.endswith(".bddl") else name
    for b in bases:
        if stem == b or stem.startswith(b + "_"):
            return b
    raise SystemExit(
        f"task {stem!r} matches none of the {len(bases)} base tasks. "
        "The suite's naming changed; re-check canonical_bases()."
    )


def load_suite(suite: str):
    import contextlib
    import io

    from liberoplus.liberoplus.benchmark import get_benchmark

    # The Benchmark constructor prints all 2402 task indices to stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        bench = get_benchmark(suite)()
    axes = load_axis_map(suite)
    raw = [bench.get_task(i) for i in range(bench.n_tasks)]
    bases = canonical_bases([t.bddl_file for t in raw])
    tasks = []
    for i, t in enumerate(raw):
        tasks.append({"id": i, "axis": axes[i], "bddl": t.bddl_file,
                      "folder": t.problem_folder, "language": t.language,
                      "base": base_stem(t.bddl_file, bases)})
    # base index = position in the sorted unique base stems. Arbitrary but stable,
    # and it only has to be consistent within one output directory.
    order = {b: k for k, b in enumerate(sorted({t["base"] for t in tasks}))}
    for t in tasks:
        t["base_id"] = order[t["base"]]
    return bench, tasks


def bddl_path(folder: str, bddl: str) -> str:
    import liberoplus.liberoplus as lp

    return os.path.join(lp.get_libero_path("bddl_files"), folder, bddl)


def actions(kind: str, steps: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """A 7-DoF action sequence: 6 arm deltas in [-1, 1] plus a gripper command.

    `white` is what "random policy" literally means, but iid noise makes the arm
    vibrate in place -- every frame undoes the last, so the scene never changes
    and the clip is useless for showing a perturbation. `smooth` low-passes the
    same noise so the arm actually travels across the table.
    """
    if kind == "zero":
        return np.zeros((steps, 7))
    raw = rng.uniform(-1, 1, size=(steps, 7))
    if kind == "white":
        out = raw
    else:
        out = np.zeros_like(raw)
        acc = np.zeros(7)
        for i in range(steps):  # exponential moving average = temporally correlated
            acc = 0.9 * acc + 0.1 * raw[i]
            out[i] = acc / 0.1 * 0.35  # rescale: the EMA shrinks the amplitude
    out = np.clip(out * scale, -1, 1)
    out[:, 6] = np.where(out[:, 6] > 0, 1.0, -1.0)  # gripper is effectively binary
    return out


def record(task: dict, steps: int, seed: int, policy: str, scale: float,
           out_path: str, fps: int) -> bool:
    import imageio.v2 as imageio
    from liberoplus.liberoplus.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=bddl_path(task["folder"], task["bddl"]),
                             camera_heights=CAM_H, camera_widths=CAM_W,
                             camera_names=CAMERAS)
    try:
        env.seed(seed)
        obs = env.reset()
        rng = np.random.default_rng(seed)
        acts = actions(policy, steps, rng, scale)
        frames, success = [], False
        for a in acts:
            # `[::-1, ::-1]` is a 180-degree rotation, which is what the policy
            # actually receives: rlinf/envs/libero/utils.py:90 applies exactly
            # this to both cameras ("rotate 180 degrees to match train
            # preprocessing"), and the cluster clips are in that orientation too.
            # Doing only the vertical half -- enough to undo MuJoCo's bottom-up
            # render -- leaves the scene mirrored, with the cabinet on the wrong
            # side.
            frames.append(np.concatenate(
                [obs[f"{c}_image"][::-1, ::-1] for c in CAMERAS], axis=1))
            obs, _reward, done, _info = env.step(a)
            success = success or bool(env.check_success())
            if done:
                break
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with imageio.get_writer(out_path, fps=fps, codec="libx264",
                                macro_block_size=1, quality=8) as w:
            for f in frames:
                w.append_data(f)
        return success
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--axis", help=f"one of {', '.join(VISUAL_AXES)} (or robot_init/language)")
    ap.add_argument("--task-id", type=int, help="record exactly this task id")
    ap.add_argument("--n", type=int, default=1, help="clips per axis")
    ap.add_argument("--figure-set", action="store_true",
                    help="one clip per visual axis plus the matching IND clips, "
                         "each axis on a different base task -- what make_env_figure.py wants")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="smooth", choices=["smooth", "white", "zero"])
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=20)  # = control_freq, so realtime
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="show task counts, record nothing")
    args = ap.parse_args()

    if not os.environ.get("MAGICK_HOME"):
        sys.exit("MAGICK_HOME is unset; run: export MAGICK_HOME=\"$(brew --prefix imagemagick)\"\n"
                 "(python-wand loads libMagickWand at import, so this fails before any render)")

    _bench, tasks = load_suite(args.suite)

    if args.list:
        by_axis: dict[str, list] = {}
        for t in tasks:
            by_axis.setdefault(t["axis"], []).append(t)
        print(f"{args.suite}: {len(tasks)} tasks, "
              f"{len({t['base'] for t in tasks})} base tasks")
        for axis in list(VISUAL_AXES) + ["robot_init", "language"]:
            got = by_axis.get(axis, [])
            bases = sorted({t["base_id"] for t in got})
            print(f"  {axis:11s} {len(got):5d} tasks   bases {bases}")
        return

    out_dir = os.path.abspath(args.out_dir)
    jobs: list[tuple[str, dict]] = []  # (filename stem, task)

    if args.task_id is not None:
        t = tasks[args.task_id]
        jobs.append((f"{t['axis']}_base{t['base_id']}_task{t['id']:04d}", t))
    elif args.figure_set:
        used: set[int] = set()
        for axis in VISUAL_AXES:
            pool = [t for t in tasks if t["axis"] == axis and t["base_id"] not in used]
            if not pool:
                sys.exit(f"no unused base task left for axis {axis}")
            t = pool[0]
            used.add(t["base_id"])
            jobs.append((f"{axis}_base{t['base_id']}_task{t['id']:04d}", t))
            ind = {**t, "bddl": t["base"] + ".bddl",
                   "language": f"(unperturbed base of task {t['id']})"}
            jobs.append((f"IND_base{t['base_id']}", ind))
    else:
        axes = [args.axis] if args.axis else VISUAL_AXES
        for axis in axes:
            pool = [t for t in tasks if t["axis"] == axis]
            if not pool:
                sys.exit(f"no tasks on axis {axis!r}")
            for t in pool[: args.n]:
                jobs.append((f"{axis}_base{t['base_id']}_task{t['id']:04d}", t))

    for stem, t in jobs:
        tmp = os.path.join(out_dir, stem + ".mp4")
        ok = record(t, args.steps, args.seed, args.policy, args.action_scale, tmp, args.fps)
        final = os.path.join(out_dir, f"{stem}_{'ok' if ok else 'fail'}.mp4")
        os.replace(tmp, final)
        print(f"{os.path.basename(final):46s}  {t['language']}")

    print(f"\nwrote {len(jobs)} clip(s) to {out_dir}")


if __name__ == "__main__":
    main()
