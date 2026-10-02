#!/usr/bin/env python3
"""Regenerate the appendix figure that shows what the OOD perturbations look like.

One row per LIBERO-Plus axis: the same base task in distribution on the left and
perturbed on the right, so a reader can see what "camera viewpoints" or "sensor
noise" actually did to the observation.

    python vla_augm/analyze/make_env_figure.py
    python vla_augm/analyze/make_env_figure.py --list      # what is available, pick nothing
    python vla_augm/analyze/make_env_figure.py --frame 20  # later in the episode

Writes one PNG per row (env_row_<axis>.png) rather than a single montage, so the
row labels and column headers live in main.tex and can be restyled without
re-rendering. It prints the exact clips it chose: paste that into the commit
message or the caption if you need the figure to be auditable later.

WHY A MATCHING AND NOT A FIXED LIST. Every row must use a different base task --
five rows of the same tabletop would show the perturbations against an identical
background and look like a rendering bug rather than a benchmark. The clips do
not cover every (axis, base) combination, so the pairing is solved rather than
hardcoded: pick one base per axis such that no base repeats and an IND clip of
that base exists. Hardcoding a list breaks silently the moment the video set is
re-recorded, which it already has been once (round2 has different coverage from
the first run).

GEOMETRY. Clips are 512x256: main|wrist at 256x256, so x=0..255 is the main
camera, which is the view the perturbations are visible in. The recorder burns a
HUD onto the render -- reward and termination across the top, the instruction
across the bottom -- which is not part of the observation. The default crop drops
both bands. Note the instruction also carries the perturbation as a suffix
("... place it on the plate light 10"), so leaving the bottom band in would
label the rows for you, but it would also put a raw task string in a figure.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VIDEOS = os.path.join(
    REPO, "..", "paper", "videos", "liberoplus_critic_a05_ep300_round2", "seed_0"
)
DEFAULT_OUT = os.path.join(REPO, "..", "paper", "vla_augm", "figures")

# Filename axis -> the name the paper uses for it, in the order the paper lists
# them (Section 3, and the columns of Table 1).
AXES = [
    ("textures", "background textures"),
    ("lighting", "light conditions"),
    ("layout", "objects layout"),
    ("camera", "camera viewpoints"),
    ("noise", "sensor noise"),
]

IND_RE = re.compile(r"^IND_base(\d+)_(ok|fail)\.mp4$")
OOD_RE = re.compile(r"^([a-z]+)_base(\d+)_task(\d+)_(ok|fail)\.mp4$")


def scan(video_dir: str) -> tuple[dict[int, str], dict[str, dict[int, list[str]]]]:
    """Index the clips by base task, and by (axis, base task)."""
    if not os.path.isdir(video_dir):
        sys.exit(f"no such video directory: {video_dir}")
    ind: dict[int, str] = {}
    ood: dict[str, dict[int, list[str]]] = {}
    for name in sorted(os.listdir(video_dir)):
        m = IND_RE.match(name)
        if m:
            # Prefer a successful episode: a failure still shows the scene, but a
            # reader who notices the arm never finishes will wonder whether that
            # is the point of the figure.
            base, status = int(m.group(1)), m.group(2)
            if base not in ind or status == "ok":
                ind[base] = name
            continue
        m = OOD_RE.match(name)
        if m:
            axis, base, _task, status = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            ood.setdefault(axis, {}).setdefault(base, [])
            (ood[axis][base].insert(0, name) if status == "ok"
             else ood[axis][base].append(name))
    return ind, ood


def solve(ind, ood, axes, forced=None) -> dict[str, tuple[int, str]] | None:
    """One base task per axis, no base reused, IND clip must exist.

    Backtracking, not greedy: with five axes and a handful of shared base tasks,
    a greedy pass can consume the only base an later axis could have used.
    Axes are tried most-constrained first so the search settles immediately.
    """
    forced = dict(forced or {})
    order = sorted(axes, key=lambda a: len(set(ood.get(a, {})) & set(ind)))

    def search(i: int, used: set[int], out: dict) -> dict | None:
        if i == len(order):
            return dict(out)
        axis = order[i]
        for base in sorted(set(ood.get(axis, {})) & set(ind)):
            if base in used:
                continue
            out[axis] = (base, ood[axis][base][0])
            got = search(i + 1, used | {base}, out)
            if got:
                return got
            del out[axis]
        return None

    return search(0, {b for b, _ in forced.values()}, dict(forced))


def grab(video: str, frame: int, crop: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video,
         "-vf", f"select='eq(n\\,{frame})',{crop}",
         "-frames:v", "1", "-vsync", "0", out],
        check=True,
    )
    if not os.path.exists(out):
        sys.exit(f"ffmpeg produced no frame {frame} from {video}; is the clip shorter?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=DEFAULT_VIDEOS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--frame", type=int, default=5,
                    help="frame index to grab; early enough that the arm has not "
                         "yet occluded the scene (default: 5)")
    ap.add_argument("--crop", default="crop=256:168:0:40",
                    help="main camera, HUD bands removed")
    ap.add_argument("--gap", type=int, default=6, help="white pixels between the two columns")
    ap.add_argument("--pick", action="append", default=[], metavar="AXIS=SUBSTRING",
                    help="force an axis to use a particular clip, e.g. "
                         "--pick textures=task0257. Repeatable; the solver assigns "
                         "the remaining axes around it.")
    ap.add_argument("--list", action="store_true", help="print coverage and exit")
    args = ap.parse_args()

    video_dir = os.path.abspath(args.videos)
    ind, ood = scan(video_dir)

    if args.list:
        print(f"{video_dir}\n")
        print("IND clips:      " + " ".join(f"base{b}" for b in sorted(ind)))
        for axis, _label in AXES:
            bases = sorted(ood.get(axis, {}))
            paired = [b for b in bases if b in ind]
            print(f"{axis:<10}bases: {' '.join('base%d' % b for b in bases) or '(none)'}"
                  f"   with an IND clip: {' '.join('base%d' % b for b in paired) or '(none)'}")
        return

    missing = [a for a, _ in AXES if a not in ood]
    if missing:
        sys.exit(f"no clips for axis/axes: {', '.join(missing)}. Run --list to see coverage.")

    forced: dict[str, tuple[int, str]] = {}
    for spec in args.pick:
        if "=" not in spec:
            sys.exit(f"--pick wants AXIS=SUBSTRING, got {spec!r}")
        axis, needle = spec.split("=", 1)
        hits = [(b, n) for b, names in ood.get(axis, {}).items()
                for n in names if needle in n]
        if not hits:
            sys.exit(f"--pick {spec}: no {axis} clip matching {needle!r}")
        if len({b for b, _ in hits}) > 1:
            sys.exit(f"--pick {spec}: matches several base tasks; be more specific")
        base, name = sorted(hits)[0]
        if base not in ind:
            sys.exit(f"--pick {spec}: no IND clip for base{base}, so the row has no left half")
        forced[axis] = (base, name)

    chosen = solve(ind, ood, [a for a, _ in AXES if a not in forced], forced)
    if chosen is None:
        sys.exit("cannot give every axis its own base task with the clips available.\n"
                 "Run --list: some axis probably shares its only base task with another.")

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, ".env_tmp")
    os.makedirs(work, exist_ok=True)
    print(f"{'axis':<12}{'base':<7}{'in distribution':<26}perturbed")
    for axis, _label in AXES:
        base, ood_clip = chosen[axis]
        ind_clip = ind[base]
        left = os.path.join(work, f"{axis}_l.png")
        right = os.path.join(work, f"{axis}_r.png")
        grab(os.path.join(video_dir, ind_clip), args.frame, args.crop, left)
        grab(os.path.join(video_dir, ood_clip), args.frame, args.crop, right)
        out = os.path.join(args.out_dir, f"env_row_{axis}.png")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", left, "-i", right,
             "-filter_complex",
             f"[0]pad=iw+{args.gap}:ih:0:0:white[p];[p][1]hstack=inputs=2[o]",
             "-map", "[o]", out],
            check=True,
        )
        print(f"{axis:<12}{base:<7}{ind_clip:<26}{ood_clip}")

    for f in os.listdir(work):
        os.remove(os.path.join(work, f))
    os.rmdir(work)
    print(f"\nwrote env_row_*.png to {os.path.abspath(args.out_dir)}")
    print(f"frame {args.frame}, {args.crop}")


if __name__ == "__main__":
    main()
