#!/usr/bin/env python
"""Rename the video-reel clips to carry their axis and task id.

RecordVideo names clips ``seed_<n>/step<k>.mp4`` -- the flush counter, nothing
about what is in them. The eval runs one env for fifty rollout epochs and
flushes once per epoch, so step<k> is the k-th episode, and the k-th row of the
per-episode dump names its task. This turns that into

    <axis>_base<b>_task<id>_<ok|fail>.mp4        (perturbed)
    IND_base<b>_<ok|fail>.mp4                    (unperturbed twin)

The base index is in both names so a pair is obvious at a glance: every
``textures_base0_*`` clip is the OOD partner of ``IND_base0_*``.

and refuses to rename anything if the two sides disagree, because a mislabelled
clip in a paper figure is worse than no clip.

Usage:
    python vla_augm/analyze/name_video_clips.py \
        --videos /home/USER/rlinf_videos/liberoplus_critic_a05_ep300 \
        --episodes vla_augm/results/video_reel_episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SELECTION = REPO / "vla_augm/results/video_task_selection.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=pathlib.Path, required=True)
    parser.add_argument("--selection", type=pathlib.Path, default=SELECTION)
    parser.add_argument(
        "--ind",
        action="store_true",
        help="name the unperturbed twins instead; episode task ids are base indices",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes = [json.loads(l) for l in open(args.episodes) if l.strip()]
    selection = json.loads(args.selection.read_text())
    axis_of = {int(r["task_id"]): r["axis"] for r in selection}
    base_of = {int(r["task_id"]): int(r["base_task"]) for r in selection}

    # One flush per episode, so the clip count must equal the episode count. A
    # mismatch means the one-env assumption broke (extra envs, or an epoch that
    # fitted two episodes) and the positional mapping is meaningless.
    clips = sorted(
        args.videos.rglob("step*.mp4"),
        key=lambda p: (p.parent.name, int(re.search(r"step(\d+)", p.stem).group(1))),
    )
    if len(clips) != len(episodes):
        sys.exit(
            f"{len(clips)} clips but {len(episodes)} episodes -- refusing to rename. "
            "Expected one clip per episode (single env, one flush per epoch)."
        )

    if not args.ind:
        unknown = {int(e["task_id"]) for e in episodes} - set(axis_of)
        if unknown:
            sys.exit(f"episodes contain task ids outside the selection: {sorted(unknown)}")

    for clip, ep in zip(clips, episodes):
        task_id = int(ep["task_id"])
        ok = float(ep.get("success_once", 0.0)) > 0.5
        if args.ind:
            # In a standard-suite eval the task id IS the base index.
            name = f"IND_base{task_id}_{'ok' if ok else 'fail'}.mp4"
        else:
            name = (
                f"{axis_of[task_id]}_base{base_of[task_id]}"
                f"_task{task_id:04d}_{'ok' if ok else 'fail'}.mp4"
            )
        target = clip.parent / name
        print(f"{clip.name:>14}  ->  {name}")
        if not args.dry_run:
            clip.rename(target)

    n_ok = sum(float(e.get("success_once", 0.0)) > 0.5 for e in episodes)
    print(f"\n{len(clips)} clips: {n_ok} success, {len(clips) - n_ok} failure")


if __name__ == "__main__":
    main()
