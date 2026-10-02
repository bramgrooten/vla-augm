#!/usr/bin/env python3
"""Render what the model actually sees under observation augmentation.

Rollouts and evals are never augmented, so no augmented video exists on disk.
This takes real eval frames and pushes them through the same code path the
training forward uses -- [-1, 1] normalization, the real overlay bank, the
same alpha and crop -- so the arms can be eyeballed before trusting them.

Usage:
  python vla_augm/utils/preview_augmentation.py \
      --video logs/<run>/video/eval/seed_1/3.mp4 \
      --out vla_augm/augmentation_preview.png \
      --alpha 0.5 --crop-scale 0.3

Column 1 is the clean frame; the rest are independent draws.
"""

from __future__ import annotations

import argparse
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from rlinf.algorithms.augmentation import OverlayImageBank, RandomOverlay

DEFAULT_BANK = (
    "/projects/PROJECT_ID/users/USER/overlay_banks/imagenet_winter21_10perclass.pt"
)


def detect_grid(height: int, width: int, tile_views: int) -> tuple[int, int, int]:
    """Infer the montage layout, returning (rows, cols, cell_height).

    Eval videos tile every env of ONE env rank via ``tile_images``, which uses
    int(sqrt(N)) rows. So the layout depends on envs-per-rank, not on the total,
    and it has been guessed wrong twice: 100 eval envs over 4 ranks is 25 per
    video (5x5), and an 8-env eval is 2x4.

    Env renders are square, so with ``tile_views`` cameras packed side by side a
    cell is ``tile_views * cell_height`` wide. Search row counts for the one
    that makes both divide exactly.
    """
    # Search plausible render sizes rather than row counts. Scanning rows from 1
    # upward returns the first layout that merely divides evenly -- for a
    # 512x2048 two-camera frame that is "1x2 envs, cell 1024x512", which is
    # arithmetically valid and completely wrong. LIBERO renders at
    # init_params.camera_heights (256), sometimes downscaled, so try those.
    for cell_h in (256, 224, 128, 512, 96, 64):
        cell_w = cell_h * tile_views
        if cell_h <= height and cell_w <= width:
            if height % cell_h == 0 and width % cell_w == 0:
                return height // cell_h, width // cell_w, cell_h

    raise SystemExit(
        f"Could not infer a {tile_views}-camera grid for a {width}x{height} frame; "
        "pass --tile explicitly."
    )


def extract_cell(frame, cell_h: int, cell_w: int, row: int = 0, col: int = 0):
    """Crop one env's cell out of the montage."""
    return frame[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]


def write_video_pair(frames, aug, args) -> int:
    """Write the clean and augmented videos of one env tile, side by side views.

    Mirrors what the recorder now does: each camera is augmented in its own
    frame, never across a concatenation of them.
    """
    import imageio.v3 as iio

    stem = args.out_video[:-4] if args.out_video.endswith(".mp4") else args.out_video
    _rows, _cols, cell_h = args._grid
    cell_w = cell_h * args.tile_views
    clean_frames, augm_frames = [], []

    for frame in frames:
        cell = extract_cell(frame, cell_h, cell_w, row=args.row, col=args.col)
        if args.tile_views == 2:
            half = cell.shape[1] // 2
            cameras = [cell[:, :half], cell[:, half:]]
        else:
            cameras = [cell]
        clean_frames.append(np.concatenate(cameras, axis=1))

        views = [
            torch.from_numpy(v.astype("float32"))
            .div_(127.5)
            .sub_(1.0)
            .permute(2, 0, 1)[None]
            for v in cameras
        ]
        augm_frames.append(
            np.concatenate(
                [
                    ((v[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
                    for v in aug(views)
                ],
                axis=1,
            )
        )

    # Write each clip in one call: the incremental writer silently dropped
    # frames (242 in, 46 out) and left the two clips different lengths.
    # The clean video does not depend on alpha, so rendering one per alpha just
    # produces identical files. --clean-out writes it once, under its own name.
    written = [f"{stem}_augm.mp4"]
    iio.imwrite(f"{stem}_augm.mp4", np.stack(augm_frames), fps=args.fps)
    if args.clean_out:
        clean_path = (
            args.clean_out
            if args.clean_out.endswith(".mp4")
            else f"{args.clean_out}.mp4"
        )
        iio.imwrite(clean_path, np.stack(clean_frames), fps=args.fps)
        written.append(clean_path)
    print(f"wrote {', '.join(written)} ({len(clean_frames)} frames)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="An eval .mp4 from a run")
    parser.add_argument("--out", default="vla_augm/augmentation_preview.png")
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--crop-scale", type=float, default=0.3, help="0 disables crop+flip"
    )
    parser.add_argument("--draws", type=int, default=3, help="Augmented columns")
    parser.add_argument(
        "--row", type=int, default=0, help="Which env row of the montage to take"
    )
    parser.add_argument(
        "--col", type=int, default=0, help="Which env column of the montage to take"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-video",
        default=None,
        help="Write <stem>.mp4 and <stem>_augm.mp4 instead of a still grid",
    )
    parser.add_argument(
        "--sequence",
        default="random",
        help="random-per-angle | random | chunk | episode",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--clean-out",
        default=None,
        help="Where to write the unaugmented video. Omit to skip it: the clean "
        "video is alpha-independent, so it only needs writing once per source.",
    )
    parser.add_argument(
        "--tile-views",
        type=int,
        default=1,
        choices=(1, 2),
        help="Cameras packed inside one tile of the montage. 1 for videos "
        "recorded before views:[main,wrist] existed (a tile is one camera); "
        "2 when a tile is main|wrist.",
    )
    parser.add_argument(
        "--bank-shards",
        type=int,
        default=32,
        help="Load 1/N of the bank. A preview needs variety, not full class coverage; "
        "the full 26 GB bank would be slow to page in for a few hundred frames.",
    )
    parser.add_argument("--size", type=int, default=224, help="Model resolution")
    args = parser.parse_args()

    frames = iio.imread(args.video, plugin="pyav")
    rows, cols, cell_h = detect_grid(frames.shape[1], frames.shape[2], args.tile_views)
    print(
        f"video {frames.shape} -> {rows}x{cols} envs, cell "
        f"{cell_h * args.tile_views}x{cell_h}px, {args.tile_views} camera(s)",
        flush=True,
    )
    args._grid = (rows, cols, cell_h)

    bank = OverlayImageBank.from_cache(
        args.bank, shard_index=0, num_shards=max(args.bank_shards, 1)
    )
    from rlinf.algorithms.augmentation import AugmentationSequence

    aug = RandomOverlay(
        bank=bank,
        alpha=args.alpha,
        crop_scale=args.crop_scale if args.crop_scale > 0 else None,
        sequence=AugmentationSequence(args.sequence),
        period=5,
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.out_video:
        return write_video_pair(frames, aug, args)

    rows = []
    for fraction in (0.05, 0.3, 0.55, 0.8):
        rows, cols, cell_h = args._grid
        frame = extract_cell(
            frames[int(len(frames) * fraction)],
            cell_h,
            cell_h * args.tile_views,
            args.row,
            args.col,
        )
        img = np.array(
            Image.fromarray(frame).resize((args.size, args.size), Image.BILINEAR)
        )
        # uint8 -> [-1, 1] NCHW, exactly as Observation.from_dict does.
        x = torch.from_numpy(img).float().div(255.0).mul(2).sub(1).permute(2, 0, 1)[None]

        columns = [img]
        for _ in range(args.draws):
            out = aug([x.clone()])[0][0]
            columns.append(
                ((out.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
            )
        rows.append(np.concatenate(columns, axis=1))

    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(args.out)
    print(f"wrote {args.out} {grid.shape} (col 1 clean, rest augmented)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
