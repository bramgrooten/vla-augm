#!/usr/bin/env python3
"""Pull a handful of overlay images out of a built bank, small enough to copy.

The banks `build_overlay_bank.py` writes are tens of GB, which is fine on a
cluster filesystem and hopeless over scp. This takes a seeded random sample and
writes it in the same format -- a bare uint8 [N, 224, 224, 3] tensor -- so
`OverlayImageBank.from_cache` loads the sample exactly like the full bank, and a
figure built from it uses the real augmentation code path.

On the cluster (any venv with torch; on Cluster A the pi one lives outside the
repo, under ~/rlinf_venvs/):

    source ~/rlinf_venvs/.venv-pi/bin/activate
    cd ~/rl_inf && git pull
    python vla_augm/utils/sample_overlay_bank.py --n 16 --out ~/overlay_sample.pt

The bank is looked up in the known `overlay_banks/` locations (project space
first, since scratch-shared is purged); `--bank` overrides it.

Then, from a laptop:

    scp cluster_a:~/overlay_sample.pt vla_augm/assets/overlay_sample.pt

Sixteen images is ~2.4 MB, small enough to commit, and this repo is private. The
sample lives in `vla_augm/assets/` so the augmented figures can be regenerated
without cluster access -- the seed alone would not be enough, since it indexes
into a bank that only exists on the cluster. Do not copy these images into anything
public: they are ImageNet crops.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

BANK_NAME = "imagenet_winter21_10perclass.pt"

# Where banks have lived, newest first. The checkout moved from scratch to home
# and the banks then moved to project space, so a single hardcoded default just
# produces a "no bank" error at the moment the user is furthest from the code.
# Project space is listed first because scratch-shared is purged periodically --
# which is how the earlier bank was lost.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANK_DIRS = [
    "/projects/PROJECT_ID/users/USER/overlay_banks",
    os.path.join(REPO, "overlay_banks"),
    "/scratch/USER/rl_inf/overlay_banks",
]


def default_bank() -> str:
    """First known bank location that actually holds the bank."""
    for directory in BANK_DIRS:
        candidate = os.path.join(directory, BANK_NAME)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(BANK_DIRS[0], BANK_NAME)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=None, help=f"default: {BANK_NAME} in the "
                    "first of " + ", ".join(BANK_DIRS) + " that has it")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bank_path = args.bank or default_bank()
    if not os.path.isfile(bank_path):
        # The bank has moved three times, so say what is actually on disk rather
        # than leaving a bare FileNotFoundError to be deciphered.
        print(f"no bank at {bank_path}", file=sys.stderr)
        for directory in BANK_DIRS:
            found = sorted(f for f in os.listdir(directory)
                           if f.endswith(".pt")) if os.path.isdir(directory) else None
            print(f"  {directory}: "
                  f"{', '.join(found) if found else '(missing)' if found is None else '(no .pt)'}",
                  file=sys.stderr)
        print("pass the right one with --bank", file=sys.stderr)
        return 1

    # mmap, so this pages in only the rows it samples instead of materialising
    # the whole bank -- the same reason from_cache uses it.
    bank = torch.load(bank_path, map_location="cpu", mmap=True)
    if bank.ndim != 4 or bank.shape[-1] != 3 or bank.dtype != torch.uint8:
        print(f"unexpected bank format: {tuple(bank.shape)} {bank.dtype}", file=sys.stderr)
        return 1
    if args.n > bank.shape[0]:
        print(f"bank has only {bank.shape[0]} images", file=sys.stderr)
        return 1

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(bank.shape[0], generator=g)[: args.n]
    sample = bank[idx.sort().values].clone()  # sorted: fewer random seeks over mmap
    torch.save(sample, args.out)
    print(f"{bank.shape[0]} images in bank -> {args.n} sampled (seed {args.seed})")
    print(f"wrote {args.out}  ({sample.numel() / 1e6:.1f} MB, {tuple(sample.shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
