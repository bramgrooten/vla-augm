#!/usr/bin/env python3
"""Rebuild the "what the policy actually receives" figure by rendering locally.

Local counterpart to `build_policyview` in `make_augmentation_figures.py`, which
pulls the frame out of a cluster clip. That clip has the recorder's HUD burned
into it, and the HUD sits inside the observation, so the cluster version has to
either crop it away (changing the framing, which is the whole point of the
figure) or leave the text in and apologise for it in the caption. Rendering here
sidesteps the choice: nothing is overlaid, so the frames are both uncropped and
clean.

    export MAGICK_HOME="$(brew --prefix imagemagick)"      # see vla_augm/install.md
    python vla_augm/analyze/make_policyview_local.py

Writes `policyview_roll<N>_{main,wrist}.png` at 224x224 and leaves the cluster's
`policyview_{main,wrist}.png` alone, so the two can sit side by side.

Both cameras come from a single observation of an unperturbed base task, so the
pair really is one moment seen two ways rather than two separate renders. The
default is base 0 after a 20-step random rollout -- the same scene, seed and
action sequence as the top-left panel of the perturbation-axes figure, so the
base view here is that panel with a wrist view added. The rollout matters more
for the wrist camera than the base one: parked in its home pose the gripper
stares at empty table. Vary it with `--rollout`; every value shares one seed, so
the rollouts are prefixes of each other rather than unrelated trajectories.

`--bank` additionally writes an augmented copy of both views, blended with a
random overlay the way training does. It runs the real `RandomOverlay` from
`rlinf/algorithms/augmentation.py` rather than reimplementing the blend, so the
figure cannot drift from what the policy is actually trained on. Get a bank small
enough to hold locally with `vla_augm/utils/sample_overlay_bank.py`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from make_env_figure_local import DEFAULT_OUT, SETTLE, render_obs  # noqa: E402
from record_libero_videos import load_suite  # noqa: E402

# The resolution both models resize to: DEFAULT_IMAGE_SIZE in
# rlinf/algorithms/augmentation.py. Rendering happens at CAM_H/CAM_W (256) and is
# scaled down here, matching the cluster script's crop-then-scale.
POLICY_SIZE = 224

CAMERAS = {"main": "agentview_image", "wrist": "robot0_eye_in_hand_image"}

# Defaults from the training configs (vla_augm/config/libero/*_augm_*.yaml), so the
# figure shows the augmentation the experiments actually ran:
#   alpha 0.5        SVEA and MaDi both use it
#   crop_scale 0.3   per-draw random resized crop + flip of the overlay
#   sequence random  one bank image shared by both views, cropped separately
ALPHA = 0.5
CROP_SCALE = 0.3


def augment(views: dict, bank_path: str, alpha: float, seed: int) -> tuple[dict, int, "np.ndarray"]:
    """Blend a random overlay into each view, exactly as training does.

    Calls into `rlinf.algorithms.augmentation` rather than reimplementing
    `(1 - alpha) * x + alpha * overlay`: the blend is easy to write and easy to
    get subtly wrong (it runs in [-1, 1], not [0, 1], and the overlay is
    randomly cropped and flipped first), and a figure that quietly diverges from
    the training code would be worse than no figure.
    """
    import logging
    import types

    import torch

    # `augmentation.py` calls `get_logger()` at import time, which reaches into
    # the Ray worker for the per-worker logger. Ray is a training dependency and
    # is not in the Mac venv, so stub that one function out. The module has no
    # other tie to the training stack, and stubbing here rather than installing
    # Ray keeps the local venv lightweight.
    stub = types.ModuleType("rlinf.utils.logging")
    stub.get_logger = lambda: logging.getLogger("augmentation")
    sys.modules.setdefault("rlinf.utils.logging", stub)

    # Load the file directly instead of `from rlinf.algorithms.augmentation
    # import ...`: the package __init__ eagerly imports its siblings, which pull
    # in Ray again. `augmentation.py` itself needs only torch and the logger, so
    # by-path loading gets the real code with none of the stack behind it.
    import importlib.util

    src = os.path.join(REPO, "rlinf", "algorithms", "augmentation.py")
    spec = importlib.util.spec_from_file_location("_augmentation", src)
    aug_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aug_mod)
    AugmentationSequence = aug_mod.AugmentationSequence
    OverlayImageBank = aug_mod.OverlayImageBank
    RandomOverlay = aug_mod.RandomOverlay

    bank = OverlayImageBank.from_cache(bank_path)
    gen = torch.Generator().manual_seed(seed)
    aug = RandomOverlay(bank, alpha=alpha, crop_scale=CROP_SCALE,
                        sequence=AugmentationSequence.RANDOM, generator=gen)

    names = list(views)
    # uint8 -> float [-1, 1], the range the model sees and the blend runs in.
    batch = [torch.from_numpy(views[n].copy()).float().div(127.5).sub(1.0).unsqueeze(0)
             for n in names]
    out = aug(batch)
    # Read the drawn index off the augmenter rather than re-deriving it from the
    # seed: under sequence=RANDOM one index is shared by every view and cached
    # here, so this is the image that was actually blended, not a guess that
    # could drift if the sampling changes.
    index = int(aug._held_indices[0])
    source = bank.images[index].numpy()
    augmented = {n: ((t[0] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
                 for n, t in zip(names, out)}
    return augmented, index, source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--base", type=int, default=0, help="which base task to render")
    ap.add_argument("--settle", type=int, default=SETTLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=20,
                    help="steps of the random policy to run after settling (0 = home pose)")
    ap.add_argument("--size", type=int, default=POLICY_SIZE)
    ap.add_argument("--tag", help="filename tag: policyview_<tag>_<cam>.png "
                    "(default: roll<rollout>)")
    ap.add_argument("--bank", help="overlay bank .pt; also writes an augmented copy of "
                    "each view as policyview_<tag>_aug_<cam>.png")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="overlay blend weight")
    ap.add_argument("--aug-seed", type=int, default=0,
                    help="which overlay is drawn, and how it is cropped")
    args = ap.parse_args()

    if not os.environ.get("MAGICK_HOME"):
        sys.exit('MAGICK_HOME is unset; run: export MAGICK_HOME="$(brew --prefix imagemagick)"')

    _bench, tasks = load_suite(args.suite)
    pool = [t for t in tasks if t["base_id"] == args.base]
    if not pool:
        sys.exit(f"no task with base {args.base}")
    task = {**pool[0], "bddl": pool[0]["base"] + ".bddl"}

    tag = args.tag or f"roll{args.rollout}"
    obs = render_obs(task, args.settle, args.seed, args.rollout)

    from PIL import Image

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"base{args.base}  {task['bddl'][-46:]}\n  \"{task['language']}\"")

    # Resize first, then augment: the policy resizes to 224 before the blend, so
    # augmenting at 256 would show overlay detail the model never receives.
    # BILINEAR, not a nearest-neighbour resize, for the same reason -- the
    # aliasing would be an artefact of the figure, not of the pipeline.
    views = {cam: np.asarray(Image.fromarray(np.asarray(obs[key]))
                             .resize((args.size, args.size), Image.BILINEAR))
             for cam, key in CAMERAS.items()}
    groups = {"": views}
    overlay_index = None
    if args.bank:
        groups["aug_"], overlay_index, source = augment(
            views, args.bank, args.alpha, args.aug_seed)
        # The overlay before the per-view crop and flip. Neither augmented panel
        # shows exactly this, which is the point: one source image, two crops.
        groups["overlay"] = {"": source}

    for prefix, group in groups.items():
        for cam, img in group.items():
            path = os.path.join(out_dir, f"policyview_{tag}_{prefix}{cam}.png")
            Image.fromarray(img).save(path)
            print(f"  {prefix or 'raw_':4s} {cam:5s} {args.size}x{args.size}  "
                  f"brightness {img.mean():5.1f}  {os.path.basename(path)}")

    if args.bank:
        # A near-zero difference means the blend silently did nothing.
        for cam in views:
            d = np.abs(views[cam].astype(float)
                       - groups["aug_"][cam].astype(float)).mean()
            print(f"  mean|clean-aug| {cam:5s} {d:6.2f}")
        print(f"  overlay: bank image {overlay_index} of {args.bank}")
    print(f"\nwrote to {out_dir}")


if __name__ == "__main__":
    main()
