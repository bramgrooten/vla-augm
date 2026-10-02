#!/usr/bin/env python3
"""Build an overlay image bank for random-overlay augmentation.

Decodes a sample of JPEGs into a single uint8 tensor [N, size, size, 3] so
training never touches the source dataset -- one .pt load instead of N JPEG
decodes per run, and no dependency on a shared filesystem staying mounted.

Sources on Cluster A (see vla_augm/install.md):
  COCO      /scratch-nvme/ml-datasets/coco/train/data           (open)
  ImageNet  /scratch-nvme/ml-datasets/imagenet21k/tars          (imagenet_access
  ImageNet  /scratch-nvme/ml-datasets/imagenet                  group)

Note the directory named ``imagenet21k`` holds the **Winter21** release --
19,167 synsets, ~13.0M images -- not the 21,841-synset fall11 release that
"ImageNet-21k" usually denotes; winter21 dropped the person subtree. Cite it
as Winter21. ``imagenet`` is ILSVRC2012: 1,000 classes, 1,281,167 train /
50,000 val / 100,000 test.

Two source layouts:

  --source DIR       a directory tree of image files (COCO). Every image is
                     pooled and sampled uniformly.
  --tar-source DIR   a directory of per-class tars (Winter21 ships 19,167 of
                     them, one per WordNet synset). Classes are sampled
                     without replacement and one image is drawn from each, so
                     a 2048-image bank spans 2048 distinct classes rather than
                     over-representing whichever classes happen to be large.

Usage:
  python vla_augm/utils/build_overlay_bank.py \
      --tar-source /scratch-nvme/ml-datasets/imagenet21k/tars \
      --out overlay_banks/imagenet21k_2048.pt \
      --num-images 2048

  # a small bank, e.g. for a figure: opens 16 tars, not 19,167
  python vla_augm/utils/build_overlay_bank.py \
      --tar-source /scratch-nvme/ml-datasets/imagenet21k/tars \
      --out overlay_banks/imagenet_winter21_16.pt \
      --num-classes 16

REPRODUCIBILITY. Which images land in a bank is fixed by --seed: the tar list is
sorted before the seeded shuffle, and each class's own draw is seeded by its
position in that list. A full --per-class sweep reproduces the same *set* of
images but not their row order, because it collects results with
imap_unordered; --num-classes is serial and so reproduces the tensor exactly,
rows included. (The .pt files still differ byte for byte -- torch.save writes a
zip whose entries carry timestamps -- but the tensors compare equal.) Row order
only matters if you need a specific index to mean the same image again; the
augmentation itself draws uniformly either way.

ImageNet needs the imagenet_access group. Group membership is fixed at login,
so an already-open shell will not see a fresh grant -- run under
`sg imagenet_access -c '...'` or log in again.

Nothing downstream cares which dataset a bank came from; swapping sources is
purely a rebuild.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import random
import sys
import tarfile
from typing import Iterator

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG")
# How far into a class tar to look for the image to draw. Reading a member
# header is cheap (tarfile seeks over the data), but scanning a whole archive
# is not, so cap it: within-class identity matters far less than class spread.
MAX_TAR_SCAN = 64

# Arachnid synsets, excluded from preview banks on request -- nobody wants a
# spider filling the screen while debugging. Best-effort: the WordNet arachnid
# subtree is larger than this, so it covers the common ones rather than all.
# Not excluded from training banks, where the overlay content is irrelevant.
ARACHNID_SYNSETS = (
    "n01769347", "n01770081", "n01770393", "n01772222", "n01772664",
    "n01773157", "n01773549", "n01773797", "n01774384", "n01774750",
    "n01775062", "n01775370", "n01776192", "n01776313", "n01776705",
    "n01777304", "n01777467", "n01777649", "n01777909", "n01778217",
    "n01778487", "n01778621", "n01778801", "n01779148", "n01779629",
    "n01779939", "n01780142", "n01780426", "n01780696", "n01781071",
    "n01781570", "n01781875", "n01782209", "n01782516", "n01782650",
    "n01782969", "n01783017", "n01783236", "n01783384",
)


def find_images(source: str, limit: int, seed: int) -> list[str]:
    """Sample `limit` image paths from `source`, recursing into subdirectories.

    Collect, shuffle with a fixed seed, truncate -- so the same seed always
    yields the same bank.
    """
    paths = []
    for root, _dirs, files in os.walk(source):
        for name in files:
            if name.endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, name))

    if not paths:
        raise SystemExit(f"No images found under {source}")

    random.Random(seed).shuffle(paths)
    return paths[:limit]


def iter_dir_images(source: str, limit: int, seed: int) -> Iterator[tuple[str, object]]:
    """Yield ``(name, path)`` for images sampled uniformly from a directory."""
    for path in find_images(source, limit, seed):
        yield path, path


def iter_tar_images(
    tar_dir: str, limit: int, seed: int
) -> Iterator[tuple[str, object]]:
    """Yield ``(name, file object)``, one image per randomly chosen class tar.

    ImageNet-21k ships one tar per WordNet synset. Sampling classes without
    replacement and taking a single image from each maximises class spread:
    a 2048-image bank covers 2048 of the 19,168 synsets, instead of being
    dominated by whichever classes hold the most photos. If more images than
    classes are requested, classes are revisited in further passes.
    """
    tars = sorted(f for f in os.listdir(tar_dir) if f.endswith(".tar"))
    if not tars:
        raise SystemExit(f"No .tar files in {tar_dir}")

    rng = random.Random(seed)
    rng.shuffle(tars)
    print(f"  {len(tars)} class tars available; drawing from {min(limit, len(tars))}")

    yielded = 0
    while yielded < limit:
        for name in tars:
            if yielded >= limit:
                return
            path = os.path.join(tar_dir, name)
            try:
                with tarfile.open(path, "r:") as tar:
                    members = []
                    for member in tar:
                        if member.isfile() and member.name.endswith(IMAGE_EXTENSIONS):
                            members.append(member)
                        if len(members) >= MAX_TAR_SCAN:
                            break
                    if not members:
                        continue
                    member = rng.choice(members)
                    data = tar.extractfile(member)
                    if data is None:
                        continue
                    # Read eagerly: the handle dies with the tar context.
                    yield f"{name}:{member.name}", io.BytesIO(data.read())
                    yielded += 1
            except Exception as exc:  # noqa: BLE001 - a bad archive is not fatal
                print(f"skip tar {name}: {exc}", file=sys.stderr)


def _decode(handle, size: int) -> "np.ndarray":
    """Decode one image to a `size`x`size` RGB centre crop."""
    with Image.open(handle) as img:
        img = img.convert("RGB")
        # Resize the short side, then centre crop: preserves aspect ratio, so
        # overlays are not distorted. Random cropping happens at draw time.
        width, height = img.size
        scale = size / min(width, height)
        img = img.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.BILINEAR,
        )
        width, height = img.size
        left, top = (width - size) // 2, (height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        return np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(size, size, 3)


def _decode_tar(args: tuple[str, int, int, int]) -> "np.ndarray":
    """Decode up to `per_class` images from one class tar. Runs in a worker.

    Returns a stacked ``[k, size, size, 3]`` array (possibly empty). Opening a
    tar dominates the cost, so drawing several images per open is what makes a
    large bank affordable.
    """
    path, per_class, size, seed = args
    try:
        with tarfile.open(path, "r:") as tar:
            members = []
            for member in tar:
                if member.isfile() and member.name.endswith(IMAGE_EXTENSIONS):
                    members.append(member)
                if len(members) >= MAX_TAR_SCAN:
                    break
            if not members:
                return np.empty((0, size, size, 3), dtype=np.uint8)

            rng = random.Random(seed)
            chosen = rng.sample(members, min(per_class, len(members)))
            decoded = []
            for member in chosen:
                data = tar.extractfile(member)
                if data is None:
                    continue
                try:
                    decoded.append(_decode(io.BytesIO(data.read()), size))
                except Exception:  # noqa: BLE001 - a corrupt image is not fatal
                    continue
            if not decoded:
                return np.empty((0, size, size, 3), dtype=np.uint8)
            return np.stack(decoded)
    except Exception as exc:  # noqa: BLE001 - a bad archive is not fatal
        print(f"skip tar {os.path.basename(path)}: {exc}", file=sys.stderr)
        return np.empty((0, size, size, 3), dtype=np.uint8)


def build_from_tars(
    tar_dir: str,
    per_class: int,
    size: int,
    seed: int,
    workers: int,
    exclude: frozenset = frozenset(),
) -> torch.Tensor:
    """Build a bank by drawing `per_class` images from every class tar.

    Parallel over tars: a single process manages roughly 4 images/s because
    each tar open costs more than the decode, and 19k opens is the whole job.
    """
    tars = sorted(f for f in os.listdir(tar_dir) if f.endswith(".tar"))
    if not tars:
        raise SystemExit(f"No .tar files in {tar_dir}")
    if exclude:
        before = len(tars)
        tars = [t for t in tars if t[: -len(".tar")] not in exclude]
        print(f"  excluded {before - len(tars)} synset(s)")
    print(f"  {len(tars)} class tars, {per_class} image(s) each, {workers} workers")

    jobs = [
        (os.path.join(tar_dir, name), per_class, size, seed + i)
        for i, name in enumerate(tars)
    ]
    chunks, kept, done = [], 0, 0
    with multiprocessing.Pool(workers) as pool:
        for arr in pool.imap_unordered(_decode_tar, jobs, chunksize=8):
            done += 1
            if arr.shape[0]:
                chunks.append(arr)
                kept += arr.shape[0]
            if done % 1000 == 0:
                print(f"  {done}/{len(tars)} tars, {kept} images", flush=True)

    if not kept:
        raise SystemExit("No images decoded successfully")
    print(f"  stacking {kept} images ...", flush=True)
    return torch.from_numpy(np.concatenate(chunks))


def build(sources: Iterator[tuple[str, object]], total: int, size: int) -> torch.Tensor:
    """Decode and center-crop to `size`, returning uint8 [N, size, size, 3]."""
    out = torch.empty((total, size, size, 3), dtype=torch.uint8)
    kept = 0
    for path, handle in sources:
        if kept >= total:
            break
        try:
            with Image.open(handle) as img:
                img = img.convert("RGB")
                # Resize the short side, then center crop: preserves aspect
                # ratio, so overlays are not distorted.
                width, height = img.size
                scale = size / min(width, height)
                img = img.resize(
                    (max(size, round(width * scale)), max(size, round(height * scale))),
                    Image.BILINEAR,
                )
                width, height = img.size
                left, top = (width - size) // 2, (height - size) // 2
                img = img.crop((left, top, left + size, top + size))
                # bytearray (not bytes) so torch gets a writable buffer.
                out[kept] = torch.frombuffer(
                    bytearray(img.tobytes()), dtype=torch.uint8
                ).reshape(size, size, 3)
            kept += 1
        except Exception as exc:  # noqa: BLE001 - a few corrupt files are fine
            print(f"skip {path}: {exc}", file=sys.stderr)

        if kept and kept % 250 == 0:
            print(f"  decoded {kept}/{total}", flush=True)

    if kept == 0:
        raise SystemExit("No images decoded successfully")
    # Avoid the clone in the common case: a full-size bank is GBs, and cloning
    # would briefly double peak RSS for no reason.
    return out if kept == total else out[:kept].clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Directory tree of image files (COCO)")
    group.add_argument(
        "--tar-source",
        help="Directory of per-class tars (ImageNet-21k); samples classes randomly",
    )
    parser.add_argument("--out", required=True, help="Destination .pt file")
    parser.add_argument(
        "--num-images", type=int, default=2048, help="--source mode only"
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="--tar-source mode: images drawn from each of the 19,167 classes",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="--tar-source mode: decode processes"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="--tar-source mode: build a small bank from this many classes, one "
        "image each, instead of sweeping all 19,167. Opens only the tars it "
        "needs, so a preview bank takes seconds rather than hours, and the "
        "result is byte-reproducible (see --seed).",
    )
    parser.add_argument(
        "--no-arachnids",
        action="store_true",
        help="Drop spider/scorpion/tick synsets. For preview banks used in "
        "debugging videos; training banks do not need it.",
    )
    parser.add_argument(
        "--size", type=int, default=224, help="Model image resolution (openpi: 224)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Fixes which images land in the bank"
    )
    args = parser.parse_args()

    if args.tar_source and args.num_classes:
        # Serial, and reproducible: the tar list is sorted before the seeded
        # shuffle and images land in draw order, so two builds give an identical
        # tensor. The full sweep below cannot promise that -- it collects with
        # imap_unordered, so a rebuild holds the same images at different rows.
        print(f"Sampling {args.num_classes} classes from {args.tar_source} ...",
              flush=True)
        sources = iter_tar_images(args.tar_source, args.num_classes, args.seed)
        bank = build(sources, args.num_classes, args.size)
    elif args.tar_source:
        print(f"Sampling classes from {args.tar_source} ...", flush=True)
        exclude = frozenset(ARACHNID_SYNSETS) if args.no_arachnids else frozenset()
        bank = build_from_tars(
            args.tar_source,
            args.per_class,
            args.size,
            args.seed,
            args.workers,
            exclude,
        )
    else:
        print(f"Scanning {args.source} ...", flush=True)
        sources = iter_dir_images(args.source, args.num_images, args.seed)
        print(f"Decoding {args.num_images} images at {args.size}x{args.size} ...")
        bank = build(sources, args.num_images, args.size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(bank, args.out)
    megabytes = bank.numel() / 1e6
    print(f"Wrote {args.out}: {tuple(bank.shape)} uint8, {megabytes:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
