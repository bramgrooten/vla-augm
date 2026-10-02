# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation augmentation for RL fine-tuning of VLAs.

Random overlay, per SVEA (arXiv:2107.00644) and MaDi (arXiv:2312.15339):

    aug(x) = (1 - alpha) * x + alpha * overlay

with ``overlay`` drawn per sample from a bank of natural images.

Observations reach the model in [-1, 1] (``openpi.models.model.Observation``
``from_dict`` scales uint8 by ``/255 * 2 - 1``), and the blend runs in that
space. Blending is affine-equivariant, so this produces the same *image* as
converting to [0, 1], blending, and converting back -- the output stays in
[-1, 1], it is not rescaled to [0, 1]. See :func:`RandomOverlay.__call__`.

Where the augmented images are *used* is the experiment; see
:class:`AugmentationPlacement`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Sequence

import torch

from rlinf.utils.logging import get_logger

logger = get_logger()

# Images reach the augmentation already resized by openpi's preprocessing.
DEFAULT_IMAGE_SIZE = 224


class AugmentationPlacement(str, Enum):
    """Which forward pass sees augmented observations.

    pi0/pi0.5 run one VLM prefix pass whose output feeds *both* heads: the KV
    cache conditions the action expert (actor) and the prefix tokens feed the
    value head (critic). Placement therefore decides how many VLM passes a
    training step costs.

    NONE:
        Baseline. One pass, no augmentation.
    UNIFORM:
        Augment once; actor and critic both see the augmented view. One pass,
        so this is nearly free -- the naive arm Exp 0 needs to rule out.
    CRITIC_ONLY:
        The SVEA analog. Two passes: the actor is conditioned on the clean
        view, and the critic is fit on clean *and* augmented views
        (``critic_loss = (1 - beta) * L(V_clean) + beta * L(V_aug)``, SVEA's
        alpha/beta form). Roughly doubles the VLM cost of a training step.
    ACTOR_ONLY:
        The mirror image of CRITIC_ONLY, and the arm that isolates *which
        module* the augmentation has to reach. Two passes: the actor is
        conditioned on the augmented view, while the critic reads a clean
        prefix. SVEA's claim is that this is the harmful placement --
        augmenting the policy input perturbs the behaviour the value targets
        were collected under, so the critic chases a moving target. Without
        it, CRITIC_ONLY beating UNIFORM cannot be attributed to the placement
        rather than to seeing fewer augmented views.

        Unlike CRITIC_ONLY there is no blending: the second pass produces the
        *clean* value, the critic is fit on that view alone, so no
        ``values_aug`` is emitted, the critic loss keeps its single-view form,
        and ``aug_beta`` does not apply.
    """

    NONE = "none"
    UNIFORM = "uniform"
    CRITIC_ONLY = "critic_only"
    ACTOR_ONLY = "actor_only"


class AugmentationScope(str, Enum):
    """Which phases of training see augmented observations.

    UPDATES_ONLY:
        Default. Only the training forward is augmented. Rollouts and evals
        run on clean observations, so the behaviour policy and every reported
        success rate are unaffected by the augmentation, and the value targets
        stay clean -- the property SVEA relies on.
    ALL:
        Rollouts and evals are augmented too, so the policy acts on augmented
        observations. Expected to be worse, and it changes what eval measures
        (success under overlay, not in-distribution success); useful as an
        ablation. Note the saved videos still show the *env* render, which is
        clean -- use vla_augm/utils/preview_augmentation.py to see model input.
    """

    UPDATES_ONLY = "updates-only"
    ALL = "all"


class AugmentationSequence(str, Enum):
    """How often a fresh overlay image is drawn.

    RANDOM_PER_ANGLE:
        A new image per camera and per draw. Cameras see unrelated overlays,
        which is the strongest perturbation.
    RANDOM:
        A new image per draw, shared across cameras. The scene is corrupted
        consistently across viewpoints, which is closer to a real background
        change than giving each camera its own.
    CHUNK:
        Held for ``period`` consecutive draws. Set ``period`` to the action
        chunk length so the overlay is fixed while one chunk executes.
    EPISODE:
        Held until :meth:`RandomOverlay.reset` is called at an episode
        boundary.

    Note the training update sees a shuffled batch of independent timesteps,
    so CHUNK and EPISODE have no meaning there and behave as RANDOM. They
    matter on the rollout/eval path, where draws are consecutive in time.
    """

    RANDOM_PER_ANGLE = "random-per-angle"
    RANDOM = "random"
    CHUNK = "chunk"
    EPISODE = "episode"


class ImageAugmentation(ABC):
    """Maps a list of per-camera image batches to augmented copies.

    Each element is ``[B, 3, H, W]`` float in [-1, 1] -- what openpi's
    ``preprocess_observation_pytorch`` emits, since it restores channels-first
    before returning. ``[B, H, W, 3]`` is accepted as well.
    """

    @abstractmethod
    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Return augmented copies; the inputs are never modified in place."""


class OverlayImageBank:
    """A fixed pool of natural images to blend into observations.

    Held as uint8 ``[N, H, W, 3]`` (~150 KB per image at 224x224), so a few
    thousand fit comfortably in GPU memory and sampling costs one index_select.
    """

    def __init__(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected an image bank of shape [N, H, W, 3], got {tuple(images.shape)}"
            )
        if images.dtype != torch.uint8:
            raise ValueError(f"Expected a uint8 image bank, got {images.dtype}")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def nbytes(self) -> int:
        return self.images.numel() * self.images.element_size()

    def to(self, device: torch.device) -> "OverlayImageBank":
        self.images = self.images.to(device)
        return self

    def pin(self) -> "OverlayImageBank":
        """Page-lock the bank so host-to-device copies of samples are faster.

        Only meaningful while the bank stays in host memory; pinning several GB
        can fail on a constrained node, so a failure is not fatal.
        """
        if not self.images.is_cuda and not self.images.is_pinned():
            try:
                self.images = self.images.pin_memory()
            except RuntimeError as exc:  # noqa: BLE001 - fall back to pageable
                logger.warning(f"Could not pin the overlay bank ({exc}); continuing.")
        return self

    def draw_indices(
        self, num: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Pick ``num`` random bank positions, so a draw can be reused."""
        return torch.randint(
            len(self), (num,), device=self.images.device, generator=generator
        )

    @classmethod
    def from_cache(
        cls,
        cache_path: str,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> "OverlayImageBank":
        """Load a bank built by ``vla_augm/utils/build_overlay_bank.py``.

        Args:
            cache_path: The ``.pt`` file.
            shard_index: Which contiguous slice this process takes.
            num_shards: How many processes split the bank. Every actor rank
                otherwise holds a full copy, so a 29 GB bank would cost 115 GB
                of host memory across 4 ranks -- more than a busy training node
                has spare. Sharding keeps the union of overlays seen during a
                run at full size while each rank pays a quarter.

        Raises:
            FileNotFoundError: If the bank has not been built.
            ValueError: If the shard would be empty.
        """
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"Overlay bank cache not found: {cache_path}. Build it with "
                "`python vla_augm/utils/build_overlay_bank.py --source <image dir> "
                f"--out {cache_path}`."
            )
        # mmap so a rank pages in only its own slice; without it every rank
        # would first materialise the whole bank and then throw most away.
        images = torch.load(cache_path, map_location="cpu", mmap=True)
        if num_shards > 1:
            total = images.shape[0]
            if num_shards > total:
                raise ValueError(
                    f"Cannot split a {total}-image bank across {num_shards} shards."
                )
            start = (total * shard_index) // num_shards
            end = (total * (shard_index + 1)) // num_shards
            images = images[start:end]
        # clone() forces the mapped pages into anonymous memory, so the tensor
        # does not fault back to a scratch filesystem mid-training.
        return cls(images.clone())

    def sample(
        self,
        num: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``num`` random images as float in [-1, 1], shape [num, H, W, 3]."""
        if indices is None:
            indices = self.draw_indices(num, generator)
        idx = indices.to(self.images.device)
        # Move uint8 and convert on the target device: a third of the traffic
        # of converting first, which matters when the bank stays in host memory.
        raw = self.images.index_select(0, idx).to(device)
        # uint8 [0, 255] -> float [-1, 1], matching the model's image range.
        return raw.float().div_(127.5).sub_(1.0)


def random_resized_crop_flip(
    images: torch.Tensor,
    min_scale: float = 0.3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random resized crop + horizontal flip, applied per image.

    SVEA and MaDi feed overlays through ``RandomResizedCrop`` and
    ``RandomHorizontalFlip`` on every draw, so the overlay distribution is far
    larger than the number of source files. Doing the same here means bank size
    sets *class* coverage while this sets within-image variety, and the two can
    be tuned independently.

    Args:
        images: ``[N, C, H, W]``.
        min_scale: Smallest crop area as a fraction of the image.
        generator: Seeded RNG (must live on ``images.device``).

    Returns:
        ``[N, C, H, W]``, same shape and dtype.
    """
    num, _, height, width = images.shape
    device = images.device

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    # Sample a crop box per image, as normalized centre + half-extent, then
    # realize it with one affine grid_sample -- cheaper than N slice+resize
    # ops and it keeps everything on-device.
    scale = min_scale + (1.0 - min_scale) * rand(num)
    extent = scale.sqrt()  # side length as a fraction of the image
    # Centre must keep the crop inside the image: |centre| <= 1 - extent.
    slack = 1.0 - extent
    centre_x = (rand(num) * 2.0 - 1.0) * slack
    centre_y = (rand(num) * 2.0 - 1.0) * slack
    # Horizontal flip = negating the x scale of the affine map.
    flip = torch.where(rand(num) < 0.5, -1.0, 1.0)

    theta = torch.zeros(num, 2, 3, device=device, dtype=images.dtype)
    theta[:, 0, 0] = extent * flip
    theta[:, 0, 2] = centre_x
    theta[:, 1, 1] = extent
    theta[:, 1, 2] = centre_y

    grid = torch.nn.functional.affine_grid(
        theta, (num, images.shape[1], height, width), align_corners=False
    )
    return torch.nn.functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


class RandomOverlay(ImageAugmentation):
    """Blend a random natural image into each observation.

    Args:
        bank: Pool of overlay images.
        alpha: Blend weight. 0 is a no-op, 1 replaces the observation
            entirely. SVEA and MaDi both use 0.5.
        view_indices: Positions in the image list to augment. openpi orders
            views by ``IMAGE_KEYS`` -- ``(base_0_rgb, left_wrist_0_rgb,
            right_wrist_0_rgb)`` -- so ``[0]`` is the base camera alone.
            None augments every view, which on LIBERO includes index 2: that
            is not a camera but an all-zero padding image, attention-masked
            off by ``LiberoInputs`` (``right_wrist_0_rgb: np.False_``).
            Prefer ``[0, 1]`` there.
        crop_scale: Minimum crop area for the per-draw random resized crop and
            flip, as a fraction of the overlay. None disables it, blending the
            stored centre crop as-is. SVEA/MaDi always crop, which is why their
            overlay distribution is far larger than their file count.
        generator: Seeded RNG, so an arm is reproducible across restarts.
    """

    def __init__(
        self,
        bank: OverlayImageBank,
        alpha: float = 0.5,
        view_indices: Optional[Sequence[int]] = None,
        crop_scale: Optional[float] = 0.3,
        sequence: "AugmentationSequence" = AugmentationSequence.RANDOM_PER_ANGLE,
        period: int = 1,
        generator: Optional[torch.Generator] = None,
        crop_generator: Optional[torch.Generator] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if crop_scale is not None and not 0.0 < crop_scale <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}")
        self.bank = bank
        self.alpha = alpha
        self.view_indices = None if view_indices is None else set(view_indices)
        self.crop_scale = crop_scale
        self.sequence = AugmentationSequence(sequence)
        # How many draws one image survives, for CHUNK. The caller sets this
        # because a "step" differs by path: one model inference covers a whole
        # action chunk, while the video recorder is called per rendered frame.
        self.period = max(int(period), 1)
        self._held_indices: Optional[torch.Tensor] = None
        self._draws_since_refresh = 0
        # Index sampling happens on the bank's device, cropping on the training
        # device. Those differ whenever the bank stays in host memory, and a
        # generator may only be used on its own device -- hence two of them.
        self.generator = generator
        self._crop_generator = crop_generator
        # Cropping runs on the images' device, which is unknown until the first
        # call, so a caller who supplies only `generator` gets a matching crop
        # generator derived from it. Without this the crops would silently fall
        # back to the global RNG and the arm would not be reproducible.
        self._crop_seed = None if generator is None else generator.initial_seed()

    def reset(self) -> None:
        """Drop the held overlay, so EPISODE draws a fresh one next call."""
        self._held_indices = None
        self._draws_since_refresh = 0

    def _indices_for(self, batch_size: int) -> Optional[torch.Tensor]:
        """Bank positions to use for this call, or None to draw per view.

        RANDOM_PER_ANGLE returns None so each view draws independently. The
        others return one index set shared by every view, refreshed according
        to the sequence policy.
        """
        if self.sequence is AugmentationSequence.RANDOM_PER_ANGLE:
            return None

        stale = (
            self._held_indices is None
            or self._held_indices.shape[0] != batch_size
            or (
                self.sequence is AugmentationSequence.RANDOM
                or (
                    self.sequence is AugmentationSequence.CHUNK
                    and self._draws_since_refresh >= self.period
                )
            )
        )
        if stale:
            self._held_indices = self.bank.draw_indices(batch_size, self.generator)
            self._draws_since_refresh = 0
        return self._held_indices

    def _crop_generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        """Crop RNG for `device`, derived from the sampling seed if needed."""
        if self._crop_generator is not None and self._crop_generator.device == device:
            return self._crop_generator
        if self._crop_seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(self._crop_seed)
        self._crop_generator = generator
        return generator

    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        batch_size = images[0].shape[0] if len(images) else 0
        shared_indices = self._indices_for(batch_size) if batch_size else None
        self._draws_since_refresh += 1
        for view, image in enumerate(images):
            if self.view_indices is not None and view not in self.view_indices:
                # Pass the tensor through untouched -- callers may rely on
                # identity for unaugmented views.
                out.append(image)
                continue

            # openpi hands us [B, C, H, W]: preprocess_observation_pytorch
            # restores channels-first before returning. Detect it the same way
            # upstream does, and support [B, H, W, C] too since that is the
            # layout it uses internally.
            channels_first = image.shape[1] == 3
            if channels_first:
                batch_size, _, height, width = image.shape
            else:
                batch_size, height, width, _ = image.shape

            # Bank is [N, H, W, 3]; move to [N, 3, H, W] so resizing is direct.
            overlay = self.bank.sample(
                batch_size, image.device, self.generator, indices=shared_indices
            ).permute(0, 3, 1, 2)
            if self.crop_scale is not None:
                # Per-draw crop + flip, as SVEA and MaDi do: the overlay
                # distribution is then much larger than the bank.
                overlay = random_resized_crop_flip(
                    overlay.contiguous(),
                    self.crop_scale,
                    self._crop_generator_for(overlay.device),
                )
            if overlay.shape[-2:] != (height, width):
                overlay = torch.nn.functional.interpolate(
                    overlay,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            if not channels_first:
                overlay = overlay.permute(0, 2, 3, 1)

            # Blend in [-1,1] rather than round-tripping through [0,1]: the
            # two agree exactly, since blending commutes with the rescale.
            #   2*[(1-a)*(x+1)/2 + a*(o+1)/2] - 1 == (1-a)*x + a*o
            # Output is in [-1,1] (both operands are), which is what the model
            # expects -- it is not rescaled to [0,1].
            overlay = overlay.to(image.dtype)
            out.append(torch.lerp(image, overlay, self.alpha))
        return out


class RandomShift(ImageAugmentation):
    """Pad and randomly crop back, the DrQ/SVEA observation augmentation.

    This perturbs the *observation's* framing, which is what makes a policy
    tolerant of a moved camera -- the LIBERO-Plus axis where pi0.5 collapses to
    ~17%. Do not confuse it with :class:`RandomOverlay`'s ``crop_scale``, which
    random-resized-crops the *overlay image* and does nothing for viewpoint.

    DrQ uses 4 px on 84x84 (4.8%); the default 10 px on 224x224 (4.5%) matches
    that in relative terms.

    Args:
        pad: Pixels of padding before the crop, i.e. the maximum shift.
        pad_mode: How the border is filled. ``replicate`` (default) extends the
            edge pixels, as DrQ does. ``constant`` pads with ``pad_value``
            (black by default), which introduces a border the policy can key
            off -- that border is itself a distribution shift, so it is not the
            default, but it is available as an ablation. ``reflect`` mirrors and
            ``circular`` wraps; both invent structure that no camera motion
            would produce.
        pad_value: Fill for ``constant`` mode, in the model's [-1, 1] range, so
            -1 is black and 0 mid-grey.
        per_view: Shift each camera independently. True by default because the
            cameras are physically separate -- a wrist camera moving does not
            imply the base camera moved the same way.
        generator: Seeded RNG (on the images' device).
    """

    PAD_MODES = ("replicate", "constant", "reflect", "circular")

    def __init__(
        self,
        pad: int = 10,
        view_indices: Optional[Sequence[int]] = None,
        per_view: bool = True,
        pad_mode: str = "replicate",
        pad_value: float = -1.0,
        generator: Optional[torch.Generator] = None,
    ):
        if pad < 1:
            raise ValueError(f"pad must be >= 1, got {pad}")
        if pad_mode not in self.PAD_MODES:
            raise ValueError(
                f"pad_mode must be one of {self.PAD_MODES}, got {pad_mode!r}"
            )
        self.pad = int(pad)
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)
        self.view_indices = None if view_indices is None else set(view_indices)
        self.per_view = per_view
        self.generator = generator

    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out, offsets = [], None
        for view, image in enumerate(images):
            if self.view_indices is not None and view not in self.view_indices:
                out.append(image)
                continue

            channels_first = image.shape[1] == 3
            chw = image if channels_first else image.permute(0, 3, 1, 2)
            batch, _, height, width = chw.shape

            # Default is replicate rather than a constant fill: a black border
            # is itself a distribution shift the policy would learn to key off.
            if self.pad_mode == "constant":
                padded = torch.nn.functional.pad(
                    chw, (self.pad,) * 4, mode="constant", value=self.pad_value
                )
            else:
                padded = torch.nn.functional.pad(
                    chw, (self.pad,) * 4, mode=self.pad_mode
                )
            if offsets is None or self.per_view:
                offsets = torch.randint(
                    0,
                    2 * self.pad + 1,
                    (batch, 2),
                    device=image.device,
                    generator=self.generator,
                )
            # Gather each sample's own crop window.
            rows = offsets[:, 0, None] + torch.arange(height, device=image.device)
            cols = offsets[:, 1, None] + torch.arange(width, device=image.device)
            idx = torch.arange(batch, device=image.device)[:, None, None]
            cropped = padded[idx, :, rows[:, :, None], cols[:, None, :]]
            # The advanced index puts the gathered dims first; restore NCHW.
            cropped = cropped.permute(0, 3, 1, 2)
            out.append(cropped if channels_first else cropped.permute(0, 2, 3, 1))
        return out


class ComposeAugmentation(ImageAugmentation):
    """Apply augmentations in order, e.g. overlay then shift."""

    def __init__(self, *augmentations: ImageAugmentation):
        self.augmentations = [a for a in augmentations if a is not None]

    def reset(self) -> None:
        for aug in self.augmentations:
            if hasattr(aug, "reset"):
                aug.reset()

    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        result = list(images)
        for aug in self.augmentations:
            result = aug(result)
        return result


def split_views_for_placement(
    images: Sequence[torch.Tensor],
    placement: AugmentationPlacement,
    image_aug: Optional[ImageAugmentation],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Decide which view the actor and the critic each see.

    This is the whole placement experiment in one function, kept separate from
    the model so it can be tested without a VLM:

        NONE         (clean, clean)  -- caller runs one prefix pass
        UNIFORM      (aug, aug)      -- one pass, both heads share the tensor
        CRITIC_ONLY  (clean, aug)    -- two passes
        ACTOR_ONLY   (aug, clean)    -- two passes

    Callers detect the two-pass case with ``critic is not actor``: identity,
    not equality, so an augmentation that happens to be a no-op (alpha=0)
    still costs only one pass.

    The two-pass placements differ in what the second pass is *for*, which is
    the caller's business: under CRITIC_ONLY it yields the extra augmented
    value (``values_aug``) that SVEA's two-term critic loss consumes, while
    under ACTOR_ONLY it yields the one clean value the critic is fit on. Both
    return ``(actor, critic)``, so a caller that only needs to know how many
    passes to run does not have to branch.

    Raises:
        ValueError: If a placement other than NONE has no augmentation.
    """
    images = list(images)
    if placement is AugmentationPlacement.NONE:
        return images, images

    if image_aug is None:
        raise ValueError(
            f"placement={placement.value} requires an augmentation, got None."
        )

    if placement is AugmentationPlacement.UNIFORM:
        augmented = image_aug(images)
        return augmented, augmented

    if placement is AugmentationPlacement.CRITIC_ONLY:
        # The actor keeps the clean view; only the critic sees the overlay.
        return images, image_aug(images)

    if placement is AugmentationPlacement.ACTOR_ONLY:
        # Mirror of CRITIC_ONLY: only the actor sees the overlay, the critic
        # reads a clean prefix. Callers must therefore take the critic's value
        # from the *second* view, not from the actor's pass.
        return image_aug(images), images

    raise ValueError(f"Unhandled placement {placement!r}")


def build_image_augmentation(
    aug_cfg,
    device: torch.device,
    seed: Optional[int] = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[AugmentationPlacement, Optional[ImageAugmentation]]:
    """Build the augmentation for an ``algorithm.augmentation`` config block.

    Returns ``(placement, augmentation)``; ``augmentation`` is None exactly
    when the placement is NONE, so callers can branch on either.

    ``shard_index``/``num_shards`` split the bank across data-parallel ranks;
    see :meth:`OverlayImageBank.from_cache`.

    Raises:
        ValueError: On an unknown placement or augmentation type.
    """
    if aug_cfg is None:
        return AugmentationPlacement.NONE, None

    placement = AugmentationPlacement(str(aug_cfg.get("placement", "none")).lower())
    if placement is AugmentationPlacement.NONE:
        return placement, None

    # Validated even though only the caller uses it, so a typo fails at
    # startup rather than silently leaving rollouts unaugmented.
    AugmentationScope(str(aug_cfg.get("apply_during", "updates-only")).lower())

    aug_type = str(aug_cfg.get("type", "random_overlay")).lower()
    valid_types = ("random_overlay", "random_shift", "random_overlay_shift")
    if aug_type not in valid_types:
        raise ValueError(
            f"Unknown augmentation type {aug_type!r}; expected one of {valid_types}."
        )
    wants_overlay = aug_type in ("random_overlay", "random_overlay_shift")
    wants_shift = aug_type in ("random_shift", "random_overlay_shift")

    bank_path = aug_cfg.get("bank_path", None)
    if wants_overlay and not bank_path:
        raise ValueError(
            f"type={aug_type} needs algorithm.augmentation.bank_path "
            "(build one with vla_augm/utils/build_overlay_bank.py)."
        )
    # Each actor rank takes a distinct slice, so the run still draws on the
    # whole bank while host memory holds one copy rather than one per rank.
    bank = (
        OverlayImageBank.from_cache(bank_path, shard_index, num_shards)
        if wants_overlay
        else None
    )

    # The bank stays in host memory by default: a large one is tens of GB, and
    # the training nodes have little spare (FreeMem on a busy H100 node runs to
    # ~15 GB). Sampling costs one uint8 host-to-device copy per micro-batch --
    # ~19 MB per view, against a ~130 s training forward, so it does not
    # register. Deliberately *not* page-locked: pinning tens of GB takes memory
    # the kernel can never reclaim, to save a copy that was never the problem.
    # Set bank_device: cuda for a small bank if the copy ever does matter.
    bank_device = str(aug_cfg.get("bank_device", "cpu")).lower()
    if bank_device not in ("cpu", "cuda", "auto"):
        raise ValueError(f"bank_device must be cpu, cuda or auto; got {bank_device!r}")
    if bank is not None and (
        bank_device == "cuda" or (bank_device == "auto" and bank.nbytes < 512e6)
    ):
        bank = bank.to(device)

    generator = crop_generator = None
    if seed is not None:
        # Index sampling reads the bank wherever it ended up, so the generator
        # must match that device -- not the training device.
        bank_device_actual = bank.images.device if bank is not None else device
        generator = torch.Generator(device=bank_device_actual)
        generator.manual_seed(seed)
        crop_generator = torch.Generator(device=device)
        crop_generator.manual_seed(seed)

    views = aug_cfg.get("views", None)
    view_indices = None if views in (None, "all") else list(views)
    crop_scale = aug_cfg.get("crop_scale", 0.3)
    sequence = AugmentationSequence(
        str(aug_cfg.get("sequence", "random-per-angle")).lower()
    )
    # One model inference covers a whole action chunk, so CHUNK holds the
    # overlay for exactly one call on this path.
    period = int(aug_cfg.get("chunk_period", 1))

    shift_pad = int(aug_cfg.get("shift_pad", 10))
    shift_per_view = bool(aug_cfg.get("shift_per_view", True))

    parts = []
    if wants_overlay:
        parts.append(
            RandomOverlay(
                bank=bank,
                alpha=float(aug_cfg.get("alpha", 0.5)),
                view_indices=view_indices,
                crop_scale=None if crop_scale is None else float(crop_scale),
                sequence=sequence,
                period=period,
                generator=generator,
                crop_generator=crop_generator,
            )
        )
    if wants_shift:
        parts.append(
            RandomShift(
                pad=shift_pad,
                view_indices=view_indices,
                per_view=shift_per_view,
                pad_mode=str(aug_cfg.get("shift_pad_mode", "replicate")).lower(),
                pad_value=float(aug_cfg.get("shift_pad_value", -1.0)),
                generator=crop_generator,
            )
        )

    bank_desc = (
        f"bank={len(bank)} images ({bank.nbytes / 1e9:.2f} GB on "
        f"{bank.images.device}) alpha={aug_cfg.get('alpha', 0.5)} "
        f"crop_scale={crop_scale} sequence={sequence.value} "
        if wants_overlay
        else ""
    )
    shift_desc = (
        f"shift_pad={shift_pad} per_view={shift_per_view} "
        f"pad_mode={aug_cfg.get('shift_pad_mode', 'replicate')} "
        if wants_shift
        else ""
    )
    logger.info(
        f"Image augmentation: placement={placement.value} type={aug_type} "
        f"{bank_desc}{shift_desc}"
        f"views={'all' if view_indices is None else view_indices}"
    )
    return placement, (parts[0] if len(parts) == 1 else ComposeAugmentation(*parts))


def get_augmentation_scope(aug_cfg) -> AugmentationScope:
    """Read ``algorithm.augmentation.apply_during``, defaulting to updates-only."""
    if aug_cfg is None:
        return AugmentationScope.UPDATES_ONLY
    return AugmentationScope(str(aug_cfg.get("apply_during", "updates-only")).lower())
