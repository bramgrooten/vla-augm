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

"""Random-overlay augmentation and the SVEA critic loss term.

The placement study only means anything if the augmented view reaches exactly
the intended forward pass, so these tests pin the blend arithmetic, which
camera views are touched, and that the critic loss reduces to its original
form when no augmented view is supplied.
"""

import json
import pathlib

import pytest
import torch

from rlinf.algorithms.augmentation import (
    AugmentationPlacement,
    AugmentationScope,
    AugmentationSequence,
    ComposeAugmentation,
    ImageAugmentation,
    OverlayImageBank,
    RandomOverlay,
    RandomShift,
    build_image_augmentation,
    get_augmentation_scope,
    random_resized_crop_flip,
    split_views_for_placement,
)
from rlinf.algorithms.losses import compute_ppo_critic_loss


def _bank(num=8, size=4, fill=None):
    if fill is not None:
        images = torch.full((num, size, size, 3), fill, dtype=torch.uint8)
    else:
        images = torch.randint(0, 256, (num, size, size, 3), dtype=torch.uint8)
    return OverlayImageBank(images)


def _views(num_views=3, batch=2, size=4):
    return [torch.zeros(batch, size, size, 3) for _ in range(num_views)]


def test_bank_rejects_wrong_shape_and_dtype():
    with pytest.raises(ValueError, match=r"\[N, H, W, 3\]"):
        OverlayImageBank(torch.zeros(4, 3, 8, 8, dtype=torch.uint8))
    with pytest.raises(ValueError, match="uint8"):
        OverlayImageBank(torch.zeros(4, 8, 8, 3, dtype=torch.float32))


def test_bank_maps_uint8_to_model_image_range():
    """Overlays must arrive in [-1, 1] -- the range openpi hands the model."""
    black = _bank(fill=0).sample(4, torch.device("cpu"))
    white = _bank(fill=255).sample(4, torch.device("cpu"))

    assert torch.allclose(black, torch.full_like(black, -1.0))
    assert torch.allclose(white, torch.full_like(white, 1.0), atol=1e-2)


def test_bank_normalization_matches_openpi_exactly():
    """Overlays must be normalized the same way observations are.

    ``Observation.from_dict`` scales uint8 by ``/255 * 2 - 1``; the bank uses
    the algebraically identical ``/127.5 - 1`` to avoid a second pass over the
    tensor. Blending images normalized by different conventions would shift
    the augmented view relative to the clean one, which would confound the
    whole placement comparison.
    """
    raw = torch.randint(0, 256, (16, 4, 4, 3), dtype=torch.uint8)
    openpi_formula = raw.float() / 255.0 * 2.0 - 1.0

    ours = OverlayImageBank(raw).sample(16, torch.device("cpu"))

    # Same values, up to which rows sample() happened to draw; compare the
    # transform directly rather than the sampled order.
    assert torch.equal(raw.float().div(127.5).sub(1.0), openpi_formula)
    assert ours.min() >= -1.0 and ours.max() <= 1.0


def test_overlay_blend_is_a_linear_interpolation():
    # Observation all -1, overlay all +1, alpha 0.25 -> -1 + 0.25*2 = -0.5.
    aug = RandomOverlay(bank=_bank(fill=255), alpha=0.25)
    images = [torch.full((2, 4, 4, 3), -1.0)]

    out = aug(images)[0]

    assert torch.allclose(out, torch.full_like(out, -0.5), atol=1e-2)


def test_blending_in_minus_one_to_one_matches_a_round_trip_through_zero_to_one():
    """Blending commutes with the [-1,1] <-> [0,1] rescale.

    Observations arrive in [-1, 1] and the blend runs there. The output is a
    [-1, 1] image -- e.g. x=-1 blended half-and-half with a mid-grey overlay
    (o=0) gives -0.5, which is 0.25 in [0, 1]. Same pixel, two coordinate
    systems; what must hold is that both routes agree.
    """
    torch.manual_seed(0)
    image = torch.rand(2, 4, 4, 3) * 2 - 1
    # A mid-grey bank (uint8 128 -> ~0.0 in [-1,1]) covers the case above.
    aug = RandomOverlay(bank=_bank(fill=128), alpha=0.5)

    in_model_space = aug([image])[0]

    overlay = torch.zeros_like(image)  # what uint8 128 maps to
    via_zero_one = 0.5 * ((image + 1) / 2) + 0.5 * ((overlay + 1) / 2)
    assert torch.allclose(in_model_space, via_zero_one * 2 - 1, atol=1e-2)
    assert in_model_space.min() >= -1.0 and in_model_space.max() <= 1.0


def test_alpha_zero_is_a_no_op_and_alpha_one_replaces():
    images = [torch.full((2, 4, 4, 3), -1.0)]

    untouched = RandomOverlay(bank=_bank(fill=255), alpha=0.0)(images)[0]
    replaced = RandomOverlay(bank=_bank(fill=255), alpha=1.0)(images)[0]

    assert torch.allclose(untouched, images[0])
    assert torch.allclose(replaced, torch.ones_like(replaced), atol=1e-2)


def test_only_selected_views_are_augmented():
    """openpi orders views (base, left_wrist, right_wrist); [0] is base only."""
    aug = RandomOverlay(bank=_bank(fill=255), alpha=1.0, view_indices=[0])
    images = _views()

    out = aug(images)

    assert not torch.allclose(out[0], images[0]), "base camera must be augmented"
    assert out[1] is images[1], "wrist views must pass through untouched"
    assert out[2] is images[2]


def test_augmentation_does_not_mutate_its_input():
    images = [torch.full((2, 4, 4, 3), -1.0)]
    before = images[0].clone()

    RandomOverlay(bank=_bank(fill=255), alpha=0.5)(images)

    assert torch.allclose(images[0], before), "the clean view is reused by the actor"


def test_overlay_handles_the_real_channels_first_observation_layout():
    """The layout the model actually passes: [B, 3, 224, 224], not [B, H, W, 3].

    openpi's preprocess_observation_pytorch converts to channels-last
    internally but permutes back before returning, so the augmentation is
    handed NCHW. Bank images are stored NHWC, so they must be transposed to
    match rather than blended as-is.
    """
    aug = RandomOverlay(bank=_bank(size=224, fill=255), alpha=0.5)
    images = [torch.full((2, 3, 224, 224), -1.0)]

    out = aug(images)[0]

    assert out.shape == (2, 3, 224, 224), "layout must be preserved"
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-2)


def test_overlay_preserves_channels_last_layout_too():
    aug = RandomOverlay(bank=_bank(size=8, fill=255), alpha=0.5)

    out = aug([torch.full((2, 16, 16, 3), -1.0)])[0]

    assert out.shape == (2, 16, 16, 3)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-2)


def test_overlay_resizes_a_bank_that_does_not_match_the_observation():
    aug = RandomOverlay(bank=_bank(size=8, fill=255), alpha=1.0)

    out = aug([torch.full((2, 4, 4, 3), -1.0)])[0]

    assert out.shape == (2, 4, 4, 3)


def test_seeded_generators_reproduce_the_same_overlays():
    images = [torch.full((2, 4, 4, 3), -1.0)]
    bank = _bank(num=64)

    def run(seed):
        generator = torch.Generator().manual_seed(seed)
        return RandomOverlay(bank=bank, alpha=0.5, generator=generator)(images)[0]

    assert torch.allclose(run(0), run(0))
    assert not torch.allclose(run(0), run(1))


def test_build_returns_none_for_the_baseline_arm():
    placement, aug = build_image_augmentation(
        {"placement": "none"}, torch.device("cpu")
    )

    assert placement is AugmentationPlacement.NONE
    assert aug is None, "the baseline arm must not construct an augmenter"


def test_build_rejects_an_unknown_placement():
    with pytest.raises(ValueError):
        build_image_augmentation({"placement": "encoder_only"}, torch.device("cpu"))


# --- placement --------------------------------------------------------------
#
# pi0.5 conditions the action expert on the VLM prefix KV cache and reads the
# value head off the same prefix tokens, so which tensor each head receives is
# the entire experiment. Getting it wrong is silent: training still runs.


def _overlay(alpha=1.0):
    return RandomOverlay(bank=_bank(fill=255), alpha=alpha)


def test_placement_none_leaves_both_heads_on_the_clean_view():
    images = _views()

    actor, critic = split_views_for_placement(
        images, AugmentationPlacement.NONE, _overlay()
    )

    assert critic is actor, "baseline must cost a single prefix pass"
    assert all(torch.allclose(a, b) for a, b in zip(actor, images))


def test_placement_uniform_gives_both_heads_the_same_augmented_view():
    images = _views()

    actor, critic = split_views_for_placement(
        images, AugmentationPlacement.UNIFORM, _overlay()
    )

    assert critic is actor, "uniform must cost a single prefix pass"
    assert not torch.allclose(actor[0], images[0]), "the actor sees the overlay"


def test_placement_critic_only_keeps_the_actor_clean():
    """The SVEA property: augmenting the critic must not touch the actor."""
    images = _views()

    actor, critic = split_views_for_placement(
        images, AugmentationPlacement.CRITIC_ONLY, _overlay()
    )

    assert critic is not actor, "critic_only must trigger a second prefix pass"
    for clean, actor_view in zip(images, actor):
        assert torch.allclose(actor_view, clean), "actor must keep the clean view"
    for clean, critic_view in zip(images, critic):
        assert not torch.allclose(critic_view, clean), "critic must see the overlay"


def test_placement_actor_only_keeps_the_critic_clean():
    """Mirror of critic_only: the arm isolating "augmenting the actor hurts".

    The overlay must reach the actor alone.
    """
    images = _views()

    actor, critic = split_views_for_placement(
        images, AugmentationPlacement.ACTOR_ONLY, _overlay()
    )

    assert critic is not actor, "actor_only must trigger a second prefix pass"
    for clean, actor_view in zip(images, actor):
        assert not torch.allclose(actor_view, clean), "actor must see the overlay"
    for clean, critic_view in zip(images, critic):
        assert torch.allclose(critic_view, clean), "critic must keep the clean view"


def test_actor_only_and_critic_only_are_exact_mirrors():
    """Same draw, swapped roles -- so the pair isolates placement alone."""
    images = _views()

    actor_aug, actor_clean = split_views_for_placement(
        images, AugmentationPlacement.ACTOR_ONLY, _overlay()
    )
    critic_clean, critic_aug = split_views_for_placement(
        images, AugmentationPlacement.CRITIC_ONLY, _overlay()
    )

    for a, b in zip(actor_clean, critic_clean):
        assert torch.allclose(a, b), "both must leave the clean view untouched"
    # _overlay() is seeded, so the two augmented views come from the same draw:
    # the arms differ in which module receives it, nothing else.
    for a, b in zip(actor_aug, critic_aug):
        assert torch.allclose(a, b)


def test_actor_only_redraws_overlays_each_call():
    images = _views(num_views=1, batch=16)
    aug = RandomOverlay(bank=_bank(num=64), alpha=0.5)

    first, _ = split_views_for_placement(images, AugmentationPlacement.ACTOR_ONLY, aug)
    second, _ = split_views_for_placement(images, AugmentationPlacement.ACTOR_ONLY, aug)

    assert not torch.allclose(first[0], second[0])


def test_actor_only_without_an_augmentation_is_rejected():
    with pytest.raises(ValueError, match="requires an augmentation"):
        split_views_for_placement(_views(), AugmentationPlacement.ACTOR_ONLY, None)


def test_critic_only_redraws_overlays_each_call():
    """Fresh noise per update epoch, as in DrQ/SVEA -- not one fixed overlay."""
    images = _views(num_views=1, batch=16)
    aug = RandomOverlay(bank=_bank(num=64), alpha=0.5)

    _, first = split_views_for_placement(images, AugmentationPlacement.CRITIC_ONLY, aug)
    _, second = split_views_for_placement(
        images, AugmentationPlacement.CRITIC_ONLY, aug
    )

    assert not torch.allclose(first[0], second[0])


def test_placement_without_an_augmentation_is_rejected():
    """Fails loudly rather than silently running the unaugmented baseline."""
    with pytest.raises(ValueError, match="requires an augmentation"):
        split_views_for_placement(_views(), AugmentationPlacement.CRITIC_ONLY, None)


# --- SVEA critic loss -------------------------------------------------------


def _critic_kwargs(**overrides):
    torch.manual_seed(0)
    kwargs = {
        "values": torch.randn(8),
        "returns": torch.randn(8),
        "prev_values": torch.zeros(8),
        "value_clip": 0.2,
        "huber_delta": 10.0,
        "loss_mask": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_critic_loss_unchanged_when_no_augmented_view_is_given():
    """Every arm but critic_only hits this path -- actor_only included, since
    its critic is fit on the clean view alone and passes no values_aug."""
    kwargs = _critic_kwargs()

    loss, metrics = compute_ppo_critic_loss(**kwargs)

    assert "critic/value_loss_aug" not in metrics
    assert torch.isfinite(loss)


def test_critic_loss_interpolates_between_the_clean_and_augmented_views():
    kwargs = _critic_kwargs()
    clean, _ = compute_ppo_critic_loss(**kwargs)
    aug_values = kwargs["values"] + 1.0
    only_aug, _ = compute_ppo_critic_loss(**{**kwargs, "values": aug_values})

    both, metrics = compute_ppo_critic_loss(
        **kwargs, values_aug=aug_values, aug_beta=0.5
    )

    assert torch.allclose(both, 0.5 * clean + 0.5 * only_aug)
    assert torch.allclose(metrics["critic/value_loss_aug"], only_aug)
    assert torch.allclose(metrics["critic/value_aug_gap"], torch.tensor(1.0))


def test_aug_beta_zero_recovers_the_clean_loss():
    """A guard against the augmented term leaking in when it is switched off."""
    kwargs = _critic_kwargs()
    clean, _ = compute_ppo_critic_loss(**kwargs)

    both, _ = compute_ppo_critic_loss(
        **kwargs, values_aug=kwargs["values"] + 5.0, aug_beta=0.0
    )

    assert torch.allclose(both, clean)


def test_gradients_reach_both_views():
    """SVEA fits the critic on both views, so both must carry gradient."""
    values = torch.randn(8, requires_grad=True)
    values_aug = torch.randn(8, requires_grad=True)
    kwargs = _critic_kwargs(values=values)

    loss, _ = compute_ppo_critic_loss(**kwargs, values_aug=values_aug, aug_beta=0.5)
    loss.backward()

    assert values.grad is not None and values.grad.abs().sum() > 0
    assert values_aug.grad is not None and values_aug.grad.abs().sum() > 0


# --- bank residency ---------------------------------------------------------
#
# A full-coverage ImageNet Winter21 bank (one image per synset) is ~2.9 GB.
# Holding that on every rank competes with the actor and the env workers' EGL
# contexts, so it stays in host memory by default.


def _write_bank(tmp_path, num=4, size=4):
    path = tmp_path / "bank.pt"
    torch.save(torch.randint(0, 256, (num, size, size, 3), dtype=torch.uint8), path)
    return str(path)


def test_bank_nbytes_reports_actual_footprint():
    bank = _bank(num=100, size=224)

    assert bank.nbytes == 100 * 224 * 224 * 3


def test_bank_stays_in_host_memory_by_default(tmp_path):
    """Default must not pin GBs of overlays into every rank's GPU memory."""
    cfg = {"placement": "uniform", "bank_path": _write_bank(tmp_path)}

    _, aug = build_image_augmentation(cfg, torch.device("cpu"))

    assert not aug.bank.images.is_cuda


def test_bank_device_rejects_an_unknown_value(tmp_path):
    cfg = {
        "placement": "uniform",
        "bank_path": _write_bank(tmp_path),
        "bank_device": "gpu",
    }

    with pytest.raises(ValueError, match="bank_device must be"):
        build_image_augmentation(cfg, torch.device("cpu"))


def test_host_resident_bank_still_samples_and_blends(tmp_path):
    """The CPU-bank path must produce the same shapes as a device-resident one."""
    cfg = {
        "placement": "uniform",
        "bank_path": _write_bank(tmp_path, num=32),
        "alpha": 0.5,
    }
    _, aug = build_image_augmentation(cfg, torch.device("cpu"), seed=0)

    out = aug([torch.full((2, 3, 8, 8), -1.0)])[0]

    assert out.shape == (2, 3, 8, 8)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_bank_shards_are_disjoint_and_cover_everything(tmp_path):
    """Ranks must partition the bank, not duplicate or drop images.

    Each actor rank holding a full copy would cost 4x host memory on a node
    that has little to spare; sharding is only sound if the union is intact.
    """
    path = tmp_path / "bank.pt"
    images = torch.arange(8, dtype=torch.uint8).reshape(8, 1, 1, 1).repeat(1, 2, 2, 3)
    torch.save(images, path)

    shards = [
        OverlayImageBank.from_cache(str(path), shard_index=i, num_shards=4)
        for i in range(4)
    ]

    assert [len(s) for s in shards] == [2, 2, 2, 2]
    seen = torch.cat([s.images[:, 0, 0, 0] for s in shards]).tolist()
    assert sorted(seen) == list(range(8)), "shards must tile the bank exactly"


def test_bank_rejects_more_shards_than_images(tmp_path):
    path = tmp_path / "bank.pt"
    torch.save(torch.zeros(2, 2, 2, 3, dtype=torch.uint8), path)

    with pytest.raises(ValueError, match="Cannot split"):
        OverlayImageBank.from_cache(str(path), shard_index=0, num_shards=4)


def test_crop_flip_preserves_shape_and_range():
    images = torch.rand(8, 3, 32, 32) * 2 - 1

    out = random_resized_crop_flip(images, 0.3, torch.Generator().manual_seed(0))

    assert out.shape == images.shape
    # Reflection padding cannot invent values outside the input range.
    assert out.min() >= images.min() - 1e-4
    assert out.max() <= images.max() + 1e-4


def test_crop_flip_makes_one_source_image_yield_distinct_overlays():
    """This is what makes bank size and overlay variety independent knobs."""
    repeated = (torch.rand(1, 3, 32, 32) * 2 - 1).repeat(16, 1, 1, 1)

    out = random_resized_crop_flip(repeated, 0.3, torch.Generator().manual_seed(0))

    flat = out.reshape(16, -1)
    pairwise = torch.cdist(flat, flat) + torch.eye(16) * 1e9
    assert pairwise.min() > 1e-3, "every draw must differ"


def test_crop_scale_none_leaves_the_overlay_uncropped():
    aug = RandomOverlay(bank=_bank(fill=255), alpha=1.0, crop_scale=None)

    out = aug([torch.full((2, 3, 4, 4), -1.0)])[0]

    # An all-white overlay at alpha=1 stays exactly white without resampling.
    assert torch.allclose(out, torch.ones_like(out), atol=1e-2)


def test_crop_scale_is_validated():
    with pytest.raises(ValueError, match="crop_scale"):
        RandomOverlay(bank=_bank(), crop_scale=0.0)


# --- apply_during scope -----------------------------------------------------


def test_scope_defaults_to_updates_only():
    """Rollouts and evals must stay clean unless explicitly asked otherwise.

    If this ever flipped, the behaviour policy and every reported success rate
    would silently be measured under overlay.
    """
    assert get_augmentation_scope(None) is AugmentationScope.UPDATES_ONLY
    assert get_augmentation_scope({}) is AugmentationScope.UPDATES_ONLY
    assert (
        get_augmentation_scope({"placement": "uniform"})
        is AugmentationScope.UPDATES_ONLY
    )


def test_scope_all_is_selectable():
    cfg = {"placement": "uniform", "apply_during": "all"}

    assert get_augmentation_scope(cfg) is AugmentationScope.ALL


def test_unknown_scope_fails_at_startup(tmp_path):
    """A typo must not silently fall back to leaving rollouts unaugmented."""
    with pytest.raises(ValueError):
        get_augmentation_scope({"apply_during": "eval-only"})

    path = tmp_path / "bank.pt"
    torch.save(torch.zeros(4, 4, 4, 3, dtype=torch.uint8), path)
    with pytest.raises(ValueError):
        build_image_augmentation(
            {
                "placement": "uniform",
                "bank_path": str(path),
                "apply_during": "sometimes",
            },
            torch.device("cpu"),
        )


# --- sequence ---------------------------------------------------------------


def _seq_overlay(sequence, period=1, num=64):
    return RandomOverlay(
        bank=_bank(num=num),
        alpha=1.0,
        crop_scale=None,
        sequence=sequence,
        period=period,
        generator=torch.Generator().manual_seed(0),
    )


def test_random_per_angle_gives_each_camera_a_different_overlay():
    aug = _seq_overlay(AugmentationSequence.RANDOM_PER_ANGLE)
    views = [torch.full((4, 3, 4, 4), -1.0) for _ in range(2)]

    out = aug(views)

    assert not torch.allclose(out[0], out[1]), "cameras must differ"


def test_random_shares_one_overlay_across_cameras():
    """The same image, applied separately to each camera -- not one image
    stretched across a concatenation of them."""
    aug = _seq_overlay(AugmentationSequence.RANDOM)
    views = [torch.full((4, 3, 4, 4), -1.0) for _ in range(2)]

    out = aug(views)

    assert torch.allclose(out[0], out[1]), "both cameras get the same overlay"


def test_random_redraws_every_call():
    aug = _seq_overlay(AugmentationSequence.RANDOM)
    views = [torch.full((4, 3, 4, 4), -1.0)]

    first, second = aug(views)[0], aug(views)[0]

    assert not torch.allclose(first, second)


def test_chunk_holds_the_overlay_for_period_calls():
    aug = _seq_overlay(AugmentationSequence.CHUNK, period=3)
    views = [torch.full((4, 3, 4, 4), -1.0)]

    draws = [aug(views)[0].clone() for _ in range(4)]

    assert torch.allclose(draws[0], draws[1]), "held within the chunk"
    assert torch.allclose(draws[0], draws[2])
    assert not torch.allclose(draws[0], draws[3]), "redrawn after period calls"


def test_episode_holds_until_reset():
    aug = _seq_overlay(AugmentationSequence.EPISODE)
    views = [torch.full((4, 3, 4, 4), -1.0)]

    first = aug(views)[0].clone()
    held = aug(views)[0].clone()
    aug.reset()
    after_reset = aug(views)[0].clone()

    assert torch.allclose(first, held), "constant for the whole episode"
    assert not torch.allclose(first, after_reset), "new episode, new overlay"


def test_sequence_defaults_to_random_per_angle(tmp_path):
    path = tmp_path / "bank.pt"
    torch.save(torch.randint(0, 256, (16, 4, 4, 3), dtype=torch.uint8), path)

    _, aug = build_image_augmentation(
        {"placement": "uniform", "bank_path": str(path)}, torch.device("cpu")
    )

    assert aug.sequence is AugmentationSequence.RANDOM_PER_ANGLE


# --- random shift and composition ------------------------------------------
#
# random_shift targets Camera Viewpoints, the LIBERO-Plus axis where pi0.5 sits
# at ~17% against ~93% in-distribution. It perturbs the OBSERVATION's framing,
# unlike crop_scale which crops the overlay image.


def test_shift_preserves_shape_and_layout():
    nchw = torch.rand(4, 3, 32, 32) * 2 - 1
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    shift = RandomShift(pad=6, generator=torch.Generator().manual_seed(0))

    assert shift([nchw])[0].shape == nchw.shape
    assert shift([nhwc])[0].shape == nhwc.shape


def test_shift_actually_moves_content():
    # A ramp, so any translation is detectable.
    ramp = torch.linspace(-1, 1, 32).expand(4, 3, 32, 32).contiguous()
    shift = RandomShift(pad=8, generator=torch.Generator().manual_seed(0))

    assert not torch.allclose(shift([ramp])[0], ramp)


def test_shift_uses_replicate_not_zero_padding():
    """A black border is itself a shift the policy could key off."""
    flat = torch.full((4, 3, 16, 16), 0.7)
    shift = RandomShift(pad=5, generator=torch.Generator().manual_seed(0))

    out = shift([flat])[0]

    assert torch.allclose(out, torch.full_like(out, 0.7)), "no border introduced"


def test_shift_per_view_controls_whether_cameras_move_together():
    image = torch.rand(4, 3, 24, 24) * 2 - 1
    views = [image.clone(), image.clone()]

    independent = RandomShift(
        pad=6, per_view=True, generator=torch.Generator().manual_seed(1)
    )(views)
    shared = RandomShift(
        pad=6, per_view=False, generator=torch.Generator().manual_seed(1)
    )(views)

    assert not torch.allclose(independent[0], independent[1])
    assert torch.allclose(shared[0], shared[1])


def test_shift_rejects_a_zero_pad():
    with pytest.raises(ValueError, match="pad must be"):
        RandomShift(pad=0)


def test_compose_applies_in_order_and_forwards_reset():
    calls = []

    class _Tag(ImageAugmentation):
        def __init__(self, name):
            self.name = name

        def __call__(self, images):
            calls.append(self.name)
            return list(images)

        def reset(self):
            calls.append(f"reset:{self.name}")

    composed = ComposeAugmentation(_Tag("a"), _Tag("b"))
    composed([torch.zeros(1, 3, 4, 4)])
    composed.reset()

    assert calls == ["a", "b", "reset:a", "reset:b"]


def test_builder_rejects_an_unknown_type(tmp_path):
    path = tmp_path / "bank.pt"
    torch.save(torch.zeros(4, 4, 4, 3, dtype=torch.uint8), path)

    with pytest.raises(ValueError, match="Unknown augmentation type"):
        build_image_augmentation(
            {"placement": "uniform", "type": "random_cutout", "bank_path": str(path)},
            torch.device("cpu"),
        )


def test_random_shift_needs_no_overlay_bank():
    """The only type usable where no ImageNet bank exists, e.g. Cluster B."""
    _, aug = build_image_augmentation(
        {"placement": "uniform", "type": "random_shift", "shift_pad": 4},
        torch.device("cpu"),
        seed=0,
    )

    assert isinstance(aug, RandomShift)


def test_overlay_shift_composes_both(tmp_path):
    path = tmp_path / "bank.pt"
    torch.save(torch.randint(0, 256, (8, 8, 8, 3), dtype=torch.uint8), path)

    _, aug = build_image_augmentation(
        {
            "placement": "uniform",
            "type": "random_overlay_shift",
            "bank_path": str(path),
            "shift_pad": 3,
        },
        torch.device("cpu"),
        seed=0,
    )

    assert isinstance(aug, ComposeAugmentation)
    assert [type(a).__name__ for a in aug.augmentations] == [
        "RandomOverlay",
        "RandomShift",
    ]


def test_shift_pad_mode_constant_introduces_a_border():
    """The ablation: a constant fill is visible, which is why it is not default."""
    flat = torch.full((8, 3, 16, 16), 0.7)
    shift = RandomShift(
        pad=5,
        pad_mode="constant",
        pad_value=-1.0,
        generator=torch.Generator().manual_seed(0),
    )

    out = shift([flat])[0]

    assert out.min() < 0.0, "black border should appear somewhere in the batch"


def test_shift_pad_mode_is_validated():
    with pytest.raises(ValueError, match="pad_mode must be"):
        RandomShift(pad=4, pad_mode="wrap")


def test_shift_pad_mode_defaults_to_replicate():
    assert RandomShift(pad=4).pad_mode == "replicate"


def test_builder_passes_pad_mode_through():
    _, aug = build_image_augmentation(
        {
            "placement": "uniform",
            "type": "random_shift",
            "shift_pad": 4,
            "shift_pad_mode": "constant",
            "shift_pad_value": 0.0,
        },
        torch.device("cpu"),
        seed=0,
    )

    assert aug.pad_mode == "constant"
    assert aug.pad_value == 0.0


# --- GR00T view plumbing ----------------------------------------------------


def test_gr00t_eagle_preprocessing_matches_the_augmentation_image_range():
    """GR00T augments ``eagle_pixel_values`` in place, with no conversion.

    That is only correct because Eagle2 preprocesses with mean/std 0.5 and
    rescale 1/255, i.e. ``uint8 / 255 * 2 - 1`` -- the same [-1, 1] range
    openpi's ``Observation.from_dict`` produces and the range this module
    documents. If NVIDIA ever changes those constants the blend would run in
    the wrong space and silently corrupt every augmented arm, so pin them.
    """
    gr00t = pytest.importorskip("gr00t")
    config_path = (
        pathlib.Path(gr00t.__file__).parent
        / "model"
        / "backbone"
        / "eagle2_hg_model"
        / "preprocessor_config.json"
    )
    if not config_path.is_file():
        pytest.skip(f"Eagle2 preprocessor config not found at {config_path}")
    cfg = json.loads(config_path.read_text())

    assert cfg["do_rescale"] and cfg["do_normalize"]
    assert cfg["rescale_factor"] == pytest.approx(1 / 255)
    assert cfg["image_mean"] == [0.5, 0.5, 0.5]
    assert cfg["image_std"] == [0.5, 0.5, 0.5]


def test_gr00t_camera_split_and_restack_round_trips():
    """The actor pass must see exactly the tensor it saw before this change.

    ``default_forward`` splits ``[B, cameras, 3, H, W]`` into per-camera views
    so the augmentation can draw a separate overlay per camera, then restacks
    and folds cameras into the batch. A transposed restack would feed the wrist
    image where the agentview belongs -- no crash, just a quietly wrong run.
    """
    pixel_values = torch.randn(4, 2, 3, 8, 8)

    actor, critic = split_views_for_placement(
        pixel_values.unbind(dim=1), AugmentationPlacement.NONE, None
    )
    restacked = torch.stack(actor, dim=1)

    assert critic is actor, "the baseline must not pay for a second pass"
    assert torch.equal(restacked, pixel_values)
    assert torch.equal(
        restacked.reshape(-1, *restacked.shape[2:]),
        pixel_values.reshape(-1, *pixel_values.shape[2:]),
    )
