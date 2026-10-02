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

"""The cached value-head prefix mask must select exactly what it replaced.

get_value_from_vlm rebuilt a 968-element Python list on every call and used it
to index a [bs, 968, 2048] tensor. Sampling put that function at ~25% of the
rollout worker -- the critical path -- so the list is now cached as a device
tensor. That is only a valid optimisation if the selection is bit-identical,
which is what these check.
"""

import pytest
import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)


class _Config:
    def __init__(self, mode="mean_token", num_images=2):
        self.value_vlm_mode = mode
        self.num_images_in_input = num_images


class _Model:
    """Bare object carrying just the method under test and its config."""

    def __init__(self, mode="mean_token", num_images=2):
        self.config = _Config(mode, num_images)

    # _value_prefix_mask reads these off the class.
    TOKENS_PER_IMAGE = OpenPi0ForRLActionPrediction.TOKENS_PER_IMAGE
    NUM_IMAGE_SLOTS = OpenPi0ForRLActionPrediction.NUM_IMAGE_SLOTS
    _value_prefix_mask = OpenPi0ForRLActionPrediction._value_prefix_mask


def reference_mask(mode, num_images, lang_len, all_len):
    """The exact list construction that was inlined in get_value_from_vlm."""
    if mode == "mean_token":
        return (
            [True] * 256 * num_images
            + [False] * 256 * (3 - num_images)
            + [True] * lang_len
        )
    if mode == "last_token":
        return [False] * (all_len - 1) + [True] * 1
    if mode == "first_token":
        return [True] * 1 + [False] * (all_len - 1)
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", ["mean_token", "last_token", "first_token"])
@pytest.mark.parametrize(
    "lang_len,all_len", [(200, 968), (48, 816)]  # pi05 and pi0 token layouts
)
def test_cached_mask_matches_the_original_list(mode, lang_len, all_len):
    model = _Model(mode)

    got = model._value_prefix_mask(torch.device("cpu"), lang_len, all_len)
    want = reference_mask(mode, 2, lang_len, all_len)

    assert got.dtype == torch.bool
    assert got.tolist() == want


def test_masked_selection_is_identical():
    """What actually matters: the gathered tensor, not just the mask."""
    torch.manual_seed(0)
    prefix = torch.randn(3, 968, 16)
    model = _Model("mean_token")

    cached = prefix[:, model._value_prefix_mask(torch.device("cpu"), 200, 968), :]
    original = prefix[:, reference_mask("mean_token", 2, 200, 968), :]

    assert torch.equal(cached, original)
    # And the mean the value head consumes.
    assert torch.equal(cached.mean(dim=1), original.mean(dim=1))


def test_mask_is_cached_not_rebuilt():
    model = _Model()

    first = model._value_prefix_mask(torch.device("cpu"), 200, 968)
    second = model._value_prefix_mask(torch.device("cpu"), 200, 968)

    assert first is second, "the whole point is to build it once"


def test_cache_keys_on_mode_and_layout():
    """A shared cache must not hand back another mode's mask."""
    model = _Model("mean_token")
    mean_mask = model._value_prefix_mask(torch.device("cpu"), 200, 968)

    model.config.value_vlm_mode = "last_token"
    last_mask = model._value_prefix_mask(torch.device("cpu"), 200, 968)

    assert not torch.equal(mean_mask, last_mask)
    assert int(last_mask.sum()) == 1


def test_unknown_mode_raises():
    model = _Model("median_token")

    with pytest.raises(ValueError, match="Unknown value_vlm_mode"):
        model._value_prefix_mask(torch.device("cpu"), 200, 968)


# --- benchmark-agnostic token layout ---------------------------------------
#
# The lengths used to be matched from config_name against "pi05_"/"pi0_", which
# hardcodes LIBERO's numbers. Any other benchmark with a different language
# budget would silently get pi0's 48-token layout, or leave the names undefined.
# They are now read off prefix_output.


class _ModelWithMask(_Model):
    get_value_from_vlm = OpenPi0ForRLActionPrediction.get_value_from_vlm
    TOKENS_PER_IMAGE = OpenPi0ForRLActionPrediction.TOKENS_PER_IMAGE
    NUM_IMAGE_SLOTS = OpenPi0ForRLActionPrediction.NUM_IMAGE_SLOTS

    def __init__(self, mode="mean_token", num_images=2, width=8):
        super().__init__(mode, num_images)
        self.value_head = lambda x: x[:, :1]

    def _selected(self, prefix):
        mask = self._value_prefix_mask(
            prefix.device,
            prefix.shape[1] - self.NUM_IMAGE_SLOTS * self.TOKENS_PER_IMAGE,
            prefix.shape[1],
        )
        return prefix[:, mask, :]


@pytest.mark.parametrize(
    "tokens,expect_lang",
    [
        (968, 200),  # pi0.5 on LIBERO
        (816, 48),   # pi0 on LIBERO
        (868, 100),  # some other benchmark's language budget
    ],
)
def test_language_length_is_derived_from_the_tensor(tokens, expect_lang):
    model = _ModelWithMask()
    prefix = torch.randn(2, tokens, 8)

    selected = model._selected(prefix)

    # 2 real cameras + the derived language tokens.
    assert selected.shape[1] == 2 * 256 + expect_lang


def test_value_matches_a_hand_built_mask_on_a_novel_layout():
    """Bit-identity must hold for a layout the old string match never saw."""
    torch.manual_seed(0)
    prefix = torch.randn(3, 868, 8)
    model = _ModelWithMask()

    got = model._selected(prefix)
    want = prefix[:, reference_mask("mean_token", 2, 868 - 768, 868), :]

    assert torch.equal(got, want)


def test_too_few_tokens_raises_rather_than_silently_miscounting():
    model = _ModelWithMask()

    with pytest.raises(ValueError, match="fewer than the"):
        model.get_value_from_vlm(torch.randn(2, 100, 8))


def test_num_images_still_controls_which_slots_are_kept():
    prefix = torch.randn(2, 968, 8)

    two = _ModelWithMask(num_images=2)._selected(prefix)
    three = _ModelWithMask(num_images=3)._selected(prefix)

    assert two.shape[1] == 2 * 256 + 200
    assert three.shape[1] == 3 * 256 + 200
