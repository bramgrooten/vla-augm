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

"""Recorded frames: camera selection and overlay augmentation.

The recorder previously only ever read ``main_images``, so videos showed one
camera and always the clean render -- neither matches what the policy consumes
when observation augmentation is on. These pin the frame the recorder builds,
without needing a LIBERO env.
"""

import numpy as np
import torch

from rlinf.algorithms.augmentation import OverlayImageBank, RandomOverlay
from rlinf.envs.wrappers.record_video import RecordVideo


class _Recorder(RecordVideo):
    """Bypass __init__: only the frame-building helpers are under test."""

    def __init__(self, views, frame_aug=None):
        self._views = views
        self._frame_aug = frame_aug
        self.env = object()
        self.video_cfg = {}


def _obs(height=8, width=8, batch=2):
    return {
        "main_images": np.full((batch, height, width, 3), 40, dtype=np.uint8),
        "wrist_images": np.full((batch, height, width, 3), 200, dtype=np.uint8),
    }


def test_default_records_the_main_camera_only():
    frame = _Recorder(["main"])._get_image_from_dict(_obs())

    assert frame.shape == (2, 8, 8, 3)
    assert (frame == 40).all()


def test_both_views_are_concatenated_side_by_side():
    frame = _Recorder(["main", "wrist"])._get_image_from_dict(_obs())

    assert frame.shape == (2, 8, 16, 3), "width should double, height unchanged"
    assert (frame[:, :, :8] == 40).all(), "main camera on the left"
    assert (frame[:, :, 8:] == 200).all(), "wrist camera on the right"


def test_mismatched_heights_fall_back_to_one_view():
    """Never emit a malformed frame just because a camera has another shape."""
    obs = _obs()
    obs["wrist_images"] = np.zeros((2, 4, 8, 3), dtype=np.uint8)

    frame = _Recorder(["main", "wrist"])._get_image_from_dict(obs)

    assert frame.shape == (2, 8, 8, 3)


def test_augmented_recording_changes_the_frame_and_keeps_dtype():
    bank = OverlayImageBank(torch.full((8, 8, 8, 3), 255, dtype=torch.uint8))
    aug = RandomOverlay(bank=bank, alpha=0.5, crop_scale=None)

    frame = _Recorder(["main"], frame_aug=aug)._get_image_from_dict(_obs())

    assert frame.dtype == np.uint8
    assert frame.shape == (2, 8, 8, 3)
    # 40 blended halfway to 255 lands near 147; must be neither input.
    assert not (frame == 40).all()
    assert abs(int(frame.mean()) - 147) < 4


def test_augmented_recording_covers_both_views():
    bank = OverlayImageBank(torch.full((8, 8, 8, 3), 255, dtype=torch.uint8))
    aug = RandomOverlay(bank=bank, alpha=1.0, crop_scale=None)

    frame = _Recorder(["main", "wrist"], frame_aug=aug)._get_image_from_dict(_obs())

    assert frame.shape == (2, 8, 16, 3)
    assert (frame > 250).all(), "alpha=1 replaces both halves"


def test_each_camera_is_augmented_separately_not_as_one_strip():
    """The reported bug: one overlay was stretched across both cameras.

    Augmenting after concatenation treats main|wrist as two halves of a single
    photo, so a giraffe's neck runs continuously across the seam. Each camera
    must get the overlay applied to it in its own frame.
    """
    from rlinf.algorithms.augmentation import AugmentationSequence

    # A bank whose images are a left-right gradient: if one overlay were
    # stretched over the concatenation, the two halves would differ.
    ramp = np.tile(np.linspace(0, 255, 8, dtype=np.uint8)[None, :, None], (8, 1, 3))
    bank = OverlayImageBank(torch.from_numpy(np.stack([ramp] * 4)))
    aug = RandomOverlay(
        bank=bank, alpha=1.0, crop_scale=None, sequence=AugmentationSequence.RANDOM
    )

    frame = _Recorder(["main", "wrist"], frame_aug=aug)._get_image_from_dict(_obs())

    assert frame.shape == (2, 8, 16, 3)
    left, right = frame[:, :, :8], frame[:, :, 8:]
    assert np.array_equal(left, right), (
        "each camera should carry the whole overlay, not half of a stretched one"
    )


def test_clean_stream_is_recorded_alongside_the_augmented_one():
    """Every augmented video must ship with its unaugmented counterpart."""
    bank = OverlayImageBank(torch.full((4, 8, 8, 3), 255, dtype=torch.uint8))
    rec = _Recorder(["main"], frame_aug=RandomOverlay(bank=bank, alpha=1.0))

    clean = rec._get_image_from_dict(_obs(), augment=False)
    augmented = rec._get_image_from_dict(_obs(), augment=True)

    assert (clean == 40).all(), "clean pass must not touch the augmentation"
    assert (augmented > 250).all()
