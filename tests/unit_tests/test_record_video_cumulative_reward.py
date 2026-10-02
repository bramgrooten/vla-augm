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

"""The ``cumulative reward`` video overlay.

LIBERO pays its reward as a single +1 on the success frame. At 30 fps that is
one frame, so a viewer scrubbing a clip can miss the only evidence the episode
succeeded. ``video_cfg.cumulative_reward_on_video`` swaps the per-step reward
for a running total, which steps to 1 and stays.

The interesting case is the double-buffer one: with augmentation enabled the
wrapper draws every frame twice, once per stream, and a naive accumulator would
count each step twice and show 2.000 on success.
"""

from types import SimpleNamespace

import pytest

from rlinf.envs.wrappers.record_video import RecordVideo


class _StubEnv:
    """The minimum RecordVideo.__init__ touches."""

    seed = 0
    num_envs = 1
    metadata: dict = {}


def _wrapper(cumulative: bool) -> RecordVideo:
    cfg = SimpleNamespace(
        save_video=True,
        info_on_video=True,
        cumulative_reward_on_video=cumulative,
        video_base_dir="/tmp",
        fps=30,
    )
    # SimpleNamespace has no .get(); RecordVideo reaches config through
    # _cfg_get, so give it the dict-style accessor it expects.
    cfg.get = lambda key, default=None: getattr(cfg, key, default)
    return RecordVideo(_StubEnv(), cfg)


def _rewards_over(wrapper: RecordVideo, seq, stream: str = "main") -> list[float]:
    """Overlay value for each step of a one-env episode."""
    out = []
    for r in seq:
        item = wrapper._build_info_item(None, [r], None, env_id=0, stream=stream)
        out.append(item.get("cumulative reward", item.get("reward")))
    return out


def test_per_step_reward_is_the_default():
    """Untouched config keeps the old overlay, for the training clips."""
    w = _wrapper(cumulative=False)
    assert _rewards_over(w, [0.0, 0.0, 1.0, 0.0]) == [0.0, 0.0, 1.0, 0.0]


def test_cumulative_reward_latches_after_success():
    """0 until the success frame, then 1 and stays -- the whole point."""
    w = _wrapper(cumulative=True)
    assert _rewards_over(w, [0.0, 0.0, 1.0, 0.0, 0.0]) == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_streams_do_not_double_count():
    """Clean and augmented streams each accumulate once, not twice.

    With augmentation on, add_new_frames draws each frame into both buffers, so
    _build_info_item runs twice per step. A shared accumulator would reach 2.0
    on a single +1.
    """
    w = _wrapper(cumulative=True)
    for r in [0.0, 1.0]:
        for stream in ("clean", "main"):
            w._build_info_item(None, [r], None, env_id=0, stream=stream)
    assert w._cum_reward["main"][0] == pytest.approx(1.0)
    assert w._cum_reward["clean"][0] == pytest.approx(1.0)


def test_envs_are_accumulated_separately():
    """One env succeeding must not mark the others as successful."""
    w = _wrapper(cumulative=True)
    for env_id, reward in enumerate([1.0, 0.0, 0.0]):
        w._build_info_item(None, [1.0, 0.0, 0.0], None, env_id=env_id)
    assert w._cum_reward["main"] == {0: 1.0, 1: 0.0, 2: 0.0}


def test_total_resets_between_episodes():
    """Otherwise the second clip of a run opens at 1.000 and reads as success."""
    w = _wrapper(cumulative=True)
    _rewards_over(w, [0.0, 1.0])
    assert w._cum_reward["main"][0] == pytest.approx(1.0)

    # A flush is the episode boundary that fires reliably: auto_reset means the
    # env rolls into the next episode without reset() reaching this wrapper.
    w.render_images = []
    w.flush_video()
    assert _rewards_over(w, [0.0, 0.0]) == [0.0, 0.0]
