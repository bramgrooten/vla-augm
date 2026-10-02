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

"""Best-checkpoint selection for the embodied runner.

Covers the pure decision (``is_new_best``) and the ``_maybe_save_best_checkpoint``
state machine (improvement saves, non-improvement/tie/disabled/missing-metric
does not), without standing up Ray workers: the runner is built via ``__new__``
and ``_save_checkpoint`` is replaced with a recorder.
"""

import logging

import pytest

from rlinf.runners.embodied_runner import EmbodiedRunner, is_new_best


def _make_runner(*, enabled=True, metric="eval/success_once", mode="max"):
    """Build an EmbodiedRunner shell with only the best-tracking state wired.

    ``__init__`` is skipped (it spins up Ray), so every attribute the method
    touches is set by hand here. ``_save_checkpoint`` records (is_best) calls
    instead of writing disk.
    """
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner._save_best_enabled = enabled
    runner._best_metric_key = metric
    runner._best_mode = mode
    runner._best_metric_value = float("-inf") if mode == "max" else float("inf")
    runner._best_metric_step = -1
    runner.logger = logging.getLogger("test")
    saves: list[bool] = []
    runner._save_checkpoint = lambda is_best=False: saves.append(is_best)
    runner._saves = saves
    return runner


# --------------------------------------------------------------------------- #
# is_new_best (pure)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,current,mode,expected",
    [
        (0.9, 0.8, "max", True),
        (0.8, 0.8, "max", False),  # tie -> not better (earliest wins)
        (0.7, 0.8, "max", False),
        (0.1, 0.2, "min", True),
        (0.2, 0.2, "min", False),
        (0.3, 0.2, "min", False),
    ],
)
def test_is_new_best_modes_and_ties(value, current, mode, expected):
    assert is_new_best(value, current, mode) is expected


def test_is_new_best_rejects_unknown_mode():
    with pytest.raises(ValueError):
        is_new_best(1.0, 0.0, "median")


# --------------------------------------------------------------------------- #
# _maybe_save_best_checkpoint (state machine)
# --------------------------------------------------------------------------- #


def test_first_eval_always_saves_best():
    runner = _make_runner()
    saved = runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.90})

    assert saved is True
    assert runner._saves == [True]
    assert runner._best_metric_value == 0.90
    assert runner._best_metric_step == 20


def test_improvement_saves_new_best():
    runner = _make_runner()
    runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.90})
    saved = runner._maybe_save_best_checkpoint(40, {"eval/success_once": 0.94})

    assert saved is True
    assert runner._saves == [True, True]
    assert runner._best_metric_value == 0.94
    assert runner._best_metric_step == 40


def test_regression_does_not_save():
    runner = _make_runner()
    runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.94})
    saved = runner._maybe_save_best_checkpoint(40, {"eval/success_once": 0.88})

    assert saved is False
    assert runner._saves == [True]  # only the first
    assert runner._best_metric_value == 0.94
    assert runner._best_metric_step == 20


def test_tie_keeps_earliest():
    runner = _make_runner()
    runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.94})
    saved = runner._maybe_save_best_checkpoint(200, {"eval/success_once": 0.94})

    assert saved is False
    assert runner._saves == [True]
    assert runner._best_metric_step == 20  # unchanged


def test_disabled_never_saves():
    runner = _make_runner(enabled=False)
    saved = runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.99})

    assert saved is False
    assert runner._saves == []
    assert runner._best_metric_step == -1


def test_missing_metric_key_does_not_save():
    runner = _make_runner(metric="eval/success_once")
    saved = runner._maybe_save_best_checkpoint(20, {"eval/return": 0.5})

    assert saved is False
    assert runner._saves == []


def test_non_numeric_metric_does_not_save():
    runner = _make_runner()
    saved = runner._maybe_save_best_checkpoint(
        20, {"eval/success_once": "not-a-number"}
    )

    assert saved is False
    assert runner._saves == []


def test_min_mode_tracks_lower_values():
    runner = _make_runner(mode="min", metric="eval/loss")
    runner._maybe_save_best_checkpoint(20, {"eval/loss": 0.5})
    saved_higher = runner._maybe_save_best_checkpoint(40, {"eval/loss": 0.6})
    saved_lower = runner._maybe_save_best_checkpoint(60, {"eval/loss": 0.3})

    assert saved_higher is False
    assert saved_lower is True
    assert runner._best_metric_value == 0.3
    assert runner._best_metric_step == 60
    assert runner._saves == [True, True]


def test_save_checkpoint_called_with_is_best_true():
    """The best path must pass is_best=True so it lands in best_model/ and is
    exempt from latest-checkpoint pruning."""
    runner = _make_runner()
    runner._maybe_save_best_checkpoint(20, {"eval/success_once": 0.9})

    assert runner._saves == [True]  # is_best=True
