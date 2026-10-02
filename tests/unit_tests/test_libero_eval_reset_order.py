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

"""Eval reset-state ordering, including under a task filter.

A LIBERO eval pass is normally truncated: it runs
``total_num_envs x rollout_epoch`` episodes against a much larger pool of
(task, trial) reset states. The pool must therefore be trial-major, so a
truncated pass covers one trial of every task instead of every trial of the
first few tasks. This was silently wrong for the filtered path, which is used
by the periodic LIBERO-Plus OOD eval: a 960-episode pass covered ~19 tasks
rather than the intended 935.
"""

import numpy as np

from rlinf.envs.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
)


def _task_of(reset_state_id, cumsum):
    """Map a reset-state id back to the task that owns it."""
    return int(np.searchsorted(cumsum, reset_state_id, side="right"))


def test_unfiltered_pool_is_trial_major():
    trials_per_task = [50] * 6
    cumsum = np.cumsum(trials_per_task)

    pool = build_interleaved_eval_reset_state_ids(trials_per_task, cumsum)

    assert len(pool) == 300
    # First pass over the pool touches every task once, in order.
    assert [_task_of(r, cumsum) for r in pool[:6]] == [0, 1, 2, 3, 4, 5]
    assert [_task_of(r, cumsum) for r in pool[6:12]] == [0, 1, 2, 3, 4, 5]


def test_filtered_pool_is_trial_major_over_the_filtered_tasks():
    trials_per_task = [50] * 6
    cumsum = np.cumsum(trials_per_task)
    task_ids = [0, 2, 5]

    pool = build_interleaved_eval_reset_state_ids(
        trials_per_task, cumsum, task_ids=task_ids
    )

    assert len(pool) == 150, "pool should hold every trial of the filtered tasks"
    assert [_task_of(r, cumsum) for r in pool[:3]] == task_ids
    assert [_task_of(r, cumsum) for r in pool[3:6]] == task_ids
    # The regression: a pass as short as the task count must still cover them all.
    assert sorted({_task_of(r, cumsum) for r in pool[: len(task_ids)]}) == task_ids


def test_filtered_pool_respects_ragged_trial_counts():
    """Tasks with fewer trials drop out of later rounds without shifting others."""
    trials_per_task = [3, 1, 2]
    cumsum = np.cumsum(trials_per_task)

    pool = build_interleaved_eval_reset_state_ids(
        trials_per_task, cumsum, task_ids=[0, 1, 2]
    )

    assert len(pool) == sum(trials_per_task)
    assert [_task_of(r, cumsum) for r in pool] == [0, 1, 2, 0, 2, 0]


def test_round_robin_distribution_loses_no_reset_state():
    trials_per_task = [50] * 6
    cumsum = np.cumsum(trials_per_task)
    pool = build_interleaved_eval_reset_state_ids(
        trials_per_task, cumsum, task_ids=[0, 2, 5]
    )

    distributed = distribute_reset_state_ids_round_robin(pool, 4)

    valid = distributed[distributed >= 0]
    assert len(valid) == len(pool)
    assert sorted(valid.tolist()) == sorted(pool.tolist())


def test_truncated_multi_rank_pass_covers_every_filtered_task():
    """The property the OOD eval depends on, across ranks."""
    num_tasks, num_ranks = 935, 4
    trials_per_task = [50] * num_tasks
    cumsum = np.cumsum(trials_per_task)
    task_ids = list(range(num_tasks))

    pool = build_interleaved_eval_reset_state_ids(
        trials_per_task, cumsum, task_ids=task_ids
    )
    distributed = distribute_reset_state_ids_round_robin(pool, num_ranks)

    # Each rank consumes the first (total_num_envs / num_ranks) * rollout_epoch
    # entries of its own slice: 40 envs over 4 ranks, 24 epochs.
    per_rank_budget = (40 // num_ranks) * 24
    seen = set()
    for rank in range(num_ranks):
        slice_ = distributed[rank][:per_rank_budget]
        seen.update(_task_of(r, cumsum) for r in slice_ if r >= 0)

    assert seen == set(task_ids), "a full-budget pass must reach every task once"
