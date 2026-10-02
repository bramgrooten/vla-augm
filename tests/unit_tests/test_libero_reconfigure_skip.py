# Copyright 2026 The RLinf Authors.
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

"""LiberoEnv._reconfigure must rebuild exactly the envs whose params changed.

Rebuilding closes the env and constructs a fresh OffScreenRenderEnv (MuJoCo XML
load plus model compile), which is ~92% of a reset by wall clock. Training used
to rebuild every env on every reset even though use_fixed_reset_state_ids keeps
the task fixed, so the rebuild produced an identical env.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

# libero_env pulls robosuite/libero at import time; stub them when absent so the
# reconfigure logic can be tested without a rendering stack.
for module in ("robosuite", "libero", "libero.libero", "mujoco"):
    if module not in sys.modules:
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - any import failure means "stub it"
            sys.modules[module] = MagicMock()

from rlinf.envs.libero.libero_env import LiberoEnv  # noqa: E402


class FakeVecEnv:
    def __init__(self):
        self.reconfigure_calls = []

    def reconfigure_env_fns(self, params, ids):
        self.reconfigure_calls.append((list(ids), list(params)))

    def seed(self, _seed):
        pass

    def reset(self, id=None):
        pass

    def set_init_state(self, init_state=None, id=None):
        pass


class ReconfigureHarness(LiberoEnv):
    """LiberoEnv with construction skipped and the env stack faked out."""

    def __init__(self, num_envs, params_by_task):
        self.num_envs = num_envs
        self.seed = 0
        self.is_eval = False
        self._active_env_fn_params = {}
        self.task_ids = [-1] * num_envs
        self.trial_ids = [-1] * num_envs
        self.env = FakeVecEnv()
        self.cfg = MagicMock()
        self.cfg.get.return_value = "standard"
        self._params_by_task = params_by_task
        self._next_task_ids = None

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        ids = list(reset_state_ids)
        return ids, [0] * len(ids)

    def get_env_fn_params(self, env_idx):
        # Mirrors the real method: params depend only on the env's task id.
        return [{"bddl_file_name": self._params_by_task[self.task_ids[i]]} for i in env_idx]

    def _get_reset_states(self, env_idx=None):
        return None


class TestReconfigureSkip(unittest.TestCase):
    def setUp(self):
        os.environ["LIBERO_TYPE"] = "standard"
        self.params = {t: f"task_{t}.bddl" for t in range(8)}

    def test_first_reset_builds_every_env(self):
        env = ReconfigureHarness(4, self.params)

        env._reconfigure([0, 1, 2, 3], np.arange(4))

        ids, _ = env.env.reconfigure_calls[0]
        self.assertEqual(ids, [0, 1, 2, 3])

    def test_repeat_reset_with_same_tasks_rebuilds_nothing(self):
        """The training case: fixed reset_state_ids, so nothing changes."""
        env = ReconfigureHarness(4, self.params)
        env._reconfigure([0, 1, 2, 3], np.arange(4))
        env.env.reconfigure_calls.clear()

        for _ in range(3):
            env._reconfigure([0, 1, 2, 3], np.arange(4))

        self.assertEqual(env.env.reconfigure_calls, [])

    def test_only_changed_envs_are_rebuilt(self):
        env = ReconfigureHarness(4, self.params)
        env._reconfigure([0, 1, 2, 3], np.arange(4))
        env.env.reconfigure_calls.clear()

        # env 1 and env 3 move to new tasks; 0 and 2 stay put.
        env._reconfigure([0, 5, 2, 6], np.arange(4))

        ids, params = env.env.reconfigure_calls[0]
        self.assertEqual(ids, [1, 3])
        self.assertEqual(
            params, [{"bddl_file_name": "task_5.bddl"}, {"bddl_file_name": "task_6.bddl"}]
        )

    def test_task_ids_are_updated_even_when_no_rebuild_happens(self):
        env = ReconfigureHarness(4, self.params)
        env._reconfigure([0, 1, 2, 3], np.arange(4))
        env.env.reconfigure_calls.clear()

        env._reconfigure([0, 1, 2, 3], np.arange(4))

        self.assertEqual(list(env.task_ids), [0, 1, 2, 3])

    def test_changed_params_for_same_task_still_rebuild(self):
        """LIBERO-Pro/Plus training redraws a perturbation bddl per reset."""
        env = ReconfigureHarness(2, self.params)
        env._reconfigure([0, 1], np.arange(2))
        env.env.reconfigure_calls.clear()

        # Same task ids, but the variant picked a different bddl for env 0.
        env._params_by_task = dict(self.params)
        env._params_by_task[0] = "task_0_light_3.bddl"
        env._reconfigure([0, 1], np.arange(2))

        ids, params = env.env.reconfigure_calls[0]
        self.assertEqual(ids, [0])
        self.assertEqual(params, [{"bddl_file_name": "task_0_light_3.bddl"}])

    def test_subset_reset_leaves_other_envs_alone(self):
        env = ReconfigureHarness(4, self.params)
        env._reconfigure([0, 1, 2, 3], np.arange(4))
        env.env.reconfigure_calls.clear()

        env._reconfigure([7], np.array([2]))

        ids, _ = env.env.reconfigure_calls[0]
        self.assertEqual(ids, [2])
        self.assertEqual(list(env.task_ids), [0, 1, 7, 3])


if __name__ == "__main__":
    unittest.main()
