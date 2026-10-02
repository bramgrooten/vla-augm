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

"""Reconfiguring LIBERO envs must dispatch to every subprocess before waiting.

A rebuild closes the robosuite env and constructs a fresh OffScreenRenderEnv
(MuJoCo XML load plus model compile). Each env owns a subprocess, so the
rebuilds can run concurrently -- but only if the parent sends every command
before it blocks on the first reply.
"""

import unittest

import numpy as np

from rlinf.envs.libero.venv import ReconfigureSubprocEnv, ReconfigureSubprocEnvWorker


class FakePipe:
    """Records the order of send/recv without a real subprocess."""

    def __init__(self, log, index):
        self.log = log
        self.index = index

    def send(self, payload):
        self.log.append(("send", self.index, payload[1]))

    def recv(self):
        self.log.append(("recv", self.index))
        return None


class FakeWorker(ReconfigureSubprocEnvWorker):
    """A worker with the pipe stubbed out; no process is spawned."""

    def __init__(self, log, index):
        self.parent_remote = FakePipe(log, index)


def make_env(num_workers, log):
    env = object.__new__(ReconfigureSubprocEnv)
    env.workers = [FakeWorker(log, i) for i in range(num_workers)]
    env.is_async = False
    env._assert_is_not_closed = lambda: None
    env._wrap_id = lambda i: np.arange(num_workers) if i is None else np.asarray(i)
    return env


class TestReconfigureDispatch(unittest.TestCase):
    def test_all_sends_precede_all_recvs(self):
        log = []
        env = make_env(4, log)

        env.reconfigure_env_fns([{"p": i} for i in range(4)], id=None)

        kinds = [entry[0] for entry in log]
        self.assertEqual(kinds, ["send"] * 4 + ["recv"] * 4)

    def test_every_worker_is_reconfigured_with_its_own_param(self):
        log = []
        env = make_env(3, log)

        env.reconfigure_env_fns([{"p": 10}, {"p": 11}, {"p": 12}], id=None)

        sends = [(entry[1], entry[2]) for entry in log if entry[0] == "send"]
        self.assertEqual(sends, [(0, {"p": 10}), (1, {"p": 11}), (2, {"p": 12})])
        recvs = sorted(entry[1] for entry in log if entry[0] == "recv")
        self.assertEqual(recvs, [0, 1, 2])

    def test_subset_of_envs_only_touches_those_workers(self):
        """_reconfigure passes just the envs whose task changed."""
        log = []
        env = make_env(4, log)

        env.reconfigure_env_fns([{"p": "a"}, {"p": "b"}], id=[1, 3])

        self.assertEqual(
            log,
            [
                ("send", 1, {"p": "a"}),
                ("send", 3, {"p": "b"}),
                ("recv", 1),
                ("recv", 3),
            ],
        )

    def test_single_worker_helper_still_round_trips(self):
        """reconfigure_env_fn keeps its send-then-wait contract for callers."""
        log = []
        worker = FakeWorker(log, 0)

        worker.reconfigure_env_fn({"p": 1})

        self.assertEqual(log, [("send", 0, {"p": 1}), ("recv", 0)])


if __name__ == "__main__":
    unittest.main()
