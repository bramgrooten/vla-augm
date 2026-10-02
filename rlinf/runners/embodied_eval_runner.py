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

import typing
from collections import defaultdict

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        self._episode_results = eval_metrics_list
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _dump_episode_results(self, path: str):
        """Write one JSON line per episode, including its ``task_id`` tag.

        The aggregate in ``compute_evaluate_metrics`` averages over the whole
        sweep, which cannot be split by LIBERO-Plus axis afterwards. Keeping the
        per-episode rows lets a full-suite sweep be scored per axis offline, so
        the axis breakdown costs no extra environment stepping.
        """
        import json
        import os

        import torch

        episodes: dict[str, list] = defaultdict(list)
        for rank_metrics in self._episode_results:
            for key, value in rank_metrics.items():
                episodes[key].append(value.reshape(-1).float())
        if "task_id" not in episodes:
            self.logger.warning(
                "Eval returned no task_id tag; skipping per-episode dump."
            )
            return

        merged = {k: torch.cat(v, dim=0).tolist() for k, v in episodes.items()}
        keys = sorted(merged)
        num_episodes = len(merged["task_id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for i in range(num_episodes):
                row = {k: merged[k][i] for k in keys}
                row["task_id"] = int(row["task_id"])
                f.write(json.dumps(row) + "\n")
        self.logger.info(f"Wrote {num_episodes} episode results to {path}")

    def run(self):
        eval_metrics = self.evaluate()

        # Written before the aggregate is logged: the aggregate is the cheap
        # part, and a failure here should not cost the whole sweep.
        dump_path = self.cfg.runner.get("episode_results_path", None)
        if dump_path:
            self._dump_episode_results(dump_path)

        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
