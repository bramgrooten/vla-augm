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

import json
import logging
import os
import queue
import shutil
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

import torch
from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def is_new_best(value: float, current_best: float, mode: str) -> bool:
    """Return True when ``value`` beats ``current_best`` for the given mode.

    ``mode="max"`` -> strictly greater (first occurrence of a peak wins, so the
    best checkpoint does not thrash on ties); ``mode="min"`` -> strictly less.
    Extracted as a pure function so it can be unit-tested without Ray.
    """
    if mode == "max":
        return value > current_best
    if mode == "min":
        return value < current_best
    raise ValueError(f"best_metric_mode must be 'max' or 'min', got {mode!r}")


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
        env_ood: Union["EnvWorker", None] = None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.env_ood = env_ood
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )

        # Step-gated profiling: ``cluster.profiling.steps`` lists the global step
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None
        )
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.env_ood is not None:
            # Dedicated channels so an OOD pass never shares a queue with the
            # in-distribution eval, which uses a different env group.
            self.env_ood_channel = Channel.create("EnvOOD")
            self.rollout_ood_channel = Channel.create("RolloutOOD")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self._init_ood_eval()
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Best-checkpoint tracking. The metric is read from the deterministic
        # in-distribution eval dict (keys prefixed with ``eval/``); defaults to
        # ``eval/success_once``. ``mode`` is "max" (higher is better) or "min".
        self._save_best_enabled = bool(
            self.cfg.runner.get("save_best_checkpoint", True)
        )
        self._best_metric_key = self.cfg.runner.get("best_metric", "eval/success_once")
        self._best_mode = self.cfg.runner.get("best_metric_mode", "max")
        self._best_metric_value: float = float("inf")
        self._best_metric_step: int = -1
        if self._best_mode == "max":
            self._best_metric_value = float("-inf")

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (
                print_metrics_table,
                (
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

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
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        # task_id/trial_id tag episodes for grouping; their mean is meaningless.
        return {
            k: v for k, v in eval_metrics.items() if k not in ("task_id", "trial_id")
        }

    def _init_ood_eval(self):
        """Resolve the OOD eval schedule and the task-id -> axis mapping."""
        self._ood_axis_of_task: dict[int, str] = {}
        self._ood_axes: list[str] = []
        self._ood_interval = -1
        self._ood_rollout_epoch = 1
        self._ood_eval_batch_size = None
        self._ood_log_per_task = False
        self._ood_task_log_dir = None
        self._ood_failures = 0
        self._ood_max_failures = 3
        if self.env_ood is None:
            return

        from rlinf.envs.libero.libero_plus_axes import build_task_id_to_axis

        ood_cfg = self.cfg.env.eval_ood
        self._ood_axes = list(ood_cfg.axes)
        self._ood_rollout_epoch = int(ood_cfg.rollout_epoch)
        # Must match the OOD env group's own send size, which can differ from
        # the in-distribution eval's (OOD runs fewer envs to save GPU memory).
        self._ood_eval_batch_size = int(ood_cfg.total_num_envs) // int(
            self.cfg.rollout.pipeline_stage_num
        )
        # Default to the in-distribution eval cadence so the ID and OOD curves
        # land on the same x-grid (the ID-minus-OOD gap is the headline plot).
        self._ood_interval = int(
            self.cfg.runner.get("val_ood_check_interval", None)
            or self.cfg.runner.val_check_interval
        )
        self._ood_log_per_task = bool(ood_cfg.get("log_per_task_wandb", False))
        # Must use the same fraction as the env's task filter, or episodes get
        # labelled with an axis map that does not match what was evaluated.
        self._ood_axis_of_task = build_task_id_to_axis(
            ood_cfg.task_suite_name,
            self._ood_axes,
            ood_cfg.get("task_classification_path", None),
            float(ood_cfg.get("subsample_frac", 1.0)),
        )
        self._ood_task_log_dir = os.path.join(
            self.cfg.runner.logger.log_path, "ood_eval"
        )
        os.makedirs(self._ood_task_log_dir, exist_ok=True)
        self.logger.info(
            f"OOD eval enabled: axes={self._ood_axes}, "
            f"{len(self._ood_axis_of_task)} tasks, every {self._ood_interval} steps."
        )

    def _evaluate_ood_guarded(self, step: int) -> dict:
        """Run the OOD eval without letting its failure abort training.

        Training is the expensive artefact of a multi-day run; a missed eval
        point is not worth losing it. Repeated failures switch the OOD eval off
        rather than burning eval time on every checkpoint.
        """
        try:
            metrics = self.evaluate_ood(step)
        except Exception:
            self._ood_failures += 1
            self.logger.exception(
                f"OOD eval failed at step {step} "
                f"({self._ood_failures}/{self._ood_max_failures}); training continues."
            )
            if self._ood_failures >= self._ood_max_failures:
                self._ood_interval = -1
                self.logger.error(
                    "Disabling OOD eval after "
                    f"{self._ood_failures} consecutive failures."
                )
            return {}
        self._ood_failures = 0
        return metrics

    def evaluate_ood(self, step: int) -> dict:
        """Evaluate on the LIBERO-Plus OOD axes and return flat wandb metrics."""
        env_handle: Handle = self.env_ood.evaluate(
            input_channel=self.env_ood_channel,
            rollout_channel=self.rollout_ood_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_ood_channel,
            output_channel=self.env_ood_channel,
            env_group_name=self.cfg.env.eval_ood.group_name,
            num_rollout_epochs=self._ood_rollout_epoch,
            eval_batch_size=self._ood_eval_batch_size,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        results = [r for r in env_results if r]
        return self._compute_ood_metrics(results, step)

    def _compute_ood_metrics(self, results: list[dict], step: int) -> dict:
        """Split per-episode eval tensors by OOD axis and aggregate.

        ``results`` holds one dict of aligned per-episode tensors per env rank,
        including the ``task_id`` tag added by ``LiberoEnv``. Grouping happens
        here, so the per-axis numbers cost no extra environment stepping.
        """
        episodes: dict[str, list] = defaultdict(list)
        for rank_metrics in results:
            for key, value in rank_metrics.items():
                episodes[key].append(value.reshape(-1).float())
        if "task_id" not in episodes:
            self.logger.warning(
                "OOD eval returned no task_id tag; skipping per-axis breakdown."
            )
            return {}

        merged = {k: torch.cat(v, dim=0) for k, v in episodes.items()}
        task_ids = merged["task_id"].to(torch.int64).tolist()
        # task_id/trial_id are grouping keys, not metrics worth averaging.
        metric_keys = [k for k in merged if k not in ("task_id", "trial_id")]
        axis_of_episode = [
            self._ood_axis_of_task.get(int(t), "unknown") for t in task_ids
        ]

        self._dump_ood_task_results(
            merged, task_ids, axis_of_episode, metric_keys, step
        )

        metrics: dict[str, float] = {}
        # Per-axis means, collected so the unweighted (macro) average over axes
        # can be logged alongside the pooled (micro) one under "all".
        per_axis_values: dict[str, list[float]] = {}
        for axis in self._ood_axes + ["all"]:
            mask = torch.tensor(
                [
                    a != "unknown" if axis == "all" else a == axis
                    for a in axis_of_episode
                ],
                dtype=torch.bool,
            )
            count = int(mask.sum())
            metrics[f"eval_ood/{axis}/num_trajectories"] = count
            if count == 0:
                self.logger.warning(f"OOD eval axis {axis!r} produced no episodes.")
                continue
            for key in metric_keys:
                value = float(merged[key][mask].mean())
                metrics[f"eval_ood/{axis}/{key}"] = value
                # success_once is the headline number and there is one per axis,
                # so mirror it into its own namespace. eval_ood/ carries every
                # metric for every axis, which makes the per-axis success curves
                # hard to find among them.
                if key == "success_once":
                    metrics[f"eval_ood_success_once/{axis}"] = value
                if axis != "all":
                    per_axis_values.setdefault(key, []).append(value)

        # "all" pools every episode and so weights an axis by its task count
        # (micro). The axes have very unequal sizes and the largest are the
        # hardest, so that is not the same as treating each axis equally. Log the
        # unweighted mean over axes (macro) next to it -- the two differ by 1-2
        # points, and a run compared against a paper number needs the matching
        # convention rather than whichever one happened to be plotted.
        for key, values in per_axis_values.items():
            if values:
                macro = sum(values) / len(values)
                metrics[f"eval_ood/all/{key}_macro"] = macro
                if key == "success_once":
                    metrics["eval_ood_success_once/all_macro"] = macro

        if self._ood_log_per_task:
            metrics.update(self._per_task_metrics(merged, task_ids, axis_of_episode))
        return metrics

    def _per_task_metrics(self, merged, task_ids, axis_of_episode) -> dict:
        """Optional per-task success series (one metric per task; off by default)."""
        success = merged.get("success_once", None)
        if success is None:
            return {}
        per_task: dict[str, list[float]] = defaultdict(list)
        for i, task_id in enumerate(task_ids):
            axis = axis_of_episode[i]
            if axis == "unknown":
                continue
            per_task[f"eval_ood_tasks/{axis}/task_{int(task_id)}"].append(
                float(success[i])
            )
        return {k: sum(v) / len(v) for k, v in per_task.items()}

    def _dump_ood_task_results(
        self, merged, task_ids, axis_of_episode, metric_keys, step: int
    ):
        """Write one JSON line per OOD episode for post-hoc per-task analysis.

        ~100 KB per eval point, so full granularity stays available without
        putting hundreds of near-empty series into the metric backend.
        """
        if self._ood_task_log_dir is None:
            return
        path = os.path.join(self._ood_task_log_dir, f"step_{step}.jsonl")
        trial_ids = merged.get("trial_id", None)
        try:
            with open(path, "w") as f:
                for i, task_id in enumerate(task_ids):
                    record = {
                        "step": step,
                        "task_id": int(task_id),
                        "axis": axis_of_episode[i],
                    }
                    if trial_ids is not None:
                        record["trial_id"] = int(trial_ids[i])
                    for key in metric_keys:
                        record[key] = float(merged[key][i])
                    f.write(json.dumps(record) + "\n")
        except OSError as e:
            self.logger.warning(f"Could not write OOD per-task results to {path}: {e}")

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        run_ood = self.env_ood is not None and (
            self._ood_interval > 0 and self.global_step % self._ood_interval == 0
        )

        eval_metrics = {}
        if run_val or run_ood:
            # One weight sync covers both passes.
            self.update_rollout_weights()

        if run_val:
            with self.timer("eval"):
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step)
                self._maybe_save_best_checkpoint(step, eval_metrics)
                # Log the running best so the best-so-far curve is visible even
                # on steps that did not improve it.
                if self._save_best_enabled and self._best_metric_step >= 0:
                    self.metric_logger.log(
                        data={
                            f"best/{self._best_metric_key.split('/', 1)[-1]}": (
                                self._best_metric_value
                            ),
                            "best/step": self._best_metric_step,
                        },
                        step=step,
                    )

        if run_ood:
            with self.timer("eval_ood"):
                ood_metrics = self._evaluate_ood_guarded(step)
                if ood_metrics:
                    self.metric_logger.log(data=ood_metrics, step=step)
                    eval_metrics.update(ood_metrics)

        if save_model:
            self._save_checkpoint()

        return eval_metrics

    def _log_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handle: Handle,
        rollout_handle: Handle,
        actor_training_handle: Handle,
        reward_handle: Handle | None,
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        logging_metrics = time_metrics
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    def _finish_run(self) -> None:
        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    def _open_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``start_profile`` to all compute worker groups for this step."""
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    def _close_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``stop_profile`` to all compute worker groups."""
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    def run(self):
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            self.logger.info(f"Epoch {_step + 1}/{self.max_steps}")
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_training_metrics = actor_training_handle.wait()
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            self.logger.info(f"Epoch {_step + 1}/{self.max_steps}")
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )
                # actor training.
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel
                )
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_results = actor_training_handle.wait()
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _save_checkpoint(self, is_best: bool = False):
        # Checkpoints can live on a separate (e.g. scratch) filesystem via
        # runner.checkpoint_path; fall back to the logger path when unset so
        # videos/logs stay put while heavy checkpoints go elsewhere.
        ckpt_root = (
            self.cfg.runner.get("checkpoint_path", None)
            or self.cfg.runner.logger.log_path
        )
        checkpoints_dir = os.path.join(
            ckpt_root,
            self.cfg.runner.logger.experiment_name,
            "checkpoints",
        )
        if is_best:
            # Stable name so the best checkpoint survives latest-checkpoint
            # pruning (its dir does not start with ``global_step_``).
            base_output_dir = os.path.join(checkpoints_dir, "best_model")
            tag = "best"
        else:
            base_output_dir = os.path.join(
                checkpoints_dir, f"global_step_{self.global_step}"
            )
            tag = f"step {self.global_step}"
        self.logger.info(f"Saving {tag} checkpoint at step {self.global_step}.")
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        # Prune older checkpoints once the new one is fully written, so a
        # failed save never takes the previous good checkpoint with it. Only
        # ``global_step_*`` dirs are pruned, so ``best_model`` is never removed.
        if not is_best and self.cfg.runner.get("keep_only_last_checkpoint", False):
            self._prune_old_checkpoints(checkpoints_dir, keep=base_output_dir)

    def _maybe_save_best_checkpoint(self, step: int, eval_metrics: dict) -> bool:
        """Save a ``best_model`` checkpoint when the tracked metric improves.

        Uses the deterministic in-distribution eval (``eval_metrics`` keys are
        prefixed with ``eval/``); the metric and mode come from
        ``runner.best_metric`` / ``runner.best_metric_mode``. Returns True when a
        new best was saved.
        """
        if not self._save_best_enabled:
            return False
        value = eval_metrics.get(self._best_metric_key)
        if not isinstance(value, (int, float)):
            return False
        if not is_new_best(value, self._best_metric_value, self._best_mode):
            return False
        self._best_metric_value = float(value)
        self._best_metric_step = step
        self._save_checkpoint(is_best=True)
        self.logger.info(
            f"New best {self._best_metric_key}={value:.4f} at step {step}; "
            f"saved to {os.path.join('checkpoints', 'best_model')}."
        )
        return True

    def _prune_old_checkpoints(self, checkpoints_dir: str, keep: str):
        # runner.keep_checkpoint_interval N (>0): never prune global_step_<k*N>
        # checkpoints, so e.g. every-100-epoch snapshots survive next to the
        # keep-only-last policy. Steps not divisible by N are pruned as usual.
        keep_interval = self.cfg.runner.get("keep_checkpoint_interval", None)
        for name in os.listdir(checkpoints_dir):
            path = os.path.join(checkpoints_dir, name)
            if not (
                name.startswith("global_step_")
                and os.path.isdir(path)
                and os.path.abspath(path) != os.path.abspath(keep)
            ):
                continue
            if keep_interval:
                try:
                    step = int(name.removeprefix("global_step_"))
                except ValueError:
                    step = -1
                if step > 0 and step % keep_interval == 0:
                    continue
            shutil.rmtree(path, ignore_errors=True)
            self.logger.info(f"Pruned old checkpoint {path}.")

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
