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

import copy
import json

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "rlt_ac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for rlt_ac."
            )
        from rlinf.workers.actor.rlt_ac_policy_worker import RLTACFSDPPolicy

        actor_worker_cls = RLTACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        if use_training_pipeline:
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # Optional out-of-distribution eval env group (e.g. LIBERO-Plus axes).
    # It needs its own process-wide LIBERO_TYPE, which libero_env.py reads at
    # import time, so it must be a separate worker group rather than a second
    # env inside the training workers.
    env_ood_group = None
    ood_cfg_node = cfg.env.get("eval_ood", None)
    if ood_cfg_node is not None and ood_cfg_node.get("enable", False):
        from rlinf.envs.libero.libero_plus_axes import get_axis_task_ids

        env_world_size = component_placement.get_world_size("env")
        assert ood_cfg_node.total_num_envs % env_world_size == 0, (
            f"env.eval_ood.total_num_envs ({ood_cfg_node.total_num_envs}) must be "
            f"divisible by the env world size ({env_world_size})."
        )

        axis_task_ids = get_axis_task_ids(
            ood_cfg_node.task_suite_name,
            list(ood_cfg_node.axes),
            ood_cfg_node.get("task_classification_path", None),
            float(ood_cfg_node.get("subsample_frac", 1.0)),
        )
        task_id_filter = sorted(t for ids in axis_task_ids.values() for t in ids)
        capacity = ood_cfg_node.total_num_envs * ood_cfg_node.rollout_epoch
        if capacity < len(task_id_filter):
            msg = (
                f"OOD eval capacity {capacity} (total_num_envs x rollout_epoch) is "
                f"below the {len(task_id_filter)} selected tasks, so only the first "
                f"{capacity} would be evaluated. Raise env.eval_ood.rollout_epoch."
            )
            # Partial coverage is fine for a smoke test but silently biases a
            # real run: the pool is ordered, so a short pass only ever sees the
            # lowest task ids (one axis). Require opting in.
            assert ood_cfg_node.get("allow_partial_coverage", False), msg
            print(f"[OOD eval] WARNING: {msg}")

        ood_cfg = copy.deepcopy(cfg)
        with open_dict(ood_cfg):
            # EnvWorker builds its eval env from cfg.env.eval, so swap the OOD
            # block in and drop env.train: this group only ever evaluates.
            ood_cfg.env.eval = ood_cfg.env.eval_ood
            ood_cfg.env.eval.task_id_filter = task_id_filter
            ood_cfg.env.group_name = cfg.env.eval_ood.group_name
            ood_cfg.env.pop("train", None)
            ood_cfg.env.pop("eval_ood", None)
        print(
            f"[OOD eval] {len(task_id_filter)} tasks over axes "
            f"{ {a: len(i) for a, i in axis_task_ids.items()} }"
        )
        env_ood_group = EnvWorker.create_group(ood_cfg).launch(
            cluster,
            name=cfg.env.eval_ood.group_name,
            placement_strategy=env_placement,
            extra_env_vars=dict(cfg.env.eval_ood.get("env_vars", {})),
        )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
        env_ood=env_ood_group,
    )

    if env_ood_group is not None:
        env_ood_group.init_worker().wait()
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
