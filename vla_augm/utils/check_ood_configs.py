#!/usr/bin/env python
"""Validate env.eval_ood across all vla_augm LIBERO configs without launching a job.

For every config that enables the OOD eval, checks that the axis task pool fits
in ``total_num_envs x rollout_epoch`` and that the OOD episode horizon matches
the suite's in-distribution eval. Catches the mistakes that would otherwise only
surface after a queue wait.

Usage: python vla_augm/utils/check_ood_configs.py
Exits non-zero if any config is misconfigured.
"""

from __future__ import annotations

import glob
import os
import sys

# vla_augm/utils/<file> -> repo root is three levels up.
REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.environ.setdefault("EMBODIED_PATH", os.path.join(REPO, "examples/embodiment"))
os.environ.setdefault("REPO_PATH", REPO)
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir  # noqa: E402

from rlinf.envs.libero.libero_plus_axes import get_axis_task_ids  # noqa: E402

CONFIG_DIR = os.path.join(REPO, "vla_augm/config/libero")


def main() -> int:
    failures = 0
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))):
        name = os.path.basename(path)[:-5]
        try:
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
                cfg = compose(config_name=name)
        except Exception as e:  # noqa: BLE001 - report and keep checking others
            print(f"  COMPOSE FAIL {name}: {e}")
            failures += 1
            continue

        # Eval-only configs have no training loop, so no periodic OOD eval to
        # check. Skip rather than report a failure.
        if cfg.runner.get("task_type", "") == "embodied_eval" or cfg.runner.get(
            "only_eval", False
        ):
            print(f"  {name:38s} eval-only, skipped")
            continue

        ood = cfg.env.get("eval_ood", None)
        if ood is None:
            print(f"  {name:38s} NO eval_ood")
            failures += 1
            continue
        if not ood.get("enable", False):
            print(f"  {name:38s} disabled")
            continue

        # Same fraction the env will filter on, or the capacity check compares
        # the full axis pool against a budget sized for a subsample.
        axes = get_axis_task_ids(
            ood.task_suite_name,
            list(ood.axes),
            ood.get("task_classification_path", None),
            float(ood.get("subsample_frac", 1.0)),
        )
        n_tasks = sum(len(v) for v in axes.values())
        capacity = ood.total_num_envs * ood.rollout_epoch
        fits = capacity >= n_tasks
        # A horizon different from the suite's eval horizon makes the
        # in-distribution and OOD success rates non-comparable.
        same_horizon = ood.max_episode_steps == cfg.env.eval.max_episode_steps
        if not fits or not same_horizon:
            failures += 1
        print(
            f"  {name:38s} ENABLED {ood.task_suite_name:14s} "
            f"tasks={n_tasks:4d} cap={capacity:4d} {'OK' if fits else 'TOO SMALL'} "
            f"steps={ood.max_episode_steps}"
            f"{'' if same_horizon else ' MISMATCH vs eval'} "
            f"interval={cfg.runner.get('val_ood_check_interval')}"
        )

    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
