#!/usr/bin/env python3
"""Assert the Exp 0(b) arm configs differ only in the variable under study.

The arms are standalone files rather than one config plus a command-line
override, because Hydra will not let a non-primary config set
hydra.searchpath, so they cannot inherit from the baseline. Standalone copies
invite drift, and a placement study is worthless if the arms differ in
anything else -- so check it mechanically.

Run: python vla_augm/utils/check_arm_configs.py
Exits non-zero on any unintended difference.
"""

from __future__ import annotations

import os
import sys

BASE = "libero_spatial_ppo_pi05_4gpu"
ARMS = {
    f"{BASE}_augm_unif": "uniform",
    f"{BASE}_augm_critic": "critic_only",
}
# The only key allowed to differ, plus fields that name the arm.
ALLOWED = {
    "algorithm.augmentation.placement",
    # critic_only runs a second VLM prefix pass and cannot fit the baseline's
    # micro_batch_size. global_batch_size is identical, so grad accumulation
    # absorbs the difference and the optimisation is unchanged -- this is a
    # memory knob, not a hyperparameter of the experiment.
    "actor.micro_batch_size",
}


def flatten(node, prefix=""):
    out = {}
    if hasattr(node, "items"):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


def main() -> int:
    os.environ.setdefault(
        "EMBODIED_PATH", os.path.abspath("examples/embodiment")
    )
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("vla_augm/config/libero")
    problems = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = flatten(compose(config_name=BASE))
        for arm, expected in ARMS.items():
            cfg = flatten(compose(config_name=arm))

            actual = cfg.get("algorithm.augmentation.placement")
            if actual != expected:
                problems.append(f"{arm}: placement is {actual!r}, expected {expected!r}")

            for key in sorted(set(base) | set(cfg)):
                if key in ALLOWED:
                    continue
                if base.get(key) != cfg.get(key):
                    problems.append(
                        f"{arm}: {key} = {cfg.get(key)!r}, baseline has {base.get(key)!r}"
                    )
            print(f"{arm:46s} placement={actual}")

    for problem in problems:
        print(f"  DRIFT: {problem}")
    print(f"\n{len(ARMS)} arms checked, {len(problems)} unintended difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
