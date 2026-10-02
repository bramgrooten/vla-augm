#!/bin/bash
# Gate before the GR00T augmentation grid goes out. Must complete END TO END.
#
# Usage:
#   bash vla_augm/scripts/cluster_b/smoke_gr00t_augm.sh [placement] [epochs]
#     placement  default critic_only (the anchor arm, and the most demanding:
#                two Eagle2 passes plus the overlay bank)
#     epochs     default 2
#
# What counts as passing -- all four, and nothing less:
#   1. an "Epoch N/N" line for every epoch. A rollout progress bar reaching
#      100% is NOT an epoch completing: nothing is logged until the epoch
#      boundary, and reading the bar as success has twice produced a false
#      "training works" on Cluster A.
#   2. an "eval=" entry in the Time table (the 500-episode in-distribution
#      eval actually ran).
#   3. a global_step_N directory on disk with actor/ inside it.
#   4. exit status 0.
#
# It also produces the two numbers the budget rests on: seconds per epoch under
# pipeline_stage_num 2, and the overlay bank load time. Compare epoch time
# against the 810.7 s measured at stage 1 (job 5020181) before trusting the
# ~-13% that stage 2 is supposed to give.
#
# 90 minutes of walltime, deliberately generous. Several "hangs" on Cluster A
# turned out to be jobs given less time than startup + bank load + epoch +
# eval + a multi-GB checkpoint write needs. Before diagnosing a deadlock here,
# check the walltime and compare rollout progress against a known-good run --
# it is a 30-second check. Equally, a py-spy dump showing half the workers idle
# at one instant is normal, not a deadlock.

set -euo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}"

PLACEMENT="${1:-critic_only}"
EPOCHS="${2:-2}"
SEED=52
TAG="smoke_${PLACEMENT}"

OVERRIDES=(
    "algorithm.augmentation.placement=${PLACEMENT}"
    "algorithm.augmentation.type=random_overlay"
    "algorithm.augmentation.alpha=0.5"
    "runner.max_epochs=${EPOCHS}"
    # Eval and checkpoint at every epoch so the gate exercises both without
    # needing the arms' 25-epoch cadence. They still share a grid.
    "runner.save_interval=1"
    "runner.val_check_interval=1"
)
if [[ "${PLACEMENT}" == "critic_only" || "${PLACEMENT}" == "actor_only" ]]; then
    OVERRIDES+=("actor.micro_batch_size=8")
fi

echo "smoke: ${TAG}, ${EPOCHS} epoch(s), seed ${SEED}"
echo "overrides: ${OVERRIDES[*]}"

sbatch --parsable \
    --time=01:30:00 \
    --export=ALL,RUN_TAG="${TAG}" \
    --job-name="gr00t-smoke-${PLACEMENT}" \
    vla_augm/scripts/cluster_b/run_gr00t_augm_4gpu.sh "${SEED}" "${OVERRIDES[@]}"
