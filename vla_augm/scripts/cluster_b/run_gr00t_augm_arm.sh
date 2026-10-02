#!/bin/bash
# Launch one arm of the GR00T N1.5 augmentation placement study on Cluster B.
#
# Wraps run_gr00t_augm_4gpu.sh so an arm is named by what it varies, rather
# than by a hand-assembled list of Hydra overrides that is easy to get subtly
# wrong. The GR00T counterpart of vla_augm/scripts/cluster_a/run_augm_arm.sh, which runs
# the pi0.5 arms of the same grid on Cluster A.
#
# Usage:
#   bash vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh <placement> <type> [alpha] [seed]
#
#   placement  none | uniform | critic_only | actor_only
#   type       random_overlay | random_shift | random_overlay_shift
#   alpha      overlay strength; ignored by random_shift. Default 0.5.
#   seed       default 52
#
# Examples:
#   bash vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh critic_only random_overlay 0.5
#   bash vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh actor_only random_overlay
#   bash vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh critic_only random_shift
#
# SEGMENTS. An arm is ~58-62 h of wall time and no Cluster B QOS gives a slot
# that long (`default` caps at 48 h, and in practice jobs end sooner), so each
# arm is submitted as a chain of SEGMENTS jobs linked by --dependency=afterany.
# Every segment runs the same self-resuming script, so segment N+1 picks up
# from the last checkpoint segment N wrote and continues the same wandb run --
# and the config checkpoints every 5 epochs, so a segment that ends early
# loses at most ~1 h rather than a whole 25-epoch block.
#
# A segment that starts after the arm has already reached max_epochs exits
# within a few minutes (the epoch loop simply has no steps left), so a spare
# tail segment is cheap insurance rather than wasted budget -- hence 4 by
# default rather than the 2 the wall-clock arithmetic alone would suggest.
# `afterany` rather than `afterok` so a crash is also resumed from.
#
# Nothing here varies augmentation strength or the overlay bank: those are the
# anchor values in the config, and shrinking one arm's bank alone would
# confound the whole alpha sweep. Only placement, type and alpha move.

set -euo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}"

PLACEMENT="${1:?placement: none | uniform | critic_only | actor_only}"
AUG_TYPE="${2:?type: random_overlay | random_shift | random_overlay_shift}"
ALPHA="${3:-0.5}"
SEED="${4:-52}"

SEGMENTS="${SEGMENTS:-4}"

case "${PLACEMENT}" in
    none|uniform|critic_only|actor_only) ;;
    *) echo "bad placement: ${PLACEMENT}" >&2; exit 1 ;;
esac
case "${AUG_TYPE}" in
    random_overlay|random_shift|random_overlay_shift) ;;
    *) echo "bad type: ${AUG_TYPE}" >&2; exit 1 ;;
esac

# Tag: placement, then a short type code, then alpha when it applies. Matches
# the Cluster A naming so the two model families line up in wandb.
case "${AUG_TYPE}" in
    random_overlay)       TYPE_TAG="ovl" ;;
    random_shift)         TYPE_TAG="shift" ;;
    random_overlay_shift) TYPE_TAG="ovlshift" ;;
esac
TAG="augm_${PLACEMENT}_${TYPE_TAG}"
[[ "${AUG_TYPE}" != "random_shift" ]] && TAG="${TAG}_alpha${ALPHA/./}"
[[ "${PLACEMENT}" == "none" ]] && TAG="augm_none"

OVERRIDES=(
    "algorithm.augmentation.placement=${PLACEMENT}"
    "algorithm.augmentation.type=${AUG_TYPE}"
    "algorithm.augmentation.alpha=${ALPHA}"
)
# The two-pass placements run a second Eagle2 forward whose activations are
# retained for the value head's backward, on top of a config that already had
# to drop micro_batch_size 32 -> 16 to fit the VLM lm_head logits under
# allocator fragmentation. Halve it again for those arms. global_batch_size is
# unchanged, so the optimisation is identical -- gradient accumulation absorbs
# it, and every arm still optimises the same objective.
if [[ "${PLACEMENT}" == "critic_only" || "${PLACEMENT}" == "actor_only" ]]; then
    OVERRIDES+=("actor.micro_batch_size=8")
fi

echo "arm: ${TAG}  (seed ${SEED}, ${SEGMENTS} segments)"
echo "overrides: ${OVERRIDES[*]}"

DEP=""
for ((i = 1; i <= SEGMENTS; i++)); do
    JOB_ID=$(sbatch --parsable ${DEP:+--dependency=afterany:${DEP}} \
        --export=ALL,RUN_TAG="${TAG}" \
        --job-name="gr00t-spatial-${TAG}-4gpu" \
        vla_augm/scripts/cluster_b/run_gr00t_augm_4gpu.sh "${SEED}" \
        "${OVERRIDES[@]}")
    echo "  segment ${i}: job ${JOB_ID}${DEP:+ (after ${DEP})}"
    DEP="${JOB_ID}"
done
