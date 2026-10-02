#!/bin/bash
# Launch one GR00T N1.5 augmentation arm on Cluster A (4x H100).
#
#   bash vla_augm/scripts/cluster_a/run_gr00t_augm_arm.sh <placement> <type> [alpha] [seed] [extra...]
#
#   placement  none | uniform | critic_only | actor_only
#   type       random_overlay | random_shift | random_overlay_shift
#   alpha      overlay strength; ignored by random_shift. Default 0.5.
#   seed       default 52
#
# Env knobs: LORA=1 [LORA_RANK=32] [MICRO_BS=n] [TAG_EXTRA=label] [TIME=d-hh:mm:ss]
#
# Mirrors vla_augm/scripts/cluster_a/run_augm_arm.sh (pi0.5) so the two model families
# are launched the same way. Slurm reserves the whole walltime against the
# budget when the job starts, so keep TIME realistic.
TIME="${TIME:-4-00:00:00}"

set -euo pipefail

REPO_PATH="${REPO_PATH:-${HOME}/rl_inf}"
cd "${REPO_PATH}"

PLACEMENT="${1:?placement: none | uniform | critic_only | actor_only}"
AUG_TYPE="${2:?type: random_overlay | random_shift | random_overlay_shift}"
ALPHA="${3:-0.5}"
SEED="${4:-52}"
EXTRA=("${@:5}")

case "${PLACEMENT}" in
    none|uniform|critic_only|actor_only) ;;
    *) echo "bad placement: ${PLACEMENT}" >&2; exit 1 ;;
esac
case "${AUG_TYPE}" in
    random_overlay|random_shift|random_overlay_shift) ;;
    *) echo "bad type: ${AUG_TYPE}" >&2; exit 1 ;;
esac

case "${AUG_TYPE}" in
    random_overlay)       TYPE_TAG="ovl" ;;
    random_shift)         TYPE_TAG="shift" ;;
    random_overlay_shift) TYPE_TAG="ovlshift" ;;
esac
TAG="augm_${PLACEMENT}_${TYPE_TAG}"
[[ "${AUG_TYPE}" != "random_shift" ]] && TAG="${TAG}_alpha${ALPHA/./}"
[[ "${PLACEMENT}" == "none" ]] && TAG="augm_none"

# LORA=1 adds rank-r adapters to the Eagle VLM only (37.7M at r=32: 12 LLM
# layers survive select_layer 12, plus 27 vision layers). The action head stays
# fully trainable -- see the GR00T branch in rlinf/models/__init__.py, which
# exists because get_peft_model on the whole model would freeze the DiT.
LORA="${LORA:-0}"
LORA_RANK="${LORA_RANK:-32}"
[[ "${LORA}" == "1" ]] && TAG="${TAG}_lora${LORA_RANK}"
[[ -n "${TAG_EXTRA:-}" ]] && TAG="${TAG}_${TAG_EXTRA}"

OVERRIDES=(
    "algorithm.augmentation.placement=${PLACEMENT}"
    "algorithm.augmentation.type=${AUG_TYPE}"
    "algorithm.augmentation.alpha=${ALPHA}"
)
if [[ "${LORA}" == "1" ]]; then
    OVERRIDES+=(
        "actor.model.is_lora=True"
        "actor.model.lora_rank=${LORA_RANK}"
    )
fi

# micro_batch_size is a MEMORY knob only: global_batch_size is identical across
# arms, so grad accumulation absorbs any change and the optimisation is
# unchanged. Config ships 16; the two-pass placements run a second Eagle2
# forward retained for the value head's backward, so they halve it. LoRA
# additionally retains the whole Eagle activation graph, so it halves again.
if [[ "${LORA}" == "1" ]]; then
    case "${PLACEMENT}" in
        critic_only|actor_only) MICRO_BS="${MICRO_BS:-4}" ;;
        *)                      MICRO_BS="${MICRO_BS:-8}" ;;
    esac
else
    case "${PLACEMENT}" in
        critic_only|actor_only) MICRO_BS="${MICRO_BS:-8}" ;;
        *)                      MICRO_BS="${MICRO_BS:-}" ;;
    esac
fi
[[ -n "${MICRO_BS}" ]] && OVERRIDES+=("actor.micro_batch_size=${MICRO_BS}")

echo "arm: ${TAG}  (seed ${SEED})"
echo "overrides: ${OVERRIDES[*]} ${EXTRA[*]}"

sbatch \
    --export=ALL,RUN_TAG="${TAG}" \
    --job-name="gr00t-spatial-${TAG}-4gpu" \
    --time="${TIME}" \
    vla_augm/scripts/cluster_a/run_gr00t_4gpu.sh "${SEED}" \
    "${OVERRIDES[@]}" "${EXTRA[@]}"
