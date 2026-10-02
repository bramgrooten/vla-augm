#!/bin/bash
# Launch one Exp 0(b) augmentation arm on Cluster A (4x H100).
#
# Wraps run_pi05_4gpu.sh so an arm is named by what it varies, rather than by a
# hand-assembled list of hydra overrides that is easy to get subtly wrong.
#
# Usage:
#   bash vla_augm/scripts/cluster_a/run_augm_arm.sh <placement> <type> [alpha] [seed] [extra...]
#
#   placement  none | uniform | critic_only | actor_only
#   type       random_overlay | random_shift | random_overlay_shift
#   alpha      overlay strength; ignored by random_shift. Default 0.5.
#   seed       default 42
#
# Examples:
#   bash vla_augm/scripts/cluster_a/run_augm_arm.sh critic_only random_shift
#   bash vla_augm/scripts/cluster_a/run_augm_arm.sh uniform random_overlay 0.25
#   bash vla_augm/scripts/cluster_a/run_augm_arm.sh critic_only random_overlay_shift 0.5 43
#
# SLURM reserves the whole walltime against the budget when the job starts, so
# the 5-day default in run_pi05_4gpu.sh reserves 92,160 SBU (768/h on 4 GPUs).
# With four such jobs already running that check fails and root scancels the
# job seconds after it starts, with no console log. Set TIME to something the
# remaining budget covers:
#   TIME=4-00:00:00 bash vla_augm/scripts/cluster_a/run_augm_arm.sh uniform random_overlay
# A 300-epoch arm needs ~67 h, so 4 days leaves real headroom without asking
# for the 5-day maximum, which is harder to schedule on a busy queue.
# Reserving more does not cost more -- Slurm bills elapsed time; the
# reservation only gates whether the submission is accepted.
# 78 h: a 300-epoch arm needs ~70 h at pipeline_stage_num 1, so a plain 3 days
# leaves only 2 h of margin. Checkpoints every 25 mean a timeout costs the last
# <=25 epochs rather than the run, but 8 h of slack is free -- Slurm bills
# elapsed time and the reservation only gates admission.
TIME="${TIME:-3-06:00:00}"

set -euo pipefail

REPO_PATH="${REPO_PATH:-${HOME}/rl_inf}"
cd "${REPO_PATH}"

PLACEMENT="${1:?placement: none | uniform | critic_only | actor_only}"
AUG_TYPE="${2:?type: random_overlay | random_shift | random_overlay_shift}"
ALPHA="${3:-0.5}"
SEED="${4:-42}"
EXTRA=("${@:5}")

case "${PLACEMENT}" in
    none|uniform|critic_only|actor_only) ;;
    *) echo "bad placement: ${PLACEMENT}" >&2; exit 1 ;;
esac
case "${AUG_TYPE}" in
    random_overlay|random_shift|random_overlay_shift) ;;
    *) echo "bad type: ${AUG_TYPE}" >&2; exit 1 ;;
esac

# Tag: placement, then a short type code, then alpha when it applies. Lands in
# the wandb run name, the log dir and the checkpoint dir.
case "${AUG_TYPE}" in
    random_overlay)       TYPE_TAG="ovl" ;;
    random_shift)         TYPE_TAG="shift" ;;
    random_overlay_shift) TYPE_TAG="ovlshift" ;;
esac
TAG="augm_${PLACEMENT}_${TYPE_TAG}"
[[ "${AUG_TYPE}" != "random_shift" ]] && TAG="${TAG}_alpha${ALPHA/./}"
[[ "${PLACEMENT}" == "none" ]] && TAG="augm_none"

# LORA=1 adds rank-r adapters to PaliGemma (SigLIP + Gemma-2B), so the value
# and policy losses reach the visual encoder instead of dying at a frozen
# prefix. 64.9M adapter params at r=32. The arm keeps its own tag so its logs,
# wandb run and checkpoints never collide with the frozen twin.
LORA="${LORA:-0}"
LORA_RANK="${LORA_RANK:-32}"
# LORA_LAYERS=top6 confines the adapters to Gemma blocks 12-17 (the last third
# of 18) and excludes the vision tower entirely. That is what makes the arm
# affordable: autograd keeps activations only between the loss and the deepest
# trainable parameter, so blocks 0-11 and all of SigLIP run without a graph.
# 13.07M adapter params instead of 64.91M. Empty = adapt the whole VLM.
LORA_LAYERS="${LORA_LAYERS:-}"
[[ "${LORA}" == "1" ]] && TAG="${TAG}_lora${LORA_RANK}"
[[ -n "${LORA_LAYERS}" ]] && TAG="${TAG}_${LORA_LAYERS}"
# Free-form tag suffix, e.g. TAG_EXTRA=smoke, so a throwaway run never
# shares a wandb name or a checkpoint directory with a real arm.
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
    if [[ "${LORA_LAYERS}" == "top6" ]]; then
        # Gemma blocks 12-17 of 18; the lookahead keeps the vision tower out.
        # The value is wrapped in single quotes because Hydra's override
        # grammar cannot lex a bare regex (LexerNoViableAltException).
        LORA_RX='^(?!.*vision_tower).*layers\.(?:1[2-7])\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
        OVERRIDES+=("actor.model.lora_target_regex='${LORA_RX}'")
    elif [[ -n "${LORA_LAYERS}" ]]; then
        echo "unknown LORA_LAYERS: ${LORA_LAYERS}" >&2; exit 1
    fi
fi

# micro_batch_size is a MEMORY knob, never a hyperparameter: global_batch_size
# is identical across arms, so grad accumulation absorbs any change and the
# optimisation is unchanged. Two regimes:
#   frozen VLM -- critic_only/actor_only keep a second prefix graph and OOM at
#     the baseline 128 (job 25023696), so 64.
#   LoRA -- the whole PaliGemma activation graph is now retained for backward
#     and openpi cannot gradient-checkpoint the prefix pass (its patched Gemma
#     decoder loop calls the layers directly, and setting the HF flag would
#     silently disable use_cache and break the expert's prefix KV). So the
#     batch has to come down instead. Override with MICRO_BS= once the smoke
#     test reports real peak memory.
if [[ "${LORA}" == "1" ]]; then
    case "${PLACEMENT}" in
        critic_only|actor_only) MICRO_BS="${MICRO_BS:-8}" ;;
        *)                      MICRO_BS="${MICRO_BS:-16}" ;;
    esac
else
    case "${PLACEMENT}" in
        critic_only|actor_only) MICRO_BS="${MICRO_BS:-64}" ;;
        *)                      MICRO_BS="${MICRO_BS:-}" ;;
    esac
fi
[[ -n "${MICRO_BS}" ]] && OVERRIDES+=("actor.micro_batch_size=${MICRO_BS}")

echo "arm: ${TAG}"
echo "overrides: ${OVERRIDES[*]} ${EXTRA[*]}"

# SEGMENTS>1 submits a chain of afterany-dependent jobs sharing one stable
# log/checkpoint directory, so each picks up where the last stopped. gpu_h100
# has been giving whole-node slots only after many hours, and a 5-day
# reservation is far harder to schedule than several shorter ones; a chain also
# means a timeout costs at most the epochs since the last save rather than the
# run. STABLE_DIRS=1 is what makes the resume work (see run_pi05_4gpu.sh).
SEGMENTS="${SEGMENTS:-1}"
DEP=""
for ((i = 1; i <= SEGMENTS; i++)); do
    JOB_ID=$(sbatch --parsable ${DEP:+--dependency=afterany:${DEP}} \
        --export=ALL,RUN_TAG="${TAG}",STABLE_DIRS="${STABLE_DIRS:-0}" \
        --job-name="pi05-spatial-${TAG}-4gpu" \
        --time="${TIME}" \
        vla_augm/scripts/cluster_a/run_pi05_4gpu.sh "${SEED}" \
        "${OVERRIDES[@]}" "${EXTRA[@]}")
    echo "  segment ${i}: job ${JOB_ID}${DEP:+ (after ${DEP})}"
    DEP="${JOB_ID}"
done
