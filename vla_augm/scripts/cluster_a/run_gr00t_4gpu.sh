#!/bin/bash
# One arm of the GR00T N1.5 augmentation study, on Cluster A (4x H100, one node).
#
# Do not submit this directly -- use vla_augm/scripts/cluster_a/run_gr00t_augm_arm.sh,
# which names an arm by what it varies instead of by a hand-assembled list of
# Hydra overrides that is easy to get subtly wrong.
#
# Port of vla_augm/scripts/cluster_b/run_gr00t_augm_4gpu.sh. The Cluster B-specific
# parts (Lmod stack, squashfs LIBERO-Plus assets, expandable_segments:False for
# that node's older kernel) are replaced by the Cluster A equivalents already
# validated in run_pi05_4gpu.sh; the self-resume logic is kept verbatim.
#
# SELF-RESUMING. Every arm has a *stable* log and checkpoint directory keyed on
# its RUN_TAG, not on a timestamp, so resubmitting after a timeout picks up
# from the newest committed checkpoint and reattaches to the same wandb run
# (rlinf/utils/metric_logger.py recovers wandb_run_id.txt from the ckpt dir).
#
#SBATCH --job-name=gr00t-spatial-augm-4gpu
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=4-00:00:00
#SBATCH --output=/projects/PROJECT_ID/users/%u/rl_inf/console_logs/%x_%j.log

set -e

# Code on $HOME (scratch is purged by access time).
# Everything a run produces -- logs, wandb, videos, console output and
# checkpoints -- goes to project space. Scratch is purged by access time
# (it ate the repo and .venv-pi once) and $HOME is near its inode quota.
# %u is Slurm's username expansion, valid in #SBATCH --output.
REPO_PATH="${REPO_PATH:-${HOME}/rl_inf}"
VENV="${VENV:-/scratch/USER/rl_inf/.venv-gr00t}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/PROJECT_ID/users/${USER}/rl_inf}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/gr00t_augm}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/ckpts/gr00t_augm}"
VLA_AUGM_DIR="${REPO_PATH}/vla_augm"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
CONFIG_NAME="${CONFIG_NAME:-libero_spatial_ppo_gr00t_4gpu_augm}"
SEED="${1:-52}"
EXTRA_OVERRIDES=("${@:2}")

RUN_TAG="${RUN_TAG:?RUN_TAG must be set; submit via run_gr00t_augm_arm.sh}"

# Mesa/EGL instead of apt mesa libs (no sudo on Cluster A). ImageMagick is
# needed by python-wand, imported when the liberoplus env package loads.
module load 2025
module load Mesa/25.1.3-GCCcore-14.2.0
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"

# CRITICAL: slim the environment. RLinf serializes the FULL driver env into each
# Ray worker's runtime-env, passed to execve as ONE argument. Linux caps a
# single arg at 128 KiB (MAX_ARG_STRLEN); the ImageMagick module's build-time
# vars push it over, and every env/rollout worker then dies at exec (E2BIG)
# with no logs.
unset CPATH LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    ACLOCAL_PATH _LMFILES_ LOADEDMODULES MANPATH XDG_DATA_DIRS
for _v in $(env | grep -oE '^__LMOD_REF_COUNT[A-Za-z_]*'); do unset "$_v"; done
echo "Driver env size: $(env | wc -c) bytes (must stay well under 131072)"

source "${VENV}/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/USER/.cache/huggingface}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EMBODIED_PATH
export REPO_PATH

EXP_NAME="${CONFIG_NAME}_lr2e-5_${RUN_TAG}_seed${SEED}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}-seed${SEED}"
CKPT_DIR="${CKPT_ROOT}/${RUN_TAG}-seed${SEED}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Resume from the newest COMPLETE global_step_N this arm has written. The
# embodied runner has no resume_dir: auto, so resolve it here. `.metadata` is
# DCP's completeness marker -- the coordinator writes it only after every rank
# committed its shard, so a job killed mid-save is skipped rather than taking
# out the whole remaining chain. sort -n so global_step_250 beats _45.
CKPTS_DIR="${CKPT_DIR}/${EXP_NAME}/checkpoints"
RESUME_OVERRIDES=()
RESUME_STEP=""
if [[ -d "${CKPTS_DIR}" ]]; then
    while read -r step; do
        [[ -n "${step}" ]] || continue
        dcp="${CKPTS_DIR}/global_step_${step}/actor/dcp_checkpoint"
        if [[ -f "${dcp}/.metadata" ]] && compgen -G "${dcp}/*.distcp" >/dev/null; then
            RESUME_STEP="${step}"
            break
        fi
        echo "Skipping incomplete checkpoint global_step_${step} (no committed DCP)."
    done < <(find "${CKPTS_DIR}" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
        | sed 's/global_step_//' | sort -rn)
fi
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_OVERRIDES+=("runner.resume_dir=${CKPTS_DIR}/global_step_${RESUME_STEP}")
    echo "Resuming ${RUN_TAG} from global_step_${RESUME_STEP} (same wandb run)."
else
    echo "No usable checkpoint under ${CKPTS_DIR}; starting ${RUN_TAG} from the SFT model."
fi

CONSOLE_LOG="${PROJECT_ROOT}/console_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Arm: ${RUN_TAG}"
echo "Log dir: ${LOG_DIR}"
echo "Ckpt dir: ${CKPT_DIR}"
echo "Using Python at $(which python)"
echo "Seed: ${SEED}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
nvidia-smi -L

# Opt-in peak-GPU-memory probe (see run_pi05_4gpu.sh): the LoRA arms are
# memory-bound and nothing in RLinf logs memory.
if [[ -n "${GPU_MEM_POLL:-}" ]]; then
    (
        peak=0
        while true; do
            cur="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                   | sort -rn | head -1 || true)"
            if [[ -n "${cur}" ]] && (( cur > peak )); then
                peak="${cur}"
                echo "[gpu-mem] new peak ${peak} MiB ($(date +%H:%M:%S))"
            fi
            sleep "${GPU_MEM_POLL}"
        done
    ) &
    trap 'kill %% 2>/dev/null || true' EXIT
fi

python "${SRC_FILE}" \
    --config-path "${VLA_AUGM_DIR}/config/libero" \
    --config-name "${CONFIG_NAME}" \
    seed="${SEED}" \
    runner.logger.log_path="${LOG_DIR}" \
    runner.checkpoint_path="${CKPT_DIR}" \
    runner.logger.experiment_name="${EXP_NAME}" \
    +runner.logger.console_log_file="${CONSOLE_LOG}" \
    "${RESUME_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
