#!/bin/bash
# Run pi0.5 LIBERO-Spatial PPO on Cluster A (4x H100, one node).
#
# Run with: sbatch vla_augm/scripts/cluster_a/run_pi05_4gpu.sh [seed] [extra hydra overrides...]
#   [seed] defaults to 42 (the config's value). Env workers use seed..seed+3.
#   Any args after the seed are forwarded verbatim as Hydra overrides, e.g.
#     sbatch --time=01:00:00 vla_augm/scripts/cluster_a/run_pi05_4gpu.sh \
#         42 runner.max_epochs=2 runner.save_interval=1

#SBATCH --job-name=pi05-libero-spatial-4gpu-lr5e-6
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=5-00:00:00
#SBATCH --output=/projects/PROJECT_ID/users/%u/rl_inf/console_logs/%x_%j.log

set -e

# Code lives on $HOME: scratch is purged by access time and ate the repo (and
# .venv-pi) once already.
# Everything a run produces -- logs, wandb, videos, console output and
# checkpoints -- goes to project space. Scratch is purged by access time
# (it ate the repo and .venv-pi once) and $HOME is near its inode quota.
# %u is Slurm's username expansion, valid in #SBATCH --output.
REPO_PATH="${REPO_PATH:-${HOME}/rl_inf}"
VENV="${VENV:-${HOME}/rlinf_venvs/.venv-pi}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/PROJECT_ID/users/${USER}/rl_inf}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
VLA_AUGM_DIR="${REPO_PATH}/vla_augm"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
CONFIG_NAME="libero_spatial_ppo_pi05_4gpu"
SEED="${1:-42}"
EXTRA_OVERRIDES=("${@:2}")
# Optional label for an experiment arm, e.g.
#   sbatch --export=ALL,RUN_TAG=augm_unif --job-name=... run_pi05_4gpu.sh 42 ...
# It lands in the wandb run name, the log dir, and the checkpoint dir, so arms
# stay distinguishable everywhere.
RUN_TAG="${RUN_TAG:-}"
TAG_SUFFIX=""
[[ -n "${RUN_TAG}" ]] && TAG_SUFFIX="-${RUN_TAG}"

# Mesa/EGL instead of apt mesa libs (no sudo on Cluster A).
# ImageMagick is needed by python-wand, imported by the liberoplus env
# package that the OOD eval worker group loads.
module load 2025
module load Mesa/25.1.3-GCCcore-14.2.0
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"

# CRITICAL: slim the environment. RLinf serializes the FULL driver env into each
# Ray worker's runtime-env, passed to execve as ONE argument. Linux caps a single
# arg at 128 KiB (MAX_ARG_STRLEN); the ImageMagick module's build-time vars push
# the env to ~147 KB, so every env/rollout worker dies instantly at exec (E2BIG)
# with no logs ("crashed during start"). Runtime only needs LD_LIBRARY_PATH and
# MAGICK_HOME; drop compile-time and Lmod bookkeeping vars.
unset CPATH LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    ACLOCAL_PATH _LMFILES_ LOADEDMODULES MANPATH XDG_DATA_DIRS
for _v in $(env | grep -oE '^__LMOD_REF_COUNT[A-Za-z_]*'); do unset "$_v"; done
echo "Driver env size: $(env | wc -c) bytes (must stay well under 131072)"

source "${VENV}/bin/activate"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
# Training and in-distribution eval run standard LIBERO. The OOD eval group
# overrides LIBERO_TYPE=plus for its own workers (env.eval_ood.env_vars).
export LIBERO_TYPE=standard
# The OOD eval group needs LIBERO-Plus's task_classification.json. Its
# auto-discovery falls back to REPO_PATH/venv-pi, which no longer exists
# now that the venv lives under $HOME/rlinf_venvs, so name it explicitly.
export LIBERO_PLUS_TASK_CLASSIFICATION="${LIBERO_PLUS_TASK_CLASSIFICATION:-${REPO_PATH}/vla_augm/utils/task_classification.json}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

# Compile inductor kernels in-process. The default spawns a compile pool per
# torch process, whose parallel imports from the GPFS-hosted venv can stall
# the node's I/O and freeze the whole job.
export TORCHINDUCTOR_COMPILE_THREADS=1

# One OMP thread per process: otherwise every Ray worker gets an
# nproc-sized OpenMP spin-wait pool that saturates the CPU allocation
# and starves the MuJoCo sim subprocesses (7x slower env stepping).
export OMP_NUM_THREADS=1

# The OOD eval env group adds MuJoCo/EGL contexts on every GPU for the whole
# run, so the actor has less headroom; expandable segments avoid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EMBODIED_PATH
export REPO_PATH

# STABLE_DIRS=1 keys the log and checkpoint directories on RUN_TAG instead of a
# timestamp, which is what makes a run resumable: resubmitting the same script
# finds the newest committed checkpoint and continues from it, and MetricLogger
# recovers wandb_run_id.txt from the (unchanged) checkpoint directory so the
# curve stays one continuous wandb run across however many SLURM segments it
# took. Needed because gpu_h100 has been handing out whole-node slots only
# after many hours, so a single 5-day reservation may never be scheduled --
# a chain of shorter segments backfills far more easily. Default (unset) keeps
# the old timestamped, non-resuming behaviour for one-off runs.
STAMP="$(date +'%Y%m%d-%H:%M:%S')"
if [[ "${STABLE_DIRS:-0}" == "1" ]]; then
    : "${RUN_TAG:?STABLE_DIRS=1 needs RUN_TAG -- the directories are keyed on it}"
    RUN_NAME="${CONFIG_NAME}${TAG_SUFFIX}-seed${SEED}"
else
    RUN_NAME="${STAMP}-${CONFIG_NAME}${TAG_SUFFIX}-seed${SEED}"
fi
# Mirror the tag into the wandb experiment name (config default has no tag).
EXP_NAME="${CONFIG_NAME}_lr5e-6_${RUN_TAG}_seed${SEED}"
NAME_OVERRIDES=()
if [[ -n "${RUN_TAG}" ]]; then
    NAME_OVERRIDES+=("runner.logger.experiment_name=${EXP_NAME}")
fi
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
# Heavy checkpoints go to personal scratch (8 TiB quota); logs, videos, and
# wandb stay in the project space. scratch-shared is auto-purged, so copy any
# checkpoint you want to keep long-term back into the project space.
# Checkpoints go to PROJECT space, not scratch: scratch is purged after ~2
# weeks idle, and the pinned every-100-epoch checkpoints have to outlive a
# run for the offline evals. A checkpoint is only ~10 inodes, so this is
# safe against the tight PROJECT_ID inode quota (~342k free group-wide);
# it is the 572k-inode venv that must stay on scratch.
CKPT_DIR="${PROJECT_ROOT}/ckpts/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Resume from the newest COMPLETE global_step_N this arm has written. The
# embodied runner has no resume_dir: auto, so resolve it here. `.metadata` is
# DCP's completeness marker -- the coordinator writes it only after every rank
# committed its shard, so a segment killed mid-save is skipped rather than
# taking out the whole remaining chain. sort -n so global_step_250 beats _45.
RESUME_OVERRIDES=()
if [[ "${STABLE_DIRS:-0}" == "1" ]]; then
    CKPTS_DIR="${CKPT_DIR}/${EXP_NAME}/checkpoints"
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
fi

# Reconstruct the SLURM console log path (%x_%j.log) to record in wandb config.
CONSOLE_LOG="${PROJECT_ROOT}/console_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Log dir: ${LOG_DIR}"
echo "Console log: ${CONSOLE_LOG}"
echo "Using Python at $(which python)"
echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "Seed: ${SEED}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
nvidia-smi -L

# Opt-in hang diagnosis: STACK_DUMP_AFTER=<seconds> dumps a py-spy stack for
# every python process once, then exits. Unset for real arms, so they pay
# nothing. Used to find where EnvOODGroup blocks (job 25335248 sat 80 min in
# env_ood_group.init_worker().wait() with no output at all).
if [[ -n "${STACK_DUMP_AFTER:-}" ]]; then
    (
        sleep "${STACK_DUMP_AFTER}"
        echo "===== py-spy stack dump at +${STACK_DUMP_AFTER}s ====="
        # Ray actor processes rename themselves "ray::ActorGroup..." -- those
        # are the ones that matter. A plain `pgrep -f python | head` returns
        # Ray infra and env subprocesses instead and never reaches them.
        # Dump every ray:: process, no head/tail. Ray renames its workers via
        # setproctitle, so their cmdline is "ray::IDLE" / "ray::ClassName" and
        # NOT default_worker (that filter matched only the raylet, job
        # 25346890); and slicing the list kept hitting the idle pool instead of
        # the actors (jobs 25339175, 25346306).
        for pid in $(pgrep -u "${USER}" -f "ray::"); do
            echo "----- pid ${pid}: $(tr '\0' ' ' < /proc/${pid}/cmdline 2>/dev/null | cut -c1-120)"
            timeout 25 py-spy dump --pid "${pid}" --nonblocking 2>&1 | head -40
        done
        echo "===== end stack dump ====="
    ) &
fi

# Opt-in peak-GPU-memory probe: GPU_MEM_POLL=<seconds> reports every new high
# water mark across the four GPUs. Nothing in RLinf logs memory, and the LoRA
# arms are memory-bound (the whole PaliGemma activation graph is retained for
# backward and openpi cannot gradient-checkpoint the prefix pass), so the smoke
# test needs this to pick actor.micro_batch_size for the real arms.
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
    +runner.logger.console_log_file="${CONSOLE_LOG}" \
    "${NAME_OVERRIDES[@]}" \
    "${RESUME_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
