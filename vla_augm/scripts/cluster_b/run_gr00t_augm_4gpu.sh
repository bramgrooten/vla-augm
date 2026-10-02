#!/bin/bash -l
# One arm of the GR00T N1.5 augmentation placement study, on Cluster B
# (4x A100-40GB, one node). Login shell (-l) so Lmod's `module` is available.
#
# Do not submit this directly -- use vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh,
# which names an arm by what it varies instead of by a hand-assembled list of
# Hydra overrides that is easy to get subtly wrong.
#
# SELF-RESUMING. Every arm has a *stable* log and checkpoint directory keyed on
# its RUN_TAG, not on a timestamp. On startup this script looks for the highest
# global_step_N already in that arm's checkpoint directory and, if it finds
# one, resumes from it. So the same job script both starts an arm and continues
# it: submit it again after a timeout and it picks up where it left off.
#
# Because experiment_name and checkpoint_path are unchanged across a resume,
# MetricLogger recovers wandb_run_id.txt from the checkpoint directory and
# reattaches to the *same* wandb run -- one continuous curve per arm however
# many SLURM segments it took. (rlinf/utils/metric_logger.py:210.)
#
#SBATCH --job-name=gr00t-spatial-augm-4gpu
#SBATCH --account=PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -e

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
VLA_AUGM_DIR="${REPO_PATH}/vla_augm"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
CONFIG_NAME="libero_spatial_ppo_gr00t_cluster_b_4gpu_augm"
SEED="${1:-52}"
EXTRA_OVERRIDES=("${@:2}")

# Arm label, set by run_gr00t_augm_arm.sh via --export. It keys the wandb run
# name, the log dir and the checkpoint dir, so arms stay distinguishable and,
# more importantly, a resume lands in the same place as the run it continues.
RUN_TAG="${RUN_TAG:?RUN_TAG must be set; submit via run_gr00t_augm_arm.sh}"

# Cluster B software stack: Mesa provides the EGL/GL loader libs for MuJoCo
# offscreen rendering (MUJOCO_GL=egl). The NVIDIA driver supplies the actual
# GPU EGL backend (NVIDIA_DRIVER_CAPABILITIES=all is exported by the venv).
module load env/release/2024.1
module load Mesa
# ImageMagick: python-wand needs it to import the liberoplus env package.
# env.eval_ood.enable is False for every arm here, but the package is still
# imported when the config tree is built, so this stays loaded.
module load ImageMagick/7.1.1-38-GCCcore-13.3.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"

# CRITICAL: slim the environment. RLinf serializes the FULL driver env into each
# Ray worker's runtime-env, passed to execve as ONE argument. Linux caps a single
# arg at 128 KiB (MAX_ARG_STRLEN); the ImageMagick module's build-time vars push
# the env over that, so every env/rollout worker dies instantly at exec (E2BIG,
# "crashed during start"). Runtime only needs LD_LIBRARY_PATH and MAGICK_HOME;
# drop compile-time and Lmod bookkeeping vars.
unset CPATH LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    ACLOCAL_PATH _LMFILES_ LOADEDMODULES MANPATH XDG_DATA_DIRS
for _v in $(env | grep -oE '^__LMOD_REF_COUNT[A-Za-z_]*'); do unset "$_v"; done
echo "Driver env size: $(env | wc -c) bytes (must stay well under 131072)"

source "${REPO_PATH}/.venv-gr00t/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

# LIBERO-Plus assets: extract the squashfs image to node-local tmpfs (inode
# quota) and point liberoplus at it for this job. Not needed while
# env.eval_ood.enable is False, but it costs ~1 min and keeps the environment
# identical to the runs that produced the timings we budgeted from.
source "${VLA_AUGM_DIR}/scripts/cluster_b/libero_plus_local.sh"

# Keep any HF lookups off the near-full $HOME.
export HF_HOME="/project/scratch/PROJECT_ID/vla_augm/.cache/huggingface"

# Compile inductor kernels in-process. The default spawns a compile pool per
# torch process, whose parallel imports from the shared-FS venv can stall the
# node's I/O and freeze the whole job.
export TORCHINDUCTOR_COMPILE_THREADS=1

# One OMP thread per process: otherwise every Ray worker gets an nproc-sized
# OpenMP spin-wait pool that saturates the CPU allocation and starves the
# MuJoCo sim subprocesses (much slower env stepping).
export OMP_NUM_THREADS=1

# expandable_segments:True would cut fragmentation but breaks CUDA-IPC on
# Cluster B's older kernel (no pidfd_open syscall), so keep it False.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

export EMBODIED_PATH
export REPO_PATH

EXP_NAME="${CONFIG_NAME}_lr2e-5_${RUN_TAG}_seed${SEED}"

# Stable per-arm directories -- the whole resume mechanism rests on these not
# moving between segments.
#
# Checkpoints go to persistent project storage, NOT to a purged scratch: the
# snapshots are the deliverable and must survive weeks for the offline OOD
# evals. /project/scratch/PROJECT_ID is project allocation on Cluster B and is not
# pruned here (unlike Cluster A's /scratch-shared); it has the most headroom of
# the three quotas, which matters at ~14.2 GiB x 10 snapshots x 8 arms.
LOG_DIR="${REPO_PATH}/logs/gr00t_augm/${RUN_TAG}-seed${SEED}"
CKPT_DIR="/project/scratch/PROJECT_ID/vla_augm/rl_inf/ckpts/gr00t_augm/${RUN_TAG}-seed${SEED}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Resume from the newest COMPLETE global_step_N this arm has written, if any.
# The embodied runner has no resume_dir: auto (only the reasoning runner does,
# rlinf/runners/embodied_runner.py:213), so resolve it here.
#
# Walk candidates newest-first and skip torn writes. A job killed mid-save (a
# timeout or a watchdog scancel landing during _save_checkpoint) leaves a
# global_step_N with an actor/ directory but an incomplete dcp_checkpoint/.
# Resuming from that raises inside dcp.load, and because every later segment
# would pick the same broken newest checkpoint, one torn write would otherwise
# take out the whole remaining chain for that arm.
#
# `.metadata` is the completeness marker: DCP's coordinator writes it after all
# ranks have finished their shards, so its presence means the save committed.
# `sort -n` (not lexicographic) so global_step_250 beats global_step_45.
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

# Reconstruct the SLURM console log path (%x_%j.log) to record in wandb config.
CONSOLE_LOG="vla_augm/console_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Arm: ${RUN_TAG}"
echo "Log dir: ${LOG_DIR}"
echo "Ckpt dir: ${CKPT_DIR}"
echo "Console log: ${CONSOLE_LOG}"
echo "Using Python at $(which python)"
echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "Seed: ${SEED}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
nvidia-smi -L

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
