#!/bin/bash
# Run with: sbatch vla_augm/scripts/cluster_a/run_pi0_1gpu.sh

#SBATCH --job-name=pi0-libero-spatial
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -e

REPO_PATH="/scratch/USER/rl_inf"
VLA_AUGM_DIR="${REPO_PATH}/vla_augm"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
CONFIG_NAME="libero_spatial_ppo_pi0_1gpu"

# Mesa/EGL instead of apt mesa libs (no sudo on Cluster A)
module load 2025
module load Mesa/25.1.3-GCCcore-14.2.0

source "${REPO_PATH}/.venv-pi/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

# Compile inductor kernels in-process; the default per-process compile pool
# can stall GPFS with parallel venv imports (see run_pi0_2gpu.sh).
export TORCHINDUCTOR_COMPILE_THREADS=1

# One OMP thread per process: otherwise every Ray worker gets an
# nproc-sized OpenMP spin-wait pool that saturates the CPU allocation
# and starves the MuJoCo sim subprocesses (7x slower env stepping).
export OMP_NUM_THREADS=1

export EMBODIED_PATH
export REPO_PATH

STAMP="$(date +'%Y%m%d-%H:%M:%S')"
RUN_NAME="${STAMP}-${CONFIG_NAME}"
LOG_DIR="${REPO_PATH}/logs/${RUN_NAME}"
# Heavy checkpoints go to personal scratch (8 TiB quota); logs, videos, and
# wandb stay in the project space. scratch-shared is auto-purged, so copy any
# checkpoint you want to keep long-term back into the project space.
# Checkpoints go to PROJECT space, not scratch: scratch is purged after ~2
# weeks idle, and the pinned every-100-epoch checkpoints have to outlive a
# run for the offline evals. A checkpoint is only ~10 inodes, so this is
# safe against the tight PROJECT_ID inode quota (~342k free group-wide);
# it is the 572k-inode venv that must stay on scratch.
CKPT_DIR="/projects/PROJECT_ID/users/${USER}/rl_inf/ckpts/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Reconstruct the SLURM console log path (%x_%j.log) to record in wandb config.
CONSOLE_LOG="vla_augm/console_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Log dir: ${LOG_DIR}"
echo "Console log: ${CONSOLE_LOG}"
echo "Using Python at $(which python)"
echo "ROBOT_PLATFORM=${ROBOT_PLATFORM}"

python "${SRC_FILE}" \
    --config-path "${VLA_AUGM_DIR}/config/libero" \
    --config-name "${CONFIG_NAME}" \
    runner.logger.log_path="${LOG_DIR}" \
    runner.checkpoint_path="${CKPT_DIR}" \
    +runner.logger.console_log_file="${CONSOLE_LOG}"
