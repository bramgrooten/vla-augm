#!/bin/bash -l
# Install the .venv-gr00t environment (GR00T N1.5 model + LIBERO env) on Cluster B.
# Runs on a GPU node: it has internet, gcc, and a visible A100 so uv's CUDA
# wheels resolve correctly.
#
# TOOLCHAIN NOTE: same as install_venv_pi.sh -- we deliberately do NOT load the
# GCCcore/CMake/CUDA modules. The system gcc (8.5) + system as (2.30) are a
# self-consistent pair that builds every source dep in this env (evdev, decord,
# torch-memory-saver, ...). uv installs any cmake/ninja a package needs into
# its isolated build env. flash-attn uses a prebuilt wheel, so no nvcc needed.
# Unlike openpi, the gr00t dep set pulls no rerun-sdk (no lerobot), so the
# rerun_sdk_stub override is not needed here.
#
# Submit with: sbatch vla_augm/scripts/cluster_b/install_venv_gr00t.sh
# Tip: the cluster is often full; add `--qos=dev --time=00:20:00` (and, if it
# still pends, `--reservation=gpudev`) so this short job backfills quickly.
#
#SBATCH --job-name=install-venv-gr00t
#SBATCH --account=PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -eo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}"

# uv lives on scratch (compute nodes have no system pip/python to bootstrap it).
export PATH="/project/scratch/PROJECT_ID/vla_augm/.local/bin:${PATH}"

# Keep the large uv wheel cache off $HOME (which is nearly full on inodes).
export UV_CACHE_DIR="/project/scratch/PROJECT_ID/vla_augm/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

# Deterministic CUDA wheels for torch==2.6.0 (repo pin). The A100 driver
# (CUDA 12.8) is backward-compatible with the cu124 runtime.
export UV_TORCH_BACKEND=cu124

echo "Host: $(hostname)"
echo "gcc: $(which gcc) $(gcc -dumpversion)  as: $(as --version | head -1)"
echo "nvidia-smi:"; nvidia-smi -L
echo "UV_CACHE_DIR=${UV_CACHE_DIR}  UV_TORCH_BACKEND=${UV_TORCH_BACKEND}"
echo "Starting install at $(date)"

bash requirements/install.sh embodied \
    --model gr00t --env libero \
    --venv .venv-gr00t --no-root

echo "Install finished at $(date)"
echo "Venv python: $(readlink -f ${REPO_PATH}/.venv-gr00t/bin/python)"
