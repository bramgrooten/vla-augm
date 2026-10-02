#!/bin/bash -l
# Install the .venv-pi environment (openpi model + LIBERO env) on Cluster B.
# Runs on a GPU node: it has internet, gcc, and a visible A100 so uv's CUDA
# wheels resolve correctly.
#
# TOOLCHAIN NOTE: we deliberately do NOT load the GCCcore/CMake/CUDA modules.
# GCCcore 13.3.0's gcc emits `--gdwarf-5` to the assembler, but Cluster B's
# GCCcore module ships no matching `as`, so builds fall back to the system
# `as` (binutils 2.30) which rejects that flag (evdev/robosuite build fails).
# The system gcc (8.5) + system as (2.30) are a self-consistent pair that
# builds every source dep in this env (evdev, decord, torch-memory-saver, ...).
# uv installs any cmake/ninja a package needs into its isolated build env.
# The libero+openpi path needs no nvcc (flash-attn uses a prebuilt wheel;
# no apex/Megatron is installed), so CUDA at build time is unnecessary.
#
# Submit with: sbatch vla_augm/scripts/cluster_b/install_venv_pi.sh
# Tip: the cluster is often full; add `--qos=dev --time=00:20:00` (and, if it
# still pends, `--reservation=gpudev`) so this short job backfills quickly.
#
#SBATCH --job-name=install-venv-pi
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

# Force rerun-sdk to the in-repo no-op stub. openpi -> lerobot pulls rerun-sdk,
# whose Rust wheels need glibc >= 2.31; Cluster B (RHEL 8) has glibc 2.28, so
# no rerun-sdk wheel is installable. rerun is visualization-only and unused by
# the pi0 RL path, so the stub is safe. Honored by install.sh's internal
# `uv pip install git+openpi` (uv reads UV_OVERRIDE like `--override`).
STUB_DIR="${REPO_PATH}/vla_augm/scripts/cluster_b/rerun_sdk_stub"
UV_OVERRIDE_FILE="$(mktemp)"
echo "rerun-sdk @ file://${STUB_DIR}" > "${UV_OVERRIDE_FILE}"
export UV_OVERRIDE="${UV_OVERRIDE_FILE}"

echo "Host: $(hostname)"
echo "gcc: $(which gcc) $(gcc -dumpversion)  as: $(as --version | head -1)"
echo "nvidia-smi:"; nvidia-smi -L
echo "UV_CACHE_DIR=${UV_CACHE_DIR}  UV_TORCH_BACKEND=${UV_TORCH_BACKEND}"
echo "Starting install at $(date)"

bash requirements/install.sh embodied \
    --model openpi --env libero \
    --venv .venv-pi --no-root

echo "Install finished at $(date)"
echo "Venv python: $(readlink -f ${REPO_PATH}/.venv-pi/bin/python)"
