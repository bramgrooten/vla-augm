#!/bin/bash
# Build .venv-gr00t on Cluster A so the GR00T snapshots trained on Cluster B can
# be evaluated here. The Cluster A counterpart of
# vla_augm/scripts/cluster_b/install_venv_gr00t.sh.
#
#   sbatch vla_augm/scripts/cluster_a/install_venv_gr00t.sh
#
# The install only needs a *visible* CUDA device, so uv resolves the right
# wheels and the closing render test can open EGL -- a MIG slice would do, and
# bills 64 SBU/h against the H100's 192. But gpu_mig is four nodes deep and job
# 25649950 sat there 27 min with no start estimate while gpu_h100 had 22
# partially-free nodes and an empty queue. This install blocks 15 eval jobs, so
# the ~150 SBU premium buys back hours. Switch back to gpu_mig if h100 is busy.
#
# TOOLCHAIN: as on Cluster B, no GCCcore/CMake/CUDA modules are loaded. The
# system gcc and as are a self-consistent pair that builds the source deps, and
# uv brings its own cmake/ninja into each isolated build env. Loading the
# modules is also what pushes the driver env past the 128 KiB execve limit that
# silently kills Ray workers later.
#
# LIBERO-Plus is deliberately NOT installed as a second copy -- see the tail of
# this script.

#SBATCH --job-name=install-venv-gr00t
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -eo pipefail

REPO_PATH="/scratch/USER/rl_inf"
cd "${REPO_PATH}"

export PATH="${HOME}/.local/bin:${PATH}"

# Keep the wheel cache off $HOME: it holds ~1M inodes of quota and a torch-sized
# cache eats a visible slice of that. Scratch has 2.1M inodes free.
export UV_CACHE_DIR="/scratch/USER/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

# Deterministic CUDA wheels for the repo's torch==2.6.0 pin. The H100 driver is
# backward-compatible with the cu124 runtime, so the eval nodes run these fine.
export UV_TORCH_BACKEND=cu124

echo "Host: $(hostname)"
echo "gcc: $(which gcc) $(gcc -dumpversion)  as: $(as --version | head -1)"
nvidia-smi -L
echo "Starting install at $(date)"

bash requirements/install.sh embodied \
    --model gr00t --env libero \
    --venv .venv-gr00t --no-root

# LIBERO-Plus: reuse the tree .venv-pi already has (8.8 GB, 468k inodes) rather
# than cloning it again. Two things make that safe:
#   * ~/.liberoplus/config.yaml is per-USER, not per-venv, and already points at
#     .venv-pi/libero_plus/..., so both venvs resolve the same assets, bddl
#     files and init states with no extra configuration.
#   * an editable install only adds a path entry, so pointing two venvs at one
#     source tree is a supported layout, not a hack.
# Only the python deps are per-venv; those are packages, not data.
PLUS_SRC="${REPO_PATH}/.venv-pi/libero_plus"
if [ ! -d "${PLUS_SRC}" ]; then
    echo "[install] ERROR: ${PLUS_SRC} is gone -- .venv-gr00t would have no LIBERO-Plus." >&2
    exit 1
fi
source "${REPO_PATH}/.venv-gr00t/bin/activate"
uv pip install -r "${PLUS_SRC}/extra_requirements.txt"
uv pip install -e "${PLUS_SRC}"
uv pip install "mujoco<=3.9.0"

# Nothing pins hydra, so it resolves to a pre-release: 1.4.0.dev8 here against
# the 1.4.0.dev6 the Cluster B training venv froze. dev8 rejects the
# version_base="1.1" that both RLinf entry points declare --
# "version_base='1.1' is not supported in Hydra 1.4" -- and job 25655923 burned
# all five retry attempts on it. Pin to what .venv-pi runs, which is the pair
# that has actually completed eval sweeps. These are config libraries; the
# GR00T model code does not touch them, so this cannot move the numbers.
uv pip install "hydra-core==1.3.2" "omegaconf==2.3.1"

# Match .venv-pi's GL stack. Kept for consistency with the venv that has
# actually completed sweeps, NOT as a bug fix: the EGLError that motivated it
# turned out to be downstream cleanup noise, and pinning these did not stop the
# crash. The real cause is patched below.
uv pip install "mujoco==3.8.1" "pyopengl==4.0.0a1"

# LIBERO-Plus is not numpy-2 clean, and .venv-gr00t resolves numpy 2.4.6 where
# .venv-pi has 1.26.4. Its `noise` axis calls np.fromstring() in binary mode,
# which numpy 2 removed:
#   ValueError: The binary mode of fromstring is removed, use frombuffer instead
# raised inside env.reset(), killing the env subprocess and taking the sweep
# down via EOFError -> ray.kill. Every GR00T sweep died on the FIRST noise task
# it reached -- epoch 26/70 on the 5-axis order, 57/101 on the 7-axis one --
# which is why it looked deterministic and checkpoint-independent.
#
# TWO separate removals bite, in two different noise functions, so fixing the
# first alone just moves the crash from epoch 57 to 58:
#   motion_blur -> np.fromstring(binary)  -> np.frombuffer
#   fog -> plasma_fractal -> np.float_    -> np.float64
# Both replacements are the documented equivalents, so patching the tree
# .venv-pi shares does not disturb the pi0.5 results already collected.
# Verified afterwards by running all five noise functions (motion_blur,
# gaussian_blur, zoom_blur, fog, glass_blur) at the env's 256x256 render size
# under numpy 2.4.6.
PLUS_PY="${PLUS_SRC}/liberoplus/liberoplus/envs/env_wrapper.py"
sed -i -e "s/np\.fromstring(/np.frombuffer(/g" \
       -e "s/np\.float_/np.float64/g" "${PLUS_PY}"
echo "[install] patched numpy-2 removals in ${PLUS_PY}"

echo "Install finished at $(date)"
echo "Venv python: $(readlink -f "${REPO_PATH}/.venv-gr00t/bin/python")"

# Functional check: import the model side and render one LIBERO-Plus env, so a
# broken install fails here rather than 15 eval jobs from now.
module load 2025
module load Mesa/25.1.3-GCCcore-14.2.0
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=plus LIBERO_SUFFIX=all
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

# By absolute path, NOT bare `python`: the module loads above put their own
# Python ahead of the venv on PATH, so the activated venv is shadowed and the
# check dies with ModuleNotFoundError while the venv itself is perfectly fine
# (job 25653210 failed exactly this way with liberoplus installed). The eval
# scripts are unaffected -- they load modules first and activate afterwards.
"${REPO_PATH}/.venv-gr00t/bin/python" - <<'PY'
import liberoplus  # noqa: F401
from rlinf.envs.libero.libero_plus_axes import AXIS_CATEGORIES, get_axis_task_ids
ids = get_axis_task_ids("libero_spatial", sorted(AXIS_CATEGORIES), None, 1.0)
print("LIBERO_PLUS_IMPORT_OK; tasks per axis:", {k: len(v) for k, v in ids.items()})
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "[install] .venv-gr00t ready at $(date)"
