#!/bin/bash -l
# Install LIBERO-Plus (OOD generalization eval suite) onto the existing
# .venv-gr00t on Cluster B -- the gr00t counterpart of install_libero_plus.sh.
#
# KEY DIFFERENCE from the .venv-pi install: NO asset-pack download. Scratch is
# at ~99% of its inode quota and the asset pack alone is ~458k files
# (.venv-pi/libero_plus/liberoplus/liberoplus/assets). The shared
# ~/.liberoplus/config.yaml already points assets/bddl_files/init_states at
# the .venv-pi copy (written on first import there, same mechanism the
# standard-LIBERO ~/.libero/config.yaml already uses for .venv-gr00t's runs),
# so an editable install here transparently reuses those files. Total cost:
# the git clone + a few installed packages (~2k inodes).
# Consequence: .venv-gr00t OOD evals break if .venv-pi/libero_plus is deleted.
#
# Submit with (short/backfillable; add --reservation=gpudev if it pends):
#   sbatch --qos=dev --reservation=gpudev --time=01:00:00 \
#       vla_augm/scripts/cluster_b/install_libero_plus_gr00t.sh
#
# Needs internet + a GPU (the functional test renders a LIBERO-Plus env via EGL).
#
#SBATCH --job-name=install-libero-plus-gr00t
#SBATCH --account=PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -eo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}"

# uv on scratch; caches off the near-full $HOME. System gcc (not GCCcore) in
# case any dep source-builds -- see install_venv_pi.sh for why.
export PATH="/project/scratch/PROJECT_ID/vla_augm/.local/bin:${PATH}"
export UV_CACHE_DIR="/project/scratch/PROJECT_ID/vla_augm/.cache/uv"
export HF_HOME="/project/scratch/PROJECT_ID/vla_augm/.cache/huggingface"
mkdir -p "${UV_CACHE_DIR}"

source "${REPO_PATH}/.venv-gr00t/bin/activate"

echo "Host: $(hostname) | python: $(which python)"

# (safety) snapshot the working env so we can roll back if LIBERO-Plus deps
# ever disturb it: uv pip install -r <this file>
uv pip freeze > "vla_augm/venv-gr00t-freeze-preplus-$(date +%Y%m%d).txt"

# 1) Clone the RLinf LIBERO-plus fork into the venv dir (matches install layout).
if [ ! -d ".venv-gr00t/libero_plus/.git" ]; then
    git clone --depth 1 https://github.com/RLinf/LIBERO-plus.git .venv-gr00t/libero_plus
else
    echo "[install] .venv-gr00t/libero_plus already cloned; reusing."
fi

# 2) Python deps -- additive only (wand; scikit-image already in .venv-gr00t),
#    editable liberoplus, and the mujoco<=3.9.0 pin (no-op if already satisfied).
uv pip install -r .venv-gr00t/libero_plus/extra_requirements.txt
uv pip install -e .venv-gr00t/libero_plus
uv pip install "mujoco<=3.9.0"

# 3) Assets: shared, NOT downloaded. Two valid sources, in order:
#    a) the .venv-pi in-package assets dir (via ~/.liberoplus/config.yaml),
#    b) the squashfs image extracted to node-local tmpfs (the default since
#       the in-venv tree was packed away for inode quota) -- source the same
#       helper the run scripts use, which sets LIBERO_CONFIG_PATH for us.
SQFS="/project/scratch/PROJECT_ID/vla_augm/libero_plus_assets.sqfs"
CFG="${LIBERO_CONFIG_PATH:-${HOME}/.liberoplus}/config.yaml"
ASSETS=""
[ -f "${CFG}" ] && ASSETS=$(sed -n 's/^assets: //p' "${CFG}")
if [ -n "${ASSETS}" ] && [ -d "${ASSETS}" ]; then
    echo "[install] sharing assets via ${CFG}: ${ASSETS}"
elif [ -f "${SQFS}" ]; then
    echo "[install] no shared assets dir; using squashfs image ${SQFS} via libero_plus_local.sh"
    source "${REPO_PATH}/vla_augm/scripts/cluster_b/libero_plus_local.sh"
else
    echo "[install] ERROR: no assets source found (neither ${CFG} assets dir nor ${SQFS})." >&2
    exit 1
fi

# 4) Functional test: ImageMagick (for python-wand) + EGL rendering of a
#    LIBERO-Plus env. Loading ImageMagick also pulls GCCcore libs onto
#    LD_LIBRARY_PATH (harmless for this single-process import test).
module load env/release/2024.1
module load Mesa ImageMagick/7.1.1-38-GCCcore-13.3.0
export MAGICK_HOME="${EBROOTIMAGEMAGICK}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=plus LIBERO_SUFFIX=all
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

echo "[test] import sanity: wand/ImageMagick + liberoplus + shared assets ..."
python - <<'PY'
import wand.image  # noqa: F401 -- forces the ImageMagick shared lib to load
print("wand/ImageMagick OK")
from liberoplus.liberoplus.benchmark import Benchmark  # noqa: F401
print("liberoplus Benchmark import OK")
from liberoplus.liberoplus import get_libero_path
import pathlib
assets = pathlib.Path(get_libero_path("assets"))
entries = sorted(p.name for p in assets.iterdir()) if assets.is_dir() else []
assert assets.is_dir() and entries, f"assets missing at {assets}"
print(f"assets OK at {assets}: {entries[:8]}")
print("LIBERO_PLUS_IMPORT_OK")
PY

echo "[install] LIBERO-Plus (gr00t, shared assets) install + test finished at $(date)"
