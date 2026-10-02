#!/bin/bash -l
# Install LIBERO-Plus (OOD generalization eval suite) onto the existing
# .venv-pi on Cluster B -- the counterpart of the Cluster A steps in
# vla_augm/install.md. Layers the RLinf/LIBERO-plus fork + its 6.4 GB asset pack so
# pi0/pi0.5 checkpoints can be evaluated on the perturbed suites.
#
# Submit with (short/backfillable; add --reservation=gpudev if it pends):
#   sbatch --qos=dev --reservation=gpudev --time=01:00:00 \
#       vla_augm/scripts/cluster_b/install_libero_plus.sh
#
# Needs internet + a GPU (the functional test renders a LIBERO-Plus env via EGL).
# ~8.7 GB and ~480k files land under .venv-pi/libero_plus -- mostly the asset
# pack, so this bumps scratch INODE usage a lot (check `myquota`).
#
#SBATCH --job-name=install-libero-plus
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

source "${REPO_PATH}/.venv-pi/bin/activate"

echo "Host: $(hostname) | python: $(which python)"

# (safety) snapshot the working env so we can roll back if LIBERO-Plus deps
# ever disturb it: uv pip install -r <this file>
uv pip freeze > "vla_augm/venv-pi-freeze-preplus-$(date +%Y%m%d).txt"

# 1) Clone the RLinf LIBERO-plus fork into the venv dir (matches install layout).
if [ ! -d ".venv-pi/libero_plus/.git" ]; then
    git clone --depth 1 https://github.com/RLinf/LIBERO-plus.git .venv-pi/libero_plus
else
    echo "[install] .venv-pi/libero_plus already cloned; reusing."
fi

# 2) Python deps -- additive only (wand, scikit-image; both wheels), editable
#    liberoplus, and the mujoco<=3.9.0 pin (a no-op if already satisfied).
uv pip install -r .venv-pi/libero_plus/extra_requirements.txt
uv pip install -e .venv-pi/libero_plus
uv pip install "mujoco<=3.9.0"

# 3) Asset pack: normally a 6.4 GB / ~458k-file download into the package dir.
#    On Cluster B that tree is packed as ONE squashfs image
#    (/project/scratch/PROJECT_ID/vla_augm/libero_plus_assets.sqfs, 1 inode) and run
#    scripts extract it to node-local tmpfs -- see libero_plus_local.sh. Only
#    download into the venv if the image is absent.
SQFS="/project/scratch/PROJECT_ID/vla_augm/libero_plus_assets.sqfs"
PKG=$(python -c "import pathlib, liberoplus.liberoplus as l; print('PKGPATH='+str(pathlib.Path(l.__file__).resolve().parent))" 2>/dev/null | sed -n 's/^PKGPATH=//p')
[ -d "${PKG}" ] || { echo "[install] ERROR: could not resolve liberoplus package dir (got '${PKG}')" >&2; exit 1; }
echo "[install] liberoplus package dir: ${PKG}"
if [ -d "${PKG}/assets" ]; then
    echo "[install] assets already present at ${PKG}/assets; skipping download."
elif [ -f "${SQFS}" ]; then
    echo "[install] no in-venv assets, using squashfs image at runtime: ${SQFS}"
else
    python - "$PKG" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Sylvest/LIBERO-plus", filename="assets.zip",
                repo_type="dataset", local_dir=sys.argv[1])
print("downloaded assets.zip")
PY
    unzip -oq "${PKG}/assets.zip" -d "${PKG}"
    rm -f "${PKG}/assets.zip"

    # The zip is mis-packed with a baked-in absolute path, so assets land under
    # inspire/hdd/.../LIBERO-plus-0/assets instead of ${PKG}/assets (where the
    # code looks: __init__.py -> benchmark_root_path/assets). Move it there.
    if [ ! -d "${PKG}/assets" ]; then
        src=$(find "${PKG}" -maxdepth 14 -type d -name assets -path '*LIBERO-plus-0*' | head -1)
        if [ -n "${src}" ]; then
            mv "${src}" "${PKG}/assets"
            rm -rf "${PKG}/inspire"
        else
            echo "[install] ERROR: could not locate extracted assets dir." >&2
            exit 1
        fi
    fi
fi
if [ -d "${PKG}/assets" ]; then
    echo "[install] assets/ contents (expect articulated_objects/ scenes/ textures/ ...):"
    ls "${PKG}/assets" | head
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

# Resolve assets the same way the run scripts do: extract the squashfs image
# to node-local tmpfs and export LIBERO_CONFIG_PATH (no-op for the venv copy
# if in-venv assets exist -- the helper always uses the image, so skip it
# then to avoid a pointless 7 GB extraction).
if [ ! -d "${PKG}/assets" ]; then
    source "${REPO_PATH}/vla_augm/scripts/cluster_b/libero_plus_local.sh"
fi

echo "[test] import sanity: wand/ImageMagick + liberoplus + assets ..."
python - <<'PY'
import wand.image  # noqa: F401 -- forces the ImageMagick shared lib to load
print("wand/ImageMagick OK")
from liberoplus.liberoplus.benchmark import Benchmark  # noqa: F401
print("liberoplus Benchmark import OK")
import pathlib
from liberoplus.liberoplus import get_libero_path
assets = pathlib.Path(get_libero_path("assets"))
entries = sorted(p.name for p in assets.iterdir()) if assets.is_dir() else []
assert assets.is_dir() and entries, f"assets missing at {assets}"
print(f"assets OK at {assets}: {entries[:8]}")
print("LIBERO_PLUS_IMPORT_OK")
PY

echo "[install] LIBERO-Plus install + test finished at $(date)"
