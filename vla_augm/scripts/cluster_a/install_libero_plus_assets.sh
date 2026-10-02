#!/bin/bash
# Download the LIBERO-Plus asset pack into .venv-pi on Cluster A.
#
# requirements/install.sh clones RLinf/LIBERO-plus and pip-installs it, but the
# assets live in a separate 6.4 GB HuggingFace zip that nothing fetches. Without
# them every EnvOODGroup worker dies at construction with
#   FileNotFoundError: .../liberoplus/assets/scenes/libero_tabletop_base_style.xml
# and the run hangs (training itself is unaffected -- only eval_ood). This is the
# Cluster A counterpart of vla_augm/scripts/cluster_b/install_libero_plus.sh and of the
# manual steps in vla_augm/install.md.
#
#   sbatch vla_augm/scripts/cluster_a/install_libero_plus_assets.sh
#
# CPU partition, so it bills the L1 (CPU-only) budget, not the GPU experiment
# budget. ~8.7 GB and ~458k files land under .venv-pi/libero_plus -- scratch has
# ~2.4M free inodes, so that fits, but it is the reason this tree must never be
# copied to project space (see the archive script).
#
# Do NOT run this while archive_to_project.sh is tarring .venv-pi: tar exits 1
# on "file changed as we read it" and the archive job dies (job 25326359).
#
#SBATCH --job-name=install-libero-plus-assets
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

# Not `set -u`: the venv's own bin/activate reads PYTHONPATH unguarded.
set -eo pipefail

REPO_PATH="/scratch/USER/rl_inf"
cd "${REPO_PATH}"

module load 2025
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HF_HOME="/scratch/USER/.cache/huggingface"
source "${REPO_PATH}/.venv-pi/bin/activate"

# Call the interpreter by path, never bare `python`: the ImageMagick module below
# pulls in EasyBuild's Python 3.13.1 and prepends it to PATH, shadowing the venv
# (job 25328657 died on ModuleNotFoundError: wand for exactly this reason).
# run_pi05_4gpu.sh dodges it by loading modules before activating; being explicit
# is immune to the ordering either way.
VENV_PY="${REPO_PATH}/.venv-pi/bin/python"

PKG=$("${VENV_PY}" -c "import pathlib, liberoplus.liberoplus as l; print(pathlib.Path(l.__file__).resolve().parent)")
[ -d "${PKG}" ] || { echo "[install] ERROR: liberoplus package dir not found ('${PKG}')" >&2; exit 1; }
echo "[install] liberoplus package dir: ${PKG}"

if [ -d "${PKG}/assets/scenes" ]; then
    echo "[install] assets already present; nothing to do."
else
    "${VENV_PY}" - "${PKG}" <<'PY'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="Sylvest/LIBERO-plus", filename="assets.zip",
                    repo_type="dataset", local_dir=sys.argv[1])
print("downloaded", p)
PY
    unzip -oq "${PKG}/assets.zip" -d "${PKG}"
    rm -f "${PKG}/assets.zip"

    # The zip is mis-packed with the packer's absolute path baked in, so the
    # tree lands under inspire/hdd/.../LIBERO-plus-0/assets rather than at
    # ${PKG}/assets, which is where __init__.py's benchmark_root_path looks.
    if [ ! -d "${PKG}/assets" ]; then
        src=$(find "${PKG}" -maxdepth 14 -type d -name assets -path '*LIBERO-plus-0*' | head -1)
        [ -n "${src}" ] || { echo "[install] ERROR: extracted assets dir not found." >&2; exit 1; }
        mv "${src}" "${PKG}/assets"
        rm -rf "${PKG}/inspire"
    fi
    # hf_hub_download leaves its blob cache behind; the zip is 6.4 GB and we
    # already have the extracted copy.
    rm -rf "${PKG}/.cache"
fi

echo "[install] assets/ contents:"
ls "${PKG}/assets"

# python-wand needs the ImageMagick shared lib from a module, not the venv --
# same modules the run scripts load, so this test matches runtime conditions.
module load Mesa/25.1.3-GCCcore-14.2.0
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="${EBROOTIMAGEMAGICK}"
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=plus LIBERO_SUFFIX=all

# Deliberately does NOT import liberoplus.benchmark: that pulls in robosuite ->
# mujoco -> PyOpenGL, which cannot initialise on a CPU partition (job 25334859:
# AttributeError: 'NoneType' object has no attribute 'glGetError'). Rendering is
# the GPU smoke test's job; this script only owns the assets.
"${VENV_PY}" - <<'PY'
import pathlib
import wand.image  # noqa: F401 -- forces the ImageMagick shared lib to load
from liberoplus.liberoplus import get_libero_path

assets = pathlib.Path(get_libero_path("assets"))
scene = assets / "scenes" / "libero_tabletop_base_style.xml"
assert scene.is_file(), f"missing the scene file the OOD envs failed on: {scene}"
print("assets OK at", assets)
print(sorted(p.name for p in assets.iterdir())[:10])
print("LIBERO_PLUS_ASSETS_OK")
PY

echo "[install] done at $(date)"
