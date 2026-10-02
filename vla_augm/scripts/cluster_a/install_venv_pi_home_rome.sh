#!/bin/bash
# Build .venv-pi on $HOME, where the scratch purge cannot reach it.
#
#   sbatch vla_augm/scripts/cluster_a/install_venv_pi_home.sh
#
# The scratch copy did not "break", it was eaten: the ~14-day access-time purge
# deletes individual files that nothing has read lately. Confirmed casualties in
# .venv-pi included typing_extensions (so torch stopped importing) and 10 of the
# 28 .py files in the bddl package -- among them bddl/parsing.py, which is
# 19,849 bytes in the install RECORD, absent from disk, and still has its
# __pycache__ .pyc from 06 Aug. That missing parser is the prime suspect for the
# LIBERO-Plus textures and lighting axes rendering an unperturbed scene, because
# liberoplus's get_problem_info() falls back silently to the BASE problem class
# on any parse exception. See vla_augm/handoff_texture_lighting_bug.md.
#
# CPU partition: this bills the L1 budget (1.6M SBU, 0.9% used), not the GPU one.
# Nothing here needs a GPU -- and rendering can use osmesa in software.

#SBATCH --job-name=install-venv-pi-rome
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# 1.5 h, not 4: a shorter job backfills into gaps a long one cannot, and the
# so101 build took 7.5 min. The download dominates, not the compile.
#SBATCH --time=01:30:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -euo pipefail

REPO_PATH="${HOME}/rl_inf"
VENV_DIR="${HOME}/rlinf_venvs/.venv-pi"
cd "${REPO_PATH}"
mkdir -p "$(dirname "${VENV_DIR}")" vla_augm/console_logs

# UV's cache is disposable, so it stays on scratch and off the $HOME inode quota.
export UV_CACHE_DIR="/scratch/USER/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

echo "[install] repo=${REPO_PATH} venv=${VENV_DIR} $(date)"
echo "[install] home inodes before: $(myquota 2>/dev/null | awk '/home6/{getline;getline;getline;print $3}')"

bash requirements/install.sh embodied \
    --model openpi --env libero \
    --venv "${VENV_DIR}" --no-root

echo "[install] base venv done $(date)"
"${VENV_DIR}/bin/python" -c "import torch, typing_extensions; print('torch', torch.__version__)"

# ---- LIBERO-Plus layer (vla_augm/install.md documents this by hand) -------------
if [ ! -d "${VENV_DIR}/libero_plus" ]; then
    git clone --depth 1 https://github.com/RLinf/LIBERO-plus.git "${VENV_DIR}/libero_plus"
fi
"${VENV_DIR}/bin/uv" pip install -r "${VENV_DIR}/libero_plus/extra_requirements.txt" 2>/dev/null \
    || "${VENV_DIR}/bin/python" -m uv pip install -r "${VENV_DIR}/libero_plus/extra_requirements.txt"
"${VENV_DIR}/bin/python" -m uv pip install -e "${VENV_DIR}/libero_plus"
"${VENV_DIR}/bin/python" -m uv pip install "mujoco<=3.9.0"

# The parser that the purge ate. Assert it is present rather than discover its
# absence again through a silently unperturbed eval.
"${VENV_DIR}/bin/python" - <<'PY'
import pathlib, bddl
p = pathlib.Path(bddl.__file__).parent / "parsing.py"
assert p.is_file(), f"bddl/parsing.py missing at {p} -- the textures bug would recur"
from bddl.parsing import scan_tokens
print("[install] bddl.parsing OK, scan_tokens present")
PY

echo "[install] .venv-pi ready at ${VENV_DIR} ($(date))"
echo "[install] NEXT: assets -> sbatch vla_augm/scripts/cluster_a/install_libero_plus_assets.sh"
