#!/bin/bash -l
# Re-record the six supplementary videos on Cluster B -- recording round 3.
#
#   PI05_CKPT=/path/to/full_weights.pt sbatch vla_augm/scripts/cluster_b/record_supplementary_videos.sh
#
# WHY. The six clips in vla_augm/supplementary_videos/ came from a Cluster A sweep
# whose LIBERO-Plus asset tree was incomplete. Textures, lighting and layout are
# delivered by swapping the scene XML, which silently renders the DEFAULT scene
# when assets/scenes/ is missing: the overlay still reads "table 1" / "light 10"
# but the table is plain wood. Camera and noise do not touch those assets and
# were unaffected. All six are re-recorded anyway so the set comes from one
# checkpoint and one environment.
#
# WHY MELUXINA. Its assets are complete. Verified in libero_plus_assets.sqfs:
# assets/scenes/ holds 15739 entries including scenes/ph/Cobblestone01_GLOSS_6K
# and scenes/phs/CarbonFiber001_GLOSS_6K -- two DIFFERENT directories, which is
# exactly what a partial sync gets wrong while still passing a "scene_xml refs
# exist" check. vla_augm/results/texture_debug/ holds renders proving it end to end.
#
# THE ONE INPUT THIS SCRIPT CANNOT PROVIDE. The pi0.5 critic-only/overlay/
# alpha0.5 seed-52 epoch-300 checkpoint does not exist on Cluster B -- there is
# no pi0.5 checkpoint here at all, only the 92 GR00T ones. It lives on Cluster A:
#   /projects/PROJECT_ID/users/USER/rl_inf/ckpts/
#     *augm_critic_only_ovl_alpha05-seed52/*/checkpoints/global_step_300/actor/
#     model_state_dict/full_weights.pt
# Cluster A refuses inbound port 22 from Cluster B, so the transfer must be PUSHED
# from Cluster A (Cluster B accepts inbound SSH normally) -- the same asymmetry
# that made pull_ckpts_from_cluster_b.sh run on the Cluster A side. Run ON SNELLIUS:
#
#   mkdir -p /project/scratch/PROJECT_ID/vla_augm/rl_inf/ckpts/pi05_cluster_a   # on Cluster B
#   rsync -av --partial --append-verify --progress \
#     <the full_weights.pt above> \
#     USER@cluster-b.example.org:/project/scratch/PROJECT_ID/vla_augm/rl_inf/ckpts/pi05_cluster_a/
#
# --partial --append-verify so a drop resumes mid-file instead of restarting:
# it is ~7 GB over a login node, Cluster B has degraded CPU capacity as of
# 2026-09-09 (the cluster status page), and login sessions get dropped. Run it under tmux
# or nohup.
#
# ------------------------------------------------------------------------
#SBATCH --job-name=pi05-supp-videos
#SBATCH --account=PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=vla_augm/console_logs/%x_%j.log

set -uo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}" || exit 1
export REPO_PATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

# ---------------------------------------------------------------- checkpoint
PI05_CKPT="${PI05_CKPT:-}"
if [ -z "${PI05_CKPT}" ]; then
    echo "ERROR: set PI05_CKPT to the epoch-300 full_weights.pt (see header)." >&2
    exit 1
fi
if [ ! -f "${PI05_CKPT}" ]; then
    echo "ERROR: PI05_CKPT does not exist: ${PI05_CKPT}" >&2
    exit 1
fi
export PI05_CKPT
echo "checkpoint: ${PI05_CKPT} ($(du -h "${PI05_CKPT}" | cut -f1))"

# ------------------------------------------------------------------- modules
# ImageMagick is load-bearing, not cosmetic: liberoplus reaches MagickMotionBlur
# through wand for the sensor-noise axis. With wand unusable the axis becomes a
# silent no-op and clip 5 would show an unperturbed scene -- the same failure
# class this whole re-record exists to undo. Its deps (libjbig and friends) come
# from sibling modules, so MAGICK_HOME alone is not enough; the module system
# has to be used.
module load env/release/2024.1
module load Mesa
module load ImageMagick/7.1.1-38-GCCcore-13.3.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"

# RLinf serializes the whole driver env into each Ray worker's runtime_env as a
# SINGLE execve argument, so the module system's path variables blow past
# MAX_ARG_STRLEN (128 KiB) and every worker dies. Drop what only the compiler
# toolchain needs.
unset CPATH LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    ACLOCAL_PATH _LMFILES_ LOADEDMODULES MANPATH XDG_DATA_DIRS
for _v in $(env | grep -oE '^__LMOD_REF_COUNT[A-Za-z_]*'); do unset "$_v"; done
echo "Driver env size: $(env | wc -c) bytes (must stay well under 131072)"

source "${REPO_PATH}/.venv-pi/bin/activate"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HF_HOME="/project/scratch/PROJECT_ID/vla_augm/.cache/huggingface"
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# ---------------------------------------------------------------- LIBERO-Plus
# Unpacks libero_plus_assets.sqfs to node-local tmpfs and exports LP_DIR.
source "${REPO_PATH}/vla_augm/scripts/cluster_b/libero_plus_local.sh"

# LIBERO_CONFIG_PATH cannot redirect the scene XMLs: BDDLBaseDomain and every
# object module build their asset paths from __file__. So point the package's
# assets/ at the extracted tree for the life of the job.
PKG_ASSETS="${REPO_PATH}/.venv-pi/libero_plus/liberoplus/liberoplus/assets"
if [ -e "${PKG_ASSETS}" ]; then
    echo "NOTE: ${PKG_ASSETS} already exists; leaving it alone."
else
    ln -s "${LP_DIR}/assets" "${PKG_ASSETS}"
    # libero_plus_local.sh installed its own EXIT trap to free the tmpfs copy;
    # a bare `trap ... EXIT` would silently replace it, so chain both.
    trap 'rm -f "${PKG_ASSETS}"; _libero_plus_local_cleanup' EXIT
    echo "symlinked ${PKG_ASSETS} -> ${LP_DIR}/assets"
fi

# ------------------------------------------------------------- preflight
# Fail before burning an allocation on clips that would be silently wrong.
python - <<'PY' || exit 1
import os, pathlib, sys
ok = True

try:
    from wand.api import library
    assert hasattr(library, "MagickMotionBlurImage")
    print("wand OK: motion blur available (noise axis will perturb)")
except Exception as e:
    print(f"FATAL: wand unusable ({e}); the noise axis would silently no-op")
    ok = False

pkg = pathlib.Path(os.environ["REPO_PATH"]) / ".venv-pi/libero_plus/liberoplus/liberoplus"
for sub, why in [("assets/scenes", "textures + lighting axes"),
                 ("assets/scenes/ph", "Cobblestone01_GLOSS_6K"),
                 ("assets/scenes/phs", "CarbonFiber001_GLOSS_6K")]:
    d = pkg / sub
    n = len(list(d.iterdir())) if d.is_dir() else 0
    print(f"{'OK  ' if n else 'FATAL'} {sub}: {n} entries ({why})")
    ok = ok and n > 0

sys.exit(0 if ok else 1)
PY

# ------------------------------------------------------------------ recording
STAMP="$(date +'%Y%m%d-%H%M%S')"
OUT_ROOT="${REPO_PATH}/vla_augm/results/supp_videos_round3_${STAMP}"
mkdir -p "${OUT_ROOT}"
CLIP_ROOT="${OUT_ROOT}/clips"
mkdir -p "${CLIP_ROOT}"
echo "output root: ${OUT_ROOT}"

# Two processes, not one. LIBERO_TYPE is read at import time and is
# process-wide, and the in-distribution twins need the STANDARD suite while the
# perturbed clips need `plus`. Same checkpoint, same seed, same node, so "one
# run" in the sense that matters holds.
run_one () {
    local phase="$1" libero_type="$2" config="$3"
    local log="${OUT_ROOT}/${phase}.log"
    local episodes="${OUT_ROOT}/${phase}_episodes.jsonl"
    # A DIRECTORY PER PHASE, and this is load-bearing. RecordVideo writes
    # seed_<eval_seed>/step<k>.mp4, both configs share eval_seed 12345 on
    # purpose (the twins must draw the same flow prior), and each phase's flush
    # counter restarts at 0. Pointed at one directory the in-distribution pass
    # would overwrite all 80 perturbed clips with its 9, and the naming step
    # would then label whatever survived.
    export VIDEO_DIR="${CLIP_ROOT}/${phase}"
    mkdir -p "${VIDEO_DIR}"
    echo "=== ${phase}: LIBERO_TYPE=${libero_type} config=${config}"
    echo "    clips -> ${VIDEO_DIR}"
    LIBERO_TYPE="${libero_type}" python -u \
        "${REPO_PATH}/evaluations/eval_embodied_agent.py" \
        --config-path "${REPO_PATH}/evaluations/libero/" \
        --config-name "${config}" \
        runner.logger.log_path="${OUT_ROOT}/${phase}_logs" \
        +runner.episode_results_path="${episodes}" \
        > "${log}" 2>&1
    local rc=$?
    echo "=== ${phase}: exit ${rc}, $(ls "${VIDEO_DIR}"/seed_*/*.mp4 2>/dev/null | wc -l) clips"
    [ ${rc} -ne 0 ] && tail -30 "${log}"
    return ${rc}
}

run_one plus     plus     libero_spatial_plus_video_pi05_cluster_b_eval || exit 1
run_one ind      standard libero_spatial_ind_video_pi05_cluster_b_eval  || exit 1

# ---------------------------------------------------------------- label clips
# RecordVideo names clips seed_<n>/step<k>.mp4 -- the flush counter and nothing
# else. name_video_clips.py joins the k-th clip to the k-th row of the episode
# dump and refuses to rename if the two sides disagree.
# --ind is not optional for the twins. In a standard-suite eval the episode's
# task id IS the base index, where a LIBERO-Plus id indexes the 2402-task
# perturbed pool. Without the flag the twins get looked up in the perturbed
# selection, and ids 0-5 exist there as TEXTURES tasks -- so six clips would be
# renamed textures_base*. That is only caught because ids 6-9 are absent from
# the selection and the script refuses the whole batch.
python "${REPO_PATH}/vla_augm/analyze/name_video_clips.py" \
    --videos "${CLIP_ROOT}/plus" \
    --episodes "${OUT_ROOT}/plus_episodes.jsonl" 2>&1 | tail -8
python "${REPO_PATH}/vla_augm/analyze/name_video_clips.py" --ind \
    --videos "${CLIP_ROOT}/ind" \
    --episodes "${OUT_ROOT}/ind_episodes.jsonl" 2>&1 | tail -14

# ------------------------------------------------------------------ verify
# The whole point of round 3. A perturbed clip must differ from its
# in-distribution twin by a mean absolute pixel delta of ~50; the round-2 clips
# managed ~5. If this prints FAIL the recording is worthless -- do not ship it.
python "${REPO_PATH}/vla_augm/scripts/cluster_b/tests_utils/verify_video_perturbation.py" \
    --clips "${CLIP_ROOT}" \
    --report "${OUT_ROOT}/perturbation_check.txt"

# ---------------------------------------------------------------- ship dir
# A NEW directory beside the round-2 clips, which stay where they are: they are
# what the current draft references, and until the report below is all-PASS
# there is nothing better to replace them with.
SHIP_DIR="${REPO_PATH}/vla_augm/supplementary_videos/round3"
mkdir -p "${SHIP_DIR}"

echo
echo "done."
echo "  clips:   ${CLIP_ROOT}/{plus,ind}"
echo "  report:  ${OUT_ROOT}/perturbation_check.txt"
echo "  ship to: ${SHIP_DIR}"
echo
echo "Curate the six only after the report reads all-PASS -- one per axis plus"
echo "an in-distribution reference, same base task, all successful episodes:"
echo "  cp ${CLIP_ROOT}/ind/IND_base<b>_ok.mp4            ${SHIP_DIR}/<b>_0_in_distribution.mp4"
echo "  cp ${CLIP_ROOT}/plus/textures_base<b>_*_ok.mp4 ${SHIP_DIR}/<b>_1_background_textures.mp4"
echo "  cp ${CLIP_ROOT}/plus/lighting_base<b>_*_ok.mp4 ${SHIP_DIR}/<b>_2_light_conditions.mp4"
echo "  cp ${CLIP_ROOT}/plus/layout_base<b>_*_ok.mp4 ${SHIP_DIR}/<b>_3_objects_layout.mp4"
echo "  cp ${CLIP_ROOT}/plus/camera_base<b>_*_ok.mp4 ${SHIP_DIR}/<b>_4_camera_viewpoints.mp4"
echo "  cp ${CLIP_ROOT}/plus/noise_base<b>_*_ok.mp4 ${SHIP_DIR}/<b>_5_sensor_noise.mp4"
