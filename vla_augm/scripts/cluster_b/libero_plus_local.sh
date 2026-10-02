#!/bin/bash
# Source (do not execute) from a cluster_b run/test script AFTER the venv is
# active and BEFORE python launches:
#   source "${REPO_PATH}/vla_augm/scripts/cluster_b/libero_plus_local.sh"
#
# Extracts the LIBERO-Plus asset pack from a squashfs image on scratch to
# node-local tmpfs and points liberoplus at it via LIBERO_CONFIG_PATH for
# this job only.
#
# Why: the extracted assets/ tree is ~458k files -- at the time of writing
# ~46% of the project scratch inode quota (1M). As one .sqfs it costs 1
# inode; /tmp on Cluster B GPU nodes is a 252 GB tmpfs, so extraction is free
# of quota and reads come from RAM (faster than Lustre for many small files).
# Assets are only read at env build / render-context creation, never per sim
# step, so training throughput is unaffected either way.
#
# bddl_files / init_states / datasets keep pointing at the .venv-pi repo copy
# (shared read-only data, exactly as ~/.liberoplus/config.yaml has them) --
# only the giant assets/ tree moves to tmpfs.
#
# Ray workers are separate processes on the same node; they inherit
# LIBERO_CONFIG_PATH through the serialized driver env, and the /tmp path is
# node-local, so all worker groups (incl. EnvOODGroup) see the files.
#
# Override the image location with LIBERO_PLUS_SQFS if ever needed.

SQFS="${LIBERO_PLUS_SQFS:-/project/scratch/PROJECT_ID/vla_augm/libero_plus_assets.sqfs}"
_LP_JOB_TAG="${SLURM_JOB_ID:-$$}"
LP_DIR="/tmp/liberoplus_assets_${_LP_JOB_TAG}"
LP_CFG="/tmp/liberoplus_cfg_${_LP_JOB_TAG}"
_VPI_PKG="/project/scratch/PROJECT_ID/vla_augm/rl_inf/.venv-pi/libero_plus/liberoplus/liberoplus"

if [ ! -f "${SQFS}" ]; then
    echo "[libero_plus_local] ERROR: image ${SQFS} missing." >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "${LP_DIR}/assets" ]; then
    mkdir -p "${LP_DIR}"
    echo "[libero_plus_local] extracting ${SQFS} -> ${LP_DIR} ..."
    # NB: Cluster B's unsquashfs is old: short options only (-n = no progress,
    # -f = overwrite, -p N = processors). Output goes to a file, not console.
    unsquashfs -n -f -p "${SLURM_CPUS_PER_TASK:-16}" -d "${LP_DIR}" "${SQFS}" > "${LP_DIR}.extract.log" 2>&1 || {
        echo "[libero_plus_local] ERROR: unsquashfs failed." >&2
        return 1 2>/dev/null || exit 1
    }
fi
echo "[libero_plus_local] assets at ${LP_DIR}/assets ($(find "${LP_DIR}/assets" -maxdepth 1 | wc -l) top-level entries)"

mkdir -p "${LP_CFG}"
cat > "${LP_CFG}/config.yaml" <<EOF
assets: ${LP_DIR}/assets
bddl_files: ${_VPI_PKG}/bddl_files
benchmark_root: ${_VPI_PKG}
datasets: ${_VPI_PKG}/../datasets
init_states: ${_VPI_PKG}/init_files
EOF
export LIBERO_CONFIG_PATH="${LP_CFG}"
echo "[libero_plus_local] LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"

# tmpfs is RAM -- do not leave ~7 GB behind after the job.
_libero_plus_local_cleanup() { rm -rf "${LP_DIR}" "${LP_CFG}"; }
trap _libero_plus_local_cleanup EXIT
