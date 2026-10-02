#!/bin/bash
# Full-suite LIBERO-Plus OOD eval of ONE checkpoint, all 7 axes (2402 tasks).
#
# Scores the every-25-epoch snapshots offline, which is where the paper's OOD
# numbers come from (in-run OOD is disabled -- it hangs the rollout->actor
# handoff). Writes per-episode rows so the axis breakdown can be computed
# afterwards without stepping the environments again; nothing is sent to wandb
# from here (vla_augm/utils/upload_ood_to_wandb.py attaches the results to the
# training run they belong to).
#
#   sbatch --job-name=ood-<tag>-ep<N> \
#       --export=ALL,MODEL=pi05,CKPT=<...>/global_step_<N>/actor/model_state_dict/full_weights.pt,OUT=<...>.jsonl \
#       vla_augm/scripts/cluster_a/eval/eval_ood_ckpt.sh
#
# CKPT empty = the SFT base model, i.e. the epoch-0 point every arm starts from.
# MODEL picks the venv and eval config (pi05 | gr00t); normally you go through
# the per-model submitter rather than calling this directly.

#SBATCH --job-name=ood-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
# Room for the in-job retry loop: a hung attempt is killed after ~15 min, so
# four hangs plus one real 4.5 h sweep fits inside 8 h. The 7-axis sweep is
# ~1.5x the 5-axis one, hence the headroom.
#SBATCH --time=08:00:00
#SBATCH --output=/projects/PROJECT_ID/users/%u/rl_inf/console_logs/%x_%j.log

set -e

# $HOME, not scratch: the scratch checkout lost git objects and tracked files to
# the access-time purge, and its .venv-pi was destroyed outright.
REPO_PATH="${HOME}/rl_inf"
: "${OUT:?set OUT to the per-episode jsonl path}"

# MODEL selects a profile; everything below it is model-independent. The
# per-model submitters (submit_ood_evals.sh, submit_gr00t_ood_evals.sh) set it.
#
# Deliberately a profile rather than one script per model: the retry/watchdog
# body below took ~10 fixes to get right, and a second copy of it would drift.
# We already lost ~14k SBU to 15 jobs running a *stale copy* of this script.
MODEL="${MODEL:-pi05}"
case "${MODEL}" in
    pi05)
        VENV="${HOME}/rlinf_venvs/.venv-pi"
        CONFIG_NAME="libero_spatial_openpi_pi05_plus_eval"
        ;;
    gr00t)
        VENV="${SCRATCH_VENV_GR00T:-/scratch/USER/rl_inf/.venv-gr00t}"
        CONFIG_NAME="libero_spatial_gr00t_plus_eval"
        ;;
    *)
        echo "unknown MODEL '${MODEL}' (expected pi05 or gr00t)" >&2
        exit 2
        ;;
esac

# EGL rendering (Mesa) + ImageMagick shared lib for python-wand (LIBERO-Plus dep).
module load 2025
module load Mesa/25.1.3-GCCcore-14.2.0
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"

# Slim the driver env: RLinf serializes it into each Ray worker's runtime-env and
# passes it to execve as one argument, which Linux caps at 128 KiB. The
# ImageMagick module pushes it to ~147 KB and every worker dies at exec (E2BIG)
# with no logs. See the note in eval_pi05_spatial_plus.sh.
unset CPATH LIBRARY_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    ACLOCAL_PATH _LMFILES_ LOADEDMODULES MANPATH XDG_DATA_DIRS
for _v in $(env | grep -oE '^__LMOD_REF_COUNT[A-Za-z_]*'); do unset "$_v"; done
echo "Driver env size: $(env | wc -c) bytes (must stay well under 131072)"

source "${VENV}/bin/activate"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=plus
export LIBERO_SUFFIX=all
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1
export RAY_worker_register_timeout_seconds=600
# These jobs take one GPU, so four land on the same node and each starts its own
# Ray. Without this, ray.init() auto-discovers the neighbours' clusters and dies
# with "Found multiple active Ray instances" (jobs 25499733/34/35). RAY_ADDRESS
# forces a fresh local instance; the per-job temp dir keeps the sockets and
# session files from colliding under the shared /tmp/ray.
export RAY_ADDRESS="local"
# Must be SHORT and node-local: Ray puts its plasma socket under this path and
# AF_UNIX caps the whole filename at 107 bytes, so a /scratch-shared path blew
# the limit and every job died in 3 min (25512255 and the rest of that batch).
export RAY_TMPDIR="/tmp/ray${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

STAMP="$(date +'%Y%m%d-%H:%M:%S')"
# Run output goes to project space, not $HOME (near its inode quota) and not
# scratch (purged by access time).
PROJECT_ROOT="${PROJECT_ROOT:-/projects/PROJECT_ID/users/${USER}/rl_inf}"
LOG_DIR="${PROJECT_ROOT}/logs/${STAMP}-ood-eval-$(basename "${OUT}" .jsonl)"
mkdir -p "${LOG_DIR}"

echo "Job: ${SLURM_JOB_ID} on $(hostname)"
echo "Checkpoint: ${CKPT:-<SFT base>}"
echo "Episode results: ${OUT}"

# add_value_head must match the checkpoint: RL snapshots carry value_head.* keys
# and the strict load fails without the head, while the SFT base has no critic.
if [ -n "${CKPT:-}" ]; then
    CKPT_ARGS=(runner.ckpt_path="${CKPT}" rollout.model.add_value_head=True)
elif [ "${MODEL}" = "gr00t" ]; then
    # GR00T is the exception: its rollout path calls the value branch
    # unconditionally, so add_value_head=False dies in 7 s with
    # "'FlowMatchingActionHeadForRLActionPrediction' object has no attribute
    # 'value_head'" -- which looked like the 0/N handshake stall and cost the
    # anchor three jobs before the traceback was read. Enabling the head on the
    # SFT base leaves it randomly initialised, which is harmless here: it feeds
    # only values_vlm, never the flow-matching action prediction, so success
    # rates are untouched. pi0.5 guards the call and keeps False.
    CKPT_ARGS=(rollout.model.add_value_head=True)
else
    CKPT_ARGS=(rollout.model.add_value_head=False)
fi

# A LoRA checkpoint cannot be loaded into a plain model. full_weights.pt is the
# FULL state dict, so with adapters attached its keys carry peft's rewriting
# (...base_model.model...q_proj.base_layer.weight, plus lora_A/lora_B), and the
# rollout worker's load_state_dict is strict -- it would fail outright. The
# eval model therefore has to be built with the SAME adapter layout as the arm
# that produced the checkpoint. LORA=1 does that; LORA_LAYERS=top6 must match
# whatever the arm used (see vla_augm/scripts/cluster_a/run_augm_arm.sh).
if [ "${LORA:-0}" = "1" ]; then
    CKPT_ARGS+=(
        rollout.model.is_lora=True
        "rollout.model.lora_rank=${LORA_RANK:-32}"
    )
    if [ "${LORA_LAYERS:-top6}" = "top6" ]; then
        LORA_RX='^(?!.*vision_tower).*layers\.(?:1[2-7])\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
        CKPT_ARGS+=("rollout.model.lora_target_regex='${LORA_RX}'")
    fi
fi

# AXES=5 restricts the sweep to the five visual axes (1662 of 2402 tasks, -31%
# compute); anything else runs the full 7. The intermediate snapshots only need
# the curve shape, so they run at 5; epoch 0 and epoch 300 -- the two points the
# paper reports -- run the full 7. rollout_epoch must be recomputed with the
# task count, since one epoch covers total_num_envs tasks.
# 24 envs rather than the config's 32: at 32 a sweep peaked at 195 GB against
# the 180 GB a single-GPU share gets (job 25474720, OUT_OF_MEMORY), and the
# silent EOFError subprocess deaths look like the same pressure killing one
# worker instead of the whole job. Fewer envs = more epochs for the same episode
# count, which is why the walltime is 8 h.
# The axis task lists are read from task_classification.json, which is normally
# discovered inside the installed liberoplus package. .venv-gr00t no longer has
# one -- the scratch purge took its libero_plus tree, the same way it took
# .venv-pi's bddl/parsing.py -- so every GR00T sweep died in seconds on
# FileNotFoundError. Point at the copy committed in this repo, which is
# venv-independent and the same file either way.
export LIBERO_PLUS_TASK_CLASSIFICATION="${LIBERO_PLUS_TASK_CLASSIFICATION:-${REPO_PATH}/vla_augm/utils/task_classification.json}"

ENVS="${ENVS:-24}"
if [ "${AXES:-7}" = "layout" ]; then
    # layout alone: 385 of 2402 tasks. Redirected 385/385 by the bddl-swap bug
    # fixed in libero_env.py, so like textures and lighting it needs redoing.
    # It reads as its own AXES value rather than a count because the axis set,
    # not the number of axes, is what identifies a sweep.
    read -r TASK_IDS N_TASKS < <("${VENV}/bin/python" -c "
from rlinf.envs.libero.libero_plus_axes import get_axis_task_ids
ids = sorted(t for v in get_axis_task_ids('libero_spatial',
    ['layout'], None, 1.0).values() for t in v)
print('[' + ','.join(map(str, ids)) + ']', len(ids))
")
    AXIS_ARGS=(+env.eval.task_id_filter="${TASK_IDS}")
elif [ "${AXES:-7}" = "2" ]; then
    # textures + lighting only: 550 of 2402 tasks. These are the two axes that
    # a damaged .venv-pi rendered unperturbed -- both deliver through scene_xml,
    # and the other three do not, which is why only these need redoing.
    read -r TASK_IDS N_TASKS < <("${VENV}/bin/python" -c "
from rlinf.envs.libero.libero_plus_axes import get_axis_task_ids
ids = sorted(t for v in get_axis_task_ids('libero_spatial',
    ['textures','lighting'], None, 1.0).values() for t in v)
print('[' + ','.join(map(str, ids)) + ']', len(ids))
")
    AXIS_ARGS=(+env.eval.task_id_filter="${TASK_IDS}")
elif [ "${AXES:-7}" = "5" ]; then
    read -r TASK_IDS N_TASKS < <("${VENV}/bin/python" -c "
from rlinf.envs.libero.libero_plus_axes import get_axis_task_ids
ids = sorted(t for v in get_axis_task_ids('libero_spatial',
    ['textures','lighting','layout','camera','noise'], None, 1.0).values() for t in v)
print('[' + ','.join(map(str, ids)) + ']', len(ids))
")
    # task_id_filter is read via cfg.get() in LiberoEnv but absent from this
    # config, so it must be ADDED (+) -- a plain override dies on
    # "Key 'task_id_filter' is not in struct". rollout_epoch already exists.
    AXIS_ARGS=(+env.eval.task_id_filter="${TASK_IDS}")
else
    N_TASKS=2402
    AXIS_ARGS=()
fi
# One epoch runs one episode per env, so this many epochs sweep every task once.
EPOCHS=$(( (N_TASKS + ENVS - 1) / ENVS ))
AXIS_ARGS+=(env.eval.total_num_envs="${ENVS}" env.eval.rollout_epoch="${EPOCHS}")

# VERIFY_VIDEO=1 records clips from the real eval path, which is the only way to
# prove a perturbation reaches the policy's observation rather than merely the
# standalone diagnostic. Do NOT set extra_info_on_video: it hangs the env worker
# before the first rollout epoch (jobs 25824230 vs 25824231 bisected it).
if [ "${VERIFY_VIDEO:-0}" = "1" ]; then
    VID_DIR="${REPO_PATH}/vla_augm/results/ood_textures_lighting_rerun/verify_video"
    mkdir -p "${VID_DIR}"
    AXIS_ARGS+=(
        env.eval.video_cfg.save_video=True
        env.eval.video_cfg.info_on_video=True
        env.eval.video_cfg.video_base_dir="${VID_DIR}"
    )
    echo "VERIFY_VIDEO on -> ${VID_DIR}"
fi
echo "${AXES:-7}-axis sweep: ${N_TASKS} tasks, ${ENVS} envs, rollout_epoch=${EPOCHS}"

# Retry inside the allocation instead of across Slurm jobs.
#
# Measured over 323 sweeps: even with every config bug fixed, only ~40% of
# attempts finish. The rest either hang at "Rollout Epochs 0/N" with both worker
# groups initialised (the Ray channel handshake never completes -- the same
# stall the training arms hit) or lose an env subprocess to EOFError. Neither
# correlates with the node, the checkpoint, or how many jobs share the node, so
# there is nothing to avoid: the only reliable lever is another attempt.
#
# Retrying across jobs wastes the whole walltime per failure and needs a human
# in the loop. Retrying here costs STALL_LIMIT minutes for a hang, so five
# attempts turn a 40% job into ~92%.
#
# WHAT THIS CANNOT FIX: the stall is node-dependent, and every attempt reuses
# the same allocation, so a bad node fails all of them identically. Seven
# straight stalls on gcn120 while all fourteen sibling sweeps completed on
# other nodes (job 25694952). When one sweep alone keeps stalling at 0/N,
# compare its node against the ones that succeeded and resubmit with
#   sbatch --exclude=<node> ...
# rather than raising ATTEMPTS, which buys nothing on a bad node.
# The retry loop expects failing commands: `wait` reports the signal from a
# killed attempt, and `grep` exits 1 before the progress bar appears. Under the
# script's `set -e` either one aborted the job, so the first watchdog kill ended
# the run after a single attempt instead of retrying (job 25590503).
set +e

# This job's own process group, so the watchdog can refuse to kill it.
SELF_PGID=$(ps -o pgid= $$ | tr -d ' ')

# 8, not 5: measured on the GR00T wave, an attempt clears the handshake only
# about half the time, and the anchor stalled four times running. A clean sweep
# is ~85 min (5 axes) to ~2 h (7 axes) against an 8 h walltime, so even eight
# attempts at the stall limit below fit inside the allocation with room to
# spare -- the walltime, not this number, is the real ceiling.
ATTEMPTS="${ATTEMPTS:-8}"
# Minutes without the progress bar advancing before an attempt is declared hung.
# A GR00T epoch takes ~1.2 min at 24 envs and startup ~5 min, so 10 clears
# normal jitter while detecting the stall a third sooner than 15 did. Raise it
# again if a model with slower epochs starts getting killed mid-sweep (every
# kill so far has been at 0/N, i.e. the handshake, never in flight).
STALL_LIMIT="${STALL_LIMIT:-10}"

for attempt in $(seq 1 "${ATTEMPTS}"); do
    rm -f "${OUT}"
    ATTEMPT_LOG="${LOG_DIR}/attempt_${attempt}.log"
    # Fresh Ray state per attempt: a wedged GCS from the previous try would
    # otherwise be discovered and inherited.
    export RAY_TMPDIR="/tmp/ray${SLURM_JOB_ID}_${attempt}"
    mkdir -p "${RAY_TMPDIR}"

    # setsid puts the driver and every Ray worker it spawns in one process
    # group, so a hung attempt can be killed as a unit without touching the
    # other jobs sharing this node.
    #
    # The inner shell reports its OWN pid, which after setsid IS the new process
    # group id. Asking `ps` for the child's pgid instead is a race: setsid()
    # has often not taken effect yet, ps returns the pgid this SCRIPT is in, and
    # the watchdog's kill then takes down the batch job itself -- one attempt
    # instead of five. Five sweeps died that way (25657698 and siblings) and it
    # is the real cause of the earlier single-attempt abort blamed on `set -e`.
    PGID_FILE="${LOG_DIR}/pgid_${attempt}"
    rm -f "${PGID_FILE}"
    setsid bash -c 'echo $$ > "$0"; exec "$@"' "${PGID_FILE}" \
        python -u "${REPO_PATH}/evaluations/eval_embodied_agent.py" \
        --config-path "${REPO_PATH}/evaluations/libero/" \
        --config-name "${CONFIG_NAME}" \
        runner.logger.log_path="${LOG_DIR}" \
        runner.logger.logger_backends=null \
        +runner.episode_results_path="${OUT}" \
        "${CKPT_ARGS[@]}" "${AXIS_ARGS[@]}" > "${ATTEMPT_LOG}" 2>&1 &
    pid=$!
    pgid=""
    for _ in $(seq 1 30); do
        [ -s "${PGID_FILE}" ] && { pgid=$(tr -d ' \n' < "${PGID_FILE}"); break; }
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    echo "attempt ${attempt}/${ATTEMPTS}: pid ${pid} pgid ${pgid:-<unknown>}"

    last_progress=""
    stalled=0
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 60
        current=$(grep -aoE "Rollout Epochs: +[0-9]+%[^0-9]*[0-9]+/[0-9]+" \
            "${ATTEMPT_LOG}" 2>/dev/null | tail -1)
        if [ "${current}" = "${last_progress}" ]; then
            stalled=$((stalled + 1))
        else
            stalled=0
            last_progress="${current}"
        fi
        if [ "${stalled}" -ge "${STALL_LIMIT}" ]; then
            echo "attempt ${attempt}: no progress for ${STALL_LIMIT} min at '${current:-<none>}', killing"
            # Never group-kill our own process group: that is suicide, and it is
            # exactly what the old ps-based pgid read did when it lost the race.
            if [ -n "${pgid}" ] && [ "${pgid}" != "${SELF_PGID}" ]; then
                kill -9 -- "-${pgid}" 2>/dev/null
            else
                echo "attempt ${attempt}: no distinct pgid (${pgid:-<unknown>}); killing pid only"
                kill -9 "${pid}" 2>/dev/null
            fi
            break
        fi
    done
    wait "${pid}" 2>/dev/null
    rm -rf "${RAY_TMPDIR}"

    if [ -s "${OUT}" ]; then
        echo "attempt ${attempt}: wrote $(wc -l < "${OUT}") episodes to ${OUT}"
        tail -5 "${ATTEMPT_LOG}"
        exit 0
    fi
    echo "attempt ${attempt}: no results, last progress '${last_progress:-<none>}'"
done

echo "all ${ATTEMPTS} attempts failed for ${OUT}" >&2
exit 1
