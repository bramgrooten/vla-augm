#!/bin/bash
# Submit LIBERO-Plus OOD evals for the GR00T augmentation grid.
#
#   bash vla_augm/scripts/cluster_a/eval/submit_gr00t_ood_evals.sh          # everything ready
#   DRY=1 bash vla_augm/scripts/cluster_a/eval/submit_gr00t_ood_evals.sh    # print, submit nothing
#
# The GR00T entry point; pi0.5 has its own (submit_ood_evals.sh). Both hand off
# to eval_ood_ckpt.sh, which carries the retry/watchdog loop, via MODEL=.
#
# WAVE 1 is epochs 0, 100 and 300 only -- 14 snapshots plus the shared SFT base,
# 15 sweeps. The every-25 points can be added later by pulling more checkpoints
# and widening STEPS in the submitter below; the epoch-0 sweep is reused rather
# than repeated, so a later wave costs only its own new points.
#
# ORDER MATTERS. Jobs are submitted anchor-first: the SFT base (which every
# arm's curve starts from and which nothing else can be read without), then the
# no-augmentation baseline, then the augmented arms. Within an arm, epoch 300
# goes before epoch 100 -- the endpoint is what the paper reports. If the
# allocation runs dry partway through, what is finished is still a publishable
# comparison rather than a scatter of unanchored points.
#
# Skips a checkpoint whose jsonl already exists, and skips one whose weights
# have not landed from Cluster B yet, so it is safe to re-run while the transfer
# is still going.

set -uo pipefail

REPO_PATH="/scratch/USER/rl_inf"
CKPT_ROOT="/scratch/USER/ckpts/gr00t_augm"
RESULTS="${REPO_PATH}/vla_augm/results/ood_gr00t"
cd "${REPO_PATH}"
mkdir -p "${RESULTS}"

# Anchor first, then the baseline, then the augmented arms. uniform and
# actor_only collapsed at epoch 100 and have no 300.
ARMS_ORDER=(
    augm_none
    augm_critic_only_ovl_alpha05
    augm_critic_only_ovl_alpha025
    augm_critic_only_ovl_alpha075
    augm_critic_only_ovlshift_alpha05
    augm_critic_only_shift
    augm_uniform_ovl_alpha05
    augm_actor_only_ovl_alpha05
)

# Names of this model's eval jobs already queued or running, so re-running the
# submitter to fill gaps does not double-submit the sweeps still in flight.
IN_FLIGHT=$(squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep '^g4ood-' || true)

submit() {  # name, out, ckpt, axes
    local name="$1" out="$2" ckpt="$3" axes="$4"
    if [ -s "${out}" ]; then
        echo "skip ${name} (have $(basename "${out}"))"
        return
    fi
    if grep -qx "g4ood-${name}" <<<"${IN_FLIGHT}"; then
        echo "skip ${name} (already queued)"
        return
    fi
    # Existence is not enough. rsync --partial writes an interrupted file under
    # its FINAL name, so a checkpoint still in flight looks present but is
    # truncated -- observed at 1.38 GB of 5.08 GB mid-transfer. Loading one
    # would burn a GPU allocation to die at load_state_dict, so require the file
    # to be within a hair of full size before believing it.
    local min_bytes=5368709120  # 5 GiB; a complete snapshot is 5.08 GiB
    if [ -n "${ckpt}" ]; then
        local size
        size=$(stat -c %s "${ckpt}" 2>/dev/null || echo 0)
        if [ "${size}" -lt "${min_bytes}" ]; then
            echo "WAIT ${name}: weights incomplete ($(( size / 1024 / 1024 )) MB)"
            return
        fi
    fi
    if [ -n "${DRY:-}" ]; then
        echo "DRY ${name} (${axes} axes) <- ${ckpt:-<SFT base>}"
        return
    fi
    local jid
    jid=$(sbatch --parsable --job-name="g4ood-${name}" \
        --export=ALL,MODEL=gr00t,CKPT="${ckpt}",OUT="${out}",AXES="${axes}" \
        vla_augm/scripts/cluster_a/eval/eval_ood_ckpt.sh)
    echo "submitted ${jid} ${name} (${axes} axes)"
}

# Epoch 0: the shared SFT starting point, scored once for all arms, full 7 axes.
submit "base-ep0" "${RESULTS}/base.jsonl" "" 7

for arm in "${ARMS_ORDER[@]}"; do
    run_dir=$(echo "${CKPT_ROOT}/${arm}-seed52"/*/checkpoints)
    if [ ! -d "${run_dir}" ]; then
        echo "WAIT ${arm}: nothing transferred yet"
        continue
    fi
    tag=${arm#augm_}

    # 300 before 100: the endpoint is the number the paper reports, and it also
    # carries the 7-axis sweep.
    for step in 300 100; do
        weights="${run_dir}/global_step_${step}/actor/model_state_dict/full_weights.pt"
        [ -e "${run_dir}/global_step_${step}" ] || continue
        # 7 axes at the endpoints the paper reports (epoch 0 and epoch 300),
        # 5 at epoch 100 -- 31% cheaper and enough to carry the curve, since
        # avg_5axes is comparable across every point.
        axes=5
        [ "${step}" = "300" ] && axes=7
        submit "${tag}-ep${step}" "${RESULTS}/${tag}_ep${step}.jsonl" "${weights}" "${axes}"
    done
done
