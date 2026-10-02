#!/bin/bash
# Submit full-suite OOD evals for every every-25 snapshot of the finished arms.
#
#   bash vla_augm/scripts/cluster_a/eval/submit_ood_evals.sh            # all finished arms
#   bash vla_augm/scripts/cluster_a/eval/submit_ood_evals.sh alpha05    # arms matching a tag
#   DRY=1 bash vla_augm/scripts/cluster_a/eval/submit_ood_evals.sh      # print, submit nothing
#
# One job per checkpoint, ~3-4 h each on a single H100. The epoch-0 point is the
# SFT base every arm starts from, so it is evaluated ONCE (base.jsonl) and reused
# by every arm rather than re-run twelve times.
#
# Skips a checkpoint whose jsonl already exists, so re-running after a failed or
# preempted job submits only the gaps.

set -uo pipefail

REPO_PATH="/scratch/USER/rl_inf"
CKPT_ROOT="/projects/PROJECT_ID/users/USER/rl_inf/ckpts"
RESULTS="${REPO_PATH}/vla_augm/results/ood_full"
FILTER="${1:-}"
cd "${REPO_PATH}"
mkdir -p "${RESULTS}"

submit() {  # name, out, ckpt, axes
    local name="$1" out="$2" ckpt="$3" axes="$4"
    if [ -s "${out}" ]; then
        echo "skip ${name} (have $(basename "${out}"))"
        return
    fi
    if [ -n "${DRY:-}" ]; then
        echo "DRY ${name} (${axes} axes) <- ${ckpt:-<SFT base>}"
        return
    fi
    local jid
    jid=$(sbatch --parsable --job-name="ood-${name}" \
        --export=ALL,CKPT="${ckpt}",OUT="${out}",AXES="${axes}" \
        vla_augm/scripts/cluster_a/eval/eval_ood_ckpt.sh)
    echo "submitted ${jid} ${name} (${axes} axes)"
}

# Epoch 0: the shared SFT starting point, scored once for all arms, full 7 axes.
submit "base-ep0" "${RESULTS}/base.jsonl" "" 7

for run in "${CKPT_ROOT}"/*-seed52; do
    tag=$(basename "${run}")
    tag=${tag#*-libero_spatial_ppo_pi05_4gpu-augm_}
    tag=${tag%-seed52}
    [ -n "${FILTER}" ] && [[ "${tag}" != *"${FILTER}"* ]] && continue

    # Only arms that reached 300. A still-training arm would get its early
    # snapshots swept now and the rest later, which is the same total compute
    # but spreads one arm's curve across days of queue. FORCE=1 to override
    # (e.g. actor_only, deliberately stopped at 100).
    if [ -z "${FORCE:-}" ] && ! compgen -G "${run}/*/checkpoints/global_step_300" >/dev/null; then
        echo "skip ${tag}: not finished (no global_step_300)"
        continue
    fi

    for ckpt_dir in "${run}"/*/checkpoints/global_step_*; do
        [ -d "${ckpt_dir}" ] || continue
        step=$(basename "${ckpt_dir}")
        step=${step#global_step_}
        weights="${ckpt_dir}/actor/model_state_dict/full_weights.pt"
        if [ ! -f "${weights}" ]; then
            echo "WARN ${tag} ep${step}: no full_weights.pt, skipping"
            continue
        fi
        # The two points the paper reports (epoch 0, epoch 300) get all 7 axes;
        # the intermediate snapshots only carry the curve, so 5 axes is enough
        # and costs 31% less. avg_5axes is comparable across all of them;
        # avg_all7 exists only at the endpoints.
        axes=5
        [ "${step}" = "300" ] && axes=7
        submit "${tag}-ep${step}" "${RESULTS}/${tag}_ep${step}.jsonl" "${weights}" "${axes}"
    done
done
