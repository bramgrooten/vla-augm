#!/bin/bash
# Submit the whole GR00T N1.5 augmentation grid on Cluster B, in priority order.
#
#   bash vla_augm/scripts/cluster_b/launch_gr00t_augm_grid.sh
#
# Run the gate first -- vla_augm/scripts/cluster_b/smoke_gr00t_augm.sh must have
# completed an epoch, an in-distribution eval and a checkpoint end to end.
#
# All arms go out at once: the queue is busy, and submission order sets
# priority, so the ordering below is the experiment's priority order rather
# than a schedule. Do NOT stagger these into timed waves.
#
# The grid. Every arm is LIBERO-Spatial, GR00T N1.5, 300 epochs, seed 52.
# critic_only / random_overlay / alpha 0.5 is the anchor, shared by all three
# subsections -- which is what lets 8 arms cover the grid rather than 8+ per
# section.
#
#   4.1 which module   none, critic_only (anchor), uniform, actor_only
#   4.2 which type     random_shift, random_overlay_shift
#   4.3 how strong     alpha 0.25, 0.75
#
# alpha 0.1 and 0.9 are cut from 4.3, and the run is 250 epochs rather than
# 300, because the full 10-arm 300-epoch grid costs ~740 GPU node-hours
# against 618 left in the August allocation. This costs the extremes of the
# strength sweep; everything else matches the Cluster A pi0.5 protocol.
#
# Nothing here varies strength or the overlay bank outside 4.3's alpha: the
# arms differ only in the three fields passed to run_gr00t_augm_arm.sh.

set -euo pipefail

REPO_PATH="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "${REPO_PATH}"

SEED="${SEED:-52}"
ARM="vla_augm/scripts/cluster_b/run_gr00t_augm_arm.sh"

# The anchor goes out FIRST, ahead of the rest of its own subsection. It is the
# only arm that appears in all three of 4.1, 4.2 and 4.3 -- every other arm is
# read against it -- so it is the one arm that must not be the one left short.
# Submission order sets queue priority here, so this ordering is the insurance.
echo "=== 4.1 which module (first priority) ==="
bash "${ARM}" critic_only random_overlay 0.5  "${SEED}"   # anchor: 4.1 + 4.2 + 4.3
bash "${ARM}" none        random_overlay 0.5  "${SEED}"   # baseline for every comparison
bash "${ARM}" uniform     random_overlay 0.5  "${SEED}"
bash "${ARM}" actor_only  random_overlay 0.5  "${SEED}"

echo "=== 4.2 which type (second priority) ==="
bash "${ARM}" critic_only random_shift         0.5 "${SEED}"
bash "${ARM}" critic_only random_overlay_shift 0.5 "${SEED}"

echo "=== 4.3 how strong (third priority) ==="
bash "${ARM}" critic_only random_overlay 0.25 "${SEED}"
bash "${ARM}" critic_only random_overlay 0.75 "${SEED}"

echo
echo "8 arms submitted. Watch with: squeue -u \$USER"
echo "Snapshots land in /project/scratch/PROJECT_ID/vla_augm/rl_inf/ckpts/gr00t_augm/<tag>-seed${SEED}/"
