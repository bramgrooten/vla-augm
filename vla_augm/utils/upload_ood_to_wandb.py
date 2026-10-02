"""Attach offline LIBERO-Plus OOD results to the wandb run that trained them.

The full-suite sweeps run as separate Slurm jobs and write one jsonl of
per-episode rows per checkpoint (``vla_augm/scripts/cluster_a/eval/eval_ood_ckpt.sh``).
This scores those rows per OOD axis and logs the curves back into the arm's
original wandb run, so the OOD panels sit next to the training curves instead
of in a detached eval run.

Two details make that work:

* The training run already logged ~1200 steps, and wandb drops out-of-order
  steps. So every OOD series is bound to its own ``ood_epoch`` step metric via
  ``define_metric``; the x-axis is the training epoch regardless of when the
  eval was uploaded.
* Epoch 0 is the shared SFT starting point (``base.jsonl``), uploaded to every
  arm so each curve starts where its training did.

Usage:
    python vla_augm/utils/upload_ood_to_wandb.py [--dry-run] [--filter TAG]
"""

import argparse
import json
import pathlib
from collections import defaultdict

import wandb

from rlinf.envs.libero.libero_plus_axes import AXIS_CATEGORIES, get_axis_task_ids

REPO = pathlib.Path("/scratch/USER/rl_inf")
PROJECT = "rl_inf"
SUITE = "libero_spatial"

# Per-model layout. The GR00T arms trained on Cluster B and their snapshots were
# pulled to scratch, so both the results dir and the checkpoint root differ from
# pi0.5's -- but they log into the same wandb project, so the upload itself is
# identical. `glob` is relative to `ckpt_root` and takes the arm tag.
MODELS = {
    "pi05": {
        "results": REPO / "vla_augm/results/ood_full",
        "ckpt_root": pathlib.Path("/projects/PROJECT_ID/users/USER/rl_inf/ckpts"),
        "glob": "*augm_{tag}-seed52/*/checkpoints/wandb_run_id.txt",
    },
    "gr00t": {
        "results": REPO / "vla_augm/results/ood_gr00t",
        "ckpt_root": pathlib.Path("/scratch/USER/ckpts/gr00t_augm"),
        "glob": "augm_{tag}-seed52/*/checkpoints/wandb_run_id.txt",
    },
}

# The five axes the periodic in-run eval used. Kept as a named subset so the
# paper can report avg_5axes against earlier numbers, with avg_all7 alongside.
FIVE_AXES = ["textures", "lighting", "layout", "camera", "noise"]


def axis_of_task() -> dict[int, str]:
    """Map every LIBERO-Plus task id to its axis (full suite, no subsampling)."""
    per_axis = get_axis_task_ids(SUITE, sorted(AXIS_CATEGORIES), None, 1.0)
    return {t: axis for axis, ids in per_axis.items() for t in ids}


def score(path: pathlib.Path, task_axis: dict[int, str]) -> dict[str, float]:
    """Per-axis means plus the two averages, from one checkpoint's episodes."""
    by_axis: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            axis = task_axis.get(int(row["task_id"]))
            if axis is None:
                continue
            for key, value in row.items():
                if key in ("task_id", "trial_id"):
                    continue
                by_axis[axis][key].append(float(value))

    metrics: dict[str, float] = {}
    for axis, keys in by_axis.items():
        for key, values in keys.items():
            metrics[f"eval_ood/{axis}/{key}"] = sum(values) / len(values)
        metrics[f"eval_ood/{axis}/num_trajectories"] = len(keys["success_once"])
        metrics[f"eval_ood_success_once/{axis}"] = metrics[
            f"eval_ood/{axis}/success_once"
        ]

    # Unweighted mean over axes, not over episodes: the axes have different task
    # counts (258 textures vs 390 language) and an episode-weighted mean would
    # quietly let the big axes dominate the headline number.
    def axis_mean(axes: list[str]) -> float | None:
        vals = [
            metrics[f"eval_ood_success_once/{a}"]
            for a in axes
            if f"eval_ood_success_once/{a}" in metrics
        ]
        return sum(vals) / len(vals) if len(vals) == len(axes) else None

    for name, axes in (("avg_all7", sorted(AXIS_CATEGORIES)), ("avg_5axes", FIVE_AXES)):
        value = axis_mean(axes)
        if value is not None:
            metrics[f"eval_ood_success_once/{name}"] = value
            metrics[f"eval_ood/{name}/success_once"] = value
    return metrics


def run_id_of(tag: str, ckpt_root: pathlib.Path, glob: str) -> str | None:
    """The wandb run id the arm's training wrote next to its checkpoints."""
    # Several run dirs share a tag: the arms that stalled on startup were
    # resubmitted, leaving short-lived directories behind. The real run is the
    # one that reached global_step_300 (or, for actor_only, has the most
    # snapshots), not simply the newest.
    matches = sorted(ckpt_root.glob(glob.format(tag=tag)))
    if not matches:
        return None
    best = max(matches, key=lambda p: len(list(p.parent.glob("global_step_*"))))
    return best.read_text().strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--filter", default="", help="only arms containing this")
    # Each run logs one point per epoch, so re-running the upload would put a
    # second point at every epoch it already covered. An arm whose sweeps are
    # still in flight is therefore excluded until complete, then uploaded once.
    parser.add_argument("--exclude", default="", help="skip arms containing this")
    parser.add_argument("--model", default="pi05", choices=sorted(MODELS))
    args = parser.parse_args()

    profile = MODELS[args.model]
    results, ckpt_root = profile["results"], profile["ckpt_root"]

    task_axis = axis_of_task()

    # {tag: {epoch: metrics}}; epoch 0 comes from the shared SFT sweep.
    per_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    base = results / "base.jsonl"
    base_metrics = score(base, task_axis) if base.is_file() else None

    for path in sorted(results.glob("*_ep*.jsonl")):
        tag, _, epoch = path.stem.rpartition("_ep")
        if args.filter and args.filter not in tag:
            continue
        if args.exclude and args.exclude in tag:
            continue
        per_arm[tag][int(epoch)] = score(path, task_axis)

    for tag, by_epoch in sorted(per_arm.items()):
        run_id = run_id_of(tag, ckpt_root, profile["glob"])
        if run_id is None:
            print(f"SKIP {tag}: no wandb_run_id.txt")
            continue
        if base_metrics is not None:
            by_epoch.setdefault(0, base_metrics)
        epochs = sorted(by_epoch)
        print(f"{tag} -> run {run_id}: epochs {epochs}")
        if args.dry_run:
            for epoch in epochs:
                # avg_all7 is absent on the 5-axis sweeps by design; only epoch
                # 0 and epoch 300 run the full seven.
                def fmt(key: str) -> str:
                    value = by_epoch[epoch].get(f"eval_ood_success_once/{key}")
                    return f"{value:.3f}" if value is not None else "  -  "

                print(f"    ep{epoch:<4} avg_all7={fmt('avg_all7')} avg_5axes={fmt('avg_5axes')}")
            continue

        run = wandb.init(project=PROJECT, id=run_id, resume="must")
        # Bind every OOD series to ood_epoch. Without this the points would be
        # placed at the run's global step, which already ran past 1200, and
        # wandb would drop them as out of order.
        wandb.define_metric("ood_epoch")
        wandb.define_metric("eval_ood/*", step_metric="ood_epoch")
        wandb.define_metric("eval_ood_success_once/*", step_metric="ood_epoch")
        for epoch in epochs:
            wandb.log({**by_epoch[epoch], "ood_epoch": epoch})
        run.finish()


if __name__ == "__main__":
    main()
