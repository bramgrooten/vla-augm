#!/usr/bin/env python3
"""Cache per-epoch training curves from wandb, so figures build offline.

The eval JSONL files under vla_augm/results/ hold OOD success at 25-epoch intervals,
but nothing else: in-distribution success, training reward and the loss terms
live only in the wandb run. This pulls them down once per run and writes a TSV,
so `make_training_figures.py` needs no network and a rebuild months from now
produces the same figure even if a run is later deleted.

    python vla_augm/analyze/fetch_wandb_curves.py            # missing runs only
    python vla_augm/analyze/fetch_wandb_curves.py --force    # refetch everything

Run ids come from `vla_augm/results/experiment_overview.tsv`, which is also the only
place the mapping from a run name to its wandb id is recorded.

EPOCH INDEXING. wandb's `_step` counts logging calls, not epochs, and different
metrics are logged at different cadences: `env/reward` lands once per training
epoch while `eval/success_once` lands once per evaluation. Both are dense in
their own series, so each metric is numbered by its own position -- reward gets
epochs 1..N, evaluations get `eval_every`, 2*`eval_every`, ... -- rather than by
`_step`, which would put them on incomparable axes.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "vla_augm", "results")
OUT_DIR = os.path.join(RESULTS, "wandb_curves")
ENTITY_PROJECT = "WANDB_ENTITY/WANDB_PROJECT"

# Metric -> cadence. "epoch" is logged every training epoch; "eval" every
# evaluation interval, which the caller converts with --eval-every.
METRICS = {
    "env/reward": "epoch",
    "train/actor/policy_loss": "epoch",
    "train/actor/entropy_loss": "epoch",
    "train/actor/approx_kl": "epoch",
    "train/actor/clip_fraction": "epoch",
    "train/critic/value_loss": "epoch",
    "eval/success_once": "eval",
    "eval/reward": "eval",
}


def runs_from_tsv() -> list[tuple[str, str, str]]:
    """(model, run_name, wandb_id) for every row that has a link."""
    out = []
    with open(os.path.join(RESULTS, "experiment_overview.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            link = (row.get("wandb link(s)") or "").strip()
            m = re.search(r"/runs/([A-Za-z0-9]+)", link)
            if row.get("run_name") and m:
                out.append((row["model"], row["run_name"], m.group(1)))
    return out


def fetch(run_id: str, eval_every: int) -> dict[int, dict[str, float]]:
    """{epoch: {metric: value}} for one run."""
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    table: dict[int, dict[str, float]] = {}
    for metric, cadence in METRICS.items():
        if metric not in run.summary:
            continue
        series = [r[metric] for r in run.scan_history(keys=["_step", metric],
                                                      page_size=2000)
                  if r.get(metric) is not None]
        for i, value in enumerate(series):
            epoch = (i + 1) if cadence == "epoch" else (i + 1) * eval_every
            table.setdefault(epoch, {})[metric] = float(value)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--eval-every", type=int, default=25,
                    help="training epochs between evaluations")
    ap.add_argument("--force", action="store_true", help="refetch runs already cached")
    ap.add_argument("--only", help="substring filter on the run name")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["epoch"] + list(METRICS)
    for model, name, run_id in runs_from_tsv():
        if args.only and args.only not in name:
            continue
        tag = f"{'pi05' if model.startswith('pi') else 'gr00t'}_{name}"
        path = os.path.join(args.out_dir, f"{tag}.tsv")
        if os.path.exists(path) and not args.force:
            print(f"  have {tag}")
            continue
        try:
            table = fetch(run_id, args.eval_every)
        except Exception as exc:  # noqa: BLE001 - one dead run should not stop the sweep
            print(f"  FAILED {tag} ({run_id}): {exc}", file=sys.stderr)
            continue
        with open(path, "w") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(cols)
            for epoch in sorted(table):
                row = table[epoch]
                w.writerow([epoch] + ["" if m not in row else f"{row[m]:.6g}"
                                      for m in METRICS])
        print(f"  wrote {tag}: {len(table)} epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
