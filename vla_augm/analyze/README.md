# vla_augm/analyze

Analysis that produces numbers **in the paper**. Everything here recomputes from
the per-episode eval JSONL in `vla_augm/results/`, so no figure or table number is
ever hand-copied.

For interactive poking at a *training run's* periodic OOD evals, use
`vla_augm/utils/analyze_ood_eval.py` instead — that reads the runner's
`logs/<run>/ood_eval/step_*.jsonl`, which carry their own `axis` field.

## Files

| file | what it does |
|---|---|
| `ood_table.py` | Per-axis + aggregate OOD success for any eval JSONL; arm-vs-arm deltas with sigmas; `--floor` for the evaluation noise floor. |
| `ablation_tables.py` | Regenerates Tables 2 and 3 (Sections 4.2 and 4.3: augmentation type and strength). `--check` compares every recomputed OOD value against the one recorded in `vla_augm/results/experiment_overview.tsv`. |
| `make_env_figure.py` | Regenerates the appendix figure of the five perturbation axes, from the clips in `paper/videos/`. `--list` shows coverage, `--pick` forces a clip. |
| `make_env_figure_local.py` | Same figure as `make_env_figure.py`, but rendered straight from the simulator on a Mac: no video, no frame search, no crop. `--survey AXIS` reports how different and how bright each variant is, for choosing one. `--rollout N` steps a random policy first, so the arm is in the workspace. Writes one panel per cell, `env_<tag>_<ind\|ood>_<axis>.png`. |
| `make_policyview_local.py` | Local counterpart to `make_augmentation_figures.py policyview`: base and wrist views of one observation at 224x224, rendered from the simulator, so they are uncropped *and* have no HUD. `--rollout N` picks how far into a random rollout to sample; the wrist camera needs it, since at home pose the gripper faces empty table. `--bank` also writes an augmented copy of each view, blended by the real `RandomOverlay` from `rlinf/algorithms/augmentation.py` (get a bank with `vla_augm/utils/sample_overlay_bank.py`). Writes `policyview_roll<N>_<main\|wrist>.png`. |
| `make_augmentation_figures.py` | `teaser` builds Figure 1 (observations at three strengths plus the OOD-vs-epoch curve); `ladder` builds the appendix strength sweep; `survey` prints a contact sheet for choosing frames. |
| `record_libero_videos.py` | Records LIBERO-Plus episodes locally on a Mac with a random policy, as 512x256 clips named like the cluster's -- so `make_env_figure.py` reads them unchanged, but without the HUD. Needs the venv from vla_augm/install.md ("Rendering LIBERO-Plus locally on Mac"). |
| `placement_table.py` | Regenerates Table 1 (Section 4.1, augmentation placement): per-axis OOD, the five-axis average, and the per-axis task counts. `--latex` emits the table body. |

## The one thing to get right: macro vs micro

The final-checkpoint evals (`vla_augm/scripts/cluster_a/eval/submit_ood_evals.sh`) write a
`task_id` but **no** `axis` field, so the axis has to be recovered from
`vla_augm/utils/task_classification.json`. Once you have per-axis rates there are two
ways to aggregate, and on the same file they differ by 1–2 points:

- **micro** — pool every episode, then take the mean.
- **macro** — take each axis's success rate, then average those rates unweighted.

They disagree because the axes have very unequal sizes (258 tasks for background
textures vs. 390 for language instructions on `libero_spatial`) and the largest
axes are the hardest. Those sizes are an artifact of how LIBERO-Plus was built,
not a claim about which shifts matter, so **the paper reports macro**.

Beware: wandb's `eval_ood/all/success_once` is **micro** — `embodied_runner.py`
builds one mask over episodes and takes `merged[key][mask].mean()`. A wandb
number and a paper number for the same checkpoint will therefore not match, and
neither is wrong. Always state which convention a number uses.

`ood_table.py` prints both, so a mismatch is visible rather than mysterious.

## Which axes

The headline OOD metric is the macro average over the **five visual axes**:
background textures, light conditions, objects layout, camera viewpoints, sensor
noise. Robot initial states and language instructions are excluded — no image
augmentation can address either, so including them only dilutes the effect.

Pass `--axes all` to get all seven, or e.g. `--axes camera,noise` for one slice.

## Evaluation noise floor

Evaluation is not deterministic even with `use_fixed_reset_state_ids` (GPU
nondeterminism, env subprocess scheduling, and the policy's own sampling all move
individual episodes). Re-evaluate one checkpoint twice and take the spread:

```
python vla_augm/analyze/ood_table.py --floor \
    vla_augm/results/ood_gr00t/none_ep300.jsonl \
    vla_augm/results/ood_gr00t/noise_floor/none_ep300_run2.jsonl
```

Current floors (macro over the five visual axes):

- **GR00T N1: 1.77 pts**, from two sweeps of the `none` epoch-300 RL checkpoint.
- **pi0.5: 0.06 pts**, from two sweeps of the SFT checkpoint.

Use the GR00T figure as the conservative threshold for both. The pi0.5 pair
measures the *SFT* checkpoint, not an RL one, so it is an optimistic estimate of
the noise an RL arm actually carries.

This is **evaluation** noise, not seed noise. Every arm is a single training seed
(52), so none of these numbers bound run-to-run variation from retraining. Say
"above the evaluation noise floor", never "significant".

## Examples

```bash
# per-axis breakdown for one checkpoint
python vla_augm/analyze/ood_table.py vla_augm/results/ood_full/none_ep300.jsonl

# arm vs baseline (first file is the baseline)
python vla_augm/analyze/ood_table.py \
    vla_augm/results/ood_full/none_ep300.jsonl \
    vla_augm/results/ood_full/critic_only_ovl_alpha05_ep300.jsonl \
    --labels none,critic-only

# regenerate the paper's Table 1
python vla_augm/analyze/placement_table.py --latex
```

## IND numbers in Tables 2 and 3

`ablation_tables.py` recomputes OOD from the JSONLs but reads IND from
`vla_augm/results/experiment_overview.tsv`, because only the two SFT checkpoints have
IND eval JSONLs on disk -- for a trained arm the in-distribution number lives only
in wandb. `--check` is the guard against that TSV drifting: it recomputes the macro
OOD for all 18 rows and compares each against the value entered there by hand. All
18 agree as of this writing, so the TSV and the JSONLs are consistent.

## Recording your own clips

`make_env_figure.py` consumes clips; `record_libero_videos.py` produces them. The
clips already in `paper/videos/` came off the cluster and carry a burned-in HUD
(reward, termination, instruction), which is why the figure script crops. Clips
recorded locally have no HUD and need no crop.

A random policy never solves a task, so every local clip is named `_fail` and
`make_env_figure.py` will prefer a cluster `_ok` clip over it when both exist.
That only affects which clip is picked, not whether the figure builds. Nothing
here produces a success rate -- no policy is run.

## Figure 1

`make_augmentation_figures.py` works from the already-blended clips in
`vla_augm/example_videos/`, so it needs only ffmpeg and the standard library -- no
torch and no overlay bank, unlike `vla_augm/utils/preview_augmentation.py`, which does
the blending itself. Reach for that one when you need an alpha or a bank no example
clip covers; reach for this one when you need a paper figure.

The teaser's curve is emitted as a pgfplots fragment (`teaser_curve.tex`), not a
raster, so it keeps the paper's fonts and needs no matplotlib.

Two things it will not guess for you:

- **The HUD is not part of the observation.** The recorder burns reward,
  termination and the instruction onto the render. The default crop drops those
  bands; change the crop and check that they are still gone.
- **Frames are chosen by content.** The overlay bank is unfiltered ImageNet, so a
  draw can blend in legible text, a recognizable person, or an animal close-up
  nobody wants in a paper -- the bank contains an octopus, a shark, a scuba diver,
  boxers and cell microscopy, and one earlier pick showed a photo caption in mirror
  image. The defaults are restricted to water, grass, flowers and stone. Run
  `survey` and re-pick whenever the source clip changes.

Alphas other than 0.25/0.5/0.75 are derived from the 0.75 clip by exploiting the
linearity of the blend; the script header records the PSNR checks that validate it.

`--latex` emits the table body exactly as it appears between `\toprule` and
`\bottomrule` in `paper/vla_augm/main.tex` -- count row, rules, `\phantom{0}`
alignment and the bold column winners included -- so it can be pasted in
wholesale. Never hand-edit those rows in the paper: do it here and re-paste,
or the two drift apart silently.

## Evaluation sizes

**IND** — 500 episodes: 10 LIBERO-Spatial tasks x 50 fixed initial states
(`use_fixed_reset_state_ids: True`), capped at 240 steps. pi0.5 deals these over
104 envs x 5 epochs = 520 slots, GR00T over 20 x 25 = 500; surplus slots are
skipped, not rerun, so nothing is double-counted.

**OOD** — one episode per LIBERO-Plus task. The full suite is 2402 tasks; the five
visual axes are 1662 of them (textures 258, lighting 292, layout 385, camera 376,
noise 351). A sweep runs `ceil(n_tasks / total_num_envs)` epochs, so the slot
count slightly exceeds the task count and a few tasks get a second trial — 2432
episodes for 2402 tasks at 32 envs, 2424 at 24 envs. `ood_table.py` pools every
episode, so those repeats are included rather than dropped.

**Episode-step budgets differ by model**: pi0.5 OOD sweeps run at
`max_episode_steps: 320` (`evaluations/libero/libero_spatial_openpi_pi05_plus_eval.yaml`),
GR00T at 240 (`..._gr00t_plus_eval.yaml`). Compare arms *within* a model; a
cross-model OOD comparison is not apples-to-apples.

## IND numbers: recomputed where a JSONL exists

`placement_table.py` recomputes the OOD columns from the per-episode JSONLs, and
does the same for IND wherever an IND JSONL exists under `vla_augm/results/ind/`.
The pi0.5 SFT (epoch-0) row now takes that path: `ind/pi05_sft_ep0.jsonl`, 500
episodes, 78.8% `success_once`.

The arms that were *trained* have no such file -- their in-distribution eval curve
lives only in wandb -- so their IND is still transcribed into the `ARMS` table at
the top of the script, each next to its run id. If you re-run one of those, update
both the OOD JSONL path and that number.

To move an arm onto the recompute path, drop its IND eval JSONL into
`vla_augm/results/ind/` and put the relative path in the IND field instead of the
float. `ind_success()` asserts the file holds exactly 500 episodes, so a partial
or double sweep fails loudly rather than quietly shifting the number.

## Known evaluation quirks

Two things about the OOD sweeps that are worth remembering before comparing numbers.

**Repeated tasks are pooled.** A sweep runs `ceil(n_tasks / n_envs)` epochs, so surplus
environment slots re-run a few tasks rather than idling: a 258-task axis yields 288
episodes at 32 environments and 280 at 24. Every trial is pooled, so those repeats are
counted twice and the effective weighting depends on the environment count. Taking only
the first occurrence of each task would be cleaner and would make a sweep independent of
how many environments it ran with. The published numbers keep the pooled convention;
changing it would move every table slightly.

**Macro, not micro.** `ood_table.py` and the tables report the unweighted mean over axis
success rates. `eval_ood/all/*` in wandb is the pooled (micro) mean, which sits 1-2 points
lower; `eval_ood/all/*_macro` was added later and is the one that matches the paper.

## Training-curve figures

`fetch_wandb_curves.py` caches per-epoch training metrics from wandb into
`vla_augm/results/wandb_curves/*.tsv` (reward, IND success, and the PPO loss terms), so
`make_training_figures.py` builds offline and a rebuild later reproduces the same figure
even if a run is deleted. Fetch once, then:

```
python vla_augm/analyze/fetch_wandb_curves.py      # ~20 s per run, skips cached ones
python vla_augm/analyze/make_training_figures.py all
```

| figure | what it shows |
|---|---|
| `dynamics` | reward, IND success and four PPO diagnostics per placement -- the mechanism behind the actor-augmentation collapse |
| `oodcurves` | macro OOD against epoch for every critic-only configuration, by type and by strength |
| `tradeoff` | IND against OOD per configuration, cropped to the runs that trained |
