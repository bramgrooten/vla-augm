# vla_augm/utils

Standalone scripts. All are run from the repo root, e.g.
`python vla_augm/utils/check_ood_configs.py`.

## Pre-launch checks

Run these before submitting a run; each exits non-zero on a problem.

| script | checks |
|---|---|
| `check_ood_configs.py` | every config's OOD eval can cover its task pool, and the OOD horizon matches the suite eval horizon |
| `check_arm_configs.py` | the Exp 0(b) arms differ from the baseline *only* in `algorithm.augmentation.placement` |
| `check_comment_refs.py` | config/script headers do not name the wrong model or the wrong script (copy-paste rot) |

## Tools

| script | does |
|---|---|
| `build_overlay_bank.py` | builds an overlay image bank from COCO or the ImageNet Winter21 per-class tars |
| `preview_augmentation.py` | renders real eval frames through the training augmentation path, since rollouts are never augmented and no augmented video exists |
| `analyze_ood_eval.py` | per-axis OOD success rates with standard errors; compares two runs |
| `parse_libero_eval_log.py` | converts a standalone `only_eval` console log into the same JSONL `analyze_ood_eval.py` reads |
