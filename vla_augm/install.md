# Installation 

To keep track of what I've done.

```
salloc -p gpu_h100 --gpus-per-node=1 -t 4:00:00      # optional, on gpu
bash requirements/install.sh embodied --model openpi --env libero --venv .venv-pi --no-root
source .venv-pi/bin/activate
```


## Quick start
Trying: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html

```
pip install huggingface-hub
hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

# pi0.5 SFT checkpoint (7.0 GB), for run_pi05_4gpu.sh
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir models/RLinf-Pi05-LIBERO-SFT

# GR00T N1.5 SFT checkpoint (3B params), for run_gr00t_4gpu.sh and the
# GR00T augmentation arms. The gr00t configs set model_path to a local copy of
# this, so download it before running them.
hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir models/RLinf-Gr00t-SFT-Spatial
```

Try things in interactive mode:
```
salloc -p gpu_h100 --gpus-per-node=1 -t 4:00:00 
source .venv-pi/bin/activate
```

To run exp:
```
sbatch vla_augm/scripts/cluster_a/run_pi0_1gpu.sh     # pi0, LIBERO-Spatial
sbatch vla_augm/scripts/cluster_a/run_pi05_4gpu.sh    # pi0.5, LIBERO-Spatial
```


## LIBERO-Plus (OOD generalization eval)

Layers the LIBERO-Plus perturbation suite (10,030 tasks, 7 OOD axes: camera,
lighting, textures, distractors, robot init, language, sensor noise) onto the
existing `.venv-pi` so pi0 checkpoints can be evaluated on it. Installed 2026-07-20.

### One-time install (run from repo root, on any node — no GPU needed)

```
source .venv-pi/bin/activate

# (safety) snapshot the working env first, so we can roll back if needed
pip freeze > vla_augm/venv-pi-freeze-preplus-$(date +%Y%m%d).txt

# clone the RLinf LIBERO-plus fork into the venv dir (matches install.sh layout)
git clone --depth 1 https://github.com/RLinf/LIBERO-plus.git .venv-pi/libero_plus

# python deps — additive only (wand, scikit-image); mujoco pin is a no-op (already 3.8.1)
uv pip install -r .venv-pi/libero_plus/extra_requirements.txt
uv pip install -e .venv-pi/libero_plus
uv pip install "mujoco<=3.9.0"

# download + extract the 6.4 GB asset pack into the liberoplus package dir, then drop the zip
PKG=$(python -c "import pathlib, liberoplus.liberoplus as l; print(pathlib.Path(l.__file__).resolve().parent)")
hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "$PKG"
unzip -oq "$PKG/assets.zip" -d "$PKG"
rm -f "$PKG/assets.zip"

# IMPORTANT: this asset zip is mis-packed with a baked-in absolute path, so it
# lands under inspire/hdd/project/.../LIBERO-plus-0/assets instead of $PKG/assets
# (which is where the code looks: __init__.py -> benchmark_root_path/assets).
# Move it to the expected location (fast same-fs rename) and drop the wrapper:
mv "$PKG"/inspire/hdd/project/*/public/*/libero_new/release/dataset/LIBERO-plus-0/assets "$PKG/assets"
rm -rf "$PKG/inspire"
# sanity: should list articulated_objects/ new_objects/ scenes/ textures/ ...
ls "$PKG/assets"
```

Footprint: ~8.7 GB and ~480k files in `.venv-pi/libero_plus` (the asset pack is
mostly small files — it raised project-space **inode** usage from ~25% to ~72%,
so inodes are now the tighter quota than bytes; check with `myquota PROJECT_ID`).

### Runtime deps — REQUIRED every time you run LIBERO-Plus

`python-wand` needs the ImageMagick shared library, which lives in an env module
(not the venv). Put this in any eval sbatch script, after `source .venv-pi/bin/activate`:

```
module load 2025
module load ImageMagick/7.1.1-47-GCCcore-14.2.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"   # so wand finds libMagickWand-7.Q16HDRI.so

# select the LIBERO-Plus variant
export LIBERO_TYPE=plus
export LIBERO_SUFFIX=all
```

Sanity check the install:

```
python -c "from liberoplus.liberoplus.benchmark import Benchmark; print('liberoplus OK')"
```

### Periodic OOD eval during training (`eval_ood/*`)

Training runs can evaluate on LIBERO-Plus every `runner.val_ood_check_interval`
steps while still training on standard LIBERO. Opt in from a config with:

```yaml
defaults:
  - env/libero_plus_ood@env.eval_ood   # examples/embodiment/config/env/
...
runner:
  val_ood_check_interval: 20
env:
  eval_ood:
    task_suite_name: libero_spatial
```

Already enabled in `libero_spatial_ppo_pi05_4gpu` and `libero_spatial_ppo_pi0_4gpu`.

Why a separate worker group: `rlinf/envs/libero/libero_env.py` resolves
`LIBERO_TYPE` at *module import* and it decides which `Benchmark` class is
imported, so one process cannot host standard and plus envs. The runner launches
an extra `EnvOODGroup` whose workers get `LIBERO_TYPE=plus` via
`env.eval_ood.env_vars` (plumbed through `WorkerGroup.launch(extra_env_vars=...)`).
A single-variant job is not an option either: the plus suite shares zero task
names with standard `libero_spatial`.

Axes evaluated are textures / lighting / layout (935 tasks for `libero_spatial`).
The other four LIBERO-Plus axes score ~0% for pi0-class policies, so they have no
dynamic range; see the table in `config/env/libero_plus_ood.yaml`.

The launch script must load ImageMagick **and** slim the environment (see the
E2BIG note in `eval_pi0_spatial_plus.sh`) — both are already in
`run_pi05_4gpu.sh` and `run_pi0_4gpu.sh`.

Outputs:
- wandb: `eval_ood/{textures,lighting,layout,all}/success_once` etc.
- per-episode JSONL: `logs/<run>/ood_eval/step_<N>.jsonl` (~100 KB per point)
- cost: read `time/eval_ood` from wandb

Analyse the JSONL (per-axis rates, SEs, and a two-run diff with sigmas):

```
python vla_augm/utils/analyze_ood_eval.py logs/<run>/ood_eval
python vla_augm/utils/analyze_ood_eval.py logs/<runA>/ood_eval logs/<runB>/ood_eval --labels SFT,RL
```

### Notes

- Selected via `LIBERO_TYPE=plus` (env var) or `env.eval.libero_variant: plus` (config);
  `standard`/`pro`/`plus` all supported by `rlinf/envs/libero/`.
  Note the env var **wins** over the config key (`libero_env.py` `get_env_fn_params`),
  which is why the OOD group sets the env var rather than only `libero_variant`.
- No pi0/OpenPI LIBERO-Plus config ships yet — reuse a pi0 libero eval config
  (e.g. `evaluations/libero/libero_goal_openpi_eval.yaml`) with the env vars above.
  Confirm which base suites the fork perturbs before comparing to Spatial/Object/Goal runs.
- Rollback if the env ever breaks: `uv pip install -r vla_augm/venv-pi-freeze-preplus-<date>.txt`.

### On Cluster B

Same layering onto `.venv-pi`, scripted for Cluster B (glibc 2.28, no CPU
allocation, module-based ImageMagick). Install + functional test in one job:

```
sbatch --qos=dev --reservation=gpudev --time=01:00:00 \
    vla_augm/scripts/cluster_b/install_libero_plus.sh
```

It clones the fork into `.venv-pi/libero_plus`, installs `wand`+`scikit-image`,
downloads/extracts the 6.4 GB asset pack to the correct package dir, and runs an
import sanity check. Then verify perturbed-scene rendering on a GPU:

```
srun -A PROJECT_ID -p gpu -q dev --reservation=gpudev -t 00:15:00 -N1 -n1 -c8 --gpus=1 \
    bash -l vla_augm/scripts/cluster_b/test_libero_plus.sh   # -> LIBERO_PLUS_RENDER_OK, 2402 tasks
```

Runtime deps for any LIBERO-Plus job (analogous to the Cluster A block above):
`module load env/release/2024.1 ImageMagick/7.1.1-38-GCCcore-13.3.0`,
`export MAGICK_HOME="$EBROOTIMAGEMAGICK"`, `LIBERO_TYPE=plus LIBERO_SUFFIX=all`,
plus the E2BIG env-slim from `eval_pi0_spatial_plus.sh` for Ray-worker jobs.

Footprint note: the asset pack pushes `/project/scratch/PROJECT_ID` **inode** usage
to ~88% (check `myquota`) — inodes are the tight quota, not bytes.

## Testing on Mac

Can't build the full `.venv-pi` on Mac (pi0 stack is Linux/CUDA only). But
CPU-only tests like `vla_augm/test_instruction_overlay.py` run with a minimal venv.

installing the .venv-pi-mac:
```
uv venv .venv-pi-mac --python 3.11
source .venv-pi-mac/bin/activate
uv pip install numpy torch pillow "omegaconf<2.4" imageio imageio-ffmpeg
```

running a test (from the repo root; PYTHONPATH=. so `rlinf` is importable):
```
source .venv-pi-mac/bin/activate
PYTHONPATH=. python vla_augm/test_instruction_overlay.py \
    --input vla_augm/42.mp4 \
    --output vla_augm/overlay_preview10.mp4 \
    --font-size 10
```

## Rendering LIBERO-Plus locally on Mac (no GPU, no VLA)

Layers the LIBERO-Plus suite onto `.venv-pi-mac` so the perturbed environments can
be *rendered* on the laptop and recorded to video with a random policy. This is
for looking at scenes -- paper figures, sanity-checking that a perturbation is
actually applied -- not for evaluation: no policy runs, so no success rate this
produces means anything. Installed 2026-08-25 on an M3.

It works because MuJoCo has a native offscreen GL backend on macOS (`MUJOCO_GL=cgl`),
so nothing here needs EGL, Mesa or a GPU. Rendering runs at roughly 5 env steps/s
at 256x256 with two cameras, i.e. ~25 s for a 120-step clip.

### One-time install (from the repo root)

```
source .venv-pi-mac/bin/activate
uv pip freeze > vla_augm/venv-pi-mac-freeze-prelibero-$(date +%Y%m%d).txt   # rollback point

# versions match vla_augm/venv-gr00t-freeze-augm-20260808.txt, i.e. what the cluster
# actually ran -- including numpy 2.x, so no downgrade is needed
uv pip install "mujoco==3.3.7" "robosuite==1.4.1" "bddl==3.6.0" "easydict==1.13" \
    "termcolor==3.3.0" "cloudpickle==3.1.2" "h5py==3.14.0" "scipy==1.17.1" \
    "scikit-image==0.26.0" "matplotlib==3.11.1" "tqdm==4.69.0" "Wand==0.7.2" \
    "opencv-python==5.0.0.93" "gym==0.26.2" huggingface-hub

git clone --depth 1 https://github.com/RLinf/LIBERO-plus.git .venv-pi-mac/libero_plus
uv pip install -e .venv-pi-mac/libero_plus

# same two numpy-2 removals the cluster patches (install_venv_gr00t.sh); note the
# BSD sed `-i ''`, GNU sed's bare `-i` is a syntax error here
PLUS_PY=.venv-pi-mac/libero_plus/liberoplus/liberoplus/envs/env_wrapper.py
sed -i '' -e "s/np\.fromstring(/np.frombuffer(/g" -e "s/np\.float_/np.float64/g" "$PLUS_PY"

brew install imagemagick    # python-wand needs libMagickWand

# 6.0 GB asset pack -> 9.4 GB extracted
PKG="$PWD/.venv-pi-mac/libero_plus/liberoplus/liberoplus"
hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "$PKG"
unzip -oq "$PKG/assets.zip" -d "$PKG" && rm -f "$PKG/assets.zip"

# the zip is mis-packed with a baked-in absolute path (same as on the clusters):
# it lands under inspire/hdd/project/... instead of $PKG/assets, where the code looks
mv "$PKG"/inspire/hdd/project/*/public/*/libero_new/release/dataset/LIBERO-plus-0/assets "$PKG/assets"
rm -rf "$PKG/inspire"
ls "$PKG/assets"   # -> articulated_objects/ new_objects/ scenes/ textures/ ...
```

### Runtime

Only `MAGICK_HOME` has to be exported; `record_libero_videos.py` sets the rest itself.

```
source .venv-pi-mac/bin/activate
export MAGICK_HOME="$(brew --prefix imagemagick)"

python vla_augm/analyze/record_libero_videos.py --list
python vla_augm/analyze/record_libero_videos.py --figure-set     # 5 axes + matching IND
python vla_augm/analyze/record_libero_videos.py --axis noise --n 3
```

Clips are 512x256 (main | wrist), named like the cluster's
(`<axis>_base<N>_task<ID>_<ok|fail>.mp4`), so `make_env_figure.py` reads them
unchanged -- except that these carry **no HUD**, so they need no crop.

### Two traps, both macOS-specific

**`MAGICK_THREAD_LIMIT=1` is required, not an optimisation.** Homebrew's
ImageMagick 7 is OpenMP-built and torch/numba load their own OpenMP runtime; two
in one process segfault the interpreter -- but only once a MuJoCo GL context
exists, so `motion_blur` tested on its own looks perfectly fine. It crashed 3/3
without and 0/3 with (`OMP_NUM_THREADS=1` works equally well). This bites only on
the sensor-noise axis, the one axis that calls ImageMagick. The script exports it
at import time, before wand loads the library, which is the only moment
ImageMagick reads it.

**A missing `MAGICK_HOME` fails at import, not at render.** python-wand loads
libMagickWand when the module is imported, so forgetting it kills the run before
any env is built, with `ImportError: MagickWand shared library not found`.
