#!/bin/bash -l
# Functional test: render a perturbed LIBERO-Plus scene end-to-end on a GPU
# (assets + bddl + EGL), the way an OOD eval would. Run on a GPU node:
#   srun -A PROJECT_ID -p gpu -q dev --reservation=gpudev -t 00:15:00 -N1 -n1 -c8 --gpus=1 \
#     bash -l vla_augm/scripts/cluster_b/_test_libero_plus_render.sh
set -e
REPO="/project/scratch/PROJECT_ID/vla_augm/rl_inf"
cd "$REPO"
module load env/release/2024.1
module load Mesa ImageMagick/7.1.1-38-GCCcore-13.3.0
export MAGICK_HOME="$EBROOTIMAGEMAGICK"
source .venv-pi/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=plus LIBERO_SUFFIX=all
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
# Resolve assets the way the run scripts do: squashfs image -> node-local tmpfs.
source "${REPO}/vla_augm/scripts/cluster_b/libero_plus_local.sh"
python - <<'PY'
import numpy as np
from liberoplus.liberoplus.benchmark import get_benchmark
from liberoplus.liberoplus.envs import OffScreenRenderEnv

# Perturbed libero_spatial suite (LIBERO_TYPE=plus selects the perturbed assets).
BenchCls = None
for name in ("libero_spatial", "LIBERO_SPATIAL"):
    try:
        BenchCls = get_benchmark(name); break
    except Exception as e:
        last = e
assert BenchCls is not None, f"get_benchmark failed: {last}"
bench = BenchCls()
n = bench.get_num_tasks()
task = bench.get_task(0)
bddl = bench.get_task_bddl_file_path(0)
init_states = bench.get_task_init_states(0)
print(f"benchmark OK: {n} perturbed tasks; task0='{getattr(task,'language',task)}'")

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
r = env.reset()
env.set_init_state(init_states[0])
step = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
obs = step[0]
img = obs["agentview_image"]
assert img.shape[0] == 128 and img.ndim == 3, img.shape
print(f"EGL render of perturbed scene OK: agentview_image shape={img.shape}, dtype={img.dtype}")
env.close()
print("LIBERO_PLUS_RENDER_OK")
PY
