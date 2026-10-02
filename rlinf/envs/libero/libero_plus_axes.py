# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map LIBERO-Plus task ids to their OOD perturbation axis.

LIBERO-Plus ships ``benchmark/task_classification.json``, which labels every
task with one of seven perturbation categories. This module reads that file
directly (never importing ``liberoplus``, whose package import pulls in
ImageMagick via ``wand``) so the training driver -- which runs with
``LIBERO_TYPE=standard`` -- can still resolve the axis of an eval task id.

Task ids are 0-based positions in the suite's task list, matching
``task_id`` as reported by :class:`~rlinf.envs.libero.libero_env.LiberoEnv`.
The JSON's ``id`` field is 1-based, hence the ``id - 1`` conversion.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache

# Short config-facing names -> the categories used in task_classification.json.
AXIS_CATEGORIES: dict[str, str] = {
    "textures": "Background Textures",
    "lighting": "Light Conditions",
    "layout": "Objects Layout",
    "camera": "Camera Viewpoints",
    "noise": "Sensor Noise",
    "robot_init": "Robot Initial States",
    "language": "Language Instructions",
}

# Fixed seed for axis subsampling. Deliberately NOT the run seed: every run and
# every arm must evaluate the identical subset, or the OOD curves are not
# comparable. Changing this invalidates comparisons against earlier runs.
SUBSAMPLE_SEED = 20260730

_CLASSIFICATION_RELPATH = os.path.join(
    "liberoplus", "liberoplus", "benchmark", "task_classification.json"
)


def _candidate_paths() -> list[str]:
    """Search locations for task_classification.json, most specific first."""
    candidates = []
    explicit = os.environ.get("LIBERO_PLUS_TASK_CLASSIFICATION", None)
    if explicit:
        candidates.append(explicit)

    # Installed alongside the repo's venv (see vla_augm/install.md), or anywhere
    # LIBERO_PLUS_ROOT points at.
    roots = [os.environ.get("LIBERO_PLUS_ROOT", None), os.environ.get("REPO_PATH", None)]
    for root in roots:
        if not root:
            continue
        candidates.append(os.path.join(root, _CLASSIFICATION_RELPATH))
        candidates.append(os.path.join(root, "venv-pi", "libero_plus", _CLASSIFICATION_RELPATH))
    return candidates


def resolve_classification_path(path: str | None = None) -> str:
    """Return a readable path to task_classification.json.

    Args:
        path: Explicit path from config. Checked first when given.

    Raises:
        FileNotFoundError: If no candidate path exists.
    """
    candidates = ([path] if path else []) + _candidate_paths()
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    # Last resort: locate the installed liberoplus package without importing it.
    try:
        import importlib.util

        spec = importlib.util.find_spec("liberoplus")
        if spec is not None and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                candidate = os.path.join(
                    location, "liberoplus", "benchmark", "task_classification.json"
                )
                if os.path.isfile(candidate):
                    return candidate
    except Exception:  # noqa: BLE001 - discovery is best-effort
        pass

    raise FileNotFoundError(
        "Could not locate LIBERO-Plus task_classification.json. Set "
        "env.eval_ood.task_classification_path in the config, or the "
        "LIBERO_PLUS_TASK_CLASSIFICATION environment variable. Tried: "
        f"{[c for c in candidates if c]}"
    )


@lru_cache(maxsize=8)
def _load_classification(path: str, suite_name: str) -> tuple[tuple[int, str], ...]:
    with open(path) as f:
        data = json.load(f)
    if suite_name not in data:
        raise KeyError(
            f"Suite {suite_name!r} not in {path}. Available: {sorted(data)}"
        )
    return tuple((int(t["id"]) - 1, str(t["category"])) for t in data[suite_name])


def get_axis_task_ids(
    suite_name: str,
    axes: list[str],
    path: str | None = None,
    subsample_frac: float = 1.0,
) -> dict[str, list[int]]:
    """Return ``{axis_name: sorted task ids}`` for the requested axes.

    Args:
        suite_name: e.g. ``"libero_spatial"``.
        axes: Short axis names, keys of :data:`AXIS_CATEGORIES`.
        path: Optional explicit path to task_classification.json.
        subsample_frac: Keep this fraction of each axis, drawn once with
            :data:`SUBSAMPLE_SEED`. Five axes at full size is ~1700 tasks per
            eval; a quarter of each keeps the periodic eval affordable while
            still covering every axis. Sampling per axis rather than over the
            pooled set keeps the axis proportions intact. Use 1.0 for the
            final full-suite eval.

    Raises:
        ValueError: On an unknown axis, an axis that matches no tasks, or a
            fraction outside (0, 1].
    """
    if not 0.0 < subsample_frac <= 1.0:
        raise ValueError(f"subsample_frac must be in (0, 1], got {subsample_frac}")
    resolved = resolve_classification_path(path)
    entries = _load_classification(resolved, suite_name)

    unknown = [a for a in axes if a not in AXIS_CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown OOD axes {unknown}. Valid: {sorted(AXIS_CATEGORIES)}"
        )

    wanted = {AXIS_CATEGORIES[a]: a for a in axes}
    result: dict[str, list[int]] = {a: [] for a in axes}
    for task_id, category in entries:
        axis = wanted.get(category)
        if axis is not None:
            result[axis].append(task_id)
    for axis, ids in result.items():
        if not ids:
            raise ValueError(
                f"Axis {axis!r} matched no tasks in suite {suite_name!r}; "
                "the classification file may be for a different LIBERO-Plus release."
            )
        ids.sort()
        if subsample_frac < 1.0:
            keep = max(1, round(len(ids) * subsample_frac))
            # Seeded from the axis name as well, so adding an axis does not
            # reshuffle the subsets already chosen for the others.
            rng = random.Random(f"{SUBSAMPLE_SEED}:{suite_name}:{axis}")
            result[axis] = sorted(rng.sample(ids, keep))
    return result


def build_task_id_to_axis(
    suite_name: str,
    axes: list[str],
    path: str | None = None,
    subsample_frac: float = 1.0,
) -> dict[int, str]:
    """Inverse of :func:`get_axis_task_ids`, for labelling eval episodes."""
    return {
        task_id: axis
        for axis, ids in get_axis_task_ids(
            suite_name, axes, path, subsample_frac
        ).items()
        for task_id in ids
    }
