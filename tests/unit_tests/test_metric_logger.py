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

"""wandb resume + step-overwrite behaviour for MetricLogger.

Pins the three things that make a resumed run append to the same cloud link:

* on a fresh run, ``wandb.init`` is called with ``reinit=True`` and no id;
* on resume, the id recovered from the checkpoint dir is passed with
  ``resume="allow"`` (and ``reinit`` is dropped);
* ``global_step`` is registered as the custom x-axis and injected into every
  payload, so re-logged steps overwrite on the server instead of being dropped
  by wandb's monotonic-step guard;
* the run id is snapshotted next to the checkpoints so the next resume finds it.
"""

import sys

import pytest
from omegaconf import OmegaConf

from rlinf.utils.metric_logger import (
    MetricLogger,
    _WandbLogger,
    resolve_auto_resume_dir,
)


class _FakeRun:
    def __init__(self, run_id="fake-run-id"):
        self.id = run_id


class _FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeWandb:
    """A module-shaped recorder: assigned to ``sys.modules["wandb"]``."""

    def __init__(self, run_id="fake-run-id"):
        self.init_calls = []
        self.define_metric_calls = []
        self.log_calls = []
        self.Settings = _FakeSettings
        self.Table = _Table
        self._run = _FakeRun(run_id)

    @property
    def run(self):
        return self._run

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self._run

    def define_metric(self, name, *args, **kwargs):
        self.define_metric_calls.append((name, kwargs))

    def log(self, payload, *args, **kwargs):
        self.log_calls.append((dict(payload), kwargs))

    def finish(self):
        pass


class _Table:
    def __init__(self, dataframe=None, data=None, **kwargs):
        self.dataframe = dataframe
        self.data = data


def _install_fake_wandb(run_id="fake-run-id"):
    """Install a fake ``wandb`` module that records every call.

    Returns the recorder; it is also reachable as ``sys.modules["wandb"]`` so
    ``import wandb`` inside the adapter picks it up.
    """
    recorder = _FakeWandb(run_id=run_id)
    sys.modules["wandb"] = recorder
    return recorder


@pytest.fixture(autouse=True)
def _clean_wandb_module():
    yield
    sys.modules.pop("wandb", None)


def _make_cfg(tmp_path, resume_dir=None, checkpoint_path=None):
    return OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path / "logs"),
                    "project_name": "rl_inf",
                    "experiment_name": "exp",
                    "logger_backends": ["wandb"],
                },
                "checkpoint_path": str(checkpoint_path or tmp_path / "ckpts"),
                "resume_dir": str(resume_dir) if resume_dir else None,
            }
        }
    )


def test_fresh_run_uses_reinit_and_registers_global_step_axis(tmp_path):
    fake = _install_fake_wandb()
    cfg = _make_cfg(tmp_path)
    MetricLogger(cfg)

    assert len(fake.init_calls) == 1
    kwargs = fake.init_calls[0]
    assert kwargs.get("reinit") is True
    assert "id" not in kwargs and "resume" not in kwargs

    # global_step is registered as the x-axis for every metric.
    names = [name for name, _ in fake.define_metric_calls]
    assert "global_step" in names
    assert ("*", {"step_metric": "global_step"}) in fake.define_metric_calls


def test_resume_passes_recovered_id_with_allow(tmp_path):
    resume_dir = tmp_path / "global_step_160"
    resume_dir.mkdir()
    (resume_dir / _WandbLogger._RUN_ID_FILENAME).write_text("oxknjtf2")

    fake = _install_fake_wandb()
    cfg = _make_cfg(tmp_path, resume_dir=str(resume_dir))
    MetricLogger(cfg)

    kwargs = fake.init_calls[0]
    assert kwargs.get("id") == "oxknjtf2"
    assert kwargs.get("resume") == "allow"
    assert "reinit" not in kwargs  # reinit must be dropped on resume


def test_recover_run_id_falls_back_to_parent_dir(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / _WandbLogger._RUN_ID_FILENAME).write_text("parent-id")
    resume_dir = checkpoints / "global_step_40"
    resume_dir.mkdir()

    cfg = _make_cfg(tmp_path, resume_dir=str(resume_dir))
    # Build without touching wandb: recover directly via the staticmethod.
    rid = MetricLogger._recover_wandb_run_id(cfg)
    assert rid == "parent-id"


def test_log_injects_global_step_and_omits_wandb_step(tmp_path):
    fake = _install_fake_wandb()
    cfg = _make_cfg(tmp_path)
    logger = MetricLogger(cfg)
    logger.log({"train/loss": 0.1, "episode_len": 240.0}, step=173)

    payload, kwargs = fake.log_calls[-1]
    assert payload["global_step"] == 173
    assert payload["train/loss"] == 0.1
    # No step= forwarded: the step_metric drives the x-axis.
    assert "step" not in kwargs


def test_snapshot_writes_run_id_next_to_checkpoints(tmp_path):
    _install_fake_wandb(run_id="snap-id")
    checkpoint_path = tmp_path / "ckpts"
    cfg = _make_cfg(tmp_path, checkpoint_path=str(checkpoint_path))
    logger = MetricLogger(cfg)

    # Simulate the first checkpoint save creating the checkpoints dir, then log.
    checkpoints_dir = checkpoint_path / "exp" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    logger.log({"train/loss": 0.1}, step=40)

    rid_file = checkpoints_dir / _WandbLogger._RUN_ID_FILENAME
    assert rid_file.is_file()
    assert rid_file.read_text() == "snap-id"


def test_snapshot_is_noop_until_checkpoints_dir_exists(tmp_path):
    _install_fake_wandb(run_id="snap-id")
    cfg = _make_cfg(tmp_path)
    logger = MetricLogger(cfg)
    logger.log({"train/loss": 0.1}, step=1)  # no checkpoints dir yet

    rid_file = (
        tmp_path / "ckpts" / "exp" / "checkpoints" / _WandbLogger._RUN_ID_FILENAME
    )
    assert not rid_file.exists()


def test_resolve_auto_resume_dir_picks_latest_under_log_path(tmp_path):
    # reasoning-runner layout: <log_path>/checkpoints/global_step_N
    cfg = _make_cfg(tmp_path, checkpoint_path=None)
    checkpoints = tmp_path / "logs" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "global_step_40").mkdir()
    (checkpoints / "global_step_160").mkdir()
    (checkpoints / "global_step_80").mkdir()

    assert resolve_auto_resume_dir(cfg) == str(checkpoints / "global_step_160")


def test_resolve_auto_resume_dir_picks_latest_under_checkpoint_path(tmp_path):
    # embodied-runner layout: <checkpoint_path>/<exp>/checkpoints/global_step_N
    ckpt_root = tmp_path / "ckpts"
    cfg = _make_cfg(tmp_path, checkpoint_path=str(ckpt_root))
    checkpoints = ckpt_root / "exp" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "global_step_40").mkdir()

    assert resolve_auto_resume_dir(cfg) == str(checkpoints / "global_step_40")


def test_resolve_auto_resume_dir_returns_none_when_empty(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert resolve_auto_resume_dir(cfg) is None


def test_resume_with_auto_recovers_run_id(tmp_path):
    # resume_dir: auto must resolve and find the snapshotted wandb run id, so
    # an auto-resume reattaches to the same cloud run as an explicit one.
    ckpt_root = tmp_path / "ckpts"
    cfg_factory = lambda rd: OmegaConf.create(  # noqa: E731
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path / "logs"),
                    "project_name": "rl_inf",
                    "experiment_name": "exp",
                    "logger_backends": ["wandb"],
                },
                "checkpoint_path": str(ckpt_root),
                "resume_dir": rd,
            }
        }
    )

    # Seed a checkpoint at step 160 with a persisted run id at the checkpoints
    # root (where MetricLogger snapshots it).
    checkpoints = ckpt_root / "exp" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "global_step_160").mkdir()
    (checkpoints / _WandbLogger._RUN_ID_FILENAME).write_text("auto-resume-id")

    fake = _install_fake_wandb()
    MetricLogger(cfg_factory("auto"))

    kwargs = fake.init_calls[0]
    assert kwargs.get("id") == "auto-resume-id"
    assert kwargs.get("resume") == "allow"
    assert "reinit" not in kwargs
