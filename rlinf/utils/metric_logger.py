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

import os

from omegaconf import DictConfig, OmegaConf


class _TensorboardLogger:
    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class _WandbLogger:
    """Adapter around the ``wandb`` module with two resume-safe behaviours.

    1. **Resume into the same cloud run.** When ``run_id`` is supplied (recovered
       from the checkpoint directory we are resuming from) it is passed to
       ``wandb.init(id=..., resume="allow")`` so the resumed process appends to
       the existing run rather than starting a new one -- one link per
       experiment, even across many restarts.

    2. **Step overwrite instead of drop.** ``global_step`` is registered as the
       custom x-axis for every metric and injected into every payload. wandb's
       default monotonic-step guard would otherwise *drop* points re-logged
       after resuming from an older checkpoint (e.g. resume from step 160 after
       a walltime kill at step 173: steps 161-173 are retrained and re-logged).
       Driving the x-axis via a ``step_metric`` routes those points through
       wandb's last-write-wins path on the server, so the retrained values
       overwrite the pre-crash ones and the curve stays contiguous.
    """

    _RUN_ID_FILENAME = "wandb_run_id.txt"

    def __init__(
        self,
        *,
        entity,
        project,
        name,
        config,
        settings,
        dir,
        run_id=None,
    ) -> None:
        import wandb

        if run_id:
            # Resume an existing run. ``reinit=True`` is intentionally omitted:
            # combined with an explicit id it can finalise the resumed run.
            wandb.init(
                entity=entity,
                project=project,
                name=name,
                config=config,
                settings=settings,
                dir=dir,
                id=run_id,
                resume="allow",
            )
        else:
            wandb.init(
                entity=entity,
                project=project,
                name=name,
                config=config,
                settings=settings,
                dir=dir,
                reinit=True,
            )

        # Register global_step as the x-axis for every metric. Do this on every
        # process (including resume): define_metric is idempotent and per-run
        # config is only persisted once it has been called locally.
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        self._wandb = wandb

    @property
    def run_id(self) -> str | None:
        run = getattr(self._wandb, "run", None)
        return getattr(run, "id", None)

    def log(self, data: dict, step: int) -> None:
        payload = dict(data)
        payload.setdefault("global_step", int(step))
        # Do not pass ``step=``: with a step_metric registered, the x-axis comes
        # from the global_step field and the internal monotonic counter is left
        # to auto-increment (so backwards global_step values overwrite instead
        # of being dropped).
        self._wandb.log(payload)

    def log_table(self, df_data, name: str, step: int) -> None:
        table = self._wandb.Table(dataframe=df_data)
        self._wandb.log({name: table, "global_step": int(step)})

    def finish(self) -> None:
        self._wandb.finish()


def resolve_auto_resume_dir(cfg: DictConfig) -> str | None:
    """Resolve ``runner.resume_dir: auto`` to the latest ``global_step_N`` dir.

    Scans every checkpoint root the runners write to and returns the
    highest-step checkpoint directory found, or ``None`` when no checkpoint
    exists yet. Covers both layouts in the codebase:

    * embodied / offline / sft runners write under
      ``<checkpoint_path|log_path>/<experiment_name>/checkpoints``;
    * the reasoning runner writes under ``<log_path>/checkpoints``.

    Centralising this here means :class:`MetricLogger` recovers the wandb run
    id for an ``auto`` resume at construction time (before the runner runs its
    own resolution), and runners can reuse the same logic instead of each
    keeping a copy.
    """
    log_path = str(cfg.runner.logger.get("log_path", "logs"))
    exp = str(cfg.runner.logger.get("experiment_name", "default"))
    candidates = []
    ckpt_root = cfg.runner.get("checkpoint_path", None)
    if ckpt_root:
        candidates.append(os.path.join(str(ckpt_root), exp, "checkpoints"))
    candidates.append(os.path.join(log_path, exp, "checkpoints"))
    candidates.append(os.path.join(log_path, "checkpoints"))

    best_step, best_dir = -1, None
    for checkpoints_dir in candidates:
        if not os.path.isdir(checkpoints_dir):
            continue
        for name in os.listdir(checkpoints_dir):
            if not name.startswith("global_step_"):
                continue
            full = os.path.join(checkpoints_dir, name)
            if not os.path.isdir(full):
                continue
            try:
                step = int(name.split("global_step_")[-1])
            except ValueError:
                continue
            if step > best_step:
                best_step, best_dir = step, full
    return best_dir


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger_cfg = cfg.runner.logger

        self.log_path = logger_cfg.get("log_path", "logs")
        self.project_name = logger_cfg.get("project_name", "rlinf")
        self.experiment_name = logger_cfg.get("experiment_name", "default")
        self.per_worker_log = bool(cfg.runner.get("per_worker_log", False))
        self.per_worker_log_root = cfg.runner.get(
            "per_worker_log_path", os.path.join(self.log_path, "worker_logs")
        )

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        self.wandb_proxy = logger_cfg.get("wandb_proxy", None)
        self.wandb_entity = logger_cfg.get("wandb_entity", None)
        self.swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._all_loggers = []
        self._worker_loggers: dict[tuple[str, int], dict] = {}
        # When resuming, reattach to the same wandb cloud run that wrote the
        # checkpoint. May be None (fresh run, or old checkpoint written before
        # this feature existed) in which case wandb starts a new run.
        self._resume_wandb_id = self._recover_wandb_run_id(cfg)
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )

    @staticmethod
    def _recover_wandb_run_id(cfg: DictConfig) -> str | None:
        """Read the wandb run id persisted next to the resume checkpoint.

        Looks first inside ``runner.resume_dir`` (the ``global_step_N`` dir) and
        then in its parent (the ``checkpoints`` dir), which is where the id is
        snapshotted every step. ``resume_dir == "auto"`` is resolved first via
        :func:`resolve_auto_resume_dir` so the id is recovered regardless of how
        the resume was requested.
        """
        resume_dir = cfg.runner.get("resume_dir") if cfg.runner else None
        if not resume_dir:
            return None
        if str(resume_dir) == "auto":
            resume_dir = resolve_auto_resume_dir(cfg)
            if not resume_dir:
                return None
        resume_dir = str(resume_dir).rstrip(os.sep)
        candidates = [
            os.path.join(resume_dir, _WandbLogger._RUN_ID_FILENAME),
            os.path.join(os.path.dirname(resume_dir), _WandbLogger._RUN_ID_FILENAME),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        run_id = f.read().strip()
                    if run_id:
                        return run_id
                except OSError:
                    return None
        return None

    def _wandb_run_id_snapshot_path(self) -> str:
        """Stable path next to the checkpoints where the run id is cached.

        Uses the same ``checkpoint_path``/``experiment_name`` layout as the
        runners' ``_save_checkpoint``; falls back to the log path when no
        checkpoint path is configured. The parent ``checkpoints`` directory is
        never pruned (only ``global_step_*`` subdirs are), so a value written
        here survives until the next resume.
        """
        ckpt_root = self.cfg.runner.get("checkpoint_path", None) or self.log_path
        return os.path.join(
            str(ckpt_root),
            str(self.cfg.runner.logger.get("experiment_name", "default")),
            "checkpoints",
            _WandbLogger._RUN_ID_FILENAME,
        )

    def _maybe_snapshot_wandb_run_id(self) -> None:
        """Persist the current wandb run id next to the checkpoints.

        Called on every log so the ordering between ``_save_checkpoint`` and
        ``log`` within a step does not matter: as soon as the ``checkpoints``
        directory has been created by the first save, the id is (re)written here
        on every subsequent step. The write is idempotent.
        """
        wandb_logger = self.logger.get("wandb")
        if wandb_logger is None:
            return
        run_id = wandb_logger.run_id
        if run_id is None:
            return
        path = self._wandb_run_id_snapshot_path()
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            return
        try:
            with open(path, "w") as f:
                f.write(run_id)
        except OSError:
            pass

    def _create_logger_bundle(
        self, log_path: str, experiment_name: str, log_path_suffix: str = ""
    ) -> dict:
        logger = {}
        if "wandb" in self.logger_backends:
            wandb_log_path = os.path.join(log_path, "wandb", log_path_suffix)
            os.makedirs(wandb_log_path, exist_ok=True)

            settings = None
            if self.wandb_proxy:
                import wandb

                settings = wandb.Settings(https_proxy=self.wandb_proxy)
            logger["wandb"] = _WandbLogger(
                entity=self.wandb_entity,
                project=self.project_name,
                name=experiment_name,
                config=self.config,
                settings=settings,
                dir=wandb_log_path,
                run_id=self._resume_wandb_id,
            )

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = os.path.join(log_path, "swanlab", log_path_suffix)
            os.makedirs(swanlab_log_path, exist_ok=True)

            swanlab.init(
                project=self.project_name,
                experiment_name=experiment_name,
                config=self.config,
                logdir=swanlab_log_path,
                mode=self.swanlab_mode,
            )
            logger["swanlab"] = swanlab

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = os.path.join(
                log_path, "tensorboard", log_path_suffix
            )
            os.makedirs(tensorboard_log_path, exist_ok=True)

            config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
            OmegaConf.save(self.cfg, config_yaml_path, resolve=True)

            logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)
        self._all_loggers.append(logger)
        return logger

    def _get_scoped_logger(self, worker_group_name: str, rank: int) -> dict:
        key = (worker_group_name, int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            worker_group_name,
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data,
        step,
        backend=None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ):
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        for default_backend, logger_instance in target_logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)
        if "wandb" in target_logger:
            self._maybe_snapshot_wandb_run_id()

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            self.logger["wandb"].log_table(df_data=df_data, name=name, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        self.finish()

    def finish(self):
        for logger in self._all_loggers:
            for logger_instance in logger.values():
                logger_instance.finish()
