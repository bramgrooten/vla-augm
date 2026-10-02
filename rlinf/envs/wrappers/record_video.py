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

import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import (
    put_info_on_image,
    put_text_on_image_bottom,
    tile_images,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render).
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        # State for the opt-in per-epoch video cap (see flush_video).
        self._video_epoch = None
        self._epoch_video_cnt = 0
        self.render_images: list[np.ndarray] = []
        # Clean counterpart, filled only when augmentation is on, so every
        # augmented video ships with the unaugmented view of the same episode.
        self.render_images_clean: list[np.ndarray] = []
        self.video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

        # Which camera(s) to record. Default is the single third-person view,
        # matching the previous behaviour; ["main", "wrist"] records both side
        # by side, which is what the policy actually consumes.
        self._views = list(self._cfg_get("views", ["main"]) or ["main"])
        self._frame_aug = self._build_frame_augmentation()

        # Running reward total per stream per env, for the "cumulative reward"
        # overlay (video_cfg.cumulative_reward_on_video).
        #
        # WHY IT IS KEYED BY STREAM. With augmentation on, add_new_frames draws
        # every frame TWICE -- once into render_images_clean, once into
        # render_images -- so a single accumulator would count each step twice
        # and the overlay would read 2.000 on success. Each buffer therefore
        # carries its own total and each advances exactly once per frame.
        self._cum_reward: dict[str, dict[int, float]] = {}

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _cfg_get(self, key: str, default=None):
        """Read a field from video_cfg, which may be a dict or an OmegaConf node."""
        if self.video_cfg is None:
            return default
        if hasattr(self.video_cfg, "get"):
            return self.video_cfg.get(key, default)
        return getattr(self.video_cfg, key, default)

    def _build_frame_augmentation(self):
        """Overlay augmentation for recorded frames, or None.

        Videos otherwise show the clean environment render, which is not what
        the policy sees when observation augmentation is on. Enabling this
        records the augmented view instead, for debugging.

        The overlays are drawn from the same bank with the same alpha as
        training, but this is an independent draw applied to the raw render --
        the same distribution as the policy's input, not the identical frame,
        which is applied after the resize to 224.
        """
        cfg = self._cfg_get("augmentation", None)
        if not cfg or not cfg.get("bank_path", None):
            return None
        try:
            import torch

            from rlinf.algorithms.augmentation import OverlayImageBank, RandomOverlay

            # Every env worker would otherwise hold a full copy of a bank that
            # can be tens of GB; take a slice, since a debug video does not
            # need full class coverage.
            shards = int(cfg.get("bank_shards", 32))
            bank = OverlayImageBank.from_cache(
                cfg["bank_path"], shard_index=0, num_shards=max(shards, 1)
            )
            generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
            from rlinf.algorithms.augmentation import AugmentationSequence

            return RandomOverlay(
                bank=bank,
                alpha=float(cfg.get("alpha", 0.5)),
                crop_scale=cfg.get("crop_scale", 0.3),
                sequence=AugmentationSequence(
                    str(cfg.get("sequence", "random-per-angle")).lower()
                ),
                # The recorder runs once per rendered frame, so a chunk spans
                # num_action_chunks frames here -- unlike the model path, where
                # one inference already covers a whole chunk.
                period=int(cfg.get("chunk_period", 5)),
                generator=generator,
            )
        except Exception as exc:  # noqa: BLE001 - a debug aid must not kill a run
            logger.warning(f"Could not build video augmentation ({exc}); recording clean frames.")
            return None

    def _augment_frames(self, frames: list) -> list:
        """Overlay each camera's [N, H, W, 3] uint8 batch, given as a list.

        Passing all views in one call lets the sequence policy share a single
        draw across cameras: `random` gives both angles the same overlay image
        applied separately to each, `random-per-angle` gives them different
        ones.
        """
        import torch

        arrays = [self._to_numpy(f) for f in frames]
        if any(a.ndim != 4 or a.shape[-1] != 3 for a in arrays):
            return frames
        tensors = [
            torch.from_numpy(a.astype("float32"))
            .div_(127.5)
            .sub_(1.0)
            .permute(0, 3, 1, 2)
            for a in arrays
        ]
        return [
            o.permute(0, 2, 3, 1)
            .add_(1.0)
            .mul_(127.5)
            .clamp_(0, 255)
            .to(torch.uint8)
            .numpy()
            for o in self._frame_aug(tensors)
        ]

    def _get_image_from_dict(self, obs: dict, augment: bool = True) -> Optional[Any]:
        """Pick the image field(s) from an observation dict.

        With ``views: ["main", "wrist"]`` the two camera batches are
        concatenated along width, so one video shows both angles.
        """
        if hasattr(self.env, "capture_image"):
            return self.env.capture_image()

        view_keys = {
            "main": ("main_images", "images", "rgb", "full_image", "main_image"),
            "wrist": ("wrist_images", "wrist_image"),
        }
        frames = []
        for view in self._views:
            for key in view_keys.get(view, ()):
                if key in obs and obs[key] is not None:
                    frames.append(self._to_numpy(obs[key]))
                    break

        if not frames:
            return None
        if augment and self._frame_aug is not None:
            # Per camera, before concatenating. Augmenting the concatenated
            # strip stretches one overlay across both views as if they were
            # halves of a single photo; the model augments each camera
            # separately and the video must match.
            frames = self._augment_frames(frames)
        if len(frames) > 1:
            heights = {f.shape[1] for f in frames}
            if len(heights) == 1:
                frames = [np.concatenate(frames, axis=2)]  # side by side
        return frames[0]

    def _extract_frame_batches(
        self, obs: Any, augment: bool = True
    ) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            image_src = self._get_image_from_dict(obs, augment=augment)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    image_src = self._get_image_from_dict(item, augment=augment)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
        stream: str = "main",
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            if self._cfg_get("cumulative_reward_on_video", False):
                # LIBERO pays its reward as a single +1 on the success frame.
                # At 30 fps that is one frame, ~33 ms -- easy to miss entirely
                # when watching, and invisible in a still. The running total
                # steps from 0 to 1 and stays there, so a clip's outcome is
                # readable at any point after success rather than only during
                # the one frame that earned it.
                totals = self._cum_reward.setdefault(stream, {})
                totals[env_id] = totals.get(env_id, 0.0) + (
                    float(value) if value is not None else 0.0
                )
                info_item["cumulative reward"] = totals[env_id]
            else:
                info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
        obs: Optional[Any] = None,
        buffer: Optional[list] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame to ``buffer``."""
        if buffer is None:
            buffer = self.render_images
        if not images:
            return
        if self.video_cfg.get("info_on_video", True):
            overlaid_images = []
            for env_id, img in enumerate(images):
                img = put_info_on_image(
                    img,
                    self._build_info_item(
                        infos,
                        rewards,
                        terminations,
                        env_id,
                        time_idx,
                        stream="clean"
                        if buffer is self.render_images_clean
                        else "main",
                    ),
                )
                # Draw the per-env language instruction at the bottom-left.
                task_desc = self._get_task_description(obs, env_id)
                if task_desc:
                    img = put_text_on_image_bottom(img, str(task_desc))
                overlaid_images.append(img)
            images = overlaid_images
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            buffer.append(full_image)
        else:
            buffer.append(images[0])

    def _append_stream(
        self,
        frames: list,
        obs: Any,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        buffer: list,
    ) -> None:
        """Append an already-extracted frame stream to ``buffer``."""
        if not frames:
            return
        list_infos = isinstance(infos, (list, tuple))
        for time_idx, images in enumerate(frames):
            step_info = (
                infos[time_idx] if list_infos and len(infos) > time_idx else infos
            )
            step_obs = (
                obs[time_idx]
                if isinstance(obs, (list, tuple)) and len(obs) > time_idx
                else obs
            )
            self._append_frame(
                images,
                step_info,
                rewards,
                terminations,
                time_idx,
                obs=step_obs,
                buffer=buffer,
            )

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer(s).

        The clean stream is extracted first, and never touches the
        augmentation, so the augmented pass draws exactly what it would have
        without a clean copy being recorded.
        """
        if self._frame_aug is not None:
            self._append_stream(
                self._extract_frame_batches(obs, augment=False),
                obs,
                infos,
                rewards,
                terminations,
                buffer=self.render_images_clean,
            )

        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                step_info = infos[time_idx] if len(infos) > time_idx else None
                step_obs = (
                    obs[time_idx]
                    if isinstance(obs, (list, tuple)) and len(obs) > time_idx
                    else obs
                )
                self._append_frame(
                    images, step_info, rewards, terminations, time_idx, obs=step_obs
                )
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(
                images, infos, rewards, terminations, time_idx, obs=obs
            )

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        # New episode, new reward total -- otherwise the overlay keeps climbing
        # across episodes and reads 4.000 on the fourth success of a run.
        self._cum_reward.clear()
        obs, info = self.env.reset(*args, **kwargs)
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def record_video_in_result(self, result) -> None:
        """Record video frames from a chunk_step / async_chunk_step result tuple."""
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]

            # Some envs may skip intermediate observations for performance and return
            # None entries. Filter them out for video collection.
            if isinstance(obs_list, (list, tuple)):
                valid_indices = [i for i, obs in enumerate(obs_list) if obs is not None]
                if len(valid_indices) == 0:
                    return
                if len(valid_indices) != len(obs_list):
                    obs_list = [obs_list[i] for i in valid_indices]
                    if isinstance(infos_list, (list, tuple)):
                        infos_list = [infos_list[i] for i in valid_indices]
                    if (
                        torch is not None
                        and isinstance(rewards, torch.Tensor)
                        and rewards.ndim == 2
                    ):
                        rewards = rewards[:, valid_indices]
                    if (
                        torch is not None
                        and isinstance(terminations, torch.Tensor)
                        and terminations.ndim == 2
                    ):
                        terminations = terminations[:, valid_indices]

            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        self.record_video_in_result(result)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to an MP4 file.

        The encode happens on the background thread pool, but we wait for the
        just-submitted write to complete before returning. The wait is required
        so the MP4 has a finalized ``moov`` atom on disk: ``imageio`` only writes
        it during ``writer.close()``, and the pool's worker threads are daemon
        threads that get killed mid-task at interpreter exit (no ``atexit``
        handler is run under Ray actor shutdown either). Without this wait,
        eval videos end at ``mdat`` and no player can open them.
        """
        # A flush IS the episode boundary for recording purposes, and the only
        # one that fires reliably: with auto_reset and ignore_terminations the
        # env rolls straight into the next episode without reset() reaching this
        # wrapper, so clearing only there would let the total run on and the
        # next clip would open already reading 1.000. Ahead of the early return
        # below, because an empty flush still ends the episode.
        self._cum_reward.clear()

        if not self.render_images:
            return

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # video_cnt alone said nothing about when a clip was recorded: it is a
        # per-worker save counter, so it climbs into the thousands over a run.
        # Name by the training epoch this eval belongs to, plus that counter.
        epoch = self._cfg_get("current_epoch", None)
        if epoch is None:
            epoch = getattr(self.env, "current_epoch", None)
        stem = (
            f"epoch{int(epoch)}_step{self.video_cnt}"
            if epoch is not None
            else f"step{self.video_cnt}"
        )

        # Opt-in sampling. The OOD eval runs ~480 episodes per eval point and
        # fires every 25 epochs, so recording all of them would be thousands of
        # clips and a real slice of the eval's wall time (each encode is
        # serialized into the rollout loop by the flush wait above).
        # video_sample_every keeps 1 in N episodes -- the pool is ordered by
        # task id, so a stride spreads the kept clips across all axes instead
        # of taking the first N from whichever axis sorts first --
        # and max_videos_per_epoch caps how many survive per eval point per
        # worker. Both default to off, so every existing caller is unaffected.
        sample_every = int(self._cfg_get("video_sample_every", 1) or 1)
        max_per_epoch = self._cfg_get("max_videos_per_epoch", None)
        keep = (self.video_cnt % sample_every) == 0
        if keep and max_per_epoch is not None:
            if epoch != self._video_epoch:
                self._video_epoch = epoch
                self._epoch_video_cnt = 0
            if self._epoch_video_cnt >= int(max_per_epoch):
                keep = False
            else:
                self._epoch_video_cnt += 1
        if not keep:
            self.render_images = []
            self.render_images_clean = []
            self.video_cnt += 1
            return

        frames = list(self.render_images)
        clean_frames = list(self.render_images_clean)
        self.render_images = []
        self.render_images_clean = []
        self.video_cnt += 1

        futures = [self._submit_save(frames, os.path.join(
            output_dir, f"{stem}_augm.mp4" if clean_frames else f"{stem}.mp4"))]
        if clean_frames:
            # Augmentation is on, so the unaugmented view of the same episode
            # sits next to it under the same stem.
            futures.append(
                self._submit_save(clean_frames, os.path.join(output_dir, f"{stem}.mp4"))
            )
        # Block until encode + writer.close() returns so the MP4s are valid on
        # disk before the rollout loop continues (or the process exits).
        for future in futures:
            future.result()

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> Future:
        """Submit a background job to save the video, return its Future."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)
        return future

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        try:
            video_writer = imageio.get_writer(mp4_path, fps=self._fps)
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
