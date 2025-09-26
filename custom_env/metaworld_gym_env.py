"""MetaWorld environment wrappers for DUSDi.

This module provides a gym-compatible environment that exposes MetaWorld
benchmarks with a consistent API for the rest of the codebase. It also
contains helper utilities for building environments from a Hydra
configuration and for inferring the observation/action dimensions.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import gym
from gym import spaces
import numpy as np
from omegaconf import OmegaConf

import metaworld


def _to_float_box(space: spaces.Box) -> spaces.Box:
    """Convert a Box space to float32 while preserving bounds."""
    low = np.array(space.low, dtype=np.float32)
    high = np.array(space.high, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _flatten_observation(obs: np.ndarray | Dict[str, np.ndarray],
                         keys: Optional[Sequence[str]] = None) -> np.ndarray:
    """Flatten dictionary observations into a single float32 vector."""
    if isinstance(obs, dict):
        if keys is None:
            items = obs.items()
        else:
            items = ((key, obs[key]) for key in keys)
        pieces: List[np.ndarray] = []
        for _, value in items:
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            pieces.append(np.asarray(value, dtype=np.float32).ravel())
        obs_array = np.concatenate(pieces, axis=0)
    else:
        obs_array = np.asarray(obs, dtype=np.float32).ravel()
    return obs_array.astype(np.float32)


class MetaWorldGymEnv(gym.Env):
    """Wrap MetaWorld tasks to expose a standard gym interface."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 suite: str,
                 task: str,
                 mode: str = "train",
                 max_episode_steps: Optional[int] = None,
                 randomize_task_on_reset: bool = True,
                 obs_keys: Optional[Sequence[str]] = None,
                 obs_clip: Optional[float] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__()

        self.suite = suite.upper()
        self.task_name = task
        self.mode = mode
        self.randomize_task_on_reset = randomize_task_on_reset
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else None
        self.obs_clip = obs_clip
        self._rng = np.random.RandomState()

        benchmark = self._build_benchmark()
        self._tasks, env_cls = self._resolve_tasks(benchmark)
        if len(self._tasks) == 0:
            raise ValueError(f"No MetaWorld tasks found for '{self.task_name}' in suite '{self.suite}'.")

        self._env = env_cls()
        if seed is not None:
            self.seed(seed)
        if hasattr(self._env, "max_path_length"):
            default_horizon = int(self._env.max_path_length)
        else:
            default_horizon = 200
        self.max_episode_steps = max_episode_steps or default_horizon
        self._step_count = 0
        self._last_success = 0.0

        self.action_space = _to_float_box(self._env.action_space)
        if not isinstance(self._env.observation_space, spaces.Box):
            raise ValueError("MetaWorld environments are expected to expose Box observation spaces.")
        self.observation_space = _to_float_box(self._env.observation_space)

        self._current_task_index = 0
        self._select_task()

    # ---------------------------------------------------------------------
    # Benchmark construction helpers
    # ---------------------------------------------------------------------
    def _build_benchmark(self):
        if self.suite == "ML1":
            return metaworld.ML1(self.task_name)
        if self.suite == "MT1":
            return metaworld.MT1(self.task_name)
        if self.suite == "ML45":
            return metaworld.ML45()
        if self.suite == "MT10":
            return metaworld.MT10()
        if self.suite == "MT50":
            return metaworld.MT50()
        raise ValueError(f"Unsupported MetaWorld suite '{self.suite}'.")

    def _resolve_tasks(self, benchmark) -> Tuple[Sequence, type]:
        if self.suite in {"ML1", "MT1"}:
            if self.mode == "train":
                task_pool = benchmark.train_tasks
                class_pool = benchmark.train_classes
            else:
                task_pool = benchmark.test_tasks
                class_pool = benchmark.test_classes
            env_cls = class_pool[self.task_name]
            tasks = [task for task in task_pool if task.env_name == self.task_name]
        else:
            if self.mode == "train":
                task_pool = benchmark.train_tasks
                class_pool = benchmark.train_classes
            else:
                task_pool = benchmark.test_tasks
                class_pool = benchmark.test_classes
            if self.task_name not in class_pool:
                raise ValueError(f"Task '{self.task_name}' is not part of suite '{self.suite}'.")
            env_cls = class_pool[self.task_name]
            tasks = [task for task in task_pool if task.env_name == self.task_name]
        return tasks, env_cls

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> Sequence[int]:
        self._rng.seed(seed)
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        return [seed] if seed is not None else []

    def _select_task(self) -> None:
        if not self._tasks:
            return
        if self.randomize_task_on_reset:
            self._current_task_index = int(self._rng.randint(len(self._tasks)))
        task = self._tasks[self._current_task_index]
        self._env.set_task(task)

    def reset(self):  # type: ignore[override]
        self._step_count = 0
        self._last_success = 0.0
        self._select_task()
        obs = self._env.reset()
        return self._process_observation(obs)

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        success = float(info.get("success", info.get("Success", 0.0)))
        self._last_success = success
        if self._step_count >= self.max_episode_steps:
            done = True
        processed_obs = self._process_observation(obs)
        info = dict(info)
        info["success"] = success
        return processed_obs, float(reward), bool(done), info

    def render(self, mode: str = "human"):
        return self._env.render(mode)

    def close(self):  # type: ignore[override]
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _process_observation(self, obs):
        obs_array = _flatten_observation(obs, self.obs_keys)
        if self.obs_clip is not None:
            np.clip(obs_array, -self.obs_clip, self.obs_clip, out=obs_array)
        return obs_array

    def get_additional_states(self) -> np.ndarray:
        progress = self._step_count / float(self.max_episode_steps)
        return np.array([progress, self._last_success], dtype=np.float32)


def _config_to_dict(cfg) -> Dict[str, object]:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError("Unsupported configuration type for MetaWorld environment.")


def create_metaworld_env_from_cfg(cfg,
                                  task: Optional[str] = None,
                                  mode: Optional[str] = None,
                                  seed: Optional[int] = None) -> MetaWorldGymEnv:
    cfg_dict = _config_to_dict(cfg)
    task_name = task or cfg_dict.get("task")
    if task_name is None:
        raise ValueError("MetaWorld configuration must define a 'task'.")
    env = MetaWorldGymEnv(
        suite=cfg_dict.get("suite", "ML1"),
        task=task_name,
        mode=mode or cfg_dict.get("mode", "train"),
        max_episode_steps=cfg_dict.get("episode_length"),
        randomize_task_on_reset=cfg_dict.get("randomize_task_on_reset", True),
        obs_keys=cfg_dict.get("obs_keys"),
        obs_clip=cfg_dict.get("obs_clip"),
        seed=seed,
    )
    return env


def get_metaworld_space_info(cfg,
                             task: Optional[str] = None,
                             mode: Optional[str] = None) -> Tuple[int, int]:
    env = create_metaworld_env_from_cfg(cfg, task=task, mode=mode)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    env.close()
    return obs_dim, action_dim
