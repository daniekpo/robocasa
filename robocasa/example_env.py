"""Example downstream Gymnasium adapter for configured RoboCasa scenes."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import robosuite
from gymnasium import spaces
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from robocasa.scene_config import SceneConfig, load_scene_config

ROBOT_STATE_SUFFIXES = (
    "joint_pos",
    "joint_vel",
    "eef_pos",
    "eef_quat",
    "gripper_qpos",
    "gripper_qvel",
    "base_pos",
    "base_quat",
    "base_to_eef_pos",
    "base_to_eef_quat",
)


class SceneGymEnv(gym.Env):
    """Gymnasium interface exposing visual, robot, and object scene state."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, env: MujocoEnv, config: SceneConfig):
        """Wrap a configured RoboCasa environment.

        Args:
            env: Initialized RoboCasa environment.
            config: Scene configuration used to build the environment.
        """
        super().__init__()
        self.env = env
        self.config = config
        sample = self._filter_observation(self.env.reset())
        self._last_observation = sample
        self.observation_space = spaces.Dict(
            {name: _space_for_observation(value) for name, value in sample.items()}
        )
        action_low, action_high = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_low.astype(np.float32),
            high=action_high.astype(np.float32),
            dtype=np.float32,
        )

    def _filter_observation(
        self, observation: dict[str, np.ndarray]
    ) -> OrderedDict[str, np.ndarray]:
        """Select stable algorithm-facing observations from RoboCasa output."""
        selected: OrderedDict[str, np.ndarray] = OrderedDict()
        camera_names = self.config.cameras.names if self.config.cameras else ()
        for camera_name in camera_names:
            for suffix in ("image", "depth"):
                key = f"{camera_name}_{suffix}"
                if key in observation:
                    camera_observation = observation[key][::-1].copy()
                    if suffix == "depth":
                        camera_observation = get_real_depth_map(
                            self.env.sim, camera_observation
                        ).astype(np.float32)
                    selected[key] = camera_observation

        robot_prefixes = tuple(
            robot.robot_model.naming_prefix for robot in self.env.robots
        )
        for key, value in observation.items():
            if any(
                key == f"{prefix}{suffix}"
                for prefix in robot_prefixes
                for suffix in ROBOT_STATE_SUFFIXES
            ):
                selected[key] = value

        for object_config in self.config.objects:
            for suffix in ("pos", "quat"):
                key = f"{object_config.name}_{suffix}"
                selected[key] = observation[key]
        return selected

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OrderedDict[str, np.ndarray], dict[str, Any]]:
        """Reset the scene and return a Gymnasium observation."""
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        observation = self._filter_observation(self.env.reset())
        self._last_observation = observation
        return observation, {"scene_config": self.config}

    def step(
        self, action: np.ndarray
    ) -> tuple[OrderedDict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance the simulator by one control step."""
        observation, reward, done, info = self.env.step(action)
        filtered_observation = self._filter_observation(observation)
        self._last_observation = filtered_observation
        return filtered_observation, reward, False, done, info

    def render(self) -> np.ndarray:
        """Return the first configured RGB camera image."""
        if self.config.cameras is None:
            raise RuntimeError("render() requires a configured camera")
        return self._last_observation[f"{self.config.cameras.names[0]}_image"]

    def get_camera_calibration(self) -> dict[str, dict[str, np.ndarray]]:
        """Return intrinsics and camera-to-world poses for configured cameras."""
        camera_group = self.config.cameras
        if camera_group is None:
            return {}
        return {
            camera_name: {
                "intrinsics": get_camera_intrinsic_matrix(
                    self.env.sim,
                    camera_name,
                    camera_group.height,
                    camera_group.width,
                ),
                "camera_to_world": get_camera_extrinsic_matrix(
                    self.env.sim, camera_name
                ),
            }
            for camera_name in camera_group.names
        }

    def close(self) -> None:
        """Close the underlying RoboCasa environment."""
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Forward simulator-specific attributes to the wrapped environment."""
        return getattr(self.env, name)


def make_scene_env(
    scene: str | Path | SceneConfig,
    *,
    as_gym: bool = True,
    **env_kwargs: Any,
) -> SceneGymEnv | MujocoEnv:
    """Build a configured scene by bundled name, path, or parsed config.

    Args:
        scene: Bundled config name, JSON/YAML path, or parsed scene config.
        as_gym: Wrap the environment with the Gymnasium API when true.
        **env_kwargs: Additional robosuite environment options such as ``horizon``.

    Returns:
        A :class:`SceneGymEnv` by default, or the raw RoboCasa environment.
    """
    config = scene if isinstance(scene, SceneConfig) else load_scene_config(scene)
    controlled_keys = {
        "camera_depths",
        "camera_heights",
        "camera_names",
        "camera_widths",
        "control_freq",
        "controller_configs",
        "env_name",
        "robots",
        "scene_config",
        "seed",
        "use_camera_obs",
    }
    conflicts = sorted(controlled_keys.intersection(env_kwargs))
    if conflicts:
        raise ValueError(
            "Scene config controls these environment options: " + ", ".join(conflicts)
        )

    camera_group = config.cameras
    has_cameras = camera_group is not None
    kwargs = {
        "env_name": config.environment,
        "robots": config.robot,
        "controller_configs": load_composite_controller_config(
            controller=config.controller, robot=config.robot
        ),
        "scene_config": config,
        "has_renderer": False,
        "has_offscreen_renderer": has_cameras,
        "use_camera_obs": has_cameras,
        "ignore_done": False,
        "control_freq": config.control_freq,
        "seed": config.seed,
    }
    if camera_group is not None:
        kwargs.update(
            camera_names=list(camera_group.names),
            camera_widths=camera_group.width,
            camera_heights=camera_group.height,
            camera_depths=camera_group.depth,
        )
    kwargs.update(env_kwargs)
    env = robosuite.make(**kwargs)
    return SceneGymEnv(env, config) if as_gym else env


def _space_for_observation(observation: np.ndarray) -> spaces.Box:
    """Create a Gymnasium box matching one simulator observation."""
    if np.issubdtype(observation.dtype, np.integer):
        dtype_info = np.iinfo(observation.dtype)
        return spaces.Box(
            low=dtype_info.min,
            high=dtype_info.max,
            shape=observation.shape,
            dtype=observation.dtype,
        )
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=observation.shape,
        dtype=observation.dtype,
    )
