"""Deterministic replay for task datasets collected in LeRobot v3 format."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from robocasa.example_env import SceneGymEnv, make_scene_env
from robocasa.scene_config import SceneConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Summary of an exact-state or action replay."""

    episode_index: int
    frame_count: int
    mode: str
    maximum_state_error: float | None = None


def load_episode_states(dataset_dir: str | Path, episode_index: int) -> np.ndarray:
    """Load the N+1 deterministic state trace for an episode."""
    path = _episode_dir(dataset_dir, episode_index) / "states.npz"
    with np.load(path) as archive:
        return np.asarray(archive["states"])


def load_episode_actions(dataset_dir: str | Path, episode_index: int) -> np.ndarray:
    """Load one episode's actions through the native LeRobot v3 reader."""
    dataset_path = Path(dataset_dir)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise ImportError("Replay requires lerobot==0.4.4") from error
    dataset = LeRobotDataset(
        repo_id=dataset_path.name,
        root=dataset_path,
        episodes=[episode_index],
        download_videos=False,
    )
    return np.asarray(dataset.hf_dataset["action"])


def replay_episode(
    dataset_dir: str | Path,
    episode_index: int = 0,
    *,
    use_actions: bool = False,
    video_path: str | Path | None = None,
    camera_names: tuple[str, ...] | None = None,
) -> ReplayResult:
    """Replay an episode, using exact states by default.

    Args:
        dataset_dir: Local root of one task's LeRobot v3 dataset.
        episode_index: Zero-based episode number.
        use_actions: Step saved actions and measure successor-state divergence
            instead of setting every saved state directly.
        video_path: Optional MP4 output path.
        camera_names: Configured cameras to render. Defaults to every camera.

    Returns:
        Replay summary, including maximum action-replay state error.
    """
    dataset_path = Path(dataset_dir)
    states = load_episode_states(dataset_path, episode_index)
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("Replay states must be a two-dimensional N+1 trace")
    scene_data = _read_json(dataset_path / "extras" / "scene.json")
    scene_config = SceneConfig.from_dict(scene_data)
    env = make_scene_env(scene_config, horizon=max(len(states) + 1, 2))
    if not isinstance(env, SceneGymEnv):
        raise TypeError("make_scene_env() did not return SceneGymEnv")

    configured_cameras = (
        scene_config.cameras.names if scene_config.cameras is not None else ()
    )
    selected_cameras = camera_names or configured_cameras
    unknown = sorted(set(selected_cameras) - set(configured_cameras))
    if unknown:
        env.close()
        raise ValueError(f"Unknown replay cameras: {', '.join(unknown)}")
    if video_path is not None and not selected_cameras:
        env.close()
        raise ValueError("Video replay requires at least one configured camera")

    actions = load_episode_actions(dataset_path, episode_index) if use_actions else None
    if actions is not None and len(actions) != len(states) - 1:
        env.close()
        raise ValueError("Episode actions do not align with the N+1 state trace")

    writer = None
    maximum_error = 0.0 if use_actions else None
    try:
        _restore_episode_model(env, dataset_path, episode_index)
        _set_simulator_state(env, states[0])
        if video_path is not None:
            output_path = Path(video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(output_path, fps=scene_config.control_freq)
            writer.append_data(_render_cameras(env, selected_cameras))

        for frame_index in range(len(states) - 1):
            if actions is None:
                _set_simulator_state(env, states[frame_index + 1])
            else:
                env.step(np.asarray(actions[frame_index]))
                actual_state = np.asarray(env.sim.get_state().flatten())
                error = float(np.linalg.norm(actual_state - states[frame_index + 1]))
                maximum_error = max(maximum_error or 0.0, error)
            if writer is not None:
                writer.append_data(_render_cameras(env, selected_cameras))
    finally:
        if writer is not None:
            writer.close()
        env.close()

    return ReplayResult(
        episode_index=episode_index,
        frame_count=len(states) - 1,
        mode="actions" if use_actions else "states",
        maximum_state_error=maximum_error,
    )


def _restore_episode_model(
    env: SceneGymEnv, dataset_dir: Path, episode_index: int
) -> None:
    model_path = _episode_dir(dataset_dir, episode_index) / "model.xml.gz"
    with gzip.open(model_path, "rt", encoding="utf-8") as stream:
        model_xml = stream.read()
    raw_env = env.env
    raw_env.reset()
    edited_xml = raw_env.edit_model_xml(model_xml)
    raw_env.reset_from_xml_string(edited_xml)
    raw_env.sim.reset()


def _set_simulator_state(env: SceneGymEnv, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state))
    env.sim.forward()


def _render_cameras(env: SceneGymEnv, camera_names: tuple[str, ...]) -> np.ndarray:
    assert env.config.cameras is not None
    frames = [
        env.sim.render(
            camera_name=camera_name,
            height=env.config.cameras.height,
            width=env.config.cameras.width,
        )[::-1]
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1)


def _episode_dir(dataset_dir: str | Path, episode_index: int) -> Path:
    return Path(dataset_dir) / "extras" / f"episode_{episode_index:06d}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
