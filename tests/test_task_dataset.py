"""Tests for native LeRobot v3 task dataset preparation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robocasa.data_collection.task_dataset import (
    DiskEpisodeRecorder,
    EpisodeRecorder,
    TaskDatasetWriter,
    build_lerobot_features,
    depth_meters_to_uint16,
    get_robot_state_layout,
    require_supported_lerobot,
)
from robocasa.scene_config import load_scene_config


class FakeDatasetBackend:
    """In-memory implementation of the LeRobot writer subset."""

    def __init__(self) -> None:
        self.active_frames: list[dict] = []
        self.episodes: list[list[dict]] = []
        self.clear_count = 0
        self.finalize_count = 0

    def add_frame(self, frame: dict) -> None:
        self.active_frames.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self.active_frames)
        self.active_frames = []

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        assert delete_images
        self.active_frames = []
        self.clear_count += 1

    def finalize(self) -> None:
        self.finalize_count += 1


def _observation() -> dict[str, np.ndarray]:
    config = load_scene_config("expanded_example_scene")
    observation = {
        "robot0_joint_pos": np.arange(7, dtype=np.float64),
        "robot0_joint_vel": np.arange(7, dtype=np.float64),
        "robot0_eef_pos": np.arange(3, dtype=np.float64),
        "robot0_eef_quat": np.arange(4, dtype=np.float64),
        "robot0_gripper_qpos": np.arange(2, dtype=np.float64),
        "robot0_gripper_qvel": np.arange(2, dtype=np.float64),
        "robot0_base_pos": np.arange(3, dtype=np.float64),
        "robot0_base_quat": np.arange(4, dtype=np.float64),
        "robot0_base_to_eef_pos": np.arange(3, dtype=np.float64),
        "robot0_base_to_eef_quat": np.arange(4, dtype=np.float64),
    }
    assert config.cameras is not None
    for camera_name in config.cameras.names:
        observation[f"{camera_name}_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
        observation[f"{camera_name}_depth"] = np.full(
            (480, 640, 1), 1.25, dtype=np.float32
        )
    for obj in config.objects:
        observation[f"{obj.name}_pos"] = np.zeros(3, dtype=np.float64)
        observation[f"{obj.name}_quat"] = np.asarray([1, 0, 0, 0], dtype=np.float64)
    return observation


def test_depth_is_saved_as_saturated_uint16_millimeters() -> None:
    depth = np.asarray([[0.0, 1.2346, np.nan, np.inf, 70.0]], dtype=np.float32)
    converted = depth_meters_to_uint16(depth)
    np.testing.assert_array_equal(converted, [[0, 1235, 0, 65535, 65535]])
    assert converted.dtype == np.uint16


def test_unsupported_lerobot_version_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "robocasa.data_collection.task_dataset.importlib.metadata.version",
        lambda package: "0.3.3",
    )

    with pytest.raises(RuntimeError, match="environment has lerobot==0.3.3"):
        require_supported_lerobot()


def test_robot_layout_does_not_duplicate_eef_suffixes() -> None:
    layout = get_robot_state_layout(_observation())
    assert len(layout) == len(set(layout)) == 10
    assert layout[-2:] == (
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
    )


def test_features_use_video_and_native_array2d_depth() -> None:
    config = load_scene_config("expanded_example_scene")
    features = build_lerobot_features(config, 39, 12, 20)
    assert features["observation.images.robot_left"]["dtype"] == "video"
    assert features["observation.images.robot_left"]["shape"] == (480, 640, 3)
    assert features["observation.depth.robot_left"] == {
        "dtype": "uint16",
        "shape": (480, 640),
        "names": ["height", "width"],
    }
    assert features["observation.object_state"]["names"][:7] == [
        "avocado.x",
        "avocado.y",
        "avocado.z",
        "avocado.qx",
        "avocado.qy",
        "avocado.qz",
        "avocado.qw",
    ]


def test_writer_commits_aligned_episode_and_sidecars(tmp_path: Path) -> None:
    config = load_scene_config("expanded_example_scene")
    observation = _observation()
    backend = FakeDatasetBackend()
    writer = TaskDatasetWriter(
        tmp_path / "dataset",
        "put_objects_in_basket",
        config,
        observation,
        action_size=12,
        environment_state_size=5,
        task_snapshot={"name": "put_objects_in_basket"},
        scene_snapshot=_scene_snapshot(),
        backend=backend,
    )
    recorder = EpisodeRecorder(seed=13, initial_simulator_state=np.zeros(5))
    recorder.record_transition(
        observation,
        np.zeros(12),
        np.ones(5),
        command_index=0,
        command_text="move(can, in, basket)",
        command_completed=False,
    )
    recorder.mark_command_completed(0)
    recorder.record_transition(
        observation,
        np.ones(12),
        np.full(5, 2),
        command_index=1,
        command_text="move(basket, left_of, bagged_food)",
        command_completed=False,
    )
    recorder.mark_command_completed(1)
    assert recorder.frame_count == 2
    assert recorder.simulator_state_index == 2
    episode = recorder.finish(
        model_xml="<mujoco/>",
        camera_calibration={
            "robot_left": {"intrinsics": np.eye(3), "camera_to_world": np.eye(4)}
        },
        scene_graph_ground_truth=_graph_trace([0, 1, 2]),
        scene_graph_symbolic=_graph_trace([0, 1, 2]),
    )

    assert writer.save_episode(episode) == 0
    writer.finalize()
    writer.finalize()

    assert len(backend.episodes) == 1
    first, second = backend.episodes[0]
    assert first["next.reward"].item() == 1.0
    assert not first["next.done"].item()
    assert second["next.done"].item()
    assert first["observation.environment_state"].tolist() == [0.0] * 5
    assert first["observation.depth.robot_left"].dtype == np.uint16
    assert backend.finalize_count == 1

    extras = tmp_path / "dataset" / "extras" / "episode_000000"
    with np.load(extras / "states.npz") as archive:
        assert archive["states"].shape == (3, 5)
    assert json.loads((extras / "episode.json").read_text())["seed"] == 13


def test_failure_log_is_append_only(tmp_path: Path) -> None:
    config = load_scene_config("expanded_example_scene")
    writer = TaskDatasetWriter(
        tmp_path / "dataset",
        "task",
        config,
        _observation(),
        action_size=12,
        environment_state_size=5,
        task_snapshot={"name": "task"},
        scene_snapshot=_scene_snapshot(),
        backend=FakeDatasetBackend(),
    )
    writer.log_failure(4, RuntimeError("grasp failed"))
    writer.log_failure(5, "timeout")
    lines = (
        (tmp_path / "dataset" / "extras" / "failures.jsonl")
        .read_text()
        .splitlines()
    )
    assert [json.loads(line)["attempt_seed"] for line in lines] == [4, 5]


def test_disk_recorder_spools_and_lazily_restores_frames() -> None:
    observation = _observation()
    transition = type(
        "Transition",
        (),
        {
            "observation": observation,
            "action": np.ones(12),
            "next_simulator_state": np.ones(5),
            "command_index": 0,
            "command_text": "move(can, in, basket)",
        },
    )()
    with DiskEpisodeRecorder(7, np.zeros(5)) as recorder:
        recorder.record_oracle_transition(transition)
        recorder.mark_command_completed(0)
        episode = recorder.finish(
            model_xml="<mujoco/>",
            camera_calibration={},
            scene_graph_ground_truth={},
            scene_graph_symbolic={},
        )

        assert len(episode.frames) == 1
        assert episode.frames[0].command_completed
        np.testing.assert_array_equal(
            episode.frames[0].observation["robot0_joint_pos"],
            observation["robot0_joint_pos"],
        )
        assert episode.frames[0].observation["robot_left_depth"].dtype == np.uint16
        assert episode.simulator_states.shape == (2, 5)


def test_writer_rejects_scene_graph_not_aligned_to_completion(tmp_path: Path) -> None:
    config = load_scene_config("expanded_example_scene")
    observation = _observation()
    writer = TaskDatasetWriter(
        tmp_path / "dataset",
        "task",
        config,
        observation,
        action_size=12,
        environment_state_size=5,
        task_snapshot={"name": "task"},
        scene_snapshot=_scene_snapshot(),
        backend=FakeDatasetBackend(),
    )
    recorder = EpisodeRecorder(0, np.zeros(5))
    recorder.record_transition(
        observation,
        np.zeros(12),
        np.ones(5),
        command_index=0,
        command_text="move(can, in, basket)",
        command_completed=True,
    )
    episode = recorder.finish(
        model_xml="<mujoco/>",
        camera_calibration={},
        scene_graph_ground_truth=_graph_trace([0]),
        scene_graph_symbolic=_graph_trace([0]),
    )

    with pytest.raises(ValueError, match="one initial and one snapshot"):
        writer.save_episode(episode)


def test_sidecar_failure_does_not_commit_backend_episode(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_scene_config("expanded_example_scene")
    observation = _observation()
    backend = FakeDatasetBackend()
    writer = TaskDatasetWriter(
        tmp_path / "dataset",
        "task",
        config,
        observation,
        action_size=12,
        environment_state_size=5,
        task_snapshot={"name": "task"},
        scene_snapshot=_scene_snapshot(),
        backend=backend,
    )
    recorder = EpisodeRecorder(0, np.zeros(5))
    recorder.record_transition(
        observation,
        np.zeros(12),
        np.ones(5),
        command_index=0,
        command_text="move(can, in, basket)",
        command_completed=True,
    )
    episode = recorder.finish(
        model_xml="<mujoco/>",
        camera_calibration={},
        scene_graph_ground_truth=_graph_trace([0, 1]),
        scene_graph_symbolic=_graph_trace([0, 1]),
    )
    def fail_sidecar_write(episode_index: int, record: object) -> None:
        del episode_index, record
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_episode_extras", fail_sidecar_write)

    with pytest.raises(OSError, match="disk full"):
        writer.save_episode(episode)

    assert not backend.episodes
    assert backend.clear_count == 1


def test_writer_rejects_misaligned_state_trace(tmp_path: Path) -> None:
    config = load_scene_config("expanded_example_scene")
    observation = _observation()
    backend = FakeDatasetBackend()
    writer = TaskDatasetWriter(
        tmp_path / "dataset",
        "task",
        config,
        observation,
        action_size=12,
        environment_state_size=5,
        task_snapshot={"name": "task"},
        scene_snapshot=_scene_snapshot(),
        backend=backend,
    )
    recorder = EpisodeRecorder(seed=1, initial_simulator_state=np.zeros(5))
    recorder.record_transition(
        observation,
        np.zeros(12),
        np.ones(5),
        command_index=0,
        command_text="move(can, in, basket)",
    )
    episode = recorder.finish(
        model_xml="<mujoco/>",
        camera_calibration={},
        scene_graph_ground_truth={},
        scene_graph_symbolic={},
    )
    episode = replace(episode, simulator_states=np.zeros((1, 5)))

    with pytest.raises(ValueError, match="N\\+1"):
        writer.save_episode(episode)
    assert not backend.episodes


def _scene_snapshot() -> dict:
    import yaml

    scene_path = (
        Path(__file__).parents[1]
        / "robocasa/scene_configs/expanded_example_scene.yaml"
    )
    return yaml.safe_load(scene_path.read_text())


def _graph_trace(state_indices: list[int]) -> dict:
    return {
        "snapshots": [
            {"stage_index": stage_index, "simulator_state_index": state_index}
            for stage_index, state_index in enumerate(state_indices)
        ]
    }
