"""Tests for deterministic task-dataset replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from robocasa.data_collection import task_replay


class FakeSim:
    """Minimal simulator state interface used by replay."""

    def __init__(self) -> None:
        self.state = np.zeros(2)
        self.restored_states: list[np.ndarray] = []

    def reset(self) -> None:
        """Accept model reset."""

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        self.state = np.asarray(state).copy()
        self.restored_states.append(self.state.copy())

    def forward(self) -> None:
        """Accept forward dynamics."""

    def get_state(self) -> SimpleNamespace:
        return SimpleNamespace(flatten=lambda: self.state.copy())


class FakeRawEnv:
    """Minimal XML restoration interface."""

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim
        self.loaded_xml: str | None = None

    def reset(self) -> None:
        """Accept environment reset."""

    def edit_model_xml(self, model_xml: str) -> str:
        return model_xml

    def reset_from_xml_string(self, model_xml: str) -> None:
        self.loaded_xml = model_xml


class FakeReplayEnv:
    """SceneGymEnv-shaped replay target."""

    last_instance: FakeReplayEnv | None = None

    def __init__(self, config: object) -> None:
        self.config = config
        self.sim = FakeSim()
        self.env = FakeRawEnv(self.sim)
        self.closed = False
        FakeReplayEnv.last_instance = self

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.sim.state = self.sim.state + np.asarray(action)
        return {}, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def _write_episode(dataset_dir: Path) -> np.ndarray:
    extras = dataset_dir / "extras"
    episode_dir = extras / "episode_000000"
    episode_dir.mkdir(parents=True)
    scene_path = (
        Path(__file__).parents[1]
        / "robocasa"
        / "scene_configs"
        / "expanded_example_scene.yaml"
    )
    (extras / "scene.json").write_text(
        json.dumps(yaml.safe_load(scene_path.read_text())),
        encoding="utf-8",
    )
    states = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]])
    np.savez_compressed(episode_dir / "states.npz", states=states)
    with gzip.open(episode_dir / "model.xml.gz", "wt", encoding="utf-8") as stream:
        stream.write("<mujoco/>")
    return states


def test_state_replay_restores_every_saved_state(
    tmp_path: Path, monkeypatch
) -> None:
    states = _write_episode(tmp_path)
    monkeypatch.setattr(task_replay, "SceneGymEnv", FakeReplayEnv)
    monkeypatch.setattr(
        task_replay,
        "make_scene_env",
        lambda config, horizon: FakeReplayEnv(config),
    )

    result = task_replay.replay_episode(tmp_path)

    env = FakeReplayEnv.last_instance
    assert env is not None
    np.testing.assert_array_equal(env.sim.restored_states, states)
    assert env.env.loaded_xml == "<mujoco/>"
    assert env.closed
    assert result.frame_count == 2
    assert result.mode == "states"


def test_action_replay_reports_successor_state_error(
    tmp_path: Path, monkeypatch
) -> None:
    _write_episode(tmp_path)
    monkeypatch.setattr(task_replay, "SceneGymEnv", FakeReplayEnv)
    monkeypatch.setattr(
        task_replay,
        "make_scene_env",
        lambda config, horizon: FakeReplayEnv(config),
    )
    monkeypatch.setattr(
        task_replay,
        "load_episode_actions",
        lambda dataset_dir, episode_index: np.asarray([[1.0, 2.0], [2.0, 3.0]]),
    )

    result = task_replay.replay_episode(tmp_path, use_actions=True)

    assert result.mode == "actions"
    assert result.maximum_state_error == 0.0
