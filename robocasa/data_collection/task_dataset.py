"""LeRobot v3 storage for task-oracle demonstrations.

The simulator-facing recorder deliberately has no dependency on the oracle. An
oracle produces :class:`EpisodeFrame` transitions and a complete
:class:`EpisodeRecord`; this module validates and commits only successful
episodes.
"""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robocasa.example_env import ROBOT_STATE_SUFFIXES
from robocasa.scene_config import SceneConfig

LOGGER = logging.getLogger(__name__)

DATASET_SCHEMA_VERSION = 1
DEPTH_METERS_TO_MILLIMETERS = 1000.0
MAX_DEPTH_MILLIMETERS = np.iinfo(np.uint16).max
SUPPORTED_LEROBOT_VERSION = "0.4.4"


def require_supported_lerobot() -> None:
    """Raise an actionable error unless the LeRobot v3 dependency is installed."""
    try:
        installed_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Task collection requires lerobot==0.4.4, but LeRobot is not installed. "
            f"Install it with: {sys.executable} -m pip install --upgrade "
            "'lerobot==0.4.4'"
        ) from error
    if installed_version != SUPPORTED_LEROBOT_VERSION:
        raise RuntimeError(
            "Task collection requires lerobot==0.4.4, but the active Python "
            f"environment has lerobot=={installed_version}. Install the supported "
            f"version with: {sys.executable} -m pip install --upgrade "
            "'lerobot==0.4.4'"
        )


class DatasetBackend(Protocol):
    """Subset of the LeRobot writer API used by :class:`TaskDatasetWriter`."""

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Buffer one frame."""

    def save_episode(self) -> None:
        """Commit the active episode."""

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """Discard the active episode, including temporary videos."""

    def finalize(self) -> None:
        """Flush metadata and close dataset writers."""


@dataclass(frozen=True)
class EpisodeFrame:
    """One pre-action observation and its resulting task annotation.

    Args:
        observation: Filtered :class:`~robocasa.example_env.SceneGymEnv`
            observation before applying ``action``.
        action: Native robot action applied to that observation.
        command_index: Zero-based index of the active task command.
        command_text: Human-readable active task command.
        command_completed: Whether this transition completed the active command.
    """

    observation: Mapping[str, np.ndarray]
    action: np.ndarray
    command_index: int
    command_text: str
    command_completed: bool = False


@dataclass(frozen=True)
class EpisodeRecord:
    """Complete successful oracle rollout and deterministic replay data.

    ``simulator_states`` follows an N+1 contract: initial state, then one
    post-action successor for every frame.
    """

    seed: int
    frames: Sequence[EpisodeFrame]
    simulator_states: np.ndarray
    model_xml: str
    camera_calibration: Mapping[str, Mapping[str, np.ndarray]]
    scene_graph_ground_truth: Mapping[str, Any] | Sequence[Any]
    scene_graph_symbolic: Mapping[str, Any] | Sequence[Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    """Collect oracle transition callbacks into an :class:`EpisodeRecord`."""

    def __init__(self, seed: int, initial_simulator_state: np.ndarray) -> None:
        """Start a recording with the state preceding the first action."""
        self.seed = seed
        self._frames: list[EpisodeFrame] = []
        self._states = [np.asarray(initial_simulator_state, dtype=np.float64).copy()]

    @property
    def frame_count(self) -> int:
        """Return the number of recorded actions."""
        return len(self._frames)

    @property
    def simulator_state_index(self) -> int:
        """Return the current index in the N+1 simulator-state trace."""
        return len(self._states) - 1

    def record_transition(
        self,
        observation: Mapping[str, np.ndarray],
        action: np.ndarray,
        next_simulator_state: np.ndarray,
        command_index: int,
        command_text: str,
        command_completed: bool = False,
    ) -> None:
        """Record one pre-observation/action and its post-action simulator state."""
        self._frames.append(
            EpisodeFrame(
                observation={
                    key: np.asarray(value).copy()
                    for key, value in observation.items()
                },
                action=np.asarray(action).copy(),
                command_index=command_index,
                command_text=command_text,
                command_completed=command_completed,
            )
        )
        self._states.append(
            np.asarray(next_simulator_state, dtype=np.float64).copy()
        )

    def record_oracle_transition(self, transition: Any) -> None:
        """Adapt an oracle ``OracleTransition`` to the storage callback."""
        if transition.command_index is None or transition.command_text is None:
            raise ValueError(
                "Recorded oracle transitions must belong to a task command"
            )
        self.record_transition(
            transition.observation,
            transition.action,
            transition.next_simulator_state,
            transition.command_index,
            transition.command_text,
        )

    def mark_command_completed(self, command_index: int) -> None:
        """Attach a sparse reward to the latest transition of a command."""
        if not self._frames or self._frames[-1].command_index != command_index:
            raise ValueError(
                f"No latest transition exists for command index {command_index}"
            )
        self._frames[-1] = replace(self._frames[-1], command_completed=True)

    def finish(
        self,
        *,
        model_xml: str,
        camera_calibration: Mapping[str, Mapping[str, np.ndarray]],
        scene_graph_ground_truth: Mapping[str, Any] | Sequence[Any],
        scene_graph_symbolic: Mapping[str, Any] | Sequence[Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeRecord:
        """Freeze the accumulated successful rollout for dataset storage."""
        return EpisodeRecord(
            seed=self.seed,
            frames=tuple(self._frames),
            simulator_states=np.stack(self._states),
            model_xml=model_xml,
            camera_calibration=camera_calibration,
            scene_graph_ground_truth=scene_graph_ground_truth,
            scene_graph_symbolic=scene_graph_symbolic,
            metadata=metadata or {},
        )


class _DiskFrameSequence(Sequence[EpisodeFrame]):
    """Load spooled RGB-D frames one at a time during dataset commit."""

    def __init__(
        self,
        paths: list[Path],
        annotations: list[tuple[int, str, bool]],
    ) -> None:
        self.paths = paths
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> EpisodeFrame:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with np.load(self.paths[index], allow_pickle=False) as archive:
            observation = {
                key.removeprefix("observation__"): np.asarray(archive[key])
                for key in archive.files
                if key.startswith("observation__")
            }
            action = np.asarray(archive["action"])
        command_index, command_text, command_completed = self.annotations[index]
        return EpisodeFrame(
            observation=observation,
            action=action,
            command_index=command_index,
            command_text=command_text,
            command_completed=command_completed,
        )


class DiskEpisodeRecorder:
    """Spool large RGB-D observations to disk while retaining small states.

    Use this recorder for real collection. It has the same callback and finish
    interface as :class:`EpisodeRecorder`, but bounds rollout RAM by loading
    only one frame at a time when the successful episode is committed.
    """

    def __init__(self, seed: int, initial_simulator_state: np.ndarray) -> None:
        """Create an attempt-local temporary spool directory."""
        self.seed = seed
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="robocasa-task-attempt-"
        )
        self._directory = Path(self._temporary_directory.name)
        self._paths: list[Path] = []
        self._annotations: list[tuple[int, str, bool]] = []
        self._states = [np.asarray(initial_simulator_state, dtype=np.float64).copy()]

    @property
    def frame_count(self) -> int:
        """Return the number of spooled actions."""
        return len(self._paths)

    @property
    def simulator_state_index(self) -> int:
        """Return the current index in the N+1 simulator-state trace."""
        return len(self._states) - 1

    def record_oracle_transition(self, transition: Any) -> None:
        """Spool one oracle transition without retaining its RGB-D arrays."""
        if transition.command_index is None or transition.command_text is None:
            raise ValueError("Recorded oracle transitions must belong to a task command")
        frame_path = self._directory / f"frame_{len(self._paths):08d}.npz"
        arrays = {
            f"observation__{key}": (
                depth_meters_to_uint16(value)
                if key.endswith("_depth")
                else np.asarray(value)
            )
            for key, value in transition.observation.items()
        }
        np.savez_compressed(frame_path, action=np.asarray(transition.action), **arrays)
        self._paths.append(frame_path)
        self._annotations.append(
            (transition.command_index, transition.command_text, False)
        )
        self._states.append(
            np.asarray(transition.next_simulator_state, dtype=np.float64).copy()
        )

    def mark_command_completed(self, command_index: int) -> None:
        """Attach a sparse reward to the latest spooled transition."""
        if not self._annotations or self._annotations[-1][0] != command_index:
            raise ValueError(
                f"No latest transition exists for command index {command_index}"
            )
        _, command_text, _ = self._annotations[-1]
        self._annotations[-1] = (command_index, command_text, True)

    def finish(
        self,
        *,
        model_xml: str,
        camera_calibration: Mapping[str, Mapping[str, np.ndarray]],
        scene_graph_ground_truth: Mapping[str, Any] | Sequence[Any],
        scene_graph_symbolic: Mapping[str, Any] | Sequence[Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeRecord:
        """Create a lazy successful episode backed by the temporary spool."""
        return EpisodeRecord(
            seed=self.seed,
            frames=_DiskFrameSequence(self._paths, self._annotations),
            simulator_states=np.stack(self._states),
            model_xml=model_xml,
            camera_calibration=camera_calibration,
            scene_graph_ground_truth=scene_graph_ground_truth,
            scene_graph_symbolic=scene_graph_symbolic,
            metadata=metadata or {},
        )

    def close(self) -> None:
        """Remove all attempt-local spooled frames."""
        self._temporary_directory.cleanup()

    def __enter__(self) -> DiskEpisodeRecorder:
        """Return this recorder as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Remove spooled frames when the attempt ends."""
        self.close()


def build_lerobot_features(
    scene_config: SceneConfig,
    robot_state_size: int,
    action_size: int,
    environment_state_size: int,
) -> dict[str, dict[str, Any]]:
    """Build native LeRobot v3 features for one configured scene.

    Numeric depth is stored as a custom two-dimensional ``uint16`` feature,
    which LeRobot maps to a Hugging Face ``Array2D`` column.
    """
    camera_group = scene_config.cameras
    if camera_group is None or not camera_group.depth:
        raise ValueError("Task datasets require depth-enabled configured cameras")
    if scene_config.robot != "PandaOmron":
        raise ValueError("Task dataset collection currently supports PandaOmron only")
    if min(robot_state_size, action_size, environment_state_size) <= 0:
        raise ValueError("State and action dimensions must be positive")

    features: dict[str, dict[str, Any]] = {}
    for camera_name in camera_group.names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (camera_group.height, camera_group.width, 3),
            "names": ["height", "width", "channel"],
        }
        features[f"observation.depth.{camera_name}"] = {
            "dtype": "uint16",
            "shape": (camera_group.height, camera_group.width),
            "names": ["height", "width"],
        }

    object_state_names = [
        f"{obj.name}.{component}"
        for obj in scene_config.objects
        # RoboCasa observation quaternions use xyzw order.
        for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
    ]
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (robot_state_size,),
                "names": None,
            },
            "observation.object_state": {
                "dtype": "float32",
                "shape": (len(object_state_names),),
                "names": object_state_names,
            },
            "observation.environment_state": {
                "dtype": "float64",
                "shape": (environment_state_size,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (action_size,),
                "names": None,
            },
            "annotation.command_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "annotation.active_command": {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
        }
    )
    return features


def get_robot_state_layout(observation: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return stable robot state keys in documented suffix order."""
    robot_prefixes = sorted(
        key[: -len("joint_pos")]
        for key in observation
        if key.endswith("joint_pos")
    )
    keys = [
        f"{prefix}{suffix}"
        for prefix in robot_prefixes
        for suffix in ROBOT_STATE_SUFFIXES
        if f"{prefix}{suffix}" in observation
    ]
    if not keys:
        raise ValueError("Observation contains no supported robot state fields")
    return tuple(keys)


def flatten_robot_state(
    observation: Mapping[str, np.ndarray], layout: Sequence[str]
) -> np.ndarray:
    """Flatten robot fields according to a fixed dataset-level layout."""
    missing = [key for key in layout if key not in observation]
    if missing:
        raise ValueError(f"Observation is missing robot state fields: {missing}")
    return np.concatenate(
        [np.asarray(observation[key]).reshape(-1) for key in layout]
    ).astype(np.float32)


def flatten_object_state(
    observation: Mapping[str, np.ndarray], scene_config: SceneConfig
) -> np.ndarray:
    """Flatten object position and quaternion in scene-config order."""
    values: list[np.ndarray] = []
    for obj in scene_config.objects:
        for suffix in ("pos", "quat"):
            key = f"{obj.name}_{suffix}"
            if key not in observation:
                raise ValueError(f"Observation is missing object state field '{key}'")
            values.append(np.asarray(observation[key]).reshape(-1))
    return np.concatenate(values).astype(np.float32)


def depth_meters_to_uint16(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth to millimeters with finite uint16 saturation."""
    metric_depth = np.asarray(depth)
    if metric_depth.dtype == np.uint16:
        if metric_depth.ndim != 2:
            raise ValueError(
                f"Millimeter depth must have shape (H, W), got {depth.shape}"
            )
        return metric_depth.copy()
    if metric_depth.ndim == 3 and metric_depth.shape[-1] == 1:
        metric_depth = metric_depth[..., 0]
    if metric_depth.ndim != 2:
        raise ValueError(
            f"Depth must have shape (H, W) or (H, W, 1), got {depth.shape}"
        )
    millimeters = np.nan_to_num(
        metric_depth * DEPTH_METERS_TO_MILLIMETERS,
        nan=0.0,
        posinf=float(MAX_DEPTH_MILLIMETERS),
        neginf=0.0,
    )
    return np.rint(np.clip(millimeters, 0, MAX_DEPTH_MILLIMETERS)).astype(np.uint16)


class TaskDatasetWriter:
    """Commit successful task rollouts to a finalized LeRobot v3 dataset."""

    def __init__(
        self,
        output_dir: str | Path,
        task_name: str,
        scene_config: SceneConfig,
        initial_observation: Mapping[str, np.ndarray],
        action_size: int,
        environment_state_size: int,
        *,
        task_snapshot: Mapping[str, Any],
        scene_snapshot: Mapping[str, Any],
        backend: DatasetBackend | None = None,
    ) -> None:
        """Create a new, non-resumable task dataset."""
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(
                "Dataset output already exists; refusing to overwrite: "
                f"{self.output_dir}"
            )
        self.task_name = task_name
        self.scene_config = scene_config
        self.robot_state_layout = get_robot_state_layout(initial_observation)
        robot_state_size = flatten_robot_state(
            initial_observation, self.robot_state_layout
        ).size
        self.features = build_lerobot_features(
            scene_config,
            robot_state_size,
            action_size,
            environment_state_size,
        )
        if backend is None:
            backend = self._create_lerobot_backend()
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self._episode_count = 0
        self._finalized = False
        self._write_dataset_metadata(task_snapshot, scene_snapshot)

    def _create_lerobot_backend(self) -> DatasetBackend:
        require_supported_lerobot()
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError(
                "Task collection requires lerobot==0.4.4. Install RoboCasa's "
                "declared dependencies before collecting data."
            ) from error
        return LeRobotDataset.create(
            repo_id=self.task_name,
            root=self.output_dir,
            robot_type=self.scene_config.robot,
            fps=self.scene_config.control_freq,
            features=self.features,
            use_videos=True,
            streaming_encoding=True,
            vcodec="h264",
        )

    @property
    def episode_count(self) -> int:
        """Return the number of committed successful episodes."""
        return self._episode_count

    def save_episode(self, episode: EpisodeRecord) -> int:
        """Validate and atomically commit one successful episode."""
        self._ensure_open()
        if not episode.frames:
            raise ValueError("Cannot save an empty episode")
        states = np.asarray(episode.simulator_states)
        if states.ndim != 2 or states.shape[0] != len(episode.frames) + 1:
            raise ValueError(
                "simulator_states must be a 2-D N+1 array with one successor "
                "state per frame"
            )
        expected_state_size = self.features["observation.environment_state"]["shape"][0]
        if states.shape[1] != expected_state_size:
            raise ValueError(
                f"Simulator state width is {states.shape[1]}, expected "
                f"{expected_state_size}"
            )
        self._validate_graph_alignment(episode)

        episode_index = self._episode_count
        episode_dir = self.output_dir / "extras" / f"episode_{episode_index:06d}"
        try:
            for frame_index, frame in enumerate(episode.frames):
                self.backend.add_frame(
                    self._build_frame(
                        frame,
                        states[frame_index],
                        frame_index,
                        len(episode.frames),
                    )
                )
            self._write_episode_extras(episode_index, episode)
            self.backend.save_episode()
            self._episode_count += 1
            return episode_index
        except Exception:
            self.backend.clear_episode_buffer(delete_images=True)
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
            raise

    def _validate_graph_alignment(self, episode: EpisodeRecord) -> None:
        """Validate command rewards and their aligned graph boundaries."""
        completed_boundaries: dict[int, int] = {}
        command_indices: set[int] = set()
        for frame_index, frame in enumerate(episode.frames):
            command_indices.add(frame.command_index)
            if frame.command_completed:
                if frame.command_index in completed_boundaries:
                    raise ValueError(
                        f"Command {frame.command_index} has multiple completion frames"
                    )
                completed_boundaries[frame.command_index] = frame_index + 1

        expected_commands = list(range(len(command_indices)))
        if sorted(command_indices) != expected_commands:
            raise ValueError("Episode command indices must be consecutive from zero")
        if sorted(completed_boundaries) != expected_commands:
            raise ValueError("Every episode command must have one completion frame")
        expected_state_indices = [0] + [
            completed_boundaries[index] for index in expected_commands
        ]

        for label, trace in (
            ("ground-truth", episode.scene_graph_ground_truth),
            ("symbolic", episode.scene_graph_symbolic),
        ):
            if not isinstance(trace, Mapping) or not isinstance(
                trace.get("snapshots"), Sequence
            ):
                raise ValueError(f"{label} scene graph must contain snapshots")
            snapshots = trace["snapshots"]
            if len(snapshots) != len(expected_state_indices):
                raise ValueError(
                    f"{label} scene graph must have one initial and one snapshot "
                    "per command"
                )
            for stage_index, (snapshot, state_index) in enumerate(
                zip(snapshots, expected_state_indices)
            ):
                if not isinstance(snapshot, Mapping):
                    raise ValueError(f"{label} scene graph snapshots must be mappings")
                if snapshot.get("stage_index") != stage_index:
                    raise ValueError(f"{label} scene graph stages must be consecutive")
                if snapshot.get("simulator_state_index") != state_index:
                    raise ValueError(
                        f"{label} scene graph stage {stage_index} is not aligned "
                        "with its command-boundary simulator state"
                    )

    def discard_episode(self) -> None:
        """Discard an unsuccessful in-progress backend episode."""
        self._ensure_open()
        self.backend.clear_episode_buffer(delete_images=True)

    def log_failure(self, seed: int, error: BaseException | str) -> None:
        """Append a rejected collection attempt to ``extras/failures.jsonl``."""
        failure_path = self.output_dir / "extras" / "failures.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "attempt_seed": seed,
            "error_type": (
                type(error).__name__
                if isinstance(error, BaseException)
                else "Error"
            ),
            "message": str(error),
        }
        with failure_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def finalize(self) -> None:
        """Finalize the v3 parquet and metadata writers exactly once."""
        if self._finalized:
            return
        self.backend.finalize()
        self._finalized = True

    def __enter__(self) -> TaskDatasetWriter:
        """Return this writer as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Finalize committed episodes on controlled shutdown."""
        self.finalize()

    def _build_frame(
        self,
        frame: EpisodeFrame,
        simulator_state: np.ndarray,
        frame_index: int,
        episode_length: int,
    ) -> dict[str, Any]:
        observation = frame.observation
        item: dict[str, Any] = {}
        assert self.scene_config.cameras is not None
        for camera_name in self.scene_config.cameras.names:
            image_key = f"{camera_name}_image"
            depth_key = f"{camera_name}_depth"
            item[f"observation.images.{camera_name}"] = np.asarray(
                observation[image_key], dtype=np.uint8
            )
            item[f"observation.depth.{camera_name}"] = depth_meters_to_uint16(
                observation[depth_key]
            )
        item.update(
            {
                "observation.state": flatten_robot_state(
                    observation, self.robot_state_layout
                ),
                "observation.object_state": flatten_object_state(
                    observation, self.scene_config
                ),
                "observation.environment_state": np.asarray(
                    simulator_state, dtype=np.float64
                ),
                "action": np.asarray(frame.action, dtype=np.float32),
                "annotation.command_index": np.asarray(
                    [frame.command_index], dtype=np.int64
                ),
                "annotation.active_command": frame.command_text,
                "next.reward": np.asarray(
                    [float(frame.command_completed)], dtype=np.float32
                ),
                "next.done": np.asarray(
                    [frame_index == episode_length - 1], dtype=bool
                ),
                "task": self.task_name,
            }
        )
        return item

    def _write_dataset_metadata(
        self,
        task_snapshot: Mapping[str, Any],
        scene_snapshot: Mapping[str, Any],
    ) -> None:
        extras = self.output_dir / "extras"
        extras.mkdir(parents=True, exist_ok=True)
        _write_json(extras / "task.json", task_snapshot)
        _write_json(extras / "scene.json", scene_snapshot)
        _write_json(
            extras / "schema.json",
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "frame_alignment": "observation_t_action_t",
                "simulator_states": "initial_plus_post_action_successors",
                "depth_unit": "millimeter",
                "robot_state_layout": list(self.robot_state_layout),
                "object_state_layout": [obj.name for obj in self.scene_config.objects],
            },
        )

    def _write_episode_extras(self, episode_index: int, episode: EpisodeRecord) -> None:
        episode_dir = self.output_dir / "extras" / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(episode_dir / "states.npz", states=episode.simulator_states)
        with gzip.open(episode_dir / "model.xml.gz", "wt", encoding="utf-8") as stream:
            stream.write(episode.model_xml)
        np.savez_compressed(
            episode_dir / "camera_calibration.npz",
            **{
                f"{camera_name}.{matrix_name}": np.asarray(matrix)
                for camera_name, calibration in episode.camera_calibration.items()
                for matrix_name, matrix in calibration.items()
            },
        )
        _write_json(
            episode_dir / "episode.json",
            {
                "seed": episode.seed,
                "num_frames": len(episode.frames),
                "metadata": episode.metadata,
            },
        )
        _write_json(
            episode_dir / "scene_graph_gt.json", episode.scene_graph_ground_truth
        )
        _write_json(
            episode_dir / "scene_graph_symbolic.json", episode.scene_graph_symbolic
        )

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("Dataset writer has already been finalized")


def _write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON while accepting NumPy values."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")
