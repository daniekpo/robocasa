"""Collect successful task-oracle demonstrations as a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
import cv2
import numpy as np

from robocasa.data_collection.task_dataset import (
    DiskEpisodeRecorder,
    TaskDatasetWriter,
    require_supported_lerobot,
)
from robocasa.example_env import SceneGymEnv, make_scene_env
from robocasa.oracles import MotionConfig, TaskOracle
from robocasa.scene_config import resolve_scene_config
from robocasa.scene_graph import (
    SceneGraphTrace,
    build_ground_truth_scene_graph,
    update_symbolic_scene_graph,
)
from robocasa.task_config import load_task_config, resolve_task_config

LOGGER = logging.getLogger(__name__)


class LiveCameraViewer:
    """Display one configured RGB camera while the oracle is running."""

    def __init__(self, camera_name: str) -> None:
        """Create a named OpenCV window for one scene camera."""
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError(
                "--view requires a graphical display (DISPLAY or WAYLAND_DISPLAY)"
            )
        self.camera_name = camera_name
        self.window_name = f"RoboCasa oracle: {camera_name}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def append(
        self,
        observation: dict[str, np.ndarray],
        stage: str,
        overlays: object = None,
    ) -> None:
        """Show the newest RGB observation and current oracle stage."""
        del overlays
        image = np.asarray(observation[f"{self.camera_name}_image"]).copy()
        cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            image,
            stage,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt("Live viewing stopped by user")

    def close(self) -> None:
        """Close the live camera window."""
        cv2.destroyWindow(self.window_name)


def collect_task_demos(
    task_reference: str | Path,
    output_dir: str | Path,
    num_demos: int,
    base_seed: int,
    *,
    max_attempts: int | None = None,
    detection_cameras: tuple[str, ...] | None = None,
    view: bool = False,
    view_camera: str | None = None,
) -> int:
    """Collect a fixed number of successful demonstrations for one task."""
    require_supported_lerobot()
    if view and not os.environ.get("DISPLAY") and not os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        raise RuntimeError(
            "--view requires a graphical display (DISPLAY or WAYLAND_DISPLAY)"
        )
    if num_demos <= 0:
        raise ValueError("num_demos must be positive")
    attempt_limit = max_attempts if max_attempts is not None else 3 * num_demos
    if attempt_limit < num_demos:
        raise ValueError("max_attempts must be at least num_demos")
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    task_path = resolve_task_config(task_reference)
    task_config = load_task_config(task_path)
    if view and view_camera is not None:
        camera_group = task_config.scene_config.cameras
        camera_names = camera_group.names if camera_group is not None else ()
        if view_camera not in camera_names:
            raise ValueError(
                f"Unknown view camera '{view_camera}'. Expected one of: "
                + ", ".join(camera_names)
            )
    task_snapshot = _load_mapping(task_path)
    scene_snapshot = _load_scene_snapshot(task_path, task_snapshot["scene_name"])
    motion_config = MotionConfig()
    horizon = len(task_config.commands) * (
        6 * motion_config.max_waypoint_steps
        + 2 * motion_config.gripper_settle_steps
    ) + 100

    writer: TaskDatasetWriter | None = None
    successful_demos = 0
    try:
        for attempt_index in range(attempt_limit):
            if successful_demos >= num_demos:
                break
            seed = base_seed + attempt_index
            # Seed construction as well as reset: random object styles can be
            # selected while the MuJoCo model is being built.
            attempt_scene = replace(task_config.scene_config, seed=seed)
            env = make_scene_env(attempt_scene, horizon=horizon)
            if not isinstance(env, SceneGymEnv):
                raise TypeError("make_scene_env() did not return SceneGymEnv")
            viewer: LiveCameraViewer | None = None
            try:
                observation, _ = env.reset(seed=seed)
                initial_state = env.sim.get_state().flatten().copy()
                if writer is None:
                    writer = TaskDatasetWriter(
                        output_path,
                        task_config.name,
                        task_config.scene_config,
                        observation,
                        action_size=env.action_space.shape[0],
                        environment_state_size=initial_state.size,
                        task_snapshot=task_snapshot,
                        scene_snapshot=scene_snapshot,
                    )
                with DiskEpisodeRecorder(seed, initial_state) as recorder:
                    cameras = detection_cameras or _default_detection_cameras(env)
                    if view:
                        camera_name = _resolve_view_camera(env, view_camera)
                        viewer = LiveCameraViewer(camera_name)
                        viewer.append(observation, f"seed {seed}: initial scene")
                    oracle = TaskOracle(
                        env,
                        cameras,
                        motion_config=motion_config,
                        recorder=viewer,
                        transition_callback=recorder.record_oracle_transition,
                    )

                    initial_graph = build_ground_truth_scene_graph(
                        env,
                        task_config,
                        simulator_state_index=recorder.simulator_state_index,
                    )
                    ground_truth_trace = SceneGraphTrace(
                        "ground_truth", (initial_graph,)
                    )
                    symbolic_trace = SceneGraphTrace("symbolic", (initial_graph,))

                    for command_index, command in enumerate(task_config.commands):
                        observation, _ = oracle.execute_command(
                            observation, command, command_index
                        )
                        recorder.mark_command_completed(command_index)
                        state_index = recorder.simulator_state_index
                        ground_truth_trace = ground_truth_trace.append(
                            build_ground_truth_scene_graph(
                                env,
                                task_config,
                                stage_index=command_index + 1,
                                simulator_state_index=state_index,
                                completed_command_index=command_index,
                                completed_command=command,
                            )
                        )
                        symbolic_trace = symbolic_trace.append(
                            update_symbolic_scene_graph(
                                symbolic_trace.snapshots[-1],
                                command,
                                command_index=command_index,
                                simulator_state_index=state_index,
                            )
                        )

                    episode = recorder.finish(
                        model_xml=env.model.get_xml(),
                        camera_calibration=env.get_camera_calibration(),
                        scene_graph_ground_truth=ground_truth_trace.to_dict(),
                        scene_graph_symbolic=symbolic_trace.to_dict(),
                        metadata={"attempt_index": attempt_index},
                    )
                    writer.save_episode(episode)
                successful_demos += 1
                LOGGER.info(
                    "Collected demonstration %d/%d with seed %d",
                    successful_demos,
                    num_demos,
                    seed,
                )
            except Exception as error:
                LOGGER.exception("Rejected task rollout with seed %d", seed)
                if writer is None:
                    raise
                writer.discard_episode()
                writer.log_failure(seed, error)
            finally:
                if viewer is not None:
                    viewer.close()
                env.close()
    finally:
        if writer is not None:
            writer.finalize()

    if successful_demos != num_demos:
        raise RuntimeError(
            f"Collected {successful_demos}/{num_demos} successful demonstrations "
            f"after {attempt_limit} attempts"
        )
    return successful_demos


def _default_detection_cameras(env: SceneGymEnv) -> tuple[str, ...]:
    camera_group = env.config.cameras
    if camera_group is None:
        raise ValueError("Task collection requires configured cameras")
    return camera_group.names[:2]


def _resolve_view_camera(env: SceneGymEnv, requested: str | None) -> str:
    camera_group = env.config.cameras
    if camera_group is None:
        raise ValueError("Live viewing requires configured cameras")
    camera_name = requested or camera_group.names[0]
    if camera_name not in camera_group.names:
        raise ValueError(
            f"Unknown view camera '{camera_name}'. Expected one of: "
            + ", ".join(camera_group.names)
        )
    return camera_name


def _load_scene_snapshot(task_path: Path, scene_reference: str) -> dict[str, Any]:
    candidate = Path(scene_reference).expanduser()
    if not candidate.is_absolute():
        relative_candidate = task_path.parent / candidate
        if relative_candidate.is_file():
            candidate = relative_candidate
        else:
            candidate = resolve_scene_config(scene_reference)
    return _load_mapping(candidate)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            value = json.load(stream)
        else:
            value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def main() -> None:
    """Parse command-line options and collect task demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task YAML path or bundled name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-demos", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--detection-cameras", nargs="+")
    parser.add_argument(
        "--view",
        action="store_true",
        help="Show a live RGB camera window; press q to stop",
    )
    parser.add_argument(
        "--view-camera",
        help="Camera shown by --view (defaults to the first scene camera)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    collect_task_demos(
        args.task,
        args.output,
        args.num_demos,
        args.seed,
        max_attempts=args.max_attempts,
        detection_cameras=(
            tuple(args.detection_cameras) if args.detection_cameras else None
        ),
        view=args.view,
        view_camera=args.view_camera,
    )


if __name__ == "__main__":
    main()
