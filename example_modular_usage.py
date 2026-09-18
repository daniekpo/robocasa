"""Detect, pick, and place an object with reusable modular oracle components.

Replace ``GroundTruthObjectDetector`` with an implementation of
``ObjectDetector`` to retain localization, motion, and transition recording
while using learned perception.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from robocasa.example_env import SceneGymEnv, make_scene_env
from robocasa.oracles.pick_place import (
    Detection,
    GroundTruthObjectDetector,
    MotionConfig,
    MotionExecutor,
    ObjectDetector,
    ObjectEstimate,
    TaskOracle,
    ViewEstimate,
    get_detection_center,
    get_object_contact_pairs,
    localize_object,
    project_pixel_to_3d,
    warn_about_initial_object_contacts,
    world_error_to_delta_action,
)
from robocasa.scene_config import load_scene_config

DEFAULT_SCENE = "expanded_example_scene"
DEFAULT_SOURCE_OBJECT = "can"
DEFAULT_DESTINATION_OBJECT = "basket"
DEFAULT_DETECTION_CAMERAS = ("robot_left", "robot_right")
DEFAULT_VIDEO_PATH = Path("outputs/modular_pick_place.mp4")
VIDEO_TILE_WIDTH = 640

LOGGER = logging.getLogger(__name__)

# Keep the original example's reusable symbols importable for downstream code.
__all__ = [
    "Detection",
    "GroundTruthObjectDetector",
    "MotionConfig",
    "MotionExecutor",
    "ObjectDetector",
    "ObjectEstimate",
    "TaskOracle",
    "ViewEstimate",
    "build_detection_overlays",
    "create_camera_grid",
    "get_detection_center",
    "get_object_contact_pairs",
    "localize_object",
    "project_pixel_to_3d",
    "run_modular_pick_and_place",
    "warn_about_initial_object_contacts",
    "world_error_to_delta_action",
]


@dataclass(frozen=True)
class DemoMoveCommand:
    """Minimal move command used by this standalone example."""

    object: str
    relation: str
    target: str
    action: str = "move"

    def __str__(self) -> str:
        """Return the task-YAML command notation."""
        return f"move({self.object}, {self.relation}, {self.target})"


class VideoRecorder:
    """Write labeled multi-camera observations to an MP4 file."""

    def __init__(
        self, output_path: Path, fps: int, camera_names: tuple[str, ...]
    ) -> None:
        """Create an imageio writer and remember the camera display order."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.camera_names = camera_names
        self._writer = imageio.get_writer(output_path, fps=fps)

    def append(
        self,
        observation: dict[str, np.ndarray],
        stage: str,
        overlays: dict[str, list[tuple[str, Detection, tuple[int, int, int]]]]
        | None = None,
    ) -> None:
        """Append one labeled grid frame."""
        self._writer.append_data(
            create_camera_grid(
                observation, self.camera_names, stage, overlays=overlays
            )
        )

    def close(self) -> None:
        """Flush and close the output video."""
        self._writer.close()


def create_camera_grid(
    observation: dict[str, np.ndarray],
    camera_names: tuple[str, ...],
    stage: str,
    overlays: dict[str, list[tuple[str, Detection, tuple[int, int, int]]]]
    | None = None,
) -> np.ndarray:
    """Create a compact RGB grid from configured camera observations."""
    tiles: list[np.ndarray] = []
    for camera_name in camera_names:
        image = observation[f"{camera_name}_image"].copy()
        for label, detection, color in (overlays or {}).get(camera_name, []):
            if detection.mask is not None:
                tint = np.zeros_like(image)
                tint[detection.mask] = color
                image = cv2.addWeighted(image, 1.0, tint, 0.35, 0.0)
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            center_x, center_y = get_detection_center(detection)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
            cv2.circle(image, (center_x, center_y), 6, color, -1)
            cv2.putText(
                image,
                label,
                (x_min, max(24, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        source_height, source_width = image.shape[:2]
        tile_height = round(source_height * VIDEO_TILE_WIDTH / source_width)
        tile = cv2.resize(image, (VIDEO_TILE_WIDTH, tile_height))
        cv2.putText(
            tile,
            camera_name,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    columns = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    blank_tile = np.zeros_like(tiles[0])
    tiles.extend(blank_tile.copy() for _ in range(rows * columns - len(tiles)))
    grid = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    cv2.rectangle(grid, (0, 0), (grid.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        grid,
        stage,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return grid


def build_detection_overlays(
    estimates: tuple[ObjectEstimate, ...],
) -> dict[str, list[tuple[str, Detection, tuple[int, int, int]]]]:
    """Collect detections for optional video annotation."""
    colors = ((255, 80, 80), (80, 255, 80))
    overlays: dict[str, list[tuple[str, Detection, tuple[int, int, int]]]] = {}
    for estimate, color in zip(estimates, colors):
        for view in estimate.views:
            overlays.setdefault(view.camera_name, []).append(
                (estimate.object_name, view.detection, color)
            )
    return overlays


def run_modular_pick_and_place(
    scene: str,
    seed: int,
    source_object: str,
    destination_object: str,
    detection_cameras: tuple[str, ...],
    video_path: Path,
) -> None:
    """Run one ``in`` command with the reusable task oracle."""
    motion_config = MotionConfig()
    horizon = (
        7 * motion_config.max_waypoint_steps
        + 2 * motion_config.gripper_settle_steps
        + 100
    )
    scene_config = replace(load_scene_config(scene), seed=seed)
    env = make_scene_env(scene_config, horizon=horizon)
    if not isinstance(env, SceneGymEnv):
        raise TypeError("make_scene_env() did not return a Gymnasium environment")
    recorder: VideoRecorder | None = None
    try:
        observation, _ = env.reset(seed=seed)
        warn_about_initial_object_contacts(env)
        camera_group = env.config.cameras
        if camera_group is None:
            raise ValueError("The scene must define cameras")
        recorder = VideoRecorder(video_path, env.config.control_freq, camera_group.names)
        oracle = TaskOracle(
            env,
            detection_cameras,
            motion_config=motion_config,
            recorder=recorder,
        )
        oracle.execute_command(
            observation,
            DemoMoveCommand(source_object, "in", destination_object),
            command_index=0,
        )
    finally:
        if recorder is not None:
            recorder.close()
            LOGGER.info("Saved video to %s", recorder.output_path)
        env.close()


def main() -> None:
    """Parse command-line arguments and run the modular example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-object", default=DEFAULT_SOURCE_OBJECT)
    parser.add_argument("--destination-object", default=DEFAULT_DESTINATION_OBJECT)
    parser.add_argument(
        "--detection-cameras",
        nargs="+",
        default=list(DEFAULT_DETECTION_CAMERAS),
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_modular_pick_and_place(
        scene=args.scene,
        seed=args.seed,
        source_object=args.source_object,
        destination_object=args.destination_object,
        detection_cameras=tuple(args.detection_cameras),
        video_path=args.video,
    )


if __name__ == "__main__":
    main()
