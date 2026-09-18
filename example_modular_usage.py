"""Detect, pick, and place an object with modular RGB-D components.

The ground-truth detector is the only simulator-specific perception component.
Replace ``GroundTruthObjectDetector`` with an implementation of
``ObjectDetector`` to use a learned detector while retaining the localization,
motion, and recording code.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from robosuite.utils.camera_utils import get_camera_segmentation

from robocasa.example_env import SceneGymEnv, make_scene_env
from robocasa.utils.object_utils import check_obj_in_receptacle

DEFAULT_SCENE = "expanded_example_scene"
DEFAULT_SOURCE_OBJECT = "can"
DEFAULT_DESTINATION_OBJECT = "basket"
DEFAULT_DETECTION_CAMERAS = ("robot_left", "robot_right")
DEFAULT_VIDEO_PATH = Path("outputs/modular_pick_place.mp4")
VIDEO_TILE_WIDTH = 640

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One object detection in an upright RGB image.

    Args:
        bbox_xyxy: Inclusive ``(x_min, y_min, x_max, y_max)`` pixel bounds.
        mask: Optional boolean segmentation mask. Localization prefers the mask
            center and falls back to the bounding-box center when absent.
    """

    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray | None = None


class ObjectDetector(Protocol):
    """Interface downstream users implement for their object detector."""

    def detect(
        self, image: np.ndarray, camera_name: str, object_name: str
    ) -> Detection:
        """Detect a named object in one RGB image."""


@dataclass(frozen=True)
class ViewEstimate:
    """Detection and reconstructed point from one camera."""

    camera_name: str
    detection: Detection
    pixel_xy: tuple[int, int]
    camera_point: np.ndarray
    world_point: np.ndarray


@dataclass(frozen=True)
class ObjectEstimate:
    """Multi-view world position and its individual camera estimates."""

    object_name: str
    world_point: np.ndarray
    views: tuple[ViewEstimate, ...]


@dataclass(frozen=True)
class MotionConfig:
    """Small set of motion parameters for the demonstrated scene."""

    approach_height: float = 0.20
    grasp_overlap: float = 0.01
    drop_clearance: float = 0.20
    position_tolerance: float = 0.012
    grasp_tolerance: float = 0.020
    max_waypoint_steps: int = 120
    gripper_settle_steps: int = 30


class GroundTruthObjectDetector:
    """Generate instance masks from MuJoCo's ground-truth geom renderer."""

    def __init__(self, env: SceneGymEnv):
        """Store the configured environment used for segmentation rendering."""
        self.env = env
        self._geom_ids: dict[str, np.ndarray] = {}

    def detect(
        self, image: np.ndarray, camera_name: str, object_name: str
    ) -> Detection:
        """Return the visible mask and bounding box for a configured object."""
        height, width = image.shape[:2]
        segmentation = get_camera_segmentation(self.env.sim, camera_name, height, width)
        mask = (segmentation[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(
            segmentation[..., 1], self._get_geom_ids(object_name)
        )
        rows, columns = np.nonzero(mask)
        if columns.size == 0:
            raise RuntimeError(
                f"Object '{object_name}' is not visible in camera '{camera_name}'"
            )
        bbox = (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        )
        return Detection(bbox_xyxy=bbox, mask=mask)

    def _get_geom_ids(self, object_name: str) -> np.ndarray:
        """Return geom IDs belonging to the object's complete body subtree."""
        if object_name in self._geom_ids:
            return self._geom_ids[object_name]
        if object_name not in self.env.objects:
            raise ValueError(f"Unknown configured object: {object_name}")

        model = self.env.sim.model
        root_body_id = model.body_name2id(self.env.objects[object_name].root_body)
        body_ids: list[int] = []
        for body_id in range(model.nbody):
            ancestor_id = body_id
            while ancestor_id != 0:
                if ancestor_id == root_body_id:
                    body_ids.append(body_id)
                    break
                ancestor_id = int(model.body_parentid[ancestor_id])
        geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, body_ids))
        self._geom_ids[object_name] = geom_ids
        return geom_ids


def get_detection_center(detection: Detection) -> tuple[int, int]:
    """Return an ``(x, y)`` center pixel guaranteed to lie on the mask."""
    if detection.mask is not None:
        rows, columns = np.nonzero(detection.mask)
        if columns.size == 0:
            raise ValueError("Detection mask must contain at least one pixel")
        center_x = float(columns.mean())
        center_y = float(rows.mean())
        closest_index = int(
            np.argmin((columns - center_x) ** 2 + (rows - center_y) ** 2)
        )
        return int(columns[closest_index]), int(rows[closest_index])

    x_min, y_min, x_max, y_max = detection.bbox_xyxy
    if x_min > x_max or y_min > y_max:
        raise ValueError("Detection bounding box has inverted bounds")
    return round((x_min + x_max) / 2), round((y_min + y_max) / 2)


def project_pixel_to_3d(
    pixel_xy: tuple[int, int],
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one RGB-D pixel into camera and world coordinates."""
    pixel_x, pixel_y = pixel_xy
    metric_depth = float(depth_map[pixel_y, pixel_x])
    if not np.isfinite(metric_depth) or metric_depth <= 0:
        raise ValueError(f"Depth at pixel {pixel_xy} must be finite and positive")

    image_ray = np.linalg.solve(
        intrinsics, np.asarray([pixel_x, pixel_y, 1.0], dtype=np.float64)
    )
    camera_point = image_ray * metric_depth
    homogeneous_point = np.concatenate((camera_point, [1.0]))
    world_point = (camera_to_world @ homogeneous_point)[:3]
    return camera_point, world_point


def world_error_to_delta_action(
    world_error: np.ndarray,
    base_rotation: np.ndarray,
    position_scale: np.ndarray,
) -> np.ndarray:
    """Convert a world-frame position error to a clipped OSC pose action."""
    base_error = base_rotation.T @ world_error
    action = np.zeros(6, dtype=np.float64)
    action[:3] = np.clip(base_error / position_scale, -1.0, 1.0)
    return action


def get_object_contact_pairs(env: SceneGymEnv) -> tuple[tuple[str, str], ...]:
    """Read configured object-object pairs from MuJoCo's current contacts."""
    geom_to_object: dict[int, str] = {}
    for object_name, obj in env.objects.items():
        for geom_name in obj.contact_geoms:
            geom_to_object[env.sim.model.geom_name2id(geom_name)] = object_name

    pairs: set[tuple[str, str]] = set()
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        first = geom_to_object.get(int(contact.geom1))
        second = geom_to_object.get(int(contact.geom2))
        if first is not None and second is not None and first != second:
            pairs.add(tuple(sorted((first, second))))
    return tuple(sorted(pairs))


def warn_about_initial_object_contacts(env: SceneGymEnv) -> None:
    """Warn about allowed initial object contacts that may be unintended."""
    contact_pairs = get_object_contact_pairs(env)
    if not contact_pairs:
        return
    formatted_pairs = ", ".join(f"{first}/{second}" for first, second in contact_pairs)
    LOGGER.warning(
        "Configured objects begin in contact: %s. Contacts are allowed; adjust "
        "relationships or absolute positions if they are unintended.",
        formatted_pairs,
    )


def localize_object(
    detector: ObjectDetector,
    observation: dict[str, np.ndarray],
    calibration: dict[str, dict[str, np.ndarray]],
    camera_names: tuple[str, ...],
    object_name: str,
) -> ObjectEstimate:
    """Detect an object in each view and average its reconstructed world points."""
    views: list[ViewEstimate] = []
    for camera_name in camera_names:
        image = observation[f"{camera_name}_image"]
        depth = observation[f"{camera_name}_depth"][..., 0]
        detection = detector.detect(image, camera_name, object_name)
        pixel_xy = get_detection_center(detection)
        camera_point, world_point = project_pixel_to_3d(
            pixel_xy,
            depth,
            calibration[camera_name]["intrinsics"],
            calibration[camera_name]["camera_to_world"],
        )
        views.append(
            ViewEstimate(
                camera_name=camera_name,
                detection=detection,
                pixel_xy=pixel_xy,
                camera_point=camera_point,
                world_point=world_point,
            )
        )
        LOGGER.info(
            "%s from %s: pixel=%s camera=%s world=%s",
            object_name,
            camera_name,
            pixel_xy,
            np.array2string(camera_point, precision=3),
            np.array2string(world_point, precision=3),
        )

    fused_point = np.mean([view.world_point for view in views], axis=0)
    LOGGER.info(
        "%s fused world point: %s",
        object_name,
        np.array2string(fused_point, precision=3),
    )
    return ObjectEstimate(object_name, fused_point, tuple(views))


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
        frame = create_camera_grid(
            observation, self.camera_names, stage, overlays=overlays
        )
        self._writer.append_data(frame)

    def close(self) -> None:
        """Flush and close the output video."""
        self._writer.close()


class MotionExecutor:
    """Execute world-frame waypoints with PandaOmron's delta OSC controller."""

    def __init__(
        self,
        env: SceneGymEnv,
        recorder: VideoRecorder,
        config: MotionConfig,
    ) -> None:
        """Validate and retain the controller, recorder, and motion settings."""
        if len(env.robots) != 1:
            raise ValueError("This example expects exactly one robot")
        self.env = env
        self.robot = env.robots[0]
        self.recorder = recorder
        self.config = config
        self.arm_name = "right"
        self.gripper_name = "right_gripper"
        self.eef_position_key = f"{self.robot.robot_model.naming_prefix}eef_pos"

        controller = self.robot.composite_controller.get_controller(self.arm_name)
        if controller.input_type != "delta" or controller.input_ref_frame != "base":
            raise ValueError("This example requires a base-frame delta arm controller")
        if controller.control_dim != 6:
            raise ValueError("This example requires a six-dimensional pose controller")
        self.position_scale = np.maximum(
            np.abs(controller.output_min[:3]), np.abs(controller.output_max[:3])
        )

    def move_to_position(
        self,
        observation: dict[str, np.ndarray],
        target_world: np.ndarray,
        gripper_closed: bool,
        stage: str,
        tolerance: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Servo the end effector to a world point or raise on timeout."""
        target_world = np.asarray(target_world, dtype=np.float64)
        target_tolerance = tolerance or self.config.position_tolerance
        for step_index in range(self.config.max_waypoint_steps):
            world_error = target_world - observation[self.eef_position_key]
            if np.linalg.norm(world_error) <= target_tolerance:
                LOGGER.info("%s reached in %d steps", stage, step_index)
                return observation
            action = self._create_action(world_error, gripper_closed)
            observation, _, terminated, truncated, _ = self.env.step(action)
            self.recorder.append(observation, stage)
            if terminated or truncated:
                raise RuntimeError(f"Episode ended while executing '{stage}'")
        final_error = np.linalg.norm(target_world - observation[self.eef_position_key])
        raise RuntimeError(
            f"Waypoint '{stage}' timed out with {final_error:.3f} m error"
        )

    def hold_gripper(
        self,
        observation: dict[str, np.ndarray],
        closed: bool,
        stage: str,
    ) -> dict[str, np.ndarray]:
        """Hold the current pose while the gripper opens or closes."""
        for _ in range(self.config.gripper_settle_steps):
            action = self._create_action(np.zeros(3), closed)
            observation, _, terminated, truncated, _ = self.env.step(action)
            self.recorder.append(observation, stage)
            if terminated or truncated:
                raise RuntimeError(f"Episode ended while executing '{stage}'")
        return observation

    def _create_action(
        self, world_error: np.ndarray, gripper_closed: bool
    ) -> np.ndarray:
        """Convert a world translation error into the native robot action."""
        _, base_rotation = self.robot.composite_controller.get_controller_base_pose(
            self.arm_name
        )
        arm_action = world_error_to_delta_action(
            world_error, base_rotation, self.position_scale
        )
        gripper_action = np.asarray([1.0 if gripper_closed else -1.0])
        return self.robot.create_action_vector(
            {
                self.arm_name: arm_action,
                self.gripper_name: gripper_action,
                "base_mode": -1,
            }
        )


def create_camera_grid(
    observation: dict[str, np.ndarray],
    camera_names: tuple[str, ...],
    stage: str,
    overlays: dict[str, list[tuple[str, Detection, tuple[int, int, int]]]]
    | None = None,
) -> np.ndarray:
    """Create a compact RGB grid from all configured camera observations."""
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
    """Collect initial detections for video annotation."""
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
    """Run the complete perception, action, and recording example."""
    motion_config = MotionConfig()
    horizon = (
        5 * motion_config.max_waypoint_steps
        + 2 * motion_config.gripper_settle_steps
        + 100
    )
    env = make_scene_env(scene, horizon=horizon)
    if not isinstance(env, SceneGymEnv):
        raise TypeError("make_scene_env() did not return a Gymnasium environment")
    recorder: VideoRecorder | None = None
    success = False
    try:
        camera_group = env.config.cameras
        if camera_group is None or not camera_group.depth:
            raise ValueError("The scene must define cameras with depth enabled")
        if not detection_cameras:
            raise ValueError("At least one detection camera is required")
        unknown_cameras = sorted(set(detection_cameras) - set(camera_group.names))
        if unknown_cameras:
            raise ValueError(
                "Unknown detection camera(s): " + ", ".join(unknown_cameras)
            )

        observation, _ = env.reset(seed=seed)
        warn_about_initial_object_contacts(env)
        calibration = env.get_camera_calibration()
        detector = GroundTruthObjectDetector(env)
        source = localize_object(
            detector,
            observation,
            calibration,
            detection_cameras,
            source_object,
        )
        destination = localize_object(
            detector,
            observation,
            calibration,
            detection_cameras,
            destination_object,
        )

        recorder = VideoRecorder(
            video_path, env.config.control_freq, camera_group.names
        )
        recorder.append(
            observation,
            "detected pick and drop targets",
            build_detection_overlays((source, destination)),
        )
        motion = MotionExecutor(env, recorder, motion_config)

        safe_height = source.world_point[2] + motion_config.approach_height
        above_source = np.asarray(
            [source.world_point[0], source.world_point[1], safe_height]
        )
        grasp_position = source.world_point.copy()
        grasp_position[2] -= motion_config.grasp_overlap
        above_destination = np.asarray(
            [destination.world_point[0], destination.world_point[1], safe_height]
        )
        drop_position = destination.world_point.copy()
        drop_position[2] += motion_config.drop_clearance

        observation = motion.move_to_position(
            observation, above_source, False, "move above source"
        )
        observation = motion.move_to_position(
            observation,
            grasp_position,
            False,
            "descend to grasp",
            tolerance=motion_config.grasp_tolerance,
        )
        observation = motion.hold_gripper(observation, True, "close gripper")
        observation = motion.move_to_position(
            observation, above_source, True, "lift vertically"
        )
        observation = motion.move_to_position(
            observation, above_destination, True, "translate above destination"
        )
        observation = motion.move_to_position(
            observation, drop_position, True, "descend to drop height"
        )
        observation = motion.hold_gripper(observation, False, "open gripper")
        motion.move_to_position(
            observation, above_destination, False, "retreat vertically"
        )
        success = check_obj_in_receptacle(env.env, source_object, destination_object)
    finally:
        if recorder is not None:
            recorder.close()
            LOGGER.info("Saved video to %s", recorder.output_path)
        env.close()

    if not success:
        raise RuntimeError(
            f"Pick-and-place failed: '{source_object}' is not inside "
            f"'{destination_object}'"
        )
    LOGGER.info(
        "Success: '%s' was placed inside '%s'", source_object, destination_object
    )


def main() -> None:
    """Parse command-line arguments and run the modular example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Bundled scene name or path to a JSON/YAML scene config",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    parser.add_argument(
        "--source-object", default=DEFAULT_SOURCE_OBJECT, help="Object to pick"
    )
    parser.add_argument(
        "--destination-object",
        default=DEFAULT_DESTINATION_OBJECT,
        help="Receptacle to place the source object inside",
    )
    parser.add_argument(
        "--detection-cameras",
        nargs="+",
        default=list(DEFAULT_DETECTION_CAMERAS),
        help="Camera names whose projected center points are averaged",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Output MP4 path",
    )
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
