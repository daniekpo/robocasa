"""Ground-truth perception and pick-place oracle for configured scenes.

The module deliberately keeps perception, low-level motion, and command dispatch
separate. A learned detector can replace :class:`GroundTruthObjectDetector`, and
dataset collectors can subscribe to transitions without enabling video output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import mujoco
import numpy as np
from robosuite.utils import transform_utils as transform_utils
from robosuite.utils.camera_utils import get_camera_segmentation

from robocasa.example_env import SceneGymEnv
from robocasa.utils.object_utils import check_obj_in_receptacle

if TYPE_CHECKING:
    from robocasa.task_config import MoveCommand


LOGGER = logging.getLogger(__name__)
BASKET_RIM_INSET = 0.015
BASKET_RIM_VERTICAL_INSET = 0.03
SUPPORTED_RELATIONS = frozenset(
    {"in", "on", "left_of", "right_of", "in_front_of", "behind"}
)


@dataclass(frozen=True)
class Detection:
    """One object detection in an upright RGB image.

    Args:
        bbox_xyxy: Inclusive ``(x_min, y_min, x_max, y_max)`` pixel bounds.
        mask: Optional boolean segmentation mask.
    """

    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray | None = None


class ObjectDetector(Protocol):
    """Interface implemented by named-object detectors."""

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
    """Motion and relation tolerances for the pick-place oracle."""

    approach_height: float = 0.20
    grasp_overlap: float = 0.01
    drop_clearance: float = 0.20
    placement_clearance: float = 0.02
    directional_clearance: float = 0.04
    basket_transport_height: float = 0.03
    relation_tolerance: float = 0.025
    position_tolerance: float = 0.012
    grasp_tolerance: float = 0.020
    max_waypoint_steps: int = 120
    gripper_settle_steps: int = 30


@dataclass(frozen=True)
class OracleTransition:
    """One simulator transition emitted by the motion executor.

    ``observation`` and ``simulator_state`` describe the state immediately
    before applying ``action``. This aligns recorded actions with LeRobot frame
    semantics while ``next_observation`` is available for online consumers.
    """

    observation: Mapping[str, np.ndarray]
    action: np.ndarray
    simulator_state: np.ndarray
    next_observation: Mapping[str, np.ndarray]
    next_simulator_state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    stage: str
    command_index: int | None
    command_text: str | None


@dataclass(frozen=True)
class CommandBoundary:
    """Metadata captured after a command has passed its postcondition."""

    command_index: int
    command_text: str
    simulator_state: np.ndarray


class FrameRecorder(Protocol):
    """Optional visualization sink used by :class:`MotionExecutor`."""

    def append(
        self, observation: Mapping[str, np.ndarray], stage: str, overlays: Any = None
    ) -> None:
        """Append one rendered observation."""


TransitionCallback = Callable[[OracleTransition], None]
GraspStrategy = Callable[[SceneGymEnv, str, ObjectEstimate], np.ndarray]


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
        return Detection(
            bbox_xyxy=(
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            ),
            mask=mask,
        )

    def _get_geom_ids(self, object_name: str) -> np.ndarray:
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


def localize_object(
    detector: ObjectDetector,
    observation: Mapping[str, np.ndarray],
    calibration: Mapping[str, Mapping[str, np.ndarray]],
    camera_names: tuple[str, ...],
    object_name: str,
) -> ObjectEstimate:
    """Detect an object in each view and average reconstructed world points."""
    views: list[ViewEstimate] = []
    for camera_name in camera_names:
        image = observation[f"{camera_name}_image"]
        depth = observation[f"{camera_name}_depth"][..., 0]
        try:
            detection = detector.detect(image, camera_name, object_name)
        except RuntimeError as error:
            LOGGER.debug(
                "Could not localize %s in %s: %s",
                object_name,
                camera_name,
                error,
            )
            continue
        pixel_xy = get_detection_center(detection)
        camera_point, world_point = project_pixel_to_3d(
            pixel_xy,
            depth,
            calibration[camera_name]["intrinsics"],
            calibration[camera_name]["camera_to_world"],
        )
        views.append(
            ViewEstimate(
                camera_name,
                detection,
                pixel_xy,
                camera_point,
                world_point,
            )
        )
    if not views:
        raise RuntimeError(
            f"Object '{object_name}' is not visible in any detection camera"
        )
    fused_point = np.mean([view.world_point for view in views], axis=0)
    return ObjectEstimate(object_name, fused_point, tuple(views))


def get_object_contact_pairs(env: SceneGymEnv) -> tuple[tuple[str, str], ...]:
    """Read configured object-object pairs from current MuJoCo contacts."""
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


def _simulation_env(env: SceneGymEnv) -> Any:
    """Return the RoboCasa environment beneath the Gymnasium adapter."""
    return getattr(env, "env", env)


def get_simulator_state(env: SceneGymEnv) -> np.ndarray:
    """Return a detached flattened MuJoCo state."""
    state = env.sim.get_state()
    flattened = state.flatten() if hasattr(state, "flatten") else np.asarray(state)
    return np.asarray(flattened, dtype=np.float64).copy()


def get_object_bounds(env: SceneGymEnv, object_name: str) -> np.ndarray:
    """Return world-aligned min/max bounds as a ``(2, 3)`` array."""
    simulation_env = _simulation_env(env)
    if object_name not in simulation_env.objects:
        raise ValueError(f"Unknown configured object: {object_name}")
    body_id = simulation_env.obj_body_id[object_name]
    position = simulation_env.sim.data.body_xpos[body_id]
    quaternion = transform_utils.convert_quat(
        simulation_env.sim.data.body_xquat[body_id], to="xyzw"
    )
    points = np.asarray(
        simulation_env.objects[object_name].get_bbox_points(
            trans=position, rot=quaternion
        )
    )
    return np.stack((points.min(axis=0), points.max(axis=0)))


def is_object_grasped(env: SceneGymEnv, object_name: str) -> bool:
    """Return whether the gripper has stable contact for object transport."""
    simulation_env = _simulation_env(env)
    robot = simulation_env.robots[0]
    if "right" not in robot.gripper:
        raise AttributeError("Robot does not expose a right gripper")
    gripper = robot.gripper["right"]
    obj = simulation_env.objects[object_name]
    if _object_category(env, object_name) == "basket":
        # A thin rim often contacts only one pad after the basket settles; the
        # closed gripper still provides a stable hook for the low slide motion.
        return bool(simulation_env.check_contact(gripper, obj))
    return bool(simulation_env._check_grasp(gripper, obj))


def _has_planar_overlap(first: np.ndarray, second: np.ndarray, axis: int) -> bool:
    orthogonal_axis = 1 - axis
    return bool(
        min(first[1, orthogonal_axis], second[1, orthogonal_axis])
        >= max(first[0, orthogonal_axis], second[0, orthogonal_axis])
    )


def verify_relation(
    env: SceneGymEnv,
    source_object: str,
    relation: str,
    target_object: str,
    tolerance: float = 0.025,
) -> bool:
    """Evaluate a supported object relation against current simulator state."""
    relation = _relation_value(relation)
    simulation_env = _simulation_env(env)
    if relation == "in":
        return bool(
            check_obj_in_receptacle(simulation_env, source_object, target_object)
        )

    source = get_object_bounds(env, source_object)
    target = get_object_bounds(env, target_object)
    if relation == "on":
        touching = simulation_env.check_contact(
            simulation_env.objects[source_object],
            simulation_env.objects[target_object],
        )
        vertical_gap = abs(source[0, 2] - target[1, 2])
        overlap_x = source[1, 0] >= target[0, 0] and target[1, 0] >= source[0, 0]
        overlap_y = source[1, 1] >= target[0, 1] and target[1, 1] >= source[0, 1]
        return bool(touching and vertical_gap <= tolerance and overlap_x and overlap_y)

    checks = {
        "left_of": (0, source[1, 0] <= target[0, 0] + tolerance),
        "right_of": (0, source[0, 0] >= target[1, 0] - tolerance),
        "behind": (1, source[1, 1] <= target[0, 1] + tolerance),
        "in_front_of": (1, source[0, 1] >= target[1, 1] - tolerance),
    }
    if relation not in checks:
        raise ValueError(f"Unsupported move relation: {relation}")
    axis, separated = checks[relation]
    return bool(separated and _has_planar_overlap(source, target, axis))


class GraspStrategyRegistry:
    """Resolve small category-specific grasp rules with a center fallback."""

    def __init__(self) -> None:
        """Register the known edge-grasp exception."""
        self._strategies: dict[str, GraspStrategy] = {"basket": _basket_edge_grasp}

    def register(self, category: str, strategy: GraspStrategy) -> None:
        """Register or replace the strategy for one object category."""
        self._strategies[category] = strategy

    def get_grasp_point(
        self,
        env: SceneGymEnv,
        object_name: str,
        estimate: ObjectEstimate,
    ) -> np.ndarray:
        """Return the grasp point selected for a configured object."""
        category = _object_category(env, object_name)
        strategy = self._strategies.get(category, _center_grasp)
        return strategy(env, object_name, estimate)


def _object_category(env: SceneGymEnv, object_name: str) -> str:
    for object_config in env.config.objects:
        if object_config.name == object_name:
            return object_config.object_type
    raise ValueError(f"Unknown configured object: {object_name}")


def _center_grasp(
    env: SceneGymEnv, object_name: str, estimate: ObjectEstimate
) -> np.ndarray:
    del env, object_name
    return estimate.world_point.copy()


def _basket_edge_grasp(
    env: SceneGymEnv, object_name: str, estimate: ObjectEstimate
) -> np.ndarray:
    del estimate
    bounds = get_object_bounds(env, object_name)
    center = bounds.mean(axis=0)
    eef_key = f"{env.robots[0].robot_model.naming_prefix}eef_quat"
    eef_rotation = transform_utils.quat2mat(env._last_observation[eef_key])
    closing_axis = eef_rotation[:2, 1]
    if abs(closing_axis[0]) >= abs(closing_axis[1]):
        edges = np.asarray(
            [
                [bounds[0, 0] + BASKET_RIM_INSET, center[1]],
                [bounds[1, 0] - BASKET_RIM_INSET, center[1]],
            ]
        )
    else:
        edges = np.asarray(
            [
                [center[0], bounds[0, 1] + BASKET_RIM_INSET],
                [center[0], bounds[1, 1] - BASKET_RIM_INSET],
            ]
        )
    controller = env.robots[0].composite_controller
    robot_position, _ = controller.get_controller_base_pose("right")
    nearest_edge = edges[np.argmin(np.linalg.norm(edges - robot_position[:2], axis=1))]
    return np.asarray(
        [
            nearest_edge[0],
            nearest_edge[1],
            bounds[1, 2] - BASKET_RIM_VERTICAL_INSET,
        ]
    )


class MotionExecutor:
    """Execute world-frame waypoints with PandaOmron's delta OSC controller."""

    def __init__(
        self,
        env: SceneGymEnv,
        recorder: FrameRecorder | None = None,
        config: MotionConfig | None = None,
        transition_callback: TransitionCallback | None = None,
    ) -> None:
        """Validate and retain the controller and optional recording hooks."""
        if len(env.robots) != 1:
            raise ValueError("The pick-place oracle expects exactly one robot")
        self.env = env
        self.robot = env.robots[0]
        self.recorder = recorder
        self.config = config or MotionConfig()
        self.transition_callback = transition_callback
        self.arm_name = "right"
        self.gripper_name = "right_gripper"
        self.eef_position_key = f"{self.robot.robot_model.naming_prefix}eef_pos"

        controller = self.robot.composite_controller.get_controller(self.arm_name)
        if controller.input_type != "delta" or controller.input_ref_frame != "base":
            raise ValueError("The oracle requires a base-frame delta arm controller")
        if controller.control_dim != 6:
            raise ValueError("The oracle requires a six-dimensional pose controller")
        self.position_scale = np.maximum(
            np.abs(controller.output_min[:3]), np.abs(controller.output_max[:3])
        )

    def move_to_position(
        self,
        observation: Mapping[str, np.ndarray],
        target_world: np.ndarray,
        gripper_closed: bool,
        stage: str,
        tolerance: float | None = None,
        command_index: int | None = None,
        command_text: str | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Servo the end effector to a world point or raise on timeout."""
        target_world = np.asarray(target_world, dtype=np.float64)
        target_tolerance = tolerance or self.config.position_tolerance
        for step_index in range(self.config.max_waypoint_steps):
            world_error = target_world - observation[self.eef_position_key]
            if np.linalg.norm(world_error) <= target_tolerance:
                LOGGER.info("%s reached in %d steps", stage, step_index)
                return observation
            action = self._create_action(world_error, gripper_closed)
            observation = self._step(
                observation, action, stage, command_index, command_text
            )
        final_error = np.linalg.norm(target_world - observation[self.eef_position_key])
        raise RuntimeError(
            f"Waypoint '{stage}' timed out with {final_error:.3f} m error"
        )

    def hold_gripper(
        self,
        observation: Mapping[str, np.ndarray],
        closed: bool,
        stage: str,
        command_index: int | None = None,
        command_text: str | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Hold the current pose while the gripper opens or closes."""
        for _ in range(self.config.gripper_settle_steps):
            action = self._create_action(np.zeros(3), closed)
            observation = self._step(
                observation, action, stage, command_index, command_text
            )
        return observation

    def _step(
        self,
        observation: Mapping[str, np.ndarray],
        action: np.ndarray,
        stage: str,
        command_index: int | None,
        command_text: str | None,
    ) -> Mapping[str, np.ndarray]:
        pre_state = get_simulator_state(self.env)
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        next_state = get_simulator_state(self.env)
        if self.transition_callback is not None:
            self.transition_callback(
                OracleTransition(
                    observation=observation,
                    action=action.copy(),
                    simulator_state=pre_state,
                    next_observation=next_observation,
                    next_simulator_state=next_state,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    stage=stage,
                    command_index=command_index,
                    command_text=command_text,
                )
            )
        if self.recorder is not None:
            self.recorder.append(next_observation, stage)
        if terminated or truncated:
            raise RuntimeError(f"Episode ended while executing '{stage}'")
        return next_observation

    def _create_action(
        self, world_error: np.ndarray, gripper_closed: bool
    ) -> np.ndarray:
        _, base_rotation = self.robot.composite_controller.get_controller_base_pose(
            self.arm_name
        )
        arm_action = world_error_to_delta_action(
            world_error, base_rotation, self.position_scale
        )
        return self.robot.create_action_vector(
            {
                self.arm_name: arm_action,
                self.gripper_name: np.asarray(
                    [1.0 if gripper_closed else -1.0]
                ),
                "base_mode": -1,
            }
        )


class TaskOracle:
    """Execute task move commands with live relocalization and verification."""

    def __init__(
        self,
        env: SceneGymEnv,
        detection_cameras: tuple[str, ...],
        *,
        detector: ObjectDetector | None = None,
        motion_config: MotionConfig | None = None,
        recorder: FrameRecorder | None = None,
        transition_callback: TransitionCallback | None = None,
        grasp_registry: GraspStrategyRegistry | None = None,
    ) -> None:
        """Create an oracle for one reset configured-scene environment."""
        if not detection_cameras:
            raise ValueError("At least one detection camera is required")
        camera_group = env.config.cameras
        if camera_group is None or not camera_group.depth:
            raise ValueError("The scene must define cameras with depth enabled")
        unknown = sorted(set(detection_cameras) - set(camera_group.names))
        if unknown:
            raise ValueError("Unknown detection camera(s): " + ", ".join(unknown))
        self.env = env
        self.detection_cameras = detection_cameras
        self.detector = detector or GroundTruthObjectDetector(env)
        self.config = motion_config or MotionConfig()
        self.motion = MotionExecutor(
            env, recorder, self.config, transition_callback=transition_callback
        )
        self.grasp_registry = grasp_registry or GraspStrategyRegistry()
        self.calibration = env.get_camera_calibration()

    def execute_command(
        self,
        observation: Mapping[str, np.ndarray],
        command: MoveCommand,
        command_index: int,
    ) -> tuple[Mapping[str, np.ndarray], CommandBoundary]:
        """Execute one move command and return its verified state boundary."""
        action = getattr(command, "action", "move")
        action_value = getattr(action, "value", action)
        if action_value != "move":
            raise ValueError(f"Unsupported oracle action: {action_value}")
        source_object = command.object
        target_object = command.target
        relation = _relation_value(command.relation)
        if relation not in SUPPORTED_RELATIONS:
            raise ValueError(f"Unsupported move relation: {relation}")
        command_text = str(command)

        source = localize_object(
            self.detector,
            observation,
            self.calibration,
            self.detection_cameras,
            source_object,
        )
        target = localize_object(
            self.detector,
            observation,
            self.calibration,
            self.detection_cameras,
            target_object,
        )
        grasp_position = self.grasp_registry.get_grasp_point(
            self.env, source_object, source
        )
        grasp_position[2] -= self.config.grasp_overlap
        drop_position = self._get_drop_position(
            source_object, grasp_position, target_object, target, relation
        )
        if _object_category(self.env, source_object) == "basket":
            source_safe_height = (
                grasp_position[2] + self.config.basket_transport_height
            )
            target_safe_height = max(
                source_safe_height,
                drop_position[2] + self.config.placement_clearance,
            )
        else:
            source_safe_height = grasp_position[2] + self.config.approach_height
            target_safe_height = max(
                source_safe_height,
                drop_position[2] + self.config.approach_height,
            )
        above_source = grasp_position.copy()
        above_source[2] = source_safe_height
        above_target = drop_position.copy()
        above_target[2] = target_safe_height

        context = {"command_index": command_index, "command_text": command_text}
        observation = self.motion.move_to_position(
            observation, above_source, False, "move above source", **context
        )
        observation = self.motion.move_to_position(
            observation,
            grasp_position,
            False,
            "descend to grasp",
            tolerance=self.config.grasp_tolerance,
            **context,
        )
        observation = self.motion.hold_gripper(
            observation, True, "close gripper", **context
        )
        observation = self.motion.move_to_position(
            observation, above_source, True, "lift vertically", **context
        )
        if not is_object_grasped(self.env, source_object):
            raise RuntimeError(f"Failed to grasp '{source_object}'")
        observation = self.motion.move_to_position(
            observation, above_target, True, "translate above target", **context
        )
        observation = self.motion.move_to_position(
            observation, drop_position, True, "descend to drop height", **context
        )
        observation = self.motion.hold_gripper(
            observation, False, "open gripper", **context
        )
        observation = self.motion.move_to_position(
            observation, above_target, False, "retreat vertically", **context
        )
        if not verify_relation(
            self.env,
            source_object,
            relation,
            target_object,
            tolerance=self.config.relation_tolerance,
        ):
            raise RuntimeError(
                f"Postcondition failed: {source_object} {relation} {target_object}"
            )
        boundary = CommandBoundary(
            command_index=command_index,
            command_text=command_text,
            simulator_state=get_simulator_state(self.env),
        )
        return observation, boundary

    def _get_drop_position(
        self,
        source_object: str,
        grasp_position: np.ndarray,
        target_object: str,
        target: ObjectEstimate,
        relation: str,
    ) -> np.ndarray:
        source_bounds = get_object_bounds(self.env, source_object)
        target_bounds = get_object_bounds(self.env, target_object)
        source_center = source_bounds.mean(axis=0)
        target_center = target_bounds.mean(axis=0)
        grasp_offset = grasp_position - source_center

        desired_center = source_center.copy()
        if relation == "in":
            drop = target.world_point.copy()
            drop[:2] = target_center[:2] + grasp_offset[:2]
            drop[2] = target_bounds[1, 2] + self.config.drop_clearance
            return drop
        if relation == "on":
            desired_center[:2] = target_center[:2]
            desired_center[2] = (
                target_bounds[1, 2]
                + (source_center[2] - source_bounds[0, 2])
                + self.config.placement_clearance
            )
            return desired_center + grasp_offset

        source_half_size = (source_bounds[1] - source_bounds[0]) / 2
        target_half_size = (target_bounds[1] - target_bounds[0]) / 2
        if relation == "left_of":
            desired_center[0] = target_center[0] - source_half_size[0] - target_half_size[0] - self.config.directional_clearance
            desired_center[1] = target_center[1]
        elif relation == "right_of":
            desired_center[0] = target_center[0] + source_half_size[0] + target_half_size[0] + self.config.directional_clearance
            desired_center[1] = target_center[1]
        elif relation == "behind":
            desired_center[1] = target_center[1] - source_half_size[1] - target_half_size[1] - self.config.directional_clearance
            desired_center[0] = target_center[0]
        elif relation == "in_front_of":
            desired_center[1] = target_center[1] + source_half_size[1] + target_half_size[1] + self.config.directional_clearance
            desired_center[0] = target_center[0]
        desired_center[2] += self.config.placement_clearance
        return desired_center + grasp_offset


def _relation_value(relation: Any) -> str:
    value = getattr(relation, "value", relation)
    return str(value)
