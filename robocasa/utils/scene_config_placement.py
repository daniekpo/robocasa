"""Exact and randomized object placement for config-driven RoboCasa scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from robosuite.utils.transform_utils import convert_quat, quat2mat

from robocasa.scene_config import (
    PlacementRelation,
    SceneConfig,
    SceneConfigError,
    SceneObjectConfig,
)
from robocasa.utils.errors import PlacementError
from robocasa.utils.object_utils import objs_intersect_bbox

IDENTITY_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0])
MAX_RANDOM_OBJECT_ATTEMPTS = 500
MAX_RANDOM_SCENE_ATTEMPTS = 100


@dataclass(frozen=True)
class PlacementWorkspace:
    """World geometry for a robot-local random-placement workspace."""

    robot_position: np.ndarray
    robot_yaw: float
    forward_range: tuple[float, float]
    lateral_range: tuple[float, float]
    support_z: float
    fixture_p0: np.ndarray
    fixture_px: np.ndarray
    fixture_py: np.ndarray
    margin: float


Placement = tuple[np.ndarray, np.ndarray, Any]


class ConfiguredScenePlacementSampler:
    """Resolve scene placements using sampled object bounds and scene constraints."""

    def __init__(
        self,
        scene_config: SceneConfig,
        objects: dict[str, Any],
        rng: np.random.Generator | None = None,
        workspace: PlacementWorkspace | None = None,
    ):
        self.scene_config = scene_config
        self.objects = objects
        self.rng = rng if rng is not None else np.random.default_rng()
        self.workspace = workspace

    def sample(
        self,
        placed_objects: dict[str, tuple[Any, Any, Any]] | None = None,
        reference: str | tuple[float, float, float] | None = None,
        on_top: bool = True,
    ) -> dict[str, Placement]:
        """Calculate all configured object poses.

        Args:
            placed_objects: Existing fixture placements. They remain managed by Kitchen
                and are not returned by this sampler.
            reference: Unused compatibility argument for RoboCasa samplers.
            on_top: Unused compatibility argument for RoboCasa samplers.

        Returns:
            Object names mapped to `(position, quaternion, object)` tuples.

        Raises:
            PlacementError: If random objects cannot be placed without overlap.
            SceneConfigError: If any object bounding box crosses the workspace.
        """
        del placed_objects, reference, on_top
        has_random_objects = any(
            object_config.random_placement
            for object_config in self.scene_config.objects
        )
        attempt_count = MAX_RANDOM_SCENE_ATTEMPTS if has_random_objects else 1
        for _ in range(attempt_count):
            placements = self._sample_all_objects()
            if _random_objects_are_collision_free(self.scene_config, placements):
                return placements

        raise PlacementError("Could not find collision-free random scene placements")

    def _sample_all_objects(self) -> dict[str, Placement]:
        placements: dict[str, Placement] = {}
        for object_config in self.scene_config.objects_in_placement_order():
            obj = self.objects[object_config.name]
            quaternion = _get_quaternion(object_config)
            local_minimum, local_maximum = _get_rotated_bounds(obj, quaternion)

            if object_config.absolute_position is not None:
                position = _position_from_contact_point(
                    np.asarray(object_config.absolute_position, dtype=float),
                    local_minimum,
                    local_maximum,
                )
            elif object_config.random_placement:
                position = self._sample_random_position(
                    obj,
                    quaternion,
                    local_minimum,
                    local_maximum,
                    placements,
                )
            else:
                position = _resolve_relative_position(
                    object_config,
                    local_minimum,
                    local_maximum,
                    placements,
                    self.rng,
                )

            placements[object_config.name] = (position, quaternion, obj)
        self._validate_workspace(placements)
        return placements

    def _validate_workspace(self, placements: dict[str, Placement]) -> None:
        """Reject any configured object whose bounding box leaves the workspace."""
        if self.workspace is None:
            return
        for object_name, (position, quaternion, obj) in placements.items():
            violation = _get_workspace_violation(
                obj, position, quaternion, self.workspace
            )
            if violation is not None:
                raise SceneConfigError(
                    f"Object '{object_name}' is outside the configured workspace: "
                    f"{violation}"
                )

    def _sample_random_position(
        self,
        obj: Any,
        quaternion: np.ndarray,
        local_minimum: np.ndarray,
        local_maximum: np.ndarray,
        placements: dict[str, Placement],
    ) -> np.ndarray:
        if self.workspace is None:
            raise PlacementError("Random placement requires a configured workspace")

        workspace = self.workspace
        rotation = _rotation_2d(workspace.robot_yaw)
        for _ in range(MAX_RANDOM_OBJECT_ATTEMPTS):
            robot_local_xy = np.asarray(
                [
                    self.rng.uniform(*workspace.forward_range),
                    self.rng.uniform(*workspace.lateral_range),
                ]
            )
            world_xy = workspace.robot_position[:2] + rotation @ robot_local_xy
            contact_point = np.asarray([world_xy[0], world_xy[1], workspace.support_z])
            position = _position_from_contact_point(
                contact_point, local_minimum, local_maximum
            )

            if not _object_is_in_workspace(obj, position, quaternion, workspace):
                continue
            if _intersects_placed_object(obj, position, quaternion, placements):
                continue
            return position

        raise PlacementError(
            f"Could not randomly place object '{obj.name}' within the workspace"
        )


def _get_quaternion(object_config: SceneObjectConfig) -> np.ndarray:
    if object_config.absolute_quat is None:
        return IDENTITY_QUATERNION.copy()
    return np.asarray(object_config.absolute_quat, dtype=float)


def _get_rotated_bounds(
    obj: Any, quaternion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotated_points = _rotated_local_points(obj, quaternion)
    return rotated_points.min(axis=0), rotated_points.max(axis=0)


def _rotated_local_points(obj: Any, quaternion: np.ndarray) -> np.ndarray:
    points = np.asarray(obj.get_bbox_points(), dtype=float)
    rotation = quat2mat(convert_quat(quaternion, to="xyzw"))
    return points @ rotation.T


def _position_from_contact_point(
    contact_point: np.ndarray,
    local_minimum: np.ndarray,
    local_maximum: np.ndarray,
) -> np.ndarray:
    local_center = (local_minimum + local_maximum) / 2
    return np.asarray(
        [
            contact_point[0] - local_center[0],
            contact_point[1] - local_center[1],
            contact_point[2] - local_minimum[2],
        ]
    )


def _resolve_relative_position(
    object_config: SceneObjectConfig,
    local_minimum: np.ndarray,
    local_maximum: np.ndarray,
    placements: dict[str, Placement],
    rng: np.random.Generator,
) -> np.ndarray:
    assert object_config.relative_to is not None
    assert object_config.relation is not None

    reference_position, reference_quaternion, reference_object = placements[
        object_config.relative_to
    ]
    reference_local_minimum, reference_local_maximum = _get_rotated_bounds(
        reference_object, reference_quaternion
    )
    reference_minimum = reference_position + reference_local_minimum
    reference_maximum = reference_position + reference_local_maximum
    reference_center = (reference_minimum + reference_maximum) / 2
    local_center = (local_minimum + local_maximum) / 2

    distance = object_config.distance
    if distance is None:
        assert object_config.distance_range is not None
        distance = rng.uniform(*object_config.distance_range)
    jitter = 0.0
    if object_config.orthogonal_jitter_range is not None:
        jitter = rng.uniform(*object_config.orthogonal_jitter_range)

    position = np.zeros(3)
    relation = object_config.relation

    if relation in {PlacementRelation.LEFT, PlacementRelation.RIGHT}:
        position[1] = reference_center[1] - local_center[1] + jitter
        position[2] = reference_minimum[2] - local_minimum[2]
        if relation is PlacementRelation.LEFT:
            position[0] = reference_minimum[0] - distance - local_maximum[0]
        else:
            position[0] = reference_maximum[0] + distance - local_minimum[0]
    elif relation in {PlacementRelation.IN_FRONT_OF, PlacementRelation.BEHIND}:
        position[0] = reference_center[0] - local_center[0] + jitter
        position[2] = reference_minimum[2] - local_minimum[2]
        if relation is PlacementRelation.BEHIND:
            position[1] = reference_minimum[1] - distance - local_maximum[1]
        else:
            position[1] = reference_maximum[1] + distance - local_minimum[1]
    else:
        position[0] = reference_center[0] - local_center[0]
        position[1] = reference_center[1] - local_center[1]
        position[2] = reference_maximum[2] + distance - local_minimum[2]

    return position


def _object_is_in_workspace(
    obj: Any,
    position: np.ndarray,
    quaternion: np.ndarray,
    workspace: PlacementWorkspace,
) -> bool:
    return _get_workspace_violation(obj, position, quaternion, workspace) is None


def _get_workspace_violation(
    obj: Any,
    position: np.ndarray,
    quaternion: np.ndarray,
    workspace: PlacementWorkspace,
) -> str | None:
    """Describe the first workspace boundary crossed by an object bounding box."""
    world_points = _rotated_local_points(obj, quaternion) + position
    robot_local_points = (
        world_points[:, :2] - workspace.robot_position[:2]
    ) @ _rotation_2d(workspace.robot_yaw)

    forward_minimum, forward_maximum = workspace.forward_range
    lateral_minimum, lateral_maximum = workspace.lateral_range
    margin = workspace.margin
    effective_limits = (
        ("forward minimum", 0, forward_minimum + margin, np.min),
        ("forward maximum", 0, forward_maximum - margin, np.max),
        ("lateral minimum", 1, lateral_minimum + margin, np.min),
        ("lateral maximum", 1, lateral_maximum - margin, np.max),
    )
    for label, axis, limit, reducer in effective_limits:
        extent = float(reducer(robot_local_points[:, axis]))
        crosses_limit = extent < limit if "minimum" in label else extent > limit
        if crosses_limit:
            return f"{label} is {limit:.4f} m, bounding-box extent is {extent:.4f} m"

    fixture_u = workspace.fixture_px[:2] - workspace.fixture_p0[:2]
    fixture_v = workspace.fixture_py[:2] - workspace.fixture_p0[:2]
    fixture_u_length = np.linalg.norm(fixture_u)
    fixture_v_length = np.linalg.norm(fixture_v)
    relative_points = world_points[:, :2] - workspace.fixture_p0[:2]
    u_projection = relative_points @ fixture_u / fixture_u_length**2
    v_projection = relative_points @ fixture_v / fixture_v_length**2
    u_margin = margin / fixture_u_length
    v_margin = margin / fixture_v_length
    if not np.all((u_projection >= u_margin) & (u_projection <= 1 - u_margin)):
        return "bounding box crosses the supporting fixture's first horizontal axis"
    if not np.all((v_projection >= v_margin) & (v_projection <= 1 - v_margin)):
        return "bounding box crosses the supporting fixture's second horizontal axis"
    return None


def _intersects_placed_object(
    obj: Any,
    position: np.ndarray,
    quaternion: np.ndarray,
    placements: dict[str, Placement],
) -> bool:
    object_points = _rotated_local_points(obj, quaternion) + position
    for other_position, other_quaternion, other_object in placements.values():
        other_points = (
            _rotated_local_points(other_object, other_quaternion) + other_position
        )
        if objs_intersect_bbox(object_points, other_points):
            return True
    return False


def _random_objects_are_collision_free(
    scene_config: SceneConfig, placements: dict[str, Placement]
) -> bool:
    random_names = {
        object_config.name
        for object_config in scene_config.objects
        if object_config.random_placement
    }
    placement_items = list(placements.items())
    for index, (object_name, placement) in enumerate(placement_items):
        for other_name, other_placement in placement_items[index + 1 :]:
            if object_name not in random_names and other_name not in random_names:
                continue
            position, quaternion, obj = placement
            other_position, other_quaternion, other_object = other_placement
            object_points = _rotated_local_points(obj, quaternion) + position
            other_points = (
                _rotated_local_points(other_object, other_quaternion) + other_position
            )
            if objs_intersect_bbox(object_points, other_points):
                return False
    return True


def _rotation_2d(yaw: float) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray([[cosine, -sine], [sine, cosine]])
