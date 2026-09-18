"""Typed loading and validation for config-driven RoboCasa scenes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

DEFAULT_ENVIRONMENT = "ConfiguredKitchen"
SCENE_CONFIG_DIRECTORY = Path(__file__).parent / "scene_configs"
SUPPORTED_DEVICES = {"keyboard", "spacemouse"}
SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}
RANDOM_PLACEMENT = "random"
RANDOM_STYLE = "random"


class SceneConfigError(ValueError):
    """Raised when a scene configuration is malformed or inconsistent."""


class PlacementRelation(str, Enum):
    """Supported object-to-object spatial relationships in world coordinates."""

    LEFT = "left"
    RIGHT = "right"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ON = "on"


@dataclass(frozen=True)
class WorkspaceConfig:
    """Robot-local sampling bounds constrained to a supporting fixture."""

    fixture: str
    forward_range: tuple[float, float]
    lateral_range: tuple[float, float]
    margin: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceConfig:
        """Parse and validate random-placement workspace settings."""
        if not isinstance(data, dict):
            raise SceneConfigError("workspace must be a mapping")
        _reject_unknown_keys(
            data,
            {"fixture", "forward_range", "lateral_range", "margin"},
            "workspace",
        )
        fixture = _require_nonempty_string(data, "fixture", "workspace")
        forward_range = _parse_range(
            data.get("forward_range"), "workspace forward_range"
        )
        lateral_range = _parse_range(
            data.get("lateral_range"), "workspace lateral_range"
        )
        margin = _parse_number(data.get("margin", 0.02), "workspace margin")
        if margin < 0:
            raise SceneConfigError("workspace margin must be non-negative")
        return cls(
            fixture=fixture,
            forward_range=forward_range,
            lateral_range=lateral_range,
            margin=margin,
        )


@dataclass(frozen=True)
class CameraConfig:
    """Placement of a fixed camera in the world or a fixture-local frame."""

    name: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fixture: str | None = None
    fovy: float = 60.0
    roll: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> CameraConfig:
        """Parse and validate one camera placement."""
        if not isinstance(data, dict):
            raise SceneConfigError(f"Camera at index {index} must be a mapping")
        _reject_unknown_keys(
            data,
            {"name", "position", "look_at", "fixture", "fovy", "roll"},
            f"camera at index {index}",
        )
        name = _require_nonempty_string(data, "name", f"camera at index {index}")
        position = _parse_vector(data.get("position"), 3, f"Camera '{name}' position")
        look_at = _parse_vector(data.get("look_at"), 3, f"Camera '{name}' look_at")
        if position == look_at:
            raise SceneConfigError(
                f"Camera '{name}' position and look_at must be different"
            )
        fixture = data.get("fixture")
        if fixture is not None and (
            not isinstance(fixture, str) or not fixture.strip()
        ):
            raise SceneConfigError(
                f"Camera '{name}' fixture must be a non-empty string or null"
            )
        fovy = _parse_number(data.get("fovy", 60.0), f"Camera '{name}' fovy")
        if not 0 < fovy < 180:
            raise SceneConfigError(f"Camera '{name}' fovy must be between 0 and 180")
        roll = _parse_number(data.get("roll", 0.0), f"Camera '{name}' roll")
        return cls(
            name=name,
            position=position,
            look_at=look_at,
            fixture=fixture,
            fovy=fovy,
            roll=roll,
        )


@dataclass(frozen=True)
class CameraGroupConfig:
    """Rendering settings and placements for algorithm observations."""

    width: int
    height: int
    depth: bool
    placements: tuple[CameraConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraGroupConfig:
        """Parse and validate the camera observation group."""
        if not isinstance(data, dict):
            raise SceneConfigError("cameras must be a mapping")
        _reject_unknown_keys(
            data, {"width", "height", "depth", "placements"}, "cameras"
        )
        width = _require_positive_integer(data, "width", "cameras")
        height = _require_positive_integer(data, "height", "cameras")
        depth = data.get("depth", False)
        if not isinstance(depth, bool):
            raise SceneConfigError("cameras field 'depth' must be a boolean")
        raw_placements = data.get("placements")
        if not isinstance(raw_placements, list) or not raw_placements:
            raise SceneConfigError("cameras placements must be a non-empty list")
        placements = tuple(
            CameraConfig.from_dict(camera, index)
            for index, camera in enumerate(raw_placements)
        )
        names = [camera.name for camera in placements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise SceneConfigError(
                f"Duplicate camera name(s): {', '.join(duplicates)}"
            )
        return cls(width=width, height=height, depth=depth, placements=placements)

    @property
    def names(self) -> tuple[str, ...]:
        """Return camera names in observation order."""
        return tuple(camera.name for camera in self.placements)


@dataclass(frozen=True)
class SceneObjectConfig:
    """Configuration for one RoboCasa object and its initial placement."""

    name: str
    object_type: str
    object_style: int | str
    absolute_position: tuple[float, float, float] | None = None
    absolute_quat: tuple[float, float, float, float] | None = None
    random_placement: bool = False
    relative_to: str | None = None
    relation: PlacementRelation | None = None
    distance: float | None = None
    distance_range: tuple[float, float] | None = None
    orthogonal_jitter_range: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> SceneObjectConfig:
        """Parse and validate an object configuration.

        Args:
            data: Raw object mapping.
            index: Object index, used to produce actionable error messages.

        Returns:
            Validated object configuration.
        """
        if not isinstance(data, dict):
            raise SceneConfigError(f"Object at index {index} must be a mapping")

        allowed_keys = {"name", "type", "style", "placement"}
        _reject_unknown_keys(data, allowed_keys, f"object at index {index}")

        name = _require_nonempty_string(data, "name", f"object at index {index}")
        object_type = _require_nonempty_string(data, "type", f"object '{name}'")
        if object_type not in OBJ_CATEGORIES:
            raise SceneConfigError(
                f"Unknown RoboCasa object category '{object_type}' for object '{name}'"
            )
        object_style = data.get("style")
        if object_style != RANDOM_STYLE and (
            isinstance(object_style, bool)
            or not isinstance(object_style, int)
            or object_style < 1
        ):
            raise SceneConfigError(
                f"Object '{name}' style must be a positive integer or '{RANDOM_STYLE}'"
            )

        placement = data.get("placement")
        if not isinstance(placement, dict):
            raise SceneConfigError(f"Object '{name}' placement must be a mapping")
        placement_type = _require_nonempty_string(
            placement, "type", f"object '{name}' placement"
        )

        absolute_position = None
        absolute_quat = None
        random_placement = False
        relative_to = None
        relation = None
        distance = None
        distance_range = None
        orthogonal_jitter_range = None

        if placement_type == "absolute":
            _reject_unknown_keys(
                placement,
                {"type", "absolute_position", "absolute_quat"},
                f"object '{name}' absolute placement",
            )
            if "absolute_position" not in placement:
                raise SceneConfigError(
                    f"Object '{name}' absolute placement is missing: absolute_position"
                )
            absolute_position = _parse_vector(
                placement["absolute_position"],
                3,
                f"Object '{name}' absolute_position",
            )
            if "absolute_quat" in placement:
                absolute_quat = _parse_vector(
                    placement["absolute_quat"],
                    4,
                    f"Object '{name}' absolute_quat",
                )
                norm = math.sqrt(sum(component**2 for component in absolute_quat))
                if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    raise SceneConfigError(
                        f"Object '{name}' absolute_quat must be normalized in wxyz order"
                    )
        elif placement_type == RANDOM_PLACEMENT:
            _reject_unknown_keys(
                placement, {"type"}, f"object '{name}' random placement"
            )
            random_placement = True
        elif placement_type == "relation":
            _reject_unknown_keys(
                placement,
                {"type", "relative_to", "relation"},
                f"object '{name}' placement",
            )
            relative_to = _require_nonempty_string(
                placement, "relative_to", f"object '{name}' placement"
            )
            relation_config = placement.get("relation")
            if not isinstance(relation_config, dict):
                raise SceneConfigError(f"Object '{name}' relation must be a mapping")
            _reject_unknown_keys(
                relation_config,
                {"type", "distance", "distance_range", "orthogonal_jitter_range"},
                f"object '{name}' relation",
            )
            relation_type = _require_nonempty_string(
                relation_config, "type", f"object '{name}' relation"
            )
            try:
                relation = PlacementRelation(relation_type)
            except (TypeError, ValueError) as error:
                supported = ", ".join(relation.value for relation in PlacementRelation)
                raise SceneConfigError(
                    f"Unsupported relation {relation_type!r} for object '{name}'. "
                    f"Expected one of: {supported}"
                ) from error
            has_distance = "distance" in relation_config
            has_distance_range = "distance_range" in relation_config
            if has_distance == has_distance_range:
                raise SceneConfigError(
                    f"Object '{name}' must specify exactly one of distance or "
                    "distance_range"
                )
            if has_distance:
                distance = _parse_number(
                    relation_config["distance"], f"Object '{name}' distance"
                )
                if distance < 0:
                    raise SceneConfigError(
                        f"Object '{name}' distance must be non-negative"
                    )
            else:
                distance_range = _parse_range(
                    relation_config["distance_range"],
                    f"Object '{name}' distance_range",
                )
                if distance_range[0] < 0:
                    raise SceneConfigError(
                        f"Object '{name}' distance_range must be non-negative"
                    )

            if "orthogonal_jitter_range" in relation_config:
                if relation is PlacementRelation.ON:
                    raise SceneConfigError(
                        f"Object '{name}' orthogonal_jitter_range is only supported "
                        "for lateral relations"
                    )
                orthogonal_jitter_range = _parse_range(
                    relation_config["orthogonal_jitter_range"],
                    f"Object '{name}' orthogonal_jitter_range",
                )
        else:
            raise SceneConfigError(
                f"Unsupported placement type {placement_type!r} for object '{name}'. "
                "Expected one of: absolute, relation, random"
            )

        return cls(
            name=name,
            object_type=object_type,
            object_style=object_style,
            absolute_position=absolute_position,
            absolute_quat=absolute_quat,
            random_placement=random_placement,
            relative_to=relative_to,
            relation=relation,
            distance=distance,
            distance_range=distance_range,
            orthogonal_jitter_range=orthogonal_jitter_range,
        )


@dataclass(frozen=True)
class SceneConfig:
    """Complete configuration for an interactive RoboCasa kitchen scene."""

    environment: str
    layout_id: int
    style_id: int
    robot: str
    robot_base_fixture: str | None
    workspace: WorkspaceConfig | None
    seed: int | None
    controller: str | None
    device: str
    render_camera: str | None
    renderer: str
    show_walls: bool
    control_freq: int
    objects: tuple[SceneObjectConfig, ...]
    cameras: CameraGroupConfig | None = None
    robot_start_joints: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneConfig:
        """Parse and validate a complete scene mapping.

        Args:
            data: Raw scene mapping loaded from JSON or YAML.

        Returns:
            Validated scene configuration.
        """
        if not isinstance(data, dict):
            raise SceneConfigError("Scene config must contain a top-level mapping")

        allowed_keys = {
            "scene",
            "robot",
            "workspace",
            "seed",
            "device",
            "render_camera",
            "renderer",
            "show_walls",
            "cameras",
            "objects",
        }
        _reject_unknown_keys(data, allowed_keys, "scene config")

        scene = data.get("scene")
        if not isinstance(scene, dict):
            raise SceneConfigError("scene must be a mapping")
        _reject_unknown_keys(scene, {"environment", "layout_id", "style_id"}, "scene")
        environment = _require_nonempty_string(scene, "environment", "scene")
        if environment != DEFAULT_ENVIRONMENT:
            raise SceneConfigError(
                f"environment must be '{DEFAULT_ENVIRONMENT}', got {environment!r}"
            )

        layout_id = _require_integer(scene, "layout_id", "scene")
        style_id = _require_integer(scene, "style_id", "scene")
        if layout_id not in {item.value for item in LayoutType if item.value > 0}:
            raise SceneConfigError(f"Unknown RoboCasa layout_id: {layout_id}")
        if style_id not in {item.value for item in StyleType if item.value > 0}:
            raise SceneConfigError(f"Unknown RoboCasa style_id: {style_id}")
        robot_config = data.get("robot")
        if not isinstance(robot_config, dict):
            raise SceneConfigError("robot must be a mapping")
        _reject_unknown_keys(
            robot_config,
            {"type", "base_fixture", "start_joints", "controller", "control_freq"},
            "robot",
        )
        robot = _require_nonempty_string(robot_config, "type", "robot")
        robot_base_fixture = robot_config.get("base_fixture")
        if robot_base_fixture is not None and (
            not isinstance(robot_base_fixture, str) or not robot_base_fixture.strip()
        ):
            raise SceneConfigError(
                "robot base_fixture must be a non-empty string or null"
            )
        raw_start_joints = robot_config.get("start_joints")
        robot_start_joints = (
            _parse_number_sequence(raw_start_joints, "robot start_joints")
            if raw_start_joints is not None
            else None
        )

        raw_workspace = data.get("workspace")
        workspace = (
            WorkspaceConfig.from_dict(raw_workspace)
            if raw_workspace is not None
            else None
        )

        seed = data.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise SceneConfigError("seed must be an integer or null")

        controller = robot_config.get("controller")
        if controller is not None and not isinstance(controller, str):
            raise SceneConfigError("robot controller must be a string or null")

        device = data.get("device", "keyboard")
        if device not in SUPPORTED_DEVICES:
            raise SceneConfigError(
                f"device must be one of {sorted(SUPPORTED_DEVICES)}, got {device!r}"
            )

        render_camera = data.get("render_camera")
        if render_camera is not None and not isinstance(render_camera, str):
            raise SceneConfigError("render_camera must be a string or null")

        renderer = data.get("renderer", "mjviewer")
        if not isinstance(renderer, str) or not renderer:
            raise SceneConfigError("renderer must be a non-empty string")

        show_walls = data.get("show_walls", False)
        if not isinstance(show_walls, bool):
            raise SceneConfigError("show_walls must be a boolean")

        control_freq = robot_config.get("control_freq", 20)
        if (
            isinstance(control_freq, bool)
            or not isinstance(control_freq, int)
            or control_freq <= 0
        ):
            raise SceneConfigError("robot control_freq must be a positive integer")

        raw_cameras = data.get("cameras")
        cameras = (
            CameraGroupConfig.from_dict(raw_cameras)
            if raw_cameras is not None
            else None
        )

        raw_objects = data.get("objects")
        if not isinstance(raw_objects, list) or not raw_objects:
            raise SceneConfigError("objects must be a non-empty list")
        objects = tuple(
            SceneObjectConfig.from_dict(object_data, index)
            for index, object_data in enumerate(raw_objects)
        )

        names = [obj.name for obj in objects]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise SceneConfigError(
                f"Duplicate object name(s): {', '.join(duplicate_names)}"
            )

        known_names = set(names)
        for obj in objects:
            if obj.relative_to is not None and obj.relative_to not in known_names:
                raise SceneConfigError(
                    f"Object '{obj.name}' references unknown object '{obj.relative_to}'"
                )
            if obj.relative_to == obj.name:
                raise SceneConfigError(f"Object '{obj.name}' cannot reference itself")

        if any(obj.random_placement for obj in objects):
            if workspace is None:
                raise SceneConfigError(
                    "workspace is required when an object uses placement type: random"
                )
            if robot_base_fixture is None:
                raise SceneConfigError(
                    "robot base_fixture is required for robot-relative random placement"
                )

        config = cls(
            environment=environment,
            layout_id=layout_id,
            style_id=style_id,
            robot=robot,
            robot_base_fixture=robot_base_fixture,
            robot_start_joints=robot_start_joints,
            workspace=workspace,
            seed=seed,
            controller=controller,
            device=device,
            render_camera=render_camera,
            renderer=renderer,
            show_walls=show_walls,
            control_freq=control_freq,
            cameras=cameras,
            objects=objects,
        )
        config.objects_in_placement_order()
        return config

    def objects_in_placement_order(self) -> tuple[SceneObjectConfig, ...]:
        """Return objects in stable dependency order.

        Returns:
            Objects ordered so every reference appears before its dependents.

        Raises:
            SceneConfigError: If the dependency graph contains a cycle.
        """
        remaining_dependencies = {
            obj.name: ({obj.relative_to} if obj.relative_to is not None else set())
            for obj in self.objects
        }
        ordered: list[SceneObjectConfig] = []
        ordered_names: set[str] = set()

        while len(ordered) < len(self.objects):
            ready = [
                obj
                for obj in self.objects
                if obj.name not in ordered_names
                and remaining_dependencies[obj.name].issubset(ordered_names)
            ]
            if not ready:
                cycle_names = [
                    obj.name for obj in self.objects if obj.name not in ordered_names
                ]
                raise SceneConfigError(
                    "Object placement dependency cycle detected involving: "
                    + ", ".join(cycle_names)
                )
            ordered.extend(ready)
            ordered_names.update(obj.name for obj in ready)

        return tuple(ordered)


def resolve_scene_config(scene: str | Path) -> Path:
    """Resolve a path or a bundled scene config name to a file.

    Bare names are looked up in :data:`SCENE_CONFIG_DIRECTORY`. The `.yaml`,
    `.yml`, and `.json` suffixes are tried in that order.

    Args:
        scene: Existing config path or bundled config name.

    Returns:
        Resolved scene config path.
    """
    path = Path(scene).expanduser()
    if path.is_file():
        return path
    if path.parent != Path("."):
        raise SceneConfigError(f"Scene config does not exist: '{path}'")

    candidates = (
        [SCENE_CONFIG_DIRECTORY / path.name]
        if path.suffix
        else [
            SCENE_CONFIG_DIRECTORY / f"{path.name}{suffix}"
            for suffix in (".yaml", ".yml", ".json")
        ]
    )
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if not matches:
        raise SceneConfigError(
            f"Unknown scene config '{scene}'. Expected a path or a config name in "
            f"'{SCENE_CONFIG_DIRECTORY}'"
        )
    return matches[0]


def load_scene_config(scene_path: str | Path) -> SceneConfig:
    """Load and validate a JSON or YAML scene configuration.

    Args:
        scene_path: Path to a `.json`, `.yaml`, or `.yml` scene file.

    Returns:
        Validated scene configuration.
    """
    path = resolve_scene_config(scene_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise SceneConfigError(
            f"Unsupported scene config extension '{path.suffix}'. Expected JSON or YAML"
        )

    try:
        with path.open(encoding="utf-8") as scene_file:
            raw_config = (
                json.load(scene_file)
                if suffix == ".json"
                else yaml.safe_load(scene_file)
            )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise SceneConfigError(
            f"Could not load scene config '{path}': {error}"
        ) from error

    return SceneConfig.from_dict(raw_config)


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise SceneConfigError(f"Unknown field(s) in {context}: {', '.join(unknown)}")


def _require_nonempty_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SceneConfigError(f"{context} field '{key}' must be a non-empty string")
    return value


def _require_integer(
    data: dict[str, Any], key: str, context: str = "scene config"
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneConfigError(f"{context} field '{key}' must be an integer")
    return value


def _require_positive_integer(data: dict[str, Any], key: str, context: str) -> int:
    value = _require_integer(data, key, context)
    if value <= 0:
        raise SceneConfigError(f"{context} field '{key}' must be a positive integer")
    return value


def _parse_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneConfigError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise SceneConfigError(f"{context} must be finite")
    return number


def _parse_vector(value: Any, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SceneConfigError(f"{context} must contain exactly {length} numbers")
    return tuple(_parse_number(component, context) for component in value)


def _parse_number_sequence(value: Any, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SceneConfigError(f"{context} must be a non-empty list of numbers")
    return tuple(_parse_number(component, context) for component in value)


def _parse_range(value: Any, context: str) -> tuple[float, float]:
    parsed = _parse_vector(value, 2, context)
    if parsed[0] > parsed[1]:
        raise SceneConfigError(f"{context} minimum must not exceed its maximum")
    return parsed
