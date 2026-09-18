"""Typed loading and validation for task-driven RoboCasa rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from robocasa.scene_config import SceneConfig, SceneConfigError, load_scene_config

MIN_COMMANDS = 2
MAX_COMMANDS = 5
TASK_CONFIG_DIRECTORY = Path(__file__).parent / "task_configs"
SUPPORTED_SUFFIXES = {".yaml", ".yml"}


class TaskConfigError(ValueError):
    """Raised when a task configuration is malformed or inconsistent."""


class TaskAction(str, Enum):
    """Actions supported by the task command schema."""

    MOVE = "move"


class TaskRelation(str, Enum):
    """Relations supported by move commands."""

    IN = "in"
    ON = "on"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"


@dataclass(frozen=True)
class MoveCommand:
    """Move one configured object into a relation with another object."""

    object: str
    relation: TaskRelation
    target: str

    @property
    def action(self) -> TaskAction:
        """Return the command action."""
        return TaskAction.MOVE

    def __str__(self) -> str:
        """Return a compact representation for logs and dataset annotations."""
        return f"move({self.object}, {self.relation.value}, {self.target})"

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> MoveCommand:
        """Parse and validate one move command."""
        if not isinstance(data, dict):
            raise TaskConfigError(f"Command at index {index} must be a mapping")
        _reject_unknown_keys(
            data,
            {"action", "object", "relation", "target"},
            f"command at index {index}",
        )
        action = _require_nonempty_string(data, "action", f"command at index {index}")
        try:
            TaskAction(action)
        except ValueError as error:
            raise TaskConfigError(
                f"Unsupported action {action!r} at command index {index}. "
                f"Expected: {TaskAction.MOVE.value}"
            ) from error

        object_name = _require_nonempty_string(
            data, "object", f"command at index {index}"
        )
        target = _require_nonempty_string(data, "target", f"command at index {index}")
        if object_name == target:
            raise TaskConfigError(
                f"Command at index {index} cannot move an object relative to itself"
            )
        raw_relation = _require_nonempty_string(
            data, "relation", f"command at index {index}"
        )
        try:
            relation = TaskRelation(raw_relation)
        except ValueError as error:
            supported = ", ".join(relation.value for relation in TaskRelation)
            raise TaskConfigError(
                f"Unsupported relation {raw_relation!r} at command index {index}. "
                f"Expected one of: {supported}"
            ) from error
        return cls(object=object_name, relation=relation, target=target)


@dataclass(frozen=True)
class TaskConfig:
    """A validated sequence of manipulation commands for one scene."""

    name: str
    scene_name: str
    commands: tuple[MoveCommand, ...]
    scene_config: SceneConfig

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        task_directory: Path | None = None,
    ) -> TaskConfig:
        """Parse a task mapping and validate its object references against its scene."""
        if not isinstance(data, dict):
            raise TaskConfigError("Task config must contain a top-level mapping")
        _reject_unknown_keys(data, {"name", "scene_name", "commands"}, "task config")
        name = _require_nonempty_string(data, "name", "task config")
        scene_name = _require_nonempty_string(data, "scene_name", "task config")
        raw_commands = data.get("commands")
        if not isinstance(raw_commands, list):
            raise TaskConfigError("task config field 'commands' must be a list")
        if not MIN_COMMANDS <= len(raw_commands) <= MAX_COMMANDS:
            raise TaskConfigError(
                f"Task must contain between {MIN_COMMANDS} and {MAX_COMMANDS} commands"
            )
        commands = tuple(
            MoveCommand.from_dict(command, index)
            for index, command in enumerate(raw_commands)
        )

        scene_reference = _resolve_scene_reference(scene_name, task_directory)
        try:
            scene_config = load_scene_config(scene_reference)
        except SceneConfigError as error:
            raise TaskConfigError(
                f"Task '{name}' references an invalid scene '{scene_name}': {error}"
            ) from error

        object_names = {obj.name for obj in scene_config.objects}
        for index, command in enumerate(commands):
            for role, object_name in (
                ("object", command.object),
                ("target", command.target),
            ):
                if object_name not in object_names:
                    raise TaskConfigError(
                        f"Command at index {index} references unknown scene {role} "
                        f"'{object_name}'"
                    )
        return cls(
            name=name,
            scene_name=scene_name,
            commands=commands,
            scene_config=scene_config,
        )


def resolve_task_config(task: str | Path) -> Path:
    """Resolve an existing path or a bundled task config name."""
    path = Path(task).expanduser()
    if path.is_file():
        return path
    if path.parent != Path("."):
        raise TaskConfigError(f"Task config does not exist: '{path}'")

    candidates = (
        [TASK_CONFIG_DIRECTORY / path.name]
        if path.suffix
        else [
            TASK_CONFIG_DIRECTORY / f"{path.name}{suffix}"
            for suffix in (".yaml", ".yml")
        ]
    )
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if not matches:
        raise TaskConfigError(
            f"Unknown task config '{task}'. Expected a path or a config name in "
            f"'{TASK_CONFIG_DIRECTORY}'"
        )
    return matches[0]


def load_task_config(task_path: str | Path) -> TaskConfig:
    """Load and validate a YAML task configuration."""
    path = resolve_task_config(task_path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise TaskConfigError(
            f"Unsupported task config extension '{path.suffix}'. Expected YAML"
        )
    try:
        with path.open(encoding="utf-8") as task_file:
            raw_config = yaml.safe_load(task_file)
    except (OSError, yaml.YAMLError) as error:
        raise TaskConfigError(
            f"Could not load task config '{path}': {error}"
        ) from error
    return TaskConfig.from_dict(raw_config, task_directory=path.parent)


def _resolve_scene_reference(
    scene_name: str, task_directory: Path | None
) -> str | Path:
    candidate = Path(scene_name).expanduser()
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    if task_directory is not None:
        relative_candidate = task_directory / candidate
        if relative_candidate.is_file():
            return relative_candidate
    return scene_name


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise TaskConfigError(f"Unknown field(s) in {context}: {', '.join(unknown)}")


def _require_nonempty_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskConfigError(f"{context} field '{key}' must be a non-empty string")
    return value
