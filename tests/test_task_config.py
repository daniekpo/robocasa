"""Tests for structured long-horizon task configuration."""

from pathlib import Path

import pytest
import yaml

from robocasa.task_config import (
    MoveCommand,
    TaskAction,
    TaskConfigError,
    TaskRelation,
    load_task_config,
    resolve_task_config,
)


def write_task(path: Path, commands: list[dict[str, str]], **updates: object) -> None:
    task: dict[str, object] = {
        "name": "test_task",
        "scene_name": "expanded_example_scene",
        "commands": commands,
    }
    task.update(updates)
    path.write_text(yaml.safe_dump(task), encoding="utf-8")


def move(
    object_name: str = "can",
    relation: str = "in",
    target: str = "basket",
) -> dict[str, str]:
    return {
        "action": "move",
        "object": object_name,
        "relation": relation,
        "target": target,
    }


def test_load_bundled_task_and_format_commands() -> None:
    task = load_task_config("example_put_objects_in_basket")

    assert resolve_task_config("example_put_objects_in_basket").suffix == ".yaml"
    assert task.name == "example_put_objects_in_basket"
    assert task.scene_config.objects[0].name == "avocado"
    assert task.commands[0] == MoveCommand("can", TaskRelation.IN, "basket")
    assert task.commands[0].action is TaskAction.MOVE
    assert str(task.commands[0]) == "move(can, in, basket)"


@pytest.mark.parametrize("command_count", [2, 5])
def test_command_count_boundaries_are_accepted(
    tmp_path: Path, command_count: int
) -> None:
    path = tmp_path / "task.yaml"
    write_task(path, [move() for _ in range(command_count)])

    assert len(load_task_config(path).commands) == command_count


@pytest.mark.parametrize("command_count", [0, 1, 6])
def test_command_count_outside_boundaries_is_rejected(
    tmp_path: Path, command_count: int
) -> None:
    path = tmp_path / "task.yaml"
    write_task(path, [move() for _ in range(command_count)])

    with pytest.raises(TaskConfigError, match="between 2 and 5"):
        load_task_config(path)


@pytest.mark.parametrize(
    ("commands", "message"),
    [
        ([move(relation="beside"), move()], "Unsupported relation"),
        ([{**move(), "action": "open"}, move()], "Unsupported action"),
        ([move(object_name="missing"), move()], "unknown scene object"),
        ([move(target="missing"), move()], "unknown scene target"),
        ([move(object_name="can", target="can"), move()], "relative to itself"),
        ([{**move(), "extra": "no"}, move()], "Unknown field"),
    ],
)
def test_invalid_commands_are_rejected_before_execution(
    tmp_path: Path,
    commands: list[dict[str, str]],
    message: str,
) -> None:
    path = tmp_path / "task.yaml"
    write_task(path, commands)

    with pytest.raises(TaskConfigError, match=message):
        load_task_config(path)


def test_scene_path_is_resolved_relative_to_task(tmp_path: Path) -> None:
    source_scene = (
        Path(__file__).parents[1]
        / "robocasa/scene_configs/expanded_example_scene.yaml"
    )
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(source_scene.read_text(encoding="utf-8"), encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    write_task(task_path, [move(), move()], scene_name="scene.yaml")

    assert load_task_config(task_path).scene_name == "scene.yaml"
