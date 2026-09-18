"""Tests for event-aligned ground-truth and symbolic scene graphs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robocasa.scene_config import load_scene_config
from robocasa.scene_graph import (
    DIRECTIONAL_RELATIONS,
    SceneGraphEdge,
    SceneGraphSnapshot,
    SceneGraphTrace,
    build_ground_truth_scene_graph,
    save_scene_graph_trace,
    update_symbolic_scene_graph,
)
from robocasa.task_config import MoveCommand, TaskConfig, TaskRelation


class FakeObject:
    """Axis-aligned unit object exposing RoboCasa's bounding-box API."""

    def __init__(self, name: str):
        self.name = name

    def get_bbox_points(
        self, trans: np.ndarray, rot: np.ndarray
    ) -> np.ndarray:
        del rot
        offsets = np.asarray(
            [
                [x, y, z]
                for x in (-0.1, 0.1)
                for y in (-0.1, 0.1)
                for z in (-0.1, 0.1)
            ]
        )
        return offsets + trans


class FakeEnv:
    """Minimal live-state interface used by graph extraction."""

    def __init__(self, positions: dict[str, tuple[float, float, float]]):
        names = list(positions)
        self.objects = {name: FakeObject(name) for name in names}
        self.obj_body_id = {name: index for index, name in enumerate(names)}
        self.sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.asarray([positions[name] for name in names]),
                body_xquat=np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in names]),
            )
        )


def make_task() -> TaskConfig:
    scene = load_scene_config("expanded_example_scene")
    return TaskConfig(
        name="graph_test",
        scene_name="expanded_example_scene",
        commands=(
            MoveCommand("can", TaskRelation.IN, "basket"),
            MoveCommand("basket", TaskRelation.LEFT_OF, "bagged_food"),
        ),
        scene_config=scene,
    )


def make_env() -> FakeEnv:
    positions = {
        "avocado": (-2.0, 0.0, 0.2),
        "bagel": (-1.0, 0.0, 0.2),
        "bagged_food": (0.0, 0.0, 0.2),
        "basket": (1.0, 0.0, 0.2),
        "can": (2.0, 0.0, 0.2),
        "boxed_drink": (1.0, 2.0, 0.2),
    }
    return FakeEnv(positions)


def test_ground_truth_graph_has_all_nodes_support_and_nearest_peer_edges() -> None:
    snapshot = build_ground_truth_scene_graph(
        make_env(),
        make_task(),
        in_predicate=lambda source, target: (source, target) == ("can", "basket"),
        on_predicate=lambda source, target: (source, target) == ("bagel", "basket"),
    )

    assert [(node.name, node.category) for node in snapshot.nodes][:2] == [
        ("avocado", "avocado"),
        ("bagel", "bagel"),
    ]
    assert SceneGraphEdge("can", TaskRelation.IN, "basket") in snapshot.edges
    assert SceneGraphEdge("bagel", TaskRelation.ON, "basket") in snapshot.edges
    directional = [
        edge for edge in snapshot.edges if edge.relation in DIRECTIONAL_RELATIONS
    ]
    assert SceneGraphEdge("avocado", TaskRelation.LEFT_OF, "bagel") in directional
    assert SceneGraphEdge("basket", TaskRelation.BEHIND, "boxed_drink") in directional
    assert snapshot.to_dict()["edges"] == sorted(
        snapshot.to_dict()["edges"],
        key=lambda edge: (edge["source"], edge["relation"], edge["target"]),
    )


def test_symbolic_support_update_preserves_incoming_support() -> None:
    initial = SceneGraphSnapshot(
        stage_index=0,
        simulator_state_index=0,
        nodes=(),
        edges=(
            SceneGraphEdge("can", TaskRelation.IN, "basket"),
            SceneGraphEdge("basket", TaskRelation.ON, "bagged_food"),
        ),
    )

    updated = update_symbolic_scene_graph(
        initial,
        MoveCommand("basket", TaskRelation.IN, "boxed_drink"),
        command_index=0,
        simulator_state_index=17,
    )

    assert SceneGraphEdge("can", TaskRelation.IN, "basket") in updated.edges
    assert SceneGraphEdge(
        "basket",
        TaskRelation.IN,
        "boxed_drink",
        introduced_stage=1,
        from_command=True,
    ) in updated.edges
    assert not any(
        edge.source == "basket" and edge.relation is TaskRelation.ON
        for edge in updated.edges
    )
    assert updated.stage_index == 1
    assert updated.completed_command_index == 0
    assert updated.simulator_state_index == 17


def test_symbolic_direction_update_removes_stale_relations() -> None:
    initial = SceneGraphSnapshot(
        stage_index=0,
        simulator_state_index=0,
        nodes=(),
        edges=(
            SceneGraphEdge("can", TaskRelation.LEFT_OF, "basket"),
            SceneGraphEdge("can", TaskRelation.BEHIND, "bagel"),
            SceneGraphEdge("boxed_drink", TaskRelation.RIGHT_OF, "can"),
        ),
    )

    updated = update_symbolic_scene_graph(
        initial,
        MoveCommand("can", TaskRelation.RIGHT_OF, "basket"),
        command_index=0,
        simulator_state_index=8,
    )
    incident = [
        edge
        for edge in updated.edges
        if edge.relation in DIRECTIONAL_RELATIONS
        and "can" in {edge.source, edge.target}
    ]

    assert incident == [
        SceneGraphEdge(
            "can",
            TaskRelation.RIGHT_OF,
            "basket",
            introduced_stage=1,
            from_command=True,
        )
    ]


def test_symbolic_direction_move_removes_old_support() -> None:
    initial = SceneGraphSnapshot(
        stage_index=0,
        simulator_state_index=0,
        nodes=(),
        edges=(SceneGraphEdge("can", TaskRelation.IN, "basket"),),
    )

    updated = update_symbolic_scene_graph(
        initial,
        MoveCommand("can", TaskRelation.LEFT_OF, "bagel"),
        command_index=0,
        simulator_state_index=5,
    )

    assert not any(edge.relation is TaskRelation.IN for edge in updated.edges)


def test_moving_receptacle_invalidates_contents_directional_edges() -> None:
    initial = SceneGraphSnapshot(
        stage_index=0,
        simulator_state_index=0,
        nodes=(),
        edges=(
            SceneGraphEdge("can", TaskRelation.IN, "basket"),
            SceneGraphEdge("can", TaskRelation.LEFT_OF, "bagel"),
        ),
    )

    updated = update_symbolic_scene_graph(
        initial,
        MoveCommand("basket", TaskRelation.RIGHT_OF, "bagged_food"),
        command_index=0,
        simulator_state_index=9,
    )

    assert SceneGraphEdge("can", TaskRelation.IN, "basket") in updated.edges
    assert not any(
        edge.relation in DIRECTIONAL_RELATIONS and edge.source == "can"
        for edge in updated.edges
    )


def test_trace_enforces_alignment_and_saves_versioned_json(tmp_path: Path) -> None:
    initial = build_ground_truth_scene_graph(
        make_env(),
        make_task(),
        in_predicate=lambda _source, _target: False,
        on_predicate=lambda _source, _target: False,
    )
    symbolic = update_symbolic_scene_graph(
        initial,
        make_task().commands[0],
        command_index=0,
        simulator_state_index=12,
    )
    trace = SceneGraphTrace("symbolic", (initial,)).append(symbolic)
    path = tmp_path / "scene_graph_symbolic.json"

    save_scene_graph_trace(trace, path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["version"] == 1
    assert saved["graph_type"] == "symbolic"
    assert len(saved["snapshots"]) == 2
    assert saved["snapshots"][1]["completed_command"] == "move(can, in, basket)"
    assert saved["snapshots"][1]["simulator_state_index"] == 12
