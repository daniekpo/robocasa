"""Event-level scene graph extraction and symbolic task updates."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robocasa.task_config import MoveCommand, TaskConfig, TaskRelation

SCENE_GRAPH_VERSION = 1
SUPPORT_RELATIONS = {TaskRelation.IN, TaskRelation.ON}
DIRECTIONAL_RELATIONS = {
    TaskRelation.LEFT_OF,
    TaskRelation.RIGHT_OF,
    TaskRelation.IN_FRONT_OF,
    TaskRelation.BEHIND,
}

RelationPredicate = Callable[[str, str], bool]


@dataclass(frozen=True)
class SceneGraphNode:
    """An object represented in a scene graph."""

    name: str
    category: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible node mapping."""
        return {"name": self.name, "category": self.category}


@dataclass(frozen=True)
class SceneGraphEdge:
    """A directed relation between two scene objects."""

    source: str
    relation: TaskRelation
    target: str
    introduced_stage: int = 0
    from_command: bool = False

    def to_dict(self) -> dict[str, str]:
        """Return the stable public JSON representation of this edge."""
        return {
            "source": self.source,
            "relation": self.relation.value,
            "target": self.target,
        }


@dataclass(frozen=True)
class SceneGraphSnapshot:
    """A scene graph aligned with one task-stage boundary and simulator state."""

    stage_index: int
    simulator_state_index: int
    nodes: tuple[SceneGraphNode, ...]
    edges: tuple[SceneGraphEdge, ...]
    completed_command_index: int | None = None
    completed_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this snapshot in the versioned JSON schema."""
        return {
            "stage_index": self.stage_index,
            "completed_command_index": self.completed_command_index,
            "completed_command": self.completed_command,
            "simulator_state_index": self.simulator_state_index,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=_edge_key)],
        }


@dataclass(frozen=True)
class SceneGraphTrace:
    """A complete initial-plus-post-command graph trace for one episode."""

    graph_type: Literal["ground_truth", "symbolic"]
    snapshots: tuple[SceneGraphSnapshot, ...]
    version: int = SCENE_GRAPH_VERSION

    def append(self, snapshot: SceneGraphSnapshot) -> SceneGraphTrace:
        """Return a trace with a consecutive snapshot appended."""
        expected_stage = len(self.snapshots)
        if snapshot.stage_index != expected_stage:
            raise ValueError(
                f"Expected scene graph stage {expected_stage}, got "
                f"{snapshot.stage_index}"
            )
        return replace(self, snapshots=(*self.snapshots, snapshot))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned JSON document."""
        return {
            "version": self.version,
            "graph_type": self.graph_type,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }


def build_ground_truth_scene_graph(
    env: Any,
    task_config: TaskConfig,
    *,
    stage_index: int = 0,
    simulator_state_index: int = 0,
    completed_command_index: int | None = None,
    completed_command: MoveCommand | None = None,
    in_predicate: RelationPredicate | None = None,
    on_predicate: RelationPredicate | None = None,
) -> SceneGraphSnapshot:
    """Extract support and sparse directional relations from a live environment.

    Args:
        env: RoboCasa environment exposing configured objects and simulator state.
        task_config: Task whose validated scene defines node order and categories.
        stage_index: Number of successfully completed task commands.
        simulator_state_index: Index of the aligned state in the saved state sequence.
        completed_command_index: Index of the command completed at this boundary.
        completed_command: Command completed at this boundary, or ``None`` initially.
        in_predicate: Optional containment predicate for tests or custom environments.
        on_predicate: Optional vertical-support predicate for tests or custom
            environments.

    Returns:
        Deterministically ordered scene graph snapshot.
    """
    nodes = tuple(
        SceneGraphNode(name=obj.name, category=obj.object_type)
        for obj in task_config.scene_config.objects
    )
    names = [node.name for node in nodes]
    bounds = {name: _get_object_bounds(env, name) for name in names}
    is_in = in_predicate or (lambda source, target: _is_in(env, source, target))
    is_on = on_predicate or (
        lambda source, target: _is_on(env, source, target, bounds)
    )

    edges: list[SceneGraphEdge] = []
    for source in names:
        for target in names:
            if source == target:
                continue
            if is_in(source, target):
                edges.append(SceneGraphEdge(source, TaskRelation.IN, target))
            if is_on(source, target):
                edges.append(SceneGraphEdge(source, TaskRelation.ON, target))

    for first, second in _nearest_peer_pairs(names, bounds):
        relation = _dominant_axis_relation(bounds[first], bounds[second])
        edges.append(SceneGraphEdge(first, relation, second))

    return SceneGraphSnapshot(
        stage_index=stage_index,
        simulator_state_index=simulator_state_index,
        completed_command_index=completed_command_index,
        completed_command=(
            str(completed_command) if completed_command is not None else None
        ),
        nodes=nodes,
        edges=tuple(sorted(set(edges), key=_edge_key)),
    )


def update_symbolic_scene_graph(
    snapshot: SceneGraphSnapshot,
    command: MoveCommand,
    *,
    command_index: int,
    simulator_state_index: int,
) -> SceneGraphSnapshot:
    """Apply one successful task command to a symbolic graph snapshot.

    Moving the source removes its old outgoing support and directional edges.
    Incoming support is retained so objects inside a moved receptacle remain
    attached to it, then the verified command relation is added.
    """
    stage_index = snapshot.stage_index + 1
    edges = list(snapshot.edges)
    carried_objects = {
        edge.source
        for edge in edges
        if edge.relation is TaskRelation.IN and edge.target == command.object
    }
    moved_objects = {command.object, *carried_objects}
    # Moving an object invalidates its old outgoing support and every
    # directional relation involving it. Incoming support is preserved so the
    # contents of a moved receptacle remain symbolically attached.
    edges = [
        edge
        for edge in edges
        if not (
            (edge.source == command.object and edge.relation in SUPPORT_RELATIONS)
            or (
                edge.relation in DIRECTIONAL_RELATIONS
                and moved_objects.intersection({edge.source, edge.target})
            )
        )
    ]

    new_edge = SceneGraphEdge(
        source=command.object,
        relation=command.relation,
        target=command.target,
        introduced_stage=stage_index,
        from_command=True,
    )
    edges = [edge for edge in edges if edge != new_edge]
    edges.append(new_edge)

    return SceneGraphSnapshot(
        stage_index=stage_index,
        simulator_state_index=simulator_state_index,
        completed_command_index=command_index,
        completed_command=str(command),
        nodes=snapshot.nodes,
        edges=tuple(sorted(set(edges), key=_edge_key)),
    )


def save_scene_graph_trace(trace: SceneGraphTrace, path: str | Path) -> None:
    """Write a graph trace as deterministic, human-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(trace.to_dict(), output_file, indent=2)
        output_file.write("\n")


def _get_object_bounds(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    obj = env.objects[name]
    body_id = env.obj_body_id[name]
    position = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
    quaternion_wxyz = np.asarray(env.sim.data.body_xquat[body_id], dtype=float)
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    points = np.asarray(
        obj.get_bbox_points(trans=position, rot=quaternion_xyzw), dtype=float
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Object '{name}' bounding box must have shape (N, 3)")
    return np.min(points, axis=0), np.max(points, axis=0)


def _is_in(env: Any, source: str, target: str) -> bool:
    from robocasa.utils.object_utils import check_obj_in_receptacle

    return bool(check_obj_in_receptacle(env, source, target))


def _is_on(
    env: Any,
    source: str,
    target: str,
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> bool:
    if not env.check_contact(env.objects[source], env.objects[target]):
        return False
    source_minimum, source_maximum = bounds[source]
    target_minimum, target_maximum = bounds[target]
    overlaps_x = (
        source_maximum[0] >= target_minimum[0]
        and source_minimum[0] <= target_maximum[0]
    )
    overlaps_y = (
        source_maximum[1] >= target_minimum[1]
        and source_minimum[1] <= target_maximum[1]
    )
    vertical_gap = abs(source_minimum[2] - target_maximum[2])
    return bool(overlaps_x and overlaps_y and vertical_gap <= 0.025)


def _nearest_peer_pairs(
    names: Sequence[str],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[str, str], ...]:
    centers = {
        name: (bounds[name][0][:2] + bounds[name][1][:2]) / 2 for name in names
    }
    pair_set: set[tuple[str, str]] = set()
    peer_count = min(2, len(names) - 1)
    for name in names:
        peers = sorted(
            (other for other in names if other != name),
            key=lambda other: (
                float(np.linalg.norm(centers[name] - centers[other])),
                other,
            ),
        )[:peer_count]
        pair_set.update(tuple(sorted((name, peer))) for peer in peers)
    return tuple(sorted(pair_set))


def _dominant_axis_relation(
    source_bounds: tuple[np.ndarray, np.ndarray],
    target_bounds: tuple[np.ndarray, np.ndarray],
) -> TaskRelation:
    source_center = (source_bounds[0] + source_bounds[1]) / 2
    target_center = (target_bounds[0] + target_bounds[1]) / 2
    delta = source_center - target_center
    if abs(delta[0]) >= abs(delta[1]):
        return TaskRelation.LEFT_OF if delta[0] < 0 else TaskRelation.RIGHT_OF
    return TaskRelation.BEHIND if delta[1] < 0 else TaskRelation.IN_FRONT_OF


def _edge_key(edge: SceneGraphEdge) -> tuple[str, str, str]:
    return edge.source, edge.relation.value, edge.target
