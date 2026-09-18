import json
from pathlib import Path

import numpy as np
import pytest

from robocasa.scene_config import (
    CameraGroupConfig,
    PlacementRelation,
    SceneConfig,
    SceneConfigError,
    load_scene_config,
    resolve_scene_config,
)
from robocasa.utils.object_utils import objs_intersect_bbox
from robocasa.utils.scene_config_placement import (
    ConfiguredScenePlacementSampler,
    PlacementWorkspace,
)


class FakeObject:
    """Minimal object model exposing the bounding-box interface used by the sampler."""

    def __init__(
        self,
        name: str,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
    ):
        self.name = name
        self._minimum = np.asarray(minimum, dtype=float)
        self._maximum = np.asarray(maximum, dtype=float)

    def get_bbox_points(self) -> np.ndarray:
        minimum = self._minimum
        maximum = self._maximum
        return np.asarray(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        )


def make_scene(objects: list[dict[str, object]]) -> dict[str, object]:
    normalized_objects = []
    for raw_object in objects:
        object_config = dict(raw_object)
        normalized_object = {
            "name": object_config.pop("name"),
            "type": object_config.pop("type"),
            "style": object_config.pop("style", 1),
        }
        if isinstance(object_config.get("placement"), dict):
            placement = object_config.pop("placement")
        elif object_config.pop("placement", None) == "random":
            placement = {"type": "random"}
        elif "absolute_position" in object_config:
            placement = {
                "type": "absolute",
                "absolute_position": object_config.pop("absolute_position"),
            }
            if "absolute_quat" in object_config:
                placement["absolute_quat"] = object_config.pop("absolute_quat")
            placement.update(object_config)
            object_config.clear()
        else:
            placement = {
                "type": "relation",
                "relative_to": object_config.pop("relative_to", None),
                "relation": {
                    "type": object_config.pop("relation", None),
                    **object_config,
                },
            }
            object_config.clear()
        assert not object_config
        normalized_object["placement"] = placement
        normalized_objects.append(normalized_object)

    return {
        "scene": {
            "environment": "ConfiguredKitchen",
            "layout_id": 48,
            "style_id": 41,
        },
        "robot": {
            "type": "PandaOmron",
            "base_fixture": "island_island_group_1",
            "controller": None,
            "control_freq": 20,
        },
        "device": "keyboard",
        "render_camera": None,
        "objects": normalized_objects,
    }


def write_scene(path: Path, scene: dict[str, object]) -> None:
    path.write_text(json.dumps(scene), encoding="utf-8")


def test_load_scene_and_order_dependencies(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    write_scene(
        scene_path,
        make_scene(
            [
                {
                    "name": "right",
                    "type": "bagel",
                    "relative_to": "anchor",
                    "relation": "right",
                    "distance": 0.05,
                },
                {
                    "name": "anchor",
                    "type": "avocado",
                    "absolute_position": [1.0, 2.0, 0.9],
                },
                {
                    "name": "top",
                    "type": "can",
                    "relative_to": "right",
                    "relation": "on",
                    "distance": 0.01,
                },
            ]
        ),
    )

    config = load_scene_config(scene_path)

    assert [obj.name for obj in config.objects_in_placement_order()] == [
        "anchor",
        "right",
        "top",
    ]
    assert config.objects[0].relation is PlacementRelation.RIGHT
    assert config.robot_base_fixture == "island_island_group_1"


@pytest.mark.parametrize(
    ("objects", "message"),
    [
        (
            [
                {"name": "same", "type": "can", "absolute_position": [0, 0, 0]},
                {"name": "same", "type": "can", "absolute_position": [1, 0, 0]},
            ],
            "Duplicate object name",
        ),
        (
            [
                {
                    "name": "child",
                    "type": "can",
                    "relative_to": "missing",
                    "relation": "left",
                    "distance": 0.1,
                }
            ],
            "unknown object",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "can",
                    "relative_to": "b",
                    "relation": "left",
                    "distance": 0.1,
                },
                {
                    "name": "b",
                    "type": "can",
                    "relative_to": "a",
                    "relation": "right",
                    "distance": 0.1,
                },
            ],
            "cycle",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "can",
                    "absolute_position": [0, 0, 0],
                    "relative_to": "b",
                    "relation": "left",
                    "distance": 0.1,
                },
                {"name": "b", "type": "can", "absolute_position": [1, 0, 0]},
            ],
            "Unknown field",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "not_a_category",
                    "absolute_position": [0, 0, 0],
                }
            ],
            "Unknown RoboCasa object category",
        ),
        (
            [
                {"name": "a", "type": "can", "absolute_position": [0, 0, 0]},
                {
                    "name": "b",
                    "type": "can",
                    "relative_to": "a",
                    "relation": "diagonal",
                    "distance": 0.1,
                },
            ],
            "Unsupported relation",
        ),
        (
            [
                {"name": "a", "type": "can", "absolute_position": [0, 0, 0]},
                {
                    "name": "b",
                    "type": "can",
                    "relative_to": "a",
                    "relation": "right",
                },
            ],
            "exactly one of distance or distance_range",
        ),
        (
            [
                {"name": "a", "type": "can", "absolute_position": [0, 0, 0]},
                {
                    "name": "b",
                    "type": "can",
                    "relative_to": "a",
                    "relation": "right",
                    "distance": -0.1,
                },
            ],
            "distance must be non-negative",
        ),
        (
            [
                {
                    "name": "a",
                    "type": "can",
                    "absolute_position": [0, 0, 0],
                    "absolute_quat": [1, 1, 0, 0],
                }
            ],
            "absolute_quat must be normalized",
        ),
    ],
)
def test_invalid_scenes_fail_before_environment_creation(
    tmp_path: Path, objects: list[dict[str, object]], message: str
) -> None:
    scene_path = tmp_path / "invalid.json"
    write_scene(scene_path, make_scene(objects))

    with pytest.raises(SceneConfigError, match=message):
        load_scene_config(scene_path)


def test_yaml_and_json_are_equivalent(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    scene = make_scene(
        [{"name": "anchor", "type": "avocado", "absolute_position": [0, 0, 0.9]}]
    )
    json_path = tmp_path / "scene.json"
    yaml_path = tmp_path / "scene.yaml"
    write_scene(json_path, scene)
    yaml_path.write_text(yaml.safe_dump(scene), encoding="utf-8")

    assert load_scene_config(json_path) == load_scene_config(yaml_path)


def test_load_bundled_scene_by_name() -> None:
    config = load_scene_config("expanded_example_scene")

    assert resolve_scene_config("expanded_example_scene").suffix == ".yaml"
    assert config.workspace is not None
    assert config.workspace.forward_range[1] - config.workspace.forward_range[
        0
    ] == pytest.approx(0.4826)
    assert config.workspace.lateral_range[1] - config.workspace.lateral_range[
        0
    ] == pytest.approx(0.8128)
    assert isinstance(config.cameras, CameraGroupConfig)
    assert config.cameras.names == (
        "robot_left",
        "robot_right",
        "robot_and_counter",
    )
    assert (config.cameras.width, config.cameras.height) == (640, 480)
    assert config.cameras.depth is True
    assert config.cameras.placements[0].roll == -7
    assert config.cameras.placements[1].roll == 7
    assert config.robot_start_joints == (
        -0.085558,
        0.231276,
        -0.066809,
        -1.315765,
        0.015312,
        1.546556,
        0.530547,
    )


def test_camera_group_validation() -> None:
    scene = make_scene(
        [{"name": "anchor", "type": "avocado", "absolute_position": [0, 0, 1]}]
    )
    scene["cameras"] = {
        "width": 256,
        "height": 256,
        "depth": True,
        "placements": [
            {
                "name": "fixed",
                "position": [1, 0, 1],
                "look_at": [0, 0, 1],
            },
            {
                "name": "fixed",
                "position": [-1, 0, 1],
                "look_at": [0, 0, 1],
            },
        ],
    }

    with pytest.raises(SceneConfigError, match="Duplicate camera name"):
        SceneConfig.from_dict(scene)


def test_robot_start_joints_must_be_numeric() -> None:
    scene = make_scene(
        [{"name": "anchor", "type": "avocado", "absolute_position": [0, 0, 1]}]
    )
    scene["robot"]["start_joints"] = [0.0, "invalid"]

    with pytest.raises(SceneConfigError, match="robot start_joints must be a number"):
        SceneConfig.from_dict(scene)


@pytest.mark.parametrize("style", [None, 0, -1, "fixed", True])
def test_object_style_must_be_positive_integer_or_random(
    tmp_path: Path, style: object
) -> None:
    scene = make_scene(
        [{"name": "anchor", "type": "avocado", "absolute_position": [0, 0, 0]}]
    )
    if style is None:
        scene["objects"][0].pop("style")
    else:
        scene["objects"][0]["style"] = style
    scene_path = tmp_path / "invalid_style.json"
    write_scene(scene_path, scene)

    with pytest.raises(SceneConfigError, match="style must be a positive integer"):
        load_scene_config(scene_path)


def test_random_object_style_is_explicitly_supported(tmp_path: Path) -> None:
    scene_path = tmp_path / "random_style.json"
    write_scene(
        scene_path,
        make_scene(
            [
                {
                    "name": "anchor",
                    "type": "avocado",
                    "style": "random",
                    "absolute_position": [0, 0, 0],
                }
            ]
        ),
    )

    config = load_scene_config(scene_path)

    assert config.objects[0].object_style == "random"


def test_absolute_position_is_bbox_bottom_center_contact_point(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    write_scene(
        scene_path,
        make_scene(
            [
                {
                    "name": "offset",
                    "type": "can",
                    "absolute_position": [1.0, 2.0, 0.9],
                }
            ]
        ),
    )
    config = load_scene_config(scene_path)
    objects = {"offset": FakeObject("offset", (-0.1, -0.2, -0.3), (0.3, 0.2, 0.5))}

    placements = ConfiguredScenePlacementSampler(config, objects).sample()
    position, quaternion, _ = placements["offset"]

    np.testing.assert_allclose(position, [0.9, 2.0, 1.2])
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])


def test_absolute_position_accounts_for_rotated_bounds(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    half_sqrt_two = np.sqrt(0.5)
    write_scene(
        scene_path,
        make_scene(
            [
                {
                    "name": "rotated",
                    "type": "can",
                    "absolute_position": [1.0, 2.0, 0.9],
                    "absolute_quat": [half_sqrt_two, 0, 0, half_sqrt_two],
                }
            ]
        ),
    )
    config = load_scene_config(scene_path)
    objects = {"rotated": FakeObject("rotated", (-0.1, -0.2, -0.3), (0.3, 0.4, 0.5))}

    placements = ConfiguredScenePlacementSampler(config, objects).sample()
    position = placements["rotated"][0]

    np.testing.assert_allclose(position, [1.1, 1.9, 1.2])


@pytest.mark.parametrize(
    ("relation", "expected_position"),
    [
        ("left", [-0.8, 0.0, 0.2]),
        ("right", [0.8, 0.0, 0.2]),
        ("in_front_of", [0.0, 0.9, 0.2]),
        ("behind", [0.0, -0.9, 0.2]),
        ("on", [0.0, 0.0, 1.0]),
    ],
)
def test_relative_placements_use_free_gap_between_bounds(
    tmp_path: Path, relation: str, expected_position: list[float]
) -> None:
    scene_path = tmp_path / "scene.json"
    write_scene(
        scene_path,
        make_scene(
            [
                {"name": "anchor", "type": "can", "absolute_position": [0, 0, 0]},
                {
                    "name": "child",
                    "type": "bagel",
                    "relative_to": "anchor",
                    "relation": relation,
                    "distance": 0.1,
                },
            ]
        ),
    )
    config = load_scene_config(scene_path)
    objects = {
        "anchor": FakeObject("anchor", (-0.2, -0.3, -0.3), (0.2, 0.3, 0.4)),
        "child": FakeObject("child", (-0.5, -0.5, -0.2), (0.5, 0.5, 0.2)),
    }

    placements = ConfiguredScenePlacementSampler(config, objects).sample()
    child_position = placements["child"][0]

    np.testing.assert_allclose(child_position, expected_position)


def test_relative_distance_and_orthogonal_jitter_are_sampled(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    write_scene(
        scene_path,
        make_scene(
            [
                {"name": "anchor", "type": "can", "absolute_position": [0, 0, 0]},
                {
                    "name": "child",
                    "type": "bagel",
                    "relative_to": "anchor",
                    "relation": "right",
                    "distance_range": [0.1, 0.2],
                    "orthogonal_jitter_range": [-0.05, 0.05],
                },
            ]
        ),
    )
    config = load_scene_config(scene_path)
    objects = {
        "anchor": FakeObject("anchor", (-0.2, -0.3, -0.3), (0.2, 0.3, 0.4)),
        "child": FakeObject("child", (-0.5, -0.5, -0.2), (0.5, 0.5, 0.2)),
    }

    placements = ConfiguredScenePlacementSampler(
        config, objects, rng=np.random.default_rng(7)
    ).sample()
    anchor_position = placements["anchor"][0]
    child_position = placements["child"][0]
    anchor_maximum_x = anchor_position[0] + 0.2
    child_minimum_x = child_position[0] - 0.5
    gap = child_minimum_x - anchor_maximum_x

    assert 0.1 <= gap <= 0.2
    assert -0.05 <= child_position[1] <= 0.05
    assert child_position[1] != 0.0


def test_random_placement_stays_in_workspace_and_avoids_objects(
    tmp_path: Path,
) -> None:
    scene = make_scene(
        [
            {"name": "random", "type": "bagel", "placement": "random"},
            {"name": "anchor", "type": "can", "absolute_position": [0.35, 0, 0]},
        ]
    )
    scene["workspace"] = {
        "fixture": "island_island_group_1",
        "forward_range": [0.2, 1.0],
        "lateral_range": [-0.4, 0.4],
        "margin": 0.02,
    }
    scene_path = tmp_path / "scene.json"
    write_scene(scene_path, scene)
    config = load_scene_config(scene_path)
    objects = {
        "anchor": FakeObject("anchor", (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        "random": FakeObject("random", (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
    }
    workspace = PlacementWorkspace(
        robot_position=np.zeros(3),
        robot_yaw=0.0,
        forward_range=(0.2, 1.0),
        lateral_range=(-0.4, 0.4),
        support_z=0.0,
        fixture_p0=np.asarray([0.0, -1.0, 0.0]),
        fixture_px=np.asarray([2.0, -1.0, 0.0]),
        fixture_py=np.asarray([0.0, 1.0, 0.0]),
        margin=0.02,
    )

    placements = ConfiguredScenePlacementSampler(
        config,
        objects,
        rng=np.random.default_rng(3),
        workspace=workspace,
    ).sample()
    random_position = placements["random"][0]
    random_points = objects["random"].get_bbox_points() + random_position
    anchor_points = objects["anchor"].get_bbox_points() + placements["anchor"][0]

    assert np.all(random_points[:, 0] >= 0.22)
    assert np.all(random_points[:, 0] <= 0.98)
    assert np.all(random_points[:, 1] >= -0.38)
    assert np.all(random_points[:, 1] <= 0.38)
    assert not objs_intersect_bbox(random_points, anchor_points)


@pytest.mark.parametrize(
    ("position", "boundary"),
    [
        ([0.1, 0.0, 0.0], "forward minimum"),
        ([0.9, 0.0, 0.0], "forward maximum"),
        ([0.5, -0.35, 0.0], "lateral minimum"),
        ([0.5, 0.35, 0.0], "lateral maximum"),
    ],
)
def test_configured_placement_outside_workspace_raises(
    tmp_path: Path, position: list[float], boundary: str
) -> None:
    scene = make_scene(
        [{"name": "outside", "type": "can", "absolute_position": position}]
    )
    scene["workspace"] = {
        "fixture": "island_island_group_1",
        "forward_range": [0.2, 1.0],
        "lateral_range": [-0.4, 0.4],
        "margin": 0.02,
    }
    scene_path = tmp_path / "scene.json"
    write_scene(scene_path, scene)
    config = load_scene_config(scene_path)
    objects = {"outside": FakeObject("outside", (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))}
    workspace = PlacementWorkspace(
        robot_position=np.zeros(3),
        robot_yaw=0.0,
        forward_range=(0.2, 1.0),
        lateral_range=(-0.4, 0.4),
        support_z=0.0,
        fixture_p0=np.asarray([0.0, -1.0, 0.0]),
        fixture_px=np.asarray([2.0, -1.0, 0.0]),
        fixture_py=np.asarray([0.0, 1.0, 0.0]),
        margin=0.02,
    )

    with pytest.raises(
        SceneConfigError,
        match=f"Object 'outside' is outside the configured workspace: {boundary}",
    ):
        ConfiguredScenePlacementSampler(config, objects, workspace=workspace).sample()


def test_relative_placement_outside_workspace_raises(tmp_path: Path) -> None:
    scene = make_scene(
        [
            {"name": "anchor", "type": "can", "absolute_position": [0.5, 0, 0]},
            {
                "name": "outside",
                "type": "can",
                "relative_to": "anchor",
                "relation": "right",
                "distance": 0.5,
            },
        ]
    )
    scene["workspace"] = {
        "fixture": "island_island_group_1",
        "forward_range": [0.2, 1.0],
        "lateral_range": [-0.4, 0.4],
        "margin": 0.02,
    }
    scene_path = tmp_path / "scene.json"
    write_scene(scene_path, scene)
    config = load_scene_config(scene_path)
    objects = {
        name: FakeObject(name, (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
        for name in ("anchor", "outside")
    }
    workspace = PlacementWorkspace(
        robot_position=np.zeros(3),
        robot_yaw=0.0,
        forward_range=(0.2, 1.0),
        lateral_range=(-0.4, 0.4),
        support_z=0.0,
        fixture_p0=np.asarray([0.0, -1.0, 0.0]),
        fixture_px=np.asarray([2.0, -1.0, 0.0]),
        fixture_py=np.asarray([0.0, 1.0, 0.0]),
        margin=0.02,
    )

    with pytest.raises(
        SceneConfigError,
        match="Object 'outside' is outside the configured workspace: forward maximum",
    ):
        ConfiguredScenePlacementSampler(config, objects, workspace=workspace).sample()
