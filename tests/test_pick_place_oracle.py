"""Unit tests for reusable pick-place oracle components."""

from types import SimpleNamespace

import numpy as np

from robocasa.oracles import pick_place
from robocasa.oracles.pick_place import (
    GraspStrategyRegistry,
    MotionConfig,
    MotionExecutor,
    ObjectEstimate,
    TaskOracle,
    verify_relation,
)
from robocasa.scene_config import load_scene_config


class FakeController:
    """Minimal base-frame delta controller."""

    input_type = "delta"
    input_ref_frame = "base"
    control_dim = 6
    output_min = np.full(6, -0.1)
    output_max = np.full(6, 0.1)


class FakeCompositeController:
    """Provide the controller and identity robot base pose."""

    def get_controller(self, arm_name: str) -> FakeController:
        assert arm_name == "right"
        return FakeController()

    def get_controller_base_pose(
        self, arm_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        assert arm_name == "right"
        return np.zeros(3), np.eye(3)


class FakeRobot:
    """Create a compact action vector from controller parts."""

    robot_model = SimpleNamespace(naming_prefix="robot0_")
    composite_controller = FakeCompositeController()

    def create_action_vector(self, action_parts: dict[str, object]) -> np.ndarray:
        arm = np.asarray(action_parts["right"])
        gripper = np.asarray(action_parts["right_gripper"])
        return np.concatenate((arm, gripper))


class FakeState:
    """State object matching robosuite's flatten API."""

    def __init__(self, value: float) -> None:
        self.value = value

    def flatten(self) -> np.ndarray:
        return np.asarray([self.value])


class FakeMotionEnv:
    """Move the fake end effector according to delta actions."""

    def __init__(self) -> None:
        self.robots = [FakeRobot()]
        self.position = np.zeros(3)
        self.step_count = 0
        self.sim = SimpleNamespace(get_state=lambda: FakeState(self.step_count))

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.position = self.position + action[:3] * 0.1
        self.step_count += 1
        return (
            {"robot0_eef_pos": self.position.copy()},
            0.0,
            False,
            False,
            {},
        )


def test_motion_executor_emits_aligned_pre_and_next_states() -> None:
    env = FakeMotionEnv()
    transitions = []
    executor = MotionExecutor(
        env,
        config=MotionConfig(max_waypoint_steps=2),
        transition_callback=transitions.append,
    )

    result = executor.move_to_position(
        {"robot0_eef_pos": np.zeros(3)},
        np.asarray([0.05, 0.0, 0.0]),
        gripper_closed=False,
        stage="test waypoint",
        command_index=2,
        command_text="move(can, in, basket)",
    )

    np.testing.assert_allclose(result["robot0_eef_pos"], [0.05, 0.0, 0.0])
    assert len(transitions) == 1
    np.testing.assert_allclose(transitions[0].simulator_state, [0.0])
    np.testing.assert_allclose(transitions[0].next_simulator_state, [1.0])
    assert transitions[0].command_index == 2
    assert transitions[0].stage == "test waypoint"


def test_directional_relation_requires_separation_and_orthogonal_overlap(
    monkeypatch,
) -> None:
    bounds = {
        "source": np.asarray([[0.0, 0.2, 0.0], [0.2, 0.8, 0.2]]),
        "target": np.asarray([[0.5, 0.0, 0.0], [0.8, 1.0, 0.2]]),
    }
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, object_name: bounds[object_name],
    )

    assert verify_relation(SimpleNamespace(), "source", "left_of", "target")
    bounds["source"] = np.asarray([[0.0, 1.2, 0.0], [0.2, 1.4, 0.2]])
    assert not verify_relation(SimpleNamespace(), "source", "left_of", "target")


def test_on_relation_requires_contact_and_vertical_support(monkeypatch) -> None:
    bounds = {
        "source": np.asarray([[0.2, 0.2, 0.5], [0.4, 0.4, 0.7]]),
        "target": np.asarray([[0.0, 0.0, 0.2], [1.0, 1.0, 0.5]]),
    }
    simulation_env = SimpleNamespace(
        objects={"source": object(), "target": object()},
        check_contact=lambda source, target: True,
    )
    env = SimpleNamespace(env=simulation_env)
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, object_name: bounds[object_name],
    )

    assert verify_relation(env, "source", "on", "target")
    simulation_env.check_contact = lambda source, target: False
    assert not verify_relation(env, "source", "on", "target")


def test_basket_grasp_uses_edge_nearest_robot(monkeypatch) -> None:
    env = SimpleNamespace(
        config=load_scene_config("expanded_example_scene"),
        robots=[FakeRobot()],
        _last_observation={
            "robot0_eef_quat": np.asarray([0.0, 1.0, 0.0, 0.0])
        },
    )
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, name: np.asarray([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.5]]),
    )
    estimate = ObjectEstimate("basket", np.zeros(3), ())

    point = GraspStrategyRegistry().get_grasp_point(env, "basket", estimate)

    np.testing.assert_allclose(point, [0.0, -1.985, 0.47])


def test_directional_drop_target_clears_target_and_preserves_grasp_offset(
    monkeypatch,
) -> None:
    bounds = {
        "source": np.asarray([[-0.1, -0.2, 0.0], [0.1, 0.2, 0.4]]),
        "target": np.asarray([[0.8, 0.8, 0.0], [1.2, 1.2, 0.4]]),
    }
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, object_name: bounds[object_name],
    )
    oracle = TaskOracle.__new__(TaskOracle)
    oracle.env = SimpleNamespace()
    oracle.config = MotionConfig(
        directional_clearance=0.05, placement_clearance=0.02
    )
    target = ObjectEstimate("target", np.asarray([1.0, 1.0, 0.4]), ())

    drop = oracle._get_drop_position(
        "source",
        np.asarray([0.05, 0.0, 0.3]),
        "target",
        target,
        "right_of",
    )

    # Desired source center is one source half-width + target half-width +
    # clearance to the right; the off-center grasp remains off-center.
    np.testing.assert_allclose(drop, [1.4, 1.0, 0.32])


def test_on_drop_target_places_source_bottom_above_target(monkeypatch) -> None:
    bounds = {
        "source": np.asarray([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.4]]),
        "target": np.asarray([[0.8, 0.8, 0.0], [1.2, 1.2, 0.5]]),
    }
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, object_name: bounds[object_name],
    )
    oracle = TaskOracle.__new__(TaskOracle)
    oracle.env = SimpleNamespace()
    oracle.config = MotionConfig(placement_clearance=0.02)
    target = ObjectEstimate("target", np.asarray([1.0, 1.0, 0.5]), ())

    drop = oracle._get_drop_position(
        "source", np.asarray([0.0, 0.0, 0.3]), "target", target, "on"
    )

    np.testing.assert_allclose(drop, [1.0, 1.0, 0.82])


def test_in_drop_target_uses_receptacle_center_and_grasp_offset(monkeypatch) -> None:
    bounds = {
        "source": np.asarray([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.4]]),
        "target": np.asarray([[0.5, 0.8, 0.0], [1.5, 1.2, 0.3]]),
    }
    monkeypatch.setattr(
        pick_place,
        "get_object_bounds",
        lambda env, object_name: bounds[object_name],
    )
    oracle = TaskOracle.__new__(TaskOracle)
    oracle.env = SimpleNamespace()
    oracle.config = MotionConfig(drop_clearance=0.2)
    off_center_detection = ObjectEstimate(
        "target", np.asarray([0.6, 0.9, 0.3]), ()
    )

    drop = oracle._get_drop_position(
        "source",
        np.asarray([0.05, -0.025, 0.3]),
        "target",
        off_center_detection,
        "in",
    )

    np.testing.assert_allclose(drop, [1.05, 0.975, 0.5])
