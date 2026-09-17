from pathlib import Path

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.transform_utils import convert_quat, quat2mat

import robocasa  # noqa: F401 - imports and registers RoboCasa environments
from robocasa.scene_config import load_scene_config


def test_configured_kitchen_headless_reset() -> None:
    scene_path = (
        Path(__file__).parents[1]
        / "robocasa"
        / "scene_configs"
        / "expanded_example_scene.yaml"
    )
    config = load_scene_config(scene_path)
    env = robosuite.make(
        env_name=config.environment,
        robots=config.robot,
        controller_configs=load_composite_controller_config(
            controller=config.controller, robot=config.robot
        ),
        scene_config=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=0,
    )

    try:
        env.reset()
        first_model_paths = {
            object_name: obj.mjcf_path for object_name, obj in env.objects.items()
        }
        avocado_position, avocado_quaternion, avocado = env.object_placements["avocado"]
        rotation = quat2mat(convert_quat(avocado_quaternion, to="xyzw"))
        world_bounds = (
            np.asarray(avocado.get_bbox_points()) @ rotation.T + avocado_position
        )
        np.testing.assert_allclose(world_bounds[:, :2].mean(axis=0), [1.55, -3.9])
        np.testing.assert_allclose(world_bounds[:, 2].min(), 0.92)

        island = env.fixtures[config.robot_base_fixture]
        p0, px, py, _ = island.get_ext_sites(relative=False)
        island_corners = np.asarray([p0, px, py, px + py - p0])
        island_minimum = island_corners[:, :2].min(axis=0)
        island_maximum = island_corners[:, :2].max(axis=0)

        assert set(env.objects) == {obj.name for obj in config.objects}
        for object_name, obj in env.objects.items():
            position = env.sim.data.get_body_xpos(obj.root_body)
            assert np.isfinite(position).all()

            initial_position, initial_quaternion, _ = env.object_placements[object_name]
            rotation = quat2mat(convert_quat(initial_quaternion, to="xyzw"))
            initial_bounds = (
                np.asarray(obj.get_bbox_points()) @ rotation.T + initial_position
            )
            assert np.all(initial_bounds[:, :2].min(axis=0) >= island_minimum)
            assert np.all(initial_bounds[:, :2].max(axis=0) <= island_maximum)

        assert env.init_robot_base_ref == config.robot_base_fixture
        assert env.init_robot_base_pos[0] > island_maximum[0]
        assert np.linalg.norm(avocado_position[:2] - env.init_robot_base_pos[:2]) < 1.0

        random_position, random_quaternion, random_object = env.object_placements[
            "boxed_drink"
        ]
        random_rotation = quat2mat(convert_quat(random_quaternion, to="xyzw"))
        random_points = (
            np.asarray(random_object.get_bbox_points()) @ random_rotation.T
            + random_position
        )
        robot_yaw = env.init_robot_base_ori[2]
        robot_rotation = np.asarray(
            [
                [np.cos(robot_yaw), -np.sin(robot_yaw)],
                [np.sin(robot_yaw), np.cos(robot_yaw)],
            ]
        )
        robot_local_points = (
            random_points[:, :2] - env.init_robot_base_pos[:2]
        ) @ robot_rotation
        assert config.workspace is not None
        assert np.all(
            robot_local_points[:, 0]
            >= config.workspace.forward_range[0] + config.workspace.margin
        )
        assert np.all(
            robot_local_points[:, 0]
            <= config.workspace.forward_range[1] - config.workspace.margin
        )
        assert np.all(
            robot_local_points[:, 1]
            >= config.workspace.lateral_range[0] + config.workspace.margin
        )
        assert np.all(
            robot_local_points[:, 1]
            <= config.workspace.lateral_range[1] - config.workspace.margin
        )

        env.reset()
        second_model_paths = {
            object_name: obj.mjcf_path for object_name, obj in env.objects.items()
        }
        assert second_model_paths == first_model_paths
    finally:
        env.close()
