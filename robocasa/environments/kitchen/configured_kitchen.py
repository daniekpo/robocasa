"""Kitchen environment whose objects and poses come from a scene config."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
from robocasa.scene_config import SceneConfig, SceneConfigError
from robocasa.utils.scene_config_placement import (
    ConfiguredScenePlacementSampler,
    PlacementWorkspace,
)


class ConfiguredKitchen(Kitchen):
    """Interactive kitchen populated from a validated scene configuration."""

    def __init__(self, scene_config: SceneConfig, *args: Any, **kwargs: Any):
        """Initialize a configured kitchen.

        Args:
            scene_config: Validated scene configuration.
            *args: Positional arguments forwarded to :class:`Kitchen`.
            **kwargs: Keyword arguments forwarded to :class:`Kitchen`.
        """
        if not isinstance(scene_config, SceneConfig):
            raise TypeError("scene_config must be a validated SceneConfig")

        self.scene_config = scene_config
        controlled_keys = {
            "init_robot_base_ref",
            "layout_and_style_ids",
            "layout_ids",
            "robot_spawn_deviation_pos_x",
            "robot_spawn_deviation_pos_y",
            "robot_spawn_deviation_rot",
            "style_ids",
        }
        if controlled_keys.intersection(kwargs):
            raise ValueError(
                "Layout, style, and robot base are controlled by scene_config"
            )
        if (
            scene_config.robot_start_joints is not None
            and "initialization_noise" in kwargs
        ):
            raise ValueError(
                "Robot initialization noise is disabled when start_joints is configured"
            )
        kwargs["layout_and_style_ids"] = [
            (scene_config.layout_id, scene_config.style_id)
        ]
        kwargs["init_robot_base_ref"] = scene_config.robot_base_fixture
        kwargs["robot_spawn_deviation_pos_x"] = 0.0
        kwargs["robot_spawn_deviation_pos_y"] = 0.0
        kwargs["robot_spawn_deviation_rot"] = 0.0
        if scene_config.robot_start_joints is not None:
            kwargs["initialization_noise"] = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        """Build the kitchen and validate its configured robot fixture."""
        super()._setup_model()
        start_joints = self.scene_config.robot_start_joints
        if start_joints is not None:
            robot = self.robots[0]
            expected_joint_count = len(robot.init_qpos)
            if len(start_joints) != expected_joint_count:
                raise SceneConfigError(
                    f"Robot '{self.scene_config.robot}' start_joints must contain "
                    f"{expected_joint_count} values, got {len(start_joints)}"
                )
            robot.init_qpos = np.asarray(start_joints)
        fixture_name = self.scene_config.robot_base_fixture
        if fixture_name is not None and fixture_name not in self.fixtures:
            raise SceneConfigError(
                f"Robot base fixture '{fixture_name}' does not exist in "
                f"layout {self.scene_config.layout_id}"
            )
        workspace = self.scene_config.workspace
        if workspace is not None and workspace.fixture not in self.fixtures:
            raise SceneConfigError(
                f"Workspace fixture '{workspace.fixture}' does not exist in "
                f"layout {self.scene_config.layout_id}"
            )
        self._configure_scene_cameras()

    def _setup_references(self) -> None:
        """Set simulator references and validate configured robot joint limits."""
        super()._setup_references()
        start_joints = self.scene_config.robot_start_joints
        if start_joints is None:
            return
        robot = self.robots[0]
        joint_ranges = self.sim.model.jnt_range[robot._ref_joint_indexes]
        start_joints_array = np.asarray(start_joints)
        invalid = np.flatnonzero(
            (start_joints_array < joint_ranges[:, 0])
            | (start_joints_array > joint_ranges[:, 1])
        )
        if invalid.size:
            joint_index = int(invalid[0])
            lower, upper = joint_ranges[joint_index]
            raise SceneConfigError(
                f"Robot start_joints[{joint_index}]={start_joints[joint_index]} is "
                f"outside joint limits [{lower}, {upper}]"
            )

    def _configure_scene_cameras(self) -> None:
        """Add config-defined cameras after fixture poses have been resolved."""
        camera_group = self.scene_config.cameras
        if camera_group is None:
            return

        for camera in camera_group.placements:
            position = np.asarray(camera.position)
            look_at = np.asarray(camera.look_at)
            if camera.fixture is not None:
                if camera.fixture not in self.fixtures:
                    raise SceneConfigError(
                        f"Camera '{camera.name}' fixture '{camera.fixture}' does not "
                        f"exist in layout {self.scene_config.layout_id}"
                    )
                fixture = self.fixtures[camera.fixture]
                fixture_position = np.asarray(fixture.pos)
                fixture_rotation = Rotation.from_euler("z", fixture.rot).as_matrix()
                position = fixture_position + fixture_rotation @ position
                look_at = fixture_position + fixture_rotation @ look_at

            self._cam_configs[camera.name] = {
                "pos": position.tolist(),
                "quat": _camera_quaternion(position, look_at, camera.roll),
                "camera_attribs": {"fovy": str(camera.fovy)},
            }

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """Convert configured categories into RoboCasa object configurations."""
        return [
            {
                "name": object_config.name,
                "obj_groups": self._get_object_group(
                    object_config.object_type, object_config.object_style
                ),
                "placement": {},
            }
            for object_config in self.scene_config.objects
        ]

    def _get_object_group(self, object_type: str, object_style: int | str) -> str:
        """Resolve an object style to a category or one deterministic model path."""
        if object_style == "random":
            return object_type

        model_paths = [
            model_path
            for registry in self.obj_registries
            if registry in OBJ_CATEGORIES[object_type]
            for model_path in OBJ_CATEGORIES[object_type][registry].mjcf_paths
        ]
        style_index = object_style - 1
        if style_index >= len(model_paths):
            raise SceneConfigError(
                f"Object category '{object_type}' has {len(model_paths)} styles in "
                f"registries {self.obj_registries}, but style {object_style} was requested"
            )
        return model_paths[style_index]

    def _get_placement_initializer(self) -> ConfiguredScenePlacementSampler:
        """Create the exact, relationship-aware scene placement sampler."""
        workspace = self._get_placement_workspace()
        return ConfiguredScenePlacementSampler(
            self.scene_config,
            self.objects,
            rng=self.rng,
            workspace=workspace,
        )

    def _get_placement_workspace(self) -> PlacementWorkspace | None:
        """Resolve robot-local workspace bounds into world geometry."""
        workspace_config = self.scene_config.workspace
        if workspace_config is None:
            return None

        assert self.scene_config.robot_base_fixture is not None
        robot_fixture = self.fixtures[self.scene_config.robot_base_fixture]
        robot_position, robot_orientation = EnvUtils.compute_robot_base_placement_pose(
            self, robot_fixture
        )
        workspace_fixture = self.fixtures[workspace_config.fixture]
        fixture_p0, fixture_px, fixture_py, fixture_pz = (
            workspace_fixture.get_ext_sites(relative=False)
        )
        return PlacementWorkspace(
            robot_position=robot_position,
            robot_yaw=float(robot_orientation[2]),
            forward_range=workspace_config.forward_range,
            lateral_range=workspace_config.lateral_range,
            support_z=float(fixture_pz[2]),
            fixture_p0=fixture_p0,
            fixture_px=fixture_px,
            fixture_py=fixture_py,
            margin=workspace_config.margin,
        )


def _camera_quaternion(
    position: np.ndarray, look_at: np.ndarray, roll: float = 0.0
) -> list[float]:
    """Return a MuJoCo wxyz quaternion for a camera aimed at a point."""
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    rotation = rotation @ Rotation.from_euler("z", roll, degrees=True).as_matrix()
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]].tolist()
