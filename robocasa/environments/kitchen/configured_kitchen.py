"""Kitchen environment whose objects and poses come from a scene config."""

from __future__ import annotations

from typing import Any

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
        kwargs["layout_and_style_ids"] = [
            (scene_config.layout_id, scene_config.style_id)
        ]
        kwargs["init_robot_base_ref"] = scene_config.robot_base_fixture
        kwargs["robot_spawn_deviation_pos_x"] = 0.0
        kwargs["robot_spawn_deviation_pos_y"] = 0.0
        kwargs["robot_spawn_deviation_rot"] = 0.0
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        """Build the kitchen and validate its configured robot fixture."""
        super()._setup_model()
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
