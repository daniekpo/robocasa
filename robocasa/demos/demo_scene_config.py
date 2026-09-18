"""Launch an interactive RoboCasa kitchen from a JSON or YAML scene config."""

from __future__ import annotations

import argparse
import logging

import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa  # noqa: F401 - imports and registers RoboCasa environments
from robocasa import macros
from robocasa.scene_config import SceneConfig, load_scene_config

LOGGER = logging.getLogger(__name__)


def create_device(scene_config: SceneConfig, env: object) -> object:
    """Create the teleoperation device selected by the scene config.

    Args:
        scene_config: Validated scene settings.
        env: Interactive RoboCasa environment.

    Returns:
        Configured robosuite input device.
    """
    if scene_config.device == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)

    from robosuite.devices import SpaceMouse

    return SpaceMouse(
        env=env,
        pos_sensitivity=4.0,
        rot_sensitivity=4.0,
        vendor_id=macros.SPACEMOUSE_VENDOR_ID,
        product_id=macros.SPACEMOUSE_PRODUCT_ID,
    )


def run_scene(scene_config: SceneConfig) -> None:
    """Create and run an interactive configured kitchen.

    Args:
        scene_config: Validated scene to run.
    """
    from robocasa.scripts.collect_demos import collect_human_trajectory
    from robocasa.wrappers.enclosing_wall_render_wrapper import (
        EnclosingWallRenderWrapper,
        install_enclosing_wall_hotkeys,
    )

    controller_config = load_composite_controller_config(
        controller=scene_config.controller,
        robot=scene_config.robot,
    )
    env = robosuite.make(
        env_name=scene_config.environment,
        robots=scene_config.robot,
        controller_configs=controller_config,
        scene_config=scene_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=scene_config.render_camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=scene_config.control_freq,
        renderer=scene_config.renderer,
        seed=scene_config.seed,
    )
    env = EnclosingWallRenderWrapper(
        env, alpha=0.1, enabled=not scene_config.show_walls
    )
    install_enclosing_wall_hotkeys(env)
    device = create_device(scene_config, env)

    LOGGER.info(
        "Running layout %s, style %s with %s control",
        scene_config.layout_id,
        scene_config.style_id,
        scene_config.device,
    )
    try:
        while True:
            collect_human_trajectory(
                env,
                device,
                "right",
                "single-arm-opposed",
                mirror_actions=True,
                render=(scene_config.renderer != "mjviewer"),
                max_fr=30,
                print_info=False,
            )
    finally:
        env.close()


def main() -> None:
    """Parse command-line arguments and launch the configured scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        required=True,
        help="Bundled config name or path to a JSON/YAML scene configuration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_scene(load_scene_config(args.scene_config))


if __name__ == "__main__":
    main()
