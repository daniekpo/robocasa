"""Run random actions in a configured scene while displaying every camera."""

from __future__ import annotations

import argparse
import logging
import math

import cv2
import numpy as np

from robocasa.example_env import SceneGymEnv, make_scene_env

DEFAULT_SCENE = "expanded_example_scene"
DEFAULT_STEPS = 10_000
WINDOW_NAME = "RoboCasa configured scene cameras"

LOGGER = logging.getLogger(__name__)


def create_camera_grid(
    observation: dict[str, np.ndarray], camera_names: tuple[str, ...]
) -> np.ndarray:
    """Create a labeled BGR image containing every configured RGB camera.

    Args:
        observation: Current Gymnasium observation dictionary.
        camera_names: Camera names in display order.

    Returns:
        Camera images arranged in a nearly square grid for OpenCV display.
    """
    frames: list[np.ndarray] = []
    for camera_name in camera_names:
        rgb_image = observation[f"{camera_name}_image"]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        labeled_image = bgr_image.copy()
        cv2.putText(
            labeled_image,
            camera_name,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(labeled_image)

    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    blank_frame = np.zeros_like(frames[0])
    frames.extend(blank_frame.copy() for _ in range(rows * columns - len(frames)))
    return np.vstack(
        [np.hstack(frames[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def run_random_actions(scene: str, steps: int, seed: int) -> None:
    """Run a configured scene with bounded random actions and live cameras.

    Args:
        scene: Bundled scene name or path to a JSON/YAML scene config.
        steps: Number of simulation control steps to run.
        seed: Environment and action-space random seed.
    """
    env = make_scene_env(scene, horizon=steps)
    if not isinstance(env, SceneGymEnv):
        raise TypeError("make_scene_env() did not return a Gymnasium environment")
    if env.config.cameras is None:
        env.close()
        raise ValueError("The selected scene does not define any cameras")

    camera_names = env.config.cameras.names
    env.action_space.seed(seed)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        observation, _ = env.reset(seed=seed)
        for step_index in range(steps):
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)

            camera_grid = create_camera_grid(observation, camera_names)
            cv2.imshow(WINDOW_NAME, camera_grid)
            if cv2.waitKey(1) & 0xFF == 27:
                LOGGER.info("Stopped after %d steps", step_index + 1)
                break
            if terminated or truncated:
                LOGGER.info("Episode ended after %d steps", step_index + 1)
                break
        else:
            LOGGER.info("Completed %d random-action steps", steps)
    finally:
        env.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Parse command-line arguments and run the visualization example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Bundled scene name or path to a JSON/YAML scene config",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of random-action control steps to run",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_random_actions(args.scene, args.steps, args.seed)


if __name__ == "__main__":
    main()
