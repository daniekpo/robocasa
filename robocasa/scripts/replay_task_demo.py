"""Replay a task-oracle demonstration from a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from robocasa.data_collection.task_replay import replay_episode


def main() -> None:
    """Parse command-line options and replay one episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local LeRobot v3 dataset root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--actions",
        action="store_true",
        help="Replay actions and report divergence instead of setting exact states",
    )
    parser.add_argument("--video", type=Path, help="Optional MP4 output path")
    parser.add_argument("--cameras", nargs="+", help="Configured cameras to render")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = replay_episode(
        args.dataset,
        args.episode,
        use_actions=args.actions,
        video_path=args.video,
        camera_names=tuple(args.cameras) if args.cameras else None,
    )
    logging.info(
        "Replayed episode %d (%d frames, mode=%s, maximum_state_error=%s)",
        result.episode_index,
        result.frame_count,
        result.mode,
        result.maximum_state_error,
    )


if __name__ == "__main__":
    main()
