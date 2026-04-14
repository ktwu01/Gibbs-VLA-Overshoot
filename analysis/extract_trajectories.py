"""Extract aligned windows around instruction switches from trajectory data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.language_table import (
    LANGUAGE_TABLE_CONTROL_HZ,
    instruction_switch_indices,
    iter_language_table_shard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a Language-Table TFRecord shard.")
    parser.add_argument("--output", required=True, help="Output `.npz` file.")
    parser.add_argument(
        "--channel",
        choices=["action", "effector_translation", "effector_target_translation"],
        default="action",
        help="Trajectory channel to export.",
    )
    parser.add_argument("--window-before", type=int, default=8, help="Frames before the switch.")
    parser.add_argument("--window-after", type=int, default=12, help="Frames after the switch.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Optional episode cap.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    windows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    episode_count = 0
    switch_count = 0

    for episode in iter_language_table_shard(args.input, limit_episodes=args.limit_episodes):
        episode_count += 1
        switches = instruction_switch_indices(episode)
        trajectory = getattr(episode, args.channel)
        for switch_index in switches:
            start = switch_index - args.window_before
            stop = switch_index + args.window_after + 1
            if start < 0 or stop > episode.num_steps:
                continue
            windows.append(trajectory[start:stop])
            metadata.append(
                {
                    "episode_id": episode.episode_id,
                    "switch_index": int(switch_index),
                    "instruction_before": episode.instruction_texts[switch_index - 1],
                    "instruction_after": episode.instruction_texts[switch_index],
                }
            )
            switch_count += 1

    info = {
        "input": args.input,
        "channel": args.channel,
        "fs": LANGUAGE_TABLE_CONTROL_HZ,
        "episode_count": episode_count,
        "switch_count": switch_count,
        "window_before": args.window_before,
        "window_after": args.window_after,
    }

    if windows:
        stacked = np.stack(windows, axis=0)
    else:
        stacked = np.empty((0, args.window_before + args.window_after + 1, 2), dtype=np.float32)

    np.savez_compressed(
        output_path,
        windows=stacked.astype(np.float32),
        metadata=np.array(metadata, dtype=object),
        info=np.array(info, dtype=object),
    )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
