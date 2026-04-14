"""Plot velocity or action-magnitude distributions from trajectory data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.language_table import LANGUAGE_TABLE_CONTROL_HZ, iter_language_table_shard
from analysis.metrics import summarize_distribution, vector_magnitude


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a Language-Table TFRecord shard.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--channel",
        choices=["action", "effector_translation", "effector_target_translation"],
        default="action",
        help="Signal source for the distribution.",
    )
    parser.add_argument("--limit-episodes", type=int, default=None, help="Optional episode cap.")
    return parser


def collect_samples(input_path: str, channel: str, limit_episodes: int | None) -> tuple[np.ndarray, dict[str, int]]:
    values: list[np.ndarray] = []
    episode_count = 0
    switching_episode_count = 0

    for episode in iter_language_table_shard(input_path, limit_episodes=limit_episodes):
        episode_count += 1
        if episode.has_instruction_switch:
            switching_episode_count += 1

        signal = getattr(episode, channel)
        if channel == "action":
            speed = vector_magnitude(signal, axis=1)
        else:
            velocity = np.gradient(signal, axis=0) * LANGUAGE_TABLE_CONTROL_HZ
            speed = vector_magnitude(velocity, axis=1)
        values.append(speed)

    if values:
        stacked = np.concatenate(values)
    else:
        stacked = np.array([], dtype=float)

    return stacked, {
        "episode_count": episode_count,
        "switching_episode_count": switching_episode_count,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, counts = collect_samples(args.input, args.channel, args.limit_episodes)
    stats = summarize_distribution(samples)
    report = {
        "input": args.input,
        "channel": args.channel,
        **counts,
        **stats,
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if samples.size:
        ax.hist(samples, bins=40, color="#1f77b4", alpha=0.9, edgecolor="white")
        ax.axvline(stats["mean"], color="#d62728", linestyle="--", linewidth=2, label="mean")
        ax.axvline(stats["p95"], color="#2ca02c", linestyle=":", linewidth=2, label="p95")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No samples found", ha="center", va="center", transform=ax.transAxes)

    title_prefix = "Action Magnitude" if args.channel == "action" else "Velocity Magnitude"
    ax.set_title(f"{title_prefix} Distribution")
    ax.set_xlabel("Per-step magnitude")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)

    note = (
        f"Episodes: {counts['episode_count']} | "
        f"Episodes with instruction switches: {counts['switching_episode_count']} | "
        f"Samples: {stats['count']}"
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
