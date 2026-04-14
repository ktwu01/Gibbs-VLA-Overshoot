"""
Bridge V2 data loader for the instruction-swap experiment.

Loads episodes from the ``bridge`` TFDS dataset (BridgeData V2,
RAIL Berkeley) and extracts observation images + language instructions.

The dataset provides 256x256 RGB images from up to 4 camera views.
We use ``image_0`` (primary) by default.

Requirements: ``pip install tensorflow tensorflow-datasets``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class BridgeV2Observation:
    """A single observation frame extracted from Bridge V2."""

    image: np.ndarray  # (256, 256, 3) uint8
    instruction: str
    state: np.ndarray  # (7,) EEF state
    episode_id: int
    step_index: int


def iter_bridge_v2_observations(
    split: str = "train",
    camera: str = "image_0",
    limit_episodes: int | None = None,
    steps_per_episode: int = 1,
    data_dir: str | None = None,
) -> Iterator[BridgeV2Observation]:
    """
    Yield observation frames from Bridge V2.

    Parameters
    ----------
    split:
        TFDS split, e.g. ``"train"`` or ``"train[:10]"``.
    camera:
        Which camera view to use (``image_0`` through ``image_3``).
    limit_episodes:
        Stop after this many episodes.
    steps_per_episode:
        How many evenly-spaced frames to sample per episode.
        Use ``1`` to get a single mid-episode frame (good for the swap
        experiment where we want a static observation).
    data_dir:
        Optional TFDS data directory override.
    """
    import tensorflow_datasets as tfds

    ds = tfds.load("bridge", split=split, data_dir=data_dir)

    for ep_idx, episode in enumerate(ds):
        if limit_episodes is not None and ep_idx >= limit_episodes:
            break

        steps = list(episode["steps"])
        n_steps = len(steps)
        if n_steps == 0:
            continue

        # Sample evenly spaced indices
        if steps_per_episode >= n_steps:
            indices = list(range(n_steps))
        else:
            indices = np.linspace(0, n_steps - 1, steps_per_episode, dtype=int).tolist()

        ep_id = int(episode.get("episode_metadata", {}).get("episode_id", ep_idx))

        for step_idx in indices:
            step = steps[step_idx]
            obs = step["observation"]
            image = obs[camera].numpy()  # (256, 256, 3) uint8
            instruction = step["language_instruction"].numpy().decode("utf-8")
            state = obs["state"].numpy()  # (7,)

            yield BridgeV2Observation(
                image=image,
                instruction=instruction,
                state=state,
                episode_id=ep_id,
                step_index=step_idx,
            )


def get_unique_instructions(
    split: str = "train[:50]",
    data_dir: str | None = None,
) -> list[str]:
    """Return unique instruction strings from the first N episodes."""
    seen: set[str] = set()
    for obs in iter_bridge_v2_observations(split=split, data_dir=data_dir, limit_episodes=50):
        seen.add(obs.instruction)
    return sorted(seen)


def load_single_observation(
    episode_index: int = 0,
    step_index: int | None = None,
    camera: str = "image_0",
    split: str = "train",
    data_dir: str | None = None,
) -> BridgeV2Observation:
    """
    Load one specific observation frame.

    If ``step_index`` is None, picks the middle frame of the episode.
    """
    import tensorflow_datasets as tfds

    ds = tfds.load("bridge", split=split, data_dir=data_dir)

    for ep_idx, episode in enumerate(ds):
        if ep_idx < episode_index:
            continue

        steps = list(episode["steps"])
        n_steps = len(steps)
        if n_steps == 0:
            raise ValueError(f"Episode {episode_index} has no steps")

        idx = step_index if step_index is not None else n_steps // 2
        idx = max(0, min(idx, n_steps - 1))

        step = steps[idx]
        obs = step["observation"]
        ep_id = int(episode.get("episode_metadata", {}).get("episode_id", ep_idx))

        return BridgeV2Observation(
            image=obs[camera].numpy(),
            instruction=step["language_instruction"].numpy().decode("utf-8"),
            state=obs["state"].numpy(),
            episode_id=ep_id,
            step_index=idx,
        )

    raise ValueError(f"Episode index {episode_index} out of range")
