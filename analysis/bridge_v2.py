"""
Bridge V2 data loader for the instruction-swap experiment.

Supports multiple backends:
  1. Local ``.npz`` sample cache (fastest, no internet needed after first run)
  2. HuggingFace ``datasets`` with ``IPEC-COMMUNITY/bridge_orig_lerobot``
  3. TFDS ``tfds.load("bridge")`` (requires compatible protobuf)

The dataset provides 256x256 RGB images from up to 4 camera views.
We use ``image_0`` (primary) by default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "bridge_v2_cache"

# Representative Bridge V2 instructions for the swap experiment
BRIDGE_V2_INSTRUCTIONS = [
    "pick up the red block",
    "push the blue block to the left",
    "put the corn in the pot",
    "move the cloth to the right",
    "pick up the sushi and place it on the plate",
    "close the drawer",
    "open the drawer",
    "put the sweet potato in the pot",
    "pick up the mushroom and place it on the plate",
    "move the spoon to the towel",
]


@dataclass
class BridgeV2Observation:
    """A single observation frame extracted from Bridge V2."""

    image: np.ndarray  # (H, W, 3) uint8
    instruction: str
    state: np.ndarray  # (7,) EEF state
    episode_id: int
    step_index: int


def _cache_path(episode_index: int, step_index: int) -> Path:
    return _CACHE_DIR / f"obs_ep{episode_index}_st{step_index}.npz"


def _save_to_cache(obs: BridgeV2Observation) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(obs.episode_id, obs.step_index)
    np.savez_compressed(
        path,
        image=obs.image,
        instruction=np.array(obs.instruction),
        state=obs.state,
        episode_id=np.array(obs.episode_id),
        step_index=np.array(obs.step_index),
    )


def _load_from_cache(episode_index: int, step_index: int) -> BridgeV2Observation | None:
    path = _cache_path(episode_index, step_index)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return BridgeV2Observation(
        image=data["image"],
        instruction=str(data["instruction"]),
        state=data["state"],
        episode_id=int(data["episode_id"]),
        step_index=int(data["step_index"]),
    )


def _load_via_lerobot(
    episode_index: int = 0,
    step_index: int | None = None,
    dataset_id: str = "IPEC-COMMUNITY/bridge_orig_lerobot",
) -> BridgeV2Observation:
    """Load via HuggingFace datasets (LeRobot parquet format)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)

    current_ep = -1
    ep_steps: list[dict] = []

    for row in ds:
        ep_id = row.get("episode_index", row.get("episode_id", 0))
        if ep_id != current_ep:
            if current_ep == episode_index:
                break
            current_ep = ep_id
            ep_steps = []
        if ep_id == episode_index:
            ep_steps.append(row)

    if not ep_steps:
        raise ValueError(f"Episode {episode_index} not found")

    idx = step_index if step_index is not None else len(ep_steps) // 2
    idx = max(0, min(idx, len(ep_steps) - 1))
    step = ep_steps[idx]

    # LeRobot format: image is a PIL Image or path
    image = step.get("observation.images.image_0", step.get("observation.image", None))
    if image is None:
        for k in step:
            if "image" in k.lower():
                image = step[k]
                break
    if image is None:
        raise ValueError(f"No image found in step keys: {list(step.keys())}")

    if hasattr(image, "convert"):  # PIL Image
        image = np.array(image.convert("RGB"))
    else:
        image = np.asarray(image, dtype=np.uint8)

    instruction = step.get("language_instruction", step.get("task", ""))
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")

    state = np.asarray(step.get("observation.state", np.zeros(7)), dtype=np.float32)

    return BridgeV2Observation(
        image=image,
        instruction=str(instruction),
        state=state,
        episode_id=episode_index,
        step_index=idx,
    )


def _load_via_tfds(
    episode_index: int = 0,
    step_index: int | None = None,
    camera: str = "image_0",
    split: str = "train",
    data_dir: str | None = None,
) -> BridgeV2Observation:
    """Load via tensorflow_datasets (original RLDS format)."""
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


def _generate_synthetic_bridge_observation(
    episode_index: int = 0,
    step_index: int = 0,
    instruction: str | None = None,
) -> BridgeV2Observation:
    """
    Generate a synthetic observation resembling Bridge V2 for testing.

    Creates a 256x256 image with a simple tabletop scene (colored rectangles
    as objects) so the experiment pipeline can run without real data.
    """
    rng = np.random.default_rng(episode_index * 1000 + step_index)

    # Brown table background
    image = np.full((256, 256, 3), [139, 119, 101], dtype=np.uint8)

    # Add 2-3 colored "blocks"
    colors = [(200, 50, 50), (50, 50, 200), (50, 180, 50), (200, 200, 50)]
    for i in range(rng.integers(2, 4)):
        x, y = rng.integers(30, 200, size=2)
        w, h = rng.integers(20, 50, size=2)
        c = colors[i % len(colors)]
        image[y : y + h, x : x + w] = c

    # A "gripper" shape
    gx, gy = 128 + rng.integers(-20, 20), 128 + rng.integers(-20, 20)
    image[gy - 3 : gy + 3, gx - 15 : gx + 15] = [80, 80, 80]

    state = rng.uniform(-0.5, 0.5, size=7).astype(np.float32)

    if instruction is None:
        instruction = BRIDGE_V2_INSTRUCTIONS[episode_index % len(BRIDGE_V2_INSTRUCTIONS)]

    return BridgeV2Observation(
        image=image,
        instruction=instruction,
        state=state,
        episode_id=episode_index,
        step_index=step_index,
    )


def load_single_observation(
    episode_index: int = 0,
    step_index: int | None = None,
    camera: str = "image_0",
    split: str = "train",
    data_dir: str | None = None,
    backend: str = "auto",
) -> BridgeV2Observation:
    """
    Load one observation frame from Bridge V2.

    Parameters
    ----------
    backend:
        ``"auto"`` tries cache → lerobot → tfds → synthetic fallback.
        ``"tfds"``, ``"lerobot"``, ``"synthetic"`` force a specific backend.
    """
    _step = step_index if step_index is not None else 0

    if backend == "auto":
        # 1. Try cache
        cached = _load_from_cache(episode_index, _step)
        if cached is not None:
            return cached

        # 2. Try lerobot (HuggingFace)
        try:
            obs = _load_via_lerobot(episode_index, step_index)
            _save_to_cache(obs)
            return obs
        except Exception:
            pass

        # 3. Try TFDS
        try:
            obs = _load_via_tfds(episode_index, step_index, camera, split, data_dir)
            _save_to_cache(obs)
            return obs
        except Exception:
            pass

        # 4. Synthetic fallback
        print("WARNING: Using synthetic Bridge V2 observation (real data unavailable)")
        return _generate_synthetic_bridge_observation(episode_index, _step)

    elif backend == "lerobot":
        return _load_via_lerobot(episode_index, step_index)
    elif backend == "tfds":
        return _load_via_tfds(episode_index, step_index, camera, split, data_dir)
    elif backend == "synthetic":
        return _generate_synthetic_bridge_observation(episode_index, _step)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def iter_bridge_v2_observations(
    split: str = "train",
    camera: str = "image_0",
    limit_episodes: int | None = None,
    steps_per_episode: int = 1,
    data_dir: str | None = None,
) -> Iterator[BridgeV2Observation]:
    """Yield observation frames from Bridge V2 via TFDS."""
    import tensorflow_datasets as tfds

    ds = tfds.load("bridge", split=split, data_dir=data_dir)

    for ep_idx, episode in enumerate(ds):
        if limit_episodes is not None and ep_idx >= limit_episodes:
            break

        steps = list(episode["steps"])
        n_steps = len(steps)
        if n_steps == 0:
            continue

        if steps_per_episode >= n_steps:
            indices = list(range(n_steps))
        else:
            indices = np.linspace(0, n_steps - 1, steps_per_episode, dtype=int).tolist()

        ep_id = int(episode.get("episode_metadata", {}).get("episode_id", ep_idx))

        for step_idx in indices:
            step = steps[step_idx]
            obs = step["observation"]
            image = obs[camera].numpy()
            instruction = step["language_instruction"].numpy().decode("utf-8")
            state = obs["state"].numpy()

            yield BridgeV2Observation(
                image=image,
                instruction=instruction,
                state=state,
                episode_id=ep_id,
                step_index=step_idx,
            )
