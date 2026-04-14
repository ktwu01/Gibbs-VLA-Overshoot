"""Minimal Language-Table RLDS shard reader built on top of TFRecord."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tfrecord import sequence_loader

LANGUAGE_TABLE_CONTROL_HZ = 10.0
INSTRUCTION_WIDTH = 512
XY_DIM = 2


@dataclass
class LanguageTableEpisode:
    """Decoded per-episode arrays from a TFDS/RLDS shard."""

    episode_id: str
    instruction_texts: list[str]
    action: np.ndarray
    effector_translation: np.ndarray
    effector_target_translation: np.ndarray
    reward: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.reward.shape[0])

    @property
    def has_instruction_switch(self) -> bool:
        return len(set(self.instruction_texts)) > 1


def _decode_instruction_row(row: np.ndarray) -> str:
    payload = bytes(int(x) for x in row if int(x) != 0)
    return payload.decode("utf-8", errors="ignore")


def _reshape_steps(flat: np.ndarray, num_steps: int, width: int) -> np.ndarray:
    arr = np.asarray(flat)
    expected = num_steps * width
    if arr.size != expected:
        raise ValueError(f"expected {expected} values, found {arr.size}")
    return arr.reshape(num_steps, width)


def iter_language_table_shard(
    shard_path: str | Path,
    limit_episodes: int | None = None,
) -> Iterator[LanguageTableEpisode]:
    """Yield decoded episodes from a public Language-Table shard."""
    path = str(shard_path)
    loader = iter(sequence_loader(path, None))
    index = 0
    while True:
        try:
            context, _ = next(loader)
        except StopIteration:
            break
        except RuntimeError as exc:
            # Useful when inspecting HTTP-range partial shards during dataset probing.
            if "Failed to read the record" in str(exc):
                break
            raise

        num_steps = int(np.asarray(context["steps/is_first"]).shape[0])
        action = _reshape_steps(context["steps/action"], num_steps, XY_DIM).astype(np.float32)
        effector = _reshape_steps(
            context["steps/observation/effector_translation"],
            num_steps,
            XY_DIM,
        ).astype(np.float32)
        target = _reshape_steps(
            context["steps/observation/effector_target_translation"],
            num_steps,
            XY_DIM,
        ).astype(np.float32)
        instructions = _reshape_steps(
            context["steps/observation/instruction"],
            num_steps,
            INSTRUCTION_WIDTH,
        )
        texts = [_decode_instruction_row(row) for row in instructions]

        episode_id_raw = context.get("episode_id", b"")
        episode_id = (
            episode_id_raw.decode("utf-8", errors="ignore")
            if isinstance(episode_id_raw, (bytes, bytearray))
            else str(episode_id_raw)
        )

        yield LanguageTableEpisode(
            episode_id=episode_id,
            instruction_texts=texts,
            action=action,
            effector_translation=effector,
            effector_target_translation=target,
            reward=np.asarray(context["steps/reward"], dtype=np.float32),
            is_first=np.asarray(context["steps/is_first"], dtype=bool),
            is_last=np.asarray(context["steps/is_last"], dtype=bool),
            is_terminal=np.asarray(context["steps/is_terminal"], dtype=bool),
        )

        index += 1
        if limit_episodes is not None and index >= limit_episodes:
            break


def instruction_switch_indices(episode: LanguageTableEpisode) -> np.ndarray:
    """Indices where the per-step instruction string changes."""
    texts = np.asarray(episode.instruction_texts, dtype=object)
    if texts.size < 2:
        return np.array([], dtype=int)
    return np.flatnonzero(texts[1:] != texts[:-1]) + 1
