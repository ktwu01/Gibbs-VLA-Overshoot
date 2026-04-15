"""
Chunked-action policy adapter for pi0 / SmolVLA instruction-swap experiments.

These models predict a CHUNK of future actions (e.g., 50 steps) at once.
When the instruction changes mid-chunk, the remaining buffered actions
still reflect the OLD instruction — creating the temporal coupling that
may produce Gibbs-like overshoot.

The key parameter is ``n_action_steps``: how many actions from each chunk
are executed before re-querying the model.  Larger = more "inertia".
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import torch
from PIL import Image


def _patch_lerobot_groot():
    """Patch around LeRobot groot dataclass bug in Python 3.12."""
    if 'lerobot.policies.groot' not in sys.modules:
        mock = types.ModuleType('lerobot.policies.groot')
        mock_cfg = types.ModuleType('lerobot.policies.groot.configuration_groot')
        mock_cfg.GrootConfig = type('GrootConfig', (), {})
        sys.modules['lerobot.policies.groot'] = mock
        sys.modules['lerobot.policies.groot.configuration_groot'] = mock_cfg
        sys.modules['lerobot.policies.groot.__init__'] = mock


class ChunkedVLAPolicy:
    """
    Wraps a LeRobot pi0 or SmolVLA policy for instruction-swap experiments.

    Unlike OpenVLA (memoryless, 1 action per call), these models predict
    ``chunk_size`` future actions per forward pass.  This adapter:

      1. Maintains an internal action buffer (the current chunk).
      2. Only re-queries the model every ``n_action_steps`` calls.
      3. Returns ``action_buffer[local_step]`` at each call to ``predict``.
      4. When the instruction changes mid-chunk, the buffer may still
         contain actions from the OLD instruction — this is the source
         of potential Gibbs-like overshoot.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier, e.g. ``"lerobot/smolvla_base"``.
    n_action_steps:
        Number of actions to execute from each chunk before re-planning.
        Set to 1 for near-memoryless behavior, chunk_size for full open-loop.
    device:
        Torch device string.
    """

    def __init__(
        self,
        model_id: str = "lerobot/smolvla_base",
        n_action_steps: int = 50,
        device: str = "auto",
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

        self.device = device
        self.n_action_steps = n_action_steps

        # Patch groot import bug before loading LeRobot policies
        _patch_lerobot_groot()

        # Detect model type and load
        if "smolvla" in model_id.lower():
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            self.policy = SmolVLAPolicy.from_pretrained(model_id)
        elif "pi0" in model_id.lower():
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            self.policy = PI0Policy.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown model type in '{model_id}'. Expected 'pi0' or 'smolvla'.")

        self.policy.to(device)
        self.policy.eval()

        # Override n_action_steps in config
        self.policy.config.n_action_steps = n_action_steps

        self._chunk_size = self.policy.config.chunk_size
        self._action_buffer = None
        self._buffer_idx = 0
        self._last_instruction = None

    @property
    def action_dim(self) -> int:
        return self.policy.config.action_dim

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def reset(self):
        """Clear the action buffer and reset state."""
        self._action_buffer = None
        self._buffer_idx = 0
        self._last_instruction = None
        self.policy.reset()

    def _prepare_batch(self, observation: np.ndarray, instruction: str) -> dict:
        """Build the batch dict expected by LeRobot policies."""
        # Image: (H, W, 3) uint8 -> (1, H, W, 3) float32 [0, 1]
        image = np.asarray(observation, dtype=np.uint8)
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # State: zeros (we don't have real robot state)
        state_dim = getattr(self.policy.config, 'max_state_dim',
                           getattr(self.policy.config, 'state_dim', 7))
        state = torch.zeros(1, state_dim, device=self.device)

        # Tokenize instruction
        processor = self.policy.processor
        task_text = instruction.strip()

        # Use the processor's tokenizer step
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        tokenizer = None
        for step in processor.steps:
            if hasattr(step, 'tokenizer'):
                tokenizer = step.tokenizer
                break

        if tokenizer is None:
            raise RuntimeError("Could not find tokenizer in policy processor")

        encoded = tokenizer(
            task_text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        batch = {
            "observations.images.0": image_tensor,
            "observations.state": state,
            OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(self.device),
        }
        return batch

    def _generate_chunk(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """Generate a full action chunk from the model."""
        batch = self._prepare_batch(observation, instruction)

        with torch.no_grad():
            # predict_action_chunk returns (1, n_action_steps, action_dim)
            actions = self.policy.predict_action_chunk(batch)

        return actions[0].cpu().numpy()  # (n_action_steps, action_dim)

    def predict(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """
        Return one action, managing the chunk buffer internally.

        If the buffer is empty or exhausted, generate a new chunk.
        If the instruction changed mid-chunk, we do NOT regenerate —
        this is intentional, as the "stale" buffer is what creates
        the temporal coupling for Gibbs-like overshoot.
        """
        need_new_chunk = (
            self._action_buffer is None
            or self._buffer_idx >= self._action_buffer.shape[0]
        )

        if need_new_chunk:
            self._action_buffer = self._generate_chunk(observation, instruction)
            self._buffer_idx = 0
            self._last_instruction = instruction

        action = self._action_buffer[self._buffer_idx]
        self._buffer_idx += 1

        return np.asarray(action, dtype=np.float64)


class ChunkedVLAPolicyForceReplan(ChunkedVLAPolicy):
    """
    Variant that forces re-planning when the instruction changes.

    This should show LESS overshoot than the base ChunkedVLAPolicy
    (which intentionally continues executing the old chunk).
    Useful as a control condition.
    """

    def predict(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        need_new_chunk = (
            self._action_buffer is None
            or self._buffer_idx >= self._action_buffer.shape[0]
            or instruction != self._last_instruction  # Force replan on instruction change
        )

        if need_new_chunk:
            self._action_buffer = self._generate_chunk(observation, instruction)
            self._buffer_idx = 0
            self._last_instruction = instruction

        action = self._action_buffer[self._buffer_idx]
        self._buffer_idx += 1

        return np.asarray(action, dtype=np.float64)
