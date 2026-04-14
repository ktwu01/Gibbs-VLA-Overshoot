"""
OpenVLA policy adapter for the instruction-swap experiment.

Wraps ``openvla/openvla-7b`` (HuggingFace) behind the :class:`PolicyAdapter`
protocol so it plugs directly into :class:`InstructionSwapRunner`.

Requirements: ``pip install transformers torch Pillow``
GPU with >=16 GB VRAM recommended (bfloat16).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class OpenVLAPolicy:
    """
    Loads OpenVLA-7B and exposes :meth:`predict` for single-step inference.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.
    unnorm_key:
        Dataset key used by OpenVLA to de-normalize actions.
        ``"bridge_orig"`` for Bridge V2 (7-DOF EEF deltas).
        Set to ``None`` to get raw [-1, 1] normalized actions.
    device:
        Torch device string.
    use_flash_attn:
        Whether to use flash-attention-2 (requires ``flash_attn`` package).
    """

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        unnorm_key: str | None = "bridge_orig",
        device: str = "cuda:0",
        use_flash_attn: bool = False,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()

        self.unnorm_key = unnorm_key
        self.device = device
        self._torch_dtype = torch.bfloat16

    @property
    def action_dim(self) -> int:
        return 7

    def predict(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """
        Run one inference step.

        Parameters
        ----------
        observation:
            RGB image as uint8 array, any spatial size (resized internally).
        instruction:
            Natural-language task instruction, e.g. ``"pick up the red block"``.

        Returns
        -------
        np.ndarray of shape ``(7,)`` — EEF deltas ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.
        """
        import torch

        image = Image.fromarray(np.asarray(observation, dtype=np.uint8)).convert("RGB")
        prompt = f"In: What action should the robot take to {instruction.lower().strip()}?\nOut:"

        inputs = self.processor(prompt, image).to(self.device, dtype=self._torch_dtype)

        with torch.no_grad():
            action = self.vla.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )

        return np.asarray(action, dtype=np.float64)
