"""
OpenVLA policy adapter for the instruction-swap experiment.

Wraps ``openvla/openvla-7b`` (HuggingFace) behind the :class:`PolicyAdapter`
protocol so it plugs directly into :class:`InstructionSwapRunner`.

Requirements: ``pip install "transformers>=4.40,<4.50" torch Pillow "timm>=0.9.10,<1"``
GPU with >=16 GB VRAM recommended (bfloat16).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _patch_llama_attention_mask(model):
    """
    Patch the LLaMA causal model to ignore the external attention_mask.

    OpenVLA's ``modeling_prismatic.py`` builds a ``multimodal_attention_mask``
    of length ``len(input_ids) + num_patches`` but newer transformers (>4.40)
    build an internal ``causal_mask`` from ``cache_position`` which has length
    ``len(inputs_embeds)`` on the first step and ``1`` on subsequent steps,
    creating a size mismatch. Dropping the mask lets LLaMA build its own
    causal mask from ``inputs_embeds`` alone, which is correct for inference.
    """
    llama_model = model.language_model.model  # LlamaModel
    _orig_forward = llama_model.forward

    def _patched_forward(*args, **kwargs):
        kwargs.pop("attention_mask", None)
        return _orig_forward(*args, **kwargs)

    llama_model.forward = _patched_forward


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
        device: str = "auto",
        use_flash_attn: bool = False,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Auto-detect device: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # MPS doesn't support bfloat16 well; use float32 on MPS/CPU
        if device == "mps" or device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()

        # Fix attention mask size mismatch with transformers >4.40
        _patch_llama_attention_mask(self.vla)

        self.unnorm_key = unnorm_key
        self.device = device
        self._torch_dtype = dtype

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

        if self.unnorm_key is not None:
            # Standard path: model handles de-normalization
            with torch.no_grad():
                action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False,
                )
            return np.asarray(action, dtype=np.float64)

        # Raw action path: manually generate tokens and decode to [-1, 1]
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1
            )
        with torch.no_grad():
            generated_ids = self.vla.generate(
                input_ids, max_new_tokens=self.action_dim, do_sample=False,
            )
        predicted_token_ids = generated_ids[0, -self.action_dim:].cpu().numpy()
        discretized = self.vla.vocab_size - predicted_token_ids
        discretized = np.clip(discretized - 1, a_min=0,
                              a_max=self.vla.bin_centers.shape[0] - 1)
        normalized_actions = self.vla.bin_centers[discretized]
        return np.asarray(normalized_actions, dtype=np.float64)
