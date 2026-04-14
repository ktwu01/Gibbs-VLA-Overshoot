"""
Simplified VLA wrapper: treat policy output as a multivariate time series.

No ROS dependency — trajectories are `(T, D)` arrays (e.g., end-effector pose deltas
or joint commands at each control step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class TrajectoryBatch:
    """Batch of trajectories aligned for analysis."""

    trajectories: np.ndarray  # shape (B, T, D)
    fs: float = 100.0  # Hz, assumed control rate
    metadata: dict[str, Any] | None = None


class VLATrajectoryExtractor:
    """
    Placeholder interface: plug in OpenVLA / RT-2 style policies later.

    For now, accepts precomputed numpy arrays or loads from disk in HDF5/NPZ.
    """

    def __init__(self, control_hz: float = 100.0):
        self.control_hz = control_hz

    def from_array(self, arr: np.ndarray) -> TrajectoryBatch:
        """Wrap a single `(T, D)` or batch `(B, T, D)` trajectory."""
        a = np.asarray(arr, dtype=float)
        if a.ndim == 2:
            a = a[np.newaxis, ...]
        if a.ndim != 3:
            raise ValueError("Expected shape (T, D) or (B, T, D).")
        return TrajectoryBatch(trajectories=a, fs=self.control_hz)

    def end_effector_z(self, batch: TrajectoryBatch, axis: int = 2) -> np.ndarray:
        """Example: extract z-axis channel for overshoot analysis (7-DOF arm)."""
        return batch.trajectories[..., axis]


# ---------------------------------------------------------------------------
# Policy adapter protocol and instruction-swap infrastructure
# ---------------------------------------------------------------------------


class PolicyAdapter(Protocol):
    """Interface any VLA policy must satisfy for the swap experiment."""

    @property
    def action_dim(self) -> int: ...

    def predict(
        self,
        observation: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Return a single action vector of shape ``(action_dim,)``."""
        ...


class MockPolicy:
    """
    Deterministic mock: maps each unique instruction to a distinct steady-state
    action vector, with configurable Gaussian noise to mimic policy stochasticity.

    The action for an instruction is a repeatable hash-seeded random vector so
    that different instructions produce measurably different outputs.
    """

    def __init__(
        self,
        action_dim: int = 7,
        noise_std: float = 0.02,
        seed: int = 42,
    ):
        self._action_dim = action_dim
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _steady_state_for(self, instruction: str) -> np.ndarray:
        if instruction not in self._cache:
            inst_seed = hash(instruction) % (2**31)
            rng = np.random.default_rng(inst_seed)
            self._cache[instruction] = rng.uniform(-1, 1, size=self._action_dim)
        return self._cache[instruction]

    def predict(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        base = self._steady_state_for(instruction)
        noise = self._rng.normal(0, self.noise_std, size=self._action_dim)
        return base + noise


@dataclass
class SwapResult:
    """Output of a single instruction-swap run."""

    actions: np.ndarray  # (T, D) full action trace
    switch_step: int  # index where instruction changed
    instruction_a: str
    instruction_b: str
    fs: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InstructionSwapRunner:
    """
    Phase-1 experiment: hold observation fixed, flip instruction at ``switch_step``.

    Parameters
    ----------
    policy:
        Anything satisfying :class:`PolicyAdapter`.
    observation:
        A single observation frame held constant throughout the rollout.
        If *None*, a zero image of shape ``(224, 224, 3)`` is used.
    fs:
        Control rate in Hz (for downstream spectral analysis).
    """

    def __init__(
        self,
        policy: PolicyAdapter,
        observation: np.ndarray | None = None,
        fs: float = 10.0,
    ):
        self.policy = policy
        self.observation = (
            observation if observation is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        self.fs = fs

    def run(
        self,
        instruction_a: str,
        instruction_b: str,
        total_steps: int = 50,
        switch_step: int = 25,
    ) -> SwapResult:
        """Execute the swap rollout and return the full action trace."""
        if switch_step < 1 or switch_step >= total_steps:
            raise ValueError("switch_step must be in [1, total_steps)")

        actions = np.empty((total_steps, self.policy.action_dim), dtype=float)
        for t in range(total_steps):
            instr = instruction_a if t < switch_step else instruction_b
            actions[t] = self.policy.predict(self.observation, instr)

        return SwapResult(
            actions=actions,
            switch_step=switch_step,
            instruction_a=instruction_a,
            instruction_b=instruction_b,
            fs=self.fs,
        )
