"""
Simplified VLA wrapper: treat policy output as a multivariate time series.

No ROS dependency — trajectories are `(T, D)` arrays (e.g., end-effector pose deltas
or joint commands at each control step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
