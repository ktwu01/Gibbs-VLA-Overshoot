"""Gibbs overshoot and transient metrics for trajectory analysis."""

from __future__ import annotations

import numpy as np


def vector_magnitude(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Euclidean norm for multivariate trajectories."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.linalg.norm(arr, axis=axis)


def calculate_gibbs_overshoot(
    trajectory: np.ndarray,
    baseline_value: float,
    jump_magnitude: float,
) -> float:
    """
    Legacy helper used by the README formula.

    Parameters
    ----------
    trajectory:
        1D step-response signal.
    baseline_value:
        Value before the jump.
    jump_magnitude:
        Signed jump size to the steady-state value.
    """
    signal = np.asarray(trajectory, dtype=float).ravel()
    if signal.size == 0 or abs(jump_magnitude) < 1e-12:
        return float("nan")

    target_value = baseline_value + jump_magnitude
    signed_signal = np.sign(jump_magnitude) * (signal - baseline_value)
    peak_value = float(np.max(signed_signal))
    overshoot = peak_value - abs(jump_magnitude)
    return float(overshoot / abs(jump_magnitude))


def overshoot_ratio(
    signal: np.ndarray,
    initial_value: float | None = None,
    steady_value: float | None = None,
) -> float:
    """
    Overshoot ratio relative to the initial-to-steady jump.

    Positive values indicate peak overshoot. Values near 0.089 correspond
    to the classical Gibbs constant.
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size == 0:
        return float("nan")

    initial = float(x[0] if initial_value is None else initial_value)
    steady = float(x[-1] if steady_value is None else steady_value)
    jump = steady - initial
    if abs(jump) < 1e-12:
        return float("nan")

    signed_response = np.sign(jump) * (x - initial)
    peak = float(np.max(signed_response))
    return float((peak - abs(jump)) / abs(jump))


def summarize_distribution(x: np.ndarray) -> dict[str, float]:
    """Small numeric summary for histograms and report tables."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def spectral_analysis(signal: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """One-sided FFT magnitude for quick transient inspection."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("signal must have at least 2 samples")

    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    return freqs, spectrum
