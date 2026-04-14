"""Derivative and spectral utilities for Gibbs-style transient analysis."""

from __future__ import annotations

import numpy as np

from .fourier import band_energy_ratio, fft_magnitude, power_spectral_density
from .metrics import overshoot_ratio


def finite_difference(x: np.ndarray, fs: float, order: int = 1, axis: int = 0) -> np.ndarray:
    """Numerical derivative using repeated central differences."""
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    for _ in range(order):
        out = np.gradient(out, axis=axis) * fs
    return out


def trajectory_derivatives(trajectory: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity and acceleration for a `(T, D)` trajectory."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[:, None]
    velocity = finite_difference(traj, fs=fs, order=1, axis=0)
    acceleration = finite_difference(traj, fs=fs, order=2, axis=0)
    return velocity, acceleration


def estimate_power_law_exponent(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fmin: float,
    fmax: float | None = None,
) -> float:
    """
    Fit `magnitude ~ f^slope` in log-log space.

    A slope near `-1` is a useful heuristic fingerprint for step-like transients.
    """
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(magnitude, dtype=float)
    upper = np.inf if fmax is None else float(fmax)
    mask = (f > max(fmin, 0.0)) & (f <= upper) & (m > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    x = np.log(f[mask])
    y = np.log(m[mask])
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def compare_frequency_bands(
    signal: np.ndarray,
    fs: float,
    split_hz: float | None = None,
) -> dict[str, float]:
    """Compare low-band and high-band energy in a transient segment."""
    nyquist = fs / 2.0
    split = split_hz if split_hz is not None else nyquist / 3.0
    split = min(max(split, 1e-6), nyquist)
    high_ratio = band_energy_ratio(signal, fs=fs, f_low=split, f_high=nyquist)
    return {
        "split_hz": float(split),
        "high_band_ratio": float(high_ratio),
        "low_band_ratio": float(1.0 - high_ratio),
    }


def analyze_step_signal(
    signal: np.ndarray,
    fs: float,
    switch_index: int | None = None,
) -> dict[str, float]:
    """Summary metrics for a 1D step-response candidate."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 4:
        raise ValueError("signal must have at least 4 samples")

    if switch_index is None:
        switch_index = x.size // 2

    initial = float(np.mean(x[: max(1, switch_index // 2)]))
    steady = float(np.mean(x[switch_index + max(2, (x.size - switch_index) // 3) :]))
    velocity = finite_difference(x, fs=fs, order=1)
    acceleration = finite_difference(x, fs=fs, order=2)
    freqs, mag = fft_magnitude(x - np.mean(x), fs=fs)
    band_stats = compare_frequency_bands(x, fs=fs)

    return {
        "switch_index": int(switch_index),
        "initial_value": initial,
        "steady_value": steady,
        "overshoot_ratio": overshoot_ratio(x, initial_value=initial, steady_value=steady),
        "peak_velocity": float(np.max(np.abs(velocity))),
        "peak_acceleration": float(np.max(np.abs(acceleration))),
        "power_law_exponent": estimate_power_law_exponent(
            freqs,
            mag,
            fmin=max(fs / x.size, 1e-6),
            fmax=fs / 2.0,
        ),
        **band_stats,
    }


def transient_psd(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD convenience wrapper."""
    return power_spectral_density(signal, fs=fs)
