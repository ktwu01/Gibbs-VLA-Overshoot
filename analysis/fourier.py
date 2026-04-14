"""FFT and spectral density analysis for multivariate time series (e.g., robot trajectories)."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def fft_magnitude(x: np.ndarray, fs: float = 100.0):
    """
    One-sided magnitude spectrum (positive frequencies) for a real signal.

    Parameters
    ----------
    x : ndarray
        1D time series.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    freqs : ndarray
        Positive frequency bins (Hz).
    mag : ndarray
        Magnitude of the DFT (same length as freqs).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("Signal must have at least 2 samples.")

    yf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(yf)
    return freqs, mag


def power_spectral_density(
    x: np.ndarray,
    fs: float = 100.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
):
    """
    Welch PSD estimate — useful for detecting spectral leakage and ringing.

    Parameters
    ----------
    x : ndarray
        1D time series.
    fs : float
        Sampling rate (Hz).
    nperseg, noverlap : optional
        Passed to `scipy.signal.welch`.

    Returns
    -------
    freqs : ndarray
    psd : ndarray
        One-sided power spectral density.
    """
    x = np.asarray(x, dtype=float).ravel()
    nperseg = min(x.size, nperseg or min(256, max(8, x.size // 4)))
    noverlap = noverlap if noverlap is not None else nperseg // 2

    freqs, psd = scipy_signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


def band_energy_ratio(
    x: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    **welch_kw,
) -> float:
    """
    Fraction of total Welch PSD energy in [f_low, f_high] (Hz).
    """
    freqs, psd = power_spectral_density(x, fs=fs, **welch_kw)
    if f_high <= f_low:
        raise ValueError("f_high must exceed f_low.")

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total = _trapz(psd, freqs)
    if total <= 0:
        return 0.0

    mask = (freqs >= f_low) & (freqs <= f_high)
    band = _trapz(psd[mask], freqs[mask])
    return float(band / total)
