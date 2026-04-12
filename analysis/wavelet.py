"""Time–frequency analysis for task boundaries and transient ringing."""

from __future__ import annotations

import numpy as np


def cwt_scalogram(
    x: np.ndarray,
    widths: np.ndarray,
    wavelet: str = "morl",
    sampling_period: float = 1.0 / 100.0,
):
    """
    Continuous wavelet transform magnitude (scalogram) using PyWavelets.

    Use high magnitude near discontinuities / task switches as a proxy for
    transient energy at multiple scales.

    Parameters
    ----------
    x : ndarray
        1D time series.
    widths : ndarray
        Wavelet scales (see PyWavelets `cwt` documentation).
    wavelet : str
        Wavelet name (default: complex Morlet `morl`).
    sampling_period : float
        Sample spacing (1/fs).

    Returns
    -------
    coefs : ndarray
        Complex CWT coefficients, shape (len(widths), len(x)).
    freqs : ndarray
        Approximate pseudo-frequencies for each scale (PyWavelets).
    """
    import pywt

    x = np.asarray(x, dtype=float).ravel()
    coefs, freqs = pywt.cwt(x, widths, wavelet, sampling_period=sampling_period)
    return coefs, freqs


def scalogram_magnitude(x: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Returns |CWT| and frequency vector for quick visualization."""
    coefs, freqs = cwt_scalogram(x, **kwargs)
    return np.abs(coefs), freqs
