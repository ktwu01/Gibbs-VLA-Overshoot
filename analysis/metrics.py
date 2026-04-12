"""Gibbs overshoot and spectral helpers for trajectory analysis."""

import numpy as np
from scipy.signal import find_peaks


def calculate_gibbs_overshoot(trajectory, baseline_value, jump_magnitude):
    """
    Calculates the ratio of the peak overshoot relative to the jump magnitude.
    Target: ~8.9% (0.089)
    """
    # 1. Identify the peak following a task jump
    peaks, _ = find_peaks(trajectory)
    if len(peaks) == 0:
        return 0

    actual_max = trajectory[peaks[0]]
    overshoot = actual_max - (baseline_value + jump_magnitude)

    # 2. Calculate the Gibbs Ratio
    gibbs_ratio = overshoot / jump_magnitude
    return gibbs_ratio


def spectral_analysis(signal, fs=100):
    """
    Computes Power Spectral Density to look for high-frequency ringing.
    """
    from scipy.fft import fft, fftfreq

    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)
    return xf, np.abs(yf)
