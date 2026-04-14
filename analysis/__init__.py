"""Math engine: Fourier, wavelet, and Gibbs metrics."""

from .fourier import band_energy_ratio, fft_magnitude, power_spectral_density
from .metrics import calculate_gibbs_overshoot, overshoot_ratio, spectral_analysis
from .spectral_engine import (
    analyze_step_signal,
    finite_difference,
    trajectory_derivatives,
    transient_psd,
)
from .wavelet import cwt_scalogram, scalogram_magnitude

__all__ = [
    "analyze_step_signal",
    "band_energy_ratio",
    "calculate_gibbs_overshoot",
    "cwt_scalogram",
    "fft_magnitude",
    "finite_difference",
    "overshoot_ratio",
    "power_spectral_density",
    "scalogram_magnitude",
    "spectral_analysis",
    "trajectory_derivatives",
    "transient_psd",
]
