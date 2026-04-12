"""Math engine: Fourier, wavelet, and Gibbs metrics."""

from .fourier import band_energy_ratio, fft_magnitude, power_spectral_density
from .metrics import calculate_gibbs_overshoot, spectral_analysis
from .wavelet import cwt_scalogram, scalogram_magnitude

__all__ = [
    "band_energy_ratio",
    "calculate_gibbs_overshoot",
    "cwt_scalogram",
    "fft_magnitude",
    "power_spectral_density",
    "scalogram_magnitude",
    "spectral_analysis",
]
