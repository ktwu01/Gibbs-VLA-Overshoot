"""Synthetic Gibbs-style step experiment with analysis plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import summarize_distribution
from analysis.spectral_engine import (
    analyze_step_signal,
    finite_difference,
    transient_psd,
)


def truncated_fourier_step(x: np.ndarray, harmonics: int) -> np.ndarray:
    """Fourier-series approximation of a unit step on `[-pi, pi]`."""
    result = 0.5 * np.ones_like(x, dtype=float)
    for k in range(harmonics):
        n = 2 * k + 1
        result += (2.0 / np.pi) * np.sin(n * x) / n
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output figure path.")
    parser.add_argument("--harmonics", type=int, default=80, help="Odd harmonics in the approximation.")
    parser.add_argument("--samples", type=int, default=1500, help="Number of time samples.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-np.pi, np.pi, args.samples)
    fs = args.samples / (2.0 * np.pi)
    t = x / np.pi
    ideal = (x >= 0).astype(float)
    signal = truncated_fourier_step(x, harmonics=args.harmonics)
    switch_index = int(np.searchsorted(x, 0.0))

    velocity = finite_difference(signal, fs=fs, order=1)
    acceleration = finite_difference(signal, fs=fs, order=2)
    psd_freqs, psd = transient_psd(signal - np.mean(signal), fs=fs)
    metrics = analyze_step_signal(signal, fs=fs, switch_index=switch_index)
    velocity_stats = summarize_distribution(np.abs(velocity))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    ax.plot(t, ideal, color="black", linewidth=2, label="ideal step")
    ax.plot(t, signal, color="#1f77b4", linewidth=2, label=f"Fourier approximation ({args.harmonics} harmonics)")
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Synthetic Step Response")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Action output")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    ax.text(
        0.03,
        0.93,
        f"Overshoot: {metrics['overshoot_ratio'] * 100:.2f}%",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    ax = axes[0, 1]
    ax.plot(t, velocity, color="#d62728", linewidth=1.6, label="velocity")
    ax.plot(t, acceleration, color="#2ca02c", linewidth=1.2, label="acceleration")
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Transient Derivatives")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Derivative magnitude")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.loglog(psd_freqs[1:], psd[1:], color="#9467bd", linewidth=1.8)
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.2, which="both")
    ax.text(
        0.03,
        0.93,
        f"Power-law slope: {metrics['power_law_exponent']:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    ax = axes[1, 1]
    ax.hist(np.abs(velocity), bins=40, color="#ff7f0e", alpha=0.9, edgecolor="white")
    ax.set_title("Velocity Magnitude Distribution")
    ax.set_xlabel("|velocity|")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.text(
        0.03,
        0.93,
        f"mean={velocity_stats['mean']:.3f}\np95={velocity_stats['p95']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Synthetic Gibbs Experiment", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    report = {
        "harmonics": args.harmonics,
        "samples": args.samples,
        "overshoot_ratio": metrics["overshoot_ratio"],
        "overshoot_percent": metrics["overshoot_ratio"] * 100.0,
        "peak_velocity": metrics["peak_velocity"],
        "peak_acceleration": metrics["peak_acceleration"],
        "power_law_exponent": metrics["power_law_exponent"],
        "high_band_ratio": metrics["high_band_ratio"],
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
