# Gibbs-VLA-Overshoot

> **Hypothesis:** The "jitter" in Vision-Language-Action (VLA) models is not mere noise, but a fundamental mathematical artifact—the **Gibbs Phenomenon**—arising from the mapping of discrete semantic tokens onto continuous physical manifolds.

## ⭐ Overview

Current VLA architectures (e.g., RT-2, OpenVLA) use discrete text tokens to condition continuous robotic actions. In signal processing, approximating a step function (the semantic jump) with a finite sum of continuous basis functions (the neural network) leads to a predictable ~8.9% overshoot.

This repository provides the mathematical framework and empirical evidence to quantify this "Semantic-to-Physical Impedance Mismatch."

## Core Research Questions

1. Does the "overshoot" in robot end-effector trajectories correlate with the theoretical Gibbs constant ($G \approx 0.0895$)?
2. Can we observe "Ringing Artifacts" in the frequency domain at task-switching boundaries?
3. Does a JEPA-style (Joint-Embedding Predictive Architecture) approach mitigate this overshoot by removing the discrete discontinuity?

## Framework

- **Domain:** Signal Processing meets Embodied AI.
- **Data:** Analysis performed on `Language-Table` and `CALVIN`.
- **Metrics:** Spectral leakage, L2-norm of the 3rd-order derivative (Jerk), and Overshoot Ratio.

## Repository layout

| Path | Role |
|------|------|
| `data/` | Symlinks to CALVIN or Language-Table datasets (see `data/README.md`) |
| `models/` | Simplified VLA wrapper for trajectory extraction |
| `analysis/` | FFT, wavelets, Gibbs-ratio metrics |
| `experiments/` | Jupyter notebooks for visualization |
| `paper/` | Manuscript outline and LaTeX source |

## License

Apache License 2.0 — see `LICENSE`.
