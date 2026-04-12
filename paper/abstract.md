# Manuscript outline — Signal Processing Paradox

## Abstract (working notes)

This work frames Vision-Language-Action (VLA) **jitter** as a problem in **signal theory**, not merely as an engineering artifact. We argue that mapping discrete semantic conditioning to continuous physical trajectories induces behavior analogous to the **Gibbs phenomenon** in Fourier analysis.

---

## Introduction — The "Blind Philosopher" problem

Large language models and VLAs exhibit strong high-level reasoning ("philosopher") while **low-level sensory-motor grounding** remains brittle. The mismatch between discrete symbolic decisions and continuous control manifests as oscillatory transients and overshoot—patterns familiar from approximating discontinuous signals with smooth bases.

---

## The discontinuity

We model the transition from language-conditioned intent to action as an **\(L \to A\)** map. Where semantic intent changes abruptly (task switch, new verb), the idealized target on the physical manifold may resemble a **step** in the relevant coordinates. Approximating such a discontinuity with a finite, smooth representational capacity (neural network rollouts) parallels **Dirichlet / Fourier partial sums** near a jump: **Gibbs overshoot** becomes a quantitative prediction, not a metaphor.

---

## Mathematical proof (sketch)

Let the semantic jump be represented by \(f(x) = \operatorname{sign}(x)\). Its Fourier partial sums \(S_n(x)\) exhibit Gibbs overshoot near \(x = 0\). In particular,

\[
\lim_{n \to \infty} S_n\left(\frac{\pi}{n}\right) = \frac{2}{\pi} \int_0^\pi \frac{\sin t}{t}\, dt \approx 1.08949
\]

so the **relative overshoot** above the step height is about **8.9%** — the classical Gibbs constant. We test whether empirical robot trajectories show **overshoot ratios** clustered near this value at semantic boundaries.

---

## Proposed direction

**Discrete-conditioning** (token-to-action) may be inherently prone to this mismatch. A complementary direction is **manifold-flow matching** or **JEPA-style** prediction in a continuous latent space, reducing the severity of symbolic discontinuities before they are realized as motor commands.

---

## Cross-domain validation (NeurIPS / CVPR angle)

1. **Robotics:** 7-DOF arm \(z\)-axis overshoot when the command changes (e.g., "Hover" \(\to\) "Press").
2. **Earth systems:** An LLM-based weather agent predicting a sharp **pressure front** — if a similar **~8.9% overshoot** appears in the predicted scalar field across the front, it would support a **universal discretization law** spanning embodied and geophysical domains.

Treat all signals as **multivariate time series**; the physics domain is secondary to the **wave** structure.
