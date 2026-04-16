# Codex Review: Determine best architecture for Gibbs overshoot reproduction

## Context

We ran **20 experiments across 2 rounds** trying to reproduce the ~8.95% Gibbs overshoot phenomenon in OpenVLA-7B instruction-swap experiments. **All experiments showed 0% overshoot.**

See full failure analysis: [`PLAN_v2.md`](https://github.com/ktwu01/Gibbs-VLA-Overshoot/blob/main/PLAN_v2.md)
See experiment record: [`experiments/exp_record.md`](https://github.com/ktwu01/Gibbs-VLA-Overshoot/blob/main/experiments/exp_record.md)

## Root Cause

**OpenVLA is a memoryless (Markov) policy.** Each action `a_t = f(image, instruction)` is independent — no temporal state, no recurrence, no action history. The instruction swap produces a **perfect step function** with zero transient. The Gibbs phenomenon requires truncated Fourier series approximation, which doesn't exist in discrete-token VLAs.

### Evidence

- 15 completed OpenVLA experiments: max overshoot = 1.57e-14% (numerical noise)
- Peak velocities up to 3.53 (actions DO change at switch) — but transition is instantaneous
- velocity_p95 = 0.0 in ALL experiments (95% of timesteps have zero velocity = constant action)
- Synthetic Gibbs ground truth: 9.05% overshoot (confirms the measurement code works)
- Mock policy: 6.26% overshoot (confirms the framework works)

## Questions for Codex Review

### 1. Is the memoryless-policy explanation correct?
Are there any hidden temporal dependencies in OpenVLA we missed? (e.g., KV-cache reuse across timesteps, image preprocessing buffers, attention patterns that create implicit memory)

### 2. Which architecture should we try next?
Rank these by likelihood of showing Gibbs ringing AND feasibility on NCAR (A100 80GB):

- **Diffusion-based action heads** (Octo, Diffusion Policy) — denoising = truncated basis expansion?
- **Action-chunking models** (ACT, VQ-BeT) — chunk boundary = temporal discontinuity?
- **Real closed-loop rollouts** (OpenVLA + MuJoCo) — robot dynamics create temporal coupling?

### 3. What should the theoretical overshoot target be?
The 8.95% Gibbs constant is specific to truncated Fourier series. For diffusion/chunking models, what's the analogous constant? Is there a theoretical prediction?

### 4. Could stochastic sampling create apparent Gibbs ringing?
If we use `do_sample=True` (temperature > 0), the token noise might create a noisy step. Is this genuine Gibbs-like behavior or just sampling variance? How do we distinguish?

### 5. Should we use a toy model instead?
Would a small custom transformer with explicit action chunking (trained from scratch on a simple task) be more convincing than wrestling with a 7B model? What's the minimal architecture that could exhibit this phenomenon?

## Deliverable

Please provide:
1. A ranked recommendation of which architecture to try first
2. A concrete experiment design for the top choice
3. Any corrections to our failure analysis
4. Suggested modifications to `PLAN_v2.md`
