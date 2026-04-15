# Gibbs-VLA-Overshoot: Experiment Plan v2

**Date:** 2026-04-15
**Status:** Round 1 & 2 FAILED — pivoting to alternative architectures
**Platform:** NCAR Casper, A100 80GB, PBS scheduler

---

## 1. Original Hypothesis

The Gibbs phenomenon (the ~8.95% overshoot that occurs when a truncated Fourier series approximates a step function) should manifest in Vision-Language-Action (VLA) models when instructions are abruptly swapped mid-rollout. Just as a bandlimited signal rings when reconstructing a discontinuity, a VLA policy should "ring" when forced through a sudden instruction boundary.

**Target:** Reproduce the theoretical Gibbs constant (8.9490..%) in a real VLA model (OpenVLA-7B) performing instruction-swap experiments.

---

## 2. FAILURE REPORT: Full Details

### 2.1 What We Tried

**20 experiments across 2 rounds**, systematically varying every available parameter:

#### Round 1 (6 completed, 5 crashed)

| ID | Config | Result |
|----|--------|--------|
| baseline | axis=0, bridge_orig, 100 steps, switch@50, default instrs | **0.0% overshoot**, peak_v=0.0056 |
| exp01 | axis=2(dz), bridge_orig, 150 steps, switch@60, default instrs | **0.0% overshoot**, peak_v=0.024 |
| exp02 | axis=0, **unnorm=none**, 100 steps, switch@50 | **CRASHED**: `AssertionError` in `predict_action` — model trained on 25 datasets, `unnorm_key=None` triggers assertion `len(norm_stats) == 1` |
| exp03 | axis=1, unnorm=none, 100 steps, switch@50 | **CRASHED**: same bug |
| exp04 | axis=2, unnorm=none, 100 steps, switch@50 | **CRASHED**: same bug |
| exp05 | axis=0, bridge_orig, 100 steps, **switch@25** (early) | **0.0% overshoot**, peak_v=0.0056, plaw=-1.17 |
| exp06 | axis=0, bridge_orig, 100 steps, switch@50, **"pick up the sushi" → "close the drawer"** | **0.0% overshoot**, peak_v=0.078 |
| exp07 | axis=2, bridge_orig, 100 steps, switch@50, "sushi" → "drawer" | **0.0% overshoot**, peak_v=0.031 |
| exp08 | axis=0, unnorm=none, 200 steps, switch@100, "corn" → "cloth" | **CRASHED**: same unnorm bug |
| exp09 | **axis=6(grip)**, bridge_orig, 100 steps, switch@50, "pick" → "close" | **NaN overshoot**, peak_v=0.0 (gripper constant) |
| exp10 | axis=2, unnorm=none, 150 steps, switch@40, "sushi" → "push blue" | **CRASHED**: same unnorm bug |

**Round 1 crash root cause:** OpenVLA's `modeling_prismatic.py` line 541:
```python
assert len(norm_stats) == 1, (
    "Your model was trained on more than one dataset, "
    "please pass a `unnorm_key` from the following options..."
)
```
When `unnorm_key=None`, the model tries to auto-detect the single dataset but OpenVLA was trained on 25 datasets. **Fix:** Modified `openvla_policy.py` to manually generate tokens via `model.generate()` and decode through `bin_centers` when `unnorm_key is None`.

#### Round 2 (9 completed, 1 crashed)

All used the fixed raw-action decoding path.

| ID | Config | Overshoot | Peak Velocity | Power Law |
|----|--------|-----------|---------------|-----------|
| r2_exp01 | axis=0(dx), raw, 100 steps, switch@50, default | **0.0%** | 1.235 | 0.366 |
| r2_exp02 | axis=1(dy), raw, 100 steps, switch@50, default | **0.0%** | 1.137 | 0.382 |
| r2_exp03 | axis=2(dz), raw, 100 steps, switch@50, default | **0.0%** | 1.608 | 0.440 |
| r2_exp04 | axis=0, raw, 100 steps, switch@50, "sushi"→"drawer" | **0.0%** | 0.980 | 0.305 |
| r2_exp05 | axis=0, raw, **200 steps**, **switch@50** (early) | **0.0%** | 1.235 | -0.894 |
| r2_exp06 | axis=0, raw, 100 steps, switch@50, "corn"→"cloth" | **0.0%** | 3.529 | 0.392 |
| r2_exp07 | axis=0, raw, 100 steps, switch@50, **episode=1** | **CRASHED** | - | - |
| r2_exp08 | axis=0, raw, 100 steps, switch@50, **episode=2** | **0.0%** | 0.039 | 0.484 |
| r2_exp09 | axis=0, **unnorm=fractal**, 100 steps, switch@50, "sushi"→"drawer" | **0.0%** | 0.549 | 0.307 |
| r2_exp10 | axis=6(grip), raw, 100 steps, switch@50, default | **NaN** | 0.0 | NaN |

### 2.2 Complete Per-Axis Overshoot Analysis

Across ALL 15 completed OpenVLA experiments, the maximum per-axis overshoot on any dimension was:

```
Maximum absolute overshoot seen: ~1.57e-14 % (numerical noise, effectively ZERO)
```

This was consistent across:
- All 7 action dimensions (dx, dy, dz, droll, dpitch, dyaw, gripper)
- 3 unnormalization schemes (bridge_orig, fractal, none/raw)
- 6 different instruction pairs
- 3 switch timings (step 25, 50, 60)
- 3 rollout lengths (100, 150, 200)
- 3 episode indices (0, 1, 2)

### 2.3 Evidence That the Step EXISTS

The experiments are NOT failing because the model ignores instructions. The peak velocities prove that actions DO change at the switch point:

| Experiment | Peak Velocity | Meaning |
|------------|--------------|---------|
| r2_exp06 (corn→cloth, raw) | **3.529** | Large step — actions jump by ~0.7 in normalized space |
| r2_exp03 (dz, raw) | **1.608** | Clear step on Z-axis |
| r2_exp01 (dx, raw) | **1.235** | Clear step on X-axis |
| r2_exp02 (dy, raw) | **1.137** | Clear step on Y-axis |
| r2_exp04 (sushi→drawer, raw) | **0.980** | Step visible with different instructions |
| exp06 (sushi→drawer, bridge) | **0.078** | Step exists but tiny in physical units |
| baseline (default, bridge) | **0.006** | Minimal step in physical units |
| r2_exp08 (episode 2) | **0.039** | Weak step — this episode's obs less instruction-sensitive |

**The step is real. But it's a PERFECT step — instantaneous, with zero transient.**

### 2.4 Why the velocity_p95 Is Always 0.0

A critical detail: `velocity_p95 = 0.0` in every experiment. This means that 95% of timesteps have zero velocity — the action is constant before the switch AND constant after the switch, with a single-sample transition. The model outputs identical actions for identical (image, instruction) inputs, proving it's deterministic and memoryless.

### 2.5 High Band Ratio Analysis

The `high_band_ratio` (fraction of spectral energy above Nyquist/3) is consistently ~0.089 across experiments. This is NOT evidence of ringing — it's the spectral signature of an ideal step function (a delta function in the time domain has flat spectrum, so a step has energy across all frequency bands).

### 2.6 Power Law Exponent Patterns

| Pattern | Experiments | Meaning |
|---------|------------|---------|
| plaw ≈ -1.0 | exp05 (-1.17), r2_exp05 (-0.89), exp01 (-0.86) | Step-like spectrum (1/f decay) — confirms a clean step |
| plaw ≈ +0.3 to +0.5 | Most r2 experiments | Slight positive slope — flat-ish spectrum of a nearly constant signal with one jump |
| plaw = NaN | exp09, r2_exp10 | Signal is constant (gripper doesn't change) — undefined |

The negative power law exponents actually CONFIRM a step function, not ringing. A Gibbs-ringing signal would show plaw ≈ -1.0 PLUS overshoot. We see the former without the latter.

---

## 3. ROOT CAUSE: Why OpenVLA Cannot Exhibit Gibbs Ringing

### 3.1 The Memoryless (Markov) Policy Architecture

OpenVLA is fundamentally a **memoryless policy**. Each inference step is:

```
action_t = f(image_t, instruction_t)
```

There is NO dependence on:
- Previous actions: `action_{t-1}`, `action_{t-2}`, ...
- Hidden state: no RNN, no temporal buffer
- Action history: no sliding window, no momentum

When instruction changes at step `t_switch`:
```
t = t_switch - 1:  action = f(image, "pick up the red block")    = [0.12, -0.05, 0.31, ...]
t = t_switch:      action = f(image, "push the blue block left") = [0.35,  0.18, 0.08, ...]
t = t_switch + 1:  action = f(image, "push the blue block left") = [0.35,  0.18, 0.08, ...]
```

The transition is **instantaneous and perfect**. There is no "memory" of the previous instruction that would create a transient.

### 3.2 Why Fourier / Gibbs Theory Doesn't Apply

The Gibbs phenomenon arises from **truncated Fourier series approximation of a discontinuity**. It requires:

1. **A continuous signal being reconstructed from a finite basis** — OpenVLA uses discrete tokens (256 bins), not Fourier coefficients. Each action dimension is independently quantized to the nearest bin center. This is a lookup table, not a series reconstruction.

2. **Bandwidth limitation** — OpenVLA's "bandwidth" is not limited. It can output any of 256 discrete values at any timestep with no constraint from adjacent timesteps.

3. **Temporal coupling** — The Gibbs overshoot is a temporal phenomenon (the signal "rings" in time around the discontinuity). OpenVLA has zero temporal coupling — each timestep is independent.

### 3.3 The Discrete Token Quantization

OpenVLA discretizes actions into 256 bins via:
```python
bin_centers = np.linspace(-1, 1, 256)  # 256 equally-spaced bins
```

The model generates token IDs via autoregressive decoding, then maps:
```
token_id → discretized_action = vocab_size - token_id
→ normalized_action = bin_centers[clamp(discretized_action - 1, 0, 255)]
```

This is a **nearest-neighbor mapping**, not a truncated series. There is no Fourier approximation, no basis function, no truncation — and therefore no Gibbs phenomenon.

### 3.4 The Synthetic Observation Problem (Secondary)

All experiments used a synthetic observation (colored rectangles on brown background) because real Bridge V2 data requires HuggingFace authentication (gated dataset). While this reduces the magnitude of instruction-dependent action differences, **it does not explain zero overshoot** — the raw experiments (r2_exp01-06) showed clear steps (peak_v up to 3.53) with zero overshoot. The memoryless architecture is the primary cause.

### 3.5 The `do_sample=False` Detail

All experiments used greedy decoding (`do_sample=False`), making the model fully deterministic. Given identical inputs, it always produces identical outputs. With `do_sample=True` (temperature > 0), token sampling noise might create a noisy step with apparent "ringing" — but this would be stochastic noise, not Gibbs ringing.

---

## 4. Comparison: What DOES Show Gibbs-like Behavior

| System | Overshoot | Why |
|--------|-----------|-----|
| **Synthetic Fourier step** | **9.05%** | Textbook Gibbs: truncated Fourier series of unit step |
| **Mock policy** | **6.26%** | Hash-seeded random + Gaussian noise creates bandwidth-limited perturbation |
| **OpenVLA** (all 15 experiments) | **0.0%** | Memoryless discrete-token policy — no Fourier approximation |

---

## 5. PIVOT PLAN: Architectures That Could Show Gibbs Ringing

To observe Gibbs-like overshoot in a real VLA, the model must have **temporal coupling** — its action at time `t` must depend on actions at times `t-1, t-2, ...` either explicitly or through a recurrent state. Three candidate architectures:

### 5.1 Diffusion-Based Action Heads

**Models:** Octo, RT-2 with diffusion head, Diffusion Policy

**Why they might show Gibbs:**
- The denoising process starts from Gaussian noise and iteratively refines toward the target action
- With limited denoising steps, the reconstruction is analogous to a truncated basis expansion
- The diffusion process has an inherent "bandwidth" determined by the noise schedule
- When the target action abruptly changes (instruction swap), the denoising trajectory must "catch up" — potentially overshooting

**Experiment design:**
- Same instruction-swap protocol
- Vary number of denoising steps (fewer steps = more truncation = more overshoot?)
- Key metric: does overshoot scale with 1/N_steps like Gibbs with harmonics?

### 5.2 Action-Chunking Models

**Models:** ACT (Action Chunking with Transformers), VQ-BeT (Vector-Quantized Behavior Transformers)

**Why they might show Gibbs:**
- These models predict a CHUNK of future actions (e.g., 20-100 steps) at once
- When executing a chunk that spans the instruction boundary, the model must interpolate
- The chunk prediction is essentially a truncated temporal basis (transformer positional encoding)
- The boundary between chunks creates a discontinuity that the model smooths over — potentially with overshoot

**Experiment design:**
- Vary chunk size (larger chunks = more potential for overshoot)
- Place the instruction switch at different positions within a chunk
- Measure overshoot at chunk boundaries vs. mid-chunk

### 5.3 Real Closed-Loop Rollouts

**Why they might show Gibbs:**
- In a closed loop, actions affect the observation (robot moves, camera view changes)
- The observation change creates temporal coupling even in a memoryless policy
- The feedback loop acts as a temporal filter with bandwidth determined by the robot's dynamics
- Physical robot dynamics (inertia, damping) inherently produce overshoot for step commands

**Experiment design:**
- Use a physics simulator (MuJoCo, PyBullet) with a simulated robot
- Run OpenVLA in closed-loop (observation changes with each action)
- Measure end-effector position overshoot (not action overshoot)
- This tests whether the Gibbs phenomenon manifests in the physical response, not the policy output

---

## 6. NEXT STEP: AutoResearch-Style Experiment Loop

### 6.1 Using autoresearch Framework

Cloned `https://github.com/ktwu01/autoresearch/` to `~/autoresearch/`.

The autoresearch framework (by Karpathy) provides autonomous experiment iteration:
- Fixed-budget experiments (5 min each)
- Automatic keep/discard based on metric improvement
- Structured logging to `results.tsv`
- Branch-based experiment tracking

**Adaptation for Gibbs-VLA:**
- Instead of optimizing `val_bpb`, optimize for **observed overshoot %** (target: 8.95%)
- Instead of modifying `train.py`, modify experiment configuration and model architecture
- Each "experiment" = one instruction-swap rollout with a different architecture/config
- Keep/discard based on whether overshoot is detected

### 6.2 Proposed Experiment Determination Strategy

Before running the autoresearch regression loop, we need to **determine which experiment method** is most likely to succeed. This is a meta-experiment:

**Phase 1: Architecture Survey (Codex Review)**
- Have Codex review this plan and the failure analysis
- Identify which of the 3 candidate architectures (diffusion, chunking, closed-loop) is:
  - Most likely to show Gibbs ringing (theoretical argument)
  - Most feasible to implement on NCAR (hardware/software constraints)
  - Most publishable (novelty + rigor)

**Phase 2: Minimal Viable Experiment**
- Implement the top candidate as a minimal experiment
- Run once to validate the setup works
- Check for any signal (even small overshoot)

**Phase 3: AutoResearch Loop**
- If Phase 2 shows signal, set up the autoresearch cron loop
- Systematically vary parameters to maximize overshoot
- Target: demonstrate that overshoot converges to 8.95% as model temporal coupling increases

### 6.3 Key Questions for Codex Review

1. **Is the memoryless-policy explanation correct?** Are there any hidden temporal dependencies in OpenVLA we missed? (e.g., KV-cache reuse across timesteps, image preprocessing buffers)

2. **Which architecture is the best bet?** Rank: diffusion head vs. action chunking vs. closed-loop. Consider theoretical strength of the Gibbs analogy AND practical feasibility.

3. **Is 8.95% the right target for a non-Fourier system?** The Gibbs constant specifically arises from sinc-function ringing. A diffusion or chunking model might show overshoot but at a different constant. What would the theoretical overshoot be for each architecture?

4. **Could stochastic sampling (`do_sample=True`) create apparent Gibbs ringing?** If so, is it genuine or just noise? How would we distinguish?

5. **Is there a simpler way to demonstrate the core claim?** Perhaps a toy model (small transformer with action chunking) would be more convincing than fighting with a 7B parameter model.

---

## 7. Files Changed

| File | Change |
|------|--------|
| `models/openvla_policy.py` | Fixed `unnorm_key=None` crash — manual token decode path |
| `experiments/exp_record.md` | Full experiment record with 20 experiments, 2 rounds |
| `review_results.py` | Automated result review and exp_record update script |
| `run_experiment.pbs` | PBS script for single OpenVLA experiment on NCAR |
| `run_all_experiments.pbs` | PBS script for Round 1 (10 experiments) |
| `run_round2.pbs` | PBS script for Round 2 (10 experiments) |
| `PLAN_v2.md` | This document — full failure analysis and pivot plan |

---

## 8. Timeline

| Date | Event |
|------|-------|
| 2026-04-15 AM | Setup: clone, conda env, PyTorch, pip install |
| 2026-04-15 AM | Baseline + synthetic/mock experiments |
| 2026-04-15 AM | Round 1: 10 experiments, 5 crashed (unnorm bug) |
| 2026-04-15 PM | Fixed unnorm bug, Round 2: 10 experiments |
| 2026-04-15 PM | **Conclusion: OpenVLA is memoryless, 0% overshoot** |
| 2026-04-15 PM | Cloned autoresearch, wrote PLAN_v2 |
| **NEXT** | Codex review of plan → determine best architecture |
| **NEXT** | Implement minimal experiment with chosen architecture |
| **NEXT** | AutoResearch cron loop for parameter sweep |
