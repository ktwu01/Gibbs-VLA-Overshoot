# Gibbs Overshoot Reproduction Experiment Record

**Goal:** Reproduce the ~8.95% Gibbs phenomenon in OpenVLA instruction-swap experiments.

**Baseline results (2026-04-15):**
- Synthetic Gibbs: 9.05% overshoot (CONFIRMED - ground truth)
- Mock policy: 6.26% overshoot
- OpenVLA default (axis=0, bridge_orig, 100 steps, switch@50): 0.0% overshoot

---

## Round 1: Varying axis, switch timing, instruction pairs

| ID | Axis | Unnorm | Steps | Switch | Instruction A | Instruction B | Rationale |
|----|------|--------|-------|--------|---------------|---------------|-----------|
| exp01 | 2 (dz) | bridge_orig | 150 | 60 | (default) | (default) | Z-axis largest response, longer capture |
| exp02 | 0 (dx) | none | 100 | 50 | (default) | (default) | Raw normalized actions, clean signal |
| exp03 | 1 (dy) | none | 100 | 50 | (default) | (default) | Raw Y-axis |
| exp04 | 2 (dz) | none | 100 | 50 | (default) | (default) | Raw Z-axis |
| exp05 | 0 (dx) | bridge_orig | 100 | 25 | (default) | (default) | Early switch, more transient time |
| exp06 | 0 (dx) | bridge_orig | 100 | 50 | "pick up the sushi" | "close the drawer" | Max semantic contrast pair |
| exp07 | 2 (dz) | bridge_orig | 100 | 50 | "pick up the sushi" | "close the drawer" | Max contrast + Z-axis |
| exp08 | 0 (dx) | none | 200 | 100 | "put the corn in the pot" | "move the cloth to the right" | Long rollout, different pair |
| exp09 | 6 (grip) | bridge_orig | 100 | 50 | "pick up the red block" | "close the drawer" | Gripper axis (pick vs close) |
| exp10 | 2 (dz) | none | 150 | 40 | "pick up the sushi" | "push the blue block to the left" | Early switch, raw Z, high contrast |

### Round 1 Results

| ID | Overshoot % | Power Law Exp | Peak Velocity | Status |
|----|-------------|---------------|---------------|--------|
| baseline | 0.0% | 0.347 | 0.0056 | DONE |
| exp01 | 0.0000% | -0.8575 | 0.024034 | DONE |
| exp02 | - | - | - | CRASHED (unnorm-key none bug) |
| exp03 | - | - | - | CRASHED (unnorm-key none bug) |
| exp04 | - | - | - | CRASHED (unnorm-key none bug) |
| exp05 | 0.0000% | -1.1747 | 0.005592 | DONE |
| exp06 | -0.0000% | 0.3790 | 0.077727 | DONE |
| exp07 | 0.0000% | -0.0010 | 0.030529 | DONE |
| exp08 | - | - | - | CRASHED (unnorm-key none bug) |
| exp09 | nan% | nan | 0.000000 | DONE |
| exp10 | - | - | - | CRASHED (unnorm-key none bug) |

### Round 1 Observations

1. **Zero overshoot across all completed experiments.** No Gibbs phenomenon detected.
2. **5 of 10 experiments crashed** due to `--unnorm-key none` bug: OpenVLA's `predict_action` requires a valid unnorm_key when model was trained on multiple datasets. **Fixed in `openvla_policy.py`** — now manually decodes tokens and returns raw normalized actions.
3. **Synthetic observation is the bottleneck.** All experiments used a fake image (colored rectangles on brown background). The model outputs near-constant actions regardless of instruction — there is no meaningful "step" to ring on.
4. Power law exponents (-0.86 to -1.17) in some experiments suggest step-like spectral structure exists in the signal, but the jump magnitude is effectively zero.
5. Bridge V2 real data via HuggingFace LeRobot requires authentication (gated dataset).

---

## Round 2: Fixed unnorm-key + raw token-space analysis

**Changes from Round 1:**
- Fixed `openvla_policy.py` to handle `unnorm_key=None` by manually decoding action tokens
- Focus on raw normalized actions (token space) where Gibbs ringing is most likely visible
- Vary episode index (different synthetic observations) to find instruction-sensitive cases
- Try different dataset unnorm keys (fractal) to amplify action differences

| ID | Axis | Unnorm | Steps | Switch | Instruction A | Instruction B | Rationale |
|----|------|--------|-------|--------|---------------|---------------|-----------|
| r2_exp01 | 0 (dx) | none | 100 | 50 | (default) | (default) | Raw X-axis, fixed code |
| r2_exp02 | 1 (dy) | none | 100 | 50 | (default) | (default) | Raw Y-axis |
| r2_exp03 | 2 (dz) | none | 100 | 50 | (default) | (default) | Raw Z-axis |
| r2_exp04 | 0 (dx) | none | 100 | 50 | "pick up the sushi" | "close the drawer" | High contrast, raw |
| r2_exp05 | 0 (dx) | none | 200 | 50 | (default) | (default) | Early switch, long transient capture |
| r2_exp06 | 0 (dx) | none | 100 | 50 | "put the corn in the pot" | "move the cloth to the right" | Different pair |
| r2_exp07 | 0 (dx) | none | 100 | 50 | (default) | (default) | Episode 1 (CRASHED) |
| r2_exp08 | 0 (dx) | none | 100 | 50 | (default) | (default) | Episode 2 (different synth obs) |
| r2_exp09 | 0 (dx) | fractal | 100 | 50 | "pick up the sushi" | "close the drawer" | Fractal dataset unnorm (wider range) |
| r2_exp10 | 6 (grip) | none | 100 | 50 | "pick up the red block" | "push the blue block" | Raw gripper axis |

### Round 2 Results

| ID | Overshoot % | Peak Velocity | Power Law Exp | Status |
|----|-------------|---------------|---------------|--------|
| r2_exp01 | 0.0000% | 1.235 | 0.366 | DONE |
| r2_exp02 | 0.0000% | 1.137 | 0.382 | DONE |
| r2_exp03 | 0.0000% | 1.608 | 0.440 | DONE |
| r2_exp04 | 0.0000% | 0.980 | 0.305 | DONE |
| r2_exp05 | 0.0000% | 1.235 | -0.894 | DONE |
| r2_exp06 | 0.0000% | 3.529 | 0.392 | DONE |
| r2_exp07 | - | - | - | CRASHED |
| r2_exp08 | 0.0000% | 0.039 | 0.484 | DONE |
| r2_exp09 | 0.0000% | 0.549 | 0.307 | DONE |
| r2_exp10 | nan% | 0.000 | nan | DONE |

### Round 2 Observations

1. **Still zero overshoot across all experiments**, even with raw token-space actions.
2. **Peak velocities are non-zero** (0.04 to 3.53), confirming the model DOES produce different actions for different instructions — there IS a step transition.
3. **But the step is instantaneous and clean** — no ringing, no overshoot, no transient.
4. The `unnorm-key none` fix works correctly. Raw normalized actions are in ~[-1, 1] range.
5. r2_exp07 (episode 1) crashed for unknown reason; r2_exp10 (gripper) shows NaN because gripper is constant.

---

## Final Analysis & Conclusion (2026-04-15)

### Why OpenVLA does NOT exhibit Gibbs ringing

After 20 experiments across 2 rounds, varying axis (0-6), unnorm key (bridge_orig, fractal, none), switch timing (25-100), rollout length (100-200), instruction pairs (6 combinations), and episode indices (0-2):

**Result: 0% overshoot in all experiments. The Gibbs phenomenon does not occur in OpenVLA.**

**Root cause: OpenVLA is a memoryless (Markov) policy.**

The Gibbs phenomenon requires a **truncated Fourier series approximation** of a discontinuous function. This arises in systems with:
- Temporal memory / recurrence (RNNs, temporal convolutions)
- State-dependent smoothing (low-pass filtering, moving averages)
- Bandwidth-limited signal reconstruction

OpenVLA has **none of these**. Each inference step is:
```
action_t = f(image, instruction)  # no dependence on action_{t-1}
```

The model receives a fixed image + instruction and independently generates 7 action tokens via greedy decoding. When the instruction changes at the switch step:
- Step t=49: `f(image, "pick up the red block")` = action_A
- Step t=50: `f(image, "push the blue block")` = action_B

This is a **perfect step function** — instantaneous transition with zero transient. There is no Fourier approximation involved, no bandwidth limitation, and no temporal smoothing. The discrete token quantization (256 bins) doesn't introduce ringing because it's a nearest-bin mapping, not a truncated series.

### Conditions that WOULD produce Gibbs ringing in a VLA

For a VLA to exhibit Gibbs-like overshoot, it would need:
1. **Temporal autoregression across timesteps** — e.g., action chunking where the model predicts a sequence and must smoothly interpolate
2. **Recurrent hidden state** — where the instruction change creates a transient in the hidden dynamics
3. **Continuous action spaces with bandwidth-limited decoding** — e.g., a diffusion-based action head with limited denoising steps
4. **Real closed-loop interaction** — where the observation changes in response to actions, creating feedback dynamics

### Recommendations for future work

- Test on **diffusion-based VLAs** (e.g., Octo, RT-2 with diffusion head) where the denoising process may introduce Fourier-like transients
- Test on **action-chunking models** (ACT, VQ-BeT) where temporal interpolation between chunks could produce overshoot
- Test with **real closed-loop rollouts** where observation feedback creates temporal coupling
- The mock policy (6.26% overshoot) succeeds because it uses hash-seeded random generation with noise — the noise acts as a bandwidth-limited perturbation

---

## Round 3: pi0 action-chunking experiments (2026-04-15)

**Model:** pi0_base (4B params, flow-matching, 50-action chunks, 32-dim actions)
**Env:** gibbs312 (Python 3.12, transformers 5.5.4, lerobot 0.5.2)

### Results

| ID | n_action_steps | Overshoot % | Peak Velocity | Power Law | Instructions |
|----|---------------|-------------|---------------|-----------|-------------|
| r3_exp07 | 50 | **1027.5%** | 1.031 | -0.073 | default |
| r3_exp_c25 | 25 | **923.9%** | 1.245 | -0.069 | default |
| r3_exp_c10 | 10 | **1367.8%** | 1.263 | -0.098 | default |
| r3_exp_c1 | 1 | **1048.1%** | 1.383 | -0.193 | default |
| r3_exp_sushi | 50 | **527.7%** | 1.061 | -0.140 | sushi/drawer |

### Analysis

**Massive overshoot detected (500-1400%) but NOT Gibbs-like.**

Key observations:
1. **Overshoot does NOT scale with n_action_steps**: c=1 (1048%) ≈ c=50 (1028%). If this were chunk-boundary Gibbs ringing, c=1 should show near-zero overshoot (re-plan every step).
2. **Per-axis overshoot is chaotic**: dim13 shows 41610%, dim12 shows 41058%. This is noise, not a clean ~9% overshoot.
3. **high_band_ratio ~0.5**: half the spectral energy is high-frequency, indicating chaotic/noisy output, not step-like transient.
4. **velocity_p95 > 0**: unlike OpenVLA (always 0), pi0 produces continuously varying actions, but this is noise not signal.

**Root cause: Out-of-distribution inputs.**
- Synthetic observation (colored rectangles) ≠ real robot scene
- Zero robot state (32 dims) ≠ real joint positions
- Same image on all 3 cameras (base, left wrist, right wrist) ≠ real multi-view
- pi0 was trained on bimanual manipulation (32-DOF), not Bridge V2 (7-DOF)
- The model produces effectively random/chaotic actions on these OOD inputs

**Conclusion:** The overshoot is model instability on OOD inputs, not Gibbs ringing from temporal coupling. To test the Gibbs hypothesis properly, we need:
1. Real robot observations (or a realistic simulator)
2. A model trained on the same observation distribution
3. In-distribution state inputs

---

## Round 4: INTACT pi0-finetune-bridge — PENDING (2026-04-16)

**Model:** juexzz/INTACT-pi0-finetune-bridge (Bridge V2-trained, single camera `observation.images.top`, 7-dim EEF delta actions)
**Env:** gibbs312 (Python 3.12)
**Script:** `run_round4.pbs` — PBS job submitted to NCAR Casper (A100 80GB, 2h walltime)

### Planned experiments

| ID | n_action_steps | chunk_size | Axis | Instructions | Rationale |
|----|---------------|------------|------|-------------|-----------|
| r4_pi0bridge_c4 | 4 | 4 | 0 (dx) | default | Full-chunk execution, in-distribution model |
| r4_pi0bridge_c2 | 2 | 4 | 0 (dx) | default | Half-chunk — tests overshoot vs coupling depth |
| r4_pi0bridge_c1 | 1 | 4 | 0 (dx) | default | Memoryless baseline — should show 0% if chunking drives it |
| r4_pi0bridge_c4_sushi | 4 | 4 | 0 (dx) | sushi / drawer | High semantic contrast |
| r4_pi0bridge_c4_corn | 4 | 4 | 0 (dx) | corn / cloth | Alternative contrast pair |
| r4_pi0bridge_c4_ax1 | 4 | 4 | 1 (dy) | default | Y-axis sweep |
| r4_pi0bridge_c4_ax2 | 4 | 4 | 2 (dz) | default | Z-axis sweep |
| r4_pi0bridge_c4_ax6 | 4 | 4 | 6 (grip) | default | Gripper axis |

### Status

**Results: NOT YET AVAILABLE** as of 2026-04-16 (status check at 15:05 UTC).

- `gibbs-round4.out` does not exist — PBS job has not yet run or not yet written output
- No `experiments/outputs/r4_*.json` files found
- Last commit: `2801477` (2026-04-16) — added Round 4 pending entry to exp_record.md
- Previous commit: `d766270` (2026-04-15 PM) — added lessons learned + paper strategy; Round 4 submitted

**Status check log:**

| Check Time | r4 JSONs | gibbs-round4.out | PBS Status |
|-----------|----------|-----------------|------------|
| 2026-04-16 15:05 UTC | 0 files | absent | NOT STARTED / IN QUEUE |

### What to look for when results arrive

- **SUCCESS:** Any axis showing overshoot within 5.95%–11.95% (±3% of 8.95%)
- **PARTIAL:** Non-zero overshoot outside that range
- **CHAOTIC:** Overshoot >100% (still OOD noise despite in-distribution model)
- **ZERO:** ~0% overshoot (model is also memoryless / chunking not sufficient)

Key diagnostic: if `r4_pi0bridge_c1` (n_action_steps=1) shows ~0% and `r4_pi0bridge_c4` shows >0%, that is strong evidence the chunk temporal coupling is the driver — consistent with the Gibbs mechanism.
