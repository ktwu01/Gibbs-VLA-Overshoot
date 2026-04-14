"""
OpenVLA instruction-swap experiment on Bridge V2.

Phase 1 of the Gibbs transient analysis:
  1. Load a real observation frame from Bridge V2.
  2. Run OpenVLA inference for N steps with instruction A.
  3. At step t0, flip to instruction B (observation stays fixed).
  4. Record the full action trace.
  5. Run spectral analysis: overshoot ratio, PSD, derivatives.
  6. Save diagnostic figure + JSON report.

Usage
-----
    python -m experiments.openvla_swap_experiment \\
        --output experiments/outputs/openvla_swap.png \\
        --total-steps 100 --switch-step 50

    # With a custom instruction pair:
    python -m experiments.openvla_swap_experiment \\
        --output experiments/outputs/openvla_swap.png \\
        --instruction-a "pick up the red block" \\
        --instruction-b "push the blue block to the left" \\
        --episode-index 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.bridge_v2 import load_single_observation
from analysis.metrics import summarize_distribution
from analysis.spectral_engine import (
    analyze_step_signal,
    finite_difference,
    transient_psd,
)
from models.openvla_policy import OpenVLAPolicy
from models.vla_wrapper import InstructionSwapRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Output figure path.")
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--switch-step", type=int, default=50)
    p.add_argument("--fs", type=float, default=5.0, help="Control rate Hz (Bridge V2 ~5Hz).")
    p.add_argument("--instruction-a", default=None,
                    help="Pre-switch instruction. If None, uses the episode's own instruction.")
    p.add_argument("--instruction-b", default=None,
                    help="Post-switch instruction. If None, uses a default contrasting task.")
    p.add_argument("--episode-index", type=int, default=0, help="Bridge V2 episode to sample from.")
    p.add_argument("--step-index", type=int, default=None, help="Step within the episode for observation.")
    p.add_argument("--device", default="auto", help="auto | cuda:0 | mps | cpu")
    p.add_argument("--unnorm-key", default="bridge_orig",
                    help="OpenVLA de-normalization key. Use 'none' for raw [-1,1].")
    p.add_argument("--data-dir", default=None, help="TFDS data directory for Bridge V2.")
    p.add_argument("--axis", type=int, default=0, help="Action axis for 1-D spectral analysis.")
    p.add_argument("--use-flash-attn", action="store_true")
    return p


# Default contrasting instructions for Bridge V2 tasks
_DEFAULT_INSTRUCTION_PAIRS = [
    ("pick up the red block", "push the blue block to the left"),
    ("put the corn in the pot", "move the cloth to the right"),
    ("pick up the sushi", "close the drawer"),
]


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unnorm_key = None if args.unnorm_key.lower() == "none" else args.unnorm_key

    # --- Load a real observation from Bridge V2 ---
    print(f"Loading Bridge V2 observation (episode={args.episode_index})...")
    obs = load_single_observation(
        episode_index=args.episode_index,
        step_index=args.step_index,
        data_dir=args.data_dir,
    )
    print(f"  Image shape: {obs.image.shape}")
    print(f"  Original instruction: {obs.instruction}")

    instruction_a = args.instruction_a or obs.instruction
    instruction_b = args.instruction_b or _DEFAULT_INSTRUCTION_PAIRS[0][1]
    if instruction_a.lower().strip() == instruction_b.lower().strip():
        raise ValueError("instruction_a and instruction_b must differ")

    print(f"  Instruction A: {instruction_a}")
    print(f"  Instruction B: {instruction_b}")

    # --- Load OpenVLA ---
    print("Loading OpenVLA-7B...")
    t0 = time.time()
    policy = OpenVLAPolicy(
        unnorm_key=unnorm_key,
        device=args.device,
        use_flash_attn=args.use_flash_attn,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # --- Run the swap experiment ---
    runner = InstructionSwapRunner(policy=policy, observation=obs.image, fs=args.fs)
    print(f"Running {args.total_steps}-step rollout (switch at step {args.switch_step})...")
    t0 = time.time()
    result = runner.run(
        instruction_a=instruction_a,
        instruction_b=instruction_b,
        total_steps=args.total_steps,
        switch_step=args.switch_step,
    )
    print(f"  Inference done in {time.time() - t0:.1f}s")

    actions = result.actions
    t = np.arange(actions.shape[0]) / args.fs

    # --- Spectral analysis on each axis ---
    per_axis_metrics: list[dict] = []
    for d in range(actions.shape[1]):
        m = analyze_step_signal(actions[:, d], fs=args.fs, switch_index=result.switch_step)
        per_axis_metrics.append(m)

    # Detailed analysis on the chosen axis
    signal = actions[:, args.axis]
    metrics = per_axis_metrics[args.axis]
    velocity = finite_difference(signal, fs=args.fs, order=1)
    acceleration = finite_difference(signal, fs=args.fs, order=2)
    psd_freqs, psd = transient_psd(signal - np.mean(signal), fs=args.fs)
    vel_stats = summarize_distribution(np.abs(velocity))

    action_norm = np.linalg.norm(actions, axis=1)

    # --- Plotting (6-panel) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) all action dims
    ax = axes[0, 0]
    dim_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    for d in range(min(actions.shape[1], 7)):
        label = dim_labels[d] if d < len(dim_labels) else f"dim{d}"
        ax.plot(t, actions[:, d], linewidth=1.2, alpha=0.8, label=label)
    ax.axvline(result.switch_step / args.fs, color="red", ls="--", lw=1.5, label="switch")
    ax.set_title("Action Trace (all dims)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action value")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.grid(alpha=0.2)

    # (0,1) single axis step response
    ax = axes[0, 1]
    ax.plot(t, signal, color="#1f77b4", linewidth=2)
    ax.axvline(result.switch_step / args.fs, color="red", ls="--", lw=1.5)
    ax.set_title(f"{dim_labels[args.axis]} — Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.grid(alpha=0.2)
    ax.text(
        0.03, 0.93,
        f"Overshoot: {metrics['overshoot_ratio'] * 100:.2f}%\n"
        f"(Gibbs ≈ 8.95%)",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # (0,2) action norm
    ax = axes[0, 2]
    ax.plot(t, action_norm, color="#2ca02c", linewidth=1.8)
    ax.axvline(result.switch_step / args.fs, color="red", ls="--", lw=1.5)
    ax.set_title("Action Norm ||a||")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||action||")
    ax.grid(alpha=0.2)

    # (1,0) velocity + acceleration
    ax = axes[1, 0]
    ax.plot(t, velocity, color="#d62728", lw=1.4, label="velocity")
    ax.plot(t, acceleration, color="#ff7f0e", lw=1.0, label="acceleration")
    ax.axvline(result.switch_step / args.fs, color="red", ls="--", lw=1.5)
    ax.set_title("Transient Derivatives")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # (1,1) PSD
    ax = axes[1, 1]
    mask = psd_freqs > 0
    ax.loglog(psd_freqs[mask], psd[mask], color="#9467bd", linewidth=1.8)
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.2, which="both")
    ax.text(
        0.03, 0.93,
        f"Slope: {metrics['power_law_exponent']:.2f}\n"
        f"(step-like ≈ -1.0)",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # (1,2) overshoot summary across all 7 dims
    ax = axes[1, 2]
    overshoots = [m["overshoot_ratio"] * 100 for m in per_axis_metrics]
    colors = ["#e74c3c" if abs(o - 8.95) < 3 else "#3498db" for o in overshoots]
    bars = ax.bar(dim_labels[: len(overshoots)], overshoots, color=colors, edgecolor="white")
    ax.axhline(8.95, color="black", ls="--", lw=1, label="Gibbs constant (8.95%)")
    ax.set_title("Overshoot % by Action Dim")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Overshoot %")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"OpenVLA Instruction Swap on Bridge V2\n"
        f"\"{instruction_a}\" → \"{instruction_b}\"  (switch @ step {result.switch_step})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Figure saved: {output_path}")

    # --- JSON report ---
    report = {
        "model": "openvla/openvla-7b",
        "dataset": "bridge_v2",
        "episode_index": args.episode_index,
        "original_instruction": obs.instruction,
        "instruction_a": instruction_a,
        "instruction_b": instruction_b,
        "total_steps": args.total_steps,
        "switch_step": result.switch_step,
        "fs_hz": args.fs,
        "unnorm_key": args.unnorm_key,
        "per_axis_overshoot_percent": {
            dim_labels[d]: per_axis_metrics[d]["overshoot_ratio"] * 100
            for d in range(len(per_axis_metrics))
        },
        "analysis_axis": args.axis,
        "overshoot_ratio": metrics["overshoot_ratio"],
        "overshoot_percent": metrics["overshoot_ratio"] * 100.0,
        "peak_velocity": metrics["peak_velocity"],
        "peak_acceleration": metrics["peak_acceleration"],
        "power_law_exponent": metrics["power_law_exponent"],
        "high_band_ratio": metrics["high_band_ratio"],
        "velocity_mean": vel_stats["mean"],
        "velocity_p95": vel_stats["p95"],
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
