"""
Phase-1 instruction-swap inference experiment.

Hold a single observation frame constant, run a policy for `total_steps`,
and flip the text instruction at `switch_step`.  Then feed the resulting
action trace into the spectral / Gibbs analysis pipeline.

Usage
-----
    python -m experiments.instruction_swap_inference \
        --output experiments/outputs/instruction_swap.png \
        --total-steps 100 --switch-step 50
"""

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
    trajectory_derivatives,
    transient_psd,
)
from models.vla_wrapper import InstructionSwapRunner, MockPolicy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Path for the output figure.")
    p.add_argument("--total-steps", type=int, default=100, help="Rollout length.")
    p.add_argument("--switch-step", type=int, default=50, help="Step at which instruction flips.")
    p.add_argument("--action-dim", type=int, default=7, help="Action dimensionality.")
    p.add_argument("--noise-std", type=float, default=0.02, help="Policy noise std dev.")
    p.add_argument("--fs", type=float, default=10.0, help="Control rate (Hz).")
    p.add_argument("--instruction-a", default="push the red block to the blue block")
    p.add_argument("--instruction-b", default="push the blue block to the green block")
    p.add_argument("--axis", type=int, default=0, help="Action axis for 1-D analysis.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- run the swap experiment ---
    policy = MockPolicy(
        action_dim=args.action_dim,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    runner = InstructionSwapRunner(policy=policy, fs=args.fs)
    result = runner.run(
        instruction_a=args.instruction_a,
        instruction_b=args.instruction_b,
        total_steps=args.total_steps,
        switch_step=args.switch_step,
    )

    actions = result.actions  # (T, D)
    t = np.arange(actions.shape[0]) / args.fs  # time in seconds

    # --- per-axis 1-D analysis on the chosen axis ---
    signal = actions[:, args.axis]
    metrics = analyze_step_signal(signal, fs=args.fs, switch_index=result.switch_step)

    velocity = finite_difference(signal, fs=args.fs, order=1)
    acceleration = finite_difference(signal, fs=args.fs, order=2)
    psd_freqs, psd = transient_psd(signal - np.mean(signal), fs=args.fs)
    vel_stats = summarize_distribution(np.abs(velocity))

    # --- multi-axis action norm ---
    action_norm = np.linalg.norm(actions, axis=1)

    # --- plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) raw action trace, all axes
    ax = axes[0, 0]
    for d in range(min(actions.shape[1], 7)):
        ax.plot(t, actions[:, d], linewidth=1.2, alpha=0.8, label=f"dim {d}")
    ax.axvline(result.switch_step / args.fs, color="red", linestyle="--", linewidth=1.5, label="switch")
    ax.set_title("Action Trace (all dims)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action value")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.2)

    # (0,1) single-axis signal with overshoot annotation
    ax = axes[0, 1]
    ax.plot(t, signal, color="#1f77b4", linewidth=2)
    ax.axvline(result.switch_step / args.fs, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"Axis {args.axis} — Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.grid(alpha=0.2)
    ax.text(
        0.03, 0.93,
        f"Overshoot: {metrics['overshoot_ratio'] * 100:.2f}%",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # (0,2) action norm
    ax = axes[0, 2]
    ax.plot(t, action_norm, color="#2ca02c", linewidth=1.8)
    ax.axvline(result.switch_step / args.fs, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Action Norm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||action||")
    ax.grid(alpha=0.2)

    # (1,0) velocity + acceleration
    ax = axes[1, 0]
    ax.plot(t, velocity, color="#d62728", linewidth=1.4, label="velocity")
    ax.plot(t, acceleration, color="#ff7f0e", linewidth=1.0, label="acceleration")
    ax.axvline(result.switch_step / args.fs, color="red", linestyle="--", linewidth=1.5)
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
        f"Slope: {metrics['power_law_exponent']:.2f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # (1,2) velocity histogram
    ax = axes[1, 2]
    ax.hist(np.abs(velocity), bins=30, color="#ff7f0e", alpha=0.9, edgecolor="white")
    ax.set_title("|Velocity| Distribution")
    ax.set_xlabel("|velocity|")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.text(
        0.03, 0.93,
        f"mean={vel_stats['mean']:.3f}\np95={vel_stats['p95']:.3f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    fig.suptitle(
        f"Instruction Swap: \"{args.instruction_a}\" → \"{args.instruction_b}\"  "
        f"(switch @ step {result.switch_step})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    # --- JSON report ---
    report = {
        "instruction_a": args.instruction_a,
        "instruction_b": args.instruction_b,
        "total_steps": args.total_steps,
        "switch_step": result.switch_step,
        "fs_hz": args.fs,
        "action_dim": args.action_dim,
        "noise_std": args.noise_std,
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
