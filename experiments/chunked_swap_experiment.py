"""
Chunked-VLA instruction-swap experiment (pi0 / SmolVLA).

Unlike OpenVLA (memoryless, 1 action per step), these models predict
a CHUNK of future actions.  When the instruction changes mid-chunk,
the buffered actions still reflect the old instruction — creating
temporal coupling that may produce Gibbs-like overshoot.

Usage
-----
    python -m experiments.chunked_swap_experiment \
        --model lerobot/smolvla_base \
        --n-action-steps 50 \
        --output experiments/outputs/smolvla_chunk50.png

    python -m experiments.chunked_swap_experiment \
        --model lerobot/pi0_base \
        --n-action-steps 25 \
        --output experiments/outputs/pi0_chunk25.png
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
from models.chunked_policy import ChunkedVLAPolicy
from models.vla_wrapper import InstructionSwapRunner


_DEFAULT_INSTRUCTION_PAIRS = [
    ("pick up the red block", "push the blue block to the left"),
    ("put the corn in the pot", "move the cloth to the right"),
    ("pick up the sushi", "close the drawer"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="lerobot/smolvla_base",
                    help="LeRobot model ID (e.g. lerobot/smolvla_base, lerobot/pi0_base)")
    p.add_argument("--n-action-steps", type=int, default=50,
                    help="Actions executed per chunk before re-planning. Key variable for Gibbs.")
    p.add_argument("--output", required=True, help="Output figure path.")
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--switch-step", type=int, default=50)
    p.add_argument("--fs", type=float, default=5.0, help="Control rate Hz.")
    p.add_argument("--instruction-a", default=None)
    p.add_argument("--instruction-b", default=None)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--step-index", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--axis", type=int, default=0, help="Action axis for 1-D spectral analysis.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load observation ---
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

    # --- Load chunked policy ---
    print(f"Loading {args.model} (n_action_steps={args.n_action_steps})...")
    t0 = time.time()
    policy = ChunkedVLAPolicy(
        model_id=args.model,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Chunk size: {policy.chunk_size}")
    print(f"  Action dim: {policy.action_dim}")

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
    n_dims = actions.shape[1]
    t = np.arange(actions.shape[0]) / args.fs

    # --- Spectral analysis on each axis ---
    dim_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    if n_dims > len(dim_labels):
        dim_labels += [f"dim{i}" for i in range(len(dim_labels), n_dims)]
    dim_labels = dim_labels[:n_dims]

    per_axis_metrics: list[dict] = []
    for d in range(n_dims):
        m = analyze_step_signal(actions[:, d], fs=args.fs, switch_index=result.switch_step)
        per_axis_metrics.append(m)

    analysis_axis = min(args.axis, n_dims - 1)
    signal = actions[:, analysis_axis]
    metrics = per_axis_metrics[analysis_axis]
    velocity = finite_difference(signal, fs=args.fs, order=1)
    acceleration = finite_difference(signal, fs=args.fs, order=2)
    psd_freqs, psd = transient_psd(signal - np.mean(signal), fs=args.fs)
    vel_stats = summarize_distribution(np.abs(velocity))

    action_norm = np.linalg.norm(actions, axis=1)

    # --- Plotting (6-panel) ---
    model_short = args.model.split("/")[-1]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) all action dims
    ax = axes[0, 0]
    for d in range(min(n_dims, 7)):
        ax.plot(t, actions[:, d], linewidth=1.2, alpha=0.8, label=dim_labels[d])
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
    # Mark chunk boundaries
    for cb in range(0, args.total_steps, args.n_action_steps):
        ax.axvline(cb / args.fs, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"{dim_labels[analysis_axis]} — Step Response (chunk={args.n_action_steps})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.grid(alpha=0.2)
    ax.text(
        0.03, 0.93,
        f"Overshoot: {metrics['overshoot_ratio'] * 100:.2f}%\n"
        f"(Gibbs target: 8.95%)\n"
        f"n_action_steps: {args.n_action_steps}",
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
    if np.any(psd[mask] > 0):
        ax.loglog(psd_freqs[mask], psd[mask], color="#9467bd", linewidth=1.8)
    else:
        ax.plot(psd_freqs[mask], psd[mask], color="#9467bd", linewidth=1.8)
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.2, which="both")
    ax.text(
        0.03, 0.93,
        f"Slope: {metrics['power_law_exponent']:.2f}\n"
        f"(step-like: -1.0)",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # (1,2) overshoot summary across dims
    ax = axes[1, 2]
    overshoots = [m["overshoot_ratio"] * 100 for m in per_axis_metrics]
    colors = ["#e74c3c" if abs(o - 8.95) < 3 else "#3498db" for o in overshoots]
    ax.bar(dim_labels, overshoots, color=colors, edgecolor="white")
    ax.axhline(8.95, color="black", ls="--", lw=1, label="Gibbs constant (8.95%)")
    ax.set_title("Overshoot % by Action Dim")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Overshoot %")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"{model_short} Instruction Swap (n_action_steps={args.n_action_steps})\n"
        f"\"{instruction_a}\" -> \"{instruction_b}\"  (switch @ step {result.switch_step})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Figure saved: {output_path}")

    # --- JSON report ---
    report = {
        "model": args.model,
        "n_action_steps": args.n_action_steps,
        "chunk_size": policy.chunk_size,
        "instruction_a": instruction_a,
        "instruction_b": instruction_b,
        "total_steps": args.total_steps,
        "switch_step": result.switch_step,
        "fs_hz": args.fs,
        "per_axis_overshoot_percent": {
            dim_labels[d]: per_axis_metrics[d]["overshoot_ratio"] * 100
            for d in range(n_dims)
        },
        "analysis_axis": analysis_axis,
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
