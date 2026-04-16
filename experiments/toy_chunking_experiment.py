"""
Toy action-chunking experiment: controlled Gibbs overshoot test.

Trains a tiny MLP with sinusoidal positional encodings on two synthetic
tasks (constant action targets), then runs an instruction-swap experiment
to observe Gibbs-like ringing at the boundary.

The sinusoidal positional encoding acts as a truncated Fourier basis.
When the model must reconstruct a step function (sudden instruction
change), the finite basis produces Gibbs ringing naturally -- overshoot
near the theoretical 8.95% Gibbs constant.

No GPU needed. Runs on CPU in under 1 minute.

Usage
-----
    python -m experiments.toy_chunking_experiment \\
        --output experiments/outputs/toy_chunk.png

    # With custom parameters:
    python -m experiments.toy_chunking_experiment \\
        --output experiments/outputs/toy_chunk.png \\
        --chunk-size 50 --action-dim 7 --n-epochs 3000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import summarize_distribution
from analysis.spectral_engine import (
    analyze_step_signal,
    finite_difference,
    transient_psd,
)
from models.toy_chunked_model import ToyChunkedModel, ToyChunkedPolicy
from models.vla_wrapper import InstructionSwapRunner


# ---------------------------------------------------------------------------
# Synthetic task definitions
# ---------------------------------------------------------------------------

TASK_A = "task_A"
TASK_B = "task_B"

# Target action vectors for each task (7-DOF robot arm style)
TASK_TARGETS = {
    TASK_A: np.array([0.30, 0.10, 0.25, -0.05, 0.15, -0.10, 0.80]),
    TASK_B: np.array([-0.20, 0.40, -0.15, 0.30, -0.25, 0.20, -0.60]),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Toy action-chunking Gibbs overshoot experiment."
    )
    p.add_argument("--output", required=True, help="Output figure path.")
    p.add_argument("--chunk-size", type=int, default=50,
                   help="Actions per chunk (temporal window).")
    p.add_argument("--action-dim", type=int, default=7,
                   help="Action dimensionality.")
    p.add_argument("--embed-dim", type=int, default=32,
                   help="Instruction/positional embedding dimension.")
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="MLP hidden layer width.")
    p.add_argument("--n-epochs", type=int, default=3000,
                   help="Training epochs.")
    p.add_argument("--lr", type=float, default=0.005,
                   help="Learning rate.")
    p.add_argument("--total-steps", type=int, default=200,
                   help="Total rollout steps for the swap experiment.")
    p.add_argument("--switch-step", type=int, default=100,
                   help="Step at which instruction changes (should be a chunk boundary).")
    p.add_argument("--fs", type=float, default=10.0,
                   help="Control rate Hz (for spectral analysis).")
    p.add_argument("--axis", type=int, default=0,
                   help="Action axis for detailed 1-D spectral analysis.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Trim targets to requested action_dim
    task_targets = {
        k: v[: args.action_dim] for k, v in TASK_TARGETS.items()
    }
    # If action_dim > 7, pad with zeros
    for k in task_targets:
        if task_targets[k].shape[0] < args.action_dim:
            task_targets[k] = np.pad(
                task_targets[k],
                (0, args.action_dim - task_targets[k].shape[0]),
            )

    # ---- 1. Build and train the model ----
    print(f"Building toy chunked model (chunk={args.chunk_size}, "
          f"action_dim={args.action_dim}, embed={args.embed_dim}, "
          f"hidden={args.hidden_dim})")

    model = ToyChunkedModel(
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    print(f"Training on {len(task_targets)} synthetic tasks for {args.n_epochs} epochs...")
    t0 = time.time()
    losses = model.train(
        task_targets=task_targets,
        n_epochs=args.n_epochs,
        lr=args.lr,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s  (final loss={losses[-1]:.6f})")

    # Verify training quality: predict chunks and check error
    for task_name, target in task_targets.items():
        pred = model.predict_chunk(task_name)
        err = np.max(np.abs(pred - target))
        print(f"  {task_name}: max_abs_error={err:.4f}")

    # ---- 2. Run instruction-swap experiment ----
    policy = ToyChunkedPolicy(model)

    # Align switch_step to chunk boundary for clean overshoot
    switch_step = args.switch_step
    if switch_step % args.chunk_size != 0:
        aligned = (switch_step // args.chunk_size) * args.chunk_size
        if aligned < 1:
            aligned = args.chunk_size
        print(f"  Aligning switch_step {switch_step} -> {aligned} (chunk boundary)")
        switch_step = aligned

    total_steps = args.total_steps
    if total_steps <= switch_step:
        total_steps = switch_step + args.chunk_size * 2
        print(f"  Extending total_steps to {total_steps}")

    runner = InstructionSwapRunner(
        policy=policy,
        observation=None,  # toy model ignores observations
        fs=args.fs,
    )

    print(f"Running {total_steps}-step rollout (switch at step {switch_step})...")
    t0 = time.time()
    result = runner.run(
        instruction_a=TASK_A,
        instruction_b=TASK_B,
        total_steps=total_steps,
        switch_step=switch_step,
    )
    infer_time = time.time() - t0
    print(f"  Inference done in {infer_time:.2f}s")

    actions = result.actions  # (T, D)
    n_dims = actions.shape[1]
    t = np.arange(actions.shape[0]) / args.fs

    # ---- 3. Spectral analysis ----
    dim_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
    if n_dims > len(dim_labels):
        dim_labels += [f"dim{i}" for i in range(len(dim_labels), n_dims)]
    dim_labels = dim_labels[:n_dims]

    per_axis_metrics: list[dict] = []
    for d in range(n_dims):
        m = analyze_step_signal(
            actions[:, d], fs=args.fs, switch_index=result.switch_step
        )
        per_axis_metrics.append(m)

    analysis_axis = min(args.axis, n_dims - 1)
    signal = actions[:, analysis_axis]
    metrics = per_axis_metrics[analysis_axis]
    velocity = finite_difference(signal, fs=args.fs, order=1)
    acceleration = finite_difference(signal, fs=args.fs, order=2)
    psd_freqs, psd = transient_psd(signal - np.mean(signal), fs=args.fs)
    vel_stats = summarize_distribution(np.abs(velocity))

    action_norm = np.linalg.norm(actions, axis=1)

    # ---- 4. Plotting (6-panel figure) ----
    fig, axes_arr = plt.subplots(2, 3, figsize=(16, 9))

    # -- (0,0) All action dims --
    ax = axes_arr[0, 0]
    for d in range(min(n_dims, 7)):
        ax.plot(t, actions[:, d], linewidth=1.2, alpha=0.8, label=dim_labels[d])
    ax.axvline(switch_step / args.fs, color="red", ls="--", lw=1.5, label="switch")
    # Show target levels
    for d in range(min(n_dims, 3)):
        ax.axhline(task_targets[TASK_A][d], color="gray", ls=":", lw=0.5, alpha=0.4)
        ax.axhline(task_targets[TASK_B][d], color="gray", ls=":", lw=0.5, alpha=0.4)
    ax.set_title("Action Trace (all dims)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action value")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.grid(alpha=0.2)

    # -- (0,1) Single axis step response --
    ax = axes_arr[0, 1]
    ax.plot(t, signal, color="#1f77b4", linewidth=2, label="predicted")
    ax.axvline(switch_step / args.fs, color="red", ls="--", lw=1.5)
    # Mark chunk boundaries
    for cb in range(0, total_steps, args.chunk_size):
        ax.axvline(cb / args.fs, color="gray", ls=":", lw=0.8, alpha=0.4)
    # Target levels
    ax.axhline(task_targets[TASK_A][analysis_axis], color="#2ca02c", ls="--",
               lw=1, alpha=0.7, label=f"target A ({task_targets[TASK_A][analysis_axis]:.2f})")
    ax.axhline(task_targets[TASK_B][analysis_axis], color="#ff7f0e", ls="--",
               lw=1, alpha=0.7, label=f"target B ({task_targets[TASK_B][analysis_axis]:.2f})")
    ax.set_title(f"{dim_labels[analysis_axis]} -- Step Response (chunk={args.chunk_size})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2)
    ax.text(
        0.03, 0.93,
        f"Overshoot: {metrics['overshoot_ratio'] * 100:.2f}%\n"
        f"(Gibbs target: 8.95%)\n"
        f"chunk_size: {args.chunk_size}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    # -- (0,2) Action norm --
    ax = axes_arr[0, 2]
    ax.plot(t, action_norm, color="#2ca02c", linewidth=1.8)
    ax.axvline(switch_step / args.fs, color="red", ls="--", lw=1.5)
    ax.set_title("Action Norm ||a||")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||action||")
    ax.grid(alpha=0.2)

    # -- (1,0) Velocity + acceleration --
    ax = axes_arr[1, 0]
    ax.plot(t, velocity, color="#d62728", lw=1.4, label="velocity")
    ax.plot(t, acceleration, color="#ff7f0e", lw=1.0, label="acceleration")
    ax.axvline(switch_step / args.fs, color="red", ls="--", lw=1.5)
    ax.set_title("Transient Derivatives")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # -- (1,1) PSD --
    ax = axes_arr[1, 1]
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

    # -- (1,2) Overshoot summary across dims --
    ax = axes_arr[1, 2]
    overshoots = [m["overshoot_ratio"] * 100 for m in per_axis_metrics]
    colors = ["#e74c3c" if abs(o - 8.95) < 5 else "#3498db" for o in overshoots]
    ax.bar(dim_labels, overshoots, color=colors, edgecolor="white")
    ax.axhline(8.95, color="black", ls="--", lw=1, label="Gibbs constant (8.95%)")
    ax.set_title("Overshoot % by Action Dim")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Overshoot %")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"Toy Chunked Model -- Gibbs Overshoot Test\n"
        f"\"{TASK_A}\" -> \"{TASK_B}\"  "
        f"(switch @ step {switch_step}, chunk={args.chunk_size})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Figure saved: {output_path}")

    # ---- 5. JSON report ----
    report = {
        "model": "toy_chunked_model",
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "n_epochs": args.n_epochs,
        "final_training_loss": losses[-1],
        "training_time_s": round(train_time, 2),
        "instruction_a": TASK_A,
        "instruction_b": TASK_B,
        "target_a": task_targets[TASK_A].tolist(),
        "target_b": task_targets[TASK_B].tolist(),
        "total_steps": total_steps,
        "switch_step": switch_step,
        "fs_hz": args.fs,
        "per_axis_overshoot_percent": {
            dim_labels[d]: round(per_axis_metrics[d]["overshoot_ratio"] * 100, 4)
            for d in range(n_dims)
        },
        "analysis_axis": analysis_axis,
        "overshoot_ratio": metrics["overshoot_ratio"],
        "overshoot_percent": round(metrics["overshoot_ratio"] * 100, 4),
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
