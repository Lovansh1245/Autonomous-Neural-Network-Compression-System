#!/usr/bin/env python3
"""
plot_gate_distribution.py — Extract all final gate values from the best-trained
model and plot a histogram of their distribution.

The plot shows the characteristic bimodal distribution: a large spike near 0
(pruned weights) and a cluster near 1 (active weights).

Outputs: gate_distribution.png in the project root directory.

Usage:
    python plot_gate_distribution.py
    python plot_gate_distribution.py --lambda_val 0.001
    python plot_gate_distribution.py --model_path outputs/models/model_lambda_0.0001.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from model import PrunableCNN
from layers import PrunableLinear


# ─── Style Constants ─────────────────────────────────────────────────────────────

BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"
ACCENT_1 = "#6366f1"   # Indigo (active gates)
ACCENT_2 = "#f43f5e"   # Rose (pruned region)
THRESHOLD_COLOR = "#ef4444"


def load_model(model_path: str | Path) -> tuple[PrunableCNN, dict]:
    """Load a trained PrunableCNN model from a checkpoint file."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PrunableCNN(num_classes=10, temperature=1.0)
    model.load_state_dict(ckpt["model_state_dict"])

    # Restore the final training temperature
    config = ckpt.get("config", {})
    temp_final = config.get("gate_temp_final", 0.5)
    model.set_temperature(temp_final)
    model.eval()

    return model, ckpt


def extract_all_gate_values(model: PrunableCNN) -> np.ndarray:
    """
    Extract all gate activation values from every PrunableLinear layer.

    Gate values are computed as sigmoid(gate_scores / temperature) where
    temperature is the model's current temperature setting.

    Returns:
        1-D numpy array of all gate values across all PrunableLinear layers.
    """
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers:
            if isinstance(layer, PrunableLinear):
                gate_vals = layer.gate_activations.cpu().numpy().flatten()
                all_gates.append(gate_vals)
    return np.concatenate(all_gates)


def compute_sparsity_level(model: PrunableCNN) -> float:
    """
    Compute Sparsity Level (%) strictly as:
      percentage of all weights across all PrunableLinear layers
      where the final gate value (sigmoid of gate_scores / temperature) < 1e-2.
    """
    total = 0
    pruned = 0
    with torch.no_grad():
        for layer in model.prunable_layers:
            if isinstance(layer, PrunableLinear):
                gate_vals = layer.gate_activations
                total += gate_vals.numel()
                pruned += (gate_vals < 1e-2).sum().item()
    return (pruned / total) * 100.0 if total > 0 else 0.0


def plot_gate_histogram(
    gate_values: np.ndarray,
    lambda_val: float,
    sparsity_pct: float,
    output_path: Path,
) -> None:
    """
    Plot a publication-quality histogram of gate activation values.

    The histogram demonstrates the bimodal distribution characteristic of
    L1-penalized gates:
      - Large spike at 0: pruned weights (gate → 0)
      - Cluster near 1: active weights (gate → 1)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dark theme
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)

    # ── Histogram ──
    bins = np.linspace(0, 1, 80)
    counts, bin_edges, patches = ax.hist(
        gate_values,
        bins=bins,
        color=ACCENT_1,
        alpha=0.85,
        edgecolor="none",
        zorder=3,
    )

    # Color bins below threshold in rose/red
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < 0.01:
            patch.set_facecolor(ACCENT_2)
            patch.set_alpha(0.9)

    # ── Threshold line ──
    ax.axvline(
        x=0.01,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Prune threshold (0.01)",
        zorder=5,
    )

    # ── Statistics box ──
    n_total = len(gate_values)
    n_pruned = (gate_values < 1e-2).sum()
    n_active = n_total - n_pruned
    mean_val = gate_values.mean()
    median_val = np.median(gate_values)

    stats_text = (
        f"Total gates: {n_total:,}\n"
        f"Pruned (<0.01): {n_pruned:,} ({sparsity_pct:.1f}%)\n"
        f"Active (≥0.01): {n_active:,} ({100-sparsity_pct:.1f}%)\n"
        f"Mean: {mean_val:.4f}\n"
        f"Median: {median_val:.4f}"
    )
    ax.text(
        0.97, 0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right", va="top",
        color=TEXT_COLOR,
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1e293b",
            edgecolor=GRID_COLOR,
            alpha=0.9,
        ),
        zorder=10,
    )

    # ── Labels ──
    ax.set_xlabel("Gate Activation Value", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Gate Value Distribution — λ = {lambda_val}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.legend(
        loc="upper center",
        fontsize=11,
        framealpha=0.6,
        facecolor="#1e293b",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    # ── Annotations ──
    # Arrow pointing to the spike at 0
    if n_pruned > 0:
        ax.annotate(
            "Pruned\nweights",
            xy=(0.005, counts[0] * 0.8),
            xytext=(0.15, counts[0] * 0.9),
            fontsize=11,
            fontweight="bold",
            color=ACCENT_2,
            arrowprops=dict(arrowstyle="->", color=ACCENT_2, lw=2),
            zorder=10,
        )

    # Arrow pointing to active cluster (if present)
    active_gates = gate_values[gate_values >= 0.01]
    if len(active_gates) > 0:
        active_median = np.median(active_gates)
        # Find the bin that contains the active median
        active_bin_idx = np.digitize(active_median, bins) - 1
        active_bin_idx = min(active_bin_idx, len(counts) - 1)
        if counts[active_bin_idx] > 0:
            ax.annotate(
                "Active\nweights",
                xy=(active_median, counts[active_bin_idx] * 0.8),
                xytext=(active_median - 0.15, counts[active_bin_idx] * 1.2),
                fontsize=11,
                fontweight="bold",
                color=ACCENT_1,
                arrowprops=dict(arrowstyle="->", color=ACCENT_1, lw=2),
                zorder=10,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✅ Gate distribution plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gate value distribution from a trained self-pruning model."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--lambda_val", type=float, default=0.0001,
        help="Lambda value to load (used to find model if --model_path not given)",
    )
    parser.add_argument(
        "--output", type=str, default="gate_distribution.png",
        help="Output filename for the plot",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(f"outputs/models/model_lambda_{args.lambda_val}.pt")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run training first: python run_experiments.py")
        return

    print(f"📦 Loading model from: {model_path}")
    model, ckpt = load_model(model_path)

    lambda_val = ckpt.get("lambda_value", args.lambda_val)
    print(f"   Lambda: {lambda_val}")

    # Extract gate values from PrunableLinear layers
    gate_values = extract_all_gate_values(model)
    print(f"   Extracted {len(gate_values):,} gate values from PrunableLinear layers")

    # Compute strict sparsity metric
    sparsity_pct = compute_sparsity_level(model)
    print(f"   Sparsity Level: {sparsity_pct:.2f}%")

    # Plot
    output_path = Path(args.output)
    plot_gate_histogram(gate_values, lambda_val, sparsity_pct, output_path)


if __name__ == "__main__":
    main()
