"""
visualize.py — Visualization engine for pruning experiment results.

Generates publication-quality plots:
  1. Gate distribution histograms (per-λ, showing spike near 0)
  2. Accuracy vs Sparsity trade-off plot
  3. Training curves (loss + accuracy over epochs)
  4. λ-effective schedule visualization
  5. Results summary table (printed + saved)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger("visualize")

# ─── Style ───────────────────────────────────────────────────────────────────────

COLORS = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6"]
BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure | None = None) -> None:
    """Apply consistent dark theme styling."""
    ax.set_facecolor(BG_COLOR)
    if fig:
        fig.patch.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


# ─── 1. Gate Distribution Histograms ────────────────────────────────────────────

def plot_gate_distributions(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Plot gate activation histograms for each λ value.
    Shows the characteristic spike near 0 for higher λ.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.patch.set_facecolor(BG_COLOR)

    if n == 1:
        axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        _apply_dark_style(ax, fig if idx == 0 else None)

        gate_values = result.get("gate_values", {})
        all_gates = []
        for layer_name, values in gate_values.items():
            all_gates.extend(values)

        if all_gates:
            all_gates = np.array(all_gates)
            ax.hist(
                all_gates,
                bins=50,
                color=COLORS[idx % len(COLORS)],
                alpha=0.85,
                edgecolor="none",
                density=True,
            )

            # Mark threshold
            ax.axvline(x=0.01, color="#ef4444", linestyle="--", alpha=0.7, label="Prune threshold")

            # Stats annotation
            pruned_pct = (all_gates < 0.01).mean() * 100
            ax.text(
                0.95, 0.95,
                f"Pruned: {pruned_pct:.1f}%",
                transform=ax.transAxes,
                ha="right", va="top",
                color=TEXT_COLOR,
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.8),
            )

        lam = result.get("lambda_value", "?")
        ax.set_title(f"λ = {lam}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Gate Activation Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper center", fontsize=9, framealpha=0.5)

    fig.suptitle(
        "Gate Activation Distributions",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "gate_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved gate distributions → {path}")
    return path


# ─── 2. Accuracy vs Sparsity ────────────────────────────────────────────────────

def plot_accuracy_vs_sparsity(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot accuracy vs sparsity trade-off across λ values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_style(ax, fig)

    lambdas = [r["lambda_value"] for r in results]
    accuracies = [r["final_accuracy"] * 100 for r in results]
    sparsities = [r["final_sparsity"] * 100 for r in results]

    # Calculate optimal point (composite scoring heuristic)
    scores = [(acc/100) - 0.5*(1 - (sp/100)) for acc, sp in zip(accuracies, sparsities)]
    best_idx = np.argmax(scores) if scores else 0

    # Draw Pareto Frontier mathematically
    points = sorted(zip(sparsities, accuracies, lambdas), key=lambda x: x[0], reverse=True)
    pareto_sp = []
    pareto_acc = []
    max_acc = -1
    for sp, acc, lam in points:
        if acc > max_acc:
            pareto_sp.append(sp)
            pareto_acc.append(acc)
            max_acc = acc
    
    # Plot Pareto Frontier as a dashed boundary line
    ax.plot(pareto_sp, pareto_acc, "--", color="#10b981", linewidth=2.5, zorder=3, alpha=0.8, label="Pareto Frontier (Boundary)")

    # Scatter with connecting line
    ax.plot(sparsities, accuracies, "-o", color=COLORS[0], linewidth=2.5,
            markersize=10, markeredgecolor="white", markeredgewidth=1.5, zorder=5, label="Experimental Configurations")
            
    # Highlight optimal lambda
    ax.scatter([sparsities[best_idx]], [accuracies[best_idx]], color="#fbbf24", marker="*", 
               s=400, edgecolors="white", linewidths=1.5, zorder=6, label="Optimal Trade-off")

    # Annotate each point with λ
    for i, lam in enumerate(lambdas):
        ax.annotate(
            f"λ={lam}",
            (sparsities[i], accuracies[i]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            color=TEXT_COLOR,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Sparsity (%)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Sparsity Trade-off", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")

    # Format axes
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    path = output_dir / "accuracy_vs_sparsity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved accuracy-vs-sparsity → {path}")
    return path


# ─── 3. Training Curves ─────────────────────────────────────────────────────────

def plot_training_curves(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot training loss and test accuracy curves for all λ values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_style(ax1, fig)
    _apply_dark_style(ax2)

    for idx, result in enumerate(results):
        lam = result["lambda_value"]
        history = result.get("epoch_history", [])
        if not history:
            continue

        epochs = [h["epoch"] + 1 for h in history]
        losses = [h["train_loss"] for h in history]
        test_accs = [h["test_accuracy"] * 100 for h in history]

        color = COLORS[idx % len(COLORS)]
        ax1.plot(epochs, losses, color=color, linewidth=2, label=f"λ={lam}", alpha=0.9)
        ax2.plot(epochs, test_accs, color=color, linewidth=2, label=f"λ={lam}", alpha=0.9)

    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.legend(fontsize=10, framealpha=0.5)

    ax2.set_title("Test Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(fontsize=10, framealpha=0.5)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved training curves → {path}")
    return path


# ─── 4. Lambda Schedule Visualization ───────────────────────────────────────────

def plot_lambda_schedule(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot effective λ and temperature over training epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(ax1, fig)
    _apply_dark_style(ax2)

    for idx, result in enumerate(results):
        lam = result["lambda_value"]
        history = result.get("epoch_history", [])
        if not history:
            continue

        epochs = [h["epoch"] + 1 for h in history]
        eff_lambdas = [h["effective_lambda"] for h in history]
        temps = [h["temperature"] for h in history]

        color = COLORS[idx % len(COLORS)]
        ax1.plot(epochs, eff_lambdas, color=color, linewidth=2, label=f"λ={lam}")
        ax2.plot(epochs, temps, color=color, linewidth=2, label=f"λ={lam}")

    ax1.set_title("Effective λ Schedule", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Effective λ")
    ax1.legend(fontsize=10, framealpha=0.5)

    ax2.set_title("Temperature Annealing", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Temperature (τ)")
    ax2.legend(fontsize=10, framealpha=0.5)

    plt.tight_layout()
    path = output_dir / "lambda_schedule.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved λ schedule → {path}")
    return path


# ─── 5. Sparsity Over Training ──────────────────────────────────────────────────

def plot_sparsity_over_training(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot sparsity % evolution over epochs for each λ."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_style(ax, fig)

    for idx, result in enumerate(results):
        lam = result["lambda_value"]
        history = result.get("epoch_history", [])
        if not history:
            continue

        epochs = [h["epoch"] + 1 for h in history]
        sparsities = [h["sparsity"] * 100 for h in history]

        color = COLORS[idx % len(COLORS)]
        ax.plot(epochs, sparsities, color=color, linewidth=2.5, label=f"λ={lam}")

    ax.set_title("Sparsity Growth Over Training", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity (%)")
    ax.legend(fontsize=11, framealpha=0.5)

    plt.tight_layout()
    path = output_dir / "sparsity_over_training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved sparsity-over-training → {path}")
    return path


# ─── 6. FLOPs Reduction Bar Chart ───────────────────────────────────────────────

def plot_flops_reduction(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot bar chart showing FLOPs reduction for each lambda."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(ax, fig)

    lambdas = []
    original_flops = []
    pruned_flops = []
    for r in results:
        lambdas.append(f"λ={r['lambda_value']}")
        flops_data = r.get("flops_reduction", {})
        original_flops.append(flops_data.get("total_original_flops", 1) / 1e6) # MFLOPs
        pruned_flops.append(flops_data.get("total_pruned_flops", 1) / 1e6)

    x = np.arange(len(lambdas))
    width = 0.35

    ax.bar(x - width/2, original_flops, width, label='Original MFLOPs', color=GRID_COLOR, alpha=0.6)
    bars = ax.bar(x + width/2, pruned_flops, width, label='Pruned MFLOPs', color=COLORS[2])

    # Annotate reduction %
    for i, bar in enumerate(bars):
        red_pct = (1 - pruned_flops[i] / max(original_flops[i], 1e-6)) * 100
        ax.annotate(f'-{red_pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color="#10b981", fontweight="bold", fontsize=11)

    ax.set_ylabel('Compute (MFLOPs)', fontsize=13)
    ax.set_title('FLOPs Reduction via Pruning', fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lambdas, fontsize=12)
    ax.legend()

    plt.tight_layout()
    path = output_dir / "flops_reduction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved FLOPs reduction → {path}")
    return path


# ─── Print Summary Table ────────────────────────────────────────────────────────

def print_results_table(results: list[dict[str, Any]]) -> str:
    """Print and return a formatted results table."""
    header = f"{'Lambda':>10} │ {'Accuracy':>10} │ {'Sparsity':>10} │ {'FLOPs Red.':>10} │ {'Dense ms':>10} │ {'Pruned ms':>10}"
    sep = "─" * len(header)

    lines = [sep, header, sep]
    for r in sorted(results, key=lambda x: x["lambda_value"]):
        flops = r.get("flops_reduction", {}).get("total_reduction_pct", 0.0)
        d_ms = r.get("inference_ms_baseline", 0.0)
        p_ms = r.get("inference_ms_pruned", 0.0)
        lines.append(
            f"{r['lambda_value']:>10} │ "
            f"{r['final_accuracy']:>9.2%} │ "
            f"{r['final_sparsity']:>9.2%} │ "
            f"{flops:>9.1f}% │ "
            f"{d_ms:>8.2f}ms │ "
            f"{p_ms:>8.2f}ms"
        )
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table


# ─── 7. Model Compression Visualization ──────────────────────────────────────────

def plot_model_compression(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Visualize Dense -> Soft-Pruned -> Hard-Pruned physical resource impacts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_style(ax, fig)

    if not results:
        return Path()

    # Find the best result by composite score
    best_idx = 0
    best_score = -999.0
    for i, r in enumerate(results):
        score = r["final_accuracy"] - 0.5 * (1 - r["final_sparsity"])
        if score > best_score:
            best_score = score
            best_idx = i
            
    r = results[best_idx]
    flop_data = r.get("flops_reduction", {})
    orig_total = sum(l["original_flops"] for l in flop_data.get("layers", []))
    pruned_total = sum(l["pruned_flops"] for l in flop_data.get("layers", []))
    
    if orig_total == 0: orig_total = 1

    stages = ["Dense\n(Training Base)", "Soft-Pruned\n(Zeroed Mats)", "Hard-Pruned\n(Physical Export)"]
    
    # Dense is 100%. Soft pruned takes same FLOP matrix (100%). Hard pruned takes pruned %.
    flop_percentages = [100.0, 100.0, (pruned_total / orig_total) * 100.0]
    
    bars = ax.bar(stages, flop_percentages, color=[COLORS[0], "#f59e0b", "#10b981"], width=0.6, edgecolor=TEXT_COLOR)
    
    ax.set_title(f"Model Compression Pipeline (λ={r['lambda_value']})", fontsize=15, fontweight="bold", pad=20)
    ax.set_ylabel("Hardware FLOP Utilization Limit (%)", fontsize=11)
    ax.set_ylim(0, 115)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 3,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        if i == 1:
            ax.text(bar.get_x() + bar.get_width() / 2, height - (height*0.1), "No physical\nacceleration", ha="center", va="top", color="#0f172a", fontsize=9, fontweight="bold")
        if i == 2:
            ax.text(bar.get_x() + bar.get_width() / 2, height - 10 if height > 20 else height + 15, "Device\nReady!", ha="center", va="top" if height > 20 else "bottom", color="#0f172a" if height > 20 else TEXT_COLOR, fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "compression_pipeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved model compression → {path}")
    return path


# ─── Generate All Plots ─────────────────────────────────────────────────────────

def generate_all_visualizations(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate all visualization plots and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(plot_gate_distributions(results, output_dir))
    paths.append(plot_accuracy_vs_sparsity(results, output_dir))
    paths.append(plot_training_curves(results, output_dir))
    paths.append(plot_flops_reduction(results, output_dir))
    paths.append(plot_lambda_schedule(results, output_dir))
    paths.append(plot_sparsity_over_training(results, output_dir))
    paths.append(plot_model_compression(results, output_dir))

    print_results_table([vars(r) if hasattr(r, '__dict__') and not isinstance(r, dict) else r for r in results])

    logger.info(f"Generated {len(paths)} visualizations in {output_dir}")
    return paths
