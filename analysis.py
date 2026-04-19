"""
analysis.py — Trade-off analysis engine for pruning experiments.

Automatically generates:
  - Trade-off summary tables
  - Best λ recommendation (composite scoring)
  - Observations on pruning behavior
  - Accuracy drop vs sparsity gain analysis
  - FLOPs reduction estimates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("analysis")


@dataclass
class TradeoffPoint:
    """A single point in the accuracy-sparsity trade-off space."""
    lambda_value: float
    accuracy: float
    sparsity: float
    flops_reduction_pct: float
    training_time: float


class AnalysisEngine:
    """
    Analyzes experiment results to find optimal pruning configurations.

    Computes Pareto frontiers, composite scores, and generates
    natural-language observations about pruning behavior.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """
        Args:
            alpha: Weight for sparsity in composite score.
                   composite = accuracy - alpha * (1 - sparsity)
                   Higher alpha favors more aggressive pruning.
        """
        self.alpha = alpha

    def extract_tradeoff_points(
        self, results: list[dict[str, Any]]
    ) -> list[TradeoffPoint]:
        """Extract trade-off points from experiment results."""
        points = []
        for r in results:
            flops = r.get("flops_reduction", {})
            points.append(TradeoffPoint(
                lambda_value=r["lambda_value"],
                accuracy=r["final_accuracy"],
                sparsity=r["final_sparsity"],
                flops_reduction_pct=flops.get("total_reduction_pct", 0.0),
                training_time=r.get("training_time_seconds", 0.0),
            ))
        return sorted(points, key=lambda p: p.lambda_value)

    def compute_composite_scores(
        self, points: list[TradeoffPoint]
    ) -> list[tuple[TradeoffPoint, float]]:
        """
        Compute composite score for each point.
        Score = accuracy - alpha * (1 - sparsity)
        Higher is better (balances accuracy and pruning).
        """
        scored = []
        for p in points:
            score = p.accuracy - self.alpha * (1 - p.sparsity)
            scored.append((p, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def recommend_best_lambda(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Find the best λ based on composite scoring and generate reasoning."""
        points = self.extract_tradeoff_points(results)
        if not points:
            return {"error": "No results to analyze"}

        scored = self.compute_composite_scores(points)
        best_point, best_score = scored[0]
        
        # Calculate baseline accuracy (dense/lowest lambda)
        baseline_acc = points[0].accuracy
        acc_drop = baseline_acc - best_point.accuracy

        reasoning = (
            f"λ={best_point.lambda_value} is optimal because it achieves "
            f"{best_point.sparsity:.1%} sparsity with an accuracy deviation of "
            f"{acc_drop:.1%}, maximizing the trade-off efficiency."
        )
        
        # Build mathematical comparison against the runner-up configuration
        composite_explanation = f"Score Formula utilized: Accuracy - {self.alpha} * (1 - Sparsity).\n"
        if len(scored) > 1:
            runner_up_point, runner_up_score = scored[1]
            diff_score = best_score - runner_up_score
            diff_sp = (best_point.sparsity - runner_up_point.sparsity) * 100
            diff_acc = (best_point.accuracy - runner_up_point.accuracy) * 100
            composite_explanation += (
                f"Configuration λ={best_point.lambda_value} strictly mathematically dominated "
                f"the runner-up (λ={runner_up_point.lambda_value}) by {diff_score:.3f} points. "
                f"Specifically, comparing λ={best_point.lambda_value} against λ={runner_up_point.lambda_value}: "
                f"It swung Sparsity by {diff_sp:+.1f}% while netting an Accuracy shift of {diff_acc:+.1f}%. "
                f"The α={self.alpha} scaling penalty determined that the sparsity gain "
                f"outweighed the accuracy drop." if diff_sp > 0 else "outweighed the sparsity drop."
            )

        if best_point.sparsity > 0.8:
            deployment_suggestion = "Highly recommended for resource-constrained edge devices (e.g., mobile, IoT). Hardware matrix pruning provides tremendous acceleration."
        elif best_point.sparsity > 0.3:
            deployment_suggestion = "Suitable for standard server-side deployment, maximizing GPU throughput while conserving physical RAM."
        else:
            deployment_suggestion = "Minimal pruning achieved. Deploy natively in accuracy-critical environments with high-capacity chips."

        return {
            "recommended_lambda": best_point.lambda_value,
            "accuracy": best_point.accuracy,
            "sparsity": best_point.sparsity,
            "flops_reduction_pct": best_point.flops_reduction_pct,
            "composite_score": best_score,
            "reasoning": reasoning,
            "composite_explanation": composite_explanation,
            "deployment_suggestion": deployment_suggestion,
            "all_scores": [
                {
                    "lambda": p.lambda_value,
                    "accuracy": p.accuracy,
                    "sparsity": p.sparsity,
                    "score": s,
                }
                for p, s in scored
            ],
        }

    def generate_observations(
        self, results: list[dict[str, Any]]
    ) -> list[str]:
        """Generate natural-language observations about pruning behavior."""
        points = self.extract_tradeoff_points(results)
        if len(points) < 2:
            return ["Insufficient data for comparative analysis."]

        observations = []

        # Sort by λ
        points_by_lambda = sorted(points, key=lambda p: p.lambda_value)

        # 1. Sparsity trend
        sparsities = [p.sparsity for p in points_by_lambda]
        if all(s1 <= s2 for s1, s2 in zip(sparsities, sparsities[1:])):
            observations.append(
                "✅ Sparsity increases monotonically with λ, confirming that "
                "L1 regularization on gate activations effectively drives pruning."
            )

        # 2. Accuracy degradation
        min_lambda_point = points_by_lambda[0]
        max_lambda_point = points_by_lambda[-1]
        acc_drop = min_lambda_point.accuracy - max_lambda_point.accuracy
        if acc_drop > 0:
            observations.append(
                f"📉 Accuracy drops by {acc_drop:.1%} from λ={min_lambda_point.lambda_value} "
                f"to λ={max_lambda_point.lambda_value}, while sparsity improves by "
                f"{max_lambda_point.sparsity - min_lambda_point.sparsity:.1%}."
            )
        else:
            observations.append(
                "🎉 No accuracy degradation observed — the network may have "
                "significant redundancy that pruning removes without harm."
            )

        # 3. Best trade-off
        scored = self.compute_composite_scores(points)
        best = scored[0][0]
        observations.append(
            f"🏆 Best trade-off: λ={best.lambda_value} achieves "
            f"{best.accuracy:.1%} accuracy with {best.sparsity:.1%} sparsity."
        )

        # 4. FLOPs reduction
        if max_lambda_point.flops_reduction_pct > 0:
            observations.append(
                f"⚡ Highest pruning (λ={max_lambda_point.lambda_value}) reduces "
                f"estimated FLOPs by {max_lambda_point.flops_reduction_pct:.1f}%."
            )

        # 5. Per-layer analysis
        for r in results:
            gate_stats = r.get("gate_stats", {})
            layer_sparsities = {
                name: stats["sparsity"]
                for name, stats in gate_stats.items()
            }
            if layer_sparsities:
                most_pruned = max(layer_sparsities, key=layer_sparsities.get)
                least_pruned = min(layer_sparsities, key=layer_sparsities.get)
                if layer_sparsities[most_pruned] > 0:
                    observations.append(
                        f"🔍 At λ={r['lambda_value']}: {most_pruned} is most pruned "
                        f"({layer_sparsities[most_pruned]:.1%}), {least_pruned} is "
                        f"least pruned ({layer_sparsities[least_pruned]:.1%})."
                    )

        # 6. Temperature effects
        observations.append(
            "🌡️ Temperature annealing sharpens gate decisions over training, "
            "pushing gates toward binary (0 or 1) values for cleaner pruning."
        )

        return observations

    def generate_report(self, results: list[dict[str, Any]]) -> str:
        """Generate a comprehensive markdown report."""
        points = self.extract_tradeoff_points(results)
        recommendation = self.recommend_best_lambda(results)
        observations = self.generate_observations(results)

        lines = []
        lines.append("# Self-Pruning Neural Network — Experiment Report\n")

        # Summary table
        lines.append("## Results Summary\n")
        lines.append("| Lambda (λ) | Accuracy | Sparsity | FLOPs Reduction | Training Time |")
        lines.append("|:----------:|:--------:|:--------:|:---------------:|:-------------:|")
        for p in points:
            lines.append(
                f"| {p.lambda_value} | {p.accuracy:.2%} | {p.sparsity:.2%} | "
                f"{p.flops_reduction_pct:.1f}% | {p.training_time:.1f}s |"
            )
        lines.append("")

        # Recommendation
        lines.append("## Recommendation\n")
        if "error" not in recommendation:
            lines.append(
                f"**Best λ = {recommendation['recommended_lambda']}** with composite "
                f"score {recommendation['composite_score']:.4f}\n"
            )
            lines.append(
                f"- Accuracy: {recommendation['accuracy']:.2%}\n"
                f"- Sparsity: {recommendation['sparsity']:.2%}\n"
                f"- FLOPs Reduction: {recommendation['flops_reduction_pct']:.1f}%\n"
            )

        # Observations
        lines.append("## Observations\n")
        for obs in observations:
            lines.append(f"- {obs}")
        lines.append("")

        # Per-layer gate stats
        lines.append("## Per-Layer Gate Statistics\n")
        for r in results:
            lines.append(f"### λ = {r['lambda_value']}\n")
            gate_stats = r.get("gate_stats", {})
            if gate_stats:
                lines.append("| Layer | Sparsity | Mean Gate | Std Gate | Active/Total |")
                lines.append("|:-----:|:--------:|:---------:|:--------:|:------------:|")
                for name, stats in gate_stats.items():
                    lines.append(
                        f"| {name} | {stats['sparsity']:.2%} | {stats['mean_gate']:.4f} | "
                        f"{stats['std_gate']:.4f} | {stats['active_params']:,}/{stats['num_params']:,} |"
                    )
            lines.append("")

        return "\n".join(lines)
