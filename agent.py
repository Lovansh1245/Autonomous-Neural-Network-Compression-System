"""
agent.py — AI Agent workflow for automated pruning experiments.

Implements an agent pipeline that:
  1. Runs experiments across multiple λ values
  2. Stores results in the RAG system
  3. Analyzes trade-offs via the analysis engine
  4. Recommends the best model configuration
  5. Answers natural-language queries about past experiments

Pipeline: User Query → Experiment Execution → Storage → Retrieval → Analysis → Response
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import SystemConfig, get_config
from train import train_model, ExperimentResult
from analysis import AnalysisEngine
from rag import ExperimentStore
from visualize import generate_all_visualizations, print_results_table

logger = logging.getLogger("agent")


@dataclass
class AgentState:
    """Tracks the agent's state across the pipeline."""
    experiments_run: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    recommendation: Optional[dict[str, Any]] = None
    report: Optional[str] = None
    total_time: float = 0.0


class PruningAgent:
    """
    An AI agent that orchestrates the pruning experiment pipeline.

    Responsibilities:
      - Run experiments with different λ values
      - Store results in the RAG system
      - Analyze trade-offs and generate insights
      - Recommend the best model configuration
      - Answer queries about experiments
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self.analysis_engine = AnalysisEngine()
        self.rag_store = ExperimentStore(
            embedding_model_name=self.config.rag.embedding_model,
            embedding_dim=self.config.rag.embedding_dim,
            persist_dir=self.config.paths.rag_dir,
        )
        self.state = AgentState()

        logger.info("🤖 PruningAgent initialized")

    # ── Step 1: Run experiments ──────────────────────────────────────────────

    def run_experiments(
        self,
        lambda_values: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """
        Run training experiments for each λ value.

        Args:
            lambda_values: List of λ values to try. Defaults to config values.

        Returns:
            List of experiment result dicts.
        """
        lambdas = lambda_values or self.config.train.lambda_values
        logger.info(f"🧪 Running {len(lambdas)} experiments: λ ∈ {lambdas}")

        results = []
        for i, lam in enumerate(lambdas, 1):
            logger.info(f"\n{'━'*60}")
            logger.info(f"  Experiment {i}/{len(lambdas)} — λ = {lam}")
            logger.info(f"{'━'*60}")

            result = train_model(
                config=self.config.train,
                paths=self.config.paths,
                lambda_val=lam,
            )
            result_dict = vars(result)
            results.append(result_dict)

            self.state.experiments_run += 1
            logger.info(
                f"  ✅ λ={lam}: accuracy={result.final_accuracy:.2%}, "
                f"sparsity={result.final_sparsity:.2%}"
            )

        self.state.results = results
        return results

    # ── Step 2: Store results in RAG ─────────────────────────────────────────

    def store_results(self, results: Optional[list[dict[str, Any]]] = None) -> None:
        """Store experiment results in the RAG system for later retrieval."""
        results = results or self.state.results
        if not results:
            logger.warning("No results to store")
            return

        logger.info(f"📚 Storing {len(results)} experiments in RAG system...")
        for r in results:
            self.rag_store.add_experiment(r)

        # Persist to disk
        self.rag_store.persist(self.config.paths.rag_dir)
        logger.info("  ✅ Results stored and persisted")

    # ── Step 3: Analyze results ──────────────────────────────────────────────

    def analyze(self, results: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Analyze experiment results and generate recommendation + report."""
        results = results or self.state.results
        if not results:
            logger.warning("No results to analyze")
            return {"error": "No results"}

        logger.info("🔬 Analyzing trade-offs...")

        recommendation = self.analysis_engine.recommend_best_lambda(results)
        observations = self.analysis_engine.generate_observations(results)
        report = self.analysis_engine.generate_report(results)

        self.state.recommendation = recommendation
        self.state.report = report

        logger.info(f"  🏆 Best λ = {recommendation.get('recommended_lambda')}")
        logger.info(f"  📊 Generated {len(observations)} observations")

        return {
            "recommendation": recommendation,
            "observations": observations,
            "report": report,
        }

    # ── Step 4: Visualize ────────────────────────────────────────────────────

    def visualize(self, results: Optional[list[dict[str, Any]]] = None) -> list[Path]:
        """Generate all visualization plots."""
        results = results or self.state.results
        if not results:
            logger.warning("No results to visualize")
            return []

        logger.info("📊 Generating visualizations...")
        paths = generate_all_visualizations(results, self.config.paths.plots_dir)
        logger.info(f"  ✅ Generated {len(paths)} plots")
        return paths

    # ── Full Pipeline ────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        lambda_values: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Execute the complete agent pipeline:
          1. Run experiments → 2. Store in RAG → 3. Analyze → 4. Visualize

        Returns:
            Complete pipeline results including recommendation and report.
        """
        pipeline_start = time.time()
        logger.info("🚀 Starting full agent pipeline")
        logger.info(f"{'═'*60}")

        # Step 1: Run experiments
        results = self.run_experiments(lambda_values)

        # Step 2: Store in RAG
        self.store_results(results)

        # Step 3: Analyze
        analysis = self.analyze(results)

        # Step 4: Visualize
        plot_paths = self.visualize(results)

        # Step 5: Print summary
        print("\n" + "═"*60)
        print("  EXPERIMENT RESULTS")
        print("═"*60 + "\n")
        print_results_table(results)
        print()

        if analysis.get("recommendation"):
            rec = analysis["recommendation"]
            print(f"🏆 Recommended λ = {rec.get('recommended_lambda')}")
            print(f"   Accuracy: {rec.get('accuracy', 0):.2%}")
            print(f"   Sparsity: {rec.get('sparsity', 0):.2%}")
            print(f"   FLOPs Reduction: {rec.get('flops_reduction_pct', 0):.1f}%")
            if "reasoning" in rec:
                print(f"   Reasoning: {rec['reasoning']}")
            if "deployment_suggestion" in rec:
                print(f"   Deployment: {rec['deployment_suggestion']}")
            print()

        if analysis.get("observations"):
            print("📝 Key Observations:")
            for obs in analysis["observations"]:
                print(f"   {obs}")
            print()

        # Save report
        report_path = self.config.paths.output_dir / "experiment_report.md"
        if analysis.get("report"):
            with open(report_path, "w") as f:
                f.write(analysis["report"])
            logger.info(f"Report saved to {report_path}")

        self.state.total_time = time.time() - pipeline_start
        logger.info(f"✅ Pipeline complete in {self.state.total_time:.1f}s")

        return {
            "results": results,
            "analysis": analysis,
            "plot_paths": [str(p) for p in plot_paths],
            "report_path": str(report_path),
            "total_time": self.state.total_time,
        }

    # ── Query Interface ──────────────────────────────────────────────────────

    def answer_query(self, question: str) -> str:
        """
        Answer a natural-language question about experiments using RAG.

        Args:
            question: User's question (e.g., "Which λ gives best accuracy?")

        Returns:
            Natural-language response based on retrieved experiments.
        """
        logger.info(f"❓ Query: '{question}'")
        response = self.rag_store.answer(question, top_k=self.config.rag.top_k)
        logger.info(f"  → Response generated ({len(response)} chars)")
        return response

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the agent's current state."""
        return {
            "experiments_run": self.state.experiments_run,
            "results_count": len(self.state.results),
            "has_recommendation": self.state.recommendation is not None,
            "has_report": self.state.report is not None,
            "total_time": self.state.total_time,
            "rag_documents": len(self.rag_store.documents),
        }
