#!/usr/bin/env python3
"""
run_experiments.py — CLI entry point for the Self-Pruning Neural Network System.

Usage:
  python run_experiments.py                          # Full pipeline (train + analyze + visualize)
  python run_experiments.py --api                    # Start FastAPI server
  python run_experiments.py --query "Which λ best?"  # RAG query on past experiments
  python run_experiments.py --lambdas 0.0001 0.01    # Custom λ values
  python run_experiments.py --epochs 50              # Custom epoch count
"""

from __future__ import annotations

import argparse
import sys
import logging

logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                          # Full pipeline
  python run_experiments.py --api                    # Start API server
  python run_experiments.py --query "Best lambda?"   # Query RAG system
  python run_experiments.py --lambdas 0.001 0.01     # Custom lambdas
  python run_experiments.py --epochs 50 --batch 256  # Custom training params
        """,
    )

    parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI server instead of running experiments",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query the RAG system about past experiments",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=None,
        help="Lambda values to experiment with (default: 0.0001 0.001 0.01)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--schedule", type=str, default=None,
        choices=["constant", "linear_warmup", "cosine"],
        help="Lambda schedule strategy (default: linear_warmup)",
    )
    parser.add_argument(
        "--no-anneal", action="store_true",
        help="Disable temperature annealing",
    )

    return parser.parse_args()


def run_api_server() -> None:
    """Start the FastAPI server."""
    import uvicorn
    from config import get_config

    cfg = get_config()
    logger.info(f"🚀 Starting API server on {cfg.api.host}:{cfg.api.port}")
    uvicorn.run(
        "api:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        log_level="info",
    )


def run_query(question: str) -> None:
    """Run a RAG query on past experiments."""
    from config import get_config
    from rag import ExperimentStore
    import json

    cfg = get_config()
    store = ExperimentStore(
        embedding_model_name=cfg.rag.embedding_model,
        embedding_dim=cfg.rag.embedding_dim,
        persist_dir=cfg.paths.rag_dir,
    )

    # Try to load persisted data
    try:
        store.load(cfg.paths.rag_dir)
    except Exception:
        pass

    # Also try loading from JSON logs
    if not store.documents:
        logs_dir = cfg.paths.logs_dir
        if logs_dir.exists():
            for log_file in sorted(logs_dir.glob("experiment_lambda_*.json")):
                with open(log_file) as f:
                    result = json.load(f)
                store.add_experiment(result)

    if not store.documents:
        print("❌ No experiments found. Run the full pipeline first:")
        print("   python run_experiments.py")
        return

    print(f"\n{'═'*60}")
    print(f"  RAG QUERY")
    print(f"{'═'*60}\n")

    answer = store.answer(question)
    print(answer)
    print()


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the complete experiment pipeline via the agent."""
    from config import get_config
    from agent import PruningAgent

    cfg = get_config()

    # Apply CLI overrides
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch:
        cfg.train.batch_size = args.batch
    if args.lr:
        cfg.train.lr = args.lr
    if args.schedule:
        cfg.train.lambda_schedule = args.schedule
    if args.no_anneal:
        cfg.train.gate_temp_anneal = False

    lambda_values = args.lambdas or cfg.train.lambda_values

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Self-Pruning Neural Network System                    ║")
    print("║   Production-Grade AI Engineering Pipeline              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   Device:    {str(cfg.train.device):>42} ║")
    print(f"║   Epochs:    {cfg.train.epochs:>42} ║")
    print(f"║   Batch:     {cfg.train.batch_size:>42} ║")
    print(f"║   LR:        {cfg.train.lr:>42} ║")
    print(f"║   Schedule:  {cfg.train.lambda_schedule:>42} ║")
    print(f"║   Lambdas:   {str(lambda_values):>42} ║")
    print(f"║   Temp:      {cfg.train.gate_temp_initial} → {cfg.train.gate_temp_final!s:>35} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    agent = PruningAgent(config=cfg)
    pipeline_result = agent.run_full_pipeline(lambda_values=lambda_values)

    # Print final summary
    print("\n" + "═"*60)
    print("  PIPELINE COMPLETE")
    print("═"*60)
    print(f"\n  Total time: {pipeline_result['total_time']:.1f}s")
    print(f"  Report: {pipeline_result['report_path']}")
    print(f"  Plots: {len(pipeline_result['plot_paths'])} generated")
    for p in pipeline_result["plot_paths"]:
        print(f"    → {p}")

    # Demo RAG queries
    print("\n" + "─"*60)
    print("  RAG QUERY DEMO")
    print("─"*60 + "\n")

    demo_queries = [
        "Which lambda gives best sparsity vs accuracy tradeoff?",
        "Show best model configuration",
        "What happens to accuracy at high sparsity?",
    ]

    for q in demo_queries:
        print(f"Q: {q}")
        answer = agent.answer_query(q)
        print(f"A: {answer}\n")


def main() -> None:
    args = parse_args()

    if args.api:
        run_api_server()
    elif args.query:
        run_query(args.query)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
