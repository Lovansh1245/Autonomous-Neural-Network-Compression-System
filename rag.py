"""
rag.py — RAG-style Model Intelligence Layer.

Provides similarity-based retrieval over experiment results using:
  - sentence-transformers for local embeddings (no API key needed)
  - FAISS for fast vector similarity search

Users can query the system with natural language:
  - "Which λ gives best accuracy?"
  - "Show results for high sparsity experiments"
  - "What is the tradeoff between accuracy and pruning?"

The system converts experiment results to textual documents, embeds them,
indexes in FAISS, and retrieves relevant context for answering queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("rag")


class ExperimentStore:
    """
    RAG-style retrieval system for experiment results.

    Stores experiment data as embedded text documents in a FAISS index.
    Supports natural-language queries with similarity-based retrieval.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir

        # Lazy-loaded to avoid import overhead at startup
        self._model = None
        self._index = None

        # Document store: maps index position → document text + metadata
        self.documents: list[dict[str, Any]] = []

        logger.info(f"ExperimentStore initialized — model: {embedding_model_name}")

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        return self._model

    @property
    def index(self):
        """Lazy-load or create the FAISS index."""
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after L2 norm)
            logger.info(f"Created FAISS index — dim={self.embedding_dim}")
        return self._index

    def _result_to_document(self, result: dict[str, Any]) -> str:
        """Convert an experiment result to a searchable text document."""
        flops = result.get("flops_reduction", {})
        lv = result["lambda_value"]
        fa = result["final_accuracy"]
        fs = result["final_sparsity"]
        ba = result.get("best_accuracy", fa)
        flops = result.get("flops_reduction", {}).get("total_reduction_pct", 0)
        baseline_ms = result.get("inference_ms_baseline", 0.0)
        pruned_ms = result.get("inference_ms_pruned", 0.0)
        t_time = result.get("training_time_seconds", 0)
        
        # Format the document text cleanly
        doc_parts = [
            f"EXPERIMENT SUMMARY | Lambda (λ) = {lv}",
            f"- Accuracy: {fa:.2%} (Best: {ba:.2%})",
            f"- Sparsity: {fs:.2%}",
            f"- FLOPs Reduction: {flops:.1f}%",
            f"- Inference Latency: {pruned_ms:.2f}ms per image (Base: {baseline_ms:.2f}ms)",
            f"- Training Time: {t_time:.1f}s",
        ]

        # Config details
        cfg = result.get("config", {})
        if cfg:
            c_ep = cfg.get("epochs", "?")
            c_ls = cfg.get("lambda_schedule", "?")
            c_ti = cfg.get("gate_temp_initial", "?")
            doc_parts.append(f"- Setup: {c_ep} epochs, '{c_ls}' schedule, {c_ti} base temp")

        # Trade-off characterization
        if result["final_sparsity"] > 0.5:
            doc_parts.append("- Impact: Extreme pruning. Highly memory efficient.")
        elif result["final_sparsity"] > 0.1:
            doc_parts.append("- Impact: Balanced pruning. Good throughput gains.")
        else:
            doc_parts.append("- Impact: Dense network. Retains full parametric capacity.")

        return "\n".join(doc_parts)

    def add_experiment(self, result: dict[str, Any]) -> int:
        """
        Add an experiment result to the store.

        Args:
            result: Experiment result dict (from ExperimentResult).

        Returns:
            Index of the added document.
        """
        doc_text = self._result_to_document(result)

        # Embed
        embedding = self.model.encode(
            [doc_text], normalize_embeddings=True, show_progress_bar=False
        )

        # Add to FAISS
        self.index.add(embedding.astype(np.float32))

        # Store document
        doc_idx = len(self.documents)
        self.documents.append({
            "text": doc_text,
            "result": result,
            "lambda_value": result["lambda_value"],
        })

        logger.info(
            f"Added experiment λ={result['lambda_value']} to store "
            f"(idx={doc_idx}, {len(doc_text)} chars)"
        )
        return doc_idx

    def query(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Query the experiment store with a natural-language question.

        Args:
            question: Natural language question about experiments.
            top_k: Number of most relevant results to return.

        Returns:
            List of relevant documents with scores.
        """
        if not self.documents:
            return []

        # Embed query
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )

        # Search
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "score": float(score),
                    "text": doc["text"],
                    "lambda_value": doc["lambda_value"],
                    "result": doc["result"],
                })

        logger.info(f"Query: '{question}' → {len(results)} results")
        return results

    def generate_response(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        """
        Generate a natural-language response based on retrieved documents.

        Uses template-based generation (no LLM API required).
        """
        if not retrieved_docs:
            return "No experiments found matching your query. Please run experiments first."

        response_parts = [f"**Query:** {question}\n"]
        response_parts.append(f"**Based on {len(retrieved_docs)} relevant experiment(s):**\n")

        for i, doc in enumerate(retrieved_docs, 1):
            r = doc["result"]
            r_lv = r["lambda_value"]
            r_fa = r["final_accuracy"]
            r_fs = r["final_sparsity"]
            r_fr = r.get("flops_reduction", {}).get("total_reduction_pct", 0)
            response_parts.append(
                f"{i}. **\u03bb={r_lv}**: "
                f"Accuracy={r_fa:.2%}, "
                f"Sparsity={r_fs:.2%}, "
                f"FLOPs reduction={r_fr:.1f}%"
            )

        # Derive answer based on question type
        question_lower = question.lower()
        best_result = retrieved_docs[0]["result"]

        if "best" in question_lower or "recommend" in question_lower:
            br_lv = best_result["lambda_value"]
            br_fa = best_result["final_accuracy"]
            br_fs = best_result["final_sparsity"]
            response_parts.append(
                f"\n**Recommendation:** \u03bb={br_lv} provides the "
                f"best match for your query with {br_fa:.2%} accuracy "
                f"and {br_fs:.2%} sparsity."
            )
        elif "tradeoff" in question_lower or "trade-off" in question_lower:
            all_results = [d["result"] for d in retrieved_docs]
            acc_range = max(r["final_accuracy"] for r in all_results) - min(r["final_accuracy"] for r in all_results)
            spar_range = max(r["final_sparsity"] for r in all_results) - min(r["final_sparsity"] for r in all_results)
            response_parts.append(
                f"\n**Trade-off Analysis:** Across these experiments, "
                f"accuracy varies by {acc_range:.1%} while sparsity varies by {spar_range:.1%}. "
                f"Higher λ values achieve more sparsity at the cost of some accuracy."
            )
        elif "sparsity" in question_lower:
            most_sparse = max(retrieved_docs, key=lambda d: d["result"]["final_sparsity"])
            ms_r = most_sparse["result"]
            response_parts.append(
                f"\n**Sparsity Focus:** The most pruned model uses "
                f"\u03bb={ms_r['lambda_value']} achieving "
                f"{ms_r['final_sparsity']:.2%} sparsity."
            )
        elif "accuracy" in question_lower:
            most_accurate = max(retrieved_docs, key=lambda d: d["result"]["final_accuracy"])
            ma_r = most_accurate["result"]
            response_parts.append(
                f"\n**Accuracy Focus:** The most accurate model uses "
                f"\u03bb={ma_r['lambda_value']} achieving "
                f"{ma_r['final_accuracy']:.2%} test accuracy."
            )
        else:
            br_lv2 = best_result["lambda_value"]
            br_fa2 = best_result["final_accuracy"]
            br_fs2 = best_result["final_sparsity"]
            response_parts.append(
                f"\n**Summary:** The most relevant experiment uses \u03bb={br_lv2} "
                f"with accuracy={br_fa2:.2%} and "
                f"sparsity={br_fs2:.2%}."
            )

        scores_str = ", ".join(format(d["score"], ".3f") for d in retrieved_docs)
        response_parts.append(
            f"\n*Relevance scores: {scores_str}*"
        )

        return "\n".join(response_parts)

    def answer(self, question: str, top_k: int = 3) -> str:
        """Full RAG pipeline: query → retrieve → respond."""
        docs = self.query(question, top_k=top_k)
        return self.generate_response(question, docs)

    def persist(self, path: Path) -> None:
        """Save the index and documents to disk."""
        import faiss
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f, indent=2)
        logger.info(f"Persisted store to {path}")

    def load(self, path: Path) -> None:
        """Load a persisted index and documents."""
        import faiss
        faiss_path = path / "index.faiss"
        docs_path = path / "documents.json"
        if faiss_path.exists() and docs_path.exists():
            self._index = faiss.read_index(str(faiss_path))
            with open(docs_path) as f:
                self.documents = json.load(f)
            logger.info(f"Loaded store from {path} — {len(self.documents)} documents")
