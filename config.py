"""
config.py — Central configuration for the Self-Pruning Neural Network System.

All hyperparameters, paths, and system settings are defined here as dataclasses.
Config-driven design ensures reproducibility and easy experimentation.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

# ─── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("config")


# ─── Device Detection ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device: MPS (Apple Silicon) → CUDA → CPU."""
    if torch.backends.mps.is_available():
        logger.info("🍎 Using Apple MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("🟢 Using CUDA device")
        return torch.device("cuda")
    logger.info("💻 Using CPU device")
    return torch.device("cpu")


# ─── Training Configuration ─────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Hyperparameters for training the prunable network."""

    # Core training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    # Sparsity
    lambda_values: list[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])

    # Gate temperature scaling (advanced feature #1)
    # Temperature anneals from initial → final over training.
    # For short runs (< 30 epochs), keep anneal=False or use a mild final temp.
    # Aggressive annealing (τ→0.1) can cause gate collapse.
    gate_temp_initial: float = 1.0
    gate_temp_final: float = 0.5
    gate_temp_anneal: bool = True

    # Lambda scheduling (advanced feature #2)
    lambda_schedule: Literal["constant", "linear_warmup", "cosine"] = "linear_warmup"
    lambda_warmup_fraction: float = 0.3  # fraction of epochs for warmup

    # Scheduler
    scheduler_type: Literal["cosine", "step"] = "cosine"
    step_lr_step_size: int = 10
    step_lr_gamma: float = 0.5

    # Device
    device: torch.device = field(default_factory=get_device)

    def __post_init__(self) -> None:
        # Set MPS fallback env var for safety
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ─── Paths Configuration ────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    """All filesystem paths used by the system."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def rag_dir(self) -> Path:
        return self.output_dir / "rag"

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.models_dir, self.plots_dir, self.logs_dir, self.rag_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ─── RAG Configuration ──────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """Configuration for the RAG intelligence layer."""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # dimension of all-MiniLM-L6-v2
    top_k: int = 3


# ─── API Configuration ──────────────────────────────────────────────────────────

@dataclass
class APIConfig:
    """Configuration for the FastAPI service."""

    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Self-Pruning Neural Network API"
    version: str = "1.0.0"


# ─── Master Configuration ───────────────────────────────────────────────────────

@dataclass
class SystemConfig:
    """Master configuration aggregating all sub-configs."""

    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self) -> None:
        self.paths.ensure_dirs()
        logger.info(f"System config initialized — device: {self.train.device}")


# ─── Default global config instance ─────────────────────────────────────────────

def get_config() -> SystemConfig:
    """Get the default system configuration."""
    return SystemConfig()
