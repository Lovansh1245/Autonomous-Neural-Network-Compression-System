"""
train.py — Training loop and evaluation for the self-pruning neural network.

Handles:
  - CIFAR-10 data loading with augmentation
  - Training with combined CE + λ×L1 sparsity loss
  - Temperature annealing for gate sharpening
  - Gradual λ scheduling (linear warmup)
  - Per-epoch metric tracking
  - Model checkpointing + JSON log export
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from config import TrainConfig, PathConfig, get_device
from model import PrunableCNN
from layers import PrunableLinear

logger = logging.getLogger("train")


# ─── Data Loading ────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders with standard augmentation."""

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit from pin_memory
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader


# ─── Sparsity Level Metric ───────────────────────────────────────────────────────

def compute_sparsity_level(model: PrunableCNN) -> float:
    """
    Compute the Sparsity Level (%) metric.

    Strictly defined as: the percentage of all weights across ALL PrunableLinear
    layers where the final gate value (sigmoid of gate_scores / temperature)
    is < 1e-2.

    This is the canonical sparsity metric for the case study evaluation.
    """
    total_weights = 0
    pruned_weights = 0
    with torch.no_grad():
        for layer in model.prunable_layers:
            if isinstance(layer, PrunableLinear):
                gate_values = layer.gate_activations  # sigmoid(gate_scores / temp)
                total_weights += gate_values.numel()
                pruned_weights += (gate_values < 1e-2).sum().item()
    if total_weights == 0:
        return 0.0
    return (pruned_weights / total_weights) * 100.0


# ─── Lambda Scheduling ──────────────────────────────────────────────────────────

def get_effective_lambda(
    base_lambda: float,
    epoch: int,
    total_epochs: int,
    schedule: str = "linear_warmup",
    warmup_fraction: float = 0.3,
) -> float:
    """
    Compute effective λ based on scheduling strategy.

    - constant: λ_eff = base_lambda throughout
    - linear_warmup: λ ramps from 0 → base_lambda over warmup_fraction of epochs
    - cosine: λ follows cosine curve from 0 → base_lambda
    """
    if schedule == "constant":
        return base_lambda

    warmup_epochs = int(total_epochs * warmup_fraction)

    if schedule == "linear_warmup":
        if epoch < warmup_epochs:
            return base_lambda * (epoch / max(warmup_epochs, 1))
        return base_lambda

    if schedule == "cosine":
        if epoch < warmup_epochs:
            # Cosine warmup: 0 → base_lambda
            progress = epoch / max(warmup_epochs, 1)
            return base_lambda * 0.5 * (1 - np.cos(np.pi * progress))
        return base_lambda

    return base_lambda


# ─── Temperature Annealing ───────────────────────────────────────────────────────

def get_temperature(
    epoch: int,
    total_epochs: int,
    temp_initial: float = 1.0,
    temp_final: float = 0.1,
    anneal: bool = True,
) -> float:
    """Linearly anneal temperature from initial to final over training."""
    if not anneal:
        return temp_initial
    progress = epoch / max(total_epochs - 1, 1)
    return temp_initial + (temp_final - temp_initial) * progress


# ─── Training Metrics ────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    ce_loss: float
    sparsity_loss: float
    effective_lambda: float
    temperature: float
    train_accuracy: float
    test_accuracy: float
    sparsity: float
    active_params: int
    total_params: int
    elapsed_seconds: float


@dataclass
class ExperimentResult:
    """Complete result from a single training run."""
    lambda_value: float
    final_accuracy: float
    final_sparsity: float
    best_accuracy: float
    flops_reduction: dict[str, Any]
    gate_stats: dict[str, dict[str, float]]
    gate_values: dict[str, list[float]]
    epoch_history: list[dict[str, Any]]
    config: dict[str, Any]
    training_time_seconds: float
    inference_ms_baseline: float = 0.0
    inference_ms_pruned: float = 0.0


# ─── Training Loop ───────────────────────────────────────────────────────────────

def train_model(
    config: TrainConfig,
    paths: PathConfig,
    lambda_val: float,
) -> ExperimentResult:
    """
    Train a PrunableCNN on CIFAR-10 with the given λ sparsity penalty.

    Returns an ExperimentResult with full metrics and trained model state.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Training with λ={lambda_val} | schedule={config.lambda_schedule}")
    logger.info(f"Device: {config.device} | Epochs: {config.epochs} | Batch: {config.batch_size}")
    logger.info(f"Temperature annealing: {config.gate_temp_initial} → {config.gate_temp_final}")
    logger.info(f"{'='*60}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Model
    model = PrunableCNN(
        num_classes=10,
        temperature=config.gate_temp_initial,
    ).to(config.device)

    # Optimizer + Scheduler
    # Exclude gate_scores from weight decay so they are only penalized by L1
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate_scores" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=config.lr)
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_lr_step_size,
            gamma=config.step_lr_gamma,
        )

    ce_criterion = nn.CrossEntropyLoss()

    epoch_history: list[dict[str, Any]] = []
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # ── Update temperature ──
        temp = get_temperature(
            epoch, config.epochs,
            config.gate_temp_initial, config.gate_temp_final,
            config.gate_temp_anneal,
        )
        model.set_temperature(temp)

        # ── Compute effective λ ──
        eff_lambda = get_effective_lambda(
            lambda_val, epoch, config.epochs,
            config.lambda_schedule, config.lambda_warmup_fraction,
        )

        # ── Train one epoch ──
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sparse = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = ce_criterion(outputs, labels)
            sparsity_loss = model.get_gate_l1_loss()
            loss = ce_loss + eff_lambda * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_sparse += sparsity_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # ── Evaluate ──
        train_acc = correct / total
        test_acc = evaluate_accuracy(model, test_loader, config.device)
        sparsity = model.get_sparsity()
        sparsity_level_pct = compute_sparsity_level(model)
        best_accuracy = max(best_accuracy, test_acc)

        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_sparse = total_sparse / len(train_loader)
        elapsed = time.time() - epoch_start

        # Log
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            ce_loss=avg_ce,
            sparsity_loss=avg_sparse,
            effective_lambda=eff_lambda,
            temperature=temp,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            sparsity=sparsity,
            active_params=model.active_params_count,
            total_params=model.total_params,
            elapsed_seconds=elapsed,
        )
        epoch_history.append(vars(metrics))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{config.epochs} │ "
                f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f} + λ·S: {eff_lambda*avg_sparse:.4f}) │ "
                f"Train: {train_acc:.3f} │ Test: {test_acc:.3f} │ "
                f"Sparsity: {sparsity:.1%} │ SparsityLevel: {sparsity_level_pct:.1f}% │ Temp: {temp:.3f}"
            )

        # Periodic MPS memory cleanup
        if config.device.type == "mps" and (epoch + 1) % 10 == 0:
            torch.mps.empty_cache()

    total_time = time.time() - start_time
    logger.info(f"  Training complete in {total_time:.1f}s | Best accuracy: {best_accuracy:.3f}")

    # ── Collect final stats ──
    gate_stats = model.get_gate_stats()
    gate_values = {k: v.tolist() for k, v in model.get_all_gate_values().items()}
    flops = model.get_flops_reduction()

    # ── Compression & Export Phase ──
    logger.info("  Starting compression & benchmarking...")
    baseline_ms = model.measure_inference_ms(config.device)
    hard_model = model.export_hard_pruned(threshold=1e-2)
    pruned_ms = hard_model.measure_inference_ms(config.device)
    
    logger.info(f"  Inference Latency: {baseline_ms:.2f}ms (Dense) → {pruned_ms:.2f}ms (Pruned)")

    # ── Save models ──
    model_path = paths.models_dir / f"model_lambda_{lambda_val}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "lambda_value": lambda_val,
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "lambda_schedule": config.lambda_schedule,
            "gate_temp_initial": config.gate_temp_initial,
            "gate_temp_final": config.gate_temp_final,
        },
        "final_accuracy": evaluate_accuracy(model, test_loader, config.device),
        "final_sparsity": model.get_sparsity(),
        "sparsity_level_pct": compute_sparsity_level(model),
        "inference_ms_baseline": baseline_ms,
        "inference_ms_pruned": pruned_ms,
    }, model_path)
    
    hard_model_path = paths.models_dir / f"model_lambda_{lambda_val}_hard.pt"
    torch.save({
        "model_state_dict": hard_model.state_dict(),
        "lambda_value": lambda_val,
        "is_hard_pruned": True,
    }, hard_model_path)
    
    logger.info(f"  Standard model saved to {model_path.name}")
    logger.info(f"  Hard-pruned model saved to {hard_model_path.name}")

    result = ExperimentResult(
        lambda_value=lambda_val,
        final_accuracy=evaluate_accuracy(model, test_loader, config.device),
        final_sparsity=model.get_sparsity(),
        best_accuracy=best_accuracy,
        flops_reduction=flops,
        gate_stats=gate_stats,
        gate_values=gate_values,
        epoch_history=epoch_history,
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "lambda_schedule": config.lambda_schedule,
            "gate_temp_initial": config.gate_temp_initial,
            "gate_temp_final": config.gate_temp_final,
        },
        training_time_seconds=total_time,
        inference_ms_baseline=baseline_ms,
        inference_ms_pruned=pruned_ms,
    )

    # Save JSON log
    log_path = paths.logs_dir / f"experiment_lambda_{lambda_val}.json"
    with open(log_path, "w") as f:
        json.dump(vars(result), f, indent=2)
    logger.info(f"  Log saved to {log_path}")

    return result


# ─── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_accuracy(
    model: PrunableCNN,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a test dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total
