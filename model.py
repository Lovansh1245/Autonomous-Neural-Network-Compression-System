"""
model.py — PrunableCNN for CIFAR-10 with learnable gate-based self-pruning.

Architecture:
  3× Conv blocks (PrunableConv2d → BatchNorm → ReLU → MaxPool)
  2× FC layers (PrunableLinear)
  Channels: 64 → 128 → 256, FC: 256×4×4 → 512 → 10

Provides aggregate methods for sparsity tracking, gate L1 loss,
and FLOPs reduction estimation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from layers import PrunableConv2d, PrunableLinear

logger = logging.getLogger("model")


class PrunableCNN(nn.Module):
    """
    A convolutional neural network for CIFAR-10 with prunable layers.

    All Conv2d and Linear layers are replaced with their prunable variants,
    enabling end-to-end differentiable pruning via learnable gates.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10).
        temperature: Initial gate temperature for all prunable layers.
    """

    def __init__(self, num_classes: int = 10, temperature: float = 1.0) -> None:
        super().__init__()

        # ── Conv Block 1: 3 → 64 ──
        self.conv1 = PrunableConv2d(3, 64, 3, padding=1, temperature=temperature)
        self.bn1 = nn.BatchNorm2d(64)

        # ── Conv Block 2: 64 → 128 ──
        self.conv2 = PrunableConv2d(64, 128, 3, padding=1, temperature=temperature)
        self.bn2 = nn.BatchNorm2d(128)

        # ── Conv Block 3: 128 → 256 ──
        self.conv3 = PrunableConv2d(128, 256, 3, padding=1, temperature=temperature)
        self.bn3 = nn.BatchNorm2d(256)

        # ── Classifier ──
        self.fc1 = PrunableLinear(256 * 4 * 4, 512, temperature=temperature)
        self.fc2 = PrunableLinear(512, num_classes, temperature=temperature)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        logger.info(f"PrunableCNN initialized — {self.total_params:,} params, temp={temperature}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: (B, 3, 32, 32) → (B, 64, 16, 16)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # Block 2: (B, 64, 16, 16) → (B, 128, 8, 8)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Block 3: (B, 128, 8, 8) → (B, 256, 4, 4)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten + FC
        x = x.reshape(x.size(0), -1)  # (B, 256*4*4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    # ── Prunable layer access ──────────────────────────────────────────────────

    @property
    def prunable_layers(self) -> list[PrunableConv2d | PrunableLinear]:
        """All prunable layers in the model."""
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    @property
    def prunable_layer_names(self) -> list[str]:
        return ["conv1", "conv2", "conv3", "fc1", "fc2"]

    # ── Aggregate metrics ──────────────────────────────────────────────────────

    def get_gate_l1_loss(self) -> torch.Tensor:
        """Total L1 loss across all gate activations (for sparsity regularization)."""
        return sum(layer.gate_l1 for layer in self.prunable_layers)

    def get_sparsity(self) -> float:
        """Overall sparsity: fraction of all gates below threshold."""
        total_params = sum(layer.num_params for layer in self.prunable_layers)
        pruned_params = sum(
            layer.num_params - layer.active_params for layer in self.prunable_layers
        )
        return pruned_params / max(total_params, 1)

    @property
    def total_params(self) -> int:
        """Total parameters in prunable layers."""
        return sum(layer.num_params for layer in self.prunable_layers)

    @property
    def active_params_count(self) -> int:
        """Currently active (non-pruned) parameters."""
        return sum(layer.active_params for layer in self.prunable_layers)

    def get_gate_stats(self) -> dict[str, dict[str, float]]:
        """Per-layer gate statistics."""
        stats: dict[str, dict[str, float]] = {}
        for name, layer in zip(self.prunable_layer_names, self.prunable_layers):
            activations = layer.gate_activations.cpu().numpy().flatten()
            stats[name] = {
                "sparsity": layer.sparsity,
                "mean_gate": float(np.mean(activations)),
                "std_gate": float(np.std(activations)),
                "min_gate": float(np.min(activations)),
                "max_gate": float(np.max(activations)),
                "num_params": layer.num_params,
                "active_params": layer.active_params,
            }
        return stats

    def get_all_gate_values(self) -> dict[str, np.ndarray]:
        """All gate activation values per layer (for visualization)."""
        return {
            name: layer.gate_activations.cpu().numpy().flatten()
            for name, layer in zip(self.prunable_layer_names, self.prunable_layers)
        }

    def get_flops_reduction(self) -> dict[str, Any]:
        """
        Estimate FLOPs reduction from pruning.

        For conv layers: FLOPs ∝ active_filters × in_channels × k² × H × W
        For FC layers: FLOPs ∝ active_params
        """
        layer_info: list[dict[str, Any]] = []
        total_original = 0
        total_pruned = 0

        # Conv layers — estimate based on spatial dimensions at each stage
        spatial_sizes = [16, 8, 4]  # after each MaxPool
        for i, (name, layer) in enumerate(
            zip(["conv1", "conv2", "conv3"], [self.conv1, self.conv2, self.conv3])
        ):
            h = w = spatial_sizes[i]
            k = layer.kernel_size
            original_flops = layer.out_channels * layer.in_channels * k * k * h * w
            active_ratio = layer.active_params / max(layer.num_params, 1)
            pruned_flops = int(original_flops * active_ratio)
            total_original += original_flops
            total_pruned += pruned_flops
            layer_info.append({
                "layer": name,
                "original_flops": original_flops,
                "pruned_flops": pruned_flops,
                "reduction_pct": (1 - active_ratio) * 100,
            })

        # FC layers
        for name, layer in zip(["fc1", "fc2"], [self.fc1, self.fc2]):
            original_flops = layer.in_features * layer.out_features
            active_ratio = layer.active_params / max(layer.num_params, 1)
            pruned_flops = int(original_flops * active_ratio)
            total_original += original_flops
            total_pruned += pruned_flops
            layer_info.append({
                "layer": name,
                "original_flops": original_flops,
                "pruned_flops": pruned_flops,
                "reduction_pct": (1 - active_ratio) * 100,
            })

        return {
            "layers": layer_info,
            "total_original_flops": total_original,
            "total_pruned_flops": total_pruned,
            "total_reduction_pct": (1 - total_pruned / max(total_original, 1)) * 100,
        }

    def set_temperature(self, temperature: float) -> None:
        """Update temperature for all prunable layers (for annealing)."""
        for layer in self.prunable_layers:
            layer.temperature = temperature

    def get_temperature(self) -> float:
        """Get current temperature (all layers share the same temperature)."""
        return self.prunable_layers[0].temperature

    def export_hard_pruned(self, threshold: float = 1e-2) -> PrunableCNN:
        """
        Creates a hard-pruned copy of this model for deployment.
        Permanently zeros out weights where gate < threshold.
        Forces gate_scores to extreme values (+20 or -20) so they statically
        evaluate to 1.0 or 0.0 with zero runtime overhead or gradient drift.
        """
        import copy
        hard_model = copy.deepcopy(self)
        
        with torch.no_grad():
            for layer in hard_model.prunable_layers:
                # 1. Identify active gates
                gate_acts = torch.sigmoid(layer.gate_scores / max(layer.temperature, 1e-6))
                
                if isinstance(layer, PrunableConv2d):
                    mask = (gate_acts.squeeze() >= threshold).view(-1, 1, 1, 1)
                else:
                    mask = gate_acts >= threshold
                    
                mask_float = mask.float()
                
                # 2. Hard zero the weights
                layer.weight.data.mul_(mask_float)
                
                # 3. Force gate_scores to structural clamps (+20 for open, -20 for closed)
                # sigmoid(20) = 1.0, sigmoid(-20) = 0.0
                clamped_gates = torch.where(mask, torch.tensor(20.0, device=mask.device), torch.tensor(-20.0, device=mask.device))
                
                if isinstance(layer, PrunableConv2d):
                    layer.gate_scores.data.copy_(clamped_gates.view_as(layer.gate_scores))
                else:
                    layer.gate_scores.data.copy_(clamped_gates)
                    
        return hard_model

    def measure_inference_ms(self, device: torch.device, batch_size: int = 64, num_runs: int = 50) -> float:
        """
        Measure average inference latency (milliseconds per batch).
        Includes warmup runs to ensure accurate GPU/MPS benchmarking.
        """
        self.eval()
        dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self(dummy_input)
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
                
            import time
            start = time.perf_counter()
            
            for _ in range(num_runs):
                _ = self(dummy_input)
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
                
            end = time.perf_counter()
            
        return ((end - start) / num_runs) * 1000.0  # convert to ms
