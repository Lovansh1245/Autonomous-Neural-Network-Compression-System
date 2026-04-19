"""
layers.py — Prunable layers with learnable gates for self-pruning neural networks.

Implements PrunableLinear and PrunableConv2d with:
  - Learnable gate_scores per weight / per filter
  - Temperature-scaled sigmoid gating
  - Full gradient flow through soft gates
  - Numerically stable implementation
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A fully-connected layer with learnable gates on weights.

    Each weight element has a corresponding gate_score. During forward:
        gates = sigmoid(gate_scores / temperature)
        pruned_weight = weight * gates
        output = input @ pruned_weight.T + bias

    As training progresses with L1 sparsity loss on gate activations,
    gate_scores are pushed toward -∞, making gates → 0 and effectively
    pruning the corresponding weights.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        temperature: Initial temperature for sigmoid scaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable gate scores — one per weight element
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Temperature for sigmoid scaling (settable externally for annealing)
        self.temperature = temperature

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights (Kaiming) and gates (positive init so gates start ~open)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate_scores to ~5.0 so sigmoid(5) ≈ 0.99 — gates start wide open.
        # Higher init resists premature pruning; L1 must work harder to prune.
        nn.init.constant_(self.gate_scores, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temperature-scaled sigmoid gating
        gates = torch.sigmoid(self.gate_scores / max(self.temperature, 1e-6))
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    @property
    def gate_activations(self) -> torch.Tensor:
        """Current gate activation values (between 0 and 1)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores / max(self.temperature, 1e-6))

    @property
    def sparsity(self) -> float:
        """Fraction of gates below threshold (1e-2) — i.e., effectively pruned."""
        activations = self.gate_activations
        return float((activations < 1e-2).sum().item() / activations.numel())

    @property
    def gate_l1(self) -> torch.Tensor:
        """L1 norm of gate activations (used for sparsity loss)."""
        gates = torch.sigmoid(self.gate_scores / max(self.temperature, 1e-6))
        return gates.sum()

    @property
    def num_params(self) -> int:
        """Total number of weight parameters (excluding gates)."""
        return self.weight.numel()

    @property
    def active_params(self) -> int:
        """Number of weight parameters with gate > 1e-2."""
        return int((self.gate_activations >= 1e-2).sum().item())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, temperature={self.temperature:.4f}"
        )


class PrunableConv2d(nn.Module):
    """
    A convolutional layer with per-filter learnable gates.

    Each output filter has a single gate_score. During forward:
        gates = sigmoid(gate_scores / temperature)  # shape: (out_channels, 1, 1, 1)
        pruned_weight = weight * gates
        output = conv2d(input, pruned_weight, bias)

    This implements filter-level (structured) pruning — entire filters can be
    removed when their gate → 0.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides.
        bias: Whether to include a bias term.
        temperature: Initial temperature for sigmoid scaling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Standard conv weight: (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Per-filter gate score: (out_channels, 1, 1, 1) for broadcasting
        self.gate_scores = nn.Parameter(torch.empty(out_channels, 1, 1, 1))

        self.temperature = temperature
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores / max(self.temperature, 1e-6))
        pruned_weight = self.weight * gates
        return F.conv2d(x, pruned_weight, self.bias, self.stride, self.padding)

    @property
    def gate_activations(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(
                self.gate_scores.squeeze() / max(self.temperature, 1e-6)
            )

    @property
    def sparsity(self) -> float:
        activations = self.gate_activations
        return float((activations < 1e-2).sum().item() / activations.numel())

    @property
    def gate_l1(self) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores / max(self.temperature, 1e-6))
        return gates.sum()

    @property
    def num_params(self) -> int:
        return self.weight.numel()

    @property
    def active_params(self) -> int:
        """Params in filters with gate > 1e-2."""
        active_filters = (self.gate_activations >= 1e-2).sum().item()
        params_per_filter = self.in_channels * self.kernel_size * self.kernel_size
        return int(active_filters * params_per_filter)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}, "
            f"temperature={self.temperature:.4f}"
        )
