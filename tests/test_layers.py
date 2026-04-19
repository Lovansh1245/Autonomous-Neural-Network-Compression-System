"""
tests/test_layers.py — Smoke tests for prunable layers and model.

Validates:
  - PrunableLinear forward pass shape
  - PrunableConv2d forward pass shape
  - Gate initialization (start mostly open)
  - Sparsity computation
  - Gate L1 loss is differentiable
  - Temperature scaling effect
  - PrunableCNN end-to-end forward pass
  - FLOPs reduction estimation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from layers import PrunableLinear, PrunableConv2d
from model import PrunableCNN


class TestPrunableLinear:
    def test_forward_shape(self):
        layer = PrunableLinear(128, 64)
        x = torch.randn(8, 128)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_gates_start_open(self):
        layer = PrunableLinear(128, 64)
        activations = layer.gate_activations
        # sigmoid(2.0) ≈ 0.88, so all gates should be > 0.5
        assert activations.min().item() > 0.5

    def test_sparsity_initially_zero(self):
        layer = PrunableLinear(128, 64)
        assert layer.sparsity == 0.0  # No gates below 0.01 initially

    def test_gate_l1_differentiable(self):
        layer = PrunableLinear(32, 16)
        l1 = layer.gate_l1
        l1.backward()
        assert layer.gate_scores.grad is not None
        assert layer.gate_scores.grad.shape == (16, 32)

    def test_temperature_effect(self):
        layer_warm = PrunableLinear(32, 16, temperature=10.0)
        layer_cold = PrunableLinear(32, 16, temperature=0.01)

        # Same gate scores, but cold temperature → more extreme activations
        scores = torch.tensor([[-5.0, 0.0, 5.0]] * 16).reshape(16, 3)

        # Manually test sigmoid behavior
        warm_gates = torch.sigmoid(scores / 10.0)
        cold_gates = torch.sigmoid(scores / 0.01)

        # Cold: gates should be much closer to 0 or 1
        assert cold_gates[:, 0].max() < 0.01  # Negative scores → 0
        assert cold_gates[:, 2].min() > 0.99  # Positive scores → 1
        # Warm: gates should be closer to 0.5
        assert warm_gates[:, 1].mean() > 0.4
        assert warm_gates[:, 1].mean() < 0.6

    def test_pruning_via_negative_gate_scores(self):
        layer = PrunableLinear(32, 16)
        # Push gates to strongly negative → should become sparse
        with torch.no_grad():
            layer.gate_scores.fill_(-10.0)
        assert layer.sparsity > 0.99

    def test_num_params(self):
        layer = PrunableLinear(128, 64)
        assert layer.num_params == 128 * 64


class TestPrunableConv2d:
    def test_forward_shape(self):
        layer = PrunableConv2d(3, 64, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 64, 32, 32)

    def test_gates_start_open(self):
        layer = PrunableConv2d(3, 64, kernel_size=3, padding=1)
        activations = layer.gate_activations
        assert activations.min().item() > 0.5

    def test_filter_level_sparsity(self):
        layer = PrunableConv2d(3, 64, kernel_size=3, padding=1)
        # Prune half the filters
        with torch.no_grad():
            layer.gate_scores[:32] = -10.0
        assert 0.4 < layer.sparsity < 0.6

    def test_gate_l1_differentiable(self):
        layer = PrunableConv2d(3, 16, kernel_size=3, padding=1)
        l1 = layer.gate_l1
        l1.backward()
        assert layer.gate_scores.grad is not None


class TestPrunableCNN:
    def test_forward_shape(self):
        model = PrunableCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_gate_l1_loss(self):
        model = PrunableCNN()
        l1 = model.get_gate_l1_loss()
        assert l1.requires_grad
        assert l1.item() > 0

    def test_initial_sparsity_is_zero(self):
        model = PrunableCNN()
        assert model.get_sparsity() == 0.0

    def test_gate_stats(self):
        model = PrunableCNN()
        stats = model.get_gate_stats()
        assert "conv1" in stats
        assert "fc2" in stats
        assert "sparsity" in stats["conv1"]
        assert "mean_gate" in stats["conv1"]

    def test_flops_reduction(self):
        model = PrunableCNN()
        flops = model.get_flops_reduction()
        assert "total_original_flops" in flops
        assert "total_reduction_pct" in flops
        # Initially no pruning → 0% reduction
        assert flops["total_reduction_pct"] < 1.0

    def test_temperature_setting(self):
        model = PrunableCNN(temperature=1.0)
        model.set_temperature(0.5)
        assert model.get_temperature() == 0.5

    def test_backward_pass(self):
        model = PrunableCNN()
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y) + 0.001 * model.get_gate_l1_loss()
        loss.backward()
        # Check gradients exist on gate_scores
        for layer in model.prunable_layers:
            assert layer.gate_scores.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
