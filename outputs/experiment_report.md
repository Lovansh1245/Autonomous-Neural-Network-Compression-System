# Self-Pruning Neural Network — Experiment Report

## Results Summary

| Lambda (λ) | Accuracy | Sparsity | FLOPs Reduction | Training Time |
|:----------:|:--------:|:--------:|:---------------:|:-------------:|
| 0.0001 | 85.90% | 82.43% | 69.5% | 1054.7s |
| 0.001 | 83.03% | 99.60% | 98.4% | 1056.3s |
| 0.01 | 78.79% | 99.98% | 99.8% | 1054.7s |

## Recommendation

**Best λ = 0.001** with composite score 0.8291

- Accuracy: 83.03%
- Sparsity: 99.60%
- FLOPs Reduction: 98.4%

## Observations

- ✅ Sparsity increases monotonically with λ, confirming that L1 regularization on gate activations effectively drives pruning.
- 📉 Accuracy drops by 7.1% from λ=0.0001 to λ=0.01, while sparsity improves by 17.5%.
- 🏆 Best trade-off: λ=0.001 achieves 83.0% accuracy with 99.6% sparsity.
- ⚡ Highest pruning (λ=0.01) reduces estimated FLOPs by 99.8%.
- 🔍 At λ=0.0001: conv2 is most pruned (93.8%), conv1 is least pruned (0.0%).
- 🔍 At λ=0.001: conv2 is most pruned (100.0%), conv1 is least pruned (59.4%).
- 🔍 At λ=0.01: conv2 is most pruned (100.0%), fc2 is least pruned (91.0%).
- 🌡️ Temperature annealing sharpens gate decisions over training, pushing gates toward binary (0 or 1) values for cleaner pruning.

## Per-Layer Gate Statistics

### λ = 0.0001

| Layer | Sparsity | Mean Gate | Std Gate | Active/Total |
|:-----:|:--------:|:---------:|:--------:|:------------:|
| conv1 | 0.00% | 0.0254 | 0.0135 | 1,728/1,728 |
| conv2 | 93.75% | 0.0085 | 0.0008 | 4,608/73,728 |
| conv3 | 43.75% | 0.0102 | 0.0009 | 165,888/294,912 |
| fc1 | 87.63% | 0.0200 | 0.0889 | 259,497/2,097,152 |
| fc2 | 47.46% | 0.2294 | 0.3871 | 2,690/5,120 |

### λ = 0.001

| Layer | Sparsity | Mean Gate | Std Gate | Active/Total |
|:-----:|:--------:|:---------:|:--------:|:------------:|
| conv1 | 59.38% | 0.0104 | 0.0033 | 702/1,728 |
| conv2 | 100.00% | 0.0055 | 0.0003 | 0/73,728 |
| conv3 | 100.00% | 0.0077 | 0.0006 | 0/294,912 |
| fc1 | 99.62% | 0.0048 | 0.0011 | 7,885/2,097,152 |
| fc2 | 73.01% | 0.1178 | 0.2998 | 1,382/5,120 |

### λ = 0.01

| Layer | Sparsity | Mean Gate | Std Gate | Active/Total |
|:-----:|:--------:|:---------:|:--------:|:------------:|
| conv1 | 93.75% | 0.0062 | 0.0014 | 108/1,728 |
| conv2 | 100.00% | 0.0048 | 0.0001 | 0/73,728 |
| conv3 | 100.00% | 0.0062 | 0.0003 | 0/294,912 |
| fc1 | 100.00% | 0.0046 | 0.0000 | 0/2,097,152 |
| fc2 | 91.00% | 0.0158 | 0.0701 | 461/5,120 |
