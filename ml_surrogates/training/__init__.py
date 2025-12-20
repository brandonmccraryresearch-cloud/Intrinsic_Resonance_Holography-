"""
Training module for IRH surrogate models.

Provides data loading, loss functions, training loops, and evaluation
for ML surrogate training.

Key Components:
- RGTrajectoryDataset: Training data from numerical RG simulations
- FixedPointDataset: Fixed point classification data
- CombinedLoss: Multi-task loss function
- Training loop: Gradient-based optimization (Phase 4 in progress)
- Evaluation metrics: Accuracy, speedup benchmarking
"""

from .data_loader import RGTrajectoryDataset, FixedPointDataset
from .loss_functions import (
    CombinedLoss,
    CouplingPredictionLoss,
    FixedPointClassificationLoss,
    ActionPredictionLoss,
    TrajectoryConsistencyLoss,
    mse_loss,
    mae_loss,
    binary_cross_entropy_loss
)

__all__ = [
    'RGTrajectoryDataset',
    'FixedPointDataset',
    'CombinedLoss',
    'CouplingPredictionLoss',
    'FixedPointClassificationLoss',
    'ActionPredictionLoss',
    'TrajectoryConsistencyLoss',
    'mse_loss',
    'mae_loss',
    'binary_cross_entropy_loss',
]
