"""
Training module for IRH surrogate models.

Provides data loading, loss functions, training loops, and evaluation
for ML surrogate training.

Key Components:
- RGTrajectoryDataset: Training data from numerical RG simulations
- FixedPointDataset: Fixed point classification data
- CombinedLoss: Multi-task loss function
- Trainer: Complete training loop with scheduling and checkpointing
- ModelEvaluator: Comprehensive evaluation and benchmarking
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
from .train_surrogate import (
    Trainer,
    LearningRateScheduler,
    EarlyStopping
)
from .evaluation import (
    ModelEvaluator,
    SpeedupBenchmark,
    TrajectoryErrorMetrics,
    FixedPointMetrics
)

__all__ = [
    # Data loading
    'RGTrajectoryDataset',
    'FixedPointDataset',
    # Loss functions
    'CombinedLoss',
    'CouplingPredictionLoss',
    'FixedPointClassificationLoss',
    'ActionPredictionLoss',
    'TrajectoryConsistencyLoss',
    'mse_loss',
    'mae_loss',
    'binary_cross_entropy_loss',
    # Training
    'Trainer',
    'LearningRateScheduler',
    'EarlyStopping',
    # Evaluation
    'ModelEvaluator',
    'SpeedupBenchmark',
    'TrajectoryErrorMetrics',
    'FixedPointMetrics',
]
