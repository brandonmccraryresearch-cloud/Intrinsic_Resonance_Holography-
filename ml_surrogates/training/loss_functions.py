"""
Loss Functions for IRH Surrogate Training

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Custom loss functions for multi-task learning

Loss components:
1. Coupling prediction (MSE): Accuracy on (λ̃, γ̃, μ̃)
2. Fixed point classification (BCE): Binary detection
3. Action prediction (MAE): cGFT action functional
4. Trajectory consistency: Multi-step prediction error

Combined multi-task loss with task weighting.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean squared error loss.
    
    Args:
        predictions: Predicted values (batch, ...)
        targets: Target values (batch, ...)
        
    Returns:
        MSE loss value
    """
    return np.mean((predictions - targets) ** 2)


def mae_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean absolute error loss.
    
    Args:
        predictions: Predicted values (batch, ...)
        targets: Target values (batch, ...)
        
    Returns:
        MAE loss value
    """
    return np.mean(np.abs(predictions - targets))


def binary_cross_entropy_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-7
) -> float:
    """
    Binary cross-entropy loss with numerical stability.
    
    Args:
        predictions: Predicted probabilities (batch,)
        targets: Target labels 0/1 (batch,)
        eps: Small constant for numerical stability
        
    Returns:
        BCE loss value
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1 - eps)
    
    return -np.mean(
        targets * np.log(predictions) + 
        (1 - targets) * np.log(1 - predictions)
    )


class CouplingPredictionLoss:
    """
    Loss for coupling constant predictions.
    
    Measures accuracy of (λ̃, γ̃, μ̃) predictions using MSE.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize loss.
        
        Args:
            weight: Loss weight in multi-task setting
        """
        self.weight = weight
    
    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute loss.
        
        Args:
            predictions: Predicted couplings (batch, 3)
            targets: Target couplings (batch, 3)
            
        Returns:
            Weighted MSE loss
        """
        return self.weight * mse_loss(predictions, targets)


class FixedPointClassificationLoss:
    """
    Loss for fixed point classification.
    
    Binary classification: Is this trajectory at a fixed point?
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize loss.
        
        Args:
            weight: Loss weight in multi-task setting
        """
        self.weight = weight
    
    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute loss.
        
        Args:
            predictions: Predicted probabilities (batch,)
            targets: Target labels 0/1 (batch,)
            
        Returns:
            Weighted BCE loss
        """
        return self.weight * binary_cross_entropy_loss(predictions, targets)


class ActionPredictionLoss:
    """
    Loss for cGFT action functional predictions.
    
    Measures accuracy of action value predictions using MAE.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize loss.
        
        Args:
            weight: Loss weight (typically smaller than coupling loss)
        """
        self.weight = weight
    
    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute loss.
        
        Args:
            predictions: Predicted actions (batch,)
            targets: Target actions (batch,)
            
        Returns:
            Weighted MAE loss
        """
        return self.weight * mae_loss(predictions, targets)


class TrajectoryConsistencyLoss:
    """
    Loss for multi-step trajectory prediction consistency.
    
    Ensures predicted trajectories are physically consistent
    with RG flow evolution.
    """
    
    def __init__(self, weight: float = 0.5):
        """
        Initialize loss.
        
        Args:
            weight: Loss weight for trajectory consistency
        """
        self.weight = weight
    
    def __call__(
        self,
        predicted_trajectory: np.ndarray,
        target_trajectory: np.ndarray
    ) -> float:
        """
        Compute loss.
        
        Args:
            predicted_trajectory: Predicted states (num_steps, 3)
            target_trajectory: Target states (num_steps, 3)
            
        Returns:
            Weighted trajectory MSE loss
        """
        return self.weight * mse_loss(predicted_trajectory, target_trajectory)


class CombinedLoss:
    """
    Combined multi-task loss for IRH surrogate training.
    
    Combines:
    - Coupling prediction loss (primary)
    - Fixed point classification loss
    - Action prediction loss
    - Trajectory consistency loss (optional)
    
    With configurable task weights.
    """
    
    def __init__(
        self,
        coupling_weight: float = 1.0,
        fixed_point_weight: float = 0.5,
        action_weight: float = 0.1,
        trajectory_weight: float = 0.3,
        use_trajectory_loss: bool = False
    ):
        """
        Initialize combined loss.
        
        Args:
            coupling_weight: Weight for coupling predictions
            fixed_point_weight: Weight for fixed point classification
            action_weight: Weight for action predictions
            trajectory_weight: Weight for trajectory consistency
            use_trajectory_loss: Whether to include trajectory loss
        """
        self.coupling_loss = CouplingPredictionLoss(weight=coupling_weight)
        self.fixed_point_loss = FixedPointClassificationLoss(weight=fixed_point_weight)
        self.action_loss = ActionPredictionLoss(weight=action_weight)
        self.trajectory_loss = TrajectoryConsistencyLoss(weight=trajectory_weight)
        self.use_trajectory_loss = use_trajectory_loss
    
    def __call__(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dictionary of predictions with keys:
                - 'couplings': (batch, 3)
                - 'is_fixed_point': (batch,)
                - 'action': (batch,)
                - 'trajectory': (batch, num_steps, 3) [optional]
            targets: Dictionary of targets with same keys
            
        Returns:
            (total_loss, loss_components)
            - total_loss: Combined scalar loss
            - loss_components: Dict of individual loss values
        """
        loss_components = {}
        
        # Coupling prediction loss
        if 'couplings' in predictions and 'couplings' in targets:
            loss_components['coupling'] = self.coupling_loss(
                predictions['couplings'],
                targets['couplings']
            )
        else:
            loss_components['coupling'] = 0.0
        
        # Fixed point classification loss
        if 'is_fixed_point' in predictions and 'is_fixed_point' in targets:
            # Reshape if needed
            pred_fp = predictions['is_fixed_point'].flatten()
            target_fp = targets['is_fixed_point'].flatten()
            
            loss_components['fixed_point'] = self.fixed_point_loss(
                pred_fp,
                target_fp
            )
        else:
            loss_components['fixed_point'] = 0.0
        
        # Action prediction loss
        if 'action' in predictions and 'action' in targets:
            pred_action = predictions['action'].flatten()
            target_action = targets['action'].flatten()
            
            loss_components['action'] = self.action_loss(
                pred_action,
                target_action
            )
        else:
            loss_components['action'] = 0.0
        
        # Trajectory consistency loss (optional)
        if self.use_trajectory_loss:
            if 'trajectory' in predictions and 'trajectory' in targets:
                loss_components['trajectory'] = self.trajectory_loss(
                    predictions['trajectory'],
                    targets['trajectory']
                )
            else:
                loss_components['trajectory'] = 0.0
        
        # Total loss
        total_loss = sum(loss_components.values())
        loss_components['total'] = total_loss
        
        return total_loss, loss_components


# Example usage and validation
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # Test 1: MSE loss
    print("\n1. Testing MSE loss...")
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.5])
    mse = mse_loss(pred, target)
    print(f"  ✓ MSE: {mse:.4f}")
    assert mse > 0
    
    # Test 2: MAE loss
    print("\n2. Testing MAE loss...")
    mae = mae_loss(pred, target)
    print(f"  ✓ MAE: {mae:.4f}")
    assert mae > 0
    
    # Test 3: BCE loss
    print("\n3. Testing BCE loss...")
    pred_prob = np.array([0.8, 0.3, 0.6])
    target_label = np.array([1.0, 0.0, 1.0])
    bce = binary_cross_entropy_loss(pred_prob, target_label)
    print(f"  ✓ BCE: {bce:.4f}")
    assert bce > 0
    
    # Test 4: Coupling prediction loss
    print("\n4. Testing CouplingPredictionLoss...")
    coupling_loss = CouplingPredictionLoss(weight=1.0)
    
    pred_couplings = np.random.randn(10, 3)
    target_couplings = np.random.randn(10, 3)
    
    loss = coupling_loss(pred_couplings, target_couplings)
    print(f"  ✓ Coupling loss: {loss:.4f}")
    
    # Test 5: Fixed point classification loss
    print("\n5. Testing FixedPointClassificationLoss...")
    fp_loss = FixedPointClassificationLoss(weight=0.5)
    
    pred_fp = np.random.rand(10)
    target_fp = np.random.randint(0, 2, size=10).astype(float)
    
    loss = fp_loss(pred_fp, target_fp)
    print(f"  ✓ FP classification loss: {loss:.4f}")
    
    # Test 6: Action prediction loss
    print("\n6. Testing ActionPredictionLoss...")
    action_loss = ActionPredictionLoss(weight=0.1)
    
    pred_action = np.random.randn(10)
    target_action = np.random.randn(10)
    
    loss = action_loss(pred_action, target_action)
    print(f"  ✓ Action loss: {loss:.4f}")
    
    # Test 7: Trajectory consistency loss
    print("\n7. Testing TrajectoryConsistencyLoss...")
    traj_loss = TrajectoryConsistencyLoss(weight=0.3)
    
    pred_traj = np.random.randn(5, 3)
    target_traj = np.random.randn(5, 3)
    
    loss = traj_loss(pred_traj, target_traj)
    print(f"  ✓ Trajectory loss: {loss:.4f}")
    
    # Test 8: Combined loss
    print("\n8. Testing CombinedLoss...")
    combined_loss = CombinedLoss(
        coupling_weight=1.0,
        fixed_point_weight=0.5,
        action_weight=0.1,
        use_trajectory_loss=False
    )
    
    predictions = {
        'couplings': np.random.randn(10, 3),
        'is_fixed_point': np.random.rand(10),
        'action': np.random.randn(10)
    }
    
    targets = {
        'couplings': np.random.randn(10, 3),
        'is_fixed_point': np.random.randint(0, 2, size=10).astype(float),
        'action': np.random.randn(10)
    }
    
    total_loss, components = combined_loss(predictions, targets)
    
    print(f"  ✓ Total loss: {total_loss:.4f}")
    print(f"  ✓ Components: {list(components.keys())}")
    print(f"    - Coupling: {components['coupling']:.4f}")
    print(f"    - Fixed point: {components['fixed_point']:.4f}")
    print(f"    - Action: {components['action']:.4f}")
    
    # Test 9: Combined loss with trajectory
    print("\n9. Testing CombinedLoss with trajectory...")
    combined_loss_traj = CombinedLoss(use_trajectory_loss=True)
    
    predictions['trajectory'] = np.random.randn(10, 5, 3)
    targets['trajectory'] = np.random.randn(10, 5, 3)
    
    total_loss, components = combined_loss_traj(predictions, targets)
    
    print(f"  ✓ Total loss with trajectory: {total_loss:.4f}")
    print(f"  ✓ Trajectory component: {components.get('trajectory', 0.0):.4f}")
    
    print("\n✅ All loss function tests passed!")
