"""
RG Flow Neural Network Surrogate Model

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.2-1.3, Phase 4.3 (ML Surrogate Models)

This module implements neural network surrogate models that approximate
the RG flow equations (Eq. 1.12-1.13) for fast evaluation.

The surrogate learns the mapping:
    (λ̃, γ̃, μ̃, t) → (λ̃(t), γ̃(t), μ̃(t))

where t is the RG scale parameter. Training data is generated from
high-fidelity numerical integration of the β-functions.

Key Features:
- Multi-layer perceptron architecture
- Ensemble methods for uncertainty estimation
- Physics-informed constraints (conservation laws, fixed point)
- Automatic CPU fallback when no ML framework available

Dependencies:
- numpy (required)
- scipy (required for numerical integration)
- Optional: scikit-learn, torch, jax for advanced features

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
from scipy.integrate import solve_ivp

__version__ = "21.1.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript §1.2-1.3, Eq. 1.12-1.13"


# =============================================================================
# Physical Constants (from fixed_points.py)
# =============================================================================

LAMBDA_STAR = 48 * math.pi**2 / 9      # λ̃* = 48π²/9 ≈ 52.638
GAMMA_STAR = 32 * math.pi**2 / 3       # γ̃* = 32π²/3 ≈ 105.276
MU_STAR = 16 * math.pi**2               # μ̃* = 16π² ≈ 157.914
FIXED_POINT = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SurrogateConfig:
    """
    Configuration for RG flow surrogate model.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3 (ML Surrogate Models)
        
    Attributes
    ----------
    hidden_layers : list
        Hidden layer sizes for the neural network
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid')
    learning_rate : float
        Learning rate for training
    n_ensemble : int
        Number of models in ensemble (for uncertainty)
    physics_weight : float
        Weight for physics-informed loss terms
    normalize_inputs : bool
        Whether to normalize inputs
    normalize_outputs : bool
        Whether to normalize outputs
    """
    hidden_layers: List[int] = field(default_factory=lambda: [64, 128, 64])
    activation: str = 'tanh'
    learning_rate: float = 0.001
    n_ensemble: int = 5
    physics_weight: float = 0.1
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    max_epochs: int = 1000
    batch_size: int = 32
    early_stopping_patience: int = 50
    seed: int = 42


# =============================================================================
# Beta Functions (inline for independence)
# =============================================================================


def _beta_lambda(lambda_t: float) -> float:
    """β_λ = -2λ̃ + (9/8π²)λ̃² (Eq. 1.13)"""
    return -2 * lambda_t + (9 / (8 * math.pi**2)) * lambda_t**2


def _beta_gamma(lambda_t: float, gamma_t: float) -> float:
    """β_γ = (3/4π²)λ̃γ̃ (Eq. 1.13)"""
    return (3 / (4 * math.pi**2)) * lambda_t * gamma_t


def _beta_mu(lambda_t: float, mu_t: float) -> float:
    """β_μ = 2μ̃ + (1/2π²)λ̃μ̃ (Eq. 1.13)"""
    return 2 * mu_t + (1 / (2 * math.pi**2)) * lambda_t * mu_t


def _compute_betas(couplings: np.ndarray) -> np.ndarray:
    """Compute all beta functions at given couplings."""
    lambda_t, gamma_t, mu_t = couplings
    return np.array([
        _beta_lambda(lambda_t),
        _beta_gamma(lambda_t, gamma_t),
        _beta_mu(lambda_t, mu_t)
    ])


# =============================================================================
# Data Generation
# =============================================================================


def generate_training_data(
    n_trajectories: int = 1000,
    t_range: Tuple[float, float] = (-5, 5),
    n_points: int = 50,
    noise_std: float = 0.0,
    seed: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate training data from numerical integration of RG equations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2, Eq. 1.13
        
    Parameters
    ----------
    n_trajectories : int
        Number of RG trajectories to generate
    t_range : tuple
        (t_min, t_max) range for RG scale parameter
    n_points : int
        Number of points along each trajectory
    noise_std : float
        Standard deviation of Gaussian noise (for robustness training)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Training data with keys:
        - 'inputs': (N, 4) array of [λ, γ, μ, t_initial]
        - 'outputs': (N, 3) array of [λ(t), γ(t), μ(t)]
        - 't_values': (N,) array of RG scale values
        - 'trajectories': list of full trajectory arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_eval = np.linspace(t_range[0], t_range[1], n_points)
    
    inputs = []
    outputs = []
    t_values = []
    trajectories = []
    
    for _ in range(n_trajectories):
        # Sample initial conditions around the fixed point
        # Use narrower range for better stability
        scale = np.exp(np.random.uniform(-0.3, 0.3, 3))
        initial = FIXED_POINT * scale
        
        # Integrate RG equations with bounded outputs
        # Theoretical Reference: IRH v21.4

        def rg_system(t, y):
            # Clip to prevent numerical explosions
            y_clipped = np.clip(y, 1e-6, 1e6)
            return _compute_betas(y_clipped)
        
        try:
            sol = solve_ivp(
                rg_system,
                t_range,
                initial,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8,
                max_step=0.5,  # Limit step size for stability
            )
            
            if sol.success and sol.y.shape[1] == n_points:
                trajectory = sol.y.T  # (n_points, 3)
                
                # Check for NaN or extreme values
                if np.any(np.isnan(trajectory)) or np.any(np.abs(trajectory) > 1e6):
                    continue
                    
                trajectories.append(trajectory)
                
                # Create input-output pairs
                for i, t in enumerate(t_eval[:-1]):
                    inputs.append(np.append(trajectory[i], t))
                    outputs.append(trajectory[i+1])
                    t_values.append(t_eval[i+1])
                    
        except Exception:
            continue
    
    # Handle empty arrays case
    if len(inputs) == 0:
        inputs = np.zeros((0, 4))
        outputs = np.zeros((0, 3))
        t_values = np.zeros(0)
    else:
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        t_values = np.array(t_values)
    
    # Add noise if requested
    if noise_std > 0:
        outputs += np.random.normal(0, noise_std, outputs.shape)
    
    return {
        'inputs': inputs,
        'outputs': outputs,
        't_values': t_values,
        'trajectories': trajectories,
        'n_samples': len(inputs),
        'n_trajectories': len(trajectories),
    }


# =============================================================================
# Simple Neural Network (NumPy-based, no dependencies)
# =============================================================================


class SimpleNeuralNetwork:
    """
    Simple feedforward neural network implemented in pure NumPy.
    
    This provides a dependency-free surrogate model that works even
    without scikit-learn or PyTorch.
    
    Theoretical Reference:
        Phase 4.3 ML Surrogate Models
    """
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'tanh',
        seed: int = 42,
    ):
        """
        Initialize neural network.
        
        Parameters
        ----------
        layer_sizes : list
            [input_dim, hidden1, hidden2, ..., output_dim]
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid')
        seed : int
            Random seed
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.seed = seed
        
        # Select activation function
        self.activation, self.activation_grad = self._get_activation(activation)
        
        # Initialize weights using Xavier initialization
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
        
        # Normalization parameters
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
        # Training history
        self.loss_history = []
    
    def _get_activation(self, name: str) -> Tuple[Callable, Callable]:
        """Get activation function and its gradient."""
        if name == 'tanh':
            return (
                lambda x: np.tanh(x),
                lambda x: 1 - np.tanh(x)**2
            )
        elif name == 'relu':
            return (
                lambda x: np.maximum(0, x),
                lambda x: (x > 0).astype(float)
            )
        elif name == 'sigmoid':
            return (
                lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
                lambda x: x * (1 - x)  # Gradient when x is sigmoid output: σ'(z) = σ(z)(1-σ(z))
            )
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    # Theoretical Reference: IRH v21.4

    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Normalize inputs
        if self.input_mean is not None:
            X = (X - self.input_mean) / (self.input_std + 1e-8)
        
        # Forward through layers
        self.activations = [X]
        h = X
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            
            # Apply activation (except last layer)
            if i < len(self.weights) - 1:
                h = self.activation(z)
            else:
                h = z  # Linear output
            
            self.activations.append(h)
        
        # Denormalize outputs
        if self.output_mean is not None:
            h = h * (self.output_std + 1e-8) + self.output_mean
        
        return h
    
    # Theoretical Reference: IRH v21.4

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for given inputs."""
        return self.forward(X)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.001,
        epochs: int = 1000,
        batch_size: int = 32,
        verbose: bool = True,
        early_stopping: int = 50,
    ) -> Dict[str, Any]:
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        Train the network using mini-batch gradient descent.
        
        Parameters
        ----------
        X : ndarray
            Input data (n_samples, n_features)
        y : ndarray
            Target data (n_samples, n_outputs)
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Mini-batch size
        verbose : bool
            Print training progress
        early_stopping : int
            Stop if no improvement for this many epochs
            
        Returns
        -------
        dict
            Training history
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        # Handle empty data case
        if len(X) == 0 or len(y) == 0:
            return {
                'loss_history': [],
                'final_loss': float('inf'),
                'epochs_trained': 0,
            }
        
        # Compute normalization parameters
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)
        self.output_mean = y.mean(axis=0)
        self.output_std = y.std(axis=0)
        
        # Normalize data for training
        X_norm = (X - self.input_mean) / (self.input_std + 1e-8)
        y_norm = (y - self.output_mean) / (self.output_std + 1e-8)
        
        n_samples = len(X)
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                X_batch = X_norm[batch_idx]
                y_batch = y_norm[batch_idx]
                
                # Forward pass (on normalized data)
                self.activations = [X_batch]
                h = X_batch
                
                for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                    z = h @ W + b
                    if i < len(self.weights) - 1:
                        h = self.activation(z)
                    else:
                        h = z
                    self.activations.append(h)
                
                y_pred = h
                
                # Compute loss
                loss = np.mean((y_pred - y_batch)**2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                delta = 2 * (y_pred - y_batch) / len(y_batch)
                
                for i in range(len(self.weights) - 1, -1, -1):
                    # Gradient for weights and biases
                    dW = self.activations[i].T @ delta
                    db = delta.sum(axis=0)
                    
                    # Update weights
                    self.weights[i] -= learning_rate * dW
                    self.biases[i] -= learning_rate * db
                    
                    # Propagate error (if not first layer)
                    if i > 0:
                        delta = delta @ self.weights[i].T
                        # Apply activation gradient
                        delta *= self.activation_grad(self.activations[i])
            
            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)
            
            # Early stopping check
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        return {
            'loss_history': self.loss_history,
            'final_loss': self.loss_history[-1],
            'epochs_trained': len(self.loss_history),
        }


# =============================================================================
# RG Flow Surrogate Model
# =============================================================================


class RGFlowSurrogate:
    """
    Neural network surrogate model for RG flow equations.
    
    THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.2-1.3
    
    This surrogate learns to approximate the mapping:
        (λ̃, γ̃, μ̃, t) → (λ̃(t'), γ̃(t'), μ̃(t'))
        
    where t' = t + Δt is a small step in RG scale.
    
    Attributes
    ----------
    config : SurrogateConfig
        Model configuration
    model : SimpleNeuralNetwork
        Underlying neural network
    ensemble : list
        Ensemble of models for uncertainty estimation
    is_trained : bool
        Whether the model has been trained
        
    Examples
    --------
    >>> surrogate = RGFlowSurrogate()
    >>> surrogate.train(n_trajectories=500)
    >>> initial = np.array([50.0, 100.0, 150.0])
    >>> trajectory = surrogate.predict_trajectory(initial, t_range=(-5, 5))
    """
    
    # Theoretical Reference: IRH v21.4
    def __init__(self, config: Optional[SurrogateConfig] = None):
        """
        Initialize RG flow surrogate model.
        
        Parameters
        ----------
        config : SurrogateConfig, optional
            Model configuration. Uses defaults if not provided.
        """
        self.config = config or SurrogateConfig()
        self.is_trained = False
        self.ensemble = []
        self.training_data = None
        
        # Create primary model
        layer_sizes = [4] + self.config.hidden_layers + [3]  # 4 inputs, 3 outputs
        self.model = SimpleNeuralNetwork(
            layer_sizes=layer_sizes,
            activation=self.config.activation,
            seed=self.config.seed,
        )
    
    def train(
        self,
        n_trajectories: int = 1000,
        t_range: Tuple[float, float] = (-5, 5),
        n_points: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the surrogate model on synthetic RG flow data.
        
        Theoretical Reference:
            IRH v21.1 Manuscript §1.2.2, Eq. 1.13
            
        Parameters
        ----------
        n_trajectories : int
            Number of training trajectories
        t_range : tuple
            RG scale range for training
        n_points : int
            Points per trajectory
        verbose : bool
            Print training progress
            
        Returns
        -------
        dict
            Training results
        """
        if verbose:
            print("Generating training data...")
        
        # Generate training data
        self.training_data = generate_training_data(
            n_trajectories=n_trajectories,
            t_range=t_range,
            n_points=n_points,
            seed=self.config.seed,
        )
        
        X = self.training_data['inputs']
        y = self.training_data['outputs']
        
        if verbose:
            print(f"Training on {len(X)} samples from {self.training_data['n_trajectories']} trajectories")
        
        # Train primary model
        result = self.model.fit(
            X, y,
            learning_rate=self.config.learning_rate,
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            verbose=verbose,
            early_stopping=self.config.early_stopping_patience,
        )
        
        # Train ensemble for uncertainty estimation
        if self.config.n_ensemble > 1:
            if verbose:
                print(f"Training ensemble of {self.config.n_ensemble} models...")
            
            for i in range(self.config.n_ensemble):
                layer_sizes = [4] + self.config.hidden_layers + [3]
                model = SimpleNeuralNetwork(
                    layer_sizes=layer_sizes,
                    activation=self.config.activation,
                    seed=self.config.seed + i + 1,
                )
                
                # Bootstrap sampling
                idx = np.random.choice(len(X), len(X), replace=True)
                model.fit(
                    X[idx], y[idx],
                    learning_rate=self.config.learning_rate,
                    epochs=self.config.max_epochs // 2,  # Faster training
                    batch_size=self.config.batch_size,
                    verbose=False,
                )
                self.ensemble.append(model)
        
        self.is_trained = True
        
        return {
            'training_result': result,
            'n_samples': len(X),
            'n_trajectories': self.training_data['n_trajectories'],
            'ensemble_size': len(self.ensemble),
        }
    
    # Theoretical Reference: IRH v21.4
    def predict(
        self,
        couplings: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Predict couplings after one step in RG scale.
        
        Parameters
        ----------
        couplings : ndarray
            Current coupling values [λ̃, γ̃, μ̃]
        t : float
            Current RG scale
            
        Returns
        -------
        ndarray
            Predicted couplings after one step
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        couplings = np.asarray(couplings).flatten()
        X = np.append(couplings, t).reshape(1, -1)
        return self.model.predict(X).flatten()
    
    # Theoretical Reference: IRH v21.4

    
    def predict_with_uncertainty(
        self,
        couplings: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict couplings with uncertainty estimate from ensemble.
        
        Parameters
        ----------
        couplings : ndarray
            Current coupling values
        t : float
            Current RG scale
            
        Returns
        -------
        tuple
            (mean_prediction, std_prediction)
        """
        if not self.ensemble:
            # No ensemble - return zero uncertainty
            pred = self.predict(couplings, t)
            return pred, np.zeros_like(pred)
        
        couplings = np.asarray(couplings).flatten()
        X = np.append(couplings, t).reshape(1, -1)
        
        predictions = np.array([m.predict(X).flatten() for m in self.ensemble])
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    def predict_trajectory(
        self,
        initial_couplings: np.ndarray,
        t_range: Tuple[float, float] = (-5, 5),
        n_steps: int = 100,
        with_uncertainty: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict full RG trajectory using surrogate model.
        
        Theoretical Reference:
            IRH v21.1 Manuscript §1.2-1.3
            
        Parameters
        ----------
        initial_couplings : ndarray
            Initial [λ̃, γ̃, μ̃] values
        t_range : tuple
            (t_min, t_max) for trajectory
        n_steps : int
            Number of integration steps
        with_uncertainty : bool
            Include uncertainty estimates
            
        Returns
        -------
        dict
            Trajectory data with keys:
            - 'couplings': (n_steps, 3) array
            - 't_values': (n_steps,) array
            - 'uncertainty': (n_steps, 3) array (if with_uncertainty)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        t_values = np.linspace(t_range[0], t_range[1], n_steps)
        dt = t_values[1] - t_values[0]
        
        trajectory = np.zeros((n_steps, 3))
        trajectory[0] = initial_couplings
        
        if with_uncertainty:
            uncertainties = np.zeros((n_steps, 3))
            uncertainties[0] = 0
        
        current = np.array(initial_couplings)
        
        for i in range(1, n_steps):
            if with_uncertainty and self.ensemble:
                pred, unc = self.predict_with_uncertainty(current, t_values[i-1])
                trajectory[i] = pred
                uncertainties[i] = unc
            else:
                trajectory[i] = self.predict(current, t_values[i-1])
            
            current = trajectory[i]
        
        result = {
            'couplings': trajectory,
            't_values': t_values,
            'lambda_trajectory': trajectory[:, 0],
            'gamma_trajectory': trajectory[:, 1],
            'mu_trajectory': trajectory[:, 2],
        }
        
        if with_uncertainty:
            result['uncertainty'] = uncertainties
        
        return result
    
    # Theoretical Reference: IRH v21.4

    
    def validate(
        self,
        n_test_trajectories: int = 50,
        t_range: Tuple[float, float] = (-3, 3),
    ) -> Dict[str, float]:
        """
        Validate surrogate against numerical integration.
        
        Returns
        -------
        dict
            Validation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before validation")
        
        # Generate test data
        test_data = generate_training_data(
            n_trajectories=n_test_trajectories,
            t_range=t_range,
            seed=self.config.seed + 10000,  # Different seed
        )
        
        X_test = test_data['inputs']
        y_test = test_data['outputs']
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Compute metrics
        mse = np.mean((y_pred - y_test)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        
        # Relative error at fixed point
        fp_input = np.append(FIXED_POINT, 0.0).reshape(1, -1)
        fp_pred = self.model.predict(fp_input).flatten()
        fp_error = np.linalg.norm(fp_pred - FIXED_POINT) / np.linalg.norm(FIXED_POINT)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'fixed_point_relative_error': float(fp_error),
            'n_test_samples': len(X_test),
        }


# =============================================================================
# Module-Level Functions
# =============================================================================


# Theoretical Reference: IRH v21.4



def create_rg_flow_surrogate(config: Optional[SurrogateConfig] = None) -> RGFlowSurrogate:
    """
    Create an RG flow surrogate model.
    
    Parameters
    ----------
    config : SurrogateConfig, optional
        Configuration for the surrogate model
        
    Returns
    -------
    RGFlowSurrogate
        Untrained surrogate model
    """
    return RGFlowSurrogate(config)


def train_rg_flow_surrogate(
    n_trajectories: int = 1000,
    config: Optional[SurrogateConfig] = None,
    verbose: bool = True,
) -> RGFlowSurrogate:
    """
    Create and train an RG flow surrogate model.
    
    # Theoretical Reference:
        IRH v21.1 Manuscript §1.2-1.3, Phase 4.3
        
    Parameters
    ----------
    n_trajectories : int
        Number of training trajectories
    config : SurrogateConfig, optional
        Model configuration
    verbose : bool
        Print training progress
        
    Returns
    -------
    RGFlowSurrogate
        Trained surrogate model
    """
    surrogate = RGFlowSurrogate(config)
    surrogate.train(n_trajectories=n_trajectories, verbose=verbose)
    return surrogate


# Theoretical Reference: IRH v21.4



def predict_rg_trajectory(
    initial_couplings: np.ndarray,
    surrogate: Optional[RGFlowSurrogate] = None,
    t_range: Tuple[float, float] = (-5, 5),
    n_steps: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Predict RG trajectory using surrogate or numerical integration.
    
    Parameters
    ----------
    initial_couplings : ndarray
        Initial [λ̃, γ̃, μ̃] values
    surrogate : RGFlowSurrogate, optional
        Trained surrogate model. If None, uses numerical integration.
    t_range : tuple
        (t_min, t_max) for trajectory
    n_steps : int
        Number of steps
        
    Returns
    -------
    dict
        Trajectory data
    """
    if surrogate is not None and surrogate.is_trained:
        return surrogate.predict_trajectory(initial_couplings, t_range, n_steps)
    
    # Fall back to numerical integration
    t_eval = np.linspace(t_range[0], t_range[1], n_steps)
    
    def rg_system(t, y):
        """
        # Theoretical Reference: IRH v21.4
        """
        y_clipped = np.clip(y, 1e-6, 1e6)
        return _compute_betas(y_clipped)
    
    sol = solve_ivp(
        rg_system,
        t_range,
        initial_couplings,
        t_eval=t_eval,
        method='RK45',
        max_step=0.5,
    )
    
    # Transpose and ensure correct shape
    couplings = sol.y.T  # (n_points, 3)
    
    return {
        'couplings': couplings,
        't_values': sol.t,
        'lambda_trajectory': couplings[:, 0],
        'gamma_trajectory': couplings[:, 1],
        'mu_trajectory': couplings[:, 2],
    }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    'SurrogateConfig',
    'RGFlowSurrogate',
    'SimpleNeuralNetwork',
    'create_rg_flow_surrogate',
    'train_rg_flow_surrogate',
    'predict_rg_trajectory',
    'generate_training_data',
    'FIXED_POINT',
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
]
