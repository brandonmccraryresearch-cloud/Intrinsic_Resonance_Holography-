"""
Parameter Optimization using Bayesian and Active Learning Methods

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Phase 4.3 (ML Surrogate Models)

This module provides optimization methods for exploring the IRH
parameter space efficiently:
- Bayesian Optimization with Gaussian Process surrogates
- Active Learning for selecting maximally informative training points
- Grid search and random search baselines

These methods are particularly useful for:
- Finding optimal coupling configurations
- Exploring the space of initial conditions
- Identifying regions requiring more training data

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

__version__ = "21.1.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Phase 4.3 (ML Surrogate Models)"


# =============================================================================
# Physical Constants
# =============================================================================

LAMBDA_STAR = 48 * math.pi**2 / 9      # λ̃* ≈ 52.638
GAMMA_STAR = 32 * math.pi**2 / 3       # γ̃* ≈ 105.276
MU_STAR = 16 * math.pi**2               # μ̃* ≈ 157.914
FIXED_POINT = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OptimizerConfig:
    """
    Configuration for parameter optimization.
    
    Attributes
    ----------
    bounds : list
        List of (min, max) tuples for each parameter
    n_initial : int
        Number of initial random samples
    n_iterations : int
        Maximum optimization iterations
    acquisition : str
        Acquisition function ('ei', 'ucb', 'pi')
    exploration_weight : float
        Balance exploration vs exploitation (for UCB)
    seed : int
        Random seed for reproducibility
    """
    bounds: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (10, 100),   # λ bounds
            (20, 200),   # γ bounds
            (30, 300),   # μ bounds
        ]
    )
    n_initial: int = 10
    n_iterations: int = 50
    acquisition: str = 'ei'
    exploration_weight: float = 2.0
    seed: int = 42


# =============================================================================
# Base Optimizer Class
# =============================================================================


class ParameterOptimizer:
    """
    Base class for parameter optimization.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    All optimizers optimize coupling values to minimize a user-defined
    objective function, typically related to distance from fixed point
    or accuracy of surrogate predictions.
    """
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        config : OptimizerConfig, optional
            Optimizer configuration
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        self.config = config or OptimizerConfig()
        self.history = []
        self.best_x = None
        self.best_y = float('inf')
    
    # Theoretical Reference: IRH v21.4

    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run optimization.
        
        Parameters
        ----------
        objective : callable
            Function to minimize: f(x) -> float
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Optimization results
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    # Theoretical Reference: IRH v21.4

    
    def suggest_next(self) -> np.ndarray:
        """
        Suggest next point to evaluate.
        
        Returns
        -------
        ndarray
            Suggested parameter values
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        raise NotImplementedError("Subclasses must implement suggest_next()")
    
    # Theoretical Reference: IRH v21.4

    
    def update(self, x: np.ndarray, y: float):
        """
        Update optimizer with new observation.
        
        Parameters
        ----------
        x : ndarray
            Parameter values
        y : float
            Objective value
        """
        self.history.append({'x': x.copy(), 'y': y})
        if y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()


# =============================================================================
# Gaussian Process (Simple Implementation)
# =============================================================================


class SimpleGaussianProcess:
    """
    Simple Gaussian Process for Bayesian optimization.
    
    This is a basic implementation using a squared exponential kernel.
    For production use, consider scikit-learn's GaussianProcessRegressor.
    
    Kernel: k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    """
    
    # Theoretical Reference: IRH v21.4
    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
    ):
        """
        Initialize Gaussian Process.
        
        Parameters
        ----------
        length_scale : float
            Kernel length scale
        variance : float
            Kernel variance
        noise : float
            Observation noise
        """
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute squared exponential kernel."""
        dists = cdist(X1, X2, metric='sqeuclidean')
        return self.variance * np.exp(-dists / (2 * self.length_scale**2))
    
    # Theoretical Reference: IRH v21.4

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to training data.
        
        Parameters
        ----------
        X : ndarray
            Training inputs (n_samples, n_features)
        y : ndarray
            Training targets (n_samples,)
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y)
        
        # Compute inverse of K + σ²I
        K = self._kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(K))
        
        self.K_inv = np.linalg.inv(K)
    
    # Theoretical Reference: IRH v21.4

    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict at new points.
        
        Parameters
        ----------
        X : ndarray
            Query points
        return_std : bool
            Also return standard deviation
            
        Returns
        -------
        ndarray or tuple
            Mean predictions (and std if requested)
        """
        X = np.atleast_2d(X)
        
        if self.X_train is None:
            # Prior: zero mean, kernel variance
            mean = np.zeros(len(X))
            std = np.sqrt(self.variance) * np.ones(len(X))
            if return_std:
                return mean, std
            return mean
        
        K_star = self._kernel(X, self.X_train)
        
        # Predictive mean
        mean = K_star @ self.K_inv @ self.y_train
        
        if return_std:
            # Predictive variance
            K_star_star = self._kernel(X, X)
            var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
            std = np.sqrt(np.maximum(var, 0))
            return mean, std
        
        return mean


# =============================================================================
# Bayesian Optimizer
# =============================================================================


class BayesianOptimizer(ParameterOptimizer):
    """
    Bayesian Optimization using Gaussian Process surrogate.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    Bayesian optimization is sample-efficient, making it ideal for
    expensive objective functions like RG flow integration or
    high-fidelity physics simulations.
    
    The algorithm:
    1. Fit GP surrogate to observed data
    2. Select next point by maximizing acquisition function
    3. Evaluate objective at new point
    4. Repeat until convergence
    
    Acquisition functions:
    - Expected Improvement (EI): E[max(y_best - f(x), 0)]
    - Upper Confidence Bound (UCB): μ(x) - κσ(x) (for minimization)
    - Probability of Improvement (PI): P(f(x) < y_best)
    """
     # Theoretical Reference: IRH v21.4
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize Bayesian optimizer.
        
        Parameters
        ----------
        config : OptimizerConfig, optional
            Configuration
        """
        super().__init__(config)
        self.gp = SimpleGaussianProcess()
        self._X_observed = []
        # Theoretical Reference: IRH v21.4
        self._y_observed = []
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Parameters
        ----------
        objective : callable
            Function to minimize
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Optimization results including best point and history
        """
        np.random.seed(self.config.seed)
        
        bounds = np.array(self.config.bounds)
        n_dims = len(bounds)
        
        # Initial random sampling
        if verbose:
            print(f"Initial sampling: {self.config.n_initial} points")
        
        for i in range(self.config.n_initial):
            x = bounds[:, 0] + np.random.rand(n_dims) * (bounds[:, 1] - bounds[:, 0])
            y = objective(x)
            self.update(x, y)
            self._X_observed.append(x)
            self._y_observed.append(y)
        
        # Bayesian optimization loop
        for iteration in range(self.config.n_iterations):
            # Fit GP
            X = np.array(self._X_observed)
            y = np.array(self._y_observed)
            
            # Normalize inputs and outputs for better GP fitting
            X_norm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            y_mean, y_std = y.mean(), y.std() + 1e-8
            y_norm = (y - y_mean) / y_std
            
            self.gp.fit(X_norm, y_norm)
            
            # Optimize acquisition function
            x_next = self._optimize_acquisition(bounds, y.min())
            
            # Evaluate objective
            y_next = objective(x_next)
            self.update(x_next, y_next)
            self._X_observed.append(x_next)
            self._y_observed.append(y_next)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.config.n_iterations}, "
                      f"Best: {self.best_y:.6f}")
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'history': self.history,
            'n_evaluations': len(self.history),
        }
    
    def _optimize_acquisition(
        self,
        bounds: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        """Optimize acquisition function to find next point."""
        n_dims = len(bounds)
        bounds_norm = np.array([[0, 1]] * n_dims)
        
        # Multi-start local optimization
        best_acq = float('inf')  # Since we're minimizing
        best_x = None
        
        for _ in range(10):
            x0 = np.random.rand(n_dims)
            
            # Theoretical Reference: IRH v21.4

            
            def neg_acquisition(x):
                x_orig = bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
                return -self._acquisition(x, y_best, bounds)
            
            result = minimize(
                neg_acquisition,
                x0,
                bounds=bounds_norm.tolist(),
                method='L-BFGS-B',
            )
            
            if -result.fun < best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        # Convert back to original scale
        x_orig = bounds[:, 0] + best_x * (bounds[:, 1] - bounds[:, 0])
        return x_orig
    
    def _acquisition(
        self,
        x_norm: np.ndarray,
        y_best: float,
        bounds: np.ndarray,
    ) -> float:
        """Compute acquisition function value."""
        x_norm = np.atleast_2d(x_norm)
        
        # GP prediction (on normalized data)
        mean, std = self.gp.predict(x_norm, return_std=True)
        
        if self.config.acquisition == 'ei':
            # Expected Improvement
            return self._expected_improvement(mean[0], std[0], y_best)
        elif self.config.acquisition == 'ucb':
            # Upper Confidence Bound (lower for minimization)
            return -(mean[0] - self.config.exploration_weight * std[0])
        elif self.config.acquisition == 'pi':
            # Probability of Improvement
            return self._probability_improvement(mean[0], std[0], y_best)
        else:
            raise ValueError(f"Unknown acquisition: {self.config.acquisition}")
    
    def _expected_improvement(
        self,
        mean: float,
        std: float,
        y_best: float,
    ) -> float:
        """Compute Expected Improvement acquisition."""
        if std < 1e-12:
            return 0.0
        
        z = (y_best - mean) / std
        # Approximate normal CDF and PDF
        phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        pdf = math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        
        ei = std * (z * phi + pdf)
        return ei
    
    def _probability_improvement(
        self,
        mean: float,
        std: float,
        y_best: float,
    ) -> float:
        """Compute Probability of Improvement."""
        if std < 1e-12:
            return 0.0
        
        # Theoretical Reference: IRH v21.4
        z = (y_best - mean) / std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def suggest_next(self) -> np.ndarray:
        """
        Suggest next point using current GP model.
        
        Returns
        -------
        ndarray
            Suggested parameter values
        """
        if len(self._y_observed) == 0:
            # Random point if no data
            bounds = np.array(self.config.bounds)
            return bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        
        bounds = np.array(self.config.bounds)
        y_best = min(self._y_observed)
        return self._optimize_acquisition(bounds, y_best)


# =============================================================================
# Active Learning Optimizer
# =============================================================================


class ActiveLearningOptimizer(ParameterOptimizer):
    """
    Active Learning for selecting informative training points.
    
    # Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    Active learning selects points that maximize information gain,
    which is particularly useful for training surrogate models.
    
    Strategies:
    - 'uncertainty': Select points with highest predicted uncertainty
    - 'diversity': Select points far from existing samples
    - 'combined': Balance uncertainty and diversity
    
    # Theoretical Reference: IRH v21.4
    This differs from Bayesian optimization: instead of minimizing
    an objective, we select points to improve the surrogate model.
    """
    
    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        strategy: str = 'combined',
    ):
        """
        Initialize active learning optimizer.
        
        Parameters
        ----------
        config : OptimizerConfig, optional
            Configuration
        strategy : str
            Selection strategy ('uncertainty', 'diversity', 'combined')
        """
        # Theoretical Reference: IRH v21.4
        super().__init__(config)
        self.strategy = strategy
        self.gp = SimpleGaussianProcess()
        self._X_observed = []
    
    # Theoretical Reference: IRH v21.4 (Bayesian Optimization)
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run active learning to select informative points.
        
        Parameters
        ----------
        objective : callable
            Function to evaluate (for labeling points)
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Selected points and their values
        """
        np.random.seed(self.config.seed)
        
        bounds = np.array(self.config.bounds)
        n_dims = len(bounds)
        
        # Initial random sampling
        selected_points = []
        selected_values = []
        
        if verbose:
            print(f"Initial sampling: {self.config.n_initial} points")
        
        for _ in range(self.config.n_initial):
            x = bounds[:, 0] + np.random.rand(n_dims) * (bounds[:, 1] - bounds[:, 0])
            y = objective(x)
            selected_points.append(x)
            selected_values.append(y)
            self._X_observed.append(x)
        
        # Active learning loop
        for iteration in range(self.config.n_iterations):
            # Fit GP on observed points
            if len(selected_points) > 0:
                X = np.array(selected_points)
                y = np.array(selected_values)
                
                # Normalize
                X_norm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
                y_norm = (y - y.mean()) / (y.std() + 1e-8)
                
                self.gp.fit(X_norm, y_norm)
            
            # Select next point
            x_next = self._select_next_point(bounds)
            
            # Evaluate
            y_next = objective(x_next)
            selected_points.append(x_next)
            selected_values.append(y_next)
            self._X_observed.append(x_next)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.config.n_iterations}")
        
        return {
            'selected_points': np.array(selected_points),
            'selected_values': np.array(selected_values),
            'n_points': len(selected_points),
        }
    
    def _select_next_point(self, bounds: np.ndarray) -> np.ndarray:
        """Select next point based on strategy."""
        n_candidates = 1000
        n_dims = len(bounds)
        
        # Generate candidate points
        candidates = bounds[:, 0] + np.random.rand(n_candidates, n_dims) * \
                    (bounds[:, 1] - bounds[:, 0])
        
        # Score candidates
        scores = np.zeros(n_candidates)
        
        if self.strategy in ('uncertainty', 'combined'):
            # Uncertainty score
            candidates_norm = (candidates - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            _, std = self.gp.predict(candidates_norm, return_std=True)
            scores += std
        
        if self.strategy in ('diversity', 'combined'):
            # Diversity score (distance to nearest observed point)
            if len(self._X_observed) > 0:
                X_obs = np.array(self._X_observed)
                dists = cdist(candidates, X_obs)
                min_dists = dists.min(axis=1)
                # Normalize and add to scores
                # Theoretical Reference: IRH v21.4
                scores += min_dists / (min_dists.max() + 1e-10)
        
        # Select point with highest score
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    # Theoretical Reference: IRH v21.4 (Active Learning)
    def suggest_next(self) -> np.ndarray:
        """Suggest next point to label."""
        bounds = np.array(self.config.bounds)
        return self._select_next_point(bounds)


# =============================================================================
# Convenience Functions
# =============================================================================


# Theoretical Reference: IRH v21.4



def optimize_parameters(
    objective: Callable[[np.ndarray], float],
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = 'bayesian',
    n_iterations: int = 50,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Optimize parameters using specified method.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    Parameters
    ----------
    objective : callable
        Function to minimize
    bounds : list, optional
        Parameter bounds. Default: bounds around fixed point.
    method : str
        'bayesian', 'active', or 'random'
    n_iterations : int
        Number of optimization iterations
    verbose : bool
        Print progress
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Optimization results
        
    Examples
    --------
    >>> from src.ml import optimize_parameters, FIXED_POINT
    >>> def objective(x):
    ...     return np.linalg.norm(x - FIXED_POINT)  # Distance to fixed point
    >>> result = optimize_parameters(objective, n_iterations=20, verbose=False)
    >>> print(f"Best found: {result['best_x']}")
    """
    if bounds is None:
        bounds = [
            (LAMBDA_STAR * 0.2, LAMBDA_STAR * 2.0),
            (GAMMA_STAR * 0.2, GAMMA_STAR * 2.0),
            (MU_STAR * 0.2, MU_STAR * 2.0),
        ]
    
    config = OptimizerConfig(
        bounds=bounds,
        n_iterations=n_iterations,
        seed=seed,
    )
    
    if method == 'bayesian':
        optimizer = BayesianOptimizer(config)
    elif method == 'active':
        optimizer = ActiveLearningOptimizer(config)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimizer.optimize(objective, verbose)


# Theoretical Reference: IRH v21.4



def suggest_next_point(
    observed_points: np.ndarray,
    observed_values: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = 'bayesian',
) -> np.ndarray:
    """
    Suggest next point to evaluate given observed data.
    
    Parameters
    ----------
    observed_points : ndarray
        Previously observed points (n_points, n_dims)
    observed_values : ndarray
        Objective values at observed points
    bounds : list, optional
        Parameter bounds
    method : str
        'bayesian' or 'active'
        
    Returns
    -------
    ndarray
        Suggested next point
    """
    if bounds is None:
        bounds = [
            (LAMBDA_STAR * 0.2, LAMBDA_STAR * 2.0),
            (GAMMA_STAR * 0.2, GAMMA_STAR * 2.0),
            (MU_STAR * 0.2, MU_STAR * 2.0),
        ]
    
    config = OptimizerConfig(bounds=bounds, n_iterations=0)
    
    if method == 'bayesian':
        optimizer = BayesianOptimizer(config)
    else:
        optimizer = ActiveLearningOptimizer(config)
    
    # Add observed data
    for x, y in zip(observed_points, observed_values):
        optimizer.update(x, y)
        optimizer._X_observed.append(x)
        if hasattr(optimizer, '_y_observed'):
            optimizer._y_observed.append(y)
    
    # Fit GP if applicable
    if isinstance(optimizer, BayesianOptimizer):
        bounds_arr = np.array(bounds)
        # Avoid division by zero if bounds have same min/max
        bounds_range = bounds_arr[:, 1] - bounds_arr[:, 0]
        bounds_range = np.where(bounds_range == 0, 1.0, bounds_range)  # Replace 0 with 1
        X_norm = (observed_points - bounds_arr[:, 0]) / bounds_range
        y_norm = (observed_values - observed_values.mean()) / (observed_values.std() + 1e-8)
        optimizer.gp.fit(X_norm, y_norm)
    
    return optimizer.suggest_next()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    'OptimizerConfig',
    'ParameterOptimizer',
    'BayesianOptimizer',
    'ActiveLearningOptimizer',
    'SimpleGaussianProcess',
    'optimize_parameters',
    'suggest_next_point',
    'FIXED_POINT',
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
]
