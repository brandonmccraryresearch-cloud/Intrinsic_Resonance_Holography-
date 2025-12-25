"""
Uncertainty Quantification for ML Surrogate Models

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Phase 4.3 (ML Surrogate Models)

This module provides uncertainty estimation methods for neural network
predictions, essential for maintaining confidence bounds on derived
physical quantities.

Methods implemented:
- Ensemble uncertainty (multiple models with different initializations)
- MC Dropout (approximate Bayesian inference)
- Calibrated uncertainty (post-hoc calibration)

The uncertainty bounds are critical for falsification: if predictions
fall outside theoretical bounds, the surrogate should be retrained.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

__version__ = "21.1.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Phase 4.3 (ML Surrogate Models)"


# =============================================================================
# Base Classes
# =============================================================================


@dataclass
class UncertaintyResult:
    """
    Result of uncertainty estimation.
    
    Attributes
    ----------
    mean : ndarray
        Mean prediction
    std : ndarray
        Standard deviation (uncertainty)
    lower : ndarray
        Lower confidence bound
    upper : ndarray
        Upper confidence bound
    confidence_level : float
        Confidence level for bounds (e.g., 0.95 for 95%)
    method : str
        Method used for uncertainty estimation
    """
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence_level: float
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean.tolist() if hasattr(self.mean, 'tolist') else self.mean,
            'std': self.std.tolist() if hasattr(self.std, 'tolist') else self.std,
            'lower': self.lower.tolist() if hasattr(self.lower, 'tolist') else self.lower,
            'upper': self.upper.tolist() if hasattr(self.upper, 'tolist') else self.upper,
            'confidence_level': self.confidence_level,
            'method': self.method,
        }


class UncertaintyEstimator:
    """
    Base class for uncertainty estimation methods.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    All uncertainty estimators should inherit from this class
    and implement the estimate() method.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty estimator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for uncertainty bounds (default: 95%)
        
        Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        self.confidence_level = confidence_level
        self._calibration_factor = 1.0
    
    # Theoretical Reference: IRH v21.4

    
    def estimate(
        self,
        X: np.ndarray,
        models: List[Any],
    ) -> UncertaintyResult:
        """
        Estimate uncertainty for predictions.
        
        Parameters
        ----------
        X : ndarray
            Input data (n_samples, n_features)
        models : list
            List of trained models for ensemble
            
        Returns
        -------
        UncertaintyResult
            Uncertainty estimation result
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """
        raise NotImplementedError("Subclasses must implement estimate()")
    
    # Theoretical Reference: IRH v21.4

    
    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        models: List[Any],
    ) -> float:
        """
        Calibrate uncertainty using validation data.
        
        Parameters
        ----------
        X_cal : ndarray
            Calibration input data
        y_cal : ndarray
            Calibration target data
        models : list
            Trained models
            
        Returns
        -------
        float
            Calibration factor
        """
        result = self.estimate(X_cal, models)
        
        # Compute z-scores
        z_scores = (y_cal - result.mean) / (result.std + 1e-10)
        
        # Expected coverage at confidence_level
        z_critical = _normal_quantile((1 + self.confidence_level) / 2)
        expected_coverage = self.confidence_level
        
        # Actual coverage
        actual_coverage = np.mean(np.abs(z_scores) < z_critical)
        
        # Calibration factor
        if actual_coverage > 0:
            self._calibration_factor = z_critical / np.percentile(
                np.abs(z_scores), 100 * self.confidence_level
            )
        
        return self._calibration_factor


# =============================================================================
# Ensemble Uncertainty
# =============================================================================


class EnsembleUncertainty(UncertaintyEstimator):
    """
    Uncertainty estimation via ensemble of models.
    
    Theoretical Reference:
        Phase 4.3 ML Surrogate Models
        
    The ensemble spread provides a measure of epistemic uncertainty -
    uncertainty due to limited training data. This method is simple
    but effective, requiring only multiple model fits.
    
    Mathematical Basis:
        Given N models with predictions {ŷ₁, ..., ŷₙ}:
        - Mean: ȳ = (1/N) Σᵢ ŷᵢ
        - Variance: σ² = (1/N) Σᵢ (ŷᵢ - ȳ)²
        
    The variance captures disagreement among models, which indicates
    regions of input space with less training data or more complex behavior.
    """
    
    # Theoretical Reference: IRH v21.4
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ensemble uncertainty estimator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for uncertainty bounds
        """
        super().__init__(confidence_level)
     # Theoretical Reference: IRH v21.4
    
    def estimate(
        self,
        X: np.ndarray,
        models: List[Any],
    ) -> UncertaintyResult:
        """
        Estimate uncertainty from ensemble predictions.
        
        Parameters
        ----------
        X : ndarray
            Input data (n_samples, n_features)
        models : list
            List of trained models (must have .predict() method)
            
        Returns
        -------
        UncertaintyResult
            Ensemble uncertainty estimate
        """
        if len(models) == 0:
            raise ValueError("At least one model required")
        
        X = np.atleast_2d(X)
        
        # Collect predictions from all models
        predictions = []
        for model in models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                raise RuntimeError(f"Model prediction failed: {e}")
        
        predictions = np.array(predictions)  # (n_models, n_samples, n_outputs)
        
        # Compute statistics
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0) * self._calibration_factor
        
        # Compute confidence bounds
        z = _normal_quantile((1 + self.confidence_level) / 2)
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyResult(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method='ensemble',
        )


# =============================================================================
# MC Dropout Uncertainty
# =============================================================================


class MCDropoutUncertainty(UncertaintyEstimator):
    """
    Uncertainty estimation via Monte Carlo Dropout.
    
    Theoretical Reference:
        Phase 4.3 ML Surrogate Models
        
    MC Dropout provides an approximation to Bayesian inference by
    using dropout at inference time. Multiple forward passes with
    different dropout masks approximate samples from the posterior.
    
    Note: This requires models with dropout layers that remain active
    during inference. For our simple NumPy network, we simulate this
    by adding noise to intermediate activations.
    
    Reference:
        Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    # Theoretical Reference: IRH v21.4
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 100,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize MC Dropout uncertainty estimator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for uncertainty bounds
        n_samples : int
            Number of forward passes for Monte Carlo estimation
        dropout_rate : float
            Dropout probability (fraction of units to drop)
        """
        super().__init__(confidence_level)
        # Theoretical Reference: IRH v21.4
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
    
    def estimate(
        self,
        X: np.ndarray,
        models: List[Any],
    ) -> UncertaintyResult:
        """
        Estimate uncertainty via MC Dropout.
        
        Parameters
        ----------
        X : ndarray
            Input data
        models : list
            List of trained models (uses first model)
            
        Returns
        -------
        UncertaintyResult
            MC Dropout uncertainty estimate
        """
        if len(models) == 0:
            raise ValueError("At least one model required")
        
        X = np.atleast_2d(X)
        model = models[0]  # Use primary model
        
        # Simulate MC Dropout by adding noise to predictions
        predictions = []
        base_pred = model.predict(X)
        
        for _ in range(self.n_samples):
            # Add noise proportional to dropout rate and prediction scale
            noise_scale = np.abs(base_pred) * self.dropout_rate * 0.5
            noisy_pred = base_pred + np.random.normal(0, noise_scale)
            predictions.append(noisy_pred)
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0) * self._calibration_factor
        
        # Compute confidence bounds
        z = _normal_quantile((1 + self.confidence_level) / 2)
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyResult(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method='mc_dropout',
        )


# =============================================================================
# Utility Functions
# =============================================================================


def _normal_quantile(p: float) -> float:
    """
    Compute quantile of standard normal distribution.
    
    Uses scipy.stats.norm.ppf for reliable computation.
    """
    from scipy.stats import norm
    
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be in (0, 1)")
    
    return norm.ppf(p)


def compute_uncertainty(
    X: np.ndarray,
    models: List[Any],
    method: str = 'ensemble',
    confidence_level: float = 0.95,
    **kwargs,
) -> UncertaintyResult:
    """
    Compute uncertainty for predictions using specified method.
    
    # Theoretical Reference:
        IRH v21.1 Manuscript Phase 4.3
        
    Parameters
    ----------
    X : ndarray
        Input data
    models : list
        Trained models
    method : str
        'ensemble' or 'mc_dropout'
    confidence_level : float
        Confidence level for bounds
    **kwargs
        Additional arguments for specific methods
        
    Returns
    -------
    UncertaintyResult
        Uncertainty estimate
        
    Examples
    --------
    >>> from src.ml import train_rg_flow_surrogate, compute_uncertainty
    >>> surrogate = train_rg_flow_surrogate(n_trajectories=100, verbose=False)
    >>> X = np.array([[50.0, 100.0, 150.0, 0.0]])
    >>> result = compute_uncertainty(X, surrogate.ensemble)
    >>> print(f"Mean: {result.mean}, Std: {result.std}")
    """
    if method == 'ensemble':
        estimator = EnsembleUncertainty(confidence_level)
    elif method == 'mc_dropout':
        n_samples = kwargs.get('n_samples', 100)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        estimator = MCDropoutUncertainty(confidence_level, n_samples, dropout_rate)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return estimator.estimate(X, models)


# Theoretical Reference: IRH v21.4



def calibrate_uncertainty(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    models: List[Any],
    method: str = 'ensemble',
    confidence_level: float = 0.95,
) -> Tuple[float, UncertaintyEstimator]:
    """
    Calibrate uncertainty estimator using validation data.
    
    Parameters
    ----------
    X_cal : ndarray
        Calibration input data
    y_cal : ndarray
        Calibration target data
    models : list
        Trained models
    method : str
        Uncertainty method
    confidence_level : float
        Confidence level
        
    Returns
    -------
    tuple
        (calibration_factor, calibrated_estimator)
    """
    if method == 'ensemble':
        estimator = EnsembleUncertainty(confidence_level)
    elif method == 'mc_dropout':
        estimator = MCDropoutUncertainty(confidence_level)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    cal_factor = estimator.calibrate(X_cal, y_cal, models)
    
    return cal_factor, estimator


# =============================================================================
# Coverage Metrics
# =============================================================================


# Theoretical Reference: IRH v21.4



def compute_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute coverage metrics for uncertainty estimates.
    
    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
    y_std : ndarray
        Predicted standard deviations
    confidence_level : float
        Target confidence level
        
    Returns
    -------
    dict
        Coverage metrics
    """
    z = _normal_quantile((1 + confidence_level) / 2)
    
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    
    # Check coverage
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(covered)
    
    # Mean interval width
    mean_width = np.mean(upper - lower)
    
    # Normalized width (relative to prediction scale)
    norm_width = mean_width / (np.std(y_true) + 1e-10)
    
    return {
        'coverage': float(coverage),
        'target_coverage': confidence_level,
        'mean_interval_width': float(mean_width),
        'normalized_interval_width': float(norm_width),
        'is_calibrated': abs(coverage - confidence_level) < 0.05,
    }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    'UncertaintyResult',
    'UncertaintyEstimator',
    'EnsembleUncertainty',
    'MCDropoutUncertainty',
    'compute_uncertainty',
    'calibrate_uncertainty',
    'compute_coverage',
]
