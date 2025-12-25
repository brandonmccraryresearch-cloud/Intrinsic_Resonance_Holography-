"""
Integration Utilities for Intrinsic Resonance Holography v21.0

This module provides numerical integration on group manifolds
and general quadrature routines.

Key Features:
    - Integration on SU(2) and U(1)
    - Monte Carlo integration
    - Adaptive quadrature

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np
from scipy import integrate


__version__ = "21.0.0"


# Theoretical Reference: IRH v21.4



def integrate_SU2(
    f: Callable[[float, float, float], float],
    n_points: int = 100,
    method: str = 'monte_carlo'
) -> Dict[str, Any]:
    """
    Integrate a function over SU(2) with Haar measure.
    
    SU(2) is parameterized by Euler angles (α, β, γ).
    Haar measure: dg = (1/8π²) sin(β) dα dβ dγ
    
    Parameters
    ----------
    f : Callable
        Function f(α, β, γ) to integrate
    n_points : int
        Number of sample points
    method : str
        'monte_carlo' or 'quadrature'
        
    Returns
    -------
    Dict
        Integration result with 'value', 'error', 'n_points'
    """
    if method == 'monte_carlo':
        # Monte Carlo integration with Haar measure
        rng = np.random.default_rng()
        
        # Sample uniformly on SU(2)
        alpha = rng.uniform(0, 2*np.pi, n_points)
        beta = np.arccos(rng.uniform(-1, 1, n_points))  # For Haar measure
        gamma = rng.uniform(0, 2*np.pi, n_points)
        
        values = np.array([f(a, b, g) for a, b, g in zip(alpha, beta, gamma)])
        
        # Volume of SU(2) with Haar normalization
        volume = 1.0  # Normalized
        
        result = volume * np.mean(values)
        error = volume * np.std(values) / np.sqrt(n_points)
        
    elif method == 'quadrature':
        # Gauss-Legendre quadrature
        # Theoretical Reference: IRH v21.4

        def integrand(alpha, beta, gamma):
            # Haar measure factor
            return f(alpha, beta, gamma) * np.sin(beta) / (8 * np.pi**2)
        
        result, error = integrate.tplquad(
            integrand,
            0, 2*np.pi,  # gamma
            lambda g: 0, lambda g: np.pi,  # beta
            lambda g, b: 0, lambda g, b: 2*np.pi,  # alpha
        )
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'value': result,
        'error': error,
        'n_points': n_points,
        'method': method,
    }


# Theoretical Reference: IRH v21.4



def integrate_U1(
    f: Callable[[float], float],
    n_points: int = 100,
    method: str = 'quadrature'
) -> Dict[str, Any]:
    """
    Integrate a function over U(1) with Haar measure.
    
    U(1) is parameterized by angle φ ∈ [0, 2π).
    Haar measure: dg = (1/2π) dφ
    
    Parameters
    ----------
    f : Callable
        Function f(φ) to integrate
    n_points : int
        Number of sample points
    method : str
        'quadrature' or 'monte_carlo'
        
    Returns
    -------
    Dict
        Integration result
    """
    if method == 'quadrature':
        def integrand(phi):
            """
            # Theoretical Reference: IRH v21.4
            """
            return f(phi) / (2 * np.pi)
        
        result, error = integrate.quad(integrand, 0, 2*np.pi)
        
    elif method == 'monte_carlo':
        rng = np.random.default_rng()
        phi = rng.uniform(0, 2*np.pi, n_points)
        values = np.array([f(p) for p in phi])
        
        result = np.mean(values)
        error = np.std(values) / np.sqrt(n_points)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'value': result,
        'error': error,
        'n_points': n_points,
        'method': method,
    }


# Theoretical Reference: IRH v21.4



def integrate_G_inf(
    f: Callable[[float, float, float, float], float],
    n_points: int = 1000,
    method: str = 'monte_carlo'
) -> Dict[str, Any]:
    """
    Integrate a function over G_inf = SU(2) × U(1).
    
    Parameters
    ----------
    f : Callable
        Function f(α, β, γ, φ) on G_inf
    n_points : int
        Number of sample points
    method : str
        Integration method
        
    Returns
    -------
    Dict
        Integration result
    """
    rng = np.random.default_rng()
    
    # Sample SU(2)
    alpha = rng.uniform(0, 2*np.pi, n_points)
    beta = np.arccos(rng.uniform(-1, 1, n_points))
    gamma = rng.uniform(0, 2*np.pi, n_points)
    
    # Sample U(1)
    phi = rng.uniform(0, 2*np.pi, n_points)
    
    values = np.array([
        f(a, b, g, p) for a, b, g, p in zip(alpha, beta, gamma, phi)
    ])
    
    result = np.mean(values)
    error = np.std(values) / np.sqrt(n_points)
    
    return {
        'value': result,
        'error': error,
        'n_points': n_points,
        'method': method,
    }


# Theoretical Reference: IRH v21.4



def monte_carlo_integrate(
    f: Callable,
    bounds: list,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    General Monte Carlo integration.
    
    Parameters
    ----------
    f : Callable
        Function to integrate
    bounds : list
        List of (low, high) tuples for each dimension
    n_samples : int
        Number of samples
    seed : int, optional
        Random seed
        
    Returns
    -------
    Dict
        Integration result
    """
    rng = np.random.default_rng(seed)
    n_dims = len(bounds)
    
    # Sample uniformly in hypercube
    samples = np.array([
        rng.uniform(low, high, n_samples)
        for low, high in bounds
    ]).T
    
    # Evaluate function
    values = np.array([f(*s) for s in samples])
    
    # Volume of integration domain
    volume = np.prod([high - low for low, high in bounds])
    
    result = volume * np.mean(values)
    error = volume * np.std(values) / np.sqrt(n_samples)
    
    return {
        'value': result,
        'error': error,
        'n_samples': n_samples,
        'n_dims': n_dims,
    }


__all__ = [
    'integrate_SU2',
    'integrate_U1',
    'integrate_G_inf',
    'monte_carlo_integrate',
]
