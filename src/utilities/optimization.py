"""
Optimization Utilities for Intrinsic Resonance Holography v21.0

This module provides numerical optimization routines including
fixed-point solvers and minimizers.

Key Features:
    - Newton-Raphson for fixed points
    - Functional minimization
    - Root finding

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Callable, Tuple, Optional, Dict, Any, List
import numpy as np
from scipy import optimize


__version__ = "21.0.0"


# Theoretical Reference: IRH v21.4



def find_fixed_point_newton(
    f: Callable[[np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
    jacobian: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Find fixed point using Newton-Raphson method.
    
    Solves f(x) = x, equivalently g(x) = f(x) - x = 0.
    
    Parameters
    ----------
    f : Callable
        Function whose fixed point to find
    initial_guess : np.ndarray
        Initial guess
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum iterations
    jacobian : Callable, optional
        Jacobian of f
        
    Returns
    -------
    Dict
        Fixed point result
    """
    # Theoretical Reference: IRH v21.4

    def g(x):
        return f(x) - x
    
    x = np.asarray(initial_guess, dtype=float)
    
    for i in range(max_iter):
        fx = f(x)
        residual = fx - x
        
        if np.linalg.norm(residual) < tol:
            return {
                'fixed_point': x,
                'converged': True,
                'iterations': i + 1,
                'residual': np.linalg.norm(residual),
            }
        
        # Newton step
        if jacobian is not None:
            J = jacobian(x) - np.eye(len(x))
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                delta = -0.1 * residual
        else:
            # Numerical Jacobian
            eps = 1e-8
            J = np.zeros((len(x), len(x)))
            for j in range(len(x)):
                x_plus = x.copy()
                x_plus[j] += eps
                J[:, j] = (g(x_plus) - g(x)) / eps
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                delta = -0.1 * residual
        
        x = x + delta
    
    return {
        'fixed_point': x,
        'converged': False,
        'iterations': max_iter,
        'residual': np.linalg.norm(f(x) - x),
    }


# Theoretical Reference: IRH v21.4



def minimize_functional(
    functional: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    method: str = 'BFGS',
    bounds: Optional[List[Tuple[float, float]]] = None,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Minimize a functional.
    
    Parameters
    ----------
    functional : Callable
        Functional to minimize
    initial_guess : np.ndarray
        Initial guess
    method : str
        Optimization method
    bounds : list, optional
        Parameter bounds
    tol : float
        Tolerance
        
    Returns
    -------
    Dict
        Optimization result
    """
    result = optimize.minimize(
        functional,
        initial_guess,
        method=method,
        bounds=bounds,
        tol=tol,
    )
    
    return {
        'minimum': result.x,
        'value': result.fun,
        'converged': result.success,
        'iterations': result.nit if hasattr(result, 'nit') else None,
        'message': result.message,
    }


# Theoretical Reference: IRH v21.4



def root_find(
    f: Callable[[np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    method: str = 'hybr',
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Find root of a vector function.
    
    Solves f(x) = 0.
    
    Parameters
    ----------
    f : Callable
        Function whose root to find
    initial_guess : np.ndarray
        Initial guess
    method : str
        Root-finding method
    tol : float
        Tolerance
        
    Returns
    -------
    Dict
        Root-finding result
    """
    result = optimize.root(
        f,
        initial_guess,
        method=method,
        tol=tol,
    )
    
    return {
        'root': result.x,
        'residual': np.linalg.norm(result.fun),
        'converged': result.success,
        'message': result.message,
    }


__all__ = [
    'find_fixed_point_newton',
    'minimize_functional',
    'root_find',
]
