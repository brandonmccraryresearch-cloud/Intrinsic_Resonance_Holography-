"""
cGFT Operators Module for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §1.1

This module implements the kinetic operator infrastructure for cGFT,
including the Laplace-Beltrami operator on SU(2) and functional derivatives.

Key Operators:
    - Δₐ^(i): Laplace-Beltrami on SU(2) for each argument
    - δ/δφ: Functional derivative
    - Hessian: Second functional derivative

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np


__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §1.1"


# SU(2) generators (Pauli matrices / 2)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex) / 2
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex) / 2

SU2_GENERATORS = [SIGMA_X, SIGMA_Y, SIGMA_Z]


def laplace_beltrami_SU2(
    field: np.ndarray,
    generator_idx: int = 0,
    arg_idx: int = 0,
    lattice_spacing: float = 0.1,
) -> np.ndarray:
    """
    Compute Laplace-Beltrami operator on SU(2).
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.1
    
    The kinetic term uses:
        Σₐ Σᵢ Δₐ^(i)
    
    where Δₐ^(i) is the Laplacian in generator direction a
    acting on argument i of the field.
    
    Parameters
    ----------
    field : np.ndarray
        Field values φ on lattice
    generator_idx : int
        Which SU(2) generator (0, 1, 2 for X, Y, Z)
    arg_idx : int
        Which field argument (0, 1, 2, 3)
    lattice_spacing : float
        Lattice spacing h
        
    Returns
    -------
    np.ndarray
        Laplacian applied to field
    """
    h = lattice_spacing
    
    # For a field on a lattice, compute second derivative
    # Using central difference: ∂²f/∂x² ≈ (f_{i+1} - 2f_i + f_{i-1})/h²
    
    result = np.zeros_like(field, dtype=complex)
    
    if field.ndim == 1:
        # 1D case
        n = len(field)
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            result[i] = (field[ip] - 2*field[i] + field[im]) / h**2
            
    elif field.ndim == 4:
        # 4D case: φ(g₁, g₂, g₃, g₄)
        shape = field.shape
        
        for i in range(shape[arg_idx]):
            # Create slice along the specified argument
            slices_center = [slice(None)] * 4
            slices_center[arg_idx] = i
            
            slices_plus = [slice(None)] * 4
            slices_plus[arg_idx] = (i + 1) % shape[arg_idx]
            
            slices_minus = [slice(None)] * 4
            slices_minus[arg_idx] = (i - 1) % shape[arg_idx]
            
            result[tuple(slices_center)] = (
                field[tuple(slices_plus)] - 2*field[tuple(slices_center)] + 
                field[tuple(slices_minus)]
            ) / h**2
    else:
        # Apply along specified axis
        result = np.gradient(np.gradient(field, h, axis=arg_idx), h, axis=arg_idx)
    
    return result


def sum_laplacians(
    field: np.ndarray,
    lattice_spacing: float = 0.1,
) -> np.ndarray:
    """
    Compute full kinetic operator: Σₐ Σᵢ Δₐ^(i).
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.1
    
    Parameters
    ----------
    field : np.ndarray
        Field values
    lattice_spacing : float
        Lattice spacing
        
    Returns
    -------
    np.ndarray
        Total Laplacian
    """
    result = np.zeros_like(field, dtype=complex)
    
    n_generators = 3  # SU(2) has 3 generators
    n_args = 4 if field.ndim >= 4 else min(field.ndim, 4)
    
    for a in range(n_generators):
        for i in range(n_args):
            result += laplace_beltrami_SU2(field, a, i, lattice_spacing)
    
    return result


def functional_derivative(
    action: Callable[[np.ndarray], float],
    field: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute functional derivative δS/δφ.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2
    
    Uses numerical differentiation:
        (δS/δφ)_i ≈ (S[φ + εδ_i] - S[φ - εδ_i]) / (2ε)
    
    Parameters
    ----------
    action : Callable
        Action functional S[φ]
    field : np.ndarray
        Field configuration
    eps : float
        Numerical step size
        
    Returns
    -------
    np.ndarray
        Functional derivative
    """
    result = np.zeros_like(field, dtype=complex)
    
    flat_field = field.ravel()
    flat_result = result.ravel()
    
    for i in range(len(flat_field)):
        # Perturb +
        field_plus = flat_field.copy()
        field_plus[i] += eps
        S_plus = action(field_plus.reshape(field.shape))
        
        # Perturb -
        field_minus = flat_field.copy()
        field_minus[i] -= eps
        S_minus = action(field_minus.reshape(field.shape))
        
        flat_result[i] = (S_plus - S_minus) / (2 * eps)
    
    return flat_result.reshape(field.shape)


def hessian_operator(
    action: Callable[[np.ndarray], float],
    field: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Hessian (second functional derivative) δ²S/δφδφ.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12
    
    Parameters
    ----------
    action : Callable
        Action functional
    field : np.ndarray
        Field configuration
    eps : float
        Numerical step size
        
    Returns
    -------
    np.ndarray
        Hessian matrix
    """
    n = field.size
    hessian = np.zeros((n, n), dtype=complex)
    
    flat_field = field.ravel()
    
    for i in range(n):
        for j in range(i, n):
            # Four-point stencil
            f_pp = flat_field.copy()
            f_pp[i] += eps
            f_pp[j] += eps
            
            f_pm = flat_field.copy()
            f_pm[i] += eps
            f_pm[j] -= eps
            
            f_mp = flat_field.copy()
            f_mp[i] -= eps
            f_mp[j] += eps
            
            f_mm = flat_field.copy()
            f_mm[i] -= eps
            f_mm[j] -= eps
            
            S_pp = action(f_pp.reshape(field.shape))
            S_pm = action(f_pm.reshape(field.shape))
            S_mp = action(f_mp.reshape(field.shape))
            S_mm = action(f_mm.reshape(field.shape))
            
            d2S = (S_pp - S_pm - S_mp + S_mm) / (4 * eps**2)
            
            hessian[i, j] = d2S
            hessian[j, i] = d2S
    
    return hessian


# Theoretical Reference: IRH v21.4



def casimir_operator(
    field: np.ndarray,
    representation: str = 'fundamental',
) -> np.ndarray:
    """
    Compute SU(2) Casimir operator C² = J·J.
    
    Parameters
    ----------
    field : np.ndarray
        Field values
    representation : str
        'fundamental', 'adjoint', etc.
        
    Returns
    -------
    np.ndarray
        Casimir eigenvalue times field
    """
    if representation == 'fundamental':
        # j = 1/2: C² = j(j+1) = 3/4
        return 0.75 * field
    elif representation == 'adjoint':
        # j = 1: C² = 2
        return 2.0 * field
    else:
        raise ValueError(f"Unknown representation: {representation}")


__all__ = [
    # Generators
    'SU2_GENERATORS',
    'SIGMA_X',
    'SIGMA_Y',
    'SIGMA_Z',
    
    # Operators
    'laplace_beltrami_SU2',
    'sum_laplacians',
    'functional_derivative',
    'hessian_operator',
    'casimir_operator',
]
