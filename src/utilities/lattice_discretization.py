"""
Lattice Discretization Utilities for Intrinsic Resonance Holography v21.0

This module provides finite-volume approximations for computations
on group manifolds.

Key Features:
    - SU(2) and U(1) lattice discretization
    - Discrete Laplacian matrices
    - Finite-volume regularization

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Tuple, Optional, Dict, Any
import numpy as np


__version__ = "21.0.0"


# Theoretical Reference: IRH v21.4



def discretize_SU2(
    n_points: int = 20,
    method: str = 'uniform',
) -> Dict[str, Any]:
    """
    Discretize SU(2) on a finite lattice.
    
    SU(2) is parameterized by unit quaternions (a, b, c, d) with a² + b² + c² + d² = 1.
    Equivalently, by Euler angles (α, β, γ).
    
    Parameters
    ----------
    n_points : int
        Number of lattice points per dimension
    method : str
        'uniform' or 'fibonacci'
        
    Returns
    -------
    Dict
        Lattice points and metadata
    """
    if method == 'uniform':
        # Uniform grid in Euler angles
        n = int(np.cbrt(n_points))
        alpha = np.linspace(0, 2*np.pi, n, endpoint=False)
        beta = np.linspace(0, np.pi, n)
        gamma = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Create grid
        A, B, G = np.meshgrid(alpha, beta, gamma, indexing='ij')
        points = np.stack([A.ravel(), B.ravel(), G.ravel()], axis=-1)
        
        # Haar measure weights
        weights = np.sin(B.ravel()) / (8 * np.pi**2)
        
    elif method == 'fibonacci':
        # Fibonacci sphere for uniform distribution
        # SU(2) ≅ S³, so use 4D Fibonacci
        n = n_points
        points = []
        for i in range(n):
            # Generate point on S³
            phi = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)  # Golden angle
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            
            # Convert to Euler angles
            alpha = phi
            beta = np.arccos(z)
            gamma = (phi + np.pi) % (2 * np.pi)
            points.append([alpha, beta, gamma])
        
        points = np.array(points)
        weights = np.ones(n) / n
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'points': points,
        'weights': weights,
        'n_points': len(points),
        'method': method,
        'manifold': 'SU(2)',
    }


# Theoretical Reference: IRH v21.4



def discretize_U1(
    n_points: int = 100,
) -> Dict[str, Any]:
    """
    Discretize U(1) on a finite lattice.
    
    U(1) is parameterized by angle φ ∈ [0, 2π).
    
    Parameters
    ----------
    n_points : int
        Number of lattice points
        
    Returns
    -------
    Dict
        Lattice points and metadata
    """
    phi = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    weights = np.ones(n_points) / n_points
    
    return {
        'points': phi,
        'weights': weights,
        'n_points': n_points,
        'manifold': 'U(1)',
    }


# Theoretical Reference: IRH v21.4



def laplacian_matrix(
    n_points: int,
    manifold: str = 'U1',
    boundary: str = 'periodic',
) -> np.ndarray:
    """
    Construct discrete Laplacian matrix.
    
    Parameters
    ----------
    n_points : int
        Number of lattice points
    manifold : str
        'U1', 'SU2', or 'G_inf'
    boundary : str
        'periodic' or 'dirichlet'
        
    Returns
    -------
    np.ndarray
        Laplacian matrix
    """
    if manifold == 'U1':
        # 1D Laplacian on circle
        h = 2 * np.pi / n_points
        L = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            L[i, i] = -2.0 / h**2
            L[i, (i + 1) % n_points] = 1.0 / h**2
            L[i, (i - 1) % n_points] = 1.0 / h**2
            
    elif manifold == 'SU2':
        # 3D Laplacian (simplified)
        n = int(np.cbrt(n_points))
        n_points = n**3
        h = 1.0 / n
        
        L = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            L[i, i] = -6.0 / h**2
            
            # Neighbors in 3D cubic lattice
            ix, iy, iz = np.unravel_index(i, (n, n, n))
            
            for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                jx = (ix + di) % n
                jy = (iy + dj) % n
                jz = (iz + dk) % n
                j = np.ravel_multi_index((jx, jy, jz), (n, n, n))
                L[i, j] = 1.0 / h**2
                
    elif manifold == 'G_inf':
        # Product of SU(2) and U(1) Laplacians
        n_SU2 = int(n_points * 0.75)
        n_U1 = int(n_points * 0.25)
        
        L_SU2 = laplacian_matrix(n_SU2, 'SU2', boundary)
        L_U1 = laplacian_matrix(n_U1, 'U1', boundary)
        
        # Tensor sum (Kronecker sum)
        I_SU2 = np.eye(L_SU2.shape[0])
        I_U1 = np.eye(L_U1.shape[0])
        
        L = np.kron(L_SU2, I_U1) + np.kron(I_SU2, L_U1)
        
    else:
        raise ValueError(f"Unknown manifold: {manifold}")
    
    return L


# Theoretical Reference: IRH v21.4



def lattice_volume(
    n_points: int,
    manifold: str = 'U1',
) -> float:
    """
    Compute lattice volume element.
    
    Parameters
    ----------
    n_points : int
        Number of points
    manifold : str
        Manifold type
        
    Returns
    -------
    float
        Volume element
    """
    if manifold == 'U1':
        return 2 * np.pi / n_points
    elif manifold == 'SU2':
        return 1.0 / n_points  # Normalized
    elif manifold == 'G_inf':
        return 2 * np.pi / n_points  # Approximate
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


__all__ = [
    'discretize_SU2',
    'discretize_U1',
    'laplacian_matrix',
    'lattice_volume',
]
