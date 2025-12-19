"""
cGFT Interactions Module for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §1.1

This module implements the interaction kernels for cGFT,
including QNCD-weighted kernels and phase coherence.

Key Components:
    - K(g,g'): QNCD-weighted interaction kernel (Eq. 1.3)
    - Phase coherence terms
    - Multi-particle interaction vertices

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Callable, Dict, Optional, Tuple, Any, List
import numpy as np


__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §1.1"


def QNCD(
    g1: np.ndarray,
    g2: np.ndarray,
) -> float:
    """
    Compute Quantum Normalized Compression Distance.
    
    THEORETICAL REFERENCE: IRH v21.1 Appendix A
    
    QNCD is a metric on G_inf that satisfies:
    1. Positivity: d(g,g') >= 0
    2. Symmetry: d(g,g') = d(g',g)
    3. Triangle inequality
    4. Bi-invariance: d(kg,kg') = d(gk,g'k) = d(g,g')
    
    Parameters
    ----------
    g1, g2 : np.ndarray
        Group elements (as quaternions or Euler angles)
        
    Returns
    -------
    float
        QNCD distance
    """
    # Simplified implementation using Frobenius norm
    # Full version would use compression-based metric
    
    g1_flat = np.asarray(g1).ravel()
    g2_flat = np.asarray(g2).ravel()
    
    # Normalize to same length
    max_len = max(len(g1_flat), len(g2_flat))
    g1_padded = np.zeros(max_len)
    g2_padded = np.zeros(max_len)
    g1_padded[:len(g1_flat)] = g1_flat
    g2_padded[:len(g2_flat)] = g2_flat
    
    # Euclidean distance normalized
    diff = g1_padded - g2_padded
    norm = np.linalg.norm(diff)
    
    # Normalize to [0, 1]
    max_norm = max(np.linalg.norm(g1_padded), np.linalg.norm(g2_padded))
    if max_norm > 0:
        return norm / (2 * max_norm)
    return 0.0


def interaction_kernel(
    g1: np.ndarray,
    g2: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """
    Compute interaction kernel K(g,g').
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.3
    
    The interaction kernel couples field modes through:
        K(g,g') = exp(-γ × QNCD(g,g'))
    
    Parameters
    ----------
    g1, g2 : np.ndarray
        Group elements
    gamma : float
        QNCD coupling strength
        
    Returns
    -------
    float
        Kernel value
    """
    distance = QNCD(g1, g2)
    return np.exp(-gamma * distance)


def QNCD_weighted_kernel(
    g1: np.ndarray,
    g2: np.ndarray,
    gamma: float = 1.0,
    mu: float = 1.0,
) -> complex:
    """
    Compute QNCD-weighted interaction kernel with phase.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.3
    
    K_QNCD(g,g') = exp(-γ × QNCD(g,g')) × exp(iμ × θ(g,g'))
    
    where θ(g,g') is a phase coherence term.
    
    Parameters
    ----------
    g1, g2 : np.ndarray
        Group elements
    gamma : float
        QNCD coupling
    mu : float
        Phase coupling
        
    Returns
    -------
    complex
        Kernel value
    """
    distance = QNCD(g1, g2)
    amplitude = np.exp(-gamma * distance)
    
    # Phase coherence from angle between elements
    g1_flat = np.asarray(g1).ravel()
    g2_flat = np.asarray(g2).ravel()
    
    # Compute relative phase
    if len(g1_flat) > 0 and len(g2_flat) > 0:
        min_len = min(len(g1_flat), len(g2_flat))
        dot = np.dot(g1_flat[:min_len], g2_flat[:min_len])
        norm1 = np.linalg.norm(g1_flat)
        norm2 = np.linalg.norm(g2_flat)
        if norm1 > 0 and norm2 > 0:
            cos_theta = np.clip(dot / (norm1 * norm2), -1, 1)
            theta = np.arccos(cos_theta)
        else:
            theta = 0.0
    else:
        theta = 0.0
    
    phase = np.exp(1j * mu * theta)
    
    return complex(amplitude * phase)


def four_point_interaction(
    phi: np.ndarray,
    g1: int, g2: int, g3: int, g4: int,
    lambda_coupling: float,
) -> complex:
    """
    Compute four-point interaction vertex.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.2
    
    λ̃ × φ̄(g₁,g₂,g₃,g₄)φ(g₁,g₂,g₃,g₄)φ̄(g₁,g₂,g₃,g₄)φ(g₁,g₂,g₃,g₄)
    
    Parameters
    ----------
    phi : np.ndarray
        Field configuration
    g1, g2, g3, g4 : int
        Group element indices
    lambda_coupling : float
        Quartic coupling
        
    Returns
    -------
    complex
        Interaction term
    """
    # Get field values
    if phi.ndim >= 4:
        phi_val = phi[g1 % phi.shape[0], g2 % phi.shape[1], 
                      g3 % phi.shape[2], g4 % phi.shape[3]]
    else:
        # Simplified for lower-dimensional fields
        phi_val = phi[g1 % len(phi)]
    
    phi_bar = np.conj(phi_val)
    
    # Quartic interaction
    return lambda_coupling * phi_bar * phi_val * phi_bar * phi_val


def holographic_interaction(
    phi: np.ndarray,
    mu_coupling: float,
) -> float:
    """
    Compute holographic interaction term.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.4
    
    S_hol = μ̃ × H[φ]
    
    where H is the harmony functional.
    
    Parameters
    ----------
    phi : np.ndarray
        Field configuration
    mu_coupling : float
        Holographic coupling
        
    Returns
    -------
    float
        Holographic interaction energy
    """
    # Compute harmony functional (simplified)
    # Full version would compute spectral zeta function
    
    phi_flat = phi.ravel()
    
    # Phase coherence measure
    phases = np.angle(phi_flat)
    phase_variance = np.var(phases)
    
    # Amplitude regularity
    amplitudes = np.abs(phi_flat)
    amplitude_variance = np.var(amplitudes)
    
    # Harmony: low variance = high harmony
    harmony = 1.0 / (1.0 + phase_variance + amplitude_variance)
    
    return float(mu_coupling * harmony)


def compute_interaction_matrix(
    n_modes: int,
    gamma: float = 1.0,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Compute full interaction matrix K_{ij}.
    
    Parameters
    ----------
    n_modes : int
        Number of field modes
    gamma : float
        QNCD coupling
    mu : float
        Phase coupling
        
    Returns
    -------
    np.ndarray
        n_modes × n_modes interaction matrix
    """
    K = np.zeros((n_modes, n_modes), dtype=complex)
    
    for i in range(n_modes):
        for j in range(n_modes):
            # Map mode indices to group elements
            g_i = np.array([i / n_modes * 2 * np.pi])
            g_j = np.array([j / n_modes * 2 * np.pi])
            
            K[i, j] = QNCD_weighted_kernel(g_i, g_j, gamma, mu)
    
    return K


__all__ = [
    # QNCD
    'QNCD',
    
    # Kernels
    'interaction_kernel',
    'QNCD_weighted_kernel',
    
    # Interactions
    'four_point_interaction',
    'holographic_interaction',
    'compute_interaction_matrix',
]
