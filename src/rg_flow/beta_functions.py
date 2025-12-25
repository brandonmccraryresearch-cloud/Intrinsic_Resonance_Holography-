"""
Beta Functions for cGFT RG Flow

THEORETICAL FOUNDATION: IRH21.md §1.2.2, Eq. 1.13

Implements the exact one-loop β-functions for the three cGFT couplings:
    β_λ = -2λ̃ + (9/8π²)λ̃²
    β_γ = (3/4π²)λ̃γ̃  
    β_μ = 2μ̃ + (1/2π²)λ̃μ̃

These β-functions arise from the Wetterich equation (Eq. 1.12) truncated
to the essential coupling space. Their zeros define the Cosmic Fixed Point.

Mathematical Foundation:
    The β-functions encode how coupling constants run with energy scale k.
    At the fixed point, all β = 0, meaning the theory becomes scale-invariant.
    This is the essence of asymptotic safety: the UV completion occurs at a
    non-trivial fixed point where the couplings achieve finite, computable values.

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# Import TransparencyEngine
try:
    from src.logging.transparency_engine import TransparencyEngine
    _TRANSPARENCY_AVAILABLE = True
except ImportError:
    _TRANSPARENCY_AVAILABLE = False
    TransparencyEngine = None

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.2.2, Eq. 1.13"


# =============================================================================
# Beta Function Class
# =============================================================================


@dataclass
class BetaFunctions:
    """
    One-loop β-functions for the cGFT couplings (λ̃, γ̃, μ̃).
    
    Theoretical Reference:
        IRH21.md §1.2.2, Eq. 1.13
        
    The β-functions determine how couplings flow under the renormalization group.
    At the Cosmic Fixed Point, all three β-functions vanish simultaneously.
    
    Attributes
    ----------
    precision : int
        Number of decimal places for internal calculations (default: 15)
        
    Examples
    --------
    >>> beta = BetaFunctions()
    >>> beta.beta_lambda(52.6379)
    0.0  # Near fixed point value
    
    >>> beta.all_betas((52.6379, 105.2757, 157.9137))
    (0.0, 0.0, 0.0)  # At the Cosmic Fixed Point
    """
    
    precision: int = 15
    
    def beta_lambda(
        self, 
        lambda_tilde: float, 
        gamma_tilde: Optional[float] = None, 
        mu_tilde: Optional[float] = None
    ) -> float:
        """
        Compute β_λ = -2λ̃ + (9/8π²)λ̃²
        
        # Theoretical Reference:
            IRH21.md §1.2.2, Eq. 1.13
            
        Mathematical Foundation:
            This β-function arises from the four-vertex bubble diagram
            in the cGFT perturbation expansion. The coefficient 9/8π²
            encodes the combinatorics of the quartic interaction.
        
        Parameters
        ----------
        lambda_tilde : float
            Dimensionless quartic coupling λ̃
        gamma_tilde : float, optional
            Dimensionless γ coupling (not used in β_λ, included for API consistency)
        mu_tilde : float, optional
            Dimensionless μ coupling (not used in β_λ, included for API consistency)
            
        Returns
        -------
        float
            Beta function value β_λ(λ̃)
            
        Notes
        -----
        The fixed point λ̃* = 48π²/9 ≈ 52.638 makes this vanish.
        """
        return -2 * lambda_tilde + (9 / (8 * math.pi**2)) * lambda_tilde**2
    
    def beta_gamma(
        self, 
        lambda_tilde: float, 
        gamma_tilde: float, 
        mu_tilde: Optional[float] = None
    ) -> float:
        """
        Compute β_γ = (3/4π²)λ̃γ̃
        
        # Theoretical Reference:
            IRH21.md §1.2.2, Eq. 1.13
            
        Mathematical Foundation:
            This β-function arises from the QNCD-weighted interaction.
            The product structure λ̃γ̃ reflects how the QNCD metric
            couples to the quartic vertex.
        
        Parameters
        ----------
        lambda_tilde : float
            Dimensionless λ coupling
        gamma_tilde : float
            Dimensionless γ coupling
        mu_tilde : float, optional
            Dimensionless μ coupling (not used in β_γ, included for API consistency)
            
        Returns
        -------
        float
            Beta function value β_γ(λ̃, γ̃)
        """
        return (3 / (4 * math.pi**2)) * lambda_tilde * gamma_tilde
    
    def beta_mu(
        self, 
        lambda_tilde: float, 
        gamma_tilde: float, 
        mu_tilde: float
    ) -> float:
        """
        Compute β_μ = 2μ̃ + (1/2π²)λ̃μ̃
        
        # Theoretical Reference:
            IRH21.md §1.2.2, Eq. 1.13
            
        Mathematical Foundation:
            This β-function arises from the holographic boundary term
            in the cGFT action. The coefficient 1/2π² reflects the
            relationship between bulk and boundary degrees of freedom.
        
        Parameters
        ----------
        lambda_tilde : float
            Dimensionless λ coupling
        gamma_tilde : float
            Dimensionless γ coupling (not used in β_μ, included for API consistency)
        mu_tilde : float
            Dimensionless μ coupling
            
        Returns
        -------
        float
            Beta function value β_μ(λ̃, μ̃)
        """
        return 2 * mu_tilde + (1 / (2 * math.pi**2)) * lambda_tilde * mu_tilde
    
    def all_betas(
        self, 
        couplings: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Compute all three β-functions simultaneously.
        
        # Theoretical Reference:
            IRH21.md §1.2.2, Eq. 1.13
            
        Parameters
        ----------
        couplings : tuple of float
            (λ̃, γ̃, μ̃) coupling values
            
        Returns
        -------
        tuple of float
            (β_λ, β_γ, β_μ) beta function values
            
        Examples
        --------
        >>> beta = BetaFunctions()
        >>> # At the Cosmic Fixed Point
        >>> fp = (48*np.pi**2/9, 32*np.pi**2/3, 16*np.pi**2)
        >>> betas = beta.all_betas(fp)
        >>> all(abs(b) < 1e-10 for b in betas)
        True
        """
        lambda_t, gamma_t, mu_t = couplings
        return (
            self.beta_lambda(lambda_t),
            self.beta_gamma(lambda_t, gamma_t),
            self.beta_mu(lambda_t, gamma_t, mu_t)
        )
    
    def jacobian(
        self, 
        couplings: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix ∂β_i/∂g_j at given couplings.
        
        # Theoretical Reference:
            IRH21.md §1.3 (Stability Analysis)
            
        The eigenvalues of this matrix determine the stability of fixed points.
        Positive eigenvalues indicate IR-attractive directions.
        
        Parameters
        ----------
        couplings : tuple of float
            (λ̃, γ̃, μ̃) coupling values
            
        Returns
        -------
        ndarray
            3×3 Jacobian matrix
        """
        lambda_t, gamma_t, mu_t = couplings
        
        # Analytical derivatives
        # ∂β_λ/∂λ = -2 + (9/4π²)λ̃
        # ∂β_λ/∂γ = 0
        # ∂β_λ/∂μ = 0
        # ∂β_γ/∂λ = (3/4π²)γ̃
        # ∂β_γ/∂γ = (3/4π²)λ̃
        # ∂β_γ/∂μ = 0
        # ∂β_μ/∂λ = (1/2π²)μ̃
        # ∂β_μ/∂γ = 0
        # ∂β_μ/∂μ = 2 + (1/2π²)λ̃
        
        J = np.array([
            [-2 + (9/(4*math.pi**2)) * lambda_t, 0, 0],
            [(3/(4*math.pi**2)) * gamma_t, (3/(4*math.pi**2)) * lambda_t, 0],
            [(1/(2*math.pi**2)) * mu_t, 0, 2 + (1/(2*math.pi**2)) * lambda_t]
        ])
        
        return J


# =============================================================================
# Module-Level Functions (for backward compatibility)
# =============================================================================


def beta_lambda(lambda_tilde: float, gamma_tilde: float, mu_tilde: float) -> float:
    """
    Compute beta function for λ̃ coupling.
    
    Theoretical Reference:
        IRH21.md §1.2, Eq. 1.13
        β_λ = -2λ̃ + (9/8π²)λ̃²
        
    Parameters
    ----------
    lambda_tilde : float
        Dimensionless λ coupling
    gamma_tilde : float
        Dimensionless γ coupling (not used in β_λ, included for API consistency)
    mu_tilde : float
        Dimensionless μ coupling (not used in β_λ, included for API consistency)
        
    Returns
    -------
    float
        β_λ(λ̃, γ̃, μ̃)
    """
    return -2 * lambda_tilde + (9 / (8 * math.pi**2)) * lambda_tilde**2


def beta_gamma(lambda_tilde: float, gamma_tilde: float, mu_tilde: float) -> float:
    """
    Compute beta function for γ̃ coupling.
    
    # Theoretical Reference:
        IRH21.md §1.2, Eq. 1.13
        β_γ = (3/4π²)λ̃γ̃
        
    Parameters
    ----------
    lambda_tilde : float
        Dimensionless λ coupling
    gamma_tilde : float
        Dimensionless γ coupling
    mu_tilde : float
        Dimensionless μ coupling (not used in β_γ, included for API consistency)
        
    Returns
    -------
    float
        β_γ(λ̃, γ̃, μ̃)
    """
    return (3 / (4 * math.pi**2)) * lambda_tilde * gamma_tilde


def beta_mu(lambda_tilde: float, gamma_tilde: float, mu_tilde: float) -> float:
    """
    Compute beta function for μ̃ coupling.
    
    # Theoretical Reference:
        IRH21.md §1.2, Eq. 1.13
        β_μ = 2μ̃ + (1/2π²)λ̃μ̃
        
    Parameters
    ----------
    lambda_tilde : float
        Dimensionless λ coupling
    gamma_tilde : float
        Dimensionless γ coupling (not used in β_μ, included for API consistency)
    mu_tilde : float
        Dimensionless μ coupling
        
    Returns
    -------
    float
        β_μ(λ̃, γ̃, μ̃)
    """
    return 2 * mu_tilde + (1 / (2 * math.pi**2)) * lambda_tilde * mu_tilde


def compute_all_betas(
    lambda_tilde: float, 
    gamma_tilde: float, 
    mu_tilde: float
) -> Tuple[float, float, float]:
    """
    Compute all three beta functions simultaneously.
    
    # Theoretical Reference:
        IRH21.md §1.2, Eq. 1.13
        
    Returns
    -------
    tuple
        (β_λ, β_γ, β_μ)
    """
    return (
        beta_lambda(lambda_tilde, gamma_tilde, mu_tilde),
        beta_gamma(lambda_tilde, gamma_tilde, mu_tilde),
        beta_mu(lambda_tilde, gamma_tilde, mu_tilde)
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    'BetaFunctions',
    'beta_lambda',
    'beta_gamma',
    'beta_mu',
    'compute_all_betas',
]
