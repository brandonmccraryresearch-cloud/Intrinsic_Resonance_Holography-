"""
Cosmic Fixed Point Computation

THEORETICAL FOUNDATION: IRH21.md §1.2.3, Eq. 1.14

This module computes and verifies the Cosmic Fixed Point - the unique
non-Gaussian infrared fixed point where all β-functions vanish.

Fixed-point values (Eq. 1.14):
    λ̃* = 48π²/9 ≈ 52.637
    γ̃* = 32π²/3 ≈ 105.276
    μ̃* = 16π²  ≈ 157.914

Mathematical Foundation:
    The Cosmic Fixed Point is the unique attractor of the RG flow in the IR.
    At this point, the theory becomes scale-invariant and all observable
    physics emerges from the fixed-point couplings. This is the computational
    heart of asymptotic safety.

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from .beta_functions import compute_all_betas, BetaFunctions

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.2.3, Eq. 1.14"


# =============================================================================
# Analytical Fixed-Point Constants (Eq. 1.14)
# =============================================================================

LAMBDA_STAR = 48 * math.pi**2 / 9      # λ̃* = 48π²/9 ≈ 52.6379...
GAMMA_STAR = 32 * math.pi**2 / 3       # γ̃* = 32π²/3 ≈ 105.2759...
MU_STAR = 16 * math.pi**2               # μ̃* = 16π² ≈ 157.9137...

# Universal exponent (Eq. 1.16)
# NOTE: The ratio formula C_H = 3λ̃*/(2γ̃*) gives exactly 0.75
# The value 0.045935703598 cited in IRH21.md comes from a more
# complex spectral zeta function calculation (see Appendix B).
C_H_RATIO = 3 * LAMBDA_STAR / (2 * GAMMA_STAR)  # = 0.75
C_H_SPECTRAL = 0.045935703598  # From spectral zeta evaluation


# =============================================================================
# Fixed Point Data Classes
# =============================================================================


@dataclass
class CosmicFixedPoint:
    """
    The unique non-Gaussian infrared fixed point of the cGFT.
    
    Theoretical Reference:
        IRH21.md §1.2.3, Eq. 1.14
        
    This dataclass represents the Cosmic Fixed Point, which is the unique
    infrared attractor of the renormalization group flow. All observable
    physics - quantum mechanics, general relativity, and the Standard Model -
    emerges from the couplings at this fixed point.
    
    Attributes
    ----------
    lambda_star : float
        Fixed-point value of quartic coupling λ̃*
    gamma_star : float
        Fixed-point value of QNCD coupling γ̃*
    mu_star : float
        Fixed-point value of holographic coupling μ̃*
        
    Examples
    --------
    >>> fp = CosmicFixedPoint.analytical()
    >>> print(f"λ̃* = {fp.lambda_star:.6f}")
    λ̃* = 52.637890
    """
    
    lambda_star: float
    gamma_star: float
    mu_star: float
    
    @classmethod
    def analytical(cls) -> 'CosmicFixedPoint':
        """
        Create fixed point using analytical values from Eq. 1.14.
        
        Returns
        -------
        CosmicFixedPoint
            Fixed point with exact analytical values
        """
        return cls(
            lambda_star=LAMBDA_STAR,
            gamma_star=GAMMA_STAR,
            mu_star=MU_STAR
        )
    
    def verify(self, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Verify this is indeed a fixed point by checking β-functions vanish.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed |β| for verification
            
        Returns
        -------
        dict
            Verification results
        """
        betas = compute_all_betas(self.lambda_star, self.gamma_star, self.mu_star)
        max_beta = max(abs(b) for b in betas)
        
        return {
            'is_fixed_point': max_beta < tolerance,
            'beta_values': betas,
            'max_beta': max_beta,
            'tolerance': tolerance,
            'lambda_match': abs(self.lambda_star - LAMBDA_STAR) < tolerance * LAMBDA_STAR,
            'gamma_match': abs(self.gamma_star - GAMMA_STAR) < tolerance * GAMMA_STAR,
            'mu_match': abs(self.mu_star - MU_STAR) < tolerance * MU_STAR,
        }
    
    def compute_C_H(self, method: str = 'ratio') -> float:
        """
        Compute universal exponent C_H from fixed-point values.
        
        Theoretical Reference:
            IRH21.md §1.3, Eq. 1.16
            
        Parameters
        ----------
        method : str
            'ratio' - Use 3λ̃*/(2γ̃*) formula (gives 0.75)
            'spectral' - Return the spectral zeta value (0.045935703598)
            
        Returns
        -------
        float
            Universal exponent C_H
        """
        if method == 'ratio':
            if self.gamma_star > 0:
                return 3 * self.lambda_star / (2 * self.gamma_star)
            return float('nan')
        elif method == 'spectral':
            return C_H_SPECTRAL
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'lambda_star': self.lambda_star,
            'gamma_star': self.gamma_star,
            'mu_star': self.mu_star,
            'C_H_ratio': self.compute_C_H('ratio'),
            'C_H_spectral': self.compute_C_H('spectral'),
        }
    
    def __repr__(self) -> str:
        return (
            f"CosmicFixedPoint(\n"
            f"  λ̃* = {self.lambda_star:.10f},\n"
            f"  γ̃* = {self.gamma_star:.10f},\n"
            f"  μ̃* = {self.mu_star:.10f},\n"
            f"  C_H = {self.compute_C_H('ratio'):.10f}\n"
            f")"
        )


@dataclass
class FixedPointResult:
    """
    Result of fixed-point computation or verification.
    
    Theoretical Reference:
        IRH21.md §1.3, Eq. 1.14
    """
    lambda_star: float
    gamma_star: float
    mu_star: float
    is_fixed_point: bool
    beta_values: Tuple[float, float, float]
    tolerance: float
    C_H: float = field(init=False)
    
    def __post_init__(self):
        """Compute derived quantities."""
        if self.gamma_star > 0:
            self.C_H = 3 * self.lambda_star / (2 * self.gamma_star)
        else:
            self.C_H = float('nan')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lambda_star': self.lambda_star,
            'gamma_star': self.gamma_star,
            'mu_star': self.mu_star,
            'is_fixed_point': self.is_fixed_point,
            'beta_values': self.beta_values,
            'C_H': self.C_H,
            'tolerance': self.tolerance
        }


# =============================================================================
# Fixed Point Finding Functions
# =============================================================================


def find_fixed_point(
    method: str = 'analytical',
    initial_guess: Optional[Tuple[float, float, float]] = None,
    tolerance: float = 1e-12
) -> CosmicFixedPoint:
    """
    Find the Cosmic Fixed Point.
    
    Theoretical Reference:
        IRH21.md §1.2.3, Eq. 1.14
        
    Parameters
    ----------
    method : str
        'analytical' - Use exact formulas from Eq. 1.14 (default)
        'numerical' - Use Newton-Raphson iteration
    initial_guess : tuple, optional
        Initial (λ̃, γ̃, μ̃) guess for numerical method
    tolerance : float
        Solver tolerance for numerical method
        
    Returns
    -------
    CosmicFixedPoint
        The unique IR fixed point
        
    Examples
    --------
    >>> fp = find_fixed_point()
    >>> print(f"λ̃* = {fp.lambda_star:.6f}")
    λ̃* = 52.637890
    
    >>> fp_num = find_fixed_point(method='numerical')
    >>> verification = fp_num.verify()
    >>> print(f"Is fixed point: {verification['is_fixed_point']}")
    Is fixed point: True
    """
    if method == 'analytical':
        return CosmicFixedPoint.analytical()
    
    elif method == 'numerical':
        return _find_fixed_point_numerical(initial_guess, tolerance)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'numerical'.")


def _find_fixed_point_numerical(
    initial_guess: Optional[Tuple[float, float, float]] = None,
    tolerance: float = 1e-12
) -> CosmicFixedPoint:
    """
    Numerically find the Cosmic Fixed Point using Newton-Raphson.
    
    Parameters
    ----------
    initial_guess : tuple, optional
        Initial (λ̃, γ̃, μ̃) guess. Default uses 90% of analytical values.
    tolerance : float
        Solver tolerance
        
    Returns
    -------
    CosmicFixedPoint
        Numerically found fixed point
    """
    if initial_guess is None:
        # Start near analytical solution
        initial_guess = (LAMBDA_STAR * 0.9, GAMMA_STAR * 0.9, MU_STAR * 0.9)
    
    def beta_system(couplings):
        """System of equations: β = 0."""
        l, g, m = couplings
        return list(compute_all_betas(l, g, m))
    
    solution, info, ier, msg = fsolve(
        beta_system, 
        initial_guess, 
        full_output=True,
        xtol=tolerance
    )
    
    return CosmicFixedPoint(
        lambda_star=solution[0],
        gamma_star=solution[1],
        mu_star=solution[2]
    )


def verify_fixed_point(
    lambda_val: float,
    gamma_val: float,
    mu_val: float,
    tolerance: float = 1e-10
) -> FixedPointResult:
    """
    Verify that given couplings constitute a fixed point.
    
    Theoretical Reference:
        IRH21.md §1.3, Eq. 1.14
        Fixed point requires: β_λ = β_γ = β_μ = 0
        
    Parameters
    ----------
    lambda_val : float
        λ̃ coupling value
    gamma_val : float
        γ̃ coupling value
    mu_val : float
        μ̃ coupling value
    tolerance : float
        Maximum allowed |β| for fixed-point classification
        
    Returns
    -------
    FixedPointResult
        Verification result with beta values and status
    """
    betas = compute_all_betas(lambda_val, gamma_val, mu_val)
    max_beta = max(abs(b) for b in betas)
    is_fp = max_beta < tolerance
    
    return FixedPointResult(
        lambda_star=lambda_val,
        gamma_star=gamma_val,
        mu_star=mu_val,
        is_fixed_point=is_fp,
        beta_values=betas,
        tolerance=tolerance
    )


def compute_universal_exponent() -> Dict[str, Any]:
    """
    Compute universal exponent C_H from fixed-point values.
    
    Theoretical Reference:
        IRH21.md §1.3, Eq. 1.16
        
        Note: There are two related quantities:
        1. C_H_ratio = 3λ̃*/(2γ̃*) = 0.75 (simple algebraic ratio)
        2. C_H_spectral = 0.045935703598 (from spectral zeta function)
        
        The manuscript uses C_H = 0.045935703598, which comes from a more
        complex calculation involving the spectral zeta function, not the
        simple ratio formula.
        
    Returns
    -------
    dict
        C_H values from both methods with comparison
    """
    computed_ratio = 3 * LAMBDA_STAR / (2 * GAMMA_STAR)
    
    return {
        'computed_ratio': computed_ratio,
        'analytical_spectral': C_H_SPECTRAL,
        'ratio_value': computed_ratio,  # 0.75
        'spectral_value': C_H_SPECTRAL,  # 0.045935703598
        'agreement': False,  # These are different physical quantities
        'relative_difference': abs(computed_ratio - C_H_SPECTRAL) / C_H_SPECTRAL,
        'note': (
            'The ratio formula 3λ̃*/(2γ̃*) gives 0.75. '
            'The spectral zeta value 0.045935703598 comes from Appendix B.'
        )
    }


# =============================================================================
# Stability Analysis
# =============================================================================


def compute_stability_matrix(
    lambda_val: Optional[float] = None,
    gamma_val: Optional[float] = None,
    mu_val: Optional[float] = None,
    delta: float = 1e-8
) -> np.ndarray:
    """
    Compute stability matrix M_ij = ∂β_i/∂g_j at given couplings.
    
    Theoretical Reference:
        IRH21.md §1.3
        Eigenvalues determine IR attractiveness of fixed point.
        
    Parameters
    ----------
    lambda_val, gamma_val, mu_val : float, optional
        Coupling values. Defaults to fixed-point values.
    delta : float
        Finite difference step
        
    Returns
    -------
    ndarray
        3×3 stability matrix
    """
    if lambda_val is None:
        lambda_val = LAMBDA_STAR
    if gamma_val is None:
        gamma_val = GAMMA_STAR
    if mu_val is None:
        mu_val = MU_STAR
    
    # Use analytical Jacobian
    beta = BetaFunctions()
    return beta.jacobian((lambda_val, gamma_val, mu_val))


def analyze_fixed_point_stability(tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Analyze stability of Cosmic Fixed Point.
    
    Theoretical Reference:
        IRH21.md §1.3
        Fixed point is IR-attractive if all eigenvalues have positive real parts
        
    Returns
    -------
    dict
        Stability analysis with eigenvalues
    """
    M = compute_stability_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    # IRH21.md predicts: λ₁ = 10, λ₂ = 4, λ₃ = 14/3
    expected_eigenvalues = np.array([10.0, 4.0, 14/3])
    
    # Sort both for comparison
    sorted_computed = np.sort(eigenvalues.real)
    sorted_expected = np.sort(expected_eigenvalues)
    
    eigenvalue_agreement = np.allclose(
        sorted_computed, sorted_expected, rtol=1e-6
    )
    
    return {
        'stability_matrix': M,
        'eigenvalues': eigenvalues,
        'expected_eigenvalues': expected_eigenvalues,
        'eigenvalue_agreement': eigenvalue_agreement,
        'is_ir_attractive': all(e.real > 0 for e in eigenvalues),
        'eigenvectors': eigenvectors
    }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
    'C_H_RATIO',
    'C_H_SPECTRAL',
    
    # Classes
    'CosmicFixedPoint',
    'FixedPointResult',
    
    # Functions
    'find_fixed_point',
    'verify_fixed_point',
    'compute_universal_exponent',
    'compute_stability_matrix',
    'analyze_fixed_point_stability',
]
