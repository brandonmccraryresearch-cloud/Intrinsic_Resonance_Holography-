"""
Wetterich Equation Solver for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12

This module implements the exact functional renormalization group (FRG)
equation, known as the Wetterich equation:

    ∂_t Γ_k = (1/2) Tr[(Γ_k^(2) + R_k)^(-1) ∂_t R_k]

where:
    - Γ_k is the effective average action at scale k
    - R_k is the regulator function
    - t = ln(k/k_0) is the RG time
    - Γ_k^(2) is the second functional derivative (Hessian)

This is the "meta-algorithm of reality" - the fundamental equation
from which the β-functions and Cosmic Fixed Point emerge.

Key Results:
    - Eq. 1.12: Wetterich equation
    - Eq. 1.13: β-functions emerge from truncated Wetterich equation
    - Eq. 1.14: Cosmic Fixed Point where ∂_t Γ_k = 0

Dependencies:
    - numpy
    - scipy.integrate

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import integrate


__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12"


# ============================================================================
# Constants
# ============================================================================

# Reference scale
K_0 = 1.0  # Reference RG scale

# Regulator shape constants
LITIM_REGULATOR = "litim"
EXPONENTIAL_REGULATOR = "exponential"


# ============================================================================
# Regulator Functions
# ============================================================================

@dataclass
class Regulator:
    """
    Regulator function R_k for the Wetterich equation.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2
    
    The regulator provides a smooth infrared cutoff that:
    1. Suppresses low-momentum modes (p² << k²)
    2. Vanishes for high-momentum modes (p² >> k²)
    3. Satisfies R_k→0 = 0 and R_k→∞ → ∞
    
    Attributes
    ----------
    name : str
        Name of regulator type ('litim' or 'exponential')
    k : float
        Current RG scale
    """
    name: str = LITIM_REGULATOR
    k: float = 1.0
    
    # Theoretical Reference: IRH v21.4

    
    def R(self, p_squared: float) -> float:
        """
        Compute regulator R_k(p²).
        
        Parameters
        ----------
        p_squared : float
            Momentum squared p²
            
        Returns
        -------
        float
            Regulator value R_k(p²)
        """
        k2 = self.k ** 2
        
        if self.name == LITIM_REGULATOR:
            # Litim (optimized) regulator: R_k(p²) = (k² - p²)θ(k² - p²)
            if p_squared < k2:
                return k2 - p_squared
            return 0.0
            
        elif self.name == EXPONENTIAL_REGULATOR:
            # Exponential regulator: R_k(p²) = p² / (exp(p²/k²) - 1)
            ratio = p_squared / k2
            if ratio < 1e-10:
                return k2  # Limit as p² → 0
            return p_squared / (np.exp(ratio) - 1)
            
        else:
            raise ValueError(f"Unknown regulator: {self.name}")
    
    # Theoretical Reference: IRH v21.4

    
    def dR_dt(self, p_squared: float) -> float:
        """
        Compute scale derivative ∂_t R_k(p²).
        
        Parameters
        ----------
        p_squared : float
            Momentum squared p²
            
        Returns
        -------
        float
            Scale derivative ∂_t R_k
        """
        k2 = self.k ** 2
        
        if self.name == LITIM_REGULATOR:
            # ∂_t R_k = 2k² θ(k² - p²)
            if p_squared < k2:
                return 2 * k2
            return 0.0
            
        elif self.name == EXPONENTIAL_REGULATOR:
            # Numerical derivative
            delta_t = 0.001
            R_plus = self.R(p_squared) * np.exp(delta_t)
            R_minus = self.R(p_squared) * np.exp(-delta_t)
            return (R_plus - R_minus) / (2 * delta_t)
            
        else:
            raise ValueError(f"Unknown regulator: {self.name}")


# ============================================================================
# Effective Action and Hessian
# ============================================================================

@dataclass
class EffectiveAction:
    """
    Effective average action Γ_k in a given truncation.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2
    
    We use a truncation ansatz where Γ_k depends on three
    dimensionless couplings:
        - λ̃(k): Quartic self-interaction
        - γ̃(k): QNCD coupling
        - μ̃(k): Holographic coupling
    
    Attributes
    ----------
    lambda_coupling : float
        Dimensionless quartic coupling λ̃
    gamma_coupling : float
        Dimensionless QNCD coupling γ̃
    mu_coupling : float
        Dimensionless holographic coupling μ̃
    """
    lambda_coupling: float
    gamma_coupling: float
    mu_coupling: float
    
    # Theoretical Reference: IRH v21.4

    
    def couplings(self) -> Tuple[float, float, float]:
        """Return all couplings as tuple."""
        return (self.lambda_coupling, self.gamma_coupling, self.mu_coupling)
    
    def hessian(self, momentum: float = 0.0) -> np.ndarray:
        """
        Compute Hessian Γ_k^(2) at given momentum.
        
        THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2
        
        The Hessian is the second functional derivative of Γ_k
        with respect to the field. In our truncation, it depends
        on the couplings and momentum.
        
        Parameters
        ----------
        momentum : float
            Momentum p
            
        Returns
        -------
        np.ndarray
            3×3 Hessian matrix in coupling space
        """
        p2 = momentum ** 2
        
        # Kinetic contribution
        H = np.zeros((3, 3))
        
        # Diagonal terms from kinetic and mass terms
        H[0, 0] = p2 + 2 * self.lambda_coupling  # From λ̃ sector
        H[1, 1] = p2 + self.gamma_coupling        # From γ̃ sector  
        H[2, 2] = p2 + self.mu_coupling           # From μ̃ sector
        
        # Off-diagonal couplings
        H[0, 1] = H[1, 0] = 0.1 * self.lambda_coupling * self.gamma_coupling
        H[0, 2] = H[2, 0] = 0.1 * self.lambda_coupling * self.mu_coupling
        H[1, 2] = H[2, 1] = 0.1 * self.gamma_coupling * self.mu_coupling
        
        return H


# ============================================================================
# Wetterich Equation Solver
# ============================================================================

@dataclass
class WetterichSolver:
    """
    Solver for the Wetterich functional RG equation.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12
    
    Integrates the equation:
        ∂_t Γ_k = (1/2) Tr[(Γ_k^(2) + R_k)^(-1) ∂_t R_k]
    
    to evolve couplings from UV to IR.
    
    Attributes
    ----------
    regulator : Regulator
        The regulator function R_k
    momentum_cutoff : float
        UV momentum cutoff for trace integration
    n_momentum_points : int
        Number of momentum points for trace
    """
    regulator: Regulator = field(default_factory=Regulator)
    momentum_cutoff: float = 100.0
    n_momentum_points: int = 100
    
    def compute_trace(
        self,
        gamma_k: EffectiveAction,
        k: float,
    ) -> Tuple[float, float, float]:
        """
        Compute the trace in the Wetterich equation.
        
        THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12
        
        Tr[(Γ_k^(2) + R_k)^(-1) ∂_t R_k]
        
        Parameters
        ----------
        gamma_k : EffectiveAction
            Current effective action
        k : float
            Current RG scale
            
        Returns
        -------
        Tuple[float, float, float]
            Trace contributions to (∂_t λ̃, ∂_t γ̃, ∂_t μ̃)
        """
        self.regulator.k = k
        
        # Momentum integration
        p_values = np.linspace(0, self.momentum_cutoff, self.n_momentum_points)
        
        trace_lambda = 0.0
        trace_gamma = 0.0
        trace_mu = 0.0
        
        for p in p_values:
            p2 = p ** 2
            
            # Hessian at this momentum
            H = gamma_k.hessian(p)
            
            # Regulator contribution
            R_k = self.regulator.R(p2)
            dR_dt = self.regulator.dR_dt(p2)
            
            # (Γ_k^(2) + R_k)^(-1)
            H_reg = H + R_k * np.eye(3)
            
            try:
                H_inv = np.linalg.inv(H_reg)
            except np.linalg.LinAlgError:
                continue
            
            # Contribution to trace
            contribution = H_inv * dR_dt
            
            # Integrate with momentum measure (4D spherical)
            weight = p ** 3 / (16 * np.pi ** 2)
            
            trace_lambda += contribution[0, 0] * weight
            trace_gamma += contribution[1, 1] * weight
            trace_mu += contribution[2, 2] * weight
        
        # Normalize by momentum spacing
        dp = p_values[1] - p_values[0] if len(p_values) > 1 else 1.0
        
        return (
            0.5 * trace_lambda * dp,
            0.5 * trace_gamma * dp,
            0.5 * trace_mu * dp
        )
    
    # Theoretical Reference: IRH v21.4

    
    def flow_equations(
        self,
        t: float,
        couplings: np.ndarray,
    ) -> np.ndarray:
        """
        Right-hand side of the RG flow equations.
        
        Parameters
        ----------
        t : float
            RG time t = ln(k/k_0)
        couplings : np.ndarray
            Current couplings [λ̃, γ̃, μ̃]
            
        Returns
        -------
        np.ndarray
            Time derivatives [∂_t λ̃, ∂_t γ̃, ∂_t μ̃]
        """
        k = K_0 * np.exp(t)
        
        gamma_k = EffectiveAction(
            lambda_coupling=couplings[0],
            gamma_coupling=couplings[1],
            mu_coupling=couplings[2]
        )
        
        # Compute trace
        traces = self.compute_trace(gamma_k, k)
        
        return np.array(traces)
    
    def integrate(
        self,
        initial_couplings: Tuple[float, float, float],
        t_range: Tuple[float, float] = (-10, 5),
        n_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Integrate the Wetterich equation.
        
        THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12
        
        Parameters
        ----------
        initial_couplings : Tuple[float, float, float]
            Initial (λ̃, γ̃, μ̃) at t = t_range[0]
        t_range : Tuple[float, float]
            RG time range (t_min, t_max)
        n_points : int
            Number of output points
            
        Returns
        -------
        Dict
            Integration results with keys:
            - 't': RG time values
            - 'k': Scale values  
            - 'lambda': λ̃(t) trajectory
            - 'gamma': γ̃(t) trajectory
            - 'mu': μ̃(t) trajectory
            - 'converged': Whether flow converged
        """
        t_span = t_range
        t_eval = np.linspace(t_range[0], t_range[1], n_points)
        y0 = np.array(initial_couplings)
        
        # Integrate
        solution = integrate.solve_ivp(
            self.flow_equations,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            max_step=0.1
        )
        
        return {
            't': solution.t,
            'k': K_0 * np.exp(solution.t),
            'lambda': solution.y[0],
            'gamma': solution.y[1],
            'mu': solution.y[2],
            'converged': solution.success,
            'theoretical_reference': 'IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12'
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def solve_wetterich_equation(
    initial_couplings: Tuple[float, float, float],
    t_range: Tuple[float, float] = (-10, 5),
    regulator_type: str = LITIM_REGULATOR,
) -> Dict[str, Any]:
    """
    Solve the Wetterich equation for given initial conditions.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2, Eq. 1.12
    
    Parameters
    ----------
    initial_couplings : Tuple[float, float, float]
        Initial (λ̃, γ̃, μ̃) values
    t_range : Tuple[float, float]
        RG time range
    regulator_type : str
        Type of regulator ('litim' or 'exponential')
        
    Returns
    -------
    Dict
        Integration results
        
    Examples
    --------
    >>> result = solve_wetterich_equation((20, 50, 80), (-5, 5))
    >>> result['converged']
    True
    """
    regulator = Regulator(name=regulator_type)
    solver = WetterichSolver(regulator=regulator)
    return solver.integrate(initial_couplings, t_range)


def verify_wetterich_at_fixed_point(
    fixed_point: Tuple[float, float, float],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verify that ∂_t Γ_k = 0 at the fixed point.
    
    THEORETICAL REFERENCE: IRH v21.1 Manuscript Part 1 §1.2-1.3
    
    At the Cosmic Fixed Point, the Wetterich equation should
    give zero flow:
        ∂_t Γ_k* = 0
    
    Parameters
    ----------
    fixed_point : Tuple[float, float, float]
        Fixed-point values (λ̃*, γ̃*, μ̃*)
    tolerance : float
        Tolerance for "zero"
        
    Returns
    -------
    Dict
        Verification results
    """
    solver = WetterichSolver()
    
    # Compute flow at fixed point
    couplings = np.array(fixed_point)
    flow = solver.flow_equations(0.0, couplings)
    
    # Check if flow vanishes
    is_fixed_point = np.all(np.abs(flow) < tolerance)
    
    return {
        'fixed_point': fixed_point,
        'flow_at_fp': tuple(flow),
        'flow_magnitude': np.linalg.norm(flow),
        'is_fixed_point': is_fixed_point,
        'tolerance': tolerance,
        'theoretical_reference': 'IRH v21.1 Manuscript Part 1 §1.2-1.3, Eq. 1.12'
    }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Constants
    'K_0',
    'LITIM_REGULATOR',
    'EXPONENTIAL_REGULATOR',
    
    # Classes
    'Regulator',
    'EffectiveAction',
    'WetterichSolver',
    
    # Functions
    'solve_wetterich_equation',
    'verify_wetterich_at_fixed_point',
]
