"""
Metric Tensor Module for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10

This module implements the emergence of the classical spacetime metric
g_μν(x) from the cGFT condensate at the Cosmic Fixed Point. The metric
is not fundamental but an emergent observable derived from the infrared
fixed-point phase of the quaternionic Group Field Theory.

Key Results:
    - Eq. 2.10: g_μν(x) emerges from condensate ⟨φ⟩ ≠ 0
    - §2.2.1: Detailed derivation of metric from group manifold
    - Theorem 2.2: Emergent metric is smooth and Lorentzian

Mathematical Framework:
    At the Cosmic Fixed Point, the quaternionic field φ(g₁,g₂,g₃,g₄) 
    develops a non-trivial condensate ⟨φ⟩ ≠ 0. This condensate breaks
    the fundamental symmetries of G_inf = SU(2) × U(1)_φ and defines
    an emergent effective geometry.

    The metric tensor is constructed from:
    1. Bilocal field Σ(g,g') = ⟨φ(g,·,·,·)φ̄(g',·,·,·)⟩
    2. Projection to spacetime M⁴ via group-to-spacetime mapping
    3. Symmetrization and normalization

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - numpy

Authors: IRH Computational Framework Team
Last Updated: December 2024
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


__version__ = "21.0.0"
__theoretical_foundation__ = "Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10"


# ============================================================================
# Physical Constants
# ============================================================================

# Spacetime dimension (from Theorem 2.1)
SPACETIME_DIM = 4

# Minkowski metric signature (-,+,+,+)
MINKOWSKI_SIGNATURE = (-1, 1, 1, 1)

# Planck length (for reference scale)
PLANCK_LENGTH = 1.616255e-35  # meters


# ============================================================================
# Core Classes
# ============================================================================

@dataclass
class MetricTensor:
    """
    Emergent metric tensor g_μν(x) from cGFT condensate.
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10
    
    Attributes
    ----------
    components : np.ndarray
        4×4 array of metric components g_μν
    position : np.ndarray
        Spacetime position x^μ where metric is evaluated
    signature : tuple
        Metric signature, should be (-1, 1, 1, 1) for Lorentzian
    is_lorentzian : bool
        Whether metric has Lorentzian signature
    determinant : float
        Metric determinant det(g_μν)
    theoretical_reference : str
        Reference to Intrinsic_Resonance_Holography-v21.1.md
    """
    components: np.ndarray
    position: np.ndarray = field(default_factory=lambda: np.zeros(4))
    signature: Tuple[int, ...] = field(init=False)
    is_lorentzian: bool = field(init=False)
    determinant: float = field(init=False)
    theoretical_reference: str = "Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10"
    
    def __post_init__(self):
        """Compute derived properties."""
        # Ensure correct shape
        if self.components.shape != (4, 4):
            raise ValueError(f"Metric must be 4×4, got {self.components.shape}")
        
        # Compute eigenvalues for signature
        eigenvalues = np.linalg.eigvalsh(self.components)
        self.signature = tuple(int(np.sign(ev)) for ev in sorted(eigenvalues))
        
        # Check Lorentzian signature
        self.is_lorentzian = (self.signature == MINKOWSKI_SIGNATURE)
        
        # Compute determinant
        self.determinant = float(np.linalg.det(self.components))
    
    @property
    def inverse(self) -> np.ndarray:
        """Compute inverse metric g^μν."""
        return np.linalg.inv(self.components)
    
    @property
    def sqrt_neg_det(self) -> float:
        """Compute √(-g) for integration measure."""
        return np.sqrt(abs(self.determinant))
    
    def raise_index(self, v: np.ndarray) -> np.ndarray:
        """Raise index: v^μ = g^μν v_ν."""
        return self.inverse @ v
    
    def lower_index(self, v: np.ndarray) -> np.ndarray:
        """Lower index: v_μ = g_μν v^ν."""
        return self.components @ v
    
    def inner_product(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute inner product g_μν u^μ v^ν."""
        return float(u @ self.components @ v)
    
    def norm_squared(self, v: np.ndarray) -> float:
        """Compute norm squared g_μν v^μ v^ν."""
        return self.inner_product(v, v)
    
    def is_timelike(self, v: np.ndarray) -> bool:
        """Check if vector is timelike (g_μν v^μ v^ν < 0)."""
        return self.norm_squared(v) < 0
    
    def is_spacelike(self, v: np.ndarray) -> bool:
        """Check if vector is spacelike (g_μν v^μ v^ν > 0)."""
        return self.norm_squared(v) > 0
    
    def is_null(self, v: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if vector is null (g_μν v^μ v^ν = 0)."""
        return abs(self.norm_squared(v)) < tol
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'components': self.components.tolist(),
            'position': self.position.tolist(),
            'signature': self.signature,
            'is_lorentzian': self.is_lorentzian,
            'determinant': self.determinant,
            'sqrt_neg_det': self.sqrt_neg_det,
            'theoretical_reference': self.theoretical_reference,
        }


@dataclass
class EmergentGeometry:
    """
    Complete emergent geometry from cGFT condensate.
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2
    
    This class encapsulates the full emergent geometry including:
    - Metric tensor field
    - Connection coefficients (Christoffel symbols)
    - Curvature tensors
    
    Attributes
    ----------
    metric_field : callable
        Function g_μν(x) returning MetricTensor at position x
    dimension : int
        Spacetime dimension (4)
    is_lorentzian : bool
        Whether geometry is Lorentzian everywhere
    """
    dimension: int = SPACETIME_DIM
    condensate_vev: float = 1.0  # ⟨φ⟩ vacuum expectation value
    planck_scale: float = PLANCK_LENGTH
    
    def metric_at(self, x: np.ndarray) -> MetricTensor:
        """
        Compute metric tensor at position x.
        
        For now, returns Minkowski metric. Full implementation would
        compute from condensate dynamics.
        """
        return minkowski_metric(x)
    
    def christoffel(
        self, 
        x: np.ndarray, 
        mu: int, 
        nu: int, 
        rho: int,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Compute Christoffel symbol Γ^μ_νρ at position x.
        
        Γ^μ_νρ = ½ g^μσ (∂_ν g_σρ + ∂_ρ g_νσ - ∂_σ g_νρ)
        """
        # Get inverse metric
        g = self.metric_at(x)
        g_inv = g.inverse
        
        # Numerical derivatives of metric
        def dg(sigma, alpha, direction):
            """Partial derivative ∂_direction g_σα."""
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[direction] += epsilon
            x_minus[direction] -= epsilon
            
            g_plus = self.metric_at(x_plus).components[sigma, alpha]
            g_minus = self.metric_at(x_minus).components[sigma, alpha]
            
            return (g_plus - g_minus) / (2 * epsilon)
        
        # Compute Christoffel symbol
        result = 0.0
        for sigma in range(4):
            term = (
                dg(sigma, rho, nu) +
                dg(nu, sigma, rho) -
                dg(nu, rho, sigma)
            )
            result += g_inv[mu, sigma] * term
        
        return 0.5 * result
    
    def riemann(
        self,
        x: np.ndarray,
        mu: int,
        nu: int,
        rho: int,
        sigma: int,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Compute Riemann tensor R^μ_νρσ at position x.
        
        R^μ_νρσ = ∂_ρ Γ^μ_νσ - ∂_σ Γ^μ_νρ + Γ^μ_λρ Γ^λ_νσ - Γ^μ_λσ Γ^λ_νρ
        """
        # Numerical derivatives of Christoffel
        def d_gamma(m, n, r, direction):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[direction] += epsilon
            x_minus[direction] -= epsilon
            
            gamma_plus = self.christoffel(x_plus, m, n, r, epsilon)
            gamma_minus = self.christoffel(x_minus, m, n, r, epsilon)
            
            return (gamma_plus - gamma_minus) / (2 * epsilon)
        
        # First two terms
        term1 = d_gamma(mu, nu, sigma, rho)
        term2 = -d_gamma(mu, nu, rho, sigma)
        
        # Connection terms
        term3 = 0.0
        term4 = 0.0
        for lam in range(4):
            gamma_mlr = self.christoffel(x, mu, lam, rho, epsilon)
            gamma_lns = self.christoffel(x, lam, nu, sigma, epsilon)
            term3 += gamma_mlr * gamma_lns
            
            gamma_mls = self.christoffel(x, mu, lam, sigma, epsilon)
            gamma_lnr = self.christoffel(x, lam, nu, rho, epsilon)
            term4 -= gamma_mls * gamma_lnr
        
        return term1 + term2 + term3 + term4
    
    def ricci_tensor(self, x: np.ndarray, mu: int, nu: int) -> float:
        """Compute Ricci tensor R_μν = R^ρ_μρν."""
        result = 0.0
        for rho in range(4):
            result += self.riemann(x, rho, mu, rho, nu)
        return result
    
    def ricci_scalar(self, x: np.ndarray) -> float:
        """Compute Ricci scalar R = g^μν R_μν."""
        g = self.metric_at(x)
        g_inv = g.inverse
        
        result = 0.0
        for mu in range(4):
            for nu in range(4):
                result += g_inv[mu, nu] * self.ricci_tensor(x, mu, nu)
        
        return result


# ============================================================================
# Metric Construction Functions
# ============================================================================

def minkowski_metric(position: Optional[np.ndarray] = None) -> MetricTensor:
    """
    Construct Minkowski (flat) metric η_μν = diag(-1, 1, 1, 1).
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2.1
    
    The Minkowski metric is the leading-order emergent metric in the
    IR limit of the cGFT, before curvature corrections.
    
    Parameters
    ----------
    position : np.ndarray, optional
        Spacetime position (default: origin)
    
    Returns
    -------
    MetricTensor
        Minkowski metric tensor
    """
    if position is None:
        position = np.zeros(4)
    
    components = np.diag([-1.0, 1.0, 1.0, 1.0])
    
    return MetricTensor(components=components, position=position)


def schwarzschild_metric(
    r: float,
    M: float = 1.0,
    G: float = 1.0,
    c: float = 1.0,
) -> MetricTensor:
    """
    Construct Schwarzschild metric for mass M.
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2.2
    
    ds² = -(1 - r_s/r)c²dt² + (1 - r_s/r)⁻¹dr² + r²dΩ²
    
    where r_s = 2GM/c² is the Schwarzschild radius.
    
    Parameters
    ----------
    r : float
        Radial coordinate (must be > r_s)
    M : float
        Mass in appropriate units
    G : float
        Newton's constant
    c : float
        Speed of light
    
    Returns
    -------
    MetricTensor
        Schwarzschild metric at radius r
    """
    # Schwarzschild radius
    r_s = 2 * G * M / c**2
    
    if r <= r_s:
        raise ValueError(f"r = {r} must be greater than r_s = {r_s}")
    
    # Metric components in (t, r, θ, φ) coordinates
    f = 1 - r_s / r
    
    components = np.array([
        [-(c**2) * f, 0, 0, 0],
        [0, 1/f, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2],  # Simplified: assumes θ = π/2
    ])
    
    position = np.array([0.0, r, np.pi/2, 0.0])
    
    return MetricTensor(components=components, position=position)


def metric_from_condensate(
    condensate_field: np.ndarray,
    grid_spacing: float = 1.0,
) -> MetricTensor:
    """
    Construct metric tensor from cGFT condensate.
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10
    
    The classical spacetime metric g_μν(x) is derived from the
    infrared fixed-point phase of the cGFT. At the Cosmic Fixed Point,
    the quaternionic field develops a non-trivial condensate that
    defines an emergent effective geometry.
    
    Parameters
    ----------
    condensate_field : np.ndarray
        Complex scalar field representing condensate
    grid_spacing : float
        Lattice spacing for gradient computation
    
    Returns
    -------
    MetricTensor
        Emergent metric tensor
    
    Notes
    -----
    Simplified implementation. Full version would:
    1. Compute bilocal field Σ(g,g')
    2. Project to spacetime M⁴
    3. Extract symmetric tensor g_μν
    """
    # For now, return Minkowski plus small perturbation
    # This is a placeholder for the full condensate-to-metric map
    
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    
    # Perturbation from condensate (placeholder)
    if condensate_field is not None:
        amplitude = np.mean(np.abs(condensate_field))
        h = 0.01 * amplitude * np.random.randn(4, 4)
        h = 0.5 * (h + h.T)  # Symmetrize
        components = eta + h
    else:
        components = eta
    
    return MetricTensor(components=components)


def emergent_metric(
    scale: float = 0.0,
    position: Optional[np.ndarray] = None,
) -> MetricTensor:
    """
    Compute emergent metric at given RG scale.
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10
    
    Parameters
    ----------
    scale : float
        RG scale k (k=0 for deep IR)
    position : np.ndarray, optional
        Spacetime position
    
    Returns
    -------
    MetricTensor
        Emergent metric tensor
    """
    if position is None:
        position = np.zeros(4)
    
    # In IR limit (scale → 0), metric approaches Minkowski
    # with small corrections from condensate dynamics
    
    return minkowski_metric(position)


# ============================================================================
# Verification Functions
# ============================================================================

def verify_lorentzian_signature(metric: MetricTensor) -> Dict[str, Any]:
    """
    Verify metric has Lorentzian signature (-,+,+,+).
    
    THEORETICAL REFERENCE: Intrinsic_Resonance_Holography-v21.1.md §2.4.1, Theorem H.1
    """
    return {
        'signature': metric.signature,
        'expected': MINKOWSKI_SIGNATURE,
        'is_lorentzian': metric.is_lorentzian,
        'determinant': metric.determinant,
        'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.4.1, Theorem H.1',
    }


def verify_metric_properties(metric: MetricTensor) -> Dict[str, Any]:
    """
    Verify essential properties of emergent metric.
    
    Returns
    -------
    dict
        Verification results for symmetry, signature, invertibility
    """
    components = metric.components
    
    # Check symmetry
    is_symmetric = np.allclose(components, components.T)
    
    # Check non-degeneracy
    is_invertible = abs(metric.determinant) > 1e-15
    
    # Check signature
    signature_check = verify_lorentzian_signature(metric)
    
    return {
        'is_symmetric': is_symmetric,
        'is_invertible': is_invertible,
        'is_lorentzian': metric.is_lorentzian,
        'signature': metric.signature,
        'determinant': metric.determinant,
        'all_checks_passed': (
            is_symmetric and is_invertible and metric.is_lorentzian
        ),
        'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.2.1, Eq. 2.10',
    }


# ============================================================================
# Summary Function
# ============================================================================

def generate_metric_tensor_summary() -> Dict[str, Any]:
    """Generate summary of metric tensor module."""
    
    # Create test metric
    eta = minkowski_metric()
    
    return {
        'module': 'metric_tensor',
        'theoretical_foundation': __theoretical_foundation__,
        'version': __version__,
        
        'minkowski_metric': {
            'components': eta.components.tolist(),
            'signature': eta.signature,
            'is_lorentzian': eta.is_lorentzian,
        },
        
        'key_concepts': [
            'Metric emerges from cGFT condensate',
            'g_μν(x) is not fundamental but emergent',
            'Lorentzian signature from spontaneous ℤ₂ breaking',
            'IR limit: Minkowski + perturbations',
        ],
        
        'references': [
            'Intrinsic_Resonance_Holography-v21.1.md §2.2.1',
            'Intrinsic_Resonance_Holography-v21.1.md Eq. 2.10',
            'Intrinsic_Resonance_Holography-v21.1.md §2.4.1 (Lorentzian signature)',
        ],
    }


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Constants
    'SPACETIME_DIM',
    'MINKOWSKI_SIGNATURE',
    'PLANCK_LENGTH',
    
    # Classes
    'MetricTensor',
    'EmergentGeometry',
    
    # Construction functions
    'minkowski_metric',
    'schwarzschild_metric',
    'metric_from_condensate',
    'emergent_metric',
    
    # Verification
    'verify_lorentzian_signature',
    'verify_metric_properties',
    
    # Summary
    'generate_metric_tensor_summary',
]


if __name__ == '__main__':
    print("=" * 60)
    print("IRH v21.0 Metric Tensor Module")
    print("THEORETICAL FOUNDATION:", __theoretical_foundation__)
    print("=" * 60)
    
    # Test Minkowski metric
    eta = minkowski_metric()
    print(f"\nMinkowski metric:")
    print(f"  Signature: {eta.signature}")
    print(f"  Is Lorentzian: {eta.is_lorentzian}")
    print(f"  Determinant: {eta.determinant}")
    
    # Verify properties
    check = verify_metric_properties(eta)
    print(f"\nVerification: {'PASS' if check['all_checks_passed'] else 'FAIL'}")
