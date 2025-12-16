"""
Group Manifold Implementation for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1

This module implements the fundamental group manifold G_inf = SU(2) × U(1)_φ
on which the cGFT field φ(g₁,g₂,g₃,g₄) is defined.

Key Components:
    - SU(2): Special unitary group, realized via unit quaternions
    - U(1)_φ: Holonomic phase group φ ∈ [0, 2π)
    - G_inf: Direct product G_inf = SU(2) × U(1)_φ
    - Haar measures for integration

Theoretical Significance:
    The group manifold G_inf encodes the primordial informational degrees
    of freedom. SU(2) provides the quaternionic structure (3 generators)
    and U(1)_φ provides the holonomic phase (1 generator), together
    yielding 4 degrees of freedom → 4D spacetime.

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .quaternions import Quaternion, quaternion_product

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.1"


# =============================================================================
# SU(2) Implementation via Unit Quaternions
# =============================================================================


@dataclass
class SU2Element:
    """
    Element of SU(2) realized as unit quaternion on S³.
    
    Theoretical Reference:
        IRH21.md §1.1
        SU(2) ≅ S³ is the 3-sphere of unit quaternions.
        
    Mathematical Foundation:
        SU(2) = {u ∈ ℍ : |u| = 1}
        Group operation: quaternion multiplication
        Generators: τₐ = iσₐ/2 where σₐ are Pauli matrices
        
    The isomorphism SU(2) ≅ S³ identifies:
        u = q₀ + iq₁ + jq₂ + kq₃  with  |u|² = 1
        
    Attributes
    ----------
    quaternion : Quaternion
        Unit quaternion representing the SU(2) element
    """
    
    quaternion: Quaternion
    
    def __post_init__(self):
        """Normalize to ensure unit quaternion."""
        norm = self.quaternion.norm()
        if norm < 1e-12:
            raise ValueError("Cannot create SU(2) element from zero quaternion")
        if abs(norm - 1.0) > 1e-10:
            # Normalize
            self.quaternion = self.quaternion.normalize()
    
    @classmethod
    def identity(cls) -> SU2Element:
        """Return identity element e = 1 + 0i + 0j + 0k."""
        return cls(quaternion=Quaternion.identity())
    
    @classmethod
    def from_quaternion(cls, q: Quaternion) -> SU2Element:
        """Create SU(2) element from quaternion (will be normalized)."""
        return cls(quaternion=q)
    
    @classmethod
    def from_components(cls, w: float, x: float, y: float, z: float) -> SU2Element:
        """Create SU(2) element from components (will be normalized)."""
        q = Quaternion(w=w, x=x, y=y, z=z)
        return cls(quaternion=q)
    
    @classmethod
    def from_axis_angle(cls, axis: NDArray[np.float64], angle: float) -> SU2Element:
        """
        Create SU(2) element from axis-angle representation.
        
        Corresponds to rotation by angle θ around unit axis n:
            u = cos(θ/2) + sin(θ/2)(n₁i + n₂j + n₃k)
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return cls.identity()
        axis = axis / axis_norm
        
        half_angle = angle / 2
        w = math.cos(half_angle)
        sin_half = math.sin(half_angle)
        
        q = Quaternion(
            w=w,
            x=sin_half * axis[0],
            y=sin_half * axis[1],
            z=sin_half * axis[2]
        )
        return cls(quaternion=q)
    
    @classmethod
    def from_euler_angles(cls, alpha: float, beta: float, gamma: float) -> SU2Element:
        """
        Create SU(2) element from Euler angles (ZYZ convention).
        
        u = exp(-i α σ₃/2) exp(-i β σ₂/2) exp(-i γ σ₃/2)
        """
        # Rotation around z by alpha
        u1 = cls.from_axis_angle(np.array([0, 0, 1]), alpha)
        # Rotation around y by beta
        u2 = cls.from_axis_angle(np.array([0, 1, 0]), beta)
        # Rotation around z by gamma
        u3 = cls.from_axis_angle(np.array([0, 0, 1]), gamma)
        
        return u1 * u2 * u3
    
    @classmethod
    def random(cls, rng: np.random.Generator = None) -> SU2Element:
        """Generate uniformly random SU(2) element (Haar measure)."""
        return cls(quaternion=Quaternion.random(rng))
    
    def to_quaternion(self) -> Quaternion:
        """Extract underlying quaternion."""
        return self.quaternion
    
    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Convert to 2×2 unitary matrix representation.
        
        SU(2) matrix:
            U = [[ α, -β̄],
                 [ β,  ᾱ]]
        where α = w + iz, β = y + ix
        """
        q = self.quaternion
        alpha = complex(q.w, q.z)
        beta = complex(q.y, q.x)
        
        return np.array([
            [alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]
        ], dtype=np.complex128)
    
    def inverse(self) -> SU2Element:
        """
        Compute group inverse u⁻¹ = ū (conjugate for unit quaternions).
        """
        return SU2Element(quaternion=self.quaternion.conjugate())
    
    def __mul__(self, other: SU2Element) -> SU2Element:
        """Group multiplication: u₁ · u₂ via quaternion product."""
        if not isinstance(other, SU2Element):
            return NotImplemented
        return SU2Element(quaternion=self.quaternion * other.quaternion)
    
    def __eq__(self, other: object) -> bool:
        """SU(2) equality (note: u and -u represent same rotation in SO(3))."""
        if not isinstance(other, SU2Element):
            return False
        return self.quaternion == other.quaternion
    
    def __repr__(self) -> str:
        q = self.quaternion
        return f"SU2Element({q.w:.4f} + {q.x:.4f}i + {q.y:.4f}j + {q.z:.4f}k)"


def haar_measure_SU2_sample(n_samples: int, rng: np.random.Generator = None) -> list[SU2Element]:
    """
    Generate samples from Haar measure on SU(2).
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
        Integration ∫dg uses the bi-invariant Haar measure.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    list[SU2Element]
        Uniformly distributed samples on SU(2) ≅ S³
    """
    if rng is None:
        rng = np.random.default_rng()
    return [SU2Element.random(rng) for _ in range(n_samples)]


def haar_integrate_SU2(
    f: Callable[[SU2Element], float],
    n_samples: int = 10000,
    rng: np.random.Generator = None
) -> Tuple[float, float]:
    """
    Monte Carlo integration over SU(2) with Haar measure.
    
    Computes ∫_{SU(2)} f(u) du where du is normalized Haar measure.
    
    Parameters
    ----------
    f : callable
        Function SU(2) → ℝ to integrate
    n_samples : int
        Number of Monte Carlo samples
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    tuple
        (mean, std_error) of the integral estimate
    """
    samples = haar_measure_SU2_sample(n_samples, rng)
    values = np.array([f(u) for u in samples])
    
    mean = np.mean(values)
    std_error = np.std(values) / np.sqrt(n_samples)
    
    return float(mean), float(std_error)


# =============================================================================
# U(1)_φ Implementation (Holonomic Phase)
# =============================================================================


@dataclass
class U1Phase:
    """
    Element of U(1)_φ - the holonomic phase group.
    
    Theoretical Reference:
        IRH21.md §1.1
        U(1)_φ encodes the holonomic phase φ ∈ [0, 2π).
        
    Mathematical Foundation:
        U(1) = {e^{iφ} : φ ∈ [0, 2π)}
        Group operation: phase addition mod 2π
        
    Holonomic Interpretation:
        In IRH, "holonomic" refers to the phase acquired by parallel
        transport around closed loops in the group manifold. The U(1)_φ
        factor captures this geometric phase, analogous to Berry phase
        in quantum mechanics. Together with SU(2), it forms G_inf which
        encodes the 4 degrees of freedom underlying emergent spacetime.
        
    Attributes
    ----------
    phase : float
        Phase angle φ ∈ [0, 2π)
    """
    
    phase: float
    
    def __post_init__(self):
        """Normalize phase to [0, 2π)."""
        self.phase = float(self.phase) % (2 * math.pi)
    
    @classmethod
    def identity(cls) -> U1Phase:
        """Return identity element φ = 0."""
        return cls(phase=0.0)
    
    @classmethod
    def from_complex(cls, z: complex) -> U1Phase:
        """Create U(1) element from unit complex number."""
        return cls(phase=np.angle(z))
    
    @classmethod
    def random(cls, rng: np.random.Generator = None) -> U1Phase:
        """Generate uniformly random U(1) element."""
        if rng is None:
            rng = np.random.default_rng()
        return cls(phase=rng.uniform(0, 2 * math.pi))
    
    def to_complex(self) -> complex:
        """Convert to unit complex number e^{iφ}."""
        return complex(np.exp(1j * self.phase))
    
    def inverse(self) -> U1Phase:
        """Compute group inverse: (e^{iφ})⁻¹ = e^{-iφ}."""
        return U1Phase(phase=-self.phase)
    
    def __mul__(self, other: U1Phase) -> U1Phase:
        """Group multiplication: e^{iφ₁} · e^{iφ₂} = e^{i(φ₁+φ₂)}."""
        if not isinstance(other, U1Phase):
            return NotImplemented
        return U1Phase(phase=self.phase + other.phase)
    
    def __eq__(self, other: object) -> bool:
        """U(1) equality."""
        if not isinstance(other, U1Phase):
            return False
        # Compare on circle
        diff = abs(self.phase - other.phase) % (2 * math.pi)
        return min(diff, 2 * math.pi - diff) < 1e-10
    
    def __repr__(self) -> str:
        return f"U1Phase(φ={self.phase:.6f})"


def haar_measure_U1_sample(n_samples: int, rng: np.random.Generator = None) -> list[U1Phase]:
    """Generate samples from Haar measure on U(1)."""
    if rng is None:
        rng = np.random.default_rng()
    return [U1Phase.random(rng) for _ in range(n_samples)]


def haar_integrate_U1(
    f: Callable[[U1Phase], float],
    n_samples: int = 10000,
    rng: np.random.Generator = None
) -> Tuple[float, float]:
    """Monte Carlo integration over U(1) with Haar measure."""
    samples = haar_measure_U1_sample(n_samples, rng)
    values = np.array([f(phi) for phi in samples])
    
    mean = np.mean(values)
    std_error = np.std(values) / np.sqrt(n_samples)
    
    return float(mean), float(std_error)


# =============================================================================
# G_inf = SU(2) × U(1)_φ Implementation
# =============================================================================


@dataclass
class GInfElement:
    """
    Element of G_inf = SU(2) × U(1)_φ - the fundamental group manifold.
    
    Theoretical Reference:
        IRH21.md §1.1
        The cGFT field φ(g₁,g₂,g₃,g₄) takes arguments gᵢ ∈ G_inf.
        
    Mathematical Foundation:
        G_inf is the direct product group:
        - dim(G_inf) = dim(SU(2)) + dim(U(1)) = 3 + 1 = 4
        - This 4-dimensionality is the source of 4D emergent spacetime
        
    Attributes
    ----------
    su2 : SU2Element
        SU(2) component
    u1 : U1Phase
        U(1)_φ component
    """
    
    su2: SU2Element
    u1: U1Phase
    
    @classmethod
    def identity(cls) -> GInfElement:
        """Return identity element (e, 0)."""
        return cls(su2=SU2Element.identity(), u1=U1Phase.identity())
    
    @classmethod
    def from_components(
        cls,
        w: float, x: float, y: float, z: float,
        phase: float
    ) -> GInfElement:
        """Create G_inf element from raw components."""
        su2 = SU2Element.from_components(w, x, y, z)
        u1 = U1Phase(phase=phase)
        return cls(su2=su2, u1=u1)
    
    @classmethod
    def random(cls, rng: np.random.Generator = None) -> GInfElement:
        """Generate uniformly random G_inf element (product Haar measure)."""
        if rng is None:
            rng = np.random.default_rng()
        return cls(su2=SU2Element.random(rng), u1=U1Phase.random(rng))
    
    def inverse(self) -> GInfElement:
        """Compute group inverse: (u, φ)⁻¹ = (u⁻¹, -φ)."""
        return GInfElement(su2=self.su2.inverse(), u1=self.u1.inverse())
    
    def __mul__(self, other: GInfElement) -> GInfElement:
        """Group multiplication: (u₁,φ₁)·(u₂,φ₂) = (u₁u₂, φ₁+φ₂)."""
        if not isinstance(other, GInfElement):
            return NotImplemented
        return GInfElement(
            su2=self.su2 * other.su2,
            u1=self.u1 * other.u1
        )
    
    def __eq__(self, other: object) -> bool:
        """G_inf equality."""
        if not isinstance(other, GInfElement):
            return False
        return self.su2 == other.su2 and self.u1 == other.u1
    
    def __repr__(self) -> str:
        return f"GInfElement(su2={self.su2}, u1={self.u1})"


def haar_measure_GInf_sample(n_samples: int, rng: np.random.Generator = None) -> list[GInfElement]:
    """
    Generate samples from Haar measure on G_inf = SU(2) × U(1).
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
        The integration measure ∫[∏dg_i] is the product Haar measure.
    """
    if rng is None:
        rng = np.random.default_rng()
    return [GInfElement.random(rng) for _ in range(n_samples)]


def haar_integrate_GInf(
    f: Callable[[GInfElement], float],
    n_samples: int = 10000,
    rng: np.random.Generator = None
) -> Tuple[float, float]:
    """Monte Carlo integration over G_inf with product Haar measure."""
    samples = haar_measure_GInf_sample(n_samples, rng)
    values = np.array([f(g) for g in samples])
    
    mean = np.mean(values)
    std_error = np.std(values) / np.sqrt(n_samples)
    
    return float(mean), float(std_error)


def compute_GInf_distance(g1: GInfElement, g2: GInfElement) -> float:
    """
    Compute bi-invariant distance on G_inf.
    
    Theoretical Reference:
        IRH21.md §1.1, Appendix A
        The QNCD metric is constructed from this group distance.
        
    The distance uses the product metric:
        d(g1, g2)² = d_SU2(u1, u2)² + d_U1(φ1, φ2)²
    
    where:
        d_SU2(u1, u2) = arccos(|〈u1, u2〉|) - geodesic on S³
        d_U1(φ1, φ2) = min(|φ1-φ2|, 2π-|φ1-φ2|) - geodesic on S¹
    """
    # SU(2) distance via quaternion dot product
    q1 = g1.su2.quaternion
    q2 = g2.su2.quaternion
    dot = abs(q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z)
    dot = min(1.0, dot)  # Clamp for numerical stability
    d_su2 = math.acos(dot)
    
    # U(1) distance (geodesic on circle)
    phi_diff = abs(g1.u1.phase - g2.u1.phase)
    d_u1 = min(phi_diff, 2 * math.pi - phi_diff)
    
    return math.sqrt(d_su2**2 + d_u1**2)


def verify_group_axioms() -> dict:
    """
    Verify G_inf group axioms computationally.
    
    Tests:
        1. Closure: g1 * g2 ∈ G_inf
        2. Associativity: (g1 * g2) * g3 = g1 * (g2 * g3)
        3. Identity: e * g = g * e = g
        4. Inverse: g * g⁻¹ = g⁻¹ * g = e
        5. Bi-invariance of distance: d(kg1, kg2) = d(g1k, g2k) = d(g1, g2)
    
    Returns
    -------
    dict
        Test results with pass/fail status
    """
    rng = np.random.default_rng(42)
    results = {}
    
    g1 = GInfElement.random(rng)
    g2 = GInfElement.random(rng)
    g3 = GInfElement.random(rng)
    e = GInfElement.identity()
    k = GInfElement.random(rng)
    
    # Test 1: Closure (implicit - type checking)
    product = g1 * g2
    results['closure'] = {
        'passed': isinstance(product, GInfElement),
        'type': type(product).__name__
    }
    
    # Test 2: Associativity
    lhs = (g1 * g2) * g3
    rhs = g1 * (g2 * g3)
    assoc_error = compute_GInf_distance(lhs, rhs)
    results['associativity'] = {
        'passed': assoc_error < 1e-10,
        'error': assoc_error
    }
    
    # Test 3: Identity
    left_id = e * g1
    right_id = g1 * e
    id_error = max(compute_GInf_distance(left_id, g1), compute_GInf_distance(right_id, g1))
    results['identity'] = {
        'passed': id_error < 1e-10,
        'error': id_error
    }
    
    # Test 4: Inverse
    left_inv = g1 * g1.inverse()
    right_inv = g1.inverse() * g1
    inv_error = max(compute_GInf_distance(left_inv, e), compute_GInf_distance(right_inv, e))
    results['inverse'] = {
        'passed': inv_error < 1e-10,
        'error': inv_error
    }
    
    # Test 5: Bi-invariance of distance
    d_original = compute_GInf_distance(g1, g2)
    d_left = compute_GInf_distance(k * g1, k * g2)
    d_right = compute_GInf_distance(g1 * k, g2 * k)
    bi_inv_error = max(abs(d_left - d_original), abs(d_right - d_original))
    results['bi_invariance'] = {
        'passed': bi_inv_error < 1e-10,
        'error': bi_inv_error
    }
    
    # Summary
    all_passed = all(r['passed'] for r in results.values())
    results['all_passed'] = all_passed
    results['theoretical_reference'] = 'IRH21.md §1.1'
    
    return results


__all__ = [
    # SU(2) exports
    'SU2Element',
    'haar_measure_SU2_sample',
    'haar_integrate_SU2',
    
    # U(1) exports
    'U1Phase',
    'haar_measure_U1_sample',
    'haar_integrate_U1',
    
    # G_inf exports
    'GInfElement',
    'haar_measure_GInf_sample',
    'haar_integrate_GInf',
    'compute_GInf_distance',
    
    # Verification
    'verify_group_axioms',
]
