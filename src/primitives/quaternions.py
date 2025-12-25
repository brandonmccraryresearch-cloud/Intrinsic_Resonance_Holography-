"""
Quaternion Algebra Implementation for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1.1, §2.1.1

This module implements the quaternion algebra ℍ which provides the algebraic
structure necessitating 4D emergent spacetime (Quaternionic Necessity Principle).

Key Properties:
    - Non-commutative division algebra over ℝ
    - Basis: {1, i, j, k} with i² = j² = k² = ijk = -1
    - Conjugation: q̄ = q₀ - iq₁ - jq₂ - kq₃
    - Norm: |q|² = qq̄ = q₀² + q₁² + q₂² + q₃²

Theoretical Significance:
    The quaternionic nature of the cGFT field φ ∈ ℍ algebraically determines
    that emergent spacetime must be 4-dimensional (Theorem 2.1, Appendix C).

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.1.1, §2.1.1"


@dataclass
class Quaternion:
    """
    Quaternion number q = q₀ + iq₁ + jq₂ + kq₃ ∈ ℍ.
    
    # Theoretical Reference:
        IRH21.md §1.1.1 (Quaternionic cGFT Action)
        The field φ(g₁,g₂,g₃,g₄) takes values in ℍ, the quaternion algebra.
        
    Mathematical Foundation:
        - Basis: {1, i, j, k}
        - Multiplication rules: i² = j² = k² = ijk = -1
        - ij = k, jk = i, ki = j (cyclic)
        - ji = -k, kj = -i, ik = -j (anti-cyclic)
        
    Attributes
    ----------
    w : float
        Scalar (real) component q₀
    x : float
        First imaginary component q₁ (coefficient of i)
    y : float
        Second imaginary component q₂ (coefficient of j)
    z : float
        Third imaginary component q₃ (coefficient of k)
    """
    
    w: float  # q₀ - scalar part
    x: float  # q₁ - i component
    y: float  # q₂ - j component
    z: float  # q₃ - k component
    
    def __post_init__(self):
        """Validate quaternion components are real numbers."""
        for attr in ['w', 'x', 'y', 'z']:
            val = getattr(self, attr)
            if not isinstance(val, (int, float, np.floating)):
                raise TypeError(f"Quaternion component {attr} must be real, got {type(val)}")
            # Convert to float
            setattr(self, attr, float(val))
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    def from_scalar(cls, scalar: float) -> Quaternion:
        """Create quaternion from real number: q = scalar + 0i + 0j + 0k."""
        return cls(w=scalar, x=0.0, y=0.0, z=0.0)
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    def from_vector(cls, v: NDArray[np.float64]) -> Quaternion:
        """Create pure quaternion from 3-vector: q = 0 + v₁i + v₂j + v₃k."""
        if len(v) != 3:
            raise ValueError(f"Vector must have 3 components, got {len(v)}")
        return cls(w=0.0, x=float(v[0]), y=float(v[1]), z=float(v[2]))
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    def from_array(cls, arr: NDArray[np.float64]) -> Quaternion:
        """Create quaternion from array [w, x, y, z]."""
        if len(arr) != 4:
            raise ValueError(f"Array must have 4 components, got {len(arr)}")
        return cls(w=float(arr[0]), x=float(arr[1]), y=float(arr[2]), z=float(arr[3]))
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    def identity(cls) -> Quaternion:
        """Return multiplicative identity: 1 + 0i + 0j + 0k."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
    def zero(cls) -> Quaternion:
        """Return additive identity: 0 + 0i + 0j + 0k."""
        return cls(w=0.0, x=0.0, y=0.0, z=0.0)
    
    @classmethod
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    def random(cls, rng: np.random.Generator = None) -> Quaternion:
        
        # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
        """Generate random unit quaternion (uniform on S³)."""
        if rng is None:
            rng = np.random.default_rng()
        # Use Gaussian distribution for uniform sampling on S³
        components = rng.standard_normal(4)
        norm = np.linalg.norm(components)
        if norm < 1e-10:
            return cls.identity()
        components = components / norm
        return cls(w=components[0], x=components[1], y=components[2], z=components[3])
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def to_vector(self) -> NDArray[np.float64]:
        """Extract vector part [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    @property
    def scalar_part(self) -> float:
        """Return scalar (real) part q₀."""
        return self.w
    
    @property
    def vector_part(self) -> NDArray[np.float64]:
        """Return vector (imaginary) part [q₁, q₂, q₃]."""
        return self.to_vector()
    
    def conjugate(self) -> Quaternion:
        """
        Compute quaternion conjugate q̄ = q₀ - iq₁ - jq₂ - kq₃.
        
        Theoretical Reference:
            IRH21.md §1.1, Eq. 1.1
            The conjugate field φ̄ appears in the kinetic term.
        """
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def norm_squared(self) -> float:
        """
        Compute squared norm |q|² = qq̄ = q₀² + q₁² + q₂² + q₃².
        
        This is real and non-negative, making ℍ a normed division algebra.
        """
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def norm(self) -> float:
        """Compute norm |q| = √(qq̄)."""
        return math.sqrt(self.norm_squared())
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
    def normalize(self) -> Quaternion:
        """Return unit quaternion q/|q| on S³."""
        n = self.norm()
        if n < 1e-12:
            raise ValueError("Cannot normalize zero quaternion")
        return self / n
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def inverse(self) -> Quaternion:
        
        # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
        """
        Compute multiplicative inverse q⁻¹ = q̄/|q|².
        
        For non-zero q: q · q⁻¹ = q⁻¹ · q = 1
        """
        n2 = self.norm_squared()
        if n2 < 1e-24:
            raise ValueError("Cannot invert zero quaternion")
        conj = self.conjugate()
        return Quaternion(w=conj.w/n2, x=conj.x/n2, y=conj.y/n2, z=conj.z/n2)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def __add__(self, other: Union[Quaternion, float]) -> Quaternion:
        """Quaternion addition."""
        if isinstance(other, (int, float)):
            return Quaternion(w=self.w + other, x=self.x, y=self.y, z=self.z)
        if isinstance(other, Quaternion):
            return Quaternion(
                w=self.w + other.w,
                x=self.x + other.x,
                y=self.y + other.y,
                z=self.z + other.z
            )
        return NotImplemented
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def __radd__(self, other: float) -> Quaternion:
        """Right addition with scalar."""
        return self.__add__(other)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
    def __sub__(self, other: Union[Quaternion, float]) -> Quaternion:
        """Quaternion subtraction."""
        if isinstance(other, (int, float)):
            return Quaternion(w=self.w - other, x=self.x, y=self.y, z=self.z)
        if isinstance(other, Quaternion):
            return Quaternion(
                w=self.w - other.w,
                x=self.x - other.x,
                y=self.y - other.y,
                z=self.z - other.z
            )
        return NotImplemented
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def __neg__(self) -> Quaternion:
        
        # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
        """Quaternion negation."""
        return Quaternion(w=-self.w, x=-self.x, y=-self.y, z=-self.z)
    
    def __mul__(self, other: Union[Quaternion, float]) -> Quaternion:
        """
        Quaternion multiplication (Hamilton product).
        
        Theoretical Reference:
            IRH21.md §1.1.1
            Quaternionic multiplication appears in the interaction terms.
            
        Mathematical Foundation:
            (a + bi + cj + dk)(e + fi + gj + hk) =
            (ae - bf - cg - dh) +
            (af + be + ch - dg)i +
            (ag - bh + ce + df)j +
            (ah + bg - cf + de)k
        
        NOTE: Non-commutative! q₁ * q₂ ≠ q₂ * q₁ in general.
        """
        if isinstance(other, (int, float)):
            return Quaternion(
                w=self.w * other,
                x=self.x * other,
                y=self.y * other,
                z=self.z * other
            )
        if isinstance(other, Quaternion):
            # Hamilton product formula
            return Quaternion(
                w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        return NotImplemented
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def __rmul__(self, other: float) -> Quaternion:
        """Right multiplication with scalar."""
        return self.__mul__(other)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
    def __truediv__(self, other: Union[Quaternion, float]) -> Quaternion:
        
        # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
        """Quaternion division: q₁ / q₂ = q₁ * q₂⁻¹."""
        if isinstance(other, (int, float)):
            if abs(other) < 1e-24:
                raise ValueError("Cannot divide by zero")
            return Quaternion(
                w=self.w / other,
                x=self.x / other,
                y=self.y / other,
                z=self.z / other
            )
        if isinstance(other, Quaternion):
            return self * other.inverse()
        return NotImplemented
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1

    
    def __abs__(self) -> float:
        """Return norm |q|."""
        return self.norm()
    
    def __eq__(self, other: object) -> bool:
        """Quaternion equality (approximate)."""
        if not isinstance(other, Quaternion):
            return False
        return np.allclose(self.to_array(), other.to_array(), rtol=1e-10, atol=1e-12)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion(w={self.w:.6f}, x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"
    
    def __str__(self) -> str:
        """Human-readable representation."""
        parts = []
        if abs(self.w) > 1e-10:
            parts.append(f"{self.w:.4f}")
        if abs(self.x) > 1e-10:
            parts.append(f"{self.x:+.4f}i")
        if abs(self.y) > 1e-10:
            parts.append(f"{self.y:+.4f}j")
        if abs(self.z) > 1e-10:
            parts.append(f"{self.z:+.4f}k")
        return "".join(parts) if parts else "0"


# =============================================================================
# Module-level functions
# =============================================================================


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    """
    Compute quaternion conjugate.
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
        φ̄ denotes the quaternionic conjugate of the field.
    """
    return q.conjugate()


def quaternion_product(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Compute Hamilton product of two quaternions.
    
    Theoretical Reference:
        IRH21.md §1.1.1
        Quaternionic multiplication preserves the algebraic structure
        necessary for 4D spacetime emergence.
    
    WARNING: Non-commutative! quaternion_product(q1, q2) ≠ quaternion_product(q2, q1)
    """
    return q1 * q2


# Theoretical Reference: IRH v21.4 Part 1, §1.1.1



def quaternion_norm(q: Quaternion) -> float:
    """Compute quaternion norm |q| = √(qq̄)."""
    return q.norm()


def quaternion_dot(q1: Quaternion, q2: Quaternion) -> float:
    """
    Compute quaternion inner product: q1 · q2 = Re(q̄₁ q₂).
    
    This defines the metric on S³ when q1, q2 are unit quaternions.
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1.1
    """
    conj_q1 = q1.conjugate()
    product = conj_q1 * q2
    return product.scalar_part


# Theoretical Reference: IRH v21.4 Part 1, §1.1.1



def quaternion_exp(q: Quaternion) -> Quaternion:
    """
    Compute quaternion exponential: exp(q) = exp(w)[cos|v| + v̂ sin|v|].
    
    Where q = w + v with v = xi + yj + zk.
    
    Used in SU(2) parameterization via unit quaternions.
    """
    w = q.w
    v = q.to_vector()
    v_norm = np.linalg.norm(v)
    
    exp_w = math.exp(w)
    
    if v_norm < 1e-12:
        # Pure scalar: exp(w) = exp(w) * 1
        return Quaternion(w=exp_w, x=0.0, y=0.0, z=0.0)
    
    cos_v = math.cos(v_norm)
    sin_v = math.sin(v_norm)
    v_hat = v / v_norm
    
    return Quaternion(
        w=exp_w * cos_v,
        x=exp_w * sin_v * v_hat[0],
        y=exp_w * sin_v * v_hat[1],
        z=exp_w * sin_v * v_hat[2]
    )


# Theoretical Reference: IRH v21.4 Part 1, §1.1.1



def quaternion_log(q: Quaternion) -> Quaternion:
    """
    Compute quaternion logarithm (principal branch).
    
    For q = |q|(cos θ + û sin θ): log(q) = ln|q| + ûθ
    """
    norm = q.norm()
    if norm < 1e-12:
        raise ValueError("Cannot compute log of zero quaternion")
    
    v = q.to_vector()
    v_norm = np.linalg.norm(v)
    
    ln_norm = math.log(norm)
    
    if v_norm < 1e-12:
        # Pure real quaternion
        if q.w > 0:
            return Quaternion(w=ln_norm, x=0.0, y=0.0, z=0.0)
        else:
            # log(-|q|) = ln|q| + πi (choosing i direction)
            return Quaternion(w=ln_norm, x=math.pi, y=0.0, z=0.0)
    
    theta = math.atan2(v_norm, q.w)
    v_hat = v / v_norm
    
    return Quaternion(
        w=ln_norm,
        x=theta * v_hat[0],
        y=theta * v_hat[1],
        z=theta * v_hat[2]
    )


# Theoretical Reference: IRH v21.4 Part 1, §1.1.1



def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical linear interpolation between unit quaternions.
    
    Returns the quaternion at parameter t ∈ [0,1] along the geodesic
    from q1 (t=0) to q2 (t=1) on S³.
    """
    q1 = q1.normalize()
    q2 = q2.normalize()
    
    # Compute the cosine of the angle between quaternions
    dot = quaternion_dot(q1, q2)
    
    # If q1 and q2 are very close, use linear interpolation
    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return result.normalize()
    
    # Choose shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # Clamp for numerical stability
    dot = min(1.0, max(-1.0, dot))
    
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    
    q_perp = (q2 - q1 * dot).normalize()
    
    return q1 * math.cos(theta) + q_perp * math.sin(theta)


# Theoretical Reference: IRH v21.4 Part 1, §1.1.1



def verify_quaternion_algebra() -> dict:
    """
    Verify quaternion algebra axioms computationally.
    
    Tests:
        1. Associativity: (q1 * q2) * q3 = q1 * (q2 * q3)
        2. Distributivity: q1 * (q2 + q3) = q1*q2 + q1*q3
        3. Conjugation involution: conj(conj(q)) = q
        4. Norm multiplicativity: |q1 * q2| = |q1| * |q2|
        5. Non-commutativity: q1 * q2 ≠ q2 * q1 (generally)
        6. Division algebra: q * q⁻¹ = 1
    
    Returns
    -------
    dict
        Test results with pass/fail status
    """
    rng = np.random.default_rng(42)
    results = {}
    
    # Generate random test quaternions
    q1 = Quaternion.random(rng)
    q2 = Quaternion.random(rng)
    q3 = Quaternion.random(rng)
    
    # Test 1: Associativity
    lhs = (q1 * q2) * q3
    rhs = q1 * (q2 * q3)
    results['associativity'] = {
        'passed': lhs == rhs,
        'max_error': np.max(np.abs(lhs.to_array() - rhs.to_array()))
    }
    
    # Test 2: Left distributivity
    lhs = q1 * (q2 + q3)
    rhs = q1 * q2 + q1 * q3
    results['left_distributivity'] = {
        'passed': lhs == rhs,
        'max_error': np.max(np.abs(lhs.to_array() - rhs.to_array()))
    }
    
    # Test 3: Conjugation involution
    double_conj = q1.conjugate().conjugate()
    results['conjugation_involution'] = {
        'passed': q1 == double_conj,
        'max_error': np.max(np.abs(q1.to_array() - double_conj.to_array()))
    }
    
    # Test 4: Norm multiplicativity
    norm_product = (q1 * q2).norm()
    product_norms = q1.norm() * q2.norm()
    results['norm_multiplicativity'] = {
        'passed': np.isclose(norm_product, product_norms, rtol=1e-10),
        'max_error': abs(norm_product - product_norms)
    }
    
    # Test 5: Non-commutativity (verify it's NOT commutative)
    commutator = q1 * q2 - q2 * q1
    results['non_commutativity'] = {
        'passed': commutator.norm() > 1e-10,  # Should be non-zero
        'commutator_norm': commutator.norm()
    }
    
    # Test 6: Division algebra
    q_inv_q = q1 * q1.inverse()
    identity = Quaternion.identity()
    results['division_algebra'] = {
        'passed': q_inv_q == identity,
        'max_error': np.max(np.abs(q_inv_q.to_array() - identity.to_array()))
    }
    
    # Summary
    all_passed = all(r['passed'] for r in results.values())
    results['all_passed'] = all_passed
    results['theoretical_reference'] = 'IRH21.md §1.1.1, Appendix C'
    
    return results


__all__ = [
    'Quaternion',
    'quaternion_conjugate',
    'quaternion_product',
    'quaternion_norm',
    'quaternion_dot',
    'quaternion_exp',
    'quaternion_log',
    'quaternion_slerp',
    'verify_quaternion_algebra',
]
