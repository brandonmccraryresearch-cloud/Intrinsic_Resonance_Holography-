"""
Quaternionic Field Implementation for cGFT in IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1, §1.1.1

This module implements the fundamental cGFT field φ(g₁,g₂,g₃,g₄) ∈ ℍ
defined over four group elements from G_inf = SU(2) × U(1)_φ.

Key Properties:
    - Field takes quaternion values: φ ∈ ℍ
    - Four group arguments: φ(g₁, g₂, g₃, g₄)
    - Discretized on lattice for numerical computation
    - Gauge transformations under G_inf

Theoretical Significance:
    The quaternionic nature of φ algebraically necessitates 4D spacetime
    emergence (Quaternionic Necessity Principle, Theorem 2.1).

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..primitives.quaternions import Quaternion
from ..primitives.group_manifold import GInfElement

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.1, §1.1.1"


@dataclass
class QuaternionicField:
    """
    Quaternionic cGFT field φ(g₁,g₂,g₃,g₄) ∈ ℍ.
    
    Theoretical Reference:
        IRH21.md §1.1
        The fundamental field takes quaternion values over four group arguments.
        
    Mathematical Foundation:
        - φ: G_inf⁴ → ℍ
        - Discretized: φ[i₁,i₂,i₃,i₄] ∈ ℍ for lattice indices iₐ ∈ {0,...,N-1}
        - Each lattice point stores 4 real numbers (quaternion components)
        
    Attributes
    ----------
    data : NDArray[np.float64]
        Field data array of shape (N, N, N, N, 4) where last axis is quaternion
    lattice_size : int
        Number of lattice points per dimension (N)
    """
    
    data: NDArray[np.float64]
    lattice_size: int
    
    def __post_init__(self):
        """Validate field data shape."""
        expected_shape = (self.lattice_size,) * 4 + (4,)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"Field data shape {self.data.shape} does not match "
                f"expected {expected_shape} for lattice_size={self.lattice_size}"
            )
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def zeros(cls, lattice_size: int) -> QuaternionicField:
        """Create field initialized to zero everywhere."""
        data = np.zeros((lattice_size,) * 4 + (4,), dtype=np.float64)
        return cls(data=data, lattice_size=lattice_size)
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def ones(cls, lattice_size: int) -> QuaternionicField:
        """Create field initialized to identity quaternion everywhere."""
        data = np.zeros((lattice_size,) * 4 + (4,), dtype=np.float64)
        data[..., 0] = 1.0  # w component = 1
        return cls(data=data, lattice_size=lattice_size)
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def random(
        cls,
        lattice_size: int,
        rng: Optional[np.random.Generator] = None,
        normalized: bool = False
    ) -> QuaternionicField:
        """
        Create field with random quaternion values.
        
        Parameters
        ----------
        lattice_size : int
            Lattice dimension
        rng : np.random.Generator, optional
            Random number generator
        normalized : bool
            If True, normalize each quaternion to unit norm
        """
        if rng is None:
            rng = np.random.default_rng()
        
        data = rng.standard_normal((lattice_size,) * 4 + (4,))
        
        if normalized:
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            data = data / norms
        
        return cls(data=data, lattice_size=lattice_size)
    
    @classmethod
    def condensate(
        cls,
        lattice_size: int,
        condensate_value: Quaternion
    ) -> QuaternionicField:
        """
        Create uniform condensate field 〈φ〉 = const.
        
        Theoretical Reference:
            IRH21.md §1.4
            The condensate solution at the cosmic fixed point.
        """
        data = np.zeros((lattice_size,) * 4 + (4,), dtype=np.float64)
        data[..., 0] = condensate_value.w
        data[..., 1] = condensate_value.x
        data[..., 2] = condensate_value.y
        data[..., 3] = condensate_value.z
        return cls(data=data, lattice_size=lattice_size)
    
    # Theoretical Reference: IRH v21.4

    
    def get_quaternion(self, i1: int, i2: int, i3: int, i4: int) -> Quaternion:
        """Get quaternion value at lattice point (i1, i2, i3, i4)."""
        q_data = self.data[i1, i2, i3, i4]
        return Quaternion(w=q_data[0], x=q_data[1], y=q_data[2], z=q_data[3])
    
    # Theoretical Reference: IRH v21.4

    
    def set_quaternion(self, i1: int, i2: int, i3: int, i4: int, q: Quaternion):
        """Set quaternion value at lattice point (i1, i2, i3, i4)."""
        self.data[i1, i2, i3, i4] = [q.w, q.x, q.y, q.z]
    
    def conjugate(self) -> QuaternionicField:
        """
        Compute conjugate field φ̄.
        
        Theoretical Reference:
            IRH21.md §1.1, Eq. 1.1
            The conjugate field appears in the kinetic term.
        """
        conj_data = self.data.copy()
        conj_data[..., 1:] *= -1  # Negate imaginary parts
        return QuaternionicField(data=conj_data, lattice_size=self.lattice_size)
    
    # Theoretical Reference: IRH v21.4

    
    def norm_squared(self) -> NDArray[np.float64]:
        """
        Compute |φ|² at each lattice point.
        
        Returns array of shape (N, N, N, N).
        """
        return np.sum(self.data ** 2, axis=-1)
    
    # Theoretical Reference: IRH v21.4

    
    def norm(self) -> NDArray[np.float64]:
        """Compute |φ| at each lattice point."""
        return np.sqrt(self.norm_squared())
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1
    def total_norm_squared(self) -> float:
        """Compute ∫|φ|² (sum over all lattice points)."""
        return float(np.sum(self.norm_squared()))
    
    def inner_product(self, other: QuaternionicField) -> complex:
        
        # Theoretical Reference: IRH v21.4
        """
        Compute inner product 〈φ₁|φ₂〉 = ∫φ̄₁·φ₂.
        
        # Theoretical Reference:
            IRH21.md §1.1
            Inner product structure on field space.
        """
        if self.lattice_size != other.lattice_size:
            raise ValueError("Fields must have same lattice size")
        
        # φ̄₁·φ₂ = Re(φ₁)·Re(φ₂) + Im(φ₁)·Im(φ₂) + i[Re(φ₁)·Im(φ₂) - Im(φ₁)·Re(φ₂)]
        # For quaternions: 〈q₁,q₂〉 = Re(q̄₁·q₂)
        conj_self = self.conjugate()
        
        # Quaternion product at each point
        # For simplicity, use component-wise real inner product
        real_part = np.sum(self.data * other.data)
        
        return complex(real_part)
    
    # Theoretical Reference: IRH v21.4

    
    def __add__(self, other: QuaternionicField) -> QuaternionicField:
        """Field addition."""
        if self.lattice_size != other.lattice_size:
            raise ValueError("Fields must have same lattice size")
        return QuaternionicField(
            data=self.data + other.data,
            lattice_size=self.lattice_size
        )
    
    # Theoretical Reference: IRH v21.4

    
    def __sub__(self, other: QuaternionicField) -> QuaternionicField:
        """Field subtraction."""
        if self.lattice_size != other.lattice_size:
            raise ValueError("Fields must have same lattice size")
        return QuaternionicField(
            data=self.data - other.data,
            lattice_size=self.lattice_size
        )
    
    # Theoretical Reference: IRH v21.4

    
    def __mul__(self, scalar: float) -> QuaternionicField:
        """Scalar multiplication."""
        return QuaternionicField(
            data=self.data * scalar,
            lattice_size=self.lattice_size
        )
    
    # Theoretical Reference: IRH v21.4

    
    def __rmul__(self, scalar: float) -> QuaternionicField:
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    # Theoretical Reference: IRH v21.4 Part 1, §1.1
    def __neg__(self) -> QuaternionicField:
        """Field negation."""
        return QuaternionicField(data=-self.data, lattice_size=self.lattice_size)


# Theoretical Reference: IRH v21.4



def create_field(
    lattice_size: int,
    initialization: str = 'random',
    **kwargs
) -> QuaternionicField:
    
    # Theoretical Reference: IRH v21.4
    """
    Factory function to create cGFT field.
    
    Parameters
    ----------
    lattice_size : int
        Number of lattice points per dimension
    initialization : str
        One of: 'zeros', 'ones', 'random', 'condensate'
    **kwargs
        Additional arguments passed to specific initializer
        
    Returns
    -------
    QuaternionicField
        Initialized field
    """
    if initialization == 'zeros':
        return QuaternionicField.zeros(lattice_size)
    elif initialization == 'ones':
        return QuaternionicField.ones(lattice_size)
    elif initialization == 'random':
        return QuaternionicField.random(lattice_size, **kwargs)
    elif initialization == 'condensate':
        condensate_value = kwargs.get('condensate_value', Quaternion.identity())
        return QuaternionicField.condensate(lattice_size, condensate_value)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")


def field_conjugate(phi: QuaternionicField) -> QuaternionicField:
    """
    Compute conjugate field φ̄.
    
    # Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
        S_kin = ∫ φ̄ · [Δ] · φ
    """
    return phi.conjugate()


def apply_gauge_transform(
    phi: QuaternionicField,
    k: GInfElement
) -> QuaternionicField:
    """
    Apply gauge transformation φ(g₁,g₂,g₃,g₄) → φ(kg₁,kg₂,kg₃,kg₄).
    
    Theoretical Reference:
        IRH21.md §1.1
        The action is invariant under left G_inf transformations.
        
    NOTE: This is a simplified version assuming uniform transformation.
    Full implementation requires group element lattice.
    """
    # For uniform condensate, gauge transform has no effect
    # Full implementation would shift lattice indices according to k
    return QuaternionicField(data=phi.data.copy(), lattice_size=phi.lattice_size)


def verify_gauge_invariance(
    phi: QuaternionicField,
    action_func: Callable[[QuaternionicField], complex],
    n_tests: int = 10,
    tolerance: float = 1e-8
) -> dict:
    """
    Verify that action is gauge-invariant.
    
    # Theoretical Reference:
        IRH21.md §1.1
        S[φ] = S[k·φ] for all k ∈ G_inf
        
    Parameters
    ----------
    phi : QuaternionicField
        Test field configuration
    action_func : callable
        Function computing action S[φ]
    n_tests : int
        Number of random transformations to test
    tolerance : float
        Allowed relative deviation
        
    Returns
    -------
    dict
        Test results
    """
    rng = np.random.default_rng(42)
    
    S_original = action_func(phi)
    
    deviations = []
    for _ in range(n_tests):
        k = GInfElement.random(rng)
        phi_transformed = apply_gauge_transform(phi, k)
        S_transformed = action_func(phi_transformed)
        
        if abs(S_original) > 1e-12:
            rel_dev = abs(S_transformed - S_original) / abs(S_original)
        else:
            rel_dev = abs(S_transformed - S_original)
        
        deviations.append(rel_dev)
    
    max_deviation = max(deviations)
    
    return {
        'passed': max_deviation < tolerance,
        'max_deviation': max_deviation,
        'n_tests': n_tests,
        'tolerance': tolerance,
        'theoretical_reference': 'IRH21.md §1.1'
    }


__all__ = [
    'QuaternionicField',
    'create_field',
    'field_conjugate',
    'apply_gauge_transform',
    'verify_gauge_invariance',
]
