"""
Complex Group Field Theory (cGFT) Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1

This module implements the fundamental dynamics of the cGFT, defining the action
functional S[φ,φ̄] (Eqs. 1.1-1.4) that governs the evolution of quantum information.
This is where IRH's unique structural commitments first appear.

Key Components:
    - G_inf = SU(2) × U(1)_φ: The fundamental group manifold
    - φ(g₁,g₂,g₃,g₄) ∈ ℍ: Quaternionic field over four group elements
    - QNCD-weighted interaction kernel with phase coherence

Modules:
    fields: φ(g₁,g₂,g₃,g₄) ∈ ℍ field representations
    actions: S_kin, S_int, S_hol (Eqs. 1.1-1.4)
    operators: Laplace-Beltrami operators, functional derivatives
    interactions: QNCD-weighted kernels, phase coherence
    symmetries: Gauge transformations, Weyl ordering

Dependencies:
    - src.primitives (Layer 0)

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.1"

# Import from actions module (Eqs. 1.1-1.4)
from .actions import (
    compute_kinetic_action,
    compute_interaction_action,
    compute_holographic_action,
    compute_total_action,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)

# Import from fields module (§1.1.1)
from .fields import (
    QuaternionicField,
    create_field,
    field_conjugate,
    apply_gauge_transform,
    verify_gauge_invariance,
)

__all__ = [
    # actions exports (Eqs. 1.1-1.4)
    'compute_kinetic_action',
    'compute_interaction_action',
    'compute_holographic_action',
    'compute_total_action',
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
    
    # fields exports (§1.1.1)
    'QuaternionicField',
    'create_field',
    'field_conjugate',
    'apply_gauge_transform',
    'verify_gauge_invariance',
    
    # operators exports (placeholder)
    'laplace_beltrami_SU2',
    'functional_derivative',
    
    # interactions exports (placeholder)
    'interaction_kernel',
    'QNCD_weighted_kernel',
]
