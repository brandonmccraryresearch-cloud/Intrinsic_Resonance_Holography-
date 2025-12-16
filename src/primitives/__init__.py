"""
Primitive Ontological Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §1.0.1 (Revised Foundational Axiom)

This module implements the axiomatically primitive quantum-informational structures
from which all emergent physics derives. It contains NO phenomenology—only the
mathematical scaffolding upon which the cGFT dynamics operate.

Ontological Commitments:
    - Reality is fundamentally quantum-informational
    - States inhabit Hilbert space ℋ_fund with complexity functional K_Q
    - Group manifold G_inf = SU(2) × U(1)_φ encodes primordial degrees of freedom
    - Quaternionic algebra ℍ provides algebraic necessity for 4D spacetime

Modules:
    quaternions: ℍ arithmetic, conjugation, quaternionic products
    group_manifold: SU(2) and U(1) Lie group operations, Haar measure
    qncd: QNCD metric, algorithmic distance on G_inf

Design Principles:
    1. Theory-agnostic: Could be reused for other quantum information theories
    2. No emergent concepts: No reference to spacetime, particles, or forces
    3. Maximum mathematical rigor: Formal group theory, functional analysis
    4. Provably correct: Every operation has mathematical theorem backing

Dependencies:
    - NumPy (numerical linear algebra)
    - SciPy (special functions, optimization)

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.0.1"

# Import quaternion algebra (§1.1.1)
from .quaternions import (
    Quaternion,
    quaternion_conjugate,
    quaternion_product,
    quaternion_norm,
    quaternion_dot,
    quaternion_exp,
    quaternion_log,
    quaternion_slerp,
    verify_quaternion_algebra,
)

# Import group manifold (§1.1)
from .group_manifold import (
    SU2Element,
    U1Phase,
    GInfElement,
    haar_measure_SU2_sample,
    haar_measure_U1_sample,
    haar_measure_GInf_sample,
    haar_integrate_SU2,
    haar_integrate_U1,
    haar_integrate_GInf,
    compute_GInf_distance,
    verify_group_axioms,
)

# Import QNCD metric (Appendix A)
from .qncd import (
    encode_quaternion,
    encode_GInf_element,
    compute_QNCD,
    compute_QNCD_from_product,
    compute_pairwise_QNCD_sum,
    verify_QNCD_metric_axioms,
    verify_QUCC_theorem,
)

__all__ = [
    # Quaternion exports
    'Quaternion',
    'quaternion_conjugate',
    'quaternion_product',
    'quaternion_norm',
    'quaternion_dot',
    'quaternion_exp',
    'quaternion_log',
    'quaternion_slerp',
    'verify_quaternion_algebra',
    
    # Group manifold exports
    'SU2Element',
    'U1Phase',
    'GInfElement',
    'haar_measure_SU2_sample',
    'haar_measure_U1_sample',
    'haar_measure_GInf_sample',
    'haar_integrate_SU2',
    'haar_integrate_U1',
    'haar_integrate_GInf',
    'compute_GInf_distance',
    'verify_group_axioms',
    
    # QNCD exports
    'encode_quaternion',
    'encode_GInf_element',
    'compute_QNCD',
    'compute_QNCD_from_product',
    'compute_pairwise_QNCD_sum',
    'verify_QNCD_metric_axioms',
    'verify_QUCC_theorem',
]


def verify_all_primitives() -> dict:
    """
    Comprehensive verification of all primitive mathematical structures.
    
    This function validates:
    - Quaternion algebra axioms
    - Group manifold axioms (closure, associativity, identity, inverse)
    - QNCD metric axioms (positivity, symmetry, triangle inequality)
    - QUCC-Theorem compliance
    
    Returns
    -------
    dict
        Complete verification results
    """
    results = {
        'quaternion_algebra': verify_quaternion_algebra(),
        'group_axioms': verify_group_axioms(),
        'qncd_metric': verify_QNCD_metric_axioms(),
        'qucc_theorem': verify_QUCC_theorem(),
    }
    
    results['all_passed'] = all(
        r.get('all_passed', r.get('passed', False))
        for r in results.values()
        if isinstance(r, dict)
    )
    
    results['theoretical_reference'] = 'IRH21.md §1.0.1, §1.1, Appendix A'
    
    return results
