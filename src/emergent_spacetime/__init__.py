"""
Emergent Spacetime Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §2.1-2.5

This module implements the emergence of 4-dimensional Lorentzian spacetime
from the RG flow of the cGFT. The spectral dimension flows from fractal UV
behavior to exactly 4 in the infrared due to asymptotic safety.

Key Results:
    - Eq. 2.8-2.9: Spectral dimension d_spec flows to exactly 4
    - Eq. 2.10: Emergent metric g_μν(x) from condensate
    - Theorem 2.1: Exact 4D spacetime from quaternionic cGFT
    - §2.1.1: Quaternionic Necessity Principle (algebraic derivation of d=4)

Modules:
    spectral_dimension: d_spec(k) flow (Eq. 2.8-2.9)
    metric_tensor: g_μν(x) from condensate (Eq. 2.10)
    lorentzian_signature: Spontaneous ℤ₂ breaking, timelike direction
    graviton: Two-point function, propagator (Appendix C)
    einstein_equations: Variational derivation from Harmony Functional

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §2.1-2.5"

# Target spectral dimension (Eq. 2.9)
SPECTRAL_DIMENSION_IR = 4.0  # Exact in the infrared limit

# One-loop spectral dimension before graviton corrections
SPECTRAL_DIMENSION_ONE_LOOP = 42 / 11  # ≈ 3.818

__all__ = [
    # Constants
    'SPECTRAL_DIMENSION_IR',
    'SPECTRAL_DIMENSION_ONE_LOOP',
    
    # spectral_dimension exports
    'compute_spectral_dimension',
    'spectral_dimension_flow',
    'graviton_correction',
    
    # metric_tensor exports
    'emergent_metric',
    'metric_from_condensate',
    
    # lorentzian_signature exports
    'compute_signature',
    'verify_lorentzian',
    'z2_symmetry_breaking',
    
    # graviton exports
    'graviton_propagator',
    'graviton_two_point_function',
    
    # einstein_equations exports
    'derive_einstein_equations',
    'einstein_hilbert_action',
]
