"""
Observable Extraction Infrastructure for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §3.2

This module provides the final interface for extracting experimentally
comparable values from the theoretical framework. It computes physical
constants and compares them with experimental data.

Key Observables:
    - Eq. 3.4-3.5: Fine-structure constant α⁻¹ = 137.035999084(1)
    - Eq. 1.16: Universal exponent C_H = 0.045935703598...
    - Tables 3.1, 3.2: Complete physical constants

Modules:
    alpha_inverse: Fine-structure constant (Eq. 3.4-3.5) ✓ COMPLETE
    universal_exponent: C_H = 0.045935703598... (Eq. 1.16) ✓ COMPLETE
    physical_constants: Complete constant database (TODO)
    experimental_comparison: Theory vs. data σ-analysis (TODO)

Dependencies:
    - All previous layers (0-8)

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.2"

# Predicted fine-structure constant inverse (Eq. 3.5)
ALPHA_INVERSE_PREDICTED = 137.035999084  # 12-digit accuracy target

# Universal exponent (Eq. 1.16)
C_H = 0.045935703598

# Predicted dark energy equation of state
W_0 = -0.91234567

# Import from alpha_inverse module
from .alpha_inverse import (
    AlphaInverseResult,
    compute_fine_structure_constant,
    alpha_inverse_from_fixed_point,
    verify_alpha_inverse_precision,
    ALPHA_INVERSE_EXPERIMENTAL,
    ALPHA_INVERSE_UNCERTAINTY,
    BETA_1,
    N_INST,
)

# Import from universal_exponent module
from .universal_exponent import (
    UniversalExponentResult,
    compute_C_H,
    verify_C_H_precision,
    compute_C_H_from_spectral_zeta,
    get_C_H_comparison_table,
    C_H_ANALYTICAL,
    C_H_RATIO_VALUE,
)

__all__ = [
    # Constants
    'ALPHA_INVERSE_PREDICTED',
    'ALPHA_INVERSE_EXPERIMENTAL',
    'ALPHA_INVERSE_UNCERTAINTY',
    'C_H',
    'C_H_ANALYTICAL',
    'C_H_RATIO_VALUE',
    'W_0',
    'BETA_1',
    'N_INST',
    
    # alpha_inverse exports
    'AlphaInverseResult',
    'compute_fine_structure_constant',
    'alpha_inverse_from_fixed_point',
    'verify_alpha_inverse_precision',
    
    # universal_exponent exports
    'UniversalExponentResult',
    'compute_C_H',
    'verify_C_H_precision',
    'compute_C_H_from_spectral_zeta',
    'get_C_H_comparison_table',
]
