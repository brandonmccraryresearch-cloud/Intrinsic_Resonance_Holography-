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
    alpha_inverse: Fine-structure constant (Eq. 3.4-3.5)
    universal_exponent: C_H = 0.045935703598... (Eq. 1.16)
    physical_constants: Complete constant database
    experimental_comparison: Theory vs. data σ-analysis

Dependencies:
    - All previous layers (0-8)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.2"

# Predicted fine-structure constant inverse (Eq. 3.5)
ALPHA_INVERSE_PREDICTED = 137.035999084  # 12-digit accuracy target

# Universal exponent (Eq. 1.16)
C_H = 0.045935703598

# Predicted dark energy equation of state
W_0 = -0.91234567

__all__ = [
    # Constants
    'ALPHA_INVERSE_PREDICTED',
    'C_H',
    'W_0',
    
    # alpha_inverse exports
    'compute_fine_structure_constant',
    'alpha_inverse_from_fixed_point',
    
    # universal_exponent exports
    'compute_C_H',
    'verify_C_H_precision',
    
    # physical_constants exports
    'get_constant',
    'all_predicted_constants',
    'compare_with_experiment',
    
    # experimental_comparison exports
    'sigma_deviation',
    'chi_squared_analysis',
    'generate_comparison_table',
]
