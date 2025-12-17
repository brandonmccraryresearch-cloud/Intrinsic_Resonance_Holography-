"""
Emergent Spacetime Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §2.1-2.5

This module implements the emergence of 4-dimensional Lorentzian spacetime
from the RG flow of the cGFT. The spectral dimension flows from fractal UV
behavior to exactly 4 in the infrared due to asymptotic safety.

Key Results:
    - Eq. 2.8-2.9: Spectral dimension d_spec flows to exactly 4
    - Eq. 2.10: Emergent metric g_μν(x) from condensate
    - Theorem 2.1: Exact 4D spacetime from quaternionic cGFT
    - §2.1.1: Quaternionic Necessity Principle (algebraic derivation of d=4)
    - Theorem H.1: Lorentzian signature emergence
    - Theorem C.3: Einstein-Hilbert from Harmony Functional

Modules:
    spectral_dimension: d_spec(k) flow (Eq. 2.8-2.9)
    metric_tensor: g_μν(x) from condensate (Eq. 2.10)
    lorentzian_signature: Spontaneous ℤ₂ breaking, timelike direction
    einstein_equations: Variational derivation from Harmony Functional

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)

Authors: IRH Computational Framework Team
Last Updated: December 2024
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §2.1-2.5"

# Spectral dimension constants
from .spectral_dimension import (
    D_SPEC_ONE_LOOP,
    D_SPEC_IR,
    DELTA_GRAV_COEFFICIENT,
    D_SPEC_UV,
    ETA_STAR,
    SpectralDimensionResult,
    SpectralDimensionFlow,
    compute_spectral_dimension,
    spectral_dimension_flow,
    anomalous_dimension,
    graviton_correction,
    verify_theorem_2_1,
    heat_kernel_spectral_dimension,
    generate_spectral_dimension_summary,
)

# Metric tensor
from .metric_tensor import (
    SPACETIME_DIM,
    MINKOWSKI_SIGNATURE,
    PLANCK_LENGTH,
    MetricTensor,
    EmergentGeometry,
    minkowski_metric,
    schwarzschild_metric,
    metric_from_condensate,
    emergent_metric,
    verify_lorentzian_signature,
    verify_metric_properties,
    generate_metric_tensor_summary,
)

# Lorentzian signature
from .lorentzian_signature import (
    LORENTZIAN_SIGNATURE,
    EUCLIDEAN_SIGNATURE,
    N_TIMELIKE,
    N_SPACELIKE,
    SignatureResult,
    Z2SymmetryBreaking,
    compute_signature,
    verify_lorentzian,
    z2_symmetry_breaking,
    signature_from_condensate,
    timelike_direction,
    light_cone_structure,
    verify_theorem_h1,
    generate_lorentzian_signature_summary,
)

# Einstein equations
from .einstein_equations import (
    G_NEWTON,
    M_PLANCK,
    L_PLANCK as PLANCK_LENGTH_SI,
    LAMBDA_OBSERVED,
    C_H_SPECTRAL,
    EinsteinTensor,
    EinsteinFieldEquations,
    HarmonyFunctional,
    compute_einstein_tensor,
    einstein_hilbert_action,
    derive_einstein_equations,
    compute_cosmological_constant,
    verify_theorem_c3,
    vacuum_einstein_equations,
    generate_einstein_equations_summary,
)

# Legacy constants (backward compatibility)
SPECTRAL_DIMENSION_IR = D_SPEC_IR  # 4.0
SPECTRAL_DIMENSION_ONE_LOOP = D_SPEC_ONE_LOOP  # 42/11

__all__ = [
    # Version info
    '__version__',
    '__theoretical_foundation__',
    
    # Spectral dimension
    'D_SPEC_ONE_LOOP',
    'D_SPEC_IR',
    'DELTA_GRAV_COEFFICIENT',
    'D_SPEC_UV',
    'ETA_STAR',
    'SPECTRAL_DIMENSION_IR',
    'SPECTRAL_DIMENSION_ONE_LOOP',
    'SpectralDimensionResult',
    'SpectralDimensionFlow',
    'compute_spectral_dimension',
    'spectral_dimension_flow',
    'anomalous_dimension',
    'graviton_correction',
    'verify_theorem_2_1',
    'heat_kernel_spectral_dimension',
    'generate_spectral_dimension_summary',
    
    # Metric tensor
    'SPACETIME_DIM',
    'MINKOWSKI_SIGNATURE',
    'PLANCK_LENGTH',
    'MetricTensor',
    'EmergentGeometry',
    'minkowski_metric',
    'schwarzschild_metric',
    'metric_from_condensate',
    'emergent_metric',
    'verify_lorentzian_signature',
    'verify_metric_properties',
    'generate_metric_tensor_summary',
    
    # Lorentzian signature
    'LORENTZIAN_SIGNATURE',
    'EUCLIDEAN_SIGNATURE',
    'N_TIMELIKE',
    'N_SPACELIKE',
    'SignatureResult',
    'Z2SymmetryBreaking',
    'compute_signature',
    'verify_lorentzian',
    'z2_symmetry_breaking',
    'signature_from_condensate',
    'timelike_direction',
    'light_cone_structure',
    'verify_theorem_h1',
    'generate_lorentzian_signature_summary',
    
    # Einstein equations
    'G_NEWTON',
    'M_PLANCK',
    'PLANCK_LENGTH_SI',
    'LAMBDA_OBSERVED',
    'C_H_SPECTRAL',
    'EinsteinTensor',
    'EinsteinFieldEquations',
    'HarmonyFunctional',
    'compute_einstein_tensor',
    'einstein_hilbert_action',
    'derive_einstein_equations',
    'compute_cosmological_constant',
    'verify_theorem_c3',
    'vacuum_einstein_equations',
    'generate_einstein_equations_summary',
]
