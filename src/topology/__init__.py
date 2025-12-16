"""
Topological Structures Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §3.1, Appendix D

This module computes the topological invariants of the emergent 3-manifold M³
that encode the gauge structure of particle physics:
    - β₁ = 12: First Betti number → 12 gauge generators → SU(3)×SU(2)×U(1)
    - n_inst = 3: Instanton number → 3 fermion generations

Key Results:
    - Appendix D.1: Proof that β₁(M³) = 12 (gauge group emergence)
    - Appendix D.2: Proof that n_inst = 3 (three generations)
    - VWPs: Vortex Wave Patterns as fermionic defects

Modules:
    betti_numbers: β₁ = 12 computation (Appendix D.1)
    instanton_number: n_inst = 3 calculation (Appendix D.2)
    vortex_wave_patterns: Fermionic defects, topological complexity
    homology: Persistent homology, Morse theory
    manifold_construction: Resonance quotient M³ from condensate

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)

Authors: IRH Computational Framework Team
Last Updated: December 2024
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.1, Appendix D"

# Betti numbers module
from .betti_numbers import (
    BETTI_1,
    SU3_GENERATORS,
    SU2_GENERATORS,
    U1_GENERATORS,
    TOTAL_GENERATORS,
    BettiNumberResult,
    GaugeGroupDecomposition,
    ResonanceQuotient,
    compute_betti_1,
    verify_betti_12,
    gauge_group_from_betti,
    compute_homology_groups,
    generate_betti_number_summary,
)

# Instanton number module
from .instanton_number import (
    N_INST,
    K_1, K_2, K_3,
    GENERATION_1, GENERATION_2, GENERATION_3,
    MORSE_MINIMA,
    InstantonResult,
    TopologicalCharge,
    VortexWavePatternConfig,
    FermionGeneration,
    compute_instanton_number,
    verify_three_generations,
    get_fermion_generations,
    topological_complexity,
    compute_mass_hierarchy_ratios,
    find_stable_vwps,
    generate_instanton_number_summary,
)

# Vortex Wave Patterns module
from .vortex_wave_patterns import (
    M_ELECTRON,
    HIGGS_VEV,
    TOPOLOGICAL_COMPLEXITIES,
    VortexWavePattern,
    VWPSpectrum,
    ComplexityOperator,
    create_standard_model_vwps,
    compute_vwp_mass,
    topological_complexity_operator,
    vwp_overlap_integral,
    verify_vwp_stability,
    generate_vwp_summary,
)

# Homology module
from .homology import (
    M3_BETTI_NUMBERS,
    M3_DIMENSION,
    HomologyGroup,
    HomologyComputation,
    PersistentHomologyResult,
    compute_homology,
    persistent_homology,
    compute_euler_characteristic,
    verify_poincare_duality,
    generate_homology_summary,
)

# Manifold construction module
from .manifold_construction import (
    G_INF_DIM,
    GAMMA_R_DIM,
    M3_DIM,
    M3_EULER,
    GroupManifold,
    SU2Manifold,
    U1Manifold,
    GInfManifold,
    ResonanceQuotientM3,
    construct_M3,
    resonance_quotient,
    verify_manifold_properties,
    compute_fundamental_group,
    generate_manifold_summary,
)

__all__ = [
    # Version info
    '__version__',
    '__theoretical_foundation__',
    
    # Betti numbers
    'BETTI_1',
    'SU3_GENERATORS',
    'SU2_GENERATORS',
    'U1_GENERATORS',
    'TOTAL_GENERATORS',
    'BettiNumberResult',
    'GaugeGroupDecomposition',
    'ResonanceQuotient',
    'compute_betti_1',
    'verify_betti_12',
    'gauge_group_from_betti',
    'compute_homology_groups',
    'generate_betti_number_summary',
    
    # Instanton number
    'N_INST',
    'K_1', 'K_2', 'K_3',
    'GENERATION_1', 'GENERATION_2', 'GENERATION_3',
    'MORSE_MINIMA',
    'InstantonResult',
    'TopologicalCharge',
    'VortexWavePatternConfig',
    'FermionGeneration',
    'compute_instanton_number',
    'verify_three_generations',
    'get_fermion_generations',
    'topological_complexity',
    'compute_mass_hierarchy_ratios',
    'find_stable_vwps',
    'generate_instanton_number_summary',
    
    # Vortex Wave Patterns
    'M_ELECTRON',
    'HIGGS_VEV',
    'TOPOLOGICAL_COMPLEXITIES',
    'VortexWavePattern',
    'VWPSpectrum',
    'ComplexityOperator',
    'create_standard_model_vwps',
    'compute_vwp_mass',
    'topological_complexity_operator',
    'vwp_overlap_integral',
    'verify_vwp_stability',
    'generate_vwp_summary',
    
    # Homology
    'M3_BETTI_NUMBERS',
    'M3_DIMENSION',
    'HomologyGroup',
    'HomologyComputation',
    'PersistentHomologyResult',
    'compute_homology',
    'persistent_homology',
    'compute_euler_characteristic',
    'verify_poincare_duality',
    'generate_homology_summary',
    
    # Manifold construction
    'G_INF_DIM',
    'GAMMA_R_DIM',
    'M3_DIM',
    'M3_EULER',
    'GroupManifold',
    'SU2Manifold',
    'U1Manifold',
    'GInfManifold',
    'ResonanceQuotientM3',
    'construct_M3',
    'resonance_quotient',
    'verify_manifold_properties',
    'compute_fundamental_group',
    'generate_manifold_summary',
]
