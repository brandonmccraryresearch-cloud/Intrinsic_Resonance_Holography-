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
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.1, Appendix D"

# Topological invariants at the Cosmic Fixed Point
BETTI_1 = 12    # First Betti number → SU(3)×SU(2)×U(1) = 8+3+1 generators
N_INST = 3      # Instanton number → 3 fermion generations

__all__ = [
    # Constants
    'BETTI_1',
    'N_INST',
    
    # betti_numbers exports
    'compute_betti_1',
    'verify_betti_12',
    
    # instanton_number exports
    'compute_instanton_number',
    'verify_three_generations',
    
    # vortex_wave_patterns exports
    'VortexWavePattern',
    'find_stable_vwps',
    'topological_complexity',
    
    # homology exports
    'compute_homology',
    'persistent_homology',
    
    # manifold_construction exports
    'construct_M3',
    'resonance_quotient',
]
