"""
Standard Model Emergence Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §3.1-3.4

This module derives the Standard Model of particle physics from the topological
structure of the IR fixed point. Gauge groups emerge from β₁=12, fermion
generations from n_inst=3, and all masses from topological complexity.

Key Results:
    - §3.1: SU(3)×SU(2)×U(1) from β₁ = 12
    - §3.2: Fermion masses from topological complexity 𝒦_f
    - §3.3: Gauge boson masses and Higgs sector
    - §3.4: Strong CP problem resolution

Modules:
    gauge_groups: SU(3)×SU(2)×U(1) from β₁=12
    fermion_masses: Yukawa couplings, 𝒦_f values (Table 3.1)
    gauge_bosons: W, Z, γ, g masses and couplings
    higgs_sector: VEV, λ_H, electroweak symmetry breaking
    neutrinos: Masses, mixing, Majorana nature (Appendix E.3)
    strong_cp: Algorithmic axion, θ-angle resolution

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)
    - src.topology (Layer 4)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.1-3.4"

# Topological complexity eigenvalues (Appendix E.1)
K_1 = 1.000          # First generation (electron, u, d)
K_2 = 206.77         # Second generation (muon, c, s)
K_3 = 3477.15        # Third generation (tau, t, b)

__all__ = [
    # Constants
    'K_1',
    'K_2',
    'K_3',
    
    # gauge_groups exports
    'derive_gauge_group',
    'verify_su3_su2_u1',
    
    # fermion_masses exports
    'compute_fermion_mass',
    'yukawa_coupling',
    'mass_hierarchy',
    
    # gauge_bosons exports
    'w_boson_mass',
    'z_boson_mass',
    'gluon_properties',
    
    # higgs_sector exports
    'higgs_vev',
    'higgs_mass',
    'higgs_quartic_coupling',
    
    # neutrinos exports
    'neutrino_masses',
    'pmns_matrix',
    'verify_normal_hierarchy',
    
    # strong_cp exports
    'algorithmic_axion',
    'theta_angle',
]
