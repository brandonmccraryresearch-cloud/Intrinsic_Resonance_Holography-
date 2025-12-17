"""
Standard Model Emergence Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §3.1-3.4

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
    fermion_masses: Yukawa couplings (Eq. 3.6), 𝒦_f values (Table 3.1)
    mixing_matrices: CKM and PMNS matrices from VWP overlaps
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
Last Updated: December 2024 (synchronized with IRH v21.1 Manuscript v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §3.1-3.4"

# Import from fermion_masses module (§3.2, Eq. 3.6)
from .fermion_masses import (
    compute_fermion_mass,
    yukawa_coupling,
    mass_hierarchy,
    verify_mass_ratios,
    TOPOLOGICAL_COMPLEXITY,
    HIGGS_VEV,
)

# Import from gauge_groups module (§3.1.1, Appendix D.1)
from .gauge_groups import (
    GaugeGroup,
    StandardModelGaugeStructure,
    GaugeCouplingUnification,
    SU3_COLOR,
    SU2_WEAK,
    U1_HYPERCHARGE,
    derive_gauge_group,
    verify_su3_su2_u1,
    compute_gauge_coupling_running,
    BETTI_1,
)

# Import from mixing_matrices module (§3.2.3)
from .mixing_matrices import (
    CKMMatrix,
    PMNSMatrix,
    compute_ckm_matrix,
    compute_pmns_matrix,
    verify_mixing_matrices,
)

# Import from higgs_sector module (§3.3)
from .higgs_sector import (
    HiggsSector,
    GaugeBosonMasses,
    compute_higgs_sector,
    compute_gauge_boson_masses,
    verify_electroweak_sector,
    HIGGS_MASS_EXP,
)

# Import from neutrinos module (§3.2.4, Appendix E.3)
from .neutrinos import (
    NeutrinoMasses,
    MajoranaNature,
    compute_neutrino_masses,
    compute_majorana_nature,
    verify_neutrino_sector,
    neutrino_hierarchy,
    K_NU,
)

# Import from strong_cp module (§3.4)
from .strong_cp import (
    StrongCPResolution,
    AlgorithmicAxion,
    compute_strong_cp_resolution,
    compute_algorithmic_axion,
    verify_strong_cp_sector,
)

# Topological complexity eigenvalues (Appendix E.1)
K_1 = 1.000          # First generation (electron, u, d)
K_2 = 206.77         # Second generation (muon, c, s)
K_3 = 3477.15        # Third generation (tau, t, b)

__all__ = [
    # Constants
    'K_1',
    'K_2',
    'K_3',
    'BETTI_1',
    'HIGGS_VEV',
    'HIGGS_MASS_EXP',
    'K_NU',
    
    # fermion_masses exports (§3.2, Eq. 3.6)
    'compute_fermion_mass',
    'yukawa_coupling',
    'mass_hierarchy',
    'verify_mass_ratios',
    'TOPOLOGICAL_COMPLEXITY',
    
    # gauge_groups exports (§3.1.1)
    'GaugeGroup',
    'StandardModelGaugeStructure',
    'GaugeCouplingUnification',
    'SU3_COLOR',
    'SU2_WEAK',
    'U1_HYPERCHARGE',
    'derive_gauge_group',
    'verify_su3_su2_u1',
    'compute_gauge_coupling_running',
    
    # mixing_matrices exports (§3.2.3)
    'CKMMatrix',
    'PMNSMatrix',
    'compute_ckm_matrix',
    'compute_pmns_matrix',
    'verify_mixing_matrices',
    
    # higgs_sector exports (§3.3)
    'HiggsSector',
    'GaugeBosonMasses',
    'compute_higgs_sector',
    'compute_gauge_boson_masses',
    'verify_electroweak_sector',
    
    # neutrinos exports (§3.2.4, Appendix E.3)
    'NeutrinoMasses',
    'MajoranaNature',
    'compute_neutrino_masses',
    'compute_majorana_nature',
    'verify_neutrino_sector',
    'neutrino_hierarchy',
    
    # strong_cp exports (§3.4)
    'StrongCPResolution',
    'AlgorithmicAxion',
    'compute_strong_cp_resolution',
    'compute_algorithmic_axion',
    'verify_strong_cp_sector',
]
