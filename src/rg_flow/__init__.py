"""
Renormalization Group Flow Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §1.2-1.3

This module implements the meta-algorithm of reality: the Wetterich equation
(Eq. 1.12) and its consequences. The RG flow represents the process ontology
wherein laws themselves emerge through asymptotic safety.

This layer contains NO direct reference to spacetime or particles—only
abstract coupling dynamics that determine the Cosmic Fixed Point.

Key Equations:
    - Eq. 1.12: Wetterich equation ∂_t Γ_k = (1/2) Tr[(Γ_k^(2) + R_k)^(-1) ∂_t R_k]
    - Eq. 1.13: β-functions for (λ̃, γ̃, μ̃)
    - Eq. 1.14: Fixed-point values (λ̃*, γ̃*, μ̃*)
    - Eq. 1.16: Universal exponent C_H = 0.045935703598...

Modules:
    wetterich: Exact RG equation integrator
    beta_functions: β_λ, β_γ, β_μ (Eq. 1.13)
    fixed_points: Cosmic Fixed Point solver
    running_couplings: Scale-dependent parameter evolution
    stability_analysis: Eigenvalue spectrum, IR attractiveness

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.2-1.3"

# Fixed-point coupling values (Eq. 1.14)
import math

LAMBDA_STAR = 48 * math.pi**2 / 9    # λ̃* = 48π²/9
GAMMA_STAR = 32 * math.pi**2 / 3     # γ̃* = 32π²/3
MU_STAR = 16 * math.pi**2             # μ̃* = 16π²

# Universal exponent (Eq. 1.16)
C_H = 0.045935703598  # First analytically computed constant of Nature

__all__ = [
    # Constants
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
    'C_H',
    
    # wetterich exports
    'WetterichEquation',
    'integrate_rg_flow',
    
    # beta_functions exports
    'beta_lambda',
    'beta_gamma',
    'beta_mu',
    'compute_all_betas',
    
    # fixed_points exports
    'find_fixed_point',
    'verify_fixed_point',
    'compute_C_H',
    
    # running_couplings exports
    'lambda_running',
    'gamma_running',
    'mu_running',
    
    # stability_analysis exports
    'stability_matrix',
    'compute_eigenvalues',
    'verify_ir_attractiveness',
]
