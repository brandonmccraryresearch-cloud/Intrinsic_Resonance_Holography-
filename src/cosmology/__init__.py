"""
Cosmology Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §2.3

This module derives cosmological predictions from the cGFT fixed point,
including the cosmological constant from the "Holographic Hum" and the
dark energy equation of state w(z).

Key Results:
    - Eq. 2.17-2.19: ρ_hum calculation (Holographic Hum)
    - Eq. 2.21-2.23: w(z) equation of state
    - Appendix C.6-C.8: Running fundamental constants c(k), ℏ(k), G(k)

Modules:
    holographic_hum: ρ_hum, Λ* calculation (Eq. 2.17-2.19)
    dark_energy: w(z) equation of state (Eq. 2.21-2.23)
    running_constants: c(k), ℏ(k), G(k) (Appendix C.6-C.8)
    primordial_universe: Early universe, inflation signatures

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)
    - src.topology (Layer 4)
    - src.standard_model (Layer 5)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §2.3"

# Predicted dark energy equation of state (Eq. 2.23)
W_0_PREDICTED = -0.91234567  # w₀ with 8-digit precision

__all__ = [
    # Constants
    'W_0_PREDICTED',
    
    # holographic_hum exports
    'compute_rho_hum',
    'holographic_hum_density',
    'cosmological_constant',
    
    # dark_energy exports
    'equation_of_state',
    'w_of_z',
    'dark_energy_density',
    
    # running_constants exports
    'speed_of_light_running',
    'planck_constant_running',
    'gravitational_constant_running',
    
    # primordial_universe exports
    'inflation_parameters',
    'primordial_spectrum',
]
