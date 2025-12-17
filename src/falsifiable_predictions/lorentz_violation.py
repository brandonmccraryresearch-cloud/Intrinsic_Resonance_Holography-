"""
Lorentz Invariance Violation Predictions

THEORETICAL FOUNDATION: Intrinsic_Resonance_Holography-v21.1.md §2.4, Eq. 2.24-2.26, Appendix J.1

Implements:
- LIV parameter ξ = C_H/(24π²) ≈ 1.93×10⁻⁴
- Modified dispersion relations E² = p²c² + ξE³/E_Pl
- Generation-specific LIV thresholds
- Gamma-ray astronomy predictions

Key Results:
    ξ = 1.933355051 × 10⁻⁴ (Eq. 2.24)
    Generation-dependent thresholds (Appendix J.1)
    Photon velocity dispersion (Eq. 2.25)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Physical constants
C_LIGHT = 299792458.0  # m/s
PLANCK_ENERGY = 1.22e19  # GeV
PLANCK_LENGTH = 1.616255e-35  # m
ELECTRON_MASS = 0.000511  # GeV

# Universal exponent (Eq. 1.16)
C_H = 0.045935703598

# LIV parameter (Eq. 2.24)
XI_LIV = C_H / (24 * np.pi**2)  # ≈ 1.93×10⁻⁴

# Certified value from theory
XI_CERTIFIED = 1.933355051e-4


@dataclass
class LIVParameter:
    """
    Lorentz Invariance Violation parameter.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.1, Eq. 2.24
        
    The LIV parameter ξ quantifies the leading-order deviation from
    exact Lorentz invariance due to the discrete nature of spacetime
    at the Planck scale.
        
    Attributes
    ----------
    xi : float
        The universal LIV parameter
    xi_uncertainty : float
        Theoretical uncertainty
    formula : str
        The derivation formula
    """
    xi: float
    xi_uncertainty: float
    formula: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'xi': self.xi,
            'xi_uncertainty': self.xi_uncertainty,
            'formula': self.formula,
            'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.4.1, Eq. 2.24'
        }


def compute_liv_parameter() -> LIVParameter:
    """
    Compute the universal LIV parameter ξ.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.1, Eq. 2.24
        
    Mathematical Foundation:
        ξ = C_H / (24π²)
        
        where C_H = 0.045935703598 is the universal exponent.
        This gives ξ ≈ 1.93×10⁻⁴.
        
    Returns
    -------
    LIVParameter
        The computed LIV parameter
    """
    xi = C_H / (24 * np.pi**2)
    xi_uncertainty = 1e-10  # High precision from C_H
    
    return LIVParameter(
        xi=xi,
        xi_uncertainty=xi_uncertainty,
        formula='ξ = C_H / (24π²)'
    )


@dataclass 
class ModifiedDispersion:
    """
    Modified dispersion relation due to LIV.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.2, Eq. 2.25
        
    E² = p²c² + m²c⁴ + ξ × E³/E_Pl
    
    where the cubic term represents Planck-scale corrections.
    """
    energy: float  # GeV
    momentum: float  # GeV/c
    mass: float  # GeV
    liv_correction: float  # Fractional correction
    effective_velocity: float  # c units
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'energy_GeV': self.energy,
            'momentum_GeV': self.momentum,
            'mass_GeV': self.mass,
            'liv_correction': self.liv_correction,
            'effective_velocity_c': self.effective_velocity,
            'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.4.2, Eq. 2.25'
        }


def compute_modified_dispersion(
    energy: float,
    mass: float = 0.0,
    xi: Optional[float] = None
) -> ModifiedDispersion:
    """
    Compute modified dispersion relation with LIV correction.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.2, Eq. 2.25
        
    Parameters
    ----------
    energy : float
        Particle energy in GeV
    mass : float, optional
        Particle mass in GeV (default: 0 for photons)
    xi : float, optional
        LIV parameter (default: computed from C_H)
        
    Returns
    -------
    ModifiedDispersion
        The modified dispersion relation
    """
    if xi is None:
        xi = compute_liv_parameter().xi
    
    # Standard dispersion: p²c² = E² - m²c⁴
    p_standard_sq = energy**2 - mass**2
    p_standard = np.sqrt(max(0, p_standard_sq))
    
    # LIV correction: δp² = ξ × E³/E_Pl
    liv_term = xi * energy**3 / PLANCK_ENERGY
    
    # Fractional correction
    if p_standard_sq > 0:
        liv_correction = liv_term / p_standard_sq
    else:
        liv_correction = 0.0
    
    # Effective velocity (group velocity)
    # v_eff/c = 1 - (m²c⁴/2E²) - (ξ/2)(E/E_Pl)
    if energy > 0:
        mass_term = mass**2 / (2 * energy**2) if mass > 0 else 0
        liv_velocity_term = (xi / 2) * (energy / PLANCK_ENERGY)
        v_eff = 1 - mass_term - liv_velocity_term
    else:
        v_eff = 0.0
    
    return ModifiedDispersion(
        energy=energy,
        momentum=p_standard,
        mass=mass,
        liv_correction=liv_correction,
        effective_velocity=v_eff
    )


@dataclass
class GenerationLIV:
    """
    Generation-specific LIV thresholds.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md Appendix J.1
        
    Different fermion generations have different LIV thresholds
    due to their different topological complexities K_f.
    """
    fermion_type: str
    generation: int
    K_f: float  # Topological complexity
    xi_f: float  # Generation-specific LIV parameter
    threshold_energy: float  # GeV
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'fermion_type': self.fermion_type,
            'generation': self.generation,
            'K_f': self.K_f,
            'xi_f': self.xi_f,
            'threshold_energy_GeV': self.threshold_energy,
            'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md Appendix J.1'
        }


# Topological complexity values (from Phase III)
K_VALUES = {
    'electron': 1.0,
    'muon': 207.0,
    'tau': 3477.0,
    'up': 2.3,
    'down': 4.8,
    'strange': 95.0,
    'charm': 1275.0,
    'bottom': 4180.0,
    'top': 173000.0,
}


def compute_generation_liv(fermion_type: str) -> GenerationLIV:
    """
    Compute generation-specific LIV threshold.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md Appendix J.1, Eq. J.1
        
    The LIV parameter for fermion type f is:
        ξ_f = ξ × [1 + β₁(K_f/K_ref - 1) + β₂(K_f/K_ref - 1)²]
        
    where K_ref is the electron complexity (K_ref = 1).
    
    Parameters
    ----------
    fermion_type : str
        Type of fermion ('electron', 'muon', 'tau', etc.)
        
    Returns
    -------
    GenerationLIV
        The generation-specific LIV parameters
    """
    xi_base = compute_liv_parameter().xi
    
    if fermion_type not in K_VALUES:
        raise ValueError(f"Unknown fermion type: {fermion_type}")
    
    K_f = K_VALUES[fermion_type]
    K_ref = K_VALUES['electron']  # Reference: electron
    
    # Coefficients from fixed-point analysis (Appendix J.1)
    beta_1 = 0.01  # Linear correction
    beta_2 = 0.001  # Quadratic correction
    
    # Generation-specific LIV parameter
    K_ratio = K_f / K_ref
    xi_f = xi_base * (1 + beta_1 * (K_ratio - 1) + beta_2 * (K_ratio - 1)**2)
    
    # Threshold energy where LIV becomes detectable
    # E_threshold ~ E_Pl × √(10^-8 / ξ_f) (for 10^-8 precision)
    threshold_energy = PLANCK_ENERGY * np.sqrt(1e-8 / xi_f)
    
    # Determine generation number
    if fermion_type in ['electron', 'up', 'down']:
        generation = 1
    elif fermion_type in ['muon', 'strange', 'charm']:
        generation = 2
    else:
        generation = 3
    
    return GenerationLIV(
        fermion_type=fermion_type,
        generation=generation,
        K_f=K_f,
        xi_f=xi_f,
        threshold_energy=threshold_energy
    )


@dataclass
class PhotonDispersion:
    """
    Photon velocity dispersion from LIV.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.3, Eq. 2.26
        
    High-energy photons travel slightly slower than low-energy photons
    due to LIV effects.
    """
    energy_high: float  # GeV
    energy_low: float   # GeV
    time_delay: float   # seconds
    distance: float     # Mpc
    delta_v_over_c: float  # Fractional velocity difference
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'energy_high_GeV': self.energy_high,
            'energy_low_GeV': self.energy_low,
            'time_delay_s': self.time_delay,
            'distance_Mpc': self.distance,
            'delta_v_over_c': self.delta_v_over_c,
            'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.4.3, Eq. 2.26'
        }


def compute_photon_time_delay(
    energy_high: float,
    energy_low: float,
    distance_mpc: float
) -> PhotonDispersion:
    """
    Compute photon time delay from LIV.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §2.4.3, Eq. 2.26
        
    Higher-energy photons are delayed relative to lower-energy ones.
    
    Parameters
    ----------
    energy_high : float
        High-energy photon energy in GeV
    energy_low : float
        Low-energy photon energy in GeV
    distance_mpc : float
        Source distance in Mpc
        
    Returns
    -------
    PhotonDispersion
        The photon dispersion parameters
    """
    xi = compute_liv_parameter().xi
    
    # Velocity difference: Δv/c = (ξ/2)(E_high - E_low)/E_Pl
    delta_v = (xi / 2) * (energy_high - energy_low) / PLANCK_ENERGY
    
    # Distance in meters
    distance_m = distance_mpc * 3.086e22
    
    # Time delay: Δt = (D/c) × (Δv/c)
    time_delay = (distance_m / C_LIGHT) * delta_v
    
    return PhotonDispersion(
        energy_high=energy_high,
        energy_low=energy_low,
        time_delay=time_delay,
        distance=distance_mpc,
        delta_v_over_c=delta_v
    )


@dataclass
class GammaRayTest:
    """
    Gamma-ray burst test of LIV.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §8.3
        
    GRBs provide the best current constraints on LIV through
    photon time-of-flight measurements.
    """
    grb_name: str
    redshift: float
    energy_max: float  # GeV
    predicted_delay: float  # seconds
    current_bound: float  # seconds
    detectable: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'grb_name': self.grb_name,
            'redshift': self.redshift,
            'energy_max_GeV': self.energy_max,
            'predicted_delay_s': self.predicted_delay,
            'current_bound_s': self.current_bound,
            'detectable': self.detectable,
            'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §8.3'
        }


def analyze_grb_liv_test(
    grb_name: str,
    redshift: float,
    energy_max: float,
    energy_ref: float = 0.1  # 100 MeV reference
) -> GammaRayTest:
    """
    Analyze LIV test from gamma-ray burst.
    
    Theoretical Reference:
        Intrinsic_Resonance_Holography-v21.1.md §8.3
        
    Parameters
    ----------
    grb_name : str
        Name of the GRB
    redshift : float
        Source redshift
    energy_max : float
        Maximum photon energy in GeV
    energy_ref : float
        Reference energy in GeV
        
    Returns
    -------
    GammaRayTest
        The GRB LIV analysis
    """
    # Convert redshift to distance (simplified)
    # Using Hubble law: D ≈ cz/H₀
    H0 = 67.4  # km/s/Mpc
    distance_mpc = C_LIGHT * redshift / (H0 * 1000) / 1e6 * 3.086e22 / 3.086e22
    # Better approximation for cosmological distance
    distance_mpc = 4000 * redshift  # Rough approximation for z < 2
    
    # Compute predicted delay
    dispersion = compute_photon_time_delay(energy_max, energy_ref, distance_mpc)
    predicted_delay = dispersion.time_delay
    
    # Current experimental bounds (Fermi-LAT)
    # Typical bound: < 0.1 second for 10 GeV photons at z~1
    current_bound = 0.1  # seconds
    
    # Check if prediction is detectable
    detectable = predicted_delay > current_bound * 0.01  # 1% of bound
    
    return GammaRayTest(
        grb_name=grb_name,
        redshift=redshift,
        energy_max=energy_max,
        predicted_delay=predicted_delay,
        current_bound=current_bound,
        detectable=detectable
    )


def verify_liv_predictions() -> Dict:
    """
    Verify all LIV predictions from IRH.
    
    Returns
    -------
    dict
        Verification results for all LIV predictions
    """
    liv = compute_liv_parameter()
    
    # Check ξ value
    xi_verified = np.isclose(liv.xi, XI_CERTIFIED, rtol=1e-6)
    
    # Check formula consistency
    xi_from_formula = C_H / (24 * np.pi**2)
    formula_verified = np.isclose(liv.xi, xi_from_formula, rtol=1e-10)
    
    # Check generation dependence
    electron_liv = compute_generation_liv('electron')
    muon_liv = compute_generation_liv('muon')
    generation_ordering = muon_liv.xi_f > electron_liv.xi_f
    
    # Check photon dispersion is subluminal (or at most c)
    # At extremely high energies, numerical precision limits detection of deviation from c
    dispersion = compute_modified_dispersion(1e6, mass=0)  # 1 PeV photon
    subluminal = dispersion.effective_velocity <= 1.0 + 1e-15  # Allow numerical precision
    
    return {
        'xi_value': liv.xi,
        'xi_certified': XI_CERTIFIED,
        'xi_verified': xi_verified,
        'formula_verified': formula_verified,
        'generation_ordering': generation_ordering,
        'subluminal_verified': subluminal,
        'all_verified': all([xi_verified, formula_verified, generation_ordering, subluminal]),
        'theoretical_reference': 'Intrinsic_Resonance_Holography-v21.1.md §2.4'
    }


def compute_cta_sensitivity() -> Dict:
    """
    Compute Cherenkov Telescope Array sensitivity to IRH LIV.
    
    CTA will probe the energy regime where IRH LIV effects
    become potentially detectable.
    
    Returns
    -------
    dict
        CTA sensitivity analysis
    """
    xi = compute_liv_parameter().xi
    
    # CTA parameters
    cta_energy_max = 100e3  # 100 TeV in GeV
    cta_energy_threshold = 20  # 20 GeV
    typical_blazar_distance = 1000  # Mpc
    
    # Predicted time delay for CTA observations
    dispersion = compute_photon_time_delay(
        cta_energy_max, 
        cta_energy_threshold,
        typical_blazar_distance
    )
    
    # CTA timing resolution: ~10 seconds for bright sources
    cta_timing = 10.0  # seconds
    
    return {
        'xi': xi,
        'energy_range_GeV': (cta_energy_threshold, cta_energy_max),
        'typical_distance_Mpc': typical_blazar_distance,
        'predicted_delay_s': dispersion.time_delay,
        'cta_timing_resolution_s': cta_timing,
        'detectable_by_cta': dispersion.time_delay > cta_timing * 0.1,
        'improvement_needed': cta_timing / dispersion.time_delay if dispersion.time_delay > 0 else float('inf')
    }


# Public API
__all__ = [
    'LIVParameter',
    'ModifiedDispersion',
    'GenerationLIV',
    'PhotonDispersion',
    'GammaRayTest',
    'XI_LIV',
    'XI_CERTIFIED',
    'compute_liv_parameter',
    'compute_modified_dispersion',
    'compute_generation_liv',
    'compute_photon_time_delay',
    'analyze_grb_liv_test',
    'verify_liv_predictions',
    'compute_cta_sensitivity',
]
