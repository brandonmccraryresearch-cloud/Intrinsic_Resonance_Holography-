"""
Gravitational Wave Sidebands Predictions

THEORETICAL FOUNDATION: IRH21.md §8.4

Implements:
- GW sideband structure from discrete spacetime
- Frequency modulation predictions
- LIGO/Virgo/LISA detectability analysis

Key Results:
    Sideband frequency: f_sideband = f_GW × (1 ± ξ × (f_GW/f_Pl))
    Planck-scale modulation signatures
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Physical constants
C_LIGHT = 299792458.0  # m/s
PLANCK_FREQUENCY = 1.85e43  # Hz
PLANCK_LENGTH = 1.616255e-35  # m
PLANCK_TIME = 5.391e-44  # s

# Fixed-point values
LAMBDA_STAR = 48 * np.pi**2 / 9
GAMMA_STAR = 32 * np.pi**2 / 3
MU_STAR = 16 * np.pi**2
C_H = 0.045935703598

# LIV parameter
XI_LIV = C_H / (24 * np.pi**2)


@dataclass
class GWSideband:
    """
    Gravitational wave sideband structure.
    
    Theoretical Reference:
        IRH21.md §8.4
        
    Attributes
    ----------
    f_gw : float
        Primary GW frequency (Hz)
    f_sideband_plus : float
        Upper sideband frequency (Hz)
    f_sideband_minus : float
        Lower sideband frequency (Hz)
    sideband_amplitude : float
        Relative amplitude of sidebands
    modulation_index : float
        Frequency modulation index
    """
    f_gw: float
    f_sideband_plus: float
    f_sideband_minus: float
    sideband_amplitude: float
    modulation_index: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'f_gw_Hz': self.f_gw,
            'f_sideband_plus_Hz': self.f_sideband_plus,
            'f_sideband_minus_Hz': self.f_sideband_minus,
            'sideband_amplitude': self.sideband_amplitude,
            'modulation_index': self.modulation_index,
            'theoretical_reference': 'IRH21.md §8.4'
        }


def compute_gw_sidebands(f_gw: float) -> GWSideband:
    """
    Compute gravitational wave sidebands from discrete spacetime.
    
    Theoretical Reference:
        IRH21.md §8.4
        
    The discrete nature of spacetime at the Planck scale induces
    small frequency modulations in gravitational waves.
    
    Parameters
    ----------
    f_gw : float
        Primary GW frequency in Hz
        
    Returns
    -------
    GWSideband
        The sideband structure
    """
    # Modulation index from LIV parameter
    # β = ξ × (f_GW / f_Pl)
    modulation_index = XI_LIV * (f_gw / PLANCK_FREQUENCY)
    
    # Sideband frequencies
    # f± = f_GW × (1 ± β)
    f_plus = f_gw * (1 + modulation_index)
    f_minus = f_gw * (1 - modulation_index)
    
    # Sideband amplitude (relative to primary)
    # A_sideband/A_primary ~ β²
    sideband_amplitude = modulation_index**2
    
    return GWSideband(
        f_gw=f_gw,
        f_sideband_plus=f_plus,
        f_sideband_minus=f_minus,
        sideband_amplitude=sideband_amplitude,
        modulation_index=modulation_index
    )


@dataclass
class GWDetectorSensitivity:
    """
    GW detector sensitivity for sideband detection.
    
    Attributes
    ----------
    detector_name : str
        Name of the detector
    frequency_range : Tuple[float, float]
        Sensitive frequency range (Hz)
    strain_sensitivity : float
        Best strain sensitivity (dimensionless)
    integration_time : float
        Required integration time (seconds)
    """
    detector_name: str
    frequency_range: Tuple[float, float]
    strain_sensitivity: float
    integration_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'detector': self.detector_name,
            'frequency_range_Hz': self.frequency_range,
            'strain_sensitivity': self.strain_sensitivity,
            'integration_time_s': self.integration_time
        }


# GW detector parameters
DETECTORS = {
    'LIGO': GWDetectorSensitivity(
        detector_name='LIGO',
        frequency_range=(10.0, 5000.0),
        strain_sensitivity=1e-23,
        integration_time=3600.0  # 1 hour
    ),
    'Virgo': GWDetectorSensitivity(
        detector_name='Virgo',
        frequency_range=(10.0, 5000.0),
        strain_sensitivity=3e-23,
        integration_time=3600.0
    ),
    'LISA': GWDetectorSensitivity(
        detector_name='LISA',
        frequency_range=(1e-4, 0.1),
        strain_sensitivity=1e-21,
        integration_time=31536000.0  # 1 year
    ),
    'Einstein_Telescope': GWDetectorSensitivity(
        detector_name='Einstein Telescope',
        frequency_range=(1.0, 10000.0),
        strain_sensitivity=1e-25,
        integration_time=3600.0
    ),
}


# Theoretical Reference: IRH v21.4



def analyze_detectability(
    f_gw: float,
    h_strain: float,
    detector: str = 'LIGO'
) -> Dict:
    """
    Analyze detectability of GW sidebands.
    
    Parameters
    ----------
    f_gw : float
        Primary GW frequency in Hz
    h_strain : float
        GW strain amplitude
    detector : str
        Detector name
        
    Returns
    -------
    dict
        Detectability analysis
    """
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector: {detector}")
    
    det = DETECTORS[detector]
    sidebands = compute_gw_sidebands(f_gw)
    
    # Check if frequency is in range
    in_range = det.frequency_range[0] <= f_gw <= det.frequency_range[1]
    
    # Sideband strain
    h_sideband = h_strain * sidebands.sideband_amplitude
    
    # SNR estimate
    snr = h_sideband / det.strain_sensitivity * np.sqrt(det.integration_time)
    
    # Detectability threshold (SNR > 8)
    detectable = snr > 8 and in_range
    
    return {
        'detector': detector,
        'f_gw': f_gw,
        'h_strain': h_strain,
        'h_sideband': h_sideband,
        'snr_estimate': snr,
        'in_frequency_range': in_range,
        'detectable': detectable,
        'required_strain_improvement': 8 / snr if snr > 0 else float('inf'),
        'theoretical_reference': 'IRH21.md §8.4'
    }


@dataclass
class SpacetimeGranularity:
    """
    Signatures of spacetime granularity in GWs.
    
    Theoretical Reference:
        IRH21.md §8.4
        
    Attributes
    ----------
    granularity_scale : float
        Effective granularity scale (m)
    phase_diffusion : float
        GW phase diffusion rate
    coherence_length : float
        GW coherence length (m)
    """
    granularity_scale: float
    phase_diffusion: float
    coherence_length: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'granularity_scale_m': self.granularity_scale,
            'phase_diffusion': self.phase_diffusion,
            'coherence_length_m': self.coherence_length,
            'theoretical_reference': 'IRH21.md §8.4'
        }


def compute_spacetime_granularity(f_gw: float, distance_m: float) -> SpacetimeGranularity:
    """
    Compute spacetime granularity effects on GWs.
    
    # Theoretical Reference:
        IRH21.md §8.4
        
    Parameters
    ----------
    f_gw : float
        GW frequency in Hz
    distance_m : float
        Propagation distance in meters
        
    Returns
    -------
    SpacetimeGranularity
        Granularity signature parameters
    """
    # Effective granularity scale
    # L_gran = ℓ_Pl × (f_Pl / f_GW)^{1/2}
    granularity_scale = PLANCK_LENGTH * np.sqrt(PLANCK_FREQUENCY / f_gw)
    
    # Phase diffusion rate
    # δφ/δt ~ ξ × (f_GW / f_Pl) × f_GW
    phase_diffusion = XI_LIV * (f_gw / PLANCK_FREQUENCY) * f_gw
    
    # Total phase shift over distance
    travel_time = distance_m / C_LIGHT
    total_phase_shift = phase_diffusion * travel_time
    
    # Coherence length (where phase shift ~ 1)
    if phase_diffusion > 0:
        coherence_length = C_LIGHT / phase_diffusion
    else:
        coherence_length = float('inf')
    
    return SpacetimeGranularity(
        granularity_scale=granularity_scale,
        phase_diffusion=phase_diffusion,
        coherence_length=coherence_length
    )


# Theoretical Reference: IRH v21.4



def predict_binary_merger_sidebands(
    m1_solar: float,
    m2_solar: float,
    distance_mpc: float
) -> Dict:
    """
    Predict sidebands for a binary black hole merger.
    
    Parameters
    ----------
    m1_solar : float
        Mass of first BH in solar masses
    m2_solar : float
        Mass of second BH in solar masses
    distance_mpc : float
        Distance in Mpc
        
    Returns
    -------
    dict
        Sideband predictions for the merger
    """
    # Solar mass and Mpc in SI
    M_SUN = 1.989e30  # kg
    MPC = 3.086e22  # m
    G = 6.674e-11  # m³/(kg·s²)
    
    # Total mass
    M_total = (m1_solar + m2_solar) * M_SUN
    distance_m = distance_mpc * MPC
    
    # ISCO frequency (approximation)
    # f_ISCO ≈ c³ / (6^{3/2} π G M)
    f_isco = C_LIGHT**3 / (6**1.5 * np.pi * G * M_total)
    
    # Ringdown frequency (approximation)
    f_ring = 32000 / (m1_solar + m2_solar)  # Hz
    
    # Strain amplitude (order of magnitude)
    h_strain = 4 * G * M_total / (C_LIGHT**2 * distance_m)
    
    # Compute sidebands for ringdown
    sidebands = compute_gw_sidebands(f_ring)
    
    # Detectability with LIGO
    detectability = analyze_detectability(f_ring, h_strain, 'LIGO')
    
    return {
        'm1_solar': m1_solar,
        'm2_solar': m2_solar,
        'distance_mpc': distance_mpc,
        'f_isco_Hz': f_isco,
        'f_ringdown_Hz': f_ring,
        'h_strain': h_strain,
        'sidebands': sidebands.to_dict(),
        'detectability': detectability,
        'theoretical_reference': 'IRH21.md §8.4'
    }


# Theoretical Reference: IRH v21.4



def verify_gw_sideband_predictions() -> Dict:
    """
    Verify GW sideband predictions from IRH.
    
    Returns
    -------
    dict
        Verification results
    """
    # Test at LIGO frequencies
    f_test = 100.0  # Hz
    sidebands = compute_gw_sidebands(f_test)
    
    # Sidebands should be:
    # 1. Very small modulation index (<<1)
    # 2. Symmetric around primary
    # 3. Amplitude ~ modulation_index²
    
    modulation_small = sidebands.modulation_index < 1e-30
    symmetric = np.isclose(
        sidebands.f_sideband_plus - f_test,
        f_test - sidebands.f_sideband_minus,
        rtol=1e-10
    )
    amplitude_correct = np.isclose(
        sidebands.sideband_amplitude,
        sidebands.modulation_index**2,
        rtol=1e-10
    )
    
    return {
        'test_frequency_Hz': f_test,
        'modulation_index': sidebands.modulation_index,
        'modulation_small': modulation_small,
        'sidebands_symmetric': symmetric,
        'amplitude_correct': amplitude_correct,
        'all_verified': modulation_small and symmetric and amplitude_correct,
        'theoretical_reference': 'IRH21.md §8.4'
    }


# Public API
__all__ = [
    'GWSideband',
    'GWDetectorSensitivity',
    'SpacetimeGranularity',
    'DETECTORS',
    'compute_gw_sidebands',
    'analyze_detectability',
    'compute_spacetime_granularity',
    'predict_binary_merger_sidebands',
    'verify_gw_sideband_predictions',
]
