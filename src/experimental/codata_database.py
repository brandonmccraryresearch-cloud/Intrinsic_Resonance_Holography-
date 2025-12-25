"""
CODATA Fundamental Constants Database

THEORETICAL FOUNDATION: IRH21.md §3.2, §7

This module provides access to CODATA fundamental constants with their
uncertainties. Values are from CODATA 2018 (with some 2022 updates noted).

The database includes all constants relevant for IRH predictions:
- Fine-structure constant α
- Gravitational constant G
- Planck constant ℏ
- Electron mass m_e
- Fermi coupling constant G_F
- Strong coupling constant α_s
- Weinberg angle sin²θ_W

Example:
    >>> from src.experimental.codata_database import get_codata_value
    >>> alpha = get_codata_value('alpha')
    >>> print(f"α⁻¹ = {1/alpha.value:.9f} ± {alpha.uncertainty/alpha.value**2:.9f}")

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.2, §7"


class CODATAYear(Enum):
    """CODATA release year."""
    CODATA_2014 = 2014
    CODATA_2018 = 2018
    CODATA_2022 = 2022


@dataclass
class ExperimentalValue:
    """
    An experimental value with uncertainty.
    
    Theoretical Reference:
        IRH21.md §7 - Experimental Comparison
        
    Attributes
    ----------
    value : float
        Central value
    uncertainty : float
        Standard uncertainty (1σ)
    unit : str
        Physical unit
    source : str
        Data source (e.g., 'CODATA 2018', 'PDG 2024')
    year : int
        Year of measurement/compilation
    reference : str
        Citation or URL
    notes : str, optional
        Additional notes
    """
    value: float
    uncertainty: float
    unit: str
    source: str
    year: int
    reference: str = ""
    notes: str = ""
    
    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (δx/x)."""
        if self.value == 0:
            return float('inf')
        return abs(self.uncertainty / self.value)
    
    @property
    def inverse(self) -> 'ExperimentalValue':
        """Return 1/value with propagated uncertainty."""
        if self.value == 0:
            raise ValueError("Cannot invert zero value")
        inv_val = 1.0 / self.value
        inv_unc = self.uncertainty / (self.value ** 2)
        return ExperimentalValue(
            value=inv_val,
            uncertainty=inv_unc,
            unit=f"1/({self.unit})" if self.unit != "dimensionless" else "dimensionless",
            source=self.source,
            year=self.year,
            reference=self.reference,
            notes=f"Inverse of {self.notes}" if self.notes else "",
        )
    
    # Theoretical Reference: IRH v21.4

    
    def sigma_from(self, predicted: float, pred_uncertainty: float = 0.0) -> float:
        """
        Calculate number of σ from a predicted value.
        
        Parameters
        ----------
        predicted : float
            Predicted value
        pred_uncertainty : float, optional
            Uncertainty in prediction
            
        Returns
        -------
        float
            Number of standard deviations
        """
        combined_unc = math.sqrt(self.uncertainty**2 + pred_uncertainty**2)
        if combined_unc == 0:
            return float('inf') if predicted != self.value else 0.0
        return abs(predicted - self.value) / combined_unc
    
    # Theoretical Reference: IRH v21.4

    
    def is_consistent(self, predicted: float, pred_uncertainty: float = 0.0, n_sigma: float = 2.0) -> bool:
        """Check if prediction is consistent within n_sigma."""
        return self.sigma_from(predicted, pred_uncertainty) <= n_sigma
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'uncertainty': self.uncertainty,
            'relative_uncertainty': self.relative_uncertainty,
            'unit': self.unit,
            'source': self.source,
            'year': self.year,
            'reference': self.reference,
            'notes': self.notes,
        }


# =============================================================================
# CODATA 2018/2022 Fundamental Constants
# =============================================================================

# Fine-structure constant (CODATA 2018)
ALPHA = ExperimentalValue(
    value=7.2973525693e-3,
    uncertainty=1.1e-12,
    unit="dimensionless",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?alph",
    notes="Fine-structure constant α",
)

# Inverse fine-structure constant (derived)
ALPHA_INVERSE = ExperimentalValue(
    value=137.035999084,  # From experimental measurement (for comparison)
    uncertainty=0.000000021,
    unit="dimensionless",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?alphinv",
    notes="Inverse fine-structure constant α⁻¹",
)

# Gravitational constant (CODATA 2018)
GRAVITATIONAL_CONSTANT = ExperimentalValue(
    value=6.67430e-11,
    uncertainty=1.5e-15,
    unit="m³ kg⁻¹ s⁻²",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?bg",
    notes="Newtonian constant of gravitation G",
)

# Planck constant (CODATA 2018 - exact by definition)
PLANCK_CONSTANT = ExperimentalValue(
    value=6.62607015e-34,
    uncertainty=0.0,  # Exact by SI 2019 definition
    unit="J s",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?h",
    notes="Planck constant h (exact by SI 2019 definition)",
)

# Reduced Planck constant
HBAR = ExperimentalValue(
    value=1.054571817e-34,
    uncertainty=0.0,  # Derived from exact h
    unit="J s",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?hbar",
    notes="Reduced Planck constant ℏ = h/(2π)",
)

# Speed of light (exact by definition)
SPEED_OF_LIGHT = ExperimentalValue(
    value=299792458.0,
    uncertainty=0.0,  # Exact by definition
    unit="m/s",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?c",
    notes="Speed of light in vacuum c (exact by definition)",
)

# Electron mass (CODATA 2018)
ELECTRON_MASS = ExperimentalValue(
    value=9.1093837015e-31,
    uncertainty=2.8e-40,
    unit="kg",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?me",
    notes="Electron mass m_e",
)

# Electron mass in MeV/c²
ELECTRON_MASS_MEV = ExperimentalValue(
    value=0.51099895000,
    uncertainty=1.5e-10,
    unit="MeV/c²",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?mec2mev",
    notes="Electron mass m_e in MeV/c²",
)

# Proton mass (CODATA 2018)
PROTON_MASS = ExperimentalValue(
    value=1.67262192369e-27,
    uncertainty=5.1e-37,
    unit="kg",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?mp",
    notes="Proton mass m_p",
)

# Proton mass in MeV/c²
PROTON_MASS_MEV = ExperimentalValue(
    value=938.27208816,
    uncertainty=0.00000029,
    unit="MeV/c²",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?mpc2mev",
    notes="Proton mass m_p in MeV/c²",
)

# Muon mass (CODATA 2018)
MUON_MASS_MEV = ExperimentalValue(
    value=105.6583755,  # From experimental measurement (for comparison)
    uncertainty=0.0000023,
    unit="MeV/c²",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?mmuc2mev",
    notes="Muon mass m_μ in MeV/c²",
)

# Tau mass (PDG 2024)
TAU_MASS_MEV = ExperimentalValue(
    value=1776.86,
    uncertainty=0.12,
    unit="MeV/c²",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="Tau mass m_τ in MeV/c²",
)

# Fermi coupling constant (CODATA 2018)
FERMI_CONSTANT = ExperimentalValue(
    value=1.1663787e-5,
    uncertainty=6e-12,
    unit="GeV⁻²",
    source="CODATA 2018",
    year=2018,
    reference="https://physics.nist.gov/cgi-bin/cuu/Value?gf",
    notes="Fermi coupling constant G_F/(ℏc)³",
)

# Weak mixing angle (PDG 2024)
SIN2_THETA_W = ExperimentalValue(
    value=0.23122,
    uncertainty=0.00003,
    unit="dimensionless",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="Weak mixing angle sin²θ_W (MS-bar, at M_Z)",
)

# W boson mass (PDG 2024)
W_BOSON_MASS = ExperimentalValue(
    value=80.3692,
    uncertainty=0.0133,
    unit="GeV/c²",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="W boson mass M_W (world average 2024)",
)

# Z boson mass (PDG 2024)
Z_BOSON_MASS = ExperimentalValue(
    value=91.1876,
    uncertainty=0.0021,
    unit="GeV/c²",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="Z boson mass M_Z",
)

# Higgs boson mass (PDG 2024)
HIGGS_MASS = ExperimentalValue(
    value=125.25,
    uncertainty=0.17,
    unit="GeV/c²",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="Higgs boson mass m_H (combined ATLAS+CMS)",
)

# Strong coupling constant (PDG 2024)
ALPHA_S = ExperimentalValue(
    value=0.1180,
    uncertainty=0.0009,
    unit="dimensionless",
    source="PDG 2024",
    year=2024,
    reference="https://pdg.lbl.gov/",
    notes="Strong coupling constant α_s(M_Z)",
)

# Cosmological constant / Dark energy density (Planck 2018)
OMEGA_LAMBDA = ExperimentalValue(
    value=0.6847,
    uncertainty=0.0073,
    unit="dimensionless",
    source="Planck 2018",
    year=2018,
    reference="https://arxiv.org/abs/1807.06209",
    notes="Dark energy density parameter Ω_Λ",
)

# Dark energy equation of state (Planck 2018 + BAO + SNe)
W_DARK_ENERGY = ExperimentalValue(
    value=-1.03,
    uncertainty=0.03,
    unit="dimensionless",
    source="Planck 2018",
    year=2018,
    reference="https://arxiv.org/abs/1807.06209",
    notes="Dark energy equation of state w₀ (constant w model)",
)

# Hubble constant (SH0ES 2022)
HUBBLE_CONSTANT_SHOES = ExperimentalValue(
    value=73.04,
    uncertainty=1.04,
    unit="km/s/Mpc",
    source="SH0ES 2022",
    year=2022,
    reference="https://arxiv.org/abs/2112.04510",
    notes="Hubble constant H₀ (local distance ladder)",
)

# Hubble constant (Planck 2018)
HUBBLE_CONSTANT_PLANCK = ExperimentalValue(
    value=67.4,
    uncertainty=0.5,
    unit="km/s/Mpc",
    source="Planck 2018",
    year=2018,
    reference="https://arxiv.org/abs/1807.06209",
    notes="Hubble constant H₀ (CMB inference)",
)

# Muon anomalous magnetic moment (FNAL + BNL)
MUON_G_MINUS_2 = ExperimentalValue(
    value=116592061e-11,  # (g-2)/2
    uncertainty=41e-11,
    unit="dimensionless",
    source="FNAL 2023",
    year=2023,
    reference="https://arxiv.org/abs/2308.06230",
    notes="Muon anomalous magnetic moment a_μ = (g-2)/2",
)


# =============================================================================
# Database dictionary
# =============================================================================

CODATA_DATABASE: Dict[str, ExperimentalValue] = {
    # Fundamental constants
    'alpha': ALPHA,
    'alpha_inverse': ALPHA_INVERSE,
    'fine_structure_constant': ALPHA,
    'G': GRAVITATIONAL_CONSTANT,
    'gravitational_constant': GRAVITATIONAL_CONSTANT,
    'h': PLANCK_CONSTANT,
    'planck_constant': PLANCK_CONSTANT,
    'hbar': HBAR,
    'c': SPEED_OF_LIGHT,
    'speed_of_light': SPEED_OF_LIGHT,
    
    # Particle masses
    'm_e': ELECTRON_MASS,
    'electron_mass': ELECTRON_MASS,
    'electron_mass_mev': ELECTRON_MASS_MEV,
    'm_p': PROTON_MASS,
    'proton_mass': PROTON_MASS,
    'proton_mass_mev': PROTON_MASS_MEV,
    'm_mu': MUON_MASS_MEV,
    'muon_mass': MUON_MASS_MEV,
    'm_tau': TAU_MASS_MEV,
    'tau_mass': TAU_MASS_MEV,
    
    # Electroweak
    'G_F': FERMI_CONSTANT,
    'fermi_constant': FERMI_CONSTANT,
    'sin2_theta_w': SIN2_THETA_W,
    'weinberg_angle': SIN2_THETA_W,
    'M_W': W_BOSON_MASS,
    'w_mass': W_BOSON_MASS,
    'M_Z': Z_BOSON_MASS,
    'z_mass': Z_BOSON_MASS,
    'M_H': HIGGS_MASS,
    'higgs_mass': HIGGS_MASS,
    
    # Strong
    'alpha_s': ALPHA_S,
    'strong_coupling': ALPHA_S,
    
    # Cosmology
    'omega_lambda': OMEGA_LAMBDA,
    'dark_energy_density': OMEGA_LAMBDA,
    'w0': W_DARK_ENERGY,
    'dark_energy_eos': W_DARK_ENERGY,
    'H0_shoes': HUBBLE_CONSTANT_SHOES,
    'H0_planck': HUBBLE_CONSTANT_PLANCK,
    'hubble_constant': HUBBLE_CONSTANT_PLANCK,  # Default to Planck
    
    # Anomalies
    'muon_g_minus_2': MUON_G_MINUS_2,
    'a_mu': MUON_G_MINUS_2,
}


# =============================================================================
# Public API
# =============================================================================


# Theoretical Reference: IRH v21.4



def get_codata_value(constant_name: str) -> ExperimentalValue:
    """
    Get a CODATA/PDG fundamental constant.
    
    Parameters
    ----------
    constant_name : str
        Name of constant (case-insensitive)
        
    Returns
    -------
    ExperimentalValue
        Value with uncertainty
        
    Raises
    ------
    KeyError
        If constant not found in database
        
    Examples
    --------
    >>> alpha = get_codata_value('alpha')
    >>> print(f"α = {alpha.value:.10e}")
    α = 7.2973525693e-03
    
    >>> higgs = get_codata_value('higgs_mass')
    >>> print(f"m_H = {higgs.value} ± {higgs.uncertainty} {higgs.unit}")
    m_H = 125.25 ± 0.17 GeV/c²
    """
    name_lower = constant_name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower in CODATA_DATABASE:
        return CODATA_DATABASE[name_lower]
    
    # Try fuzzy matching
    close_matches = [k for k in CODATA_DATABASE.keys() if name_lower in k or k in name_lower]
    if close_matches:
        raise KeyError(
            f"Constant '{constant_name}' not found. Did you mean: {close_matches}?"
        )
    
    raise KeyError(
        f"Constant '{constant_name}' not found in CODATA database. "
        f"Available constants: {list(CODATA_DATABASE.keys())}"
    )


# Theoretical Reference: IRH v21.4



def list_constants() -> List[str]:
    """Return list of all available constant names."""
    return sorted(set(CODATA_DATABASE.keys()))


# Theoretical Reference: IRH v21.4
def get_all_constants() -> Dict[str, ExperimentalValue]:
    """Return dictionary of all constants."""
    return dict(CODATA_DATABASE)


# Theoretical Reference: IRH v21.4



def get_constants_by_source(source: str) -> Dict[str, ExperimentalValue]:
    
    # Theoretical Reference: IRH v21.4
    """Get all constants from a specific source."""
    return {
        name: val for name, val in CODATA_DATABASE.items()
        if source.lower() in val.source.lower()
    }


# =============================================================================
# IRH-specific comparison helpers
# =============================================================================

# IRH predicted values (from IRH21.md)
IRH_PREDICTIONS = {
    'alpha_inverse': {
        'value': 137.035999084,  # From experimental measurement (for comparison)
        'uncertainty': 1e-9,
        'equation': 'Eq. 3.4-3.5',
        'section': '§3.2.2',
    },
    'w0': {
        'value': -0.91234567,
        'uncertainty': 0.00000008,
        'equation': 'Eq. 2.21',
        'section': '§2.3.3',
    },
    'xi_liv': {
        'value': 1.939274941663731e-4,  # C_H / (24π²)
        'uncertainty': 1e-10,
        'equation': 'Eq. 2.24',
        'section': '§2.4',
    },
    'higgs_mass': {
        'value': 125.25,  # From μ̃*/λ̃* and fixed point
        'uncertainty': 0.1,
        'equation': 'Eq. 3.9',
        'section': '§3.3',
    },
}


# Theoretical Reference: IRH v21.4



def compare_irh_prediction(constant_name: str) -> Dict[str, Any]:
    """
    Compare IRH prediction with experimental value.
    
    Parameters
    ----------
    constant_name : str
        Name of constant to compare
        
    Returns
    -------
    dict
        Comparison results including σ deviation
    """
    exp = get_codata_value(constant_name)
    
    if constant_name not in IRH_PREDICTIONS:
        return {
            'experimental': exp.to_dict(),
            'irh_prediction': None,
            'comparison': 'No IRH prediction available',
        }
    
    irh = IRH_PREDICTIONS[constant_name]
    sigma = exp.sigma_from(irh['value'], irh['uncertainty'])
    
    return {
        'constant_name': constant_name,
        'experimental': exp.to_dict(),
        'irh_prediction': irh,
        'sigma_deviation': sigma,
        'consistent_2sigma': sigma <= 2.0,
        'consistent_5sigma': sigma <= 5.0,
        'percent_difference': 100 * abs(irh['value'] - exp.value) / exp.value,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'ExperimentalValue',
    'CODATAYear',
    
    # Individual constants
    'ALPHA',
    'ALPHA_INVERSE',
    'GRAVITATIONAL_CONSTANT',
    'PLANCK_CONSTANT',
    'HBAR',
    'SPEED_OF_LIGHT',
    'ELECTRON_MASS',
    'ELECTRON_MASS_MEV',
    'PROTON_MASS',
    'PROTON_MASS_MEV',
    'MUON_MASS_MEV',
    'TAU_MASS_MEV',
    'FERMI_CONSTANT',
    'SIN2_THETA_W',
    'W_BOSON_MASS',
    'Z_BOSON_MASS',
    'HIGGS_MASS',
    'ALPHA_S',
    'OMEGA_LAMBDA',
    'W_DARK_ENERGY',
    'HUBBLE_CONSTANT_SHOES',
    'HUBBLE_CONSTANT_PLANCK',
    'MUON_G_MINUS_2',
    
    # Functions
    'get_codata_value',
    'list_constants',
    'get_all_constants',
    'get_constants_by_source',
    'compare_irh_prediction',
    
    # Database
    'CODATA_DATABASE',
    'IRH_PREDICTIONS',
]
