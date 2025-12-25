"""
Particle Data Group (PDG) Data Parser

THEORETICAL FOUNDATION: IRH21.md §3.2, §7

This module provides access to PDG particle properties. It includes
a comprehensive database of particle masses, widths, and other
properties relevant for IRH predictions.

Currently supports:
- Leptons (e, μ, τ and neutrinos)
- Quarks (u, d, c, s, t, b)
- Gauge bosons (γ, W, Z, g)
- Higgs boson

Example:
    >>> from src.experimental.pdg_parser import get_pdg_value
    >>> top = get_pdg_value('top', 'mass')
    >>> print(f"m_t = {top.value} ± {top.uncertainty} {top.unit}")

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .codata_database import ExperimentalValue

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §3.2, §7"


class ParticleType(Enum):
    """Classification of particle types."""
    LEPTON = "lepton"
    QUARK = "quark"
    GAUGE_BOSON = "gauge_boson"
    SCALAR = "scalar"
    NEUTRINO = "neutrino"


@dataclass
class Particle:
    """
    Particle data container.
    
    Attributes
    ----------
    name : str
        Particle name
    symbol : str
        LaTeX symbol
    particle_type : ParticleType
        Classification
    mass : ExperimentalValue
        Mass (with uncertainty)
    charge : float
        Electric charge in units of e
    spin : float
        Spin quantum number
    properties : dict
        Additional properties (width, lifetime, etc.)
    """
    name: str
    symbol: str
    particle_type: ParticleType
    mass: ExperimentalValue
    charge: float
    spin: float
    properties: Dict[str, ExperimentalValue] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    # Theoretical Reference: IRH v21.4

    
    def get_property(self, prop_name: str) -> Optional[ExperimentalValue]:
        """Get a particle property."""
        if prop_name == 'mass':
            return self.mass
        return self.properties.get(prop_name)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'symbol': self.symbol,
            'type': self.particle_type.value,
            'mass': self.mass.to_dict(),
            'charge': self.charge,
            'spin': self.spin,
            'properties': {k: v.to_dict() for k, v in self.properties.items()},
        }


# =============================================================================
# PDG 2024 Particle Database
# =============================================================================

# Leptons
ELECTRON = Particle(
    name='electron',
    symbol=r'e^-',
    particle_type=ParticleType.LEPTON,
    mass=ExperimentalValue(
        value=0.51099895000,
        uncertainty=1.5e-10,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
    ),
    charge=-1.0,
    spin=0.5,
)

MUON = Particle(
    name='muon',
    symbol=r'\mu^-',
    particle_type=ParticleType.LEPTON,
    mass=ExperimentalValue(
        value=105.6583755,  # From experimental measurement (for comparison)
        uncertainty=0.0000023,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
    ),
    charge=-1.0,
    spin=0.5,
    properties={
        'lifetime': ExperimentalValue(
            value=2.1969811e-6,
            uncertainty=2.2e-12,
            unit='s',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
        ),
        'magnetic_moment_anomaly': ExperimentalValue(
            value=1.16592061e-3,  # a_μ = (g-2)/2
            uncertainty=4.1e-10,
            unit='dimensionless',
            source='FNAL 2023',
            year=2023,
            reference='https://arxiv.org/abs/2308.06230',
        ),
    },
)

TAU = Particle(
    name='tau',
    symbol=r'\tau^-',
    particle_type=ParticleType.LEPTON,
    mass=ExperimentalValue(
        value=1776.86,
        uncertainty=0.12,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
    ),
    charge=-1.0,
    spin=0.5,
    properties={
        'lifetime': ExperimentalValue(
            value=2.903e-13,
            uncertainty=5e-16,
            unit='s',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
        ),
    },
)

# Neutrinos (mass differences, not absolute masses)
NEUTRINO_E = Particle(
    name='electron_neutrino',
    symbol=r'\nu_e',
    particle_type=ParticleType.NEUTRINO,
    mass=ExperimentalValue(
        value=0.0,  # Upper bound < 1.1 eV (direct)
        uncertainty=1.1,
        unit='eV/c²',
        source='KATRIN 2022',
        year=2022,
        reference='https://arxiv.org/abs/2105.08533',
        notes='Upper limit, not central value',
    ),
    charge=0.0,
    spin=0.5,
)

# Mass squared differences (neutrino oscillations)
DELTA_M21_SQUARED = ExperimentalValue(
    value=7.53e-5,
    uncertainty=0.18e-5,
    unit='eV²',
    source='PDG 2024',
    year=2024,
    reference='https://pdg.lbl.gov/',
    notes='Solar mass squared difference Δm²₂₁',
)

DELTA_M32_SQUARED = ExperimentalValue(
    value=2.453e-3,  # Normal ordering
    uncertainty=0.033e-3,
    unit='eV²',
    source='PDG 2024',
    year=2024,
    reference='https://pdg.lbl.gov/',
    notes='Atmospheric mass squared difference |Δm²₃₂| (normal ordering)',
)

# Quarks
UP_QUARK = Particle(
    name='up',
    symbol='u',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=2.16,
        uncertainty=0.07,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='MS-bar mass at 2 GeV',
    ),
    charge=2/3,
    spin=0.5,
)

DOWN_QUARK = Particle(
    name='down',
    symbol='d',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=4.67,
        uncertainty=0.09,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='MS-bar mass at 2 GeV',
    ),
    charge=-1/3,
    spin=0.5,
)

STRANGE_QUARK = Particle(
    name='strange',
    symbol='s',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=93.4,
        uncertainty=0.8,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='MS-bar mass at 2 GeV',
    ),
    charge=-1/3,
    spin=0.5,
)

CHARM_QUARK = Particle(
    name='charm',
    symbol='c',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=1270,
        uncertainty=20,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='MS-bar mass m_c(m_c)',
    ),
    charge=2/3,
    spin=0.5,
)

BOTTOM_QUARK = Particle(
    name='bottom',
    symbol='b',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=4180,
        uncertainty=30,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='MS-bar mass m_b(m_b)',
    ),
    charge=-1/3,
    spin=0.5,
)

TOP_QUARK = Particle(
    name='top',
    symbol='t',
    particle_type=ParticleType.QUARK,
    mass=ExperimentalValue(
        value=172760,
        uncertainty=300,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='Direct measurement (pole mass)',
    ),
    charge=2/3,
    spin=0.5,
    properties={
        'width': ExperimentalValue(
            value=1420,
            uncertainty=190,
            unit='MeV',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
        ),
    },
)

# Gauge bosons
PHOTON = Particle(
    name='photon',
    symbol=r'\gamma',
    particle_type=ParticleType.GAUGE_BOSON,
    mass=ExperimentalValue(
        value=0.0,
        uncertainty=1e-18,  # Upper limit
        unit='eV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='Upper limit < 10⁻¹⁸ eV',
    ),
    charge=0.0,
    spin=1.0,
)

W_BOSON = Particle(
    name='W_boson',
    symbol='W^\\pm',
    particle_type=ParticleType.GAUGE_BOSON,
    mass=ExperimentalValue(
        value=80369.2,
        uncertainty=13.3,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='World average 2024',
    ),
    charge=1.0,
    spin=1.0,
    properties={
        'width': ExperimentalValue(
            value=2085,
            uncertainty=42,
            unit='MeV',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
        ),
    },
)

Z_BOSON = Particle(
    name='Z_boson',
    symbol='Z^0',
    particle_type=ParticleType.GAUGE_BOSON,
    mass=ExperimentalValue(
        value=91187.6,
        uncertainty=2.1,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
    ),
    charge=0.0,
    spin=1.0,
    properties={
        'width': ExperimentalValue(
            value=2495.2,
            uncertainty=2.3,
            unit='MeV',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
        ),
    },
)

GLUON = Particle(
    name='gluon',
    symbol='g',
    particle_type=ParticleType.GAUGE_BOSON,
    mass=ExperimentalValue(
        value=0.0,
        uncertainty=0.0,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='Massless by gauge invariance',
    ),
    charge=0.0,
    spin=1.0,
)

# Higgs
HIGGS = Particle(
    name='Higgs',
    symbol='H^0',
    particle_type=ParticleType.SCALAR,
    mass=ExperimentalValue(
        value=125250,
        uncertainty=170,
        unit='MeV/c²',
        source='PDG 2024',
        year=2024,
        reference='https://pdg.lbl.gov/',
        notes='Combined ATLAS+CMS',
    ),
    charge=0.0,
    spin=0.0,
    properties={
        'width': ExperimentalValue(
            value=3.2,
            uncertainty=0.8,
            unit='MeV',
            source='PDG 2024',
            year=2024,
            reference='https://pdg.lbl.gov/',
            notes='Experimental upper limit',
        ),
    },
)


# =============================================================================
# PDG Database
# =============================================================================

PDG_DATABASE: Dict[str, Particle] = {
    # Leptons
    'electron': ELECTRON,
    'e': ELECTRON,
    'muon': MUON,
    'mu': MUON,
    'μ': MUON,
    'tau': TAU,
    'τ': TAU,
    'neutrino_e': NEUTRINO_E,
    'nu_e': NEUTRINO_E,
    'νe': NEUTRINO_E,
    
    # Quarks
    'up': UP_QUARK,
    'u': UP_QUARK,
    'down': DOWN_QUARK,
    'd': DOWN_QUARK,
    'strange': STRANGE_QUARK,
    's': STRANGE_QUARK,
    'charm': CHARM_QUARK,
    'c': CHARM_QUARK,
    'bottom': BOTTOM_QUARK,
    'b': BOTTOM_QUARK,
    'top': TOP_QUARK,
    't': TOP_QUARK,
    
    # Gauge bosons
    'photon': PHOTON,
    'gamma': PHOTON,
    'γ': PHOTON,
    'W': W_BOSON,
    'W_boson': W_BOSON,
    'Z': Z_BOSON,
    'Z_boson': Z_BOSON,
    'gluon': GLUON,
    'g': GLUON,
    
    # Higgs
    'Higgs': HIGGS,
    'H': HIGGS,
    'higgs': HIGGS,
}

# Additional properties database
PDG_PROPERTIES: Dict[str, ExperimentalValue] = {
    'delta_m21_squared': DELTA_M21_SQUARED,
    'delta_m32_squared': DELTA_M32_SQUARED,
    'dm21': DELTA_M21_SQUARED,
    'dm32': DELTA_M32_SQUARED,
}


# =============================================================================
# Public API
# =============================================================================


# Theoretical Reference: IRH v21.4



def get_particle(name: str) -> Particle:
    """
    Get particle data from PDG database.
    
    Parameters
    ----------
    name : str
        Particle name (case-insensitive)
        
    Returns
    -------
    Particle
        Particle data
        
    Raises
    ------
    KeyError
        If particle not found
    """
    name_lower = name.lower()
    
    if name_lower in PDG_DATABASE:
        return PDG_DATABASE[name_lower]
    
    # Check for case variations
    for key, particle in PDG_DATABASE.items():
        if key.lower() == name_lower or particle.name.lower() == name_lower:
            return particle
    
    raise KeyError(
        f"Particle '{name}' not found in PDG database. "
        f"Available particles: {list(set(p.name for p in PDG_DATABASE.values()))}"
    )


# Theoretical Reference: IRH v21.4



def get_pdg_value(particle_name: str, property_name: str) -> ExperimentalValue:
    """
    Get a specific property for a particle.
    
    Parameters
    ----------
    particle_name : str
        Particle name
    property_name : str
        Property name ('mass', 'width', 'lifetime', etc.)
        
    Returns
    -------
    ExperimentalValue
        Property value with uncertainty
        
    Examples
    --------
    >>> mass = get_pdg_value('electron', 'mass')
    >>> print(f"m_e = {mass.value} {mass.unit}")
    m_e = 0.51099895 MeV/c²
    """
    # Check for standalone properties first
    prop_key = f"{particle_name}_{property_name}".lower().replace(' ', '_')
    if prop_key in PDG_PROPERTIES:
        return PDG_PROPERTIES[prop_key]
    
    # Otherwise get from particle
    particle = get_particle(particle_name)
    value = particle.get_property(property_name)
    
    if value is None:
        available = ['mass'] + list(particle.properties.keys())
        raise KeyError(
            f"Property '{property_name}' not found for {particle_name}. "
            f"Available: {available}"
        )
    
    return value


# Theoretical Reference: IRH v21.4



def list_particles() -> List[str]:
    """Return list of available particle names."""
    return sorted(set(p.name for p in PDG_DATABASE.values()))


# Theoretical Reference: IRH v21.4
def get_particles_by_type(particle_type: ParticleType) -> List[Particle]:
    """Get all particles of a given type."""
    seen = set()
    result = []
    for particle in PDG_DATABASE.values():
        if particle.particle_type == particle_type and particle.name not in seen:
            result.append(particle)
            seen.add(particle.name)
    return result


# Theoretical Reference: IRH v21.4



def get_lepton_masses() -> Dict[str, ExperimentalValue]:
    
    # Theoretical Reference: IRH v21.4
    """Get all lepton masses."""
    return {p.name: p.mass for p in get_particles_by_type(ParticleType.LEPTON)}


def get_quark_masses() -> Dict[str, ExperimentalValue]:
    """Get all quark masses."""
    return {p.name: p.mass for p in get_particles_by_type(ParticleType.QUARK)}


# Theoretical Reference: IRH v21.4



def mass_ratio(particle1: str, particle2: str) -> ExperimentalValue:
    
    # Theoretical Reference: IRH v21.4
    """
    Compute mass ratio m₁/m₂ with uncertainty propagation.
    
    Parameters
    ----------
    particle1, particle2 : str
        Particle names
        
    Returns
    -------
    ExperimentalValue
        Mass ratio with propagated uncertainty
        
    Raises
    ------
    ValueError
        If either particle has zero mass
    """
    m1 = get_pdg_value(particle1, 'mass')
    m2 = get_pdg_value(particle2, 'mass')
    
    if m2.value == 0:
        raise ValueError(f"Cannot compute mass ratio: {particle2} has zero mass")
    if m1.value == 0:
        raise ValueError(f"Cannot compute mass ratio: {particle1} has zero mass")
    
    ratio = m1.value / m2.value
    # Relative uncertainties add in quadrature for division
    rel_unc = (
        (m1.uncertainty / m1.value)**2 + 
        (m2.uncertainty / m2.value)**2
    )**0.5
    
    return ExperimentalValue(
        value=ratio,
        uncertainty=ratio * rel_unc,
        unit='dimensionless',
        source=f'{m1.source} / {m2.source}',
        year=max(m1.year, m2.year),
        reference='Derived from PDG values',
        notes=f'm({particle1})/m({particle2})',
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'Particle',
    'ParticleType',
    
    # Individual particles
    'ELECTRON',
    'MUON',
    'TAU',
    'NEUTRINO_E',
    'UP_QUARK',
    'DOWN_QUARK',
    'STRANGE_QUARK',
    'CHARM_QUARK',
    'BOTTOM_QUARK',
    'TOP_QUARK',
    'PHOTON',
    'W_BOSON',
    'Z_BOSON',
    'GLUON',
    'HIGGS',
    
    # Neutrino properties
    'DELTA_M21_SQUARED',
    'DELTA_M32_SQUARED',
    
    # Functions
    'get_particle',
    'get_pdg_value',
    'list_particles',
    'get_particles_by_type',
    'get_lepton_masses',
    'get_quark_masses',
    'mass_ratio',
    
    # Database
    'PDG_DATABASE',
    'PDG_PROPERTIES',
]
