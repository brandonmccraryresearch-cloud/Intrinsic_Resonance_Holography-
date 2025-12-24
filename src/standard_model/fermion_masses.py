"""
Fermion Mass Implementation for IRH v21.4

THEORETICAL FOUNDATION: IRH v21.4 Manuscript Part 1 §3.2

This module derives fermion masses from topological complexity eigenvalues 𝓚_f
through the Yukawa coupling formula (Eq. 3.6), incorporating full Renormalization
Group (RG) running effects.

Key Equations:
    - Eq. 3.6 (Complete): m_f = 𝓡_Y × √2 × 𝓚_f × √λ̃* × √(μ̃*/λ̃*) × ℓ_0^(-1)
    - Appendix E.1: Precise determination of 𝓚 eigenvalues via transcendental equations

The mass hierarchy emerges from the topological complexity spectrum 𝓚_f without fine-tuning.
Hardcoded values are strictly forbidden for derivation, used only for validation.

Authors: IRH Computational Framework Team
Last Updated: December 2025 (IRH v21.4 compliance)
"""

import math
from typing import Dict, Optional, List

import numpy as np

# Import transparency engine
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.logging.transparency_engine import TransparencyEngine, FULL, DETAILED
from src.topology.complexity_operator import get_topological_complexity
from src.standard_model.yukawa_rg_running import compute_fermion_mass_with_rg, YukawaRGResult

__version__ = "21.4.0"
__theoretical_foundation__ = "IRH v21.4 Manuscript Part 1 §3.2, Eq. 3.6"


# Universal exponent (Eq. 1.16)
C_H = 0.045935703598

# Fixed-point couplings (Eq. 1.14)
LAMBDA_STAR = 48 * math.pi**2 / 9
GAMMA_STAR = 32 * math.pi**2 / 3
MU_STAR = 16 * math.pi**2

# Higgs VEV in GeV (Emergent property, but used as scale reference)
HIGGS_VEV = 246.22


# =============================================================================
# Validation Data (NOT FOR DERIVATION)
# =============================================================================
# These values are strictly for validation of computed results.
# They MUST NOT be used as inputs for any calculation.
# Source: IRH v21.4 Manuscript, Table 3.1
_VALIDATION_TOPOLOGICAL_COMPLEXITY = {
    'electron': 1.0000,
    'muon': 206.768,
    'tau': 3477.150,
}

def compute_fermion_mass(fermion: str, verbosity: int = 1) -> Dict:
    """
    Compute fermion mass from topological complexity per §3.2.

    Theoretical Reference:
        IRH v21.4 Manuscript Part 1 §3.2, Eq. 3.6
        m_f = 𝓡_Y × √2 × 𝓚_f × √λ̃* × √(μ̃*/λ̃*) × ℓ_0^(-1)

    Parameters
    ----------
    fermion : str
        Fermion name (e.g., 'electron', 'top', 'tau')
    verbosity : int
        Transparency level (0=silent, 1=minimal, 3=detailed, 4=full)

    Returns
    -------
    dict
        Dictionary containing:
        - 'mass_GeV': Computed mass in GeV
        - 'K_f': Topological complexity eigenvalue (computed)
        - 'R_Y': Yukawa renormalization factor
        - 'theoretical_reference': Citation string
    """
    # Initialize transparency engine
    engine = TransparencyEngine(verbosity=verbosity)
    engine.info(
        f"Computing mass for {fermion}",
        reference="IRH v21.4 Part 1, Eq. 3.6"
    )

    # 1. Get Topological Complexity (Computed, NOT hardcoded)
    try:
        K_f = get_topological_complexity(fermion, verbosity=verbosity)
    except Exception as e:
        engine.error(f"Failed to compute topological complexity for {fermion}: {e}")
        raise

    engine.step(f"Computed topological complexity 𝓚_f = {K_f:.6f}")

    # 2. Compute Mass with full RG Running
    # This uses the rigorous Eq. 3.6 implementation in yukawa_rg_running.py
    try:
        result = compute_fermion_mass_with_rg(
            fermion=fermion,
            K_f=K_f,
            lambda_star=LAMBDA_STAR,
            mu_star=MU_STAR,
            verbosity='detailed' if verbosity >= 3 else 'minimal'
        )
    except Exception as e:
        engine.error(f"Failed to compute mass with RG running: {e}")
        raise

    mass_gev = result['mass_GeV']
    R_Y = result['R_Y']

    engine.value("mass_GeV", mass_gev, uncertainty=1e-6)
    engine.passed("Mass computation complete")

    return {
        'mass_GeV': mass_gev,
        'K_f': K_f,
        'R_Y': R_Y,
        'theoretical_reference': 'IRH v21.4 Manuscript Part 1 §3.2, Eq. 3.6',
        'components': result.get('components', {})
    }


def yukawa_coupling(fermion: str, higgs_vev: float = HIGGS_VEV) -> Dict:
    """
    Compute Yukawa coupling for a fermion per Eq. 3.6.

    Theoretical Reference:
        IRH v21.4 Manuscript Part 1 §3.2, Eq. 3.6
        y_f = √(2) × m_f / v

    Parameters
    ----------
    fermion : str
        Fermion name (e.g., 'electron', 'top', 'tau')
    higgs_vev : float
        Higgs vacuum expectation value in GeV

    Returns
    -------
    dict
        Dictionary containing:
        - 'yukawa': Dimensionless Yukawa coupling y_f
        - 'mass_GeV': Fermion mass used
        - 'K_f': Topological complexity eigenvalue
        - 'theoretical_reference': Citation string
    """
    # We must use the rigorous mass computation first
    mass_result = compute_fermion_mass(fermion, verbosity=1)
    mass_gev = mass_result['mass_GeV']

    # Eq. 3.6: y_f = √(2) × m_f / v
    # This relationship holds at the Electroweak scale (where v is defined)
    yukawa = math.sqrt(2) * mass_gev / higgs_vev

    return {
        'yukawa': yukawa,
        'mass_GeV': mass_gev,
        'K_f': mass_result['K_f'],
        'theoretical_reference': 'IRH v21.4 Manuscript Part 1 §3.2, Eq. 3.6',
    }


def mass_hierarchy() -> Dict:
    """
    Compute the full fermion mass hierarchy from topological complexity.

    Theoretical Reference:
        IRH v21.4 Manuscript Part 1 §3.2, Table 3.1

    Returns
    -------
    dict
        Complete mass spectrum with theoretical uncertainties
    """
    hierarchy = {}

    # List of known fermions to compute
    # Note: Currently complexity_operator supports generations 1, 2, 3
    # which map to electron, muon, tau (and quarks)
    fermions = ['electron', 'muon', 'tau']

    for fermion in fermions:
        try:
            result = compute_fermion_mass(fermion, verbosity=0)
            yukawa_result = yukawa_coupling(fermion)
            hierarchy[fermion] = {
                'mass_GeV': result['mass_GeV'],
                'K_f': result['K_f'],
                'R_Y': result['R_Y'],
                'yukawa': yukawa_result['yukawa'],
            }
        except Exception as e:
            hierarchy[fermion] = {'error': str(e)}

    return {
        'masses': hierarchy,
        'theoretical_reference': 'IRH v21.4 Manuscript Part 1 §3.2, Table 3.1',
    }


def verify_mass_ratios() -> Dict:
    """
    Verify predicted mass ratios against experimental values.

    Theoretical Reference:
        IRH v21.4 Manuscript Part 1 §3.2.2

    Returns
    -------
    dict
        Comparison of predicted vs experimental mass ratios
    """
    # Experimental mass ratios (Validation targets)
    experimental = {
        'm_mu / m_e': 206.7682830,
        'm_tau / m_mu': 16.8170,
        'm_tau / m_e': 3477.15,
    }

    # Compute from first principles
    try:
        mass_e = compute_fermion_mass('electron', verbosity=0)['mass_GeV']
        mass_mu = compute_fermion_mass('muon', verbosity=0)['mass_GeV']
        mass_tau = compute_fermion_mass('tau', verbosity=0)['mass_GeV']

        predicted = {
            'm_mu / m_e': mass_mu / mass_e,
            'm_tau / m_mu': mass_tau / mass_mu,
            'm_tau / m_e': mass_tau / mass_e,
        }

        comparisons = {}
        for ratio_name in experimental:
            exp_val = experimental[ratio_name]
            pred_val = predicted[ratio_name]
            relative_error = abs(pred_val - exp_val) / exp_val

            comparisons[ratio_name] = {
                'experimental': exp_val,
                'predicted': pred_val,
                'relative_error': relative_error,
                'agreement': relative_error < 0.05,  # 5% tolerance for provisional model
            }

        return {
            'comparisons': comparisons,
            'theoretical_reference': 'IRH v21.4 Manuscript Part 1 §3.2.2',
        }

    except Exception as e:
        return {'error': f"Verification failed: {str(e)}"}


__all__ = [
    'compute_fermion_mass',
    'yukawa_coupling',
    'mass_hierarchy',
    'verify_mass_ratios',
    'HIGGS_VEV',
    'C_H',
    'LAMBDA_STAR',
]
