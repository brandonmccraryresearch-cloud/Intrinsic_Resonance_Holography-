"""
Experimental Data Module

THEORETICAL FOUNDATION: IRH v21.1 Manuscript - Phase 4.5

This module provides automated experimental data updates and comparison:
    - CODATA fundamental constants
    - PDG particle properties
    - Statistical comparison (σ-analysis)
    - Report generation
    - Cache management for offline operation

CRITICAL DESIGN PRINCIPLE:
    This module performs VALIDATION ONLY, not derivation.
    All IRH predictions are derived from first principles.
    Experimental data is used solely for falsification tests.

Usage Example:
    >>> from src.experimental import update_codata_online, check_for_data_updates
    >>> 
    >>> # Check for updates
    >>> status = check_for_data_updates()
    >>> print(f"Update recommended: {status['update_recommended']}")
    >>> 
    >>> # Fetch latest CODATA
    >>> result = update_codata_online(force_refresh=True)
    >>> print(f"Updated {result.updated_count} constants")
    >>> 
    >>> # Generate alerts for significant deviations
    >>> from src.experimental import generate_alerts
    >>> irh_predictions = {'α⁻¹': 137.035999084}
    >>> alerts = generate_alerts(irh_predictions, result.constants)
    >>> for alert in alerts:
    >>>     print(f"{alert['symbol']}: {alert['deviation_sigma']:.2f}σ deviation")

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §7"

# Lazy imports for optional dependencies
_PDG_DATA = None
_CODATA_VALUES = None


def get_pdg_value(particle: str, property_name: str):
    """
    Get a PDG value for a particle property.
    
    Parameters
    ----------
    particle : str
        Particle name (e.g., 'electron', 'muon', 'W')
    property_name : str
        Property name (e.g., 'mass', 'lifetime', 'width')
        
    Returns
    -------
    ExperimentalValue
        Value with uncertainty
    """
    from .pdg_parser import get_pdg_value as _get
    return _get(particle, property_name)


def get_codata_value(constant_name: str):
    """
    Get a CODATA fundamental constant.
    
    Parameters
    ----------
    constant_name : str
        Constant name (e.g., 'alpha', 'G_F', 'hbar')
        
    Returns
    -------
    ExperimentalValue
        Value with uncertainty
    """
    from .codata_database import get_codata_value as _get
    return _get(constant_name)


def compare_with_experiment(irh_value: float, experimental_name: str, uncertainty: float = None):
    """
    Compare an IRH prediction with experimental value.
    
    Parameters
    ----------
    irh_value : float
        IRH predicted value
    experimental_name : str
        Name of experimental value to compare against
    uncertainty : float, optional
        IRH prediction uncertainty
        
    Returns
    -------
    ComparisonResult
        Statistical comparison result
    """
    from .comparison import compare_single as _compare
    return _compare(irh_value, experimental_name, uncertainty)


def update_codata_online(use_cache: bool = True, force_refresh: bool = False):
    """
    Update CODATA constants from online source.
    
    Parameters
    ----------
    use_cache : bool
        Whether to use/update cache
    force_refresh : bool
        Force refresh even if cache is fresh
        
    Returns
    -------
    UpdateResult
        Update results with changes
    """
    from .online_updater import update_codata as _update
    return _update(use_cache=use_cache, force_refresh=force_refresh)


def update_pdg_online(use_cache: bool = True, force_refresh: bool = False):
    """
    Update PDG particle data from online source.
    
    Parameters
    ----------
    use_cache : bool
        Whether to use/update cache
    force_refresh : bool
        Force refresh even if cache is fresh
        
    Returns
    -------
    UpdateResult
        Update results with changes
    """
    from .online_updater import update_pdg as _update
    return _update(use_cache=use_cache, force_refresh=force_refresh)


def check_for_data_updates():
    """
    Check for available updates without downloading full data.
    
    Returns
    -------
    dict
        Status of each data source
    """
    from .online_updater import check_for_updates as _check
    return _check()


__all__ = [
    'get_pdg_value',
    'get_codata_value', 
    'compare_with_experiment',
    # Phase 4.5: Online updates
    'update_codata_online',
    'update_pdg_online',
    'check_for_data_updates',
]
