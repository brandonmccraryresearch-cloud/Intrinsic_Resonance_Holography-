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

from .cache_manager import (
    CacheManager,
    CacheEntry,
    create_cache_manager
)

from .online_updater import (
    PhysicalConstant,
    UpdateResult,
    CODATAFetcher,
    PDGFetcher,
    update_codata_online,
    update_pdg_online,
    check_for_data_updates,
    generate_change_report,
    generate_alerts
)

__version__ = "1.0.0"
__all__ = [
    # Cache management
    'CacheManager',
    'CacheEntry',
    'create_cache_manager',
    
    # Data classes
    'PhysicalConstant',
    'UpdateResult',
    
    # Fetchers
    'CODATAFetcher',
    'PDGFetcher',
    
    # Update functions
    'update_codata_online',
    'update_pdg_online',
    'check_for_data_updates',
    
    # Reporting
    'generate_change_report',
    'generate_alerts',
]
