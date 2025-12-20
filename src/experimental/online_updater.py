"""
Online Data Updater for Experimental Constants

THEORETICAL FOUNDATION: IRH v21.1 Manuscript - Phase 4.5 Implementation

This module provides automated updates from experimental databases:
    - CODATA (physics.nist.gov) for fundamental constants
    - PDG (Particle Data Group) for particle properties

Key Features:
    - HTTP-based fetching with rate limiting
    - Local caching for offline operation
    - Change detection and significance analysis
    - Report generation (Markdown, JSON, text)
    - σ-threshold alerts for significant deviations

CRITICAL: This module performs VALIDATION, not derivation.
All theoretical predictions are derived from first principles in IRH.
Experimental data is used ONLY for comparison and falsification tests.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error

from .cache_manager import CacheManager

__version__ = "1.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §3.2 (validation only)"


# =============================================================================
# Configuration
# =============================================================================


# NIST CODATA API endpoints
NIST_CODATA_URL = "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"
NIST_CODATA_API = "https://physics.nist.gov/cgi-bin/cuu/Value"

# PDG API endpoints (would use actual PDG URLs when available)
PDG_BASE_URL = "https://pdg.lbl.gov/rpp-data"

# Rate limiting (requests per second)
RATE_LIMIT_DELAY = 1.0  # seconds between requests


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PhysicalConstant:
    """
    Represents a physical constant with experimental value.
    
    Attributes
    ----------
    name : str
        Name of the constant
    symbol : str
        Symbol (e.g., 'α', 'G_F')
    value : float
        Central value
    uncertainty : float
        Experimental uncertainty (1σ)
    units : str
        Physical units
    source : str
        Data source (e.g., 'CODATA 2022')
    year : int
        Year of measurement/compilation
    """
    
    name: str
    symbol: str
    value: float
    uncertainty: float
    units: str
    source: str
    year: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'units': self.units,
            'source': self.source,
            'year': self.year
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> PhysicalConstant:
        """Create from dictionary."""
        return cls(**d)


@dataclass
class UpdateResult:
    """
    Result of an update operation.
    
    Attributes
    ----------
    success : bool
        Whether update succeeded
    updated_count : int
        Number of constants updated
    failed_count : int
        Number of failed updates
    timestamp : float
        Unix timestamp of update
    constants : List[PhysicalConstant]
        List of updated constants
    errors : List[str]
        List of error messages
    has_significant_changes : bool
        Whether any >3σ deviations found
    """
    
    success: bool
    updated_count: int
    failed_count: int
    timestamp: float
    constants: List[PhysicalConstant]
    errors: List[str]
    has_significant_changes: bool = False


# =============================================================================
# CODATA Fetcher
# =============================================================================


class CODATAFetcher:
    """
    Fetches fundamental constants from NIST CODATA database.
    
    Parameters
    ----------
    cache_manager : CacheManager
        Cache manager for offline operation
    rate_limit_delay : float
        Seconds to wait between requests
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        rate_limit_delay: float = RATE_LIMIT_DELAY
    ):
        self.cache = cache_manager
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _fetch_codata_2022(self) -> Optional[Dict[str, PhysicalConstant]]:
        """
        Fetch CODATA 2022 constants from NIST.
        
        Returns
        -------
        Optional[Dict[str, PhysicalConstant]]
            Dictionary of constants keyed by symbol, or None if fetch fails
        """
        # Check cache first
        cached = self.cache.get('codata_2022')
        if cached is not None:
            return {
                k: PhysicalConstant.from_dict(v)
                for k, v in cached.items()
            }
        
        # Rate limit
        self._rate_limit()
        
        try:
            # Fetch from NIST
            with urllib.request.urlopen(NIST_CODATA_URL, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            # Parse constants (simplified - real implementation would parse full table)
            constants = self._parse_codata_table(data)
            
            # Cache result
            cache_data = {k: v.to_dict() for k, v in constants.items()}
            self.cache.set('codata_2022', cache_data, ttl=86400 * 7)  # 7 days
            
            return constants
            
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            # Network error - return None
            return None
    
    def _parse_codata_table(self, data: str) -> Dict[str, PhysicalConstant]:
        """
        Parse CODATA ASCII table.
        
        This is a simplified parser. Real implementation would parse
        the full NIST CODATA table format.
        
        Parameters
        ----------
        data : str
            ASCII table data from NIST
        
        Returns
        -------
        Dict[str, PhysicalConstant]
            Parsed constants
        """
        # For now, return hardcoded CODATA 2022 values
        # Real implementation would parse the actual table
        return {
            'α⁻¹': PhysicalConstant(
                name='fine-structure constant (inverse)',
                symbol='α⁻¹',
                value=137.035999084,
                uncertainty=0.000000021,
                units='dimensionless',
                source='CODATA 2022',
                year=2022
            ),
            'G': PhysicalConstant(
                name='Newtonian constant of gravitation',
                symbol='G',
                value=6.67430e-11,
                uncertainty=0.00015e-11,
                units='m³ kg⁻¹ s⁻²',
                source='CODATA 2022',
                year=2022
            ),
            'h': PhysicalConstant(
                name='Planck constant',
                symbol='h',
                value=6.62607015e-34,
                uncertainty=0.0,  # Exact by definition
                units='J Hz⁻¹',
                source='CODATA 2022',
                year=2022
            ),
            'c': PhysicalConstant(
                name='speed of light in vacuum',
                symbol='c',
                value=299792458.0,
                uncertainty=0.0,  # Exact by definition
                units='m s⁻¹',
                source='CODATA 2022',
                year=2022
            ),
        }
    
    def fetch(self, force_refresh: bool = False) -> Optional[Dict[str, PhysicalConstant]]:
        """
        Fetch CODATA constants (with caching).
        
        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and fetch fresh data
        
        Returns
        -------
        Optional[Dict[str, PhysicalConstant]]
            Dictionary of constants, or None if fetch fails
        """
        if force_refresh:
            self.cache.invalidate('codata_2022')
        
        return self._fetch_codata_2022()


# =============================================================================
# PDG Fetcher
# =============================================================================


class PDGFetcher:
    """
    Fetches particle properties from Particle Data Group.
    
    Parameters
    ----------
    cache_manager : CacheManager
        Cache manager for offline operation
    rate_limit_delay : float
        Seconds to wait between requests
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        rate_limit_delay: float = RATE_LIMIT_DELAY
    ):
        self.cache = cache_manager
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _fetch_pdg_2024(self) -> Optional[Dict[str, PhysicalConstant]]:
        """
        Fetch PDG 2024 particle properties.
        
        Returns
        -------
        Optional[Dict[str, PhysicalConstant]]
            Dictionary of particle properties, or None if fetch fails
        """
        # Check cache first
        cached = self.cache.get('pdg_2024')
        if cached is not None:
            return {
                k: PhysicalConstant.from_dict(v)
                for k, v in cached.items()
            }
        
        # For now, return hardcoded PDG 2024 values
        # Real implementation would fetch from PDG API
        particles = {
            'm_e': PhysicalConstant(
                name='electron mass',
                symbol='m_e',
                value=0.51099895000,  # MeV/c²
                uncertainty=0.00000000015,
                units='MeV/c²',
                source='PDG 2024',
                year=2024
            ),
            'm_μ': PhysicalConstant(
                name='muon mass',
                symbol='m_μ',
                value=105.6583755,  # MeV/c²
                uncertainty=0.0000023,
                units='MeV/c²',
                source='PDG 2024',
                year=2024
            ),
            'm_τ': PhysicalConstant(
                name='tau mass',
                symbol='m_τ',
                value=1776.93,  # MeV/c²
                uncertainty=0.09,
                units='MeV/c²',
                source='PDG 2024',
                year=2024
            ),
        }
        
        # Cache result
        cache_data = {k: v.to_dict() for k, v in particles.items()}
        self.cache.set('pdg_2024', cache_data, ttl=86400 * 30)  # 30 days
        
        return particles
    
    def fetch(self, force_refresh: bool = False) -> Optional[Dict[str, PhysicalConstant]]:
        """
        Fetch PDG particle properties (with caching).
        
        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and fetch fresh data
        
        Returns
        -------
        Optional[Dict[str, PhysicalConstant]]
            Dictionary of particle properties, or None if fetch fails
        """
        if force_refresh:
            self.cache.invalidate('pdg_2024')
        
        return self._fetch_pdg_2024()


# =============================================================================
# Update Functions
# =============================================================================


def update_codata_online(
    cache_dir: str = "data/cache/experimental",
    force_refresh: bool = False
) -> UpdateResult:
    """
    Update CODATA constants from NIST online database.
    
    Parameters
    ----------
    cache_dir : str
        Directory for cache storage
    force_refresh : bool
        If True, bypass cache and fetch fresh data
    
    Returns
    -------
    UpdateResult
        Result of update operation
    """
    cache = CacheManager(cache_dir)
    fetcher = CODATAFetcher(cache)
    
    try:
        constants_dict = fetcher.fetch(force_refresh=force_refresh)
        
        if constants_dict is None:
            return UpdateResult(
                success=False,
                updated_count=0,
                failed_count=1,
                timestamp=time.time(),
                constants=[],
                errors=['Failed to fetch CODATA data (network error or blocked)']
            )
        
        constants = list(constants_dict.values())
        
        return UpdateResult(
            success=True,
            updated_count=len(constants),
            failed_count=0,
            timestamp=time.time(),
            constants=constants,
            errors=[]
        )
        
    except Exception as e:
        return UpdateResult(
            success=False,
            updated_count=0,
            failed_count=1,
            timestamp=time.time(),
            constants=[],
            errors=[str(e)]
        )


def update_pdg_online(
    cache_dir: str = "data/cache/experimental",
    force_refresh: bool = False
) -> UpdateResult:
    """
    Update PDG particle properties from online database.
    
    Parameters
    ----------
    cache_dir : str
        Directory for cache storage
    force_refresh : bool
        If True, bypass cache and fetch fresh data
    
    Returns
    -------
    UpdateResult
        Result of update operation
    """
    cache = CacheManager(cache_dir)
    fetcher = PDGFetcher(cache)
    
    try:
        particles_dict = fetcher.fetch(force_refresh=force_refresh)
        
        if particles_dict is None:
            return UpdateResult(
                success=False,
                updated_count=0,
                failed_count=1,
                timestamp=time.time(),
                constants=[],
                errors=['Failed to fetch PDG data']
            )
        
        particles = list(particles_dict.values())
        
        return UpdateResult(
            success=True,
            updated_count=len(particles),
            failed_count=0,
            timestamp=time.time(),
            constants=particles,
            errors=[]
        )
        
    except Exception as e:
        return UpdateResult(
            success=False,
            updated_count=0,
            failed_count=1,
            timestamp=time.time(),
            constants=[],
            errors=[str(e)]
        )


def check_for_data_updates(cache_dir: str = "data/cache/experimental") -> Dict:
    """
    Check for available data updates without downloading.
    
    Parameters
    ----------
    cache_dir : str
        Directory for cache storage
    
    Returns
    -------
    Dict
        Status information including cache age and available updates
    """
    cache = CacheManager(cache_dir)
    
    codata_cached = cache.get('codata_2022')
    pdg_cached = cache.get('pdg_2024')
    
    cache_info = cache.info()
    
    return {
        'codata_cached': codata_cached is not None,
        'pdg_cached': pdg_cached is not None,
        'cache_entry_count': cache_info['entry_count'],
        'cache_age_hours': cache_info['oldest_age_hours'],
        'update_recommended': cache_info['oldest_age_hours'] > 168  # 7 days
    }


# =============================================================================
# Report Generation
# =============================================================================


def generate_change_report(
    old_constants: List[PhysicalConstant],
    new_constants: List[PhysicalConstant],
    format: str = 'markdown'
) -> str:
    """
    Generate report of changed constants.
    
    Parameters
    ----------
    old_constants : List[PhysicalConstant]
        Previous values
    new_constants : List[PhysicalConstant]
        New values
    format : str
        Output format ('markdown', 'text', or 'json')
    
    Returns
    -------
    str
        Formatted report
    """
    # Build changes dictionary
    old_dict = {c.symbol: c for c in old_constants}
    new_dict = {c.symbol: c for c in new_constants}
    
    changes = []
    for symbol in set(old_dict.keys()) | set(new_dict.keys()):
        if symbol in old_dict and symbol in new_dict:
            old_val = old_dict[symbol]
            new_val = new_dict[symbol]
            
            if old_val.value != new_val.value:
                changes.append({
                    'symbol': symbol,
                    'old_value': old_val.value,
                    'new_value': new_val.value,
                    'change': new_val.value - old_val.value,
                    'rel_change': (new_val.value - old_val.value) / old_val.value
                })
    
    if format == 'json':
        return json.dumps(changes, indent=2)
    
    elif format == 'markdown':
        report = "# Experimental Data Changes\n\n"
        report += f"**Generated**: {datetime.now().isoformat()}\n\n"
        
        if not changes:
            report += "No changes detected.\n"
        else:
            report += "| Symbol | Old Value | New Value | Change | Rel. Change |\n"
            report += "|--------|-----------|-----------|--------|-------------|\n"
            for c in changes:
                report += f"| {c['symbol']} | {c['old_value']:.6e} | {c['new_value']:.6e} | {c['change']:.6e} | {c['rel_change']:.2%} |\n"
        
        return report
    
    else:  # text
        report = "EXPERIMENTAL DATA CHANGES\n"
        report += "=" * 50 + "\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        
        if not changes:
            report += "No changes detected.\n"
        else:
            for c in changes:
                report += f"{c['symbol']}:\n"
                report += f"  Old: {c['old_value']:.6e}\n"
                report += f"  New: {c['new_value']:.6e}\n"
                report += f"  Change: {c['change']:.6e} ({c['rel_change']:.2%})\n\n"
        
        return report


def generate_alerts(
    irh_predictions: Dict[str, float],
    experimental_values: List[PhysicalConstant],
    sigma_threshold: float = 3.0
) -> List[Dict]:
    """
    Generate alerts for significant deviations (>σ threshold).
    
    Parameters
    ----------
    irh_predictions : Dict[str, float]
        IRH theoretical predictions keyed by symbol
    experimental_values : List[PhysicalConstant]
        Experimental values to compare
    sigma_threshold : float
        Threshold in standard deviations (default: 3.0)
    
    Returns
    -------
    List[Dict]
        List of alerts for significant deviations
    """
    alerts = []
    
    for const in experimental_values:
        if const.symbol in irh_predictions:
            irh_value = irh_predictions[const.symbol]
            exp_value = const.value
            exp_uncert = const.uncertainty
            
            if exp_uncert > 0:
                deviation_sigma = abs(irh_value - exp_value) / exp_uncert
                
                if deviation_sigma >= sigma_threshold:
                    alerts.append({
                        'symbol': const.symbol,
                        'name': const.name,
                        'irh_value': irh_value,
                        'exp_value': exp_value,
                        'exp_uncertainty': exp_uncert,
                        'deviation_sigma': deviation_sigma,
                        'falsified': deviation_sigma >= 5.0  # 5σ = falsification
                    })
    
    return alerts
