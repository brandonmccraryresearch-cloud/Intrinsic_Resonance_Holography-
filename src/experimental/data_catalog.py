"""
Version-Controlled Experimental Data Catalog

THEORETICAL FOUNDATION: IRH21.md §7 - Falsifiable Predictions

This module provides a unified, version-controlled interface for managing
experimental data. It supports:

- Multiple data sources (CODATA, PDG, experimental collaborations)
- Version tracking with timestamps
- Data provenance and citations
- Automatic updates from online sources
- Local caching for offline access

Example:
    >>> from src.experimental.data_catalog import DataCatalog
    >>> catalog = DataCatalog()
    >>> alpha = catalog.get('alpha_inverse')
    >>> print(f"α⁻¹ = {alpha.value} ({alpha.source})")

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .codata_database import ExperimentalValue, get_codata_value, CODATA_DATABASE
from .pdg_parser import get_pdg_value, get_particle, PDG_DATABASE

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §7"


@dataclass
class DataEntry:
    """
    A catalog entry with metadata.
    
    Attributes
    ----------
    key : str
        Unique identifier
    value : ExperimentalValue
        The experimental value
    category : str
        Category (e.g., 'fundamental_constant', 'particle_mass')
    tags : list[str]
        Searchable tags
    added : datetime
        When entry was added to catalog
    updated : datetime
        When entry was last updated
    checksum : str
        Hash for version control
    """
    key: str
    value: ExperimentalValue
    category: str
    tags: List[str] = field(default_factory=list)
    added: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute hash of value for versioning."""
        data = f"{self.value.value}:{self.value.uncertainty}:{self.value.source}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value.to_dict(),
            'category': self.category,
            'tags': self.tags,
            'added': self.added.isoformat(),
            'updated': self.updated.isoformat(),
            'checksum': self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataEntry':
        """Create from dictionary."""
        return cls(
            key=data['key'],
            value=ExperimentalValue(
                value=data['value']['value'],
                uncertainty=data['value']['uncertainty'],
                unit=data['value']['unit'],
                source=data['value']['source'],
                year=data['value']['year'],
                reference=data['value'].get('reference', ''),
                notes=data['value'].get('notes', ''),
            ),
            category=data['category'],
            tags=data.get('tags', []),
            added=datetime.fromisoformat(data['added']),
            updated=datetime.fromisoformat(data['updated']),
            checksum=data.get('checksum', ''),
        )


class DataCatalog:
    """
    Unified experimental data catalog.
    
    Theoretical Reference:
        IRH21.md §7 - Experimental Comparison
        
    This catalog provides a single interface for accessing experimental
    data from multiple sources (CODATA, PDG, etc.) with version control
    and caching capabilities.
    
    Attributes
    ----------
    entries : dict[str, DataEntry]
        All catalog entries
    sources : list[str]
        Active data sources
    version : str
        Catalog version
    last_updated : datetime
        Last update timestamp
    
    Examples
    --------
    >>> catalog = DataCatalog()
    >>> catalog.list_categories()
    ['fundamental_constant', 'particle_mass', 'cosmology', ...]
    
    >>> alpha = catalog.get('alpha_inverse')
    >>> print(f"α⁻¹ = {alpha.value:.9f}")
    α⁻¹ = 137.035999084
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data catalog.
        
        Parameters
        ----------
        cache_dir : Path, optional
            Directory for caching. Default: ~/.irh/data_cache
        """
        self.entries: Dict[str, DataEntry] = {}
        self.sources = ['CODATA 2018', 'PDG 2024', 'Planck 2018', 'FNAL 2023']
        self.version = __version__
        self.last_updated = datetime.now()
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.irh' / 'data_cache'
        self.cache_dir = Path(cache_dir)
        
        # Load built-in data
        self._load_builtin_data()
    
    def _load_builtin_data(self):
        """Load data from built-in databases."""
        # Load CODATA constants
        for key, value in CODATA_DATABASE.items():
            self.entries[key] = DataEntry(
                key=key,
                value=value,
                category='fundamental_constant',
                tags=['codata', value.source.lower().replace(' ', '_')],
            )
        
        # Load PDG particle masses
        seen_particles = set()
        for key, particle in PDG_DATABASE.items():
            if particle.name not in seen_particles:
                mass_key = f"{particle.name}_mass"
                self.entries[mass_key] = DataEntry(
                    key=mass_key,
                    value=particle.mass,
                    category='particle_mass',
                    tags=['pdg', particle.particle_type.value, particle.name],
                )
                seen_particles.add(particle.name)
    
    def get(self, key: str) -> ExperimentalValue:
        """
        Get experimental value by key.
        
        Parameters
        ----------
        key : str
            Entry key (case-insensitive)
            
        Returns
        -------
        ExperimentalValue
            The experimental value
            
        Raises
        ------
        KeyError
            If key not found
        """
        key_lower = key.lower()
        
        if key_lower in self.entries:
            return self.entries[key_lower].value
        
        # Try CODATA
        try:
            return get_codata_value(key)
        except KeyError:
            pass
        
        # Try PDG
        try:
            # Check if it's a particle mass
            if '_mass' in key_lower:
                particle = key_lower.replace('_mass', '')
                return get_pdg_value(particle, 'mass')
            return get_pdg_value(key, 'mass')
        except KeyError:
            pass
        
        raise KeyError(f"Key '{key}' not found in catalog")
    
    def get_entry(self, key: str) -> DataEntry:
        """Get full catalog entry (with metadata)."""
        key_lower = key.lower()
        if key_lower in self.entries:
            return self.entries[key_lower]
        raise KeyError(f"Key '{key}' not found in catalog")
    
    def search(self, query: str = "", category: str = None, tags: List[str] = None) -> List[DataEntry]:
        """
        Search catalog entries.
        
        Parameters
        ----------
        query : str, optional
            Text to search in keys and tags
        category : str, optional
            Filter by category
        tags : list[str], optional
            Filter by tags (all must match)
            
        Returns
        -------
        list[DataEntry]
            Matching entries
        """
        results = []
        query_lower = query.lower()
        
        for entry in self.entries.values():
            # Text search
            if query and query_lower not in entry.key.lower():
                if not any(query_lower in tag.lower() for tag in entry.tags):
                    continue
            
            # Category filter
            if category and entry.category != category:
                continue
            
            # Tags filter
            if tags and not all(t.lower() in [x.lower() for x in entry.tags] for t in tags):
                continue
            
            results.append(entry)
        
        return results
    
    def list_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(set(e.category for e in self.entries.values()))
    
    def list_keys(self, category: str = None) -> List[str]:
        """Get list of all keys, optionally filtered by category."""
        if category:
            return sorted(e.key for e in self.entries.values() if e.category == category)
        return sorted(self.entries.keys())
    
    def add(self, key: str, value: ExperimentalValue, category: str, tags: List[str] = None):
        """
        Add a new entry to the catalog.
        
        Parameters
        ----------
        key : str
            Unique identifier
        value : ExperimentalValue
            The value to add
        category : str
            Category name
        tags : list[str], optional
            Searchable tags
        """
        entry = DataEntry(
            key=key.lower(),
            value=value,
            category=category,
            tags=tags or [],
        )
        self.entries[key.lower()] = entry
        self.last_updated = datetime.now()
    
    def update(self, key: str, value: ExperimentalValue):
        """
        Update an existing entry.
        
        Parameters
        ----------
        key : str
            Entry key
        value : ExperimentalValue
            New value
        """
        key_lower = key.lower()
        if key_lower not in self.entries:
            raise KeyError(f"Key '{key}' not found")
        
        entry = self.entries[key_lower]
        entry.value = value
        entry.updated = datetime.now()
        entry.checksum = entry._compute_checksum()
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export catalog to dictionary."""
        return {
            'version': self.version,
            'sources': self.sources,
            'last_updated': self.last_updated.isoformat(),
            'entries': {k: v.to_dict() for k, v in self.entries.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataCatalog':
        """Create catalog from dictionary."""
        catalog = cls.__new__(cls)
        catalog.version = data.get('version', __version__)
        catalog.sources = data.get('sources', [])
        catalog.last_updated = datetime.fromisoformat(data['last_updated'])
        catalog.entries = {
            k: DataEntry.from_dict(v) 
            for k, v in data.get('entries', {}).items()
        }
        catalog.cache_dir = Path.home() / '.irh' / 'data_cache'
        return catalog
    
    def save(self, path: Optional[Path] = None):
        """
        Save catalog to JSON file.
        
        Parameters
        ----------
        path : Path, optional
            Output path. Default: cache_dir/catalog.json
        """
        if path is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / 'catalog.json'
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DataCatalog':
        """Load catalog from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """Generate catalog summary."""
        categories = self.list_categories()
        counts = {cat: len(self.search(category=cat)) for cat in categories}
        
        lines = [
            f"IRH Experimental Data Catalog v{self.version}",
            f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}",
            f"Sources: {', '.join(self.sources)}",
            f"Total entries: {len(self.entries)}",
            "",
            "Entries by category:",
        ]
        for cat, count in sorted(counts.items()):
            lines.append(f"  {cat}: {count}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __contains__(self, key: str) -> bool:
        return key.lower() in self.entries
    
    def __getitem__(self, key: str) -> ExperimentalValue:
        return self.get(key)


# =============================================================================
# Convenience functions
# =============================================================================


_DEFAULT_CATALOG: Optional[DataCatalog] = None


def get_catalog() -> DataCatalog:
    """Get the default data catalog (singleton)."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = DataCatalog()
    return _DEFAULT_CATALOG


def get_experimental_value(key: str) -> ExperimentalValue:
    """
    Get experimental value from default catalog.
    
    This is a convenience function equivalent to:
        get_catalog().get(key)
    """
    return get_catalog().get(key)


def search_catalog(query: str = "", **kwargs) -> List[DataEntry]:
    """Search the default catalog."""
    return get_catalog().search(query, **kwargs)


def compare_with_irh(
    irh_predictions: Dict[str, float],
    uncertainties: Dict[str, float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare IRH predictions with catalog values.
    
    Parameters
    ----------
    irh_predictions : dict
        Dictionary of {key: predicted_value}
    uncertainties : dict, optional
        Dictionary of {key: uncertainty}
        
    Returns
    -------
    dict
        Comparison results for each key
    """
    from .comparison import compare_single
    
    catalog = get_catalog()
    uncertainties = uncertainties or {}
    results = {}
    
    for key, pred in irh_predictions.items():
        try:
            exp = catalog.get(key)
            result = compare_single(
                pred, key, 
                uncertainties.get(key, 0.0)
            )
            results[key] = result.to_dict()
        except KeyError:
            results[key] = {'error': f"Key '{key}' not found in catalog"}
    
    return results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'DataEntry',
    'DataCatalog',
    
    # Functions
    'get_catalog',
    'get_experimental_value',
    'search_catalog',
    'compare_with_irh',
]
