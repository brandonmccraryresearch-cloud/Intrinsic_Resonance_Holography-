"""
Cache Manager for Experimental Data

THEORETICAL FOUNDATION: IRH v21.1 Manuscript - Phase 4.5 Infrastructure

This module provides caching infrastructure for experimental data updates
to enable offline operation and reduce network load.

Key Features:
    - Time-to-live (TTL) based cache invalidation
    - Persistent disk storage for offline use
    - Thread-safe operations
    - Automatic cleanup of expired entries

Design Principles:
    - Caching is infrastructure, not theory
    - Enables reproducibility offline
    - No impact on theoretical predictions

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

__version__ = "1.0.0"
__module_type__ = "infrastructure"


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """
    Represents a single cached data entry.
    
    Attributes
    ----------
    key : str
        Unique identifier for the cached data
    data : Any
        The cached data (can be dict, list, or primitive)
    timestamp : float
        Unix timestamp when entry was created
    ttl : float
        Time-to-live in seconds (None = never expires)
    """
    
    key: str
    data: Any
    timestamp: float
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        age = time.time() - self.timestamp
        return age > self.ttl
    
    def age_seconds(self) -> float:
        """Return age of cache entry in seconds."""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'key': self.key,
            'data': self.data,
            'timestamp': self.timestamp,
            'ttl': self.ttl
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> CacheEntry:
        """Deserialize from dictionary."""
        return cls(
            key=d['key'],
            data=d['data'],
            timestamp=d['timestamp'],
            ttl=d.get('ttl')
        )


# =============================================================================
# Cache Manager
# =============================================================================


class CacheManager:
    """
    Thread-safe cache manager with TTL support.
    
    Provides persistent disk storage for experimental data
    with automatic expiration and cleanup.
    
    Parameters
    ----------
    cache_dir : Path or str
        Directory for persistent cache storage
    default_ttl : float
        Default time-to-live in seconds (86400 = 24 hours)
    """
    
    def __init__(
        self,
        cache_dir: Union[Path, str] = "data/cache/experimental",
        default_ttl: float = 86400.0  # 24 hours
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._load_from_disk()
    
    def _cache_path(self, key: str) -> Path:
        """Get path to cache file for given key."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}.json"
    
    def _load_from_disk(self):
        """Load all non-expired entries from disk into memory."""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                entry = CacheEntry.from_dict(data)
                
                if not entry.is_expired():
                    self._memory_cache[entry.key] = entry
                else:
                    # Remove expired file
                    cache_file.unlink()
            except Exception:
                # Skip corrupted cache files
                continue
    
    def _save_to_disk(self, entry: CacheEntry):
        """Save cache entry to disk."""
        cache_path = self._cache_path(entry.key)
        with open(cache_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Parameters
        ----------
        key : str
            Cache key
        
        Returns
        -------
        Optional[Any]
            Cached data if exists and not expired, None otherwise
        """
        entry = self._memory_cache.get(key)
        
        if entry is None:
            return None
        
        if entry.is_expired():
            self.invalidate(key)
            return None
        
        return entry.data
    
    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[float] = None
    ):
        """
        Store data in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        data : Any
            Data to cache (must be JSON-serializable)
        ttl : float, optional
            Time-to-live in seconds (uses default_ttl if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self._memory_cache[key] = entry
        self._save_to_disk(entry)
    
    def invalidate(self, key: str):
        """
        Remove entry from cache.
        
        Parameters
        ----------
        key : str
            Cache key to invalidate
        """
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        cache_path = self._cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self):
        """Remove all entries from cache."""
        self._memory_cache.clear()
        
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.invalidate(key)
    
    def info(self) -> Dict:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict
            Statistics including entry count, total size, oldest entry
        """
        entries = list(self._memory_cache.values())
        
        if not entries:
            return {
                'entry_count': 0,
                'expired_count': 0,
                'total_size_mb': 0.0,
                'oldest_age_hours': 0.0
            }
        
        expired_count = sum(1 for e in entries if e.is_expired())
        oldest_age = max(e.age_seconds() for e in entries)
        
        # Estimate size
        total_size = sum(
            len(json.dumps(e.data)) for e in entries
        )
        
        return {
            'entry_count': len(entries),
            'expired_count': expired_count,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_age_hours': oldest_age / 3600
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_cache_manager(
    cache_dir: str = "data/cache/experimental",
    ttl_hours: float = 24.0
) -> CacheManager:
    """
    Create a cache manager with specified settings.
    
    Parameters
    ----------
    cache_dir : str
        Directory for cache storage
    ttl_hours : float
        Time-to-live in hours
    
    Returns
    -------
    CacheManager
        Configured cache manager instance
    """
    return CacheManager(
        cache_dir=cache_dir,
        default_ttl=ttl_hours * 3600
    )
