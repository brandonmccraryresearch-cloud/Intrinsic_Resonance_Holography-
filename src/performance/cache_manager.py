"""
Cache Manager for IRH Computations

THEORETICAL FOUNDATION: IRH21.md §1.6, docs/ROADMAP.md §3.2

This module provides caching infrastructure for expensive computations:
    - LRU (Least Recently Used) in-memory cache
    - Disk-based persistent cache for large results
    - Function decorator for automatic caching
    - Cache statistics and management utilities

The caching system is designed to maintain numerical precision while
avoiding redundant computations, particularly for:
    - QNCD distance calculations (Appendix A)
    - RG flow integration (§1.2)
    - Topological invariant computation (Appendix D)

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - IRH v21.1 Manuscript Appendix A.4 (QNCD caching)
    - docs/ROADMAP.md §3.2 (Caching & Memoization)
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

__all__ = [
    'CacheManager',
    'LRUCache',
    'DiskCache',
    'create_cache',
    'get_cache',
    'cached',
    'clear_all_caches',
    'get_cache_stats',
]

# Configuration constants
HASH_TRUNCATE_LENGTH = 16  # Length of hash prefix for cache filenames

T = TypeVar('T')

# Global cache registry
_cache_registry: Dict[str, 'CacheManager'] = {}
_registry_lock = threading.Lock()


@dataclass
class CacheStats:
    """
    Statistics for cache performance monitoring.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.2 - Cache performance metrics
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'total_size': self.total_size,
            'max_size': self.max_size,
        }


class LRUCache(Generic[T]):
    """
    Thread-safe Least Recently Used (LRU) cache.
    
    Theoretical Reference:
        IRH21.md §1.6 - Computational efficiency
        docs/ROADMAP.md §3.2 - In-memory caching
    
    Parameters
    ----------
    max_size : int
        Maximum number of items to cache (default: 1000)
    
    Examples
    --------
    >>> cache = LRUCache(max_size=100)
    >>> cache.put('key1', [1, 2, 3])
    >>> cache.get('key1')
    [1, 2, 3]
    """
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[Hashable, T] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=max_size)
    
    # Theoretical Reference: IRH v21.4

    
    def get(self, key: Hashable) -> Optional[T]:
        """
        Get value from cache, moving to end (most recently used).
        
        Parameters
        ----------
        key : Hashable
            Cache key
            
        Returns
        -------
        Optional[T]
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            self._stats.misses += 1
            return None
    
    # Theoretical Reference: IRH v21.4

    
    def put(self, key: Hashable, value: T) -> None:
        """
        Put value into cache, evicting LRU item if necessary.
        
        Parameters
        ----------
        key : Hashable
            Cache key
        value : T
            Value to cache
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1
            self._cache[key] = value
            self._stats.total_size = len(self._cache)
    
    # Theoretical Reference: IRH v21.4

    
    def contains(self, key: Hashable) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._stats.total_size = 0
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)


class DiskCache:
    """
    Disk-based persistent cache for large computation results.
    
    Theoretical Reference:
        IRH21.md §1.6 - Persistent storage for reproducibility
        docs/ROADMAP.md §3.2 - Disk caching
    
    Parameters
    ----------
    cache_dir : Path or str
        Directory for cache files
    max_size_mb : int
        Maximum cache size in megabytes (default: 1000)
    
    Examples
    --------
    >>> cache = DiskCache('/tmp/irh_cache')
    >>> cache.put('computation_key', large_result_array)
    >>> result = cache.get('computation_key')
    """
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def __init__(
        self,
        cache_dir: Union[Path, str] = '/tmp/irh_cache',
        max_size_mb: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._index_file = self.cache_dir / 'index.json'
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=max_size_mb)
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                self._index = json.load(f)
        else:
            self._index = {'entries': {}, 'total_size': 0}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f)
    
    def _key_to_filename(self, key: str) -> Path:
        """Convert cache key to filename."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
        return self.cache_dir / f'{key_hash}.pkl'
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from disk cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value or None if not found
        """
        with self._lock:
            if key in self._index['entries']:
                filepath = self._key_to_filename(key)
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        self._stats.hits += 1
                        self._index['entries'][key]['last_access'] = time.time()
                        self._save_index()
                        return pickle.load(f)
            self._stats.misses += 1
            return None
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def put(self, key: str, value: Any) -> None:
        """
        Put value into disk cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache (must be picklable)
        """
        with self._lock:
            # Serialize value
            serialized = pickle.dumps(value)
            size = len(serialized)
            
            # Evict if necessary
            while (
                self._index['total_size'] + size > self.max_size_bytes
                and self._index['entries']
            ):
                self._evict_lru()
            
            # Save to disk
            filepath = self._key_to_filename(key)
            with open(filepath, 'wb') as f:
                f.write(serialized)
            
            # Update index
            self._index['entries'][key] = {
                'size': size,
                'created': time.time(),
                'last_access': time.time(),
            }
            self._index['total_size'] += size
            self._stats.total_size = self._index['total_size']
            self._save_index()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._index['entries']:
            return
        
        # Find LRU entry
        lru_key = min(
            self._index['entries'],
            key=lambda k: self._index['entries'][k]['last_access']
        )
        
        # Remove from disk
        filepath = self._key_to_filename(lru_key)
        if filepath.exists():
            filepath.unlink()
        
        # Update index
        size = self._index['entries'][lru_key]['size']
        del self._index['entries'][lru_key]
        self._index['total_size'] -= size
        self._stats.evictions += 1
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            for entry in self._index['entries']:
                filepath = self._key_to_filename(entry)
                if filepath.exists():
                    filepath.unlink()
            self._index = {'entries': {}, 'total_size': 0}
            self._stats.total_size = 0
            self._save_index()
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class CacheManager:
    """
    Unified cache manager combining LRU and disk caching.
    
    Theoretical Reference:
        IRH21.md §1.6 - Computational efficiency infrastructure
        docs/ROADMAP.md §3.2 - Multi-tier caching
    
    Parameters
    ----------
    name : str
        Cache manager name for registry
    lru_size : int
        Maximum LRU cache size (default: 1000)
    disk_cache_dir : Path or str, optional
        Directory for disk cache (None to disable disk caching)
    disk_max_mb : int
        Maximum disk cache size in MB (default: 1000)
    
    Examples
    --------
    >>> cache = CacheManager('qncd_cache', lru_size=500)
    >>> cache.get_or_compute('key', lambda: expensive_computation())
    """
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def __init__(
        self,
        name: str,
        lru_size: int = 1000,
        disk_cache_dir: Optional[Union[Path, str]] = None,
        disk_max_mb: int = 1000
    ):
        self.name = name
        self._lru = LRUCache(max_size=lru_size)
        self._disk = (
            DiskCache(disk_cache_dir, disk_max_mb)
            if disk_cache_dir
            else None
        )
        self._theoretical_reference = "IRH21.md §1.6, docs/ROADMAP.md §3.2"
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def get(self, key: Hashable) -> Optional[Any]:
        """
        Get value from cache (LRU first, then disk).
        
        Parameters
        ----------
        key : Hashable
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value or None if not found
        """
        # Check LRU first
        value = self._lru.get(key)
        if value is not None:
            return value
        
        # Check disk cache
        if self._disk is not None:
            str_key = str(key)
            value = self._disk.get(str_key)
            if value is not None:
                # Promote to LRU
                self._lru.put(key, value)
                return value
        
        return None
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def put(self, key: Hashable, value: Any, persist: bool = False) -> None:
        """
        Put value into cache.
        
        Parameters
        ----------
        key : Hashable
            Cache key
        value : Any
            Value to cache
        persist : bool
            Whether to persist to disk (default: False)
        """
        self._lru.put(key, value)
        
        if persist and self._disk is not None:
            str_key = str(key)
            self._disk.put(str_key, value)
    
    # Theoretical Reference: IRH v21.4

    
    def get_or_compute(
        self,
        key: Hashable,
        compute_fn: Callable[[], T],
        persist: bool = False
    ) -> T:
        """
        Get from cache or compute and cache result.
        
        Parameters
        ----------
        key : Hashable
            Cache key
        compute_fn : Callable
            Function to compute value if not cached
        persist : bool
            Whether to persist to disk (default: False)
            
        Returns
        -------
        T
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = compute_fn()
        self.put(key, value, persist=persist)
        return value
    
    # Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
    def clear(self) -> None:
        """Clear all caches."""
        self._lru.clear()
        if self._disk is not None:
            self._disk.clear()
    
    # Theoretical Reference: IRH v21.4

    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get combined cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics from LRU and disk caches
        """
        stats = {
            'name': self.name,
            'lru': self._lru.stats.to_dict(),
            'theoretical_reference': self._theoretical_reference,
        }
        if self._disk is not None:
            stats['disk'] = self._disk.stats.to_dict()
        return stats


# Theoretical Reference: IRH v21.4



def create_cache(
    name: str,
    lru_size: int = 1000,
    disk_cache_dir: Optional[Union[Path, str]] = None,
    disk_max_mb: int = 1000
) -> CacheManager:
    """
    Create and register a cache manager.
    
    Parameters
    ----------
    name : str
        Cache name
    lru_size : int
        LRU cache size
    disk_cache_dir : Path or str, optional
        Disk cache directory
    disk_max_mb : int
        Disk cache max size in MB
        
    Returns
    -------
    CacheManager
        Registered cache manager
    """
    with _registry_lock:
        if name in _cache_registry:
            return _cache_registry[name]
        cache = CacheManager(
            name, lru_size, disk_cache_dir, disk_max_mb
        )
        _cache_registry[name] = cache
        return cache


# Theoretical Reference: IRH v21.4



def get_cache(name: str) -> Optional[CacheManager]:
    """
    Get a registered cache manager by name.
    
    Parameters
    ----------
    name : str
        Cache name
        
    Returns
    -------
    Optional[CacheManager]
        Cache manager or None if not found
    """
    with _registry_lock:
        return _cache_registry.get(name)


def cached(
    cache_name: str = 'default',
    key_fn: Optional[Callable[..., Hashable]] = None,
    persist: bool = False
):
    """
    Decorator for automatic function result caching.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.2 - Automatic memoization
    
    Parameters
    ----------
    cache_name : str
        Name of cache to use
    key_fn : Callable, optional
        Function to generate cache key from args
    persist : bool
        Whether to persist to disk
        
    Examples
    --------
    >>> @cached('qncd_cache')
    ... def compute_qncd(s1: str, s2: str) -> float:
    ...     return expensive_qncd_computation(s1, s2)
    """
    # Theoretical Reference: IRH v21.4

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create cache
            cache = get_cache(cache_name)
            if cache is None:
                cache = create_cache(cache_name)
            
            # Generate cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                key = (
                    func.__name__,
                    _make_hashable(args),
                    _make_hashable(tuple(sorted(kwargs.items())))
                )
            
            return cache.get_or_compute(
                key,
                lambda: func(*args, **kwargs),
                persist=persist
            )
        
        return wrapper
    return decorator


def _make_hashable(obj: Any) -> Hashable:
    """Convert object to hashable representation."""
    if isinstance(obj, np.ndarray):
        return (obj.tobytes(), obj.shape, str(obj.dtype))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(
            (k, _make_hashable(v))
            for k, v in sorted(obj.items())
        )
    elif isinstance(obj, (set, frozenset)):
        return frozenset(_make_hashable(item) for item in obj)
    return obj


# Theoretical Reference: IRH v21.4



def clear_all_caches() -> None:
    """Clear all registered caches."""
    with _registry_lock:
        for cache in _cache_registry.values():
            cache.clear()


# Theoretical Reference: IRH v21.4 (Memory/Cache Management Infrastructure)
def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics from all registered caches.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Statistics keyed by cache name
    """
    with _registry_lock:
        return {
            name: cache.get_stats()
            for name, cache in _cache_registry.items()
        }
