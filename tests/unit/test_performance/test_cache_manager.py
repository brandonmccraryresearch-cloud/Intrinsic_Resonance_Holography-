"""
Tests for Cache Manager Module

THEORETICAL FOUNDATION: docs/ROADMAP.md §3.2

Tests for caching infrastructure:
    - LRU cache operations
    - Disk cache operations
    - Cache manager unified interface
    - Decorator-based caching

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest


class TestLRUCache:
    """Tests for LRU in-memory cache."""
    
    def test_lru_cache_creation(self):
        """Test LRU cache instantiation."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache) == 0
    
    def test_lru_cache_put_get(self):
        """Test basic put and get operations."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache(max_size=100)
        cache.put('key1', 'value1')
        cache.put('key2', [1, 2, 3])
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == [1, 2, 3]
        assert cache.get('nonexistent') is None
    
    def test_lru_cache_eviction(self):
        """Test LRU eviction when max size reached."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache(max_size=3)
        cache.put('key1', 1)
        cache.put('key2', 2)
        cache.put('key3', 3)
        
        # Access key1 to make it recently used
        cache.get('key1')
        
        # Add key4, should evict key2 (least recently used)
        cache.put('key4', 4)
        
        assert cache.get('key1') == 1  # Still present (accessed recently)
        assert cache.get('key2') is None  # Evicted
        assert cache.get('key3') == 3
        assert cache.get('key4') == 4
    
    def test_lru_cache_stats(self):
        """Test cache statistics tracking."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache(max_size=5)
        cache.put('key1', 1)
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
    
    def test_lru_cache_contains(self):
        """Test contains check."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache()
        cache.put('key1', 1)
        
        assert cache.contains('key1')
        assert not cache.contains('key2')
    
    def test_lru_cache_clear(self):
        """Test cache clearing."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache()
        cache.put('key1', 1)
        cache.put('key2', 2)
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get('key1') is None


class TestDiskCache:
    """Tests for disk-based cache."""
    
    def test_disk_cache_creation(self):
        """Test disk cache instantiation."""
        from src.performance.cache_manager import DiskCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir, max_size_mb=10)
            assert cache.cache_dir == Path(tmpdir)
    
    def test_disk_cache_put_get(self):
        """Test basic put and get operations."""
        from src.performance.cache_manager import DiskCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            
            cache.put('key1', {'data': [1, 2, 3]})
            cache.put('key2', np.array([1.0, 2.0, 3.0]))
            
            assert cache.get('key1') == {'data': [1, 2, 3]}
            np.testing.assert_array_equal(
                cache.get('key2'),
                np.array([1.0, 2.0, 3.0])
            )
            assert cache.get('nonexistent') is None
    
    def test_disk_cache_persistence(self):
        """Test that disk cache persists across instances."""
        from src.performance.cache_manager import DiskCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write with first instance
            cache1 = DiskCache(tmpdir)
            cache1.put('persistent_key', 'persistent_value')
            
            # Read with second instance
            cache2 = DiskCache(tmpdir)
            assert cache2.get('persistent_key') == 'persistent_value'
    
    def test_disk_cache_stats(self):
        """Test disk cache statistics."""
        from src.performance.cache_manager import DiskCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            cache.put('key1', 'value1')
            cache.get('key1')  # Hit
            cache.get('missing')  # Miss
            
            stats = cache.stats
            assert stats.hits == 1
            assert stats.misses == 1


class TestCacheManager:
    """Tests for unified cache manager."""
    
    def test_cache_manager_creation(self):
        """Test cache manager instantiation."""
        from src.performance.cache_manager import CacheManager
        
        manager = CacheManager('test_cache', lru_size=100)
        assert manager.name == 'test_cache'
    
    def test_cache_manager_lru_only(self):
        """Test cache manager with LRU only."""
        from src.performance.cache_manager import CacheManager
        
        manager = CacheManager('test_lru', lru_size=50)
        
        manager.put('key1', 'value1')
        assert manager.get('key1') == 'value1'
    
    def test_cache_manager_with_disk(self):
        """Test cache manager with disk backing."""
        from src.performance.cache_manager import CacheManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(
                'test_disk',
                lru_size=50,
                disk_cache_dir=tmpdir
            )
            
            # Put without persist
            manager.put('lru_only', 'value1')
            
            # Put with persist
            manager.put('persisted', 'value2', persist=True)
            
            assert manager.get('lru_only') == 'value1'
            assert manager.get('persisted') == 'value2'
    
    def test_cache_manager_get_or_compute(self):
        """Test get_or_compute functionality."""
        from src.performance.cache_manager import CacheManager
        
        manager = CacheManager('test_compute')
        
        compute_count = [0]
        
        def expensive_fn():
            compute_count[0] += 1
            return 42
        
        # First call should compute
        result1 = manager.get_or_compute('key', expensive_fn)
        assert result1 == 42
        assert compute_count[0] == 1
        
        # Second call should use cache
        result2 = manager.get_or_compute('key', expensive_fn)
        assert result2 == 42
        assert compute_count[0] == 1  # Not computed again
    
    def test_cache_manager_get_stats(self):
        """Test statistics retrieval."""
        from src.performance.cache_manager import CacheManager
        
        manager = CacheManager('test_stats')
        manager.put('key1', 1)
        manager.get('key1')
        
        stats = manager.get_stats()
        assert stats['name'] == 'test_stats'
        assert 'lru' in stats
        assert stats['lru']['hits'] == 1


class TestCacheRegistry:
    """Tests for cache registry functions."""
    
    def test_create_and_get_cache(self):
        """Test cache creation and retrieval."""
        from src.performance.cache_manager import create_cache, get_cache, clear_all_caches
        
        # Clear any existing caches
        clear_all_caches()
        
        cache = create_cache('test_registry_cache')
        retrieved = get_cache('test_registry_cache')
        
        assert cache is retrieved
    
    def test_create_cache_returns_existing(self):
        """Test that create_cache returns existing cache."""
        from src.performance.cache_manager import create_cache, clear_all_caches
        
        clear_all_caches()
        
        cache1 = create_cache('same_name')
        cache2 = create_cache('same_name')
        
        assert cache1 is cache2
    
    def test_get_cache_stats(self):
        """Test global cache stats retrieval."""
        from src.performance.cache_manager import (
            create_cache, get_cache_stats, clear_all_caches
        )
        
        clear_all_caches()
        
        cache1 = create_cache('cache_a')
        cache2 = create_cache('cache_b')
        
        cache1.put('key', 1)
        
        stats = get_cache_stats()
        assert 'cache_a' in stats
        assert 'cache_b' in stats


class TestCachedDecorator:
    """Tests for @cached decorator."""
    
    def test_cached_decorator_basic(self):
        """Test basic decorator usage."""
        from src.performance.cache_manager import cached, clear_all_caches
        
        clear_all_caches()
        
        call_count = [0]
        
        @cached('test_decorator_cache')
        def add(a, b):
            call_count[0] += 1
            return a + b
        
        # First call
        result1 = add(1, 2)
        assert result1 == 3
        assert call_count[0] == 1
        
        # Second call with same args should use cache
        result2 = add(1, 2)
        assert result2 == 3
        assert call_count[0] == 1
        
        # Different args should compute
        result3 = add(3, 4)
        assert result3 == 7
        assert call_count[0] == 2
    
    def test_cached_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        from src.performance.cache_manager import cached, clear_all_caches
        
        clear_all_caches()
        
        call_count = [0]
        
        @cached('test_kwargs_cache')
        def multiply(a, b=2):
            call_count[0] += 1
            return a * b
        
        result1 = multiply(3, b=4)
        result2 = multiply(3, b=4)
        
        assert result1 == 12
        assert result2 == 12
        assert call_count[0] == 1


class TestMakeHashable:
    """Tests for hashability utilities."""
    
    def test_make_hashable_numpy_array(self):
        """Test hashing numpy arrays."""
        from src.performance.cache_manager import _make_hashable
        
        arr = np.array([1.0, 2.0, 3.0])
        hashable = _make_hashable(arr)
        
        # Should be hashable (tuple)
        assert isinstance(hashable, tuple)
        hash(hashable)  # Should not raise
    
    def test_make_hashable_nested(self):
        """Test hashing nested structures."""
        from src.performance.cache_manager import _make_hashable
        
        nested = {'a': [1, 2, 3], 'b': {'c': 4}}
        hashable = _make_hashable(nested)
        
        hash(hashable)  # Should not raise


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_lru_cache_thread_safety(self):
        """Test LRU cache is thread-safe."""
        from src.performance.cache_manager import LRUCache
        
        cache = LRUCache(max_size=1000)
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    cache.put(f'key_{threading.current_thread().name}_{i}', i)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for i in range(100):
                    cache.get(f'key_{threading.current_thread().name}_{i}')
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, name=f'writer_{i}'))
            threads.append(threading.Thread(target=reader, name=f'reader_{i}'))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestTheoreticalReference:
    """Tests for theoretical grounding."""
    
    def test_cache_manager_has_reference(self):
        """Test cache manager includes theoretical reference."""
        from src.performance.cache_manager import CacheManager
        
        manager = CacheManager('test')
        assert hasattr(manager, '_theoretical_reference')
        assert 'ROADMAP' in manager._theoretical_reference
    
    def test_cache_stats_to_dict(self):
        """Test cache stats serialization."""
        from src.performance.cache_manager import CacheStats
        
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        
        assert d['hits'] == 10
        assert d['misses'] == 5
        assert d['hit_rate'] == 10/15
