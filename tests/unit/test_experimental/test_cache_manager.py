"""
Tests for Cache Manager

Tests the caching infrastructure for experimental data.
"""

import json
import pytest
import time
from pathlib import Path
import tempfile
import shutil

from src.experimental.cache_manager import (
    CacheEntry,
    CacheManager,
    create_cache_manager
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create cache manager with temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)


class TestCacheEntry:
    """Tests for CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        entry = CacheEntry(
            key='test_key',
            data={'value': 42},
            timestamp=time.time(),
            ttl=3600
        )
        
        assert entry.key == 'test_key'
        assert entry.data == {'value': 42}
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create expired entry
        entry = CacheEntry(
            key='test_key',
            data={'value': 42},
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl=3600  # 1 hour TTL
        )
        
        assert entry.is_expired()
        assert entry.age_seconds() >= 7200
    
    def test_cache_entry_no_expiration(self):
        """Test cache entry with no TTL (never expires)."""
        entry = CacheEntry(
            key='test_key',
            data={'value': 42},
            timestamp=time.time() - 86400,  # 1 day ago
            ttl=None  # Never expires
        )
        
        assert not entry.is_expired()
    
    def test_cache_entry_serialization(self):
        """Test CacheEntry to_dict and from_dict."""
        entry = CacheEntry(
            key='test_key',
            data={'value': 42, 'nested': [1, 2, 3]},
            timestamp=1234567890.0,
            ttl=3600
        )
        
        # Serialize
        entry_dict = entry.to_dict()
        assert entry_dict['key'] == 'test_key'
        assert entry_dict['data'] == {'value': 42, 'nested': [1, 2, 3]}
        assert entry_dict['timestamp'] == 1234567890.0
        assert entry_dict['ttl'] == 3600
        
        # Deserialize
        restored = CacheEntry.from_dict(entry_dict)
        assert restored.key == entry.key
        assert restored.data == entry.data
        assert restored.timestamp == entry.timestamp
        assert restored.ttl == entry.ttl


class TestCacheManager:
    """Tests for CacheManager class."""
    
    def test_cache_manager_creation(self, temp_cache_dir):
        """Test CacheManager creation."""
        manager = CacheManager(cache_dir=temp_cache_dir, default_ttl=7200)
        
        assert manager.cache_dir == Path(temp_cache_dir)
        assert manager.default_ttl == 7200
        assert len(manager._memory_cache) == 0
    
    def test_set_and_get(self, cache_manager):
        """Test setting and getting cached data."""
        # Set data
        cache_manager.set('test_key', {'value': 42}, ttl=3600)
        
        # Get data
        data = cache_manager.get('test_key')
        assert data == {'value': 42}
    
    def test_get_nonexistent(self, cache_manager):
        """Test getting nonexistent key returns None."""
        data = cache_manager.get('nonexistent_key')
        assert data is None
    
    def test_cache_expiration(self, cache_manager):
        """Test cache entry expiration."""
        # Set data with very short TTL
        cache_manager.set('test_key', {'value': 42}, ttl=0.1)
        
        # Should exist immediately
        assert cache_manager.get('test_key') == {'value': 42}
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should return None after expiration
        assert cache_manager.get('test_key') is None
    
    def test_default_ttl(self, cache_manager):
        """Test that default TTL is used when not specified."""
        # Set data without TTL
        cache_manager.set('test_key', {'value': 42})
        
        # Check entry has default TTL
        entry = cache_manager._memory_cache['test_key']
        assert entry.ttl == 3600  # default from fixture
    
    def test_invalidate(self, cache_manager):
        """Test cache invalidation."""
        # Set data
        cache_manager.set('test_key', {'value': 42})
        assert cache_manager.get('test_key') == {'value': 42}
        
        # Invalidate
        cache_manager.invalidate('test_key')
        assert cache_manager.get('test_key') is None
    
    def test_clear(self, cache_manager):
        """Test clearing all cache entries."""
        # Set multiple entries
        cache_manager.set('key1', {'value': 1})
        cache_manager.set('key2', {'value': 2})
        cache_manager.set('key3', {'value': 3})
        
        assert len(cache_manager._memory_cache) == 3
        
        # Clear all
        cache_manager.clear()
        
        assert len(cache_manager._memory_cache) == 0
        assert cache_manager.get('key1') is None
        assert cache_manager.get('key2') is None
        assert cache_manager.get('key3') is None
    
    def test_cleanup_expired(self, cache_manager):
        """Test cleanup of expired entries."""
        # Set some entries with different TTLs
        cache_manager.set('short_ttl', {'value': 1}, ttl=0.1)
        cache_manager.set('long_ttl', {'value': 2}, ttl=3600)
        
        # Wait for short TTL to expire
        time.sleep(0.2)
        
        # Cleanup expired
        cache_manager.cleanup_expired()
        
        # Short TTL should be removed
        assert cache_manager.get('short_ttl') is None
        # Long TTL should still exist
        assert cache_manager.get('long_ttl') == {'value': 2}
    
    def test_persistence(self, temp_cache_dir):
        """Test cache persistence to disk."""
        # Create manager and set data
        manager1 = CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)
        manager1.set('persistent_key', {'value': 42})
        
        # Create new manager with same directory
        manager2 = CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)
        
        # Data should be loaded from disk
        assert manager2.get('persistent_key') == {'value': 42}
    
    def test_expired_persistence(self, temp_cache_dir):
        """Test that expired entries are not loaded from disk."""
        # Create manager and set data with short TTL
        manager1 = CacheManager(cache_dir=temp_cache_dir, default_ttl=0.1)
        manager1.set('test_key', {'value': 42})
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Create new manager - should not load expired entry
        manager2 = CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)
        assert manager2.get('test_key') is None
    
    def test_cache_info(self, cache_manager):
        """Test cache statistics."""
        # Empty cache
        info = cache_manager.info()
        assert info['entry_count'] == 0
        
        # Add entries
        cache_manager.set('key1', {'value': 1})
        time.sleep(0.1)
        cache_manager.set('key2', {'value': 2})
        
        # Check info
        info = cache_manager.info()
        assert info['entry_count'] == 2
        assert info['expired_count'] == 0
        assert info['total_size_mb'] > 0
        assert info['oldest_age_hours'] > 0


class TestCreateCacheManager:
    """Tests for create_cache_manager convenience function."""
    
    def test_create_cache_manager(self, temp_cache_dir):
        """Test create_cache_manager function."""
        manager = create_cache_manager(
            cache_dir=temp_cache_dir,
            ttl_hours=48
        )
        
        assert manager.cache_dir == Path(temp_cache_dir)
        assert manager.default_ttl == 48 * 3600  # Convert hours to seconds
    
    def test_create_cache_manager_defaults(self):
        """Test create_cache_manager with defaults."""
        manager = create_cache_manager()
        
        assert manager.default_ttl == 24 * 3600  # 24 hours default


class TestCacheManagerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_cache_path_sanitization(self, cache_manager):
        """Test that keys with path separators are sanitized."""
        # Set key with path separators
        cache_manager.set('path/with/slashes', {'value': 42})
        
        # Should work without issues
        assert cache_manager.get('path/with/slashes') == {'value': 42}
    
    def test_corrupted_cache_file(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        # Create corrupted cache file
        cache_dir = Path(temp_cache_dir)
        corrupted_file = cache_dir / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write("{ this is not valid JSON")
        
        # Create manager - should skip corrupted file
        manager = CacheManager(cache_dir=temp_cache_dir)
        assert len(manager._memory_cache) == 0
    
    def test_multiple_data_types(self, cache_manager):
        """Test caching different data types."""
        # Test various data types
        test_data = [
            ('dict', {'key': 'value'}),
            ('list', [1, 2, 3]),
            ('string', 'test string'),
            ('number', 42),
            ('float', 3.14159),
            ('bool', True),
            ('nested', {'list': [1, 2], 'dict': {'a': 1}}),
        ]
        
        for key, data in test_data:
            cache_manager.set(key, data)
            retrieved = cache_manager.get(key)
            assert retrieved == data, f"Failed for {key}: {retrieved} != {data}"
