"""
Tests for Memory Optimization Module

THEORETICAL FOUNDATION: docs/ROADMAP.md §3.3

Tests for memory optimization utilities:
    - Array pooling
    - Sparse field arrays
    - Memory monitoring
    - Memory optimizer
    - Utility functions

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import gc
import tempfile
import numpy as np
import pytest


class TestArrayPool:
    """Tests for ArrayPool class."""
    
    def test_pool_creation(self):
        """Test ArrayPool instantiation."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool(max_pool_size=50)
        assert pool.max_pool_size == 50
    
    def test_acquire_new_array(self):
        """Test acquiring new array when pool is empty."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        arr = pool.acquire((10, 10), np.float64)
        
        assert arr.shape == (10, 10)
        assert arr.dtype == np.float64
    
    def test_acquire_with_fill_value(self):
        """Test acquiring array with fill value."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        arr = pool.acquire((5, 5), np.float64, fill_value=1.0)
        
        assert np.all(arr == 1.0)
    
    def test_release_and_reuse(self):
        """Test releasing and reusing arrays."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        
        # Acquire and release
        arr1 = pool.acquire((10, 10), np.float64)
        arr1_id = id(arr1)
        pool.release(arr1)
        
        # Acquire same shape/dtype should reuse
        arr2 = pool.acquire((10, 10), np.float64)
        
        # Should get the same array back (from pool)
        assert id(arr2) == arr1_id
    
    def test_pool_stats(self):
        """Test pool statistics tracking."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        
        # Miss (new allocation)
        arr = pool.acquire((5, 5), np.float64)
        pool.release(arr)
        
        # Hit (from pool)
        pool.acquire((5, 5), np.float64)
        
        stats = pool.stats
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.pool_efficiency == 0.5
    
    def test_pool_clear(self):
        """Test clearing pool."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        arr = pool.acquire((10, 10), np.float64)
        pool.release(arr)
        
        pool.clear()
        assert pool.stats.pooled_arrays == 0
    
    def test_large_array_not_pooled(self):
        """Test that large arrays are not pooled."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool(max_array_size_mb=0.001)  # Very small limit
        arr = pool.acquire((1000, 1000), np.float64)  # ~8MB
        
        result = pool.release(arr)
        assert result is False  # Not pooled
    
    def test_get_pool_info(self):
        """Test pool info retrieval."""
        from src.performance.memory_optimization import ArrayPool
        
        pool = ArrayPool()
        arr = pool.acquire((10, 10), np.float64)
        pool.release(arr)
        
        info = pool.get_pool_info()
        assert 'pools' in info
        assert 'stats' in info
        assert 'theoretical_reference' in info


class TestSparseFieldArray:
    """Tests for SparseFieldArray class."""
    
    def test_sparse_creation(self):
        """Test SparseFieldArray instantiation."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10, 10, 10), np.complex128)
        assert sparse.shape == (10, 10, 10, 10)
        assert sparse.nnz == 0
    
    def test_sparse_set_get(self):
        """Test setting and getting values."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.float64)
        sparse[5, 5] = 1.0
        
        assert sparse[5, 5] == 1.0
        assert sparse[0, 0] == 0.0  # Default
    
    def test_sparse_complex(self):
        """Test complex values."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.complex128)
        sparse[3, 4] = 1.0 + 2.0j
        
        assert sparse[3, 4] == 1.0 + 2.0j
    
    def test_sparse_threshold(self):
        """Test sparsity threshold."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.float64, sparsity_threshold=0.1)
        sparse[0, 0] = 0.05  # Below threshold
        sparse[1, 1] = 0.5   # Above threshold
        
        assert sparse.nnz == 1  # Only one stored
        assert sparse[0, 0] == 0.0  # Treated as zero
        assert sparse[1, 1] == 0.5
    
    def test_sparse_density(self):
        """Test density calculation."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.float64)
        sparse[0, 0] = 1.0
        sparse[1, 1] = 1.0
        
        assert sparse.nnz == 2
        assert sparse.density == 2 / 100
    
    def test_sparse_to_dense(self):
        """Test conversion to dense array."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((5, 5), np.float64)
        sparse[2, 2] = 3.0
        sparse[4, 4] = 5.0
        
        dense = sparse.to_dense()
        assert dense[2, 2] == 3.0
        assert dense[4, 4] == 5.0
        assert dense[0, 0] == 0.0
    
    def test_sparse_from_dense(self):
        """Test creation from dense array."""
        from src.performance.memory_optimization import SparseFieldArray
        
        dense = np.zeros((10, 10))
        dense[3, 3] = 1.0
        dense[7, 7] = 2.0
        
        sparse = SparseFieldArray.from_dense(dense)
        assert sparse.nnz == 2
        assert sparse[3, 3] == 1.0
        assert sparse[7, 7] == 2.0
    
    def test_sparse_memory_savings(self):
        """Test memory compression ratio."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((100, 100), np.float64)
        sparse[50, 50] = 1.0
        
        # Should have significant compression
        assert sparse.compression_ratio > 10
    
    def test_sparse_get_info(self):
        """Test info retrieval."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.float64)
        sparse[5, 5] = 1.0
        
        info = sparse.get_info()
        assert info['shape'] == (10, 10)
        assert info['nnz'] == 1
        assert 'theoretical_reference' in info


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""
    
    def test_monitor_creation(self):
        """Test MemoryMonitor instantiation."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor()
        assert monitor._running is False
    
    def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(track_gc=False)
        monitor.start()
        assert monitor._running is True
        
        monitor.stop()
        assert monitor._running is False
    
    def test_monitor_snapshot(self):
        """Test taking snapshots."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(track_gc=False)
        monitor.start()
        
        snapshot = monitor.snapshot('test_label')
        assert snapshot['label'] == 'test_label'
        assert 'timestamp' in snapshot
        
        monitor.stop()
    
    def test_monitor_get_snapshots(self):
        """Test retrieving snapshots."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(track_gc=False)
        monitor.start()
        monitor.snapshot('snap1')
        monitor.snapshot('snap2')
        monitor.stop()
        
        snapshots = monitor.get_snapshots()
        assert len(snapshots) == 2
    
    def test_monitor_clear_snapshots(self):
        """Test clearing snapshots."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(track_gc=False)
        monitor.start()
        monitor.snapshot('snap1')
        monitor.clear_snapshots()
        
        assert len(monitor.get_snapshots()) == 0
        monitor.stop()
    
    def test_monitor_get_report(self):
        """Test report generation."""
        from src.performance.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(track_gc=False)
        report = monitor.get_report()
        
        assert 'stats' in report
        assert 'snapshots' in report
        assert 'theoretical_reference' in report


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer class."""
    
    def test_optimizer_creation(self):
        """Test MemoryOptimizer instantiation."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        assert optimizer.enable_pooling is True
        assert optimizer.enable_gc_optimization is True
    
    def test_optimizer_acquire_release(self):
        """Test array acquisition and release."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        arr = optimizer.acquire_array((10, 10), np.float64, fill_value=0.0)
        
        assert arr.shape == (10, 10)
        assert np.all(arr == 0.0)
        
        optimizer.release_array(arr)
    
    def test_optimizer_context_manager(self):
        """Test context manager usage."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        with MemoryOptimizer() as optimizer:
            arr = optimizer.acquire_array((5, 5), np.float64)
            optimizer.release_array(arr)
    
    def test_optimizer_gc_optimization(self):
        """Test GC optimization."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        original_threshold = gc.get_threshold()
        
        optimizer = MemoryOptimizer(enable_gc_optimization=True)
        optimizer.optimize_gc()
        
        # Threshold should be changed
        new_threshold = gc.get_threshold()
        assert new_threshold != original_threshold
        
        optimizer.restore_gc()
        
        # Should be restored
        restored_threshold = gc.get_threshold()
        assert restored_threshold == original_threshold
    
    def test_optimizer_get_stats(self):
        """Test statistics retrieval."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        stats = optimizer.get_stats()
        
        assert 'monitor' in stats
        assert 'pool' in stats
        assert 'theoretical_reference' in stats
    
    def test_optimizer_without_pooling(self):
        """Test optimizer without pooling."""
        from src.performance.memory_optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer(enable_pooling=False)
        arr = optimizer.acquire_array((5, 5), np.float64)
        
        # Should still work but no pool
        assert arr.shape == (5, 5)


class TestMemoryEfficientDecorator:
    """Tests for @memory_efficient decorator."""
    
    def test_decorator_basic(self):
        """Test basic decorator usage."""
        from src.performance.memory_optimization import memory_efficient
        
        @memory_efficient('test_pool')
        def compute_something():
            return np.zeros((10, 10))
        
        result = compute_something()
        assert result.shape == (10, 10)
    
    def test_decorator_preserves_result(self):
        """Test decorator preserves function result."""
        from src.performance.memory_optimization import memory_efficient
        
        @memory_efficient('test_pool')
        def add(a, b):
            return a + b
        
        assert add(1, 2) == 3
    
    def test_decorator_with_gc_optimize(self):
        """Test decorator with GC optimization."""
        from src.performance.memory_optimization import memory_efficient
        
        @memory_efficient('test_pool', gc_optimize=True)
        def allocate_arrays():
            return [np.zeros(100) for _ in range(10)]
        
        result = allocate_arrays()
        assert len(result) == 10


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_memory_stats(self):
        """Test get_memory_stats function."""
        from src.performance.memory_optimization import get_memory_stats
        
        stats = get_memory_stats()
        assert 'gc_objects' in stats
        assert 'gc_threshold' in stats
        assert 'theoretical_reference' in stats
    
    def test_optimize_gc(self):
        """Test optimize_gc function."""
        from src.performance.memory_optimization import optimize_gc
        
        original = gc.get_threshold()
        
        optimize_gc(threshold=(10000, 100, 10))
        assert gc.get_threshold() == (10000, 100, 10)
        
        # Restore
        gc.set_threshold(*original)
    
    def test_optimize_gc_disable(self):
        """Test GC disable option."""
        from src.performance.memory_optimization import optimize_gc
        
        was_enabled = gc.isenabled()
        
        optimize_gc(disable=True)
        assert not gc.isenabled()
        
        # Restore
        if was_enabled:
            gc.enable()
    
    def test_create_memory_mapped_array(self):
        """Test memory-mapped array creation."""
        from src.performance.memory_optimization import create_memory_mapped_array
        
        with tempfile.NamedTemporaryFile(suffix='.dat') as f:
            mmap = create_memory_mapped_array(f.name, (10, 10), np.float64)
            mmap[5, 5] = 1.0
            mmap.flush()
            
            assert mmap[5, 5] == 1.0
            assert mmap.shape == (10, 10)
    
    def test_estimate_memory_usage_dense(self):
        """Test memory estimation for dense arrays."""
        from src.performance.memory_optimization import estimate_memory_usage
        
        estimate = estimate_memory_usage(
            (100, 100),
            np.float64
        )
        
        assert estimate['total_elements'] == 10000
        assert estimate['dense_bytes'] == 10000 * 8
        assert 'dense_mb' in estimate
    
    def test_estimate_memory_usage_sparse(self):
        """Test memory estimation for sparse arrays."""
        from src.performance.memory_optimization import estimate_memory_usage
        
        estimate = estimate_memory_usage(
            (100, 100),
            np.float64,
            sparse=True,
            expected_density=0.01
        )
        
        assert 'sparse' in estimate
        assert estimate['sparse']['expected_density'] == 0.01
        assert estimate['sparse']['compression_ratio'] > 1


class TestMemoryStatsDataclass:
    """Tests for MemoryStats dataclass."""
    
    def test_memory_stats_creation(self):
        """Test MemoryStats instantiation."""
        from src.performance.memory_optimization import MemoryStats
        
        stats = MemoryStats(
            current_bytes=1024*1024,
            peak_bytes=2*1024*1024
        )
        
        assert stats.current_mb == 1.0
        assert stats.peak_mb == 2.0
    
    def test_memory_stats_to_dict(self):
        """Test serialization."""
        from src.performance.memory_optimization import MemoryStats
        
        stats = MemoryStats(current_bytes=1000, peak_bytes=2000)
        d = stats.to_dict()
        
        assert d['current_bytes'] == 1000
        assert d['peak_bytes'] == 2000


class TestTheoreticalGrounding:
    """Tests for theoretical grounding."""
    
    def test_sparse_array_reference(self):
        """Test SparseFieldArray has theoretical reference."""
        from src.performance.memory_optimization import SparseFieldArray
        
        sparse = SparseFieldArray((10, 10), np.float64)
        assert 'IRH' in sparse._theoretical_reference
    
    def test_utility_functions_references(self):
        """Test utility functions include references."""
        from src.performance.memory_optimization import (
            get_memory_stats, estimate_memory_usage
        )
        
        stats = get_memory_stats()
        assert 'theoretical_reference' in stats
        
        estimate = estimate_memory_usage((10, 10), np.float64)
        assert 'theoretical_reference' in estimate


class TestIntegration:
    """Integration tests for memory optimization."""
    
    def test_full_workflow(self):
        """Test complete memory optimization workflow."""
        from src.performance.memory_optimization import (
            MemoryOptimizer, SparseFieldArray
        )
        
        with MemoryOptimizer() as optimizer:
            # Use pooled arrays
            arr1 = optimizer.acquire_array((100, 100), np.float64, fill_value=0.0)
            arr2 = optimizer.acquire_array((100, 100), np.float64, fill_value=0.0)
            
            # Compute something
            arr1 += arr2
            
            # Release back to pool
            optimizer.release_array(arr1)
            optimizer.release_array(arr2)
            
            # Use sparse for large field
            sparse = SparseFieldArray((50, 50, 50, 50), np.complex128)
            sparse[10, 10, 10, 10] = 1.0 + 2.0j
            
            # Verify memory savings
            assert sparse.compression_ratio > 100
        
        stats = optimizer.get_stats()
        assert stats['pool']['stats']['allocated_arrays'] == 2
    
    def test_sparse_field_rg_flow_scenario(self):
        """Test sparse arrays for cGFT field scenario."""
        from src.performance.memory_optimization import SparseFieldArray
        
        # Simulate sparse cGFT field configuration
        # φ(g₁, g₂, g₃, g₄) on N=20 lattice
        N = 20
        sparse_field = SparseFieldArray((N, N, N, N), np.complex128)
        
        # Set some localized excitations
        for i in range(10):
            sparse_field[i, i, i, i] = np.exp(1j * i * np.pi / 5)
        
        # Memory should be much less than dense
        dense_size = N**4 * 16  # complex128 = 16 bytes
        sparse_size = sparse_field.memory_bytes
        
        assert sparse_size < dense_size / 100
        
        # Can still convert to dense when needed
        dense = sparse_field.to_dense()
        assert dense.shape == (N, N, N, N)
