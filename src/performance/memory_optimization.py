"""
Memory Optimization Module for IRH Computations

THEORETICAL FOUNDATION: IRH21.md §1.6, docs/ROADMAP.md §3.3

This module provides memory optimization utilities for IRH computations:
    - Memory-efficient array pooling and reuse
    - Sparse matrix support for large lattice computations
    - Memory monitoring and leak detection
    - Garbage collection optimization
    - Memory-mapped arrays for large datasets

The memory optimization layer enables exascale computations by
minimizing memory allocation overhead and reducing memory footprint.

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - IRH v21.1 Manuscript §1.6 (Computational Infrastructure)
    - docs/ROADMAP.md §3.3 (Memory Optimization)
"""

from __future__ import annotations

import gc
import sys
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ArrayPool',
    'SparseFieldArray',
    'MemoryMonitor',
    'MemoryOptimizer',
    'memory_efficient',
    'get_memory_stats',
    'optimize_gc',
    'create_memory_mapped_array',
    'estimate_memory_usage',
]

T = TypeVar('T')

# Global memory monitor instance
_global_monitor: Optional['MemoryMonitor'] = None
_monitor_lock = threading.Lock()


@dataclass
class MemoryStats:
    """
    Memory usage statistics.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory monitoring
    """
    current_bytes: int = 0
    peak_bytes: int = 0
    allocated_arrays: int = 0
    pooled_arrays: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gc_collections: int = 0
    
    @property
    def current_mb(self) -> float:
        """Current memory in MB."""
        return self.current_bytes / (1024 * 1024)
    
    @property
    def peak_mb(self) -> float:
        """Peak memory in MB."""
        return self.peak_bytes / (1024 * 1024)
    
    @property
    def pool_efficiency(self) -> float:
        """Pool hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_bytes': self.current_bytes,
            'current_mb': self.current_mb,
            'peak_bytes': self.peak_bytes,
            'peak_mb': self.peak_mb,
            'allocated_arrays': self.allocated_arrays,
            'pooled_arrays': self.pooled_arrays,
            'pool_efficiency': self.pool_efficiency,
            'gc_collections': self.gc_collections,
        }


class ArrayPool:
    """
    Thread-safe array pool for efficient memory reuse.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.6 - Computational efficiency
        docs/ROADMAP.md §3.3 - Array pooling
    
    The pool maintains pre-allocated arrays organized by shape and dtype,
    reducing allocation overhead for repeated computations like RG flow
    integration and QNCD batch calculations.
    
    Parameters
    ----------
    max_pool_size : int
        Maximum number of arrays to pool per (shape, dtype) key
    max_array_size_mb : float
        Maximum size of individual arrays to pool (MB)
    
    Examples
    --------
    >>> pool = ArrayPool(max_pool_size=100)
    >>> arr = pool.acquire((100, 100), np.float64)
    >>> # ... use array ...
    >>> pool.release(arr)
    """
    
    def __init__(
        self,
        max_pool_size: int = 100,
        max_array_size_mb: float = 100.0
    ):
        self.max_pool_size = max_pool_size
        self.max_array_bytes = int(max_array_size_mb * 1024 * 1024)
        self._pools: Dict[Tuple, List[NDArray]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stats = MemoryStats()
    
    def _make_key(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype
    ) -> Tuple[Tuple[int, ...], str]:
        """Create pool key from shape and dtype."""
        return (shape, str(dtype))
    
    def acquire(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        fill_value: Optional[float] = None
    ) -> NDArray:
        """
        Acquire an array from the pool or allocate new.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Array shape
        dtype : np.dtype
            Array data type
        fill_value : float, optional
            Value to fill array (None for uninitialized)
            
        Returns
        -------
        NDArray
            Array from pool or newly allocated
        """
        key = self._make_key(shape, np.dtype(dtype))
        
        with self._lock:
            pool = self._pools[key]
            
            if pool:
                arr = pool.pop()
                self._stats.cache_hits += 1
                self._stats.pooled_arrays -= 1
            else:
                arr = np.empty(shape, dtype=dtype)
                self._stats.cache_misses += 1
                self._stats.allocated_arrays += 1
                self._stats.current_bytes += arr.nbytes
                self._stats.peak_bytes = max(
                    self._stats.peak_bytes,
                    self._stats.current_bytes
                )
        
        if fill_value is not None:
            arr.fill(fill_value)
        
        return arr
    
    def release(self, arr: NDArray) -> bool:
        """
        Release an array back to the pool.
        
        Parameters
        ----------
        arr : NDArray
            Array to release
            
        Returns
        -------
        bool
            True if array was pooled, False if discarded
        """
        # Don't pool large arrays
        if arr.nbytes > self.max_array_bytes:
            return False
        
        key = self._make_key(arr.shape, arr.dtype)
        
        with self._lock:
            pool = self._pools[key]
            
            if len(pool) < self.max_pool_size:
                pool.append(arr)
                self._stats.pooled_arrays += 1
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all pooled arrays."""
        with self._lock:
            total_freed = 0
            for pool in self._pools.values():
                for arr in pool:
                    total_freed += arr.nbytes
                pool.clear()
            self._stats.pooled_arrays = 0
            self._stats.current_bytes -= total_freed
    
    @property
    def stats(self) -> MemoryStats:
        """Get pool statistics."""
        return self._stats
    
    def get_pool_info(self) -> Dict[str, Any]:
        """Get detailed pool information."""
        with self._lock:
            return {
                'pools': {
                    str(k): len(v) for k, v in self._pools.items()
                },
                'stats': self._stats.to_dict(),
                'theoretical_reference': 'docs/ROADMAP.md §3.3',
            }


class SparseFieldArray:
    """
    Memory-efficient sparse array for cGFT field configurations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1 - cGFT field φ(g₁,g₂,g₃,g₄)
        docs/ROADMAP.md §3.3 - Sparse matrix support
    
    For large lattice sizes (N >> 10), most field values may be near zero.
    This class stores only non-zero values, dramatically reducing memory
    for sparse configurations.
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        Full array shape
    dtype : np.dtype
        Data type
    sparsity_threshold : float
        Values below this threshold are considered zero
    
    Examples
    --------
    >>> sparse = SparseFieldArray((100, 100, 100, 100), np.complex128)
    >>> sparse[10, 20, 30, 40] = 1.0 + 2.0j
    >>> dense = sparse.to_dense()
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.complex128,
        sparsity_threshold: float = 1e-15
    ):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.sparsity_threshold = sparsity_threshold
        self._data: Dict[Tuple[int, ...], Any] = {}
        self._theoretical_reference = "IRH v21.1 §1.1, docs/ROADMAP.md §3.3"
    
    def __getitem__(self, index: Tuple[int, ...]) -> Any:
        """Get value at index."""
        return self._data.get(index, self.dtype.type(0))
    
    def __setitem__(self, index: Tuple[int, ...], value: Any) -> None:
        """Set value at index."""
        if abs(value) > self.sparsity_threshold:
            self._data[index] = self.dtype.type(value)
        elif index in self._data:
            del self._data[index]
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self._data)
    
    @property
    def density(self) -> float:
        """Fraction of non-zero elements."""
        total = np.prod(self.shape)
        return self.nnz / total if total > 0 else 0.0
    
    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Each entry: key (tuple of ints) + value
        key_size = len(self.shape) * 8  # int64 per dimension
        value_size = self.dtype.itemsize
        overhead = 64  # Dict entry overhead
        return self.nnz * (key_size + value_size + overhead)
    
    @property
    def dense_memory_bytes(self) -> int:
        """Memory if stored as dense array."""
        return int(np.prod(self.shape) * self.dtype.itemsize)
    
    @property
    def compression_ratio(self) -> float:
        """Memory savings ratio."""
        dense = self.dense_memory_bytes
        return dense / self.memory_bytes if self.memory_bytes > 0 else float('inf')
    
    def to_dense(self) -> NDArray:
        """Convert to dense numpy array."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        for idx, val in self._data.items():
            arr[idx] = val
        return arr
    
    @classmethod
    def from_dense(
        cls,
        arr: NDArray,
        sparsity_threshold: float = 1e-15
    ) -> 'SparseFieldArray':
        """Create from dense array."""
        sparse = cls(arr.shape, arr.dtype, sparsity_threshold)
        
        # Find non-zero indices
        if np.issubdtype(arr.dtype, np.complexfloating):
            mask = np.abs(arr) > sparsity_threshold
        else:
            mask = np.abs(arr) > sparsity_threshold
        
        indices = np.argwhere(mask)
        for idx in indices:
            sparse._data[tuple(idx)] = arr[tuple(idx)]
        
        return sparse
    
    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get array information."""
        return {
            'shape': self.shape,
            'dtype': str(self.dtype),
            'nnz': self.nnz,
            'density': self.density,
            'memory_bytes': self.memory_bytes,
            'dense_memory_bytes': self.dense_memory_bytes,
            'compression_ratio': self.compression_ratio,
            'theoretical_reference': self._theoretical_reference,
        }


class MemoryMonitor:
    """
    Memory usage monitor for tracking allocations.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory monitoring
    
    Parameters
    ----------
    track_gc : bool
        Track garbage collection events
    
    Examples
    --------
    >>> monitor = MemoryMonitor()
    >>> monitor.start()
    >>> # ... computations ...
    >>> stats = monitor.get_stats()
    >>> monitor.stop()
    """
    
    def __init__(self, track_gc: bool = True):
        self.track_gc = track_gc
        self._stats = MemoryStats()
        self._snapshots: List[Dict[str, Any]] = []
        self._running = False
        self._gc_callback_id: Optional[int] = None
    
    def start(self) -> None:
        """Start monitoring."""
        self._running = True
        self._stats = MemoryStats()
        
        if self.track_gc:
            gc.callbacks.append(self._gc_callback)
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        
        if self.track_gc and self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
    
    def _gc_callback(self, phase: str, info: Dict) -> None:
        """Callback for garbage collection events."""
        if phase == 'stop':
            self._stats.gc_collections += 1
    
    def snapshot(self, label: str = '') -> Dict[str, Any]:
        """Take memory snapshot."""
        import tracemalloc
        
        snapshot_data = {
            'label': label,
            'timestamp': __import__('time').time(),
            'gc_collections': self._stats.gc_collections,
        }
        
        # Try to get tracemalloc stats if available
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot_data['current_bytes'] = current
            snapshot_data['peak_bytes'] = peak
        
        self._snapshots.append(snapshot_data)
        return snapshot_data
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._stats
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all snapshots."""
        return list(self._snapshots)
    
    def clear_snapshots(self) -> None:
        """Clear snapshot history."""
        self._snapshots.clear()
    
    def get_report(self) -> Dict[str, Any]:
        """Generate memory report."""
        return {
            'stats': self._stats.to_dict(),
            'snapshots': self._snapshots,
            'running': self._running,
            'theoretical_reference': 'docs/ROADMAP.md §3.3',
        }


class MemoryOptimizer:
    """
    Unified memory optimization manager.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.6 - Computational efficiency
        docs/ROADMAP.md §3.3 - Memory optimization
    
    Parameters
    ----------
    enable_pooling : bool
        Enable array pooling
    enable_gc_optimization : bool
        Enable GC optimization
    pool_size : int
        Array pool size
    
    Examples
    --------
    >>> optimizer = MemoryOptimizer()
    >>> with optimizer.optimized_context():
    ...     result = compute_rg_flow(...)
    """
    
    def __init__(
        self,
        enable_pooling: bool = True,
        enable_gc_optimization: bool = True,
        pool_size: int = 100
    ):
        self.enable_pooling = enable_pooling
        self.enable_gc_optimization = enable_gc_optimization
        self._pool = ArrayPool(max_pool_size=pool_size) if enable_pooling else None
        self._monitor = MemoryMonitor()
        self._original_gc_settings: Optional[Tuple] = None
    
    def acquire_array(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        fill_value: Optional[float] = None
    ) -> NDArray:
        """Acquire array from pool or allocate."""
        if self._pool is not None:
            return self._pool.acquire(shape, dtype, fill_value)
        
        arr = np.empty(shape, dtype=dtype)
        if fill_value is not None:
            arr.fill(fill_value)
        return arr
    
    def release_array(self, arr: NDArray) -> None:
        """Release array back to pool."""
        if self._pool is not None:
            self._pool.release(arr)
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        self._monitor.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitor.stop()
    
    def optimize_gc(self) -> None:
        """Apply GC optimizations."""
        if self.enable_gc_optimization:
            self._original_gc_settings = (
                gc.get_threshold(),
                gc.isenabled()
            )
            # Increase thresholds for less frequent collection
            gc.set_threshold(50000, 500, 100)
    
    def restore_gc(self) -> None:
        """Restore original GC settings."""
        if self._original_gc_settings is not None:
            threshold, enabled = self._original_gc_settings
            gc.set_threshold(*threshold)
            if enabled:
                gc.enable()
            else:
                gc.disable()
            self._original_gc_settings = None
    
    def __enter__(self) -> 'MemoryOptimizer':
        """Enter optimized context."""
        self.start_monitoring()
        self.optimize_gc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit optimized context."""
        self.stop_monitoring()
        self.restore_gc()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {
            'monitor': self._monitor.get_report(),
            'theoretical_reference': 'IRH v21.1 §1.6, docs/ROADMAP.md §3.3',
        }
        if self._pool is not None:
            stats['pool'] = self._pool.get_pool_info()
        return stats


def memory_efficient(
    pool_name: str = 'default',
    gc_optimize: bool = False
):
    """
    Decorator for memory-efficient function execution.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory optimization
    
    Parameters
    ----------
    pool_name : str
        Name for array pool
    gc_optimize : bool
        Apply GC optimizations during execution
    
    Examples
    --------
    >>> @memory_efficient('rg_flow')
    ... def compute_rg_trajectory(initial, n_steps):
    ...     return integrate_trajectory(initial, n_steps)
    """
    def decorator(func: Callable) -> Callable:
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = MemoryOptimizer(
                enable_gc_optimization=gc_optimize
            )
            
            with optimizer:
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def get_memory_stats() -> Dict[str, Any]:
    """
    Get global memory statistics.
    
    Returns
    -------
    Dict[str, Any]
        Memory statistics including Python and NumPy usage
    """
    import sys
    
    stats = {
        'python_allocated_bytes': sys.getallocatedblocks() * 8,  # Approximate
        'gc_objects': len(gc.get_objects()),
        'gc_threshold': gc.get_threshold(),
        'gc_counts': gc.get_count(),
        'theoretical_reference': 'docs/ROADMAP.md §3.3',
    }
    
    # Try to get more detailed stats if tracemalloc available
    import tracemalloc
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        stats['tracemalloc_current'] = current
        stats['tracemalloc_peak'] = peak
    
    return stats


def optimize_gc(
    threshold: Tuple[int, int, int] = (50000, 500, 100),
    disable: bool = False
) -> None:
    """
    Configure garbage collection for optimal performance.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - GC optimization
    
    Parameters
    ----------
    threshold : Tuple[int, int, int]
        GC threshold for generations 0, 1, 2
    disable : bool
        Completely disable GC (use with caution)
    """
    if disable:
        gc.disable()
    else:
        gc.enable()
        gc.set_threshold(*threshold)


def create_memory_mapped_array(
    filename: str,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    mode: str = 'w+'
) -> np.memmap:
    """
    Create memory-mapped array for large datasets.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory-mapped arrays
    
    Memory-mapped arrays allow working with datasets larger than RAM
    by mapping file contents directly to virtual memory.
    
    Parameters
    ----------
    filename : str
        Path to memory-mapped file
    shape : Tuple[int, ...]
        Array shape
    dtype : np.dtype
        Data type
    mode : str
        File mode ('r', 'r+', 'w+', 'c')
    
    Returns
    -------
    np.memmap
        Memory-mapped array
    
    Examples
    --------
    >>> mmap_arr = create_memory_mapped_array('/tmp/large_field.dat', (1000, 1000, 1000, 1000))
    >>> mmap_arr[0, 0, 0, 0] = 1.0
    >>> mmap_arr.flush()
    """
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)


def estimate_memory_usage(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    sparse: bool = False,
    expected_density: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate memory usage for an array configuration.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory estimation
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        Array shape
    dtype : np.dtype
        Data type
    sparse : bool
        Whether to estimate sparse storage
    expected_density : float
        Expected fraction of non-zero elements (for sparse)
    
    Returns
    -------
    Dict[str, Any]
        Memory estimates
    
    Examples
    --------
    >>> estimate = estimate_memory_usage((100, 100, 100, 100), np.complex128)
    >>> print(f"Dense: {estimate['dense_gb']:.2f} GB")
    """
    dtype = np.dtype(dtype)
    total_elements = int(np.prod(shape))
    dense_bytes = total_elements * dtype.itemsize
    
    result = {
        'shape': shape,
        'dtype': str(dtype),
        'total_elements': total_elements,
        'dense_bytes': dense_bytes,
        'dense_mb': dense_bytes / (1024**2),
        'dense_gb': dense_bytes / (1024**3),
        'theoretical_reference': 'docs/ROADMAP.md §3.3',
    }
    
    if sparse:
        nnz = int(total_elements * expected_density)
        # Sparse storage: indices + values
        index_bytes = nnz * len(shape) * 8  # int64 indices
        value_bytes = nnz * dtype.itemsize
        sparse_bytes = index_bytes + value_bytes
        
        result['sparse'] = {
            'expected_nnz': nnz,
            'expected_density': expected_density,
            'sparse_bytes': sparse_bytes,
            'sparse_mb': sparse_bytes / (1024**2),
            'compression_ratio': dense_bytes / sparse_bytes if sparse_bytes > 0 else float('inf'),
        }
    
    return result
