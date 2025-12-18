"""
IRH v21.0 Performance Optimization System

THEORETICAL FOUNDATION: IRH21.md §1.6, docs/ROADMAP.md §3 (Tier 3)

This module provides comprehensive performance optimization capabilities:
    - Caching and memoization for expensive computations
    - Vectorized numerical routines for large-scale operations
    - Performance profiling and benchmarking utilities
    - Memory optimization tools (array pooling, sparse arrays, GC tuning)

The optimization layer maintains theoretical fidelity while achieving
significant speedups for exascale-ready computations.

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)

Implementation Timeline:
    Phase 3.1: NumPy Vectorization (Q1 2026) ✅
    Phase 3.2: Caching & Memoization (Q1 2026) ✅
    Phase 3.3: Memory Optimization (Q1 2026) ✅
    Phase 3.4: MPI Parallelization (Q2 2026)
    Phase 3.5: GPU Acceleration (Q3 2026)
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.6, docs/ROADMAP.md §3"

from .cache_manager import (
    CacheManager,
    LRUCache,
    DiskCache,
    create_cache,
    get_cache,
    cached,
    clear_all_caches,
    get_cache_stats,
)

from .numerical_opts import (
    vectorized_beta_functions,
    vectorized_qncd_distance,
    optimized_matrix_operations,
    batch_quaternion_multiply,
    parallel_fixed_point_search,
    VectorizedOperations,
)

from .profiling import (
    Profiler,
    profile,
    time_function,
    memory_profile,
    get_profiling_stats,
    ProfileReport,
    create_profiler,
)

from .memory_optimization import (
    ArrayPool,
    SparseFieldArray,
    MemoryMonitor,
    MemoryOptimizer,
    memory_efficient,
    get_memory_stats,
    optimize_gc,
    create_memory_mapped_array,
    estimate_memory_usage,
)

__all__ = [
    # Cache Management
    'CacheManager',
    'LRUCache',
    'DiskCache',
    'create_cache',
    'get_cache',
    'cached',
    'clear_all_caches',
    'get_cache_stats',
    
    # Vectorized Operations
    'vectorized_beta_functions',
    'vectorized_qncd_distance',
    'optimized_matrix_operations',
    'batch_quaternion_multiply',
    'parallel_fixed_point_search',
    'VectorizedOperations',
    
    # Profiling
    'Profiler',
    'profile',
    'time_function',
    'memory_profile',
    'get_profiling_stats',
    'ProfileReport',
    'create_profiler',
    
    # Memory Optimization
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
