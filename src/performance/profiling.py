"""
Performance Profiling Utilities for IRH Computations

THEORETICAL FOUNDATION: IRH21.md §1.6, docs/ROADMAP.md §3.7-3.8

This module provides comprehensive profiling capabilities:
    - Function execution timing with nanosecond precision
    - Memory usage profiling
    - Call graph analysis
    - Profile report generation
    - Bottleneck identification

The profiling tools help identify optimization opportunities while
maintaining the theoretical integrity of IRH computations.

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - docs/ROADMAP.md §3.7 (Performance Benchmarking Suite)
    - docs/ROADMAP.md §3.8 (Profiling & Bottleneck Analysis)
"""

from __future__ import annotations

import cProfile
import functools
import io
import pstats
import sys
import time
import tracemalloc
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

__all__ = [
    'Profiler',
    'profile',
    'time_function',
    'memory_profile',
    'get_profiling_stats',
    'ProfileReport',
    'create_profiler',
]

T = TypeVar('T')

# Global profiler registry
_profiler_registry: Dict[str, 'Profiler'] = {}
_registry_lock = threading.Lock()


@dataclass
class TimingResult:
    """
    Result of a timed function execution.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.7 - Performance metrics
    """
    function_name: str
    execution_time_ns: int
    execution_time_ms: float = field(init=False)
    execution_time_s: float = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.execution_time_ms = self.execution_time_ns / 1e6
        self.execution_time_s = self.execution_time_ns / 1e9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'execution_time_ns': self.execution_time_ns,
            'execution_time_ms': self.execution_time_ms,
            'execution_time_s': self.execution_time_s,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }


@dataclass
class MemoryResult:
    """
    Result of memory profiling.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.3 - Memory Optimization
    """
    function_name: str
    peak_memory_bytes: int
    peak_memory_mb: float = field(init=False)
    current_memory_bytes: int = 0
    allocation_count: int = 0
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.peak_memory_mb = self.peak_memory_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'peak_memory_bytes': self.peak_memory_bytes,
            'peak_memory_mb': self.peak_memory_mb,
            'current_memory_bytes': self.current_memory_bytes,
            'allocation_count': self.allocation_count,
            'top_allocations': self.top_allocations,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ProfileReport:
    """
    Comprehensive profiling report.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.8 - Profiling Tools
    """
    name: str
    timing_results: List[TimingResult] = field(default_factory=list)
    memory_results: List[MemoryResult] = field(default_factory=list)
    call_stats: Optional[pstats.Stats] = None
    created_at: datetime = field(default_factory=datetime.now)
    theoretical_reference: str = "docs/ROADMAP.md §3.7-3.8"
    
    # Theoretical Reference: IRH v21.4

    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        timing_stats = {}
        if self.timing_results:
            times = [t.execution_time_ms for t in self.timing_results]
            timing_stats = {
                'count': len(times),
                'total_ms': sum(times),
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'median_ms': np.median(times),
            }
        
        memory_stats = {}
        if self.memory_results:
            peaks = [m.peak_memory_mb for m in self.memory_results]
            memory_stats = {
                'count': len(peaks),
                'max_peak_mb': max(peaks),
                'mean_peak_mb': np.mean(peaks),
            }
        
        return {
            'name': self.name,
            'timing': timing_stats,
            'memory': memory_stats,
            'created_at': self.created_at.isoformat(),
            'theoretical_reference': self.theoretical_reference,
        }
    
    # Theoretical Reference: IRH v21.4

    
    def format_report(self, output: Optional[TextIO] = None) -> str:
        """
        Format report as human-readable string.
        
        Parameters
        ----------
        output : TextIO, optional
            Output stream (default: returns string)
            
        Returns
        -------
        str
            Formatted report
        """
        lines = [
            "=" * 60,
            f"Profile Report: {self.name}",
            f"Created: {self.created_at.isoformat()}",
            # f"Theoretical Reference: {self.theoretical_reference}",
            "=" * 60,
            "",
        ]
        
        # Timing section
        if self.timing_results:
            lines.append("TIMING RESULTS")
            lines.append("-" * 40)
            summary = self.get_summary()['timing']
            lines.append(f"  Total executions: {summary['count']}")
            lines.append(f"  Total time: {summary['total_ms']:.3f} ms")
            lines.append(f"  Mean time: {summary['mean_ms']:.3f} ms ± {summary['std_ms']:.3f}")
            lines.append(f"  Min/Max: {summary['min_ms']:.3f} / {summary['max_ms']:.3f} ms")
            lines.append(f"  Median: {summary['median_ms']:.3f} ms")
            lines.append("")
        
        # Memory section
        if self.memory_results:
            lines.append("MEMORY RESULTS")
            lines.append("-" * 40)
            summary = self.get_summary()['memory']
            lines.append(f"  Total profiles: {summary['count']}")
            lines.append(f"  Max peak: {summary['max_peak_mb']:.2f} MB")
            lines.append(f"  Mean peak: {summary['mean_peak_mb']:.2f} MB")
            lines.append("")
        
        # Call stats
        if self.call_stats:
            lines.append("CALL STATISTICS")
            lines.append("-" * 40)
            stream = io.StringIO()
            self.call_stats.stream = stream
            self.call_stats.print_stats(20)
            lines.append(stream.getvalue())
        
        report = "\n".join(lines)
        
        if output is not None:
            output.write(report)
        
        return report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'summary': self.get_summary(),
            'timing_results': [t.to_dict() for t in self.timing_results],
            'memory_results': [m.to_dict() for m in self.memory_results],
            'created_at': self.created_at.isoformat(),
        }


class Profiler:
    """
    Comprehensive profiler for IRH computations.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.7-3.8 - Performance Analysis
        
    Parameters
    ----------
    name : str
        Profiler name for identification
    enable_timing : bool
        Enable timing profiling (default: True)
    enable_memory : bool
        Enable memory profiling (default: True)
    enable_call_graph : bool
        Enable call graph profiling (default: False)
        
    Examples
    --------
    >>> profiler = Profiler('rg_flow')
    >>> with profiler.profile_timing('compute_betas'):
    ...     result = compute_beta_functions(couplings)
    >>> report = profiler.generate_report()
    """
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(
        self,
        name: str,
        enable_timing: bool = True,
        enable_memory: bool = True,
        enable_call_graph: bool = False
    ):
        self.name = name
        self.enable_timing = enable_timing
        self.enable_memory = enable_memory
        self.enable_call_graph = enable_call_graph
        
        self._timing_results: List[TimingResult] = []
        self._memory_results: List[MemoryResult] = []
        self._call_profiler: Optional[cProfile.Profile] = None
        self._lock = threading.Lock()
        
        if enable_call_graph:
            self._call_profiler = cProfile.Profile()
    
    @contextmanager
    # Theoretical Reference: IRH v21.4

    def profile_timing(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for timing a code block.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being timed
        metadata : dict, optional
            Additional metadata to store
            
        Yields
        ------
        None
        """
        if not self.enable_timing:
            yield
            return
        
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            result = TimingResult(
                function_name=operation_name,
                execution_time_ns=elapsed,
                metadata=metadata or {}
            )
            with self._lock:
                self._timing_results.append(result)
    
    @contextmanager
    # Theoretical Reference: IRH v21.4

    def profile_memory(
        self,
        operation_name: str,
        trace_depth: int = 10
    ):
        """
        Context manager for memory profiling.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being profiled
        trace_depth : int
            Stack trace depth for allocation tracking
            
        Yields
        ------
        None
        """
        if not self.enable_memory:
            yield
            return
        
        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            # Get top allocations
            top_stats = snapshot.statistics('lineno')[:trace_depth]
            top_allocations = [
                {
                    'file': str(stat.traceback),
                    'size_bytes': stat.size,
                    'count': stat.count,
                }
                for stat in top_stats
            ]
            
            result = MemoryResult(
                function_name=operation_name,
                peak_memory_bytes=peak,
                current_memory_bytes=current,
                allocation_count=len(top_stats),
                top_allocations=top_allocations,
            )
            with self._lock:
                self._memory_results.append(result)
    
    @contextmanager
    # Theoretical Reference: IRH v21.4

    def profile_calls(self):
        """
        Context manager for call graph profiling.
        
        Yields
        ------
        None
        """
        if not self.enable_call_graph or self._call_profiler is None:
            yield
            return
        
        self._call_profiler.enable()
        try:
            yield
        finally:
            self._call_profiler.disable()
    
    # Theoretical Reference: IRH v21.4

    
    def generate_report(self) -> ProfileReport:
        """
        Generate comprehensive profiling report.
        
        Returns
        -------
        ProfileReport
            Complete profiling report
        """
        call_stats = None
        if self._call_profiler is not None:
            stream = io.StringIO()
            stats = pstats.Stats(self._call_profiler, stream=stream)
            stats.sort_stats('cumulative')
            call_stats = stats
        
        return ProfileReport(
            name=self.name,
            timing_results=list(self._timing_results),
            memory_results=list(self._memory_results),
            call_stats=call_stats,
        )
    
    # Theoretical Reference: IRH v21.4

    
    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._timing_results.clear()
            self._memory_results.clear()
            if self._call_profiler is not None:
                self._call_profiler = cProfile.Profile()
    
    # Theoretical Reference: IRH v21.4 (Performance Infrastructure)
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.generate_report().get_summary()


# Theoretical Reference: IRH v21.4



def create_profiler(
    name: str,
    enable_timing: bool = True,
    enable_memory: bool = True,
    enable_call_graph: bool = False
) -> Profiler:
    
    # Theoretical Reference: IRH v21.4
    """
    Create and register a profiler.
    
    Parameters
    ----------
    name : str
        Profiler name
    enable_timing : bool
        Enable timing profiling
    enable_memory : bool
        Enable memory profiling
    enable_call_graph : bool
        Enable call graph profiling
        
    Returns
    -------
    Profiler
        Registered profiler
    """
    with _registry_lock:
        if name in _profiler_registry:
            return _profiler_registry[name]
        profiler = Profiler(
            name, enable_timing, enable_memory, enable_call_graph
        )
        _profiler_registry[name] = profiler
        return profiler


# Theoretical Reference: IRH v21.4



def get_profiling_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics from all registered profilers.
    
    Returns
    -------
    Dict[str, Dict]
        Statistics keyed by profiler name
    """
    with _registry_lock:
        return {
            name: profiler.get_stats()
            for name, profiler in _profiler_registry.items()
        }


def profile(
    profiler_name: str = 'default',
    enable_timing: bool = True,
    enable_memory: bool = False
):
    """
    Decorator for profiling function execution.
    
    # Theoretical Reference:
        docs/ROADMAP.md §3.7 - Performance Benchmarking
        
    Parameters
    ----------
    profiler_name : str
        Name of profiler to use
    enable_timing : bool
        Enable timing profiling
    enable_memory : bool
        Enable memory profiling
        
    Examples
    --------
    >>> @profile('rg_flow')
    ... def compute_betas(couplings):
    ...     return vectorized_beta_functions(couplings)
    """
    # Theoretical Reference: IRH v21.4

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            profiler = create_profiler(
                profiler_name,
                enable_timing=enable_timing,
                enable_memory=enable_memory
            )
            
            with profiler.profile_timing(func.__name__):
                if enable_memory:
                    with profiler.profile_memory(func.__name__):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Theoretical Reference: IRH v21.4



def time_function(func: Callable[..., T]) -> Callable[..., Tuple[T, TimingResult]]:
    """
    Decorator that returns both result and timing information.
    
    Parameters
    ----------
    func : Callable
        Function to time
        
    Returns
    -------
    Callable
        Wrapped function returning (result, TimingResult)
        
    Examples
    --------
    >>> @time_function
    ... def slow_computation(n):
    ...     return sum(range(n))
    >>> result, timing = slow_computation(1000000)
    >>> print(f"Took {timing.execution_time_ms:.2f} ms")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[T, TimingResult]:
        """
        # Theoretical Reference: IRH v21.4
        """
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        
        timing = TimingResult(
            function_name=func.__name__,
            execution_time_ns=elapsed,
            metadata={'args_count': len(args), 'kwargs_count': len(kwargs)}
        )
        
        return result, timing
    
    return wrapper


# Theoretical Reference: IRH v21.4



def memory_profile(func: Callable[..., T]) -> Callable[..., Tuple[T, MemoryResult]]:
    """
    Decorator that returns both result and memory information.
    
    Parameters
    ----------
    func : Callable
        Function to profile
        
    Returns
    -------
    Callable
        Wrapped function returning (result, MemoryResult)
        
    Examples
    --------
    >>> @memory_profile
    ... def allocate_arrays():
    ...     return np.zeros((1000, 1000))
    >>> result, memory = allocate_arrays()
    >>> print(f"Peak memory: {memory.peak_memory_mb:.2f} MB")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[T, MemoryResult]:
        """
        # Theoretical Reference: IRH v21.4
        """
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory = MemoryResult(
            function_name=func.__name__,
            peak_memory_bytes=peak,
            current_memory_bytes=current,
        )
        
        return result, memory
    
    return wrapper
