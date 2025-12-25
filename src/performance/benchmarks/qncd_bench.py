"""
QNCD Computation Benchmark Suite

THEORETICAL FOUNDATION: IRH21.md Appendix A, docs/ROADMAP.md §3.7

Benchmarks for Quantum Normalized Compression Distance:
    - Single pair QNCD computation
    - Batch QNCD computation
    - Method comparison benchmarks

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'benchmark_qncd_single',
    'benchmark_qncd_batch',
    'benchmark_qncd_methods',
    'QNCDBenchmarkSuite',
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_time_s': self.total_time_s,
            'mean_time_ms': self.mean_time_ms,
            'std_time_ms': self.std_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'throughput': self.throughput,
            'metadata': self.metadata,
        }


def _run_benchmark(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10
) -> BenchmarkResult:
    """Run a benchmark function multiple times."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Timed runs
    times_ns = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        times_ns.append(elapsed)
    
    times_ms = np.array(times_ns) / 1e6
    total_s = sum(times_ns) / 1e9
    
    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else 'anonymous',
        iterations=iterations,
        total_time_s=total_s,
        mean_time_ms=float(np.mean(times_ms)),
        std_time_ms=float(np.std(times_ms)),
        min_time_ms=float(np.min(times_ms)),
        max_time_ms=float(np.max(times_ms)),
        throughput=iterations / total_s if total_s > 0 else 0,
    )


def benchmark_qncd_single(
    vector_sizes: List[int] = [10, 100, 1000, 10000],
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark single-pair QNCD computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A, Eq. A.1
        
    Parameters
    ----------
    vector_sizes : List[int]
        Vector dimensions to benchmark
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by vector size
    """
    from ..numerical_opts import vectorized_qncd_distance
    
    results = {}
    
    for size in vector_sizes:
        # Generate test vectors
        v1 = np.random.rand(1, size)
        v2 = np.random.rand(1, size)
        
        def bench_fn():
            """
            # Theoretical Reference: IRH v21.4
            """
            return vectorized_qncd_distance(v1, v2)
        
        bench_fn.__name__ = f'qncd_single_dim_{size}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'vector_dimension': size,
            'theoretical_reference': 'IRH v21.1 Appendix A, Eq. A.1',
        }
        results[f'dim_{size}'] = result
    
    return results


def benchmark_qncd_batch(
    batch_sizes: List[int] = [1, 10, 100, 1000],
    vector_dim: int = 100,
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark batch QNCD computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A
        
    Parameters
    ----------
    batch_sizes : List[int]
        Batch sizes to benchmark
    vector_dim : int
        Vector dimension
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by batch size
    """
    from ..numerical_opts import vectorized_qncd_distance
    
    results = {}
    
    for batch_size in batch_sizes:
        # Generate batch of vector pairs
        v1 = np.random.rand(batch_size, vector_dim)
        v2 = np.random.rand(batch_size, vector_dim)
        
        def bench_fn():
            """
            # Theoretical Reference: IRH v21.4
            """
            return vectorized_qncd_distance(v1, v2)
        
        bench_fn.__name__ = f'qncd_batch_{batch_size}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'batch_size': batch_size,
            'vector_dimension': vector_dim,
            'throughput_per_pair': result.throughput * batch_size,
            'theoretical_reference': 'IRH v21.1 Appendix A',
        }
        results[f'batch_{batch_size}'] = result
    
    return results


def benchmark_qncd_methods(
    methods: List[str] = ['compression_proxy', 'entropy', 'complexity'],
    batch_size: int = 100,
    vector_dim: int = 100,
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark different QNCD computation methods.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A.4 - Method comparison
        
    Parameters
    ----------
    methods : List[str]
        QNCD methods to benchmark
    batch_size : int
        Batch size for benchmarks
    vector_dim : int
        Vector dimension
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by method name
    """
    from ..numerical_opts import vectorized_qncd_distance
    
    results = {}
    
    # Generate test data
    v1 = np.random.rand(batch_size, vector_dim)
    v2 = np.random.rand(batch_size, vector_dim)
    
    for method in methods:
        # Theoretical Reference: IRH v21.4

        def make_bench_fn(m):
            def bench_fn():
                return vectorized_qncd_distance(v1, v2, method=m)
            return bench_fn
        
        bench_fn = make_bench_fn(method)
        bench_fn.__name__ = f'qncd_{method}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'method': method,
            'batch_size': batch_size,
            'vector_dimension': vector_dim,
            'theoretical_reference': 'IRH v21.1 Appendix A.4',
        }
        results[method] = result
    
    return results


@dataclass
class QNCDBenchmarkSuite:
    """
    Complete benchmark suite for QNCD computations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A, docs/ROADMAP.md §3.7
        
    Examples
    --------
    >>> suite = QNCDBenchmarkSuite()
    >>> results = suite.run_all()
    >>> suite.print_report(results)
    """
    iterations: int = 100
    warmup: int = 10
    
    # Theoretical Reference: IRH v21.4

    
    def run_all(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all QNCD benchmarks."""
        return {
            'single_pair': benchmark_qncd_single(
                iterations=self.iterations
            ),
            'batch': benchmark_qncd_batch(
                iterations=self.iterations
            ),
            'methods': benchmark_qncd_methods(
                iterations=self.iterations
            ),
        }
    
    # Theoretical Reference: IRH v21.4 (Performance Infrastructure)
    def print_report(self, results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
        """Print formatted benchmark report."""
        print("=" * 70)
        print("QNCD BENCHMARK REPORT")
        # print("Theoretical Reference: IRH v21.1 Appendix A, docs/ROADMAP.md §3.7")
        print("=" * 70)
        
        for category, benchmarks in results.items():
            print(f"\n{category.upper()}")
            print("-" * 50)
            for name, result in benchmarks.items():
                print(f"  {name}:")
                print(f"    Mean: {result.mean_time_ms:.3f} ms ± {result.std_time_ms:.3f}")
                print(f"    Throughput: {result.throughput:.1f} ops/s")
                if 'throughput_per_pair' in result.metadata:
                    print(f"    Pairs/s: {result.metadata['throughput_per_pair']:.1f}")
    
    # Theoretical Reference: IRH v21.4 (Performance Infrastructure)
    def get_summary(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]]
    ) -> Dict[str, Any]:
        
        # Theoretical Reference: IRH v21.4
        
        # Theoretical Reference: IRH v21.4
        """Get summary statistics."""
        summary = {
            'theoretical_reference': 'IRH v21.1 Appendix A, docs/ROADMAP.md §3.7',
            'categories': {}
        }
        
        for category, benchmarks in results.items():
            category_summary = {}
            for name, result in benchmarks.items():
                category_summary[name] = {
                    'mean_ms': result.mean_time_ms,
                    'throughput': result.throughput,
                }
            summary['categories'][category] = category_summary
        
        return summary
