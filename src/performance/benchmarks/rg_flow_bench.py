"""
RG Flow Benchmark Suite

THEORETICAL FOUNDATION: IRH21.md §1.2, docs/ROADMAP.md §3.7

Benchmarks for RG flow computations:
    - Beta function evaluation
    - Fixed point search algorithms
    - RG trajectory integration

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'benchmark_beta_functions',
    'benchmark_fixed_point_search',
    'benchmark_rg_trajectory',
    'RGFlowBenchmarkSuite',
]

# Physical constants (Eq. 1.14)
LAMBDA_STAR = 48 * np.pi**2 / 9
GAMMA_STAR = 32 * np.pi**2 / 3
MU_STAR = 16 * np.pi**2


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
    throughput: float  # operations per second
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
    """
    Run a benchmark function multiple times.
    
    Parameters
    ----------
    func : Callable
        Function to benchmark (no arguments)
    iterations : int
        Number of timed iterations
    warmup : int
        Number of warmup iterations (not timed)
        
    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
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


def benchmark_beta_functions(
    batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark beta function computation at various batch sizes.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2, Eq. 1.13
        
    Parameters
    ----------
    batch_sizes : List[int]
        Batch sizes to benchmark
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by batch size
    """
    from ..numerical_opts import vectorized_beta_functions
    
    results = {}
    
    for batch_size in batch_sizes:
        # Generate random couplings
        couplings = np.random.rand(batch_size, 3) * 100
        
        # Theoretical Reference: IRH v21.4

        
        def bench_fn():
            return vectorized_beta_functions(couplings)
        
        bench_fn.__name__ = f'beta_functions_batch_{batch_size}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'batch_size': batch_size,
            'throughput_per_item': result.throughput * batch_size,
            'theoretical_reference': 'IRH v21.1 §1.2.2, Eq. 1.13',
        }
        results[f'batch_{batch_size}'] = result
    
    return results


def benchmark_fixed_point_search(
    n_points: List[int] = [1, 10, 100],
    iterations: int = 50
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark fixed point search algorithms.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.3, Eq. 1.14
        
    Parameters
    ----------
    n_points : List[int]
        Number of initial points to search from
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by point count
    """
    from ..numerical_opts import parallel_fixed_point_search
    
    results = {}
    
    for n in n_points:
        # Generate initial guesses near fixed point
        initial = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR]) + \
                  np.random.randn(n, 3) * 10
        
        def bench_fn():
            """
            # Theoretical Reference: IRH v21.4
            """
            return parallel_fixed_point_search(initial, max_iter=100)
        
        bench_fn.__name__ = f'fixed_point_search_{n}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'n_initial_points': n,
            'theoretical_reference': 'IRH v21.1 §1.2.3, Eq. 1.14',
        }
        results[f'points_{n}'] = result
    
    return results


def benchmark_rg_trajectory(
    n_steps_list: List[int] = [100, 1000, 10000],
    iterations: int = 50
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark RG trajectory integration.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2 - RG flow
        
    Parameters
    ----------
    n_steps_list : List[int]
        Number of integration steps
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by step count
    """
    from ..numerical_opts import vectorized_beta_functions
    
    results = {}
    
    for n_steps in n_steps_list:
        # Initial conditions
        x0 = np.array([[10.0, 20.0, 30.0]])
        dt = 0.01
        
        # Theoretical Reference: IRH v21.4

        
        def integrate_trajectory():
            """Simple Euler integration for benchmark."""
            x = x0.copy()
            trajectory = [x.copy()]
            for _ in range(n_steps):
                beta = vectorized_beta_functions(x)
                x = x + dt * beta
                trajectory.append(x.copy())
            return np.array(trajectory)
        
        integrate_trajectory.__name__ = f'rg_trajectory_{n_steps}'
        result = _run_benchmark(integrate_trajectory, iterations)
        result.metadata = {
            'n_steps': n_steps,
            'dt': dt,
            'steps_per_second': n_steps * result.throughput,
            'theoretical_reference': 'IRH v21.1 §1.2',
        }
        results[f'steps_{n_steps}'] = result
    
    return results


@dataclass
class RGFlowBenchmarkSuite:
    """
    Complete benchmark suite for RG flow computations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2, docs/ROADMAP.md §3.7
        
    Examples
    --------
    >>> suite = RGFlowBenchmarkSuite()
    >>> results = suite.run_all()
    >>> suite.print_report(results)
    """
    iterations: int = 100
    warmup: int = 10
    
    # Theoretical Reference: IRH v21.4

    
    def run_all(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all RG flow benchmarks."""
        return {
            'beta_functions': benchmark_beta_functions(
                iterations=self.iterations
            ),
            'fixed_point_search': benchmark_fixed_point_search(
                iterations=self.iterations // 2
            ),
            'rg_trajectory': benchmark_rg_trajectory(
                iterations=self.iterations // 2
            ),
        }
    
    # Theoretical Reference: IRH v21.4 (Performance Infrastructure)
    def print_report(self, results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
        """Print formatted benchmark report."""
        print("=" * 70)
        print("RG FLOW BENCHMARK REPORT")
        # print("Theoretical Reference: IRH v21.1 §1.2, docs/ROADMAP.md §3.7")
        print("=" * 70)
        
        for category, benchmarks in results.items():
            print(f"\n{category.upper()}")
            print("-" * 50)
            for name, result in benchmarks.items():
                print(f"  {name}:")
                print(f"    Mean: {result.mean_time_ms:.3f} ms ± {result.std_time_ms:.3f}")
                print(f"    Throughput: {result.throughput:.1f} ops/s")
                if 'throughput_per_item' in result.metadata:
                    print(f"    Items/s: {result.metadata['throughput_per_item']:.1f}")
    
    # Theoretical Reference: IRH v21.4 (Performance Infrastructure)
    def get_summary(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]]
    ) -> Dict[str, Any]:
        
        # Theoretical Reference: IRH v21.4
        
        # Theoretical Reference: IRH v21.4
        """Get summary statistics."""
        summary = {
            'theoretical_reference': 'IRH v21.1 §1.2, docs/ROADMAP.md §3.7',
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
