"""
cGFT Action Computation Benchmark Suite

THEORETICAL FOUNDATION: IRH21.md §1.1, Eq. 1.1-1.4, docs/ROADMAP.md §3.7

Benchmarks for cGFT action computations:
    - Kinetic action S_kin (Eq. 1.1)
    - Interaction action S_int (Eq. 1.2-1.3)
    - Total action evaluation

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
    'benchmark_kinetic_action',
    'benchmark_interaction_action',
    'benchmark_total_action',
    'ActionBenchmarkSuite',
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


def _compute_kinetic_action(
    phi: NDArray[np.complex128],
    laplacian_coeffs: NDArray[np.float64]
) -> np.complex128:
    """
    Compute kinetic action S_kin (Eq. 1.1).
    
    S_kin = ∫[∏dg_i] φ̄·[Σₐ Σᵢ Δₐ^(i)]·φ
    """
    # Simplified Laplacian application
    phi_bar = np.conj(phi)
    laplacian_phi = np.zeros_like(phi)
    
    # Apply discrete Laplacian along each axis
    for axis in range(len(phi.shape)):
        laplacian_phi += np.roll(phi, 1, axis=axis) - 2*phi + np.roll(phi, -1, axis=axis)
    
    # Scale by coefficients
    laplacian_phi *= np.mean(laplacian_coeffs)
    
    # Integrate (sum)
    return np.sum(phi_bar * laplacian_phi)


def _compute_interaction_action(
    phi: NDArray[np.complex128],
    lambda_coupling: float = LAMBDA_STAR,
    gamma_coupling: float = GAMMA_STAR
) -> np.complex128:
    """
    Compute interaction action S_int (Eq. 1.2-1.3).
    
    S_int = λ/4! ∫[∏dg_i] |φ|⁴ + γ/3! ∫ K[φ,φ,φ]
    """
    # Quartic term
    phi_abs_sq = np.abs(phi)**2
    quartic = (lambda_coupling / 24) * np.sum(phi_abs_sq**2)
    
    # Cubic term (simplified convolution)
    cubic = (gamma_coupling / 6) * np.sum(phi_abs_sq * np.abs(phi))
    
    return quartic + cubic


def _compute_total_action(
    phi: NDArray[np.complex128],
    laplacian_coeffs: NDArray[np.float64],
    lambda_c: float = LAMBDA_STAR,
    gamma_c: float = GAMMA_STAR,
    mu_c: float = MU_STAR
) -> Dict[str, Any]:
    """
    Compute total cGFT action S[φ,φ̄].
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1, Eq. 1.1-1.4
    """
    s_kin = _compute_kinetic_action(phi, laplacian_coeffs)
    s_int = _compute_interaction_action(phi, lambda_c, gamma_c)
    
    # Holographic term (Eq. 1.4)
    s_hol = mu_c * np.sum(np.abs(phi)**2)
    
    return {
        'S_kin': complex(s_kin),
        'S_int': complex(s_int),
        'S_hol': complex(s_hol),
        'S_total': complex(s_kin + s_int + s_hol),
    }


def benchmark_kinetic_action(
    field_sizes: List[int] = [4, 8, 16, 32],
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark kinetic action computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1, Eq. 1.1
        
    Parameters
    ----------
    field_sizes : List[int]
        Field lattice sizes (N for N⁴ lattice)
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by field size
    """
    results = {}
    
    for N in field_sizes:
        # Generate random field on N⁴ lattice
        phi = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
        laplacian_coeffs = np.ones((3, 4))  # 3 SU(2) generators × 4 arguments
        
        def bench_fn():
            return _compute_kinetic_action(phi, laplacian_coeffs)
        
        bench_fn.__name__ = f'kinetic_N{N}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'field_size': N,
            'lattice_points': N**4,
            'theoretical_reference': 'IRH v21.1 §1.1, Eq. 1.1',
        }
        results[f'N{N}'] = result
    
    return results


def benchmark_interaction_action(
    field_sizes: List[int] = [4, 8, 16, 32],
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark interaction action computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1, Eq. 1.2-1.3
        
    Parameters
    ----------
    field_sizes : List[int]
        Field lattice sizes
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by field size
    """
    results = {}
    
    for N in field_sizes:
        phi = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
        
        def bench_fn():
            return _compute_interaction_action(phi)
        
        bench_fn.__name__ = f'interaction_N{N}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'field_size': N,
            'lattice_points': N**4,
            'theoretical_reference': 'IRH v21.1 §1.1, Eq. 1.2-1.3',
        }
        results[f'N{N}'] = result
    
    return results


def benchmark_total_action(
    field_sizes: List[int] = [4, 8, 16, 32],
    iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark total action computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1, Eq. 1.1-1.4
        
    Parameters
    ----------
    field_sizes : List[int]
        Field lattice sizes
    iterations : int
        Iterations per benchmark
        
    Returns
    -------
    Dict[str, BenchmarkResult]
        Results keyed by field size
    """
    results = {}
    
    for N in field_sizes:
        phi = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
        laplacian_coeffs = np.ones((3, 4))
        
        def bench_fn():
            return _compute_total_action(phi, laplacian_coeffs)
        
        bench_fn.__name__ = f'total_N{N}'
        result = _run_benchmark(bench_fn, iterations)
        result.metadata = {
            'field_size': N,
            'lattice_points': N**4,
            'points_per_ms': N**4 / result.mean_time_ms if result.mean_time_ms > 0 else 0,
            'theoretical_reference': 'IRH v21.1 §1.1, Eq. 1.1-1.4',
        }
        results[f'N{N}'] = result
    
    return results


@dataclass
class ActionBenchmarkSuite:
    """
    Complete benchmark suite for cGFT action computations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1, docs/ROADMAP.md §3.7
        
    Examples
    --------
    >>> suite = ActionBenchmarkSuite()
    >>> results = suite.run_all()
    >>> suite.print_report(results)
    """
    iterations: int = 100
    warmup: int = 10
    field_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    
    def run_all(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all action benchmarks."""
        return {
            'kinetic': benchmark_kinetic_action(
                field_sizes=self.field_sizes,
                iterations=self.iterations
            ),
            'interaction': benchmark_interaction_action(
                field_sizes=self.field_sizes,
                iterations=self.iterations
            ),
            'total': benchmark_total_action(
                field_sizes=self.field_sizes,
                iterations=self.iterations
            ),
        }
    
    def print_report(self, results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
        """Print formatted benchmark report."""
        print("=" * 70)
        print("cGFT ACTION BENCHMARK REPORT")
        print("Theoretical Reference: IRH v21.1 §1.1, docs/ROADMAP.md §3.7")
        print("=" * 70)
        
        for category, benchmarks in results.items():
            print(f"\n{category.upper()} ACTION")
            print("-" * 50)
            for name, result in benchmarks.items():
                print(f"  {name} (N⁴ lattice):")
                print(f"    Mean: {result.mean_time_ms:.3f} ms ± {result.std_time_ms:.3f}")
                print(f"    Throughput: {result.throughput:.1f} ops/s")
                if 'points_per_ms' in result.metadata:
                    print(f"    Points/ms: {result.metadata['points_per_ms']:.1f}")
    
    def get_summary(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]]
    ) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            'theoretical_reference': 'IRH v21.1 §1.1, docs/ROADMAP.md §3.7',
            'categories': {}
        }
        
        for category, benchmarks in results.items():
            category_summary = {}
            for name, result in benchmarks.items():
                category_summary[name] = {
                    'mean_ms': result.mean_time_ms,
                    'throughput': result.throughput,
                    'lattice_points': result.metadata.get('lattice_points', 0),
                }
            summary['categories'][category] = category_summary
        
        return summary
