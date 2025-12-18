"""
IRH v21.0 Performance Benchmarks

THEORETICAL FOUNDATION: docs/ROADMAP.md §3.7

This module provides benchmark suites for IRH computations:
    - RG flow integration benchmarks
    - QNCD computation benchmarks
    - cGFT action evaluation benchmarks
    - Quaternion algebra benchmarks

Benchmarks establish baseline performance metrics and track
optimization improvements across versions.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "docs/ROADMAP.md §3.7"

from .rg_flow_bench import (
    benchmark_beta_functions,
    benchmark_fixed_point_search,
    benchmark_rg_trajectory,
    RGFlowBenchmarkSuite,
)

from .qncd_bench import (
    benchmark_qncd_single,
    benchmark_qncd_batch,
    benchmark_qncd_methods,
    QNCDBenchmarkSuite,
)

from .action_bench import (
    benchmark_kinetic_action,
    benchmark_interaction_action,
    benchmark_total_action,
    ActionBenchmarkSuite,
)

__all__ = [
    # RG Flow Benchmarks
    'benchmark_beta_functions',
    'benchmark_fixed_point_search',
    'benchmark_rg_trajectory',
    'RGFlowBenchmarkSuite',
    
    # QNCD Benchmarks
    'benchmark_qncd_single',
    'benchmark_qncd_batch',
    'benchmark_qncd_methods',
    'QNCDBenchmarkSuite',
    
    # Action Benchmarks
    'benchmark_kinetic_action',
    'benchmark_interaction_action',
    'benchmark_total_action',
    'ActionBenchmarkSuite',
]
