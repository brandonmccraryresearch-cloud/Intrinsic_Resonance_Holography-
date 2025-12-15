"""
Cross-Cutting Computational Utilities for Intrinsic Resonance Holography v21.0

This module provides shared numerical infrastructure used across all layers
of the computational framework. These utilities are theory-agnostic and can
be used independently of the IRH-specific modules.

Modules:
    integration: Numerical quadrature on group manifolds
    optimization: Fixed-point solvers, minimizers
    special_functions: Bessel, hypergeometric, etc.
    lattice_discretization: Finite-volume approximations
    parallel_computing: MPI/OpenMP infrastructure

Design Principles:
    1. No IRH-specific assumptions
    2. High numerical precision (configurable)
    3. Parallelizable where appropriate
    4. Well-documented numerical methods

Dependencies:
    - NumPy, SciPy (numerical computing)
    - Optional: JAX (GPU acceleration)
    - Optional: mpi4py (distributed computing)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"

__all__ = [
    # integration exports
    'integrate_SU2',
    'integrate_U1',
    'integrate_G_inf',
    'monte_carlo_integrate',
    
    # optimization exports
    'find_fixed_point_newton',
    'minimize_functional',
    'root_find',
    
    # special_functions exports
    'bessel_j',
    'hypergeometric_2f1',
    'wigner_d_matrix',
    
    # lattice_discretization exports
    'discretize_SU2',
    'discretize_U1',
    'laplacian_matrix',
    
    # parallel_computing exports
    'parallel_map',
    'distributed_sum',
]
