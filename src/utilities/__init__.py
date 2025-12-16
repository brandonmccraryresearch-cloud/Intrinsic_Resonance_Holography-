"""
Cross-Cutting Computational Utilities for Intrinsic Resonance Holography v21.0

This module provides shared numerical infrastructure used across all layers
of the computational framework. These utilities are theory-agnostic and can
be used independently of the IRH-specific modules.

Modules:
    instrumentation: Theoretical logging and traceability (Phase II)
    output_contextualization: Standardized outputs with provenance (Phase III)
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
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"

# Import instrumentation module (Phase II)
from .instrumentation import (
    IRHLogLevel,
    TheoreticalReference,
    ComputationContext,
    IRHLogger,
    instrumented,
    get_logger,
    configure_logging,
)

# Import output contextualization module (Phase III)
from .output_contextualization import (
    ComputationType,
    TheoreticalContext,
    ComputationalProvenance,
    ObservableResult,
    UncertaintyTracker,
    IRHOutputWriter,
    create_output_writer,
    format_observable,
)

__all__ = [
    # instrumentation exports (Phase II)
    'IRHLogLevel',
    'TheoreticalReference',
    'ComputationContext',
    'IRHLogger',
    'instrumented',
    'get_logger',
    'configure_logging',
    
    # output_contextualization exports (Phase III)
    'ComputationType',
    'TheoreticalContext',
    'ComputationalProvenance',
    'ObservableResult',
    'UncertaintyTracker',
    'IRHOutputWriter',
    'create_output_writer',
    'format_observable',
    
    # integration exports (placeholder)
    'integrate_SU2',
    'integrate_U1',
    'integrate_G_inf',
    'monte_carlo_integrate',
    
    # optimization exports (placeholder)
    'find_fixed_point_newton',
    'minimize_functional',
    'root_find',
    
    # special_functions exports (placeholder)
    'bessel_j',
    'hypergeometric_2f1',
    'wigner_d_matrix',
    
    # lattice_discretization exports (placeholder)
    'discretize_SU2',
    'discretize_U1',
    'laplacian_matrix',
    
    # parallel_computing exports (placeholder)
    'parallel_map',
    'distributed_sum',
]
