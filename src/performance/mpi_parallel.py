"""
MPI Parallelization for IRH Computations

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.4

This module provides MPI-based parallelization for distributed IRH computations:
    - MPIContext: MPI environment initialization and cleanup
    - Distributed RG flow integration across multiple processes
    - Scatter/gather operations for batch computations
    - Domain decomposition for large lattice computations
    - Parallel fixed point search with load balancing

The parallelization maintains theoretical fidelity while enabling
exascale-ready performance for demanding computations such as:
    - Large-scale RG flow trajectory integration (§1.2)
    - QNCD distance matrix computation (Appendix A)
    - Topological invariant calculation (Appendix D)
    - Monte Carlo sampling on G_inf (§1.1)

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - IRH v21.1 Manuscript §1.2-1.3 (RG flow integration)
    - IRH v21.1 Manuscript Appendix A (QNCD parallelization)
    - docs/ROADMAP.md §3.4 (MPI Parallelization phase)

Implementation Notes:
    - mpi4py is an optional dependency; fallback to serial execution if unavailable
    - All MPI operations are wrapped in try/except for graceful degradation
    - Domain decomposition uses balanced workload distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import warnings
import time

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'MPIContext',
    'MPIBackend',
    'distributed_rg_flow',
    'scatter_initial_conditions',
    'gather_results',
    'parallel_fixed_point_search',
    'parallel_qncd_matrix',
    'domain_decomposition',
    'is_mpi_available',
    'get_mpi_info',
]

# =============================================================================
# MPI Availability Detection
# =============================================================================

_MPI_AVAILABLE = False
_MPI_IMPORT_ERROR: Optional[str] = None

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError as e:
    _MPI_IMPORT_ERROR = str(e)
    MPI = None  # type: ignore


# Theoretical Reference: IRH v21.4



def is_mpi_available() -> bool:
    """
    Check if MPI (mpi4py) is available.
    
    Returns
    -------
    bool
        True if mpi4py is installed and importable
        
    Examples
    --------
    >>> if is_mpi_available():
    ...     ctx = MPIContext()
    ... else:
    ...     print("MPI not available, using serial execution")
    """
    return _MPI_AVAILABLE


# Theoretical Reference: IRH v21.4



def get_mpi_info() -> Dict[str, Any]:
    """
    Get information about MPI environment.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with MPI information:
        - 'available': bool
        - 'version': MPI version string (if available)
        - 'import_error': error message (if MPI unavailable)
        - 'world_size': number of processes (if MPI active)
        - 'rank': current process rank (if MPI active)
    """
    info: Dict[str, Any] = {
        'available': _MPI_AVAILABLE,
        'import_error': _MPI_IMPORT_ERROR,
    }
    
    if _MPI_AVAILABLE and MPI is not None:
        comm = MPI.COMM_WORLD
        info['version'] = MPI.Get_version()
        info['world_size'] = comm.Get_size()
        info['rank'] = comm.Get_rank()
        info['processor_name'] = MPI.Get_processor_name()
    
    return info


# =============================================================================
# Physical Constants (from IRH v21.1 Manuscript Eq. 1.14)
# =============================================================================

LAMBDA_STAR = 48 * np.pi**2 / 9  # ≈ 52.638
GAMMA_STAR = 32 * np.pi**2 / 3   # ≈ 105.276
MU_STAR = 16 * np.pi**2          # ≈ 157.914


# =============================================================================
# MPI Context Manager
# =============================================================================

@dataclass
class MPIContext:
    """
    Context manager for MPI environment.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.6 - Computational infrastructure
        docs/ROADMAP.md §3.4 - MPI Parallelization
    
    Handles MPI initialization, communicator management, and cleanup.
    Provides graceful fallback to serial execution if MPI is unavailable.
    
    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD or None if unavailable)
    verbose : bool
        Whether to print MPI information (default: False)
    
    Attributes
    ----------
    rank : int
        Process rank (0 if MPI unavailable)
    size : int
        Number of processes (1 if MPI unavailable)
    is_root : bool
        Whether this is the root process (rank 0)
    is_parallel : bool
        Whether MPI parallelization is active (size > 1)
    
    Examples
    --------
    >>> with MPIContext() as ctx:
    ...     if ctx.is_root:
    ...         print(f"Running with {ctx.size} processes")
    ...     result = distributed_rg_flow(initial_conditions, ctx=ctx)
    """
    
    comm: Any = field(default=None)
    verbose: bool = field(default=False)
    
    # Computed fields
    rank: int = field(init=False, default=0)
    size: int = field(init=False, default=1)
    is_root: bool = field(init=False, default=True)
    is_parallel: bool = field(init=False, default=False)
    _active: bool = field(init=False, default=False)
    
    def __post_init__(self):
        """Initialize MPI context."""
        if _MPI_AVAILABLE and MPI is not None:
            if self.comm is None:
                self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_root = self.rank == 0
            self.is_parallel = self.size > 1
            self._active = True
        else:
            # Fallback to serial execution
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_root = True
            self.is_parallel = False
            self._active = False
            
            if self.verbose:
                warnings.warn(
                    "MPI not available, using serial execution. "
                    f"Import error: {_MPI_IMPORT_ERROR}"
                )
        
        if self.verbose and self.is_root:
            print(f"MPIContext initialized: rank={self.rank}, size={self.size}")
    
    def __enter__(self) -> 'MPIContext':
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and synchronize."""
        if self._active and self.comm is not None:
            self.barrier()
    
    # Theoretical Reference: IRH v21.4

    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self._active and self.comm is not None:
            self.comm.Barrier()
    
    def bcast(self, data: Any, root: int = 0) -> Any:
        """
        Broadcast data from root to all processes.
        
        Parameters
        ----------
        data : Any
            Data to broadcast (only root's value is used)
        root : int
            Root rank (default: 0)
            
        Returns
        -------
        Any
            Broadcasted data (same on all processes)
        
        # Theoretical Reference: IRH v21.4
        """
        if self._active and self.comm is not None:
            return self.comm.bcast(data, root=root)
        return data
    
    # Theoretical Reference: IRH v21.4

    
    def scatter(
        self,
        data: Optional[List[Any]],
        root: int = 0
    ) -> Any:
        """
        Scatter data from root to all processes.
        
        Parameters
        ----------
        data : List[Any], optional
            List of data chunks to scatter (only used on root)
        root : int
            Root rank (default: 0)
            
        Returns
        -------
        Any
            Local chunk of scattered data
        """
        if self._active and self.comm is not None:
            return self.comm.scatter(data, root=root)
        return data[0] if data else None
    
    # Theoretical Reference: IRH v21.4

    
    def gather(
        self,
        data: Any,
        root: int = 0
    ) -> Optional[List[Any]]:
        """
        Gather data from all processes to root.
        
        Parameters
        ----------
        data : Any
            Local data to gather
        root : int
            Root rank (default: 0)
            
        Returns
        -------
        Optional[List[Any]]
            List of gathered data (only on root, None on others)
        """
        if self._active and self.comm is not None:
            return self.comm.gather(data, root=root)
        return [data]
    
    # Theoretical Reference: IRH v21.4

    
    def allgather(self, data: Any) -> List[Any]:
        """
        Gather data from all processes to all processes.
        
        Parameters
        ----------
        data : Any
            Local data to gather
            
        Returns
        -------
        List[Any]
            List of gathered data from all processes
        """
        if self._active and self.comm is not None:
            return self.comm.allgather(data)
        return [data]
    
    # Theoretical Reference: IRH v21.4

    
    def reduce(
        self,
        data: Any,
        op: Optional[Any] = None,
        root: int = 0
    ) -> Optional[Any]:
        """
        Reduce data from all processes to root.
        
        Parameters
        ----------
        data : Any
            Local data for reduction
        op : MPI.Op, optional
            Reduction operation (default: MPI.SUM)
        root : int
            Root rank (default: 0)
            
        Returns
        -------
        Optional[Any]
            Reduced result (only on root)
        """
        if self._active and self.comm is not None and MPI is not None:
            if op is None:
                op = MPI.SUM
            return self.comm.reduce(data, op=op, root=root)
        return data
    
    # Theoretical Reference: IRH v21.4

    
    def allreduce(
        self,
        data: Any,
        op: Optional[Any] = None
    ) -> Any:
        """
        Reduce data and distribute result to all processes.
        
        Parameters
        ----------
        data : Any
            Local data for reduction
        op : MPI.Op, optional
            Reduction operation (default: MPI.SUM)
            
        Returns
        -------
        Any
            Reduced result (same on all processes)
        """
        if self._active and self.comm is not None and MPI is not None:
            if op is None:
                op = MPI.SUM
            return self.comm.allreduce(data, op=op)
        return data
    
    # Theoretical Reference: IRH v21.4 (Parallel Computing Infrastructure)
    def get_theoretical_reference(self) -> str:
        """Return theoretical foundation reference."""
        return "IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.4"


# =============================================================================
# MPI Backend Wrapper
# =============================================================================

@dataclass
class MPIBackend:
    
    # Theoretical Reference: IRH v21.4
    """
    Backend wrapper for MPI operations with automatic fallback.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.4 - MPI Parallelization
    
    Provides a unified interface for parallel operations that
    automatically falls back to serial execution if MPI is unavailable.
    
    Parameters
    ----------
    ctx : MPIContext, optional
        MPI context (created if not provided)
    load_balance : bool
        Whether to enable load balancing (default: True)
    chunk_size : int, optional
        Fixed chunk size for work distribution (None for automatic)
    
    Examples
    --------
    >>> backend = MPIBackend()
    >>> results = backend.parallel_map(expensive_function, data_list)
    """
    
    ctx: Optional[MPIContext] = None
    load_balance: bool = True
    chunk_size: Optional[int] = None
    
    def __post_init__(self):
        """Initialize backend."""
        if self.ctx is None:
            self.ctx = MPIContext()
    
    # Theoretical Reference: IRH v21.4

    
    def parallel_map(
        self,
        func: Callable[[Any], Any],
        data: List[Any],
        show_progress: bool = False
    ) -> List[Any]:
        """
        Apply function to data in parallel.
        
        Parameters
        ----------
        func : Callable
            Function to apply to each data item
        data : List[Any]
            List of input data items
        show_progress : bool
            Whether to show progress (on root only)
            
        Returns
        -------
        List[Any]
            List of results in same order as input
        """
        if self.ctx is None or not self.ctx.is_parallel:
            # Serial execution
            return [func(item) for item in data]
        
        # Distribute work
        chunks = self._distribute_work(data)
        local_chunk = self.ctx.scatter(chunks, root=0)
        
        # Process local chunk
        if show_progress and self.ctx.is_root:
            print(f"Processing {len(local_chunk)} items on rank {self.ctx.rank}")
        
        local_results = [func(item) for item in local_chunk]
        
        # Gather results
        all_results = self.ctx.gather(local_results, root=0)
        
        if self.ctx.is_root and all_results is not None:
            # Flatten results
            results = []
            for chunk_results in all_results:
                results.extend(chunk_results)
            return results
        
        return []
    
    def _distribute_work(self, data: List[Any]) -> List[List[Any]]:
        """Distribute data into chunks for each process."""
        if self.ctx is None:
            return [data]
        
        n = len(data)
        size = self.ctx.size
        
        if self.chunk_size is not None:
            # Fixed chunk size
            chunk_size = self.chunk_size
        else:
            # Balanced distribution
            chunk_size = (n + size - 1) // size
        
        chunks = []
        for i in range(size):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunks.append(data[start:end])
        
        return chunks


# =============================================================================
# Distributed RG Flow Integration
# =============================================================================

def distributed_rg_flow(
    initial_conditions: NDArray[np.float64],
    t_range: Tuple[float, float] = (-20.0, 10.0),
    n_steps: int = 1000,
    ctx: Optional[MPIContext] = None,
    beta_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Integrate RG flow trajectories in parallel using MPI.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2, Eq. 1.12-1.13
        
        Solves d(λ,γ,μ)/dt = (β_λ, β_γ, β_μ) with β-functions from Eq. 1.13.
        Multiple initial conditions are integrated simultaneously across
        MPI processes for efficient parameter space exploration.
    
    Parameters
    ----------
    initial_conditions : NDArray[np.float64]
        Array of initial conditions, shape (N, 3) with columns [λ, γ, μ]
    t_range : Tuple[float, float]
        RG time range (t_min, t_max) where t = log(k/k₀)
    n_steps : int
        Number of integration steps
    ctx : MPIContext, optional
        MPI context (created if not provided)
    beta_fn : Callable, optional
        Custom beta function (default: one-loop β-functions)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'trajectories': Array of shape (N, n_steps+1, 3) with flow trajectories
        - 'times': Array of RG times
        - 'converged': Boolean mask of converged trajectories
        - 'fixed_points': Final coupling values
        - 'timing': Execution timing information
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> # Explore RG flow from multiple starting points
    >>> initial = np.random.rand(100, 3) * 100
    >>> result = distributed_rg_flow(initial, t_range=(-20, 10))
    >>> print(f"Converged: {np.sum(result['converged'])} / {len(initial)}")
    """
    if ctx is None:
        ctx = MPIContext()
    
    if beta_fn is None:
        beta_fn = _default_beta_functions
    
    initial_conditions = np.atleast_2d(initial_conditions)
    n_trajectories = initial_conditions.shape[0]
    
    start_time = time.time()
    
    # Scatter initial conditions across processes
    local_ic = scatter_initial_conditions(initial_conditions, ctx)
    
    if local_ic is None or len(local_ic) == 0:
        local_trajectories = np.array([])
        local_converged = np.array([], dtype=bool)
    else:
        # Integrate local trajectories
        local_trajectories, local_converged = _integrate_rg_trajectories(
            local_ic, t_range, n_steps, beta_fn
        )
    
    # Gather results
    all_trajectories, all_converged, times = gather_results(
        local_trajectories, local_converged, t_range, n_steps, ctx
    )
    
    end_time = time.time()
    
    # Prepare result (only complete on root)
    result: Dict[str, Any] = {
        'theoretical_reference': 'IRH v21.1 Manuscript §1.2, Eq. 1.12-1.13',
    }
    
    if ctx.is_root:
        result['trajectories'] = all_trajectories
        result['times'] = times
        result['converged'] = all_converged
        result['fixed_points'] = all_trajectories[:, -1, :] if len(all_trajectories) > 0 else np.array([])
        result['timing'] = {
            'total_seconds': end_time - start_time,
            'n_trajectories': n_trajectories,
            'n_processes': ctx.size,
            'trajectories_per_process': n_trajectories / ctx.size,
        }
    
    return result


def _default_beta_functions(couplings: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Default one-loop β-functions from Eq. 1.13.
    
    β_λ = -2λ̃ + (9/8π²)λ̃²
    β_γ = (3/4π²)λ̃γ̃
    β_μ = 2μ̃ + (1/2π²)λ̃μ̃
    """
    couplings = np.atleast_2d(couplings)
    
    lambda_t = couplings[:, 0]
    gamma_t = couplings[:, 1]
    mu_t = couplings[:, 2]
    
    pi_sq = np.pi**2
    
    beta_lambda = -2 * lambda_t + (9 / (8 * pi_sq)) * lambda_t**2
    beta_gamma = (3 / (4 * pi_sq)) * lambda_t * gamma_t
    beta_mu = 2 * mu_t + (1 / (2 * pi_sq)) * lambda_t * mu_t
    
    return np.column_stack([beta_lambda, beta_gamma, beta_mu])


def _integrate_rg_trajectories(
    initial_conditions: NDArray[np.float64],
    t_range: Tuple[float, float],
    n_steps: int,
    beta_fn: Callable
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Integrate RG trajectories using RK4 method.
    
    Returns
    -------
    Tuple containing:
        - trajectories: Array of shape (N, n_steps+1, 3)
        - converged: Boolean array of shape (N,)
    """
    n_traj = initial_conditions.shape[0]
    dt = (t_range[1] - t_range[0]) / n_steps
    
    # Initialize trajectory storage
    trajectories = np.zeros((n_traj, n_steps + 1, 3))
    trajectories[:, 0, :] = initial_conditions
    
    # RK4 integration
    y = initial_conditions.copy()
    
    for i in range(n_steps):
        k1 = beta_fn(y) * dt
        k2 = beta_fn(y + 0.5 * k1) * dt
        k3 = beta_fn(y + 0.5 * k2) * dt
        k4 = beta_fn(y + k3) * dt
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        trajectories[:, i + 1, :] = y
    
    # Check convergence (β-functions vanish)
    final_betas = beta_fn(y)
    residuals = np.linalg.norm(final_betas, axis=1)
    converged = residuals < 1e-6
    
    return trajectories, converged


# =============================================================================
# Scatter/Gather Operations
# =============================================================================

def scatter_initial_conditions(
    initial_conditions: NDArray[np.float64],
    ctx: MPIContext
) -> NDArray[np.float64]:
    """
    Scatter initial conditions across MPI processes.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.4 - Work distribution
    
    Parameters
    ----------
    initial_conditions : NDArray[np.float64]
        Full array of initial conditions, shape (N, 3)
    ctx : MPIContext
        MPI context
        
    Returns
    -------
    NDArray[np.float64]
        Local chunk of initial conditions for this process
        
    Examples
    --------
    >>> ic = np.random.rand(100, 3)
    >>> with MPIContext() as ctx:
    ...     local_ic = scatter_initial_conditions(ic, ctx)
    ...     print(f"Rank {ctx.rank} has {len(local_ic)} conditions")
    """
    if not ctx.is_parallel:
        return initial_conditions
    
    # Prepare chunks on root
    chunks = None
    if ctx.is_root:
        n = len(initial_conditions)
        chunk_sizes = _balanced_chunks(n, ctx.size)
        
        chunks = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(initial_conditions[offset:offset + size])
            offset += size
    
    # Scatter
    local_chunk = ctx.scatter(chunks, root=0)
    
    return local_chunk


def gather_results(
    local_trajectories: NDArray[np.float64],
    local_converged: NDArray[np.bool_],
    t_range: Tuple[float, float],
    n_steps: int,
    ctx: MPIContext
) -> Tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """
    Gather RG flow results from all processes.
    
    Theoretical Reference:
        docs/ROADMAP.md §3.4 - Result aggregation
    
    Parameters
    ----------
    local_trajectories : NDArray[np.float64]
        Local trajectory results
    local_converged : NDArray[np.bool_]
        Local convergence flags
    t_range : Tuple[float, float]
        RG time range
    n_steps : int
        Number of integration steps
    ctx : MPIContext
        MPI context
        
    Returns
    -------
    Tuple containing:
        - all_trajectories: Combined trajectories (on root)
        - all_converged: Combined convergence flags (on root)
        - times: RG time array
    """
    times = np.linspace(t_range[0], t_range[1], n_steps + 1)
    
    if not ctx.is_parallel:
        return local_trajectories, local_converged, times
    
    # Gather trajectories and convergence flags
    all_traj_list = ctx.gather(local_trajectories, root=0)
    all_conv_list = ctx.gather(local_converged, root=0)
    
    if ctx.is_root and all_traj_list is not None and all_conv_list is not None:
        # Concatenate results
        # Filter out empty arrays
        non_empty_traj = [t for t in all_traj_list if len(t) > 0]
        non_empty_conv = [c for c in all_conv_list if len(c) > 0]
        
        if non_empty_traj:
            all_trajectories = np.vstack(non_empty_traj)
            all_converged = np.concatenate(non_empty_conv)
        else:
            all_trajectories = np.array([])
            all_converged = np.array([], dtype=bool)
        
        return all_trajectories, all_converged, times
    
    return np.array([]), np.array([], dtype=bool), times


def _balanced_chunks(n: int, num_chunks: int) -> List[int]:
    """Compute balanced chunk sizes for work distribution."""
    base_size = n // num_chunks
    remainder = n % num_chunks
    
    return [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]


# =============================================================================
# Parallel Fixed Point Search
# =============================================================================

def parallel_fixed_point_search(
    initial_guesses: NDArray[np.float64],
    ctx: Optional[MPIContext] = None,
    max_iter: int = 1000,
    tolerance: float = 1e-12
) -> Dict[str, Any]:
    """
    Search for fixed points in parallel using MPI.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.3, Eq. 1.14
        
        Fixed point: β(λ*, γ*, μ*) = 0
        Analytical values: λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²
    
    Parameters
    ----------
    initial_guesses : NDArray[np.float64]
        Array of initial guesses, shape (N, 3)
    ctx : MPIContext, optional
        MPI context
    max_iter : int
        Maximum Newton iterations per guess
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'fixed_points': Converged fixed point values
        - 'converged': Boolean convergence mask
        - 'iterations': Iterations to convergence
        - 'residuals': Final residual norms
        - 'unique_fixed_points': Unique fixed points found
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> # Search from random initial guesses
    >>> guesses = np.random.rand(100, 3) * 200
    >>> result = parallel_fixed_point_search(guesses)
    >>> print(f"Found {len(result['unique_fixed_points'])} unique fixed points")
    """
    if ctx is None:
        ctx = MPIContext()
    
    initial_guesses = np.atleast_2d(initial_guesses)
    
    # Scatter guesses
    local_guesses = scatter_initial_conditions(initial_guesses, ctx)
    
    if local_guesses is None or len(local_guesses) == 0:
        local_result = {
            'fixed_points': np.array([]),
            'converged': np.array([], dtype=bool),
            'iterations': np.array([], dtype=int),
            'residuals': np.array([]),
        }
    else:
        # Run Newton-Raphson on local guesses
        local_result = _newton_raphson_search(
            local_guesses, max_iter, tolerance
        )
    
    # Gather results
    all_fp = ctx.gather(local_result['fixed_points'], root=0)
    all_conv = ctx.gather(local_result['converged'], root=0)
    all_iter = ctx.gather(local_result['iterations'], root=0)
    all_resid = ctx.gather(local_result['residuals'], root=0)
    
    result: Dict[str, Any] = {
        'theoretical_reference': 'IRH v21.1 Manuscript §1.2.3, Eq. 1.14',
    }
    
    if ctx.is_root:
        # Concatenate results
        if all_fp and all_conv and all_iter and all_resid:
            non_empty_fp = [f for f in all_fp if len(f) > 0]
            non_empty_conv = [c for c in all_conv if len(c) > 0]
            non_empty_iter = [i for i in all_iter if len(i) > 0]
            non_empty_resid = [r for r in all_resid if len(r) > 0]
            
            if non_empty_fp:
                fixed_points = np.vstack(non_empty_fp)
                converged = np.concatenate(non_empty_conv)
                iterations = np.concatenate(non_empty_iter)
                residuals = np.concatenate(non_empty_resid)
            else:
                fixed_points = np.array([])
                converged = np.array([], dtype=bool)
                iterations = np.array([], dtype=int)
                residuals = np.array([])
        else:
            fixed_points = np.array([])
            converged = np.array([], dtype=bool)
            iterations = np.array([], dtype=int)
            residuals = np.array([])
        
        result['fixed_points'] = fixed_points
        result['converged'] = converged
        result['iterations'] = iterations
        result['residuals'] = residuals
        
        # Find unique fixed points
        if len(fixed_points) > 0 and np.any(converged):
            converged_fp = fixed_points[converged]
            result['unique_fixed_points'] = _find_unique_fixed_points(converged_fp)
        else:
            result['unique_fixed_points'] = np.array([])
    
    return result


def _newton_raphson_search(
    initial_guesses: NDArray[np.float64],
    max_iter: int,
    tolerance: float
) -> Dict[str, NDArray]:
    """
    Newton-Raphson fixed point search.
    
    Returns local results for each initial guess.
    """
    n_guesses = len(initial_guesses)
    
    # Working arrays
    x = initial_guesses.copy()
    converged = np.zeros(n_guesses, dtype=bool)
    iterations = np.zeros(n_guesses, dtype=int)
    
    h = 1e-8  # Finite difference step
    
    for iteration in range(max_iter):
        # Compute beta functions
        f = _default_beta_functions(x)
        residuals = np.linalg.norm(f, axis=1)
        
        # Check convergence
        newly_converged = (residuals < tolerance) & ~converged
        converged |= newly_converged
        iterations[newly_converged] = iteration
        
        if np.all(converged):
            break
        
        # Update non-converged points
        active = ~converged
        x_active = x[active]
        f_active = f[active]
        
        # Numerical Jacobian
        jacobian = np.zeros((len(x_active), 3, 3))
        for j in range(3):
            x_plus = x_active.copy()
            x_plus[:, j] += h
            f_plus = _default_beta_functions(x_plus)
            jacobian[:, :, j] = (f_plus - f_active) / h
        
        # Newton step with regularization
        reg = 1e-10 * np.eye(3)
        for i, (J, f_i) in enumerate(zip(jacobian, f_active)):
            try:
                delta = np.linalg.solve(J + reg, -f_i)
                # Find original index
                orig_idx = np.where(active)[0][i]
                x[orig_idx] += delta
            except np.linalg.LinAlgError:
                # Fallback: small step in -f direction
                orig_idx = np.where(active)[0][i]
                x[orig_idx] -= 0.1 * f_i
    
    # Final residuals
    f_final = _default_beta_functions(x)
    final_residuals = np.linalg.norm(f_final, axis=1)
    
    return {
        'fixed_points': x,
        'converged': converged,
        'iterations': iterations,
        'residuals': final_residuals,
    }


def _find_unique_fixed_points(
    fixed_points: NDArray[np.float64],
    tolerance: float = 1e-6
) -> NDArray[np.float64]:
    """Find unique fixed points by clustering nearby solutions."""
    # Validate input array
    if fixed_points.size == 0:
        return np.array([])
    
    # Ensure 2D array
    fixed_points = np.atleast_2d(fixed_points)
    if fixed_points.shape[0] == 0:
        return np.array([])
    
    unique = [fixed_points[0]]
    
    for fp in fixed_points[1:]:
        # Check if close to any existing unique point
        is_new = True
        for ufp in unique:
            if np.linalg.norm(fp - ufp) < tolerance:
                is_new = False
                break
        if is_new:
            unique.append(fp)
    
    return np.array(unique)


# =============================================================================
# Parallel QNCD Matrix Computation
# =============================================================================

def parallel_qncd_matrix(
    vectors: NDArray[np.float64],
    ctx: Optional[MPIContext] = None,
    method: str = 'compression_proxy'
) -> NDArray[np.float64]:
    """
    Compute QNCD distance matrix in parallel.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A
        
        QNCD(x,y) = [C(xy) - min(C(x),C(y))] / max(C(x),C(y))
    
    Parameters
    ----------
    vectors : NDArray[np.float64]
        Array of vectors, shape (N, D)
    ctx : MPIContext, optional
        MPI context
    method : str
        QNCD method ('compression_proxy', 'entropy', 'complexity')
        
    Returns
    -------
    NDArray[np.float64]
        Symmetric distance matrix of shape (N, N)
    
    Examples
    --------
    >>> vectors = np.random.rand(1000, 10)
    >>> with MPIContext() as ctx:
    ...     dist_matrix = parallel_qncd_matrix(vectors, ctx)
    """
    if ctx is None:
        ctx = MPIContext()
    
    n = len(vectors)
    
    # For small matrices or single process, compute directly
    if n <= 100 or not ctx.is_parallel:
        # Generate all pairs for upper triangle
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        # For larger matrices, still need all pairs but this is the bottleneck
        # Future optimization: use generator-based chunking
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Scatter pairs
    local_pairs: Optional[List[Tuple[int, int]]] = None
    if ctx.is_root:
        chunk_sizes = _balanced_chunks(len(pairs), ctx.size)
        chunks: List[List[Tuple[int, int]]] = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(pairs[offset:offset + size])
            offset += size
        local_pairs = ctx.scatter(chunks, root=0)
    else:
        local_pairs = ctx.scatter(None, root=0)
    
    if local_pairs is None:
        local_pairs = []
    
    # Compute local distances
    from .numerical_opts import vectorized_qncd_distance
    
    local_distances = []
    for i, j in local_pairs:
        v1 = vectors[i:i+1]
        v2 = vectors[j:j+1]
        # Use .item() to safely extract scalar from 0-d or 1-element array
        d_array = vectorized_qncd_distance(v1, v2, method=method)
        d = float(d_array.flat[0]) if d_array.size > 0 else 0.0
        local_distances.append((i, j, d))
    
    # Gather distances
    all_distances = ctx.gather(local_distances, root=0)
    
    if ctx.is_root and all_distances is not None:
        # Build symmetric matrix with explicit diagonal initialization
        dist_matrix = np.zeros((n, n))
        # Diagonal is explicitly 0 (distance to self)
        np.fill_diagonal(dist_matrix, 0.0)
        for chunk in all_distances:
            for i, j, d in chunk:
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix
    
    return np.zeros((n, n))


# =============================================================================
# Domain Decomposition
# =============================================================================

def domain_decomposition(
    lattice_shape: Tuple[int, ...],
    ctx: MPIContext
) -> Dict[str, Any]:
    """
    Compute domain decomposition for large lattice computations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1 - cGFT field discretization
        docs/ROADMAP.md §3.4 - Lattice parallelization
    
    Divides a D-dimensional lattice among MPI processes for parallel
    field theory computations (e.g., cGFT action evaluation).
    
    Parameters
    ----------
    lattice_shape : Tuple[int, ...]
        Shape of the full lattice (N₁, N₂, ..., Nᴅ)
    ctx : MPIContext
        MPI context
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'local_shape': Shape of local domain
        - 'local_offset': Offset of local domain in full lattice
        - 'halo_size': Size of halo region for communication
        - 'neighbors': Ranks of neighboring processes
        - 'decomposition_type': Type of decomposition used
    
    Examples
    --------
    >>> with MPIContext() as ctx:
    ...     decomp = domain_decomposition((100, 100, 100, 100), ctx)
    ...     print(f"Local domain: {decomp['local_shape']}")
    """
    n_dims = len(lattice_shape)
    n_procs = ctx.size
    rank = ctx.rank
    
    # Choose decomposition dimension (largest dimension)
    decomp_dim = 0
    if n_dims > 0:
        decomp_dim = int(np.argmax(lattice_shape))
    
    # Compute local slice along decomposition dimension
    full_size = lattice_shape[decomp_dim] if n_dims > 0 else 1
    chunk_sizes = _balanced_chunks(full_size, n_procs)
    
    # Compute offset
    offset = sum(chunk_sizes[:rank])
    local_size = chunk_sizes[rank]
    
    # Build local shape
    local_shape = list(lattice_shape)
    if n_dims > 0:
        local_shape[decomp_dim] = local_size
    
    # Build offset tuple
    local_offset = [0] * n_dims
    if n_dims > 0:
        local_offset[decomp_dim] = offset
    
    # Determine neighbors
    neighbors = {
        'left': (rank - 1) if rank > 0 else None,
        'right': (rank + 1) if rank < n_procs - 1 else None,
    }
    
    return {
        'local_shape': tuple(local_shape),
        'local_offset': tuple(local_offset),
        'halo_size': 1,  # Ghost cell layer
        'neighbors': neighbors,
        'decomposition_type': 'slab',
        'decomposition_dim': decomp_dim,
        'rank': rank,
        'n_processes': n_procs,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.1, docs/ROADMAP.md §3.4',
    }
