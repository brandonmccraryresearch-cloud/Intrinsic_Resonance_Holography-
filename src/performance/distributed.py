"""
Distributed Computing for IRH Computations (Dask/Ray)

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.6

This module provides distributed computing capabilities for cluster-scale IRH computations:
    - DistributedContext: Cluster management and automatic serial fallback
    - Dask-based distributed RG flow integration (§1.2-1.3)
    - Ray-based parameter space exploration (§1.3, Appendix B)
    - Distributed Monte Carlo sampling on G_inf (§1.1)
    - Distributed QNCD matrix computation (Appendix A)

The distributed computing layer maintains theoretical fidelity while enabling
cluster-scale performance for demanding computations such as:
    - Large-scale RG flow trajectory integration (§1.2)
    - Parameter space exploration and sensitivity analysis (§1.3)
    - Monte Carlo sampling for ensemble averages (§1.1)
    - QNCD distance matrix for large state spaces (Appendix A)

Implementation Options:
    - Dask: Primary distributed backend with lazy evaluation
    - Ray: Alternative distributed backend for task-based parallelism
    - Serial: Fallback for single-machine execution

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - IRH v21.1 Manuscript §1.2-1.3 (RG flow integration)
    - IRH v21.1 Manuscript §1.1 (Monte Carlo on G_inf)
    - IRH v21.1 Manuscript Appendix A (QNCD parallelization)
    - IRH v21.1 Manuscript Appendix B (RG flow details)
    - docs/ROADMAP.md §3.6 (Distributed Computing phase)

Implementation Notes:
    - Dask/Ray are optional dependencies; fallback to serial execution if unavailable
    - All distributed operations maintain numerical precision equivalent to serial versions
    - Automatic scaling based on cluster resources
    - Progress tracking and result aggregation across workers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from enum import Enum
import warnings
import time
import functools

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'DistributedBackend',
    'DistributedContext',
    'dask_rg_flow',
    'ray_parameter_scan',
    'distributed_monte_carlo',
    'distributed_qncd_matrix',
    'distributed_map',
    'is_dask_available',
    'is_ray_available',
    'get_distributed_info',
    'get_available_distributed_backends',
    'create_local_cluster',
    'shutdown_cluster',
]

# =============================================================================
# Distributed Backend Availability Detection
# =============================================================================

class DistributedBackend(Enum):
    """Available distributed computing backends."""
    DASK = "dask"
    RAY = "ray"
    SERIAL = "serial"  # Single-machine fallback


_DASK_AVAILABLE = False
_DASK_IMPORT_ERROR: Optional[str] = None
_DASK_DISTRIBUTED_AVAILABLE = False

_RAY_AVAILABLE = False
_RAY_IMPORT_ERROR: Optional[str] = None

# Try to import Dask
try:
    import dask
    import dask.array as da
    from dask import delayed, compute
    _DASK_AVAILABLE = True
    
    # Check for distributed support
    try:
        from dask.distributed import Client, LocalCluster, get_client
        _DASK_DISTRIBUTED_AVAILABLE = True
    except ImportError:
        _DASK_DISTRIBUTED_AVAILABLE = False
        
except ImportError as e:
    _DASK_IMPORT_ERROR = str(e)
    dask = None  # type: ignore
    da = None  # type: ignore
    delayed = None  # type: ignore
    compute = None  # type: ignore

# Try to import Ray
try:
    import ray
    _RAY_AVAILABLE = True
except ImportError as e:
    _RAY_IMPORT_ERROR = str(e)
    ray = None  # type: ignore

# Default backend selection
_DEFAULT_BACKEND: DistributedBackend = DistributedBackend.SERIAL
if _DASK_DISTRIBUTED_AVAILABLE:
    _DEFAULT_BACKEND = DistributedBackend.DASK
elif _RAY_AVAILABLE:
    _DEFAULT_BACKEND = DistributedBackend.RAY
elif _DASK_AVAILABLE:
    _DEFAULT_BACKEND = DistributedBackend.DASK  # Dask without distributed still provides parallel


# Theoretical Reference: IRH v21.4



def is_dask_available() -> bool:
    """
    Check if Dask is available.
    
    Returns
    -------
    bool
        True if Dask is installed and importable
        
    Examples
    --------
    >>> if is_dask_available():
    ...     ctx = DistributedContext(backend=DistributedBackend.DASK)
    ... else:
    ...     print("Dask not available, using serial execution")
    """
    return _DASK_AVAILABLE


# Theoretical Reference: IRH v21.4



def is_ray_available() -> bool:
    """
    Check if Ray is available.
    
    Returns
    -------
    bool
        True if Ray is installed and importable
        
    Examples
    --------
    >>> if is_ray_available():
    ...     ctx = DistributedContext(backend=DistributedBackend.RAY)
    ... else:
    ...     print("Ray not available, using serial execution")
    """
    return _RAY_AVAILABLE


# Theoretical Reference: IRH v21.4



def get_available_distributed_backends() -> List[DistributedBackend]:
    """
    Get list of available distributed computing backends.
    
    Returns
    -------
    List[DistributedBackend]
        List of available backends (always includes SERIAL)
    """
    backends = [DistributedBackend.SERIAL]
    if _DASK_AVAILABLE:
        backends.append(DistributedBackend.DASK)
    if _RAY_AVAILABLE:
        backends.append(DistributedBackend.RAY)
    return backends


# Theoretical Reference: IRH v21.4



def get_distributed_info() -> Dict[str, Any]:
    """
    Get information about distributed computing environment.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with distributed computing information:
        - 'dask_available': bool
        - 'dask_distributed': bool
        - 'ray_available': bool
        - 'default_backend': str
        - 'available_backends': list
    """
    info: Dict[str, Any] = {
        'dask_available': _DASK_AVAILABLE,
        'dask_distributed': _DASK_DISTRIBUTED_AVAILABLE,
        'dask_import_error': _DASK_IMPORT_ERROR,
        'ray_available': _RAY_AVAILABLE,
        'ray_import_error': _RAY_IMPORT_ERROR,
        'default_backend': _DEFAULT_BACKEND.value,
        'available_backends': [b.value for b in get_available_distributed_backends()],
    }
    
    if _DASK_AVAILABLE and dask is not None:
        info['dask_version'] = dask.__version__
    
    if _RAY_AVAILABLE and ray is not None:
        info['ray_version'] = ray.__version__
    
    return info


# =============================================================================
# Physical Constants (from IRH v21.1 Manuscript Eq. 1.14)
# =============================================================================

LAMBDA_STAR = 48 * np.pi**2 / 9  # ≈ 52.638
GAMMA_STAR = 32 * np.pi**2 / 3   # ≈ 105.276
MU_STAR = 16 * np.pi**2          # ≈ 157.914


# =============================================================================
# Distributed Context Manager
# =============================================================================

@dataclass
class DistributedContext:
    """
    Context manager for distributed computing.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.6 - Computational infrastructure
        docs/ROADMAP.md §3.6 - Distributed Computing
    
    Handles cluster initialization, task submission, and cleanup.
    Provides graceful fallback to serial execution if Dask/Ray unavailable.
    
    Parameters
    ----------
    backend : DistributedBackend, optional
        Distributed backend to use (default: auto-selected best available)
    n_workers : int, optional
        Number of workers for local cluster (default: auto-detect)
    scheduler_address : str, optional
        Address of existing scheduler (for remote clusters)
    verbose : bool
        Whether to print cluster information (default: False)
    memory_limit : str, optional
        Memory limit per worker (e.g., '4GB')
        
    Attributes
    ----------
    backend : DistributedBackend
        Active backend
    n_workers : int
        Number of workers
    is_distributed : bool
        Whether true distributed computing is active
        
    Examples
    --------
    >>> with DistributedContext() as ctx:
    ...     result = dask_rg_flow(initial_conditions, ctx=ctx)
    
    >>> ctx = DistributedContext(backend=DistributedBackend.RAY, n_workers=4)
    >>> result = ray_parameter_scan(parameter_grid, ctx=ctx)
    """
    
    backend: DistributedBackend = field(default_factory=lambda: _DEFAULT_BACKEND)
    n_workers: Optional[int] = None
    scheduler_address: Optional[str] = None
    verbose: bool = False
    memory_limit: Optional[str] = None
    
    # Runtime state
    _client: Any = field(default=None, repr=False)
    _cluster: Any = field(default=None, repr=False)
    _active: bool = field(default=False, repr=False)
    _ray_initialized: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Validate backend and configuration."""
        if self.backend == DistributedBackend.DASK and not _DASK_AVAILABLE:
            if self.verbose:
                warnings.warn(f"Dask not available, falling back to serial")
            self.backend = DistributedBackend.SERIAL
            
        if self.backend == DistributedBackend.RAY and not _RAY_AVAILABLE:
            if self.verbose:
                warnings.warn(f"Ray not available, falling back to serial")
            self.backend = DistributedBackend.SERIAL
        
        # Auto-detect workers if not specified
        if self.n_workers is None:
            import os
            self.n_workers = os.cpu_count() or 4
    
    @property
    def is_distributed(self) -> bool:
        """Check if distributed computing is active."""
        return self.backend != DistributedBackend.SERIAL and self._active
    
    def __enter__(self) -> 'DistributedContext':
        """Enter distributed context and initialize cluster."""
        self._active = True
        
        if self.backend == DistributedBackend.DASK and _DASK_DISTRIBUTED_AVAILABLE:
            self._init_dask_cluster()
        elif self.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
            self._init_ray()
        
        if self.verbose:
            print(f"[Distributed] Entering context with backend: {self.backend.value}")
            print(f"[Distributed] Workers: {self.n_workers}")
            print(f"[Distributed] Distributed active: {self.is_distributed}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit distributed context and cleanup."""
        self._active = False
        
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        
        if self._cluster is not None:
            try:
                self._cluster.close()
            except Exception:
                pass
            self._cluster = None
        
        if self._ray_initialized and _RAY_AVAILABLE:
            try:
                ray.shutdown()
            except Exception:
                pass
            self._ray_initialized = False
        
        if self.verbose:
            print(f"[Distributed] Exiting context")
        
        return False
    
    def _init_dask_cluster(self):
        """Initialize Dask distributed cluster."""
        if not _DASK_DISTRIBUTED_AVAILABLE:
            return
        
        try:
            if self.scheduler_address:
                # Connect to existing cluster
                self._client = Client(self.scheduler_address)
            else:
                # Create local cluster
                cluster_kwargs = {'n_workers': self.n_workers}
                if self.memory_limit:
                    cluster_kwargs['memory_limit'] = self.memory_limit
                    
                self._cluster = LocalCluster(**cluster_kwargs)
                self._client = Client(self._cluster)
                
            if self.verbose:
                print(f"[Dask] Cluster initialized: {self._client}")
                
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Failed to initialize Dask cluster: {e}")
            self._client = None
            self._cluster = None
    
    def _init_ray(self):
        """Initialize Ray runtime."""
        if not _RAY_AVAILABLE:
            return
        
        try:
            if not ray.is_initialized():
                ray_kwargs = {}
                if self.n_workers:
                    ray_kwargs['num_cpus'] = self.n_workers
                ray.init(**ray_kwargs)
                self._ray_initialized = True
                
            if self.verbose:
                print(f"[Ray] Runtime initialized")
                
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Failed to initialize Ray: {e}")
            self._ray_initialized = False
    
    # Theoretical Reference: IRH v21.4

    
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit a task for distributed execution.
        
        Parameters
        ----------
        func : Callable
            Function to execute
        *args : Any
            Positional arguments
        **kwargs : Any
            Keyword arguments
            
        Returns
        -------
        Any
            Future or delayed object
        """
        if self.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
            return delayed(func)(*args, **kwargs)
        elif self.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
            remote_func = ray.remote(func)
            return remote_func.remote(*args, **kwargs)
        else:
            # Serial execution
            return func(*args, **kwargs)
    
    # Theoretical Reference: IRH v21.4

    
    def gather(self, futures: List[Any]) -> List[Any]:
        """
        Gather results from distributed tasks.
        
        Parameters
        ----------
        futures : List[Any]
            List of futures or delayed objects
            
        Returns
        -------
        List[Any]
            List of computed results
        """
        if self.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
            return list(compute(*futures))
        elif self.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
            return ray.get(futures)
        else:
            # Serial - futures are already computed
            return futures


# =============================================================================
# Helper Functions
# =============================================================================

# Theoretical Reference: IRH v21.4


def create_local_cluster(
    n_workers: int = 4,
    memory_limit: str = '4GB',
) -> Any:
    """
    Create a local Dask cluster.
    
    Parameters
    ----------
    n_workers : int
        Number of workers
    memory_limit : str
        Memory limit per worker
        
    Returns
    -------
    Any
        LocalCluster object or None if unavailable
    """
    if not _DASK_DISTRIBUTED_AVAILABLE:
        warnings.warn("Dask distributed not available")
        return None
    
    return LocalCluster(n_workers=n_workers, memory_limit=memory_limit)


# Theoretical Reference: IRH v21.4



def shutdown_cluster(cluster: Any) -> None:
    """
    Shutdown a Dask cluster.
    
    Parameters
    ----------
    cluster : Any
        Cluster object to shutdown
    """
    if cluster is not None:
        try:
            cluster.close()
        except Exception:
            pass


# =============================================================================
# Beta Functions (Eq. 1.13)
# =============================================================================

# Pre-computed mathematical constant
_PI_SQUARED = np.pi ** 2


def _beta_functions(couplings: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute β-functions from Eq. 1.13.
    
    β_λ = -2λ̃ + (9/8π²)λ̃²
    β_γ = (3/4π²)λ̃γ̃
    β_μ = 2μ̃ + (1/2π²)λ̃μ̃
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2, Eq. 1.13
    """
    couplings = np.atleast_2d(couplings)
    
    lambda_t = couplings[:, 0]
    gamma_t = couplings[:, 1]
    mu_t = couplings[:, 2]
    
    beta_lambda = -2 * lambda_t + (9 / (8 * _PI_SQUARED)) * lambda_t**2
    beta_gamma = (3 / (4 * _PI_SQUARED)) * lambda_t * gamma_t
    beta_mu = 2 * mu_t + (1 / (2 * _PI_SQUARED)) * lambda_t * mu_t
    
    return np.column_stack([beta_lambda, beta_gamma, beta_mu])


def _integrate_rg_trajectory(
    initial: NDArray[np.float64],
    t_range: Tuple[float, float],
    n_steps: int,
) -> Tuple[NDArray[np.float64], bool]:
    """
    Integrate single RG trajectory using RK4.
    
    Returns
    -------
    Tuple containing:
        - trajectory: Array of shape (n_steps+1, 3)
        - converged: Whether trajectory converged to fixed point
    """
    dt = (t_range[1] - t_range[0]) / n_steps
    
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = initial
    
    y = np.atleast_2d(initial.copy())
    
    for i in range(n_steps):
        k1 = _beta_functions(y) * dt
        k2 = _beta_functions(y + 0.5 * k1) * dt
        k3 = _beta_functions(y + 0.5 * k2) * dt
        k4 = _beta_functions(y + k3) * dt
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        trajectory[i + 1] = y[0]
    
    # Check convergence
    final_beta = _beta_functions(y)
    residual = np.linalg.norm(final_beta)
    converged = residual < 1e-6
    
    return trajectory, converged


# =============================================================================
# Distributed RG Flow Integration (Dask)
# =============================================================================

def dask_rg_flow(
    initial_conditions: NDArray[np.float64],
    t_range: Tuple[float, float] = (-20.0, 10.0),
    n_steps: int = 1000,
    ctx: Optional[DistributedContext] = None,
) -> Dict[str, Any]:
    """
    Distributed RG flow integration using Dask.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2, Eq. 1.12-1.13
        
        Solves d(λ,γ,μ)/dt = (β_λ, β_γ, β_μ) with β-functions from Eq. 1.13.
        Multiple initial conditions are integrated simultaneously using
        Dask delayed tasks for distributed execution.
    
    Parameters
    ----------
    initial_conditions : NDArray[np.float64]
        Array of initial conditions, shape (N, 3) with columns [λ, γ, μ]
    t_range : Tuple[float, float]
        RG time range (t_min, t_max) where t = log(k/k₀)
    n_steps : int
        Number of integration steps
    ctx : DistributedContext, optional
        Distributed context (created if not provided)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'trajectories': Array of shape (N, n_steps+1, 3) with flow trajectories
        - 'times': Array of RG times
        - 'converged': Boolean mask of converged trajectories
        - 'fixed_points': Final coupling values
        - 'timing': Execution timing information
        - 'backend': Backend used
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> initial = np.random.rand(100, 3) * 100
    >>> with DistributedContext() as ctx:
    ...     result = dask_rg_flow(initial, t_range=(-20, 10), ctx=ctx)
    >>> print(f"Converged: {np.sum(result['converged'])} / {len(initial)}")
    """
    if ctx is None:
        ctx = DistributedContext()
    
    initial_conditions = np.atleast_2d(initial_conditions)
    n_trajectories = initial_conditions.shape[0]
    
    start_time = time.time()
    
    if ctx.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
        # Create delayed tasks for each trajectory
        delayed_results = []
        for i in range(n_trajectories):
            d_result = delayed(_integrate_rg_trajectory)(
                initial_conditions[i],
                t_range,
                n_steps
            )
            delayed_results.append(d_result)
        
        # Compute all in parallel
        results = compute(*delayed_results)
        
        # Unpack results
        trajectories = np.array([r[0] for r in results])
        converged = np.array([r[1] for r in results])
        
    else:
        # Serial execution
        trajectories = np.zeros((n_trajectories, n_steps + 1, 3))
        converged = np.zeros(n_trajectories, dtype=bool)
        
        for i in range(n_trajectories):
            traj, conv = _integrate_rg_trajectory(
                initial_conditions[i],
                t_range,
                n_steps
            )
            trajectories[i] = traj
            converged[i] = conv
    
    end_time = time.time()
    times = np.linspace(t_range[0], t_range[1], n_steps + 1)
    
    return {
        'trajectories': trajectories,
        'times': times,
        'converged': converged,
        'fixed_points': trajectories[:, -1, :],
        'n_trajectories': n_trajectories,
        'n_converged': int(np.sum(converged)),
        'timing': {
            'total_seconds': end_time - start_time,
            'n_trajectories': n_trajectories,
            'trajectories_per_second': n_trajectories / (end_time - start_time),
        },
        'backend': ctx.backend.value,
        'is_distributed': ctx.is_distributed,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.2, Eq. 1.12-1.13',
    }


# =============================================================================
# Ray Parameter Space Exploration
# =============================================================================

def _evaluate_at_point(
    params: NDArray[np.float64],
    eval_func: Callable[[NDArray], float],
) -> Tuple[NDArray[np.float64], float]:
    """Evaluate function at a parameter point."""
    value = eval_func(params)
    return params, value


def ray_parameter_scan(
    parameter_grid: NDArray[np.float64],
    evaluation_function: Optional[Callable[[NDArray], float]] = None,
    ctx: Optional[DistributedContext] = None,
) -> Dict[str, Any]:
    """
    Distributed parameter space exploration using Ray.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.3, Appendix B
        
        Explores the coupling space (λ̃, γ̃, μ̃) to map out the RG flow
        structure, fixed points, and their basins of attraction.
    
    Parameters
    ----------
    parameter_grid : NDArray[np.float64]
        Array of parameter points, shape (N, D) where D is dimension
    evaluation_function : Callable, optional
        Function to evaluate at each point (default: distance to fixed point)
    ctx : DistributedContext, optional
        Distributed context (created if not provided)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'parameters': Input parameter points
        - 'values': Evaluation results
        - 'min_point': Point with minimum value
        - 'max_point': Point with maximum value
        - 'timing': Execution timing
        - 'backend': Backend used
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> # Scan parameter space around fixed point
    >>> lambda_vals = np.linspace(40, 60, 10)
    >>> gamma_vals = np.linspace(90, 120, 10)
    >>> mu_vals = np.linspace(140, 180, 10)
    >>> grid = np.array(np.meshgrid(lambda_vals, gamma_vals, mu_vals)).T.reshape(-1, 3)
    >>> with DistributedContext(backend=DistributedBackend.RAY) as ctx:
    ...     result = ray_parameter_scan(grid, ctx=ctx)
    """
    if ctx is None:
        ctx = DistributedContext()
    
    parameter_grid = np.atleast_2d(parameter_grid)
    n_points = parameter_grid.shape[0]
    
    # Default evaluation: distance to fixed point
    if evaluation_function is None:
        fixed_point = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])
        # Theoretical Reference: IRH v21.4

        def evaluation_function(params):
            return np.linalg.norm(params - fixed_point)
    
    start_time = time.time()
    
    if ctx.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
        # Create Ray remote function
        @ray.remote
        # Theoretical Reference: IRH v21.4

        def ray_evaluate(params, eval_func):
            return _evaluate_at_point(params, eval_func)
        
        # Submit all tasks
        futures = [
            ray_evaluate.remote(parameter_grid[i], evaluation_function)
            for i in range(n_points)
        ]
        
        # Gather results
        results = ray.get(futures)
        
        parameters = np.array([r[0] for r in results])
        values = np.array([r[1] for r in results])
        
    elif ctx.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
        # Use Dask delayed
        delayed_results = [
            delayed(_evaluate_at_point)(parameter_grid[i], evaluation_function)
            for i in range(n_points)
        ]
        
        results = compute(*delayed_results)
        
        parameters = np.array([r[0] for r in results])
        values = np.array([r[1] for r in results])
        
    else:
        # Serial execution
        parameters = parameter_grid.copy()
        values = np.array([evaluation_function(p) for p in parameter_grid])
    
    end_time = time.time()
    
    # Find min/max points
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    
    return {
        'parameters': parameters,
        'values': values,
        'min_point': parameters[min_idx],
        'min_value': values[min_idx],
        'max_point': parameters[max_idx],
        'max_value': values[max_idx],
        'n_points': n_points,
        'timing': {
            'total_seconds': end_time - start_time,
            'n_points': n_points,
            'points_per_second': n_points / (end_time - start_time),
        },
        'backend': ctx.backend.value,
        'is_distributed': ctx.is_distributed,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.3, Appendix B',
    }


# =============================================================================
# Distributed Monte Carlo Sampling
# =============================================================================

def _monte_carlo_sample(
    n_samples: int,
    sample_function: Callable[[int], NDArray],
    observable_function: Callable[[NDArray], float],
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Generate Monte Carlo samples and evaluate observable.
    
    Returns
    -------
    Tuple containing:
        - samples: Generated samples
        - values: Observable values
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = sample_function(n_samples)
    values = np.array([observable_function(s) for s in samples])
    
    return samples, values


def distributed_monte_carlo(
    n_samples: int,
    sample_function: Optional[Callable[[int], NDArray]] = None,
    observable_function: Optional[Callable[[NDArray], float]] = None,
    n_batches: int = 10,
    ctx: Optional[DistributedContext] = None,
) -> Dict[str, Any]:
    """
    Distributed Monte Carlo sampling on G_inf.
    
    # Theoretical Reference:
        IRH v21.1 Manuscript §1.1
        
        Monte Carlo sampling on the informational group manifold
        G_inf = SU(2) × U(1)_φ for computing ensemble averages
        of cGFT observables.
    
    Parameters
    ----------
    n_samples : int
        Total number of Monte Carlo samples
    sample_function : Callable, optional
        Function to generate samples (default: random couplings)
    observable_function : Callable, optional
        Observable to evaluate (default: beta function norm)
    n_batches : int
        Number of batches for distributed execution
    ctx : DistributedContext, optional
        Distributed context
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'mean': Mean of observable
        - 'std': Standard deviation
        - 'samples': All samples (if small)
        - 'values': Observable values
        - 'timing': Execution timing
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> with DistributedContext() as ctx:
    ...     result = distributed_monte_carlo(
    ...         n_samples=10000,
    ...         n_batches=10,
    ...         ctx=ctx
    ...     )
    >>> print(f"Observable mean: {result['mean']:.6f} ± {result['std']:.6f}")
    """
    if ctx is None:
        ctx = DistributedContext()
    
    # Default sample function: random couplings around fixed point
    if sample_function is None:
        # Theoretical Reference: IRH v21.4

        def sample_function(n):
            return np.random.randn(n, 3) * 20 + np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])
    
    # Default observable: norm of beta functions
    if observable_function is None:
        # Theoretical Reference: IRH v21.4

        def observable_function(params):
            betas = _beta_functions(params.reshape(1, -1))
            return np.linalg.norm(betas)
    
    samples_per_batch = n_samples // n_batches
    
    start_time = time.time()
    
    if ctx.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
        # Create delayed batches
        delayed_results = [
            delayed(_monte_carlo_sample)(
                samples_per_batch,
                sample_function,
                observable_function,
                seed=i * 1000
            )
            for i in range(n_batches)
        ]
        
        results = compute(*delayed_results)
        
        all_samples = np.vstack([r[0] for r in results])
        all_values = np.concatenate([r[1] for r in results])
        
    elif ctx.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
        @ray.remote
        # Theoretical Reference: IRH v21.4

        def ray_mc_sample(n, sample_func, obs_func, seed):
            return _monte_carlo_sample(n, sample_func, obs_func, seed)
        
        futures = [
            ray_mc_sample.remote(
                samples_per_batch,
                sample_function,
                observable_function,
                i * 1000
            )
            for i in range(n_batches)
        ]
        
        results = ray.get(futures)
        
        all_samples = np.vstack([r[0] for r in results])
        all_values = np.concatenate([r[1] for r in results])
        
    else:
        # Serial execution
        all_samples = []
        all_values = []
        
        for i in range(n_batches):
            samples, values = _monte_carlo_sample(
                samples_per_batch,
                sample_function,
                observable_function,
                seed=i * 1000
            )
            all_samples.append(samples)
            all_values.append(values)
        
        all_samples = np.vstack(all_samples)
        all_values = np.concatenate(all_values)
    
    end_time = time.time()
    
    return {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'variance': float(np.var(all_values)),
        'n_samples': len(all_values),
        'samples': all_samples if len(all_samples) <= 1000 else None,
        'values': all_values,
        'histogram': {
            'counts': np.histogram(all_values, bins=50)[0].tolist(),
            'bin_edges': np.histogram(all_values, bins=50)[1].tolist(),
        },
        'timing': {
            'total_seconds': end_time - start_time,
            'n_samples': len(all_values),
            'samples_per_second': len(all_values) / (end_time - start_time),
        },
        'backend': ctx.backend.value,
        'is_distributed': ctx.is_distributed,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.1',
    }


# =============================================================================
# Distributed QNCD Matrix Computation
# =============================================================================

def _compute_qncd_pairs(
    vectors: NDArray[np.float64],
    pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int, float]]:
    """
    Compute QNCD distances for a list of (i, j) pairs.
    
    Parameters
    ----------
    vectors : NDArray[np.float64]
        Full vector array
    pairs : List[Tuple[int, int]]
        List of (i, j) index pairs to compute
    
    Returns
    -------
    List[Tuple[int, int, float]]
        List of (row, col, distance) tuples
    """
    results = []
    norms = np.linalg.norm(vectors, axis=1)
    
    for i, j in pairs:
        diff = np.linalg.norm(vectors[i] - vectors[j])
        norm_sum = norms[i] + norms[j]
        if norm_sum > 0:
            d = diff / norm_sum
        else:
            d = 0.0
        results.append((int(i), int(j), float(d)))
    
    return results


def distributed_qncd_matrix(
    vectors: NDArray[np.float64],
    n_blocks: int = 10,
    ctx: Optional[DistributedContext] = None,
) -> Dict[str, Any]:
    """
    Distributed QNCD distance matrix computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A
        
        QNCD (Quantum Normalized Compression Distance) metric on G_inf.
        For algorithmic states represented by vectors, uses a proxy metric:
            d(x, y) = ||x - y|| / (||x|| + ||y||)
    
    Parameters
    ----------
    vectors : NDArray[np.float64]
        Array of state vectors, shape (N, D)
    n_blocks : int
        Number of blocks for parallel computation
    ctx : DistributedContext, optional
        Distributed context
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'distance_matrix': Symmetric NxN distance matrix
        - 'min_distance': Minimum non-zero distance
        - 'max_distance': Maximum distance
        - 'mean_distance': Mean pairwise distance
        - 'timing': Execution timing
        - 'theoretical_reference': IRH manuscript reference
    
    Examples
    --------
    >>> vectors = np.random.rand(500, 4)
    >>> with DistributedContext() as ctx:
    ...     result = distributed_qncd_matrix(vectors, n_blocks=5, ctx=ctx)
    >>> print(f"Matrix shape: {result['distance_matrix'].shape}")
    """
    if ctx is None:
        ctx = DistributedContext()
    
    N = vectors.shape[0]
    
    # Create all (i, j) pairs for upper triangle
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(all_pairs)
    
    # Distribute pairs across blocks
    pairs_per_block = (n_pairs + n_blocks - 1) // n_blocks
    
    start_time = time.time()
    
    if ctx.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
        # Create delayed block computations
        delayed_results = []
        
        for block_idx in range(n_blocks):
            start_idx = block_idx * pairs_per_block
            end_idx = min(start_idx + pairs_per_block, n_pairs)
            block_pairs = all_pairs[start_idx:end_idx]
            
            if block_pairs:
                # Pass pairs directly to compute function
                d_result = delayed(_compute_qncd_pairs)(vectors, block_pairs)
                delayed_results.append(d_result)
        
        results = compute(*delayed_results)
        
        # Flatten results
        all_distances = []
        for block_result in results:
            all_distances.extend(block_result)
        
    elif ctx.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
        @ray.remote
        # Theoretical Reference: IRH v21.4

        def ray_compute_pairs(vecs, pairs):
            return _compute_qncd_pairs(vecs, pairs)
        
        # Put vectors in object store for efficient sharing
        vectors_ref = ray.put(vectors)
        
        futures = []
        for block_idx in range(n_blocks):
            start_idx = block_idx * pairs_per_block
            end_idx = min(start_idx + pairs_per_block, n_pairs)
            block_pairs = all_pairs[start_idx:end_idx]
            
            if block_pairs:
                future = ray_compute_pairs.remote(vectors_ref, block_pairs)
                futures.append(future)
        
        results = ray.get(futures)
        
        all_distances = []
        for block_result in results:
            all_distances.extend(block_result)
        
    else:
        # Serial execution
        all_distances = []
        norms = np.linalg.norm(vectors, axis=1)
        
        for i, j in all_pairs:
            diff = np.linalg.norm(vectors[i] - vectors[j])
            norm_sum = norms[i] + norms[j]
            if norm_sum > 0:
                d = diff / norm_sum
            else:
                d = 0.0
            all_distances.append((i, j, d))
    
    end_time = time.time()
    
    # Build symmetric distance matrix
    dist_matrix = np.zeros((N, N))
    for i, j, d in all_distances:
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d
    
    # Statistics
    upper_tri = dist_matrix[np.triu_indices(N, k=1)]
    
    return {
        'distance_matrix': dist_matrix,
        'min_distance': float(np.min(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'max_distance': float(np.max(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'mean_distance': float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'n_vectors': N,
        'n_pairs': n_pairs,
        'timing': {
            'total_seconds': end_time - start_time,
            'pairs_per_second': n_pairs / (end_time - start_time) if n_pairs > 0 else 0,
        },
        'backend': ctx.backend.value,
        'is_distributed': ctx.is_distributed,
        'theoretical_reference': 'IRH v21.1 Manuscript Appendix A',
    }


# =============================================================================
# Generic Distributed Map
# =============================================================================

# Theoretical Reference: IRH v21.4


def distributed_map(
    func: Callable[[Any], Any],
    items: List[Any],
    ctx: Optional[DistributedContext] = None,
) -> List[Any]:
    """
    Apply a function to items in parallel.
    
    Parameters
    ----------
    func : Callable
        Function to apply
    items : List[Any]
        Items to process
    ctx : DistributedContext, optional
        Distributed context
        
    Returns
    -------
    List[Any]
        Results in same order as inputs
    
    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> with DistributedContext() as ctx:
    ...     results = distributed_map(square, [1, 2, 3, 4, 5], ctx)
    >>> print(results)  # [1, 4, 9, 16, 25]
    """
    if ctx is None:
        ctx = DistributedContext()
    
    if ctx.backend == DistributedBackend.DASK and _DASK_AVAILABLE:
        delayed_results = [delayed(func)(item) for item in items]
        results = list(compute(*delayed_results))
        return results
        
    elif ctx.backend == DistributedBackend.RAY and _RAY_AVAILABLE:
        remote_func = ray.remote(func)
        futures = [remote_func.remote(item) for item in items]
        return ray.get(futures)
        
    else:
        return [func(item) for item in items]
