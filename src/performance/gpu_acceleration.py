"""
GPU Acceleration for IRH Computations

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.5

This module provides GPU-accelerated computations for demanding IRH calculations:
    - GPUContext: GPU device management and automatic CPU fallback
    - GPU-accelerated RG flow integration (§1.2-1.3)
    - GPU-accelerated QNCD distance matrix computation (Appendix A)
    - GPU-accelerated beta function batch evaluation (Eq. 1.13)
    - GPU-accelerated quaternion operations (§1.1.1)

The GPU acceleration maintains theoretical fidelity while achieving significant
speedups for computationally intensive operations such as:
    - Large-scale RG flow trajectory integration (§1.2)
    - QNCD distance matrix computation (Appendix A)
    - Batch quaternion multiplication (§1.1.1)
    - Field configuration sampling on G_inf (§1.1)

Implementation Options:
    - JAX: Primary GPU backend with XLA compilation
    - CuPy: Alternative NVIDIA CUDA backend
    - NumPy: Fallback for CPU-only systems

Authors: IRH Computational Framework Team
Last Updated: December 2025

References:
    - IRH v21.1 Manuscript §1.2-1.3 (RG flow integration)
    - IRH v21.1 Manuscript §1.1.1 (Quaternionic field theory)
    - IRH v21.1 Manuscript Appendix A (QNCD metric)
    - docs/ROADMAP.md §3.5 (GPU Acceleration phase)

Implementation Notes:
    - JAX/CuPy are optional dependencies; graceful fallback to NumPy if unavailable
    - All GPU operations maintain numerical precision equivalent to CPU versions
    - Device memory management is handled automatically
    - JIT compilation is used where available for optimal performance
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
    'GPUBackend',
    'GPUContext',
    'gpu_beta_functions',
    'gpu_rg_flow_integration',
    'gpu_qncd_matrix',
    'gpu_quaternion_multiply',
    'is_gpu_available',
    'get_gpu_info',
    'get_available_backends',
    'set_default_backend',
    'benchmark_gpu_performance',
]

# =============================================================================
# GPU Backend Availability Detection
# =============================================================================

class GPUBackend(Enum):
    """Available GPU backends for acceleration."""
    JAX = "jax"
    CUPY = "cupy"
    NUMPY = "numpy"  # CPU fallback


_JAX_AVAILABLE = False
_JAX_IMPORT_ERROR: Optional[str] = None
_JAX_GPU_AVAILABLE = False

_CUPY_AVAILABLE = False
_CUPY_IMPORT_ERROR: Optional[str] = None

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    _JAX_AVAILABLE = True
    
    # Check for GPU support
    try:
        devices = jax.devices()
        _JAX_GPU_AVAILABLE = any(d.platform == 'gpu' for d in devices)
    except Exception:
        _JAX_GPU_AVAILABLE = False
        
except ImportError as e:
    _JAX_IMPORT_ERROR = str(e)
    jax = None  # type: ignore
    jnp = None  # type: ignore

# Try to import CuPy
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError as e:
    _CUPY_IMPORT_ERROR = str(e)
    cp = None  # type: ignore

# Default backend selection
_DEFAULT_BACKEND: GPUBackend = GPUBackend.NUMPY
if _JAX_GPU_AVAILABLE:
    _DEFAULT_BACKEND = GPUBackend.JAX
elif _CUPY_AVAILABLE:
    _DEFAULT_BACKEND = GPUBackend.CUPY
elif _JAX_AVAILABLE:
    _DEFAULT_BACKEND = GPUBackend.JAX  # JAX CPU is still faster than NumPy


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns
    -------
    bool
        True if JAX with GPU or CuPy is available
        
    Examples
    --------
    >>> if is_gpu_available():
    ...     ctx = GPUContext()
    ... else:
    ...     print("GPU not available, using CPU")
    """
    return _JAX_GPU_AVAILABLE or _CUPY_AVAILABLE


def get_available_backends() -> List[GPUBackend]:
    """
    Get list of available GPU backends.
    
    Returns
    -------
    List[GPUBackend]
        List of available backends (always includes NUMPY)
    """
    backends = [GPUBackend.NUMPY]
    if _JAX_AVAILABLE:
        backends.append(GPUBackend.JAX)
    if _CUPY_AVAILABLE:
        backends.append(GPUBackend.CUPY)
    return backends


def set_default_backend(backend: GPUBackend) -> None:
    """
    Set the default GPU backend.
    
    Parameters
    ----------
    backend : GPUBackend
        Backend to use by default
        
    Raises
    ------
    ValueError
        If requested backend is not available
    """
    global _DEFAULT_BACKEND
    
    if backend == GPUBackend.JAX and not _JAX_AVAILABLE:
        raise ValueError(f"JAX is not available: {_JAX_IMPORT_ERROR}")
    if backend == GPUBackend.CUPY and not _CUPY_AVAILABLE:
        raise ValueError(f"CuPy is not available: {_CUPY_IMPORT_ERROR}")
    
    _DEFAULT_BACKEND = backend


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about GPU environment.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with GPU information:
        - 'gpu_available': bool
        - 'jax_available': bool
        - 'jax_gpu': bool
        - 'cupy_available': bool
        - 'default_backend': str
        - 'devices': list of device info (if available)
    """
    info: Dict[str, Any] = {
        'gpu_available': is_gpu_available(),
        'jax_available': _JAX_AVAILABLE,
        'jax_gpu': _JAX_GPU_AVAILABLE,
        'jax_import_error': _JAX_IMPORT_ERROR,
        'cupy_available': _CUPY_AVAILABLE,
        'cupy_import_error': _CUPY_IMPORT_ERROR,
        'default_backend': _DEFAULT_BACKEND.value,
        'available_backends': [b.value for b in get_available_backends()],
    }
    
    if _JAX_AVAILABLE and jax is not None:
        try:
            devices = jax.devices()
            info['jax_devices'] = [
                {'platform': d.platform, 'id': d.id}
                for d in devices
            ]
        except Exception:
            pass
    
    if _CUPY_AVAILABLE and cp is not None:
        try:
            info['cupy_device_count'] = cp.cuda.runtime.getDeviceCount()
        except Exception:
            pass
    
    return info


# =============================================================================
# Physical Constants (from IRH v21.1 Manuscript Eq. 1.14)
# =============================================================================

LAMBDA_STAR = 48 * np.pi**2 / 9  # ≈ 52.638
GAMMA_STAR = 32 * np.pi**2 / 3   # ≈ 105.276
MU_STAR = 16 * np.pi**2          # ≈ 157.914


# =============================================================================
# GPU Context Manager
# =============================================================================

@dataclass
class GPUContext:
    """
    Context manager for GPU computations.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.6 - Computational infrastructure
        docs/ROADMAP.md §3.5 - GPU Acceleration
    
    Handles GPU device selection, memory management, and backend switching.
    Provides graceful fallback to CPU execution if GPU is unavailable.
    
    Parameters
    ----------
    backend : GPUBackend, optional
        GPU backend to use (default: auto-selected best available)
    device_id : int, optional
        GPU device ID for multi-GPU systems (default: 0)
    verbose : bool
        Whether to print GPU information (default: False)
        
    Attributes
    ----------
    backend : GPUBackend
        Active backend
    device_id : int
        Active GPU device
    is_gpu : bool
        Whether computation uses GPU
        
    Examples
    --------
    >>> with GPUContext() as ctx:
    ...     result = gpu_beta_functions(couplings, ctx=ctx)
    
    >>> ctx = GPUContext(backend=GPUBackend.JAX)
    >>> result = gpu_rg_flow_integration(initial, ctx=ctx)
    """
    backend: GPUBackend = field(default_factory=lambda: _DEFAULT_BACKEND)
    device_id: int = 0
    verbose: bool = False
    
    # Runtime state
    _active: bool = field(default=False, repr=False)
    _original_device: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate backend and device configuration."""
        if self.backend == GPUBackend.JAX and not _JAX_AVAILABLE:
            if self.verbose:
                warnings.warn(f"JAX not available, falling back to NumPy")
            self.backend = GPUBackend.NUMPY
            
        if self.backend == GPUBackend.CUPY and not _CUPY_AVAILABLE:
            if self.verbose:
                warnings.warn(f"CuPy not available, falling back to NumPy")
            self.backend = GPUBackend.NUMPY
    
    @property
    def is_gpu(self) -> bool:
        """Check if GPU is being used."""
        if self.backend == GPUBackend.JAX:
            return _JAX_GPU_AVAILABLE
        elif self.backend == GPUBackend.CUPY:
            return _CUPY_AVAILABLE
        return False
    
    @property
    def array_module(self):
        """Get the appropriate array module (jnp, cp, or np)."""
        if self.backend == GPUBackend.JAX and _JAX_AVAILABLE:
            return jnp
        elif self.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
            return cp
        return np
    
    def __enter__(self) -> 'GPUContext':
        """Enter GPU context."""
        self._active = True
        
        if self.verbose:
            print(f"[GPU] Entering context with backend: {self.backend.value}")
            print(f"[GPU] GPU available: {self.is_gpu}")
        
        # Set device for CuPy
        if self.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
            self._original_device = cp.cuda.Device().id
            cp.cuda.Device(self.device_id).use()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit GPU context and cleanup."""
        self._active = False
        
        # Restore original device for CuPy
        if self._original_device is not None and _CUPY_AVAILABLE:
            cp.cuda.Device(self._original_device).use()
        
        if self.verbose:
            print(f"[GPU] Exiting context")
        
        return False
    
    def to_device(self, array: NDArray) -> Any:
        """
        Transfer array to GPU device.
        
        Parameters
        ----------
        array : NDArray
            NumPy array to transfer
            
        Returns
        -------
        Any
            Array on device (JAX array, CuPy array, or NumPy array)
        """
        if self.backend == GPUBackend.JAX and _JAX_AVAILABLE:
            return jnp.asarray(array)
        elif self.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
            return cp.asarray(array)
        return array
    
    def to_host(self, array: Any) -> NDArray:
        """
        Transfer array from GPU to CPU.
        
        Parameters
        ----------
        array : Any
            Device array to transfer
            
        Returns
        -------
        NDArray
            NumPy array on CPU
        """
        if self.backend == GPUBackend.JAX and _JAX_AVAILABLE:
            return np.asarray(array)
        elif self.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
            return cp.asnumpy(array)
        return np.asarray(array)


# =============================================================================
# GPU-Accelerated Beta Functions (Eq. 1.13)
# =============================================================================

def _compute_beta_functions_generic(xp, lambda_t, gamma_t, mu_t):
    """
    Generic beta function computation using any array module.
    
    Parameters
    ----------
    xp : module
        Array module (numpy, jax.numpy, or cupy)
    lambda_t, gamma_t, mu_t : array-like
        Coupling values
        
    Returns
    -------
    tuple
        (beta_lambda, beta_gamma, beta_mu)
        
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2, Eq. 1.13
    """
    # β_λ = -2λ̃ + (9/8π²)λ̃²
    beta_lambda = -2 * lambda_t + (9 / (8 * xp.pi**2)) * lambda_t**2
    # β_γ = (3/4π²)λ̃γ̃
    beta_gamma = (3 / (4 * xp.pi**2)) * lambda_t * gamma_t
    # β_μ = 2μ̃ + (1/2π²)λ̃μ̃
    beta_mu = 2 * mu_t + (1 / (2 * xp.pi**2)) * lambda_t * mu_t
    return beta_lambda, beta_gamma, beta_mu


def _beta_functions_numpy(
    lambda_t: NDArray[np.float64],
    gamma_t: NDArray[np.float64],
    mu_t: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """NumPy implementation of beta functions."""
    return _compute_beta_functions_generic(np, lambda_t, gamma_t, mu_t)


if _JAX_AVAILABLE:
    @jit
    def _beta_functions_jax(
        lambda_t: Any,
        gamma_t: Any,
        mu_t: Any,
    ) -> Tuple[Any, Any, Any]:
        """JAX-JIT compiled beta functions."""
        beta_lambda = -2 * lambda_t + (9 / (8 * jnp.pi**2)) * lambda_t**2
        beta_gamma = (3 / (4 * jnp.pi**2)) * lambda_t * gamma_t
        beta_mu = 2 * mu_t + (1 / (2 * jnp.pi**2)) * lambda_t * mu_t
        return beta_lambda, beta_gamma, beta_mu


def gpu_beta_functions(
    couplings: NDArray[np.float64],
    ctx: Optional[GPUContext] = None,
) -> Dict[str, Any]:
    """
    GPU-accelerated batch evaluation of beta functions.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2, Eq. 1.13
        β_λ = -2λ̃ + (9/8π²)λ̃²
        β_γ = (3/4π²)λ̃γ̃
        β_μ = 2μ̃ + (1/2π²)λ̃μ̃
    
    Parameters
    ----------
    couplings : NDArray[np.float64]
        Array of shape (N, 3) with coupling values [λ̃, γ̃, μ̃]
    ctx : GPUContext, optional
        GPU context for device management
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'beta_lambda': array of β_λ values
        - 'beta_gamma': array of β_γ values
        - 'beta_mu': array of β_μ values
        - 'backend': backend used
        - 'execution_time_ms': computation time
        - 'theoretical_reference': citation
    
    Examples
    --------
    >>> couplings = np.array([[52.0, 105.0, 157.0], [53.0, 106.0, 158.0]])
    >>> result = gpu_beta_functions(couplings)
    >>> print(result['beta_lambda'])
    """
    if ctx is None:
        ctx = GPUContext()
    
    start_time = time.perf_counter()
    
    # Ensure 2D input
    couplings = np.atleast_2d(couplings)
    lambda_t = couplings[:, 0]
    gamma_t = couplings[:, 1]
    mu_t = couplings[:, 2]
    
    if ctx.backend == GPUBackend.JAX and _JAX_AVAILABLE:
        # Transfer to device and compute
        lambda_d = ctx.to_device(lambda_t)
        gamma_d = ctx.to_device(gamma_t)
        mu_d = ctx.to_device(mu_t)
        
        beta_lambda, beta_gamma, beta_mu = _beta_functions_jax(
            lambda_d, gamma_d, mu_d
        )
        
        # Transfer back to host
        beta_lambda = ctx.to_host(beta_lambda)
        beta_gamma = ctx.to_host(beta_gamma)
        beta_mu = ctx.to_host(beta_mu)
        
    elif ctx.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
        # Transfer to device
        lambda_d = ctx.to_device(lambda_t)
        gamma_d = ctx.to_device(gamma_t)
        mu_d = ctx.to_device(mu_t)
        
        # Compute on GPU using generic function
        beta_lambda, beta_gamma, beta_mu = _compute_beta_functions_generic(
            cp, lambda_d, gamma_d, mu_d
        )
        
        # Transfer back
        beta_lambda = ctx.to_host(beta_lambda)
        beta_gamma = ctx.to_host(beta_gamma)
        beta_mu = ctx.to_host(beta_mu)
        
    else:
        # NumPy fallback
        beta_lambda, beta_gamma, beta_mu = _beta_functions_numpy(
            lambda_t, gamma_t, mu_t
        )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        'beta_lambda': beta_lambda,
        'beta_gamma': beta_gamma,
        'beta_mu': beta_mu,
        'backend': ctx.backend.value,
        'is_gpu': ctx.is_gpu,
        'execution_time_ms': elapsed_ms,
        'n_evaluations': len(lambda_t),
        'theoretical_reference': 'IRH v21.1 Manuscript §1.2.2, Eq. 1.13',
    }


# =============================================================================
# GPU-Accelerated RG Flow Integration (§1.2-1.3)
# =============================================================================

def _rk4_step_numpy(
    couplings: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Single RK4 step using NumPy."""
    lambda_t, gamma_t, mu_t = couplings
    
    # k1
    k1_l, k1_g, k1_m = _beta_functions_numpy(
        np.array([lambda_t]), np.array([gamma_t]), np.array([mu_t])
    )
    k1 = np.array([k1_l[0], k1_g[0], k1_m[0]])
    
    # k2
    mid1 = couplings + 0.5 * dt * k1
    k2_l, k2_g, k2_m = _beta_functions_numpy(
        np.array([mid1[0]]), np.array([mid1[1]]), np.array([mid1[2]])
    )
    k2 = np.array([k2_l[0], k2_g[0], k2_m[0]])
    
    # k3
    mid2 = couplings + 0.5 * dt * k2
    k3_l, k3_g, k3_m = _beta_functions_numpy(
        np.array([mid2[0]]), np.array([mid2[1]]), np.array([mid2[2]])
    )
    k3 = np.array([k3_l[0], k3_g[0], k3_m[0]])
    
    # k4
    end = couplings + dt * k3
    k4_l, k4_g, k4_m = _beta_functions_numpy(
        np.array([end[0]]), np.array([end[1]]), np.array([end[2]])
    )
    k4 = np.array([k4_l[0], k4_g[0], k4_m[0]])
    
    # Final update
    return couplings + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


if _JAX_AVAILABLE:
    @jit
    def _rk4_step_jax(
        couplings: Any,
        dt: float,
    ) -> Any:
        """JAX-JIT compiled RK4 step."""
        lambda_t, gamma_t, mu_t = couplings[0], couplings[1], couplings[2]
        
        # k1
        k1_l, k1_g, k1_m = _beta_functions_jax(lambda_t, gamma_t, mu_t)
        k1 = jnp.array([k1_l, k1_g, k1_m])
        
        # k2
        mid1 = couplings + 0.5 * dt * k1
        k2_l, k2_g, k2_m = _beta_functions_jax(mid1[0], mid1[1], mid1[2])
        k2 = jnp.array([k2_l, k2_g, k2_m])
        
        # k3
        mid2 = couplings + 0.5 * dt * k2
        k3_l, k3_g, k3_m = _beta_functions_jax(mid2[0], mid2[1], mid2[2])
        k3 = jnp.array([k3_l, k3_g, k3_m])
        
        # k4
        end = couplings + dt * k3
        k4_l, k4_g, k4_m = _beta_functions_jax(end[0], end[1], end[2])
        k4 = jnp.array([k4_l, k4_g, k4_m])
        
        # Final update
        return couplings + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def gpu_rg_flow_integration(
    initial_couplings: NDArray[np.float64],
    t_range: Tuple[float, float] = (-10, 10),
    n_steps: int = 1000,
    ctx: Optional[GPUContext] = None,
) -> Dict[str, Any]:
    """
    GPU-accelerated RG flow integration using RK4.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.2.2-1.2.3
        Integrates the coupled ODEs:
            d(λ̃,γ̃,μ̃)/dt = (β_λ, β_γ, β_μ)
        from UV (t>0) to IR (t<0)
    
    Parameters
    ----------
    initial_couplings : NDArray[np.float64]
        Initial values [λ̃₀, γ̃₀, μ̃₀]
    t_range : Tuple[float, float]
        RG time range (t_start, t_end)
    n_steps : int
        Number of integration steps
    ctx : GPUContext, optional
        GPU context for device management
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'trajectory': array of shape (n_steps+1, 3) with coupling evolution
        - 't_values': array of RG time values
        - 'converged': whether flow converged to fixed point
        - 'final_couplings': final coupling values
        - 'fixed_point_distance': distance from Cosmic Fixed Point
        - 'backend': backend used
        - 'execution_time_ms': computation time
    
    Examples
    --------
    >>> initial = np.array([60.0, 110.0, 160.0])
    >>> result = gpu_rg_flow_integration(initial, t_range=(-20, 0))
    >>> print(f"Converged: {result['converged']}")
    """
    if ctx is None:
        ctx = GPUContext()
    
    start_time = time.perf_counter()
    
    t_start, t_end = t_range
    dt = (t_end - t_start) / n_steps
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    
    # Initialize trajectory storage
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = initial_couplings
    
    couplings = np.array(initial_couplings, dtype=np.float64)
    
    if ctx.backend == GPUBackend.JAX and _JAX_AVAILABLE:
        # Transfer to device
        couplings_d = ctx.to_device(couplings)
        
        # Integration loop
        for i in range(n_steps):
            couplings_d = _rk4_step_jax(couplings_d, dt)
            trajectory[i + 1] = ctx.to_host(couplings_d)
            
    elif ctx.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
        # CuPy implementation - transfer and compute
        couplings_d = ctx.to_device(couplings)
        
        for i in range(n_steps):
            # RK4 on GPU using generic beta functions
            lambda_t, gamma_t, mu_t = couplings_d[0], couplings_d[1], couplings_d[2]
            
            # k1
            k1_l, k1_g, k1_m = _compute_beta_functions_generic(cp, lambda_t, gamma_t, mu_t)
            k1 = cp.array([k1_l, k1_g, k1_m])
            
            # k2
            mid1 = couplings_d + 0.5 * dt * k1
            k2_l, k2_g, k2_m = _compute_beta_functions_generic(cp, mid1[0], mid1[1], mid1[2])
            k2 = cp.array([k2_l, k2_g, k2_m])
            
            # k3
            mid2 = couplings_d + 0.5 * dt * k2
            k3_l, k3_g, k3_m = _compute_beta_functions_generic(cp, mid2[0], mid2[1], mid2[2])
            k3 = cp.array([k3_l, k3_g, k3_m])
            
            # k4
            end = couplings_d + dt * k3
            k4_l, k4_g, k4_m = _compute_beta_functions_generic(cp, end[0], end[1], end[2])
            k4 = cp.array([k4_l, k4_g, k4_m])
            
            couplings_d = couplings_d + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory[i + 1] = ctx.to_host(couplings_d)
            
    else:
        # NumPy fallback
        for i in range(n_steps):
            couplings = _rk4_step_numpy(couplings, dt)
            trajectory[i + 1] = couplings
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Check convergence to fixed point
    final = trajectory[-1]
    fixed_point = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])
    distance = np.linalg.norm(final - fixed_point)
    converged = distance < 1.0  # Within 1.0 of fixed point
    
    return {
        'trajectory': trajectory,
        't_values': t_values,
        'converged': converged,
        'final_couplings': final,
        'fixed_point': fixed_point,
        'fixed_point_distance': distance,
        'backend': ctx.backend.value,
        'is_gpu': ctx.is_gpu,
        'execution_time_ms': elapsed_ms,
        'n_steps': n_steps,
        'dt': dt,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.2.2-1.2.3',
    }


# =============================================================================
# GPU-Accelerated QNCD Matrix (Appendix A)
# =============================================================================

def gpu_qncd_matrix(
    states: NDArray[np.float64],
    ctx: Optional[GPUContext] = None,
) -> Dict[str, Any]:
    """
    GPU-accelerated pairwise QNCD distance matrix computation.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Appendix A
        QNCD (Quantum Normalized Compression Distance) metric on G_inf
        
        For algorithmic states represented by vectors in R^d, we use
        a proxy metric based on normalized L2 distance:
            d(x, y) = ||x - y|| / (||x|| + ||y||)
    
    Parameters
    ----------
    states : NDArray[np.float64]
        Array of shape (N, d) with N states of dimension d
    ctx : GPUContext, optional
        GPU context for device management
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'distance_matrix': symmetric NxN distance matrix
        - 'min_distance': minimum non-zero distance
        - 'max_distance': maximum distance
        - 'mean_distance': mean pairwise distance
        - 'backend': backend used
        - 'execution_time_ms': computation time
    
    Examples
    --------
    >>> states = np.random.rand(100, 4)  # 100 states in 4D
    >>> result = gpu_qncd_matrix(states)
    >>> print(f"Matrix shape: {result['distance_matrix'].shape}")
    """
    if ctx is None:
        ctx = GPUContext()
    
    start_time = time.perf_counter()
    
    N = states.shape[0]
    
    if ctx.backend == GPUBackend.JAX and _JAX_AVAILABLE:
        # Transfer to device
        states_d = ctx.to_device(states)
        norms = jnp.linalg.norm(states_d, axis=1)
        
        # Compute pairwise distances using broadcasting
        # diff[i,j] = ||states[i] - states[j]||
        diff = states_d[:, None, :] - states_d[None, :, :]
        dist_num = jnp.linalg.norm(diff, axis=2)
        
        # Normalized: d[i,j] = dist_num[i,j] / (norms[i] + norms[j])
        norm_sum = norms[:, None] + norms[None, :]
        distance_matrix = dist_num / jnp.where(norm_sum > 0, norm_sum, 1.0)
        
        distance_matrix = ctx.to_host(distance_matrix)
        
    elif ctx.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
        # Transfer to device
        states_d = ctx.to_device(states)
        norms = cp.linalg.norm(states_d, axis=1)
        
        # Pairwise distances
        diff = states_d[:, None, :] - states_d[None, :, :]
        dist_num = cp.linalg.norm(diff, axis=2)
        norm_sum = norms[:, None] + norms[None, :]
        distance_matrix = dist_num / cp.where(norm_sum > 0, norm_sum, 1.0)
        
        distance_matrix = ctx.to_host(distance_matrix)
        
    else:
        # NumPy fallback
        norms = np.linalg.norm(states, axis=1)
        diff = states[:, None, :] - states[None, :, :]
        dist_num = np.linalg.norm(diff, axis=2)
        norm_sum = norms[:, None] + norms[None, :]
        distance_matrix = dist_num / np.where(norm_sum > 0, norm_sum, 1.0)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Statistics (excluding diagonal zeros)
    upper_tri = distance_matrix[np.triu_indices(N, k=1)]
    
    return {
        'distance_matrix': distance_matrix,
        'min_distance': float(np.min(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'max_distance': float(np.max(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'mean_distance': float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'n_states': N,
        'backend': ctx.backend.value,
        'is_gpu': ctx.is_gpu,
        'execution_time_ms': elapsed_ms,
        'theoretical_reference': 'IRH v21.1 Manuscript Appendix A',
    }


# =============================================================================
# GPU-Accelerated Quaternion Operations (§1.1.1)
# =============================================================================

def gpu_quaternion_multiply(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    ctx: Optional[GPUContext] = None,
) -> Dict[str, Any]:
    """
    GPU-accelerated batch quaternion multiplication.
    
    Theoretical Reference:
        IRH v21.1 Manuscript §1.1.1
        Hamilton product for quaternions q = w + xi + yj + zk:
            q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2) +
                      (w1*x2 + x1*w2 + y1*z2 - z1*y2)i +
                      (w1*y2 - x1*z2 + y1*w2 + z1*x2)j +
                      (w1*z2 + x1*y2 - y1*x2 + z1*w2)k
    
    Parameters
    ----------
    q1 : NDArray[np.float64]
        First quaternion array of shape (N, 4) or (4,)
    q2 : NDArray[np.float64]
        Second quaternion array of shape (N, 4) or (4,)
    ctx : GPUContext, optional
        GPU context for device management
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'product': quaternion product array
        - 'backend': backend used
        - 'execution_time_ms': computation time
    
    Examples
    --------
    >>> q1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Two quaternions
    >>> q2 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    >>> result = gpu_quaternion_multiply(q1, q2)
    >>> print(result['product'])
    """
    if ctx is None:
        ctx = GPUContext()
    
    start_time = time.perf_counter()
    
    # Ensure 2D arrays
    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)
    
    # Extract components
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    if ctx.backend == GPUBackend.JAX and _JAX_AVAILABLE:
        # Transfer to device
        w1, x1, y1, z1 = [ctx.to_device(x) for x in [w1, x1, y1, z1]]
        w2, x2, y2, z2 = [ctx.to_device(x) for x in [w2, x2, y2, z2]]
        
        # Hamilton product
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        product = jnp.stack([w, x, y, z], axis=1)
        product = ctx.to_host(product)
        
    elif ctx.backend == GPUBackend.CUPY and _CUPY_AVAILABLE:
        # Transfer to device
        w1, x1, y1, z1 = [ctx.to_device(x) for x in [w1, x1, y1, z1]]
        w2, x2, y2, z2 = [ctx.to_device(x) for x in [w2, x2, y2, z2]]
        
        # Hamilton product
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        product = cp.stack([w, x, y, z], axis=1)
        product = ctx.to_host(product)
        
    else:
        # NumPy fallback
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        product = np.stack([w, x, y, z], axis=1)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        'product': product,
        'n_products': len(product),
        'backend': ctx.backend.value,
        'is_gpu': ctx.is_gpu,
        'execution_time_ms': elapsed_ms,
        'theoretical_reference': 'IRH v21.1 Manuscript §1.1.1',
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def benchmark_gpu_performance(
    n_evaluations: int = 10000,
    n_rg_steps: int = 1000,
) -> Dict[str, Any]:
    """
    Benchmark GPU acceleration performance.
    
    Parameters
    ----------
    n_evaluations : int
        Number of beta function evaluations to benchmark
    n_rg_steps : int
        Number of RG integration steps
        
    Returns
    -------
    Dict[str, Any]
        Benchmark results comparing backends
    """
    results: Dict[str, Any] = {
        'available_backends': [b.value for b in get_available_backends()],
        'gpu_available': is_gpu_available(),
        'benchmarks': {},
    }
    
    # Generate test data
    np.random.seed(42)
    couplings = np.random.uniform(40, 60, (n_evaluations, 3))
    initial = np.array([60.0, 110.0, 160.0])
    
    # Benchmark each available backend
    for backend in get_available_backends():
        ctx = GPUContext(backend=backend)
        
        # Beta functions benchmark
        beta_result = gpu_beta_functions(couplings, ctx=ctx)
        
        # RG flow benchmark (smaller for speed)
        rg_result = gpu_rg_flow_integration(
            initial, t_range=(-5, 0), n_steps=min(n_rg_steps, 500), ctx=ctx
        )
        
        results['benchmarks'][backend.value] = {
            'beta_functions_ms': beta_result['execution_time_ms'],
            'rg_flow_ms': rg_result['execution_time_ms'],
            'is_gpu': ctx.is_gpu,
        }
    
    return results
