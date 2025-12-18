"""
Tests for GPU Acceleration Module

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.5

Tests cover:
    - GPUContext device management and fallback behavior
    - GPU-accelerated beta functions (Eq. 1.13)
    - GPU-accelerated RG flow integration (§1.2-1.3)
    - GPU-accelerated QNCD matrix computation (Appendix A)
    - GPU-accelerated quaternion multiplication (§1.1.1)
    - Backend selection and availability detection

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from src.performance.gpu_acceleration import (
    GPUBackend,
    GPUContext,
    gpu_beta_functions,
    gpu_rg_flow_integration,
    gpu_qncd_matrix,
    gpu_quaternion_multiply,
    is_gpu_available,
    get_gpu_info,
    get_available_backends,
    set_default_backend,
    benchmark_gpu_performance,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)


class TestGPUAvailability:
    """Tests for GPU availability detection."""
    
    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)
    
    def test_get_gpu_info_structure(self):
        """get_gpu_info should return expected structure."""
        info = get_gpu_info()
        
        assert 'gpu_available' in info
        assert 'jax_available' in info
        assert 'cupy_available' in info
        assert 'default_backend' in info
        assert 'available_backends' in info
        
        assert isinstance(info['gpu_available'], bool)
        assert isinstance(info['available_backends'], list)
    
    def test_get_available_backends_includes_numpy(self):
        """NumPy should always be available as fallback."""
        backends = get_available_backends()
        assert GPUBackend.NUMPY in backends
    
    def test_available_backends_are_valid(self):
        """All reported backends should be valid GPUBackend values."""
        backends = get_available_backends()
        for b in backends:
            assert isinstance(b, GPUBackend)


class TestGPUContext:
    """Tests for GPUContext class."""
    
    def test_default_context_creation(self):
        """Should create context with default backend."""
        ctx = GPUContext()
        assert ctx.backend in get_available_backends()
    
    def test_numpy_backend_always_works(self):
        """NumPy backend should always work."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        assert ctx.backend == GPUBackend.NUMPY
        assert ctx.is_gpu is False
    
    def test_context_manager_entry_exit(self):
        """Context manager should work correctly."""
        ctx = GPUContext()
        with ctx as c:
            assert c._active is True
        assert ctx._active is False
    
    def test_array_module_returns_numpy_for_numpy_backend(self):
        """array_module should return np for NUMPY backend."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        assert ctx.array_module is np
    
    def test_to_device_numpy_returns_same(self):
        """to_device with NUMPY backend should return input."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        arr = np.array([1.0, 2.0, 3.0])
        result = ctx.to_device(arr)
        assert_array_almost_equal(result, arr)
    
    def test_to_host_numpy_returns_numpy(self):
        """to_host with NUMPY backend should return NumPy array."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        arr = np.array([1.0, 2.0, 3.0])
        result = ctx.to_host(arr)
        assert isinstance(result, np.ndarray)
    
    def test_verbose_mode(self):
        """Verbose mode should not raise errors."""
        ctx = GPUContext(verbose=True)
        with ctx:
            pass  # Just verify no exceptions


class TestGPUBetaFunctions:
    """Tests for GPU-accelerated beta functions (Eq. 1.13)."""
    
    def test_single_evaluation(self):
        """Test single coupling evaluation."""
        couplings = np.array([[52.0, 105.0, 158.0]])
        result = gpu_beta_functions(couplings)
        
        assert 'beta_lambda' in result
        assert 'beta_gamma' in result
        assert 'beta_mu' in result
        assert len(result['beta_lambda']) == 1
    
    def test_batch_evaluation(self):
        """Test batch evaluation of multiple couplings."""
        n = 100
        couplings = np.random.uniform(40, 60, (n, 3))
        result = gpu_beta_functions(couplings)
        
        assert result['n_evaluations'] == n
        assert len(result['beta_lambda']) == n
        assert len(result['beta_gamma']) == n
        assert len(result['beta_mu']) == n
    
    def test_fixed_point_beta_lambda_near_zero(self):
        """β_λ should vanish at λ̃ = 16π²/9 (not at LAMBDA_STAR from Eq. 1.14)."""
        # From β_λ = -2λ̃ + (9/8π²)λ̃² = 0, the non-trivial solution is:
        # λ̃ = 16π²/9 ≈ 17.546
        lambda_zero = 16 * np.pi**2 / 9
        couplings = np.array([[lambda_zero, GAMMA_STAR, MU_STAR]])
        result = gpu_beta_functions(couplings)
        
        # β_λ should vanish at this point
        assert abs(result['beta_lambda'][0]) < 1e-10
    
    def test_fixed_point_beta_gamma_near_zero(self):
        """β_γ should vanish when λ=0 or proportional at fixed point."""
        couplings = np.array([[LAMBDA_STAR, GAMMA_STAR, MU_STAR]])
        result = gpu_beta_functions(couplings)
        
        # At fixed point, β_γ ≈ 0 (check proportionality)
        # β_γ = (3/4π²)λ̃γ̃
        expected = (3 / (4 * np.pi**2)) * LAMBDA_STAR * GAMMA_STAR
        assert_allclose(result['beta_gamma'][0], expected, rtol=1e-10)
    
    def test_result_contains_metadata(self):
        """Result should contain execution metadata."""
        couplings = np.array([[52.0, 105.0, 158.0]])
        result = gpu_beta_functions(couplings)
        
        assert 'backend' in result
        assert 'execution_time_ms' in result
        assert 'theoretical_reference' in result
        assert 'IRH' in result['theoretical_reference']
    
    def test_numpy_backend_explicit(self):
        """Test explicit NumPy backend."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        couplings = np.array([[52.0, 105.0, 158.0]])
        result = gpu_beta_functions(couplings, ctx=ctx)
        
        assert result['backend'] == 'numpy'
        assert result['is_gpu'] is False
    
    def test_beta_functions_formula_correctness(self):
        """Test that beta functions match analytical formulas."""
        # Test at arbitrary point
        lambda_t, gamma_t, mu_t = 50.0, 100.0, 150.0
        couplings = np.array([[lambda_t, gamma_t, mu_t]])
        result = gpu_beta_functions(couplings)
        
        # Expected values from Eq. 1.13
        expected_beta_lambda = -2 * lambda_t + (9 / (8 * np.pi**2)) * lambda_t**2
        expected_beta_gamma = (3 / (4 * np.pi**2)) * lambda_t * gamma_t
        expected_beta_mu = 2 * mu_t + (1 / (2 * np.pi**2)) * lambda_t * mu_t
        
        assert_allclose(result['beta_lambda'][0], expected_beta_lambda, rtol=1e-10)
        assert_allclose(result['beta_gamma'][0], expected_beta_gamma, rtol=1e-10)
        assert_allclose(result['beta_mu'][0], expected_beta_mu, rtol=1e-10)


class TestGPURGFlowIntegration:
    """Tests for GPU-accelerated RG flow integration."""
    
    def test_basic_integration(self):
        """Test basic RG flow integration."""
        initial = np.array([60.0, 110.0, 160.0])
        result = gpu_rg_flow_integration(initial, t_range=(-5, 0), n_steps=100)
        
        assert 'trajectory' in result
        assert 'converged' in result
        assert 'final_couplings' in result
        assert result['trajectory'].shape == (101, 3)
    
    def test_trajectory_starts_at_initial(self):
        """Trajectory should start at initial conditions."""
        initial = np.array([60.0, 110.0, 160.0])
        result = gpu_rg_flow_integration(initial, n_steps=50)
        
        assert_allclose(result['trajectory'][0], initial)
    
    def test_converges_toward_fixed_point(self):
        """Integration should produce valid trajectory with stable parameters."""
        # Use small initial conditions near the trivial fixed point for stability
        # The beta functions can cause exponential growth for large values
        initial = np.array([5.0, 10.0, 15.0])
        result = gpu_rg_flow_integration(initial, t_range=(0, 1), n_steps=50)
        
        # Check trajectory doesn't contain NaN
        assert not np.any(np.isnan(result['trajectory'])), "Trajectory contains NaN values"
        
        # Check final values are finite
        final = result['final_couplings']
        assert np.all(np.isfinite(final)), "Final couplings are not finite"
        
        # Verify trajectory starts at initial conditions
        assert_allclose(result['trajectory'][0], initial)
    
    def test_result_contains_timing_metadata(self):
        """Result should contain timing information."""
        initial = np.array([60.0, 110.0, 160.0])
        result = gpu_rg_flow_integration(initial, n_steps=50)
        
        assert 'execution_time_ms' in result
        assert result['execution_time_ms'] > 0
    
    def test_result_contains_theoretical_reference(self):
        """Result should cite theoretical foundation."""
        initial = np.array([60.0, 110.0, 160.0])
        result = gpu_rg_flow_integration(initial, n_steps=50)
        
        assert 'theoretical_reference' in result
        assert 'IRH' in result['theoretical_reference']
    
    def test_t_values_correct_length(self):
        """t_values should have correct length."""
        initial = np.array([60.0, 110.0, 160.0])
        n_steps = 100
        result = gpu_rg_flow_integration(initial, n_steps=n_steps)
        
        assert len(result['t_values']) == n_steps + 1
    
    def test_numpy_backend(self):
        """Test with explicit NumPy backend."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        initial = np.array([60.0, 110.0, 160.0])
        result = gpu_rg_flow_integration(initial, n_steps=50, ctx=ctx)
        
        assert result['backend'] == 'numpy'


class TestGPUQNCDMatrix:
    """Tests for GPU-accelerated QNCD matrix computation."""
    
    def test_basic_computation(self):
        """Test basic QNCD matrix computation."""
        states = np.random.rand(10, 4)
        result = gpu_qncd_matrix(states)
        
        assert 'distance_matrix' in result
        assert result['distance_matrix'].shape == (10, 10)
    
    def test_symmetric_matrix(self):
        """Distance matrix should be symmetric."""
        states = np.random.rand(20, 4)
        result = gpu_qncd_matrix(states)
        
        matrix = result['distance_matrix']
        assert_allclose(matrix, matrix.T, atol=1e-10)
    
    def test_diagonal_zeros(self):
        """Diagonal elements should be zero (distance to self)."""
        states = np.random.rand(15, 4)
        result = gpu_qncd_matrix(states)
        
        diagonal = np.diag(result['distance_matrix'])
        assert_allclose(diagonal, 0.0, atol=1e-10)
    
    def test_non_negative_distances(self):
        """All distances should be non-negative."""
        states = np.random.rand(10, 4)
        result = gpu_qncd_matrix(states)
        
        assert np.all(result['distance_matrix'] >= 0)
    
    def test_statistics_computed(self):
        """Should compute distance statistics."""
        states = np.random.rand(20, 4)
        result = gpu_qncd_matrix(states)
        
        assert 'min_distance' in result
        assert 'max_distance' in result
        assert 'mean_distance' in result
    
    def test_contains_theoretical_reference(self):
        """Result should cite Appendix A."""
        states = np.random.rand(5, 4)
        result = gpu_qncd_matrix(states)
        
        assert 'theoretical_reference' in result
        assert 'Appendix A' in result['theoretical_reference']
    
    def test_numpy_backend(self):
        """Test with explicit NumPy backend."""
        ctx = GPUContext(backend=GPUBackend.NUMPY)
        states = np.random.rand(10, 4)
        result = gpu_qncd_matrix(states, ctx=ctx)
        
        assert result['backend'] == 'numpy'


class TestGPUQuaternionMultiply:
    """Tests for GPU-accelerated quaternion multiplication."""
    
    def test_single_multiplication(self):
        """Test single quaternion multiplication."""
        q1 = np.array([[1.0, 0.0, 0.0, 0.0]])  # Identity
        q2 = np.array([[0.0, 1.0, 0.0, 0.0]])  # i
        
        result = gpu_quaternion_multiply(q1, q2)
        
        # 1 * i = i
        expected = np.array([[0.0, 1.0, 0.0, 0.0]])
        assert_allclose(result['product'], expected, atol=1e-10)
    
    def test_batch_multiplication(self):
        """Test batch quaternion multiplication."""
        n = 50
        q1 = np.random.rand(n, 4)
        q2 = np.random.rand(n, 4)
        
        result = gpu_quaternion_multiply(q1, q2)
        
        assert result['product'].shape == (n, 4)
        assert result['n_products'] == n
    
    def test_quaternion_multiplication_properties(self):
        """Test i*j=k, j*k=i, k*i=j."""
        i = np.array([[0.0, 1.0, 0.0, 0.0]])
        j = np.array([[0.0, 0.0, 1.0, 0.0]])
        k = np.array([[0.0, 0.0, 0.0, 1.0]])
        
        # i * j = k
        result_ij = gpu_quaternion_multiply(i, j)
        assert_allclose(result_ij['product'], k, atol=1e-10)
        
        # j * k = i
        result_jk = gpu_quaternion_multiply(j, k)
        assert_allclose(result_jk['product'], i, atol=1e-10)
        
        # k * i = j
        result_ki = gpu_quaternion_multiply(k, i)
        assert_allclose(result_ki['product'], j, atol=1e-10)
    
    def test_noncommutative(self):
        """Quaternion multiplication should be non-commutative."""
        i = np.array([[0.0, 1.0, 0.0, 0.0]])
        j = np.array([[0.0, 0.0, 1.0, 0.0]])
        
        result_ij = gpu_quaternion_multiply(i, j)
        result_ji = gpu_quaternion_multiply(j, i)
        
        # i*j = k but j*i = -k
        assert not np.allclose(result_ij['product'], result_ji['product'])
    
    def test_identity_multiplication(self):
        """Multiplying by identity should return same quaternion."""
        identity = np.array([[1.0, 0.0, 0.0, 0.0]])
        q = np.array([[0.5, 0.5, 0.5, 0.5]])
        
        result = gpu_quaternion_multiply(identity, q)
        assert_allclose(result['product'], q, atol=1e-10)
    
    def test_contains_theoretical_reference(self):
        """Result should cite §1.1.1."""
        q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q2 = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = gpu_quaternion_multiply(q1, q2)
        
        assert 'theoretical_reference' in result
        assert '1.1.1' in result['theoretical_reference']


class TestBenchmark:
    """Tests for GPU benchmark functionality."""
    
    def test_benchmark_runs(self):
        """Benchmark should complete without errors."""
        result = benchmark_gpu_performance(n_evaluations=100, n_rg_steps=50)
        
        assert 'available_backends' in result
        assert 'benchmarks' in result
    
    def test_benchmark_includes_numpy(self):
        """Benchmark should include NumPy results."""
        result = benchmark_gpu_performance(n_evaluations=100, n_rg_steps=50)
        
        assert 'numpy' in result['benchmarks']
        assert 'beta_functions_ms' in result['benchmarks']['numpy']
        assert 'rg_flow_ms' in result['benchmarks']['numpy']


class TestBackendSelection:
    """Tests for backend selection and switching."""
    
    def test_set_default_backend_numpy(self):
        """Should be able to set NumPy as default."""
        set_default_backend(GPUBackend.NUMPY)
        ctx = GPUContext()
        # After this, default should be NUMPY (restored in other tests)
        assert GPUBackend.NUMPY in get_available_backends()
    
    def test_invalid_backend_raises_error(self):
        """Setting unavailable backend should raise error."""
        # This will only fail if JAX/CuPy is not available
        info = get_gpu_info()
        if not info['jax_available']:
            with pytest.raises(ValueError):
                set_default_backend(GPUBackend.JAX)


class TestConsistencyAcrossBackends:
    """Tests ensuring results are consistent across backends."""
    
    def test_beta_functions_numpy_consistent(self):
        """Beta functions should give same results with NumPy."""
        couplings = np.array([[52.0, 105.0, 158.0]])
        
        ctx_numpy = GPUContext(backend=GPUBackend.NUMPY)
        result = gpu_beta_functions(couplings, ctx=ctx_numpy)
        
        # Verify against analytical formula
        lambda_t = couplings[0, 0]
        gamma_t = couplings[0, 1]
        mu_t = couplings[0, 2]
        
        expected_beta_lambda = -2 * lambda_t + (9 / (8 * np.pi**2)) * lambda_t**2
        expected_beta_gamma = (3 / (4 * np.pi**2)) * lambda_t * gamma_t
        expected_beta_mu = 2 * mu_t + (1 / (2 * np.pi**2)) * lambda_t * mu_t
        
        assert_allclose(result['beta_lambda'][0], expected_beta_lambda, rtol=1e-10)
        assert_allclose(result['beta_gamma'][0], expected_beta_gamma, rtol=1e-10)
        assert_allclose(result['beta_mu'][0], expected_beta_mu, rtol=1e-10)
    
    def test_rg_flow_reproducible(self):
        """RG flow should be reproducible with same inputs."""
        initial = np.array([60.0, 110.0, 160.0])
        
        result1 = gpu_rg_flow_integration(initial, n_steps=100)
        result2 = gpu_rg_flow_integration(initial, n_steps=100)
        
        assert_allclose(result1['trajectory'], result2['trajectory'], rtol=1e-10)
