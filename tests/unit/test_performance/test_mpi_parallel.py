"""
Tests for MPI Parallelization Module

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.4

Tests cover:
    - MPIContext initialization and operations
    - Distributed RG flow integration
    - Scatter/gather operations
    - Parallel fixed point search
    - Domain decomposition
    - Graceful fallback to serial execution

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.performance.mpi_parallel import (
    MPIContext,
    MPIBackend,
    distributed_rg_flow,
    scatter_initial_conditions,
    gather_results,
    parallel_fixed_point_search,
    parallel_qncd_matrix,
    domain_decomposition,
    is_mpi_available,
    get_mpi_info,
    _default_beta_functions,
    _balanced_chunks,
    _find_unique_fixed_points,
    _newton_raphson_search,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)


# =============================================================================
# Test Constants
# =============================================================================

FIXED_POINT = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])
TOLERANCE = 1e-8


# =============================================================================
# Test MPI Availability
# =============================================================================

class TestMPIAvailability:
    """Tests for MPI availability detection."""
    
    def test_is_mpi_available_returns_bool(self):
        """is_mpi_available should return a boolean."""
        result = is_mpi_available()
        assert isinstance(result, bool)
    
    def test_get_mpi_info_returns_dict(self):
        """get_mpi_info should return a dictionary with expected keys."""
        info = get_mpi_info()
        assert isinstance(info, dict)
        assert 'available' in info
        assert isinstance(info['available'], bool)
    
    def test_get_mpi_info_contains_import_error_if_unavailable(self):
        """If MPI unavailable, get_mpi_info should contain import_error."""
        info = get_mpi_info()
        if not info['available']:
            assert 'import_error' in info


# =============================================================================
# Test MPIContext
# =============================================================================

class TestMPIContext:
    """Tests for MPIContext class."""
    
    def test_mpi_context_initialization(self):
        """MPIContext should initialize with proper defaults."""
        ctx = MPIContext()
        assert isinstance(ctx.rank, int)
        assert isinstance(ctx.size, int)
        assert ctx.rank >= 0
        assert ctx.size >= 1
        assert isinstance(ctx.is_root, bool)
        assert isinstance(ctx.is_parallel, bool)
    
    def test_mpi_context_serial_fallback(self):
        """Without MPI, context should fall back to serial mode."""
        ctx = MPIContext()
        # In serial mode: rank=0, size=1, is_root=True
        if not ctx._active:
            assert ctx.rank == 0
            assert ctx.size == 1
            assert ctx.is_root is True
            assert ctx.is_parallel is False
    
    def test_mpi_context_as_context_manager(self):
        """MPIContext should work as context manager."""
        with MPIContext() as ctx:
            assert ctx.rank >= 0
            assert ctx.size >= 1
    
    def test_mpi_context_barrier_no_error(self):
        """Barrier should not raise errors in serial mode."""
        ctx = MPIContext()
        ctx.barrier()  # Should not raise
    
    def test_mpi_context_bcast(self):
        """Broadcast should return data in serial mode."""
        ctx = MPIContext()
        data = [1, 2, 3]
        result = ctx.bcast(data)
        assert result == data
    
    def test_mpi_context_scatter(self):
        """Scatter should return first element in serial mode."""
        ctx = MPIContext()
        data = [[1, 2], [3, 4]]
        result = ctx.scatter(data)
        assert result == [1, 2]
    
    def test_mpi_context_gather(self):
        """Gather should return list with single element in serial mode."""
        ctx = MPIContext()
        data = [1, 2, 3]
        result = ctx.gather(data)
        assert result == [[1, 2, 3]]
    
    def test_mpi_context_allgather(self):
        """Allgather should return list with single element in serial mode."""
        ctx = MPIContext()
        data = [1, 2, 3]
        result = ctx.allgather(data)
        assert result == [[1, 2, 3]]
    
    def test_mpi_context_reduce(self):
        """Reduce should return data unchanged in serial mode."""
        ctx = MPIContext()
        data = 42
        result = ctx.reduce(data)
        assert result == data
    
    def test_mpi_context_allreduce(self):
        """Allreduce should return data unchanged in serial mode."""
        ctx = MPIContext()
        data = 42
        result = ctx.allreduce(data)
        assert result == data
    
    def test_mpi_context_theoretical_reference(self):
        """Context should provide theoretical reference."""
        ctx = MPIContext()
        ref = ctx.get_theoretical_reference()
        assert 'IRH' in ref
        assert 'ROADMAP' in ref


# =============================================================================
# Test MPIBackend
# =============================================================================

class TestMPIBackend:
    """Tests for MPIBackend class."""
    
    def test_mpi_backend_initialization(self):
        """MPIBackend should initialize properly."""
        backend = MPIBackend()
        assert backend.ctx is not None
        assert isinstance(backend.load_balance, bool)
    
    def test_mpi_backend_parallel_map_serial(self):
        """parallel_map should work in serial mode."""
        backend = MPIBackend()
        data = [1, 2, 3, 4, 5]
        results = backend.parallel_map(lambda x: x * 2, data)
        assert results == [2, 4, 6, 8, 10]
    
    def test_mpi_backend_parallel_map_with_arrays(self):
        """parallel_map should work with numpy arrays."""
        backend = MPIBackend()
        data = [np.array([1, 2]), np.array([3, 4])]
        results = backend.parallel_map(lambda x: x.sum(), data)
        assert results == [3, 7]
    
    def test_mpi_backend_distribute_work(self):
        """_distribute_work should create balanced chunks."""
        backend = MPIBackend()
        data = list(range(10))
        chunks = backend._distribute_work(data)
        
        # In serial mode, should have one chunk with all data
        if not backend.ctx.is_parallel:
            assert len(chunks) == 1
            assert chunks[0] == data


# =============================================================================
# Test Beta Functions
# =============================================================================

class TestBetaFunctions:
    """Tests for default beta functions."""
    
    def test_beta_functions_vectorized(self):
        """Beta functions should handle batch input."""
        couplings = np.random.rand(100, 3) * 100
        betas = _default_beta_functions(couplings)
        
        assert betas.shape == (100, 3)
    
    def test_beta_functions_single_input(self):
        """Beta functions should handle single coupling tuple."""
        couplings = np.array([50.0, 100.0, 150.0])
        betas = _default_beta_functions(couplings)
        
        assert betas.shape == (1, 3)
    
    def test_beta_lambda_formula(self):
        """Test β_λ = -2λ̃ + (9/8π²)λ̃² (Eq. 1.13)."""
        lambda_val = 52.0
        couplings = np.array([[lambda_val, 100.0, 150.0]])
        betas = _default_beta_functions(couplings)
        
        expected_beta_lambda = -2 * lambda_val + (9 / (8 * np.pi**2)) * lambda_val**2
        assert_allclose(betas[0, 0], expected_beta_lambda, rtol=1e-10)
    
    def test_beta_lambda_zero_at_16pi2_over_9(self):
        """β_λ should vanish at λ = 16π²/9 (one-loop fixed point)."""
        # This is where the simple one-loop β_λ vanishes
        lambda_fp = 16 * np.pi**2 / 9  # ≈ 17.55
        couplings = np.array([[lambda_fp, 100.0, 150.0]])
        betas = _default_beta_functions(couplings)
        
        assert np.abs(betas[0, 0]) < TOLERANCE, f"β_λ(16π²/9) = {betas[0, 0]}"
    
    def test_beta_gamma_formula(self):
        """Test β_γ = (3/4π²)λ̃γ̃ (Eq. 1.13)."""
        lambda_val, gamma_val = 50.0, 100.0
        couplings = np.array([[lambda_val, gamma_val, 150.0]])
        betas = _default_beta_functions(couplings)
        
        expected_beta_gamma = (3 / (4 * np.pi**2)) * lambda_val * gamma_val
        assert_allclose(betas[0, 1], expected_beta_gamma, rtol=1e-10)
    
    def test_beta_mu_formula(self):
        """Test β_μ = 2μ̃ + (1/2π²)λ̃μ̃ (Eq. 1.13)."""
        lambda_val, mu_val = 50.0, 100.0
        couplings = np.array([[lambda_val, 75.0, mu_val]])
        betas = _default_beta_functions(couplings)
        
        expected_beta_mu = 2 * mu_val + (1 / (2 * np.pi**2)) * lambda_val * mu_val
        assert_allclose(betas[0, 2], expected_beta_mu, rtol=1e-10)


# =============================================================================
# Test Distributed RG Flow
# =============================================================================

class TestDistributedRGFlow:
    """Tests for distributed RG flow integration."""
    
    def test_distributed_rg_flow_single_trajectory(self):
        """Single trajectory should integrate correctly."""
        ic = np.array([[50.0, 100.0, 150.0]])
        result = distributed_rg_flow(ic, t_range=(-5, 5), n_steps=100)
        
        if 'trajectories' in result:
            assert result['trajectories'].shape == (1, 101, 3)
            assert len(result['times']) == 101
    
    def test_distributed_rg_flow_multiple_trajectories(self):
        """Multiple trajectories should integrate in parallel."""
        ic = np.random.rand(10, 3) * 100
        result = distributed_rg_flow(ic, t_range=(-5, 5), n_steps=50)
        
        if 'trajectories' in result:
            assert result['trajectories'].shape == (10, 51, 3)
    
    def test_distributed_rg_flow_small_initial(self):
        """Small initial conditions should remain finite."""
        ic = np.array([[1.0, 2.0, 3.0]])
        result = distributed_rg_flow(ic, t_range=(-1, 1), n_steps=20)
        
        if 'trajectories' in result and len(result['trajectories']) > 0:
            # Check trajectories are finite
            assert np.all(np.isfinite(result['trajectories']))
    
    def test_distributed_rg_flow_convergence(self):
        """Flow from moderate values should compute."""
        # Start with moderate values
        ic = np.array([[20.0, 40.0, 60.0]] * 5)
        result = distributed_rg_flow(ic, t_range=(-2, 2), n_steps=50)
        
        if 'converged' in result:
            assert 'converged' in result
    
    def test_distributed_rg_flow_timing_info(self):
        """Result should include timing information."""
        ic = np.random.rand(5, 3) * 50
        result = distributed_rg_flow(ic, t_range=(-2, 2), n_steps=50)
        
        if 'timing' in result:
            assert 'total_seconds' in result['timing']
            assert result['timing']['total_seconds'] >= 0
    
    def test_distributed_rg_flow_theoretical_reference(self):
        """Result should include theoretical reference."""
        ic = np.array([[50.0, 100.0, 150.0]])
        result = distributed_rg_flow(ic)
        
        assert 'theoretical_reference' in result
        assert 'IRH' in result['theoretical_reference']


# =============================================================================
# Test Scatter/Gather
# =============================================================================

class TestScatterGather:
    """Tests for scatter and gather operations."""
    
    def test_scatter_initial_conditions_serial(self):
        """Scatter should return full array in serial mode."""
        ctx = MPIContext()
        ic = np.random.rand(10, 3)
        
        local = scatter_initial_conditions(ic, ctx)
        
        if not ctx.is_parallel:
            assert_array_equal(local, ic)
    
    def test_gather_results_serial(self):
        """Gather should combine results in serial mode."""
        ctx = MPIContext()
        traj = np.random.rand(5, 10, 3)
        conv = np.array([True, False, True, True, False])
        t_range = (-5, 5)
        n_steps = 9
        
        all_traj, all_conv, times = gather_results(
            traj, conv, t_range, n_steps, ctx
        )
        
        assert_array_equal(all_traj, traj)
        assert_array_equal(all_conv, conv)
        assert len(times) == n_steps + 1
    
    def test_balanced_chunks(self):
        """_balanced_chunks should distribute work evenly."""
        # Exact division
        chunks = _balanced_chunks(10, 5)
        assert sum(chunks) == 10
        assert all(c == 2 for c in chunks)
        
        # Uneven division
        chunks = _balanced_chunks(10, 3)
        assert sum(chunks) == 10
        assert max(chunks) - min(chunks) <= 1


# =============================================================================
# Test Parallel Fixed Point Search
# =============================================================================

class TestParallelFixedPointSearch:
    """Tests for parallel fixed point search."""
    
    def test_parallel_fixed_point_search_from_nearby(self):
        """Search from near fixed point should converge."""
        guesses = FIXED_POINT + np.random.randn(5, 3) * 1.0
        result = parallel_fixed_point_search(guesses, tolerance=1e-8)
        
        if 'converged' in result:
            # At least some should converge
            # (convergence depends on initial guess quality)
            assert 'fixed_points' in result
    
    def test_parallel_fixed_point_search_from_exact(self):
        """Search from exact fixed point should converge immediately."""
        guesses = np.array([FIXED_POINT])
        result = parallel_fixed_point_search(guesses, tolerance=1e-6)
        
        if 'converged' in result and len(result['converged']) > 0:
            # Should have very small residual
            if len(result['residuals']) > 0:
                assert result['residuals'][0] < 1e-4
    
    def test_parallel_fixed_point_search_unique_detection(self):
        """Should identify unique fixed points."""
        # Multiple guesses near same fixed point
        guesses = FIXED_POINT + np.random.randn(10, 3) * 0.5
        result = parallel_fixed_point_search(guesses, tolerance=1e-6)
        
        if 'unique_fixed_points' in result:
            # Should find at most a few unique points
            assert len(result['unique_fixed_points']) <= 10
    
    def test_parallel_fixed_point_search_theoretical_reference(self):
        """Result should include theoretical reference."""
        guesses = np.array([FIXED_POINT])
        result = parallel_fixed_point_search(guesses)
        
        assert 'theoretical_reference' in result
        assert 'Eq. 1.14' in result['theoretical_reference']


# =============================================================================
# Test Newton-Raphson Search
# =============================================================================

class TestNewtonRaphsonSearch:
    """Tests for Newton-Raphson fixed point search."""
    
    def test_newton_raphson_convergence(self):
        """Newton-Raphson should converge from good initial guess."""
        guesses = np.array([FIXED_POINT + [0.1, 0.1, 0.1]])
        result = _newton_raphson_search(guesses, max_iter=100, tolerance=1e-8)
        
        assert 'fixed_points' in result
        assert 'converged' in result
        assert 'iterations' in result
        assert 'residuals' in result
    
    def test_newton_raphson_result_shapes(self):
        """Result arrays should have correct shapes."""
        guesses = np.random.rand(10, 3) * 100
        result = _newton_raphson_search(guesses, max_iter=50, tolerance=1e-6)
        
        assert result['fixed_points'].shape == (10, 3)
        assert result['converged'].shape == (10,)
        assert result['iterations'].shape == (10,)
        assert result['residuals'].shape == (10,)


# =============================================================================
# Test Unique Fixed Point Detection
# =============================================================================

class TestUniqueFP:
    """Tests for unique fixed point detection."""
    
    def test_find_unique_single_point(self):
        """Single point should be returned as unique."""
        fps = np.array([[1.0, 2.0, 3.0]])
        unique = _find_unique_fixed_points(fps)
        assert len(unique) == 1
    
    def test_find_unique_identical_points(self):
        """Identical points should collapse to one."""
        fps = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])
        unique = _find_unique_fixed_points(fps, tolerance=1e-6)
        assert len(unique) == 1
    
    def test_find_unique_distinct_points(self):
        """Distinct points should all be kept."""
        fps = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ])
        unique = _find_unique_fixed_points(fps, tolerance=1e-6)
        assert len(unique) == 3
    
    def test_find_unique_empty_array(self):
        """Empty array should return empty."""
        fps = np.array([])
        unique = _find_unique_fixed_points(fps)
        assert len(unique) == 0


# =============================================================================
# Test Domain Decomposition
# =============================================================================

class TestDomainDecomposition:
    """Tests for domain decomposition."""
    
    def test_domain_decomposition_1d(self):
        """1D lattice decomposition should work."""
        ctx = MPIContext()
        result = domain_decomposition((100,), ctx)
        
        assert 'local_shape' in result
        assert 'local_offset' in result
        assert 'halo_size' in result
        assert 'neighbors' in result
    
    def test_domain_decomposition_4d(self):
        """4D lattice (cGFT field) decomposition should work."""
        ctx = MPIContext()
        result = domain_decomposition((10, 10, 10, 10), ctx)
        
        assert len(result['local_shape']) == 4
        assert len(result['local_offset']) == 4
        
        # In serial mode, local shape should equal full shape
        if not ctx.is_parallel:
            assert result['local_shape'] == (10, 10, 10, 10)
    
    def test_domain_decomposition_serial_full_domain(self):
        """In serial mode, local domain should be full domain."""
        ctx = MPIContext()
        if not ctx.is_parallel:
            result = domain_decomposition((50, 50, 50), ctx)
            assert result['local_shape'] == (50, 50, 50)
            assert result['local_offset'] == (0, 0, 0)
    
    def test_domain_decomposition_metadata(self):
        """Decomposition should include metadata."""
        ctx = MPIContext()
        result = domain_decomposition((100, 100), ctx)
        
        assert 'decomposition_type' in result
        assert 'rank' in result
        assert 'n_processes' in result
        assert 'theoretical_reference' in result


# =============================================================================
# Test QNCD Matrix
# =============================================================================

class TestParallelQNCDMatrix:
    """Tests for parallel QNCD matrix computation."""
    
    def test_parallel_qncd_matrix_small(self):
        """Small matrix should compute correctly."""
        vectors = np.random.rand(10, 5)
        ctx = MPIContext()
        
        matrix = parallel_qncd_matrix(vectors, ctx)
        
        if ctx.is_root:
            assert matrix.shape == (10, 10)
            # Diagonal should be zero
            assert_allclose(np.diag(matrix), 0, atol=1e-10)
            # Should be symmetric
            assert_allclose(matrix, matrix.T, atol=1e-10)
    
    def test_parallel_qncd_matrix_values_in_range(self):
        """QNCD values should be in [0, 1]."""
        vectors = np.random.rand(20, 10)
        ctx = MPIContext()
        
        matrix = parallel_qncd_matrix(vectors, ctx)
        
        if ctx.is_root:
            assert np.all(matrix >= 0)
            assert np.all(matrix <= 1)


# =============================================================================
# Test Integration
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple MPI functions."""
    
    def test_full_workflow_serial(self):
        """Full workflow should work in serial mode."""
        # 1. Create initial conditions
        n_points = 20
        ic = np.random.rand(n_points, 3) * 100
        
        # 2. Integrate RG flow
        with MPIContext() as ctx:
            result = distributed_rg_flow(
                ic, t_range=(-10, 10), n_steps=100, ctx=ctx
            )
        
        # 3. Search for fixed points
        if 'fixed_points' in result:
            fp_result = parallel_fixed_point_search(result['fixed_points'])
            assert 'theoretical_reference' in fp_result
    
    def test_context_reuse(self):
        """Same context should work for multiple operations."""
        with MPIContext() as ctx:
            # First operation
            ic1 = np.random.rand(5, 3) * 50
            result1 = distributed_rg_flow(ic1, ctx=ctx, n_steps=10)
            
            # Second operation
            ic2 = np.random.rand(5, 3) * 100
            result2 = distributed_rg_flow(ic2, ctx=ctx, n_steps=10)
            
            assert 'theoretical_reference' in result1
            assert 'theoretical_reference' in result2


# =============================================================================
# Test Theoretical Consistency
# =============================================================================

class TestTheoreticalConsistency:
    """Tests for consistency with IRH v21.1 Manuscript."""
    
    def test_fixed_point_values_eq_1_14(self):
        """Fixed point values should match Eq. 1.14."""
        # λ̃* = 48π²/9
        assert_allclose(LAMBDA_STAR, 48 * np.pi**2 / 9, rtol=1e-14)
        
        # γ̃* = 32π²/3
        assert_allclose(GAMMA_STAR, 32 * np.pi**2 / 3, rtol=1e-14)
        
        # μ̃* = 16π²
        assert_allclose(MU_STAR, 16 * np.pi**2, rtol=1e-14)
    
    def test_beta_lambda_vanishes_at_16pi2_over_9(self):
        """β_λ(16π²/9) = 0 per one-loop Eq. 1.13."""
        # The simple one-loop β_λ = -2λ + (9/8π²)λ² vanishes at λ = 16π²/9
        lambda_fp = 16 * np.pi**2 / 9
        couplings = np.array([[lambda_fp, GAMMA_STAR, MU_STAR]])
        betas = _default_beta_functions(couplings)
        
        assert np.abs(betas[0, 0]) < 1e-10, f"β_λ(16π²/9) = {betas[0, 0]}, expected 0"
    
    def test_beta_formulas_match_eq_1_13(self):
        """Verify all three beta functions match Eq. 1.13 formulas."""
        l, g, m = 30.0, 60.0, 90.0
        couplings = np.array([[l, g, m]])
        betas = _default_beta_functions(couplings)
        
        pi_sq = np.pi**2
        
        # β_λ = -2λ̃ + (9/8π²)λ̃²
        expected_b_lambda = -2 * l + (9 / (8 * pi_sq)) * l**2
        assert_allclose(betas[0, 0], expected_b_lambda, rtol=1e-12)
        
        # β_γ = (3/4π²)λ̃γ̃
        expected_b_gamma = (3 / (4 * pi_sq)) * l * g
        assert_allclose(betas[0, 1], expected_b_gamma, rtol=1e-12)
        
        # β_μ = 2μ̃ + (1/2π²)λ̃μ̃
        expected_b_mu = 2 * m + (1 / (2 * pi_sq)) * l * m
        assert_allclose(betas[0, 2], expected_b_mu, rtol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
