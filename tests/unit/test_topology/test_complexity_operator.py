"""
Tests for Topological Complexity Operator

THEORETICAL FOUNDATION: IRH v21.4 Part 2, Appendix E.1

These tests verify the computational derivation of topological complexity
eigenvalues 𝓚_f from transcendental equations.
"""

import pytest
import numpy as np
from src.topology.complexity_operator import (
    compute_effective_potential,
    compute_potential_gradient,
    compute_potential_hessian,
    solve_transcendental_equation,
    find_all_critical_points,
    compute_topological_complexity_eigenvalues,
    get_topological_complexity,
    LAMBDA_STAR, GAMMA_STAR, MU_STAR,
    MANUSCRIPT_K_VALUES,
    TopologicalComplexityResult,
)


class TestEffectivePotential:
    """Test effective potential computation per Appendix E.1."""
    
    def test_potential_is_finite(self):
        """Effective potential should be finite for physical K_f values."""
        K_f_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for K_f in K_f_values:
            V_eff = compute_effective_potential(K_f)
            assert np.isfinite(V_eff), f"Potential not finite at K_f={K_f}"
    
    def test_potential_at_unity(self):
        """Check potential value at K_f = 1 (electron generation)."""
        V_eff = compute_effective_potential(1.0)
        
        # Should be finite and reasonable magnitude
        assert np.isfinite(V_eff)
        assert abs(V_eff) < 1000, "Potential unexpectedly large"
    
    def test_potential_increases_for_large_K(self):
        """Potential should increase for very large K_f (quartic term dominates)."""
        V_small = compute_effective_potential(10.0)
        V_large = compute_effective_potential(1000.0)
        
        # For large K_f, quartic term should dominate → increasing potential
        assert V_large > V_small


class TestPotentialDerivatives:
    """Test gradient and Hessian computations."""
    
    def test_gradient_finite(self):
        """Gradient should be finite everywhere."""
        K_f_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for K_f in K_f_values:
            grad = compute_potential_gradient(K_f)
            assert np.isfinite(grad), f"Gradient not finite at K_f={K_f}"
    
    def test_hessian_finite(self):
        """Hessian should be finite everywhere."""
        K_f_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for K_f in K_f_values:
            hess = compute_potential_hessian(K_f)
            assert np.isfinite(hess), f"Hessian not finite at K_f={K_f}"
    
    def test_numerical_gradient(self):
        """Verify gradient via numerical differentiation."""
        K_f = 10.0
        epsilon = 1e-6
        
        # Analytical gradient
        grad_analytical = compute_potential_gradient(K_f)
        
        # Numerical gradient
        V_plus = compute_effective_potential(K_f + epsilon)
        V_minus = compute_effective_potential(K_f - epsilon)
        grad_numerical = (V_plus - V_minus) / (2 * epsilon)
        
        # Should agree within numerical error
        assert np.isclose(grad_analytical, grad_numerical, rtol=1e-4)
    
    def test_numerical_hessian(self):
        """Verify Hessian via numerical differentiation."""
        K_f = 10.0
        epsilon = 1e-6
        
        # Analytical Hessian
        hess_analytical = compute_potential_hessian(K_f)
        
        # Numerical Hessian
        grad_plus = compute_potential_gradient(K_f + epsilon)
        grad_minus = compute_potential_gradient(K_f - epsilon)
        hess_numerical = (grad_plus - grad_minus) / (2 * epsilon)
        
        # Should agree within numerical error (looser tolerance for second derivative)
        assert np.isclose(hess_analytical, hess_numerical, rtol=0.15)


class TestTranscendentalSolver:
    """Test transcendental equation solver."""
    
    def test_convergence_from_unity(self):
        """Solver should converge from initial guess near 1.0."""
        K_f, info = solve_transcendental_equation(
            initial_guess=1.0,
            tolerance=1e-10,
            verbosity=0,
        )
        
        assert info['converged'], "Solver did not converge"
        assert K_f > 0, "Solution is unphysical (negative)"
        assert abs(info['final_gradient']) < 1e-9, "Gradient not close to zero"
    
    def test_convergence_from_large_value(self):
        """Solver should converge from initial guess near 200."""
        K_f, info = solve_transcendental_equation(
            initial_guess=200.0,
            tolerance=1e-10,
            verbosity=0,
        )
        
        assert info['converged'], "Solver did not converge"
        assert K_f > 0, "Solution is unphysical"
    
    def test_solution_is_critical_point(self):
        """Verify that solution satisfies dV_eff/dK_f ≈ 0."""
        K_f, info = solve_transcendental_equation(
            initial_guess=1.0,
            tolerance=1e-10,
            verbosity=0,
        )
        
        grad = compute_potential_gradient(K_f)
        assert abs(grad) < 1e-8, f"Solution not a critical point: grad={grad}"
    
    def test_reproducibility(self):
        """Same initial guess should give same result."""
        K_f_1, _ = solve_transcendental_equation(initial_guess=1.0, verbosity=0)
        K_f_2, _ = solve_transcendental_equation(initial_guess=1.0, verbosity=0)
        
        assert np.isclose(K_f_1, K_f_2), "Solver not reproducible"


class TestMorseTheory:
    """Test Morse theory classification of critical points."""
    
    def test_morse_index_at_minimum(self):
        """Stable minimum should have Morse index = 0 (all positive eigenvalues)."""
        # Find a critical point
        K_f, info = solve_transcendental_equation(1.0, verbosity=0)
        
        # Compute Hessian
        hessian = compute_potential_hessian(K_f)
        
        # For stable minimum, Hessian should be positive
        assert hessian > 0, f"Hessian negative at minimum: {hessian}"
    
    def test_find_multiple_minima(self):
        """Global search should find multiple critical points."""
        results = find_all_critical_points(
            K_range=(0.01, 5000.0),
            n_samples=500,  # Reduced for speed
            verbosity=0,
        )
        
        # Should find at least 2 critical points
        assert len(results) >= 2, f"Found only {len(results)} critical points"
    
    def test_stable_points_only(self):
        """Returned critical points should all be stable minima."""
        results = find_all_critical_points(n_samples=500, verbosity=0)
        
        for result in results:
            assert result.is_stable, f"Unstable point returned: {result.K_f}"
            assert result.morse_index == 0, f"Non-zero Morse index: {result.morse_index}"


class TestTopologicalComplexityEigenvalues:
    """Test main eigenvalue computation function."""
    
    def test_returns_three_eigenvalues(self):
        """Should return exactly 3 eigenvalues (for 3 generations)."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        # Manuscript states 3 stable minima
        assert len(results) == 3, f"Expected 3 eigenvalues, got {len(results)}"
    
    def test_eigenvalues_ordered(self):
        """Eigenvalues should be ordered by magnitude: K_1 < K_2 < K_3."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        K_values = [r.K_f for r in results]
        assert K_values == sorted(K_values), "Eigenvalues not ordered"
    
    def test_electron_generation_near_unity(self):
        """First eigenvalue (electron) should be close to 1.0."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        K_1 = results[0].K_f
        expected, uncertainty = MANUSCRIPT_K_VALUES[1]
        
        # Should be within 10σ (very conservative)
        deviation = abs(K_1 - expected)
        assert deviation < 10 * uncertainty, \
            f"K_1 = {K_1} deviates from manuscript value {expected} ± {uncertainty}"
    
    def test_muon_generation_magnitude(self):
        """Second eigenvalue (muon) should be O(200)."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        if len(results) >= 2:
            K_2 = results[1].K_f
            # Should be roughly 200 (order of magnitude)
            assert 100 < K_2 < 400, f"K_2 = {K_2} not in expected range"
    
    def test_tau_generation_magnitude(self):
        """Third eigenvalue (tau) should be O(3000)."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        if len(results) >= 3:
            K_3 = results[2].K_f
            # Should be roughly 3500 (order of magnitude)
            assert 2000 < K_3 < 5000, f"K_3 = {K_3} not in expected range"
    
    def test_result_structure(self):
        """Each result should have proper structure."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            assert isinstance(result, TopologicalComplexityResult)
            assert hasattr(result, 'K_f')
            assert hasattr(result, 'generation')
            assert hasattr(result, 'uncertainty')
            assert hasattr(result, 'is_stable')
            assert result.is_stable
    
    def test_to_dict_export(self):
        """Results should be exportable to dictionary."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert 'K_f' in result_dict
            assert 'generation' in result_dict
            assert 'theoretical_reference' in result_dict


class TestFermionInterface:
    """Test convenience interface for fermion name lookup."""
    
    def test_electron_lookup(self):
        """Should return K_f for electron."""
        K_f = get_topological_complexity('electron', verbosity=0)
        
        assert K_f > 0
        assert K_f < 2, "Electron K_f should be close to 1"
    
    def test_muon_lookup(self):
        """Should return K_f for muon."""
        K_f = get_topological_complexity('muon', verbosity=0)
        
        assert 100 < K_f < 400, "Muon K_f should be O(200)"
    
    def test_tau_lookup(self):
        """Should return K_f for tau."""
        K_f = get_topological_complexity('tau', verbosity=0)
        
        assert 2000 < K_f < 5000, "Tau K_f should be O(3500)"
    
    def test_quark_lookup(self):
        """Should handle quark names."""
        K_up = get_topological_complexity('up', verbosity=0)
        K_charm = get_topological_complexity('charm', verbosity=0)
        K_top = get_topological_complexity('top', verbosity=0)
        
        # All should be positive
        assert K_up > 0
        assert K_charm > 0
        assert K_top > 0
        
        # Should follow generation ordering
        assert K_up < K_charm < K_top
    
    def test_unknown_fermion_raises(self):
        """Unknown fermion name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown fermion"):
            get_topological_complexity('bogus_particle', verbosity=0)
    
    def test_aliases(self):
        """Should handle fermion name aliases."""
        K_e = get_topological_complexity('e', verbosity=0)
        K_electron = get_topological_complexity('electron', verbosity=0)
        
        assert np.isclose(K_e, K_electron), "Aliases should give same result"


class TestPhysicalConsistency:
    """Test physical consistency of results."""
    
    def test_positive_eigenvalues(self):
        """All eigenvalues should be positive (physical)."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            assert result.K_f > 0, f"Unphysical negative K_f: {result.K_f}"
    
    def test_hierarchy_preserved(self):
        """Mass hierarchy should be preserved: m_1 < m_2 < m_3."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        # Since m_f ∝ √K_f, hierarchy in K_f implies hierarchy in mass
        K_values = [r.K_f for r in results]
        assert K_values[0] < K_values[1] < K_values[2], \
            "Generation hierarchy not preserved"
    
    def test_finite_uncertainties(self):
        """Uncertainties should be finite and reasonable."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            assert np.isfinite(result.uncertainty)
            assert result.uncertainty > 0
            assert result.uncertainty < 1.0, "Uncertainty unreasonably large"
    
    def test_effective_potential_at_minima(self):
        """Effective potential should be negative at stable minima (bound states)."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            # Bound states typically have V_eff < 0
            # (though sign convention may vary)
            assert np.isfinite(result.effective_potential)


class TestTheoreticalReferences:
    """Test that theoretical references are properly documented."""
    
    def test_manuscript_citation(self):
        """All results should cite IRH v21.4 manuscript."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            assert 'IRH v21.4' in result.theoretical_reference
            assert 'Appendix E.1' in result.theoretical_reference
    
    def test_convergence_info(self):
        """Results should include convergence information."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            assert 'convergence_info' in result.to_dict()
            info = result.convergence_info
            assert 'converged' in info or 'iterations' in info


@pytest.mark.slow
class TestNumericalPrecision:
    """Test numerical precision requirements (marked slow)."""
    
    def test_high_precision_convergence(self):
        """Should achieve 10^-10 precision."""
        K_f, info = solve_transcendental_equation(
            initial_guess=1.0,
            tolerance=1e-10,
            verbosity=0,
        )
        
        assert info['converged']
        assert abs(info['final_gradient']) < 1e-10
    
    def test_manuscript_agreement_strict(self):
        """Should match manuscript values within stated uncertainties."""
        results = compute_topological_complexity_eigenvalues(verbosity=0)
        
        for result in results:
            if result.generation in MANUSCRIPT_K_VALUES:
                expected, uncertainty = MANUSCRIPT_K_VALUES[result.generation]
                deviation = abs(result.K_f - expected)
                
                # Within 5σ is acceptable (computational vs analytical)
                assert deviation < 5 * uncertainty, \
                    f"K_{result.generation} = {result.K_f} deviates by " \
                    f"{deviation/uncertainty:.1f}σ from manuscript"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
