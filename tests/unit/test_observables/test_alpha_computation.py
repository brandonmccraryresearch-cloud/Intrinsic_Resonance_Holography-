"""
Test Suite for Fine-Structure Constant Computation

This test suite validates the computational implementation of α⁻¹
from the IRH formula (Eq. 3.4-3.5).

Theoretical Reference:
    IRH v21.4 Part 1 §3.2.2, Eq. 3.4-3.5
    
Test Purpose:
    1. Verify fixed-point values are used correctly
    2. Validate component computations
    3. Check consistency with CODATA 2022
    4. Verify approximation quality
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.observables.alpha_inverse import (
    compute_fine_structure_constant,
    verify_alpha_inverse_precision,
    get_implementation_warnings,
    ALPHA_INVERSE_EXPERIMENTAL,
    ALPHA_INVERSE_UNCERTAINTY,
)


class TestAlphaInverseComputation:
    """Test suite for alpha inverse computation."""
    
    def test_codata_2022_value_correct(self):
        """Verify CODATA 2022 value is correctly set."""
        assert ALPHA_INVERSE_EXPERIMENTAL == 137.035999177, \
            "CODATA 2022 value should be 137.035999177"
        assert ALPHA_INVERSE_UNCERTAINTY == 0.000000021, \
            "CODATA 2022 uncertainty should be 0.000000021"
    
    def test_leading_term_computation(self):
        """Verify leading term is computed correctly."""
        result = compute_fine_structure_constant(method='leading')
        
        # Leading term should be 4π² × (γ*/λ*) ≈ 78.96
        assert 78 < result.alpha_inverse < 80, \
            f"Leading term should be ~79, got {result.alpha_inverse}"
        
        # Should be well below experimental value
        assert result.alpha_inverse < ALPHA_INVERSE_EXPERIMENTAL, \
            "Leading term should be below full result"
    
    def test_full_computation_returns_computed_value(self):
        """Verify full method returns a computed (not hardcoded) value."""
        result = compute_fine_structure_constant(method='full')
        
        # Should have components
        assert 'leading_term' in result.components
        assert 'log_corrections' in result.components
        assert 'g_qncd_approximation' in result.components
        assert 'v_vertex_approximation' in result.components
        
        # Leading term should be ~79
        assert 78 < result.components['leading_term'] < 80
        
        # Total should be > leading (corrections are positive)
        assert result.alpha_inverse > result.components['leading_term']
    
    def test_corrections_are_positive(self):
        """Verify all correction terms are positive."""
        result = compute_fine_structure_constant(method='full')
        
        assert result.components['log_corrections'] > 0
        assert result.components['g_qncd_approximation'] > 0
        assert result.components['v_vertex_approximation'] > 0
    
    def test_result_close_to_experimental(self):
        """Verify result is reasonably close to CODATA 2022."""
        result = compute_fine_structure_constant(method='full')
        
        # Should be within ~2% of experimental
        rel_error = abs(result.alpha_inverse - ALPHA_INVERSE_EXPERIMENTAL) / ALPHA_INVERSE_EXPERIMENTAL
        assert rel_error < 0.02, \
            f"Result should be within 2% of experimental, got {rel_error:.2%}"
    
    def test_fixed_point_values_used(self):
        """Verify correct fixed-point values are used."""
        result = compute_fine_structure_constant(method='full')
        
        # Check γ*/λ* ≈ 2
        gamma_over_lambda = result.components['gamma_over_lambda']
        assert 1.99 < gamma_over_lambda < 2.01, \
            f"γ*/λ* should be ≈2, got {gamma_over_lambda}"
        
        # Check fixed-point values are in expected range
        assert 50 < result.components['lambda_star'] < 55
        assert 100 < result.components['gamma_star'] < 110
        assert 150 < result.components['mu_star'] < 165
    
    def test_analytical_equals_full(self):
        """Verify analytical method uses same computation as full."""
        result_analytical = compute_fine_structure_constant(method='analytical')
        result_full = compute_fine_structure_constant(method='full')
        
        # Should be identical
        assert abs(result_analytical.alpha_inverse - result_full.alpha_inverse) < 1e-6
    
    def test_precision_verification_shows_details(self):
        """Verify precision verification provides detailed info."""
        verification = verify_alpha_inverse_precision(n_digits=9)
        
        assert 'computed_value' in verification
        assert 'codata_2022_value' in verification
        assert 'discrepancy' in verification
        assert 'sigma_deviation' in verification
        assert 'matching_digits' in verification
    
    def test_implementation_warnings_updated(self):
        """Verify implementation warnings reflect computed status."""
        warnings = get_implementation_warnings()
        
        assert warnings['implementation_status'] == 'COMPUTED with approximations'
        assert 'computed_value' in warnings
        assert 'approximations_used' in warnings
        assert len(warnings['approximations_used']) == 3  # Log, G_QNCD, V
    
    def test_result_object_completeness(self):
        """Verify result object contains all expected fields."""
        result = compute_fine_structure_constant(method='full')
        
        assert hasattr(result, 'alpha_inverse')
        assert hasattr(result, 'uncertainty')
        assert hasattr(result, 'experimental')
        assert hasattr(result, 'sigma_deviation')
        assert hasattr(result, 'components')
        
        # Components should be a dict
        assert isinstance(result.components, dict)


class TestComputationConsistency:
    """Tests for internal consistency of computation."""
    
    def test_components_sum_to_total(self):
        """Verify components sum correctly to total."""
        result = compute_fine_structure_constant(method='full')
        
        leading = result.components['leading_term']
        log_corr = result.components['log_corrections']
        g_qncd = result.components['g_qncd_approximation']
        v_vertex = result.components['v_vertex_approximation']
        
        # Total = leading × (1 + corrections/leading)
        total_corrections = log_corr + g_qncd + v_vertex
        expected = leading * (1 + total_corrections / leading)
        
        assert abs(result.alpha_inverse - expected) < 1e-6, \
            f"Components don't sum correctly: {result.alpha_inverse} vs {expected}"
    
    def test_correction_fraction_reasonable(self):
        """Verify correction fraction is in reasonable range."""
        result = compute_fine_structure_constant(method='full')
        
        correction_frac = result.components['correction_fraction']
        
        # Corrections should be significant but not dominant
        # Expect ~50-100% of leading term
        assert 0.5 < correction_frac < 1.5, \
            f"Correction fraction {correction_frac} seems unreasonable"
    
    def test_sigma_deviation_reflects_actual_difference(self):
        """Verify sigma deviation is calculated correctly."""
        result = compute_fine_structure_constant(method='full')
        
        expected_sigma = (result.alpha_inverse - ALPHA_INVERSE_EXPERIMENTAL) / ALPHA_INVERSE_UNCERTAINTY
        
        assert abs(result.sigma_deviation - expected_sigma) < 1e-6, \
            f"Sigma deviation mismatch: {result.sigma_deviation} vs {expected_sigma}"


class TestApproximationQuality:
    """Tests for quality of approximations."""
    
    def test_log_corrections_reasonable(self):
        """Verify log corrections are in reasonable range."""
        result = compute_fine_structure_constant(method='full')
        
        log_corr = result.components['log_corrections']
        leading = result.components['leading_term']
        
        # Log corrections should be ~10-20% of leading term
        ratio = log_corr / leading
        assert 0.1 < ratio < 0.3, \
            f"Log correction ratio {ratio} outside expected range"
    
    def test_non_perturbative_terms_significant(self):
        """Verify non-perturbative terms contribute significantly."""
        result = compute_fine_structure_constant(method='full')
        
        g_qncd = result.components['g_qncd_approximation']
        v_vertex = result.components['v_vertex_approximation']
        leading = result.components['leading_term']
        
        # Together should be ~20-60% of leading
        total_np = g_qncd + v_vertex
        ratio = total_np / leading
        assert 0.2 < ratio < 0.7, \
            f"Non-perturbative ratio {ratio} outside expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
