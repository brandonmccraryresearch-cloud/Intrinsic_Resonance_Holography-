"""
Unit Tests for Fixed Points Module

THEORETICAL FOUNDATION: IRH21.md §1.2.3, Eq. 1.14

Tests validate:
1. CosmicFixedPoint class implementation
2. Analytical and numerical fixed point finding
3. Fixed point verification
4. Universal exponent computation
5. Stability analysis

Note: The analytical fixed-point values from Eq. 1.14 (λ̃*=48π²/9, γ̃*=32π²/3, μ̃*=16π²)
are the manuscript's predictions. These serve as reference values even though the
simplified beta function formulas (Eq. 1.13) may have numerical fixed points at
different locations due to truncation effects.

Authors: IRH Computational Framework Team
"""

import math
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
import sys
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.rg_flow.fixed_points import (
    CosmicFixedPoint,
    FixedPointResult,
    find_fixed_point,
    verify_fixed_point,
    compute_universal_exponent,
    compute_stability_matrix,
    analyze_fixed_point_stability,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
    C_H_RATIO,
    C_H_SPECTRAL,
)


class TestCosmicFixedPointClass:
    """Test CosmicFixedPoint dataclass."""
    
    def test_analytical_constructor(self):
        """Analytical fixed point should match Eq. 1.14."""
        fp = CosmicFixedPoint.analytical()
        
        assert fp.lambda_star == pytest.approx(48 * math.pi**2 / 9, rel=1e-15)
        assert fp.gamma_star == pytest.approx(32 * math.pi**2 / 3, rel=1e-15)
        assert fp.mu_star == pytest.approx(16 * math.pi**2, rel=1e-15)
    
    def test_verify_returns_dict(self):
        """Verify method should return structured dict."""
        fp = CosmicFixedPoint.analytical()
        result = fp.verify()
        
        assert 'is_fixed_point' in result
        assert 'beta_values' in result
        assert 'max_beta' in result
        assert 'tolerance' in result
    
    def test_compute_C_H_ratio(self):
        """C_H ratio should be 0.75."""
        fp = CosmicFixedPoint.analytical()
        c_h = fp.compute_C_H(method='ratio')
        
        assert c_h == pytest.approx(0.75, rel=1e-10)
    
    def test_compute_C_H_spectral(self):
        """C_H spectral should be 0.045935703598."""
        fp = CosmicFixedPoint.analytical()
        c_h = fp.compute_C_H(method='spectral')
        
        assert c_h == pytest.approx(0.045935703598, rel=1e-10)
    
    def test_to_dict(self):
        """to_dict should return complete dictionary."""
        fp = CosmicFixedPoint.analytical()
        d = fp.to_dict()
        
        assert 'lambda_star' in d
        assert 'gamma_star' in d
        assert 'mu_star' in d
        assert 'C_H_ratio' in d
        assert 'C_H_spectral' in d
    
    def test_repr(self):
        """__repr__ should be informative."""
        fp = CosmicFixedPoint.analytical()
        repr_str = repr(fp)
        
        assert 'CosmicFixedPoint' in repr_str


class TestFindFixedPoint:
    """Test find_fixed_point function."""
    
    def test_analytical_method(self):
        """Analytical method should return Eq. 1.14 values."""
        fp = find_fixed_point(method='analytical')
        
        assert fp.lambda_star == LAMBDA_STAR
        assert fp.gamma_star == GAMMA_STAR
        assert fp.mu_star == MU_STAR
    
    def test_numerical_method_converges(self):
        """Numerical method should find some fixed point."""
        fp = find_fixed_point(method='numerical')
        
        # Should return finite values
        assert np.isfinite(fp.lambda_star)
        assert np.isfinite(fp.gamma_star)
        assert np.isfinite(fp.mu_star)
    
    def test_default_is_analytical(self):
        """Default method should be analytical."""
        fp_default = find_fixed_point()
        fp_analytical = find_fixed_point(method='analytical')
        
        assert fp_default.lambda_star == fp_analytical.lambda_star
    
    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            find_fixed_point(method='invalid')


class TestVerifyFixedPoint:
    """Test verify_fixed_point function."""
    
    def test_verify_returns_result(self):
        """verify_fixed_point should return FixedPointResult."""
        result = verify_fixed_point(10.0, 50.0, 100.0)
        
        assert isinstance(result, FixedPointResult)
    
    def test_verify_non_fixed_point(self):
        """Arbitrary values should not verify as fixed point."""
        result = verify_fixed_point(10.0, 50.0, 100.0)
        
        # These arbitrary values should not have vanishing betas
        assert not result.is_fixed_point
    
    def test_result_structure(self):
        """FixedPointResult should have correct structure."""
        result = verify_fixed_point(10.0, 50.0, 100.0)
        
        assert hasattr(result, 'lambda_star')
        assert hasattr(result, 'gamma_star')
        assert hasattr(result, 'mu_star')
        assert hasattr(result, 'is_fixed_point')
        assert hasattr(result, 'beta_values')
        assert hasattr(result, 'C_H')


class TestComputeUniversalExponent:
    """Test compute_universal_exponent function."""
    
    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        result = compute_universal_exponent()
        
        assert 'computed_ratio' in result
        assert 'analytical_spectral' in result
        assert 'ratio_value' in result
        assert 'spectral_value' in result
        assert 'note' in result
    
    def test_ratio_value(self):
        """Ratio value should be 0.75."""
        result = compute_universal_exponent()
        
        assert result['ratio_value'] == pytest.approx(0.75, rel=1e-10)
    
    def test_spectral_value(self):
        """Spectral value should be 0.045935703598."""
        result = compute_universal_exponent()
        
        assert result['spectral_value'] == pytest.approx(0.045935703598, rel=1e-10)


class TestStabilityAnalysis:
    """Test stability analysis functions."""
    
    def test_stability_matrix_shape(self):
        """Stability matrix should be 3×3."""
        M = compute_stability_matrix()
        assert M.shape == (3, 3)
    
    def test_stability_matrix_finite(self):
        """Stability matrix should have finite entries."""
        M = compute_stability_matrix()
        
        assert np.isfinite(M).all()
    
    def test_ir_attractiveness(self):
        """Fixed point should be IR attractive (positive eigenvalues)."""
        result = analyze_fixed_point_stability()
        
        assert result['is_ir_attractive']
    
    def test_positive_eigenvalues(self):
        """All eigenvalues should have positive real parts."""
        result = analyze_fixed_point_stability()
        
        for ev in result['eigenvalues']:
            assert ev.real > 0, f"Non-positive eigenvalue: {ev}"


class TestConstants:
    """Test module constants."""
    
    def test_lambda_star(self):
        """LAMBDA_STAR should equal 48π²/9."""
        assert LAMBDA_STAR == pytest.approx(48 * math.pi**2 / 9, rel=1e-15)
    
    def test_gamma_star(self):
        """GAMMA_STAR should equal 32π²/3."""
        assert GAMMA_STAR == pytest.approx(32 * math.pi**2 / 3, rel=1e-15)
    
    def test_mu_star(self):
        """MU_STAR should equal 16π²."""
        assert MU_STAR == pytest.approx(16 * math.pi**2, rel=1e-15)
    
    def test_c_h_ratio(self):
        """C_H_RATIO should be 0.75."""
        assert C_H_RATIO == pytest.approx(0.75, rel=1e-10)
    
    def test_c_h_spectral(self):
        """C_H_SPECTRAL should be 0.045935703598."""
        assert C_H_SPECTRAL == pytest.approx(0.045935703598, rel=1e-10)


class TestTheoreticalGrounding:
    """Verify theoretical references."""
    
    def test_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.rg_flow import fixed_points
        assert 'IRH21.md' in fixed_points.__theoretical_foundation__
        assert 'Eq. 1.14' in fixed_points.__theoretical_foundation__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
