"""
Unit Tests for Beta Functions Module

THEORETICAL FOUNDATION: IRH21.md §1.2.2, Eq. 1.13

Tests validate:
1. BetaFunctions class implementation
2. Module-level beta function helpers
3. Jacobian matrix computation
4. Formula correctness

Note: The analytical fixed-point values from Eq. 1.14 (λ̃*=48π²/9, γ̃*=32π²/3, μ̃*=16π²)
and the beta function formulas from Eq. 1.13 are computed independently. The numerical
fixed point where β=0 is found by solve_ivp/fsolve, not by direct evaluation at Eq. 1.14 values.

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

from src.rg_flow.beta_functions import (
    BetaFunctions,
    beta_lambda,
    beta_gamma,
    beta_mu,
    compute_all_betas,
)


class TestBetaFunctionsClass:
    """Test BetaFunctions dataclass implementation."""
    
    def test_instantiation(self):
        """BetaFunctions should instantiate correctly."""
        beta = BetaFunctions()
        assert beta.precision == 15
        
        beta_custom = BetaFunctions(precision=10)
        assert beta_custom.precision == 10
    
    def test_beta_lambda_formula(self):
        """Verify β_λ = -2λ̃ + (9/8π²)λ̃² (Eq. 1.13)."""
        beta = BetaFunctions()
        l = 10.0
        expected = -2 * l + (9 / (8 * math.pi**2)) * l**2
        result = beta.beta_lambda(l)
        assert result == pytest.approx(expected, rel=1e-15)
    
    def test_beta_lambda_zero_point(self):
        """β_λ should vanish at λ̃ = 16π²/9."""
        beta = BetaFunctions()
        # From β_λ = -2λ̃ + (9/8π²)λ̃² = 0, the non-trivial solution is:
        # λ̃ = 16π²/9
        lambda_zero = 16 * math.pi**2 / 9
        result = beta.beta_lambda(lambda_zero)
        assert abs(result) < 1e-10, f"β_λ = {result} at λ̃ = 16π²/9"
    
    def test_beta_gamma_formula(self):
        """Verify β_γ = (3/4π²)λ̃γ̃ (Eq. 1.13)."""
        beta = BetaFunctions()
        l, g = 10.0, 50.0
        expected = (3 / (4 * math.pi**2)) * l * g
        result = beta.beta_gamma(l, g)
        assert result == pytest.approx(expected, rel=1e-15)
    
    def test_beta_gamma_vanishes_at_origin(self):
        """β_γ should vanish when either λ or γ is zero."""
        beta = BetaFunctions()
        assert beta.beta_gamma(0.0, 50.0) == 0.0
        assert beta.beta_gamma(10.0, 0.0) == 0.0
    
    def test_beta_mu_formula(self):
        """Verify β_μ = 2μ̃ + (1/2π²)λ̃μ̃ (Eq. 1.13)."""
        beta = BetaFunctions()
        l, g, m = 10.0, 50.0, 100.0
        expected = 2 * m + (1 / (2 * math.pi**2)) * l * m
        result = beta.beta_mu(l, g, m)
        assert result == pytest.approx(expected, rel=1e-15)
    
    def test_all_betas_returns_tuple(self):
        """all_betas should return tuple of 3 floats."""
        beta = BetaFunctions()
        couplings = (10.0, 50.0, 100.0)
        betas = beta.all_betas(couplings)
        
        assert isinstance(betas, tuple)
        assert len(betas) == 3
        assert all(isinstance(b, float) for b in betas)
    
    def test_all_betas_consistency(self):
        """all_betas should match individual function calls."""
        beta = BetaFunctions()
        l, g, m = 10.0, 50.0, 100.0
        betas = beta.all_betas((l, g, m))
        
        assert betas[0] == beta.beta_lambda(l)
        assert betas[1] == beta.beta_gamma(l, g)
        assert betas[2] == beta.beta_mu(l, g, m)


class TestBetaFunctionsJacobian:
    """Test Jacobian matrix computation."""
    
    def test_jacobian_shape(self):
        """Jacobian should be 3×3."""
        beta = BetaFunctions()
        J = beta.jacobian((10.0, 50.0, 100.0))
        assert J.shape == (3, 3)
    
    def test_jacobian_at_origin(self):
        """Jacobian at origin should have specific structure."""
        beta = BetaFunctions()
        J = beta.jacobian((0.0, 0.0, 0.0))
        
        # At origin:
        # ∂β_λ/∂λ = -2
        # ∂β_γ/∂γ = 0  (since λ=0)
        # ∂β_μ/∂μ = 2  (since λ=0)
        assert J[0, 0] == pytest.approx(-2.0, rel=1e-10)
        assert J[1, 1] == pytest.approx(0.0, abs=1e-10)
        assert J[2, 2] == pytest.approx(2.0, rel=1e-10)
    
    def test_jacobian_analytical_derivatives(self):
        """Jacobian should match analytical derivatives."""
        beta = BetaFunctions()
        l, g, m = 20.0, 80.0, 120.0
        J = beta.jacobian((l, g, m))
        
        # Check specific entries
        # ∂β_λ/∂λ = -2 + (9/4π²)λ̃
        expected_00 = -2 + (9 / (4 * math.pi**2)) * l
        assert J[0, 0] == pytest.approx(expected_00, rel=1e-10)
        
        # ∂β_γ/∂γ = (3/4π²)λ̃
        expected_11 = (3 / (4 * math.pi**2)) * l
        assert J[1, 1] == pytest.approx(expected_11, rel=1e-10)


class TestModuleFunctions:
    """Test module-level helper functions."""
    
    def test_beta_lambda_consistency(self):
        """Module function should match class method."""
        l, g, m = 20.0, 80.0, 120.0
        
        module_result = beta_lambda(l, g, m)
        class_result = BetaFunctions().beta_lambda(l, g, m)
        
        assert module_result == class_result
    
    def test_beta_gamma_consistency(self):
        """Module function should match class method."""
        l, g, m = 20.0, 80.0, 120.0
        
        module_result = beta_gamma(l, g, m)
        class_result = BetaFunctions().beta_gamma(l, g, m)
        
        assert module_result == class_result
    
    def test_beta_mu_consistency(self):
        """Module function should match class method."""
        l, g, m = 20.0, 80.0, 120.0
        
        module_result = beta_mu(l, g, m)
        class_result = BetaFunctions().beta_mu(l, g, m)
        
        assert module_result == class_result
    
    def test_compute_all_betas_tuple(self):
        """compute_all_betas should return tuple of 3 floats."""
        l, g, m = 20.0, 80.0, 120.0
        result = compute_all_betas(l, g, m)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(b, float) for b in result)


class TestTheoreticalGrounding:
    """Verify theoretical references."""
    
    def test_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.rg_flow import beta_functions
        assert 'IRH21.md' in beta_functions.__theoretical_foundation__
        assert 'Eq. 1.13' in beta_functions.__theoretical_foundation__
    
    def test_class_docstring(self):
        """BetaFunctions should reference Eq. 1.13."""
        assert 'Eq. 1.13' in BetaFunctions.__doc__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
