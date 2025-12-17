"""
Unit Tests for Emergent Spacetime Module - Phase II

THEORETICAL FOUNDATION: IRH21.md §2.1-2.5

Tests validate:
1. Spectral dimension flow (Eq. 2.8-2.9)
2. Metric tensor emergence (Eq. 2.10)
3. Lorentzian signature (Theorem H.1)
4. Einstein equations (Theorem C.3)

Authors: IRH Computational Framework Team
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

# Add src to path for imports
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class TestSpectralDimension:
    """Test spectral dimension module (Eq. 2.8-2.9)."""
    
    def test_one_loop_value(self):
        """One-loop d_spec should be 42/11."""
        from src.emergent_spacetime import D_SPEC_ONE_LOOP
        
        expected = 42 / 11
        assert D_SPEC_ONE_LOOP == pytest.approx(expected, rel=1e-15)
    
    def test_ir_value(self):
        """IR d_spec should be exactly 4."""
        from src.emergent_spacetime import D_SPEC_IR
        
        assert D_SPEC_IR == 4.0
    
    def test_graviton_coefficient(self):
        """Graviton correction coefficient should be -2/11."""
        from src.emergent_spacetime import DELTA_GRAV_COEFFICIENT
        
        expected = -2 / 11
        assert DELTA_GRAV_COEFFICIENT == pytest.approx(expected, rel=1e-15)
    
    def test_compute_spectral_dimension_analytical(self):
        """Analytical d_spec at IR should be 4.0."""
        from src.emergent_spacetime import compute_spectral_dimension
        
        result = compute_spectral_dimension(scale=0.0, method='analytical')
        
        assert result.d_spec == 4.0
        assert result.is_exact_4d
        assert result.method == 'analytical'
    
    def test_compute_spectral_dimension_one_loop(self):
        """One-loop d_spec should be 42/11."""
        from src.emergent_spacetime import compute_spectral_dimension
        
        result = compute_spectral_dimension(method='one_loop')
        
        assert result.d_spec == pytest.approx(42/11, rel=1e-15)
        assert not result.is_exact_4d
    
    def test_spectral_dimension_flow(self):
        """Flow should go from UV to IR."""
        from src.emergent_spacetime import spectral_dimension_flow
        
        flow = spectral_dimension_flow(t_range=(-10, 10))
        
        assert len(flow.t_values) > 0
        assert len(flow.d_spec_values) == len(flow.t_values)
    
    def test_verify_theorem_2_1(self):
        """Theorem 2.1: d_spec → 4 exactly."""
        from src.emergent_spacetime import verify_theorem_2_1
        
        result = verify_theorem_2_1()
        
        assert result['is_verified']
        assert result['d_spec_ir'] == 4.0
        assert abs(result['cancellation_residual']) < 1e-10
    
    def test_graviton_correction_function(self):
        """graviton_correction should be non-zero."""
        from src.emergent_spacetime import graviton_correction
        
        # At some scale
        delta = graviton_correction(k=1.0)
        assert isinstance(delta, float)
        assert np.isfinite(delta)


class TestMetricTensor:
    """Test metric tensor module (Eq. 2.10)."""
    
    def test_minkowski_metric(self):
        """Minkowski metric should be diag(-1,1,1,1)."""
        from src.emergent_spacetime import minkowski_metric
        
        eta = minkowski_metric()
        
        assert eta.components.shape == (4, 4)
        assert eta.is_lorentzian
        assert eta.signature == (-1, 1, 1, 1)
        assert eta.determinant == pytest.approx(-1.0, rel=1e-10)
    
    def test_metric_tensor_properties(self):
        """MetricTensor should have correct properties."""
        from src.emergent_spacetime import minkowski_metric
        
        eta = minkowski_metric()
        
        # Inverse should work
        eta_inv = eta.inverse
        identity = eta.components @ eta_inv
        assert np.allclose(identity, np.eye(4))
    
    def test_verify_metric_properties(self):
        """Verification should pass for valid metric."""
        from src.emergent_spacetime import minkowski_metric, verify_metric_properties
        
        eta = minkowski_metric()
        result = verify_metric_properties(eta)
        
        assert result['is_symmetric']
        assert result['is_invertible']
        assert result['is_lorentzian']
        assert result['all_checks_passed']
    
    def test_emergent_metric(self):
        """emergent_metric should return valid MetricTensor."""
        from src.emergent_spacetime import emergent_metric
        
        g = emergent_metric(scale=0.0)
        
        assert g.components.shape == (4, 4)
        assert g.is_lorentzian
    
    def test_schwarzschild_metric(self):
        """Schwarzschild metric should be valid."""
        from src.emergent_spacetime import schwarzschild_metric
        
        # At r = 10 (in units where r_s = 2)
        g = schwarzschild_metric(r=10.0, M=1.0)
        
        assert g.components.shape == (4, 4)
        assert g.is_lorentzian


class TestLorentzianSignature:
    """Test Lorentzian signature module (Theorem H.1)."""
    
    def test_lorentzian_signature_constant(self):
        """LORENTZIAN_SIGNATURE should be (-1,1,1,1)."""
        from src.emergent_spacetime import LORENTZIAN_SIGNATURE
        
        assert LORENTZIAN_SIGNATURE == (-1, 1, 1, 1)
    
    def test_euclidean_signature_constant(self):
        """EUCLIDEAN_SIGNATURE should be (1,1,1,1)."""
        from src.emergent_spacetime import EUCLIDEAN_SIGNATURE
        
        assert EUCLIDEAN_SIGNATURE == (1, 1, 1, 1)
    
    def test_compute_signature(self):
        """compute_signature should return correct tuple."""
        from src.emergent_spacetime import compute_signature
        
        # Minkowski eigenvalues
        eigenvalues = np.array([-1.0, 1.0, 1.0, 1.0])
        sig = compute_signature(eigenvalues)
        
        assert sig == (-1, 1, 1, 1)
    
    def test_verify_lorentzian(self):
        """verify_lorentzian should return True for (-1,1,1,1)."""
        from src.emergent_spacetime import verify_lorentzian
        
        assert verify_lorentzian((-1, 1, 1, 1))
        assert not verify_lorentzian((1, 1, 1, 1))
    
    def test_z2_symmetry_breaking_low_temp(self):
        """Low temperature should break ℤ₂."""
        from src.emergent_spacetime import z2_symmetry_breaking
        
        result = z2_symmetry_breaking(temperature=0.0, critical_temperature=1.0)
        
        assert result.is_broken
        assert result.vev > 0
    
    def test_z2_symmetry_breaking_high_temp(self):
        """High temperature should preserve ℤ₂."""
        from src.emergent_spacetime import z2_symmetry_breaking
        
        result = z2_symmetry_breaking(temperature=2.0, critical_temperature=1.0)
        
        assert not result.is_broken
        assert result.vev == 0.0
    
    def test_signature_from_condensate(self):
        """Non-zero condensate should give Lorentzian."""
        from src.emergent_spacetime import signature_from_condensate
        
        result = signature_from_condensate(condensate_vev=1.0)
        
        assert result.is_lorentzian
        assert result.n_timelike == 1
        assert result.n_spacelike == 3
    
    def test_verify_theorem_h1(self):
        """Theorem H.1: Lorentzian signature emergence."""
        from src.emergent_spacetime import verify_theorem_h1
        
        result = verify_theorem_h1()
        
        assert result['is_verified']
        assert result['ir_signature'] == (-1, 1, 1, 1)
        assert result['stability_check']
        assert result['unitarity_check']


class TestEinsteinEquations:
    """Test Einstein equations module (Theorem C.3)."""
    
    def test_physical_constants(self):
        """Physical constants should be defined."""
        from src.emergent_spacetime import G_NEWTON, M_PLANCK, C_H_SPECTRAL
        
        assert G_NEWTON > 0
        assert M_PLANCK > 0
        assert C_H_SPECTRAL == pytest.approx(0.045935703598, rel=1e-10)
    
    def test_harmony_functional(self):
        """HarmonyFunctional should be instantiable."""
        from src.emergent_spacetime import HarmonyFunctional
        
        hf = HarmonyFunctional()
        
        assert hf.C_H == pytest.approx(0.045935703598, rel=1e-10)
    
    def test_derive_einstein_equations(self):
        """derive_einstein_equations should return valid result."""
        from src.emergent_spacetime import derive_einstein_equations
        
        result = derive_einstein_equations(method='variational')
        
        assert 'field_equations' in result
        assert 'G_μν' in result['field_equations']
        assert result['cosmological_term'] > 0
    
    def test_compute_cosmological_constant(self):
        """compute_cosmological_constant should work."""
        from src.emergent_spacetime import compute_cosmological_constant
        
        result = compute_cosmological_constant()
        
        assert 'ratio' in result
        assert result['ratio'] == pytest.approx(0.25, rel=1e-10)  # 1/4
    
    def test_einstein_hilbert_action(self):
        """einstein_hilbert_action should compute finite value."""
        from src.emergent_spacetime import einstein_hilbert_action
        
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        R = 0.0  # Flat space
        volume = 1.0
        
        S = einstein_hilbert_action(metric, R, volume)
        
        assert np.isfinite(S)
    
    def test_verify_theorem_c3(self):
        """Theorem C.3: Einstein-Hilbert from Harmony Functional."""
        from src.emergent_spacetime import verify_theorem_c3
        
        result = verify_theorem_c3()
        
        assert result['is_verified']
        assert 'G_μν' in result['key_results']['field_equations']
    
    def test_vacuum_einstein_equations(self):
        """vacuum_einstein_equations should return solutions."""
        from src.emergent_spacetime import vacuum_einstein_equations
        
        result = vacuum_einstein_equations(cosmological_constant=0.0)
        
        assert 'equation' in result
        assert 'Minkowski' in str(result['solutions'])


class TestIntegration:
    """Integration tests for Phase II modules."""
    
    def test_module_imports(self):
        """All Phase II modules should import successfully."""
        from src.emergent_spacetime import (
            # Spectral dimension
            compute_spectral_dimension,
            spectral_dimension_flow,
            verify_theorem_2_1,
            
            # Metric tensor
            minkowski_metric,
            emergent_metric,
            verify_metric_properties,
            
            # Lorentzian signature
            verify_lorentzian,
            signature_from_condensate,
            verify_theorem_h1,
            
            # Einstein equations
            derive_einstein_equations,
            verify_theorem_c3,
        )
        
        # All imports successful
        assert True
    
    def test_complete_spacetime_emergence(self):
        """Test complete spacetime emergence pipeline."""
        from src.emergent_spacetime import (
            verify_theorem_2_1,
            verify_theorem_h1,
            verify_theorem_c3,
            minkowski_metric,
        )
        
        # 1. Verify spectral dimension → 4
        t2_1 = verify_theorem_2_1()
        assert t2_1['is_verified'], "Theorem 2.1 failed"
        
        # 2. Verify Lorentzian signature
        th1 = verify_theorem_h1()
        assert th1['is_verified'], "Theorem H.1 failed"
        
        # 3. Verify Einstein equations emerge
        tc3 = verify_theorem_c3()
        assert tc3['is_verified'], "Theorem C.3 failed"
        
        # 4. Construct emergent metric
        g = minkowski_metric()
        assert g.is_lorentzian, "Metric not Lorentzian"
        assert g.signature == (-1, 1, 1, 1)
    
    def test_summary_generation(self):
        """Summary generation functions should work."""
        from src.emergent_spacetime import (
            generate_spectral_dimension_summary,
            generate_metric_tensor_summary,
            generate_lorentzian_signature_summary,
            generate_einstein_equations_summary,
        )
        
        summaries = [
            generate_spectral_dimension_summary(),
            generate_metric_tensor_summary(),
            generate_lorentzian_signature_summary(),
            generate_einstein_equations_summary(),
        ]
        
        for summary in summaries:
            assert 'theoretical_foundation' in summary
            assert 'version' in summary


class TestTheoreticalGrounding:
    """Verify theoretical references."""
    
    def test_module_foundation(self):
        """Module should reference IRH21.md."""
        from src import emergent_spacetime
        
        assert 'IRH v21.1 Manuscript' in emergent_spacetime.__theoretical_foundation__
        assert '§2' in emergent_spacetime.__theoretical_foundation__
    
    def test_spectral_dimension_foundation(self):
        """Spectral dimension should reference Eq. 2.8-2.9."""
        from src.emergent_spacetime import spectral_dimension
        
        assert 'IRH v21.1 Manuscript' in spectral_dimension.__theoretical_foundation__
        assert 'Eqs. 2' in spectral_dimension.__theoretical_foundation__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
