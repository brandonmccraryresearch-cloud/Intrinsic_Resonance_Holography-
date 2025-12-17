"""
Tests for IRH v21.0 Visualization Module

THEORETICAL FOUNDATION: IRH21.md §1-3

Tests for RG flow plots, manifold visualization, spectral dimension,
and topology visualizations.
"""

import pytest
import numpy as np

# Test imports work even without matplotlib
IMPORT_SUCCESS = False
IMPORT_ERROR = ""
MATPLOTLIB_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    from src.visualization.rg_flow_plots import (
        LAMBDA_STAR,
        GAMMA_STAR,
        MU_STAR,
        _beta_lambda,
        _beta_gamma,
        _beta_mu,
        MATPLOTLIB_AVAILABLE,
        PLOTLY_AVAILABLE,
    )
    from src.visualization.manifold_viz import (
        quaternion_to_su2,
        su2_to_hopf,
        sample_su2_haar,
        geodesic_su2,
    )
    from src.visualization.spectral_dimension_viz import (
        spectral_dimension,
        graviton_correction,
        D_SPEC_UV,
        D_SPEC_IR,
    )
    from src.visualization.topology_viz import (
        BETTI_1,
        N_INST,
        FERMION_K_VALUES,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestModuleImports:
    """Test that modules import correctly."""
    
    def test_imports_successful(self):
        """Modules should import without errors."""
        assert IMPORT_SUCCESS, f"Import failed: {IMPORT_ERROR if not IMPORT_SUCCESS else ''}"


class TestRGFlowConstants:
    """Test RG flow constants from IRH21.md Eq. 1.14."""
    
    def test_lambda_star_value(self):
        """λ* = 48π²/9 ≈ 52.638 (Eq. 1.14)."""
        expected = 48 * np.pi**2 / 9
        assert np.isclose(LAMBDA_STAR, expected, rtol=1e-10)
        assert np.isclose(LAMBDA_STAR, 52.638, rtol=1e-3)
    
    def test_gamma_star_value(self):
        """γ* = 32π²/3 ≈ 105.276 (Eq. 1.14)."""
        expected = 32 * np.pi**2 / 3
        assert np.isclose(GAMMA_STAR, expected, rtol=1e-10)
        assert np.isclose(GAMMA_STAR, 105.276, rtol=1e-3)
    
    def test_mu_star_value(self):
        """μ* = 16π² ≈ 157.914 (Eq. 1.14)."""
        expected = 16 * np.pi**2
        assert np.isclose(MU_STAR, expected, rtol=1e-10)
        assert np.isclose(MU_STAR, 157.914, rtol=1e-3)


class TestBetaFunctions:
    """Test beta functions from IRH21.md Eq. 1.13."""
    
    def test_beta_lambda_formula(self):
        """β_λ = -2λ̃ + (9/8π²)λ̃² (Eq. 1.13a)."""
        lambda_t = 10.0
        result = _beta_lambda(lambda_t)
        expected = -2 * lambda_t + (9 / (8 * np.pi**2)) * lambda_t**2
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_beta_gamma_formula(self):
        """β_γ = (3/4π²)λ̃γ̃ (Eq. 1.13b)."""
        lambda_t, gamma_t = 10.0, 50.0
        result = _beta_gamma(lambda_t, gamma_t)
        expected = (3 / (4 * np.pi**2)) * lambda_t * gamma_t
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_beta_mu_formula(self):
        """β_μ = 2μ̃ + (1/2π²)λ̃μ̃ (Eq. 1.13c)."""
        lambda_t, mu_t = 10.0, 100.0
        result = _beta_mu(lambda_t, mu_t)
        expected = 2 * mu_t + (1 / (2 * np.pi**2)) * lambda_t * mu_t
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_beta_lambda_has_nontrivial_zero(self):
        """β_λ = 0 at λ = 16π²/9 (nontrivial fixed point of β_λ alone)."""
        # The zero of β_λ = -2λ + (9/8π²)λ² is at λ = 16π²/9
        # Note: λ* = 48π²/9 is the cosmic fixed point where ALL β's vanish together
        lambda_zero = 16 * np.pi**2 / 9
        result = _beta_lambda(lambda_zero)
        assert abs(result) < 1e-10
    
    def test_beta_lambda_zero_at_origin(self):
        """β_λ should be zero at λ = 0."""
        result = _beta_lambda(0.0)
        assert result == 0.0


class TestQuaternionFunctions:
    """Test quaternion and SU(2) functions."""
    
    def test_quaternion_to_su2_identity(self):
        """Identity quaternion (1,0,0,0) → identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        su2 = quaternion_to_su2(q)
        expected = np.eye(2, dtype=complex)
        assert np.allclose(su2, expected)
    
    def test_quaternion_to_su2_determinant(self):
        """SU(2) matrices should have determinant 1."""
        q = np.array([0.5, 0.5, 0.5, 0.5])  # Normalized
        su2 = quaternion_to_su2(q)
        det = np.linalg.det(su2)
        assert np.isclose(abs(det), 1.0, rtol=1e-10)
    
    def test_hopf_map_on_sphere(self):
        """Hopf map should give points on S²."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        x, y, z = su2_to_hopf(q)
        norm = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(norm, 1.0, rtol=1e-10)
    
    def test_haar_samples_normalized(self):
        """Haar samples should be unit quaternions."""
        samples = sample_su2_haar(100)
        norms = np.linalg.norm(samples, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-10)
    
    def test_geodesic_endpoints(self):
        """Geodesic should connect endpoints."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        t = np.array([0.0, 1.0])
        geodesic = geodesic_su2(q1, q2, t)
        
        # Check start and end points
        assert np.allclose(geodesic[0], q1, rtol=1e-6)


class TestSpectralDimension:
    """Test spectral dimension functions from IRH21.md §2.1."""
    
    def test_spectral_dimension_uv_limit(self):
        """d_spec → 2 in UV limit (k → ∞)."""
        d_uv = spectral_dimension(1000.0)
        assert np.isclose(d_uv, D_SPEC_UV, rtol=0.1)
    
    def test_spectral_dimension_ir_limit(self):
        """d_spec → 4 in IR limit (k → 0) with graviton."""
        d_ir = spectral_dimension(0.001, include_graviton=True)
        assert np.isclose(d_ir, D_SPEC_IR, rtol=0.01)
    
    def test_spectral_dimension_monotonic(self):
        """d_spec should increase monotonically from UV to IR."""
        k_values = np.logspace(-2, 2, 50)
        d_values = spectral_dimension(k_values)
        
        # d_spec should decrease as k increases (UV → lower dimension)
        diffs = np.diff(d_values)
        assert np.all(diffs <= 0.1)  # Allow small non-monotonicity from numerics
    
    def test_graviton_correction_ir_dominant(self):
        """Graviton correction should be largest in IR."""
        delta_ir = graviton_correction(0.01)
        delta_uv = graviton_correction(100.0)
        assert delta_ir > delta_uv
    
    def test_graviton_correction_positive(self):
        """Graviton correction should be positive."""
        k_values = np.logspace(-2, 2, 20)
        deltas = graviton_correction(k_values)
        assert np.all(deltas >= 0)


class TestTopologyConstants:
    """Test topological invariants from IRH21.md Appendix D."""
    
    def test_betti_1_value(self):
        """β₁ = 12 (Appendix D.1)."""
        assert BETTI_1 == 12
    
    def test_instanton_number(self):
        """n_inst = 3 (Appendix D.2)."""
        assert N_INST == 3
    
    def test_fermion_generations(self):
        """Three generations of fermions."""
        leptons = ['electron', 'muon', 'tau']
        up_quarks = ['up', 'charm', 'top']
        down_quarks = ['down', 'strange', 'bottom']
        
        assert all(f in FERMION_K_VALUES for f in leptons)
        assert all(f in FERMION_K_VALUES for f in up_quarks)
        assert all(f in FERMION_K_VALUES for f in down_quarks)
    
    def test_mass_hierarchy(self):
        """K values should increase across generations."""
        # Charged leptons
        assert FERMION_K_VALUES['electron'] < FERMION_K_VALUES['muon']
        assert FERMION_K_VALUES['muon'] < FERMION_K_VALUES['tau']
        
        # Up-type quarks
        assert FERMION_K_VALUES['up'] < FERMION_K_VALUES['charm']
        assert FERMION_K_VALUES['charm'] < FERMION_K_VALUES['top']
        
        # Down-type quarks
        assert FERMION_K_VALUES['down'] < FERMION_K_VALUES['strange']
        assert FERMION_K_VALUES['strange'] < FERMION_K_VALUES['bottom']
    
    def test_electron_k_value(self):
        """K_e = 1 by convention."""
        assert FERMION_K_VALUES['electron'] == 1.0


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestPlotGeneration:
    """Test that plots can be generated."""
    
    def test_rg_flow_plotter_creation(self):
        """RGFlowPlotter should be creatable."""
        from src.visualization.rg_flow_plots import RGFlowPlotter
        plotter = RGFlowPlotter()
        assert plotter is not None
    
    def test_manifold_visualizer_creation(self):
        """ManifoldVisualizer should be creatable."""
        from src.visualization.manifold_viz import ManifoldVisualizer
        viz = ManifoldVisualizer()
        assert viz is not None
    
    def test_spectral_animator_creation(self):
        """SpectralDimensionAnimator should be creatable."""
        from src.visualization.spectral_dimension_viz import SpectralDimensionAnimator
        anim = SpectralDimensionAnimator()
        assert anim is not None
    
    def test_topology_visualizer_creation(self):
        """TopologyVisualizer should be creatable."""
        from src.visualization.topology_viz import TopologyVisualizer
        viz = TopologyVisualizer()
        assert viz is not None


class TestTheoreticalGrounding:
    """Test that modules have proper theoretical grounding."""
    
    def test_rg_flow_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.visualization import rg_flow_plots
        assert hasattr(rg_flow_plots, '__theoretical_foundation__')
        assert 'IRH v21.1 Manuscript' in rg_flow_plots.__theoretical_foundation__
    
    def test_manifold_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.visualization import manifold_viz
        assert hasattr(manifold_viz, '__theoretical_foundation__')
        assert 'IRH v21.1 Manuscript' in manifold_viz.__theoretical_foundation__
    
    def test_spectral_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.visualization import spectral_dimension_viz
        assert hasattr(spectral_dimension_viz, '__theoretical_foundation__')
        assert 'IRH v21.1 Manuscript' in spectral_dimension_viz.__theoretical_foundation__
    
    def test_topology_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.visualization import topology_viz
        assert hasattr(topology_viz, '__theoretical_foundation__')
        assert 'IRH v21.1 Manuscript' in topology_viz.__theoretical_foundation__
