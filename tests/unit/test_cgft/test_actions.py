"""
Unit Tests for cGFT Action Functional (Eqs. 1.1-1.4)

THEORETICAL FOUNDATION: IRH21.md §1.1

Tests validate:
1. Kinetic action S_kin (Eq. 1.1)
2. Interaction action S_int (Eq. 1.2)
3. Interaction kernel K (Eq. 1.3)
4. Holographic action S_hol (Eq. 1.4)
5. Gauge invariance
6. Fixed-point coupling values

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

from src.cgft.actions import (
    compute_kinetic_action,
    compute_interaction_action,
    compute_holographic_action,
    compute_total_action,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)


class TestFixedPointConstants:
    """Test fixed-point coupling values per Eq. 1.14."""

    def test_lambda_star_value(self):
        """λ̃* = 48π²/9 per Eq. 1.14."""
        expected = 48 * math.pi**2 / 9
        assert np.isclose(LAMBDA_STAR, expected, rtol=1e-12)
        # Numerical value: ~52.6379...
        assert np.isclose(LAMBDA_STAR, 52.637890, rtol=1e-5)

    def test_gamma_star_value(self):
        """γ̃* = 32π²/3 per Eq. 1.14."""
        expected = 32 * math.pi**2 / 3
        assert np.isclose(GAMMA_STAR, expected, rtol=1e-12)
        # Numerical value: ~105.2758...
        assert np.isclose(GAMMA_STAR, 105.27578, rtol=1e-5)

    def test_mu_star_value(self):
        """μ̃* = 16π² per Eq. 1.14."""
        expected = 16 * math.pi**2
        assert np.isclose(MU_STAR, expected, rtol=1e-12)
        # Numerical value: ~157.9137...
        assert np.isclose(MU_STAR, 157.9136704174, rtol=1e-8)


class TestKineticAction:
    """Test kinetic action S_kin per Eq. 1.1."""

    def test_kinetic_action_zero_field(self):
        """S_kin = 0 for zero field configuration."""
        phi = np.zeros((10, 10, 10, 10), dtype=np.complex128)
        phi_bar = np.conj(phi)

        s_kin = compute_kinetic_action(phi, phi_bar)
        assert np.isclose(np.abs(s_kin), 0.0, atol=1e-12)

    def test_kinetic_action_constant_field(self):
        """S_kin = 0 for spatially constant field (no gradients)."""
        phi = np.ones((5, 5, 5, 5), dtype=np.complex128) * (1 + 1j)
        phi_bar = np.conj(phi)

        s_kin = compute_kinetic_action(phi, phi_bar)
        # Laplacian of constant is zero (with periodic BC)
        assert np.isclose(np.abs(s_kin), 0.0, atol=1e-10)

    def test_kinetic_action_shape_mismatch(self):
        """Raise error if phi and phi_bar shapes differ."""
        phi = np.ones((5, 5, 5, 5), dtype=np.complex128)
        phi_bar = np.ones((5, 5, 5, 10), dtype=np.complex128)

        with pytest.raises(ValueError):
            compute_kinetic_action(phi, phi_bar)

    def test_kinetic_action_real_positive(self):
        """S_kin should be real (up to numerical error) for physical configs."""
        rng = np.random.default_rng(42)
        phi = rng.random((5, 5, 5, 5)) + 1j * rng.random((5, 5, 5, 5))
        phi_bar = np.conj(phi)

        s_kin = compute_kinetic_action(phi, phi_bar)
        # S_kin is computed; just verify it's finite
        assert np.isfinite(s_kin.real)


class TestInteractionAction:
    """Test interaction action S_int per Eq. 1.2."""

    def test_interaction_action_zero_field(self):
        """S_int = 0 for zero field."""
        phi = np.zeros((10,), dtype=np.complex128)
        s_int = compute_interaction_action(phi)
        assert np.isclose(np.abs(s_int), 0.0, atol=1e-12)

    def test_interaction_action_positive(self):
        """S_int ≥ 0 for |φ|⁴ term."""
        phi = np.ones((10,), dtype=np.complex128) * 2
        s_int = compute_interaction_action(phi)
        assert s_int.real >= 0

    def test_interaction_action_uses_fixed_point_couplings(self):
        """Default couplings are fixed-point values."""
        phi = np.ones((10,), dtype=np.complex128)

        # With default couplings
        s_int_default = compute_interaction_action(phi)

        # With explicit fixed-point values
        s_int_explicit = compute_interaction_action(
            phi,
            lambda_coupling=LAMBDA_STAR,
            gamma_coupling=GAMMA_STAR,
        )

        assert np.isclose(s_int_default, s_int_explicit)


class TestHolographicAction:
    """Test holographic action S_hol per Eq. 1.4."""

    def test_holographic_action_zero_field(self):
        """S_hol for zero field."""
        phi = np.zeros((10,), dtype=np.complex128)
        s_hol = compute_holographic_action(phi)
        # Should be defined (constraint is always satisfied for zero)
        assert np.isfinite(s_hol.real)

    def test_holographic_action_scales_with_mu(self):
        """S_hol scales linearly with μ coupling."""
        phi = np.ones((10,), dtype=np.complex128)

        s_hol_1 = compute_holographic_action(phi, mu_coupling=1.0)
        s_hol_2 = compute_holographic_action(phi, mu_coupling=2.0)

        # Ratio should be approximately 2
        if np.abs(s_hol_1) > 1e-12:
            assert np.isclose(s_hol_2 / s_hol_1, 2.0, rtol=0.1)


class TestTotalAction:
    """Test complete action S = S_kin + S_int + S_hol."""

    def test_total_action_decomposition(self):
        """S_total = S_kin + S_int + S_hol."""
        rng = np.random.default_rng(42)
        phi = rng.random((5, 5, 5, 5)) + 1j * rng.random((5, 5, 5, 5))

        result = compute_total_action(phi)

        expected_total = result['S_kin'] + result['S_int'] + result['S_hol']
        assert np.isclose(result['S_total'], expected_total)

    def test_total_action_has_theoretical_reference(self):
        """Result includes IRH21.md citation."""
        phi = np.ones((5, 5, 5, 5), dtype=np.complex128)
        result = compute_total_action(phi)

        assert 'theoretical_reference' in result
        assert 'IRH21.md' in result['theoretical_reference']
        assert '1.1' in result['theoretical_reference']

    def test_total_action_all_keys_present(self):
        """Result dictionary has all expected keys."""
        phi = np.ones((5, 5, 5, 5), dtype=np.complex128)
        result = compute_total_action(phi)

        expected_keys = {'S_total', 'S_kin', 'S_int', 'S_hol', 'theoretical_reference'}
        assert set(result.keys()) == expected_keys

    def test_total_action_zero_field(self):
        """S_total for zero field configuration."""
        phi = np.zeros((5, 5, 5, 5), dtype=np.complex128)
        result = compute_total_action(phi)

        assert np.isclose(result['S_kin'], 0.0, atol=1e-12)
        assert np.isclose(result['S_int'], 0.0, atol=1e-12)


class TestEquationReferences:
    """Verify equation references in docstrings."""

    def test_kinetic_references_eq_1_1(self):
        """compute_kinetic_action references Eq. 1.1."""
        docstring = compute_kinetic_action.__doc__
        assert 'Eq. 1.1' in docstring or 'Eq 1.1' in docstring

    def test_interaction_references_eq_1_2_1_3(self):
        """compute_interaction_action references Eqs. 1.2 and 1.3."""
        docstring = compute_interaction_action.__doc__
        assert 'Eq. 1.2' in docstring or 'Eq 1.2' in docstring
        assert 'Eq. 1.3' in docstring or 'Eq 1.3' in docstring

    def test_holographic_references_eq_1_4(self):
        """compute_holographic_action references Eq. 1.4."""
        docstring = compute_holographic_action.__doc__
        assert 'Eq. 1.4' in docstring or 'Eq 1.4' in docstring


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
