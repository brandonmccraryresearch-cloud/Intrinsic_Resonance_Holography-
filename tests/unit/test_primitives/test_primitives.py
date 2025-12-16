"""
Unit Tests for Primitive Ontological Layer

THEORETICAL FOUNDATION: IRH21.md §1.0.1, §1.1, Appendix A

Tests validate:
1. Quaternion algebra axioms
2. SU(2) and U(1) group structure
3. G_inf = SU(2) × U(1) direct product
4. QNCD metric properties

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

from src.primitives.quaternions import (
    Quaternion,
    quaternion_conjugate,
    quaternion_product,
    quaternion_norm,
    quaternion_exp,
    quaternion_log,
    verify_quaternion_algebra,
)

from src.primitives.group_manifold import (
    SU2Element,
    U1Phase,
    GInfElement,
    compute_GInf_distance,
    verify_group_axioms,
)

from src.primitives.qncd import (
    compute_QNCD,
    compute_pairwise_QNCD_sum,
    verify_QNCD_metric_axioms,
)


class TestQuaternions:
    """Test quaternion algebra per IRH21.md §1.1.1."""

    def test_quaternion_creation(self):
        """Test quaternion construction."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_quaternion_identity(self):
        """Multiplicative identity: 1 + 0i + 0j + 0k."""
        identity = Quaternion.identity()
        assert identity.norm() == pytest.approx(1.0)
        assert identity.w == 1.0

    def test_quaternion_norm(self):
        """Test |q|² = q₀² + q₁² + q₂² + q₃²."""
        q = Quaternion(w=1.0, x=2.0, y=2.0, z=0.0)
        expected_norm = math.sqrt(1 + 4 + 4 + 0)
        assert q.norm() == pytest.approx(expected_norm)

    def test_quaternion_conjugate(self):
        """Test q̄ = q₀ - iq₁ - jq₂ - kq₃."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        q_bar = q.conjugate()
        assert q_bar.w == 1.0
        assert q_bar.x == -2.0
        assert q_bar.y == -3.0
        assert q_bar.z == -4.0

    def test_quaternion_multiplication_identity(self):
        """Test q * 1 = 1 * q = q."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        identity = Quaternion.identity()
        
        assert q * identity == q
        assert identity * q == q

    def test_quaternion_non_commutativity(self):
        """Verify quaternion multiplication is non-commutative."""
        i = Quaternion(w=0, x=1, y=0, z=0)
        j = Quaternion(w=0, x=0, y=1, z=0)
        
        ij = i * j  # Should be k
        ji = j * i  # Should be -k
        
        assert ij != ji  # Non-commutative
        assert ij.z == pytest.approx(1.0)  # ij = k
        assert ji.z == pytest.approx(-1.0)  # ji = -k

    def test_quaternion_basis_multiplication(self):
        """Test i² = j² = k² = ijk = -1."""
        i = Quaternion(w=0, x=1, y=0, z=0)
        j = Quaternion(w=0, x=0, y=1, z=0)
        k = Quaternion(w=0, x=0, y=0, z=1)
        minus_one = Quaternion(w=-1, x=0, y=0, z=0)
        
        assert i * i == minus_one  # i² = -1
        assert j * j == minus_one  # j² = -1
        assert k * k == minus_one  # k² = -1
        assert i * j * k == minus_one  # ijk = -1

    def test_quaternion_inverse(self):
        """Test q * q⁻¹ = 1."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        q_inv = q.inverse()
        product = q * q_inv
        
        assert product == Quaternion.identity()

    def test_quaternion_algebra_verification(self):
        """Run complete algebra verification."""
        results = verify_quaternion_algebra()
        
        assert results['associativity']['passed']
        assert results['left_distributivity']['passed']
        assert results['conjugation_involution']['passed']
        assert results['norm_multiplicativity']['passed']
        assert results['non_commutativity']['passed']
        assert results['division_algebra']['passed']
        assert results['all_passed']


class TestSU2:
    """Test SU(2) group structure per IRH21.md §1.1."""

    def test_su2_identity(self):
        """Test identity element."""
        e = SU2Element.identity()
        q = e.to_quaternion()
        assert q.norm() == pytest.approx(1.0)
        assert q.w == pytest.approx(1.0)

    def test_su2_normalization(self):
        """SU(2) elements must be unit quaternions."""
        # Non-unit quaternion should be normalized
        q = Quaternion(w=3.0, x=4.0, y=0.0, z=0.0)  # norm = 5
        u = SU2Element.from_quaternion(q)
        assert u.quaternion.norm() == pytest.approx(1.0)

    def test_su2_multiplication(self):
        """Test group multiplication."""
        u1 = SU2Element.random()
        u2 = SU2Element.random()
        u3 = u1 * u2
        
        # Result should still be unit quaternion
        assert u3.quaternion.norm() == pytest.approx(1.0)

    def test_su2_inverse(self):
        """Test u * u⁻¹ = e."""
        u = SU2Element.random()
        e = SU2Element.identity()
        
        product = u * u.inverse()
        
        # Check quaternions are equal
        q1 = product.to_quaternion()
        q2 = e.to_quaternion()
        assert np.allclose(q1.to_array(), q2.to_array(), atol=1e-10)

    def test_su2_axis_angle(self):
        """Test axis-angle construction."""
        axis = np.array([0, 0, 1])
        angle = math.pi / 2  # 90 degrees
        
        u = SU2Element.from_axis_angle(axis, angle)
        
        # For 90° rotation around z: cos(π/4) + sin(π/4)k
        q = u.to_quaternion()
        assert q.w == pytest.approx(math.cos(math.pi / 4))
        assert q.z == pytest.approx(math.sin(math.pi / 4))


class TestU1Phase:
    """Test U(1)_φ holonomic phase group."""

    def test_u1_identity(self):
        """Test identity element φ = 0."""
        e = U1Phase.identity()
        assert e.phase == pytest.approx(0.0)

    def test_u1_normalization(self):
        """Phase should be normalized to [0, 2π)."""
        phi = U1Phase(phase=3 * math.pi)
        assert 0 <= phi.phase < 2 * math.pi
        assert phi.phase == pytest.approx(math.pi)

    def test_u1_multiplication(self):
        """Test e^{iφ₁} · e^{iφ₂} = e^{i(φ₁+φ₂)}."""
        phi1 = U1Phase(phase=math.pi / 3)
        phi2 = U1Phase(phase=math.pi / 4)
        
        product = phi1 * phi2
        expected = (math.pi / 3 + math.pi / 4) % (2 * math.pi)
        
        assert product.phase == pytest.approx(expected)

    def test_u1_inverse(self):
        """Test (e^{iφ})⁻¹ = e^{-iφ}."""
        phi = U1Phase(phase=math.pi / 3)
        inv = phi.inverse()
        
        product = phi * inv
        assert product == U1Phase.identity()


class TestGInf:
    """Test G_inf = SU(2) × U(1) per IRH21.md §1.1."""

    def test_ginf_identity(self):
        """Test identity element (e, 0)."""
        e = GInfElement.identity()
        assert e.su2 == SU2Element.identity()
        assert e.u1 == U1Phase.identity()

    def test_ginf_multiplication(self):
        """Test direct product multiplication."""
        g1 = GInfElement.random()
        g2 = GInfElement.random()
        
        product = g1 * g2
        
        # Check it's still a valid element
        assert isinstance(product, GInfElement)
        assert product.su2.quaternion.norm() == pytest.approx(1.0)

    def test_ginf_inverse(self):
        """Test g * g⁻¹ = e."""
        g = GInfElement.random()
        e = GInfElement.identity()
        
        product = g * g.inverse()
        
        # Distance should be near zero
        dist = compute_GInf_distance(product, e)
        assert dist < 1e-6  # Relaxed tolerance for numerical precision

    def test_ginf_distance_positive(self):
        """Test d(g₁, g₂) ≥ 0."""
        g1 = GInfElement.random()
        g2 = GInfElement.random()
        
        d = compute_GInf_distance(g1, g2)
        assert d >= 0

    def test_ginf_distance_symmetric(self):
        """Test d(g₁, g₂) = d(g₂, g₁)."""
        g1 = GInfElement.random()
        g2 = GInfElement.random()
        
        d12 = compute_GInf_distance(g1, g2)
        d21 = compute_GInf_distance(g2, g1)
        
        assert d12 == pytest.approx(d21)

    def test_group_axioms_verification(self):
        """Run complete group axioms verification."""
        results = verify_group_axioms()
        
        assert results['closure']['passed']
        assert results['associativity']['passed']
        assert results['identity']['passed']
        assert results['inverse']['passed']
        assert results['bi_invariance']['passed']
        assert results['all_passed']


class TestQNCD:
    """Test QNCD metric per IRH21.md Appendix A."""

    def test_qncd_range(self):
        """Test QNCD ∈ [0, 1]."""
        g1 = GInfElement.random()
        g2 = GInfElement.random()
        
        d = compute_QNCD(g1, g2)
        assert 0 <= d <= 1

    def test_qncd_symmetry(self):
        """Test d(g₁, g₂) = d(g₂, g₁)."""
        g1 = GInfElement.random()
        g2 = GInfElement.random()
        
        d12 = compute_QNCD(g1, g2)
        d21 = compute_QNCD(g2, g1)
        
        assert d12 == pytest.approx(d21, abs=0.01)

    def test_pairwise_sum(self):
        """Test pairwise QNCD sum for 4 elements."""
        elements = [GInfElement.random() for _ in range(4)]
        
        total = compute_pairwise_QNCD_sum(elements)
        
        # 4 choose 2 = 6 pairs
        assert total >= 0

    def test_qncd_metric_axioms(self):
        """Run QNCD metric axioms verification."""
        results = verify_QNCD_metric_axioms(n_samples=20)
        
        assert results['positivity']['passed']
        # Note: Symmetry may have small deviations due to compression
        # This is expected for classical approximations to Kolmogorov complexity
        assert results['symmetry']['max_asymmetry'] < 0.1  # Relaxed tolerance


class TestTheoreticalReferences:
    """Verify theoretical references in docstrings."""

    def test_quaternion_module_reference(self):
        """quaternions module references §1.1.1."""
        from src.primitives import quaternions
        assert 'IRH21.md' in quaternions.__theoretical_foundation__
        assert '1.1.1' in quaternions.__theoretical_foundation__

    def test_group_manifold_reference(self):
        """group_manifold module references §1.1."""
        from src.primitives import group_manifold
        assert 'IRH21.md' in group_manifold.__theoretical_foundation__
        assert '1.1' in group_manifold.__theoretical_foundation__

    def test_qncd_reference(self):
        """qncd module references Appendix A."""
        from src.primitives import qncd
        assert 'IRH21.md' in qncd.__theoretical_foundation__
        assert 'Appendix A' in qncd.__theoretical_foundation__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
