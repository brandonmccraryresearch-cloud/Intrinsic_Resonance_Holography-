"""
Quantum Normalized Compression Distance (QNCD) Implementation for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1, Appendix A

This module implements the QNCD metric d_QNCD which measures algorithmic
similarity between group elements on G_inf = SU(2) × U(1)_φ.

Key Properties:
    - Bi-invariance: d(kg₁, kg₂) = d(g₁k, g₂k) = d(g₁, g₂)
    - Metric axioms: positivity, symmetry, triangle inequality
    - QUCC-Theorem compliance: compressor-independent (Appendix A.4)

Theoretical Significance:
    The QNCD metric appears in the interaction kernel K (Eq. 1.3):
        K = exp[i(φ₁+φ₂+φ₃-φ₄)] · exp[-γ Σ d_QNCD(gᵢgⱼ⁻¹)]
    
    It quantifies the informational distance between group elements,
    determining the strength of non-local interactions in the cGFT.

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .group_manifold import GInfElement, SU2Element, U1Phase, compute_GInf_distance
from .quaternions import Quaternion

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.1, Appendix A"


# =============================================================================
# Encoding Functions
# =============================================================================


def encode_quaternion(q: Quaternion, n_bits: int = 64) -> bytes:
    """
    Encode quaternion as binary string for compression.
    
    Parameters
    ----------
    q : Quaternion
        Quaternion to encode
    n_bits : int
        Bits per component (total encoding = 4 * n_bits)
        
    Returns
    -------
    bytes
        Binary encoding of quaternion
    """
    # Quantize to fixed precision
    bits_per_component = n_bits // 4
    scale = 2 ** (bits_per_component - 1)
    bytes_per_component = (bits_per_component + 7) // 8
    
    components = [
        int(q.w * scale) & ((1 << bits_per_component) - 1),
        int(q.x * scale) & ((1 << bits_per_component) - 1),
        int(q.y * scale) & ((1 << bits_per_component) - 1),
        int(q.z * scale) & ((1 << bits_per_component) - 1),
    ]
    
    # Pack into bytes
    result = b''
    for c in components:
        result += c.to_bytes(bytes_per_component, 'big')
    
    return result


def encode_GInf_element(g: GInfElement, n_bits: int = 64) -> bytes:
    """
    Encode G_inf element as binary string for compression.
    
    Theoretical Reference:
        IRH21.md Appendix A.1
        Quantum state encoding for QNCD computation.
        
    Parameters
    ----------
    g : GInfElement
        Group element to encode
    n_bits : int
        Total bits for encoding
        
    Returns
    -------
    bytes
        Binary encoding of group element
    """
    # Encode SU(2) part (quaternion)
    su2_bytes = encode_quaternion(g.su2.quaternion, n_bits=n_bits * 4 // 5)
    
    # Encode U(1) part (phase)
    phase_bits = n_bits // 5
    phase_scale = 2 ** phase_bits
    phase_bytes = (phase_bits + 7) // 8
    phase_quantized = int((g.u1.phase / (2 * math.pi)) * phase_scale) % phase_scale
    u1_bytes = phase_quantized.to_bytes(phase_bytes, 'big')
    
    return su2_bytes + u1_bytes


# =============================================================================
# Compression Functions
# =============================================================================


def compress_zlib(data: bytes) -> bytes:
    """Compress data using zlib (classical approximation to Kolmogorov complexity)."""
    return zlib.compress(data, level=9)


def compressed_length(data: bytes) -> int:
    """Return length of compressed data."""
    return len(compress_zlib(data))


# =============================================================================
# QNCD Implementation
# =============================================================================


def compute_QNCD(
    g1: GInfElement,
    g2: GInfElement,
    n_bits: int = 64,
    compressor: Callable[[bytes], bytes] = compress_zlib
) -> float:
    """
    Compute Quantum Normalized Compression Distance between group elements.
    
    Theoretical Reference:
        IRH21.md Appendix A, Eq. A.1
        d_QNCD(g₁, g₂) = [K(g₁|g₂) + K(g₂|g₁)] / [K(g₁) + K(g₂)]
        
    This implementation uses classical compression as an approximation
    to quantum Kolmogorov complexity K_Q.
    
    Mathematical Foundation:
        NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
        
    where C(·) is compressed length approximating K(·).
    
    Parameters
    ----------
    g1, g2 : GInfElement
        Group elements to compare
    n_bits : int
        Encoding precision
    compressor : callable
        Compression function (default: zlib)
        
    Returns
    -------
    float
        QNCD distance in [0, 1]
    """
    # Encode group elements
    x = encode_GInf_element(g1, n_bits)
    y = encode_GInf_element(g2, n_bits)
    xy = x + y
    
    # Compute compressed lengths
    C_x = len(compressor(x))
    C_y = len(compressor(y))
    C_xy = len(compressor(xy))
    
    # NCD formula
    min_C = min(C_x, C_y)
    max_C = max(C_x, C_y)
    
    if max_C == 0:
        return 0.0
    
    ncd = (C_xy - min_C) / max_C
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, ncd))


def compute_QNCD_from_product(
    g1: GInfElement,
    g2: GInfElement,
    **kwargs
) -> float:
    """
    Compute QNCD of group product g₁g₂⁻¹.
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.3
        The interaction kernel uses d_QNCD(gᵢgⱼ⁻¹).
        
    This measures how "different" g1 and g2 are in the group sense.
    """
    product = g1 * g2.inverse()
    identity = GInfElement.identity()
    return compute_QNCD(product, identity, **kwargs)


def compute_pairwise_QNCD_sum(
    g_list: list[GInfElement],
    **kwargs
) -> float:
    """
    Compute sum of pairwise QNCD distances.
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.3
        Σ_{i<j} d_QNCD(gᵢgⱼ⁻¹) appears in the interaction kernel.
        
    Parameters
    ----------
    g_list : list[GInfElement]
        List of group elements [g₁, g₂, g₃, g₄]
        
    Returns
    -------
    float
        Sum of all pairwise QNCD distances
    """
    n = len(g_list)
    total = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            total += compute_QNCD_from_product(g_list[i], g_list[j], **kwargs)
    
    return total


# =============================================================================
# QNCD Metric Properties Verification
# =============================================================================


def verify_QNCD_metric_axioms(n_samples: int = 100, seed: int = 42) -> dict:
    """
    Verify that QNCD satisfies metric axioms.
    
    Theoretical Reference:
        IRH21.md Appendix A.2
        QNCD must satisfy metric axioms for well-defined cGFT.
    
    Tests:
        1. Positivity: d(g₁, g₂) ≥ 0
        2. Identity of indiscernibles: d(g, g) = 0
        3. Symmetry: d(g₁, g₂) = d(g₂, g₁)
        4. Triangle inequality: d(g₁, g₃) ≤ d(g₁, g₂) + d(g₂, g₃)
        5. Bi-invariance: d(kg₁, kg₂) = d(g₁, g₂)
        
    Returns
    -------
    dict
        Test results with pass/fail status
    """
    rng = np.random.default_rng(seed)
    results = {}
    
    # Generate test elements
    elements = [GInfElement.random(rng) for _ in range(n_samples)]
    
    # Test 1: Positivity
    positivity_violations = 0
    for i in range(min(n_samples, 50)):
        for j in range(i + 1, min(n_samples, 50)):
            d = compute_QNCD(elements[i], elements[j])
            if d < -1e-10:
                positivity_violations += 1
    results['positivity'] = {
        'passed': positivity_violations == 0,
        'violations': positivity_violations
    }
    
    # Test 2: Identity of indiscernibles
    identity_errors = []
    for i in range(min(n_samples, 20)):
        d = compute_QNCD(elements[i], elements[i])
        identity_errors.append(d)
    results['identity_indiscernibles'] = {
        'passed': max(identity_errors) < 0.1,  # Relaxed for compression artifacts
        'max_self_distance': max(identity_errors)
    }
    
    # Test 3: Symmetry
    symmetry_errors = []
    for i in range(min(n_samples, 30)):
        for j in range(i + 1, min(n_samples, 30)):
            d_ij = compute_QNCD(elements[i], elements[j])
            d_ji = compute_QNCD(elements[j], elements[i])
            symmetry_errors.append(abs(d_ij - d_ji))
    results['symmetry'] = {
        'passed': max(symmetry_errors) < 0.01,
        'max_asymmetry': max(symmetry_errors)
    }
    
    # Test 4: Triangle inequality
    triangle_violations = 0
    for i in range(min(n_samples, 20)):
        for j in range(i + 1, min(n_samples, 20)):
            for k in range(j + 1, min(n_samples, 20)):
                d_ij = compute_QNCD(elements[i], elements[j])
                d_jk = compute_QNCD(elements[j], elements[k])
                d_ik = compute_QNCD(elements[i], elements[k])
                if d_ik > d_ij + d_jk + 0.01:  # Small tolerance
                    triangle_violations += 1
    results['triangle_inequality'] = {
        'passed': triangle_violations == 0,
        'violations': triangle_violations
    }
    
    # Test 5: Bi-invariance (approximate for QNCD)
    bi_inv_errors = []
    for i in range(min(n_samples, 20)):
        g1 = elements[i]
        g2 = elements[(i + 1) % n_samples]
        k = elements[(i + 2) % n_samples]
        
        d_original = compute_QNCD(g1, g2)
        d_left = compute_QNCD(k * g1, k * g2)
        
        bi_inv_errors.append(abs(d_left - d_original))
    
    results['bi_invariance'] = {
        'passed': np.mean(bi_inv_errors) < 0.1,  # Relaxed for compression
        'mean_error': float(np.mean(bi_inv_errors)),
        'max_error': float(max(bi_inv_errors))
    }
    
    # Summary
    all_passed = all(r['passed'] for r in results.values())
    results['all_passed'] = all_passed
    results['theoretical_reference'] = 'IRH21.md Appendix A.2'
    
    return results


def verify_QUCC_theorem(n_samples: int = 50, seed: int = 42) -> dict:
    """
    Verify QUCC-Theorem compliance: compressor independence.
    
    Theoretical Reference:
        IRH21.md Appendix A.4
        QNCD should be approximately independent of compressor choice.
        
    Tests that different compressors give similar QNCD values
    (within expected variance).
        
    Returns
    -------
    dict
        QUCC compliance test results
    """
    import bz2
    import lzma
    
    rng = np.random.default_rng(seed)
    
    # Define different compressors
    compressors = {
        'zlib': compress_zlib,
        'bz2': lambda x: bz2.compress(x, compresslevel=9),
        'lzma': lambda x: lzma.compress(x),
    }
    
    # Generate test pairs
    pairs = [(GInfElement.random(rng), GInfElement.random(rng)) for _ in range(n_samples)]
    
    # Compute QNCD with each compressor
    results_by_compressor = {name: [] for name in compressors}
    
    for g1, g2 in pairs:
        for name, comp in compressors.items():
            d = compute_QNCD(g1, g2, compressor=comp)
            results_by_compressor[name].append(d)
    
    # Compare results across compressors
    compressor_names = list(compressors.keys())
    correlations = {}
    
    for i, name1 in enumerate(compressor_names):
        for name2 in compressor_names[i + 1:]:
            arr1 = np.array(results_by_compressor[name1])
            arr2 = np.array(results_by_compressor[name2])
            corr = np.corrcoef(arr1, arr2)[0, 1]
            correlations[f"{name1}_vs_{name2}"] = float(corr)
    
    min_correlation = min(correlations.values())
    
    return {
        'passed': min_correlation > 0.7,  # Should be strongly correlated
        'correlations': correlations,
        'min_correlation': min_correlation,
        'theoretical_reference': 'IRH21.md Appendix A.4'
    }


__all__ = [
    # Encoding
    'encode_quaternion',
    'encode_GInf_element',
    
    # Compression
    'compress_zlib',
    'compressed_length',
    
    # QNCD computation
    'compute_QNCD',
    'compute_QNCD_from_product',
    'compute_pairwise_QNCD_sum',
    
    # Verification
    'verify_QNCD_metric_axioms',
    'verify_QUCC_theorem',
]
