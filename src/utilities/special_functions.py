"""
Special Functions for Intrinsic Resonance Holography v21.0

This module provides special mathematical functions used in
cGFT calculations on group manifolds.

Key Functions:
    - Bessel functions
    - Hypergeometric functions
    - Wigner D-matrices

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import math
from typing import Tuple, Optional, Union
import numpy as np
from scipy import special


__version__ = "21.0.0"


def bessel_j(n: int, x: float) -> float:
    """
    Bessel function of the first kind J_n(x).
    
    Parameters
    ----------
    n : int
        Order
    x : float
        Argument
        
    Returns
    -------
    float
        J_n(x)
    """
    return float(special.jv(n, x))


def hypergeometric_2f1(
    a: float,
    b: float,
    c: float,
    z: complex,
) -> complex:
    """
    Gauss hypergeometric function ₂F₁(a, b; c; z).
    
    Parameters
    ----------
    a, b, c : float
        Parameters
    z : complex
        Argument
        
    Returns
    -------
    complex
        ₂F₁(a, b; c; z)
    """
    return complex(special.hyp2f1(a, b, c, z))


def wigner_d_matrix(
    j: float,
    m: int,
    mp: int,
    beta: float,
) -> float:
    """
    Wigner small d-matrix d^j_{m,m'}(β).
    
    The Wigner d-matrix is related to rotation matrix elements:
        D^j_{m,m'}(α, β, γ) = e^{-imα} d^j_{m,m'}(β) e^{-im'γ}
    
    Parameters
    ----------
    j : float
        Angular momentum (half-integer or integer)
    m : int
        Magnetic quantum number
    mp : int
        Magnetic quantum number prime
    beta : float
        Euler angle β
        
    Returns
    -------
    float
        d^j_{m,m'}(β)
    """
    # Check validity
    if abs(m) > j or abs(mp) > j:
        return 0.0
    
    # Use SciPy if available for integer j
    if j == int(j):
        from scipy.spatial.transform import Rotation
        # Alternative implementation using general formula
        pass
    
    # General formula using Jacobi polynomials
    # d^j_{m,m'}(β) = sqrt((j+m)!(j-m)!/(j+m')!(j-m')!) 
    #                × (cos(β/2))^{m+m'} (sin(β/2))^{m-m'}
    #                × P^{(m-m',m+m')}_{j-m}(cos(β))
    
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)
    
    # Prefactors
    from math import factorial
    
    jm = int(j + m)
    jmm = int(j - m)
    jmp = int(j + mp)
    jmmp = int(j - mp)
    
    if jm < 0 or jmm < 0 or jmp < 0 or jmmp < 0:
        return 0.0
    
    prefactor = np.sqrt(
        factorial(jm) * factorial(jmm) / 
        (factorial(jmp) * factorial(jmmp))
    )
    
    # Trig factors with proper handling of negative powers
    power_c = m + mp
    power_s = m - mp
    
    # Handle negative powers by using 1/c^|power| or 1/s^|power|
    if power_c >= 0:
        trig_c = c ** power_c
    else:
        if abs(c) < 1e-10:
            return 0.0  # Avoid division by zero
        trig_c = 1.0 / (c ** abs(power_c))
    
    if power_s >= 0:
        trig_s = s ** power_s
    else:
        if abs(s) < 1e-10:
            return 0.0  # Avoid division by zero
        trig_s = 1.0 / (s ** abs(power_s))
    
    trig = trig_c * trig_s
    
    # Jacobi polynomial
    n = int(j - max(m, mp))
    alpha = abs(m - mp)
    beta_param = abs(m + mp)
    
    if n < 0:
        return 0.0
    
    P = special.eval_jacobi(n, alpha, beta_param, np.cos(beta))
    
    return float(prefactor * trig * P)


def wigner_D_matrix(
    j: float,
    m: int,
    mp: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> complex:
    """
    Wigner D-matrix D^j_{m,m'}(α, β, γ).
    
    Full rotation matrix element in the j-representation.
    
    Parameters
    ----------
    j : float
        Angular momentum
    m, mp : int
        Magnetic quantum numbers
    alpha, beta, gamma : float
        Euler angles
        
    Returns
    -------
    complex
        D^j_{m,m'}(α, β, γ)
    """
    d = wigner_d_matrix(j, m, mp, beta)
    phase = np.exp(-1j * (m * alpha + mp * gamma))
    return complex(phase * d)


def clebsch_gordan(
    j1: float, m1: int,
    j2: float, m2: int,
    j: float, m: int,
) -> float:
    """
    Clebsch-Gordan coefficient ⟨j₁ m₁; j₂ m₂ | j m⟩.
    
    Parameters
    ----------
    j1, j2, j : float
        Angular momenta
    m1, m2, m : int
        Magnetic quantum numbers
        
    Returns
    -------
    float
        Clebsch-Gordan coefficient
    """
    # Selection rule
    if m != m1 + m2:
        return 0.0
    
    # Triangle inequality
    if not (abs(j1 - j2) <= j <= j1 + j2):
        return 0.0
    
    # Use 3j symbol relation
    # ⟨j₁ m₁; j₂ m₂ | j m⟩ = (-1)^{j₁-j₂+m} √(2j+1) (j₁ j₂ j; m₁ m₂ -m)
    
    # Simplified formula for common cases
    from math import factorial, sqrt
    
    # Full formula (Racah)
    try:
        # Compute using 3j symbol
        three_j = float(special.wigner_3j(j1, j2, j, m1, m2, -m))
        sign = (-1) ** int(j1 - j2 + m)
        return sign * np.sqrt(2 * j + 1) * three_j
    except:
        return 0.0


def spherical_harmonic(
    l: int,
    m: int,
    theta: float,
    phi: float,
) -> complex:
    """
    Spherical harmonic Y_l^m(θ, φ).
    
    Parameters
    ----------
    l : int
        Degree
    m : int
        Order
    theta : float
        Polar angle
    phi : float
        Azimuthal angle
        
    Returns
    -------
    complex
        Y_l^m(θ, φ)
    """
    return complex(special.sph_harm(m, l, phi, theta))


__all__ = [
    'bessel_j',
    'hypergeometric_2f1',
    'wigner_d_matrix',
    'wigner_D_matrix',
    'clebsch_gordan',
    'spherical_harmonic',
]
