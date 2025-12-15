# Contributing to IRH v21.0 Computational Framework

## Overview

Thank you for your interest in contributing to the Intrinsic Resonance Holography v21.0 Computational Framework. This repository instantiates a unified theoretical framework that derives fundamental physical laws from quantum-informational principles. Contributions must maintain **theoretical fidelity** to the canonical manuscript `IRH21.md`.

## Core Principles

### 1. Theoretical Traceability

**Every computational construct must be traceable to its theoretical foundation.**

All functions, classes, and modules must include:

- **Section references**: `# IRH21.md §2.3.3`
- **Equation labels**: `# Implements Eq. 2.21-2.23`
- **Appendix citations**: `# Derivation in Appendix C.6`

Example docstring:

```python
def compute_beta_lambda(lambda_k: float, gamma_k: float) -> float:
    """
    Compute the beta function for the interaction coupling λ.
    
    THEORETICAL FOUNDATION: IRH21.md §1.2.2 (Eq. 1.13)
    
    The one-loop beta function for the dimensionless coupling λ̃:
    
        β_λ = ∂_t λ̃ = -2λ̃ + (9/8π²)λ̃²
    
    This drives the flow toward the Cosmic Fixed Point.
    
    Parameters
    ----------
    lambda_k : float
        Running interaction coupling at scale k
    gamma_k : float
        Running QNCD coupling at scale k
        
    Returns
    -------
    float
        Value of β_λ at the given couplings
        
    References
    ----------
    IRH21.md §1.2.2, Eq. 1.13
    Appendix B.1.1 for canonical dimension derivation
    """
    pass
```

### 2. Ontological Stratification

The directory structure enforces the **Epistemic Stratification Principle** (§4.1):

```
primitives/ → cgft/ → rg_flow/ → emergent_spacetime/ → ...
```

**Dependencies must form a directed acyclic graph (DAG)**:
- `primitives/` depends on nothing
- `cgft/` depends only on `primitives/`
- `rg_flow/` depends on `{primitives/, cgft/}`
- And so on...

**No circular dependencies are permitted.** Any violation indicates conceptual confusion about the emergent hierarchy.

### 3. Gauge Invariance

All implementations must preserve the fundamental symmetries:

- **Gauge invariance**: `S[φ] = S[φ']` for `φ' = φ(kg)`
- **Unitarity**: Quantum probability conservation
- **Hermiticity**: Real observables from Hermitian operators
- **Diffeomorphism invariance**: General covariance (Appendix H.2)

Contributions must pass `tests/theoretical_invariants/` before merging.

### 4. Numerical Convergence

All numerical computations must demonstrate:

- **Lattice convergence**: Results stable under grid refinement
- **RG step size independence**: Consistent across integration schemes
- **Compressor independence**: QNCD results satisfy QUCC-Theorem (Appendix A.4)

Include convergence tests in `tests/convergence/` for new numerical methods.

## Contribution Workflow

### 1. Issue Creation

Before contributing, create an issue describing:
- Which theoretical section/equation you're implementing
- Proposed approach and algorithm
- Expected validation strategy

### 2. Branch Naming

Use descriptive branch names:
```
feature/implement-beta-functions-eq-1.13
fix/qncd-metric-triangle-inequality
docs/spectral-dimension-derivation
```

### 3. Implementation

1. **Start with tests**: Write tests that verify theoretical properties
2. **Implement**: Write clean, documented code
3. **Validate**: Ensure all tests pass, especially theoretical invariants
4. **Document**: Update `THEORETICAL_CORRESPONDENCE.md`

### 4. Pull Request

PRs must include:
- [ ] Theoretical traceability (equation/section references)
- [ ] Unit tests for new functionality
- [ ] Theoretical invariant tests (if applicable)
- [ ] Convergence tests (for numerical methods)
- [ ] Updated `THEORETICAL_CORRESPONDENCE.md`
- [ ] Docstrings with IRH21.md citations

## Code Style

### Python Standards

- **PEP 8** compliance
- **Type hints** for all function signatures
- **NumPy-style docstrings**
- Maximum line length: 88 characters (Black formatter)

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `beta_functions.py` |
| Classes | PascalCase | `QuaternionField` |
| Functions | snake_case | `compute_fixed_point()` |
| Constants | UPPER_SNAKE | `FIXED_POINT_LAMBDA` |
| Variables | snake_case | `spectral_dim` |

### Physical Quantities

Use consistent naming for physical quantities:

```python
# Correct
lambda_star = 48 * np.pi**2 / 9  # Fixed-point coupling (Eq. 1.14)
alpha_inv = 137.035999084        # Fine-structure constant inverse

# Incorrect
l = 48 * np.pi**2 / 9  # Unclear
a = 137.035999084      # Ambiguous
```

## Testing Requirements

### Unit Tests

Located in `tests/unit/`, mirroring `src/` structure:

```python
# tests/unit/test_rg_flow/test_beta_functions.py

def test_beta_lambda_at_fixed_point():
    """β_λ should vanish at the Cosmic Fixed Point (§1.2.3)."""
    from src.rg_flow.beta_functions import compute_beta_lambda
    from src.rg_flow.fixed_points import LAMBDA_STAR, GAMMA_STAR
    
    beta = compute_beta_lambda(LAMBDA_STAR, GAMMA_STAR)
    assert np.isclose(beta, 0.0, atol=1e-10)
```

### Theoretical Invariant Tests

Located in `tests/theoretical_invariants/`:

```python
# tests/theoretical_invariants/test_gauge_invariance.py

def test_action_gauge_invariant():
    """S[φ] = S[φ'] for gauge-transformed φ' = φ(kg) (§1.1)."""
    from src.cgft.actions import compute_action
    from src.cgft.symmetries import gauge_transform
    
    phi = random_field()
    phi_prime = gauge_transform(phi, random_group_element())
    
    assert np.isclose(compute_action(phi), compute_action(phi_prime))
```

### Convergence Tests

Located in `tests/convergence/`:

```python
# tests/convergence/test_lattice_convergence.py

@pytest.mark.parametrize("N", [25, 50, 100, 200])
def test_spectral_dimension_converges(N):
    """d_spec should converge to 4 as lattice refines (§2.1)."""
    from src.emergent_spacetime.spectral_dimension import compute_spectral_dimension
    
    d_spec = compute_spectral_dimension(lattice_size=N)
    assert np.isclose(d_spec, 4.0, atol=1/N)  # O(1/N) convergence
```

## Documentation

### Inline Documentation

Every module should have a header explaining:
- Theoretical foundation (IRH21.md section)
- Key equations implemented
- Dependencies within the ontological hierarchy

### THEORETICAL_CORRESPONDENCE.md

Update this living document when:
- Implementing new equations
- Adding new modules
- Changing theoretical interpretations

### API Reference

Docstrings are automatically extracted to `docs/api_reference/`. Ensure complete documentation:

```python
def haar_measure_SU2(f: Callable, n_samples: int = 10000) -> float:
    """
    Integrate function f over SU(2) with respect to Haar measure.
    
    THEORETICAL FOUNDATION: IRH21.md §1.1, Appendix A.5
    
    The Haar measure on SU(2) is the unique left-right invariant
    measure, normalized to total volume 2π².
    
    Parameters
    ----------
    f : Callable
        Function SU(2) → ℂ to integrate
    n_samples : int, optional
        Monte Carlo sample count (default: 10000)
        
    Returns
    -------
    float
        Estimated integral ∫_{SU(2)} f(g) dg
        
    Notes
    -----
    Uses quaternionic parameterization: g = (cos(θ/2), sin(θ/2)n̂)
    where n̂ ∈ S² and θ ∈ [0, 2π].
    
    Examples
    --------
    >>> haar_measure_SU2(lambda g: 1.0)  # Total volume
    6.283185307...  # ≈ 2π²
    
    References
    ----------
    IRH21.md §1.1, Appendix A.5
    """
    pass
```

## Review Process

### Self-Review Checklist

Before submitting:
- [ ] All tests pass (`pytest tests/`)
- [ ] Code follows style guide
- [ ] Docstrings are complete with IRH21.md references
- [ ] `THEORETICAL_CORRESPONDENCE.md` updated
- [ ] No circular dependencies introduced
- [ ] Theoretical invariants preserved

### Reviewer Responsibilities

Reviewers verify:
1. **Theoretical correctness**: Does the implementation match IRH21.md?
2. **Code quality**: Is the code clean, readable, and maintainable?
3. **Test coverage**: Are edge cases and invariants tested?
4. **Documentation**: Are citations complete and accurate?

## Getting Help

- **Theoretical questions**: Open an issue with `[Theory]` prefix
- **Implementation questions**: Open an issue with `[Implementation]` prefix
- **Documentation**: Open an issue with `[Docs]` prefix

## Recognition

Contributors will be acknowledged in:
- Repository CONTRIBUTORS.md
- Future publications arising from this framework
- The IRH Consortium membership (when established)

---

*By contributing, you agree to maintain the theoretical integrity of IRH v21.0 and help establish the computational foundation for a unified theory of emergent reality.*
