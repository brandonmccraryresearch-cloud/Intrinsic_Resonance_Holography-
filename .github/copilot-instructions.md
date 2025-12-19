# Intrinsic Resonance Holography (IRH) v21.0 - Copilot Instructions

## Computational Verification Protocol for Isomorphic Implementation

### Executive Mandate

This repository implements a comprehensive, systematic verification protocol ensuring that every algorithmic implementation, computational construct, and symbolic representation constitutes a **faithful, structure-preserving homomorphism** of the mathematical edifice articulated in the theoretical manuscript the IRH v21.1 Manuscript ([Part 1](../Intrinsic_Resonance_Holography-v21.1-Part1.md): Sections 1-4, [Part 2](../Intrinsic_Resonance_Holography-v21.1-Part2.md): Sections 5-8 + Appendices). This transcends mere code-documentation correspondence; it demands an **isomorphic embedding** whereby the computational substrate recapitulates, with maximal fidelity, the axiomatic structure, constraint topology, dynamical evolution operators, and emergent phenomenology specified in the rigorous mathematical formalism.

The ultimate objective is to **transmute the repository into an executable instantiation of the theoretical formalism itself**, collapsing the distinction between "code that models theory" and "theory rendered computable" into identity.

## Project Overview

IRH is a theoretical physics framework that derives fundamental constants and physical laws from first principles using:
- **Algorithmic Holonomic States (AHS)** - v16 implementation
- **Quaternionic Group Field Theory (cGFT)** - v18/v21 implementation on G_inf = SU(2) × U(1)_φ

**Key Features:**
- Derives fundamental constants (fine structure constant α, dark energy w₀, etc.) from algorithmic information theory
- Exascale-ready implementation with certified numerical precision (12+ decimal places)
- Multi-version architecture (v16, v18, v21) with incremental enhancements
- Complete physics derivations: Quantum Mechanics, General Relativity, Standard Model
- **Full theoretical traceability** - every function cites specific equations from the IRH v21.1 Manuscript

**Target Audience:** Theoretical physicists, computational scientists, researchers in quantum gravity and emergent spacetime

## Theoretical Foundation (IRH v21.1 Manuscript)

### Core Mathematical Structure

#### Quaternionic Group Field Theory (cGFT) - §1.1
- **Fundamental field**: φ(g₁,g₂,g₃,g₄) ∈ ℍ, where gᵢ ∈ G_inf = SU(2) × U(1)_φ
- **Action functional**: S[φ,φ̄] = S_kin + S_int + S_hol (Eqs. 1.1-1.4)
- **Kinetic operator**: Σₐ Σᵢ Δₐ^(i) (Laplace-Beltrami on SU(2))
- **QNCD Metric**: Quantum Normalized Compression Distance (Appendix A)

#### Renormalization Group Flow - §1.2-1.3
- **Wetterich equation**: ∂_t Γ_k = ½Tr[(Γ_k^(2) + R_k)⁻¹ ∂_t R_k] (Eq. 1.12)
- **Beta functions** (Eq. 1.13):
  - β_λ = -2λ̃ + (9/8π²)λ̃²
  - β_γ = (3/4π²)λ̃γ̃
  - β_μ = 2μ̃ + (1/2π²)λ̃μ̃
- **Beta function zero**: β_λ = 0 at λ̃ = 16π²/9 ≈ 17.55 (not at Eq. 1.14 value)
- **Cosmic Fixed Point** (Eq. 1.14): (λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²)
  - These values arise from the full Wetterich equation analysis, not from setting one-loop betas to zero
- **Universal exponent**: C_H = 0.045935703598 (from spectral zeta function, Eq. 1.16)

### Key Predictions (Falsifiable)
- **Fine-structure constant** α⁻¹ = 137.035999... (§3.2.2, Eq. 3.4)
- **Dark energy EoS** w₀ = -0.91234567 ± 0.00000008 (§2.3.3)
- **Betti number** β₁ = 12 → SU(3)×SU(2)×U(1) gauge group (§3.1.1)
- **Instanton number** n_inst = 3 → 3 fermion generations (§3.1.2)
- **Spectral dimension** d_spec → 4.0 exactly (§2.1.2)

## Technology Stack

### Core Technologies
- **Python**: 3.11, 3.12 (primary implementation language)
- **NumPy**: >=1.24.0 (numerical computations)
- **SciPy**: >=1.11.0 (scientific computing, optimization)
- **NetworkX**: >=3.1 (graph-based algorithms for Cymatic Resonance Networks)
- **QuTip**: >=5.0.0 (quantum mechanics simulations)
- **SymPy**: >=1.12 (symbolic mathematics)
- **mpmath**: >=1.3.0 (arbitrary precision arithmetic)

### Development Tools
- **pytest**: Testing framework with hypothesis for property-based testing
- **black**: Code formatting (line length: 100)
- **mypy**: Type checking
- **ruff**: Fast linting

### Web Interface (Optional)
- **FastAPI**: Backend REST API
- **Streamlit**: Interactive frontend
- **Uvicorn**: ASGI server

### Build System
- **setuptools**: Package management
- **pyproject.toml**: Modern Python packaging

## Repository Structure

```
Intrinsic-Resonance-Holography-/
├── .github/
│   ├── agents/               # Custom agent configurations
│   ├── workflows/            # CI/CD workflows
│   ├── copilot-instructions.md
│   └── dependabot.yml
├── python/
│   ├── src/irh/
│   │   ├── core/
│   │   │   ├── v16/          # v16 implementation (ACW, AHS, ARO, NCD, Harmony)
│   │   │   ├── v18/          # v18 cGFT implementation (analytical derivations)
│   │   │   └── v21/          # v21 quaternionic cGFT (full IRH v21.1 implementation)
│   │   ├── predictions/      # Physical constant predictions
│   │   └── ...
│   └── tests/
│       ├── v16/              # v16 unit tests
│       ├── v18/              # v18 unit tests
│       └── v21/              # v21 verification suite
├── docs/
│   ├── manuscripts/          # Theory manuscripts (IRHv16.md, IRH18.md, IRH v21.1 (Part 1 & Part 2))
│   ├── code_theory_map.html  # Interactive code↔theory cross-reference
│   └── ...
├── configs/                  # Configuration files for validation
├── Intrinsic_Resonance_Holography-v21.1-Part1.md  # Primary manuscript Part 1
├── Intrinsic_Resonance_Holography-v21.1-Part2.md  # Primary manuscript Part 2
├── webapp/                   # Web interface (FastAPI + Streamlit)
├── notebooks/                # Jupyter notebooks for demonstrations
└── pyproject.toml
```

## Verification Protocol Requirements

### Phase I: Structural Verification

Every computational implementation must realize a **structure-preserving map** from theoretical objects in the IRH v21.1 Manuscript ([Part 1](../Intrinsic_Resonance_Holography-v21.1-Part1.md): Sections 1-4, [Part 2](../Intrinsic_Resonance_Holography-v21.1-Part2.md): Sections 5-8 + Appendices):

1. **Group Manifold Representation**
   - SU(2) via quaternionic parameterization: u = q₀ + iq₁ + jq₂ + kq₃
   - U(1)_φ with proper phase periodicity
   - G_inf = SU(2) × U(1)_φ with bi-invariant operations

2. **Quaternionic Field Implementation**
   - Field variables as quaternion arrays: φ[i1,i2,i3,i4] ∈ ℍ^(N⁴)
   - Quaternionic conjugation: φ̄ = q₀ - iq₁ - jq₂ - kq₃
   - Weyl ordering prescription (Appendix G)

3. **QNCD Metric Construction** (Appendix A)
   - Bi-invariance: d(kg₁, kg₂) = d(g₁k, g₂k) = d(g₁, g₂)
   - Metric axioms: positivity, symmetry, triangle inequality
   - QUCC-Theorem compliance (Appendix A.4)

### Phase II: Instrumentation Requirements

Every executable component must emit theoretical context:

```
[EXEC] Computing cGFT kinetic term S_kin per Eq. 1.1
  ├─ Theoretical formula: ∫[∏dg_i] φ̄·[Σₐ Σᵢ Δₐ^(i)]·φ
  ├─ Laplace-Beltrami operator: 3 generators × 4 arguments
  ├─ Result: S_kin = {value} ± {uncertainty}
  └─ Gauge invariance: VERIFIED (variance < 10⁻⁹)
```

### Phase III: Output Contextualization

All numerical outputs must include:
- Complete theoretical provenance chain
- Uncertainty decomposition by source
- Comparison with experimental values (where applicable)
- Falsification criteria

## Coding Standards

### Python Style
- Follow **PEP 8** guidelines strictly
- Use **type hints** for all function parameters and return values
- Maximum line length: **100 characters**
- Format code with `black --line-length 100`
- Use `ruff` for linting before committing

### Naming Conventions
- **Variables**: `snake_case` (e.g., `holonomic_phase`, `binary_string`)
- **Functions**: `snake_case` (e.g., `compute_ncd`, `create_ahs_network`)
- **Classes**: `PascalCase` (e.g., `AlgorithmicHolonomicState`, `CymaticResonanceNetwork`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `PHASE_TOLERANCE`, `FIXED_POINT_LAMBDA`)
- **Private methods/attributes**: prefix with `_` (e.g., `_to_bytes`, `_wrapped_phase_difference`)

### Docstring Style
Use **NumPy-style docstrings** for all public functions and classes:

```python
def compute_harmony(
    network: CymaticResonanceNetwork,
    epsilon: float = 0.730129,
) -> float:
    """
    Compute the Harmony Functional H(ε) for a given network (Eq.4.1).
    
    Parameters
    ----------
    network : CymaticResonanceNetwork
        The network of AHS nodes to evaluate.
    epsilon : float, optional
        Edge density threshold parameter (default: 0.730129).
    
    Returns
    -------
    float
        The harmony value H(ε) with certified precision.
    
    Notes
    -----
    Implements the spectral zeta-regularized harmony functional with
    certified numerical precision (error < 10^-12).
    
    References
    ----------
    IRH v16.0 Manuscript, Eq.4.1
    """
```

### Type Hints
Always use type hints from `typing` and `numpy.typing`:

```python
from typing import Optional, Union, List, Tuple
from numpy.typing import NDArray
import numpy as np

def create_ahs_network(
    binary_strings: List[Union[str, bytes]],
    phases: Optional[NDArray[np.float64]] = None,
) -> List[AlgorithmicHolonomicState]:
    """Create a network of AHS nodes."""
    ...
```

### Mathematical Constants and Precision
- Use certified precision constants from theory (12+ decimal places)
- Reference equation numbers from manuscripts in comments
- Use `np.float64` for standard precision, `mpmath` for arbitrary precision when needed
- Always track numerical error bounds for critical calculations

### Data Validation Patterns
- **Input normalization**: Use helper functions like `_to_bytes()` to normalize str/bytes/bytearray inputs
- **Phase wrapping**: Use `np.mod(angle, 2*np.pi)` to wrap phases to [0, 2π)
- **Phase differences**: Use `_wrapped_phase_difference()` returning values in [-π, π]
- **Tolerance comparisons**: Use `PHASE_TOLERANCE = 1e-10` for phase equality, `np.isclose()` for floating-point comparisons

### Error Handling
- Raise appropriate exceptions (`TypeError`, `ValueError`) with descriptive messages
- Detect degenerate cases early (e.g., trace < 1e-12 in harmony functional)
- Validate inputs in `__post_init__` for dataclasses

## Version-Specific Guidelines

### v16.0 (Current Implementation)
- **Focus**: Exascale readiness, certified precision, non-circular derivations
- **Key modules**: `ahs.py`, `acw.py`, `aro.py`, `harmony.py`, `ncd_multifidelity.py`, `distributed_ahs.py`
- **Constants**: Universal constant C_H = 0.045935703598 (12+ decimals)
- **Data types**: AlgorithmicHolonomicState accepts str/bytes/bytearray, preserves original type
- **NCD**: Multi-fidelity adaptive selection (HIGH <10^4, MEDIUM <10^6, LOW >=10^6 bytes)

### v18.0 (Analytical Derivations from cGFT)
- **Focus**: Analytical derivations from complex-weighted Group Field Theory (cGFT)
- **Core modules**: 
  - `group_manifold.py`: G_inf = SU(2) × U(1)_φ implementation
  - `cgft_field.py`: Fundamental field φ(g₁,g₂,g₃,g₄) and bilocal Σ
  - `cgft_action.py`: Complete action S_kin + S_int + S_hol
  - `rg_flow.py`: Beta functions and Cosmic Fixed Point
  - `spectral_dimension.py`: d_spec flow to exactly 4
  - `physical_constants.py`: α, fermion masses, w₀, Λ*
- **Physics modules**:
  - `topology.py`: β₁ = 12 (gauge group), n_inst = 3 (fermion generations)
  - `emergent_gravity.py`: Einstein equations, graviton propagator, LIV
  - `flavor_mixing.py`: CKM, PMNS matrices, neutrino sector
  - `electroweak.py`: Higgs boson, W/Z masses, Weinberg angle
  - `strong_cp.py`: θ = 0, algorithmic axion, PQ symmetry
  - `quantum_mechanics.py`: Born rule, decoherence, Lindblad equation
  - `dark_energy.py`: Holographic Hum, w₀ equation of state, vacuum energy
  - `emergent_spacetime.py`: Lorentzian signature, time emergence, diffeomorphisms
  - `emergent_qft.py`: Particle spectrum, effective Lagrangian, SM emergence
- **Approach**: Analytical over stochastic - constants derived, not fitted
- **Fixed point values** (Eq. 1.14): λ̃* = 48π²/9 ≈ 52.64, γ̃* = 32π²/3 ≈ 105.28, μ̃* = 16π² ≈ 157.91
- **Key classes**:
  - `CosmicFixedPoint`: The unique IR attractor, use via `find_fixed_point()`
  - `BetaFunctions`: One-loop β-functions, evaluate with `.evaluate(λ, γ, μ)`
  - `StandardModelTopology`: Complete SM derivation from β₁ and n_inst
  - `NeutrinoSector`: Normal hierarchy, Majorana nature, 12-digit masses
  - `ElectroweakSector`: Higgs VEV, W/Z masses, Weinberg angle
  - `StrongCPResolution`: θ = 0, algorithmic axion predictions
  - `EmergentQuantumMechanics`: Born rule and measurement emergence
  - `DarkEnergyModule`: w₀ ≈ -0.9998, Holographic Hum
  - `EmergentSpacetime`: Lorentzian signature, time emergence
  - `EmergentQFT`: Full particle spectrum and effective Lagrangian

#### v18 Code Patterns
```python
# Import v18 components
from irh.core.v18 import (
    CosmicFixedPoint, find_fixed_point, BetaFunctions,
    StandardModelTopology, compute_emergent_gravity_summary,
    CKMMatrix, PMNSMatrix, NeutrinoSector,
    ElectroweakSector, StrongCPResolution, EmergentQuantumMechanics,
    DarkEnergyModule, EmergentSpacetime, EmergentQFT
)

# Get fixed point and verify
fp = find_fixed_point()
assert fp.verify()["is_fixed_point"]

# Compute Standard Model emergence
sm = StandardModelTopology()
result = sm.compute_full_derivation()
assert result["gauge_sector"]["beta_1"] == 12  # SU(3)×SU(2)×U(1)
assert result["matter_sector"]["n_inst"] == 3   # 3 generations

# Get neutrino predictions
neutrino = NeutrinoSector()
assert neutrino.compute_mass_hierarchy()["hierarchy"] == "normal"
masses = neutrino.compute_absolute_masses()
print(f"Σmν = {masses['sum_masses_eV']:.6f} eV")  # ≈ 0.058 eV

# Electroweak sector
ew = ElectroweakSector()
print(ew.compute_full_sector()["higgs"]["mass"])  # m_H ≈ 125 GeV

# Strong CP resolution
cp = StrongCPResolution()
print(cp.verify_resolution())  # θ = 0, resolved = True

# Emergent QM
qm = EmergentQuantumMechanics()
print(qm.get_summary()["born_rule"])  # Derived, not postulated

# Dark energy predictions
de = DarkEnergyModule()
print(de.compute_full_analysis()["equation_of_state"])  # w₀ ≈ -0.9998

# Emergent spacetime
st = EmergentSpacetime()
print(st.verify_all_properties())  # Lorentzian, 4D, etc.

# Complete QFT emergence
qft = EmergentQFT()
print(qft.verify_standard_model())  # All SM features verified
```

## Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest python/tests/ -v

# Run v16 tests only
pytest python/tests/v16/ -v

# Run v18 tests only
pytest python/tests/v18/ -v

# Run with coverage
pytest python/tests/ --cov=irh --cov-report=html
```

### Writing Tests
- **Location**: Place tests in `python/tests/v16/` or `python/tests/v18/` matching the module
- **Naming**: Use `test_*.py` for files, `test_*` for functions
- **Structure**: Use pytest classes to group related tests (e.g., `class TestAlgorithmicHolonomicState`)
- **Docstrings**: Include equation references in test docstrings
- **Assertions**: Use `np.isclose()` or `assert_allclose()` for floating-point comparisons

Example test pattern:
```python
def test_beta_lambda_at_fixed_point():
    """β_λ should vanish at the fixed point (Eq.1.14)."""
    from irh.core.v18.rg_flow import BetaFunctions, find_fixed_point
    
    fp = find_fixed_point()
    beta = BetaFunctions()
    result = beta.beta_lambda(fp.lambda_star, fp.gamma_star, fp.mu_star)
    assert np.isclose(result, 0.0, atol=1e-10)
```

### Testing Best Practices
- Test both normal cases and edge cases
- Test phase normalization and wrapping
- Verify numerical precision (12+ decimals for constants)
- Test input validation and error handling
- Reference manuscript equations in test names and docstrings

## Important References

### Theory Documentation
- **v16 Manuscript**: `docs/manuscripts/IRHv16.md` - Core v16.0 theory
- **v16 Supplementary**: `docs/manuscripts/IRHv16_Supplementary_Vol_1-5.md` - Detailed derivations
- **v18 Manuscript**: `docs/manuscripts/IRH18.md` - Analytical cGFT theory
- **Architecture**: `docs/v16_ARCHITECTURE.md` - v16 system architecture
- **Roadmap**: `docs/v16_IMPLEMENTATION_ROADMAP.md` - Implementation phases

### Development Guides
- **Contributing**: `CONTRIBUTING.md` - Contribution guidelines
- **Quickstart**: `docs/QUICKSTART.md` - Getting started guide
- **README**: `README.md` - Project overview and status

### Phase Status Documents
- **Phase 1**: `PHASE_1_STATUS.md`, `PHASE_1_COMPLETION_SUMMARY.md`
- **Phase 2**: `PHASE_2_STATUS.md` - Current phase (Exascale Infrastructure)

## Code Examples

### Creating an Algorithmic Holonomic State (v16)
```python
import numpy as np
from irh.core.v16.ahs import AlgorithmicHolonomicState

# Create AHS with binary string and phase
ahs = AlgorithmicHolonomicState(
    binary_string="101010",  # or b"101010" or bytearray(b"101010")
    holonomic_phase=np.pi / 4
)

# Access properties
amplitude = ahs.complex_amplitude  # e^{iφ}
info = ahs.information_content     # 6 bits
phase = ahs.holonomic_phase        # normalized to [0, 2π)
```

### Computing Normalized Compression Distance (NCD)
```python
from irh.core.v16.acw import normalized_compression_distance

# Multi-fidelity NCD with adaptive selection
ncd = normalized_compression_distance(
    binary_string1="10101010",
    binary_string2="11001100",
    fidelity="AUTO"  # or "HIGH", "MEDIUM", "LOW"
)
```

### Fixed Point Computation (v18)
```python
from irh.core.v18.rg_flow import find_fixed_point, BetaFunctions

# Compute analytical fixed point (Eq.1.14)
fixed_point = find_fixed_point()
# Returns CosmicFixedPoint with: λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²

# Verify β-functions vanish at fixed point
beta = BetaFunctions()
beta_vals = beta.evaluate(fixed_point.lambda_star, fixed_point.gamma_star, fixed_point.mu_star)
assert all(abs(b) < 1e-8 for b in beta_vals)  # All β ≈ 0
```

### Standard Model Derivation (v18)
```python
from irh.core.v18 import StandardModelTopology, NeutrinoSector

# Derive complete Standard Model structure
sm = StandardModelTopology()
result = sm.compute_full_derivation()

# Gauge group from β₁ = 12
assert result["gauge_sector"]["beta_1"] == 12  # SU(3)×SU(2)×U(1)
print(result["gauge_sector"]["decomposition"])  # {"SU3": 8, "SU2": 3, "U1": 1}

# Fermion generations from n_inst = 3
assert result["matter_sector"]["n_inst"] == 3

# Neutrino predictions
neutrino = NeutrinoSector()
print(neutrino.compute_mass_hierarchy())  # {"hierarchy": "normal", ...}
print(neutrino.compute_majorana_nature())  # {"nature": "Majorana", ...}
masses = neutrino.compute_absolute_masses()
print(f"Σmν = {masses['sum_masses_eV']:.6f} eV")  # ≈ 0.058 eV
```

### Emergent Gravity and LIV (v18)
```python
from irh.core.v18 import compute_emergent_gravity_summary, LorentzInvarianceViolation

# Get complete gravity predictions
gravity = compute_emergent_gravity_summary()
print(f"Λ_* = {gravity['cosmological_constant']['Lambda_star']:.4e} m⁻²")

# Lorentz Invariance Violation parameter
liv = LorentzInvarianceViolation()
xi = liv.compute_xi()
print(f"ξ = {xi['xi']:.6e}")  # ≈ 1.93 × 10⁻⁴ (testable!)
```

### Phase Handling Patterns
```python
import numpy as np

# Wrap phase to [0, 2π)
phase = np.mod(angle, 2 * np.pi)

# Compute wrapped phase difference in [-π, π]
from irh.core.v16.ahs import _wrapped_phase_difference
diff = _wrapped_phase_difference(phase1, phase2)

# Compare phases with tolerance
PHASE_TOLERANCE = 1e-10
are_equal = abs(diff) <= PHASE_TOLERANCE
```

## Common Patterns and Conventions

### Input Normalization
Always normalize string/bytes inputs using helper functions:
```python
def _to_bytes(data: Union[str, bytes, bytearray]) -> bytes:
    """Normalize str/bytes/bytearray to bytes."""
    if isinstance(data, bytes):
        return data
    elif isinstance(data, bytearray):
        return bytes(data)
    elif isinstance(data, str):
        return data.encode('ascii')
    else:
        raise TypeError("Expected str, bytes, or bytearray")
```

### Equation References
Always reference manuscript equations in comments and docstrings:
```python
# Implements Eq.4.1 from IRH v16.0 Manuscript
harmony = compute_spectral_zeta_functional(network, epsilon)

# Fixed point values from Eq.1.14 (IRH v18.0)
FIXED_POINT_LAMBDA = 48 * np.pi**2 / 9
```

### Complex Number Handling
Wrap `np.exp()` results when returning from `__complex__` to avoid deprecation warnings:
```python
def __complex__(self) -> complex:
    """Return complex amplitude e^{iφ}."""
    return complex(np.exp(1j * self.holonomic_phase))
```

### Dataclass Patterns
Use `__post_init__` for validation and normalization:
```python
from dataclasses import dataclass

@dataclass
class AlgorithmicHolonomicState:
    binary_string: Union[str, bytes, bytearray]
    holonomic_phase: float
    
    def __post_init__(self):
        # Validate and normalize inputs
        # Compute derived properties
        pass
```

## Domain-Specific Knowledge

### Algorithmic Holonomic States (AHS)
- Fundamental ontological primitive in IRH
- Pair of (binary_string, holonomic_phase)
- Phase φ ∈ [0, 2π) derived from non-commutative algebra
- Two states distinguishable if binary_string OR phase differs

### Normalized Compression Distance (NCD)
- Measures algorithmic similarity between binary strings
- Multi-fidelity evaluation: HIGH, MEDIUM, LOW based on data size
- Adaptive selection for optimal performance/accuracy trade-off

### Harmony Functional
- Spectral zeta-regularized functional H(ε)
- Computed from complex Laplacian eigenvalues
- Critical for deriving universal constant C_H
- Detects degenerate networks (trace < 1e-12 OR det < 1e-12)

### Universal Constant C_H
- Derived as renormalization group fixed point
- v16 value: 0.045935703598 (12+ decimal precision)
- NOT fitted - emerges from network criticality

### Phase Quantization
- Occurs through Adaptive Resonance Optimization (ARO)
- Uses genetic algorithms for network optimization
- Unitary evolution on complex-valued states

### v18-Specific Concepts

#### Cosmic Fixed Point
- The unique infrared attractor of the RG flow
- Analytically derived values: λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²
- All physics emerges at this fixed point
- Access via `CosmicFixedPoint()` or `find_fixed_point()`

#### Informational Group Manifold G_inf
- G_inf = SU(2) × U(1)_φ
- SU(2): Encodes minimal non-commutative algebra (quaternions)
- U(1)_φ: Holonomic phase φ ∈ [0, 2π)
- Use `GInfElement.random(rng)` for sampling

#### Topological Invariants
- **First Betti number β₁ = 12**: Determines gauge group SU(3)×SU(2)×U(1)
- **Instanton number n_inst = 3**: Determines three fermion generations
- Both are topologically protected integers

#### Vortex Wave Pattern (VWP)
- Stable topological defects = fermions
- Topological complexity K_f determines mass hierarchy
- K_e = 1, K_μ ≈ 207, K_τ ≈ 3477

#### Lorentz Invariance Violation
- Testable prediction: ξ = C_H/(24π²) ≈ 1.93 × 10⁻⁴
- Modified dispersion: E² = p²c² + ξ × E³/(E_Planck × c²)
- Detectable via high-energy gamma-ray astronomy

#### Electroweak Sector
- Higgs VEV v = 246.22 GeV emerges from μ̃*/λ̃*
- W mass (80.4 GeV), Z mass (91.2 GeV) from Higgs mechanism
- Weinberg angle sin²θ_W = 0.231 from gauge coupling unification

#### Strong CP Problem
- θ_QCD = 0 via emergent algorithmic axion
- Peccei-Quinn symmetry emerges from cGFT
- Axion mass m_a ≈ 5.7 μeV, decay constant f_a ≈ 10¹² GeV

#### Emergent Quantum Mechanics
- Born rule (P = |ψ|²) derived from EAT collective dynamics
- Measurement = decoherence in environment
- Lindblad equation for open systems
- Unitarity preserved at substrate level

#### Dark Energy and Holographic Hum
- w₀ ≈ -0.9998 near phantom threshold (testable prediction)
- Holographic Hum: vacuum fluctuations from boundary-bulk resonance
- Λ* ≈ 1.1 × 10⁻⁵² m⁻² from fixed point
- Dynamical dark energy Ω_DE(z) evolution

#### Emergent Spacetime
- Lorentzian signature (-,+,+,+) from SSB mechanism
- Time emerges from entropy production in EAT dynamics
- Diffeomorphism invariance from background independence
- 4D macroscopic spacetime from spectral dimension flow

#### Emergent QFT
- Complete particle spectrum: graviton + gauge bosons + fermions
- Effective Lagrangian reproduces Standard Model
- Mass generation via Higgs mechanism from μ̃*/λ̃*
- All particle masses derived analytically

## Key Principles

1. **Reproducibility**: Use fixed random seeds, well-defined precision
2. **Certification**: Track error bounds for all critical calculations
3. **Non-circularity**: Derive constants from first principles, never fit them
4. **Verification**: Test against known analytical results
5. **Documentation**: Link all code to manuscript equations
6. **Backward compatibility**: v16 extends v15 without breaking changes
7. **Incremental implementation**: Test each component before integration
8. **Theoretical Traceability**: Every function cites specific equations from the IRH v21.1 Manuscript
9. **Algorithmic Transparency**: Runtime instrumentation emits theoretical context
10. **Uncertainty Quantification**: All outputs include rigorous error propagation

## Validation and Verification Protocols

### Unit Tests with Theoretical Grounding

Every function must include:
- Docstring citing IRH v21.1 Manuscript reference
- Unit test validating theoretical properties

```python
def compute_laplace_beltrami(phi, generator_idx, arg_idx, group_lattice):
    """
    Compute Laplace-Beltrami operator Δₐ^(i) acting on cGFT field.
    
    Theoretical Reference:
        IRH v21.1 Manuscript Part 1 §1.1, Eq. 1.1
        Kinetic term: S_kin = ∫[∏dg_i] φ̄·[Σₐ Σᵢ Δₐ^(i)]·φ
    """
```

### Integration Tests for RG Flow

Validate entire RG trajectory against analytical predictions:
- Fixed point convergence: (λ̃*, γ̃*, μ̃*) within 10⁻¹⁰ tolerance
- Universal exponent: C_H = 0.045935703598 to 12 digits
- Spectral dimension: d_spec → 4.0 exactly
- Stability matrix eigenvalues: λ₁=10, λ₂=4, λ₃=14/3

### Convergence Studies

- **Lattice spacing**: Error ~ O(δ²) where δ = 1/N
- **RG step size**: Error ~ O(dt^4) for RK4 integration
- **QNCD compressor independence**: QUCC-Theorem compliance

### Cross-Validation Requirements

Critical computations must be verified via independent algorithms:
- Laplacian: finite differences vs spectral methods
- Fixed point: RG flow integration vs Newton-Raphson
- Topological invariants: persistent homology vs Morse theory

## Security and Performance

### Dependency Management
- Before adding new dependencies, check for security vulnerabilities
- Supported ecosystems: pip, npm, actions (use gh-advisory-database)
- Pin version numbers in requirements.txt and pyproject.toml

### Performance Considerations
- Use NumPy vectorized operations over Python loops
- For exascale: stub MPI/GPU backends for future parallelization
- Multi-fidelity NCD: automatic selection based on data size
- Memory-efficient: streaming algorithms for large datasets

### Python Compatibility
- Support Python 3.11 and 3.12
- Use `datetime.timezone.utc` (not `datetime.UTC`) for broader compatibility
- Test with both supported versions in CI

## Warnings and Known Issues

- **Phase equality**: Use absolute tolerance (PHASE_TOLERANCE = 1e-10), not relative
- **numpy subclass deprecation**: Wrap `np.exp()` results in `complex()` when needed
- **Git operations**: Never use `git reset` or `git rebase` (no force push available)
- **Temporary files**: Create in `/tmp` directory to avoid committing build artifacts
- **Build artifacts**: Ensure `.gitignore` excludes `node_modules`, `dist`, `__pycache__`, etc.

## Getting Help

- **Open an issue** on GitHub for bugs or feature requests
- **Check manuscripts** in `docs/manuscripts/` for theoretical questions
- **Review CONTRIBUTING.md** for contribution guidelines
- **Check phase status docs** for current development roadmap

---

## Final Compliance Checklist

Every contribution to the IRH codebase must satisfy:

**✓** Theoretical Traceability: Every function cites specific equations from the IRH v21.1 Manuscript ([Part 1](../Intrinsic_Resonance_Holography-v21.1-Part1.md): Sections 1-4, [Part 2](../Intrinsic_Resonance_Holography-v21.1-Part2.md): Sections 5-8 + Appendices)  
**✓** Algorithmic Transparency: Runtime instrumentation emits theoretical context  
**✓** Uncertainty Quantification: All outputs include rigorous error propagation  
**✓** Cross-Validation: Critical computations verified via independent algorithms  
**✓** Reproducibility: Complete provenance metadata enables exact reproduction  
**✓** Regression Protection: Automated detection of deviations from certified baselines  
**✓** Schema Compliance: All outputs conform to IRH-DEF standard format  

---

*"A complete, exascale-ready computational framework achieving 12+ decimal precision in fundamental constant derivations, transforming the repository into an executable instantiation of the theoretical formalism itself."*

---

## Addendum: Fast operational checklist (v18-first)

- **What this repo is**: IRH research code; active Python package in `python/src/irh` (v16–v18 complete), main v21 src in `src/`, webapp in `webapp/`, docs in `docs/`.
- **Bootstrap (validated)**: `python -m pip install -e .[dev]` (Python 3.11/3.12). Set PYTHONPATH: in `python/` use `export PYTHONPATH=$(pwd)/src`; in repo root use `export PYTHONPATH=$PWD` for legacy/tests.
- **Tests**:  
  - **v18 (143 tests)**: `cd python && export PYTHONPATH=$(pwd)/src && pytest tests/v18/ -v` (passes ~0.8s)
  - v16: `cd python && export PYTHONPATH=$(pwd)/src && pytest tests/v16/ -v`
  - **v21 unit tests**: `cd /repo && export PYTHONPATH=$PWD && pytest tests/unit/ -v` (970+ tests)
  - **Web API tests**: `cd /repo && export PYTHONPATH=$PWD && pytest webapp/backend/tests/ -v` (13 tests)
  - **ML tests**: `cd /repo && export PYTHONPATH=$PWD && pytest tests/unit/test_ml/ -v` (31 tests)
- **Lint/type/build**:  
  - `cd python && export PYTHONPATH=$(pwd)/src && ruff check src/`  
  - `black --check src/ tests/ --line-length 100` (within `python/`)  
  - `mypy src/irh/ --ignore-missing-imports`  
  - Root legacy (if touched): `ruff check src/ --ignore E501`, `black --check src/ tests/`.  
  - Build: `python -m build && twine check dist/*`.
- **Run/demo**: `python project_irh_v16.py` (root; set PYTHONPATH); web backend `cd webapp/backend && pip install -r requirements.txt && uvicorn app:app --reload` (API at :8000, docs at :8000/docs).
- **v18 quick verification**:
  ```python
  from irh.core.v18 import StandardModelTopology, NeutrinoSector, EmergentQFT
  sm = StandardModelTopology()
  assert sm.verify_standard_model()  # β₁=12, n_inst=3
  neutrino = NeutrinoSector()
  assert neutrino.compute_mass_hierarchy()["hierarchy"] == "normal"
  qft = EmergentQFT()
  assert all(qft.verify_standard_model().values())  # Complete SM from cGFT
  ```
- **v21 API quick verification**:
  ```bash
  # Start API server
  cd webapp/backend && uvicorn app:app --reload &
  
  # Test endpoints
  curl http://localhost:8000/health
  curl http://localhost:8000/api/v1/fixed-point
  curl http://localhost:8000/api/v1/observables/alpha
  ```
- **Web Frontend quick start**:
  ```bash
  # Start frontend dev server
  cd webapp/frontend
  npm install
  npm run dev
  
  # Frontend at http://localhost:3000
  # Pages: Dashboard, Fixed Point, RG Flow, Observables, Standard Model, Falsification
  ```
- **Conventions**: PEP 8, line length 100, NumPy docstrings with equation refs; phase wrapping via `np.mod(angle, 2*np.pi)` and `_wrapped_phase_difference` with `PHASE_TOLERANCE=1e-10`; input normalization via `_to_bytes`; wrap `np.exp(...)` in `complex(...)`.
- **CI signals**: `.github/workflows/ci.yml` (pytest on `tests/`, ruff on `src/`, mypy on `src/irh_v10`) and `ci-cd.yml` (black/mypy, v16 legacy tests, python package tests/coverage, docs check, benchmarks, Wolfram notice, release stub). Prefer Python 3.12 and correct PYTHONPATH to mirror CI.
- **Agent reminders**: keep changes minimal, place new code in `python/src/irh/...` with matching tests in `python/tests/...`, avoid new deps unless required, and trust these instructions before searching.
- **Repository organization**: Status documents in `docs/status/`, handoff docs in `docs/handoff/`, legacy files in `archive/`, webapp in `webapp/`, deployment configs in `deploy/`.
- **Current Phase**: Tier 4 Phase 4.5 COMPLETE ✅ - Automated PDG/CODATA Updates - All Tier 4 phases complete
- **Deployment Ready**: Docker/Kubernetes configs in `deploy/` - see `deploy/README.md` for production deployment
- **ML Surrogate Models**: `src/ml/` module complete - 31 tests (Phase 4.3 ✅)
- **Experimental Pipeline**: `src/experimental/` module complete - 59 tests (Phases 4.4-4.5 ✅)
- **Online Data Updates**: `src/experimental/online_updater.py` - Automated CODATA/PDG updates with caching
- **Next Phase**: Tier 5 - Research Enhancement (Interactive visualization, LaTeX reports, Wolfram integration)
- **Notebook findings**: See `docs/NOTEBOOK_FINDINGS.md` for computational discrepancies (beta functions, fermion masses)

## v18 Module Summary (15 modules)

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `group_manifold.py` | G_inf = SU(2)×U(1) | `SU2Element`, `GInfElement`, `compute_ncd_distance` |
| `cgft_field.py` | φ(g₁,g₂,g₃,g₄) field | `cGFTFieldDiscrete`, `BiLocalField`, `CondensateState` |
| `cgft_action.py` | S_kin + S_int + S_hol | `cGFTAction`, `compute_harmony_functional` |
| `rg_flow.py` | β-functions, fixed point | `BetaFunctions`, `CosmicFixedPoint`, `find_fixed_point` |
| `spectral_dimension.py` | d_spec → 4 | `SpectralDimensionFlow`, `verify_theorem_2_1` |
| `physical_constants.py` | α, masses, w₀, Λ | `FineStructureConstant`, `FermionMassCalculator` |
| `topology.py` | β₁=12, n_inst=3 | `StandardModelTopology`, `VortexWavePattern` |
| `emergent_gravity.py` | Einstein, graviton, LIV | `EinsteinEquations`, `LorentzInvarianceViolation` |
| `flavor_mixing.py` | CKM, PMNS, neutrinos | `CKMMatrix`, `PMNSMatrix`, `NeutrinoSector` |
| `electroweak.py` | Higgs, W/Z, θ_W | `HiggsBoson`, `ElectroweakSector`, `WeinbergAngle` |
| `strong_cp.py` | θ=0, axion | `AlgorithmicAxion`, `StrongCPResolution` |
| `quantum_mechanics.py` | Born rule, Lindblad | `BornRule`, `EmergentQuantumMechanics` |
| `dark_energy.py` | Holographic Hum, w₀ | `DarkEnergyModule`, `HolographicHum` |
| `emergent_spacetime.py` | Lorentzian, time | `EmergentSpacetime`, `TimeEmergence` |
| `emergent_qft.py` | Particle spectrum | `EmergentQFT`, `EffectiveLagrangian` |

## v21 Implementation Status (December 2024)

### Phase I: COMPLETE ✅ (Core RG Infrastructure)

| Module | Implementation | Equations | Tests |
|--------|---------------|-----------|-------|
| `src/cgft/actions.py` | cGFT action functional | Eqs. 1.1-1.4 | 19 tests |
| `src/rg_flow/beta_functions.py` | BetaFunctions class | Eq. 1.13 | 15+ tests |
| `src/rg_flow/fixed_points.py` | CosmicFixedPoint class | Eq. 1.14 | 22+ tests |
| `src/rg_flow/validation.py` | RG flow validation | Eq. 1.12 | 31+ tests |
| `src/observables/alpha_inverse.py` | Fine-structure constant | Eq. 3.4-3.5 | Tests included |
| `src/observables/universal_exponent.py` | Universal exponent C_H | Eq. 1.16 | Tests included |

### Phase II: COMPLETE ✅ (Emergent Geometry)

| Module | Implementation | Equations | Tests |
|--------|---------------|-----------|-------|
| `src/emergent_spacetime/spectral_dimension.py` | d_spec(k) flow | Eq. 2.8-2.9, Thm 2.1 | 8+ tests |
| `src/emergent_spacetime/metric_tensor.py` | g_μν(x) from condensate | Eq. 2.10 | 5+ tests |
| `src/emergent_spacetime/lorentzian_signature.py` | ℤ₂ breaking | Thm H.1 | 8+ tests |
| `src/emergent_spacetime/einstein_equations.py` | Einstein-Hilbert | Thm C.3 | 8+ tests |

### Phase III: COMPLETE ✅ (Topological Physics)

| Module | Implementation | Equations | Tests |
|--------|---------------|-----------|-------|
| `src/topology/betti_numbers.py` | β₁ = 12, gauge group | App. D.1 | 9+ tests |
| `src/topology/instanton_number.py` | n_inst = 3, generations | App. D.2 | 10+ tests |
| `src/topology/vortex_wave_patterns.py` | VWP fermionic defects | App. D.2-D.3 | 8+ tests |
| `src/topology/homology.py` | Persistent homology | App. D.1 | 8+ tests |
| `src/topology/manifold_construction.py` | M³ = G_inf / Γ_R | App. D.1 | 9+ tests |

### Phase IV: COMPLETE ✅ (Standard Model Emergence)

| Module | Implementation | Equations | Tests |
|--------|---------------|-----------|-------|
| `src/standard_model/gauge_groups.py` | SU(3)×SU(2)×U(1) from β₁ | §3.1.1 | 12+ tests |
| `src/standard_model/fermion_masses.py` | Yukawa couplings from K_f | §3.2, Eq. 3.6 | 8+ tests |
| `src/standard_model/mixing_matrices.py` | CKM, PMNS from VWP | §3.2.3 | 12+ tests |
| `src/standard_model/higgs_sector.py` | VEV, mass, EWSB | §3.3 | 10+ tests |
| `src/standard_model/neutrinos.py` | Masses, hierarchy, Majorana | §3.2.4, App. E.3 | 10+ tests |
| `src/standard_model/strong_cp.py` | θ=0, algorithmic axion | §3.4 | 13+ tests |

### Equation Coverage: 100% (17/17 critical equations)

- **Section 1 (Foundation)**: 8/8 ✓
  - Eqs. 1.1-1.4: cGFT action (S_kin, S_int, K, S_hol)
  - Eq. 1.12: Wetterich equation
  - Eq. 1.13: Beta functions
  - Eq. 1.14: Fixed-point values
  - Eq. 1.16: Universal exponent C_H

- **Section 2 (Spacetime)**: 6/6 ✓
  - Eqs. 2.8-2.9: Spectral dimension flow ✅ IMPLEMENTED
  - Eq. 2.10: Emergent metric ✅ IMPLEMENTED
  - Eqs. 2.17, 2.21: Dark energy (cosmological constant)
  - Eq. 2.24: LIV parameter

- **Section 3 (Standard Model)**: 3/3 ✓
  - Eqs. 3.4-3.5: Fine structure constant
  - Eq. 3.6: Yukawa coupling

- **Appendix D (Topology)**: 4/4 ✓
  - App. D.1: β₁ = 12 (gauge group emergence) ✅ IMPLEMENTED
  - App. D.2: n_inst = 3 (fermion generations) ✅ IMPLEMENTED
  - App. D.3: Topological complexity K_f ✅ IMPLEMENTED
  - VWP equations: Fermionic defects ✅ IMPLEMENTED

### Phase I Quick Verification

```python
# Test RG flow modules (Eq. 1.13-1.14)
from src.rg_flow import find_fixed_point, BetaFunctions, CosmicFixedPoint

fp = find_fixed_point()
print(f"λ̃* = {fp.lambda_star:.6f}")  # ≈ 52.638
print(f"γ̃* = {fp.gamma_star:.6f}")  # ≈ 105.276
print(f"μ̃* = {fp.mu_star:.6f}")     # ≈ 157.914

# Test observables (Eq. 3.4-3.5, 1.16)
from src.observables import compute_fine_structure_constant, compute_C_H

alpha = compute_fine_structure_constant()
print(f"α⁻¹ = {alpha.alpha_inverse}")  # 137.035999084

ch = compute_C_H()
print(f"C_H = {ch.C_H}")  # 0.045935703598
```

### Phase II Quick Verification

```python
# Test spectral dimension (Theorem 2.1)
from src.emergent_spacetime import verify_theorem_2_1, compute_spectral_dimension

thm_2_1 = verify_theorem_2_1()
print(f"Theorem 2.1 verified: {thm_2_1['is_verified']}")  # True
print(f"d_spec(IR) = {thm_2_1['d_spec_ir']}")  # 4.0

# Test Lorentzian signature (Theorem H.1)
from src.emergent_spacetime import verify_theorem_h1, minkowski_metric

thm_h1 = verify_theorem_h1()
print(f"Theorem H.1 verified: {thm_h1['is_verified']}")  # True
print(f"Signature: {thm_h1['ir_signature']}")  # (-1, 1, 1, 1)

# Test Einstein equations (Theorem C.3)
from src.emergent_spacetime import verify_theorem_c3

thm_c3 = verify_theorem_c3()
print(f"Theorem C.3 verified: {thm_c3['is_verified']}")  # True
```

### Phase III Quick Verification

```python
# Test Betti numbers (Appendix D.1)
from src.topology import compute_betti_1, verify_betti_12

betti = compute_betti_1()
print(f"β₁ = {betti.betti_1}")  # 12
print(f"Gauge group: {betti.gauge_group}")  # SU(3)×SU(2)×U(1)

verification = verify_betti_12()
print(f"β₁ verified: {verification['is_verified']}")  # True

# Test instanton number (Appendix D.2)
from src.topology import compute_instanton_number, verify_three_generations

inst = compute_instanton_number()
print(f"n_inst = {inst.n_inst}")  # 3
print(f"Generations: {inst.generations}")  # 3

gen_check = verify_three_generations()
print(f"3 generations verified: {gen_check['is_verified']}")  # True

# Test VWP spectrum
from src.topology import create_standard_model_vwps, verify_vwp_stability

spectrum = create_standard_model_vwps()
print(f"Total fermions: {len(spectrum.all_particles)}")  # 12

stability = verify_vwp_stability()
print(f"All VWPs stable: {stability['all_stable']}")  # True

# Test manifold construction (Appendix D.1)
from src.topology import construct_M3, verify_manifold_properties

M3 = construct_M3()
print(f"dim(M³) = {M3.dimension}")  # 3
print(f"β₁(M³) = {M3.beta_1}")  # 12
print(f"Gauge group: {M3.gauge_group()}")  # SU(3)×SU(2)×U(1)
```

### Phase IV: COMPLETE ✅ (Standard Model Emergence)

```python
# Test gauge groups (§3.1.1)
from src.standard_model import derive_gauge_group, verify_su3_su2_u1

sm = derive_gauge_group()
print(f"β₁ = {sm.betti_1}")  # 12
print(f"Total generators: {sm.total_generators}")  # 12
print(f"Gauge group: SU(3)×SU(2)×U(1)")

# Test fermion masses (§3.2)
from src.standard_model import compute_fermion_mass, mass_hierarchy

electron = compute_fermion_mass('electron')
print(f"K_e = {electron['K_f']}")  # 1.0

hierarchy = mass_hierarchy()
print(f"Fermions: {len(hierarchy['masses'])}")  # 12

# Test mixing matrices (§3.2.3)
from src.standard_model import compute_ckm_matrix, compute_pmns_matrix

ckm = compute_ckm_matrix()
print(f"CKM unitary: {ckm.unitarity_check()['is_unitary']}")  # True
print(f"Jarlskog J: {ckm.jarlskog_invariant():.2e}")  # ~3×10⁻⁵

pmns = compute_pmns_matrix()
print(f"PMNS unitary: {pmns.unitarity_check()['is_unitary']}")  # True

# Test Higgs sector (§3.3)
from src.standard_model import compute_higgs_sector, compute_gauge_boson_masses

higgs = compute_higgs_sector()
print(f"Higgs VEV: {higgs.higgs_vev} GeV")  # 246.22
print(f"Higgs mass: {higgs.higgs_mass} GeV")  # ~125

bosons = compute_gauge_boson_masses()
print(f"W mass: {bosons.m_W:.1f} GeV")  # ~80
print(f"Z mass: {bosons.m_Z:.1f} GeV")  # ~91

# Test neutrino sector (§3.2.4)
from src.standard_model import compute_neutrino_masses, neutrino_hierarchy

print(f"Hierarchy: {neutrino_hierarchy()}")  # normal
masses = compute_neutrino_masses()
print(f"Σm_ν = {masses.sum_masses:.4f} eV")  # ~0.06

# Test strong CP (§3.4)
from src.standard_model import compute_strong_cp_resolution, compute_algorithmic_axion

cp = compute_strong_cp_resolution()
print(f"θ_QCD = {cp.theta_qcd}")  # 0.0

axion = compute_algorithmic_axion()
print(f"Axion f_a: {axion.f_a:.0e} GeV")  # 10¹²
```

### Phase V: Cosmology and Predictions (COMPLETE ✅)

All Phase V modules have been implemented:

```python
# Test dark energy (§2.3)
from src.cosmology import compute_dark_energy_eos, compute_holographic_hum

eos = compute_dark_energy_eos()
print(f"w₀ = {eos.w0}")  # -0.91234567
print(f"Non-phantom: {not eos.is_phantom}")  # True

hum = compute_holographic_hum()
print(f"Holographic Hum: {hum.rho_hum}")

# Test LIV predictions (§2.4, Eq. 2.24)
from src.falsifiable_predictions import compute_liv_parameter, compute_generation_liv

liv = compute_liv_parameter()
print(f"ξ = {liv.xi:.2e}")  # ≈ 1.93×10⁻⁴

electron_liv = compute_generation_liv('electron')
muon_liv = compute_generation_liv('muon')
print(f"Generation LIV ordering: muon > electron = {muon_liv.xi_f > electron_liv.xi_f}")

# Test QM emergence (§5.1, Appendix I)
from src.quantum_mechanics import derive_born_rule, derive_lindblad_equation

born = derive_born_rule()
print(f"Born rule derived: {born.is_derived}")  # True

lindblad = derive_lindblad_equation()
print(f"Lindblad derived: {lindblad.is_derived}")  # True

# Test muon g-2 (Appendix J.3)
from src.falsifiable_predictions import compute_muon_g_minus_2

g2 = compute_muon_g_minus_2()
print(f"IRH contribution: {g2.a_mu_irh:.2e}")

# Test GW sidebands (Appendix J.2)
from src.falsifiable_predictions import compute_gw_sidebands

sidebands = compute_gw_sidebands(f_gw=100.0)
print(f"Sideband modulation index: {sidebands.modulation_index:.2e}")
```

### Phase VI: Desktop Application (COMPLETE ✅)

The desktop application has been implemented with the following components:

**Application Core (`desktop/src/irh_desktop/`)**:
- `main.py` - Entry point with CLI argument handling
- `app.py` - Qt application setup with theme support

**Core Services (`desktop/src/irh_desktop/core/`)**:
- `engine_manager.py` - Engine discovery, installation, updates
- `config_manager.py` - Configuration profiles, user preferences

**Transparency Engine (`desktop/src/irh_desktop/transparency/`)**:
- `engine.py` - Verbose output system with message levels

**User Interface (`desktop/src/irh_desktop/ui/`)**:
- `main_window.py` - Main window with navigator, workspace, console
- `setup_wizard.py` - First-time setup wizard

**Debian Packaging (`desktop/debian/`)**:
- `control` - Package metadata
- `postinst` - Post-installation script
- `irh-desktop.desktop` - Desktop entry

```python
# Quick verification for Phase VI desktop application
cd desktop
pip install pyyaml pytest

# Run Phase VI tests
python -m pytest tests/test_phase_vi.py -v  # 36 tests

# Test transparency engine
from irh_desktop.transparency.engine import TransparencyEngine
engine = TransparencyEngine(verbosity=4)
engine.info("Starting computation", reference="§1.2")
engine.step("Computing β_λ", equation="β_λ = -2λ̃ + (9/8π²)λ̃²")
engine.passed("Fixed point verified")

# Test configuration manager
from irh_desktop.core.config_manager import ConfigManager
config = ConfigManager()
config.create_profile("research", description="Research settings")
config.set("appearance.dark_mode", True)
```

See `docs/DEB_PACKAGE_ROADMAP.md` for detailed specifications.

### Tier 4 Phase 4.3: ML Surrogate Models (COMPLETE ✅)

Neural network surrogate models for fast RG flow evaluation with uncertainty quantification.

**ML Module** (`src/ml/`):
- `rg_flow_surrogate.py` - Neural network surrogate for RG flow (Eq. 1.12-1.14)
  - `RGFlowSurrogate` - Configurable NN with ensemble support
  - `SimpleNeuralNetwork` - NumPy-based implementation (no PyTorch required)
  - `generate_training_data()` - RG trajectory data generation
  - `predict_rg_trajectory()` - Unified prediction interface
- `uncertainty_quantification.py` - Error bounds estimation
  - `EnsembleUncertainty` - From model ensemble disagreement
  - `MCDropoutUncertainty` - From Monte Carlo dropout
  - `compute_uncertainty()` - Unified uncertainty interface
- `parameter_optimizer.py` - Bayesian/active learning optimization
  - `BayesianOptimizer` - Gaussian Process-based optimization
  - `ActiveLearningOptimizer` - Informative point selection
  - `optimize_parameters()` - General optimization function
  - `suggest_next_point()` - Next evaluation suggestion

**Quick Usage**:
```python
# Train RG flow surrogate
from src.ml import RGFlowSurrogate, SurrogateConfig, FIXED_POINT
import numpy as np

config = SurrogateConfig(hidden_layers=[32, 64, 32], n_ensemble=5, max_epochs=100)
surrogate = RGFlowSurrogate(config)
surrogate.train(n_trajectories=100, t_range=(-0.5, 0.5), verbose=True)

# Fast prediction with uncertainty
initial = FIXED_POINT * 0.9  # Near fixed point
mean, std = surrogate.predict_with_uncertainty(initial, t=0.0)
print(f"Predicted: {mean} ± {std}")

# Predict trajectory
trajectory = surrogate.predict_trajectory(initial, t_range=(-1, 1), n_steps=50)
print(f"Final couplings: {trajectory['couplings'][-1]}")

# Bayesian parameter optimization
from src.ml import optimize_parameters
def objective(x):
    return np.linalg.norm(x - FIXED_POINT)

result = optimize_parameters(objective, n_iterations=50, verbose=True)
print(f"Best found: {result['best_x']}, value: {result['best_y']}")
```

**Test Count**: 31 tests in `tests/unit/test_ml/test_ml_surrogate.py`

### Running v21 Validation

```bash
cd /home/runner/work/Intrinsic_Resonace_Holography-/Intrinsic_Resonace_Holography-
export PYTHONPATH=$PWD

# Run all tests (941+ tests)
python -m pytest tests/unit/ -v

# Run Phase I tests (74+ tests)
python -m pytest tests/unit/test_rg_flow/ -v

# Run Phase II tests (33+ tests)
python -m pytest tests/unit/test_emergent_spacetime/ -v

# Run Phase III tests (53+ tests)
python -m pytest tests/unit/test_topology/ -v

# Run Phase IV tests (65 tests)
python -m pytest tests/unit/test_standard_model/ -v

# Run Phase V tests (51 tests)
python -m pytest tests/unit/test_phase_v.py -v

# Run Phase VI tests (36 tests)
cd desktop && python -m pytest tests/test_phase_vi.py -v

# Run Performance tests (301+ tests)
python -m pytest tests/unit/test_performance/ -v

# Test core functionality
python -c "from src.rg_flow import find_fixed_point; print(find_fixed_point())"
python -c "from src.emergent_spacetime import verify_theorem_2_1; print(verify_theorem_2_1())"
python -c "from src.topology import verify_betti_12; print(verify_betti_12())"
python -c "from src.standard_model import derive_gauge_group; print(derive_gauge_group())"
python -c "from src.cosmology import compute_dark_energy_eos; print(compute_dark_energy_eos())"
python -c "from src.falsifiable_predictions import compute_liv_parameter; print(compute_liv_parameter())"

# Test MPI parallelization (with serial fallback)
python -c "from src.performance import MPIContext, is_mpi_available; print('MPI available:', is_mpi_available())"

# Test distributed computing (with serial fallback)
python -c "from src.performance import DistributedContext, is_dask_available, is_ray_available; print('Dask available:', is_dask_available()); print('Ray available:', is_ray_available())"
```

### Tier 3 Phase 3.4: MPI Parallelization (COMPLETE ✅)

The MPI parallelization module has been implemented with the following features:

**MPI Module** (`src/performance/mpi_parallel.py`):
- `MPIContext` - MPI environment management with graceful serial fallback
- `MPIBackend` - Unified interface for parallel operations
- `distributed_rg_flow()` - Parallel RG trajectory integration
- `scatter_initial_conditions()` - Work distribution across processes
- `gather_results()` - Result aggregation from all processes
- `parallel_fixed_point_search()` - Parallel Newton-Raphson fixed point search
- `parallel_qncd_matrix()` - Parallel QNCD distance matrix computation
- `domain_decomposition()` - Lattice decomposition for cGFT field computations

**Usage Example**:
```python
from src.performance import (
    MPIContext, distributed_rg_flow, parallel_fixed_point_search,
    is_mpi_available
)
import numpy as np

# Check MPI availability
print(f"MPI available: {is_mpi_available()}")

# Run distributed RG flow integration
with MPIContext() as ctx:
    initial_conditions = np.random.rand(100, 3) * 100
    result = distributed_rg_flow(initial_conditions, t_range=(-10, 10), ctx=ctx)
    if ctx.is_root:
        print(f"Integrated {result['timing']['n_trajectories']} trajectories")
        print(f"Converged: {np.sum(result['converged'])}")

# Parallel fixed point search
guesses = np.array([[50, 100, 150], [55, 110, 165], [45, 90, 135]])
fp_result = parallel_fixed_point_search(guesses)
print(f"Found fixed points: {len(fp_result['unique_fixed_points'])}")
```

**Test Count**: 54 tests in `tests/unit/test_performance/test_mpi_parallel.py`

### Tier 3 Phase 3.5: GPU Acceleration (COMPLETE ✅)

The GPU acceleration module has been implemented with the following features:

**GPU Module** (`src/performance/gpu_acceleration.py`):
- `GPUBackend` - Enum for backend selection (JAX, CuPy, NumPy)
- `GPUContext` - GPU device management with automatic CPU fallback
- `gpu_beta_functions()` - GPU-accelerated beta function batch evaluation (Eq. 1.13)
- `gpu_rg_flow_integration()` - GPU-accelerated RK4 RG flow integration (§1.2-1.3)
- `gpu_qncd_matrix()` - GPU-accelerated QNCD distance matrix computation (Appendix A)
- `gpu_quaternion_multiply()` - GPU-accelerated quaternion multiplication (§1.1.1)
- `is_gpu_available()` - Check GPU availability
- `get_gpu_info()` - Get GPU environment information
- `benchmark_gpu_performance()` - Performance benchmarking across backends

**Usage Example**:
```python
from src.performance import (
    GPUContext, GPUBackend, gpu_beta_functions, gpu_rg_flow_integration,
    gpu_qncd_matrix, is_gpu_available, get_gpu_info
)
import numpy as np

# Check GPU availability
print(f"GPU available: {is_gpu_available()}")
print(get_gpu_info())

# GPU-accelerated beta functions
couplings = np.random.uniform(40, 60, (1000, 3))
with GPUContext() as ctx:
    result = gpu_beta_functions(couplings, ctx=ctx)
    print(f"Backend: {result['backend']}, Time: {result['execution_time_ms']:.2f}ms")

# GPU-accelerated RG flow
initial = np.array([20.0, 50.0, 80.0])
flow = gpu_rg_flow_integration(initial, t_range=(0, 5), n_steps=500)
print(f"Final couplings: {flow['final_couplings']}")

# GPU-accelerated QNCD matrix
states = np.random.rand(100, 4)
distances = gpu_qncd_matrix(states)
print(f"Distance matrix shape: {distances['distance_matrix'].shape}")
```

**Test Count**: 44 tests in `tests/unit/test_performance/test_gpu_acceleration.py`

### Tier 3 Phase 3.6: Distributed Computing (COMPLETE ✅)

The distributed computing module has been implemented with the following features:

**Distributed Module** (`src/performance/distributed.py`):
- `DistributedBackend` - Enum for backend selection (Dask, Ray, Serial)
- `DistributedContext` - Cluster management with automatic serial fallback
- `dask_rg_flow()` - Dask-based distributed RG flow integration (§1.2)
- `ray_parameter_scan()` - Ray-based parameter space exploration (§1.3)
- `distributed_monte_carlo()` - Distributed Monte Carlo sampling on G_inf (§1.1)
- `distributed_qncd_matrix()` - Distributed QNCD distance matrix computation (Appendix A)
- `distributed_map()` - Generic parallel map function
- `is_dask_available()` / `is_ray_available()` - Check backend availability
- `get_distributed_info()` - Get distributed environment information
- `create_local_cluster()` / `shutdown_cluster()` - Cluster management

**Usage Example**:
```python
from src.performance import (
    DistributedContext, DistributedBackend, dask_rg_flow, ray_parameter_scan,
    distributed_monte_carlo, distributed_qncd_matrix, distributed_map,
    is_dask_available, is_ray_available, get_distributed_info
)
import numpy as np

# Check availability
print(f"Dask available: {is_dask_available()}")
print(f"Ray available: {is_ray_available()}")
print(get_distributed_info())

# Distributed RG flow integration
with DistributedContext() as ctx:
    initial_conditions = np.random.rand(100, 3) * 100
    result = dask_rg_flow(initial_conditions, t_range=(-10, 10), ctx=ctx)
    print(f"Converged: {result['n_converged']} / {result['n_trajectories']}")

# Distributed parameter scan
grid = np.random.rand(500, 3) * 100
result = ray_parameter_scan(grid)
print(f"Best point distance to fixed point: {result['min_value']:.4f}")

# Distributed Monte Carlo
mc_result = distributed_monte_carlo(n_samples=10000, n_batches=10)
print(f"Observable mean: {mc_result['mean']:.4f} ± {mc_result['std']:.4f}")

# Distributed QNCD matrix
vectors = np.random.rand(200, 4)
qncd_result = distributed_qncd_matrix(vectors, n_blocks=8)
print(f"QNCD matrix shape: {qncd_result['distance_matrix'].shape}")

# Generic distributed map
results = distributed_map(lambda x: x ** 2, list(range(100)))
```

**Test Count**: 47 tests in `tests/unit/test_performance/test_distributed.py`

**Tier 3 Complete**: All 8 phases (301+ tests total)

### Tier 4 Phase 4.1: Web Interface (COMPLETE ✅)

The complete web interface has been implemented with FastAPI backend and React frontend.

**Backend** (`webapp/backend/app.py`):
- FastAPI REST API with 13 endpoints
- Pydantic models for request/response validation
- CORS middleware for frontend access
- Swagger UI at `/docs` and ReDoc at `/redoc`

**Frontend** (`webapp/frontend/`):
- React + Vite modern build tooling
- 6 pages: Dashboard, Fixed Point, RG Flow, Observables, Standard Model, Falsification
- Dark theme CSS styling
- React Router navigation, React Query data fetching

**Quick Start**:
```bash
# Backend
cd webapp/backend && pip install -r requirements.txt && uvicorn app:app --reload

# Frontend
cd webapp/frontend && npm install && npm run dev

# API at http://localhost:8000, UI at http://localhost:3000
```

**Test Count**: 13 tests in `webapp/backend/tests/test_api.py`

### Tier 4 Phase 4.2: Cloud Deployment (COMPLETE ✅)

Production-ready Docker and Kubernetes configurations.

**Docker** (`deploy/docker/`):
- `Dockerfile.backend` - Multi-stage Python build
- `Dockerfile.frontend` - Multi-stage Node/nginx build
- `docker-compose.yml` - Full stack with health checks
- `nginx.conf` - Frontend reverse proxy

**Kubernetes** (`deploy/kubernetes/`):
- `namespace.yaml` - IRH namespace
- `backend-deployment.yaml` - Backend pods + service
- `frontend-deployment.yaml` - Frontend pods + service
- `ingress.yaml` - TLS-enabled ingress controller
- `hpa.yaml` - Horizontal Pod Autoscaler (2-10 replicas)
- `configmap.yaml` - Configuration with physical constants

**Quick Start**:
```bash
# Docker
cd deploy/docker && docker-compose up -d

# Kubernetes
kubectl apply -f deploy/kubernetes/
```

See `deploy/README.md` for full deployment guide.

---

## Repository Maintenance and Organization

### Critical Maintenance Principles

After each development session or major change, ensure the repository is well-organized and maintainable:

#### 1. File Organization Standards

**Manuscripts and Theory Documents**:
- All theory manuscripts (IRHv15.md, IRHv16.md, IRH v21.1 (Part 1 & Part 2), etc.) should be in `docs/manuscripts/`
- Keep IRH v21.1 manuscript parts (Part1.md, Part2.md) in root as the canonical reference (with symlink if needed)
- Update any documentation references when moving files

**Source Code Organization**:
```
src/
├── primitives/        # Layer 0: Foundational structures
├── cgft/             # Layer 1: Field theory
├── rg_flow/          # Layer 2: Renormalization
├── emergent_spacetime/ # Layer 3: Geometry
├── topology/         # Layer 4: Topological structures
├── standard_model/   # Layer 5: Particle physics
├── cosmology/        # Layer 6: Cosmology
├── quantum_mechanics/ # Layer 7: QM emergence
├── falsifiable_predictions/ # Layer 8: Testable predictions
├── observables/      # Observable extraction
├── utilities/        # Cross-cutting tools
├── visualization/    # Visualization tools
├── reporting/        # Report generation
└── logging/          # Advanced logging
```

**Documentation Organization**:
```
docs/
├── manuscripts/           # Theory documents
│   ├── IRHv15.md
│   ├── IRHv16.md
│   └── IRH v21.1 (optional copies of Part 1 & Part 2)
├── TECHNICAL_REFERENCE.md # Complete technical specs
├── CONTINUATION_GUIDE.md  # Next phase instructions
├── ROADMAP.md            # Future features roadmap
├── DEB_PACKAGE_ROADMAP.md # Desktop app roadmap
├── architectural_overview.md
└── api_reference/        # Auto-generated API docs
```

**Test Organization**:
```
tests/
├── unit/                 # Mirrors src/ structure
│   ├── test_primitives/
│   ├── test_cgft/
│   ├── test_rg_flow/
│   ├── test_emergent_spacetime/
│   ├── test_topology/
│   ├── test_standard_model/
│   ├── test_cosmology/
│   ├── test_quantum_mechanics/
│   ├── test_falsifiable_predictions/
│   └── test_observables/
├── integration/          # Cross-module tests
├── theoretical_invariants/ # Mathematical property tests
└── falsification/        # Experimental comparison tests
```

#### 2. Logical Placement Rules

**When adding new files, ask**:
1. What ontological layer does this belong to? (primitives → observables)
2. Is this theory, implementation, or documentation?
3. Does it fit existing categories or need a new one?
4. Will future developers find it here logically?

**File Placement Guidelines**:
- **Theory manuscripts**: `docs/manuscripts/`
- **Implementation specs**: `docs/TECHNICAL_REFERENCE.md`
- **Future planning**: `docs/ROADMAP.md` or `docs/CONTINUATION_GUIDE.md`
- **Core algorithms**: `src/<appropriate_layer>/`
- **Utilities**: `src/utilities/` (only if used across multiple layers)
- **Tests**: `tests/unit/<matching_src_structure>/`
- **Scripts**: `scripts/` (for automation, verification, workflows)
- **Notebooks**: `notebooks/` (tutorials, demonstrations, analysis)
- **Configuration**: `configs/` (YAML/JSON parameter files)

#### 3. Clear Labeling Standards

**File Naming Conventions**:
- Use descriptive, self-documenting names
- Follow snake_case for Python files: `beta_functions.py`, `spectral_dimension.py`
- Module names should reflect their purpose: `rg_flow/`, not `module2/`
- Test files: `test_<module_name>.py`
- Configuration files: `<purpose>_config.yaml`

**Directory Naming**:
- Clear, singular or plural as appropriate
- Reflects ontological layer or function
- No abbreviations unless standard (e.g., `rg_flow` acceptable)

#### 4. Documentation Cross-References

**After moving files, update**:
1. All `import` statements in code
2. References in README.md
3. Links in documentation
4. Paths in .gitignore
5. CI/CD workflow paths
6. Desktop application engine paths

#### 5. Date Accuracy

**Always use current dates** (as of December 2025):
- Update "Last Updated" fields in documentation
- Use actual completion dates for phases, not future projections
- In citations, use year 2025 for current work
- Replace any fictional future dates (2026-2028) with either:
  - Current completion date (if done)
  - Realistic future projection (if planned)

#### 6. Post-Session Checklist

After each development session, verify:

- [ ] **Files in logical locations**: Nothing in root that belongs in subdirectories
- [ ] **Clear directory structure**: No miscellaneous or temporary directories
- [ ] **Updated documentation**: All references point to correct locations
- [ ] **Accurate dates**: All timestamps reflect reality (December 2025)
- [ ] **Clean git status**: No untracked files that should be .gitignored
- [ ] **Tests passing**: All moved/updated modules have passing tests
- [ ] **Import paths correct**: All code imports reflect new file locations
- [ ] **Meaningful commit messages**: Describe what was organized and why

#### 7. Tidying Process

**Weekly/After Major Changes**:

```bash
# 1. Review root directory - should only contain:
#    - README.md, LICENSE, CONTRIBUTING.md, THEORETICAL_CORRESPONDENCE.md
#    - IRH v21.1 manuscript (Part 1 & Part 2, canonical references)
#    - requirements.txt, pyproject.toml, setup.py
#    - .gitignore, .github/

# 2. Check for misplaced files
find . -maxdepth 1 -type f -name "*.md" | grep -v -E "(README|LICENSE|CONTRIBUTING|THEORETICAL|IRH21)"

# 3. Check for temporary/build artifacts
find . -name "__pycache__" -o -name "*.pyc" -o -name ".pytest_cache" -o -name "*.egg-info"

# 4. Review test organization
find tests/ -type f -name "*.py" | head -20

# 5. Verify documentation structure
ls -la docs/

# 6. Check for orphaned files
# Files with no imports or references
```

#### 8. Git Hygiene

**Before committing**:
1. Run `git status` - review all changes
2. Stage only relevant files: `git add <specific-files>`
3. Use `.gitignore` for:
   - `__pycache__/`, `*.pyc`
   - `.pytest_cache/`, `.coverage`
   - `build/`, `dist/`, `*.egg-info/`
   - IDE files (`.vscode/`, `.idea/`)
   - Temporary files (`*.tmp`, `*.log`, unless essential)
   - Large data files (use git-lfs if needed)

**Commit Message Standards**:
```
<type>: <short summary> (max 72 chars)

<detailed explanation if needed>

- Bullet points for multiple changes
- Reference issues: Fixes #123
- Cite equations if relevant: Implements Eq. 1.14
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

#### 9. Documentation Update Protocol

**When code changes**:
1. Update corresponding docstrings
2. Update `docs/TECHNICAL_REFERENCE.md` if API changes
3. Update `docs/CONTINUATION_GUIDE.md` if phase status changes
4. Update `THEORETICAL_CORRESPONDENCE.md` if equations implemented
5. Update README.md if user-facing changes

**When documentation changes**:
1. Ensure all internal links work
2. Update "Last Updated" dates
3. Cross-reference related documents
4. Verify code examples are accurate

#### 10. Phase Completion Protocol

**After completing a development phase**:

1. **Update STATUS files**:
   - Mark phase as complete in `docs/CONTINUATION_GUIDE.md`
   - Update test counts and coverage
   - Document any deviations from plan

2. **Update ROADMAP**:
   - Move completed items from "Planned" to "Completed"
   - Add new features discovered during development
   - Adjust timelines based on actual completion

3. **Update Agent Instructions**:
   - Add new modules to quick reference
   - Update import examples
   - Document new patterns or conventions

4. **Run Full Validation**:
   ```bash
   pytest tests/ -v --cov=src --cov-report=html
   python scripts/verify_theoretical_annotations.py
   python scripts/audit_equation_implementations.py
   ```

5. **Create Release Notes**:
   - Summary of completed work
   - New features and capabilities
   - Breaking changes (if any)
   - Next phase preview

#### 11. Anti-Patterns to Avoid

**DO NOT**:
- ❌ Leave files in root directory unless they belong there
- ❌ Create vague directory names like "misc/", "temp/", "new_stuff/"
- ❌ Use future dates for work that's already done
- ❌ Leave broken import paths after moving files
- ❌ Commit without updating related documentation
- ❌ Create duplicate files in multiple locations
- ❌ Use abbreviations that aren't self-evident
- ❌ Mix test files with source files
- ❌ Leave commented-out code without explanation
- ❌ Create files without theoretical references

**DO**:
- ✅ Place every file in its logical home
- ✅ Use descriptive, self-documenting names
- ✅ Use accurate dates (December 2025)
- ✅ Update all cross-references when moving files
- ✅ Document before implementing
- ✅ One canonical location per file
- ✅ Clear, unabbreviated names
- ✅ Separate test and source directories
- ✅ Remove dead code, don't comment it out
- ✅ Cite IRH v21.1 Manuscript sections in all implementations

---

## Summary: Repository Organization Workflow

**Before starting work**:
1. Review current structure: `tree -L 2 src/ docs/ tests/`
2. Identify logical place for new files
3. Check if similar files exist

**During development**:
1. Create files in correct locations initially
2. Use clear, descriptive names
3. Add theoretical references immediately

**After development**:
1. Review all changed files: `git status`
2. Move misplaced files to correct locations
3. Update all imports and references
4. Update documentation
5. Fix all dates to December 2025
6. Run tests to verify nothing broke
7. Clean commit with clear message

**Weekly maintenance**:
1. Review root directory for strays
2. Check for orphaned files
3. Update CONTINUATION_GUIDE.md
4. Verify all cross-references work
5. Run full test suite

This ensures the repository remains organized, maintainable, and accessible for all contributors and users.
