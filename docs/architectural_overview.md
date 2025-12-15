# Architectural Overview: IRH v21.0 Computational Framework

## Conceptual Foundation

This document explains the architectural principles underlying the IRH v21.0 Computational Framework. The structure directly instantiates the **Epistemic Stratification Principle** (IRH21.md §4.1), which asserts that fundamental theory must decompose into:

1. **Primitive Ontology**: Axiomatic commitments about reality's basic constituents
2. **Structural Dynamics**: Mathematical laws governing primitives
3. **Phenomenological Emergence**: Observable consequences

## Ontological Stratification in Directory Structure

### Layer 0: `src/primitives/` — Ontological Bedrock

**Theoretical Foundation**: IRH21.md §1.0.1 (Revised Foundational Axiom)

This layer implements the axiomatically primitive quantum-informational structures from which all emergent physics derives. It contains **no phenomenology**—only the mathematical scaffolding upon which the cGFT dynamics operate.

**Contents**:
- `quantum_information.py`: Hilbert space representations, quantum states, K_Q complexity
- `group_manifolds.py`: SU(2) and U(1) Lie group operations, Haar measure
- `quaternions.py`: ℍ arithmetic, conjugation, quaternionic products
- `algorithmic_measures.py`: QNCD metric, universal quantum compressor infrastructure

**Design Principles**:
1. Theory-agnostic: Could be reused for other quantum information theories
2. No emergent concepts: No reference to spacetime, particles, or forces
3. Maximum mathematical rigor: Formal group theory, functional analysis
4. Provably correct: Every operation has mathematical theorem backing

### Layer 1: `src/cgft/` — Fundamental Dynamics

**Theoretical Foundation**: IRH21.md §1.1

This layer defines the action functional S[φ,φ̄] (Eqs. 1.1-1.4) that governs the evolution of quantum information. This is where IRH's unique structural commitments first appear: the specific choice of G_inf = SU(2) × U(1), the quaternionic field, the QNCD-weighted interaction kernel.

**Contents**:
- `fields.py`: φ(g₁,g₂,g₃,g₄) ∈ ℍ field representations
- `actions.py`: S_kin, S_int, S_hol (Eqs. 1.1-1.4)
- `operators.py`: Laplace-Beltrami operators, functional derivatives
- `interactions.py`: QNCD-weighted kernels, phase coherence
- `symmetries.py`: Gauge transformations, Weyl ordering

### Layer 2: `src/rg_flow/` — Meta-Algorithm of Reality

**Theoretical Foundation**: IRH21.md §1.2-1.3

The Wetterich equation (Eq. 1.12) and its consequences (β-functions, fixed points) represent the process ontology wherein laws themselves emerge through asymptotic safety. This layer contains no direct reference to spacetime or particles—only abstract coupling dynamics.

**Contents**:
- `wetterich.py`: Exact RG equation integrator
- `beta_functions.py`: β_λ, β_γ, β_μ (Eq. 1.13)
- `fixed_points.py`: Cosmic Fixed Point solver (Eq. 1.14)
- `running_couplings.py`: Scale-dependent parameter evolution
- `stability_analysis.py`: Eigenvalue spectrum, IR attractiveness

### Layer 3: `src/emergent_spacetime/` — Geometric Emergence

**Theoretical Foundation**: IRH21.md §2.1-2.5

From the RG flow emerges 4-dimensional Lorentzian spacetime with its metric structure. The spectral dimension flows from fractal UV behavior to exactly 4 in the IR.

**Contents**:
- `spectral_dimension.py`: d_spec(k) flow (Eq. 2.8-2.9)
- `metric_tensor.py`: g_μν(x) from condensate (Eq. 2.10)
- `lorentzian_signature.py`: Spontaneous ℤ₂ breaking, timelike direction
- `graviton.py`: Two-point function, propagator (Appendix C)
- `einstein_equations.py`: Variational derivation from Harmony Functional

### Layer 4: `src/topology/` — Topological Structures

**Theoretical Foundation**: IRH21.md §3.1, Appendix D

The emergent 3-manifold possesses topological invariants that encode the gauge structure of particle physics: β₁ = 12 generators yield SU(3)×SU(2)×U(1), and n_inst = 3 determines three fermion generations.

**Contents**:
- `betti_numbers.py`: β₁ = 12 computation (Appendix D.1)
- `instanton_number.py`: n_inst = 3 calculation (Appendix D.2)
- `vortex_wave_patterns.py`: Fermionic defects, topological complexity
- `homology.py`: Persistent homology, Morse theory
- `manifold_construction.py`: Resonance quotient M³ from condensate

### Layer 5: `src/standard_model/` — Particle Physics Emergence

**Theoretical Foundation**: IRH21.md §3.1-3.4

The Standard Model's gauge groups, particle masses, and mixing matrices emerge from the topological structure of the IR fixed point.

**Contents**:
- `gauge_groups.py`: SU(3)×SU(2)×U(1) from β₁=12
- `fermion_masses.py`: Yukawa couplings, 𝒦_f values (Table 3.1)
- `gauge_bosons.py`: W, Z, γ, g masses and couplings
- `higgs_sector.py`: VEV, λ_H, electroweak symmetry breaking
- `neutrinos.py`: Masses, mixing, Majorana nature (Appendix E.3)
- `strong_cp.py`: Algorithmic axion, θ-angle resolution

### Layer 6: `src/cosmology/` — Cosmological Predictions

**Theoretical Foundation**: IRH21.md §2.3

The cosmological constant emerges from the "Holographic Hum"—residual vacuum energy from exact cancellation mechanisms.

**Contents**:
- `holographic_hum.py`: ρ_hum, Λ* calculation (Eq. 2.17-2.19)
- `dark_energy.py`: w(z) equation of state (Eq. 2.21-2.23)
- `running_constants.py`: c(k), ℏ(k), G(k) (Appendix C.6-C.8)
- `primordial_universe.py`: Early universe, inflation signatures

### Layer 7: `src/quantum_mechanics/` — QM Phenomenology Emergence

**Theoretical Foundation**: IRH21.md §5, Appendix I

Quantum mechanical phenomena—Born rule, measurement, decoherence—emerge from the wave interference structure of the cGFT condensate.

**Contents**:
- `hilbert_space.py`: Emergent ℋ from wave interference (Appendix I.1)
- `born_rule.py`: Probability derivation (Appendix I.2)
- `decoherence.py`: Lindblad equation, pointer basis
- `measurement.py`: Algorithmic Selection, observer back-reaction
- `entanglement.py`: Quantum correlations from QNCD

### Layer 8: `src/falsifiable_predictions/` — Experimental Interface

**Theoretical Foundation**: IRH21.md §8, Appendix J

These modules extract testable, falsifiable predictions that connect mathematical formalism to experimental reality. This is the "tip of the iceberg" where IRH confronts Nature's tribunal.

**Contents**:
- `lorentz_violation.py`: LIV parameter ξ (Eq. 2.24-2.26)
- `generation_specific_liv.py`: 𝒦_f-dependent thresholds (Appendix J.1)
- `gravitational_sidebands.py`: Recursive VWP signatures (Appendix J.2)
- `muon_g_minus_2.py`: Anomalous magnetic moment (Appendix J.3)
- `higgs_trilinear.py`: λ_HHH prediction (Appendix J.4)
- `observer_backreaction.py`: Quantifiable measurement cost (Eq. 5.2)

## Dependency Architecture

### Critical Principle

**Dependencies must form a directed acyclic graph (DAG)** where:
- `primitives/` depends on nothing
- `cgft/` depends only on `primitives/`
- `rg_flow/` depends on `{primitives/, cgft/}`
- etc.

**Any violation indicates conceptual confusion about emergent hierarchy.**

### Dependency Matrix

```
                 Depends On
              prim cgft  rg   space topo  sm   cosm  qm   fals obs  util
primitives     -    -     -     -     -    -     -    -     -    -    -
cgft           ✓    -     -     -     -    -     -    -     -    -    -
rg_flow        ✓    ✓     -     -     -    -     -    -     -    -    -
emergent_sp    ✓    ✓     ✓     -     -    -     -    -     -    -    -
topology       ✓    ✓     ✓     ✓     -    -     -    -     -    -    -
standard_m     ✓    ✓     ✓     ✓     ✓    -     -    -     -    -    -
cosmology      ✓    ✓     ✓     ✓     ✓    ✓     -    -     -    -    -
quantum_mech   ✓    ✓     ✓     ✓     -    -     -    -     -    -    -
falsifiable    ✓    ✓     ✓     ✓     ✓    ✓     ✓    ✓     -    -    -
observables    ✓    ✓     ✓     ✓     ✓    ✓     ✓    ✓     ✓    -    -
utilities      -    -     -     -     -    -     -    -     -    -    -
```

## Cross-Cutting Concerns

### `src/utilities/` — Computational Tools

Shared numerical infrastructure used across all layers:
- `integration.py`: Numerical quadrature on group manifolds
- `optimization.py`: Fixed-point solvers, minimizers
- `special_functions.py`: Bessel, hypergeometric, etc.
- `lattice_discretization.py`: Finite-volume approximations
- `parallel_computing.py`: MPI/OpenMP infrastructure

### `src/observables/` — Observable Extraction

The final interface that computes experimentally comparable values:
- `alpha_inverse.py`: Fine-structure constant (Eq. 3.4-3.5)
- `universal_exponent.py`: C_H = 0.045935703598... (Eq. 1.16)
- `physical_constants.py`: Complete constant database
- `experimental_comparison.py`: Theory vs. data σ-analysis

## Testing Philosophy

### `tests/unit/` — Atomic Verification

Tests individual functions in isolation, mirroring src/ structure.

### `tests/integration/` — Module Interaction

Tests that multiple modules work together correctly across the ontological hierarchy.

### `tests/theoretical_invariants/` — Mathematical Properties

Verifies fundamental theoretical constraints:
- Gauge invariance
- Unitarity
- Hermiticity
- Diffeomorphism invariance

### `tests/convergence/` — Numerical Robustness

Ensures numerical methods converge appropriately:
- Lattice refinement studies
- RG step size independence
- Compressor independence (QUCC-Theorem)

### `tests/benchmarks/` — Analytical Validation

Compares against known exact solutions:
- Free field propagators
- Perturbative limits
- Abelian limit recovery (QED-like flows)

### `tests/falsification/` — Experimental Predictions

Tests that predicted values match experimental data within uncertainties:
- 12-digit α⁻¹ accuracy
- Neutrino mass hierarchy
- Cosmological parameters

## Configuration Management

### `configs/`

Parameter files controlling computational behavior:
- `standard_lattice.yaml`: Default discretization (N_SU2=50, N_U1=25)
- `high_precision_lattice.yaml`: Fine mesh for validation
- `rg_flow_settings.yaml`: Integration tolerances
- `falsification_suite.yaml`: Experimental prediction parameters

## Data Management

### `data/theoretical_predictions/`

Certified values from IRH21.md for validation:
- `fixed_point_couplings.json`
- `physical_constants.json`
- `topological_invariants.json`

### `data/experimental_values/`

External reference data:
- `particle_data_group_2024.json`
- `codata_2026_constants.json`
- `planck_2018_cosmology.json`

### `data/baselines/`

Certified computational benchmarks for regression detection.

---

*This architecture ensures that the computational framework remains a faithful instantiation of IRH v21.0's theoretical structure, where the boundary between mathematical theory and executable code dissolves into identity.*
