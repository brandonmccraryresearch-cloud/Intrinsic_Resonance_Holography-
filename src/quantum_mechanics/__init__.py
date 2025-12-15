"""
Quantum Mechanics Emergence Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §5, Appendix I

This module derives quantum mechanical phenomena from the wave interference
structure of the cGFT condensate. The Born rule, measurement, and decoherence
all emerge from the fundamental algorithmic information dynamics.

Key Results:
    - Appendix I.1: Emergent Hilbert space from wave interference
    - Appendix I.2: Born rule derivation from Algorithmic Selection
    - Appendix I.3: Lindblad equation for decoherence
    - §5.2.1: Quantifiable observer back-reaction

Modules:
    hilbert_space: Emergent ℋ from wave interference (Appendix I.1)
    born_rule: Probability derivation (Appendix I.2)
    decoherence: Lindblad equation, pointer basis
    measurement: Algorithmic Selection, observer back-reaction
    entanglement: Quantum correlations from QNCD

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)

Authors: IRH Computational Framework Team
Last Updated: 2026-Q2 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §5, Appendix I"

__all__ = [
    # hilbert_space exports
    'EmergentHilbertSpace',
    'emergent_inner_product',
    
    # born_rule exports
    'born_probability',
    'verify_born_rule',
    
    # decoherence exports
    'lindblad_equation',
    'decoherence_rate',
    'pointer_basis',
    
    # measurement exports
    'algorithmic_selection',
    'measurement_outcome',
    'observer_backreaction',
    
    # entanglement exports
    'entanglement_entropy',
    'qncd_correlations',
]
