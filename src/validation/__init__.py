# src/validation/__init__.py
"""
Phase V: Cross-Validation and Convergence Analysis

This module provides systematic validation infrastructure for IRH computations:
- Convergence studies for discretization parameters
- Algorithmic cross-validation with multiple methods
- Error propagation framework

Theoretical Reference:
    IRH21.md Appendix A.5: Convergence to continuum limit
    copilot21promtMAX.md Phase V: Cross-Validation requirements
"""

from .cross_validation import (
    ConvergenceAnalysis,
    AlgorithmicCrossValidation,
    ErrorPropagation,
    ConvergenceResult,
    CrossValidationResult,
    run_full_validation_suite,
)

__all__ = [
    "ConvergenceAnalysis",
    "AlgorithmicCrossValidation",
    "ErrorPropagation",
    "ConvergenceResult",
    "CrossValidationResult",
    "run_full_validation_suite",
]
