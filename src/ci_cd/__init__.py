# CI/CD Infrastructure for IRH v21.0
# Phase VII: Pre-commit hooks, regression detection, and workflow automation

"""
CI/CD Infrastructure Module

This module provides:
- Pre-commit validation hooks for theoretical annotations
- Regression detection against certified baselines
- Test tier execution (T1-T4)
- Coverage reporting with theoretical mapping

Theoretical Reference:
    IRH21.md copilot21promtMAX.md Phase VII: CI/CD
"""

from .ci_infrastructure import (
    PreCommitValidator,
    RegressionDetector,
    TestTierRunner,
    BaselineManager,
    CoverageReporter,
    ValidationResult,
    RegressionReport,
    TestResult,
    run_pre_commit_checks,
    detect_regressions,
    run_test_tier,
    update_baselines,
)

__all__ = [
    "PreCommitValidator",
    "RegressionDetector",
    "TestTierRunner",
    "BaselineManager",
    "CoverageReporter",
    "ValidationResult",
    "RegressionReport",
    "TestResult",
    "run_pre_commit_checks",
    "detect_regressions",
    "run_test_tier",
    "update_baselines",
]
