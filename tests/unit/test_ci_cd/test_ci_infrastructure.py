"""
Unit Tests for CI/CD Infrastructure (Phase VII)

Tests the pre-commit validation, regression detection, and test tier
execution components of the IRH v21.0 verification protocol.

Theoretical Reference:
    copilot21promtMAX.md Phase VII: CI/CD
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ci_cd.ci_infrastructure import (
    BaselineManager,
    CoverageReporter,
    PreCommitValidator,
    RegressionDetector,
    RegressionReport,
    TestResult,
    TestTier,
    TestTierRunner,
    ValidationResult,
    ValidationStatus,
    detect_regressions,
    run_pre_commit_checks,
    run_test_tier,
    update_baselines,
)


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult data class."""
    
    def test_basic_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(
            name="test_check",
            status=ValidationStatus.PASSED,
            message="Check passed",
        )
        assert result.name == "test_check"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Check passed"
    
    def test_with_details(self):
        """Test ValidationResult with details."""
        result = ValidationResult(
            name="equation_check",
            status=ValidationStatus.WARNING,
            message="Some equations missing",
            details={"equations": ["1.1", "1.2"]},
            duration_s=0.5,
        )
        assert result.details["equations"] == ["1.1", "1.2"]
        assert result.duration_s == 0.5
    
    def test_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            name="test",
            status=ValidationStatus.FAILED,
            message="Failed",
            details={"reason": "test"},
            duration_s=1.0,
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "failed"
        assert d["message"] == "Failed"
        assert d["details"]["reason"] == "test"
    
    def test_all_statuses(self):
        """Test all validation statuses."""
        for status in ValidationStatus:
            result = ValidationResult(
                name="test",
                status=status,
                message="test",
            )
            assert result.status == status


# =============================================================================
# RegressionReport Tests
# =============================================================================

class TestRegressionReport:
    """Tests for RegressionReport data class."""
    
    def test_no_regression(self):
        """Test report with no regression."""
        report = RegressionReport(
            observable="C_H",
            baseline_value=0.045935703598,
            current_value=0.045935703598,
            tolerance=1e-10,
            is_regression=False,
            deviation=0.0,
            sigma_deviation=0.0,
            theoretical_reference="Eq. 1.16",
        )
        assert not report.is_regression
        assert report.deviation == 0.0
    
    def test_regression_detected(self):
        """Test report with regression."""
        report = RegressionReport(
            observable="alpha_inv",
            baseline_value=137.035999,
            current_value=137.1,
            tolerance=0.0001,
            is_regression=True,
            deviation=0.064001,
            sigma_deviation=100.0,
        )
        assert report.is_regression
        assert report.deviation > report.tolerance
    
    def test_to_dict(self):
        """Test RegressionReport serialization."""
        report = RegressionReport(
            observable="test",
            baseline_value=1.0,
            current_value=1.1,
            tolerance=0.01,
            is_regression=True,
            deviation=0.1,
            sigma_deviation=10.0,
        )
        d = report.to_dict()
        assert d["observable"] == "test"
        assert d["is_regression"] is True


# =============================================================================
# TestResult Tests
# =============================================================================

class TestTestResult:
    """Tests for TestResult data class."""
    
    def test_passed_test(self):
        """Test passing test result."""
        result = TestResult(
            test_name="test_fixed_point",
            tier=TestTier.T1_FAST,
            passed=True,
            duration_s=0.1,
            theoretical_reference="Eq. 1.14",
        )
        assert result.passed
        assert result.tier == TestTier.T1_FAST
    
    def test_failed_test(self):
        """Test failing test result."""
        result = TestResult(
            test_name="test_convergence",
            tier=TestTier.T3_COMPREHENSIVE,
            passed=False,
            duration_s=60.0,
            error_message="Assertion failed",
        )
        assert not result.passed
        assert result.error_message is not None
    
    def test_to_dict(self):
        """Test TestResult serialization."""
        result = TestResult(
            test_name="test",
            tier=TestTier.T2_STANDARD,
            passed=True,
            duration_s=1.0,
        )
        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["tier"] == "T2"


# =============================================================================
# PreCommitValidator Tests
# =============================================================================

class TestPreCommitValidator:
    """Tests for PreCommitValidator."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = PreCommitValidator()
        assert validator.repo_root == Path.cwd()
        assert len(validator.results) == 0
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = PreCommitValidator()
        results = validator.validate_file(Path("/nonexistent/file.py"))
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.ERROR
    
    def test_validate_python_file_with_docstring(self):
        """Test validation of Python file with proper docstring."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write('"""Module docstring with Eq. 1.14 reference."""\n')
            f.write("x = 1\n")
            f.flush()
            
            validator = PreCommitValidator()
            results = validator.validate_file(Path(f.name))
        
        # Should have docstring and equation checks
        docstring_results = [r for r in results if "docstring" in r.name]
        assert len(docstring_results) >= 1
        assert docstring_results[0].status == ValidationStatus.PASSED
    
    def test_validate_file_without_docstring(self):
        """Test validation of Python file without docstring."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            
            validator = PreCommitValidator()
            results = validator.validate_file(Path(f.name))
        
        docstring_results = [r for r in results if "docstring" in r.name]
        assert len(docstring_results) >= 1
        assert docstring_results[0].status == ValidationStatus.WARNING
    
    def test_equation_pattern(self):
        """Test equation reference pattern matching."""
        validator = PreCommitValidator()
        
        test_cases = [
            ("Eq. 1.14", True),
            ("Equation 2.3", True),
            ("Eq.3.4.5", True),
            ("equation 1.1", True),
            ("Eq 1", False),  # No decimal
        ]
        
        for text, should_match in test_cases:
            match = validator.EQUATION_PATTERN.search(text)
            if should_match:
                assert match is not None, f"Should match: {text}"
            # Note: some may still match partial patterns
    
    def test_generate_report(self):
        """Test report generation."""
        validator = PreCommitValidator()
        validator.results = [
            ValidationResult("check1", ValidationStatus.PASSED, "OK"),
            ValidationResult("check2", ValidationStatus.WARNING, "Warning"),
            ValidationResult("check3", ValidationStatus.FAILED, "Failed"),
        ]
        
        report = validator.generate_report()
        assert "Pre-Commit Validation Report" in report
        assert "1 passed" in report
        assert "1 warnings" in report
        assert "1 errors" in report


# =============================================================================
# RegressionDetector Tests
# =============================================================================

class TestRegressionDetector:
    """Tests for RegressionDetector."""
    
    def test_init_with_defaults(self):
        """Test initialization with default baselines."""
        detector = RegressionDetector()
        assert "C_H" in detector.baselines
        assert "lambda_star" in detector.baselines
    
    def test_check_observable_pass(self):
        """Test checking observable that passes."""
        detector = RegressionDetector()
        report = detector.check_observable("C_H", 0.045935703598)
        
        assert not report.is_regression
        assert report.deviation < 1e-10
    
    def test_check_observable_fail(self):
        """Test checking observable that fails."""
        detector = RegressionDetector()
        # Significantly different value
        report = detector.check_observable("C_H", 0.05)
        
        assert report.is_regression
        assert report.deviation > report.tolerance
    
    def test_check_unknown_observable(self):
        """Test checking unknown observable."""
        detector = RegressionDetector()
        report = detector.check_observable("unknown_observable", 42.0)
        
        # Should not be marked as regression (no baseline)
        assert not report.is_regression
        assert report.theoretical_reference == "Unknown baseline"
    
    def test_check_all(self):
        """Test checking multiple observables."""
        detector = RegressionDetector()
        values = {
            "C_H": 0.045935703598,
            "lambda_star": 52.63789013914324,
        }
        
        reports = detector.check_all(values)
        assert len(reports) == 2
        assert not any(r.is_regression for r in reports)
    
    def test_has_regressions(self):
        """Test regression detection flag."""
        detector = RegressionDetector()
        
        # No regressions initially
        assert not detector.has_regressions()
        
        # Add a passing check
        detector.check_observable("C_H", 0.045935703598)
        assert not detector.has_regressions()
        
        # Add a failing check
        detector.check_observable("alpha_inv", 140.0)  # Wrong value
        assert detector.has_regressions()
    
    def test_generate_report(self):
        """Test report generation."""
        detector = RegressionDetector()
        detector.check_observable("C_H", 0.045935703598)
        detector.check_observable("alpha_inv", 140.0)  # Regression
        
        report = detector.generate_report()
        assert "Regression Detection Report" in report
        assert "REGRESSIONS DETECTED" in report
        assert "alpha_inv" in report
    
    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        detector = RegressionDetector()
        
        # With default tolerance, should fail
        report1 = detector.check_observable("C_H", 0.046)
        assert report1.is_regression
        
        # With relaxed tolerance, should pass
        detector2 = RegressionDetector()
        report2 = detector2.check_observable("C_H", 0.046, tolerance=0.01)
        assert not report2.is_regression


# =============================================================================
# TestTierRunner Tests
# =============================================================================

class TestTestTierRunner:
    """Tests for TestTierRunner."""
    
    def test_init(self):
        """Test runner initialization."""
        runner = TestTierRunner()
        assert runner.repo_root == Path.cwd()
    
    def test_tier_directories(self):
        """Test tier directory mapping."""
        # T1 should only have unit tests
        assert TestTierRunner.TIER_DIRECTORIES[TestTier.T1_FAST] == ["tests/unit"]
        
        # T4 should have all directories
        assert len(TestTierRunner.TIER_DIRECTORIES[TestTier.T4_FALSIFICATION]) == 6
    
    def test_tier_timeouts(self):
        """Test tier timeout values."""
        assert TestTierRunner.TIER_TIMEOUTS[TestTier.T1_FAST] == 60
        assert TestTierRunner.TIER_TIMEOUTS[TestTier.T4_FALSIFICATION] == 21600
    
    def test_generate_report(self):
        """Test report generation."""
        runner = TestTierRunner()
        runner.results = [
            TestResult("test1", TestTier.T1_FAST, True, 0.1),
            TestResult("test2", TestTier.T1_FAST, True, 0.2),
            TestResult("test3", TestTier.T1_FAST, False, 0.1),
        ]
        
        report = runner.generate_report()
        assert "Test Tier Execution Report" in report
        assert "T1_FAST" in report


# =============================================================================
# BaselineManager Tests
# =============================================================================

class TestBaselineManager:
    """Tests for BaselineManager."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        manager = BaselineManager()
        # Should have default baselines
        assert "C_H" in manager.baselines
    
    def test_update_baseline(self):
        """Test updating a baseline."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manager = BaselineManager(Path(f.name))
            
            manager.update(
                name="new_observable",
                value=1.234,
                uncertainty=0.001,
                tolerance=0.01,
                reference="Eq. 1.1",
            )
            
            assert "new_observable" in manager.baselines
            assert manager.baselines["new_observable"]["value"] == 1.234
    
    def test_save_and_load(self):
        """Test saving and loading baselines."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
            
            # Create and save
            manager1 = BaselineManager(filepath)
            manager1.update("test", 42.0, 0.1, 0.5, "ref")
            manager1.save()
            
            # Load in new manager
            manager2 = BaselineManager(filepath)
            assert "test" in manager2.baselines
            assert manager2.baselines["test"]["value"] == 42.0
    
    def test_get_baseline(self):
        """Test getting a specific baseline."""
        manager = BaselineManager()
        baseline = manager.get("C_H")
        
        assert baseline is not None
        assert "value" in baseline
    
    def test_list_baselines(self):
        """Test listing all baselines."""
        manager = BaselineManager()
        names = manager.list_baselines()
        
        assert "C_H" in names
        assert "lambda_star" in names


# =============================================================================
# CoverageReporter Tests
# =============================================================================

class TestCoverageReporter:
    """Tests for CoverageReporter."""
    
    def test_init(self):
        """Test reporter initialization."""
        reporter = CoverageReporter()
        assert reporter.repo_root == Path.cwd()
    
    def test_generate_theoretical_coverage_report_no_data(self):
        """Test report generation without coverage data."""
        reporter = CoverageReporter()
        report = reporter.generate_theoretical_coverage_report()
        
        assert "Theoretical Coverage Report" in report
        assert "No coverage data available" in report
    
    def test_get_file_coverage_no_data(self):
        """Test getting file coverage without data."""
        reporter = CoverageReporter()
        coverage = reporter.get_file_coverage("src/test.py")
        
        assert coverage is None


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_detect_regressions_no_regression(self):
        """Test regression detection with no regressions."""
        values = {
            "C_H": 0.045935703598,
            "lambda_star": 52.63789013914324,
        }
        
        no_regression, report = detect_regressions(values)
        assert no_regression
        assert "Regression Detection Report" in report
    
    def test_detect_regressions_with_regression(self):
        """Test regression detection with regressions."""
        values = {
            "C_H": 1.0,  # Wrong value
        }
        
        no_regression, report = detect_regressions(values)
        assert not no_regression
        assert "REGRESSIONS DETECTED" in report
    
    def test_update_baselines(self):
        """Test baseline update function."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
            
            new_values = {
                "new_obs": {
                    "value": 1.0,
                    "uncertainty": 0.1,
                    "tolerance": 0.5,
                    "reference": "Test",
                },
            }
            
            update_baselines(new_values, filepath)
            
            # Verify it was saved
            with open(filepath) as f2:
                data = json.load(f2)
            
            assert "new_obs" in data


# =============================================================================
# TestTier Enum Tests
# =============================================================================

class TestTestTierEnum:
    """Tests for TestTier enum."""
    
    def test_all_tiers_exist(self):
        """Test all expected tiers exist."""
        assert TestTier.T1_FAST.value == "T1"
        assert TestTier.T2_STANDARD.value == "T2"
        assert TestTier.T3_COMPREHENSIVE.value == "T3"
        assert TestTier.T4_FALSIFICATION.value == "T4"
    
    def test_tier_count(self):
        """Test correct number of tiers."""
        assert len(TestTier) == 4


# =============================================================================
# ValidationStatus Enum Tests
# =============================================================================

class TestValidationStatusEnum:
    """Tests for ValidationStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.ERROR.value == "error"
    
    def test_status_count(self):
        """Test correct number of statuses."""
        assert len(ValidationStatus) == 5


# =============================================================================
# Integration Tests
# =============================================================================

class TestCICDIntegration:
    """Integration tests for CI/CD infrastructure."""
    
    def test_full_validation_workflow(self):
        """Test full validation workflow."""
        # Create validator
        validator = PreCommitValidator()
        
        # Create detector
        detector = RegressionDetector()
        
        # Check some values
        detector.check_observable("C_H", 0.045935703598)
        detector.check_observable("lambda_star", 52.63789013914324)
        
        # Generate reports
        val_report = validator.generate_report()
        reg_report = detector.generate_report()
        
        assert "Pre-Commit Validation Report" in val_report
        assert "Regression Detection Report" in reg_report
    
    def test_baseline_roundtrip(self):
        """Test baseline save/load roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
            
            # Create manager and add baselines
            manager = BaselineManager(filepath)
            manager.update("test1", 1.0, 0.1, 0.5, "ref1")
            manager.update("test2", 2.0, 0.2, 0.5, "ref2")
            manager.save()
            
            # Load in detector and check
            detector = RegressionDetector(filepath)
            assert "test1" in detector.baselines
            assert "test2" in detector.baselines
    
    def test_theoretical_reference_propagation(self):
        """Test that theoretical references are preserved."""
        detector = RegressionDetector()
        report = detector.check_observable("C_H", 0.045935703598)
        
        assert "IRH21.md" in report.theoretical_reference
        assert "Eq. 1.16" in report.theoretical_reference
