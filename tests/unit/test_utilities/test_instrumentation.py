"""
Unit Tests for Instrumentation Module

THEORETICAL FOUNDATION: copilot21promtMAX.md Phase II

Tests validate:
1. Logger initialization and configuration
2. Theoretical context logging
3. RG flow narration
4. Verification reporting

Authors: IRH Computational Framework Team
"""

import io
import logging
from pathlib import Path

import pytest

# Add src to path for imports
import sys
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.utilities.instrumentation import (
    IRHLogLevel,
    TheoreticalReference,
    ComputationContext,
    IRHLogger,
    instrumented,
    get_logger,
    configure_logging,
)


class TestTheoreticalReference:
    """Test TheoreticalReference data class."""

    def test_reference_creation(self):
        """Test basic reference creation."""
        ref = TheoreticalReference(section="§1.1", equation="1.1")
        assert ref.section == "§1.1"
        assert ref.equation == "1.1"

    def test_reference_string(self):
        """Test string representation."""
        ref = TheoreticalReference(
            section="§1.2",
            equation="1.12",
            description="Wetterich equation"
        )
        s = str(ref)
        assert "IRH21.md" in s
        assert "§1.2" in s
        assert "Eq. 1.12" in s

    def test_reference_with_appendix(self):
        """Test reference with appendix."""
        ref = TheoreticalReference(section="§1.1", appendix="A")
        s = str(ref)
        assert "Appendix A" in s


class TestComputationContext:
    """Test ComputationContext data class."""

    def test_context_creation(self):
        """Test basic context creation."""
        ref = TheoreticalReference(section="§1.1")
        ctx = ComputationContext(
            operation="test_op",
            theoretical_ref=ref
        )
        assert ctx.operation == "test_op"

    def test_context_to_dict(self):
        """Test conversion to dictionary."""
        ref = TheoreticalReference(section="§1.1")
        ctx = ComputationContext(
            operation="test_op",
            theoretical_ref=ref,
            result=42.0,
            uncertainty=0.1
        )
        d = ctx.to_dict()
        
        assert d['operation'] == "test_op"
        assert d['result'] == 42.0
        assert d['uncertainty'] == 0.1


class TestIRHLogger:
    """Test IRHLogger class."""

    def test_logger_singleton(self):
        """Test singleton pattern."""
        logger1 = IRHLogger.get_instance()
        logger2 = IRHLogger.get_instance()
        assert logger1 is logger2

    def test_logger_init_message(self):
        """Test init logging."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        logger.init("Test initialization")
        
        output = stream.getvalue()
        assert "[INIT]" in output
        assert "Test initialization" in output

    def test_logger_exec_message(self):
        """Test execution logging."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        ref = TheoreticalReference(section="§1.1", equation="1.1")
        logger.exec("kinetic_action", ref, formula="S_kin = ∫φ̄Δφ")
        
        output = stream.getvalue()
        assert "[EXEC]" in output
        assert "kinetic_action" in output

    def test_logger_verify_pass(self):
        """Test verification logging (pass)."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        ref = TheoreticalReference(section="§1.1")
        logger.verify("gauge_invariance", passed=True, theoretical_ref=ref)
        
        output = stream.getvalue()
        assert "[VERIFY]" in output
        assert "PASS" in output

    def test_logger_verify_fail(self):
        """Test verification logging (fail)."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        ref = TheoreticalReference(section="§1.1")
        logger.verify("gauge_invariance", passed=False, theoretical_ref=ref)
        
        output = stream.getvalue()
        assert "[VERIFY]" in output
        assert "FAIL" in output

    def test_logger_rg_flow_start(self):
        """Test RG flow start logging."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        logger.rg_flow_start(
            lambda_0=1.0, gamma_0=1.0, mu_0=1.0,
            target_lambda=52.6, target_gamma=105.3, target_mu=157.9
        )
        
        output = stream.getvalue()
        assert "[RG-FLOW]" in output
        assert "Wetterich" in output

    def test_logger_result(self):
        """Test result logging."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        ref = TheoreticalReference(section="§3.2.2", equation="3.4")
        logger.result(
            name="alpha_inverse",
            value=137.036,
            theoretical_ref=ref,
            uncertainty=0.001,
            experimental=137.035999
        )
        
        output = stream.getvalue()
        assert "[RESULT]" in output
        assert "alpha_inverse" in output

    def test_logger_history(self):
        """Test log history."""
        stream = io.StringIO()
        logger = IRHLogger(stream=stream, include_timestamp=False)
        
        logger.init("Test 1")
        logger.init("Test 2")
        
        history = logger.get_history()
        assert len(history) == 2
        
        logger.clear_history()
        assert len(logger.get_history()) == 0


class TestInstrumentedDecorator:
    """Test @instrumented decorator."""

    def test_decorated_function(self):
        """Test that decorated function executes correctly."""
        ref = TheoreticalReference(section="§1.1", equation="1.1")
        
        @instrumented(ref, formula="f(x) = x²")
        def square(x):
            return x * x
        
        result = square(5)
        assert result == 25

    def test_decorated_function_preserves_name(self):
        """Test that decorator preserves function name."""
        ref = TheoreticalReference(section="§1.1")
        
        @instrumented(ref)
        def my_function():
            pass
        
        assert my_function.__name__ == "my_function"


class TestLogLevels:
    """Test log level enum."""

    def test_log_levels_exist(self):
        """Test all log levels are defined."""
        assert IRHLogLevel.INIT.value == "INIT"
        assert IRHLogLevel.EXEC.value == "EXEC"
        assert IRHLogLevel.VERIFY.value == "VERIFY"
        assert IRHLogLevel.RG_FLOW.value == "RG-FLOW"
        assert IRHLogLevel.RG_STEP.value == "RG-STEP"
        assert IRHLogLevel.RESULT.value == "RESULT"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
