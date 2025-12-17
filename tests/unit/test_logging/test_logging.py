"""
Tests for IRH v21.0 Logging Module

THEORETICAL FOUNDATION: IRH21.md Appendix K

Tests for structured logging and provenance tracking.
"""

import pytest
import json
import sys
from io import StringIO
from datetime import datetime

from src.logging.structured_logger import (
    StructuredLogger,
    LogEntry,
    LogLevel,
    configure_logging,
    get_logger,
    create_logger,
)

from src.logging.provenance import (
    ProvenanceTracker,
    ComputationRecord,
    create_provenance_tracker,
    get_provenance_tracker,
)


class TestLogLevel:
    """Test LogLevel enumeration."""
    
    def test_level_values(self):
        """Log levels should have correct ordering."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.STEP.value
        assert LogLevel.STEP.value < LogLevel.RESULT.value
        assert LogLevel.RESULT.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value
    
    def test_from_string(self):
        """Should convert string to LogLevel."""
        assert LogLevel.from_string('info') == LogLevel.INFO
        assert LogLevel.from_string('WARNING') == LogLevel.WARNING


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_creation(self):
        """Should create log entry."""
        entry = LogEntry(
            timestamp="2025-01-01T00:00:00Z",
            level="INFO",
            message="Test message"
        )
        assert entry.message == "Test message"
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        entry = LogEntry(
            timestamp="2025-01-01T00:00:00Z",
            level="INFO",
            message="Test"
        )
        d = entry.to_dict()
        assert d['message'] == "Test"
        assert d['level'] == "INFO"
    
    def test_to_json(self):
        """Should convert to JSON string."""
        entry = LogEntry(
            timestamp="2025-01-01T00:00:00Z",
            level="INFO",
            message="Test"
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        assert data['message'] == "Test"
    
    def test_to_text(self):
        """Should convert to human-readable text."""
        entry = LogEntry(
            timestamp="2025-01-01T00:00:00Z",
            level="INFO",
            message="Test message",
            module="test_module"
        )
        text = entry.to_text()
        assert "INFO" in text
        assert "Test message" in text


class TestStructuredLogger:
    """Test StructuredLogger class."""
    
    def test_creation(self):
        """Logger should be creatable."""
        logger = StructuredLogger(name="test")
        assert logger.name == "test"
    
    def test_log_levels(self):
        """Should respect log levels."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.WARNING,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        logger.info("Should not appear")
        logger.warning("Should appear")
        
        output = stream.getvalue()
        assert "Should not appear" not in output
        assert "Should appear" in output
    
    def test_info_logging(self):
        """Should log info messages."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        logger.info("Test info")
        assert "Test info" in stream.getvalue()
    
    def test_step_logging(self):
        """Should log step messages."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        logger.step("Computation step")
        assert "STEP" in stream.getvalue()
    
    def test_result_logging(self):
        """Should log results with values."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        logger.result("Computed value", value=42)
        output = stream.getvalue()
        assert "RESULT" in output
        assert "42" in output
    
    def test_context_manager(self):
        """Context should add metadata to logs."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        with logger.context(module='test_module', theoretical_ref='§1.2'):
            logger.info("Message with context")
        
        output = stream.getvalue()
        assert "§1.2" in output or "test_module" in output
    
    def test_timed_context(self):
        """Timed context should record duration."""
        stream = StringIO()
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            stream=stream,
            text_output=True,
            json_output=False
        )
        
        with logger.timed("Test operation"):
            pass
        
        output = stream.getvalue()
        assert "Starting" in output or "Completed" in output
    
    def test_get_entries(self):
        """Should retrieve logged entries."""
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            text_output=False,
            json_output=False
        )
        
        logger.info("Entry 1")
        logger.warning("Entry 2")
        
        entries = logger.get_entries()
        assert len(entries) == 2
    
    def test_get_summary(self):
        """Should generate summary statistics."""
        logger = StructuredLogger(
            name="test",
            level=LogLevel.DEBUG,
            text_output=False,
            json_output=False
        )
        
        logger.info("Info message")
        logger.warning("Warning message")
        
        summary = logger.get_summary()
        assert summary['total'] == 2
        assert 'INFO' in summary['by_level']


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_configure_logging(self):
        """Should configure default logger."""
        logger = configure_logging(name="configured", level="DEBUG")
        assert logger.name == "configured"
    
    def test_get_logger(self):
        """Should get default logger."""
        logger = get_logger()
        assert logger is not None
    
    def test_create_logger(self):
        """Should create new logger."""
        logger = create_logger("new_logger", level="WARNING")
        assert logger.name == "new_logger"
        assert logger.level == LogLevel.WARNING


class TestComputationRecord:
    """Test ComputationRecord dataclass."""
    
    def test_creation(self):
        """Should create computation record."""
        record = ComputationRecord(
            name="test_computation",
            description="Test description"
        )
        assert record.name == "test_computation"
    
    def test_auto_id_generation(self):
        """Should auto-generate ID."""
        record = ComputationRecord(name="test")
        assert record.id is not None
        assert len(record.id) == 8
    
    def test_timestamp_generation(self):
        """Should auto-generate timestamp."""
        record = ComputationRecord(name="test")
        assert record.timestamp is not None
    
    def test_environment_gathering(self):
        """Should gather environment info."""
        record = ComputationRecord(name="test")
        assert record.python_version is not None
        assert record.numpy_version is not None
    
    def test_checksum_computation(self):
        """Should compute checksum."""
        record = ComputationRecord(
            name="test",
            input_parameters={'x': 1.0}
        )
        assert record.checksum is not None
        assert len(record.checksum) == 16
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        record = ComputationRecord(name="test")
        d = record.to_dict()
        assert d['name'] == "test"
    
    def test_to_json(self):
        """Should convert to JSON."""
        record = ComputationRecord(name="test")
        json_str = record.to_json()
        data = json.loads(json_str)
        assert data['name'] == "test"
    
    def test_from_dict(self):
        """Should create from dictionary."""
        d = {
            'name': 'test',
            'description': 'desc',
            'input_parameters': {'x': 1.0}
        }
        record = ComputationRecord.from_dict(d)
        assert record.name == "test"


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""
    
    def test_creation(self):
        """Tracker should be creatable."""
        tracker = ProvenanceTracker()
        assert tracker is not None
    
    def test_start_computation(self):
        """Should start tracking computation."""
        tracker = ProvenanceTracker()
        record = tracker.start_computation(
            name="test",
            theoretical_ref="§1.2",
            input_parameters={'x': 1.0}
        )
        assert record.name == "test"
    
    def test_add_input(self):
        """Should add input parameters."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test")
        tracker.add_input("y", 2.0)
        
        record = tracker.complete_computation()
        assert record.input_parameters['y'] == 2.0
    
    def test_add_result(self):
        """Should add output results."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test")
        tracker.add_result("z", 3.0)
        
        record = tracker.complete_computation()
        assert record.output_results['z'] == 3.0
    
    def test_complete_computation(self):
        """Should complete and store record."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test")
        record = tracker.complete_computation(duration_seconds=1.5)
        
        assert record.duration_seconds == 1.5
        
        # Record should be stored
        records = tracker.get_records()
        assert len(records) == 1
    
    def test_get_record_by_id(self):
        """Should retrieve record by ID."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test")
        record = tracker.complete_computation()
        
        retrieved = tracker.get_record(record.id)
        assert retrieved is not None
        assert retrieved.name == "test"
    
    def test_verify_record(self):
        """Should verify record integrity."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test", input_parameters={'x': 1.0})
        record = tracker.complete_computation()
        
        result = tracker.verify_record(record.id)
        assert result['valid'] is True
    
    def test_chained_computations(self):
        """Should track computation chains."""
        tracker = ProvenanceTracker()
        
        # First computation
        tracker.start_computation(name="first")
        first = tracker.complete_computation()
        
        # Second computation, child of first
        tracker.start_computation(name="second", parent_id=first.id)
        second = tracker.complete_computation()
        
        # Get chain
        chain = tracker.get_chain(second.id)
        assert len(chain) == 2
        assert chain[0].name == "first"
        assert chain[1].name == "second"
    
    def test_generate_report(self):
        """Should generate provenance report."""
        tracker = ProvenanceTracker()
        tracker.start_computation(name="test")
        tracker.complete_computation()
        
        report = tracker.generate_report()
        assert "Provenance Report" in report
        assert "test" in report


class TestProvenanceModuleFunctions:
    """Test provenance module-level functions."""
    
    def test_create_provenance_tracker(self):
        """Should create tracker."""
        tracker = create_provenance_tracker()
        assert tracker is not None
    
    def test_get_provenance_tracker(self):
        """Should get default tracker."""
        tracker = get_provenance_tracker()
        assert tracker is not None


class TestTheoreticalGrounding:
    """Test that modules have proper theoretical grounding."""
    
    def test_logger_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.logging import structured_logger
        assert hasattr(structured_logger, '__theoretical_foundation__')
        assert 'IRH21' in structured_logger.__theoretical_foundation__
    
    def test_provenance_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.logging import provenance
        assert hasattr(provenance, '__theoretical_foundation__')
        assert 'IRH21' in provenance.__theoretical_foundation__
