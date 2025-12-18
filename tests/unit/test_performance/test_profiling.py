"""
Tests for Profiling Module

THEORETICAL FOUNDATION: docs/ROADMAP.md §3.7-3.8

Tests for profiling utilities:
    - Timing functionality
    - Memory profiling
    - Profile report generation
    - Decorators

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import time
import numpy as np
import pytest


class TestTimingResult:
    """Tests for TimingResult dataclass."""
    
    def test_timing_result_creation(self):
        """Test TimingResult instantiation."""
        from src.performance.profiling import TimingResult
        
        result = TimingResult(
            function_name='test_fn',
            execution_time_ns=1_000_000
        )
        
        assert result.function_name == 'test_fn'
        assert result.execution_time_ns == 1_000_000
        assert result.execution_time_ms == 1.0
        assert result.execution_time_s == 0.001
    
    def test_timing_result_to_dict(self):
        """Test serialization to dictionary."""
        from src.performance.profiling import TimingResult
        
        result = TimingResult(
            function_name='test_fn',
            execution_time_ns=2_500_000
        )
        
        d = result.to_dict()
        assert d['function_name'] == 'test_fn'
        assert d['execution_time_ns'] == 2_500_000
        assert d['execution_time_ms'] == 2.5
        assert 'timestamp' in d


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""
    
    def test_memory_result_creation(self):
        """Test MemoryResult instantiation."""
        from src.performance.profiling import MemoryResult
        
        result = MemoryResult(
            function_name='memory_fn',
            peak_memory_bytes=1024 * 1024  # 1 MB
        )
        
        assert result.function_name == 'memory_fn'
        assert result.peak_memory_bytes == 1024 * 1024
        assert result.peak_memory_mb == 1.0
    
    def test_memory_result_to_dict(self):
        """Test serialization to dictionary."""
        from src.performance.profiling import MemoryResult
        
        result = MemoryResult(
            function_name='memory_fn',
            peak_memory_bytes=2 * 1024 * 1024
        )
        
        d = result.to_dict()
        assert d['function_name'] == 'memory_fn'
        assert d['peak_memory_mb'] == 2.0


class TestProfiler:
    """Tests for Profiler class."""
    
    def test_profiler_creation(self):
        """Test Profiler instantiation."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('test_profiler')
        assert profiler.name == 'test_profiler'
        assert profiler.enable_timing == True
        assert profiler.enable_memory == True
    
    def test_profiler_timing_context(self):
        """Test timing context manager."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('timing_test')
        
        with profiler.profile_timing('sleep_operation'):
            time.sleep(0.01)  # 10ms
        
        report = profiler.generate_report()
        assert len(report.timing_results) == 1
        assert report.timing_results[0].function_name == 'sleep_operation'
        assert report.timing_results[0].execution_time_ms >= 9  # At least 9ms
    
    def test_profiler_memory_context(self):
        """Test memory profiling context manager."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('memory_test')
        
        with profiler.profile_memory('array_allocation'):
            # Allocate ~8 MB array
            _ = np.zeros((1000, 1000), dtype=np.float64)
        
        report = profiler.generate_report()
        assert len(report.memory_results) == 1
        assert report.memory_results[0].function_name == 'array_allocation'
        # Peak memory should be positive
        assert report.memory_results[0].peak_memory_bytes > 0
    
    def test_profiler_disabled_timing(self):
        """Test profiler with timing disabled."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('no_timing', enable_timing=False)
        
        with profiler.profile_timing('should_not_record'):
            time.sleep(0.001)
        
        report = profiler.generate_report()
        assert len(report.timing_results) == 0
    
    def test_profiler_reset(self):
        """Test profiler reset functionality."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('reset_test')
        
        with profiler.profile_timing('op1'):
            pass
        
        assert len(profiler._timing_results) == 1
        
        profiler.reset()
        assert len(profiler._timing_results) == 0
    
    def test_profiler_get_stats(self):
        """Test statistics retrieval."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('stats_test')
        
        with profiler.profile_timing('op1'):
            time.sleep(0.001)
        
        stats = profiler.get_stats()
        assert stats['name'] == 'stats_test'
        assert 'timing' in stats


class TestProfileReport:
    """Tests for ProfileReport class."""
    
    def test_profile_report_creation(self):
        """Test ProfileReport instantiation."""
        from src.performance.profiling import ProfileReport
        
        report = ProfileReport(name='test_report')
        assert report.name == 'test_report'
        assert len(report.timing_results) == 0
        assert len(report.memory_results) == 0
    
    def test_profile_report_summary(self):
        """Test report summary generation."""
        from src.performance.profiling import ProfileReport, TimingResult
        
        report = ProfileReport(name='summary_test')
        report.timing_results = [
            TimingResult('fn1', 1_000_000),
            TimingResult('fn2', 2_000_000),
            TimingResult('fn3', 3_000_000),
        ]
        
        summary = report.get_summary()
        assert summary['name'] == 'summary_test'
        assert summary['timing']['count'] == 3
        assert summary['timing']['total_ms'] == 6.0
        assert summary['timing']['mean_ms'] == 2.0
    
    def test_profile_report_format(self):
        """Test report formatting."""
        from src.performance.profiling import ProfileReport, TimingResult
        
        report = ProfileReport(name='format_test')
        report.timing_results = [
            TimingResult('fn1', 1_000_000),
        ]
        
        formatted = report.format_report()
        assert 'format_test' in formatted
        assert 'TIMING' in formatted
    
    def test_profile_report_to_dict(self):
        """Test serialization to dictionary."""
        from src.performance.profiling import ProfileReport
        
        report = ProfileReport(name='dict_test')
        d = report.to_dict()
        
        assert d['name'] == 'dict_test'
        assert 'summary' in d
        assert 'timing_results' in d


class TestProfilerRegistry:
    """Tests for profiler registry functions."""
    
    def test_create_profiler(self):
        """Test profiler creation via registry."""
        from src.performance.profiling import create_profiler
        
        profiler = create_profiler('registry_test_1')
        assert profiler.name == 'registry_test_1'
    
    def test_create_profiler_returns_existing(self):
        """Test that create_profiler returns existing profiler."""
        from src.performance.profiling import create_profiler
        
        p1 = create_profiler('same_profiler_name')
        p2 = create_profiler('same_profiler_name')
        
        assert p1 is p2
    
    def test_get_profiling_stats(self):
        """Test global stats retrieval."""
        from src.performance.profiling import (
            create_profiler, get_profiling_stats
        )
        
        profiler = create_profiler('stats_registry_test')
        with profiler.profile_timing('test_op'):
            pass
        
        stats = get_profiling_stats()
        assert 'stats_registry_test' in stats


class TestProfileDecorator:
    """Tests for @profile decorator."""
    
    def test_profile_decorator_basic(self):
        """Test basic decorator usage."""
        from src.performance.profiling import profile, create_profiler
        
        @profile('decorator_test_profiler')
        def test_function():
            return 42
        
        result = test_function()
        assert result == 42
        
        profiler = create_profiler('decorator_test_profiler')
        report = profiler.generate_report()
        assert len(report.timing_results) > 0
    
    def test_profile_decorator_preserves_name(self):
        """Test decorator preserves function name."""
        from src.performance.profiling import profile
        
        @profile('name_test_profiler')
        def original_name():
            pass
        
        assert original_name.__name__ == 'original_name'


class TestTimeFunctionDecorator:
    """Tests for @time_function decorator."""
    
    def test_time_function_returns_tuple(self):
        """Test decorator returns (result, timing) tuple."""
        from src.performance.profiling import time_function
        
        @time_function
        def add(a, b):
            return a + b
        
        result, timing = add(1, 2)
        
        assert result == 3
        assert hasattr(timing, 'execution_time_ms')
    
    def test_time_function_measures_time(self):
        """Test timing measurement is accurate."""
        from src.performance.profiling import time_function
        
        @time_function
        def sleep_fn():
            time.sleep(0.01)
            return 'done'
        
        result, timing = sleep_fn()
        
        assert result == 'done'
        assert timing.execution_time_ms >= 9  # At least 9ms


class TestMemoryProfileDecorator:
    """Tests for @memory_profile decorator."""
    
    def test_memory_profile_returns_tuple(self):
        """Test decorator returns (result, memory) tuple."""
        from src.performance.profiling import memory_profile
        
        @memory_profile
        def allocate():
            return np.zeros(1000)
        
        result, memory = allocate()
        
        assert len(result) == 1000
        assert hasattr(memory, 'peak_memory_mb')
    
    def test_memory_profile_measures_allocation(self):
        """Test memory measurement captures allocations."""
        from src.performance.profiling import memory_profile
        
        @memory_profile
        def allocate_large():
            # Allocate ~8 MB
            return np.zeros((1000, 1000), dtype=np.float64)
        
        result, memory = allocate_large()
        
        # Should measure some memory (at least a few KB)
        assert memory.peak_memory_bytes > 0


class TestTheoreticalReference:
    """Tests for theoretical grounding."""
    
    def test_profile_report_has_reference(self):
        """Test ProfileReport includes theoretical reference."""
        from src.performance.profiling import ProfileReport
        
        report = ProfileReport(name='ref_test')
        assert 'ROADMAP' in report.theoretical_reference
    
    def test_profiler_module_docstring(self):
        """Test module has proper docstring."""
        from src.performance import profiling
        
        assert profiling.__doc__ is not None
        assert 'ROADMAP' in profiling.__doc__


class TestProfilerIntegration:
    """Integration tests for profiling system."""
    
    def test_full_profiling_workflow(self):
        """Test complete profiling workflow."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('integration_test', enable_call_graph=False)
        
        # Profile multiple operations
        for i in range(3):
            with profiler.profile_timing(f'operation_{i}'):
                _ = np.random.rand(100, 100)
        
        with profiler.profile_memory('large_allocation'):
            _ = np.zeros((500, 500))
        
        # Generate and verify report
        report = profiler.generate_report()
        
        assert report.name == 'integration_test'
        assert len(report.timing_results) == 3
        assert len(report.memory_results) == 1
        
        # Verify summary
        summary = report.get_summary()
        assert summary['timing']['count'] == 3
        assert summary['memory']['count'] == 1
    
    def test_profiler_with_exceptions(self):
        """Test profiler handles exceptions gracefully."""
        from src.performance.profiling import Profiler
        
        profiler = Profiler('exception_test')
        
        try:
            with profiler.profile_timing('failing_op'):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Timing should still be recorded
        report = profiler.generate_report()
        assert len(report.timing_results) == 1
