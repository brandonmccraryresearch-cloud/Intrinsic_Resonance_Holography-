"""
Tests for Benchmark Suites

THEORETICAL FOUNDATION: docs/ROADMAP.md §3.7

Tests for benchmark infrastructure:
    - RG flow benchmarks
    - QNCD benchmarks
    - cGFT action benchmarks

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import numpy as np
import pytest


class TestBenchmarkModuleImports:
    """Tests for benchmark module imports."""
    
    def test_benchmarks_module_import(self):
        """Test benchmarks module imports successfully."""
        from src.performance import benchmarks
        
        assert hasattr(benchmarks, '__version__')
    
    def test_rg_flow_bench_import(self):
        """Test RG flow benchmarks import."""
        from src.performance.benchmarks import rg_flow_bench
        
        assert hasattr(rg_flow_bench, 'benchmark_beta_functions')
        assert hasattr(rg_flow_bench, 'RGFlowBenchmarkSuite')
    
    def test_qncd_bench_import(self):
        """Test QNCD benchmarks import."""
        from src.performance.benchmarks import qncd_bench
        
        assert hasattr(qncd_bench, 'benchmark_qncd_single')
        assert hasattr(qncd_bench, 'QNCDBenchmarkSuite')
    
    def test_action_bench_import(self):
        """Test action benchmarks import."""
        from src.performance.benchmarks import action_bench
        
        assert hasattr(action_bench, 'benchmark_kinetic_action')
        assert hasattr(action_bench, 'ActionBenchmarkSuite')


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult instantiation."""
        from src.performance.benchmarks.rg_flow_bench import BenchmarkResult
        
        result = BenchmarkResult(
            name='test_bench',
            iterations=100,
            total_time_s=1.0,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=8.0,
            max_time_ms=15.0,
            throughput=100.0
        )
        
        assert result.name == 'test_bench'
        assert result.iterations == 100
        assert result.mean_time_ms == 10.0
    
    def test_benchmark_result_to_dict(self):
        """Test serialization to dictionary."""
        from src.performance.benchmarks.rg_flow_bench import BenchmarkResult
        
        result = BenchmarkResult(
            name='test',
            iterations=50,
            total_time_s=0.5,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=9.0,
            max_time_ms=12.0,
            throughput=100.0
        )
        
        d = result.to_dict()
        assert d['name'] == 'test'
        assert d['iterations'] == 50


class TestRGFlowBenchmarks:
    """Tests for RG flow benchmark functions."""
    
    def test_benchmark_beta_functions_returns_dict(self):
        """Test benchmark returns dictionary of results."""
        from src.performance.benchmarks import benchmark_beta_functions
        
        results = benchmark_beta_functions(
            batch_sizes=[1, 10],
            iterations=5
        )
        
        assert isinstance(results, dict)
        assert 'batch_1' in results
        assert 'batch_10' in results
    
    def test_benchmark_beta_functions_result_structure(self):
        """Test benchmark result has correct structure."""
        from src.performance.benchmarks import benchmark_beta_functions
        
        results = benchmark_beta_functions(
            batch_sizes=[10],
            iterations=5
        )
        
        result = results['batch_10']
        assert hasattr(result, 'mean_time_ms')
        assert hasattr(result, 'throughput')
        assert 'batch_size' in result.metadata
        assert 'theoretical_reference' in result.metadata
    
    def test_benchmark_fixed_point_search(self):
        """Test fixed point search benchmarks."""
        from src.performance.benchmarks import benchmark_fixed_point_search
        
        results = benchmark_fixed_point_search(
            n_points=[1, 5],
            iterations=3
        )
        
        assert 'points_1' in results
        assert 'points_5' in results
    
    def test_benchmark_rg_trajectory(self):
        """Test RG trajectory benchmarks."""
        from src.performance.benchmarks import benchmark_rg_trajectory
        
        results = benchmark_rg_trajectory(
            n_steps_list=[10, 100],
            iterations=3
        )
        
        assert 'steps_10' in results
        assert 'steps_100' in results


class TestRGFlowBenchmarkSuite:
    """Tests for RGFlowBenchmarkSuite class."""
    
    def test_suite_creation(self):
        """Test benchmark suite instantiation."""
        from src.performance.benchmarks import RGFlowBenchmarkSuite
        
        suite = RGFlowBenchmarkSuite(iterations=5)
        assert suite.iterations == 5
    
    def test_suite_run_all(self):
        """Test running all benchmarks."""
        from src.performance.benchmarks import RGFlowBenchmarkSuite
        
        suite = RGFlowBenchmarkSuite(iterations=2)
        results = suite.run_all()
        
        assert 'beta_functions' in results
        assert 'fixed_point_search' in results
        assert 'rg_trajectory' in results
    
    def test_suite_get_summary(self):
        """Test summary generation."""
        from src.performance.benchmarks import RGFlowBenchmarkSuite
        
        suite = RGFlowBenchmarkSuite(iterations=2)
        results = suite.run_all()
        summary = suite.get_summary(results)
        
        assert 'theoretical_reference' in summary
        assert 'categories' in summary


class TestQNCDBenchmarks:
    """Tests for QNCD benchmark functions."""
    
    def test_benchmark_qncd_single(self):
        """Test single-pair QNCD benchmarks."""
        from src.performance.benchmarks import benchmark_qncd_single
        
        results = benchmark_qncd_single(
            vector_sizes=[10, 50],
            iterations=5
        )
        
        assert 'dim_10' in results
        assert 'dim_50' in results
    
    def test_benchmark_qncd_batch(self):
        """Test batch QNCD benchmarks."""
        from src.performance.benchmarks import benchmark_qncd_batch
        
        results = benchmark_qncd_batch(
            batch_sizes=[5, 20],
            vector_dim=50,
            iterations=5
        )
        
        assert 'batch_5' in results
        assert 'batch_20' in results
    
    def test_benchmark_qncd_methods(self):
        """Test method comparison benchmarks."""
        from src.performance.benchmarks import benchmark_qncd_methods
        
        results = benchmark_qncd_methods(
            methods=['compression_proxy', 'entropy'],
            batch_size=10,
            iterations=5
        )
        
        assert 'compression_proxy' in results
        assert 'entropy' in results


class TestQNCDBenchmarkSuite:
    """Tests for QNCDBenchmarkSuite class."""
    
    def test_suite_creation(self):
        """Test benchmark suite instantiation."""
        from src.performance.benchmarks import QNCDBenchmarkSuite
        
        suite = QNCDBenchmarkSuite(iterations=5)
        assert suite.iterations == 5
    
    def test_suite_run_all(self):
        """Test running all benchmarks."""
        from src.performance.benchmarks import QNCDBenchmarkSuite
        
        suite = QNCDBenchmarkSuite(iterations=2)
        results = suite.run_all()
        
        assert 'single_pair' in results
        assert 'batch' in results
        assert 'methods' in results


class TestActionBenchmarks:
    """Tests for cGFT action benchmarks."""
    
    def test_benchmark_kinetic_action(self):
        """Test kinetic action benchmarks."""
        from src.performance.benchmarks import benchmark_kinetic_action
        
        results = benchmark_kinetic_action(
            field_sizes=[4, 8],
            iterations=3
        )
        
        assert 'N4' in results
        assert 'N8' in results
    
    def test_benchmark_interaction_action(self):
        """Test interaction action benchmarks."""
        from src.performance.benchmarks import benchmark_interaction_action
        
        results = benchmark_interaction_action(
            field_sizes=[4, 8],
            iterations=3
        )
        
        assert 'N4' in results
        assert 'N8' in results
    
    def test_benchmark_total_action(self):
        """Test total action benchmarks."""
        from src.performance.benchmarks import benchmark_total_action
        
        results = benchmark_total_action(
            field_sizes=[4],
            iterations=3
        )
        
        assert 'N4' in results


class TestActionBenchmarkSuite:
    """Tests for ActionBenchmarkSuite class."""
    
    def test_suite_creation(self):
        """Test benchmark suite instantiation."""
        from src.performance.benchmarks import ActionBenchmarkSuite
        
        suite = ActionBenchmarkSuite(iterations=5, field_sizes=[4])
        assert suite.iterations == 5
    
    def test_suite_run_all(self):
        """Test running all benchmarks."""
        from src.performance.benchmarks import ActionBenchmarkSuite
        
        suite = ActionBenchmarkSuite(iterations=2, field_sizes=[4])
        results = suite.run_all()
        
        assert 'kinetic' in results
        assert 'interaction' in results
        assert 'total' in results


class TestBenchmarkTheoreticalGrounding:
    """Tests for theoretical grounding of benchmarks."""
    
    def test_rg_flow_references(self):
        """Test RG flow benchmarks include theoretical references."""
        from src.performance.benchmarks import benchmark_beta_functions
        
        results = benchmark_beta_functions(batch_sizes=[10], iterations=2)
        result = results['batch_10']
        
        assert 'theoretical_reference' in result.metadata
        assert 'IRH' in result.metadata['theoretical_reference']
        assert 'Eq. 1.13' in result.metadata['theoretical_reference']
    
    def test_qncd_references(self):
        """Test QNCD benchmarks include theoretical references."""
        from src.performance.benchmarks import benchmark_qncd_single
        
        results = benchmark_qncd_single(vector_sizes=[10], iterations=2)
        result = results['dim_10']
        
        assert 'theoretical_reference' in result.metadata
        assert 'Appendix A' in result.metadata['theoretical_reference']
    
    def test_action_references(self):
        """Test action benchmarks include theoretical references."""
        from src.performance.benchmarks import benchmark_kinetic_action
        
        results = benchmark_kinetic_action(field_sizes=[4], iterations=2)
        result = results['N4']
        
        assert 'theoretical_reference' in result.metadata
        assert 'Eq. 1.1' in result.metadata['theoretical_reference']


class TestBenchmarkPerformance:
    """Tests to verify benchmark performance characteristics."""
    
    def test_beta_functions_scaling(self):
        """Test beta functions scale approximately linearly with batch size."""
        from src.performance.benchmarks import benchmark_beta_functions
        
        results = benchmark_beta_functions(
            batch_sizes=[100, 1000],
            iterations=10
        )
        
        time_100 = results['batch_100'].mean_time_ms
        time_1000 = results['batch_1000'].mean_time_ms
        
        # Time should scale roughly linearly (with some overhead)
        # 10x data should be less than 20x time
        assert time_1000 < time_100 * 20
    
    def test_throughput_positive(self):
        """Test all benchmarks report positive throughput."""
        from src.performance.benchmarks import (
            benchmark_beta_functions,
            benchmark_qncd_single,
            benchmark_kinetic_action
        )
        
        beta_results = benchmark_beta_functions([10], iterations=3)
        assert beta_results['batch_10'].throughput > 0
        
        qncd_results = benchmark_qncd_single([10], iterations=3)
        assert qncd_results['dim_10'].throughput > 0
        
        action_results = benchmark_kinetic_action([4], iterations=3)
        assert action_results['N4'].throughput > 0
