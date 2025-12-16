"""
Unit Tests for Output Contextualization Module

THEORETICAL FOUNDATION: IRH21.md Appendix K, copilot21promptMAX.md Phase III

Tests validate:
1. TheoreticalContext data class
2. ComputationalProvenance with reproducibility hashing
3. ObservableResult with uncertainty quantification
4. UncertaintyTracker propagation
5. IRHOutputWriter standardized output

Authors: IRH Computational Framework Team
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path for imports
import sys
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.utilities.output_contextualization import (
    ComputationType,
    TheoreticalContext,
    ComputationalProvenance,
    ObservableResult,
    UncertaintyTracker,
    IRHOutputWriter,
    create_output_writer,
    format_observable,
)


class TestComputationType:
    """Test ComputationType enum."""

    def test_all_types_exist(self):
        """Verify all computation types are defined."""
        assert ComputationType.RG_FLOW.value == "rg_flow"
        assert ComputationType.OBSERVABLE_EXTRACTION.value == "observable_extraction"
        assert ComputationType.TOPOLOGICAL_INVARIANT.value == "topological_invariant"
        assert ComputationType.CONVERGENCE_STUDY.value == "convergence_study"
        assert ComputationType.FALSIFICATION_TEST.value == "falsification_test"
        assert ComputationType.BENCHMARK.value == "benchmark"


class TestTheoreticalContext:
    """Test TheoreticalContext data class."""

    def test_default_creation(self):
        """Test default context creation."""
        ctx = TheoreticalContext()
        assert ctx.manuscript_version == "IRH21.md v21.0"
        assert ctx.equations_implemented == []
        assert ctx.section_references == []
        assert ctx.theoretical_precision_target == 1e-10

    def test_add_equation(self):
        """Test adding equation references."""
        ctx = TheoreticalContext()
        ctx.add_equation("1.1", "§1.1")
        ctx.add_equation("1.2", "§1.1")
        ctx.add_equation("1.1", "§1.1")  # Duplicate
        
        assert "1.1" in ctx.equations_implemented
        assert "1.2" in ctx.equations_implemented
        assert len(ctx.equations_implemented) == 2  # No duplicates
        assert "§1.1" in ctx.section_references
        assert len(ctx.section_references) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = TheoreticalContext()
        ctx.add_equation("1.14", "§1.2")
        
        d = ctx.to_dict()
        assert d['manuscript_version'] == "IRH21.md v21.0"
        assert "1.14" in d['equations_implemented']
        assert "§1.2" in d['section_references']


class TestComputationalProvenance:
    """Test ComputationalProvenance data class."""

    def test_default_creation(self):
        """Test default provenance creation."""
        prov = ComputationalProvenance()
        assert prov.timestamp  # Should have a timestamp
        assert prov.lattice_parameters == {}
        assert prov.rg_parameters == {}

    def test_gather_environment(self):
        """Test automatic environment gathering."""
        prov = ComputationalProvenance()
        prov.gather_environment()
        
        assert prov.python_version  # Should be populated
        assert prov.numpy_version
        assert 'platform' in prov.hardware_specs

    def test_reproducibility_hash_deterministic(self):
        """Test that reproducibility hash is deterministic."""
        prov1 = ComputationalProvenance()
        prov1.lattice_parameters = {'N': 10}
        prov1.rg_parameters = {'dt': 0.01}
        prov1.random_seed = 42
        
        prov2 = ComputationalProvenance()
        prov2.lattice_parameters = {'N': 10}
        prov2.rg_parameters = {'dt': 0.01}
        prov2.random_seed = 42
        
        assert prov1.compute_reproducibility_hash() == prov2.compute_reproducibility_hash()

    def test_reproducibility_hash_changes(self):
        """Test that hash changes with parameters."""
        prov1 = ComputationalProvenance()
        prov1.random_seed = 42
        
        prov2 = ComputationalProvenance()
        prov2.random_seed = 43
        
        assert prov1.compute_reproducibility_hash() != prov2.compute_reproducibility_hash()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        prov = ComputationalProvenance()
        prov.gather_environment()
        
        d = prov.to_dict()
        assert 'timestamp' in d
        assert 'reproducibility_hash' in d
        assert 'python_version' in d


class TestObservableResult:
    """Test ObservableResult data class."""

    def test_basic_creation(self):
        """Test basic observable creation."""
        obs = ObservableResult(
            name="alpha_inverse",
            value=137.036,
            uncertainty=0.001,
            unit=""
        )
        assert obs.name == "alpha_inverse"
        assert obs.value == 137.036
        assert obs.uncertainty == 0.001

    def test_sigma_deviation_calculation(self):
        """Test σ deviation calculation."""
        obs = ObservableResult(
            name="alpha_inverse",
            value=137.036,
            uncertainty=0.001,
            unit="",
            experimental_value=137.035999,
            experimental_uncertainty=0.00001
        )
        
        sigma = obs.compute_sigma_deviation()
        assert sigma is not None
        assert sigma < 0.01  # Should be very close

    def test_agreement_status(self):
        """Test agreement status classification."""
        # Excellent agreement
        obs1 = ObservableResult(
            name="test",
            value=100.0,
            uncertainty=1.0,
            unit="",
            experimental_value=100.5,
            experimental_uncertainty=1.0
        )
        assert obs1.get_agreement_status() == "EXCELLENT"
        
        # Tension
        obs2 = ObservableResult(
            name="test",
            value=100.0,
            uncertainty=1.0,
            unit="",
            experimental_value=105.0,
            experimental_uncertainty=1.0
        )
        assert obs2.get_agreement_status() == "TENSION"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        obs = ObservableResult(
            name="alpha_inverse",
            value=137.036,
            uncertainty=0.001,
            unit="",
            experimental_value=137.035999
        )
        
        d = obs.to_dict()
        assert d['name'] == "alpha_inverse"
        assert d['value'] == 137.036
        assert 'agreement_status' in d
        assert 'theoretical_foundation' in d


class TestUncertaintyTracker:
    """Test UncertaintyTracker class."""

    def test_register_uncertainty(self):
        """Test registering uncertainty sources."""
        tracker = UncertaintyTracker()
        
        tracker.register_source_uncertainty(
            observable="lambda_star",
            value=52.64,
            uncertainty=0.001,
            source="rg_convergence"
        )
        
        value, unc = tracker.get_uncertainty("lambda_star")
        assert value == 52.64
        assert unc == pytest.approx(0.001)

    def test_combined_uncertainty(self):
        """Test combining multiple uncertainty sources."""
        tracker = UncertaintyTracker()
        
        tracker.register_source_uncertainty(
            observable="C_H",
            value=0.0459,
            uncertainty=0.0001,
            source="rg_convergence"
        )
        tracker.register_source_uncertainty(
            observable="C_H",
            value=0.0459,
            uncertainty=0.0002,
            source="discretization"
        )
        
        value, unc = tracker.get_uncertainty("C_H")
        expected_unc = np.sqrt(0.0001**2 + 0.0002**2)
        assert unc == pytest.approx(expected_unc)

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through formula."""
        tracker = UncertaintyTracker()
        
        # Register inputs
        tracker.register_source_uncertainty("x", 2.0, 0.1, "measurement")
        tracker.register_source_uncertainty("y", 3.0, 0.1, "measurement")
        
        # Propagate through f(x,y) = x + y
        def add(x, y):
            return x + y
        
        result, unc = tracker.propagate_uncertainty(
            output_name="sum",
            formula=add,
            input_names=["x", "y"]
        )
        
        assert result == pytest.approx(5.0)
        expected_unc = np.sqrt(0.1**2 + 0.1**2)  # For addition
        assert unc == pytest.approx(expected_unc, rel=0.01)

    def test_uncertainty_report(self):
        """Test uncertainty report generation."""
        tracker = UncertaintyTracker()
        tracker.register_source_uncertainty("test_obs", 100.0, 1.0, "source1")
        tracker.register_source_uncertainty("test_obs", 100.0, 2.0, "source2")
        
        report = tracker.generate_uncertainty_report()
        assert "test_obs" in report
        assert "source1" in report
        assert "source2" in report


class TestIRHOutputWriter:
    """Test IRHOutputWriter class."""

    def test_creation(self):
        """Test basic creation."""
        writer = IRHOutputWriter(ComputationType.RG_FLOW)
        assert writer.computation_type == ComputationType.RG_FLOW
        assert writer.provenance is not None

    def test_creation_from_string(self):
        """Test creation from string type."""
        writer = IRHOutputWriter("rg_flow")
        assert writer.computation_type == ComputationType.RG_FLOW

    def test_set_parameters(self):
        """Test setting parameters."""
        writer = IRHOutputWriter(ComputationType.RG_FLOW)
        
        writer.set_lattice_parameters(N=10, dim=4)
        writer.set_rg_parameters(dt=0.01, k_UV=1e10)
        writer.set_random_seed(42)
        
        assert writer.provenance.lattice_parameters['N'] == 10
        assert writer.provenance.rg_parameters['dt'] == 0.01
        assert writer.provenance.random_seed == 42

    def test_add_result(self):
        """Test adding results."""
        writer = IRHOutputWriter(ComputationType.OBSERVABLE_EXTRACTION)
        
        result = writer.add_result(
            name="alpha_inverse",
            value=137.036,
            uncertainty=0.001,
            unit=""
        )
        
        assert len(writer.results) == 1
        assert result.name == "alpha_inverse"

    def test_add_equation_reference(self):
        """Test adding equation references."""
        writer = IRHOutputWriter(ComputationType.RG_FLOW)
        writer.add_equation_reference("1.14", "§1.2")
        
        assert "1.14" in writer.theoretical_context.equations_implemented

    def test_to_dict(self):
        """Test conversion to dictionary."""
        writer = IRHOutputWriter(ComputationType.RG_FLOW)
        writer.add_result("test", 1.0, 0.1, "unit")
        
        d = writer.to_dict()
        assert d['irh_def_version'] == '1.0'
        assert d['computation_type'] == 'rg_flow'
        assert 'provenance' in d
        assert 'results' in d

    def test_to_json(self):
        """Test JSON conversion."""
        writer = IRHOutputWriter(ComputationType.RG_FLOW)
        writer.add_result("test", 1.0, 0.1, "unit")
        
        json_str = writer.to_json()
        parsed = json.loads(json_str)
        assert parsed['computation_type'] == 'rg_flow'

    def test_generate_report(self):
        """Test human-readable report generation."""
        writer = IRHOutputWriter(ComputationType.OBSERVABLE_EXTRACTION)
        writer.add_equation_reference("1.14", "§1.2")
        writer.add_result(
            name="alpha_inverse",
            value=137.036,
            uncertainty=0.001,
            unit="",
            experimental_value=137.035999,
            experimental_uncertainty=0.00001
        )
        
        report = writer.generate_report()
        assert "IRH v21.0" in report
        assert "alpha_inverse" in report
        assert "EXCELLENT" in report or "σ" in report

    def test_write_to_file(self, tmp_path):
        """Test writing output to file."""
        output_file = tmp_path / "test_output.json"
        
        writer = IRHOutputWriter(ComputationType.RG_FLOW, str(output_file))
        writer.add_result("test", 1.0, 0.1, "unit")
        writer.write()
        
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['computation_type'] == 'rg_flow'


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_output_writer(self):
        """Test factory function."""
        writer = create_output_writer("rg_flow")
        assert isinstance(writer, IRHOutputWriter)
        assert writer.computation_type == ComputationType.RG_FLOW

    def test_format_observable(self):
        """Test observable formatting."""
        result = format_observable("α⁻¹", 137.036, 0.001, "")
        assert "e+02" in result or "137" in result  # Scientific notation or regular
        assert "±" in result
        
        result_with_unit = format_observable("mass", 125.1, 0.1, "GeV")
        assert "GeV" in result_with_unit


class TestTheoreticalReferences:
    """Verify theoretical references in module."""

    def test_module_reference(self):
        """Module references Appendix K."""
        from src.utilities import output_contextualization
        assert 'IRH21.md' in output_contextualization.__theoretical_foundation__
        assert 'Appendix K' in output_contextualization.__theoretical_foundation__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
