"""
Unit tests for Phase VIII: Output Standardization.

Tests the IRH-DEF schema, output formatting, report generation,
compliance checking, and metadata management.

Theoretical Reference:
    IRH v21.1 Manuscript (Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices)
    - Intrinsic_Resonance_Holography-v21.1-Part1.md
    - Intrinsic_Resonance_Holography-v21.1-Part2.md
    Final Compliance Checklist: "All outputs conform to IRH-DEF standard format"
"""

import pytest
import json
from datetime import datetime, timezone

from src.output.output_standardization import (
    IRHDEFSchema,
    OutputFormatter,
    ReportGenerator,
    ComplianceChecker,
    MetadataManager,
    OutputFormat,
    ComplianceLevel,
    TheoreticalAnnotation,
    UncertaintyInfo,
    ProvenanceInfo,
    ValidationInfo,
    create_irh_output,
    format_output,
    check_compliance,
)


class TestUncertaintyInfo:
    """Tests for UncertaintyInfo class."""
    
    def test_basic_creation(self):
        """Test basic uncertainty info creation."""
        unc = UncertaintyInfo(value=1.0, uncertainty=0.1)
        assert unc.value == 1.0
        assert unc.uncertainty == 0.1
    
    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        unc = UncertaintyInfo(value=100.0, uncertainty=1.0)
        assert abs(unc.relative_uncertainty - 0.01) < 1e-10
    
    def test_relative_uncertainty_zero_value(self):
        """Test relative uncertainty with zero value."""
        unc = UncertaintyInfo(value=0.0, uncertainty=0.1)
        assert unc.relative_uncertainty == float('inf')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        unc = UncertaintyInfo(value=1.0, uncertainty=0.1, unit="eV")
        d = unc.to_dict()
        assert d["value"] == 1.0
        assert d["uncertainty"] == 0.1
        assert d["unit"] == "eV"
        assert "relative_uncertainty" in d


class TestProvenanceInfo:
    """Tests for ProvenanceInfo class."""
    
    def test_basic_creation(self):
        """Test basic provenance creation."""
        prov = ProvenanceInfo()
        assert prov.timestamp is not None
    
    def test_with_parameters(self):
        """Test provenance with parameters."""
        prov = ProvenanceInfo(
            random_seed=42,
            parameters={"lattice_size": 10}
        )
        assert prov.random_seed == 42
        assert prov.parameters["lattice_size"] == 10
    
    def test_compute_hash(self):
        """Test reproducibility hash computation."""
        prov1 = ProvenanceInfo(random_seed=42, parameters={"a": 1})
        prov2 = ProvenanceInfo(random_seed=42, parameters={"a": 1})
        assert prov1.compute_hash() == prov2.compute_hash()
    
    def test_different_params_different_hash(self):
        """Test that different parameters give different hash."""
        prov1 = ProvenanceInfo(random_seed=42, parameters={"a": 1})
        prov2 = ProvenanceInfo(random_seed=42, parameters={"a": 2})
        assert prov1.compute_hash() != prov2.compute_hash()
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        prov = ProvenanceInfo(random_seed=42)
        d = prov.to_dict()
        assert "timestamp" in d
        assert d["random_seed"] == 42
        assert "reproducibility_hash" in d


class TestValidationInfo:
    """Tests for ValidationInfo class."""
    
    def test_basic_creation(self):
        """Test basic validation info creation."""
        val = ValidationInfo(passed=True)
        assert val.passed is True
    
    def test_with_checks(self):
        """Test validation with checks."""
        val = ValidationInfo(
            passed=True,
            checks_performed=["fixed_point", "eigenvalues"],
            deviations={"lambda": 1e-10}
        )
        assert len(val.checks_performed) == 2
        assert val.deviations["lambda"] == 1e-10
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        val = ValidationInfo(passed=False)
        d = val.to_dict()
        assert d["passed"] is False


class TestIRHDEFSchema:
    """Tests for IRHDEFSchema class."""
    
    def test_basic_creation(self):
        """Test basic schema creation."""
        schema = IRHDEFSchema(computation_type="test")
        assert schema.computation_type == "test"
        assert schema.schema_version == "1.0.0"
    
    def test_add_result(self):
        """Test adding results."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("value", 42)
        assert schema.results["value"] == 42
    
    def test_add_result_with_uncertainty(self):
        """Test adding result with uncertainty."""
        schema = IRHDEFSchema(computation_type="test")
        unc = UncertaintyInfo(value=1.0, uncertainty=0.1)
        schema.add_result("value", 1.0, uncertainty=unc)
        assert "value" in schema.uncertainties
    
    def test_add_result_with_annotation(self):
        """Test adding result with theoretical annotation."""
        schema = IRHDEFSchema(computation_type="test")
        ann = TheoreticalAnnotation(
            equation_number="1.14",
            section="§1.2",
            description="Fixed point"
        )
        schema.add_result("lambda_star", 52.64, annotation=ann)
        assert "lambda_star" in schema.theoretical_context
        assert schema.theoretical_context["lambda_star"]["equation"] == "1.14"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        d = schema.to_dict()
        assert "irh_def_schema" in d
        assert d["irh_def_schema"]["computation_type"] == "test"
        assert d["results"]["x"] == 1.0
    
    def test_to_json(self):
        """Test JSON conversion."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert parsed["results"]["x"] == 1.0


class TestOutputFormatter:
    """Tests for OutputFormatter class."""
    
    @pytest.fixture
    def sample_schema(self):
        """Create a sample schema for testing."""
        schema = IRHDEFSchema(computation_type="fixed_point")
        unc = UncertaintyInfo(value=52.64, uncertainty=0.01)
        ann = TheoreticalAnnotation(
            equation_number="1.14",
            section="§1.2",
            description="Cosmic fixed point"
        )
        schema.add_result("lambda_star", 52.64, uncertainty=unc, annotation=ann)
        return schema
    
    def test_format_json(self, sample_schema):
        """Test JSON formatting."""
        formatter = OutputFormatter(sample_schema)
        output = formatter.format(OutputFormat.JSON)
        parsed = json.loads(output)
        assert "results" in parsed
    
    def test_format_markdown(self, sample_schema):
        """Test Markdown formatting."""
        formatter = OutputFormatter(sample_schema)
        output = formatter.format(OutputFormat.MARKDOWN)
        assert "# IRH Computation Results" in output
        assert "lambda_star" in output
    
    def test_format_latex(self, sample_schema):
        """Test LaTeX formatting."""
        formatter = OutputFormatter(sample_schema)
        output = formatter.format(OutputFormat.LATEX)
        assert "\\begin{table}" in output
        assert "lambda_star" in output
    
    def test_format_html(self, sample_schema):
        """Test HTML formatting."""
        formatter = OutputFormatter(sample_schema)
        output = formatter.format(OutputFormat.HTML)
        assert "<!DOCTYPE html>" in output
        assert "lambda_star" in output
    
    def test_format_plain(self, sample_schema):
        """Test plain text formatting."""
        formatter = OutputFormatter(sample_schema)
        output = formatter.format(OutputFormat.PLAIN)
        assert "IRH COMPUTATION RESULTS" in output
        assert "lambda_star" in output


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    def test_basic_creation(self):
        """Test basic report generator creation."""
        gen = ReportGenerator()
        assert gen.title == "IRH Computation Report"
    
    def test_set_metadata(self):
        """Test setting metadata."""
        gen = ReportGenerator()
        gen.set_metadata(title="Test Report", author="Test Author")
        assert gen.title == "Test Report"
        assert gen.author == "Test Author"
    
    def test_add_section(self):
        """Test adding sections."""
        gen = ReportGenerator()
        schema = IRHDEFSchema(computation_type="test")
        gen.add_section(schema, "Test Section")
        assert len(gen.sections) == 1
    
    def test_generate_markdown(self):
        """Test Markdown report generation."""
        gen = ReportGenerator()
        gen.set_metadata(title="Test Report")
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        gen.add_section(schema, "Results")
        output = gen.generate(OutputFormat.MARKDOWN)
        assert "# Test Report" in output
        assert "Results" in output
    
    def test_generate_json(self):
        """Test JSON report generation."""
        gen = ReportGenerator()
        schema = IRHDEFSchema(computation_type="test")
        gen.add_section(schema)
        output = gen.generate(OutputFormat.JSON)
        parsed = json.loads(output)
        assert "sections" in parsed
    
    def test_generate_html(self):
        """Test HTML report generation."""
        gen = ReportGenerator()
        schema = IRHDEFSchema(computation_type="test")
        gen.add_section(schema)
        output = gen.generate(OutputFormat.HTML)
        assert "<!DOCTYPE html>" in output
    
    def test_generate_latex(self):
        """Test LaTeX report generation."""
        gen = ReportGenerator()
        schema = IRHDEFSchema(computation_type="test")
        gen.add_section(schema)
        output = gen.generate(OutputFormat.LATEX)
        assert "\\documentclass" in output


class TestComplianceChecker:
    """Tests for ComplianceChecker class."""
    
    def test_full_compliance(self):
        """Test full compliance detection."""
        schema = IRHDEFSchema(computation_type="test")
        unc = UncertaintyInfo(value=1.0, uncertainty=0.1)
        ann = TheoreticalAnnotation(
            equation_number="1.1",
            section="§1",
            description="Test"
        )
        schema.add_result("x", 1.0, uncertainty=unc, annotation=ann)
        
        checker = ComplianceChecker()
        level = checker.check(schema)
        assert level == ComplianceLevel.FULL
    
    def test_partial_compliance_no_uncertainty(self):
        """Test partial compliance without uncertainty."""
        schema = IRHDEFSchema(computation_type="test")
        ann = TheoreticalAnnotation(
            equation_number="1.1",
            section="§1",
            description="Test"
        )
        schema.add_result("x", 1.0, annotation=ann)
        
        checker = ComplianceChecker()
        level = checker.check(schema)
        assert level == ComplianceLevel.PARTIAL
    
    def test_minimal_compliance(self):
        """Test minimal compliance."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        
        checker = ComplianceChecker()
        level = checker.check(schema)
        assert level == ComplianceLevel.MINIMAL
    
    def test_invalid_no_computation_type(self):
        """Test invalid when missing computation type."""
        schema = IRHDEFSchema()  # No computation_type
        
        checker = ComplianceChecker()
        level = checker.check(schema)
        assert level == ComplianceLevel.INVALID
    
    def test_get_issues(self):
        """Test getting compliance issues."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        
        checker = ComplianceChecker()
        checker.check(schema)
        issues = checker.get_issues()
        assert len(issues) > 0
    
    def test_validate_theoretical_coverage(self):
        """Test theoretical coverage validation."""
        schema = IRHDEFSchema(computation_type="test")
        ann = TheoreticalAnnotation(
            equation_number="1.14",
            section="§1.2",
            description="Fixed point"
        )
        schema.add_result("lambda", 52.64, annotation=ann)
        
        checker = ComplianceChecker()
        coverage = checker.validate_theoretical_coverage(
            schema, ["1.13", "1.14", "1.16"]
        )
        assert coverage["1.14"] is True
        assert coverage["1.13"] is False


class TestMetadataManager:
    """Tests for MetadataManager class."""
    
    def test_start_session(self):
        """Test starting a session."""
        manager = MetadataManager()
        session_id = manager.start_session(random_seed=42)
        assert len(session_id) == 12
    
    def test_log_computation(self):
        """Test logging computation."""
        manager = MetadataManager()
        manager.start_session()
        manager.log_computation(
            "fixed_point",
            {"method": "newton"},
            {"lambda": 52.64}
        )
        summary = manager.get_session_summary()
        assert summary["computation_count"] == 1
    
    def test_create_provenance(self):
        """Test creating provenance."""
        manager = MetadataManager()
        manager.start_session(random_seed=42)
        prov = manager.create_provenance(lattice_size=10)
        assert prov.random_seed == 42
        assert prov.parameters["lattice_size"] == 10
    
    def test_export_provenance(self):
        """Test exporting provenance."""
        manager = MetadataManager()
        manager.start_session()
        export = manager.export_provenance()
        parsed = json.loads(export)
        assert "session_id" in parsed


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_irh_output(self):
        """Test create_irh_output function."""
        schema = create_irh_output(
            "test",
            {"x": 1.0, "y": 2.0}
        )
        assert schema.computation_type == "test"
        assert schema.results["x"] == 1.0
    
    def test_create_irh_output_with_uncertainties(self):
        """Test create_irh_output with uncertainties."""
        unc = {"x": UncertaintyInfo(value=1.0, uncertainty=0.1)}
        schema = create_irh_output(
            "test",
            {"x": 1.0},
            uncertainties=unc
        )
        assert "x" in schema.uncertainties
    
    def test_format_output(self):
        """Test format_output function."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        
        output = format_output(schema, OutputFormat.JSON)
        parsed = json.loads(output)
        assert "results" in parsed
    
    def test_format_output_string_format(self):
        """Test format_output with string format."""
        schema = IRHDEFSchema(computation_type="test")
        schema.add_result("x", 1.0)
        
        output = format_output(schema, "json")
        parsed = json.loads(output)
        assert "results" in parsed
    
    def test_check_compliance(self):
        """Test check_compliance function."""
        schema = IRHDEFSchema(computation_type="test")
        unc = UncertaintyInfo(value=1.0, uncertainty=0.1)
        ann = TheoreticalAnnotation(
            equation_number="1.1",
            section="§1",
            description="Test"
        )
        schema.add_result("x", 1.0, uncertainty=unc, annotation=ann)
        
        result = check_compliance(schema)
        assert result["is_valid"] is True
        assert result["level"] == "full"


class TestIntegration:
    """Integration tests for output standardization."""
    
    def test_full_workflow(self):
        """Test complete output standardization workflow."""
        # Create metadata manager
        manager = MetadataManager()
        manager.start_session(random_seed=42)
        
        # Create schema with full context
        schema = IRHDEFSchema(computation_type="rg_fixed_point")
        
        unc = UncertaintyInfo(value=52.64, uncertainty=0.01)
        ann = TheoreticalAnnotation(
            equation_number="1.14",
            section="§1.2",
            description="Cosmic fixed point λ̃*"
        )
        schema.add_result("lambda_star", 52.64, uncertainty=unc, annotation=ann)
        
        schema.provenance = manager.create_provenance(method="analytical")
        
        # Check compliance
        compliance = check_compliance(schema)
        assert compliance["is_full"] is True
        
        # Generate outputs in all formats
        formatter = OutputFormatter(schema)
        
        json_out = formatter.format(OutputFormat.JSON)
        assert json.loads(json_out)["results"]["lambda_star"] == 52.64
        
        md_out = formatter.format(OutputFormat.MARKDOWN)
        assert "lambda_star" in md_out
        
        html_out = formatter.format(OutputFormat.HTML)
        assert "<table>" in html_out
        
        # Log computation
        manager.log_computation(
            "fixed_point",
            {"method": "analytical"},
            {"lambda_star": 52.64}
        )
        
        summary = manager.get_session_summary()
        assert summary["computation_count"] == 1
    
    def test_report_with_multiple_sections(self):
        """Test report generation with multiple sections."""
        gen = ReportGenerator()
        gen.set_metadata(
            title="IRH v21.0 Verification Report",
            author="IRH Framework",
            abstract="Complete verification of cGFT fixed point computation."
        )
        
        # Add fixed point section
        fp_schema = IRHDEFSchema(computation_type="fixed_point")
        fp_schema.add_result("lambda_star", 52.64)
        fp_schema.add_result("gamma_star", 105.28)
        gen.add_section(fp_schema, "Fixed Point Values")
        
        # Add validation section
        val_schema = IRHDEFSchema(computation_type="validation")
        val_schema.add_result("convergence", True)
        val_schema.add_result("max_deviation", 1e-10)
        gen.add_section(val_schema, "Validation Results")
        
        # Generate report
        report = gen.generate(OutputFormat.MARKDOWN)
        
        assert "IRH v21.0 Verification Report" in report
        assert "Fixed Point Values" in report
        assert "Validation Results" in report
        assert "lambda_star" in report
        assert "convergence" in report
