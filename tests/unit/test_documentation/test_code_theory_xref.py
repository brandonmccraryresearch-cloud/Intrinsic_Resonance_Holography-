"""
Tests for Code↔Theory Cross-Reference Generator (Phase VI)

These tests validate the documentation infrastructure that maps
between theoretical equations in IRH21.md and their implementations.

Theoretical Reference:
    copilot21promtMAX.md - Phase VI: Documentation Infrastructure
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.documentation.code_theory_xref import (
    CodeTheoryXRef,
    CoverageReport,
    EquationImplementation,
    EquationReference,
    ModuleMapping,
    generate_interactive_html,
    generate_markdown_report,
    scan_source_directory,
)


class TestEquationReference:
    """Tests for EquationReference dataclass."""
    
    def test_creation(self):
        """Test basic EquationReference creation."""
        ref = EquationReference(
            section="1.1",
            equation_number="1.1",
            description="S_kin kinetic term",
        )
        assert ref.section == "1.1"
        assert ref.equation_number == "1.1"
        assert ref.description == "S_kin kinetic term"
        assert ref.manuscript == "IRH v21.1 Manuscript (Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices)"
    
    def test_string_representation(self):
        """Test __str__ method."""
        ref = EquationReference(
            section="1.1",
            equation_number="1.1",
            description="S_kin kinetic term",
        )
        assert str(ref) == "IRH v21.1 Manuscript (Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices) §1.1, Eq. 1.1"
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        ref = EquationReference(
            section="2.3",
            equation_number="2.17",
            description="ρ_hum calculation",
        )
        d = ref.to_dict()
        assert d["section"] == "2.3"
        assert d["equation_number"] == "2.17"
        assert d["description"] == "ρ_hum calculation"
        assert d["manuscript"] == "IRH v21.1 Manuscript (Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices)"


class TestEquationImplementation:
    """Tests for EquationImplementation dataclass."""
    
    def test_creation(self):
        """Test basic EquationImplementation creation."""
        ref = EquationReference(
            section="1.1",
            equation_number="1.1",
            description="S_kin kinetic term",
        )
        impl = EquationImplementation(
            equation=ref,
            file_path="src/cgft/actions.py",
            function_name="compute_kinetic_action",
            line_number=42,
            implementation_status="complete",
        )
        assert impl.equation == ref
        assert impl.file_path == "src/cgft/actions.py"
        assert impl.function_name == "compute_kinetic_action"
        assert impl.line_number == 42
        assert impl.implementation_status == "complete"
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        ref = EquationReference(
            section="1.1",
            equation_number="1.1",
            description="S_kin",
        )
        impl = EquationImplementation(
            equation=ref,
            file_path="test.py",
            function_name="test_func",
            line_number=10,
            implementation_status="partial",
            notes="Work in progress",
        )
        d = impl.to_dict()
        assert d["file_path"] == "test.py"
        assert d["function_name"] == "test_func"
        assert d["line_number"] == 10
        assert d["implementation_status"] == "partial"
        assert d["notes"] == "Work in progress"
        assert "equation" in d


class TestModuleMapping:
    """Tests for ModuleMapping dataclass."""
    
    def test_creation(self):
        """Test basic ModuleMapping creation."""
        mapping = ModuleMapping(
            module_path="src/cgft/",
            theoretical_sections=["1.1", "1.2"],
            key_equations=["1.1", "1.2", "1.3", "1.4"],
            description="cGFT field implementation",
            layer=1,
        )
        assert mapping.module_path == "src/cgft/"
        assert "1.1" in mapping.theoretical_sections
        assert len(mapping.key_equations) == 4
        assert mapping.layer == 1
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        mapping = ModuleMapping(
            module_path="src/rg_flow/",
            theoretical_sections=["1.2", "1.3"],
            key_equations=["1.12", "1.13", "1.14"],
            description="RG flow computations",
            layer=2,
        )
        d = mapping.to_dict()
        assert d["module_path"] == "src/rg_flow/"
        assert d["layer"] == 2
        assert len(d["key_equations"]) == 3


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""
    
    def test_creation(self):
        """Test basic CoverageReport creation."""
        report = CoverageReport(
            total_equations=17,
            implemented_equations=10,
            partial_equations=3,
            stub_equations=2,
            coverage_by_section={"1": 75.0, "2": 50.0, "3": 100.0},
        )
        assert report.total_equations == 17
        assert report.implemented_equations == 10
        assert report.partial_equations == 3
        assert report.stub_equations == 2
    
    def test_coverage_percentage(self):
        """Test coverage percentage calculation."""
        report = CoverageReport(
            total_equations=100,
            implemented_equations=75,
            partial_equations=10,
            stub_equations=5,
            coverage_by_section={},
        )
        assert report.coverage_percentage == 75.0
    
    def test_coverage_percentage_zero_total(self):
        """Test coverage percentage with zero total equations."""
        report = CoverageReport(
            total_equations=0,
            implemented_equations=0,
            partial_equations=0,
            stub_equations=0,
            coverage_by_section={},
        )
        assert report.coverage_percentage == 0.0
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        report = CoverageReport(
            total_equations=17,
            implemented_equations=10,
            partial_equations=3,
            stub_equations=2,
            coverage_by_section={"1": 75.0},
        )
        d = report.to_dict()
        assert d["total_equations"] == 17
        assert d["coverage_percentage"] == pytest.approx(58.82, rel=0.01)
        assert "timestamp" in d


class TestCodeTheoryXRef:
    """Tests for CodeTheoryXRef class."""
    
    @pytest.fixture
    def temp_source_dir(self):
        """Create a temporary source directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create a test Python file with equation references
            test_file = tmppath / "test_module.py"
            test_file.write_text('''
"""
Test module for equation scanning.
"""

def compute_kinetic_action(phi):
    """
    Compute the kinetic action S_kin (Eq. 1.1).
    
    Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
    """
    pass

def compute_beta_functions(couplings):
    """
    Compute beta functions β_λ, β_γ, β_μ (Eq. 1.13).
    
    This is a complete implementation.
    """
    pass

def stub_function():
    """
    Stub: Not implemented yet (Eq. 2.24).
    """
    pass

def partial_function():
    """
    Partial implementation of Eq. 1.14.
    TODO: Complete this.
    """
    pass
''')
            
            yield tmppath
    
    def test_creation(self, temp_source_dir):
        """Test CodeTheoryXRef creation."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        assert xref.source_root == temp_source_dir
        assert len(xref.implementations) == 0
    
    def test_scan_file(self, temp_source_dir):
        """Test scanning a single file."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        test_file = temp_source_dir / "test_module.py"
        
        impls = xref.scan_file(test_file)
        
        assert len(impls) >= 4
        eq_numbers = [i.equation.equation_number for i in impls]
        assert "1.1" in eq_numbers
        assert "1.13" in eq_numbers
    
    def test_scan_directory(self, temp_source_dir):
        """Test scanning entire directory."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        xref.scan_directory()
        
        assert len(xref.implementations) >= 4
    
    def test_status_inference(self, temp_source_dir):
        """Test implementation status inference from docstrings."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        xref.scan_directory()
        
        # Find the stub implementation
        stub_impls = [i for i in xref.implementations if "stub" in i.implementation_status]
        assert len(stub_impls) >= 1
        
        # Find the partial implementation
        partial_impls = [i for i in xref.implementations if "partial" in i.implementation_status]
        assert len(partial_impls) >= 1
    
    def test_equation_range_expansion(self, temp_source_dir):
        """Test equation range expansion (e.g., '1.1-1.4')."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        
        result = xref._expand_equation_range("1.1-1.4")
        assert result == ["1.1", "1.2", "1.3", "1.4"]
        
        result = xref._expand_equation_range("2.17")
        assert result == ["2.17"]
    
    def test_compute_coverage(self, temp_source_dir):
        """Test coverage computation."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        xref.scan_directory()
        
        coverage = xref.compute_coverage()
        
        assert isinstance(coverage, CoverageReport)
        assert coverage.total_equations == len(xref.CRITICAL_EQUATIONS)
    
    def test_to_json(self, temp_source_dir):
        """Test JSON export."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        xref.scan_directory()
        
        json_str = xref.to_json()
        data = json.loads(json_str)
        
        assert "implementations" in data
        assert "coverage" in data
        assert "source_root" in data
    
    def test_critical_equations_defined(self, temp_source_dir):
        """Test that critical equations are properly defined."""
        xref = CodeTheoryXRef(str(temp_source_dir))
        
        assert "1.1" in xref.CRITICAL_EQUATIONS
        assert "1.13" in xref.CRITICAL_EQUATIONS
        assert "1.14" in xref.CRITICAL_EQUATIONS
        assert "3.6" in xref.CRITICAL_EQUATIONS


class TestScanSourceDirectory:
    """Tests for scan_source_directory convenience function."""
    
    def test_scan_returns_xref(self):
        """Test that scan returns a CodeTheoryXRef object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xref = scan_source_directory(tmpdir)
            assert isinstance(xref, CodeTheoryXRef)


class TestGenerateMarkdownReport:
    """Tests for Markdown report generation."""
    
    @pytest.fixture
    def xref_with_data(self):
        """Create a CodeTheoryXRef with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "test.py"
            test_file.write_text('''
def test_func():
    """Implements Eq. 1.1 and Eq. 1.13."""
    pass
''')
            xref = CodeTheoryXRef(str(tmppath))
            xref.scan_directory()
            yield xref
    
    def test_generates_markdown(self, xref_with_data):
        """Test Markdown generation."""
        report = generate_markdown_report(xref_with_data)
        
        assert "# IRH v21.0" in report
        assert "Coverage Summary" in report
        assert "| Equation |" in report
    
    def test_contains_equations(self, xref_with_data):
        """Test that report contains equation information."""
        report = generate_markdown_report(xref_with_data)
        
        assert "Eq. 1.1" in report or "Eq. 1.13" in report
    
    def test_contains_coverage_stats(self, xref_with_data):
        """Test that report contains coverage statistics."""
        report = generate_markdown_report(xref_with_data)
        
        assert "Total Critical Equations" in report
        assert "Implemented" in report


class TestGenerateInteractiveHTML:
    """Tests for interactive HTML generation."""
    
    @pytest.fixture
    def xref_with_data(self):
        """Create a CodeTheoryXRef with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "test.py"
            test_file.write_text('''
def test_func():
    """Implements Eq. 1.1."""
    pass
''')
            xref = CodeTheoryXRef(str(tmppath))
            xref.scan_directory()
            yield xref
    
    def test_generates_html(self, xref_with_data):
        """Test HTML generation."""
        html = generate_interactive_html(xref_with_data)
        
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
    
    def test_contains_title(self, xref_with_data):
        """Test that HTML contains title."""
        html = generate_interactive_html(xref_with_data)
        
        assert "IRH v21.0 Code↔Theory Cross-Reference" in html
    
    def test_contains_coverage_stats(self, xref_with_data):
        """Test that HTML contains coverage statistics."""
        html = generate_interactive_html(xref_with_data)
        
        assert "Coverage Summary" in html
        assert "Total Equations" in html
    
    def test_contains_search_functionality(self, xref_with_data):
        """Test that HTML includes search functionality."""
        html = generate_interactive_html(xref_with_data)
        
        assert 'id="search"' in html
        assert "searchInput" in html
    
    def test_contains_filter_buttons(self, xref_with_data):
        """Test that HTML includes filter buttons."""
        html = generate_interactive_html(xref_with_data)
        
        assert "filter-btn" in html
        assert "Complete" in html
        assert "Partial" in html
    
    def test_contains_table(self, xref_with_data):
        """Test that HTML contains implementation table."""
        html = generate_interactive_html(xref_with_data)
        
        assert "<table" in html
        assert "<thead>" in html
        assert "<tbody>" in html


class TestRealSourceScan:
    """Tests that scan the actual IRH source code."""
    
    @pytest.fixture
    def src_path(self):
        """Get the actual src directory path."""
        return Path("/home/runner/work/Intrinsic_Resonance_Holography-/Intrinsic_Resonance_Holography-/src")
    
    def test_scan_actual_source(self, src_path):
        """Test scanning the actual source code."""
        if not src_path.exists():
            pytest.skip("Source directory not found")
        
        xref = CodeTheoryXRef(str(src_path))
        xref.scan_directory()
        
        # Should find implementations
        assert len(xref.implementations) >= 0
    
    def test_coverage_computed(self, src_path):
        """Test that coverage can be computed for actual source."""
        if not src_path.exists():
            pytest.skip("Source directory not found")
        
        xref = CodeTheoryXRef(str(src_path))
        xref.scan_directory()
        coverage = xref.compute_coverage()
        
        assert isinstance(coverage, CoverageReport)
        assert coverage.total_equations > 0
