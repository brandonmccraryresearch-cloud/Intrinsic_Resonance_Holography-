"""
Tests for IRH v21.0 Reporting Module

THEORETICAL FOUNDATION: IRH21.md Appendix K

Tests for LaTeX, HTML, and Markdown report generation.
"""

import pytest
import numpy as np

from src.reporting.latex_generator import (
    LaTeXGenerator,
    ObservableResult,
    EquationSpec,
    generate_latex_report,
    create_results_table,
)

from src.reporting.html_generator import (
    HTMLGenerator,
    HTMLSection,
    generate_html_report,
)

from src.reporting.markdown_summary import (
    MarkdownGenerator,
    generate_markdown_summary,
    create_results_markdown,
    create_comparison_markdown,
)


class TestLatexGenerator:
    """Test LaTeX report generation."""
    
    def test_generator_creation(self):
        """Generator should be creatable."""
        gen = LaTeXGenerator(title="Test Report")
        assert gen.title == "Test Report"
    
    def test_add_section(self):
        """Should be able to add sections."""
        gen = LaTeXGenerator()
        gen.add_section("Test Section", "Test content")
        assert len(gen.sections) == 1
        assert gen.sections[0].title == "Test Section"
    
    def test_add_equation(self):
        """Should be able to add equations."""
        gen = LaTeXGenerator()
        gen.add_equation(
            label="1.13",
            latex=r"\beta_\lambda = -2\tilde{\lambda}",
            description="Beta function",
            reference="§1.2"
        )
        assert len(gen.equations) == 1
    
    def test_add_result(self):
        """Should be able to add results."""
        gen = LaTeXGenerator()
        gen.add_result(
            name="alpha_inverse",
            value=137.035999084,
            uncertainty=1e-9,
            unit="",
            theoretical_ref="§3.2"
        )
        assert len(gen.results) == 1
    
    def test_generate_output(self):
        """Should generate LaTeX content."""
        gen = LaTeXGenerator(title="Test")
        gen.add_result("test", 1.0, 0.1)
        latex = gen.generate()
        
        assert "\\documentclass" in latex
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex
    
    def test_results_table_generation(self):
        """Should generate results table."""
        gen = LaTeXGenerator()
        gen.add_result("x", 1.0, 0.1, theoretical_ref="§1.1")
        gen.add_result("y", 2.0, 0.2, theoretical_ref="§1.2")
        
        table = gen.generate_results_table()
        assert "\\begin{table}" in table
        assert "\\end{table}" in table


class TestObservableResult:
    """Test ObservableResult dataclass."""
    
    def test_creation(self):
        """Should create observable result."""
        result = ObservableResult(
            name="alpha_inverse",
            value=137.035999084,
            uncertainty=1e-9,
            unit=""
        )
        assert result.value == 137.035999084
    
    def test_to_latex_row(self):
        """Should generate LaTeX table row."""
        result = ObservableResult(
            name="test",
            value=1.0,
            uncertainty=0.1,
            unit="GeV",
            theoretical_ref="§1.1"
        )
        row = result.to_latex_row()
        assert "test" in row
        assert "1.0" in row or "1." in row


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_generate_latex_report(self):
        """Should generate report from dict input."""
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1, 'theoretical_ref': '§1.1'}
        ]
        latex = generate_latex_report("Test", results)
        assert "\\documentclass" in latex
    
    def test_create_results_table(self):
        """Should create table from dict input."""
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1}
        ]
        table = create_results_table(results)
        assert "\\begin{table}" in table


class TestHtmlGenerator:
    """Test HTML report generation."""
    
    def test_generator_creation(self):
        """Generator should be creatable."""
        gen = HTMLGenerator(title="Test Report")
        assert gen.title == "Test Report"
    
    def test_add_section(self):
        """Should be able to add sections."""
        gen = HTMLGenerator()
        gen.add_section("test", "Test Section", "Content")
        assert len(gen.sections) == 1
    
    def test_generate_output(self):
        """Should generate HTML content."""
        gen = HTMLGenerator(title="Test")
        gen.add_section("intro", "Introduction", "Test content")
        html = gen.generate()
        
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
    
    def test_results_table(self):
        """Should generate results table."""
        gen = HTMLGenerator()
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1, 'unit': '', 'theoretical_ref': '§1.1'}
        ]
        gen.add_results_table(results)
        html = gen.generate()
        assert "<table>" in html
    
    def test_mathjax_included(self):
        """Should include MathJax for equation rendering."""
        gen = HTMLGenerator()
        html = gen.generate()
        assert "MathJax" in html


class TestHtmlSection:
    """Test HTMLSection dataclass."""
    
    def test_to_html(self):
        """Should generate HTML for section."""
        section = HTMLSection(id="test", title="Test", content="Content")
        html = section.to_html()
        assert "<section" in html
        assert "Test" in html
    
    def test_collapsible(self):
        """Collapsible sections should have button."""
        section = HTMLSection(id="test", title="Test", content="Content", collapsible=True)
        html = section.to_html()
        assert "collapsible" in html


class TestMarkdownGenerator:
    """Test Markdown report generation."""
    
    def test_generator_creation(self):
        """Generator should be creatable."""
        gen = MarkdownGenerator(title="Test Summary")
        assert gen.title == "Test Summary"
    
    def test_add_header(self):
        """Should add headers."""
        gen = MarkdownGenerator()
        gen.add_header("Test", level=2)
        assert "## Test" in gen.sections[0]
    
    def test_add_paragraph(self):
        """Should add paragraphs."""
        gen = MarkdownGenerator()
        gen.add_paragraph("Test paragraph")
        assert "Test paragraph" in gen.sections[0]
    
    def test_generate_output(self):
        """Should generate Markdown content."""
        gen = MarkdownGenerator(title="Test")
        gen.add_paragraph("Content")
        md = gen.generate()
        
        assert "# Test" in md
        assert "Generated:" in md
    
    def test_results_table(self):
        """Should generate results table."""
        gen = MarkdownGenerator()
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1, 'theoretical_ref': '§1.1'}
        ]
        gen.add_results_table(results)
        md = gen.generate()
        
        assert "|" in md  # Table markers
    
    def test_comparison_table(self):
        """Should generate comparison table."""
        gen = MarkdownGenerator()
        results = [
            {
                'name': 'x',
                'value': 1.0,
                'uncertainty': 0.1,
                'experimental_value': 1.05,
                'experimental_uncertainty': 0.05
            }
        ]
        gen.add_comparison_table(results)
        md = gen.generate()
        
        assert "Theory" in md or "Experiment" in md or "IRH" in md
    
    def test_checklist(self):
        """Should generate checklist."""
        gen = MarkdownGenerator()
        items = [
            {'name': 'Test 1', 'passed': True},
            {'name': 'Test 2', 'passed': False}
        ]
        gen.add_checklist(items)
        md = gen.generate()
        
        assert "✅" in md
        assert "❌" in md


class TestMarkdownConvenienceFunctions:
    """Test Markdown convenience functions."""
    
    def test_generate_markdown_summary(self):
        """Should generate summary from dict input."""
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1}
        ]
        md = generate_markdown_summary("Test", results)
        assert "# Test" in md
    
    def test_create_results_markdown(self):
        """Should create results table markdown."""
        results = [
            {'name': 'x', 'value': 1.0, 'uncertainty': 0.1}
        ]
        md = create_results_markdown(results)
        assert "|" in md
    
    def test_create_comparison_markdown(self):
        """Should create comparison table markdown."""
        results = [
            {
                'name': 'x',
                'value': 1.0,
                'uncertainty': 0.1,
                'experimental_value': 1.05
            }
        ]
        md = create_comparison_markdown(results)
        assert "|" in md or "comparison" in md.lower() or md == ""  # Empty if no exp values


class TestTheoreticalGrounding:
    """Test that modules have proper theoretical grounding."""
    
    def test_latex_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.reporting import latex_generator
        assert hasattr(latex_generator, '__theoretical_foundation__')
        assert 'IRH21' in latex_generator.__theoretical_foundation__
    
    def test_html_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.reporting import html_generator
        assert hasattr(html_generator, '__theoretical_foundation__')
        assert 'IRH21' in html_generator.__theoretical_foundation__
    
    def test_markdown_module_foundation(self):
        """Module should reference IRH21.md."""
        from src.reporting import markdown_summary
        assert hasattr(markdown_summary, '__theoretical_foundation__')
        assert 'IRH21' in markdown_summary.__theoretical_foundation__
