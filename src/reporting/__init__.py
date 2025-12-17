"""
IRH v21.0 Report Generation System

THEORETICAL FOUNDATION: IRH21.md Appendix K, docs/ROADMAP.md

This module provides comprehensive report generation capabilities:
    - LaTeX report generation with equation rendering
    - HTML interactive reports with MathJax
    - Markdown summaries for quick reference
    - Experimental comparison tables

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K, docs/ROADMAP.md"

from .latex_generator import (
    LaTeXGenerator,
    generate_latex_report,
    create_equation_section,
    create_results_table,
)

from .html_generator import (
    HTMLGenerator,
    generate_html_report,
    create_interactive_section,
)

from .markdown_summary import (
    MarkdownGenerator,
    generate_markdown_summary,
    create_results_markdown,
    create_comparison_markdown,
)

__all__ = [
    # LaTeX
    'LaTeXGenerator',
    'generate_latex_report',
    'create_equation_section',
    'create_results_table',
    
    # HTML
    'HTMLGenerator',
    'generate_html_report',
    'create_interactive_section',
    
    # Markdown
    'MarkdownGenerator',
    'generate_markdown_summary',
    'create_results_markdown',
    'create_comparison_markdown',
]
