# Documentation infrastructure for IRH v21.0
# Provides code↔theory cross-reference generation

from .code_theory_xref import (
    CodeTheoryXRef,
    EquationImplementation,
    ModuleMapping,
    CoverageReport,
    generate_interactive_html,
    generate_markdown_report,
    scan_source_directory,
)

__all__ = [
    "CodeTheoryXRef",
    "EquationImplementation", 
    "ModuleMapping",
    "CoverageReport",
    "generate_interactive_html",
    "generate_markdown_report",
    "scan_source_directory",
]
