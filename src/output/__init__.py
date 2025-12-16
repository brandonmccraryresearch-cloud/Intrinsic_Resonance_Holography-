"""
IRH Output Standardization Module.

Phase VIII: Output Standardization from copilot21promtMAX.md verification protocol.

This module provides:
- IRH-DEF schema for standardized output structure
- Multi-format output generation (JSON, Markdown, LaTeX)
- Schema compliance validation
- Comprehensive report generation
- Reproducibility metadata management

Theoretical Reference:
    IRH21.md - Final Compliance Checklist
    All outputs must conform to IRH-DEF standard format
"""

from src.output.output_standardization import (
    IRHDEFSchema,
    OutputFormatter,
    ReportGenerator,
    ComplianceChecker,
    MetadataManager,
)

__all__ = [
    "IRHDEFSchema",
    "OutputFormatter",
    "ReportGenerator",
    "ComplianceChecker",
    "MetadataManager",
]
