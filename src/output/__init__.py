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
    IRH v21.1 Manuscript (Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices)
    - Intrinsic_Resonance_Holography-v21.1-Part1.md
    - Intrinsic_Resonance_Holography-v21.1-Part2.md
    Final Compliance Checklist: All outputs must conform to IRH-DEF standard format
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
