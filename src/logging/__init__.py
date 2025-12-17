"""
IRH v21.0 Advanced Logging System

THEORETICAL FOUNDATION: IRH21.md Appendix K, docs/ROADMAP.md

This module provides comprehensive logging capabilities:
    - Structured JSON logging for machine parsing
    - Provenance tracking for reproducibility
    - Log analysis and aggregation tools
    - Performance metrics integration

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K, docs/ROADMAP.md"

from .structured_logger import (
    StructuredLogger,
    create_logger,
    get_logger,
    LogLevel,
    LogEntry,
    configure_logging,
)

from .provenance import (
    ProvenanceTracker,
    ComputationRecord,
    create_provenance_tracker,
    get_provenance_tracker,
)

__all__ = [
    # Structured Logger
    'StructuredLogger',
    'create_logger',
    'get_logger',
    'LogLevel',
    'LogEntry',
    'configure_logging',
    
    # Provenance
    'ProvenanceTracker',
    'ComputationRecord',
    'create_provenance_tracker',
    'get_provenance_tracker',
]
