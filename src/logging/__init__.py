"""
IRH v21.4 Advanced Logging and Transparency System

THEORETICAL FOUNDATION: IRH v21.4 Algorithmic Transparency Mandate

This module provides comprehensive logging and transparency capabilities:
    - TransparencyEngine: Runtime instrumentation for all computations (PRIMARY)
    - Structured JSON logging for machine parsing
    - Provenance tracking for reproducibility
    - Log analysis and aggregation tools
    - Performance metrics integration

The TransparencyEngine is the cornerstone of IRH v21.4's "zero black boxes" commitment.

Authors: IRH Computational Framework Team
Last Updated: December 2025 (v21.4 correspondence)
"""

from __future__ import annotations

__version__ = "21.4.0"
__theoretical_foundation__ = "IRH v21.4 Algorithmic Transparency Mandate"

# Transparency Engine (PRIMARY INTERFACE for v21.4)
from .transparency_engine import (
    TransparencyEngine,
    TransparencyMessage,
    ProvenanceChain,
    VerbosityLevel,
    MessageType,
    create_engine,
    SILENT,
    MINIMAL,
    STANDARD,
    DETAILED,
    FULL
)

# Legacy structured logging (still supported)
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
    # Transparency Engine (v21.4 PRIMARY)
    'TransparencyEngine',
    'TransparencyMessage',
    'ProvenanceChain',
    'VerbosityLevel',
    'MessageType',
    'create_engine',
    'SILENT',
    'MINIMAL',
    'STANDARD',
    'DETAILED',
    'FULL',
    
    # Structured Logger (legacy)
    'StructuredLogger',
    'create_logger',
    'get_logger',
    'LogLevel',
    'LogEntry',
    'configure_logging',
    
    # Provenance (legacy)
    'ProvenanceTracker',
    'ComputationRecord',
    'create_provenance_tracker',
    'get_provenance_tracker',
]
