"""
Transparency Engine for IRH Computational Framework

THEORETICAL FOUNDATION: IRH v21.4 Mandate for Algorithmic Transparency

This module provides comprehensive runtime instrumentation for all IRH computations,
ensuring every calculation can be traced back to specific equations in the IRH v21.4
manuscript with complete provenance tracking.

The Transparency Engine is the cornerstone of IRH's commitment to "zero black boxes" —
every computation must emit:
1. Theoretical reference (manuscript section + equation number)
2. Complete formula with all terms
3. Component-by-component breakdown
4. Uncertainty propagation
5. Validation checks (dimensional consistency, gauge invariance)

Authors: IRH Computational Framework Team
Last Updated: December 2025 (v21.4 correspondence)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

__version__ = "21.4.0"
__theoretical_foundation__ = "IRH v21.4 Algorithmic Transparency Mandate"


# =============================================================================
# Verbosity Levels
# =============================================================================


class VerbosityLevel(Enum):
    """
    Verbosity levels for transparency output.
    
    - SILENT: No output (not recommended, violates transparency)
    - MINIMAL: Only final results and critical errors
    - STANDARD: Results + validation checks
    - DETAILED: Above + step-by-step derivation
    - FULL: Above + all intermediate values and formulas
    """
    SILENT = 0
    MINIMAL = 1
    STANDARD = 2
    DETAILED = 3
    FULL = 4


# Convenience constants
SILENT = VerbosityLevel.SILENT
MINIMAL = VerbosityLevel.MINIMAL
STANDARD = VerbosityLevel.STANDARD
DETAILED = VerbosityLevel.DETAILED
FULL = VerbosityLevel.FULL


# =============================================================================
# Message Types
# =============================================================================


class MessageType(Enum):
    """Types of transparency messages."""
    INFO = "INFO"
    STEP = "STEP"
    FORMULA = "FORMULA"
    VALUE = "VALUE"
    VALIDATION = "VALIDATION"
    WARNING = "WARNING"
    ERROR = "ERROR"
    RESULT = "RESULT"
    PASSED = "PASSED"
    FAILED = "FAILED"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TransparencyMessage:
    """
    Single message in the transparency log.
    
    Attributes
    ----------
    timestamp : float
        Unix timestamp of message
    message_type : MessageType
        Type of message
    content : str
        Main message content
    reference : Optional[str]
        Manuscript reference (e.g., "IRH v21.4 Part 1 §3.2.2, Eq. 3.4")
    formula : Optional[str]
        LaTeX/Unicode formula representation
    variables : Optional[Dict]
        Variable values in formula
    value : Optional[float]
        Computed numerical value
    uncertainty : Optional[float]
        Numerical uncertainty
    metadata : Dict
        Additional metadata
    """
    timestamp: float
    message_type: MessageType
    content: str
    reference: Optional[str] = None
    formula: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'type': self.message_type.value,
            'content': self.content,
            'reference': self.reference,
            'formula': self.formula,
            'variables': self.variables,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'metadata': self.metadata
        }


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a computed result.
    
    Tracks how a result was derived from inputs through theoretical formulas.
    """
    result_name: str
    final_value: float
    uncertainty: float
    theoretical_reference: str
    formula: str
    components: Dict[str, Any]
    input_sources: List[str]
    computational_method: str
    validation_checks: Dict[str, bool]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'result_name': self.result_name,
            'final_value': self.final_value,
            'uncertainty': self.uncertainty,
            'theoretical_reference': self.theoretical_reference,
            'formula': self.formula,
            'components': self.components,
            'input_sources': self.input_sources,
            'computational_method': self.computational_method,
            'validation_checks': self.validation_checks,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


# =============================================================================
# Transparency Engine
# =============================================================================


class TransparencyEngine:
    """
    Runtime transparency and provenance tracking engine.
    
    Ensures all IRH computations are fully traceable to theoretical foundations.
    
    Theoretical Reference:
        IRH v21.4 Executive Summary, Point 4:
        "The HarmonyOptimizer's role is clarified as a tool for certified 
        computational verification of analytical proofs and for the high-precision
        calculation of analytically defined non-perturbative functional integrals,
        not as a black box for tuning parameters."
    
    Parameters
    ----------
    verbosity : VerbosityLevel
        Level of output detail (default: STANDARD)
    output_file : Optional[str]
        Path to output log file (if None, no file output)
    real_time_display : bool
        Whether to print messages in real-time (default: True)
        
    Examples
    --------
    >>> engine = TransparencyEngine(verbosity=FULL)
    >>> engine.info("Computing α⁻¹", reference="IRH v21.4 Part 1 §3.2.2, Eq. 3.4")
    >>> engine.step("Step 1: Computing leading order term")
    >>> engine.formula("α⁻¹ = (4π²γ̃*/λ̃*)", variables={'γ̃*': 105.276, 'λ̃*': 52.638})
    >>> engine.value("α⁻¹_leading", 137.036, uncertainty=1e-6)
    >>> engine.passed("α⁻¹ computation complete")
    """
    
    def __init__(
        self,
        verbosity: VerbosityLevel = STANDARD,
        output_file: Optional[str] = None,
        real_time_display: bool = True
    ):
        self.verbosity = verbosity
        self.output_file = output_file
        self.real_time_display = real_time_display
        
        self.messages: List[TransparencyMessage] = []
        self.provenance_chains: List[ProvenanceChain] = []
        
        self._start_time = time.time()
        self._current_computation = None
        
    # -------------------------------------------------------------------------
    # Message Emission
    # -------------------------------------------------------------------------
    
    def _emit(
        self,
        message_type: MessageType,
        content: str,
        reference: Optional[str] = None,
        formula: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        value: Optional[float] = None,
        uncertainty: Optional[float] = None,
        **metadata
    ):
        """Internal method to emit a transparency message."""
        msg = TransparencyMessage(
            timestamp=time.time(),
            message_type=message_type,
            content=content,
            reference=reference,
            formula=formula,
            variables=variables,
            value=value,
            uncertainty=uncertainty,
            metadata=metadata
        )
        
        self.messages.append(msg)
        
        # Real-time display
        if self.real_time_display and self.verbosity != SILENT:
            self._display_message(msg)
            
        # File output
        if self.output_file:
            self._write_to_file(msg)
    
    def _display_message(self, msg: TransparencyMessage):
        """Display message based on verbosity level."""
        type_symbol = {
            MessageType.INFO: "ℹ️",
            MessageType.STEP: "▶️",
            MessageType.FORMULA: "📐",
            MessageType.VALUE: "🔢",
            MessageType.VALIDATION: "✓",
            MessageType.WARNING: "⚠️",
            MessageType.ERROR: "❌",
            MessageType.RESULT: "🎯",
            MessageType.PASSED: "✅",
            MessageType.FAILED: "🔴"
        }
        
        symbol = type_symbol.get(msg.message_type, "•")
        
        # Determine if message should be displayed at current verbosity
        display_threshold = {
            MessageType.INFO: DETAILED,
            MessageType.STEP: DETAILED,
            MessageType.FORMULA: FULL,
            MessageType.VALUE: FULL,
            MessageType.VALIDATION: STANDARD,
            MessageType.WARNING: MINIMAL,
            MessageType.ERROR: MINIMAL,
            MessageType.RESULT: STANDARD,
            MessageType.PASSED: STANDARD,
            MessageType.FAILED: MINIMAL
        }
        
        if self.verbosity.value >= display_threshold.get(msg.message_type, STANDARD).value:
            print(f"{symbol} {msg.content}")
            
            if msg.reference and self.verbosity.value >= DETAILED.value:
                print(f"   └─ Reference: {msg.reference}")
                
            if msg.formula and self.verbosity.value >= FULL.value:
                print(f"   └─ Formula: {msg.formula}")
                if msg.variables:
                    for var, val in msg.variables.items():
                        print(f"      • {var} = {val}")
                        
            if msg.value is not None and self.verbosity.value >= FULL.value:
                if msg.uncertainty is not None:
                    print(f"   └─ Value: {msg.value} ± {msg.uncertainty}")
                else:
                    print(f"   └─ Value: {msg.value}")
    
    def _write_to_file(self, msg: TransparencyMessage):
        """Append message to log file."""
        try:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(msg.to_dict()) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to write to log file: {e}")
    
    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    
    def info(self, content: str, reference: Optional[str] = None, **metadata):
        """
        Emit informational message about computation context.
        
        Parameters
        ----------
        content : str
            Informational message
        reference : Optional[str]
            Manuscript reference (e.g., "IRH v21.4 Part 1 §3.2.2")
        """
        self._emit(MessageType.INFO, content, reference=reference, **metadata)
    
    def step(self, content: str, **metadata):
        """
        Emit computation step message.
        
        Parameters
        ----------
        content : str
            Description of computational step
        """
        self._emit(MessageType.STEP, content, **metadata)
    
    def formula(
        self,
        formula: str,
        variables: Optional[Dict[str, Any]] = None,
        reference: Optional[str] = None,
        **metadata
    ):
        """
        Emit formula with variable values.
        
        Parameters
        ----------
        formula : str
            Mathematical formula (LaTeX or Unicode notation)
        variables : Optional[Dict]
            Variable names and their values
        reference : Optional[str]
            Manuscript equation reference
        """
        self._emit(
            MessageType.FORMULA,
            f"Formula: {formula}",
            formula=formula,
            variables=variables,
            reference=reference,
            **metadata
        )
    
    def value(
        self,
        name: str,
        value: float,
        uncertainty: Optional[float] = None,
        **metadata
    ):
        """
        Emit computed numerical value.
        
        Parameters
        ----------
        name : str
            Name of computed quantity
        value : float
            Numerical value
        uncertainty : Optional[float]
            Numerical uncertainty (1-sigma)
        """
        if uncertainty is not None:
            content = f"{name} = {value} ± {uncertainty}"
        else:
            content = f"{name} = {value}"
            
        self._emit(
            MessageType.VALUE,
            content,
            value=value,
            uncertainty=uncertainty,
            **metadata
        )
    
    def validate(
        self,
        check_name: str,
        passed: bool,
        details: Optional[str] = None,
        **metadata
    ):
        """
        Emit validation check result.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Validation protocol for computational integrity
        
        Parameters
        ----------
        check_name : str
            Name of validation check
        passed : bool
            Whether check passed
        details : Optional[str]
            Additional details about check
        """
        status = "PASSED" if passed else "FAILED"
        content = f"{check_name}: {status}"
        if details:
            content += f" ({details})"
            
        msg_type = MessageType.VALIDATION if passed else MessageType.FAILED
        self._emit(msg_type, content, **metadata)
    
    def warning(self, content: str, reference: Optional[str] = None, **metadata):
        """
        Emit warning message.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Module header, lines 4-16
        
        Parameters
        ----------
        content : str
            Warning message content
        reference : Optional[str]
            Manuscript reference for the warning context
        **metadata : Any
            Additional metadata
        """
        self._emit(MessageType.WARNING, content, reference=reference, **metadata)
    
    def error(self, content: str, reference: Optional[str] = None, **metadata):
        """
        Emit error message.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Module header, lines 4-16
        
        Parameters
        ----------
        content : str
            Error message content
        reference : Optional[str]
            Manuscript reference for the error context
        **metadata : Any
            Additional metadata
        """
        self._emit(MessageType.ERROR, content, reference=reference, **metadata)
    
    def result(
        self,
        name: str,
        value: float,
        uncertainty: Optional[float] = None,
        components: Optional[Dict[str, Any]] = None,
        reference: Optional[str] = None,
        **metadata
    ):
        """
        Emit final result with full context.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Result reporting with provenance tracking
        
        Parameters
        ----------
        name : str
            Name of result
        value : float
            Final value
        uncertainty : Optional[float]
            Uncertainty bound
        components : Optional[Dict]
            Breakdown of result components
        reference : Optional[str]
            Theoretical reference
        """
        if uncertainty is not None:
            content = f"RESULT: {name} = {value} ± {uncertainty}"
        else:
            content = f"RESULT: {name} = {value}"
            
        self._emit(
            MessageType.RESULT,
            content,
            value=value,
            uncertainty=uncertainty,
            reference=reference,
            components=components,
            **metadata
        )
    
    def passed(self, content: str, reference: Optional[str] = None, **metadata):
        """
        Emit success message.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Module header, lines 4-16
        
        Parameters
        ----------
        content : str
            Success message content
        reference : Optional[str]
            Manuscript reference for the validation context
        **metadata : Any
            Additional metadata
        """
        self._emit(MessageType.PASSED, content, reference=reference, **metadata)
    
    def failed(self, content: str, reference: Optional[str] = None, **metadata):
        """
        Emit failure message.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Module header, lines 4-16
        
        Parameters
        ----------
        content : str
            Failure message content
        reference : Optional[str]
            Manuscript reference for the validation context
        **metadata : Any
            Additional metadata
        """
        self._emit(MessageType.FAILED, content, reference=reference, **metadata)
    
    # -------------------------------------------------------------------------
    # Provenance Tracking
    # -------------------------------------------------------------------------
    
    def add_provenance(
        self,
        result_name: str,
        final_value: float,
        uncertainty: float,
        theoretical_reference: str,
        formula: str,
        components: Dict[str, Any],
        input_sources: List[str],
        computational_method: str,
        validation_checks: Dict[str, bool]
    ) -> ProvenanceChain:
        """
        Add complete provenance chain for a result.
        
        Theoretical Reference:
            IRH v21.4 Algorithmic Transparency Mandate §1.0
            Provenance tracking for computational reproducibility
        
        This creates a permanent record of how a result was derived.
        
        Parameters
        ----------
        result_name : str
            Name of the result
        final_value : float
            Computed value
        uncertainty : float
            Uncertainty bound
        theoretical_reference : str
            Manuscript citation
        formula : str
            Mathematical formula used
        components : Dict[str, Any]
            Result components
        input_sources : List[str]
            Data sources
        computational_method : str
            Algorithm description
        validation_checks : Dict[str, bool]
            Validation results
        
        Returns
        -------
        ProvenanceChain
            Complete provenance record
        """
        chain = ProvenanceChain(
            result_name=result_name,
            final_value=final_value,
            uncertainty=uncertainty,
            theoretical_reference=theoretical_reference,
            formula=formula,
            components=components,
            input_sources=input_sources,
            computational_method=computational_method,
            validation_checks=validation_checks
        )
        
        self.provenance_chains.append(chain)
        return chain
    
    def get_provenance(self, result_name: Optional[str] = None) -> List[ProvenanceChain]:
        """
        Retrieve provenance chains.
        
        Parameters
        ----------
        result_name : Optional[str]
            If provided, filter to chains for this result
            
        Returns
        -------
        List[ProvenanceChain]
            Matching provenance chains
        """
        if result_name is None:
            return self.provenance_chains
        else:
            return [c for c in self.provenance_chains if c.result_name == result_name]
    
    # -------------------------------------------------------------------------
    # Export & Display
    # -------------------------------------------------------------------------
    
    def export(self, format: str = 'dict') -> Union[Dict, str]:
        """
        Export complete log.
        
        Parameters
        ----------
        format : str
            Export format: 'dict', 'json', or 'text'
            
        Returns
        -------
        Union[Dict, str]
            Exported log in requested format
        """
        if format == 'dict':
            return {
                'messages': [msg.to_dict() for msg in self.messages],
                'provenance_chains': [chain.to_dict() for chain in self.provenance_chains],
                'metadata': {
                    'start_time': self._start_time,
                    'duration': time.time() - self._start_time,
                    'verbosity': self.verbosity.name
                }
            }
        elif format == 'json':
            return json.dumps(self.export('dict'), indent=2)
        elif format == 'text':
            lines = []
            lines.append("="*70)
            lines.append("TRANSPARENCY LOG")
            lines.append("="*70)
            for msg in self.messages:
                lines.append(f"[{msg.message_type.value}] {msg.content}")
                if msg.reference:
                    lines.append(f"  Reference: {msg.reference}")
                if msg.formula:
                    lines.append(f"  Formula: {msg.formula}")
            lines.append("="*70)
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def display_provenance(self, chain: ProvenanceChain):
        """Display provenance chain in human-readable format."""
        print("\n" + "="*70)
        print(f"PROVENANCE: {chain.result_name}")
        print("="*70)
        print(f"Value: {chain.final_value} ± {chain.uncertainty}")
        print(f"Reference: {chain.theoretical_reference}")
        print(f"Formula: {chain.formula}")
        print(f"\nComponents:")
        for name, value in chain.components.items():
            print(f"  • {name}: {value}")
        print(f"\nInput Sources:")
        for source in chain.input_sources:
            print(f"  • {source}")
        print(f"\nMethod: {chain.computational_method}")
        print(f"\nValidation Checks:")
        for check, passed in chain.validation_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        print("="*70 + "\n")
    
    def display_components(self, components: Dict[str, Any]):
        """Display component breakdown."""
        print("\n" + "="*70)
        print("COMPONENT BREAKDOWN")
        print("="*70)
        for name, value in components.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value}")
            else:
                print(f"  {name}: {value}")
        print("="*70 + "\n")
    
    def validate_result(
        self,
        result: Any,
        experimental_value: Optional[float] = None,
        tolerance: float = 1e-6
    ):
        """
        Validate result against experimental value if available.
        
        Parameters
        ----------
        result : Any
            Result object with .value attribute
        experimental_value : Optional[float]
            Known experimental value
        tolerance : float
            Acceptable deviation
        """
        if experimental_value is not None and hasattr(result, 'value'):
            deviation = abs(result.value - experimental_value)
            passed = deviation <= tolerance
            
            self.validate(
                "Experimental Agreement",
                passed,
                details=f"Δ = {deviation:.2e}, tolerance = {tolerance:.2e}"
            )
            
            if hasattr(result, 'uncertainty'):
                sigma_dev = deviation / result.uncertainty if result.uncertainty > 0 else float('inf')
                print(f"  Agreement: {sigma_dev:.2f}σ")
    
    def summary(self):
        """Display summary statistics."""
        n_messages = len(self.messages)
        n_provenance = len(self.provenance_chains)
        duration = time.time() - self._start_time
        
        print("\n" + "="*70)
        print("TRANSPARENCY ENGINE SUMMARY")
        print("="*70)
        print(f"Total Messages: {n_messages}")
        print(f"Provenance Chains: {n_provenance}")
        print(f"Duration: {duration:.2f}s")
        print(f"Verbosity: {self.verbosity.name}")
        print("="*70 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_engine(
    verbosity: Union[VerbosityLevel, str] = STANDARD,
    output_file: Optional[str] = None
) -> TransparencyEngine:
    """
    Create a TransparencyEngine with convenient string-based verbosity.
    
    Parameters
    ----------
    verbosity : Union[VerbosityLevel, str]
        Verbosity level: SILENT, MINIMAL, STANDARD, DETAILED, or FULL
    output_file : Optional[str]
        Path to log file
        
    Returns
    -------
    TransparencyEngine
        Configured engine
    """
    if isinstance(verbosity, str):
        verbosity = VerbosityLevel[verbosity.upper()]
        
    return TransparencyEngine(verbosity=verbosity, output_file=output_file)
