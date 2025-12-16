"""
Theoretical Instrumentation Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.6, copilot21promtMAX.md Phase II

This module provides runtime logging infrastructure that emits theoretical
context for every computational operation, establishing bidirectional
traceability between algorithmic primitives and mathematical counterparts.

Key Features:
    - Structured logging with equation references
    - Per-operation theoretical correspondence
    - RG flow real-time narration
    - Verification status reporting

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.6, copilot21promtMAX.md Phase II"


# =============================================================================
# Log Level Definitions
# =============================================================================


class IRHLogLevel(Enum):
    """IRH-specific log levels with theoretical context."""
    INIT = "INIT"           # Initialization and configuration
    EXEC = "EXEC"           # Execution of theoretical operations
    VERIFY = "VERIFY"       # Verification and validation steps
    RG_FLOW = "RG-FLOW"     # RG flow integration status
    RG_STEP = "RG-STEP"     # Individual RG step
    RESULT = "RESULT"       # Final results and predictions
    WARNING = "WARNING"     # Theoretical warnings
    ERROR = "ERROR"         # Errors in computation


# =============================================================================
# Theoretical Context Data Structures
# =============================================================================


@dataclass
class TheoreticalReference:
    """Reference to theoretical manuscript."""
    section: str
    equation: Optional[str] = None
    appendix: Optional[str] = None
    description: str = ""
    
    def __str__(self) -> str:
        parts = [f"IRH21.md {self.section}"]
        if self.equation:
            parts.append(f"Eq. {self.equation}")
        if self.appendix:
            parts.append(f"Appendix {self.appendix}")
        return ", ".join(parts)


@dataclass
class ComputationContext:
    """Context for a computational operation."""
    operation: str
    theoretical_ref: TheoreticalReference
    formula: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    uncertainty: Optional[float] = None
    verification_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'theoretical_reference': str(self.theoretical_ref),
            'formula': self.formula,
            'parameters': self.parameters,
            'result': self.result,
            'uncertainty': self.uncertainty,
            'verification_status': self.verification_status,
        }


# =============================================================================
# IRH Logger Class
# =============================================================================


class IRHLogger:
    """
    Structured logger for IRH computations with theoretical context.
    
    Theoretical Reference:
        copilot21promtMAX.md Phase II
        Every computational operation emits theoretical correspondence.
    """
    
    _instance: Optional['IRHLogger'] = None
    
    def __init__(
        self,
        name: str = "IRH",
        level: int = logging.INFO,
        stream: Any = sys.stdout,
        include_timestamp: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.include_timestamp = include_timestamp
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create handler with custom formatter
        handler = logging.StreamHandler(stream)
        handler.setFormatter(IRHFormatter(include_timestamp=include_timestamp))
        self.logger.addHandler(handler)
        
        # Context stack for nested operations
        self._context_stack: List[ComputationContext] = []
        
        # Log history for analysis
        self._log_history: List[Dict[str, Any]] = []
    
    @classmethod
    def get_instance(cls) -> 'IRHLogger':
        """Get singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def log(
        self,
        level: IRHLogLevel,
        message: str,
        context: Optional[ComputationContext] = None,
        **kwargs
    ):
        """Log message with theoretical context."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'context': context.to_dict() if context else None,
            **kwargs
        }
        
        self._log_history.append(entry)
        
        formatted = f"[{level.value}] {message}"
        if context:
            formatted += f"\n  ├─ Ref: {context.theoretical_ref}"
            if context.formula:
                formatted += f"\n  ├─ Formula: {context.formula}"
            if context.parameters:
                formatted += f"\n  ├─ Params: {context.parameters}"
            if context.result is not None:
                formatted += f"\n  ├─ Result: {context.result}"
            if context.uncertainty is not None:
                formatted += f" ± {context.uncertainty}"
            if context.verification_status:
                formatted += f"\n  └─ Status: {context.verification_status}"
        
        self.logger.info(formatted)
    
    def init(
        self,
        message: str,
        theoretical_ref: Optional[TheoreticalReference] = None,
        **kwargs
    ):
        """Log initialization event."""
        context = None
        if theoretical_ref:
            context = ComputationContext(
                operation="initialization",
                theoretical_ref=theoretical_ref,
                **kwargs
            )
        self.log(IRHLogLevel.INIT, message, context)
    
    def exec(
        self,
        operation: str,
        theoretical_ref: TheoreticalReference,
        formula: str = "",
        **kwargs
    ):
        """Log execution of theoretical operation."""
        context = ComputationContext(
            operation=operation,
            theoretical_ref=theoretical_ref,
            formula=formula,
            **kwargs
        )
        self.log(IRHLogLevel.EXEC, f"Computing {operation}", context)
    
    def verify(
        self,
        test_name: str,
        passed: bool,
        theoretical_ref: TheoreticalReference,
        **kwargs
    ):
        """Log verification result."""
        status = "PASS ✓" if passed else "FAIL ✗"
        context = ComputationContext(
            operation=f"verify_{test_name}",
            theoretical_ref=theoretical_ref,
            verification_status=status,
            **kwargs
        )
        self.log(IRHLogLevel.VERIFY, f"Verification: {test_name}", context)
    
    def rg_flow_start(
        self,
        lambda_0: float,
        gamma_0: float,
        mu_0: float,
        target_lambda: float,
        target_gamma: float,
        target_mu: float
    ):
        """Log start of RG flow integration."""
        self.log(
            IRHLogLevel.RG_FLOW,
            "Commencing integration of Wetterich equation (Eq. 1.12)",
            ComputationContext(
                operation="rg_flow_start",
                theoretical_ref=TheoreticalReference(
                    section="§1.2",
                    equation="1.12",
                    description="Wetterich equation"
                ),
                parameters={
                    'initial': (lambda_0, gamma_0, mu_0),
                    'target': (target_lambda, target_gamma, target_mu)
                }
            )
        )
    
    def rg_step(
        self,
        t: float,
        k: float,
        lambda_k: float,
        gamma_k: float,
        mu_k: float,
        beta_lambda: float,
        beta_gamma: float,
        beta_mu: float
    ):
        """Log individual RG step."""
        self.log(
            IRHLogLevel.RG_STEP,
            f"RG step at t={t:.6f}, k={k:.2e}",
            ComputationContext(
                operation="rg_step",
                theoretical_ref=TheoreticalReference(
                    section="§1.2",
                    equation="1.13",
                    description="β-functions"
                ),
                parameters={
                    'couplings': (lambda_k, gamma_k, mu_k),
                    'betas': (beta_lambda, beta_gamma, beta_mu)
                }
            )
        )
    
    def result(
        self,
        name: str,
        value: Any,
        theoretical_ref: TheoreticalReference,
        uncertainty: Optional[float] = None,
        experimental: Optional[float] = None
    ):
        """Log final result with comparison to experiment."""
        msg = f"Result: {name}"
        params = {'value': value}
        if uncertainty:
            params['uncertainty'] = uncertainty
        if experimental:
            params['experimental'] = experimental
            if value and experimental:
                params['agreement'] = abs(value - experimental) / experimental
        
        context = ComputationContext(
            operation=f"result_{name}",
            theoretical_ref=theoretical_ref,
            result=value,
            uncertainty=uncertainty,
            parameters=params
        )
        self.log(IRHLogLevel.RESULT, msg, context)
    
    @contextmanager
    def operation(
        self,
        name: str,
        theoretical_ref: TheoreticalReference,
        formula: str = ""
    ):
        """Context manager for tracking operation execution."""
        context = ComputationContext(
            operation=name,
            theoretical_ref=theoretical_ref,
            formula=formula
        )
        self._context_stack.append(context)
        self.log(IRHLogLevel.EXEC, f"Begin: {name}", context)
        
        try:
            yield context
        finally:
            self._context_stack.pop()
            self.log(IRHLogLevel.EXEC, f"End: {name}", context)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get log history for analysis."""
        return self._log_history.copy()
    
    def clear_history(self):
        """Clear log history."""
        self._log_history.clear()


class IRHFormatter(logging.Formatter):
    """Custom formatter for IRH logs."""
    
    def __init__(self, include_timestamp: bool = True):
        self.include_timestamp = include_timestamp
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"[{timestamp}] {record.getMessage()}"
        return record.getMessage()


# =============================================================================
# Decorator for Instrumented Functions
# =============================================================================


def instrumented(
    theoretical_ref: TheoreticalReference,
    formula: str = ""
) -> Callable:
    """
    Decorator to add instrumentation to functions.
    
    Usage:
        @instrumented(
            TheoreticalReference(section="§1.1", equation="1.1"),
            formula="S_kin = ∫ φ̄·Δ·φ"
        )
        def compute_kinetic_action(phi):
            ...
    """
    import functools
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = IRHLogger.get_instance()
            with logger.operation(func.__name__, theoretical_ref, formula):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


# =============================================================================
# Module-Level Functions
# =============================================================================


def get_logger() -> IRHLogger:
    """Get the singleton IRH logger instance."""
    return IRHLogger.get_instance()


def configure_logging(
    level: int = logging.INFO,
    stream: Any = sys.stdout,
    include_timestamp: bool = True
):
    """Configure the global IRH logger."""
    IRHLogger._instance = IRHLogger(
        level=level,
        stream=stream,
        include_timestamp=include_timestamp
    )


__all__ = [
    # Enums
    'IRHLogLevel',
    
    # Data classes
    'TheoreticalReference',
    'ComputationContext',
    
    # Logger class
    'IRHLogger',
    
    # Decorator
    'instrumented',
    
    # Functions
    'get_logger',
    'configure_logging',
]
