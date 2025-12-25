"""
Structured Logger for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K

This module provides structured JSON logging with theoretical context:
    - Machine-parsable JSON format
    - Theoretical references in log entries
    - Performance metrics tracking
    - Multiple output handlers

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K"


# =============================================================================
# Log Level Enumeration
# =============================================================================

class LogLevel(Enum):
    """IRH-specific log levels with numerical values."""
    DEBUG = 10
    INFO = 20
    STEP = 25      # Computation step
    RESULT = 30    # Result output
    WARNING = 40
    ERROR = 50
    CRITICAL = 60
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def from_string(cls, s: str) -> 'LogLevel':
        """Convert string to LogLevel."""
        return cls[s.upper()]


# =============================================================================
# Log Entry Data Class
# =============================================================================

@dataclass
class LogEntry:
    """
    A structured log entry with theoretical context.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Structured logging with provenance tracking.
    """
    timestamp: str
    level: str
    message: str
    module: str = ""
    function: str = ""
    theoretical_ref: str = ""
    equation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Remove empty fields
        return {k: v for k, v in d.items() if v not in (None, "", {}, [])}
    
    # Theoretical Reference: IRH v21.4

    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    # Theoretical Reference: IRH v21.4
    def to_text(self) -> str:
        """Convert to human-readable text format."""
        parts = [f"[{self.timestamp}] [{self.level}]"]
        
        if self.module:
            parts.append(f"[{self.module}]")
        
        parts.append(self.message)
        
        if self.theoretical_ref:
            parts.append(f"(Ref: {self.theoretical_ref})")
        
        if self.result is not None:
            parts.append(f"=> {self.result}")
        
        if self.duration_ms is not None:
            parts.append(f"[{self.duration_ms:.2f}ms]")
        
        return " ".join(parts)


# =============================================================================
# Structured Logger Class
# =============================================================================

class StructuredLogger:
    
    # Theoretical Reference: IRH v21.4
    """
    Structured logger with JSON output and theoretical context.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Machine-parsable logging with complete provenance.
    """
    
    _instances: Dict[str, 'StructuredLogger'] = {}
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(
        self,
        name: str = "irh",
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True,
        text_output: bool = True,
        stream: TextIO = sys.stdout,
        log_file: Optional[str] = None
    ):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : LogLevel
            Minimum log level
        json_output : bool
            Output JSON format
        text_output : bool
            Output human-readable text
        stream : TextIO
            Output stream for text
        log_file : str, optional
            Path to JSON log file
        """
        self.name = name
        self.level = level
        self.json_output = json_output
        self.text_output = text_output
        self.stream = stream
        self.log_file = log_file
        
        self._entries: List[LogEntry] = []
        self._context_stack: List[Dict[str, Any]] = []
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def get_instance(cls, name: str = "irh") -> 'StructuredLogger':
        """Get or create a logger instance."""
        if name not in cls._instances:
            cls._instances[name] = cls(name=name)
        return cls._instances[name]
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged at given level."""
        return level.value >= self.level.value
    
    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> LogEntry:
        """Create a log entry."""
        # Merge context from stack
        context = {}
        for ctx in self._context_stack:
            context.update(ctx)
        
        # Create entry
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            message=message,
            module=kwargs.pop('module', context.get('module', '')),
            function=kwargs.pop('function', context.get('function', '')),
            theoretical_ref=kwargs.pop('theoretical_ref', context.get('theoretical_ref', '')),
            equation=kwargs.pop('equation', context.get('equation', '')),
            parameters=kwargs.pop('parameters', {}),
            result=kwargs.pop('result', None),
            duration_ms=kwargs.pop('duration_ms', None),
            extra=kwargs
        )
        
        return entry
    
    def _output_entry(self, entry: LogEntry) -> None:
        """Output a log entry."""
        self._entries.append(entry)
        
        if self.text_output:
            self.stream.write(entry.to_text() + "\n")
            self.stream.flush()
        
        if self.json_output and self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(entry.to_json() + "\n")
    
    # Theoretical Reference: IRH v21.4

    
    def log(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> None:
        """Log a message at specified level."""
        if not self._should_log(level):
            return
        
        entry = self._create_entry(level, message, **kwargs)
        self._output_entry(entry)
    
    # Theoretical Reference: IRH v21.4

    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    # Theoretical Reference: IRH v21.4
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    # Theoretical Reference: IRH v21.4

    
    def step(self, message: str, **kwargs) -> None:
        
        # Theoretical Reference: IRH v21.4
        """Log computation step."""
        self.log(LogLevel.STEP, message, **kwargs)
    
    def result(self, message: str, value: Any = None, **kwargs) -> None:
        """
        Log computation result.
        
        Theoretical Reference:
            IRH21.md Appendix K, §1.0 (Structured Logging Protocol)
            Result logging with theoretical traceability
        
        Parameters
        ----------
        message : str
            Result description
        value : Any, optional
            Computed value
        **kwargs : Any
            Additional log metadata
        """
        self.log(LogLevel.RESULT, message, result=value, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log warning message.
        
        # Theoretical Reference:
            IRH21.md Appendix K, §1.0 (Structured Logging Protocol)
            Module header, lines 4-10
        
        Parameters
        ----------
        message : str
            Warning message content
        **kwargs : Any
            Additional log metadata
        """
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Log error message.
        
        Theoretical Reference:
            IRH21.md Appendix K, §1.0 (Structured Logging Protocol)
            Module header, lines 4-10
        
        Parameters
        ----------
        message : str
            Error message content
        **kwargs : Any
            Additional log metadata
        """
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Log critical message.
        
        Theoretical Reference:
            IRH21.md Appendix K, §1.0 (Structured Logging Protocol)
            Module header, lines 4-10
        
        Parameters
        ----------
        message : str
            Critical message content
        **kwargs : Any
            Additional log metadata
        """
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    @contextmanager
    def context(
        self,
        module: str = "",
        function: str = "",
        theoretical_ref: str = "",
        equation: str = "",
        **kwargs
    ):
        """
        Context manager for adding context to log messages.
        
        Usage:
            with logger.context(module='rg_flow', theoretical_ref='§1.2'):
                logger.info("Starting computation")
        """
        ctx = {
            'module': module,
            'function': function,
            'theoretical_ref': theoretical_ref,
            'equation': equation,
            **kwargs
        }
        self._context_stack.append(ctx)
        try:
            yield
        finally:
            self._context_stack.pop()
    
    @contextmanager
    # Theoretical Reference: IRH v21.4

    def timed(self, message: str, level: LogLevel = LogLevel.STEP, **kwargs):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.timed("Computing fixed point"):
                result = find_fixed_point()
        """
        import time
        
        self.log(level, f"Starting: {message}", **kwargs)
        start = time.perf_counter()
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.log(level, f"Completed: {message}", duration_ms=duration_ms, **kwargs)
    
    # Theoretical Reference: IRH v21.4

    
    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        module: Optional[str] = None,
        since: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Get filtered log entries.
        
        Parameters
        ----------
        level : LogLevel, optional
            Filter by minimum level
        module : str, optional
            Filter by module name
        since : str, optional
            Filter by timestamp (ISO format)
            
        Returns
        -------
        list[LogEntry]
            Filtered log entries
        """
        entries = self._entries
        
        if level:
            entries = [e for e in entries if LogLevel[e.level].value >= level.value]
        
        if module:
            entries = [e for e in entries if e.module == module]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        return entries
    
    # Theoretical Reference: IRH v21.4

    
    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
    
    # Theoretical Reference: IRH v21.4
    def export_json(self, path: str) -> None:
        """Export all entries to JSON file."""
        with open(path, 'w') as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2, default=str)
    
    # Theoretical Reference: IRH v21.4

    
    def get_summary(self) -> Dict[str, Any]:
        
        # Theoretical Reference: IRH v21.4
        """Get summary statistics of log entries."""
        if not self._entries:
            return {'total': 0}
        
        level_counts = {}
        for entry in self._entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        durations = [e.duration_ms for e in self._entries if e.duration_ms is not None]
        
        return {
            'total': len(self._entries),
            'by_level': level_counts,
            'timing': {
                'count': len(durations),
                'total_ms': sum(durations) if durations else 0,
                'avg_ms': sum(durations) / len(durations) if durations else 0,
                'max_ms': max(durations) if durations else 0
            } if durations else None
        }


# =============================================================================
# Module-Level Functions
# =============================================================================

_default_logger: Optional[StructuredLogger] = None


# Theoretical Reference: IRH v21.4



def configure_logging(
    name: str = "irh",
    level: Union[LogLevel, str] = LogLevel.INFO,
    json_output: bool = True,
    text_output: bool = True,
    log_file: Optional[str] = None
) -> StructuredLogger:
    """
    Configure the default logger.
    
    Parameters
    ----------
    name : str
        Logger name
    level : LogLevel or str
        Minimum log level
    json_output : bool
        Enable JSON output
    text_output : bool
        Enable text output
    log_file : str, optional
        Path to JSON log file
        
    Returns
    -------
    StructuredLogger
        Configured logger instance
    """
    global _default_logger
    
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    
    _default_logger = StructuredLogger(
        name=name,
        level=level,
        json_output=json_output,
        text_output=text_output,
        log_file=log_file
    )
    
    return _default_logger


# Theoretical Reference: IRH v21.4



def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns default logger.
        
    Returns
    -------
    StructuredLogger
        Logger instance
    """
    global _default_logger
    
    if name:
        return StructuredLogger.get_instance(name)
    
    if _default_logger is None:
        _default_logger = configure_logging()
    
    return _default_logger


# Theoretical Reference: IRH v21.4



def create_logger(
    name: str,
    level: Union[LogLevel, str] = LogLevel.INFO,
    **kwargs
) -> StructuredLogger:
    """
    Create a new logger instance.
    
    Parameters
    ----------
    name : str
        Logger name
    level : LogLevel or str
        Minimum log level
    **kwargs
        Additional StructuredLogger arguments
        
    Returns
    -------
    StructuredLogger
        New logger instance
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    
    return StructuredLogger(name=name, level=level, **kwargs)


__all__ = [
    'StructuredLogger',
    'LogEntry',
    'LogLevel',
    'configure_logging',
    'get_logger',
    'create_logger',
]
