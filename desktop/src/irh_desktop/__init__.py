"""
IRH Desktop Application

Intrinsic Resonance Holography v21.0 Desktop Interface

A feature-rich desktop application providing:
- Transparent, verbose computation output
- Interactive visualization of physics derivations
- Auto-update system for IRH engine
- Customizable configuration profiles
- Plugin system for extensions

Theoretical Foundation:
    IRH21.md - Complete unified theory implementation

Author: Brandon D. McCrary
License: MIT
"""

__version__ = "21.0.0"
__author__ = "Brandon D. McCrary"

from irh_desktop.app import IRHDesktopApp
from irh_desktop.core.engine_manager import EngineManager
from irh_desktop.core.config_manager import ConfigManager
from irh_desktop.core.computation_runner import (
    ComputationRunner,
    ComputationParameters,
    ComputationResult,
    ComputationType,
)
from irh_desktop.core.job_queue import JobQueueManager, JobPriority
from irh_desktop.core.result_exporter import ResultExporter, export_results
from irh_desktop.transparency.engine import TransparencyEngine

__all__ = [
    # Main app
    "IRHDesktopApp",
    # Core services
    "EngineManager",
    "ConfigManager",
    "ComputationRunner",
    "ComputationParameters",
    "ComputationResult",
    "ComputationType",
    "JobQueueManager",
    "JobPriority",
    "ResultExporter",
    "export_results",
    # Transparency
    "TransparencyEngine",
    # Version
    "__version__",
]
