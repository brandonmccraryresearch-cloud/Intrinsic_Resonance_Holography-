"""
IRH Desktop - Core Services Package

This package provides core services for the IRH Desktop application:
- Engine Manager: Manages IRH engine lifecycle
- Config Manager: Handles configuration files
- Computation Runner: Executes IRH computations
- Job Queue: Manages computation job queue
- Result Exporter: Exports results in various formats

Author: Brandon D. McCrary
"""

from irh_desktop.core.engine_manager import EngineManager, EngineInfo, UpdateInfo
from irh_desktop.core.config_manager import ConfigManager
from irh_desktop.core.computation_runner import (
    ComputationRunner,
    ComputationParameters,
    ComputationResult,
    ComputationJob,
    ComputationType,
    ComputationStatus,
    COMPUTATION_INFO,
    create_computation_runner,
)
from irh_desktop.core.job_queue import (
    JobQueueManager,
    JobPriority,
    JobHistory,
    create_job_queue,
)
from irh_desktop.core.result_exporter import (
    ResultExporter,
    ExportOptions,
    export_results,
)

__all__ = [
    # Engine Manager
    "EngineManager",
    "EngineInfo",
    "UpdateInfo",
    # Config Manager
    "ConfigManager",
    # Computation Runner
    "ComputationRunner",
    "ComputationParameters",
    "ComputationResult",
    "ComputationJob",
    "ComputationType",
    "ComputationStatus",
    "COMPUTATION_INFO",
    "create_computation_runner",
    # Job Queue
    "JobQueueManager",
    "JobPriority",
    "JobHistory",
    "create_job_queue",
    # Result Exporter
    "ResultExporter",
    "ExportOptions",
    "export_results",
]
