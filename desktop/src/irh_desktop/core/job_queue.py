"""
IRH Desktop - Job Queue Manager

Manages a persistent queue of computation jobs with:
- Priority scheduling
- Job persistence (resume after restart)
- Resource management
- History tracking

This module implements Phase 4 of the DEB_PACKAGE_ROADMAP.md:
- Job queue management

Author: Brandon D. McCrary
"""

import json
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum, auto
from queue import PriorityQueue

from irh_desktop.core.computation_runner import (
    ComputationRunner,
    ComputationParameters,
    ComputationResult,
    ComputationJob,
    ComputationType,
    ComputationStatus,
)

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass(order=True)
class QueuedJob:
    """
    A job in the priority queue.
    
    Attributes
    ----------
    priority : int
        Priority value (lower = higher priority)
    created_at : datetime
        Creation time (for FIFO within priority)
    job_id : str
        Unique job identifier
    params : ComputationParameters
        Computation parameters
    """
    priority: int
    created_at: datetime = field(compare=False)
    job_id: str = field(compare=False)
    params: ComputationParameters = field(compare=False)


@dataclass
class JobHistory:
    """
    Record of a completed job for history tracking.
    
    Attributes
    ----------
    job_id : str
        Job identifier
    computation_type : str
        Type of computation
    status : str
        Final status
    started_at : datetime
        Start time
    completed_at : datetime
        Completion time
    duration_seconds : float
        Total duration
    success : bool
        Whether succeeded
    result_summary : Dict[str, Any]
        Summary of results
    """
    job_id: str
    computation_type: str
    status: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    success: bool
    result_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "computation_type": self.computation_type,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "result_summary": self.result_summary,
        }


class JobQueueManager:
    """
    Manages computation job queue with persistence.
    
    Features:
    - Priority-based scheduling
    - Job persistence across restarts
    - History tracking
    - Resource management
    
    Parameters
    ----------
    runner : ComputationRunner
        Computation runner instance
    data_dir : Path
        Directory for persistent data
    max_concurrent : int
        Maximum concurrent jobs
    max_history : int
        Maximum history entries to keep
        
    Examples
    --------
    >>> runner = ComputationRunner()
    >>> queue = JobQueueManager(runner)
    >>> job_id = queue.enqueue(
    ...     ComputationParameters(ComputationType.FIXED_POINT),
    ...     priority=JobPriority.HIGH
    ... )
    >>> queue.start_processing()
    
    Theoretical Foundation
    ----------------------
    Implements job management as specified in
    docs/DEB_PACKAGE_ROADMAP.md §4 "Computation Interface"
    """
    
    def __init__(
        self,
        runner: Optional[ComputationRunner] = None,
        data_dir: Optional[Path] = None,
        max_concurrent: int = 2,
        max_history: int = 100
    ):
        """Initialize the Job Queue Manager."""
        self.runner = runner or ComputationRunner()
        self.data_dir = data_dir or Path.home() / ".local/share/irh/queue"
        self.max_concurrent = max_concurrent
        self.max_history = max_history
        
        # Queue data
        self._queue: PriorityQueue = PriorityQueue()
        self._pending: Dict[str, QueuedJob] = {}
        self._running: Dict[str, str] = {}  # job_id -> runner_job_id
        self._history: List[JobHistory] = []
        
        # Threading
        self._lock = threading.Lock()
        self._processing = False
        self._process_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._status_callbacks: List[Callable[[str, str], None]] = []
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persisted state
        self._load_state()
        
        # Connect to runner callbacks
        self.runner.add_completion_callback(self._on_job_completed)
    
    def _load_state(self) -> None:
        """Load persisted queue state."""
        state_file = self.data_dir / "queue_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file) as f:
                data = json.load(f)
            
            # Load history
            for item in data.get("history", []):
                self._history.append(JobHistory(
                    job_id=item["job_id"],
                    computation_type=item["computation_type"],
                    status=item["status"],
                    started_at=datetime.fromisoformat(item["started_at"]),
                    completed_at=datetime.fromisoformat(item["completed_at"]),
                    duration_seconds=item["duration_seconds"],
                    success=item["success"],
                    result_summary=item.get("result_summary", {}),
                ))
            
            logger.info(f"Loaded {len(self._history)} history entries")
            
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
    
    def _save_state(self) -> None:
        """Save queue state to disk."""
        state_file = self.data_dir / "queue_state.json"
        
        try:
            # Keep only recent history
            recent_history = self._history[-self.max_history:]
            
            data = {
                "history": [h.to_dict() for h in recent_history],
                "last_saved": datetime.now().isoformat(),
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def add_status_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Add callback for job status changes.
        
        Parameters
        ----------
        callback : callable
            Function(job_id, status)
        """
        self._status_callbacks.append(callback)
    
    def _emit_status(self, job_id: str, status: str) -> None:
        """Emit status change to callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(job_id, status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def enqueue(
        self,
        params: ComputationParameters,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """
        Add a computation to the queue.
        
        Parameters
        ----------
        params : ComputationParameters
            Computation parameters
        priority : JobPriority
            Job priority
            
        Returns
        -------
        str
            Job ID
        """
        with self._lock:
            # Generate job ID
            job_id = f"queue_{datetime.now():%Y%m%d_%H%M%S_%f}"
            
            # Create queued job
            queued = QueuedJob(
                priority=priority.value,
                created_at=datetime.now(),
                job_id=job_id,
                params=params,
            )
            
            # Add to queue
            self._queue.put(queued)
            self._pending[job_id] = queued
            
            logger.info(f"Enqueued job {job_id} with priority {priority.name}")
        
        self._emit_status(job_id, "queued")
        
        # Start processing if not running
        if self._processing:
            self._check_and_start_jobs()
        
        return job_id
    
    def start_processing(self) -> None:
        """Start processing the job queue."""
        if self._processing:
            return
        
        self._processing = True
        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self._process_thread.start()
        logger.info("Job queue processing started")
    
    def stop_processing(self) -> None:
        """Stop processing the job queue."""
        self._processing = False
        if self._process_thread:
            self._process_thread.join(timeout=5.0)
        logger.info("Job queue processing stopped")
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        import time
        
        while self._processing:
            self._check_and_start_jobs()
            time.sleep(0.5)  # Check every 500ms
    
    def _check_and_start_jobs(self) -> None:
        """Check queue and start jobs if capacity allows."""
        with self._lock:
            # Check if we can start more jobs
            running_count = len(self._running)
            available_slots = self.max_concurrent - running_count
            
            if available_slots <= 0:
                return
            
            # Start jobs from queue
            while available_slots > 0 and not self._queue.empty():
                try:
                    queued = self._queue.get_nowait()
                    
                    # Remove from pending
                    if queued.job_id in self._pending:
                        del self._pending[queued.job_id]
                    
                    # Submit to runner
                    runner_job_id = self.runner.submit(queued.params)
                    self._running[queued.job_id] = runner_job_id
                    
                    self._emit_status(queued.job_id, "running")
                    logger.info(f"Started job {queued.job_id} -> {runner_job_id}")
                    
                    available_slots -= 1
                    
                except Exception as e:
                    logger.error(f"Failed to start job: {e}")
                    break
    
    def _on_job_completed(
        self,
        runner_job_id: str,
        result: ComputationResult
    ) -> None:
        """Handle job completion from runner."""
        with self._lock:
            # Find our job ID
            queue_job_id = None
            for qid, rid in list(self._running.items()):
                if rid == runner_job_id:
                    queue_job_id = qid
                    del self._running[qid]
                    break
            
            if not queue_job_id:
                return
            
            # Get runner job for details
            runner_job = self.runner.get_job(runner_job_id)
            
            # Create history entry
            history = JobHistory(
                job_id=queue_job_id,
                computation_type=runner_job.params.computation_type.value if runner_job else "unknown",
                status="completed" if result.success else "failed",
                started_at=runner_job.started_at if runner_job else datetime.now(),
                completed_at=datetime.now(),
                duration_seconds=result.duration_seconds,
                success=result.success,
                result_summary=result.values,
            )
            
            self._history.append(history)
        
        # Save state
        self._save_state()
        
        # Emit status
        status = "completed" if result.success else "failed"
        self._emit_status(queue_job_id, status)
        
        logger.info(f"Job {queue_job_id} {status}")
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Parameters
        ----------
        job_id : str
            Job to cancel
            
        Returns
        -------
        bool
            True if cancelled
        """
        with self._lock:
            # Check if pending
            if job_id in self._pending:
                del self._pending[job_id]
                self._emit_status(job_id, "cancelled")
                return True
            
            # Check if running
            if job_id in self._running:
                runner_job_id = self._running[job_id]
                success = self.runner.cancel_job(runner_job_id)
                if success:
                    del self._running[job_id]
                    self._emit_status(job_id, "cancelled")
                return success
        
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns
        -------
        Dict[str, Any]
            Queue status information
        """
        with self._lock:
            return {
                "pending": len(self._pending),
                "running": len(self._running),
                "processing": self._processing,
                "history_count": len(self._history),
            }
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of pending jobs.
        
        Returns
        -------
        List[Dict[str, Any]]
            Pending job information
        """
        with self._lock:
            return [
                {
                    "job_id": job.job_id,
                    "computation_type": job.params.computation_type.value,
                    "priority": JobPriority(job.priority).name,
                    "created_at": job.created_at.isoformat(),
                }
                for job in self._pending.values()
            ]
    
    def get_running_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of running jobs.
        
        Returns
        -------
        List[Dict[str, Any]]
            Running job information
        """
        with self._lock:
            result = []
            for queue_id, runner_id in self._running.items():
                runner_job = self.runner.get_job(runner_id)
                if runner_job:
                    result.append({
                        "job_id": queue_id,
                        "computation_type": runner_job.params.computation_type.value,
                        "progress": runner_job.progress,
                        "progress_message": runner_job.progress_message,
                        "started_at": runner_job.started_at.isoformat() if runner_job.started_at else None,
                    })
            return result
    
    def get_history(
        self,
        limit: int = 20,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get job history.
        
        Parameters
        ----------
        limit : int
            Maximum entries to return
        success_only : bool
            Only return successful jobs
            
        Returns
        -------
        List[Dict[str, Any]]
            History entries
        """
        with self._lock:
            history = list(reversed(self._history))
            
            if success_only:
                history = [h for h in history if h.success]
            
            return [h.to_dict() for h in history[:limit]]
    
    def clear_history(self) -> int:
        """
        Clear job history.
        
        Returns
        -------
        int
            Number of entries cleared
        """
        with self._lock:
            count = len(self._history)
            self._history.clear()
        
        self._save_state()
        return count


# Convenience functions
def create_job_queue(
    runner: Optional[ComputationRunner] = None
) -> JobQueueManager:
    """
    Create a job queue manager.
    
    Parameters
    ----------
    runner : ComputationRunner, optional
        Computation runner to use
        
    Returns
    -------
    JobQueueManager
        Configured queue manager
    """
    return JobQueueManager(runner=runner)
