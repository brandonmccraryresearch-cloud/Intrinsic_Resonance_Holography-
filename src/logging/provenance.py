"""
Provenance Tracking for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K

This module provides computation provenance tracking:
    - Complete computation history
    - Input parameter tracking
    - Git commit hash for reproducibility
    - Result verification chains

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K"


# =============================================================================
# Computation Record Data Class
# =============================================================================

@dataclass
class ComputationRecord:
    """
    A record of a single computation with full provenance.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Complete provenance enables exact reproduction.
    """
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Computation identification
    name: str = ""
    description: str = ""
    theoretical_ref: str = ""
    equations: List[str] = field(default_factory=list)
    
    # Input parameters
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Output results
    output_results: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    git_commit: str = ""
    python_version: str = ""
    numpy_version: str = ""
    random_seed: Optional[int] = None
    
    # Performance
    duration_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    
    # Verification
    checksum: str = ""
    parent_id: Optional[str] = None  # For chained computations
    
    def __post_init__(self):
        """Gather environment info on creation."""
        if not self.python_version:
            self.python_version = sys.version.split()[0]
        if not self.numpy_version:
            self.numpy_version = np.__version__
        if not self.git_commit:
            self._gather_git_commit()
        if not self.checksum:
            self._compute_checksum()
    
    def _gather_git_commit(self) -> None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.git_commit = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError):
            self.git_commit = "unknown"
    
    def _compute_checksum(self) -> None:
        """Compute reproducibility checksum."""
        data = {
            'name': self.name,
            'input_parameters': self.input_parameters,
            'random_seed': self.random_seed,
            'git_commit': self.git_commit
        }
        canonical = json.dumps(data, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationRecord':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ComputationRecord':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Provenance Tracker Class
# =============================================================================

class ProvenanceTracker:
    """
    Track computation provenance for reproducibility.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Complete provenance chain enables verification and reproduction.
    """
    
    _instance: Optional['ProvenanceTracker'] = None
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize provenance tracker.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store provenance records (JSON)
        auto_save : bool
            Automatically save records after each computation
        """
        self.storage_path = storage_path
        self.auto_save = auto_save
        self._records: List[ComputationRecord] = []
        self._current: Optional[ComputationRecord] = None
        
        if storage_path:
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()
    
    @classmethod
    def get_instance(cls) -> 'ProvenanceTracker':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_existing(self) -> None:
        """Load existing records from storage."""
        if self.storage_path and Path(self.storage_path).exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self._records = [ComputationRecord.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._records = []
    
    def _save(self) -> None:
        """Save records to storage."""
        if self.storage_path:
            with open(self.storage_path, 'w') as f:
                json.dump([r.to_dict() for r in self._records], f, indent=2, default=str)
    
    def start_computation(
        self,
        name: str,
        description: str = "",
        theoretical_ref: str = "",
        equations: Optional[List[str]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        parent_id: Optional[str] = None
    ) -> ComputationRecord:
        """
        Start tracking a new computation.
        
        Parameters
        ----------
        name : str
            Computation name
        description : str
            Description of computation
        theoretical_ref : str
            IRH21.md reference
        equations : list[str], optional
            Implemented equations
        input_parameters : dict, optional
            Input parameters
        random_seed : int, optional
            Random seed for reproducibility
        parent_id : str, optional
            ID of parent computation
            
        Returns
        -------
        ComputationRecord
            The new computation record
        """
        self._current = ComputationRecord(
            name=name,
            description=description,
            theoretical_ref=theoretical_ref,
            equations=equations or [],
            input_parameters=input_parameters or {},
            random_seed=random_seed,
            parent_id=parent_id
        )
        
        return self._current
    
    def add_input(self, key: str, value: Any) -> None:
        """Add an input parameter to current computation."""
        if self._current:
            self._current.input_parameters[key] = value
    
    def add_result(self, key: str, value: Any) -> None:
        """Add an output result to current computation."""
        if self._current:
            self._current.output_results[key] = value
    
    def complete_computation(
        self,
        duration_seconds: Optional[float] = None,
        peak_memory_mb: Optional[float] = None
    ) -> ComputationRecord:
        """
        Complete the current computation and save.
        
        Parameters
        ----------
        duration_seconds : float, optional
            Computation duration
        peak_memory_mb : float, optional
            Peak memory usage
            
        Returns
        -------
        ComputationRecord
            The completed record
        """
        if not self._current:
            raise RuntimeError("No computation in progress")
        
        self._current.duration_seconds = duration_seconds
        self._current.peak_memory_mb = peak_memory_mb
        self._current._compute_checksum()
        
        record = self._current
        self._records.append(record)
        self._current = None
        
        if self.auto_save:
            self._save()
        
        return record
    
    def get_record(self, id: str) -> Optional[ComputationRecord]:
        """Get a computation record by ID."""
        for record in self._records:
            if record.id == id:
                return record
        return None
    
    def get_records(
        self,
        name: Optional[str] = None,
        since: Optional[str] = None,
        theoretical_ref: Optional[str] = None
    ) -> List[ComputationRecord]:
        """
        Get filtered computation records.
        
        Parameters
        ----------
        name : str, optional
            Filter by computation name
        since : str, optional
            Filter by timestamp (ISO format)
        theoretical_ref : str, optional
            Filter by theoretical reference
            
        Returns
        -------
        list[ComputationRecord]
            Filtered records
        """
        records = self._records
        
        if name:
            records = [r for r in records if r.name == name]
        
        if since:
            records = [r for r in records if r.timestamp >= since]
        
        if theoretical_ref:
            records = [r for r in records if theoretical_ref in r.theoretical_ref]
        
        return records
    
    def get_chain(self, id: str) -> List[ComputationRecord]:
        """
        Get the full provenance chain for a computation.
        
        Parameters
        ----------
        id : str
            ID of the computation
            
        Returns
        -------
        list[ComputationRecord]
            Chain of records from root to given computation
        """
        record = self.get_record(id)
        if not record:
            return []
        
        chain = [record]
        while record.parent_id:
            parent = self.get_record(record.parent_id)
            if parent:
                chain.insert(0, parent)
                record = parent
            else:
                break
        
        return chain
    
    def verify_record(self, id: str) -> Dict[str, Any]:
        """
        Verify a computation record's integrity.
        
        Parameters
        ----------
        id : str
            ID of the computation
            
        Returns
        -------
        dict
            Verification result with 'valid' key
        """
        record = self.get_record(id)
        if not record:
            return {'valid': False, 'error': 'Record not found'}
        
        # Recompute checksum
        data = {
            'name': record.name,
            'input_parameters': record.input_parameters,
            'random_seed': record.random_seed,
            'git_commit': record.git_commit
        }
        canonical = json.dumps(data, sort_keys=True, default=str)
        expected_checksum = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        
        valid = record.checksum == expected_checksum
        
        return {
            'valid': valid,
            'recorded_checksum': record.checksum,
            'computed_checksum': expected_checksum,
            'record_id': id
        }
    
    def export(self, path: str) -> None:
        """Export all records to JSON file."""
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate a provenance report."""
        lines = [
            "# IRH v21.0 Provenance Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Total computations: {len(self._records)}",
            "",
            "## Computation Summary",
            ""
        ]
        
        # Group by name
        by_name: Dict[str, List[ComputationRecord]] = {}
        for record in self._records:
            by_name.setdefault(record.name, []).append(record)
        
        for name, records in by_name.items():
            lines.append(f"### {name}")
            lines.append(f"- Count: {len(records)}")
            if records:
                latest = max(records, key=lambda r: r.timestamp)
                lines.append(f"- Latest: {latest.timestamp}")
                lines.append(f"- Ref: {latest.theoretical_ref}")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._current = None
        if self.auto_save:
            self._save()


# =============================================================================
# Module-Level Functions
# =============================================================================

_default_tracker: Optional[ProvenanceTracker] = None


def create_provenance_tracker(
    storage_path: Optional[str] = None,
    auto_save: bool = True
) -> ProvenanceTracker:
    """
    Create a new provenance tracker.
    
    Parameters
    ----------
    storage_path : str, optional
        Path to store provenance records
    auto_save : bool
        Auto-save after each computation
        
    Returns
    -------
    ProvenanceTracker
        New tracker instance
    """
    global _default_tracker
    _default_tracker = ProvenanceTracker(
        storage_path=storage_path,
        auto_save=auto_save
    )
    return _default_tracker


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get the default provenance tracker.
    
    Returns
    -------
    ProvenanceTracker
        Default tracker instance
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ProvenanceTracker()
    return _default_tracker


__all__ = [
    'ProvenanceTracker',
    'ComputationRecord',
    'create_provenance_tracker',
    'get_provenance_tracker',
]
