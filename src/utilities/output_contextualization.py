"""
Output Contextualization Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K, copilot21promptMAX.md Phase III

This module provides standardized output generation with complete theoretical
provenance, uncertainty quantification, and reproducibility tracking.

Key Features:
    - IRHOutputWriter for standardized outputs
    - Uncertainty quantification and propagation
    - Comprehensive output reports with provenance
    - IRH-DEF schema compliance

Authors: IRH Computational Framework Team
Last Updated: December 2024 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K, copilot21promptMAX.md Phase III"


# =============================================================================
# Computation Types
# =============================================================================


class ComputationType(Enum):
    """Types of IRH computations for output classification."""
    RG_FLOW = "rg_flow"
    OBSERVABLE_EXTRACTION = "observable_extraction"
    TOPOLOGICAL_INVARIANT = "topological_invariant"
    CONVERGENCE_STUDY = "convergence_study"
    FALSIFICATION_TEST = "falsification_test"
    BENCHMARK = "benchmark"


# =============================================================================
# Theoretical Context
# =============================================================================


@dataclass
class TheoreticalContext:
    """
    Metadata linking computation to theoretical foundation.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Every computation must be traceable to specific equations.
    """
    manuscript_version: str = "IRH21.md v21.0"
    equations_implemented: List[str] = field(default_factory=list)
    section_references: List[str] = field(default_factory=list)
    theoretical_precision_target: float = 1e-10
    
    def add_equation(self, equation: str, section: str = ""):
        """Add equation reference."""
        if equation not in self.equations_implemented:
            self.equations_implemented.append(equation)
        if section and section not in self.section_references:
            self.section_references.append(section)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'manuscript_version': self.manuscript_version,
            'equations_implemented': self.equations_implemented,
            'section_references': self.section_references,
            'precision_target': self.theoretical_precision_target
        }


# =============================================================================
# Computational Provenance
# =============================================================================


@dataclass
class ComputationalProvenance:
    """
    Complete specification of computational environment and parameters.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Enables exact reproduction of computational results.
    """
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit_hash: str = ""
    python_version: str = ""
    numpy_version: str = ""
    lattice_parameters: Dict[str, int] = field(default_factory=dict)
    rg_parameters: Dict[str, float] = field(default_factory=dict)
    numerical_methods: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    hardware_specs: Dict[str, str] = field(default_factory=dict)
    
    def compute_reproducibility_hash(self) -> str:
        """
        Generate deterministic hash of all parameters affecting output.
        Enables exact reproduction verification.
        """
        canonical_repr = json.dumps({
            'git_commit': self.git_commit_hash,
            'lattice': self.lattice_parameters,
            'rg': self.rg_parameters,
            'methods': self.numerical_methods,
            'seed': self.random_seed
        }, sort_keys=True)
        
        return hashlib.sha256(canonical_repr.encode()).hexdigest()[:16]
    
    def gather_environment(self):
        """Automatically collect computational environment metadata."""
        self.python_version = sys.version.split()[0]
        self.numpy_version = np.__version__
        
        # Try to get git commit
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.git_commit_hash = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError):
            self.git_commit_hash = "unknown"
        
        # Hardware specs
        self.hardware_specs = {
            'platform': platform.platform(),
            'processor': platform.processor() or 'unknown',
            'machine': platform.machine()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'git_commit': self.git_commit_hash,
            'python_version': self.python_version,
            'numpy_version': self.numpy_version,
            'lattice_parameters': self.lattice_parameters,
            'rg_parameters': self.rg_parameters,
            'numerical_methods': self.numerical_methods,
            'random_seed': self.random_seed,
            'hardware_specs': self.hardware_specs,
            'reproducibility_hash': self.compute_reproducibility_hash()
        }


# =============================================================================
# Observable Result
# =============================================================================


@dataclass
class ObservableResult:
    """
    Single physical observable with complete uncertainty quantification.
    
    Theoretical Reference:
        IRH21.md §3
        Physical observables derived from cosmic fixed point.
    """
    name: str
    value: float
    uncertainty: float
    unit: str
    theoretical_prediction: Optional[float] = None
    experimental_value: Optional[float] = None
    experimental_uncertainty: Optional[float] = None
    sigma_deviation: Optional[float] = None
    
    uncertainty_breakdown: Dict[str, float] = field(default_factory=dict)
    """Decomposition by source: 'discretization', 'integration', 'truncation', etc."""
    
    theoretical_foundation: TheoreticalContext = field(default_factory=TheoreticalContext)
    
    def compute_sigma_deviation(self) -> Optional[float]:
        """Calculate statistical significance of theory-experiment comparison."""
        if self.experimental_value is not None:
            exp_unc = self.experimental_uncertainty or 0
            total_unc = np.sqrt(self.uncertainty**2 + exp_unc**2)
            if total_unc > 0:
                self.sigma_deviation = abs(self.value - self.experimental_value) / total_unc
        return self.sigma_deviation
    
    def get_agreement_status(self) -> str:
        """Return human-readable agreement status."""
        sigma = self.compute_sigma_deviation()
        if sigma is None:
            return "NO_COMPARISON"
        elif sigma < 1:
            return "EXCELLENT"
        elif sigma < 2:
            return "GOOD"
        elif sigma < 3:
            return "MARGINAL"
        else:
            return "TENSION"
    
    def to_dict(self) -> Dict[str, Any]:
        self.compute_sigma_deviation()
        return {
            'name': self.name,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'unit': self.unit,
            'theoretical_prediction': self.theoretical_prediction,
            'experimental': {
                'value': self.experimental_value,
                'uncertainty': self.experimental_uncertainty
            } if self.experimental_value is not None else None,
            'sigma_deviation': self.sigma_deviation,
            'agreement_status': self.get_agreement_status(),
            'uncertainty_breakdown': self.uncertainty_breakdown,
            'theoretical_foundation': self.theoretical_foundation.to_dict()
        }


# =============================================================================
# Uncertainty Tracker
# =============================================================================


class UncertaintyTracker:
    """
    Comprehensive uncertainty quantification and propagation system.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Track and report uncertainty propagation through computational pipeline.
    """
    
    def __init__(self):
        self.error_registry: Dict[str, Dict[str, Any]] = {}
    
    def register_source_uncertainty(
        self,
        observable: str,
        value: float,
        uncertainty: float,
        source: str,
        theoretical_ref: str = ""
    ):
        """
        Register primitive uncertainty source with theoretical provenance.
        
        Parameters
        ----------
        observable : str
            Name of the observable
        value : float
            Central value
        uncertainty : float
            Absolute uncertainty
        source : str
            Source of uncertainty (e.g., 'rg_convergence', 'discretization')
        theoretical_ref : str
            Reference to theoretical manuscript
        """
        if observable not in self.error_registry:
            self.error_registry[observable] = {
                'value': value,
                'uncertainties': {},
                'total_uncertainty': 0.0,
                'theoretical_ref': theoretical_ref
            }
        
        self.error_registry[observable]['uncertainties'][source] = uncertainty
        
        # Recompute total
        total_sq = sum(u**2 for u in self.error_registry[observable]['uncertainties'].values())
        self.error_registry[observable]['total_uncertainty'] = np.sqrt(total_sq)
    
    def get_uncertainty(self, observable: str) -> Tuple[float, float]:
        """Get value and total uncertainty for an observable."""
        if observable not in self.error_registry:
            raise KeyError(f"Observable '{observable}' not registered")
        
        entry = self.error_registry[observable]
        return entry['value'], entry['total_uncertainty']
    
    def propagate_uncertainty(
        self,
        output_name: str,
        formula: Callable,
        input_names: List[str],
        derivatives: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Propagate uncertainties through a formula using linear error propagation.
        
        Parameters
        ----------
        output_name : str
            Name for the derived observable
        formula : callable
            Function computing output from inputs
        input_names : list[str]
            Names of input observables
        derivatives : dict, optional
            Partial derivatives ∂f/∂x_i. If None, computed numerically.
            
        Returns
        -------
        tuple
            (output_value, output_uncertainty)
        """
        # Gather inputs
        values = [self.error_registry[name]['value'] for name in input_names]
        uncertainties = [self.error_registry[name]['total_uncertainty'] for name in input_names]
        
        # Compute output value
        output_value = formula(*values)
        
        # Compute derivatives if not provided
        if derivatives is None:
            derivatives = {}
            delta = 1e-8
            for i, name in enumerate(input_names):
                values_plus = values.copy()
                values_plus[i] += delta
                values_minus = values.copy()
                values_minus[i] -= delta
                derivatives[name] = (formula(*values_plus) - formula(*values_minus)) / (2 * delta)
        
        # Linear error propagation
        output_variance = sum(
            (derivatives.get(name, 0) * unc)**2
            for name, unc in zip(input_names, uncertainties)
        )
        output_uncertainty = np.sqrt(output_variance)
        
        # Register derived observable
        self.error_registry[output_name] = {
            'value': output_value,
            'uncertainties': {'propagated': output_uncertainty},
            'total_uncertainty': output_uncertainty,
            'derived_from': input_names
        }
        
        return output_value, output_uncertainty
    
    def generate_uncertainty_report(self) -> str:
        """Generate comprehensive uncertainty budget document."""
        lines = [
            "# Uncertainty Budget Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Observable Summary",
            ""
        ]
        
        for name, data in self.error_registry.items():
            value = data['value']
            total_unc = data['total_uncertainty']
            rel_unc = total_unc / abs(value) if value != 0 else float('inf')
            
            lines.append(f"### {name}")
            lines.append(f"- Value: {value:.10e}")
            lines.append(f"- Total uncertainty: ±{total_unc:.2e}")
            lines.append(f"- Relative uncertainty: {rel_unc:.2e}")
            lines.append("")
            
            if 'uncertainties' in data and data['uncertainties']:
                lines.append("Breakdown:")
                for source, unc in data['uncertainties'].items():
                    frac = (unc / total_unc * 100) if total_unc > 0 else 0
                    lines.append(f"  - {source}: ±{unc:.2e} ({frac:.1f}%)")
                lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# IRH Output Writer
# =============================================================================


class IRHOutputWriter:
    """
    Standardized writer for all IRH computational outputs.
    Enforces IRH-DEF schema compliance.
    
    Theoretical Reference:
        IRH21.md Appendix K
        All outputs must follow standardized format for reproducibility.
    """
    
    def __init__(
        self,
        computation_type: Union[ComputationType, str],
        output_path: Optional[str] = None
    ):
        """
        Initialize output writer.
        
        Parameters
        ----------
        computation_type : ComputationType or str
            Type of computation
        output_path : str, optional
            Path for output file (JSON)
        """
        if isinstance(computation_type, str):
            computation_type = ComputationType(computation_type)
        
        self.computation_type = computation_type
        self.output_path = output_path
        self.provenance = ComputationalProvenance()
        self.provenance.gather_environment()
        self.theoretical_context = TheoreticalContext()
        self.results: List[ObservableResult] = []
        self.diagnostics: Dict[str, Any] = {}
        self.uncertainty_tracker = UncertaintyTracker()
    
    def set_lattice_parameters(self, **params):
        """Set lattice discretization parameters."""
        self.provenance.lattice_parameters.update(params)
    
    def set_rg_parameters(self, **params):
        """Set RG flow parameters."""
        self.provenance.rg_parameters.update(params)
    
    def set_numerical_methods(self, **methods):
        """Set numerical method choices."""
        self.provenance.numerical_methods.update(methods)
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.provenance.random_seed = seed
    
    def add_equation_reference(self, equation: str, section: str = ""):
        """Add theoretical equation reference."""
        self.theoretical_context.add_equation(equation, section)
    
    def add_result(
        self,
        name: str,
        value: float,
        uncertainty: float,
        unit: str,
        experimental_value: Optional[float] = None,
        experimental_uncertainty: Optional[float] = None,
        uncertainty_breakdown: Optional[Dict[str, float]] = None
    ) -> ObservableResult:
        """
        Add a computed observable result.
        
        Returns the ObservableResult for further modification.
        """
        result = ObservableResult(
            name=name,
            value=value,
            uncertainty=uncertainty,
            unit=unit,
            experimental_value=experimental_value,
            experimental_uncertainty=experimental_uncertainty,
            uncertainty_breakdown=uncertainty_breakdown or {},
            theoretical_foundation=self.theoretical_context
        )
        self.results.append(result)
        return result
    
    def add_diagnostic(self, key: str, value: Any):
        """Add diagnostic information."""
        self.diagnostics[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire output to dictionary."""
        return {
            'irh_def_version': '1.0',
            'computation_type': self.computation_type.value,
            'provenance': self.provenance.to_dict(),
            'theoretical_context': self.theoretical_context.to_dict(),
            'results': [r.to_dict() for r in self.results],
            'diagnostics': self.diagnostics
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def write(self, path: Optional[str] = None):
        """Write output to file."""
        output_path = path or self.output_path
        if output_path is None:
            raise ValueError("No output path specified")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(self.to_json())
    
    def generate_report(self) -> str:
        """
        Generate human-readable report with theoretical context.
        
        Theoretical Reference:
            copilot21promptMAX.md Phase III
            Outputs must include complete theoretical provenance.
        """
        lines = [
            "=" * 60,
            f"IRH v21.0 COMPUTATIONAL OUTPUT",
            f"Computation Type: {self.computation_type.value}",
            "=" * 60,
            "",
            "[PROVENANCE]",
            f"  Timestamp: {self.provenance.timestamp}",
            f"  Git commit: {self.provenance.git_commit_hash}",
            f"  Reproducibility hash: {self.provenance.compute_reproducibility_hash()}",
            "",
            "[THEORETICAL FOUNDATION]",
            f"  Manuscript: {self.theoretical_context.manuscript_version}",
            f"  Equations: {', '.join(self.theoretical_context.equations_implemented)}",
            f"  Sections: {', '.join(self.theoretical_context.section_references)}",
            "",
            "[RESULTS]"
        ]
        
        for result in self.results:
            sigma = result.compute_sigma_deviation()
            status = result.get_agreement_status()
            
            lines.append(f"")
            lines.append(f"  {result.name}:")
            lines.append(f"    Value: {result.value:.10e} ± {result.uncertainty:.2e} {result.unit}")
            
            if result.experimental_value is not None:
                lines.append(f"    Experimental: {result.experimental_value:.10e} ± {result.experimental_uncertainty or 0:.2e}")
                lines.append(f"    Agreement: {status} ({sigma:.2f}σ)")
            
            if result.uncertainty_breakdown:
                lines.append(f"    Uncertainty breakdown:")
                for source, unc in result.uncertainty_breakdown.items():
                    lines.append(f"      - {source}: ±{unc:.2e}")
        
        lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_output_writer(
    computation_type: str,
    output_path: Optional[str] = None
) -> IRHOutputWriter:
    """Factory function to create output writer."""
    return IRHOutputWriter(computation_type, output_path)


def format_observable(
    name: str,
    value: float,
    uncertainty: float,
    unit: str = "",
    precision: int = 10
) -> str:
    """Format observable for display."""
    if unit:
        return f"{name} = {value:.{precision}e} ± {uncertainty:.2e} {unit}"
    return f"{name} = {value:.{precision}e} ± {uncertainty:.2e}"


__all__ = [
    # Enums
    'ComputationType',
    
    # Data classes
    'TheoreticalContext',
    'ComputationalProvenance',
    'ObservableResult',
    
    # Main classes
    'UncertaintyTracker',
    'IRHOutputWriter',
    
    # Functions
    'create_output_writer',
    'format_observable',
]
