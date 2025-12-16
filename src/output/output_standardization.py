"""
IRH Output Standardization - Phase VIII Implementation.

This module implements the complete output standardization infrastructure
per the IRH v21.0 verification protocol from copilot21promtMAX.md.

Components:
- IRHDEFSchema: Standard schema for IRH outputs
- OutputFormatter: Multi-format output generation
- ReportGenerator: Comprehensive report creation
- ComplianceChecker: Schema validation
- MetadataManager: Reproducibility tracking

Theoretical Reference:
    IRH21.md - Final Compliance Checklist
    "All outputs must conform to IRH-DEF standard format"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    PLAIN = "plain"


class ComplianceLevel(Enum):
    """IRH-DEF compliance levels."""
    FULL = "full"        # All required fields present
    PARTIAL = "partial"  # Some optional fields missing
    MINIMAL = "minimal"  # Only core fields present
    INVALID = "invalid"  # Missing required fields


@dataclass
class TheoreticalAnnotation:
    """
    Theoretical annotation for output values.
    
    Theoretical Reference:
        IRH21.md - Theoretical Traceability requirement
        "Every function cites specific equations from IRH21.md"
    """
    equation_number: str
    section: str
    description: str
    manuscript: str = "IRH21.md"


@dataclass
class UncertaintyInfo:
    """
    Uncertainty information for computed values.
    
    Theoretical Reference:
        IRH21.md - Uncertainty Quantification requirement
        "All outputs include rigorous error propagation"
    """
    value: float
    uncertainty: float
    unit: str = ""
    confidence_level: float = 0.95
    method: str = "analytical"
    
    @property
    def relative_uncertainty(self) -> float:
        """Compute relative uncertainty."""
        if abs(self.value) < 1e-15:
            return float('inf')
        return abs(self.uncertainty / self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "uncertainty": self.uncertainty,
            "unit": self.unit,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "relative_uncertainty": self.relative_uncertainty,
        }


@dataclass
class ProvenanceInfo:
    """
    Computational provenance for reproducibility.
    
    Theoretical Reference:
        IRH21.md - Reproducibility requirement
        "Complete provenance metadata enables exact reproduction"
    """
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = ""
    python_version: str = ""
    numpy_version: str = ""
    random_seed: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute reproducibility hash."""
        content = json.dumps({
            "parameters": self.parameters,
            "random_seed": self.random_seed,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "random_seed": self.random_seed,
            "parameters": self.parameters,
            "reproducibility_hash": self.compute_hash(),
        }


@dataclass
class ValidationInfo:
    """
    Validation status for computed values.
    
    Theoretical Reference:
        IRH21.md - Regression Protection requirement
        "Automated detection of deviations from certified baselines"
    """
    passed: bool
    checks_performed: List[str] = field(default_factory=list)
    deviations: Dict[str, float] = field(default_factory=dict)
    baseline_comparison: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks_performed": self.checks_performed,
            "deviations": self.deviations,
            "baseline_comparison": self.baseline_comparison,
        }


@dataclass
class IRHDEFSchema:
    """
    IRH-DEF Schema for standardized output structure.
    
    This is the core schema that all IRH outputs must conform to,
    ensuring consistent structure across all computational results.
    
    Theoretical Reference:
        IRH21.md - Schema Compliance requirement
        "All outputs conform to IRH-DEF standard format"
    
    Attributes
    ----------
    schema_version : str
        Version of the IRH-DEF schema
    computation_type : str
        Type of computation (e.g., "rg_flow", "fixed_point", "observable")
    results : Dict[str, Any]
        Main computation results
    theoretical_context : Dict[str, TheoreticalAnnotation]
        Theoretical annotations for results
    uncertainties : Dict[str, UncertaintyInfo]
        Uncertainty information for values
    provenance : ProvenanceInfo
        Computational provenance
    validation : ValidationInfo
        Validation status
    metadata : Dict[str, Any]
        Additional metadata
    """
    
    schema_version: str = "1.0.0"
    computation_type: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    theoretical_context: Dict[str, Any] = field(default_factory=dict)
    uncertainties: Dict[str, UncertaintyInfo] = field(default_factory=dict)
    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    validation: ValidationInfo = field(default_factory=lambda: ValidationInfo(passed=True))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(
        self,
        name: str,
        value: Any,
        uncertainty: Optional[UncertaintyInfo] = None,
        annotation: Optional[TheoreticalAnnotation] = None,
    ) -> None:
        """
        Add a result with optional uncertainty and annotation.
        
        Parameters
        ----------
        name : str
            Result name
        value : Any
            Result value
        uncertainty : UncertaintyInfo, optional
            Uncertainty information
        annotation : TheoreticalAnnotation, optional
            Theoretical annotation
        """
        self.results[name] = value
        if uncertainty:
            self.uncertainties[name] = uncertainty
        if annotation:
            self.theoretical_context[name] = {
                "equation": annotation.equation_number,
                "section": annotation.section,
                "description": annotation.description,
                "manuscript": annotation.manuscript,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        uncertainties_dict = {
            k: v.to_dict() if isinstance(v, UncertaintyInfo) else v
            for k, v in self.uncertainties.items()
        }
        
        return {
            "irh_def_schema": {
                "version": self.schema_version,
                "computation_type": self.computation_type,
            },
            "results": self.results,
            "theoretical_context": self.theoretical_context,
            "uncertainties": uncertainties_dict,
            "provenance": self.provenance.to_dict(),
            "validation": self.validation.to_dict(),
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class OutputFormatter:
    """
    Multi-format output formatter.
    
    Converts IRHDEFSchema to various output formats.
    
    Theoretical Reference:
        IRH21.md - Output Contextualization
        All numerical outputs must include theoretical provenance
    """
    
    def __init__(self, schema: IRHDEFSchema):
        """
        Initialize formatter with schema.
        
        Parameters
        ----------
        schema : IRHDEFSchema
            Schema to format
        """
        self.schema = schema
    
    def format(self, output_format: OutputFormat = OutputFormat.JSON) -> str:
        """
        Format output in specified format.
        
        Parameters
        ----------
        output_format : OutputFormat
            Desired output format
            
        Returns
        -------
        str
            Formatted output string
        """
        formatters = {
            OutputFormat.JSON: self._format_json,
            OutputFormat.MARKDOWN: self._format_markdown,
            OutputFormat.LATEX: self._format_latex,
            OutputFormat.HTML: self._format_html,
            OutputFormat.PLAIN: self._format_plain,
        }
        return formatters[output_format]()
    
    def _format_json(self) -> str:
        """Format as JSON."""
        return self.schema.to_json(indent=2)
    
    def _format_markdown(self) -> str:
        """Format as Markdown."""
        lines = [
            f"# IRH Computation Results",
            f"",
            f"**Schema Version:** {self.schema.schema_version}",
            f"**Computation Type:** {self.schema.computation_type}",
            f"",
            "## Results",
            "",
        ]
        
        for name, value in self.schema.results.items():
            uncertainty = self.schema.uncertainties.get(name)
            if uncertainty:
                lines.append(f"- **{name}:** {value} ± {uncertainty.uncertainty}")
            else:
                lines.append(f"- **{name}:** {value}")
            
            # Add theoretical context
            context = self.schema.theoretical_context.get(name)
            if context:
                lines.append(f"  - *Reference:* {context.get('manuscript', 'IRH21.md')} "
                           f"{context.get('section', '')} {context.get('equation', '')}")
        
        lines.extend([
            "",
            "## Validation",
            f"- **Status:** {'✓ Passed' if self.schema.validation.passed else '✗ Failed'}",
            "",
            "## Provenance",
            f"- **Timestamp:** {self.schema.provenance.timestamp}",
            f"- **Hash:** {self.schema.provenance.compute_hash()}",
        ])
        
        return "\n".join(lines)
    
    def _format_latex(self) -> str:
        """Format as LaTeX."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{IRH Computation Results}",
            r"\begin{tabular}{lll}",
            r"\hline",
            r"Parameter & Value & Reference \\",
            r"\hline",
        ]
        
        for name, value in self.schema.results.items():
            uncertainty = self.schema.uncertainties.get(name)
            context = self.schema.theoretical_context.get(name, {})
            ref = context.get('equation', '-')
            
            if uncertainty:
                val_str = f"${value} \\pm {uncertainty.uncertainty}$"
            else:
                val_str = f"${value}$" if isinstance(value, (int, float)) else str(value)
            
            lines.append(f"{name} & {val_str} & Eq. {ref} \\\\")
        
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _format_html(self) -> str:
        """Format as HTML."""
        rows = []
        for name, value in self.schema.results.items():
            uncertainty = self.schema.uncertainties.get(name)
            context = self.schema.theoretical_context.get(name, {})
            
            val_str = f"{value}"
            if uncertainty:
                val_str += f" ± {uncertainty.uncertainty}"
            
            ref_str = context.get('equation', '-')
            
            rows.append(f"<tr><td>{name}</td><td>{val_str}</td><td>Eq. {ref_str}</td></tr>")
        
        status = "Passed" if self.schema.validation.passed else "Failed"
        status_class = "success" if self.schema.validation.passed else "error"
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>IRH Computation Results</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>IRH Computation Results</h1>
    <p><strong>Schema Version:</strong> {self.schema.schema_version}</p>
    <p><strong>Computation Type:</strong> {self.schema.computation_type}</p>
    
    <h2>Results</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Reference</th></tr>
        {''.join(rows)}
    </table>
    
    <h2>Validation</h2>
    <p class="{status_class}"><strong>Status:</strong> {status}</p>
    
    <h2>Provenance</h2>
    <p><strong>Timestamp:</strong> {self.schema.provenance.timestamp}</p>
    <p><strong>Hash:</strong> {self.schema.provenance.compute_hash()}</p>
</body>
</html>"""
    
    def _format_plain(self) -> str:
        """Format as plain text."""
        lines = [
            "=" * 60,
            "IRH COMPUTATION RESULTS",
            "=" * 60,
            f"Schema Version: {self.schema.schema_version}",
            f"Computation Type: {self.schema.computation_type}",
            "-" * 60,
            "RESULTS:",
        ]
        
        for name, value in self.schema.results.items():
            uncertainty = self.schema.uncertainties.get(name)
            if uncertainty:
                lines.append(f"  {name}: {value} ± {uncertainty.uncertainty}")
            else:
                lines.append(f"  {name}: {value}")
        
        lines.extend([
            "-" * 60,
            f"Validation: {'PASSED' if self.schema.validation.passed else 'FAILED'}",
            f"Hash: {self.schema.provenance.compute_hash()}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class ReportGenerator:
    """
    Comprehensive report generator.
    
    Generates full computation reports with theoretical context,
    validation results, and reproducibility information.
    
    Theoretical Reference:
        IRH21.md - Documentation requirements
        "Complete theoretical provenance chain"
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.sections: List[IRHDEFSchema] = []
        self.title: str = "IRH Computation Report"
        self.author: str = ""
        self.abstract: str = ""
    
    def add_section(self, schema: IRHDEFSchema, section_title: str = "") -> None:
        """
        Add a computation section to the report.
        
        Parameters
        ----------
        schema : IRHDEFSchema
            Schema for this section
        section_title : str
            Title for this section
        """
        schema.metadata["section_title"] = section_title or schema.computation_type
        self.sections.append(schema)
    
    def set_metadata(
        self,
        title: str = "",
        author: str = "",
        abstract: str = "",
    ) -> None:
        """
        Set report metadata.
        
        Parameters
        ----------
        title : str
            Report title
        author : str
            Report author
        abstract : str
            Report abstract
        """
        if title:
            self.title = title
        if author:
            self.author = author
        if abstract:
            self.abstract = abstract
    
    def generate(self, output_format: OutputFormat = OutputFormat.MARKDOWN) -> str:
        """
        Generate complete report.
        
        Parameters
        ----------
        output_format : OutputFormat
            Desired output format
            
        Returns
        -------
        str
            Complete report
        """
        if output_format == OutputFormat.MARKDOWN:
            return self._generate_markdown()
        elif output_format == OutputFormat.LATEX:
            return self._generate_latex()
        elif output_format == OutputFormat.HTML:
            return self._generate_html()
        elif output_format == OutputFormat.JSON:
            return self._generate_json()
        else:
            return self._generate_plain()
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {self.title}",
            "",
        ]
        
        if self.author:
            lines.append(f"**Author:** {self.author}")
            lines.append("")
        
        if self.abstract:
            lines.extend([
                "## Abstract",
                "",
                self.abstract,
                "",
            ])
        
        lines.extend([
            "## Table of Contents",
            "",
        ])
        
        for i, section in enumerate(self.sections, 1):
            section_title = section.metadata.get("section_title", f"Section {i}")
            lines.append(f"{i}. [{section_title}](#{section_title.lower().replace(' ', '-')})")
        
        lines.append("")
        
        for i, section in enumerate(self.sections, 1):
            section_title = section.metadata.get("section_title", f"Section {i}")
            formatter = OutputFormatter(section)
            
            lines.extend([
                f"## {section_title}",
                "",
                formatter.format(OutputFormat.MARKDOWN).split("# IRH Computation Results\n\n")[1],
                "",
            ])
        
        # Summary section
        total_results = sum(len(s.results) for s in self.sections)
        all_passed = all(s.validation.passed for s in self.sections)
        
        lines.extend([
            "## Summary",
            "",
            f"- **Total Sections:** {len(self.sections)}",
            f"- **Total Results:** {total_results}",
            f"- **Overall Validation:** {'✓ All Passed' if all_passed else '✗ Some Failed'}",
            "",
            "---",
            f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex(self) -> str:
        """Generate LaTeX report."""
        sections_tex = []
        for section in self.sections:
            formatter = OutputFormatter(section)
            sections_tex.append(formatter.format(OutputFormat.LATEX))
        
        return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\title{{{self.title}}}
\\author{{{self.author}}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{self.abstract}
\\end{{abstract}}

{chr(10).join(sections_tex)}

\\end{{document}}"""
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        sections_html = []
        for section in self.sections:
            formatter = OutputFormatter(section)
            # Extract just the body content
            html = formatter.format(OutputFormat.HTML)
            body_start = html.find("<body>") + 6
            body_end = html.find("</body>")
            sections_html.append(html[body_start:body_end])
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p><em>Author: {self.author}</em></p>
    
    <h2>Abstract</h2>
    <p>{self.abstract}</p>
    
    {''.join(sections_html)}
    
    <hr>
    <p><em>Generated: {datetime.now(timezone.utc).isoformat()}</em></p>
</body>
</html>"""
    
    def _generate_json(self) -> str:
        """Generate JSON report."""
        return json.dumps({
            "title": self.title,
            "author": self.author,
            "abstract": self.abstract,
            "sections": [s.to_dict() for s in self.sections],
            "generated": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str)
    
    def _generate_plain(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 70,
            self.title.center(70),
            "=" * 70,
            "",
        ]
        
        if self.author:
            lines.append(f"Author: {self.author}")
            lines.append("")
        
        if self.abstract:
            lines.extend(["Abstract:", self.abstract, ""])
        
        for i, section in enumerate(self.sections, 1):
            section_title = section.metadata.get("section_title", f"Section {i}")
            formatter = OutputFormatter(section)
            lines.extend([
                "-" * 70,
                f"SECTION {i}: {section_title}",
                "-" * 70,
                formatter.format(OutputFormat.PLAIN),
                "",
            ])
        
        return "\n".join(lines)


class ComplianceChecker:
    """
    IRH-DEF schema compliance checker.
    
    Validates outputs against the IRH-DEF standard requirements.
    
    Theoretical Reference:
        IRH21.md - Schema Compliance requirement
        "All outputs conform to IRH-DEF standard format"
    """
    
    # Required fields for full compliance
    REQUIRED_FIELDS = {
        "schema_version",
        "computation_type",
        "results",
        "provenance",
        "validation",
    }
    
    # Fields required for theoretical traceability
    TRACEABILITY_FIELDS = {
        "theoretical_context",
    }
    
    # Fields required for uncertainty quantification
    UNCERTAINTY_FIELDS = {
        "uncertainties",
    }
    
    def __init__(self):
        """Initialize compliance checker."""
        self.issues: List[str] = []
    
    def check(self, schema: IRHDEFSchema) -> ComplianceLevel:
        """
        Check schema compliance level.
        
        Parameters
        ----------
        schema : IRHDEFSchema
            Schema to check
            
        Returns
        -------
        ComplianceLevel
            Compliance level
        """
        self.issues = []
        data = schema.to_dict()
        
        # Check required fields
        irh_def = data.get("irh_def_schema", {})
        missing_required = []
        
        if not irh_def.get("version"):
            missing_required.append("schema_version")
        if not irh_def.get("computation_type"):
            missing_required.append("computation_type")
        if not data.get("results"):
            missing_required.append("results")
        if not data.get("provenance"):
            missing_required.append("provenance")
        if "validation" not in data:
            missing_required.append("validation")
        
        if missing_required:
            self.issues.extend([f"Missing required field: {f}" for f in missing_required])
            return ComplianceLevel.INVALID
        
        # Check for traceability
        has_traceability = bool(data.get("theoretical_context"))
        
        # Check for uncertainty
        has_uncertainty = bool(data.get("uncertainties"))
        
        # Determine compliance level
        if has_traceability and has_uncertainty:
            return ComplianceLevel.FULL
        elif has_traceability or has_uncertainty:
            if not has_traceability:
                self.issues.append("Missing theoretical context for full compliance")
            if not has_uncertainty:
                self.issues.append("Missing uncertainty information for full compliance")
            return ComplianceLevel.PARTIAL
        else:
            self.issues.append("No theoretical context")
            self.issues.append("No uncertainty information")
            return ComplianceLevel.MINIMAL
    
    def get_issues(self) -> List[str]:
        """Get compliance issues found."""
        return self.issues.copy()
    
    def validate_theoretical_coverage(
        self,
        schema: IRHDEFSchema,
        required_equations: List[str],
    ) -> Dict[str, bool]:
        """
        Validate that all required equations are referenced.
        
        Parameters
        ----------
        schema : IRHDEFSchema
            Schema to check
        required_equations : List[str]
            List of required equation numbers
            
        Returns
        -------
        Dict[str, bool]
            Coverage for each equation
        """
        referenced = set()
        for context in schema.theoretical_context.values():
            if isinstance(context, dict) and "equation" in context:
                referenced.add(context["equation"])
        
        return {eq: eq in referenced for eq in required_equations}


class MetadataManager:
    """
    Reproducibility metadata manager.
    
    Manages computational provenance and ensures reproducibility.
    
    Theoretical Reference:
        IRH21.md - Reproducibility requirement
        "Complete provenance metadata enables exact reproduction"
    """
    
    def __init__(self):
        """Initialize metadata manager."""
        self._session_id: str = ""
        self._global_params: Dict[str, Any] = {}
        self._computation_log: List[Dict[str, Any]] = []
    
    def start_session(self, random_seed: Optional[int] = None) -> str:
        """
        Start a new computation session.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        str
            Session ID
        """
        import sys
        
        self._session_id = hashlib.sha256(
            datetime.now(timezone.utc).isoformat().encode()
        ).hexdigest()[:12]
        
        self._global_params = {
            "python_version": sys.version,
            "random_seed": random_seed,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        
        self._computation_log = []
        
        return self._session_id
    
    def log_computation(
        self,
        computation_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """
        Log a computation for provenance tracking.
        
        Parameters
        ----------
        computation_type : str
            Type of computation
        parameters : Dict[str, Any]
            Computation parameters
        results : Dict[str, Any]
            Computation results
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "computation_type": computation_type,
            "parameters": parameters,
            "results_hash": hashlib.sha256(
                json.dumps(results, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        }
        self._computation_log.append(entry)
    
    def create_provenance(self, **kwargs) -> ProvenanceInfo:
        """
        Create provenance info with current session data.
        
        Parameters
        ----------
        **kwargs
            Additional provenance parameters
            
        Returns
        -------
        ProvenanceInfo
            Provenance information
        """
        return ProvenanceInfo(
            python_version=self._global_params.get("python_version", ""),
            random_seed=self._global_params.get("random_seed"),
            parameters=kwargs,
        )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary.
        
        Returns
        -------
        Dict[str, Any]
            Session summary including all logged computations
        """
        return {
            "session_id": self._session_id,
            "global_params": self._global_params,
            "computation_count": len(self._computation_log),
            "computations": self._computation_log,
        }
    
    def export_provenance(self) -> str:
        """
        Export full provenance chain as JSON.
        
        Returns
        -------
        str
            JSON string of full provenance
        """
        return json.dumps(self.get_session_summary(), indent=2, default=str)


# Convenience functions

def create_irh_output(
    computation_type: str,
    results: Dict[str, Any],
    uncertainties: Optional[Dict[str, UncertaintyInfo]] = None,
    annotations: Optional[Dict[str, TheoreticalAnnotation]] = None,
) -> IRHDEFSchema:
    """
    Create a standardized IRH output schema.
    
    Parameters
    ----------
    computation_type : str
        Type of computation
    results : Dict[str, Any]
        Computation results
    uncertainties : Dict[str, UncertaintyInfo], optional
        Uncertainty information
    annotations : Dict[str, TheoreticalAnnotation], optional
        Theoretical annotations
        
    Returns
    -------
    IRHDEFSchema
        Standardized output schema
    """
    schema = IRHDEFSchema(computation_type=computation_type)
    
    for name, value in results.items():
        unc = uncertainties.get(name) if uncertainties else None
        ann = annotations.get(name) if annotations else None
        schema.add_result(name, value, unc, ann)
    
    return schema


def format_output(
    schema: IRHDEFSchema,
    output_format: Union[OutputFormat, str] = OutputFormat.JSON,
) -> str:
    """
    Format IRH output in specified format.
    
    Parameters
    ----------
    schema : IRHDEFSchema
        Schema to format
    output_format : OutputFormat or str
        Desired format
        
    Returns
    -------
    str
        Formatted output
    """
    if isinstance(output_format, str):
        output_format = OutputFormat(output_format.lower())
    
    formatter = OutputFormatter(schema)
    return formatter.format(output_format)


def check_compliance(schema: IRHDEFSchema) -> Dict[str, Any]:
    """
    Check schema compliance.
    
    Parameters
    ----------
    schema : IRHDEFSchema
        Schema to check
        
    Returns
    -------
    Dict[str, Any]
        Compliance report
    """
    checker = ComplianceChecker()
    level = checker.check(schema)
    
    return {
        "level": level.value,
        "is_valid": level != ComplianceLevel.INVALID,
        "is_full": level == ComplianceLevel.FULL,
        "issues": checker.get_issues(),
    }
