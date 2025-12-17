"""
LaTeX Report Generator for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K

This module generates LaTeX documents for IRH computation results:
    - Automatic compilation of results with uncertainty
    - Theoretical references and equation citations
    - Publication-quality formatting
    - Figure inclusion support

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K"


# =============================================================================
# LaTeX Templates
# =============================================================================

DOCUMENT_TEMPLATE = r"""
\documentclass[11pt,a4paper]{{article}}

% Packages
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{fancyhdr}}
\usepackage{{xcolor}}

% Header/Footer
\pagestyle{{fancy}}
\fancyhf{{}}
\rhead{{IRH v21.0}}
\lhead{{Computation Report}}
\rfoot{{\thepage}}
\lfoot{{Generated: {timestamp}}}

% Custom commands
\newcommand{{\irhref}}[1]{{\textcolor{{blue}}{{\small[IRH21.md #1]}}}}
\newcommand{{\uncertainty}}[2]{{#1 \pm #2}}

\title{{{title}}}
\author{{IRH v21.0 Computational Framework}}
\date{{{date}}}

\begin{{document}}

\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\tableofcontents
\newpage

{content}

\end{{document}}
"""

EQUATION_TEMPLATE = r"""
\section{{{section_title}}}
\label{{sec:{section_label}}}

{description}

\begin{{equation}}
\label{{eq:{equation_label}}}
{equation}
\end{{equation}}

\irhref{{{reference}}}
"""

RESULT_TABLE_TEMPLATE = r"""
\begin{{table}}[H]
\centering
\caption{{{caption}}}
\label{{tab:{label}}}
\begin{{tabular}}{{l S[table-format=3.10] S[table-format=1.2e-2] l}}
\toprule
Observable & {{Value}} & {{Uncertainty}} & Reference \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ObservableResult:
    """A computed observable with uncertainty."""
    name: str
    value: float
    uncertainty: float
    unit: str = ""
    theoretical_ref: str = ""
    experimental_value: Optional[float] = None
    experimental_uncertainty: Optional[float] = None
    
    def to_latex_row(self) -> str:
        """Generate LaTeX table row."""
        name_tex = self.name.replace('_', r'\_')
        return f"{name_tex} & {self.value:.10e} & {self.uncertainty:.2e} & \\irhref{{{self.theoretical_ref}}} \\\\"
    
    def to_latex_display(self) -> str:
        """Generate displayable LaTeX result."""
        return f"${self.name} = {self.value:.10e} \\pm {self.uncertainty:.2e}$ {self.unit}"


@dataclass
class EquationSpec:
    """Specification for a theoretical equation."""
    label: str
    latex: str
    description: str
    reference: str
    section_title: str = ""


@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: str
    label: str = ""
    subsections: List['ReportSection'] = field(default_factory=list)
    
    def to_latex(self) -> str:
        """Generate LaTeX section."""
        label = self.label or self.title.lower().replace(' ', '_')
        lines = [
            f"\\section{{{self.title}}}",
            f"\\label{{sec:{label}}}",
            "",
            self.content,
        ]
        
        for subsec in self.subsections:
            lines.extend([
                "",
                f"\\subsection{{{subsec.title}}}",
                f"\\label{{sec:{subsec.label or subsec.title.lower().replace(' ', '_')}}}",
                "",
                subsec.content,
            ])
        
        return "\n".join(lines)


# =============================================================================
# LaTeX Generator Class
# =============================================================================

@dataclass
class LaTeXGenerator:
    """
    Generate publication-quality LaTeX reports for IRH computations.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Standardized output format with theoretical provenance.
    """
    title: str = "IRH v21.0 Computation Report"
    abstract: str = ""
    author: str = "IRH Computational Framework"
    
    sections: List[ReportSection] = field(default_factory=list)
    equations: List[EquationSpec] = field(default_factory=list)
    results: List[ObservableResult] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.abstract:
            self.abstract = (
                "This report presents computational results from the "
                "Intrinsic Resonance Holography (IRH) v21.0 framework, "
                "implementing the theoretical formalism of IRH21.md with "
                "certified numerical precision."
            )
    
    def add_section(self, title: str, content: str, label: str = "") -> None:
        """Add a section to the report."""
        self.sections.append(ReportSection(title=title, content=content, label=label))
    
    def add_equation(
        self,
        label: str,
        latex: str,
        description: str,
        reference: str,
        section_title: str = ""
    ) -> None:
        """Add a theoretical equation to the report."""
        self.equations.append(EquationSpec(
            label=label,
            latex=latex,
            description=description,
            reference=reference,
            section_title=section_title or f"Equation {label}"
        ))
    
    def add_result(
        self,
        name: str,
        value: float,
        uncertainty: float,
        unit: str = "",
        theoretical_ref: str = "",
        experimental_value: Optional[float] = None,
        experimental_uncertainty: Optional[float] = None
    ) -> None:
        """Add a computational result."""
        self.results.append(ObservableResult(
            name=name,
            value=value,
            uncertainty=uncertainty,
            unit=unit,
            theoretical_ref=theoretical_ref,
            experimental_value=experimental_value,
            experimental_uncertainty=experimental_uncertainty
        ))
    
    def generate_theoretical_section(self) -> str:
        """Generate section for theoretical equations."""
        if not self.equations:
            return ""
        
        lines = [
            "\\section{Theoretical Foundation}",
            "\\label{sec:theory}",
            "",
            "The following equations from IRH21.md are implemented in this computation:",
            ""
        ]
        
        for eq in self.equations:
            lines.extend([
                f"\\subsection{{{eq.section_title}}}",
                f"\\label{{subsec:{eq.label}}}",
                "",
                eq.description,
                "",
                "\\begin{equation}",
                f"\\label{{eq:{eq.label}}}",
                eq.latex,
                "\\end{equation}",
                "",
                f"\\irhref{{{eq.reference}}}",
                ""
            ])
        
        return "\n".join(lines)
    
    def generate_results_table(self, caption: str = "Computed Observables") -> str:
        """Generate LaTeX table of results."""
        if not self.results:
            return ""
        
        rows = "\n".join(r.to_latex_row() for r in self.results)
        
        return RESULT_TABLE_TEMPLATE.format(
            caption=caption,
            label="results",
            rows=rows
        )
    
    def generate_comparison_section(self) -> str:
        """Generate theory vs experiment comparison section."""
        compared = [r for r in self.results if r.experimental_value is not None]
        
        if not compared:
            return ""
        
        lines = [
            "\\section{Theory vs Experiment}",
            "\\label{sec:comparison}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Comparison with Experimental Values}",
            "\\label{tab:comparison}",
            "\\begin{tabular}{l S[table-format=3.10] S[table-format=3.10] S[table-format=2.2]}",
            "\\toprule",
            "Observable & {IRH Prediction} & {Experiment} & {$\\sigma$ deviation} \\\\",
            "\\midrule"
        ]
        
        for r in compared:
            exp_unc = r.experimental_uncertainty or 0
            total_unc = np.sqrt(r.uncertainty**2 + exp_unc**2)
            sigma = abs(r.value - r.experimental_value) / total_unc if total_unc > 0 else 0
            
            name_tex = r.name.replace('_', r'\_')
            lines.append(f"{name_tex} & {r.value:.10e} & {r.experimental_value:.10e} & {sigma:.2f} \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate(self) -> str:
        """Generate complete LaTeX document."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        date = datetime.now().strftime("%B %d, %Y")
        
        # Build content
        content_parts = []
        
        # Theoretical section
        theory_sec = self.generate_theoretical_section()
        if theory_sec:
            content_parts.append(theory_sec)
        
        # Custom sections
        for section in self.sections:
            content_parts.append(section.to_latex())
        
        # Results section
        if self.results:
            content_parts.append("\\section{Computational Results}")
            content_parts.append("\\label{sec:results}")
            content_parts.append("")
            content_parts.append(self.generate_results_table())
        
        # Comparison section
        comparison_sec = self.generate_comparison_section()
        if comparison_sec:
            content_parts.append(comparison_sec)
        
        content = "\n\n".join(content_parts)
        
        return DOCUMENT_TEMPLATE.format(
            timestamp=timestamp,
            title=self.title,
            date=date,
            abstract=self.abstract,
            content=content
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save LaTeX document to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate())


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def generate_latex_report(
    title: str,
    results: List[Dict[str, Any]],
    equations: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a LaTeX report from computation results.
    
    Parameters
    ----------
    title : str
        Report title
    results : list[dict]
        List of result dictionaries with keys:
        'name', 'value', 'uncertainty', 'unit', 'theoretical_ref'
    equations : list[dict], optional
        List of equation specifications
    output_path : str, optional
        Path to save LaTeX file
        
    Returns
    -------
    str
        LaTeX document content
    """
    gen = LaTeXGenerator(title=title)
    
    for r in results:
        gen.add_result(
            name=r.get('name', 'Unknown'),
            value=r.get('value', 0.0),
            uncertainty=r.get('uncertainty', 0.0),
            unit=r.get('unit', ''),
            theoretical_ref=r.get('theoretical_ref', ''),
            experimental_value=r.get('experimental_value'),
            experimental_uncertainty=r.get('experimental_uncertainty')
        )
    
    if equations:
        for eq in equations:
            gen.add_equation(
                label=eq.get('label', 'eq'),
                latex=eq.get('latex', ''),
                description=eq.get('description', ''),
                reference=eq.get('reference', ''),
                section_title=eq.get('section_title', '')
            )
    
    latex_content = gen.generate()
    
    if output_path:
        gen.save(output_path)
    
    return latex_content


def create_equation_section(
    equations: List[Dict[str, Any]],
    section_title: str = "Theoretical Equations"
) -> str:
    """
    Create a LaTeX section with theoretical equations.
    
    Parameters
    ----------
    equations : list[dict]
        List of equation dictionaries
    section_title : str
        Section title
        
    Returns
    -------
    str
        LaTeX section content
    """
    gen = LaTeXGenerator()
    for eq in equations:
        gen.add_equation(**eq)
    return gen.generate_theoretical_section()


def create_results_table(
    results: List[Dict[str, Any]],
    caption: str = "Computational Results"
) -> str:
    """
    Create a LaTeX results table.
    
    Parameters
    ----------
    results : list[dict]
        List of result dictionaries
    caption : str
        Table caption
        
    Returns
    -------
    str
        LaTeX table content
    """
    gen = LaTeXGenerator()
    for r in results:
        gen.add_result(
            name=r.get('name', 'Unknown'),
            value=r.get('value', 0.0),
            uncertainty=r.get('uncertainty', 0.0),
            unit=r.get('unit', ''),
            theoretical_ref=r.get('theoretical_ref', '')
        )
    return gen.generate_results_table(caption)


__all__ = [
    'LaTeXGenerator',
    'ObservableResult',
    'EquationSpec',
    'ReportSection',
    'generate_latex_report',
    'create_equation_section',
    'create_results_table',
]
