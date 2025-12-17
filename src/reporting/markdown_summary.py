"""
Markdown Summary Generator for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K

This module generates markdown summaries for IRH computations:
    - Quick computation summaries
    - GitHub-compatible formatting
    - Results tables with uncertainties
    - Comparison tables

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K"


# =============================================================================
# Markdown Generator Class
# =============================================================================

@dataclass
class MarkdownGenerator:
    """
    Generate markdown summaries for IRH computations.
    
    Theoretical Reference:
        IRH21.md Appendix K
        Quick summaries with GitHub-compatible formatting.
    """
    title: str = "IRH v21.0 Computation Summary"
    sections: List[str] = field(default_factory=list)
    
    def add_header(self, text: str, level: int = 2) -> None:
        """Add a header."""
        self.sections.append(f"{'#' * level} {text}\n")
    
    def add_paragraph(self, text: str) -> None:
        """Add a paragraph."""
        self.sections.append(f"{text}\n")
    
    def add_equation(
        self,
        label: str,
        latex: str,
        description: str = "",
        reference: str = ""
    ) -> None:
        """Add an equation (using code block for GitHub compatibility)."""
        parts = []
        if description:
            parts.append(f"**{label}**: {description}")
        parts.append(f"```")
        parts.append(latex)
        parts.append(f"```")
        if reference:
            parts.append(f"*[IRH21.md {reference}]*")
        
        self.sections.append("\n".join(parts) + "\n")
    
    def add_results_table(
        self,
        results: List[Dict[str, Any]],
        title: str = "Results"
    ) -> None:
        """Add a results table."""
        lines = [
            f"### {title}\n",
            "| Observable | Value | Uncertainty | Reference |",
            "|------------|-------|-------------|-----------|"
        ]
        
        for r in results:
            name = r.get('name', 'Unknown')
            value = r.get('value', 0.0)
            uncertainty = r.get('uncertainty', 0.0)
            ref = r.get('theoretical_ref', '')
            unit = r.get('unit', '')
            
            val_str = f"{value:.6e}"
            if unit:
                val_str += f" {unit}"
            
            lines.append(f"| {name} | {val_str} | ±{uncertainty:.2e} | {ref} |")
        
        self.sections.append("\n".join(lines) + "\n")
    
    def add_comparison_table(
        self,
        results: List[Dict[str, Any]],
        title: str = "Theory vs Experiment"
    ) -> None:
        """Add a comparison table."""
        # Filter to those with experimental values
        compared = [r for r in results if r.get('experimental_value') is not None]
        
        if not compared:
            return
        
        lines = [
            f"### {title}\n",
            "| Observable | IRH Prediction | Experiment | σ deviation | Status |",
            "|------------|----------------|------------|-------------|--------|"
        ]
        
        for r in compared:
            name = r.get('name', 'Unknown')
            value = r.get('value', 0.0)
            exp_val = r.get('experimental_value', 0.0)
            uncertainty = r.get('uncertainty', 0.0)
            exp_unc = r.get('experimental_uncertainty', 0.0)
            
            total_unc = np.sqrt(uncertainty**2 + exp_unc**2)
            sigma = abs(value - exp_val) / total_unc if total_unc > 0 else 0
            
            if sigma < 1:
                status = "✅ Excellent"
            elif sigma < 2:
                status = "🟡 Good"
            else:
                status = "⚠️ Tension"
            
            lines.append(f"| {name} | {value:.6e} | {exp_val:.6e} | {sigma:.2f}σ | {status} |")
        
        self.sections.append("\n".join(lines) + "\n")
    
    def add_checklist(
        self,
        items: List[Dict[str, Any]],
        title: str = "Verification"
    ) -> None:
        """Add a checklist section."""
        lines = [f"### {title}\n"]
        
        for item in items:
            name = item.get('name', 'Item')
            passed = item.get('passed', False)
            details = item.get('details', '')
            
            checkbox = "✅" if passed else "❌"
            line = f"- {checkbox} **{name}**"
            if details:
                line += f": {details}"
            lines.append(line)
        
        self.sections.append("\n".join(lines) + "\n")
    
    def add_code_block(
        self,
        code: str,
        language: str = "python"
    ) -> None:
        """Add a code block."""
        self.sections.append(f"```{language}\n{code}\n```\n")
    
    def add_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> None:
        """Add metadata section."""
        lines = ["### Computation Metadata\n"]
        
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        
        self.sections.append("\n".join(lines) + "\n")
    
    def generate(self) -> str:
        """Generate complete markdown document."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        header = [
            f"# {self.title}",
            "",
            f"*Generated: {timestamp}*",
            "",
            "---",
            ""
        ]
        
        return "\n".join(header) + "\n".join(self.sections)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save markdown document to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate())


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def generate_markdown_summary(
    title: str,
    results: List[Dict[str, Any]],
    equations: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a markdown summary from computation results.
    
    Parameters
    ----------
    title : str
        Summary title
    results : list[dict]
        List of result dictionaries
    equations : list[dict], optional
        List of equation specifications
    metadata : dict, optional
        Provenance metadata
    output_path : str, optional
        Path to save markdown file
        
    Returns
    -------
    str
        Markdown document content
    """
    gen = MarkdownGenerator(title=title)
    
    # Add abstract
    gen.add_paragraph(
        "This summary presents computational results from the "
        "Intrinsic Resonance Holography (IRH) v21.0 framework."
    )
    
    # Add equations section
    if equations:
        gen.add_header("Theoretical Equations", level=2)
        for eq in equations:
            gen.add_equation(
                label=eq.get('label', ''),
                latex=eq.get('latex', ''),
                description=eq.get('description', ''),
                reference=eq.get('reference', '')
            )
    
    # Add results
    gen.add_header("Computational Results", level=2)
    gen.add_results_table(results)
    
    # Add comparison
    gen.add_comparison_table(results)
    
    # Add metadata
    if metadata:
        gen.add_metadata(metadata)
    
    md_content = gen.generate()
    
    if output_path:
        gen.save(output_path)
    
    return md_content


def create_results_markdown(
    results: List[Dict[str, Any]],
    title: str = "Results"
) -> str:
    """
    Create a markdown table of results.
    
    Parameters
    ----------
    results : list[dict]
        List of result dictionaries
    title : str
        Table title
        
    Returns
    -------
    str
        Markdown table content
    """
    gen = MarkdownGenerator()
    gen.add_results_table(results, title=title)
    return "\n".join(gen.sections)


def create_comparison_markdown(
    results: List[Dict[str, Any]],
    title: str = "Theory vs Experiment"
) -> str:
    """
    Create a markdown comparison table.
    
    Parameters
    ----------
    results : list[dict]
        List of result dictionaries with experimental values
    title : str
        Table title
        
    Returns
    -------
    str
        Markdown table content
    """
    gen = MarkdownGenerator()
    gen.add_comparison_table(results, title=title)
    return "\n".join(gen.sections)


def create_quick_summary(
    results: List[Dict[str, Any]],
    title: str = "Quick Summary"
) -> str:
    """
    Create a quick one-line summary of key results.
    
    Parameters
    ----------
    results : list[dict]
        List of result dictionaries
    title : str
        Summary title
        
    Returns
    -------
    str
        Quick summary string
    """
    lines = [f"**{title}**\n"]
    
    for r in results[:5]:  # Top 5 results
        name = r.get('name', 'Unknown')
        value = r.get('value', 0.0)
        uncertainty = r.get('uncertainty', 0.0)
        
        lines.append(f"- {name} = {value:.6e} ± {uncertainty:.2e}")
    
    if len(results) > 5:
        lines.append(f"- ... and {len(results) - 5} more results")
    
    return "\n".join(lines)


__all__ = [
    'MarkdownGenerator',
    'generate_markdown_summary',
    'create_results_markdown',
    'create_comparison_markdown',
    'create_quick_summary',
]
