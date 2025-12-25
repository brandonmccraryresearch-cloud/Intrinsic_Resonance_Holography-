"""
HTML Report Generator for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md Appendix K

This module generates interactive HTML reports for IRH computations:
    - MathJax equation rendering
    - Collapsible sections
    - Interactive elements
    - PDF export capability

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md Appendix K"


# =============================================================================
# HTML Templates
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- MathJax for equation rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
            --border-color: #dee2e6;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
            color: var(--primary-color);
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        
        header .subtitle {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        
        nav.toc {{
            background: var(--light-bg);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        nav.toc h2 {{
            margin-top: 0;
            color: var(--secondary-color);
        }}
        
        nav.toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        nav.toc li {{
            margin: 8px 0;
        }}
        
        nav.toc a {{
            color: var(--primary-color);
            text-decoration: none;
        }}
        
        nav.toc a:hover {{
            color: var(--secondary-color);
        }}
        
        section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        section h2 {{
            color: var(--secondary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 10px;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 15px;
            background: var(--light-bg);
            border: none;
            width: 100%;
            text-align: left;
            font-size: 1.1em;
            border-radius: 5px;
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .collapsible:hover {{
            background: #e9ecef;
        }}
        
        .collapsible::after {{
            content: '+';
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .collapsible.active::after {{
            content: '-';
        }}
        
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
            padding: 0 15px;
        }}
        
        .collapsible-content.show {{
            max-height: 2000px;
            padding: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--secondary-color);
            color: white;
        }}
        
        tr:hover {{
            background: var(--light-bg);
        }}
        
        .equation-box {{
            background: var(--light-bg);
            padding: 20px;
            border-left: 4px solid var(--secondary-color);
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
        
        .reference {{
            color: var(--secondary-color);
            font-size: 0.9em;
            font-style: italic;
        }}
        
        .status-excellent {{
            color: var(--success-color);
            font-weight: bold;
        }}
        
        .status-good {{
            color: var(--warning-color);
        }}
        
        .status-tension {{
            color: var(--danger-color);
            font-weight: bold;
        }}
        
        .metadata {{
            background: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .metadata-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media print {{
            .collapsible-content {{
                max-height: none !important;
                padding: 15px !important;
            }}
            .collapsible::after {{
                content: '' !important;
            }}
        }}
    </style>
</head>
<body>

<header>
    <h1>{title}</h1>
    <div class="subtitle">IRH v21.0 Computational Framework</div>
    <div class="subtitle">Generated: {timestamp}</div>
</header>

<nav class="toc">
    <h2>Contents</h2>
    <ul>
        {toc}
    </ul>
</nav>

{content}

<footer>
    <p>Generated by IRH v21.0 Computational Framework</p>
    <p>Theoretical Foundation: IRH21.md</p>
</footer>

<script>
// Collapsible functionality
document.querySelectorAll('.collapsible').forEach(button => {{
    button.addEventListener('click', function() {{
        this.classList.toggle('active');
        const content = this.nextElementSibling;
        content.classList.toggle('show');
    }});
}});
</script>

</body>
</html>
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HTMLSection:
    """A section of the HTML report."""
    id: str
    title: str
    content: str
    collapsible: bool = False
    
    # Theoretical Reference: IRH v21.4

    
    def to_html(self) -> str:
        """Generate HTML for section."""
        if self.collapsible:
            return f"""
<section id="{self.id}">
    <button class="collapsible">{self.title}</button>
    <div class="collapsible-content">
        {self.content}
    </div>
</section>
"""
        else:
            return f"""
<section id="{self.id}">
    <h2>{self.title}</h2>
    {self.content}
</section>
"""
    
    # Theoretical Reference: IRH v21.4

    
    def toc_entry(self) -> str:
        """Generate TOC entry."""
        return f'<li><a href="#{self.id}">{self.title}</a></li>'


# =============================================================================
# HTML Generator Class
# =============================================================================

@dataclass
class HTMLGenerator:
    """
    Generate interactive HTML reports for IRH computations.
    
    # Theoretical Reference:
        IRH21.md Appendix K
        Interactive reports with MathJax equation rendering.
    """
    title: str = "IRH v21.0 Computation Report"
    sections: List[HTMLSection] = field(default_factory=list)
    
    # Theoretical Reference: IRH v21.4
    def add_section(
        self,
        id: str,
        title: str,
        content: str,
        collapsible: bool = False
    ) -> None:
        """Add a section to the report."""
        self.sections.append(HTMLSection(
            id=id,
            title=title,
            content=content,
            collapsible=collapsible
        ))
    
    # Theoretical Reference: IRH v21.4

    
    def add_equation_section(
        self,
        equations: List[Dict[str, Any]],
        collapsible: bool = True
    ) -> None:
        
        # Theoretical Reference: IRH v21.4 (ML Infrastructure)
        """Add section with theoretical equations."""
        content_parts = []
        
        for eq in equations:
            label = eq.get('label', '')
            latex = eq.get('latex', '')
            description = eq.get('description', '')
            reference = eq.get('reference', '')
            
            content_parts.append(f"""
<div class="equation-box">
    <p>{description}</p>
    <p>\\[{latex}\\]</p>
    <p class="reference">[IRH21.md {reference}]</p>
</div>
""")
        
        self.add_section(
            id="equations",
            title="Theoretical Equations",
            content="\n".join(content_parts),
            collapsible=collapsible
        )
    
    # Theoretical Reference: IRH v21.4

    
    def add_results_table(
        self,
        results: List[Dict[str, Any]],
        section_title: str = "Computational Results"
    ) -> None:
        """Add results table section."""
        rows = []
        for r in results:
            name = r.get('name', 'Unknown')
            value = r.get('value', 0.0)
            uncertainty = r.get('uncertainty', 0.0)
            unit = r.get('unit', '')
            ref = r.get('theoretical_ref', '')
            
            rows.append(f"""
<tr>
    <td>{name}</td>
    <td>{value:.10e}</td>
    <td>±{uncertainty:.2e}</td>
    <td>{unit}</td>
    <td class="reference">{ref}</td>
</tr>
""")
        
        content = f"""
<table>
    <thead>
        <tr>
            <th>Observable</th>
            <th>Value</th>
            <th>Uncertainty</th>
            <th>Unit</th>
            <th>Reference</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>
"""
        
        self.add_section(
            id="results",
            title=section_title,
            content=content,
            collapsible=False
        )
    
    # Theoretical Reference: IRH v21.4

    
    def add_comparison_table(
        self,
        results: List[Dict[str, Any]],
        section_title: str = "Theory vs Experiment"
    ) -> None:
        """Add comparison table section."""
        import numpy as np
        
        rows = []
        for r in results:
            exp_val = r.get('experimental_value')
            if exp_val is None:
                continue
            
            name = r.get('name', 'Unknown')
            value = r.get('value', 0.0)
            uncertainty = r.get('uncertainty', 0.0)
            exp_unc = r.get('experimental_uncertainty', 0.0)
            
            total_unc = np.sqrt(uncertainty**2 + exp_unc**2)
            sigma = abs(value - exp_val) / total_unc if total_unc > 0 else 0
            
            # Status class
            if sigma < 1:
                status_class = "status-excellent"
                status = "Excellent"
            elif sigma < 2:
                status_class = "status-good"
                status = "Good"
            else:
                status_class = "status-tension"
                status = "Tension"
            
            rows.append(f"""
<tr>
    <td>{name}</td>
    <td>{value:.10e}</td>
    <td>{exp_val:.10e}</td>
    <td>{sigma:.2f}σ</td>
    <td class="{status_class}">{status}</td>
</tr>
""")
        
        if rows:
            content = f"""
<table>
    <thead>
        <tr>
            <th>Observable</th>
            <th>IRH Prediction</th>
            <th>Experimental</th>
            <th>Deviation</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>
"""
            
            self.add_section(
                id="comparison",
                title=section_title,
                content=content,
                collapsible=False
            )
    
    # Theoretical Reference: IRH v21.4

    
    def add_metadata_section(
        self,
        metadata: Dict[str, Any]
    ) -> None:
        """Add provenance metadata section."""
        items = []
        for key, value in metadata.items():
            items.append(f'<span class="metadata-item"><strong>{key}:</strong> {value}</span>')
        
        content = f'<div class="metadata">{"".join(items)}</div>'
        
        self.add_section(
            id="metadata",
            title="Computation Metadata",
            content=content,
            collapsible=True
        )
    
    # Theoretical Reference: IRH v21.4

    
    def generate(self) -> str:
        """Generate complete HTML document."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        toc = "\n".join(s.toc_entry() for s in self.sections)
        content = "\n".join(s.to_html() for s in self.sections)
        
        return HTML_TEMPLATE.format(
            title=self.title,
            timestamp=timestamp,
            toc=toc,
            content=content
        )
    
    # Theoretical Reference: IRH v21.4

    
    def save(self, path: Union[str, Path]) -> None:
        """Save HTML document to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate())


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Theoretical Reference: IRH v21.4


def generate_html_report(
    title: str,
    results: List[Dict[str, Any]],
    equations: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate an HTML report from computation results.
    
    Parameters
    ----------
    title : str
        Report title
    results : list[dict]
        List of result dictionaries
    equations : list[dict], optional
        List of equation specifications
    metadata : dict, optional
        Provenance metadata
    output_path : str, optional
        Path to save HTML file
        
    Returns
    -------
    str
        HTML document content
    """
    gen = HTMLGenerator(title=title)
    
    if equations:
        gen.add_equation_section(equations)
    
    gen.add_results_table(results)
    gen.add_comparison_table(results)
    
    if metadata:
        gen.add_metadata_section(metadata)
    
    html_content = gen.generate()
    
    if output_path:
        gen.save(output_path)
    
    return html_content


# Theoretical Reference: IRH v21.4



def create_interactive_section(
    id: str,
    title: str,
    content: str,
    collapsible: bool = True
) -> str:
    """
    Create an interactive HTML section.
    
    Parameters
    ----------
    id : str
        Section ID for linking
    title : str
        Section title
    content : str
        Section content (can include HTML)
    collapsible : bool
        Whether section is collapsible
        
    Returns
    -------
    str
        HTML section content
    """
    section = HTMLSection(id=id, title=title, content=content, collapsible=collapsible)
    return section.to_html()


__all__ = [
    'HTMLGenerator',
    'HTMLSection',
    'generate_html_report',
    'create_interactive_section',
]
