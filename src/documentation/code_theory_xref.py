"""
Code↔Theory Cross-Reference Generator for IRH v21.0

This module provides tools for generating interactive documentation
that maps between theoretical equations in IRH21.md and their
computational implementations.

Theoretical Reference:
    IRH21.md - Verification Protocol Requirements
    copilot21promtMAX.md - Phase VI: Documentation Infrastructure
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class EquationReference:
    """A reference to an equation in the theoretical manuscript."""
    
    section: str
    equation_number: str
    description: str
    manuscript: str = "IRH21.md"
    
    def __str__(self) -> str:
        return f"{self.manuscript} §{self.section}, Eq. {self.equation_number}"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "section": self.section,
            "equation_number": self.equation_number,
            "description": self.description,
            "manuscript": self.manuscript,
        }


@dataclass
class EquationImplementation:
    """Records where an equation is implemented in code."""
    
    equation: EquationReference
    file_path: str
    function_name: str
    line_number: int
    implementation_status: str  # "complete", "partial", "stub"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation": self.equation.to_dict(),
            "file_path": self.file_path,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "implementation_status": self.implementation_status,
            "notes": self.notes,
        }


@dataclass
class ModuleMapping:
    """Maps a source module to its theoretical foundations."""
    
    module_path: str
    theoretical_sections: List[str]
    key_equations: List[str]
    description: str
    layer: int  # Ontological layer (0=primitives, 1=cgft, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "theoretical_sections": self.theoretical_sections,
            "key_equations": self.key_equations,
            "description": self.description,
            "layer": self.layer,
        }


@dataclass
class CoverageReport:
    """Summary of equation implementation coverage."""
    
    total_equations: int
    implemented_equations: int
    partial_equations: int
    stub_equations: int
    coverage_by_section: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def coverage_percentage(self) -> float:
        if self.total_equations == 0:
            return 0.0
        return (self.implemented_equations / self.total_equations) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_equations": self.total_equations,
            "implemented_equations": self.implemented_equations,
            "partial_equations": self.partial_equations,
            "stub_equations": self.stub_equations,
            "coverage_percentage": self.coverage_percentage,
            "coverage_by_section": self.coverage_by_section,
            "timestamp": self.timestamp,
        }


class CodeTheoryXRef:
    """
    Manages bidirectional mapping between code and theory.
    
    This class provides the infrastructure for:
    1. Scanning source code for equation references
    2. Building a cross-reference database
    3. Generating documentation in various formats
    4. Computing coverage metrics
    
    Theoretical Reference:
        copilot21promtMAX.md - Verification Protocol Requirements
    """
    
    # Pattern to match equation references in docstrings and comments
    EQUATION_PATTERN = re.compile(
        r'(?:Eq\.?|Equation)\s*(\d+\.\d+(?:-\d+\.\d+)?)',
        re.IGNORECASE
    )
    
    SECTION_PATTERN = re.compile(
        r'§(\d+(?:\.\d+)*)',
        re.IGNORECASE
    )
    
    REFERENCE_PATTERN = re.compile(
        r'IRH21\.md\s*§(\d+(?:\.\d+)*),?\s*Eq\.?\s*(\d+\.\d+)',
        re.IGNORECASE
    )
    
    # Known equations in IRH21.md
    CRITICAL_EQUATIONS = {
        "1.1": "S_kin kinetic term",
        "1.2": "S_int interaction term",
        "1.3": "Interaction kernel K",
        "1.4": "S_hol holographic term",
        "1.12": "Wetterich equation",
        "1.13": "β-functions",
        "1.14": "Fixed-point values",
        "1.16": "Universal exponent C_H",
        "2.8": "Spectral dimension flow",
        "2.9": "d_spec → 4",
        "2.10": "Emergent metric",
        "2.17": "ρ_hum calculation",
        "2.21": "w(z) equation",
        "2.24": "LIV parameter ξ",
        "3.4": "α⁻¹ derivation start",
        "3.5": "α⁻¹ derivation end",
        "3.6": "Yukawa coupling",
    }
    
    def __init__(self, source_root: str):
        """
        Initialize the cross-reference manager.
        
        Parameters
        ----------
        source_root : str
            Root directory of source code to scan.
        """
        self.source_root = Path(source_root)
        self.implementations: List[EquationImplementation] = []
        self.module_mappings: List[ModuleMapping] = []
        self._equations_found: Set[str] = set()
    
    def scan_file(self, file_path: Path) -> List[EquationImplementation]:
        """
        Scan a Python file for equation references.
        
        Parameters
        ----------
        file_path : Path
            Path to the Python file to scan.
            
        Returns
        -------
        List[EquationImplementation]
            List of equation implementations found.
        """
        implementations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return implementations
        
        # Parse the AST to find functions with equation references
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return implementations
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Find equation references in docstring
                    eq_matches = self.EQUATION_PATTERN.findall(docstring)
                    ref_matches = self.REFERENCE_PATTERN.findall(docstring)
                    
                    for eq_num in eq_matches:
                        # Normalize equation number (e.g., "1.1-1.4" -> ["1.1", "1.2", "1.3", "1.4"])
                        eq_numbers = self._expand_equation_range(eq_num)
                        
                        for eq in eq_numbers:
                            self._equations_found.add(eq)
                            impl = EquationImplementation(
                                equation=EquationReference(
                                    section=self._infer_section(eq),
                                    equation_number=eq,
                                    description=self.CRITICAL_EQUATIONS.get(eq, ""),
                                ),
                                file_path=str(file_path.relative_to(self.source_root)),
                                function_name=node.name,
                                line_number=node.lineno,
                                implementation_status=self._infer_status(docstring),
                            )
                            implementations.append(impl)
                    
                    for section, eq_num in ref_matches:
                        self._equations_found.add(eq_num)
                        impl = EquationImplementation(
                            equation=EquationReference(
                                section=section,
                                equation_number=eq_num,
                                description=self.CRITICAL_EQUATIONS.get(eq_num, ""),
                            ),
                            file_path=str(file_path.relative_to(self.source_root)),
                            function_name=node.name,
                            line_number=node.lineno,
                            implementation_status=self._infer_status(docstring),
                        )
                        implementations.append(impl)
        
        return implementations
    
    def _expand_equation_range(self, eq_range: str) -> List[str]:
        """Expand equation range like '1.1-1.4' to ['1.1', '1.2', '1.3', '1.4']."""
        if '-' not in eq_range:
            return [eq_range]
        
        parts = eq_range.split('-')
        if len(parts) != 2:
            return [eq_range]
        
        try:
            start_parts = parts[0].split('.')
            end_parts = parts[1].split('.')
            
            if len(start_parts) != 2 or len(end_parts) != 2:
                return [eq_range]
            
            section = start_parts[0]
            start_num = int(start_parts[1])
            end_num = int(end_parts[1])
            
            return [f"{section}.{i}" for i in range(start_num, end_num + 1)]
        except (ValueError, IndexError):
            return [eq_range]
    
    def _infer_section(self, eq_num: str) -> str:
        """Infer the section number from equation number."""
        if '.' in eq_num:
            return eq_num.split('.')[0]
        return "?"
    
    def _infer_status(self, docstring: str) -> str:
        """Infer implementation status from docstring content."""
        docstring_lower = docstring.lower()
        
        if "stub" in docstring_lower or "not implemented" in docstring_lower:
            return "stub"
        elif "partial" in docstring_lower or "todo" in docstring_lower:
            return "partial"
        else:
            return "complete"
    
    def scan_directory(self, directory: Optional[Path] = None) -> None:
        """
        Recursively scan a directory for Python files.
        
        Parameters
        ----------
        directory : Path, optional
            Directory to scan. Defaults to source_root.
        """
        if directory is None:
            directory = self.source_root
        
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith(('.', '__')):
                self.scan_directory(item)
            elif item.suffix == '.py':
                impls = self.scan_file(item)
                self.implementations.extend(impls)
    
    def compute_coverage(self) -> CoverageReport:
        """
        Compute equation implementation coverage.
        
        Returns
        -------
        CoverageReport
            Summary of implementation coverage.
        """
        total = len(self.CRITICAL_EQUATIONS)
        implemented = 0
        partial = 0
        stub = 0
        
        for eq_num in self.CRITICAL_EQUATIONS:
            # Find implementations for this equation
            impls = [i for i in self.implementations if i.equation.equation_number == eq_num]
            
            if not impls:
                continue
            
            # Use best status
            statuses = [i.implementation_status for i in impls]
            if "complete" in statuses:
                implemented += 1
            elif "partial" in statuses:
                partial += 1
            elif "stub" in statuses:
                stub += 1
        
        # Compute coverage by section
        sections = {}
        for eq_num, desc in self.CRITICAL_EQUATIONS.items():
            section = eq_num.split('.')[0]
            if section not in sections:
                sections[section] = {"total": 0, "implemented": 0}
            sections[section]["total"] += 1
            
            impls = [i for i in self.implementations if i.equation.equation_number == eq_num]
            if impls and any(i.implementation_status == "complete" for i in impls):
                sections[section]["implemented"] += 1
        
        coverage_by_section = {
            section: (data["implemented"] / data["total"] * 100) if data["total"] > 0 else 0.0
            for section, data in sections.items()
        }
        
        return CoverageReport(
            total_equations=total,
            implemented_equations=implemented,
            partial_equations=partial,
            stub_equations=stub,
            coverage_by_section=coverage_by_section,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export cross-reference data as dictionary."""
        return {
            "source_root": str(self.source_root),
            "implementations": [i.to_dict() for i in self.implementations],
            "module_mappings": [m.to_dict() for m in self.module_mappings],
            "coverage": self.compute_coverage().to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export cross-reference data as JSON."""
        return json.dumps(self.to_dict(), indent=indent)


def scan_source_directory(source_root: str) -> CodeTheoryXRef:
    """
    Convenience function to scan a source directory.
    
    Parameters
    ----------
    source_root : str
        Path to source root directory.
        
    Returns
    -------
    CodeTheoryXRef
        Populated cross-reference manager.
    """
    xref = CodeTheoryXRef(source_root)
    xref.scan_directory()
    return xref


def generate_markdown_report(xref: CodeTheoryXRef) -> str:
    """
    Generate a Markdown report of equation implementations.
    
    Parameters
    ----------
    xref : CodeTheoryXRef
        Cross-reference data.
        
    Returns
    -------
    str
        Markdown formatted report.
    """
    coverage = xref.compute_coverage()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        "# IRH v21.0: Code ↔ Theory Implementation Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Source Root**: `{xref.source_root}`",
        "",
        "## Coverage Summary",
        "",
        f"- **Total Critical Equations**: {coverage.total_equations}",
        f"- **Fully Implemented**: {coverage.implemented_equations}",
        f"- **Partially Implemented**: {coverage.partial_equations}",
        f"- **Stub Only**: {coverage.stub_equations}",
        f"- **Coverage**: {coverage.coverage_percentage:.1f}%",
        "",
        "## Coverage by Section",
        "",
        "| Section | Coverage |",
        "|---------|----------|",
    ]
    
    for section, pct in sorted(coverage.coverage_by_section.items()):
        status = "✅" if pct == 100 else "⚠️" if pct > 0 else "❌"
        lines.append(f"| §{section} | {status} {pct:.0f}% |")
    
    lines.extend([
        "",
        "## Equation Implementations",
        "",
        "| Equation | Status | File | Function | Line |",
        "|----------|--------|------|----------|------|",
    ])
    
    for impl in sorted(xref.implementations, key=lambda i: i.equation.equation_number):
        status_icon = {
            "complete": "✅",
            "partial": "⚠️",
            "stub": "📝",
        }.get(impl.implementation_status, "❓")
        
        lines.append(
            f"| Eq. {impl.equation.equation_number} | {status_icon} | "
            f"`{impl.file_path}` | `{impl.function_name}` | {impl.line_number} |"
        )
    
    # Add unimplemented equations
    implemented_eqs = {i.equation.equation_number for i in xref.implementations}
    unimplemented = set(xref.CRITICAL_EQUATIONS.keys()) - implemented_eqs
    
    if unimplemented:
        lines.extend([
            "",
            "## Unimplemented Equations",
            "",
            "| Equation | Description |",
            "|----------|-------------|",
        ])
        for eq in sorted(unimplemented):
            lines.append(f"| Eq. {eq} | {xref.CRITICAL_EQUATIONS[eq]} |")
    
    return "\n".join(lines)


def generate_interactive_html(xref: CodeTheoryXRef) -> str:
    """
    Generate an interactive HTML cross-reference document.
    
    Parameters
    ----------
    xref : CodeTheoryXRef
        Cross-reference data.
        
    Returns
    -------
    str
        HTML document with interactive features.
    """
    coverage = xref.compute_coverage()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Build implementation table rows
    impl_rows = []
    for impl in sorted(xref.implementations, key=lambda i: i.equation.equation_number):
        status_class = {
            "complete": "status-complete",
            "partial": "status-partial",
            "stub": "status-stub",
        }.get(impl.implementation_status, "")
        
        impl_rows.append(f'''
            <tr class="{status_class}">
                <td>Eq. {impl.equation.equation_number}</td>
                <td>{impl.equation.description}</td>
                <td><code>{impl.file_path}</code></td>
                <td><code>{impl.function_name}</code></td>
                <td>{impl.line_number}</td>
                <td class="status">{impl.implementation_status}</td>
            </tr>
        ''')
    
    # Build section coverage bars
    section_bars = []
    for section, pct in sorted(coverage.coverage_by_section.items()):
        section_bars.append(f'''
            <div class="coverage-bar">
                <span class="section-label">§{section}</span>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {pct}%"></div>
                </div>
                <span class="coverage-pct">{pct:.0f}%</span>
            </div>
        ''')
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRH v21.0 Code↔Theory Cross-Reference</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }}
        
        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .card h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--text);
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        
        .stat {{
            text-align: center;
            padding: 1rem;
            background: var(--bg);
            border-radius: 8px;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        
        .coverage-bar {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        
        .section-label {{
            width: 3rem;
            font-weight: 600;
        }}
        
        .bar-container {{
            flex: 1;
            height: 12px;
            background: var(--border);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            border-radius: 6px;
            transition: width 0.5s ease;
        }}
        
        .coverage-pct {{
            width: 3rem;
            text-align: right;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        
        .search-container {{
            margin-bottom: 1rem;
        }}
        
        .search-input {{
            width: 100%;
            padding: 0.75rem 1rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: var(--primary);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            font-weight: 600;
            background: var(--bg);
            position: sticky;
            top: 0;
        }}
        
        tr:hover {{
            background: var(--bg);
        }}
        
        .status-complete {{ background: rgba(22, 163, 74, 0.1); }}
        .status-partial {{ background: rgba(202, 138, 4, 0.1); }}
        .status-stub {{ background: rgba(220, 38, 38, 0.1); }}
        
        .status {{
            text-transform: capitalize;
            font-weight: 500;
        }}
        
        .status-complete .status {{ color: var(--success); }}
        .status-partial .status {{ color: var(--warning); }}
        .status-stub .status {{ color: var(--danger); }}
        
        code {{
            background: var(--bg);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.875rem;
        }}
        
        .filter-buttons {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .filter-btn {{
            padding: 0.5rem 1rem;
            border: 2px solid var(--border);
            border-radius: 6px;
            background: var(--card-bg);
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }}
        
        .filter-btn:hover {{
            border-color: var(--primary);
        }}
        
        .filter-btn.active {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>IRH v21.0 Code↔Theory Cross-Reference</h1>
        <p class="subtitle">Generated: {timestamp}</p>
        
        <div class="card">
            <h2>Coverage Summary</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{coverage.coverage_percentage:.0f}%</div>
                    <div class="stat-label">Overall Coverage</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{coverage.total_equations}</div>
                    <div class="stat-label">Total Equations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{coverage.implemented_equations}</div>
                    <div class="stat-label">Implemented</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{coverage.partial_equations}</div>
                    <div class="stat-label">Partial</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Coverage by Section</h2>
            {''.join(section_bars)}
        </div>
        
        <div class="card">
            <h2>Equation Implementations</h2>
            <div class="search-container">
                <input type="text" class="search-input" id="search" 
                       placeholder="Search equations, files, functions...">
            </div>
            <div class="filter-buttons">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="complete">Complete</button>
                <button class="filter-btn" data-filter="partial">Partial</button>
                <button class="filter-btn" data-filter="stub">Stub</button>
            </div>
            <table id="impl-table">
                <thead>
                    <tr>
                        <th>Equation</th>
                        <th>Description</th>
                        <th>File</th>
                        <th>Function</th>
                        <th>Line</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(impl_rows)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Search functionality
        const searchInput = document.getElementById('search');
        const table = document.getElementById('impl-table');
        const rows = table.querySelectorAll('tbody tr');
        
        searchInput.addEventListener('input', function() {{
            const query = this.value.toLowerCase();
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            }});
        }});
        
        // Filter functionality
        const filterBtns = document.querySelectorAll('.filter-btn');
        filterBtns.forEach(btn => {{
            btn.addEventListener('click', function() {{
                filterBtns.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                const filter = this.dataset.filter;
                rows.forEach(row => {{
                    if (filter === 'all') {{
                        row.style.display = '';
                    }} else {{
                        const status = row.querySelector('.status').textContent.toLowerCase();
                        row.style.display = status === filter ? '' : 'none';
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>'''
    
    return html
