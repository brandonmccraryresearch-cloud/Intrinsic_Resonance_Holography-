"""
IRH Desktop - Result Exporter

Exports computation results in various formats:
- JSON (structured data)
- CSV (tabular data)
- PDF (formatted reports)
- HTML (web-viewable)
- LaTeX (for publications)

This module implements Phase 4 of the DEB_PACKAGE_ROADMAP.md:
- Result display and export

Author: Brandon D. McCrary
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from io import StringIO

from irh_desktop.core.computation_runner import (
    ComputationResult,
    ComputationType,
    COMPUTATION_INFO,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """
    Options for result export.
    
    Attributes
    ----------
    include_metadata : bool
        Include timestamp, version, etc.
    include_verification : bool
        Include verification status
    include_reference : bool
        Include IRH21.md references
    precision : int
        Decimal places for floats
    format_equations : bool
        Format equations for display
    """
    include_metadata: bool = True
    include_verification: bool = True
    include_reference: bool = True
    precision: int = 12
    format_equations: bool = True


class ResultExporter:
    """
    Exports IRH computation results in various formats.
    
    Supports JSON, CSV, HTML, and plain text export with
    configurable options for scientific publication.
    
    Parameters
    ----------
    default_options : ExportOptions
        Default export options
        
    Examples
    --------
    >>> exporter = ResultExporter()
    >>> exporter.export_json(result, "results.json")
    >>> exporter.export_csv(results, "data.csv")
    >>> html = exporter.to_html(result)
    
    Theoretical Foundation
    ----------------------
    Implements result export as specified in
    docs/DEB_PACKAGE_ROADMAP.md §4 "Computation Interface"
    """
    
    def __init__(self, default_options: Optional[ExportOptions] = None):
        """Initialize the Result Exporter."""
        self.default_options = default_options or ExportOptions()
    
    def export_json(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export results to JSON file.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to export
        path : str or Path
            Output file path
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        bool
            True if export succeeded
        """
        opts = options or self.default_options
        
        try:
            data = self._format_for_json(result, opts)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported JSON to {path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    def export_csv(
        self,
        results: List[ComputationResult],
        path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export multiple results to CSV file.
        
        Parameters
        ----------
        results : List[ComputationResult]
            Results to export
        path : str or Path
            Output file path
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        bool
            True if export succeeded
        """
        opts = options or self.default_options
        
        try:
            # Collect all value keys
            all_keys = set()
            for result in results:
                all_keys.update(result.values.keys())
            
            all_keys = sorted(all_keys)
            
            # Build header
            header = ["timestamp", "success", "duration_s"]
            if opts.include_reference:
                header.append("reference")
            header.extend(all_keys)
            
            # Build rows
            rows = []
            for result in results:
                row = [
                    result.timestamp.isoformat() if result.timestamp else "",
                    result.success,
                    f"{result.duration_seconds:.4f}",
                ]
                
                if opts.include_reference:
                    row.append(result.reference)
                
                for key in all_keys:
                    value = result.values.get(key, "")
                    if isinstance(value, float):
                        value = f"{value:.{opts.precision}g}"
                    row.append(str(value))
                
                rows.append(row)
            
            # Write CSV
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            
            logger.info(f"Exported CSV to {path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def export_html(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        path: Union[str, Path],
        title: str = "IRH Computation Results",
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export results to HTML file.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to export
        path : str or Path
            Output file path
        title : str
            Page title
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        bool
            True if export succeeded
        """
        opts = options or self.default_options
        
        try:
            html = self.to_html(result, title=title, options=opts)
            
            with open(path, 'w') as f:
                f.write(html)
            
            logger.info(f"Exported HTML to {path}")
            return True
            
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            return False
    
    def export_text(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export results to plain text file.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to export
        path : str or Path
            Output file path
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        bool
            True if export succeeded
        """
        opts = options or self.default_options
        
        try:
            text = self.to_text(result, options=opts)
            
            with open(path, 'w') as f:
                f.write(text)
            
            logger.info(f"Exported text to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Text export failed: {e}")
            return False
    
    def export_latex(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export results to LaTeX file.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to export
        path : str or Path
            Output file path
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        bool
            True if export succeeded
        """
        opts = options or self.default_options
        
        try:
            latex = self.to_latex(result, options=opts)
            
            with open(path, 'w') as f:
                f.write(latex)
            
            logger.info(f"Exported LaTeX to {path}")
            return True
            
        except Exception as e:
            logger.error(f"LaTeX export failed: {e}")
            return False
    
    def _format_for_json(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        opts: ExportOptions
    ) -> Dict[str, Any]:
        """Format results for JSON export."""
        if isinstance(result, list):
            results_data = [self._result_to_dict(r, opts) for r in result]
            data = {"results": results_data, "count": len(results_data)}
        else:
            data = {"result": self._result_to_dict(result, opts)}
        
        if opts.include_metadata:
            data["metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "irh_version": "21.0.0",
                "format": "json",
            }
        
        return data
    
    def _result_to_dict(
        self,
        result: ComputationResult,
        opts: ExportOptions
    ) -> Dict[str, Any]:
        """Convert a single result to dictionary."""
        data = {
            "success": result.success,
            "values": self._format_values(result.values, opts),
            "duration_seconds": result.duration_seconds,
        }
        
        if opts.include_verification:
            data["verification"] = result.verification
        
        if opts.include_reference and result.reference:
            data["reference"] = result.reference
        
        if result.timestamp:
            data["timestamp"] = result.timestamp.isoformat()
        
        if result.error:
            data["error"] = result.error
        
        return data
    
    def _format_values(
        self,
        values: Dict[str, Any],
        opts: ExportOptions
    ) -> Dict[str, Any]:
        """Format numerical values with precision."""
        formatted = {}
        for key, value in values.items():
            if isinstance(value, float):
                # Keep full precision in JSON, round for display
                formatted[key] = value
            elif isinstance(value, dict):
                formatted[key] = self._format_values(value, opts)
            else:
                formatted[key] = value
        return formatted
    
    def to_json(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        options: Optional[ExportOptions] = None
    ) -> str:
        """
        Convert results to JSON string.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to convert
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        str
            JSON string
        """
        opts = options or self.default_options
        data = self._format_for_json(result, opts)
        return json.dumps(data, indent=2, default=str)
    
    def to_html(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        title: str = "IRH Computation Results",
        options: Optional[ExportOptions] = None
    ) -> str:
        """
        Convert results to HTML string.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to convert
        title : str
            Page title
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        str
            HTML string
        """
        opts = options or self.default_options
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ color: #569cd6; border-bottom: 2px solid #3a3a3a; padding-bottom: 10px; }}
        h2 {{ color: #4ec9b0; margin-top: 30px; }}
        .result-card {{
            background: #252526;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .success {{ border-left: 4px solid #4caf50; }}
        .failed {{ border-left: 4px solid #f44336; }}
        .value-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .value-table th, .value-table td {{
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid #3a3a3a;
        }}
        .value-table th {{
            color: #569cd6;
            background: #2d2d30;
        }}
        .reference {{
            color: #ce9178;
            font-style: italic;
            margin-top: 10px;
        }}
        .verification {{
            margin-top: 15px;
            padding: 10px;
            background: #2d2d30;
            border-radius: 4px;
        }}
        .check {{ color: #4caf50; }}
        .cross {{ color: #f44336; }}
        .metadata {{
            color: #808080;
            font-size: 0.9em;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid #3a3a3a;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""
        
        results = result if isinstance(result, list) else [result]
        
        for i, r in enumerate(results, 1):
            status_class = "success" if r.success else "failed"
            status_icon = "✓" if r.success else "✗"
            status_color = "check" if r.success else "cross"
            
            html += f"""
    <div class="result-card {status_class}">
        <h2>Result {i} <span class="{status_color}">{status_icon}</span></h2>
"""
            
            if opts.include_reference and r.reference:
                html += f'        <div class="reference">Reference: {r.reference}</div>\n'
            
            # Values table
            if r.values:
                html += """        <table class="value-table">
            <thead>
                <tr><th>Parameter</th><th>Value</th></tr>
            </thead>
            <tbody>
"""
                for key, value in r.values.items():
                    if isinstance(value, float):
                        value_str = f"{value:.{opts.precision}g}"
                    elif isinstance(value, dict):
                        value_str = json.dumps(value, default=str)
                    else:
                        value_str = str(value)
                    html += f"                <tr><td>{key}</td><td>{value_str}</td></tr>\n"
                
                html += """            </tbody>
        </table>
"""
            
            # Verification
            if opts.include_verification and r.verification:
                html += '        <div class="verification">\n'
                html += '            <strong>Verification:</strong><br>\n'
                for key, passed in r.verification.items():
                    icon = "✓" if passed else "✗"
                    color = "check" if passed else "cross"
                    html += f'            <span class="{color}">{icon}</span> {key}<br>\n'
                html += '        </div>\n'
            
            html += f"""        <div class="metadata">
            Duration: {r.duration_seconds:.4f}s
            | Timestamp: {r.timestamp.isoformat() if r.timestamp else 'N/A'}
        </div>
    </div>
"""
        
        if opts.include_metadata:
            html += f"""
    <div class="metadata">
        Generated by IRH Desktop v21.0 | {datetime.now():%Y-%m-%d %H:%M:%S}
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def to_text(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        options: Optional[ExportOptions] = None
    ) -> str:
        """
        Convert results to plain text string.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to convert
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        str
            Plain text string
        """
        opts = options or self.default_options
        lines = []
        
        lines.append("=" * 70)
        lines.append("IRH v21.0 Computation Results")
        lines.append("=" * 70)
        lines.append("")
        
        results = result if isinstance(result, list) else [result]
        
        for i, r in enumerate(results, 1):
            status = "SUCCESS" if r.success else "FAILED"
            lines.append(f"Result {i}: {status}")
            lines.append("-" * 40)
            
            if opts.include_reference and r.reference:
                lines.append(f"Reference: {r.reference}")
                lines.append("")
            
            lines.append("Values:")
            for key, value in r.values.items():
                if isinstance(value, float):
                    value_str = f"{value:.{opts.precision}g}"
                elif isinstance(value, dict):
                    value_str = json.dumps(value, default=str)
                else:
                    value_str = str(value)
                lines.append(f"  {key}: {value_str}")
            
            if opts.include_verification and r.verification:
                lines.append("")
                lines.append("Verification:")
                for key, passed in r.verification.items():
                    icon = "[PASS]" if passed else "[FAIL]"
                    lines.append(f"  {icon} {key}")
            
            lines.append("")
            lines.append(f"Duration: {r.duration_seconds:.4f} seconds")
            if r.timestamp:
                lines.append(f"Timestamp: {r.timestamp.isoformat()}")
            lines.append("")
        
        if opts.include_metadata:
            lines.append("=" * 70)
            lines.append(f"Generated by IRH Desktop v21.0")
            lines.append(f"Export time: {datetime.now():%Y-%m-%d %H:%M:%S}")
            lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_latex(
        self,
        result: Union[ComputationResult, List[ComputationResult]],
        options: Optional[ExportOptions] = None
    ) -> str:
        """
        Convert results to LaTeX string.
        
        Parameters
        ----------
        result : ComputationResult or List[ComputationResult]
            Result(s) to convert
        options : ExportOptions, optional
            Export options
            
        Returns
        -------
        str
            LaTeX string
        """
        opts = options or self.default_options
        
        latex = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}

\title{IRH v21.0 Computation Results}
\author{Generated by IRH Desktop}
\date{\today}

\begin{document}
\maketitle

"""
        
        results = result if isinstance(result, list) else [result]
        
        for i, r in enumerate(results, 1):
            status = r"$\checkmark$ Success" if r.success else r"$\times$ Failed"
            
            latex += f"\\section{{Result {i} ({status})}}\n\n"
            
            if opts.include_reference and r.reference:
                latex += f"\\textit{{Reference: {r.reference}}}\n\n"
            
            # Values table
            if r.values:
                latex += "\\begin{table}[h]\n"
                latex += "\\centering\n"
                latex += "\\begin{tabular}{ll}\n"
                latex += "\\toprule\n"
                latex += "Parameter & Value \\\\\n"
                latex += "\\midrule\n"
                
                for key, value in r.values.items():
                    # LaTeX-safe key
                    safe_key = key.replace("_", r"\_")
                    
                    if isinstance(value, float):
                        # Scientific notation for very small/large numbers
                        if abs(value) < 1e-4 or abs(value) > 1e6:
                            value_str = f"${value:.{opts.precision}e}$"
                        else:
                            value_str = f"${value:.{opts.precision}g}$"
                    elif isinstance(value, bool):
                        value_str = "True" if value else "False"
                    else:
                        value_str = str(value).replace("_", r"\_")
                    
                    latex += f"{safe_key} & {value_str} \\\\\n"
                
                latex += "\\bottomrule\n"
                latex += "\\end{tabular}\n"
                latex += "\\caption{Computed values}\n"
                latex += "\\end{table}\n\n"
            
            # Verification
            if opts.include_verification and r.verification:
                latex += "\\textbf{Verification:}\n"
                latex += "\\begin{itemize}\n"
                for key, passed in r.verification.items():
                    icon = r"$\checkmark$" if passed else r"$\times$"
                    safe_key = key.replace("_", r"\_")
                    latex += f"\\item {icon} {safe_key}\n"
                latex += "\\end{itemize}\n\n"
            
            latex += f"Duration: {r.duration_seconds:.4f} seconds\n\n"
        
        latex += r"\end{document}"
        
        return latex


# Convenience functions
def export_results(
    results: Union[ComputationResult, List[ComputationResult]],
    path: Union[str, Path],
    format: str = "json"
) -> bool:
    """
    Export results to file.
    
    Parameters
    ----------
    results : ComputationResult or List[ComputationResult]
        Results to export
    path : str or Path
        Output file path
    format : str
        Export format ('json', 'csv', 'html', 'text', 'latex')
        
    Returns
    -------
    bool
        True if export succeeded
    """
    exporter = ResultExporter()
    
    format = format.lower()
    
    if format == "json":
        return exporter.export_json(results, path)
    elif format == "csv":
        if not isinstance(results, list):
            results = [results]
        return exporter.export_csv(results, path)
    elif format == "html":
        return exporter.export_html(results, path)
    elif format in ("text", "txt"):
        return exporter.export_text(results, path)
    elif format in ("latex", "tex"):
        return exporter.export_latex(results, path)
    else:
        logger.error(f"Unknown export format: {format}")
        return False
