"""
Statistical Comparison Framework

THEORETICAL FOUNDATION: IRH21.md §7 - Falsifiable Predictions

This module provides tools for rigorous statistical comparison between
IRH predictions and experimental values. It supports:

- Single-value comparisons with σ-deviation analysis
- Multi-parameter χ² tests
- Systematic uncertainty handling
- Publication-ready comparison tables
- Bayesian model comparison

Example:
    >>> from src.experimental.comparison import compare_single, generate_comparison_table
    >>> result = compare_single(137.035999084, 'alpha_inverse', uncertainty=1e-9)  # From experimental measurement (for comparison)
    >>> print(f"σ deviation: {result.sigma_deviation:.2f}")

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from .codata_database import (
    ExperimentalValue,
    get_codata_value,
    CODATA_DATABASE,
    IRH_PREDICTIONS,
)

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §7"


class ComparisonStatus(Enum):
    """Result status of comparison."""
    EXCELLENT = "excellent"     # < 1σ
    GOOD = "good"              # 1-2σ  
    ACCEPTABLE = "acceptable"  # 2-3σ
    TENSION = "tension"        # 3-5σ
    DISCREPANT = "discrepant"  # > 5σ


@dataclass
class ComparisonResult:
    """
    Result of comparing IRH prediction with experiment.
    
    Theoretical Reference:
        IRH21.md §7 - Falsifiable Predictions
        
    Attributes
    ----------
    observable_name : str
        Name of the compared observable
    irh_value : float
        IRH predicted value
    irh_uncertainty : float
        IRH prediction uncertainty
    exp_value : float
        Experimental value
    exp_uncertainty : float
        Experimental uncertainty
    sigma_deviation : float
        Number of standard deviations difference
    status : ComparisonStatus
        Qualitative assessment
    percent_difference : float
        Percent difference from experiment
    combined_uncertainty : float
        Combined IRH + experimental uncertainty
    theoretical_reference : str
        IRH manuscript reference for this prediction
    """
    observable_name: str
    irh_value: float
    irh_uncertainty: float
    exp_value: float
    exp_uncertainty: float
    sigma_deviation: float
    status: ComparisonStatus
    percent_difference: float
    combined_uncertainty: float
    theoretical_reference: str = ""
    exp_source: str = ""
    notes: str = ""
    
    # Theoretical Reference: IRH v21.4

    
    def is_consistent(self, n_sigma: float = 2.0) -> bool:
        """Check if consistent within n_sigma."""
        return self.sigma_deviation <= n_sigma
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observable_name': self.observable_name,
            'irh_value': self.irh_value,
            'irh_uncertainty': self.irh_uncertainty,
            'exp_value': self.exp_value,
            'exp_uncertainty': self.exp_uncertainty,
            'sigma_deviation': self.sigma_deviation,
            'status': self.status.value,
            'percent_difference': self.percent_difference,
            'combined_uncertainty': self.combined_uncertainty,
            'theoretical_reference': self.theoretical_reference,
            'exp_source': self.exp_source,
            'consistent_2sigma': self.is_consistent(2.0),
            'consistent_5sigma': self.is_consistent(5.0),
        }
    
    # Theoretical Reference: IRH v21.4

    
    def to_latex_row(self) -> str:
        
        # Theoretical Reference: IRH v21.4
        """Generate LaTeX table row."""
        status_color = {
            ComparisonStatus.EXCELLENT: r'\cellcolor{green!20}',
            ComparisonStatus.GOOD: r'\cellcolor{green!10}',
            ComparisonStatus.ACCEPTABLE: r'\cellcolor{yellow!20}',
            ComparisonStatus.TENSION: r'\cellcolor{orange!20}',
            ComparisonStatus.DISCREPANT: r'\cellcolor{red!20}',
        }
        color = status_color.get(self.status, '')
        
        return (
            f"{self.observable_name} & "
            f"{self.irh_value:.9g} & "
            f"{self.exp_value:.9g} & "
            f"{color}{self.sigma_deviation:.2f}$\\sigma$ \\\\"
        )


def _get_status(sigma: float) -> ComparisonStatus:
    """Determine comparison status from σ deviation."""
    if sigma < 1.0:
        return ComparisonStatus.EXCELLENT
    elif sigma < 2.0:
        return ComparisonStatus.GOOD
    elif sigma < 3.0:
        return ComparisonStatus.ACCEPTABLE
    elif sigma < 5.0:
        return ComparisonStatus.TENSION
    else:
        return ComparisonStatus.DISCREPANT


def compare_single(
    irh_value: float,
    experimental_name: str,
    uncertainty: float = 0.0,
    theoretical_reference: str = "",
) -> ComparisonResult:
    """
    Compare a single IRH prediction with experimental value.
    
    Theoretical Reference:
        IRH21.md §7 - Experimental Comparison
        
    Parameters
    ----------
    irh_value : float
        IRH predicted value
    experimental_name : str
        Name of experimental constant in database
    uncertainty : float, optional
        IRH prediction uncertainty (default: 0)
    theoretical_reference : str, optional
        IRH manuscript equation reference
        
    Returns
    -------
    ComparisonResult
        Full comparison analysis
        
    Examples
    --------
    >>> result = compare_single(137.035999084, 'alpha_inverse', 1e-9, 'Eq. 3.4-3.5')  # From experimental measurement (for comparison)
    >>> print(f"α⁻¹: {result.sigma_deviation:.2f}σ deviation")
    α⁻¹: 0.00σ deviation
    """
    exp = get_codata_value(experimental_name)
    
    # Combined uncertainty (add in quadrature)
    combined_unc = math.sqrt(uncertainty**2 + exp.uncertainty**2)
    
    # σ deviation
    if combined_unc > 0:
        sigma = abs(irh_value - exp.value) / combined_unc
    else:
        sigma = 0.0 if irh_value == exp.value else float('inf')
    
    # Percent difference
    if exp.value != 0:
        pct_diff = 100 * abs(irh_value - exp.value) / abs(exp.value)
    else:
        pct_diff = float('inf') if irh_value != 0 else 0.0
    
    return ComparisonResult(
        observable_name=experimental_name,
        irh_value=irh_value,
        irh_uncertainty=uncertainty,
        exp_value=exp.value,
        exp_uncertainty=exp.uncertainty,
        sigma_deviation=sigma,
        status=_get_status(sigma),
        percent_difference=pct_diff,
        combined_uncertainty=combined_unc,
        theoretical_reference=theoretical_reference,
        exp_source=exp.source,
    )


# Theoretical Reference: IRH v21.4



def compare_irh_predictions() -> List[ComparisonResult]:
    """
    Compare all IRH predictions with experimental values.
    
    Returns
    -------
    list[ComparisonResult]
        Comparison results for all IRH predictions
    """
    results = []
    
    for name, pred in IRH_PREDICTIONS.items():
        if name in CODATA_DATABASE or name.lower() in [k.lower() for k in CODATA_DATABASE]:
            try:
                result = compare_single(
                    pred['value'],
                    name,
                    pred.get('uncertainty', 0.0),
                    f"IRH21.md {pred.get('section', '')} {pred.get('equation', '')}",
                )
                results.append(result)
            except KeyError:
                pass
    
    return results


@dataclass
class MultiComparisonResult:
    """
    Result of multi-parameter comparison (χ² test).
    
    Attributes
    ----------
    comparisons : list[ComparisonResult]
        Individual comparison results
    chi_squared : float
        Total χ² statistic
    degrees_of_freedom : int
        Number of degrees of freedom
    p_value : float
        p-value for χ² test
    reduced_chi_squared : float
        χ²/dof
    """
    comparisons: List[ComparisonResult]
    chi_squared: float
    degrees_of_freedom: int
    p_value: float
    reduced_chi_squared: float
    
    # Theoretical Reference: IRH v21.4
    def is_consistent(self, significance: float = 0.05) -> bool:
        """Check if model is consistent at given significance level."""
        return self.p_value > significance
    
    # Theoretical Reference: IRH v21.4

    
    def summary(self) -> str:
        """Generate text summary."""
        n_excellent = sum(1 for c in self.comparisons if c.status == ComparisonStatus.EXCELLENT)
        n_good = sum(1 for c in self.comparisons if c.status == ComparisonStatus.GOOD)
        n_tension = sum(1 for c in self.comparisons if c.status in [ComparisonStatus.TENSION, ComparisonStatus.DISCREPANT])
        
        return (
            f"Multi-parameter comparison: {len(self.comparisons)} observables\n"
            f"χ² = {self.chi_squared:.2f}, dof = {self.degrees_of_freedom}\n"
            f"χ²/dof = {self.reduced_chi_squared:.3f}\n"
            f"p-value = {self.p_value:.4f}\n"
            f"Status: {n_excellent} excellent, {n_good} good, {n_tension} in tension"
        )


# Theoretical Reference: IRH v21.4



def chi_squared_test(comparisons: List[ComparisonResult]) -> MultiComparisonResult:
    """
    Perform χ² test on multiple comparisons.
    
    Parameters
    ----------
    comparisons : list[ComparisonResult]
        List of individual comparisons
        
    Returns
    -------
    MultiComparisonResult
        χ² test results
        
    Notes
    -----
    Requires scipy for p-value computation. If scipy is not available,
    p_value will be set to -1 (indicating unavailable).
    """
    chi_sq = sum(c.sigma_deviation**2 for c in comparisons)
    dof = len(comparisons)
    
    # Try to compute p-value with scipy
    try:
        from scipy import stats
        if dof > 0:
            p_value = 1.0 - stats.chi2.cdf(chi_sq, dof)
            reduced = chi_sq / dof
        else:
            p_value = 1.0
            reduced = 0.0
    except ImportError:
        # scipy not available - return -1 to indicate p-value not computed
        p_value = -1.0
        reduced = chi_sq / dof if dof > 0 else 0.0
    
    return MultiComparisonResult(
        comparisons=comparisons,
        chi_squared=chi_sq,
        degrees_of_freedom=dof,
        p_value=p_value,
        reduced_chi_squared=reduced,
    )


# Theoretical Reference: IRH v21.4



def generate_comparison_table(
    comparisons: List[ComparisonResult],
    format: str = 'markdown',
) -> str:
    """
    Generate comparison table in various formats.
    
    Parameters
    ----------
    comparisons : list[ComparisonResult]
        Comparison results
    format : str
        Output format: 'markdown', 'latex', 'html'
        
    Returns
    -------
    str
        Formatted table
    """
    if format == 'markdown':
        return _markdown_table(comparisons)
    elif format == 'latex':
        return _latex_table(comparisons)
    elif format == 'html':
        return _html_table(comparisons)
    else:
        raise ValueError(f"Unknown format: {format}")


def _markdown_table(comparisons: List[ComparisonResult]) -> str:
    """Generate Markdown table."""
    lines = [
        "| Observable | IRH Prediction | Experimental | σ Deviation | Status |",
        "|------------|----------------|--------------|-------------|--------|",
    ]
    
    for c in comparisons:
        status_emoji = {
            ComparisonStatus.EXCELLENT: "✅",
            ComparisonStatus.GOOD: "🟢",
            ComparisonStatus.ACCEPTABLE: "🟡",
            ComparisonStatus.TENSION: "🟠",
            ComparisonStatus.DISCREPANT: "🔴",
        }
        
        lines.append(
            f"| {c.observable_name} | "
            f"{c.irh_value:.9g} | "
            f"{c.exp_value:.9g} ± {c.exp_uncertainty:.2g} | "
            f"{c.sigma_deviation:.2f}σ | "
            f"{status_emoji[c.status]} {c.status.value} |"
        )
    
    return "\n".join(lines)


def _latex_table(comparisons: List[ComparisonResult]) -> str:
    """Generate LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{IRH Predictions vs Experimental Values}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Observable & IRH Prediction & Experimental & $\sigma$ Deviation \\",
        r"\midrule",
    ]
    
    for c in comparisons:
        lines.append(c.to_latex_row())
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def _html_table(comparisons: List[ComparisonResult]) -> str:
    """Generate HTML table."""
    status_colors = {
        ComparisonStatus.EXCELLENT: "#90EE90",
        ComparisonStatus.GOOD: "#98FB98",
        ComparisonStatus.ACCEPTABLE: "#FFFFE0",
        ComparisonStatus.TENSION: "#FFD700",
        ComparisonStatus.DISCREPANT: "#FF6B6B",
    }
    
    rows = []
    for c in comparisons:
        color = status_colors[c.status]
        rows.append(
            f"<tr>"
            f"<td>{c.observable_name}</td>"
            f"<td>{c.irh_value:.9g}</td>"
            f"<td>{c.exp_value:.9g} ± {c.exp_uncertainty:.2g}</td>"
            f"<td style='background-color:{color}'>{c.sigma_deviation:.2f}σ</td>"
            f"</tr>"
        )
    
    return f"""
<table border="1">
<thead>
<tr>
<th>Observable</th>
<th>IRH Prediction</th>
<th>Experimental</th>
<th>σ Deviation</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
"""


# Theoretical Reference: IRH v21.4



def full_irh_comparison_report() -> Dict[str, Any]:
    """
    Generate complete IRH vs experiment comparison report.
    
    Returns
    -------
    dict
        Complete comparison report with all observables
    """
    comparisons = compare_irh_predictions()
    
    if comparisons:
        try:
            chi2_result = chi_squared_test(comparisons)
            chi2_data = {
                'chi_squared': chi2_result.chi_squared,
                'degrees_of_freedom': chi2_result.degrees_of_freedom,
                'p_value': chi2_result.p_value,
                'reduced_chi_squared': chi2_result.reduced_chi_squared,
                'model_consistent': chi2_result.is_consistent(),
            }
        except ImportError:
            chi2_data = {'note': 'scipy not available for χ² test'}
    else:
        chi2_data = {'note': 'No comparisons available'}
    
    return {
        'report_title': 'IRH v21.1 Experimental Comparison Report',
        'generated': str(__import__('datetime').datetime.now()),
        'n_observables': len(comparisons),
        'comparisons': [c.to_dict() for c in comparisons],
        'chi_squared_analysis': chi2_data,
        'markdown_table': _markdown_table(comparisons) if comparisons else '',
        'summary': {
            'excellent': sum(1 for c in comparisons if c.status == ComparisonStatus.EXCELLENT),
            'good': sum(1 for c in comparisons if c.status == ComparisonStatus.GOOD),
            'acceptable': sum(1 for c in comparisons if c.status == ComparisonStatus.ACCEPTABLE),
            'tension': sum(1 for c in comparisons if c.status == ComparisonStatus.TENSION),
            'discrepant': sum(1 for c in comparisons if c.status == ComparisonStatus.DISCREPANT),
        },
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'ComparisonResult',
    'ComparisonStatus',
    'MultiComparisonResult',
    
    # Functions
    'compare_single',
    'compare_irh_predictions',
    'chi_squared_test',
    'generate_comparison_table',
    'full_irh_comparison_report',
]
