"""
Failure Analysis and ML-Based Refactoring Suggestions

THEORETICAL FOUNDATION: IRH v21.4 Computational Transparency Mandate

This module provides automatic failure detection, logging, and ML-based
suggestions for fixing computational failures in IRH framework.

Key Features:
- Automatic failure detection with context capture
- JSON-formatted failure logs for machine analysis
- ML-based pattern recognition for common failure modes
- Integration with Gemini API (Colab) for AI-assisted debugging
- Automatic git push of failure logs to repository

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import warnings

__version__ = "21.4.0"
__theoretical_foundation__ = "IRH v21.4 Computational Transparency Mandate"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FailureContext:
    """
    Context information for a computational failure.
    
    Attributes
    ----------
    timestamp : str
        ISO 8601 timestamp of failure
    computation : str
        Name of the computation that failed
    theoretical_ref : str
        IRH manuscript reference (e.g., "IRH v21.4 Part 1 §1.2, Eq. 1.12")
    error_type : str
        Type of error (e.g., "ConvergenceError", "ValueError")
    error_message : str
        Error message
    parameters : dict
        Input parameters that caused failure
    stack_trace : str
        Full stack trace
    context : dict
        Additional context (session_id, notebook, cell_number, etc.)
    suggested_fixes : list
        List of suggested fixes (populated by ML analysis)
    """
    timestamp: str
    computation: str
    theoretical_ref: str
    error_type: str
    error_message: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# Failure Logger
# =============================================================================


class FailureLogger:
    """
    Logger for computational failures with automatic analysis.
    
    Theoretical Reference:
        IRH v21.4 Computational Transparency Mandate
    
    Parameters
    ----------
    output_dir : str or Path
        Directory for failure logs (default: "io/failures")
    auto_push : bool
        Whether to automatically push failures to git (default: False)
    verbose : bool
        Whether to print failure summaries (default: True)
    """
    
    def __init__(
        self,
        output_dir: str | Path = "io/failures",
        auto_push: bool = False,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_push = auto_push
        self.verbose = verbose
        self._failure_count = 0
    
    def log_failure(
        self,
        computation: str,
        error: Exception,
        parameters: Optional[Dict[str, Any]] = None,
        theoretical_ref: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Log a computational failure.
        
        Parameters
        ----------
        computation : str
            Name of the computation that failed
        error : Exception
            The exception that was raised
        parameters : dict, optional
            Input parameters that caused the failure
        theoretical_ref : str, optional
            IRH manuscript reference
        context : dict, optional
            Additional context information
        
        Returns
        -------
        Path
            Path to the failure log file
        
        Examples
        --------
        >>> logger = FailureLogger()
        >>> try:
        ...     result = integrate_rg_flow(initial=[50, 100, 150])
        ... except Exception as e:
        ...     log_path = logger.log_failure(
        ...         computation="rg_flow_integration",
        ...         error=e,
        ...         parameters={"initial": [50, 100, 150]},
        ...         theoretical_ref="IRH v21.4 Part 1 §1.2, Eq. 1.12"
        ...     )
        """
        self._failure_count += 1
        
        # Create failure context
        failure = FailureContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            computation=computation,
            theoretical_ref=theoretical_ref,
            error_type=type(error).__name__,
            error_message=str(error),
            parameters=parameters or {},
            stack_trace=traceback.format_exc(),
            context=context or {},
            suggested_fixes=self._analyze_failure(error, parameters)
        )
        
        # Generate filename using UTC timestamp for consistency
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{computation}_{timestamp_str}.json"
        filepath = self.output_dir / filename
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(failure.to_json())
        
        # Also write as "latest.json" for easy access
        latest_path = self.output_dir / "latest.json"
        with open(latest_path, 'w') as f:
            f.write(failure.to_json())
        
        if self.verbose:
            self._print_failure_summary(failure, filepath)
        
        # Auto-push if enabled
        if self.auto_push:
            self._push_to_git(filepath)
        
        return filepath
    
    def _analyze_failure(
        self,
        error: Exception,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Analyze failure and suggest fixes using pattern matching.
        
        This is a simple pattern-based analyzer. For more sophisticated
        analysis, use Gemini API (see FailureAnalyzer class).
        """
        suggestions = []
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # Common RG flow issues
        if "convergence" in error_msg or "diverge" in error_msg:
            suggestions.extend([
                "Try using 'Radau' solver instead of 'RK45' (better for stiff equations)",
                "Reduce t_range: use (-1, 1) instead of (-10, 10)",
                "Start closer to fixed point: perturb by 5% instead of 20%",
                "Increase max_steps or decrease rtol/atol"
            ])
        
        # Numerical stability
        if "overflow" in error_msg or "inf" in error_msg:
            suggestions.extend([
                "Couplings may be growing too large - check initial conditions",
                "Use log-scale for coupling evolution",
                "Add bounds checking: clip couplings to reasonable range"
            ])
        
        # Matrix/linear algebra issues
        if "singular" in error_msg or "ill-conditioned" in error_msg:
            suggestions.extend([
                "Matrix may be degenerate - add regularization",
                "Check for zero eigenvalues",
                "Use SVD instead of direct inversion"
            ])
        
        # Dimension mismatches
        if "shape" in error_msg or "dimension" in error_msg:
            suggestions.extend([
                "Check array dimensions match expected sizes",
                "Verify broadcasting rules are satisfied",
                "Print intermediate shapes for debugging"
            ])
        
        # Missing data
        if "none" in error_msg and error_type == "TypeError":
            suggestions.extend([
                "Check that all required parameters are provided",
                "Verify function returned a value (not None)",
                "Add explicit None checks before operations"
            ])
        
        return suggestions
    
    def _print_failure_summary(self, failure: FailureContext, filepath: Path):
        """Print a formatted failure summary."""
        print("\n" + "="*80)
        print("⚠️  COMPUTATION FAILURE LOGGED")
        print("="*80)
        print(f"Computation: {failure.computation}")
        print(f"Error: {failure.error_type}: {failure.error_message}")
        print(f"Theoretical Ref: {failure.theoretical_ref}")
        print(f"Log File: {filepath}")
        
        if failure.suggested_fixes:
            print("\n💡 Suggested Fixes:")
            for i, fix in enumerate(failure.suggested_fixes, 1):
                print(f"  {i}. {fix}")
        
        print("="*80 + "\n")
    
    def _push_to_git(self, filepath: Path):
        """Push failure log to git repository (if in git repo)."""
        try:
            import subprocess
            
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=filepath.parent
            )
            
            if result.returncode == 0:
                # Add, commit, and push
                subprocess.run(["git", "add", str(filepath)], cwd=filepath.parent)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subprocess.run(
                    ["git", "commit", "-m", f"Failure log: {timestamp}"],
                    cwd=filepath.parent
                )
                subprocess.run(["git", "push"], cwd=filepath.parent)
                
                if self.verbose:
                    print(f"✅ Failure log pushed to repository")
        
        except Exception as e:
            warnings.warn(f"Could not push to git: {e}")
    
    @property
    def failure_count(self) -> int:
        """Total number of failures logged in this session."""
        return self._failure_count


# =============================================================================
# ML-Based Failure Analyzer (with Gemini Integration)
# =============================================================================


class FailureAnalyzer:
    """
    ML-based failure analyzer with Gemini API integration.
    
    This class provides more sophisticated failure analysis using
    Large Language Models (specifically Google Gemini in Colab).
    
    Parameters
    ----------
    failure_log_path : str or Path
        Path to failure log JSON file
    use_gemini : bool
        Whether to use Gemini API (requires Colab environment)
    """
    
    def __init__(
        self,
        failure_log_path: str | Path,
        use_gemini: bool = True
    ):
        self.failure_log_path = Path(failure_log_path)
        self.use_gemini = use_gemini and self._is_colab()
        
        # Load failure context
        with open(self.failure_log_path) as f:
            self.failure = FailureContext(**json.load(f))
    
    @staticmethod
    def _is_colab() -> bool:
        """Check if running in Google Colab."""
        return 'google.colab' in sys.modules
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze failure and generate comprehensive report.
        
        Returns
        -------
        dict
            Analysis report with suggested fixes
        """
        report = {
            "failure_summary": {
                "computation": self.failure.computation,
                "error": f"{self.failure.error_type}: {self.failure.error_message}",
                "theoretical_ref": self.failure.theoretical_ref
            },
            "pattern_based_suggestions": self.failure.suggested_fixes,
            "gemini_suggestions": []
        }
        
        # Add Gemini-based analysis if available
        if self.use_gemini:
            try:
                report["gemini_suggestions"] = self._get_gemini_suggestions()
            except Exception as e:
                report["gemini_error"] = str(e)
        
        return report
    
    def _get_gemini_suggestions(self) -> List[str]:
        """
        Get suggestions from Gemini API.
        
        Note: This requires the google.colab.gemini module.
        """
        try:
            from google.colab import gemini
        except ImportError:
            return ["Gemini API not available (not in Colab environment)"]
        
        # Construct prompt
        prompt = f"""
You are analyzing a computational failure in the Intrinsic Resonance Holography (IRH) 
theoretical physics framework.

**Failure Details:**
- Computation: {self.failure.computation}
- Theoretical Reference: {self.failure.theoretical_ref}
- Error Type: {self.failure.error_type}
- Error Message: {self.failure.error_message}

**Parameters:**
```json
{json.dumps(self.failure.parameters, indent=2)}
```

**Stack Trace:**
```
{self.failure.stack_trace[:1000]}...
```

**Task:**
Provide 3-5 specific, actionable code refactoring suggestions to fix this issue.
Focus on:
1. Numerical stability improvements
2. Parameter adjustments based on theoretical constraints
3. Algorithm alternatives
4. Error handling improvements

Format your response as a numbered list of concrete suggestions.
"""
        
        # Get Gemini response
        try:
            response = gemini.generate(prompt)
            
            # Parse response into list of suggestions
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove leading number/bullet
                    suggestion = line.lstrip('0123456789.-) ')
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions
        
        except Exception as e:
            return [f"Gemini API error: {str(e)}"]
    
    def generate_refactoring_code(self) -> str:
        """
        Generate example refactored code based on suggestions.
        
        Returns
        -------
        str
            Python code implementing suggested fixes
        """
        # This is a template - would need to be more sophisticated
        # for real automated refactoring
        
        code = f"""
# Refactored code for: {self.failure.computation}
# Based on failure analysis: {self.failure.timestamp}

# Original parameters that failed:
# {json.dumps(self.failure.parameters, indent=2)}

# Suggested refactoring:
"""
        
        for i, suggestion in enumerate(self.failure.suggested_fixes, 1):
            code += f"\n# {i}. {suggestion}\n"
        
        # Add specific code examples based on computation type
        if "rg_flow" in self.failure.computation.lower():
            code += """
# Example: More robust RG flow integration
from scipy.integrate import solve_ivp

def integrate_rg_flow_robust(initial, t_range=(-1, 1)):
    '''
    More robust RG flow integration with improved numerics.
    '''
    from src.rg_flow.beta_functions import beta_lambda, beta_gamma, beta_mu
    
    def rhs(t, y):
        lambda_t, gamma_t, mu_t = y
        return [
            beta_lambda(lambda_t, gamma_t, mu_t),
            beta_gamma(lambda_t, gamma_t, mu_t),
            beta_mu(lambda_t, gamma_t, mu_t)
        ]
    
    # Use Radau (better for stiff equations)
    result = solve_ivp(
        rhs,
        t_range,
        initial,
        method='Radau',  # Changed from RK45
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1  # Limit step size
    )
    
    return result
"""
        
        return code


# =============================================================================
# Convenience Functions
# =============================================================================


def log_failure(
    computation: str,
    error: Exception,
    parameters: Optional[Dict[str, Any]] = None,
    theoretical_ref: str = "",
    context: Optional[Dict[str, Any]] = None,
    output_dir: str | Path = "io/failures",
    auto_push: bool = False
) -> Path:
    """
    Convenience function to log a failure.
    
    Parameters
    ----------
    computation : str
        Name of the computation
    error : Exception
        The exception that was raised
    parameters : dict, optional
        Input parameters
    theoretical_ref : str, optional
        IRH manuscript reference
    context : dict, optional
        Additional context
    output_dir : str or Path
        Output directory for logs
    auto_push : bool
        Whether to auto-push to git
    
    Returns
    -------
    Path
        Path to the log file
    
    Examples
    --------
    >>> try:
    ...     result = compute_alpha_inverse()
    ... except Exception as e:
    ...     log_failure(
    ...         "alpha_inverse_calculation",
    ...         e,
    ...         theoretical_ref="IRH v21.4 Part 1 §3.2.2, Eq. 3.4"
    ...     )
    """
    logger = FailureLogger(output_dir=output_dir, auto_push=auto_push)
    return logger.log_failure(
        computation=computation,
        error=error,
        parameters=parameters,
        theoretical_ref=theoretical_ref,
        context=context
    )


def analyze_latest_failure(
    failure_dir: str | Path = "io/failures",
    use_gemini: bool = True
) -> Dict[str, Any]:
    """
    Analyze the most recent failure.
    
    Parameters
    ----------
    failure_dir : str or Path
        Directory containing failure logs
    use_gemini : bool
        Whether to use Gemini API
    
    Returns
    -------
    dict
        Analysis report
    
    Examples
    --------
    >>> report = analyze_latest_failure()
    >>> print(report["pattern_based_suggestions"])
    >>> if "gemini_suggestions" in report:
    ...     print(report["gemini_suggestions"])
    """
    failure_dir = Path(failure_dir)
    latest_path = failure_dir / "latest.json"
    
    if not latest_path.exists():
        return {"error": "No failure logs found"}
    
    analyzer = FailureAnalyzer(latest_path, use_gemini=use_gemini)
    return analyzer.analyze()


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    # Example: Log a failure
    logger = FailureLogger(verbose=True, auto_push=False)
    
    try:
        # Simulate a computation that fails
        def failing_computation():
            raise ValueError("RG flow integration failed to converge")
        
        failing_computation()
    
    except Exception as e:
        log_path = logger.log_failure(
            computation="rg_flow_integration",
            error=e,
            parameters={"initial": [50, 100, 150], "t_range": [-10, 10]},
            theoretical_ref="IRH v21.4 Part 1 §1.2, Eq. 1.12",
            context={"notebook": "test", "cell": 1}
        )
        
        print(f"\nFailure logged to: {log_path}")
        print(f"Total failures in session: {logger.failure_count}")
        
        # Analyze the failure
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)
        
        report = analyze_latest_failure(use_gemini=False)
        print(json.dumps(report, indent=2))
