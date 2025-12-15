#!/usr/bin/env python3
"""
Run Full Validation Suite for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.6 (Verification Protocols)

This script orchestrates the complete validation pipeline:
1. Theoretical annotation verification
2. Unit tests
3. Integration tests
4. Theoretical invariant tests
5. Convergence tests
6. Benchmark tests
7. Falsification suite

Usage:
    python scripts/run_full_validation_suite.py [--quick]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0
    
    status = "✓ PASSED" if success else "✗ FAILED"
    print(f"\n{description}: {status}\n")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="IRH v21.0 Full Validation Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    
    print("=" * 60)
    print("IRH v21.0 Full Validation Suite")
    print("=" * 60)
    
    results = {}
    
    # 1. Theoretical annotations
    results["Annotations"] = run_command(
        ["python", str(root_dir / "scripts" / "verify_theoretical_annotations.py")],
        "Theoretical Annotation Verification"
    )
    
    # 2. Equation audit
    results["Equations"] = run_command(
        ["python", str(root_dir / "scripts" / "audit_equation_implementations.py")],
        "Equation Implementation Audit"
    )
    
    # 3. Unit tests
    results["Unit Tests"] = run_command(
        ["python", "-m", "pytest", str(root_dir / "tests" / "unit"), "-v"],
        "Unit Tests"
    )
    
    if not args.quick:
        # 4. Integration tests
        results["Integration"] = run_command(
            ["python", "-m", "pytest", str(root_dir / "tests" / "integration"), "-v"],
            "Integration Tests"
        )
        
        # 5. Theoretical invariants
        results["Invariants"] = run_command(
            ["python", "-m", "pytest", str(root_dir / "tests" / "theoretical_invariants"), "-v"],
            "Theoretical Invariant Tests"
        )
        
        # 6. Convergence tests
        results["Convergence"] = run_command(
            ["python", "-m", "pytest", str(root_dir / "tests" / "convergence"), "-v"],
            "Convergence Tests"
        )
        
        # 7. Benchmarks
        results["Benchmarks"] = run_command(
            ["python", "-m", "pytest", str(root_dir / "tests" / "benchmarks"), "-v"],
            "Benchmark Tests"
        )
        
        # 8. Falsification
        results["Falsification"] = run_command(
            ["python", "-m", "pytest", str(root_dir / "tests" / "falsification"), "-v"],
            "Falsification Tests"
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} test suites passed")
    print("=" * 60)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
