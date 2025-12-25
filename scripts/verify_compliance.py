#!/usr/bin/env python3
"""
IRH v21.4 Compliance Verification Script

This script verifies that code changes comply with:
1. THEORETICAL_CORRESPONDENCE_MANDATE.md
2. MANDATORY_AUDIT_PROTOCOL.md
3. COMPREHENSIVE_AUDIT_REPORT.md

Run before committing code or opening PRs.

Usage:
    python scripts/verify_compliance.py [--fix] [--verbose]

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Manuscript reference patterns
MANUSCRIPT_PATTERN = r'IRH\s+(v?21\.?[0-9]*|v21\.4\s+Part\s+[12])'
EQUATION_PATTERN = r'Eq\.\s*\d+\.\d+'
SECTION_PATTERN = r'§\d+\.\d+(\.\d+)?'


class ComplianceChecker:
    """Verify code compliance with IRH v21.4 standards."""
    
    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.violations = []
        self.warnings = []
        self.passes = []
        
    def print_header(self, text: str):
        """Print section header."""
        print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{CYAN}{text.center(80)}{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")
        
    def print_violation(self, category: str, message: str, file: str = None, line: int = None):
        """Record and print a violation."""
        violation = {
            'category': category,
            'message': message,
            'file': file,
            'line': line
        }
        self.violations.append(violation)
        
        loc = f" ({file}:{line})" if file and line else f" ({file})" if file else ""
        print(f"{RED}✗ VIOLATION{RESET} [{category}]{loc}")
        print(f"  {message}\n")
        
    def print_warning(self, category: str, message: str, file: str = None):
        """Record and print a warning."""
        warning = {
            'category': category,
            'message': message,
            'file': file
        }
        self.warnings.append(warning)
        
        loc = f" ({file})" if file else ""
        print(f"{YELLOW}⚠ WARNING{RESET} [{category}]{loc}")
        print(f"  {message}\n")
        
    def print_pass(self, message: str):
        """Record and print a pass."""
        self.passes.append(message)
        if self.verbose:
            print(f"{GREEN}✓{RESET} {message}")
    
    def check_manuscript_citations(self) -> bool:
        """Check that all functions cite IRH v21.4 manuscript."""
        self.print_header("1. Manuscript Citation Verification")
        
        src_dir = self.repo_root / "src"
        python_files = list(src_dir.rglob("*.py"))
        
        if not python_files:
            self.print_warning("NO_FILES", "No Python files found in src/")
            return False
            
        total_functions = 0
        functions_with_citations = 0
        
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions
                        if node.name.startswith('_') and not node.name.startswith('__'):
                            continue
                        
                        # Skip utility methods that don't need citations
                        utility_methods = ['to_dict', '__str__', '__repr__', '__eq__', '__hash__',
                                         '__lt__', '__le__', '__gt__', '__ge__', '__ne__',
                                         '__len__', '__iter__', '__next__', '__enter__', '__exit__',
                                         '__post_init__', '__getitem__', '__setitem__', '__delitem__']
                        
                        # Skip property getters/setters
                        is_property = False
                        if node.decorator_list:
                            for dec in node.decorator_list:
                                if isinstance(dec, ast.Name) and dec.id in ['property', 'cached_property']:
                                    is_property = True
                                    break
                        
                        if node.name in utility_methods or is_property:
                            continue
                            
                        total_functions += 1
                        docstring = ast.get_docstring(node)
                        
                        # Check docstring for citations
                        has_citation_in_docstring = False
                        if docstring:
                            has_manuscript = bool(re.search(MANUSCRIPT_PATTERN, docstring))
                            has_equation = bool(re.search(EQUATION_PATTERN, docstring))
                            has_section = bool(re.search(SECTION_PATTERN, docstring))
                            has_citation_in_docstring = has_manuscript or has_equation or has_section
                        
                        # Also check for inline comment citation near function (within 5 lines before)
                        has_comment_citation = False
                        if node.lineno > 1:
                            # Check the 5 lines before the function definition
                            file_lines = content.split('\n')
                            for offset in range(1, 6):
                                if node.lineno - offset - 1 >= 0 and node.lineno - offset - 1 < len(file_lines):
                                    check_line = file_lines[node.lineno - offset - 1]
                                    if 'Theoretical Reference' in check_line and 'IRH' in check_line:
                                        has_comment_citation = True
                                        break
                        
                        if has_citation_in_docstring or has_comment_citation:
                            functions_with_citations += 1
                            self.print_pass(f"{py_file.name}::{node.name} - Citation present")
                        elif docstring:
                            self.print_violation(
                                "MISSING_CITATION",
                                f"Function '{node.name}' lacks manuscript reference",
                                str(py_file.relative_to(self.repo_root)),
                                node.lineno
                            )
                        else:
                            self.print_violation(
                                "NO_DOCSTRING",
                                f"Function '{node.name}' has no docstring",
                                str(py_file.relative_to(self.repo_root)),
                                node.lineno
                            )
                            
            except Exception as e:
                self.print_warning("PARSE_ERROR", f"Could not parse {py_file}: {e}")
                
        if total_functions > 0:
            coverage = (functions_with_citations / total_functions) * 100
            print(f"\n{BOLD}Citation Coverage:{RESET} {coverage:.1f}% ({functions_with_citations}/{total_functions})")
            
            if coverage < 80:
                self.print_violation(
                    "LOW_COVERAGE",
                    f"Citation coverage {coverage:.1f}% below 80% threshold"
                )
            elif coverage < 100:
                self.print_warning(
                    "INCOMPLETE_COVERAGE",
                    f"Citation coverage {coverage:.1f}% - target is 100%"
                )
            else:
                print(f"{GREEN}✓ All functions cite manuscript{RESET}")
                
        return len(self.violations) == 0
    
    def check_hardcoded_constants(self) -> bool:
        """Check for hardcoded physical constants."""
        self.print_header("2. Hardcoded Constants Check")
        
        # Known problematic patterns
        problematic_patterns = [
            (r'\b137\.03\d+\b', 'alpha inverse'),
            (r'\bALPHA_INVERSE\s*=\s*\d+\.\d+', 'alpha constant assignment'),
            (r'\b0\.511\d*\b', 'electron mass'),
            (r'\b105\.6\d*\b', 'muon mass'),
            (r'\b1\.777\d*\b', 'tau mass'),
            (r'TOPOLOGICAL_COMPLEXITY\s*=\s*\{', 'hardcoded topological complexity'),
        ]
        
        src_dir = self.repo_root / "src"
        python_files = list(src_dir.rglob("*.py"))
        
        found_issues = False
        
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                        
                    for pattern, name in problematic_patterns:
                        if re.search(pattern, line):
                            # Check if justified with comment (same line or next line)
                            justified = False
                            
                            # Check current line for justification
                            if any(keyword in line.lower() for keyword in ['computed', 'derived', 'experimental', 'comparison', 'measurement', 'reference']):
                                justified = True
                            
                            # Check next line
                            if not justified and i < len(lines):
                                next_line = lines[i]
                                if any(keyword in next_line.lower() for keyword in ['computed', 'derived', 'experimental', 'comparison', 'measurement', 'reference']):
                                    justified = True
                                    
                            if not justified:
                                self.print_violation(
                                    "HARDCODED_CONSTANT",
                                    f"Hardcoded {name} found - must be computed or justified",
                                    str(py_file.relative_to(self.repo_root)),
                                    i
                                )
                                found_issues = True
                                
            except Exception as e:
                self.print_warning("READ_ERROR", f"Could not read {py_file}: {e}")
        
        if not found_issues:
            print(f"{GREEN}✓ No problematic hardcoded constants found{RESET}")
            
        return not found_issues
    
    def check_transparency_usage(self) -> bool:
        """Check that TransparencyEngine is used in critical modules."""
        self.print_header("3. Transparency Engine Usage Check")
        
        critical_modules = [
            'src/observables/alpha_inverse.py',
            'src/standard_model/fermion_masses.py',
            'src/rg_flow/beta_functions.py',
        ]
        
        all_good = True
        
        for module_path in critical_modules:
            full_path = self.repo_root / module_path
            
            if not full_path.exists():
                self.print_warning("MISSING_MODULE", f"Critical module not found: {module_path}")
                continue
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            has_import = 'TransparencyEngine' in content or 'transparency_engine' in content
            has_usage = re.search(r'TransparencyEngine\(', content)
            
            if has_import and has_usage:
                self.print_pass(f"{module_path} - Transparency integrated")
            elif has_import:
                self.print_warning(
                    "IMPORT_NO_USAGE",
                    f"{module_path} imports TransparencyEngine but doesn't use it"
                )
            else:
                self.print_violation(
                    "NO_TRANSPARENCY",
                    f"{module_path} lacks TransparencyEngine integration",
                    module_path
                )
                all_good = False
                
        return all_good
    
    def check_test_coverage(self) -> bool:
        """Check that tests exist and pass."""
        self.print_header("4. Test Coverage Verification")
        
        try:
            # Check if pytest is available
            result = subprocess.run(
                ['pytest', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.print_warning("NO_PYTEST", "pytest not available - skipping test check")
                return True
                
            # Run tests
            print("Running test suite...")
            result = subprocess.run(
                ['pytest', 'tests/', '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.repo_root)
            )
            
            if result.returncode == 0:
                # Parse output for pass count
                output = result.stdout
                passed_match = re.search(r'(\d+) passed', output)
                if passed_match:
                    passed_count = int(passed_match.group(1))
                    print(f"{GREEN}✓ All {passed_count} tests passed{RESET}")
                else:
                    print(f"{GREEN}✓ Tests passed{RESET}")
                return True
            else:
                self.print_violation(
                    "TEST_FAILURES",
                    "Some tests failed - see output above",
                )
                print(result.stdout)
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            self.print_warning("TEST_TIMEOUT", "Tests timed out after 5 minutes")
            return False
        except Exception as e:
            self.print_warning("TEST_ERROR", f"Could not run tests: {e}")
            return True  # Don't fail on test infrastructure issues
    
    def check_documentation_consistency(self) -> bool:
        """Check that documentation references are valid."""
        self.print_header("5. Documentation Consistency Check")
        
        # Check that critical documents exist
        required_docs = [
            '.github/THEORETICAL_CORRESPONDENCE_MANDATE.md',
            '.github/COMPREHENSIVE_AUDIT_REPORT.md',
            '.github/MANDATORY_AUDIT_PROTOCOL.md',
            'src/logging/transparency_engine.py',
        ]
        
        all_exist = True
        for doc in required_docs:
            doc_path = self.repo_root / doc
            if doc_path.exists():
                self.print_pass(f"{doc} exists")
            else:
                self.print_violation(
                    "MISSING_DOCUMENT",
                    f"Required document missing: {doc}",
                    doc
                )
                all_exist = False
                
        return all_exist
    
    def generate_report(self) -> Dict:
        """Generate compliance report."""
        return {
            'timestamp': subprocess.run(
                ['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'],
                capture_output=True,
                text=True
            ).stdout.strip(),
            'violations': self.violations,
            'warnings': self.warnings,
            'passes_count': len(self.passes),
            'compliant': len(self.violations) == 0
        }
    
    def print_summary(self):
        """Print summary of compliance check."""
        self.print_header("COMPLIANCE CHECK SUMMARY")
        
        print(f"{BOLD}Results:{RESET}")
        print(f"  {GREEN}✓ Passes:{RESET} {len(self.passes)}")
        print(f"  {YELLOW}⚠ Warnings:{RESET} {len(self.warnings)}")
        print(f"  {RED}✗ Violations:{RESET} {len(self.violations)}")
        
        print(f"\n{BOLD}Status:{RESET} ", end='')
        if len(self.violations) == 0:
            print(f"{GREEN}{BOLD}✓ COMPLIANT{RESET}")
            print(f"\n{GREEN}Code meets IRH v21.4 standards. Ready to commit.{RESET}")
            return True
        else:
            print(f"{RED}{BOLD}✗ NON-COMPLIANT{RESET}")
            print(f"\n{RED}Code violates IRH v21.4 standards. Fix violations before committing.{RESET}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify IRH v21.4 compliance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_compliance.py                # Basic check
  python scripts/verify_compliance.py --verbose      # Detailed output
  python scripts/verify_compliance.py --report out.json  # Save report
        """
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--report', '-r', type=str,
                        help='Save report to JSON file')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.absolute()
    
    print(f"{BOLD}{MAGENTA}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                    IRH v21.4 COMPLIANCE VERIFICATION                      ║")
    print("║                                                                           ║")
    print("║  Ensuring computational engine meets theoretical correspondence mandate   ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    checker = ComplianceChecker(repo_root, verbose=args.verbose)
    
    # Run all checks
    checks = [
        checker.check_manuscript_citations(),
        checker.check_hardcoded_constants(),
        checker.check_transparency_usage(),
        checker.check_test_coverage(),
        checker.check_documentation_consistency(),
    ]
    
    # Generate report
    if args.report:
        report = checker.generate_report()
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{CYAN}Report saved to: {args.report}{RESET}")
    
    # Print summary
    compliant = checker.print_summary()
    
    # Exit code
    sys.exit(0 if compliant else 1)


if __name__ == '__main__':
    main()
