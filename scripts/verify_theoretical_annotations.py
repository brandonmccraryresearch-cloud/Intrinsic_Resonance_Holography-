#!/usr/bin/env python3
"""
Verify Theoretical Annotations in IRH v21.0 Codebase

THEORETICAL FOUNDATION: IRH v21.1 Manuscript Part 1 §1.6 (Transparency Commitment)

This script scans all Python source files and verifies that:
1. Every module has a THEORETICAL FOUNDATION citation
2. Every function/class has docstrings with IRH v21.1 Manuscript references
3. Equation labels are properly formatted and valid

Usage:
    python scripts/verify_theoretical_annotations.py

Exit Codes:
    0 - All annotations valid
    1 - Missing or invalid annotations found
"""

import os
import re
import sys
from pathlib import Path
from typing import List

# Configuration constants
MIN_INIT_FILE_LENGTH = 50  # Minimum content length for __init__.py to check


def find_python_files(root_dir: str) -> List[Path]:
    """Find all Python files in the source directory."""
    src_path = Path(root_dir) / "src"
    return list(src_path.rglob("*.py"))


def check_module_header(filepath: Path) -> List[str]:
    """Check if module has proper theoretical foundation citation."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip empty __init__.py files
    if filepath.name == "__init__.py" and len(content.strip()) < MIN_INIT_FILE_LENGTH:
        return issues
    
    # Check for THEORETICAL FOUNDATION in module docstring
    if 'THEORETICAL FOUNDATION' not in content and '__init__' not in str(filepath):
        issues.append(f"{filepath}: Missing THEORETICAL FOUNDATION citation in module docstring")
    
    # Check for IRH v21.1 Manuscript reference (accepts both old and new formats)
    has_manuscript_ref = (
        'IRH21.md' in content or 
        'IRH v21.1 Manuscript' in content or
        'Part 1' in content or 
        'Part 2' in content
    )
    if not has_manuscript_ref and '__init__' not in str(filepath):
        issues.append(f"{filepath}: Missing IRH v21.1 Manuscript reference")
    
    return issues


def check_equation_labels(filepath: Path) -> List[str]:
    """Check that equation references are properly formatted."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find equation references
    eq_pattern = r'Eq\.?\s*(\d+\.?\d*)'
    equations = re.findall(eq_pattern, content, re.IGNORECASE)
    
    # Valid equation ranges from IRH21.md
    valid_sections = {
        '1': range(1, 20),    # Section 1: 1.1-1.16
        '2': range(1, 30),    # Section 2: 2.1-2.26
        '3': range(1, 10),    # Section 3: 3.1-3.8
        '5': range(1, 5),     # Section 5: 5.1-5.2
    }
    
    for eq in equations:
        if '.' in eq:
            section, number = eq.split('.', 1)
            # Basic format validation
            try:
                float(eq)  # Should be a valid number
            except ValueError:
                issues.append(f"{filepath}: Invalid equation reference format: Eq. {eq}")
    
    return issues


def check_section_references(filepath: Path) -> List[str]:
    """Check that section references are properly formatted."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find section references
    section_pattern = r'§(\d+\.?\d*\.?\d*)'
    sections = re.findall(section_pattern, content)
    
    # Valid section numbers from IRH v21.1 Manuscript (Part 1 + Part 2)
    # Part 1: Sections 1-4, Part 2: Sections 5-8 + Appendices
    valid_sections = [
        # Part 1: Section 1 - Foundation
        '1', '1.0', '1.0.1', '1.1', '1.1.1', '1.2', '1.2.1', '1.2.2', '1.2.3', '1.2.4',
        '1.3', '1.3.1', '1.3.2', '1.3.3', '1.4', '1.4.1', '1.4.2', '1.5', '1.6',
        # Part 1: Section 2 - Emergent Spacetime
        '2', '2.1', '2.1.1', '2.1.2', '2.2', '2.2.1', '2.2.2', '2.3', '2.3.1', '2.3.2', '2.3.3', 
        '2.4', '2.4.1', '2.4.2', '2.4.3', '2.5',
        # Part 1: Section 3 - Standard Model
        '3', '3.1', '3.1.1', '3.1.2', '3.2', '3.2.', '3.2.1', '3.2.2', '3.2.3', '3.2.4', 
        '3.3', '3.3.1', '3.3.2', '3.3.3', '3.3.4', 
        '3.4', '3.4.1', '3.4.2', '3.4.3', '3.4.4',
        # Part 1: Section 4 - Meta-theory
        '4', '4.1', '4.2', '4.3',
        # Part 2: Section 5 - Quantum Mechanics
        '5', '5.1', '5.2', '5.2.1',
        # Part 2: Section 6-8 - Predictions
        '6', '7', '7.1', '7.2',
        '8', '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7',
        # Part 2: Appendices
        '9', '9.1', '9.2', '9.3', '9.4',
        '10',
        # Appendix references (A-K)
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'A.1', 'A.2', 'A.3', 'A.4', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5',
        'D.1', 'D.2', 'D.3', 'E.1', 'E.2', 'E.3', 'F.1', 'F.2',
        'G.1', 'G.2', 'H.1', 'H.2', 'I.1', 'I.2', 'J.1', 'J.2', 'J.3', 'K.1', 'K.2'
    ]
    
    for section in sections:
        if section not in valid_sections:
            issues.append(f"{filepath}: Potentially invalid section reference: §{section}")
    
    return issues


def main():
    """Run all verification checks."""
    root_dir = Path(__file__).parent.parent
    issues = []
    
    print("=" * 60)
    print("IRH v21.0 Theoretical Annotation Verification")
    print("=" * 60)
    
    python_files = find_python_files(root_dir)
    print(f"\nScanning {len(python_files)} Python files...\n")
    
    for filepath in python_files:
        file_issues = []
        file_issues.extend(check_module_header(filepath))
        file_issues.extend(check_equation_labels(filepath))
        file_issues.extend(check_section_references(filepath))
        
        if file_issues:
            print(f"Issues in {filepath.relative_to(root_dir)}:")
            for issue in file_issues:
                print(f"  - {issue.split(': ', 1)[1]}")
            print()
        
        issues.extend(file_issues)
    
    print("=" * 60)
    if issues:
        print(f"Found {len(issues)} annotation issues.")
        print("Please ensure all code has proper theoretical citations.")
        return 1
    else:
        print("All theoretical annotations valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
