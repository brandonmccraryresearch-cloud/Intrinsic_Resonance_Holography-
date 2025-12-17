"""
Pytest configuration file for IRH test suite.

This file ensures that the src directory is available for imports
without shadowing Python's standard library modules.
"""

import sys
from pathlib import Path

# Add the repository root to sys.path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add src directory to sys.path for direct imports
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
