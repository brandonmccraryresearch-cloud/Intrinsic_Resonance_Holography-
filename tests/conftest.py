"""
Pytest configuration file for IRH test suite.

This file ensures that the src directory is available for imports
without shadowing Python's standard library modules.
"""

import sys
from pathlib import Path

# Add the repository root to sys.path (use absolute path)
repo_root = Path(__file__).parent.parent.resolve()
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Add src directory to sys.path for direct imports (use absolute path)
src_path = repo_root / "src"
src_path_str = str(src_path)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)
