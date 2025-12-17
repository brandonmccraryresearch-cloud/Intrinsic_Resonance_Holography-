"""
Pytest configuration file for IRH test suite.

This file ensures that imports work correctly by adding the repository
root to sys.path, allowing imports like 'from src.module import ...'.

We do NOT add 'src' directly to sys.path because that would cause
the repository's 'src/logging/' directory to shadow Python's standard
library 'logging' module, breaking pytest and other tools.
"""

import sys
from pathlib import Path

# Add the repository root to sys.path (use absolute path)
# This allows imports like: from src.output.output_standardization import ...
repo_root = Path(__file__).parent.parent.resolve()
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
