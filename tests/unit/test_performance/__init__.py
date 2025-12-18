"""
Tests for Performance Module __init__

This file ensures the performance module is properly importable.
"""

from __future__ import annotations


def test_performance_module_import():
    """Test that performance module imports successfully."""
    from src import performance
    
    assert hasattr(performance, '__version__')
    assert performance.__version__ == "21.0.0"


def test_performance_theoretical_foundation():
    """Test theoretical reference is present."""
    from src import performance
    
    assert hasattr(performance, '__theoretical_foundation__')
    assert 'ROADMAP' in performance.__theoretical_foundation__
