"""
Tests for Update Manager

THEORETICAL FOUNDATION: IRH21.md §7

Tests the unified update manager for experimental data.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import pytest
import tempfile
from pathlib import Path

from src.experimental.update_manager import (
    UpdateManager,
    UpdateSummary,
    generate_update_report,
)


class TestUpdateManager:
    """Test Update Manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create update manager with temp cache."""
        return UpdateManager(cache_dir=temp_cache_dir)
    
    def test_manager_initialization(self, manager, temp_cache_dir):
        """Test manager is initialized correctly."""
        assert manager.cache_dir == temp_cache_dir
        assert manager.codata_client is not None
        assert manager.pdg_client is not None
    
    def test_report_generation_structure(self, manager):
        """Test report generation methods exist."""
        assert hasattr(manager, 'generate_comprehensive_report')
        assert callable(manager.generate_comprehensive_report)
    
    def test_update_summary_creation(self):
        """Test creating update summary."""
        summary = UpdateSummary(
            timestamp=pytest.approx,  # Will be set
            codata_updates=[],
            pdg_updates=[],
            total_updates=0,
            significant_updates=0,
            requires_action=False,
        )
        
        assert summary.total_updates == 0
        assert not summary.requires_action


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
