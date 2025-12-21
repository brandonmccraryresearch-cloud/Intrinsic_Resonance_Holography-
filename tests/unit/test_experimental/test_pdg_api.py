"""
Tests for PDG Online API Client

THEORETICAL FOUNDATION: IRH21.md §7

Tests the online API client for PDG particle data.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import pytest
import tempfile
from pathlib import Path

from src.experimental.pdg_api import (
    PDGAPIClient,
    PDGAPIResponse,
    ParticleUpdate,
)


class TestPDGAPIClient:
    """Test PDG API client."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def client(self, temp_cache_dir):
        """Create API client with temp cache."""
        return PDGAPIClient(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
        )
    
    def test_client_initialization(self, client, temp_cache_dir):
        """Test client is initialized correctly."""
        assert client.cache_dir == temp_cache_dir
        assert client.cache_ttl == 3600
        assert temp_cache_dir.exists()
    
    def test_particle_code_mapping(self, client):
        """Test particle name to PDG code mapping."""
        assert 'electron' in client.PARTICLE_CODES
        assert client.PARTICLE_CODES['electron'] == 11
        assert 'top' in client.PARTICLE_CODES
        assert client.PARTICLE_CODES['top'] == 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
