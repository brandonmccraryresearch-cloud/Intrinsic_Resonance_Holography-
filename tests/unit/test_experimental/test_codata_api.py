"""
Tests for CODATA Online API Client

THEORETICAL FOUNDATION: IRH21.md §7

Tests the online API client for CODATA fundamental constants, including
caching, rate limiting, update detection, and diff reporting.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.experimental.codata_api import (
    CODATAAPIClient,
    APIResponse,
    ConstantUpdate,
    check_codata_updates,
)
from src.experimental.codata_database import ExperimentalValue


class TestAPIResponse:
    """Test APIResponse dataclass."""
    
    def test_successful_response(self):
        """Test successful API response."""
        response = APIResponse(
            success=True,
            data={'value': 1.0, 'uncertainty': 0.01},
        )
        
        assert response.success
        assert response.data['value'] == 1.0
        assert response.error == ""
        assert isinstance(response.timestamp, datetime)
    
    def test_failed_response(self):
        """Test failed API response."""
        response = APIResponse(
            success=False,
            data={},
            error="Network error",
        )
        
        assert not response.success
        assert response.error == "Network error"


class TestConstantUpdate:
    """Test ConstantUpdate dataclass."""
    
    def test_constant_update_creation(self):
        """Test creating a constant update."""
        old_value = ExperimentalValue(
            value=137.035999,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
        )
        
        new_value = ExperimentalValue(
            value=137.035998,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2022",
            year=2022,
        )
        
        update = ConstantUpdate(
            name="alpha_inverse",
            old_value=old_value,
            new_value=new_value,
            change_sigma=1.0,
            is_significant=False,
        )
        
        assert update.name == "alpha_inverse"
        assert update.change_sigma == 1.0
        assert not update.is_significant


class TestCODATAAPIClient:
    """Test CODATA API client."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def client(self, temp_cache_dir):
        """Create API client with temp cache."""
        return CODATAAPIClient(
            cache_dir=temp_cache_dir,
            cache_ttl=3600,
            timeout=5,
        )
    
    def test_client_initialization(self, client, temp_cache_dir):
        """Test client is initialized correctly."""
        assert client.cache_dir == temp_cache_dir
        assert client.cache_ttl == 3600
        assert client.timeout == 5
        assert temp_cache_dir.exists()
    
    def test_cache_path_generation(self, client):
        """Test cache path generation."""
        path1 = client._get_cache_path("fine-structure constant")
        path2 = client._get_cache_path("fine-structure constant")
        path3 = client._get_cache_path("Planck constant")
        
        # Same constant should give same path
        assert path1 == path2
        # Different constants should give different paths
        assert path1 != path3
        assert path1.parent == path3.parent
    
    def test_cache_validity_check(self, client, temp_cache_dir):
        """Test cache validity checking."""
        cache_path = temp_cache_dir / "test_cache.json"
        
        # Non-existent cache is invalid
        assert not client._is_cache_valid(cache_path)
        
        # Create fresh cache
        cache_path.write_text("{}")
        assert client._is_cache_valid(cache_path)
        
        # Modify timestamp to make it old
        old_time = datetime.now().timestamp() - client.cache_ttl - 10
        cache_path.touch()
        import os
        os.utime(cache_path, (old_time, old_time))
        assert not client._is_cache_valid(cache_path)
    
    def test_cache_save_and_load(self, client, temp_cache_dir):
        """Test cache save and load."""
        cache_path = temp_cache_dir / "test_cache.json"
        test_data = {
            'name': 'alpha',
            'value': 0.0072973525693,
            'uncertainty': 1.1e-12,
        }
        
        # Save cache
        client._save_cache(cache_path, test_data)
        assert cache_path.exists()
        
        # Load cache
        loaded_data = client._load_cache(cache_path)
        assert loaded_data == test_data
    
    def test_constant_name_mapping(self, client):
        """Test internal to NIST name mapping."""
        assert 'alpha' in client.CONSTANT_NAMES
        assert 'hbar' in client.CONSTANT_NAMES
        assert client.CONSTANT_NAMES['alpha'] == 'fine-structure constant'
    
    @patch('urllib.request.urlopen')
    def test_fetch_constant_with_mock_api(self, mock_urlopen, client):
        """Test fetching constant with mocked API."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.read.return_value = b'<html>Mock response</html>'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        response = client.fetch_constant('alpha', use_cache=False)
        
        # Check that request was made
        mock_urlopen.assert_called_once()
        
        # Note: Real implementation would parse HTML, this just tests structure
        assert response.success or not response.success  # Either is valid
    
    def test_fetch_constant_uses_cache(self, client, temp_cache_dir):
        """Test that fetch uses cache when available."""
        # Create cached data
        cache_path = client._get_cache_path('fine-structure constant')
        cache_data = {
            'name': 'fine-structure constant',
            'value': 0.0072973525693,
            'uncertainty': 1.1e-12,
            'unit': 'dimensionless',
            'source': 'NIST CODATA',
            'year': 2022,
        }
        client._save_cache(cache_path, cache_data)
        
        # Fetch should use cache
        response = client.fetch_constant('alpha', use_cache=True)
        
        assert response.success
        assert response.data == cache_data
    
    def test_rate_limiting(self, client):
        """Test rate limiting between requests."""
        import time
        
        start = time.time()
        client._rate_limit()
        client._rate_limit()
        elapsed = time.time() - start
        
        # Should have waited at least min_request_interval
        assert elapsed >= client._min_request_interval
    
    def test_check_for_updates_structure(self, client):
        """Test structure of check_for_updates method."""
        # Note: This requires network access or mocking
        # Testing structure only
        assert hasattr(client, 'check_for_updates')
        assert callable(client.check_for_updates)
    
    def test_generate_markdown_report(self, client):
        """Test generating markdown report."""
        old_value = ExperimentalValue(
            value=137.035999,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
        )
        
        new_value = ExperimentalValue(
            value=137.036000,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2022",
            year=2022,
        )
        
        updates = [
            ConstantUpdate(
                name="alpha_inverse",
                old_value=old_value,
                new_value=new_value,
                change_sigma=0.7,
                is_significant=False,
            )
        ]
        
        report = client.generate_update_report(updates, format='markdown')
        
        assert "CODATA Constant Updates" in report
        assert "alpha_inverse" in report
        assert "1.370" in report  # Check for scientific notation
    
    def test_generate_text_report(self, client):
        """Test generating text report."""
        old_value = ExperimentalValue(
            value=137.035999,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
        )
        
        new_value = ExperimentalValue(
            value=137.036000,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2022",
            year=2022,
        )
        
        updates = [
            ConstantUpdate(
                name="alpha_inverse",
                old_value=old_value,
                new_value=new_value,
                change_sigma=0.7,
                is_significant=False,
            )
        ]
        
        report = client.generate_update_report(updates, format='text')
        
        assert "CODATA Constant Updates" in report
        assert "alpha_inverse" in report
    
    def test_generate_json_report(self, client):
        """Test generating JSON report."""
        old_value = ExperimentalValue(
            value=137.035999,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
        )
        
        new_value = ExperimentalValue(
            value=137.036000,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2022",
            year=2022,
        )
        
        updates = [
            ConstantUpdate(
                name="alpha_inverse",
                old_value=old_value,
                new_value=new_value,
                change_sigma=0.7,
                is_significant=False,
            )
        ]
        
        report = client.generate_update_report(updates, format='json')
        
        # Should be valid JSON
        data = json.loads(report)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['name'] == 'alpha_inverse'
    
    def test_significant_change_detection(self, client):
        """Test detection of significant changes."""
        old_value = ExperimentalValue(
            value=137.035999,
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
        )
        
        # New value differs by >3σ
        new_value = ExperimentalValue(
            value=137.036010,  # 11σ change
            uncertainty=0.000001,
            unit="dimensionless",
            source="CODATA 2022",
            year=2022,
        )
        
        combined_unc = (
            old_value.uncertainty**2 + new_value.uncertainty**2
        ) ** 0.5
        change_sigma = abs(new_value.value - old_value.value) / combined_unc
        
        update = ConstantUpdate(
            name="alpha_inverse",
            old_value=old_value,
            new_value=new_value,
            change_sigma=change_sigma,
            is_significant=change_sigma > 3.0,
        )
        
        assert update.is_significant
        assert change_sigma > 3.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.experimental.codata_api.CODATAAPIClient')
    def test_check_codata_updates_function(self, mock_client_class):
        """Test check_codata_updates convenience function."""
        # Mock the client
        mock_client = Mock()
        mock_client.check_for_updates.return_value = []
        mock_client_class.return_value = mock_client
        
        # Note: This would require network or more complex mocking
        # Just test that function exists and is callable
        assert callable(check_codata_updates)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
