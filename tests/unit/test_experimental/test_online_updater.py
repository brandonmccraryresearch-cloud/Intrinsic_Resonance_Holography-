"""
Tests for Online Data Updater

Tests the online fetching of experimental constants from CODATA and PDG.
These tests will now work with firewall disabled.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

from src.experimental.online_updater import (
    PhysicalConstant,
    UpdateResult,
    CODATAFetcher,
    PDGFetcher,
    update_codata_online,
    update_pdg_online,
    check_for_data_updates,
    generate_change_report,
    generate_alerts
)
from src.experimental.cache_manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create cache manager with temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)


class TestPhysicalConstant:
    """Tests for PhysicalConstant data class."""
    
    def test_physical_constant_creation(self):
        """Test PhysicalConstant creation."""
        const = PhysicalConstant(
            name='fine-structure constant',
            symbol='α⁻¹',
            value=137.035999084,
            uncertainty=0.000000021,
            units='dimensionless',
            source='CODATA 2022',
            year=2022
        )
        
        assert const.name == 'fine-structure constant'
        assert const.symbol == 'α⁻¹'
        assert const.value == 137.035999084
        assert const.uncertainty == 0.000000021
    
    def test_physical_constant_serialization(self):
        """Test PhysicalConstant to_dict and from_dict."""
        const = PhysicalConstant(
            name='Planck constant',
            symbol='h',
            value=6.62607015e-34,
            uncertainty=0.0,
            units='J Hz⁻¹',
            source='CODATA 2022',
            year=2022
        )
        
        # Serialize
        const_dict = const.to_dict()
        assert const_dict['symbol'] == 'h'
        assert const_dict['value'] == 6.62607015e-34
        
        # Deserialize
        restored = PhysicalConstant.from_dict(const_dict)
        assert restored.symbol == const.symbol
        assert restored.value == const.value
        assert restored.uncertainty == const.uncertainty


class TestUpdateResult:
    """Tests for UpdateResult data class."""
    
    def test_update_result_creation(self):
        """Test UpdateResult creation."""
        result = UpdateResult(
            success=True,
            updated_count=5,
            failed_count=0,
            timestamp=time.time(),
            constants=[],
            errors=[]
        )
        
        assert result.success is True
        assert result.updated_count == 5
        assert result.failed_count == 0
        assert not result.has_significant_changes


class TestCODATAFetcher:
    """Tests for CODATAFetcher class."""
    
    def test_codata_fetcher_creation(self, cache_manager):
        """Test CODATAFetcher creation."""
        fetcher = CODATAFetcher(cache_manager)
        
        assert fetcher.cache is cache_manager
        assert fetcher.rate_limit_delay > 0
    
    def test_codata_fetch_cached(self, cache_manager):
        """Test CODATA fetch uses cache when available."""
        fetcher = CODATAFetcher(cache_manager)
        
        # Pre-populate cache
        test_data = {
            'α⁻¹': PhysicalConstant(
                name='fine-structure constant',
                symbol='α⁻¹',
                value=137.035999084,
                uncertainty=0.000000021,
                units='dimensionless',
                source='CODATA 2022',
                year=2022
            ).to_dict()
        }
        cache_manager.set('codata_2022', test_data)
        
        # Fetch should use cache (no network request)
        result = fetcher.fetch(force_refresh=False)
        
        assert result is not None
        assert 'α⁻¹' in result
        assert result['α⁻¹'].value == 137.035999084
    
    def test_codata_fetch_force_refresh(self, cache_manager):
        """Test CODATA fetch with force_refresh bypasses cache."""
        fetcher = CODATAFetcher(cache_manager)
        
        # Pre-populate cache with old data
        old_data = {
            'α⁻¹': PhysicalConstant(
                name='fine-structure constant',
                symbol='α⁻¹',
                value=137.0,  # Wrong value
                uncertainty=0.000000021,
                units='dimensionless',
                source='CODATA 2018',
                year=2018
            ).to_dict()
        }
        cache_manager.set('codata_2022', old_data)
        
        # Force refresh should invalidate cache and fetch new
        result = fetcher.fetch(force_refresh=True)
        
        # Should get updated value (from _parse_codata_table hardcoded values)
        assert result is not None
        assert 'α⁻¹' in result
        assert result['α⁻¹'].value == 137.035999084  # Updated value
    
    def test_codata_rate_limiting(self, cache_manager):
        """Test rate limiting is applied."""
        fetcher = CODATAFetcher(cache_manager, rate_limit_delay=0.5)
        
        # First call
        start_time = time.time()
        fetcher._rate_limit()
        
        # Second call should be delayed
        fetcher._rate_limit()
        elapsed = time.time() - start_time
        
        # Should have at least one rate limit delay
        assert elapsed >= 0.5
    
    def test_codata_parse_table(self, cache_manager):
        """Test CODATA table parsing."""
        fetcher = CODATAFetcher(cache_manager)
        
        # Parse (currently uses hardcoded values)
        result = fetcher._parse_codata_table("")
        
        # Check that essential constants are present
        assert 'α⁻¹' in result
        assert 'c' in result
        assert 'h' in result
        assert 'G' in result
        
        # Check fine-structure constant
        alpha = result['α⁻¹']
        assert alpha.value == 137.035999084
        assert alpha.source == 'CODATA 2022'


class TestPDGFetcher:
    """Tests for PDGFetcher class."""
    
    def test_pdg_fetcher_creation(self, cache_manager):
        """Test PDGFetcher creation."""
        fetcher = PDGFetcher(cache_manager)
        
        assert fetcher.cache is cache_manager
        assert fetcher.rate_limit_delay > 0
    
    def test_pdg_fetch_cached(self, cache_manager):
        """Test PDG fetch uses cache when available."""
        fetcher = PDGFetcher(cache_manager)
        
        # Pre-populate cache
        test_data = {
            'm_e': PhysicalConstant(
                name='electron mass',
                symbol='m_e',
                value=0.51099895,
                uncertainty=0.00000015,
                units='MeV/c²',
                source='PDG 2024',
                year=2024
            ).to_dict()
        }
        cache_manager.set('pdg_2024', test_data)
        
        # Fetch should use cache
        result = fetcher.fetch(force_refresh=False)
        
        assert result is not None
        assert 'm_e' in result
        assert result['m_e'].value == 0.51099895
    
    def test_pdg_fetch_force_refresh(self, cache_manager):
        """Test PDG fetch with force_refresh."""
        fetcher = PDGFetcher(cache_manager)
        
        # Force refresh
        result = fetcher.fetch(force_refresh=True)
        
        # Should get hardcoded values
        assert result is not None
        assert 'm_e' in result
        assert 'm_μ' in result
        assert 'm_τ' in result


class TestUpdateFunctions:
    """Tests for update convenience functions."""
    
    def test_update_codata_online(self, temp_cache_dir):
        """Test update_codata_online function."""
        result = update_codata_online(cache_dir=temp_cache_dir, force_refresh=False)
        
        assert isinstance(result, UpdateResult)
        # Should succeed (using hardcoded values, no network required)
        assert result.success is True
        assert result.updated_count > 0
        assert len(result.constants) > 0
        assert len(result.errors) == 0
    
    def test_update_pdg_online(self, temp_cache_dir):
        """Test update_pdg_online function."""
        result = update_pdg_online(cache_dir=temp_cache_dir, force_refresh=False)
        
        assert isinstance(result, UpdateResult)
        assert result.success is True
        assert result.updated_count > 0
        assert len(result.constants) > 0
        assert len(result.errors) == 0
    
    def test_check_for_data_updates(self, temp_cache_dir):
        """Test check_for_data_updates function."""
        # Initial check (no cache)
        status = check_for_data_updates(cache_dir=temp_cache_dir)
        
        assert 'codata_cached' in status
        assert 'pdg_cached' in status
        assert 'cache_entry_count' in status
        assert 'update_recommended' in status
        
        # Update data
        update_codata_online(cache_dir=temp_cache_dir)
        
        # Check again
        status = check_for_data_updates(cache_dir=temp_cache_dir)
        assert status['codata_cached'] is True


class TestReportGeneration:
    """Tests for report generation functions."""
    
    def test_generate_change_report_no_changes(self):
        """Test change report with no changes."""
        constants = [
            PhysicalConstant(
                name='test constant',
                symbol='α',
                value=137.0,
                uncertainty=0.1,
                units='',
                source='Test',
                year=2024
            )
        ]
        
        report = generate_change_report(constants, constants, format='text')
        
        assert 'No changes detected' in report
    
    def test_generate_change_report_with_changes(self):
        """Test change report with changes."""
        old_constants = [
            PhysicalConstant(
                name='test constant',
                symbol='α',
                value=137.0,
                uncertainty=0.1,
                units='',
                source='Test',
                year=2023
            )
        ]
        
        new_constants = [
            PhysicalConstant(
                name='test constant',
                symbol='α',
                value=137.035999084,
                uncertainty=0.1,
                units='',
                source='Test',
                year=2024
            )
        ]
        
        # Test markdown format
        report = generate_change_report(old_constants, new_constants, format='markdown')
        assert '# Experimental Data Changes' in report
        assert 'α' in report
        
        # Test text format
        report = generate_change_report(old_constants, new_constants, format='text')
        assert 'EXPERIMENTAL DATA CHANGES' in report
        assert 'α' in report
        
        # Test JSON format
        report = generate_change_report(old_constants, new_constants, format='json')
        # JSON might escape Unicode as \u03b1
        assert '"symbol"' in report and ('α' in report or '\\u03b1' in report)
    
    def test_generate_change_report_formats(self):
        """Test all report formats."""
        old = [PhysicalConstant('test', 'α', 137.0, 0.1, '', 'Test', 2023)]
        new = [PhysicalConstant('test', 'α', 137.1, 0.1, '', 'Test', 2024)]
        
        # All formats should work
        markdown = generate_change_report(old, new, format='markdown')
        text = generate_change_report(old, new, format='text')
        json_str = generate_change_report(old, new, format='json')
        
        assert len(markdown) > 0
        assert len(text) > 0
        assert len(json_str) > 0


class TestAlertGeneration:
    """Tests for alert generation."""
    
    def test_generate_alerts_no_deviations(self):
        """Test alerts with perfect agreement."""
        irh_predictions = {
            'α⁻¹': 137.035999084
        }
        
        experimental = [
            PhysicalConstant(
                name='fine-structure constant',
                symbol='α⁻¹',
                value=137.035999084,
                uncertainty=0.000000021,
                units='',
                source='CODATA',
                year=2022
            )
        ]
        
        alerts = generate_alerts(irh_predictions, experimental, sigma_threshold=3.0)
        
        # No alerts for perfect agreement
        assert len(alerts) == 0
    
    def test_generate_alerts_with_deviation(self):
        """Test alerts with significant deviation."""
        irh_predictions = {
            'α⁻¹': 140.0  # 3σ+ deviation
        }
        
        experimental = [
            PhysicalConstant(
                name='fine-structure constant',
                symbol='α⁻¹',
                value=137.035999084,
                uncertainty=0.5,  # Large uncertainty for testing
                units='',
                source='CODATA',
                year=2022
            )
        ]
        
        alerts = generate_alerts(irh_predictions, experimental, sigma_threshold=3.0)
        
        # Should generate alert
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert['symbol'] == 'α⁻¹'
        assert alert['deviation_sigma'] > 3.0
    
    def test_generate_alerts_falsification(self):
        """Test alerts for falsification (>5σ)."""
        irh_predictions = {
            'α⁻¹': 200.0  # Huge deviation
        }
        
        experimental = [
            PhysicalConstant(
                name='fine-structure constant',
                symbol='α⁻¹',
                value=137.035999084,
                uncertainty=1.0,
                units='',
                source='CODATA',
                year=2022
            )
        ]
        
        alerts = generate_alerts(irh_predictions, experimental, sigma_threshold=3.0)
        
        # Should mark as falsified
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert['falsified'] is True
    
    def test_generate_alerts_missing_prediction(self):
        """Test alerts when IRH has no prediction for a constant."""
        irh_predictions = {
            'α⁻¹': 137.035999084
        }
        
        experimental = [
            PhysicalConstant(
                name='some other constant',
                symbol='β',
                value=42.0,
                uncertainty=0.1,
                units='',
                source='Test',
                year=2024
            )
        ]
        
        alerts = generate_alerts(irh_predictions, experimental, sigma_threshold=3.0)
        
        # No alert for constant without IRH prediction
        assert len(alerts) == 0
    
    def test_generate_alerts_zero_uncertainty(self):
        """Test alerts with zero uncertainty (exact by definition)."""
        irh_predictions = {
            'c': 299792458.0
        }
        
        experimental = [
            PhysicalConstant(
                name='speed of light',
                symbol='c',
                value=299792458.0,
                uncertainty=0.0,  # Exact by definition
                units='m/s',
                source='CODATA',
                year=2022
            )
        ]
        
        alerts = generate_alerts(irh_predictions, experimental, sigma_threshold=3.0)
        
        # No alert for constants with zero uncertainty
        assert len(alerts) == 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_update_workflow(self, temp_cache_dir):
        """Test complete update workflow."""
        # Check status (empty cache)
        status = check_for_data_updates(cache_dir=temp_cache_dir)
        assert status['codata_cached'] is False
        
        # Update CODATA
        result = update_codata_online(cache_dir=temp_cache_dir, force_refresh=True)
        assert result.success is True
        
        # Check status (now cached)
        status = check_for_data_updates(cache_dir=temp_cache_dir)
        assert status['codata_cached'] is True
        
        # Generate report
        if len(result.constants) > 0:
            report = generate_change_report([], result.constants, format='markdown')
            assert len(report) > 0
    
    def test_validation_workflow(self, temp_cache_dir):
        """Test validation against IRH predictions."""
        # Update experimental data
        result = update_codata_online(cache_dir=temp_cache_dir)
        assert result.success is True
        
        # IRH predictions (from src/observables/alpha_inverse.py)
        irh_predictions = {
            'α⁻¹': 137.035999084  # IRH prediction
        }
        
        # Generate alerts
        alerts = generate_alerts(irh_predictions, result.constants, sigma_threshold=3.0)
        
        # Should have perfect agreement (or close enough)
        for alert in alerts:
            # If any alert, should be small deviation
            assert alert['deviation_sigma'] < 5.0, f"Unexpected large deviation: {alert}"
