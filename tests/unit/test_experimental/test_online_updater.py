"""
Tests for Online Data Updater (Phase 4.5)

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §7
ROADMAP REFERENCE: docs/ROADMAP.md Phase 4.5

Tests:
- CacheManager functionality
- Update result data structures
- Change detection logic
- Alert generation
- Report generation
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.experimental.online_updater import (
    # Enums
    UpdateSource,
    ChangeType,
    
    # Data classes
    DataChange,
    UpdateResult,
    CacheMetadata,
    Alert,
    
    # Cache management
    CacheManager,
    
    # Update functions
    update_codata,
    update_pdg,
    update_all,
    check_for_updates,
    
    # Reporting
    generate_change_report,
    generate_alerts,
    
    # Configuration
    CACHE_VALIDITY_HOURS,
)


class TestDataChange:
    """Tests for DataChange data class."""
    
    def test_create_value_change(self):
        """Test creating a value change."""
        change = DataChange(
            constant_name='alpha_inverse',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=137.035999084,
            new_value=137.035999085,
            old_uncertainty=0.000000021,
            new_uncertainty=0.000000021,
            percent_change=7.3e-9,
            source='CODATA',
        )
        
        assert change.constant_name == 'alpha_inverse'
        assert change.change_type == ChangeType.VALUE_CHANGE
        assert change.old_value == 137.035999084
        assert change.new_value == 137.035999085
    
    def test_significance_calculation(self):
        """Test statistical significance calculation."""
        # Small change, should not be significant
        small_change = DataChange(
            constant_name='test',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=100.0,
            new_value=100.00001,
            old_uncertainty=0.001,
            percent_change=0.00001,
        )
        assert not small_change.is_significant(threshold_sigma=2.0)
        
        # Large change, should be significant
        large_change = DataChange(
            constant_name='test',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=100.0,
            new_value=100.01,
            old_uncertainty=0.001,
            percent_change=0.01,
        )
        assert large_change.is_significant(threshold_sigma=2.0)
    
    def test_new_constant_is_significant(self):
        """Test that new constants are always significant."""
        change = DataChange(
            constant_name='new_constant',
            change_type=ChangeType.NEW_CONSTANT,
            new_value=1.234,
            source='test',
        )
        assert change.is_significant()
    
    def test_no_change_not_significant(self):
        """Test that no change is not significant."""
        change = DataChange(
            constant_name='test',
            change_type=ChangeType.NO_CHANGE,
        )
        assert not change.is_significant()
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        change = DataChange(
            constant_name='alpha',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=7.297e-3,
            new_value=7.298e-3,
            source='CODATA',
        )
        
        d = change.to_dict()
        assert d['constant_name'] == 'alpha'
        assert d['change_type'] == 'value_change'
        assert 'is_significant' in d


class TestUpdateResult:
    """Tests for UpdateResult data class."""
    
    def test_create_result(self):
        """Test creating an update result."""
        result = UpdateResult(
            source=UpdateSource.CODATA,
            success=True,
            updated_count=5,
        )
        
        assert result.source == UpdateSource.CODATA
        assert result.success
        assert result.updated_count == 5
        assert len(result.changes) == 0
        assert len(result.errors) == 0
    
    def test_has_significant_changes(self):
        """Test checking for significant changes."""
        result = UpdateResult(source=UpdateSource.CODATA)
        assert not result.has_significant_changes
        
        # Add a significant change
        result.changes.append(DataChange(
            constant_name='test',
            change_type=ChangeType.NEW_CONSTANT,
            new_value=1.0,
        ))
        assert result.has_significant_changes
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = UpdateResult(
            source=UpdateSource.PDG,
            success=True,
            updated_count=10,
        )
        
        d = result.to_dict()
        assert d['source'] == 'pdg'
        assert d['success'] == True
        assert d['updated_count'] == 10
        assert 'timestamp' in d


class TestCacheManager:
    """Tests for CacheManager."""
    
    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def cache_manager(self, cache_dir):
        """Create cache manager with temp directory."""
        return CacheManager(cache_dir)
    
    def test_cache_directory_creation(self, cache_dir):
        """Test cache directory is created."""
        manager = CacheManager(cache_dir / 'new_dir')
        assert (cache_dir / 'new_dir').exists()
    
    def test_save_and_load_cache(self, cache_manager):
        """Test saving and loading cache."""
        test_data = {
            'alpha': {'value': 7.297e-3, 'uncertainty': 1.1e-12},
            'hbar': {'value': 1.054e-34, 'uncertainty': 0.0},
        }
        
        # Save
        cache_path = cache_manager.save_cache(UpdateSource.CODATA, test_data, 'v2018')
        assert Path(cache_path).exists()
        
        # Load
        loaded = cache_manager.load_cache(UpdateSource.CODATA)
        assert loaded is not None
        assert loaded['alpha']['value'] == 7.297e-3
    
    def test_cache_freshness(self, cache_manager):
        """Test cache freshness check."""
        # Initially no cache
        assert not cache_manager.is_cache_fresh(UpdateSource.CODATA)
        
        # Save cache
        cache_manager.save_cache(UpdateSource.CODATA, {'test': 1})
        
        # Now should be fresh
        assert cache_manager.is_cache_fresh(UpdateSource.CODATA)
    
    def test_cache_metadata(self, cache_manager):
        """Test cache metadata."""
        cache_manager.save_cache(UpdateSource.PDG, {'test': 1}, 'v2024')
        
        meta = cache_manager.get_metadata(UpdateSource.PDG)
        assert meta is not None
        assert meta.version == 'v2024'
        assert meta.constant_count == 1
    
    def test_clear_cache(self, cache_manager):
        """Test clearing cache."""
        cache_manager.save_cache(UpdateSource.CODATA, {'test': 1})
        cache_manager.save_cache(UpdateSource.PDG, {'test': 2})
        
        # Clear just CODATA
        cache_manager.clear_cache(UpdateSource.CODATA)
        assert cache_manager.load_cache(UpdateSource.CODATA) is None
        assert cache_manager.load_cache(UpdateSource.PDG) is not None
        
        # Clear all
        cache_manager.clear_cache()
        assert cache_manager.load_cache(UpdateSource.PDG) is None


class TestUpdateFunctions:
    """Tests for update functions."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield CacheManager(Path(tmpdir))
    
    def test_update_codata_returns_result(self, cache_manager):
        """Test that update_codata returns a valid result."""
        # Note: This doesn't actually make network calls if cache is used
        result = update_codata(use_cache=True, cache_manager=cache_manager)
        
        assert isinstance(result, UpdateResult)
        assert result.source == UpdateSource.CODATA
        assert isinstance(result.timestamp, str)
    
    def test_update_pdg_returns_result(self, cache_manager):
        """Test that update_pdg returns a valid result."""
        result = update_pdg(use_cache=True, cache_manager=cache_manager)
        
        assert isinstance(result, UpdateResult)
        assert result.source == UpdateSource.PDG
    
    def test_update_all(self, cache_manager):
        """Test updating all sources."""
        results = update_all(use_cache=True, cache_dir=cache_manager.cache_dir)
        
        assert 'codata' in results
        assert 'pdg' in results
        assert isinstance(results['codata'], UpdateResult)
        assert isinstance(results['pdg'], UpdateResult)
    
    def test_check_for_updates(self, cache_manager):
        """Test checking for updates."""
        status = check_for_updates(cache_manager=cache_manager)
        
        assert 'checked_at' in status
        assert 'sources' in status
        assert 'codata' in status['sources']
        assert 'pdg' in status['sources']


class TestReporting:
    """Tests for report generation."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample update results."""
        codata_result = UpdateResult(
            source=UpdateSource.CODATA,
            success=True,
            updated_count=3,
        )
        codata_result.changes.append(DataChange(
            constant_name='alpha',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=7.297e-3,
            new_value=7.298e-3,
            percent_change=0.0137,
            source='CODATA',
        ))
        
        pdg_result = UpdateResult(
            source=UpdateSource.PDG,
            success=True,
            updated_count=5,
        )
        
        return {'codata': codata_result, 'pdg': pdg_result}
    
    def test_generate_markdown_report(self, sample_results):
        """Test Markdown report generation."""
        report = generate_change_report(sample_results, output_format='markdown')
        
        assert '# Experimental Data Update Report' in report
        assert 'CODATA' in report
        assert 'PDG' in report
        assert 'alpha' in report
    
    def test_generate_text_report(self, sample_results):
        """Test text report generation."""
        report = generate_change_report(sample_results, output_format='text')
        
        assert 'EXPERIMENTAL DATA UPDATE REPORT' in report
        assert 'CODATA' in report
    
    def test_generate_json_report(self, sample_results):
        """Test JSON report generation."""
        report = generate_change_report(sample_results, output_format='json')
        
        data = json.loads(report)
        assert 'codata' in data
        assert 'pdg' in data
        assert data['codata']['success'] == True


class TestAlerts:
    """Tests for alert generation."""
    
    def test_generate_alerts_for_significant_changes(self):
        """Test that alerts are generated for significant changes."""
        result = UpdateResult(source=UpdateSource.CODATA)
        result.changes.append(DataChange(
            constant_name='alpha',
            change_type=ChangeType.VALUE_CHANGE,
            old_value=100.0,
            new_value=102.0,  # 2% change - significant
            old_uncertainty=0.01,
            percent_change=2.0,
        ))
        
        alerts = generate_alerts({'codata': result}, sigma_threshold=2.0)
        
        assert len(alerts) > 0
        assert any(a.constant_name == 'alpha' for a in alerts)
    
    def test_alert_for_failed_update(self):
        """Test alert generation for failed updates."""
        result = UpdateResult(source=UpdateSource.PDG, success=False)
        result.errors.append("Connection timeout")
        
        alerts = generate_alerts({'pdg': result})
        
        assert len(alerts) > 0
        assert any(a.level == 'warning' for a in alerts)
        assert any('Connection timeout' in a.message for a in alerts)
    
    def test_no_alerts_for_no_changes(self):
        """Test no alerts when no significant changes."""
        result = UpdateResult(source=UpdateSource.CODATA, success=True)
        
        alerts = generate_alerts({'codata': result})
        
        assert len(alerts) == 0


class TestEnums:
    """Tests for enum values."""
    
    def test_update_source_values(self):
        """Test UpdateSource enum values."""
        assert UpdateSource.CODATA.value == 'codata'
        assert UpdateSource.PDG.value == 'pdg'
        assert UpdateSource.ALL.value == 'all'
    
    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.VALUE_CHANGE.value == 'value_change'
        assert ChangeType.UNCERTAINTY_CHANGE.value == 'uncertainty_change'
        assert ChangeType.NEW_CONSTANT.value == 'new_constant'
        assert ChangeType.REMOVED_CONSTANT.value == 'removed_constant'
        assert ChangeType.NO_CHANGE.value == 'no_change'


class TestIntegration:
    """Integration tests for online updater."""
    
    def test_full_update_workflow(self):
        """Test complete update workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # 1. Check for updates
            status = check_for_updates(cache_manager=CacheManager(cache_dir))
            assert 'sources' in status
            
            # 2. Perform updates
            results = update_all(use_cache=True, cache_dir=cache_dir)
            assert 'codata' in results
            assert 'pdg' in results
            
            # 3. Generate report
            report = generate_change_report(results, output_format='markdown')
            assert '# Experimental Data Update Report' in report
            
            # 4. Generate alerts
            alerts = generate_alerts(results)
            assert isinstance(alerts, list)
    
    def test_cache_persistence(self):
        """Test that cache persists across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Save with first manager
            manager1 = CacheManager(cache_dir)
            manager1.save_cache(UpdateSource.CODATA, {'test': 'data'})
            
            # Load with second manager
            manager2 = CacheManager(cache_dir)
            loaded = manager2.load_cache(UpdateSource.CODATA)
            
            assert loaded == {'test': 'data'}
