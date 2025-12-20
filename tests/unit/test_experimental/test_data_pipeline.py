"""
Tests for Experimental Data Pipeline (Phase 4.4)

Tests CODATA database, PDG parser, comparison framework, and data catalog.
"""

import pytest
import math
from datetime import datetime


class TestExperimentalValue:
    """Tests for ExperimentalValue dataclass."""
    
    def test_create_value(self):
        """Test creating experimental value."""
        from src.experimental.codata_database import ExperimentalValue
        
        val = ExperimentalValue(
            value=137.035999084,
            uncertainty=0.000000021,
            unit="dimensionless",
            source="CODATA 2018",
            year=2018,
            reference="https://physics.nist.gov",
        )
        
        assert val.value == pytest.approx(137.035999084)
        assert val.uncertainty == pytest.approx(0.000000021)
        assert val.unit == "dimensionless"
        assert val.source == "CODATA 2018"
    
    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        from src.experimental.codata_database import ExperimentalValue
        
        val = ExperimentalValue(
            value=100.0,
            uncertainty=1.0,
            unit="",
            source="test",
            year=2024,
        )
        
        assert val.relative_uncertainty == pytest.approx(0.01)
    
    def test_sigma_from(self):
        """Test σ deviation calculation."""
        from src.experimental.codata_database import ExperimentalValue
        
        val = ExperimentalValue(
            value=100.0,
            uncertainty=1.0,
            unit="",
            source="test",
            year=2024,
        )
        
        # 2σ away
        assert val.sigma_from(102.0) == pytest.approx(2.0)
        
        # With prediction uncertainty
        assert val.sigma_from(102.0, 1.0) == pytest.approx(2.0 / math.sqrt(2))
    
    def test_is_consistent(self):
        """Test consistency check."""
        from src.experimental.codata_database import ExperimentalValue
        
        val = ExperimentalValue(
            value=100.0,
            uncertainty=1.0,
            unit="",
            source="test",
            year=2024,
        )
        
        assert val.is_consistent(101.0, n_sigma=2.0)  # 1σ away
        assert not val.is_consistent(105.0, n_sigma=2.0)  # 5σ away


class TestCODATADatabase:
    """Tests for CODATA database."""
    
    def test_get_alpha(self):
        """Test retrieving fine-structure constant."""
        from src.experimental.codata_database import get_codata_value
        
        alpha = get_codata_value('alpha')
        assert alpha.value == pytest.approx(7.2973525693e-3, rel=1e-10)
    
    def test_get_alpha_inverse(self):
        """Test retrieving inverse fine-structure constant."""
        from src.experimental.codata_database import get_codata_value
        
        alpha_inv = get_codata_value('alpha_inverse')
        assert alpha_inv.value == pytest.approx(137.035999084, rel=1e-10)
    
    def test_get_electron_mass(self):
        """Test retrieving electron mass."""
        from src.experimental.codata_database import get_codata_value
        
        m_e = get_codata_value('electron_mass_mev')
        assert m_e.value == pytest.approx(0.51099895000, rel=1e-9)
        assert m_e.unit == "MeV/c²"
    
    def test_get_higgs_mass(self):
        """Test retrieving Higgs mass."""
        from src.experimental.codata_database import get_codata_value
        
        m_H = get_codata_value('higgs_mass')
        assert m_H.value == pytest.approx(125.25, rel=0.01)
        assert m_H.unit == "GeV/c²"
    
    def test_list_constants(self):
        """Test listing available constants."""
        from src.experimental.codata_database import list_constants
        
        constants = list_constants()
        assert 'alpha' in constants
        assert 'higgs_mass' in constants
        assert len(constants) > 20
    
    def test_invalid_constant(self):
        """Test error handling for invalid constant."""
        from src.experimental.codata_database import get_codata_value
        
        with pytest.raises(KeyError):
            get_codata_value('nonexistent_constant')
    
    def test_compare_irh_prediction(self):
        """Test IRH prediction comparison."""
        from src.experimental.codata_database import compare_irh_prediction
        
        result = compare_irh_prediction('alpha_inverse')
        
        assert 'experimental' in result
        assert 'irh_prediction' in result
        assert 'sigma_deviation' in result
        assert result['sigma_deviation'] < 1.0  # Should be very close


class TestPDGParser:
    """Tests for PDG parser."""
    
    def test_get_electron(self):
        """Test retrieving electron data."""
        from src.experimental.pdg_parser import get_particle
        
        electron = get_particle('electron')
        assert electron.name == 'electron'
        assert electron.charge == -1.0
        assert electron.spin == 0.5
        assert electron.mass.value == pytest.approx(0.51099895, rel=1e-8)
    
    def test_get_top_quark(self):
        """Test retrieving top quark."""
        from src.experimental.pdg_parser import get_particle
        
        top = get_particle('top')
        assert top.name == 'top'
        # Electric charge is 2/3 (exact)
        expected_charge = 2.0 / 3.0
        assert abs(top.charge - expected_charge) < 1e-10
        assert top.mass.value == pytest.approx(172760, rel=0.01)  # MeV
    
    def test_get_pdg_value(self):
        """Test getting specific property."""
        from src.experimental.pdg_parser import get_pdg_value
        
        mass = get_pdg_value('muon', 'mass')
        assert mass.value == pytest.approx(105.658, rel=1e-4)
    
    def test_list_particles(self):
        """Test listing particles."""
        from src.experimental.pdg_parser import list_particles
        
        particles = list_particles()
        assert 'electron' in particles
        assert 'muon' in particles
        assert 'top' in particles
        assert 'Higgs' in particles
    
    def test_get_particles_by_type(self):
        """Test filtering particles by type."""
        from src.experimental.pdg_parser import get_particles_by_type, ParticleType
        
        leptons = get_particles_by_type(ParticleType.LEPTON)
        lepton_names = [p.name for p in leptons]
        
        assert 'electron' in lepton_names
        assert 'muon' in lepton_names
        assert 'tau' in lepton_names
    
    def test_mass_ratio(self):
        """Test mass ratio calculation."""
        from src.experimental.pdg_parser import mass_ratio
        
        ratio = mass_ratio('muon', 'electron')
        
        # μ/e mass ratio ≈ 206.77
        assert ratio.value == pytest.approx(206.77, rel=0.01)
        assert ratio.unit == 'dimensionless'


class TestComparison:
    """Tests for comparison framework."""
    
    def test_compare_single(self):
        """Test single value comparison."""
        from src.experimental.comparison import compare_single, ComparisonStatus
        
        # Compare IRH α⁻¹ prediction with experiment
        result = compare_single(137.035999084, 'alpha_inverse', 1e-9)
        
        assert result.irh_value == pytest.approx(137.035999084)
        assert result.sigma_deviation < 1.0
        assert result.status == ComparisonStatus.EXCELLENT
    
    def test_comparison_status(self):
        """Test comparison status classification."""
        from src.experimental.comparison import compare_single, ComparisonStatus
        
        # Exact match
        result = compare_single(137.035999084, 'alpha_inverse', 0.0)
        assert result.status == ComparisonStatus.EXCELLENT
    
    def test_generate_comparison_table(self):
        """Test table generation."""
        from src.experimental.comparison import compare_single, generate_comparison_table
        
        comparisons = [
            compare_single(137.035999084, 'alpha_inverse', 1e-9),
        ]
        
        # Markdown table
        md_table = generate_comparison_table(comparisons, format='markdown')
        assert 'Observable' in md_table
        assert 'alpha_inverse' in md_table
    
    def test_full_comparison_report(self):
        """Test full comparison report."""
        from src.experimental.comparison import full_irh_comparison_report
        
        report = full_irh_comparison_report()
        
        assert 'report_title' in report
        assert 'n_observables' in report
        assert 'comparisons' in report


class TestDataCatalog:
    """Tests for data catalog."""
    
    def test_create_catalog(self):
        """Test creating catalog."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        assert len(catalog) > 0
    
    def test_get_value(self):
        """Test getting value from catalog."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        alpha = catalog.get('alpha_inverse')
        
        assert alpha.value == pytest.approx(137.035999084, rel=1e-9)
    
    def test_search(self):
        """Test searching catalog."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        results = catalog.search('mass')
        
        assert len(results) > 0
        assert any('mass' in r.key for r in results)
    
    def test_search_by_category(self):
        """Test filtering by category."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        results = catalog.search(category='particle_mass')
        
        assert len(results) > 0
        assert all(r.category == 'particle_mass' for r in results)
    
    def test_list_categories(self):
        """Test listing categories."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        categories = catalog.list_categories()
        
        assert 'fundamental_constant' in categories
        assert 'particle_mass' in categories
    
    def test_add_entry(self):
        """Test adding new entry."""
        from src.experimental.data_catalog import DataCatalog, DataEntry
        from src.experimental.codata_database import ExperimentalValue
        
        catalog = DataCatalog()
        
        catalog.add(
            key='custom_value',
            value=ExperimentalValue(
                value=42.0,
                uncertainty=0.1,
                unit='units',
                source='test',
                year=2024,
            ),
            category='test',
            tags=['custom'],
        )
        
        assert 'custom_value' in catalog
        assert catalog.get('custom_value').value == 42.0
    
    def test_catalog_to_dict(self):
        """Test exporting catalog."""
        from src.experimental.data_catalog import DataCatalog
        
        catalog = DataCatalog()
        data = catalog.to_dict()
        
        assert 'version' in data
        assert 'entries' in data
        assert len(data['entries']) > 0
    
    def test_get_experimental_value_convenience(self):
        """Test convenience function."""
        from src.experimental.data_catalog import get_experimental_value
        
        alpha = get_experimental_value('alpha_inverse')
        assert alpha.value == pytest.approx(137.035999084, rel=1e-9)


class TestIRHPredictions:
    """Tests for IRH prediction comparisons."""
    
    def test_alpha_inverse_prediction(self):
        """Test α⁻¹ = 137.035999084 prediction."""
        from src.experimental.codata_database import IRH_PREDICTIONS
        
        pred = IRH_PREDICTIONS.get('alpha_inverse')
        assert pred is not None
        assert pred['value'] == pytest.approx(137.035999084, rel=1e-10)
    
    def test_dark_energy_prediction(self):
        """Test w₀ = -0.91234567 prediction."""
        from src.experimental.codata_database import IRH_PREDICTIONS
        
        pred = IRH_PREDICTIONS.get('w0')
        assert pred is not None
        assert pred['value'] == pytest.approx(-0.91234567, rel=1e-8)
    
    def test_liv_prediction(self):
        """Test ξ = C_H/(24π²) prediction."""
        from src.experimental.codata_database import IRH_PREDICTIONS
        import math
        
        pred = IRH_PREDICTIONS.get('xi_liv')
        assert pred is not None
        
        # Verify formula: ξ = C_H / (24π²)
        C_H = 0.045935703598
        expected_xi = C_H / (24 * math.pi**2)
        assert pred['value'] == pytest.approx(expected_xi, rel=1e-8)


# Marker for pytest
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
