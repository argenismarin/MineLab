"""Tests for minelab.mineral_processing.thickening."""

import numpy as np
import pytest

from minelab.mineral_processing.thickening import (
    coe_clevenger,
    flocculant_dosage,
    kynch_analysis,
    talmage_fitch,
)


class TestKynchAnalysis:
    """Tests for Kynch batch settling test analysis."""

    def test_settling_rates(self):
        """Should return settling rates."""
        t = np.array([0, 5, 10, 20, 30, 60])
        h = np.array([1.0, 0.8, 0.65, 0.45, 0.35, 0.3])
        result = kynch_analysis(t, h)
        assert len(result["settling_rates"]) == 5
        assert np.all(result["settling_rates"] > 0)

    def test_concentrations_increase(self):
        """Concentrations should increase as height decreases."""
        t = np.array([0, 5, 10, 20])
        h = np.array([1.0, 0.8, 0.6, 0.4])
        result = kynch_analysis(t, h)
        assert np.all(np.diff(result["concentrations"]) > 0)


class TestTalmageFitch:
    """Tests for Talmage-Fitch thickener sizing."""

    def test_positive_area(self):
        """Thickener area should be positive."""
        result = talmage_fitch(1.0, 0.5, 0.1, 0.5, 100)
        assert result["thickener_area"] > 0
        assert result["diameter"] > 0

    def test_higher_feed_larger_thickener(self):
        """Higher feed rate → larger thickener."""
        small = talmage_fitch(1.0, 0.5, 0.1, 0.5, 50)
        large = talmage_fitch(1.0, 0.5, 0.1, 0.5, 100)
        assert large["thickener_area"] > small["thickener_area"]


class TestCoeClevenger:
    """Tests for Coe-Clevenger unit area method."""

    def test_positive_area(self):
        """Should return positive area."""
        rates = np.array([0.5, 0.3, 0.1, 0.05])
        concs = np.array([0.05, 0.1, 0.2, 0.3])
        result = coe_clevenger(rates, concs, 0.4, 100, 0.05)
        assert result["thickener_area"] > 0

    def test_controlling_concentration(self):
        """Should identify controlling concentration."""
        rates = np.array([0.5, 0.3, 0.1, 0.05])
        concs = np.array([0.05, 0.1, 0.2, 0.3])
        result = coe_clevenger(rates, concs, 0.4, 100, 0.05)
        assert result["controlling_concentration"] in concs


class TestFlocculantDosage:
    """Tests for flocculant consumption."""

    def test_known_value(self):
        """100 m3/h, 10% solids, 20 g/t → 0.2 kg/h."""
        consumption = flocculant_dosage(100, 0.1, 20)
        assert consumption == pytest.approx(0.2, rel=0.01)

    def test_zero_solids(self):
        """Zero solids → zero consumption."""
        consumption = flocculant_dosage(100, 0.0, 20)
        assert consumption == pytest.approx(0.0)
