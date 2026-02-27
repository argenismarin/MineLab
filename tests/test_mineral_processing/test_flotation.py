"""Tests for minelab.mineral_processing.flotation."""

import numpy as np
import pytest

from minelab.mineral_processing.flotation import (
    flotation_bank_design,
    flotation_circuit,
    flotation_first_order,
    flotation_kelsall,
    flotation_kinetics_fit,
    selectivity_index,
)


class TestFlotationFirstOrder:
    """Tests for first-order flotation model."""

    def test_known_value(self):
        """R∞=0.95, k=0.5, t=10 → R ≈ 0.944."""
        r = flotation_first_order(0.95, 0.5, 10)
        assert r == pytest.approx(0.944, rel=0.01)

    def test_zero_time(self):
        """t=0 → R=0."""
        r = flotation_first_order(0.95, 0.5, 0)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_long_time(self):
        """t→∞ → R→R∞."""
        r = flotation_first_order(0.95, 0.5, 1000)
        assert r == pytest.approx(0.95, rel=0.001)

    def test_invalid_r_inf(self):
        """R∞ > 1 should raise."""
        with pytest.raises(ValueError):
            flotation_first_order(1.5, 0.5, 10)


class TestFlotationKelsall:
    """Tests for Kelsall two-component model."""

    def test_total_less_than_one(self):
        """Total R should be ≤ 1.0."""
        r = flotation_kelsall(0.6, 2.0, 0.3, 0.2, 100)
        assert r <= 1.0

    def test_zero_time(self):
        """t=0 → R=0."""
        r = flotation_kelsall(0.6, 2.0, 0.3, 0.2, 0)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_long_time(self):
        """t→∞ → R→R∞_fast + R∞_slow."""
        r = flotation_kelsall(0.6, 2.0, 0.3, 0.2, 1000)
        assert r == pytest.approx(0.9, rel=0.01)


class TestFlotationBankDesign:
    """Tests for flotation bank design."""

    def test_positive_cells(self):
        """Should return positive number of cells."""
        result = flotation_bank_design(0.9, 0.5, 10, 5)
        assert result["n_cells"] > 0

    def test_higher_recovery_more_cells(self):
        """Higher recovery target → more cells."""
        n_low = flotation_bank_design(0.8, 0.5, 10, 5)["n_cells"]
        n_high = flotation_bank_design(0.95, 0.5, 10, 5)["n_cells"]
        assert n_high >= n_low

    def test_residence_time(self):
        """τ = V/Q."""
        result = flotation_bank_design(0.9, 0.5, 10, 5)
        assert result["residence_time"] == pytest.approx(2.0)


class TestFlotationCircuit:
    """Tests for flotation circuit recovery."""

    def test_rougher_cleaner_only(self):
        """Without scavenger: overall = rougher * cleaner."""
        result = flotation_circuit(0.9, 0.8)
        assert result["overall_recovery"] == pytest.approx(0.72, rel=0.01)

    def test_with_scavenger(self):
        """Scavenger increases overall recovery."""
        no_scav = flotation_circuit(0.9, 0.8, 0.0)["overall_recovery"]
        with_scav = flotation_circuit(0.9, 0.8, 0.5)["overall_recovery"]
        assert with_scav > no_scav

    def test_overall_less_than_one(self):
        """Overall recovery ≤ 1."""
        result = flotation_circuit(0.95, 0.95, 0.95)
        assert result["overall_recovery"] <= 1.0


class TestSelectivityIndex:
    """Tests for Gaudin selectivity index."""

    def test_known_value(self):
        """Rm=0.9, Rg=0.1 → SI=81."""
        si = selectivity_index(0.9, 0.1)
        assert si == pytest.approx(81.0, rel=0.01)

    def test_perfect_selectivity(self):
        """High mineral, low gangue → high SI."""
        si = selectivity_index(0.95, 0.05)
        assert si > 100


class TestFlotationKineticsFit:
    """Tests for flotation kinetics fitting."""

    def test_recovers_parameters(self):
        """Should recover known k and R∞."""
        t = np.array([0, 1, 2, 5, 10, 20])
        r = 0.95 * (1 - np.exp(-0.5 * t))
        result = flotation_kinetics_fit(t, r)
        assert result["r_inf"] == pytest.approx(0.95, rel=0.05)
        assert result["k"] == pytest.approx(0.5, rel=0.1)
        assert result["r_squared"] > 0.99
