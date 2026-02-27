"""Tests for minelab.equipment.fuel_consumption."""

import pytest

from minelab.equipment.fuel_consumption import (
    fuel_consumption_rate,
    fuel_cost_per_tonne,
)


class TestFuelConsumptionRate:
    """Tests for fuel consumption rate."""

    def test_positive(self):
        """Consumption should be positive."""
        rate = fuel_consumption_rate(500, 0.6)
        assert rate > 0

    def test_higher_load_more_fuel(self):
        """Higher load factor → more fuel."""
        r_low = fuel_consumption_rate(500, 0.3)
        r_high = fuel_consumption_rate(500, 0.8)
        assert r_high > r_low

    def test_known_value(self):
        """500kW * 0.6 * 0.24 = 72 L/h."""
        rate = fuel_consumption_rate(500, 0.6, 0.24)
        assert rate == pytest.approx(72.0, rel=0.01)


class TestFuelCostPerTonne:
    """Tests for fuel cost per tonne."""

    def test_positive(self):
        """Cost should be positive."""
        cost = fuel_cost_per_tonne(72, 1.5, 500)
        assert cost > 0

    def test_known_value(self):
        """72 L/h * $1.5/L / 500 t/h = $0.216/t."""
        cost = fuel_cost_per_tonne(72, 1.5, 500)
        assert cost == pytest.approx(0.216, rel=0.01)
