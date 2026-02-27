"""Tests for minelab.environmental.closure."""

import pytest

from minelab.environmental.closure import (
    acid_rock_drainage_neutralisation_cost,
    bond_amount,
    closure_cost_estimate,
    post_closure_water_management_cost,
    revegetation_success_probability,
)


# ---------------------------------------------------------------------------
# closure_cost_estimate
# ---------------------------------------------------------------------------


class TestClosureCostEstimate:
    """Tests for closure_cost_estimate."""

    def test_tailings_multiplier(self):
        """Tailings disturbance: 100 ha * 5000 * 1.5 = 750000."""
        result = closure_cost_estimate(100, "tailings", 5000)
        assert result == pytest.approx(750_000.0, rel=1e-4)

    def test_pit_multiplier(self):
        """Pit disturbance: 50 ha * 8000 * 2.0 = 800000."""
        result = closure_cost_estimate(50, "pit", 8000)
        assert result == pytest.approx(800_000.0, rel=1e-4)

    def test_waste_dump_multiplier(self):
        """Waste dump: 200 ha * 3000 * 1.2 = 720000."""
        result = closure_cost_estimate(200, "waste_dump", 3000)
        assert result == pytest.approx(720_000.0, rel=1e-4)

    def test_infrastructure_multiplier(self):
        """Infrastructure: 30 ha * 10000 * 0.8 = 240000."""
        result = closure_cost_estimate(30, "infrastructure", 10000)
        assert result == pytest.approx(240_000.0, rel=1e-4)

    def test_unknown_type_default_multiplier(self):
        """Unknown type defaults to 1.0: 100 * 5000 = 500000."""
        result = closure_cost_estimate(100, "other", 5000)
        assert result == pytest.approx(500_000.0, rel=1e-4)

    def test_invalid_area(self):
        """Negative area raises ValueError."""
        with pytest.raises(ValueError, match="disturbed_area_ha"):
            closure_cost_estimate(-10, "tailings", 5000)

    def test_invalid_unit_cost(self):
        """Zero unit cost raises ValueError."""
        with pytest.raises(ValueError, match="unit_cost"):
            closure_cost_estimate(100, "tailings", 0)

    def test_monotonic_with_area(self):
        """Larger area -> higher cost."""
        small = closure_cost_estimate(50, "tailings", 5000)
        large = closure_cost_estimate(100, "tailings", 5000)
        assert large > small


# ---------------------------------------------------------------------------
# bond_amount
# ---------------------------------------------------------------------------


class TestBondAmount:
    """Tests for bond_amount."""

    def test_known_value(self):
        """PV of $1M at 8% over 10 years = 463193.49."""
        result = bond_amount(1_000_000, 0.08, 10)
        assert result == pytest.approx(463_193.49, rel=1e-4)

    def test_zero_years_limit(self):
        """At t->0, bond approaches closure cost. Use t=0.01."""
        result = bond_amount(1_000_000, 0.10, 0.01)
        assert result == pytest.approx(1_000_000, rel=0.01)

    def test_higher_rate_lower_bond(self):
        """Higher discount rate -> lower bond."""
        low_rate = bond_amount(1_000_000, 0.05, 10)
        high_rate = bond_amount(1_000_000, 0.15, 10)
        assert high_rate < low_rate

    def test_invalid_cost(self):
        """Zero closure cost raises ValueError."""
        with pytest.raises(ValueError, match="npv_closure_cost"):
            bond_amount(0, 0.08, 10)

    def test_invalid_rate(self):
        """Negative discount rate raises ValueError."""
        with pytest.raises(ValueError, match="discount_rate"):
            bond_amount(1_000_000, -0.05, 10)

    def test_invalid_years(self):
        """Zero years raises ValueError."""
        with pytest.raises(ValueError, match="years_to_closure"):
            bond_amount(1_000_000, 0.08, 0)


# ---------------------------------------------------------------------------
# revegetation_success_probability
# ---------------------------------------------------------------------------


class TestRevegetationSuccessProbability:
    """Tests for revegetation_success_probability."""

    def test_ideal_conditions(self):
        """High rainfall, gentle slope, deep topsoil, quality=1."""
        result = revegetation_success_probability(600, 10, 300, 1.0)
        # rain=1.0, slope=max(0.3, 1-10/50)=0.8, soil=1.0, q=1.0
        assert result["probability"] == pytest.approx(0.8, rel=1e-4)
        assert result["risk_class"] == "low"

    def test_poor_conditions(self):
        """Low rainfall, steep slope, thin topsoil, poor quality."""
        result = revegetation_success_probability(150, 40, 50, 0.3)
        # rain=0.25, slope=max(0.3,1-40/50)=0.3, soil=0.167, q=0.3
        expected = 0.25 * 0.3 * (50 / 300) * 0.3
        assert result["probability"] == pytest.approx(expected, rel=1e-4)
        assert result["risk_class"] == "high"

    def test_medium_risk(self):
        """Moderate conditions yielding medium risk class."""
        result = revegetation_success_probability(500, 5, 250, 0.9)
        # rain=500/600=0.8333, slope=1-5/50=0.9, soil=250/300=0.8333
        expected = (500 / 600) * 0.9 * (250 / 300) * 0.9
        assert result["probability"] == pytest.approx(expected, rel=1e-4)
        assert result["risk_class"] == "medium"

    def test_rain_factor_capped(self):
        """Rainfall > 600 mm caps rain_factor at 1.0."""
        result = revegetation_success_probability(1200, 0, 300, 1.0)
        assert result["rain_factor"] == pytest.approx(1.0, rel=1e-4)

    def test_slope_factor_floored(self):
        """Very steep slope floors slope_factor at 0.3."""
        result = revegetation_success_probability(600, 100, 300, 1.0)
        assert result["slope_factor"] == pytest.approx(0.3, rel=1e-4)

    def test_invalid_rainfall(self):
        """Negative rainfall raises ValueError."""
        with pytest.raises(ValueError, match="rainfall_mm"):
            revegetation_success_probability(-100, 10, 300, 1.0)

    def test_invalid_seed_quality(self):
        """Seed quality > 1 raises ValueError."""
        with pytest.raises(ValueError, match="seed_mix_quality"):
            revegetation_success_probability(600, 10, 300, 1.5)


# ---------------------------------------------------------------------------
# acid_rock_drainage_neutralisation_cost
# ---------------------------------------------------------------------------


class TestAcidRockDrainageNeutralisationCost:
    """Tests for acid_rock_drainage_neutralisation_cost."""

    def test_known_value(self):
        """NAPP=30, 1Mt waste, lime=$50/t."""
        result = acid_rock_drainage_neutralisation_cost(
            30, 1_000_000, 50
        )
        # lime_ratio = 30/1000 * 1.02 = 0.0306
        # lime_tonnes = 0.0306 * 1e6 = 30600
        # total_cost = 30600 * 50 = 1530000
        assert result["lime_required_tonnes"] == pytest.approx(
            30_600.0, rel=1e-4
        )
        assert result["total_cost"] == pytest.approx(
            1_530_000.0, rel=1e-4
        )
        assert result["cost_per_tonne_waste"] == pytest.approx(
            1.53, rel=1e-4
        )

    def test_higher_napp_higher_cost(self):
        """Higher NAPP -> more lime -> higher cost."""
        low = acid_rock_drainage_neutralisation_cost(10, 1000, 50)
        high = acid_rock_drainage_neutralisation_cost(50, 1000, 50)
        assert high["total_cost"] > low["total_cost"]

    def test_cost_proportional_to_tonnage(self):
        """Doubling tonnage doubles lime and cost."""
        single = acid_rock_drainage_neutralisation_cost(20, 500, 50)
        double = acid_rock_drainage_neutralisation_cost(20, 1000, 50)
        assert double["lime_required_tonnes"] == pytest.approx(
            2 * single["lime_required_tonnes"], rel=1e-4
        )
        assert double["total_cost"] == pytest.approx(
            2 * single["total_cost"], rel=1e-4
        )

    def test_invalid_napp(self):
        """Zero NAPP raises ValueError."""
        with pytest.raises(ValueError, match="napp_kg_t"):
            acid_rock_drainage_neutralisation_cost(0, 1000, 50)

    def test_invalid_tonnage(self):
        """Negative tonnage raises ValueError."""
        with pytest.raises(ValueError, match="tonnes_acid_forming"):
            acid_rock_drainage_neutralisation_cost(30, -100, 50)


# ---------------------------------------------------------------------------
# post_closure_water_management_cost
# ---------------------------------------------------------------------------


class TestPostClosureWaterManagementCost:
    """Tests for post_closure_water_management_cost."""

    def test_known_value(self):
        """10000 m3/y * $2.5/m3 * 20 years = $500000."""
        result = post_closure_water_management_cost(10_000, 2.5, 20)
        assert result == pytest.approx(500_000.0, rel=1e-4)

    def test_proportional_to_years(self):
        """Doubling years doubles cost."""
        short = post_closure_water_management_cost(5000, 3.0, 10)
        long = post_closure_water_management_cost(5000, 3.0, 20)
        assert long == pytest.approx(2 * short, rel=1e-4)

    def test_proportional_to_rate(self):
        """Doubling seepage rate doubles cost."""
        low = post_closure_water_management_cost(5000, 3.0, 10)
        high = post_closure_water_management_cost(10_000, 3.0, 10)
        assert high == pytest.approx(2 * low, rel=1e-4)

    def test_invalid_seepage(self):
        """Zero seepage raises ValueError."""
        with pytest.raises(ValueError, match="seepage_rate_m3y"):
            post_closure_water_management_cost(0, 2.5, 20)

    def test_invalid_treatment_cost(self):
        """Negative treatment cost raises ValueError."""
        with pytest.raises(ValueError, match="treatment_cost_per_m3"):
            post_closure_water_management_cost(10_000, -1, 20)
