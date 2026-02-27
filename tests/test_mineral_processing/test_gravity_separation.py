"""Tests for minelab.mineral_processing.gravity_separation."""

import pytest

from minelab.mineral_processing.gravity_separation import (
    concentration_criterion,
    dms_cutpoint,
    humphreys_spiral_recovery,
    shaking_table_efficiency,
)


class TestConcentrationCriterion:
    """Tests for gravity separation concentration criterion."""

    def test_gold_quartz(self):
        """Gold(19.3)/quartz(2.65) in water → CC ≈ 11.1."""
        result = concentration_criterion(19.3, 2.65, 1.0)
        assert result["cc"] == pytest.approx(11.1, rel=0.02)
        assert result["feasibility"] == "Easy separation"

    def test_easy_separation(self):
        """CC > 2.5 → Easy."""
        result = concentration_criterion(7.5, 2.65, 1.0)
        assert result["cc"] > 2.5
        assert "Easy" in result["feasibility"]

    def test_difficult(self):
        """Similar densities → difficult."""
        result = concentration_criterion(2.7, 2.65, 1.0)
        assert result["cc"] < 1.25


class TestHumphreysSpiral:
    """Tests for Humphreys spiral recovery estimation."""

    def test_positive_recovery(self):
        """Should return positive recovery."""
        result = humphreys_spiral_recovery(5.0, 0.05)
        assert result["estimated_recovery"] > 0

    def test_higher_cc_higher_recovery(self):
        """Higher CC → higher recovery."""
        r_low = humphreys_spiral_recovery(1.5, 0.05)["estimated_recovery"]
        r_high = humphreys_spiral_recovery(5.0, 0.05)["estimated_recovery"]
        assert r_high > r_low


class TestDMSCutpoint:
    """Tests for DMS cut-point analysis."""

    def test_sink(self):
        """Heavy particle → sink."""
        result = dms_cutpoint(2.8, 3.5)
        assert result["reports_to"] == "sink"

    def test_float(self):
        """Light particle → float."""
        result = dms_cutpoint(2.8, 2.0)
        assert result["reports_to"] == "float"


class TestShakingTableEfficiency:
    """Tests for shaking table separation efficiency."""

    def test_known_value(self):
        """f=0.05, c=0.40, t=0.01 → E ≈ 88.5%."""
        e = shaking_table_efficiency(0.05, 0.40, 0.01)
        expected = 0.40 * (0.05 - 0.01) / (0.05 * (0.40 - 0.01)) * 100
        assert e == pytest.approx(expected, rel=0.01)

    def test_perfect_separation(self):
        """Zero tail grade → 100% efficiency."""
        e = shaking_table_efficiency(0.05, 1.0, 0.0)
        assert e == pytest.approx(100.0, rel=0.01)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestConcentrationCriterionCategories:
    """Test CC feasibility categories."""

    def test_possible_75_micrometers(self):
        """CC between 1.75 and 2.5 should report 'Possible down to 75 micrometers'."""
        # CC = (sg_heavy - 1.0) / (sg_light - 1.0)
        # Need 1.75 < CC <= 2.5
        # sg_heavy=4.5, sg_light=2.0: CC = (4.5-1)/(2.0-1) = 3.5/1 = 3.5 too high
        # sg_heavy=3.75, sg_light=2.5: CC = (3.75-1)/(2.5-1) = 2.75/1.5 = 1.833
        result = concentration_criterion(3.75, 2.5, 1.0)
        assert 1.75 < result["cc"] <= 2.5
        assert result["feasibility"] == "Possible down to 75 micrometers"

    def test_possible_6mm(self):
        """CC between 1.25 and 1.75 should report 'Possible down to 6 mm'."""
        # sg_heavy=3.0, sg_light=2.5: CC = (3.0-1)/(2.5-1) = 2.0/1.5 = 1.333
        result = concentration_criterion(3.0, 2.5, 1.0)
        assert 1.25 < result["cc"] <= 1.75
        assert result["feasibility"] == "Possible down to 6 mm"


class TestHumphreysSpiralCategories:
    """Test Humphreys spiral recovery for different CC ranges."""

    def test_moderate_cc(self):
        """CC between 1.75 and 2.5 uses the moderate recovery formula."""
        result = humphreys_spiral_recovery(2.0, 0.05)
        # recovery = 0.5 + 0.1 * 2.0 = 0.7
        assert result["estimated_recovery"] == pytest.approx(0.7, rel=1e-4)

    def test_low_cc(self):
        """CC <= 1.75 uses the low recovery formula."""
        result = humphreys_spiral_recovery(1.5, 0.05)
        # recovery = max(0.1, 0.3 * 1.5) = max(0.1, 0.45) = 0.45
        assert result["estimated_recovery"] == pytest.approx(0.45, rel=1e-4)


class TestShakingTableEdgeCases:
    """Edge case tests for shaking_table_efficiency."""

    def test_equal_conc_tail_grade(self):
        """When conc_grade == tail_grade, efficiency should be 0."""
        e = shaking_table_efficiency(0.05, 0.5, 0.5)
        assert e == 0.0

    def test_zero_feed_grade(self):
        """When feed_grade == 0, efficiency should be 0."""
        e = shaking_table_efficiency(0.0, 0.0, 0.0)
        assert e == 0.0
