"""Tests for minelab.production.reconciliation."""

import pytest

from minelab.production.reconciliation import (
    f_factors,
    reconciliation_report,
    variance_analysis,
)


class TestFFactors:
    """Tests for reconciliation F-factors."""

    def test_perfect_reconciliation(self):
        """Equal model/mined/plant → all F = 1.0."""
        result = f_factors(1000, 2.0, 1000, 2.0, 1000, 2.0)
        assert result["F1_tonnes"] == pytest.approx(1.0)
        assert result["F2_tonnes"] == pytest.approx(1.0)
        assert result["F3_tonnes"] == pytest.approx(1.0)

    def test_known_value(self):
        """Known example: model=1000, mined=1100, plant=950."""
        result = f_factors(1000, 2.0, 1100, 1.8, 950, 2.1)
        assert result["F1_tonnes"] == pytest.approx(1.1, rel=0.01)
        assert result["F2_tonnes"] == pytest.approx(950 / 1100, rel=0.01)
        assert result["F3_tonnes"] == pytest.approx(0.95, rel=0.01)


class TestReconciliationReport:
    """Tests for multi-period reconciliation."""

    def test_multiple_periods(self):
        """Should handle multiple periods."""
        data = [
            {"model_tonnes": 1000, "model_grade": 2.0,
             "mined_tonnes": 1050, "mined_grade": 1.9,
             "plant_tonnes": 980, "plant_grade": 2.1},
            {"model_tonnes": 1200, "model_grade": 1.8,
             "mined_tonnes": 1150, "mined_grade": 1.7,
             "plant_tonnes": 1100, "plant_grade": 1.9},
        ]
        result = reconciliation_report(data)
        assert len(result["periods"]) == 2
        assert "averages" in result


class TestVarianceAnalysis:
    """Tests for variance analysis."""

    def test_decomposition(self):
        """Tonnage + grade + combined should sum to total."""
        result = variance_analysis(1000, 2.0, 1100, 1.8)
        total = result["tonnage_effect"] + result["grade_effect"] + result["combined_effect"]
        assert total == pytest.approx(result["total_variance"], rel=0.01)

    def test_no_change(self):
        """Same planned/actual → zero variance."""
        result = variance_analysis(1000, 2.0, 1000, 2.0)
        assert result["total_variance"] == pytest.approx(0.0, abs=0.01)
