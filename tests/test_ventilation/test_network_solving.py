"""Tests for minelab.ventilation.network_solving."""

import pytest

from minelab.ventilation.network_solving import (
    hardy_cross,
    simple_network,
)


class TestSimpleNetwork:
    """Tests for simple series/parallel network."""

    def test_series(self):
        """Series: R_total = sum."""
        r = simple_network([1, 2, 3], "series")
        assert r == pytest.approx(6.0)

    def test_parallel(self):
        """Parallel: [4, 4] → 1.0."""
        r = simple_network([4, 4], "parallel")
        assert r == pytest.approx(1.0, rel=0.01)


class TestHardyCross:
    """Tests for Hardy Cross network solver."""

    def test_simple_converges(self):
        """Simple 2-branch parallel should converge."""
        branches = [
            {"from": 0, "to": 1, "resistance": 2.0, "initial_Q": 30.0},
            {"from": 0, "to": 1, "resistance": 8.0, "initial_Q": 20.0},
        ]
        result = hardy_cross(branches, 2)
        assert result["converged"]

    def test_flow_conservation(self):
        """Total flow in = total flow out."""
        branches = [
            {"from": 0, "to": 1, "resistance": 2.0, "initial_Q": 30.0},
            {"from": 0, "to": 1, "resistance": 8.0, "initial_Q": 20.0},
        ]
        result = hardy_cross(branches, 2)
        total = sum(result["flows"])
        assert total == pytest.approx(50.0, rel=0.01)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestHardyCrossValidation:
    """Validation tests for hardy_cross."""

    def test_empty_branches_raises(self):
        """Empty branches should raise ValueError."""
        with pytest.raises(ValueError, match="branches"):
            hardy_cross([], 2)

    def test_one_junction_raises(self):
        """junctions < 2 should raise ValueError."""
        branches = [
            {"from": 0, "to": 0, "resistance": 1.0, "Q_init": 10.0},
        ]
        with pytest.raises(ValueError, match="junctions"):
            hardy_cross(branches, 1)

    def test_missing_resistance_raises(self):
        """Branch missing 'resistance' key should raise ValueError."""
        branches = [{"from": 0, "to": 1, "Q_init": 10.0}]
        with pytest.raises(ValueError, match="resistance"):
            hardy_cross(branches, 2)

    def test_q_init_key(self):
        """Should accept 'Q_init' key for initial airflow."""
        branches = [
            {"from": 0, "to": 1, "resistance": 2.0, "Q_init": 30.0},
            {"from": 0, "to": 1, "resistance": 8.0, "Q_init": 20.0},
        ]
        result = hardy_cross(branches, 2)
        assert result["converged"]

    def test_missing_q_init_raises(self):
        """Branch missing both 'Q_init' and 'initial_Q' should raise ValueError."""
        branches = [{"from": 0, "to": 1, "resistance": 1.0}]
        with pytest.raises(ValueError, match="Q_init"):
            hardy_cross(branches, 2)

    def test_zero_resistance_denominator(self):
        """Zero resistance branches should handle zero denominator gracefully."""
        branches = [
            {"from": 0, "to": 1, "resistance": 0.0, "Q_init": 0.0},
            {"from": 0, "to": 1, "resistance": 0.0, "Q_init": 0.0},
        ]
        # Should not raise, denominator==0 means skip correction
        result = hardy_cross(branches, 2)
        assert "converged" in result


class TestSimpleNetworkValidation:
    """Validation tests for simple_network."""

    def test_empty_resistances_raises(self):
        """Empty resistances should raise ValueError."""
        with pytest.raises(ValueError, match="resistances"):
            simple_network([])

    def test_invalid_topology_raises(self):
        """Invalid topology string should raise ValueError."""
        with pytest.raises(ValueError, match="topology"):
            simple_network([1.0, 2.0], "diagonal")
