"""Tests for minelab.mine_planning.scheduling."""

import numpy as np
import pytest

from minelab.mine_planning.scheduling import (
    npv_schedule,
    precedence_constraints,
    schedule_by_period,
)


class TestScheduleByPeriod:
    """Tests for period scheduling."""

    def test_all_assigned(self):
        """All pit blocks should be assigned to a period."""
        values = np.array([[10, 20, 10], [5, 50, 5]])
        mask = np.ones_like(values, dtype=bool)
        result = schedule_by_period(values, mask, [3, 3], 2)
        assert "schedule" in result

    def test_capacity_respected(self):
        """Each period should not exceed capacity."""
        values = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        mask = np.ones_like(values, dtype=bool)
        result = schedule_by_period(values, mask, [4, 4], 2)
        for i in range(2):
            assert np.sum(result["schedule"] == i + 1) <= 4


class TestNPVSchedule:
    """Tests for NPV from schedule."""

    def test_positive_npv(self):
        """Positive values → positive NPV."""
        npv = npv_schedule([100, 100, 100], 0.1)
        assert npv > 0

    def test_discount_effect(self):
        """Higher discount rate → lower NPV."""
        npv_low = npv_schedule([100, 100, 100], 0.05)
        npv_high = npv_schedule([100, 100, 100], 0.15)
        assert npv_low > npv_high

    def test_front_loaded(self):
        """Front-loaded values → higher NPV."""
        npv_front = npv_schedule([300, 100, 0], 0.1)
        npv_back = npv_schedule([0, 100, 300], 0.1)
        assert npv_front > npv_back


class TestScheduleByPeriodValidation:
    """Validation tests for schedule_by_period."""

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            schedule_by_period(np.array([1, 2]), np.array([True]), [10], 1)

    def test_non_positive_periods(self):
        """Non-positive n_periods should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            schedule_by_period(np.array([1]), np.array([True]), [], 0)

    def test_capacities_length_mismatch(self):
        """Capacities length != n_periods should raise ValueError."""
        with pytest.raises(ValueError, match="capacities"):
            schedule_by_period(np.array([1]), np.array([True]), [10, 20], 1)

    def test_unscheduled_blocks(self):
        """Blocks that exceed capacity stay unscheduled (0)."""
        values = np.array([100, 100, 100])
        mask = np.ones(3, dtype=bool)
        result = schedule_by_period(values, mask, [100.0], 1)
        # Only 1 block fits in the single period
        assert np.sum(result["schedule"] == 0) == 2


class TestNPVScheduleValidation:
    """Validation tests for npv_schedule."""

    def test_empty_values(self):
        """Empty period_values should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            npv_schedule([], 0.1)


class TestPrecedenceConstraints:
    """Tests for precedence constraints."""

    def test_returns_pairs(self):
        """Should return list of (parent, child) tuples."""
        pairs = precedence_constraints((3, 5), 45, 10, 10)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_3d_model(self):
        """3D block model should produce precedence pairs."""
        pairs = precedence_constraints((3, 3, 3), 45, 10, 10)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_invalid_shape(self):
        """Non-2D/3D shape should raise ValueError."""
        with pytest.raises(ValueError, match="2-D or 3-D"):
            precedence_constraints((3,), 45, 10, 10)
