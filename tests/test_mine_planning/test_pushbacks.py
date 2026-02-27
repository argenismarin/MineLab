"""Tests for minelab.mine_planning.pushbacks."""

import numpy as np
import pytest

from minelab.mine_planning.pushbacks import (
    design_pushbacks,
    nested_pit_shells,
)


class TestNestedPitShells:
    """Tests for nested pit shells."""

    def test_multiple_shells(self):
        """Should return multiple shells."""
        values = np.array([[-1, -1, -1], [-1, 50, -1]])
        result = nested_pit_shells(values, [0.4, 0.6, 0.8, 1.0], (45, 45))
        assert len(result["shells"]) > 0

    def test_inner_subset_of_outer(self):
        """Inner shell ⊆ outer shell (monotonic nesting)."""
        values = np.array([[-1, -1, -1, -1, -1],
                          [-1, -1, 20, -1, -1],
                          [-1, 50, 80, 50, -1]])
        result = nested_pit_shells(values, [0.5, 1.0], (45, 45))
        if len(result["shells"]) >= 2:
            inner = result["shells"][0]
            outer = result["shells"][-1]
            # Inner should be subset of outer
            assert np.all(inner <= outer)


class TestDesignPushbacks:
    """Tests for pushback design from shells."""

    def test_returns_pushbacks(self):
        """Should return pushback phases."""
        shells = [
            np.array([[0, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 1]]),
            np.array([[1, 1, 1], [1, 1, 1]]),
        ]
        result = design_pushbacks(shells)
        assert "pushbacks" in result
        assert len(result["pushbacks"]) > 0


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestNestedPitShellsValidation:
    """Validation tests for nested_pit_shells."""

    def test_empty_revenue_factors_raises(self):
        """Empty revenue_factors list should raise ValueError."""
        bv = np.array([[-2, 10, -2], [-3, 20, -3]])
        with pytest.raises(ValueError, match="revenue_factors"):
            nested_pit_shells(bv, [], (45.0, 45.0))


class TestDesignPushbacksValidation:
    """Validation tests for design_pushbacks."""

    def test_empty_shells_raises(self):
        """Empty shells list should raise ValueError."""
        with pytest.raises(ValueError, match="shells"):
            design_pushbacks([])


class TestDesignPushbacksMerging:
    """Tests for pushback merging with min_tonnage constraints."""

    def test_flush_remaining_merges_into_last(self):
        """Remaining material below threshold merges into the last pushback."""
        # Shell 1: 4 blocks, Shell 2: 6 blocks total, Shell 3: 8 blocks total
        s1 = np.array([[False, True, True, False],
                       [False, True, True, False]])
        s2 = np.array([[True, True, True, False],
                       [True, True, True, False]])
        s3 = np.array([[True, True, True, True],
                       [True, True, True, False]])
        # min_tonnage=4 means pushbacks need at least 4 blocks
        # Incremental: s1=4 blocks (ok), s2-s1=2 blocks (<4, accumulate),
        # s3-s2=1 block (<4 total=3, still accumulating)
        # Flush: 3 accumulated blocks merge into last pushback
        result = design_pushbacks([s1, s2, s3], min_tonnage=4)
        assert len(result["pushbacks"]) == 1
        # Total blocks should be s3.sum() = 7
        assert sum(result["tonnages"]) == int(s3.sum())

    def test_flush_remaining_single_pushback(self):
        """When all incremental shells are too small, one pushback is returned."""
        s1 = np.array([[False, True, False]])
        s2 = np.array([[True, True, False]])
        # min_tonnage=10 means nothing meets threshold
        # Both incremental shells are too small, accumulate everything
        # Flush: no pushbacks exist, so create a single pushback
        result = design_pushbacks([s1, s2], min_tonnage=10)
        assert len(result["pushbacks"]) == 1
        assert result["tonnages"][0] == int(s2.sum())
