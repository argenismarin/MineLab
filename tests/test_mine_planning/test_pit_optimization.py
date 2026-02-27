"""Tests for minelab.mine_planning.pit_optimization."""

import numpy as np
import pytest

from minelab.mine_planning.pit_optimization import (
    block_economic_value,
    lerchs_grossmann_2d,
    pseudoflow_3d,
)


class TestLerchsGrossmann2D:
    """Tests for 2D Lerchs-Grossmann pit optimizer."""

    def test_all_positive(self):
        """All positive blocks → mine everything."""
        values = np.array([[1, 1, 1], [1, 1, 1]])
        result = lerchs_grossmann_2d(values, (45, 45))
        assert result["total_value"] > 0

    def test_all_negative(self):
        """All negative blocks → mine nothing."""
        values = np.array([[-10, -10, -10], [-10, -10, -10]])
        result = lerchs_grossmann_2d(values, (45, 45))
        assert result["total_value"] <= 0 or np.sum(result["pit_mask"]) == 0

    def test_high_value_center(self):
        """High value center block should be mined."""
        values = np.array([[-1, -1, -1], [-1, 100, -1]])
        result = lerchs_grossmann_2d(values, (45, 45))
        assert result["total_value"] > 0


class TestPseudoflow3D:
    """Tests for simplified 3D pit optimization."""

    def test_positive_pit(self):
        """Should find positive pit in favorable model."""
        values = np.ones((3, 3, 3)) * -1
        values[2, 1, 1] = 100  # high value at bottom center
        values[1, 1, 1] = 50
        values[0, 1, 1] = -1
        result = pseudoflow_3d(values, (45, 45, 45, 45))
        assert result["total_value"] > 0


class TestBlockEconomicValue:
    """Tests for block economic value."""

    def test_positive_value(self):
        """High-grade block → positive value."""
        v = block_economic_value(5.0, 1000, 8000, 0.9, 3.0, 15.0)
        assert v > 0

    def test_zero_grade(self):
        """Zero grade → negative value (mining cost only)."""
        v = block_economic_value(0.0, 1000, 8000, 0.9, 3.0, 15.0)
        assert v < 0

    def test_breakeven(self):
        """At breakeven grade, value ≈ 0."""
        # COG = (mining + processing) / (price * recovery)
        # COG = (3 + 15) / (8000 * 0.9) = 0.0025 = 0.25%
        v = block_economic_value(0.0025, 1000, 8000, 0.9, 3.0, 15.0)
        assert abs(v) < 1  # near zero
