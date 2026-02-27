"""Tests for minelab.geostatistics.block_model."""

import numpy as np
import pytest

from minelab.geostatistics.block_model import BlockModel, block_grade_tonnage


class TestBlockModel:
    """Tests for BlockModel class."""

    def test_n_total(self):
        """10x10x5 model should have 500 blocks."""
        bm = BlockModel([0, 0, 0], [10, 10, 5], [10, 10, 5])
        assert bm.n_total == 500

    def test_block_centers_shape(self):
        """Block centers should have shape (n_total, 3)."""
        bm = BlockModel([0, 0, 0], [10, 10, 5], [10, 10, 5])
        centers = bm.block_centers()
        assert centers.shape == (500, 3)

    def test_block_centers_values(self):
        """First center should be at half-block from origin."""
        bm = BlockModel([100, 200, 300], [10, 20, 5], [2, 2, 2])
        centers = bm.block_centers()
        # First block center
        expected_first = [105, 210, 302.5]
        assert centers[0] == pytest.approx(expected_first)

    def test_add_get_variable(self):
        """Add and get variable should roundtrip."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [3, 3, 3])
        data = np.arange(27, dtype=float)
        bm.add_variable("grade", data)
        retrieved = bm.get_variable("grade")
        np.testing.assert_array_equal(retrieved, data)

    def test_add_wrong_length_raises(self):
        """Adding variable with wrong length should raise ValueError."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [3, 3, 3])
        with pytest.raises(ValueError, match="n_total"):
            bm.add_variable("grade", np.arange(10, dtype=float))

    def test_get_missing_variable_raises(self):
        """Getting non-existent variable should raise KeyError."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [2, 2, 2])
        with pytest.raises(KeyError, match="grade"):
            bm.get_variable("grade")

    def test_filter_blocks(self):
        """Filter should return correct indices."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [2, 2, 2])
        grades = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7])
        bm.add_variable("grade", grades)

        above = bm.filter_blocks("grade", ">", 0.5)
        assert set(above) == {3, 5, 7}

        below_eq = bm.filter_blocks("grade", "<=", 0.3)
        assert set(below_eq) == {0, 2, 4}

    def test_variables_list(self):
        """variables property should list added variables."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [2, 2, 2])
        bm.add_variable("grade", np.zeros(8))
        bm.add_variable("density", np.ones(8))
        assert set(bm.variables) == {"grade", "density"}

    def test_invalid_dimensions(self):
        """Non-3D inputs should raise ValueError."""
        with pytest.raises(ValueError, match="3 elements"):
            BlockModel([0, 0], [10, 10], [5, 5])


class TestBlockGradeTonnage:
    """Tests for grade-tonnage curve computation."""

    def test_tonnage_decreasing(self):
        """Tonnage should decrease with increasing cutoff."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [5, 5, 2])
        rng = np.random.default_rng(42)
        bm.add_variable("grade", rng.lognormal(0, 0.5, bm.n_total))
        bm.add_variable("density", np.full(bm.n_total, 2.7))
        gt = block_grade_tonnage(bm, "grade", "density", [0.5, 1.0, 1.5, 2.0])
        assert gt["tonnage"].is_monotonic_decreasing

    def test_grade_increasing(self):
        """Mean grade should increase with increasing cutoff (above ore only)."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [5, 5, 2])
        rng = np.random.default_rng(42)
        bm.add_variable("grade", rng.lognormal(0, 0.5, bm.n_total))
        bm.add_variable("density", np.full(bm.n_total, 2.7))
        gt = block_grade_tonnage(bm, "grade", "density", [0.5, 1.0, 1.5, 2.0])
        nonzero = gt[gt["tonnage"] > 0]
        if len(nonzero) > 1:
            assert nonzero["mean_grade"].is_monotonic_increasing

    def test_output_columns(self):
        """DataFrame should have correct columns."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [2, 2, 2])
        bm.add_variable("grade", np.ones(8))
        bm.add_variable("density", np.full(8, 2.7))
        gt = block_grade_tonnage(bm, "grade", "density", [0.5, 1.5])
        assert list(gt.columns) == ["cutoff", "tonnage", "mean_grade", "metal"]

    def test_zero_tonnage_at_high_cutoff(self):
        """Cutoff above max grade should give zero tonnage."""
        bm = BlockModel([0, 0, 0], [10, 10, 10], [2, 2, 2])
        bm.add_variable("grade", np.ones(8) * 0.5)
        bm.add_variable("density", np.full(8, 2.7))
        gt = block_grade_tonnage(bm, "grade", "density", [1.0])
        assert gt["tonnage"].iloc[0] == 0.0
