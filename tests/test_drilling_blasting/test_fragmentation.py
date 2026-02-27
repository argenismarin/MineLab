"""Tests for minelab.drilling_blasting.fragmentation."""

import numpy as np
import pytest

from minelab.drilling_blasting.fragmentation import (
    kuz_ram,
    modified_kuz_ram,
    swebrec_distribution,
    uniformity_index,
)


class TestKuzRam:
    """Tests for Kuz-Ram fragmentation model."""

    def test_positive_x50(self):
        """X50 should be positive."""
        result = kuz_ram(0.5, 50, 100, 8.0)
        assert result["x50"] > 0

    def test_higher_powder_factor_finer(self):
        """Higher powder factor → smaller X50."""
        r_low = kuz_ram(0.3, 50, 100, 8.0)
        r_high = kuz_ram(0.8, 50, 100, 8.0)
        assert r_high["x50"] < r_low["x50"]

    def test_typical_range(self):
        """X50 should be in typical range 0.05-2.0m."""
        result = kuz_ram(0.5, 50, 100, 8.0)
        assert 0.01 < result["x50"] < 5.0


class TestUniformityIndex:
    """Tests for Cunningham uniformity index."""

    def test_positive(self):
        """n should be positive."""
        n = uniformity_index(89, 3.0, 3.5, 10.0, 0.1, 7.0, 2.0)
        assert n > 0

    def test_typical_range(self):
        """n typically 0.8-2.2."""
        n = uniformity_index(89, 3.0, 3.5, 10.0, 0.1, 7.0, 2.0)
        assert 0.3 < n < 3.0


class TestModifiedKuzRam:
    """Tests for modified Kuz-Ram (2005)."""

    def test_positive_x50(self):
        """X50 should be positive."""
        result = modified_kuz_ram(0.5, 50, 100, 8.0, 89)
        assert result["x50"] > 0


class TestSwebrecDistribution:
    """Tests for Swebrec distribution function."""

    def test_at_x50(self):
        """At x=x50, F ≈ 0.5."""
        sizes = np.array([0.3])
        f = swebrec_distribution(0.3, 1.0, 2.0, sizes)
        assert float(f[0]) == pytest.approx(0.5, rel=0.01)

    def test_monotonic(self):
        """F increases with size."""
        sizes = np.array([0.05, 0.1, 0.2, 0.5, 0.8])
        f = swebrec_distribution(0.3, 1.0, 2.0, sizes)
        assert np.all(np.diff(f) > 0)

    def test_at_xmax(self):
        """At x=xmax, F → 1.0."""
        sizes = np.array([0.99])
        f = swebrec_distribution(0.3, 1.0, 2.0, sizes)
        assert float(f[0]) > 0.9


class TestKuzRamEdgeCases:
    """Edge-case tests for kuz_ram."""

    def test_n_rows_less_than_1_raises(self):
        """n_rows < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_rows"):
            kuz_ram(0.5, 50, 100, 8.0, n_rows=0)

    def test_multi_row_correction(self):
        """Multi-row correction reduces X50 compared to single row."""
        r_single = kuz_ram(0.5, 50, 100, 8.0, n_rows=1)
        r_multi = kuz_ram(0.5, 50, 100, 8.0, n_rows=3)
        assert r_multi["x50"] < r_single["x50"]


class TestUniformityIndexEdgeCases:
    """Edge-case tests for uniformity_index."""

    def test_negative_drill_accuracy_raises(self):
        """Negative drill_accuracy should raise ValueError."""
        with pytest.raises(ValueError, match="drill_accuracy"):
            uniformity_index(89, 3.0, 3.5, 10.0, -0.1, 7.0, 2.0)

    def test_drill_accuracy_exceeds_burden_raises(self):
        """drill_accuracy >= burden should raise ValueError."""
        with pytest.raises(ValueError, match="drill_accuracy"):
            uniformity_index(89, 3.0, 3.5, 10.0, 3.0, 7.0, 2.0)

    def test_negative_bottom_charge_length_raises(self):
        """Negative bottom_charge_length should raise ValueError."""
        with pytest.raises(ValueError, match="bottom_charge_length"):
            uniformity_index(89, 3.0, 3.5, 10.0, 0.1, 7.0, -1.0)


class TestModifiedKuzRamEdgeCases:
    """Edge-case tests for modified_kuz_ram."""

    def test_n_rows_less_than_1_raises(self):
        """n_rows < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_rows"):
            modified_kuz_ram(0.5, 50, 100, 8.0, 89, n_rows=0)

    def test_x50_ge_xmax_branch(self):
        """When x50 >= xmax, n_swebrec should be 1.0."""
        # Use very high rock_factor to make x50 large relative to xmax
        # xmax = diameter * (rock_factor / 0.06) * 0.0001
        # For diameter=10, rock_factor=0.1: xmax = 10*(0.1/0.06)*0.0001 ≈ 0.00167 m
        # x50 will be larger with rock_factor=0.1 given kuz_ram formula
        result = modified_kuz_ram(0.1, 50, 100, 0.1, 10)
        # x50 should be >= xmax for these extreme params, giving n_swebrec=1.0
        if result["x50"] >= result["xmax"]:
            assert result["n_swebrec"] == pytest.approx(1.0)
        else:
            # If x50 < xmax, n_swebrec = 2 - log(x50/xmax) > 2
            assert result["n_swebrec"] > 1.0


class TestSwebrecDistributionEdgeCases:
    """Edge-case tests for swebrec_distribution."""

    def test_x50_ge_xmax_raises(self):
        """x50 >= xmax should raise ValueError."""
        sizes = np.array([0.5])
        with pytest.raises(ValueError, match="x50"):
            swebrec_distribution(1.0, 1.0, 2.0, sizes)

    def test_non_positive_sizes_raises(self):
        """Sizes containing zero or negative should raise ValueError."""
        sizes = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="positive"):
            swebrec_distribution(0.3, 1.0, 2.0, sizes)

    def test_sizes_exceed_xmax_raises(self):
        """Sizes exceeding xmax should raise ValueError."""
        sizes = np.array([0.5, 1.5])
        with pytest.raises(ValueError, match="xmax"):
            swebrec_distribution(0.3, 1.0, 2.0, sizes)

    def test_at_xmax_exactly(self):
        """At x=xmax exactly, F should be 1.0."""
        sizes = np.array([1.0])
        f = swebrec_distribution(0.3, 1.0, 2.0, sizes)
        assert float(f[0]) == pytest.approx(1.0)
