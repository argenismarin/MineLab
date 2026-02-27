"""Tests for minelab.mineral_processing.classification."""

import numpy as np
import pytest

from minelab.mineral_processing.classification import (
    gates_gaudin_schuhmann,
    lynch_rao_partition,
    plitt_model,
    rosin_rammler,
    screen_efficiency,
    tromp_curve,
)


class TestPlittModel:
    """Tests for Plitt hydrocyclone model."""

    def test_positive_d50c(self):
        """d50c should be positive."""
        result = plitt_model(0.25, 0.075, 0.1, 0.3, 0.05, 50, 0.1, 2700)
        assert result["d50c"] > 0

    def test_smaller_cyclone_finer(self):
        """Smaller cyclone → finer cut."""
        large = plitt_model(0.5, 0.15, 0.2, 0.6, 0.1, 50, 0.1, 2700)
        small = plitt_model(0.25, 0.075, 0.1, 0.3, 0.05, 50, 0.1, 2700)
        assert small["d50c"] < large["d50c"]


class TestScreenEfficiency:
    """Tests for screen efficiency."""

    def test_known_value(self):
        """Good separation → high efficiency."""
        e = screen_efficiency(1000, 600, 0.05, 0.03)
        assert e == pytest.approx(0.95 * 0.97, rel=0.01)

    def test_perfect_screen(self):
        """No misplacement → efficiency = 1.0."""
        e = screen_efficiency(1000, 600, 0.0, 0.0)
        assert e == pytest.approx(1.0)


class TestLynchRaoPartition:
    """Tests for Lynch-Rao partition curve."""

    def test_at_d50c(self):
        """At d50c, partition = 0.5."""
        y = lynch_rao_partition(75, 3.0, np.array([75.0]))
        assert float(y[0]) == pytest.approx(0.5, rel=0.01)

    def test_monotonic(self):
        """Partition increases with size."""
        sizes = np.array([25, 50, 75, 100, 150])
        y = lynch_rao_partition(75, 3.0, sizes)
        assert np.all(np.diff(y) > 0)


class TestTrompCurve:
    """Tests for Tromp partition curve."""

    def test_range(self):
        """Partition values should be in [0, 1]."""
        feed = np.array([0.1, 0.3, 0.4, 0.2])
        uf = np.array([0.02, 0.1, 0.48, 0.4])
        of = np.array([0.18, 0.5, 0.32, 0.0])
        pc = tromp_curve(feed, of, uf, 0.5)
        assert np.all(pc >= 0)
        assert np.all(pc <= 1)


class TestRosinRammler:
    """Tests for Rosin-Rammler distribution."""

    def test_at_k(self):
        """At x=k, F=0.632."""
        f = rosin_rammler(np.array([100.0]), 100.0, 1.5)
        assert float(f[0]) == pytest.approx(0.632, rel=0.01)

    def test_zero(self):
        """At x=0, F=0."""
        f = rosin_rammler(np.array([0.0]), 100.0, 1.5)
        assert float(f[0]) == pytest.approx(0.0, abs=1e-10)

    def test_monotonic(self):
        """F increases with size."""
        sizes = np.array([10, 50, 100, 200, 500])
        f = rosin_rammler(sizes, 100.0, 1.5)
        assert np.all(np.diff(f) > 0)


class TestGatesGaudinSchuhmann:
    """Tests for GGS distribution."""

    def test_at_k(self):
        """At x=k, F=1.0."""
        f = gates_gaudin_schuhmann(np.array([100.0]), 100.0, 0.5)
        assert float(f[0]) == pytest.approx(1.0, rel=0.01)

    def test_beyond_k(self):
        """At x>k, F=1.0."""
        f = gates_gaudin_schuhmann(np.array([200.0]), 100.0, 0.5)
        assert float(f[0]) == pytest.approx(1.0)
