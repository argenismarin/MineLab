"""Tests for minelab.production.stockpiles."""

import pytest

from minelab.production.stockpiles import (
    stockpile_fifo,
    stockpile_lifo,
)


class TestStockpileFIFO:
    """Tests for FIFO stockpile."""

    def test_single_reclaim(self):
        """Add 100t@2%, reclaim 50t → grade 2%."""
        adds = [{"tonnes": 100, "grade": 2.0}]
        result = stockpile_fifo(adds, [50])
        assert result["reclaimed"][0]["grade"] == pytest.approx(2.0, rel=0.01)
        assert result["reclaimed"][0]["tonnes"] == pytest.approx(50, rel=0.01)

    def test_multi_add_reclaim(self):
        """FIFO: reclaims from first addition first."""
        adds = [
            {"tonnes": 100, "grade": 1.0},
            {"tonnes": 100, "grade": 3.0},
        ]
        result = stockpile_fifo(adds, [50])
        assert result["reclaimed"][0]["grade"] == pytest.approx(1.0, rel=0.01)

    def test_remaining(self):
        """Remaining should track what's left."""
        adds = [{"tonnes": 100, "grade": 2.0}]
        result = stockpile_fifo(adds, [30])
        total_remaining = sum(r["tonnes"] for r in result["remaining"])
        assert total_remaining == pytest.approx(70, rel=0.01)


class TestStockpileLIFO:
    """Tests for LIFO stockpile."""

    def test_lifo_order(self):
        """LIFO: reclaims from last addition first."""
        adds = [
            {"tonnes": 100, "grade": 1.0},
            {"tonnes": 100, "grade": 3.0},
        ]
        result = stockpile_lifo(adds, [50])
        assert result["reclaimed"][0]["grade"] == pytest.approx(3.0, rel=0.01)

    def test_remaining(self):
        """Remaining should be correct."""
        adds = [{"tonnes": 100, "grade": 2.0}]
        result = stockpile_lifo(adds, [40])
        total_remaining = sum(r["tonnes"] for r in result["remaining"])
        assert total_remaining == pytest.approx(60, rel=0.01)
