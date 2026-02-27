"""Tests for minelab.mine_planning.reserves."""

import pytest

from minelab.mine_planning.reserves import (
    dilution_ore_loss,
    resource_to_reserve,
)


class TestResourceToReserve:
    """Tests for resource to reserve conversion."""

    def test_dilution_increases_tonnage(self):
        """Dilution increases reserve tonnage."""
        result = resource_to_reserve(1000000, 2.0, 0.10, 0.05)
        assert result["reserve_tonnes"] > 1000000 * 0.95  # after ore loss

    def test_dilution_decreases_grade(self):
        """Dilution decreases reserve grade."""
        result = resource_to_reserve(1000000, 2.0, 0.10, 0.05)
        assert result["reserve_grade"] < 2.0

    def test_no_modifiers(self):
        """Zero dilution and zero ore loss → same as resource."""
        result = resource_to_reserve(1000000, 2.0, 0.0, 0.0, 1.0)
        assert result["reserve_tonnes"] == pytest.approx(1000000, rel=0.01)
        assert result["reserve_grade"] == pytest.approx(2.0, rel=0.01)


class TestDilutionOreLoss:
    """Tests for dilution and ore loss calculation."""

    def test_dilution(self):
        """More actual tonnage → positive dilution."""
        result = dilution_ore_loss(1000, 2.0, 1100, 1.8)
        assert result["dilution_pct"] > 0

    def test_ore_loss(self):
        """Less actual tonnage → positive ore loss."""
        result = dilution_ore_loss(1000, 2.0, 900, 2.1)
        assert result["ore_loss_pct"] > 0
