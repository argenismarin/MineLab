"""Tests for minelab.mineral_processing.magnetic_separation."""

import pytest

from minelab.mineral_processing.magnetic_separation import (
    davis_tube_recovery,
    magnetic_susceptibility_classify,
)


class TestMagneticSusceptibilityClassify:
    """Tests for magnetic susceptibility classification."""

    def test_magnetite_ferromagnetic(self):
        """Magnetite → ferromagnetic."""
        result = magnetic_susceptibility_classify(["magnetite"])
        assert result[0]["class"] == "ferromagnetic"

    def test_quartz_diamagnetic(self):
        """Quartz → diamagnetic."""
        result = magnetic_susceptibility_classify(["quartz"])
        assert result[0]["class"] == "diamagnetic"

    def test_hematite_paramagnetic(self):
        """Hematite → paramagnetic."""
        result = magnetic_susceptibility_classify(["hematite"])
        assert result[0]["class"] == "paramagnetic"

    def test_unknown_mineral(self):
        """Unknown mineral → unknown."""
        result = magnetic_susceptibility_classify(["unobtainium"])
        assert result[0]["class"] == "unknown"

    def test_multiple_minerals(self):
        """Multiple minerals → list of classifications."""
        result = magnetic_susceptibility_classify(
            ["magnetite", "quartz", "hematite"]
        )
        assert len(result) == 3


class TestDavisTubeRecovery:
    """Tests for Davis tube test recovery."""

    def test_weight_recovery(self):
        """30g magnetic out of 100g → 30% recovery."""
        result = davis_tube_recovery(100, 30, 20, 55)
        assert result["weight_recovery"] == pytest.approx(0.3)

    def test_grade_recovery(self):
        """Grade recovery calculation."""
        result = davis_tube_recovery(100, 30, 20, 55)
        expected = (30 * 55) / (100 * 20)
        assert result["grade_recovery"] == pytest.approx(expected, rel=0.01)

    def test_upgrade_ratio(self):
        """Upgrade ratio = mag_grade / feed_grade."""
        result = davis_tube_recovery(100, 30, 20, 55)
        assert result["upgrade_ratio"] == pytest.approx(55 / 20, rel=0.01)
