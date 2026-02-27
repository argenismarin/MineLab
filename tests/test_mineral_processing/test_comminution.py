"""Tests for minelab.mineral_processing.comminution."""

import numpy as np
import pytest

from minelab.mineral_processing.comminution import (
    ball_mill_power,
    bond_energy,
    bond_work_index,
    crusher_reduction_ratio,
    kick_energy,
    rittinger_energy,
    rod_mill_power,
    sag_mill_power,
)


class TestBondWorkIndex:
    """Tests for Bond lab work index calculation."""

    def test_known_value(self):
        """Typical ore: Wi ≈ 13-14 kWh/t."""
        wi = bond_work_index(106, 2500, 75, 1.5)
        assert 10 < wi < 20

    def test_positive(self):
        """Wi must be positive."""
        wi = bond_work_index(150, 3000, 100, 2.0)
        assert wi > 0

    def test_invalid_grindability(self):
        """Negative grindability should raise."""
        with pytest.raises(ValueError):
            bond_work_index(106, 2500, 75, -1)


class TestBondEnergy:
    """Tests for Bond's 3rd law of comminution."""

    def test_known_value(self):
        """Wi=12, F80=2500, P80=75 → W ≈ 11.46 kWh/t."""
        w = bond_energy(12, 2500, 75)
        # W = 10*12*(1/sqrt(75) - 1/sqrt(2500)) = 11.457
        expected = 10 * 12 * (1 / np.sqrt(75) - 1 / np.sqrt(2500))
        assert w == pytest.approx(expected, rel=0.01)

    def test_finer_product_more_energy(self):
        """Finer product requires more energy."""
        w_coarse = bond_energy(12, 2500, 150)
        w_fine = bond_energy(12, 2500, 75)
        assert w_fine > w_coarse

    def test_zero_reduction(self):
        """Same F80 and P80 → near zero energy."""
        w = bond_energy(12, 1000, 1000)
        assert abs(w) < 0.01

    def test_invalid_wi(self):
        """Negative Wi should raise."""
        with pytest.raises(ValueError):
            bond_energy(-12, 2500, 75)


class TestKickEnergy:
    """Tests for Kick's law (coarse crushing)."""

    def test_known_value(self):
        """Ki=1, F/P=10 → W ≈ 2.30."""
        w = kick_energy(1.0, 1000, 100)
        assert w == pytest.approx(np.log(10), rel=0.01)

    def test_no_reduction(self):
        """Same size → zero energy."""
        w = kick_energy(1.0, 100, 100)
        assert abs(w) < 1e-10


class TestRittingerEnergy:
    """Tests for Rittinger's law (fine grinding)."""

    def test_known_value(self):
        """Kr=1, F=1000, P=100 → W = 0.009."""
        w = rittinger_energy(1.0, 1000, 100)
        expected = 1 / 100 - 1 / 1000
        assert w == pytest.approx(expected, rel=0.01)


class TestBallMillPower:
    """Tests for ball mill power draw."""

    def test_known_value(self):
        """Wi=12, 100 t/h, 0.9 eff."""
        power = ball_mill_power(12, 2500, 75, 100, 0.9)
        expected = bond_energy(12, 2500, 75) * 100 / 0.9
        assert power == pytest.approx(expected, rel=0.01)

    def test_higher_tonnage(self):
        """More tonnage → more power."""
        p100 = ball_mill_power(12, 2500, 75, 100)
        p200 = ball_mill_power(12, 2500, 75, 200)
        assert p200 > p100


class TestSAGMillPower:
    """Tests for SAG mill power."""

    def test_known_value(self):
        """SPI=10, 200 t/h → 2000 kW."""
        power = sag_mill_power(10, 150000, 2000, 200)
        assert power == pytest.approx(2000.0, rel=0.01)


class TestRodMillPower:
    """Tests for rod mill power draw."""

    def test_positive_power(self):
        """Should return positive power."""
        power = rod_mill_power(12, 15000, 1000, 100)
        assert power > 0

    def test_correction_factor(self):
        """Correction factor > 1 increases power."""
        base = rod_mill_power(12, 15000, 1000, 100, 1.0)
        corrected = rod_mill_power(12, 15000, 1000, 100, 1.1)
        assert corrected > base


class TestCrusherReductionRatio:
    """Tests for crusher reduction ratio."""

    def test_known_value(self):
        """F80=500, P80=100 → RR=5."""
        result = crusher_reduction_ratio(500, 100)
        assert result["reduction_ratio"] == 5.0

    def test_jaw_crusher(self):
        """RR 5-7 → Jaw crusher."""
        result = crusher_reduction_ratio(600, 100)
        assert "Jaw" in result["crusher_type"]

    def test_cone_crusher(self):
        """RR 3-5 → Cone crusher."""
        result = crusher_reduction_ratio(400, 100)
        assert "Cone" in result["crusher_type"]
