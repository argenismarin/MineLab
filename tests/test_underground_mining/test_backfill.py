"""Tests for minelab.underground_mining.backfill."""

import numpy as np
import pytest

from minelab.underground_mining.backfill import (
    arching_stress,
    backfill_requirement,
    cemented_paste_strength,
    fill_pour_rate,
    hydraulic_fill_transport,
)


class TestCementedPasteStrength:
    """Tests for Belem & Benzaazoua (2008) CPB strength."""

    def test_known_value(self):
        """5% cement, 28 days, w/c=7 -> UCS=100 kPa."""
        result = cemented_paste_strength(0.05, 28.0, 7.0)
        expected = 14.0 * 0.05 * np.sqrt(28.0 / 28.0) / 7.0
        assert result == pytest.approx(expected, rel=1e-4)
        assert result == pytest.approx(0.1, rel=1e-4)  # 0.1 kPa

    def test_more_cement_stronger(self):
        """Higher cement content -> higher strength."""
        low = cemented_paste_strength(0.03, 28.0, 7.0)
        high = cemented_paste_strength(0.08, 28.0, 7.0)
        assert high > low

    def test_longer_cure_stronger(self):
        """Longer cure time -> higher strength."""
        early = cemented_paste_strength(0.05, 7.0, 7.0)
        late = cemented_paste_strength(0.05, 56.0, 7.0)
        assert late > early

    def test_higher_wc_weaker(self):
        """Higher water/cement ratio -> lower strength."""
        low_wc = cemented_paste_strength(0.05, 28.0, 5.0)
        high_wc = cemented_paste_strength(0.05, 28.0, 10.0)
        assert low_wc > high_wc

    def test_invalid_cement_content(self):
        """Zero cement should raise."""
        with pytest.raises(ValueError, match="cement_content"):
            cemented_paste_strength(0.0, 28.0, 7.0)

    def test_invalid_cure_days(self):
        """Negative cure days should raise."""
        with pytest.raises(ValueError, match="cure_days"):
            cemented_paste_strength(0.05, -1.0, 7.0)


class TestArchingStress:
    """Tests for Marston (1930) arching stress."""

    def test_arching_reduces_stress(self):
        """Arching ratio should be < 1."""
        result = arching_stress(30.0, 6.0, 10.0, 35.0, 2000.0)
        assert result["arching_ratio"] < 1.0
        assert result["vertical_stress_kpa"] > 0

    def test_shallow_fill_full_overburden(self):
        """Very shallow fill -> arching ratio close to 1."""
        result = arching_stress(1.0, 6.0, 10.0, 35.0, 2000.0)
        assert result["arching_ratio"] > 0.9

    def test_tall_fill_more_arching(self):
        """Taller fill -> more arching (lower ratio)."""
        shallow = arching_stress(5.0, 6.0, 10.0, 35.0, 2000.0)
        tall = arching_stress(50.0, 6.0, 10.0, 35.0, 2000.0)
        assert tall["arching_ratio"] < shallow["arching_ratio"]

    def test_k_ratio(self):
        """K = (1-sin(phi))/(1+sin(phi))."""
        result = arching_stress(30.0, 6.0, 10.0, 30.0, 2000.0)
        phi_rad = np.radians(30.0)
        expected_k = (1.0 - np.sin(phi_rad)) / (1.0 + np.sin(phi_rad))
        assert result["K_ratio"] == pytest.approx(expected_k, rel=1e-4)

    def test_invalid_friction_angle(self):
        """Friction angle = 0 should raise."""
        with pytest.raises(ValueError, match="friction_angle"):
            arching_stress(30.0, 6.0, 10.0, 0.0, 2000.0)

    def test_invalid_density(self):
        """Zero density should raise."""
        with pytest.raises(ValueError, match="density"):
            arching_stress(30.0, 6.0, 10.0, 35.0, 0.0)


class TestHydraulicFillTransport:
    """Tests for Durand (1953) pipeline transport."""

    def test_above_critical(self):
        """High velocity -> above critical."""
        result = hydraulic_fill_transport(2.5, 0.15, 1600.0)
        assert result["is_above_critical"] is True

    def test_below_critical(self):
        """Very low velocity -> below critical."""
        result = hydraulic_fill_transport(0.1, 0.15, 1600.0)
        assert result["is_above_critical"] is False

    def test_critical_velocity_formula(self):
        """Vc = 1.8 * sqrt(2*g*D*(S-1))."""
        d = 0.15
        rho = 1600.0
        s_ratio = rho / 1000.0 - 1.0
        expected_vc = 1.8 * np.sqrt(2.0 * 9.81 * d * s_ratio)
        result = hydraulic_fill_transport(2.0, d, rho)
        assert result["critical_velocity_ms"] == pytest.approx(expected_vc, rel=1e-4)

    def test_head_loss_positive(self):
        """Head loss should be positive."""
        result = hydraulic_fill_transport(2.0, 0.15, 1600.0)
        assert result["head_loss_kpa_per_m"] > 0

    def test_invalid_velocity(self):
        """Zero velocity should raise."""
        with pytest.raises(ValueError, match="flow_velocity"):
            hydraulic_fill_transport(0.0, 0.15, 1600.0)


class TestFillPourRate:
    """Tests for fill pour scheduling."""

    def test_basic(self):
        """5000 m3 / 200 m3/day = 25 days pour."""
        result = fill_pour_rate(5000.0, 200.0, 14.0)
        assert result["pour_days"] == pytest.approx(25.0, rel=1e-4)
        assert result["total_days"] == pytest.approx(39.0, rel=1e-4)

    def test_effective_rate(self):
        """effective_rate = volume / total_days."""
        result = fill_pour_rate(5000.0, 200.0, 14.0)
        expected = 5000.0 / 39.0
        assert result["effective_fill_rate_m3_per_day"] == pytest.approx(expected, rel=1e-4)

    def test_faster_pour_shorter_time(self):
        """Faster pour rate -> fewer pour days."""
        slow = fill_pour_rate(5000.0, 100.0, 14.0)
        fast = fill_pour_rate(5000.0, 500.0, 14.0)
        assert fast["pour_days"] < slow["pour_days"]

    def test_invalid_volume(self):
        """Zero volume should raise."""
        with pytest.raises(ValueError, match="stope_volume"):
            fill_pour_rate(0, 200.0, 14.0)


class TestBackfillRequirement:
    """Tests for backfill volume and mass calculation."""

    def test_basic(self):
        """10000 m3, 95% fill, 1.8 t/m3."""
        result = backfill_requirement(10000.0, 0.95, 1.8)
        assert result["fill_volume_m3"] == pytest.approx(9500.0, rel=1e-4)
        assert result["fill_mass_tonnes"] == pytest.approx(17100.0, rel=1e-4)

    def test_full_fill(self):
        """100% fill ratio."""
        result = backfill_requirement(5000.0, 1.0, 2.0)
        assert result["fill_volume_m3"] == pytest.approx(5000.0, rel=1e-4)

    def test_higher_density_more_mass(self):
        """Higher fill density -> more mass."""
        light = backfill_requirement(5000.0, 1.0, 1.5)
        heavy = backfill_requirement(5000.0, 1.0, 2.5)
        assert heavy["fill_mass_tonnes"] > light["fill_mass_tonnes"]

    def test_invalid_filling_ratio(self):
        """Filling ratio > 1 should raise."""
        with pytest.raises(ValueError, match="void_filling_ratio"):
            backfill_requirement(10000.0, 1.5, 1.8)

    def test_invalid_density(self):
        """Zero density should raise."""
        with pytest.raises(ValueError, match="fill_density"):
            backfill_requirement(10000.0, 0.95, 0.0)
