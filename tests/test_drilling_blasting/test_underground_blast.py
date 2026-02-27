"""Tests for minelab.drilling_blasting.underground_blast."""

import math

import pytest

from minelab.drilling_blasting.underground_blast import (
    burn_cut_advance,
    controlled_blasting_ppv,
    cut_hole_design,
    delay_timing_design,
    presplit_parameters,
    tunnel_blast_powder_factor,
    underground_blast_vibration_limit,
)


# ---------------------------------------------------------------------------
# cut_hole_design
# ---------------------------------------------------------------------------


class TestCutHoleDesign:
    """Tests for cut_hole_design."""

    def test_known_value(self):
        """45 mm hole, 102 mm relief, 0.5 m burden."""
        result = cut_hole_design(45, 102, 0.5)
        expected_spacing = 0.5 * math.sqrt(2)
        assert result["spacing_m"] == pytest.approx(
            expected_spacing, rel=1e-4
        )
        assert result["pattern_type"] == "four_hole_box_cut"
        assert result["relief_hole_diameter_mm"] == pytest.approx(
            102.0, rel=1e-4
        )

    def test_charge_calculation(self):
        """Verify charge_per_hole using pi/4 * d^2 * B * rho."""
        result = cut_hole_design(45, 102, 0.5)
        d_m = 45 / 1000.0
        expected_charge = (math.pi / 4) * d_m**2 * 0.5 * 1200
        assert result["charge_per_hole_kg"] == pytest.approx(
            expected_charge, rel=1e-4
        )

    def test_larger_burden_larger_spacing(self):
        """Larger burden -> larger spacing."""
        small = cut_hole_design(45, 102, 0.3)
        large = cut_hole_design(45, 102, 0.8)
        assert large["spacing_m"] > small["spacing_m"]

    def test_relief_must_be_larger(self):
        """Relief hole diameter must exceed charged hole diameter."""
        with pytest.raises(ValueError, match="uncharged_hole_diameter"):
            cut_hole_design(102, 45, 0.5)

    def test_equal_diameters_raises(self):
        """Equal diameters should raise ValueError."""
        with pytest.raises(ValueError, match="uncharged_hole_diameter"):
            cut_hole_design(76, 76, 0.5)

    def test_invalid_burden(self):
        """Zero burden raises ValueError."""
        with pytest.raises(ValueError, match="burden"):
            cut_hole_design(45, 102, 0)


# ---------------------------------------------------------------------------
# burn_cut_advance
# ---------------------------------------------------------------------------


class TestBurnCutAdvance:
    """Tests for burn_cut_advance."""

    def test_known_value(self):
        """3.0 m * 0.90 * 0.95 = 2.565 m."""
        result = burn_cut_advance(3.0, 0.90, 0.95)
        assert result == pytest.approx(2.565, rel=1e-4)

    def test_maximum_advance(self):
        """Maximum parameters: 4.0 * 0.95 * 1.0 = 3.8 m."""
        result = burn_cut_advance(4.0, 0.95, 1.0)
        assert result == pytest.approx(3.8, rel=1e-4)

    def test_minimum_ratios(self):
        """Minimum ratios: 3.0 * 0.85 * 0.8 = 2.04 m."""
        result = burn_cut_advance(3.0, 0.85, 0.8)
        assert result == pytest.approx(2.04, rel=1e-4)

    def test_advance_less_than_drill_length(self):
        """Advance should always be <= drill length."""
        result = burn_cut_advance(3.5, 0.90, 0.90)
        assert result <= 3.5

    def test_invalid_drill_length(self):
        """Zero drill length raises ValueError."""
        with pytest.raises(ValueError, match="drill_length"):
            burn_cut_advance(0, 0.90, 0.95)

    def test_charge_ratio_out_of_range(self):
        """Charge ratio outside [0.85, 0.95] raises ValueError."""
        with pytest.raises(ValueError, match="charge_ratio"):
            burn_cut_advance(3.0, 0.50, 0.95)

    def test_rock_factor_out_of_range(self):
        """Rock factor outside [0.8, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="rock_factor"):
            burn_cut_advance(3.0, 0.90, 0.5)


# ---------------------------------------------------------------------------
# tunnel_blast_powder_factor
# ---------------------------------------------------------------------------


class TestTunnelBlastPowderFactor:
    """Tests for tunnel_blast_powder_factor."""

    def test_known_value(self):
        """50 kg / 25 m3 = 2.0 kg/m3."""
        result = tunnel_blast_powder_factor(50, 25)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_low_powder_factor(self):
        """10 kg / 100 m3 = 0.1 kg/m3."""
        result = tunnel_blast_powder_factor(10, 100)
        assert result == pytest.approx(0.1, rel=1e-4)

    def test_higher_charge_higher_pf(self):
        """More charge -> higher powder factor."""
        low = tunnel_blast_powder_factor(20, 50)
        high = tunnel_blast_powder_factor(40, 50)
        assert high > low

    def test_invalid_charge(self):
        """Zero charge raises ValueError."""
        with pytest.raises(ValueError, match="charge_per_blast_kg"):
            tunnel_blast_powder_factor(0, 25)

    def test_invalid_volume(self):
        """Negative volume raises ValueError."""
        with pytest.raises(ValueError, match="volume_blasted_m3"):
            tunnel_blast_powder_factor(50, -10)


# ---------------------------------------------------------------------------
# controlled_blasting_ppv
# ---------------------------------------------------------------------------


class TestControlledBlastingPPV:
    """Tests for controlled_blasting_ppv."""

    def test_known_value(self):
        """W=10, D=50, K=700, alpha=1.5."""
        sd = 50 / math.sqrt(10)
        expected = 700 * sd ** (-1.5)
        result = controlled_blasting_ppv(10, 50, 700, 1.5)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_closer_distance_higher_ppv(self):
        """Closer distance -> higher PPV."""
        far = controlled_blasting_ppv(10, 100, 700, 1.5)
        near = controlled_blasting_ppv(10, 20, 700, 1.5)
        assert near > far

    def test_more_charge_higher_ppv(self):
        """More charge per delay -> higher PPV."""
        light = controlled_blasting_ppv(5, 50, 700, 1.5)
        heavy = controlled_blasting_ppv(50, 50, 700, 1.5)
        assert heavy > light

    def test_invalid_charge(self):
        """Zero charge raises ValueError."""
        with pytest.raises(ValueError, match="charge_per_delay_kg"):
            controlled_blasting_ppv(0, 50, 700, 1.5)

    def test_invalid_distance(self):
        """Negative distance raises ValueError."""
        with pytest.raises(ValueError, match="distance_m"):
            controlled_blasting_ppv(10, -5, 700, 1.5)

    def test_invalid_k(self):
        """Zero k raises ValueError."""
        with pytest.raises(ValueError, match="k"):
            controlled_blasting_ppv(10, 50, 0, 1.5)


# ---------------------------------------------------------------------------
# presplit_parameters
# ---------------------------------------------------------------------------


class TestPresplitParameters:
    """Tests for presplit_parameters."""

    def test_known_value(self):
        """76 mm hole, 10 MPa tensile, 0.8 m spacing."""
        result = presplit_parameters(76, 10, 0.8)
        # decoupling_ratio = 0.5
        # charge_diam = 76/1000 * 0.5 = 0.038 m
        # linear_charge = pi/4 * 0.038^2 * 1200 = 1.36088e-3 * 1200
        expected_charge = (
            (math.pi / 4) * (0.076 * 0.5) ** 2 * 1200
        )
        assert result["linear_charge_kg_per_m"] == pytest.approx(
            expected_charge, rel=1e-4
        )

    def test_recommended_spacing(self):
        """Recommended spacing = 11 * d (mm -> m)."""
        result = presplit_parameters(76, 10, 0.8)
        expected = 11.0 * 76 / 1000.0  # 0.836 m
        assert result["recommended_spacing_m"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_spacing_to_diameter_ratio(self):
        """S/d = hole_spacing(m)*1000 / hole_diameter(mm)."""
        result = presplit_parameters(76, 10, 0.8)
        expected_ratio = 0.8 * 1000 / 76
        assert result["spacing_to_diameter_ratio"] == pytest.approx(
            expected_ratio, rel=1e-4
        )

    def test_larger_hole_larger_charge(self):
        """Larger hole diameter -> higher linear charge."""
        small = presplit_parameters(45, 10, 0.5)
        large = presplit_parameters(89, 10, 0.5)
        assert (
            large["linear_charge_kg_per_m"]
            > small["linear_charge_kg_per_m"]
        )

    def test_invalid_diameter(self):
        """Zero hole diameter raises ValueError."""
        with pytest.raises(ValueError, match="hole_diameter"):
            presplit_parameters(0, 10, 0.8)

    def test_invalid_tensile(self):
        """Negative tensile strength raises ValueError."""
        with pytest.raises(
            ValueError, match="rock_tensile_strength"
        ):
            presplit_parameters(76, -5, 0.8)


# ---------------------------------------------------------------------------
# delay_timing_design
# ---------------------------------------------------------------------------


class TestDelayTimingDesign:
    """Tests for delay_timing_design."""

    def test_sequential_default(self):
        """4 holes at 25 ms each, sequential."""
        result = delay_timing_design(4, 25)
        assert result["total_blast_time_ms"] == pytest.approx(
            75.0, rel=1e-4
        )
        assert result["number_of_delays"] == 4
        assert len(result["timing_schedule"]) == 4

    def test_grouped_sequence(self):
        """8 holes in groups of [2,2,2,2] at 50 ms."""
        result = delay_timing_design(8, 50, [2, 2, 2, 2])
        assert result["total_blast_time_ms"] == pytest.approx(
            150.0, rel=1e-4
        )
        assert result["number_of_delays"] == 4

    def test_timing_schedule_values(self):
        """Check individual timing entries."""
        result = delay_timing_design(3, 100)
        schedule = result["timing_schedule"]
        assert schedule[0]["delay_number"] == 1
        assert schedule[0]["time_ms"] == pytest.approx(0.0)
        assert schedule[1]["time_ms"] == pytest.approx(100.0)
        assert schedule[2]["time_ms"] == pytest.approx(200.0)

    def test_single_hole(self):
        """Single hole: total time = 0."""
        result = delay_timing_design(1, 50)
        assert result["total_blast_time_ms"] == pytest.approx(0.0)
        assert result["number_of_delays"] == 1

    def test_sequence_mismatch_raises(self):
        """Sequence sum != number_of_holes raises ValueError."""
        with pytest.raises(
            ValueError, match="detonation_sequence"
        ):
            delay_timing_design(8, 50, [2, 2, 2])

    def test_invalid_number_of_holes(self):
        """Zero holes raises ValueError."""
        with pytest.raises(ValueError, match="number_of_holes"):
            delay_timing_design(0, 25)

    def test_invalid_ms(self):
        """Zero ms_per_hole raises ValueError."""
        with pytest.raises(ValueError, match="ms_per_hole"):
            delay_timing_design(4, 0)


# ---------------------------------------------------------------------------
# underground_blast_vibration_limit
# ---------------------------------------------------------------------------


class TestUndergroundBlastVibrationLimit:
    """Tests for underground_blast_vibration_limit."""

    def test_residential_100m(self):
        """Residential at 100 m: 5 * sqrt(1 + 100/100) = 7.071."""
        result = underground_blast_vibration_limit(100, "residential")
        expected = 5.0 * math.sqrt(2.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_commercial_50m(self):
        """Commercial at 50 m: 20 * sqrt(1.5) = 24.495."""
        result = underground_blast_vibration_limit(50, "commercial")
        expected = 20.0 * math.sqrt(1.5)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sensitive_200m(self):
        """Sensitive at 200 m: 3 * sqrt(3.0) = 5.196."""
        result = underground_blast_vibration_limit(200, "sensitive")
        expected = 3.0 * math.sqrt(3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_deeper_allows_more_vibration(self):
        """Deeper blast -> higher allowed PPV (more attenuation)."""
        shallow = underground_blast_vibration_limit(
            50, "residential"
        )
        deep = underground_blast_vibration_limit(
            300, "residential"
        )
        assert deep > shallow

    def test_unknown_structure_raises(self):
        """Unknown structure type raises ValueError."""
        with pytest.raises(ValueError, match="structure_type"):
            underground_blast_vibration_limit(100, "church")

    def test_invalid_depth(self):
        """Zero depth raises ValueError."""
        with pytest.raises(ValueError, match="depth_below_surface"):
            underground_blast_vibration_limit(0, "residential")
