"""Tests for minelab.underground_mining.room_and_pillar."""

import numpy as np
import pytest

from minelab.underground_mining.room_and_pillar import (
    barrier_pillar_width,
    critical_span,
    pillar_safety_factor,
    room_and_pillar_geometry,
    subsidence_angle,
)


class TestPillarSafetyFactor:
    """Tests for pillar safety factor."""

    def test_basic(self):
        """SF = 50/25 = 2.0."""
        assert pillar_safety_factor(50.0, 25.0) == pytest.approx(2.0, rel=1e-4)

    def test_marginal(self):
        """SF = 30/25 = 1.2."""
        assert pillar_safety_factor(30.0, 25.0) == pytest.approx(1.2, rel=1e-4)

    def test_higher_strength_higher_sf(self):
        """Higher strength -> higher SF."""
        sf_low = pillar_safety_factor(40.0, 20.0)
        sf_high = pillar_safety_factor(60.0, 20.0)
        assert sf_high > sf_low

    def test_invalid_strength(self):
        """Zero strength should raise."""
        with pytest.raises(ValueError, match="pillar_strength"):
            pillar_safety_factor(0.0, 25.0)

    def test_invalid_stress(self):
        """Negative stress should raise."""
        with pytest.raises(ValueError, match="pillar_stress"):
            pillar_safety_factor(50.0, -5.0)


class TestRoomAndPillarGeometry:
    """Tests for extraction ratio and pillar geometry."""

    def test_equal_width(self):
        """room=pillar=6m -> e = 1 - (6/12)^2 = 0.75."""
        result = room_and_pillar_geometry(6.0, 6.0, 3.0)
        assert result["extraction_ratio"] == pytest.approx(0.75, rel=1e-4)

    def test_pillar_area(self):
        """pillar_area = Wp^2."""
        result = room_and_pillar_geometry(6.0, 8.0, 4.0)
        assert result["pillar_area_m2"] == pytest.approx(64.0, rel=1e-4)

    def test_w_over_h(self):
        """w/h = Wp / seam_height."""
        result = room_and_pillar_geometry(6.0, 8.0, 4.0)
        assert result["w_over_h"] == pytest.approx(2.0, rel=1e-4)

    def test_wider_rooms_higher_extraction(self):
        """Wider rooms -> higher extraction ratio."""
        narrow = room_and_pillar_geometry(4.0, 8.0, 3.0)
        wide = room_and_pillar_geometry(8.0, 8.0, 3.0)
        assert wide["extraction_ratio"] > narrow["extraction_ratio"]

    def test_invalid_room_width(self):
        """Zero room width should raise."""
        with pytest.raises(ValueError, match="room_width"):
            room_and_pillar_geometry(0, 6.0, 3.0)


class TestBarrierPillarWidth:
    """Tests for Obert & Duvall (1967) barrier pillar width."""

    def test_known_value(self):
        """span=50, depth=200, ucs=60, SF=2."""
        gamma_g = 2700.0 * 9.81 / 1e6
        expected = np.sqrt(200.0 * 50.0 * gamma_g * 2.0 / 60.0)
        result = barrier_pillar_width(50.0, 200.0, 60.0, 2.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_deeper_wider_pillar(self):
        """Deeper mining -> wider barrier pillar."""
        shallow = barrier_pillar_width(50.0, 100.0, 60.0, 2.0)
        deep = barrier_pillar_width(50.0, 500.0, 60.0, 2.0)
        assert deep > shallow

    def test_stronger_rock_narrower_pillar(self):
        """Higher UCS -> narrower barrier pillar."""
        weak = barrier_pillar_width(50.0, 200.0, 30.0, 2.0)
        strong = barrier_pillar_width(50.0, 200.0, 100.0, 2.0)
        assert strong < weak

    def test_higher_sf_wider_pillar(self):
        """Higher safety factor -> wider pillar."""
        sf_low = barrier_pillar_width(50.0, 200.0, 60.0, 1.5)
        sf_high = barrier_pillar_width(50.0, 200.0, 60.0, 3.0)
        assert sf_high > sf_low

    def test_invalid_span(self):
        """Zero span should raise."""
        with pytest.raises(ValueError, match="span"):
            barrier_pillar_width(0, 200.0, 60.0, 2.0)


class TestCriticalSpan:
    """Tests for Lang (1994) critical span."""

    def test_good_rock(self):
        """High RMR -> large critical span."""
        result = critical_span(80.0, 100.0, 1.0)
        assert result["critical_span_m"] > 10.0

    def test_poor_rock(self):
        """Low RMR -> smaller critical span than good rock."""
        poor = critical_span(30.0, 100.0, 1.0)
        good = critical_span(70.0, 100.0, 1.0)
        assert poor["critical_span_m"] < good["critical_span_m"]

    def test_depth_factor(self):
        """Greater depth -> lower depth factor."""
        shallow = critical_span(60.0, 100.0, 1.5)
        deep = critical_span(60.0, 500.0, 1.5)
        assert deep["depth_factor"] < shallow["depth_factor"]

    def test_stability_classes(self):
        """Different RMR values produce different classes."""
        high = critical_span(90.0, 100.0, 1.0)
        assert high["stability_class"] == "stable"
        low = critical_span(20.0, 200.0, 2.0)
        assert low["stability_class"] in ("marginal", "unstable")

    def test_invalid_rmr(self):
        """RMR > 100 should raise."""
        with pytest.raises(ValueError, match="rmr"):
            critical_span(110.0, 100.0, 1.0)

    def test_invalid_k_ratio(self):
        """k_ratio <= 0 should raise."""
        with pytest.raises(ValueError, match="k_ratio"):
            critical_span(60.0, 100.0, 0.0)


class TestSubsidenceAngle:
    """Tests for Kratzsch (1983) subsidence."""

    def test_flat_seam(self):
        """Flat seam (dip=0) -> angle_of_draw = 35."""
        result = subsidence_angle(100.0, 3.0, 0.0)
        assert result["angle_of_draw_deg"] == pytest.approx(35.0, rel=1e-4)

    def test_max_subsidence(self):
        """Max subsidence = 0.9 * seam_thickness."""
        result = subsidence_angle(100.0, 3.0, 0.0)
        assert result["max_subsidence_m"] == pytest.approx(2.7, rel=1e-4)

    def test_trough_width(self):
        """Trough width = 2 * depth * tan(draw_angle)."""
        result = subsidence_angle(100.0, 3.0, 0.0)
        expected = 2.0 * 100.0 * np.tan(np.radians(35.0))
        assert result["trough_width_m"] == pytest.approx(expected, rel=1e-4)

    def test_steeper_dip_wider_draw(self):
        """Steeper dip -> wider angle of draw."""
        flat = subsidence_angle(100.0, 3.0, 0.0)
        steep = subsidence_angle(100.0, 3.0, 30.0)
        assert steep["angle_of_draw_deg"] > flat["angle_of_draw_deg"]

    def test_invalid_overburden(self):
        """Zero overburden should raise."""
        with pytest.raises(ValueError, match="overburden_depth"):
            subsidence_angle(0, 3.0, 0.0)
