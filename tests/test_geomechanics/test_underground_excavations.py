"""Tests for minelab.geomechanics.underground_excavations."""

import numpy as np
import pytest

from minelab.geomechanics.underground_excavations import (
    cable_bolt_capacity,
    failure_criterion_mohr_coulomb_ug,
    in_situ_stress_depth,
    kirsch_elastic_stress,
    plastic_zone_radius,
    shotcrete_lining_capacity,
    strength_to_stress_ratio,
    tunnel_support_pressure,
)


class TestInSituStressDepth:
    """Tests for Brady & Brown stress from depth."""

    def test_known_value(self):
        """1000m, 2700 kg/m3 -> sigma_v = 26.487 MPa."""
        result = in_situ_stress_depth(1000.0, 2700.0, 1.5)
        expected_sv = 2700.0 * 9.81 * 1000.0 / 1e6
        assert result["sigma_v_mpa"] == pytest.approx(expected_sv, rel=1e-4)
        assert result["sigma_h_mpa"] == pytest.approx(1.5 * expected_sv, rel=1e-4)

    def test_mean_stress(self):
        """Mean = (sigma_v + 2*sigma_h) / 3."""
        result = in_situ_stress_depth(500.0, 2700.0, 1.0)
        expected_mean = (result["sigma_v_mpa"] + 2.0 * result["sigma_h_mpa"]) / 3.0
        assert result["sigma_mean_mpa"] == pytest.approx(expected_mean, rel=1e-4)

    def test_deeper_higher_stress(self):
        """Greater depth -> higher stress."""
        shallow = in_situ_stress_depth(200.0, 2700.0, 1.0)
        deep = in_situ_stress_depth(1000.0, 2700.0, 1.0)
        assert deep["sigma_v_mpa"] > shallow["sigma_v_mpa"]

    def test_k_ratio_effect(self):
        """k=2 -> sigma_h = 2 * sigma_v."""
        result = in_situ_stress_depth(500.0, 2700.0, 2.0)
        assert result["sigma_h_mpa"] == pytest.approx(2.0 * result["sigma_v_mpa"], rel=1e-4)

    def test_invalid_depth(self):
        """Zero depth should raise."""
        with pytest.raises(ValueError, match="depth"):
            in_situ_stress_depth(0, 2700.0, 1.0)


class TestKirschElasticStress:
    """Tests for Kirsch (1898) solution."""

    def test_crown_stress(self):
        """At crown (theta=90, r=a): sigma_tt = 3*sh - sv."""
        result = kirsch_elastic_stress(10.0, 5.0, 3.0, 3.0, 90.0)
        # At wall: P*(1+1) - Q*(1+3)*cos(180)
        p = (10.0 + 5.0) / 2.0
        q = (10.0 - 5.0) / 2.0
        expected = p * 2.0 - q * 4.0 * np.cos(np.radians(180.0))
        assert result["sigma_tangential"] == pytest.approx(expected, rel=1e-4)

    def test_sidewall_stress(self):
        """At sidewall (theta=0, r=a): sigma_tt = 3*sh - sv."""
        result = kirsch_elastic_stress(10.0, 5.0, 3.0, 3.0, 0.0)
        # sigma_tt = P*(1+1) - Q*(1+3)*cos(0) = 2P - 4Q = 3*sh - sv
        expected = 3.0 * 5.0 - 10.0
        assert result["sigma_tangential"] == pytest.approx(expected, rel=1e-4)

    def test_radial_stress_at_wall(self):
        """At the wall (r=a), radial stress = 0."""
        result = kirsch_elastic_stress(10.0, 10.0, 3.0, 3.0, 45.0)
        assert result["sigma_radial"] == pytest.approx(0.0, abs=1e-6)

    def test_far_field_stress(self):
        """Far from tunnel, stresses approach far-field."""
        result = kirsch_elastic_stress(10.0, 10.0, 3.0, 100.0, 0.0)
        # At r >> a, sigma_rr -> P + Q*cos(2theta) = sigma_v
        assert result["sigma_radial"] == pytest.approx(10.0, rel=0.01)

    def test_r_less_than_tunnel(self):
        """r < r_tunnel should raise."""
        with pytest.raises(ValueError, match="r"):
            kirsch_elastic_stress(10.0, 5.0, 3.0, 2.0, 0.0)


class TestPlasticZoneRadius:
    """Tests for Hoek & Brown plastic zone."""

    def test_basic(self):
        """Should return r_p > r_tunnel for unsupported case."""
        r_p = plastic_zone_radius(20.0, 10.0, 2.0, 35.0, 3.0)
        assert r_p > 3.0

    def test_higher_stress_larger_zone(self):
        """Higher in-situ stress -> larger plastic zone."""
        rp_low = plastic_zone_radius(10.0, 10.0, 2.0, 35.0, 3.0)
        rp_high = plastic_zone_radius(30.0, 10.0, 2.0, 35.0, 3.0)
        assert rp_high > rp_low

    def test_stronger_rock_smaller_zone(self):
        """Higher cohesion -> smaller plastic zone."""
        rp_weak = plastic_zone_radius(20.0, 10.0, 1.0, 35.0, 3.0)
        rp_strong = plastic_zone_radius(20.0, 10.0, 5.0, 35.0, 3.0)
        assert rp_strong < rp_weak

    def test_invalid_phi(self):
        """phi = 0 should raise."""
        with pytest.raises(ValueError, match="phi_deg"):
            plastic_zone_radius(20.0, 10.0, 2.0, 0.0, 3.0)

    def test_invalid_cohesion(self):
        """Cohesion <= 0 should raise."""
        with pytest.raises(ValueError, match="c"):
            plastic_zone_radius(20.0, 10.0, 0.0, 35.0, 3.0)


class TestStrengthToStressRatio:
    """Tests for Hoek (2007) SSR."""

    def test_stable(self):
        """UCS/sigma_1 > 2 -> stable."""
        result = strength_to_stress_ratio(100.0, 40.0)
        assert result["ratio"] == pytest.approx(2.5, rel=1e-4)
        assert result["classification"] == "stable"

    def test_minor_problems(self):
        """1 < ratio < 2 -> minor problems."""
        result = strength_to_stress_ratio(100.0, 80.0)
        assert result["classification"] == "minor problems"

    def test_severe_problems(self):
        """0.5 < ratio < 1 -> severe problems."""
        result = strength_to_stress_ratio(50.0, 70.0)
        assert result["classification"] == "severe problems"

    def test_extreme_problems(self):
        """ratio < 0.5 -> extreme."""
        result = strength_to_stress_ratio(30.0, 100.0)
        assert result["classification"] == "extreme problems"

    def test_invalid_sigma(self):
        """Zero sigma_1 should raise."""
        with pytest.raises(ValueError, match="sigma_1"):
            strength_to_stress_ratio(100.0, 0.0)


class TestTunnelSupportPressure:
    """Tests for Barton Q-system support pressure."""

    def test_basic(self):
        """Q=4, span=8, ESR=1 -> P = 5*4^(-1/3)/1."""
        result = tunnel_support_pressure(4.0, 8.0, 1.0)
        expected = 5.0 * 4.0 ** (-1.0 / 3.0) / 1.0
        assert result["support_pressure_kpa"] == pytest.approx(expected, rel=1e-4)

    def test_equivalent_dimension(self):
        """De = span / ESR."""
        result = tunnel_support_pressure(10.0, 12.0, 1.5)
        assert result["equivalent_dimension"] == pytest.approx(8.0, rel=1e-4)

    def test_q_classification_good(self):
        """Q > 10 -> good."""
        result = tunnel_support_pressure(15.0, 8.0, 1.0)
        assert result["Q_class"] == "good"

    def test_q_classification_poor(self):
        """Q = 2 -> poor."""
        result = tunnel_support_pressure(2.0, 8.0, 1.0)
        assert result["Q_class"] == "poor"

    def test_invalid_q(self):
        """Q <= 0 should raise."""
        with pytest.raises(ValueError, match="q_value"):
            tunnel_support_pressure(0.0, 8.0, 1.0)


class TestCableBoltCapacity:
    """Tests for Fuller & Cox (1975) cable bolt capacity."""

    def test_design_capacity_positive(self):
        """Design capacity should be positive."""
        result = cable_bolt_capacity(15.2, 40.0, 3.0)
        assert result["design_capacity_kn"] > 0

    def test_design_is_minimum(self):
        """Design capacity = min(bond, steel)."""
        result = cable_bolt_capacity(15.2, 40.0, 3.0)
        assert result["design_capacity_kn"] == pytest.approx(
            min(
                result["bond_capacity_kn"],
                result["steel_capacity_kn"],
            ),
            rel=1e-4,
        )

    def test_longer_embedment_more_bond(self):
        """Longer embedment -> higher bond capacity."""
        short = cable_bolt_capacity(15.2, 40.0, 1.5)
        long_ = cable_bolt_capacity(15.2, 40.0, 5.0)
        assert long_["bond_capacity_kn"] > short["bond_capacity_kn"]

    def test_stronger_grout_more_bond(self):
        """Stronger grout -> higher bond capacity."""
        weak = cable_bolt_capacity(15.2, 20.0, 3.0)
        strong = cable_bolt_capacity(15.2, 60.0, 3.0)
        assert strong["bond_capacity_kn"] > weak["bond_capacity_kn"]

    def test_invalid_diameter(self):
        """Zero diameter should raise."""
        with pytest.raises(ValueError, match="diameter_mm"):
            cable_bolt_capacity(0.0, 40.0, 3.0)


class TestShotcreteLiningCapacity:
    """Tests for Hoek et al. (1995) shotcrete lining."""

    def test_basic(self):
        """100mm, 30 MPa, 3m radius -> p = 30*0.1/3 = 1.0 MPa."""
        result = shotcrete_lining_capacity(100.0, 30.0, 3.0)
        assert result["max_pressure_mpa"] == pytest.approx(1.0, rel=1e-4)

    def test_thicker_lining_more_pressure(self):
        """Thicker shotcrete -> higher support pressure."""
        thin = shotcrete_lining_capacity(50.0, 30.0, 3.0)
        thick = shotcrete_lining_capacity(150.0, 30.0, 3.0)
        assert thick["max_pressure_mpa"] > thin["max_pressure_mpa"]

    def test_larger_tunnel_less_pressure(self):
        """Larger radius -> lower support pressure."""
        small = shotcrete_lining_capacity(100.0, 30.0, 2.0)
        large = shotcrete_lining_capacity(100.0, 30.0, 5.0)
        assert large["max_pressure_mpa"] < small["max_pressure_mpa"]

    def test_lining_stress_equals_ucs(self):
        """At capacity, lining stress = UCS."""
        result = shotcrete_lining_capacity(100.0, 30.0, 3.0)
        assert result["lining_stress_mpa"] == pytest.approx(30.0, rel=1e-4)

    def test_invalid_thickness(self):
        """Zero thickness should raise."""
        with pytest.raises(ValueError, match="thickness_mm"):
            shotcrete_lining_capacity(0.0, 30.0, 3.0)


class TestFailureCriterionMohrCoulombUG:
    """Tests for Mohr-Coulomb failure criterion."""

    def test_known_value(self):
        """sigma_3=5, c=10, phi=35 -> sigma_1."""
        phi = np.radians(35.0)
        n_phi = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
        expected = 5.0 * n_phi + 2.0 * 10.0 * np.sqrt(n_phi)
        result = failure_criterion_mohr_coulomb_ug(5.0, 10.0, 35.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_unconfined(self):
        """sigma_3=0 -> sigma_1 = 2*c*sqrt(Nphi) = UCS."""
        phi = np.radians(30.0)
        n_phi = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
        expected = 2.0 * 5.0 * np.sqrt(n_phi)
        result = failure_criterion_mohr_coulomb_ug(0.0, 5.0, 30.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_confinement_higher_strength(self):
        """Higher sigma_3 -> higher sigma_1 at failure."""
        low = failure_criterion_mohr_coulomb_ug(0.0, 10.0, 35.0)
        high = failure_criterion_mohr_coulomb_ug(10.0, 10.0, 35.0)
        assert high > low

    def test_higher_friction_higher_strength(self):
        """Higher friction angle -> higher sigma_1."""
        low_phi = failure_criterion_mohr_coulomb_ug(5.0, 10.0, 25.0)
        high_phi = failure_criterion_mohr_coulomb_ug(5.0, 10.0, 45.0)
        assert high_phi > low_phi

    def test_invalid_cohesion(self):
        """Zero cohesion should raise."""
        with pytest.raises(ValueError, match="cohesion"):
            failure_criterion_mohr_coulomb_ug(5.0, 0.0, 35.0)

    def test_invalid_friction_angle(self):
        """phi = 90 should raise."""
        with pytest.raises(ValueError, match="friction_angle_deg"):
            failure_criterion_mohr_coulomb_ug(5.0, 10.0, 90.0)
