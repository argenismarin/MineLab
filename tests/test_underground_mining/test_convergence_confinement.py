"""Tests for minelab.underground_mining.convergence_confinement."""

import pytest

from minelab.underground_mining.convergence_confinement import (
    ground_reaction_curve,
    longitudinal_deformation_profile,
    rock_burst_potential,
    rock_support_interaction,
    squeezing_index,
    support_reaction_curve,
    tunnel_deformation_strain,
)


class TestGroundReactionCurve:
    """Tests for Carranza-Torres & Fairhurst (2000) GRC."""

    def test_elastic_regime(self):
        """High p_i (above p_cr) -> elastic, no plastic zone."""
        result = ground_reaction_curve(
            p_i=8.0,
            sigma_0=10.0,
            sigma_ci=50.0,
            mi=10.0,
            gsi=80.0,
            r_tunnel=3.0,
            e_rock=10000.0,
        )
        assert result["r_plastic_m"] == 0.0
        assert result["u_max_mm"] > 0

    def test_plastic_regime(self):
        """Low p_i -> plastic zone develops."""
        result = ground_reaction_curve(
            p_i=0.0,
            sigma_0=20.0,
            sigma_ci=30.0,
            mi=10.0,
            gsi=40.0,
            r_tunnel=3.0,
            e_rock=5000.0,
        )
        assert result["r_plastic_m"] > 3.0
        assert result["u_max_mm"] > 0
        assert result["convergence_pct"] > 0

    def test_p_critical_non_negative(self):
        """Critical pressure should be non-negative."""
        result = ground_reaction_curve(
            p_i=0.5,
            sigma_0=10.0,
            sigma_ci=50.0,
            mi=10.0,
            gsi=60.0,
            r_tunnel=3.0,
            e_rock=5000.0,
        )
        assert result["p_critical_mpa"] >= 0

    def test_convergence_positive(self):
        """Convergence should be positive for any loading."""
        result = ground_reaction_curve(
            p_i=0.0,
            sigma_0=15.0,
            sigma_ci=40.0,
            mi=8.0,
            gsi=50.0,
            r_tunnel=4.0,
            e_rock=3000.0,
        )
        assert result["convergence_pct"] > 0

    def test_invalid_gsi(self):
        """GSI > 100 should raise."""
        with pytest.raises(ValueError, match="gsi"):
            ground_reaction_curve(
                p_i=0.0,
                sigma_0=10.0,
                sigma_ci=50.0,
                mi=10.0,
                gsi=110.0,
                r_tunnel=3.0,
                e_rock=5000.0,
            )

    def test_invalid_sigma_0(self):
        """sigma_0 <= 0 should raise."""
        with pytest.raises(ValueError, match="sigma_0"):
            ground_reaction_curve(
                p_i=0.0,
                sigma_0=0.0,
                sigma_ci=50.0,
                mi=10.0,
                gsi=60.0,
                r_tunnel=3.0,
                e_rock=5000.0,
            )


class TestSupportReactionCurve:
    """Tests for support reaction curve parameters."""

    def test_basic(self):
        """k=0.01, p_max=0.5, u_install=5 -> u_at_pmax=55."""
        result = support_reaction_curve(0.01, 0.5, 5.0)
        assert result["max_displacement_mm"] == pytest.approx(55.0, rel=1e-4)

    def test_stiff_support(self):
        """Stiffer support -> smaller displacement range."""
        soft = support_reaction_curve(0.01, 0.5, 5.0)
        stiff = support_reaction_curve(0.1, 0.5, 5.0)
        assert stiff["max_displacement_mm"] < soft["max_displacement_mm"]

    def test_keys_present(self):
        """All expected keys should be present."""
        result = support_reaction_curve(0.05, 1.0, 3.0)
        assert "support_stiffness" in result
        assert "max_support_pressure" in result
        assert "installation_displacement" in result

    def test_invalid_k_support(self):
        """Zero stiffness should raise."""
        with pytest.raises(ValueError, match="k_support"):
            support_reaction_curve(0, 0.5, 5.0)


class TestLongitudinalDeformationProfile:
    """Tests for Vlachopoulos & Diederichs (2009) LDP."""

    def test_at_face(self):
        """At the face (x=0), displacement ~ 1/3 of u_max."""
        u = longitudinal_deformation_profile(0.0, 3.0, 3.0, 10.0)
        # For elastic case (rp=rt), u_face_ratio = 1/3
        assert u == pytest.approx(10.0 / 3.0, rel=0.05)

    def test_far_behind_face(self):
        """Far behind face, displacement approaches u_max."""
        u = longitudinal_deformation_profile(50.0, 3.0, 3.0, 10.0)
        assert u == pytest.approx(10.0, rel=0.01)

    def test_ahead_of_face(self):
        """Ahead of face (x<0), displacement is small."""
        u = longitudinal_deformation_profile(-10.0, 3.0, 3.0, 10.0)
        assert u < 3.0
        assert u > 0

    def test_monotonic_increase(self):
        """Displacement increases with distance behind face."""
        u1 = longitudinal_deformation_profile(1.0, 3.0, 3.0, 10.0)
        u2 = longitudinal_deformation_profile(5.0, 3.0, 3.0, 10.0)
        u3 = longitudinal_deformation_profile(20.0, 3.0, 3.0, 10.0)
        assert u3 > u2 > u1

    def test_zero_umax(self):
        """If u_max is 0, displacement is 0 everywhere."""
        u = longitudinal_deformation_profile(5.0, 3.0, 3.0, 0.0)
        assert u == 0.0

    def test_invalid_r_tunnel(self):
        """Zero radius should raise."""
        with pytest.raises(ValueError, match="r_tunnel"):
            longitudinal_deformation_profile(5.0, 0.0, 3.0, 10.0)


class TestRockSupportInteraction:
    """Tests for GRC-SRC equilibrium."""

    def test_basic_equilibrium(self):
        """Finds an equilibrium point."""
        p_i = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        u_grc = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]
        result = rock_support_interaction(
            10.0,
            p_i,
            u_grc,
            0.5,
            2.0,
            2.0,
        )
        assert result["equilibrium_pressure_mpa"] >= 0
        assert result["equilibrium_displacement_mm"] >= 0
        assert result["factor_of_safety"] > 0

    def test_fos_decreases_with_weaker_support(self):
        """Weaker support -> lower FoS."""
        p_i = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        u_grc = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]
        strong = rock_support_interaction(
            10.0,
            p_i,
            u_grc,
            0.5,
            2.0,
            5.0,
        )
        weak = rock_support_interaction(
            10.0,
            p_i,
            u_grc,
            0.5,
            2.0,
            1.0,
        )
        assert strong["factor_of_safety"] >= weak["factor_of_safety"]

    def test_too_few_points(self):
        """Arrays with < 2 elements should raise."""
        with pytest.raises(ValueError, match="at least"):
            rock_support_interaction(
                10.0,
                [1.0],
                [0.5],
                0.5,
                2.0,
                1.0,
            )


class TestSqueezingIndex:
    """Tests for Hoek & Marinos (2000) squeezing index."""

    def test_extreme_squeezing(self):
        """sigma_cm/sigma_0 < 0.2 -> extreme."""
        result = squeezing_index(30.0, 5.0)
        assert result["ratio"] == pytest.approx(5.0 / 30.0, rel=1e-4)
        assert result["classification"] == "extreme squeezing"

    def test_severe_squeezing(self):
        """ratio 0.2-0.3 -> severe."""
        result = squeezing_index(20.0, 5.0)
        assert result["classification"] == "severe squeezing"

    def test_moderate_squeezing(self):
        """ratio 0.3-0.4 -> moderate."""
        result = squeezing_index(10.0, 3.5)
        assert result["classification"] == "moderate squeezing"

    def test_few_problems(self):
        """ratio > 0.4 -> few support problems."""
        result = squeezing_index(10.0, 6.0)
        assert result["classification"] == "few support problems"

    def test_strain_estimate(self):
        """Strain ~ 1/ratio^2."""
        result = squeezing_index(20.0, 4.0)
        ratio = 4.0 / 20.0
        expected_strain = 1.0 / (ratio**2)
        assert result["strain_pct_estimate"] == pytest.approx(expected_strain, rel=1e-4)

    def test_invalid_sigma_0(self):
        """sigma_0 <= 0 should raise."""
        with pytest.raises(ValueError, match="sigma_0"):
            squeezing_index(0.0, 5.0)


class TestRockBurstPotential:
    """Tests for Kaiser et al. (1996) rock burst assessment."""

    def test_high_potential(self):
        """High stress ratio and BPI -> high potential."""
        result = rock_burst_potential(80.0, 3.0, 100.0, 50.0)
        assert result["bpi"] == pytest.approx(0.8, rel=1e-4)
        assert result["potential"] == "high"
        assert result["brittleness_class"] == "brittle"

    def test_no_potential(self):
        """Low BPI -> none."""
        result = rock_burst_potential(10.0, 5.0, 100.0, 20.0)
        assert result["bpi"] == pytest.approx(0.1, rel=1e-4)
        assert result["potential"] == "none"

    def test_moderate_potential(self):
        """BPI > 0.5 but stress_ratio < 20 -> moderate."""
        result = rock_burst_potential(60.0, 10.0, 100.0, 50.0)
        assert result["potential"] == "moderate"

    def test_ductile_class(self):
        """Brittleness <= 40 -> ductile."""
        result = rock_burst_potential(50.0, 5.0, 100.0, 30.0)
        assert result["brittleness_class"] == "ductile"

    def test_zero_sigma_3(self):
        """sigma_3 = 0 -> infinite stress ratio."""
        result = rock_burst_potential(50.0, 0.0, 100.0, 50.0)
        assert result["stress_ratio"] == float("inf")

    def test_invalid_ucs(self):
        """ucs <= 0 should raise."""
        with pytest.raises(ValueError, match="ucs"):
            rock_burst_potential(50.0, 5.0, 0.0, 30.0)


class TestTunnelDeformationStrain:
    """Tests for Hoek & Marinos tunnel strain."""

    def test_known_value(self):
        """p_i=0, sigma_0=10, sigma_cm=3 -> strain=0.2*(3/10)^-2."""
        result = tunnel_deformation_strain(0.0, 10.0, 3.0, 3.0)
        expected = 0.2 * (3.0 / 10.0) ** (-2)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_stress_more_strain(self):
        """Higher sigma_0 -> more strain."""
        low = tunnel_deformation_strain(0.0, 5.0, 3.0, 3.0)
        high = tunnel_deformation_strain(0.0, 15.0, 3.0, 3.0)
        assert high > low

    def test_support_reduces_strain(self):
        """Higher p_i -> less strain."""
        unsupported = tunnel_deformation_strain(0.0, 10.0, 3.0, 3.0)
        supported = tunnel_deformation_strain(3.0, 10.0, 3.0, 3.0)
        assert unsupported > supported

    def test_zero_net_stress(self):
        """If p_i >= sigma_0, strain = 0."""
        result = tunnel_deformation_strain(10.0, 10.0, 3.0, 3.0)
        assert result == 0.0

    def test_invalid_sigma_cm(self):
        """sigma_cm <= 0 should raise."""
        with pytest.raises(ValueError, match="sigma_cm"):
            tunnel_deformation_strain(0.0, 10.0, 0.0, 3.0)
