"""Tests for minelab.hydrogeology.aquifer_tests module."""


import pytest

from minelab.hydrogeology.aquifer_tests import (
    aquifer_hydraulic_conductivity,
    cooper_jacob_drawdown,
    specific_capacity,
    theis_drawdown,
    theis_recovery,
    transmissivity_from_slug,
)

# -------------------------------------------------------------------------
# Theis Drawdown
# -------------------------------------------------------------------------


class TestTheisDrawdown:
    """Tests for the Theis (1935) well function drawdown."""

    def test_known_value(self):
        """Verify drawdown for a typical confined aquifer scenario.

        Q=500 m3/day, T=100 m2/day, S=0.001, r=50 m, t=1 day.
        u = r^2*S/(4*T*t) = 2500*0.001/400 = 0.00625
        W(u=0.00625) = 4.506 (from series expansion)
        s = 500/(4*pi*100) * W(u) = 0.3979 * 4.506 = 1.7922 m.
        """
        result = theis_drawdown(Q=500.0, T=100.0, S=0.001, r=50.0, t=1.0)
        assert result == pytest.approx(1.7922, rel=1e-3)

    def test_drawdown_increases_with_time(self):
        """Drawdown should increase as pumping time increases."""
        s1 = theis_drawdown(Q=500.0, T=100.0, S=0.001, r=50.0, t=0.5)
        s2 = theis_drawdown(Q=500.0, T=100.0, S=0.001, r=50.0, t=2.0)
        assert s2 > s1

    def test_drawdown_decreases_with_distance(self):
        """Drawdown should decrease further from the well."""
        s_near = theis_drawdown(Q=500.0, T=100.0, S=0.001, r=10.0, t=1.0)
        s_far = theis_drawdown(Q=500.0, T=100.0, S=0.001, r=100.0, t=1.0)
        assert s_near > s_far

    def test_invalid_pumping_rate(self):
        with pytest.raises(ValueError, match="Q"):
            theis_drawdown(Q=-10.0, T=100.0, S=0.001, r=50.0, t=1.0)

    def test_invalid_transmissivity(self):
        with pytest.raises(ValueError, match="T"):
            theis_drawdown(Q=500.0, T=0.0, S=0.001, r=50.0, t=1.0)

    def test_invalid_storativity(self):
        with pytest.raises(ValueError, match="S"):
            theis_drawdown(Q=500.0, T=100.0, S=-0.001, r=50.0, t=1.0)


# -------------------------------------------------------------------------
# Cooper-Jacob Drawdown
# -------------------------------------------------------------------------


class TestCooperJacobDrawdown:
    """Tests for the Cooper-Jacob (1946) log approximation."""

    def test_known_value(self):
        """Cooper-Jacob for Q=500, T=100, S=0.001, r=50, t=1 day.

        s = (2.3*500)/(4*pi*100) * log10(2.25*100*1/(50^2*0.001))
          = (1150/1256.637) * log10(2.25*100/2.5)
          = 0.9152 * log10(90)
          = 0.9152 * 1.9542
          = 1.788 m approximately.
        """
        result = cooper_jacob_drawdown(
            Q=500.0, T=100.0, S=0.001, r=50.0, t=1.0
        )
        assert result == pytest.approx(1.788, rel=1e-2)

    def test_matches_theis_for_small_u(self):
        """Cooper-Jacob should approximate Theis when u is small.

        Use large t to ensure u << 0.05.
        """
        cj = cooper_jacob_drawdown(
            Q=500.0, T=100.0, S=0.0001, r=10.0, t=10.0
        )
        th = theis_drawdown(
            Q=500.0, T=100.0, S=0.0001, r=10.0, t=10.0
        )
        assert cj == pytest.approx(th, rel=1e-2)

    def test_drawdown_increases_with_pumping_rate(self):
        """Higher Q should produce greater drawdown."""
        s1 = cooper_jacob_drawdown(
            Q=200.0, T=100.0, S=0.001, r=50.0, t=1.0
        )
        s2 = cooper_jacob_drawdown(
            Q=800.0, T=100.0, S=0.001, r=50.0, t=1.0
        )
        assert s2 > s1

    def test_invalid_r(self):
        with pytest.raises(ValueError, match="r"):
            cooper_jacob_drawdown(
                Q=500.0, T=100.0, S=0.001, r=0.0, t=1.0
            )


# -------------------------------------------------------------------------
# Theis Recovery
# -------------------------------------------------------------------------


class TestTheisRecovery:
    """Tests for the Theis recovery method."""

    def test_known_value(self):
        """Q=500, T=100, t_pump=5 days, t_since_stop=1 day.

        s' = (2.3*500)/(4*pi*100) * log10((5+1)/1)
           = 0.9152 * log10(6)
           = 0.9152 * 0.7782
           = 0.7122 m approximately.
        """
        result = theis_recovery(
            Q=500.0, T=100.0, t_pump=5.0, t_since_stop=1.0
        )
        assert result == pytest.approx(0.7122, rel=1e-2)

    def test_recovery_decreases_with_time(self):
        """Residual drawdown should decrease as recovery time increases."""
        s1 = theis_recovery(Q=500.0, T=100.0, t_pump=5.0, t_since_stop=1.0)
        s2 = theis_recovery(
            Q=500.0, T=100.0, t_pump=5.0, t_since_stop=10.0
        )
        assert s1 > s2

    def test_long_recovery_approaches_zero(self):
        """After very long recovery time, residual drawdown is near zero."""
        result = theis_recovery(
            Q=500.0, T=100.0, t_pump=5.0, t_since_stop=1e6
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_invalid_t_pump(self):
        with pytest.raises(ValueError, match="t_pump"):
            theis_recovery(Q=500.0, T=100.0, t_pump=0.0, t_since_stop=1.0)

    def test_invalid_t_since_stop(self):
        with pytest.raises(ValueError, match="t_since_stop"):
            theis_recovery(Q=500.0, T=100.0, t_pump=5.0, t_since_stop=-1.0)


# -------------------------------------------------------------------------
# Transmissivity from Slug Test
# -------------------------------------------------------------------------


class TestTransmissivityFromSlug:
    """Tests for the Bouwer-Rice (1976) slug test estimate."""

    def test_known_value(self):
        """r_casing=0.05, r_screen=0.05, wtd=5, L_screen=3, slug=0.002.

        K = 0.002 / (pi * 0.05^2 * 3 * 5)
          = 0.002 / (pi * 0.0025 * 15)
          = 0.002 / 0.11781
          = 0.01698 m/day
        T = K * 3 = 0.05093 m2/day.
        """
        result = transmissivity_from_slug(
            r_casing=0.05,
            r_screen=0.05,
            water_table_depth=5.0,
            L_screen=3.0,
            slug_volume=0.002,
        )
        assert result == pytest.approx(0.05093, rel=1e-2)

    def test_larger_slug_gives_higher_transmissivity(self):
        """A larger slug volume should yield a higher transmissivity."""
        t1 = transmissivity_from_slug(0.05, 0.05, 5.0, 3.0, 0.001)
        t2 = transmissivity_from_slug(0.05, 0.05, 5.0, 3.0, 0.005)
        assert t2 > t1

    def test_invalid_slug_volume(self):
        with pytest.raises(ValueError, match="slug_volume"):
            transmissivity_from_slug(0.05, 0.05, 5.0, 3.0, -0.001)

    def test_invalid_r_casing(self):
        with pytest.raises(ValueError, match="r_casing"):
            transmissivity_from_slug(0.0, 0.05, 5.0, 3.0, 0.002)


# -------------------------------------------------------------------------
# Specific Capacity
# -------------------------------------------------------------------------


class TestSpecificCapacity:
    """Tests for specific capacity calculation."""

    def test_basic(self):
        """SC = 500 / 5 = 100 m2/day."""
        result = specific_capacity(Q=500.0, drawdown=5.0)
        assert result == pytest.approx(100.0)

    def test_high_drawdown(self):
        """SC = 500 / 50 = 10 m2/day."""
        result = specific_capacity(Q=500.0, drawdown=50.0)
        assert result == pytest.approx(10.0)

    def test_monotonic_with_pumping_rate(self):
        """Higher Q at same drawdown gives higher SC."""
        sc1 = specific_capacity(Q=100.0, drawdown=5.0)
        sc2 = specific_capacity(Q=1000.0, drawdown=5.0)
        assert sc2 > sc1

    def test_invalid_pumping_rate(self):
        with pytest.raises(ValueError, match="Q"):
            specific_capacity(Q=0.0, drawdown=5.0)

    def test_invalid_drawdown(self):
        with pytest.raises(ValueError, match="drawdown"):
            specific_capacity(Q=500.0, drawdown=-1.0)


# -------------------------------------------------------------------------
# Aquifer Hydraulic Conductivity
# -------------------------------------------------------------------------


class TestAquiferHydraulicConductivity:
    """Tests for K = T / b."""

    def test_basic(self):
        """K = 100 / 10 = 10 m/day."""
        result = aquifer_hydraulic_conductivity(
            transmissivity=100.0, aquifer_thickness=10.0
        )
        assert result == pytest.approx(10.0)

    def test_thin_aquifer(self):
        """K = 100 / 2 = 50 m/day."""
        result = aquifer_hydraulic_conductivity(
            transmissivity=100.0, aquifer_thickness=2.0
        )
        assert result == pytest.approx(50.0)

    def test_thicker_aquifer_gives_lower_conductivity(self):
        """Same T with thicker aquifer -> lower K."""
        k1 = aquifer_hydraulic_conductivity(100.0, 10.0)
        k2 = aquifer_hydraulic_conductivity(100.0, 50.0)
        assert k1 > k2

    def test_invalid_transmissivity(self):
        with pytest.raises(ValueError, match="transmissivity"):
            aquifer_hydraulic_conductivity(
                transmissivity=-5.0, aquifer_thickness=10.0
            )

    def test_invalid_thickness(self):
        with pytest.raises(ValueError, match="aquifer_thickness"):
            aquifer_hydraulic_conductivity(
                transmissivity=100.0, aquifer_thickness=0.0
            )
