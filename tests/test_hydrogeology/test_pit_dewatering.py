"""Tests for minelab.hydrogeology.pit_dewatering module."""

import math

import pytest

from minelab.hydrogeology.pit_dewatering import (
    cone_of_depression_radius,
    darcy_pit_inflow,
    dewatering_power,
    dewatering_well_capacity,
    number_of_dewatering_wells,
    toth_seepage,
)

# -------------------------------------------------------------------------
# Darcy Pit Inflow
# -------------------------------------------------------------------------


class TestDarcyPitInflow:
    """Tests for Darcy's law pit inflow."""

    def test_known_value(self):
        """Q = K * i * A = 5.0 * 0.02 * 10000 = 1000 m3/day."""
        result = darcy_pit_inflow(K=5.0, hydraulic_gradient=0.02, area=10000.0)
        assert result == pytest.approx(1000.0)

    def test_proportional_to_conductivity(self):
        """Doubling K should double the inflow."""
        q1 = darcy_pit_inflow(K=5.0, hydraulic_gradient=0.02, area=10000.0)
        q2 = darcy_pit_inflow(K=10.0, hydraulic_gradient=0.02, area=10000.0)
        assert q2 == pytest.approx(2.0 * q1)

    def test_proportional_to_area(self):
        """Doubling area should double the inflow."""
        q1 = darcy_pit_inflow(K=5.0, hydraulic_gradient=0.02, area=5000.0)
        q2 = darcy_pit_inflow(K=5.0, hydraulic_gradient=0.02, area=10000.0)
        assert q2 == pytest.approx(2.0 * q1)

    def test_invalid_conductivity(self):
        with pytest.raises(ValueError, match="K"):
            darcy_pit_inflow(K=0.0, hydraulic_gradient=0.02, area=10000.0)

    def test_invalid_gradient(self):
        with pytest.raises(ValueError, match="hydraulic_gradient"):
            darcy_pit_inflow(K=5.0, hydraulic_gradient=-0.01, area=10000.0)

    def test_invalid_area(self):
        with pytest.raises(ValueError, match="area"):
            darcy_pit_inflow(K=5.0, hydraulic_gradient=0.02, area=0.0)


# -------------------------------------------------------------------------
# Toth Seepage
# -------------------------------------------------------------------------


class TestTothSeepage:
    """Tests for Toth (1963) regional flow seepage."""

    def test_known_value(self):
        """Q = K * head_diff * sqrt(pit_area) / pit_depth.

        K=2, head_diff=30, pit_area=40000 (sqrt=200), pit_depth=100.
        Q = 2 * 30 * 200 / 100 = 120.0 m3/day.
        """
        result = toth_seepage(
            K=2.0, head_diff=30.0, pit_depth=100.0, pit_area=40000.0
        )
        assert result == pytest.approx(120.0)

    def test_larger_pit_area_increases_seepage(self):
        """Larger pit area should increase seepage."""
        q1 = toth_seepage(K=2.0, head_diff=30.0, pit_depth=100.0, pit_area=10000.0)
        q2 = toth_seepage(K=2.0, head_diff=30.0, pit_depth=100.0, pit_area=40000.0)
        assert q2 > q1

    def test_deeper_pit_reduces_seepage(self):
        """Deeper pit should reduce seepage (denominator effect)."""
        q1 = toth_seepage(K=2.0, head_diff=30.0, pit_depth=50.0, pit_area=40000.0)
        q2 = toth_seepage(K=2.0, head_diff=30.0, pit_depth=200.0, pit_area=40000.0)
        assert q1 > q2

    def test_invalid_pit_depth(self):
        with pytest.raises(ValueError, match="pit_depth"):
            toth_seepage(K=2.0, head_diff=30.0, pit_depth=0.0, pit_area=40000.0)


# -------------------------------------------------------------------------
# Dewatering Well Capacity
# -------------------------------------------------------------------------


class TestDewateringWellCapacity:
    """Tests for the Thiem (1906) steady-state well equation."""

    def test_known_value(self):
        """Q = 2*pi*K*L*dh / ln(R/r).

        K=10, L=20, dh=15, r=0.15, R=300.
        ln(300/0.15) = ln(2000) = 7.6009
        Q = 2*pi*10*20*15 / 7.6009 = 18849.56 / 7.6009 = 2480.0 approx.
        """
        result = dewatering_well_capacity(
            K=10.0,
            screen_length=20.0,
            head_reduction=15.0,
            r_well=0.15,
            r_influence=300.0,
        )
        expected = (
            2.0 * math.pi * 10.0 * 20.0 * 15.0
            / math.log(300.0 / 0.15)
        )
        assert result == pytest.approx(expected, rel=1e-4)

    def test_higher_conductivity_gives_more_capacity(self):
        """More permeable aquifer yields higher well capacity."""
        q1 = dewatering_well_capacity(5.0, 20.0, 15.0, 0.15, 300.0)
        q2 = dewatering_well_capacity(20.0, 20.0, 15.0, 0.15, 300.0)
        assert q2 > q1

    def test_r_influence_must_exceed_r_well(self):
        """r_influence <= r_well should raise ValueError."""
        with pytest.raises(ValueError, match="r_influence"):
            dewatering_well_capacity(10.0, 20.0, 15.0, 0.15, 0.10)

    def test_invalid_screen_length(self):
        with pytest.raises(ValueError, match="screen_length"):
            dewatering_well_capacity(10.0, 0.0, 15.0, 0.15, 300.0)


# -------------------------------------------------------------------------
# Number of Dewatering Wells
# -------------------------------------------------------------------------


class TestNumberOfDewateringWells:
    """Tests for well count estimation."""

    def test_basic_no_interference(self):
        """total=5000, well=1000, interference=0 -> 5 wells."""
        result = number_of_dewatering_wells(5000.0, 1000.0, 0.0)
        assert result == 5

    def test_with_interference(self):
        """total=5000, well=1000, interference=0.2 -> eff=800 -> 7 wells."""
        result = number_of_dewatering_wells(5000.0, 1000.0, 0.2)
        # 5000 / 800 = 6.25 -> ceil -> 7
        assert result == 7

    def test_ceiling_rounding(self):
        """Verify ceiling behaviour: 3001/1000 -> 4 wells."""
        result = number_of_dewatering_wells(3001.0, 1000.0, 0.0)
        assert result == 4

    def test_more_interference_needs_more_wells(self):
        """Higher interference should require more wells."""
        n1 = number_of_dewatering_wells(5000.0, 1000.0, 0.1)
        n2 = number_of_dewatering_wells(5000.0, 1000.0, 0.5)
        assert n2 >= n1

    def test_invalid_total_flow(self):
        with pytest.raises(ValueError, match="total_Q"):
            number_of_dewatering_wells(0.0, 1000.0, 0.1)

    def test_invalid_interference_factor(self):
        with pytest.raises(ValueError, match="interference_factor"):
            number_of_dewatering_wells(5000.0, 1000.0, 1.0)


# -------------------------------------------------------------------------
# Dewatering Power
# -------------------------------------------------------------------------


class TestDewateringPower:
    """Tests for pumping power calculation."""

    def test_known_value(self):
        """Q=8640 m3/day=0.1 m3/s, TDH=50 m, eff=0.7.

        P = 1000 * 9.81 * 0.1 * 50 / 0.7 = 70071.4 W = 70.071 kW.
        """
        result = dewatering_power(
            Q_total=8640.0, total_dynamic_head=50.0, pump_efficiency=0.7
        )
        assert result == pytest.approx(70.071, rel=1e-3)

    def test_higher_head_increases_power(self):
        """Greater TDH should increase power requirement."""
        p1 = dewatering_power(8640.0, 50.0, 0.7)
        p2 = dewatering_power(8640.0, 100.0, 0.7)
        assert p2 > p1

    def test_higher_efficiency_reduces_power(self):
        """Better efficiency should reduce power requirement."""
        p1 = dewatering_power(8640.0, 50.0, 0.5)
        p2 = dewatering_power(8640.0, 50.0, 0.9)
        assert p1 > p2

    def test_invalid_efficiency(self):
        with pytest.raises(ValueError, match="pump_efficiency"):
            dewatering_power(8640.0, 50.0, 0.0)

    def test_invalid_flow_rate(self):
        with pytest.raises(ValueError, match="Q_total"):
            dewatering_power(-100.0, 50.0, 0.7)


# -------------------------------------------------------------------------
# Cone of Depression Radius
# -------------------------------------------------------------------------


class TestConeOfDepressionRadius:
    """Tests for the cone of depression radius estimation."""

    def test_known_value(self):
        """R = 1.5 * sqrt(4*K*b*t/S).

        K=10, b=20, t=30, S=0.001.
        R = 1.5 * sqrt(4*10*20*30/0.001) = 1.5 * sqrt(24000000)
          = 1.5 * 4898.98 = 7348.47 m.
        """
        result = cone_of_depression_radius(
            K=10.0, b=20.0, Q=500.0, t=30.0, S=0.001
        )
        expected = 1.5 * math.sqrt(4.0 * 10.0 * 20.0 * 30.0 / 0.001)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_radius_increases_with_time(self):
        """Cone of depression should expand with time."""
        r1 = cone_of_depression_radius(10.0, 20.0, 500.0, 1.0, 0.001)
        r2 = cone_of_depression_radius(10.0, 20.0, 500.0, 100.0, 0.001)
        assert r2 > r1

    def test_higher_storativity_reduces_radius(self):
        """Higher storativity should produce smaller cone."""
        r1 = cone_of_depression_radius(10.0, 20.0, 500.0, 30.0, 0.001)
        r2 = cone_of_depression_radius(10.0, 20.0, 500.0, 30.0, 0.1)
        assert r1 > r2

    def test_invalid_conductivity(self):
        with pytest.raises(ValueError, match="K"):
            cone_of_depression_radius(-1.0, 20.0, 500.0, 30.0, 0.001)

    def test_invalid_storativity(self):
        with pytest.raises(ValueError, match="S"):
            cone_of_depression_radius(10.0, 20.0, 500.0, 30.0, 0.0)
