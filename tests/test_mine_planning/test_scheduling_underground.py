"""Tests for minelab.mine_planning.scheduling_underground module."""

import math

import pytest

from minelab.mine_planning.scheduling_underground import (
    activity_on_node,
    block_value_underground,
    lateral_development_schedule,
    ore_pass_capacity,
    ramp_access_time,
    shaft_hoisting_capacity,
)

# -------------------------------------------------------------------------
# Activity-on-Node (CPM)
# -------------------------------------------------------------------------


class TestActivityOnNode:
    """Tests for the Critical Path Method scheduling."""

    def test_simple_serial(self):
        """Three serial activities: A->B->C, durations 3,5,2.

        Critical path: all activities, duration = 10.
        """
        result = activity_on_node(
            activities=["A", "B", "C"],
            durations=[3.0, 5.0, 2.0],
            dependencies=[[], [0], [1]],
        )
        assert result["project_duration"] == pytest.approx(10.0)
        assert result["critical_path"] == ["A", "B", "C"]

    def test_parallel_paths(self):
        """Two parallel paths from start to end.

        A(5) -> C(3)  total=8  (critical)
        B(4) -> C(3)  total=7  (not critical, B has float=1)

        Activities: A, B, C
        Deps: A=[], B=[], C=[A, B]
        """
        result = activity_on_node(
            activities=["A", "B", "C"],
            durations=[5.0, 4.0, 3.0],
            dependencies=[[], [], [0, 1]],
        )
        assert result["project_duration"] == pytest.approx(8.0)
        assert "A" in result["critical_path"]
        assert "C" in result["critical_path"]

    def test_single_activity(self):
        """Single activity should be the entire critical path."""
        result = activity_on_node(
            activities=["X"],
            durations=[10.0],
            dependencies=[[]],
        )
        assert result["project_duration"] == pytest.approx(10.0)
        assert result["critical_path"] == ["X"]
        assert result["activities"][0]["es"] == pytest.approx(0.0)
        assert result["activities"][0]["ef"] == pytest.approx(10.0)
        assert result["activities"][0]["float"] == pytest.approx(0.0)

    def test_activity_details(self):
        """Verify ES, EF, LS, LF, Float for serial tasks."""
        result = activity_on_node(
            activities=["A", "B"],
            durations=[4.0, 6.0],
            dependencies=[[], [0]],
        )
        a_info = result["activities"][0]
        b_info = result["activities"][1]
        assert a_info["es"] == pytest.approx(0.0)
        assert a_info["ef"] == pytest.approx(4.0)
        assert b_info["es"] == pytest.approx(4.0)
        assert b_info["ef"] == pytest.approx(10.0)

    def test_empty_activities_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            activity_on_node([], [], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            activity_on_node(
                activities=["A", "B"],
                durations=[3.0],
                dependencies=[[], [0]],
            )

    def test_invalid_dependency_raises(self):
        with pytest.raises(ValueError, match="Invalid dependency"):
            activity_on_node(
                activities=["A", "B"],
                durations=[3.0, 5.0],
                dependencies=[[], [5]],
            )


# -------------------------------------------------------------------------
# Lateral Development Schedule
# -------------------------------------------------------------------------


class TestLateralDevelopmentSchedule:
    """Tests for sequential zone development scheduling."""

    def test_basic(self):
        """Two zones: 300m and 200m at 100m/month, $500/m.

        Zone A: 3 months, $150,000.  Zone B: 2 months, $100,000.
        Total: 5 months, 500m, $250,000.
        """
        result = lateral_development_schedule(
            zones=["A", "B"],
            footage_per_zone=[300.0, 200.0],
            monthly_advance=100.0,
            advance_cost_per_m=500.0,
        )
        assert result["total_months"] == pytest.approx(5.0)
        assert result["total_metres"] == pytest.approx(500.0)
        assert result["total_cost"] == pytest.approx(250000.0)

    def test_schedule_timing(self):
        """Verify sequential start/end months."""
        result = lateral_development_schedule(
            zones=["A", "B", "C"],
            footage_per_zone=[200.0, 100.0, 300.0],
            monthly_advance=100.0,
            advance_cost_per_m=500.0,
        )
        sched = result["schedule"]
        assert sched[0]["start_month"] == pytest.approx(0.0)
        assert sched[0]["end_month"] == pytest.approx(2.0)
        assert sched[1]["start_month"] == pytest.approx(2.0)
        assert sched[1]["end_month"] == pytest.approx(3.0)
        assert sched[2]["start_month"] == pytest.approx(3.0)
        assert sched[2]["end_month"] == pytest.approx(6.0)

    def test_single_zone(self):
        """Single zone development."""
        result = lateral_development_schedule(
            zones=["Main"],
            footage_per_zone=[500.0],
            monthly_advance=50.0,
            advance_cost_per_m=1000.0,
        )
        assert result["total_months"] == pytest.approx(10.0)
        assert result["total_cost"] == pytest.approx(500000.0)

    def test_empty_zones_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            lateral_development_schedule([], [], 100.0, 500.0)

    def test_invalid_monthly_advance(self):
        with pytest.raises(ValueError, match="monthly_advance"):
            lateral_development_schedule(
                ["A"], [300.0], 0.0, 500.0
            )

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="advance_cost_per_m"):
            lateral_development_schedule(
                ["A"], [300.0], 100.0, -10.0
            )


# -------------------------------------------------------------------------
# Ore Pass Capacity
# -------------------------------------------------------------------------


class TestOrePassCapacity:
    """Tests for Hambley (1987) ore pass capacity."""

    def test_known_value(self):
        """D=3m, H=30m, draw_angle=60 deg, density=2.7 t/m3.

        cross_area = pi/4 * 9 = 7.0686
        total_vol = 7.0686 * 30 = 212.06 m3
        cone_height = 3 / (2*tan(60)) = 3/3.4641 = 0.8660
        live_vol = 7.0686 * (30 - 0.8660) = 7.0686 * 29.134 = 205.93
        dead_vol = 212.06 - 205.93 = 6.12
        live_cap = 205.93 * 2.7 = 556.0 t approx.
        """
        result = ore_pass_capacity(
            diameter=3.0, height=30.0,
            draw_angle=60.0, bulk_density=2.7,
        )
        cross_area = math.pi / 4.0 * 3.0**2
        cone_h = 3.0 / (2.0 * math.tan(math.radians(60.0)))
        expected_live = cross_area * (30.0 - cone_h)
        expected_total = cross_area * 30.0

        assert result["total_volume_m3"] == pytest.approx(
            expected_total, rel=1e-4
        )
        assert result["live_volume_m3"] == pytest.approx(
            expected_live, rel=1e-4
        )
        assert result["live_capacity_tonnes"] == pytest.approx(
            expected_live * 2.7, rel=1e-4
        )

    def test_larger_diameter_increases_capacity(self):
        """Wider ore pass should have greater capacity."""
        r1 = ore_pass_capacity(2.0, 30.0, 60.0, 2.7)
        r2 = ore_pass_capacity(4.0, 30.0, 60.0, 2.7)
        assert r2["live_capacity_tonnes"] > r1["live_capacity_tonnes"]

    def test_taller_pass_increases_capacity(self):
        """Taller ore pass should have greater live volume."""
        r1 = ore_pass_capacity(3.0, 20.0, 60.0, 2.7)
        r2 = ore_pass_capacity(3.0, 50.0, 60.0, 2.7)
        assert r2["live_volume_m3"] > r1["live_volume_m3"]

    def test_invalid_diameter(self):
        with pytest.raises(ValueError, match="diameter"):
            ore_pass_capacity(0.0, 30.0, 60.0, 2.7)

    def test_invalid_draw_angle(self):
        with pytest.raises(ValueError, match="draw_angle"):
            ore_pass_capacity(3.0, 30.0, 0.0, 2.7)


# -------------------------------------------------------------------------
# Shaft Hoisting Capacity
# -------------------------------------------------------------------------


class TestShaftHoistingCapacity:
    """Tests for shaft hoisting capacity calculation."""

    def test_known_value(self):
        """cage=10t, cycle=5min, hours=16, avail=0.85.

        hoists/hr = 60/5 = 12
        daily = 10 * 12 * 16 * 0.85 = 1632 t/day
        annual = 1632 * 365 = 595680 t/year.
        """
        result = shaft_hoisting_capacity(
            cage_capacity=10.0,
            cycle_time_min=5.0,
            operating_hours=16.0,
            availability=0.85,
        )
        assert result["hoists_per_hour"] == pytest.approx(12.0)
        assert result["daily_capacity_tonnes"] == pytest.approx(1632.0)
        assert result["annual_capacity_tonnes"] == pytest.approx(
            595680.0
        )

    def test_faster_cycle_increases_capacity(self):
        """Shorter cycle time should increase daily capacity."""
        r1 = shaft_hoisting_capacity(10.0, 10.0, 16.0, 0.85)
        r2 = shaft_hoisting_capacity(10.0, 5.0, 16.0, 0.85)
        assert r2["daily_capacity_tonnes"] > r1["daily_capacity_tonnes"]

    def test_higher_availability_increases_capacity(self):
        """Better availability should increase capacity."""
        r1 = shaft_hoisting_capacity(10.0, 5.0, 16.0, 0.5)
        r2 = shaft_hoisting_capacity(10.0, 5.0, 16.0, 0.95)
        assert r2["daily_capacity_tonnes"] > r1["daily_capacity_tonnes"]

    def test_invalid_cage_capacity(self):
        with pytest.raises(ValueError, match="cage_capacity"):
            shaft_hoisting_capacity(0.0, 5.0, 16.0, 0.85)

    def test_invalid_availability(self):
        with pytest.raises(ValueError, match="availability"):
            shaft_hoisting_capacity(10.0, 5.0, 16.0, 0.0)


# -------------------------------------------------------------------------
# Ramp Access Time
# -------------------------------------------------------------------------


class TestRampAccessTime:
    """Tests for ramp travel time estimation."""

    def test_known_value(self):
        """ramp=1000m, gradient=10%, speed=30 km/h.

        eff_speed = 30 * (1 - 0.10) = 27 km/h
        time = (1000/1000) / 27 * 60 = 2.222 min.
        """
        result = ramp_access_time(
            ramp_length=1000.0,
            ramp_gradient_pct=10.0,
            vehicle_speed_kmh=30.0,
        )
        assert result == pytest.approx(2.2222, rel=1e-3)

    def test_flat_ramp(self):
        """Zero gradient: time = distance / speed.

        1000m at 30 km/h => 1/30*60 = 2.0 min.
        """
        result = ramp_access_time(
            ramp_length=1000.0,
            ramp_gradient_pct=0.0,
            vehicle_speed_kmh=30.0,
        )
        assert result == pytest.approx(2.0)

    def test_steeper_gradient_increases_time(self):
        """Steeper gradient should increase travel time."""
        t1 = ramp_access_time(1000.0, 5.0, 30.0)
        t2 = ramp_access_time(1000.0, 15.0, 30.0)
        assert t2 > t1

    def test_longer_ramp_increases_time(self):
        """Longer ramp should take more time."""
        t1 = ramp_access_time(500.0, 10.0, 30.0)
        t2 = ramp_access_time(2000.0, 10.0, 30.0)
        assert t2 > t1

    def test_invalid_ramp_length(self):
        with pytest.raises(ValueError, match="ramp_length"):
            ramp_access_time(0.0, 10.0, 30.0)

    def test_invalid_gradient(self):
        with pytest.raises(ValueError, match="ramp_gradient_pct"):
            ramp_access_time(1000.0, -5.0, 30.0)

    def test_invalid_speed(self):
        with pytest.raises(ValueError, match="vehicle_speed_kmh"):
            ramp_access_time(1000.0, 10.0, 0.0)


# -------------------------------------------------------------------------
# Block Value Underground
# -------------------------------------------------------------------------


class TestBlockValueUnderground:
    """Tests for underground mining block valuation."""

    def test_known_value(self):
        """tonnes=10000, grade=5.0, nsr=50, mining=20, fill=5, diluted=4.0.

        value = 10000 * (4.0*50 - 20 - 5) = 10000 * (200 - 25) = 1750000.
        """
        result = block_value_underground(
            tonnes=10000.0,
            grade=5.0,
            nsr_per_unit=50.0,
            mining_cost=20.0,
            filling_cost=5.0,
            diluted_grade=4.0,
        )
        assert result == pytest.approx(1750000.0)

    def test_negative_block_value(self):
        """Low grade block should have negative value."""
        result = block_value_underground(
            tonnes=10000.0,
            grade=1.0,
            nsr_per_unit=50.0,
            mining_cost=40.0,
            filling_cost=15.0,
            diluted_grade=0.5,
        )
        # value = 10000 * (0.5*50 - 40 - 15) = 10000 * (25 - 55) = -300000
        assert result == pytest.approx(-300000.0)

    def test_zero_filling_cost(self):
        """No backfill required."""
        result = block_value_underground(
            tonnes=5000.0,
            grade=3.0,
            nsr_per_unit=100.0,
            mining_cost=30.0,
            filling_cost=0.0,
            diluted_grade=2.5,
        )
        # value = 5000 * (2.5*100 - 30 - 0) = 5000 * 220 = 1100000
        assert result == pytest.approx(1100000.0)

    def test_higher_nsr_increases_value(self):
        """Higher NSR should increase block value."""
        v1 = block_value_underground(
            10000.0, 5.0, 30.0, 20.0, 5.0, 4.0
        )
        v2 = block_value_underground(
            10000.0, 5.0, 80.0, 20.0, 5.0, 4.0
        )
        assert v2 > v1

    def test_invalid_tonnes(self):
        with pytest.raises(ValueError, match="tonnes"):
            block_value_underground(
                0.0, 5.0, 50.0, 20.0, 5.0, 4.0
            )

    def test_invalid_grade(self):
        with pytest.raises(ValueError, match="grade"):
            block_value_underground(
                10000.0, -1.0, 50.0, 20.0, 5.0, 4.0
            )

    def test_invalid_mining_cost(self):
        with pytest.raises(ValueError, match="mining_cost"):
            block_value_underground(
                10000.0, 5.0, 50.0, -1.0, 5.0, 4.0
            )
