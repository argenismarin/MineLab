"""Underground mine scheduling and development planning.

This module provides functions for critical path analysis, lateral
development scheduling, ore pass capacity estimation, shaft hoisting
capacity, ramp access timing, and underground block valuation.

References
----------
.. [1] Hustrulid, W. & Bullock, R. (2001). *Underground Mining Methods:
       Engineering Fundamentals and International Case Studies*. SME.
.. [2] Hambley, D.F. (1987). Design of ore pass systems for underground
       mines. *CIM Bulletin*, 80(897), 25--30.
.. [3] Brady, B.H.G. & Brown, E.T. (2006). *Rock Mechanics for
       Underground Mining*, 3rd ed. Springer.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import (
    validate_array,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Activity-on-Node (CPM)
# ---------------------------------------------------------------------------


def activity_on_node(
    activities: list,
    durations: list,
    dependencies: list,
) -> dict:
    """Critical Path Method (CPM) scheduling using activity-on-node.

    Performs forward and backward passes to compute early start (ES),
    early finish (EF), late start (LS), late finish (LF), and total
    float for each activity.  The critical path consists of activities
    with zero total float.

    Parameters
    ----------
    activities : list
        List of activity name strings.  Must have at least 1 element.
    durations : list
        List of activity durations in days.  Each must be >= 0.
        Same length as *activities*.
    dependencies : list
        List of lists, where each inner list contains the indices of
        predecessor activities.  Same length as *activities*.

    Returns
    -------
    dict
        ``project_duration`` : float
            Total project duration (max EF).
        ``critical_path`` : list
            Names of activities on the critical path.
        ``activities`` : list of dict
            Each dict has keys: ``name``, ``es``, ``ef``, ``ls``,
            ``lf``, ``float``.

    Raises
    ------
    ValueError
        If input lists have inconsistent lengths or invalid values.

    References
    ----------
    .. [1] Hustrulid, W. & Bullock, R. (2001). *Underground Mining
           Methods*. SME. Ch. 11.
    """
    n = len(activities)
    if n == 0:
        raise ValueError("'activities' must have at least 1 element.")

    dur_arr = validate_array(durations, "durations", min_length=1)

    if len(dur_arr) != n:
        raise ValueError(
            f"'activities' and 'durations' must have the same length, got {n} and {len(dur_arr)}."
        )
    if len(dependencies) != n:
        raise ValueError(
            "'activities' and 'dependencies' must have the same "
            f"length, got {n} and {len(dependencies)}."
        )

    for i, d in enumerate(dur_arr):
        if d < 0:
            raise ValueError(f"All durations must be non-negative, got durations[{i}]={d}.")

    # Validate dependency indices
    for i, deps in enumerate(dependencies):
        for dep in deps:
            if dep < 0 or dep >= n:
                raise ValueError(
                    f"Invalid dependency index {dep} for activity {i}; must be in [0, {n - 1}]."
                )

    # Forward pass
    es = [0.0] * n
    ef = [0.0] * n
    for i in range(n):
        if dependencies[i]:
            es[i] = max(ef[dep] for dep in dependencies[i])
        ef[i] = es[i] + dur_arr[i]

    project_duration = max(ef)

    # Backward pass
    lf = [project_duration] * n
    ls = [0.0] * n

    # Find successors for each activity
    successors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for dep in dependencies[i]:
            successors[dep].append(i)

    for i in range(n - 1, -1, -1):
        if successors[i]:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - dur_arr[i]

    # Compute float and identify critical path
    total_float = [ls[i] - es[i] for i in range(n)]
    critical_path = [activities[i] for i in range(n) if abs(total_float[i]) < 1e-9]

    activity_details = []
    for i in range(n):
        activity_details.append(
            {
                "name": activities[i],
                "es": float(es[i]),
                "ef": float(ef[i]),
                "ls": float(ls[i]),
                "lf": float(lf[i]),
                "float": float(total_float[i]),
            }
        )

    return {
        "project_duration": float(project_duration),
        "critical_path": critical_path,
        "activities": activity_details,
    }


# ---------------------------------------------------------------------------
# Lateral Development Schedule
# ---------------------------------------------------------------------------


def lateral_development_schedule(
    zones: list,
    footage_per_zone: list,
    monthly_advance: float,
    advance_cost_per_m: float,
) -> dict:
    """Schedule sequential lateral development for underground zones.

    Each zone is developed in sequence.  The time for each zone is
    footage / monthly_advance.

    Parameters
    ----------
    zones : list
        List of zone name strings.  Must have at least 1 element.
    footage_per_zone : list
        Development length for each zone in metres.  Each must be > 0.
        Same length as *zones*.
    monthly_advance : float
        Development advance rate in metres/month.  Must be > 0.
    advance_cost_per_m : float
        Cost per metre of development.  Must be > 0.

    Returns
    -------
    dict
        ``total_months`` : float
            Total development time.
        ``total_metres`` : float
            Total development footage.
        ``total_cost`` : float
            Total development cost.
        ``schedule`` : list of dict
            Each dict has keys: ``zone``, ``metres``, ``months``,
            ``cost``, ``start_month``, ``end_month``.

    References
    ----------
    .. [1] Hustrulid, W. & Bullock, R. (2001). *Underground Mining
           Methods*. SME. Ch. 10.
    """
    n = len(zones)
    if n == 0:
        raise ValueError("'zones' must have at least 1 element.")

    fpz_arr = validate_array(footage_per_zone, "footage_per_zone", min_length=1)

    if len(fpz_arr) != n:
        raise ValueError(
            "'zones' and 'footage_per_zone' must have the same "
            f"length, got {n} and {len(fpz_arr)}."
        )

    for i, f in enumerate(fpz_arr):
        if f <= 0:
            raise ValueError(
                f"All footage values must be positive, got footage_per_zone[{i}]={f}."
            )

    validate_positive(monthly_advance, "monthly_advance")
    validate_positive(advance_cost_per_m, "advance_cost_per_m")

    schedule = []
    cumulative_months = 0.0
    total_metres = 0.0
    total_cost = 0.0

    for i in range(n):
        metres = float(fpz_arr[i])
        months = metres / monthly_advance
        cost = metres * advance_cost_per_m
        start_month = cumulative_months
        end_month = cumulative_months + months

        schedule.append(
            {
                "zone": zones[i],
                "metres": metres,
                "months": float(months),
                "cost": float(cost),
                "start_month": float(start_month),
                "end_month": float(end_month),
            }
        )

        cumulative_months = end_month
        total_metres += metres
        total_cost += cost

    return {
        "total_months": float(cumulative_months),
        "total_metres": float(total_metres),
        "total_cost": float(total_cost),
        "schedule": schedule,
    }


# ---------------------------------------------------------------------------
# Ore Pass Capacity
# ---------------------------------------------------------------------------


def ore_pass_capacity(
    diameter: float,
    height: float,
    draw_angle: float,
    bulk_density: float,
) -> dict:
    """Estimate ore pass storage capacity (Hambley, 1987).

    live_volume = pi/4 * D^2 * (H - D / (2 * tan(draw_angle_rad)))
    dead_volume = total_volume - live_volume
    total_volume = pi/4 * D^2 * H

    Parameters
    ----------
    diameter : float
        Ore pass diameter in metres.  Must be > 0.
    height : float
        Ore pass height in metres.  Must be > 0.
    draw_angle : float
        Draw angle in degrees, in (0, 90).
    bulk_density : float
        Bulk density of the ore in tonnes/m3.  Must be > 0.

    Returns
    -------
    dict
        ``live_volume_m3`` : float
        ``dead_volume_m3`` : float
        ``total_volume_m3`` : float
        ``live_capacity_tonnes`` : float

    References
    ----------
    .. [1] Hambley, D.F. (1987). Design of ore pass systems for
           underground mines. *CIM Bulletin*, 80(897), 25--30.
    """
    validate_positive(diameter, "diameter")
    validate_positive(height, "height")
    validate_range(draw_angle, 0.1, 89.9, "draw_angle")
    validate_positive(bulk_density, "bulk_density")

    draw_angle_rad = math.radians(draw_angle)
    cross_area = math.pi / 4.0 * diameter**2
    total_volume = cross_area * height

    cone_height = diameter / (2.0 * math.tan(draw_angle_rad))
    live_volume = cross_area * (height - cone_height)
    # Ensure live volume is non-negative
    live_volume = max(live_volume, 0.0)
    dead_volume = total_volume - live_volume

    return {
        "live_volume_m3": float(live_volume),
        "dead_volume_m3": float(dead_volume),
        "total_volume_m3": float(total_volume),
        "live_capacity_tonnes": float(live_volume * bulk_density),
    }


# ---------------------------------------------------------------------------
# Shaft Hoisting Capacity
# ---------------------------------------------------------------------------


def shaft_hoisting_capacity(
    cage_capacity: float,
    cycle_time_min: float,
    operating_hours: float,
    availability: float,
) -> dict:
    """Compute shaft hoisting capacity.

    hoists_per_hour = 60 / cycle_time_min
    daily_capacity = cage_capacity * hoists_per_hour
                     * operating_hours * availability
    annual_capacity = daily_capacity * 365

    Parameters
    ----------
    cage_capacity : float
        Capacity per hoist cycle in tonnes.  Must be > 0.
    cycle_time_min : float
        Hoisting cycle time in minutes.  Must be > 0.
    operating_hours : float
        Operating hours per day.  Must be > 0 and <= 24.
    availability : float
        Mechanical availability as a fraction, in (0, 1].

    Returns
    -------
    dict
        ``hoists_per_hour`` : float
        ``daily_capacity_tonnes`` : float
        ``annual_capacity_tonnes`` : float

    References
    ----------
    .. [1] Brady, B.H.G. & Brown, E.T. (2006). *Rock Mechanics for
           Underground Mining*, 3rd ed. Springer, Ch. 15.
    """
    validate_positive(cage_capacity, "cage_capacity")
    validate_positive(cycle_time_min, "cycle_time_min")
    validate_positive(operating_hours, "operating_hours")
    validate_range(operating_hours, 0.1, 24.0, "operating_hours")
    validate_positive(availability, "availability")
    validate_range(availability, 0.01, 1.0, "availability")

    hoists_per_hour = 60.0 / cycle_time_min
    daily = cage_capacity * hoists_per_hour * operating_hours * availability
    annual = daily * 365.0

    return {
        "hoists_per_hour": float(hoists_per_hour),
        "daily_capacity_tonnes": float(daily),
        "annual_capacity_tonnes": float(annual),
    }


# ---------------------------------------------------------------------------
# Ramp Access Time
# ---------------------------------------------------------------------------


def ramp_access_time(
    ramp_length: float,
    ramp_gradient_pct: float,
    vehicle_speed_kmh: float,
) -> float:
    """Estimate travel time along a ramp with gradient adjustment.

    effective_speed = vehicle_speed_kmh * (1 - gradient_pct / 100)
    time_min = (ramp_length / 1000) / effective_speed * 60

    Parameters
    ----------
    ramp_length : float
        Ramp length in metres.  Must be > 0.
    ramp_gradient_pct : float
        Ramp gradient as a percentage, in [0, 99).
    vehicle_speed_kmh : float
        Vehicle speed on flat in km/h.  Must be > 0.

    Returns
    -------
    float
        Travel time in minutes.

    References
    ----------
    .. [1] Hustrulid, W. & Bullock, R. (2001). *Underground Mining
           Methods*. SME. Ch. 12.
    """
    validate_positive(ramp_length, "ramp_length")
    validate_range(ramp_gradient_pct, 0.0, 99.0, "ramp_gradient_pct")
    validate_positive(vehicle_speed_kmh, "vehicle_speed_kmh")

    effective_speed = vehicle_speed_kmh * (1.0 - ramp_gradient_pct / 100.0)
    if effective_speed <= 0:
        raise ValueError("Effective speed must be positive; gradient too steep.")

    ramp_km = ramp_length / 1000.0
    time_min = ramp_km / effective_speed * 60.0
    return float(time_min)


# ---------------------------------------------------------------------------
# Block Value Underground
# ---------------------------------------------------------------------------


def block_value_underground(
    tonnes: float,
    grade: float,
    nsr_per_unit: float,
    mining_cost: float,
    filling_cost: float,
    diluted_grade: float,
) -> float:
    """Compute the economic value of an underground mining block.

    value = tonnes * (diluted_grade * nsr_per_unit
                      - mining_cost - filling_cost)

    Parameters
    ----------
    tonnes : float
        Block tonnage.  Must be > 0.
    grade : float
        In-situ grade (for reference/validation).  Must be >= 0.
    nsr_per_unit : float
        Net smelter return per unit of grade.  Must be > 0.
    mining_cost : float
        Mining cost per tonne.  Must be >= 0.
    filling_cost : float
        Backfill cost per tonne.  Must be >= 0.
    diluted_grade : float
        Grade after dilution.  Must be >= 0.

    Returns
    -------
    float
        Block value in currency units.

    References
    ----------
    .. [1] Hustrulid, W. & Bullock, R. (2001). *Underground Mining
           Methods*. SME. Ch. 14.
    """
    validate_positive(tonnes, "tonnes")
    if grade < 0:
        raise ValueError(f"'grade' must be non-negative, got {grade}.")
    validate_positive(nsr_per_unit, "nsr_per_unit")
    if mining_cost < 0:
        raise ValueError(f"'mining_cost' must be non-negative, got {mining_cost}.")
    if filling_cost < 0:
        raise ValueError(f"'filling_cost' must be non-negative, got {filling_cost}.")
    if diluted_grade < 0:
        raise ValueError(f"'diluted_grade' must be non-negative, got {diluted_grade}.")

    value = tonnes * (diluted_grade * nsr_per_unit - mining_cost - filling_cost)
    return float(value)
