"""Truck cycle time analysis for open-pit haulage operations.

This module provides functions for calculating truck cycle times, rimpull-based
speeds, and segment travel times used in mine haulage fleet analysis.

References
----------
.. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
.. [2] Hustrulid, W., Kuchta, M. & Martin, R. (2013). *Open Pit Mine Planning
       and Design*, 3rd ed. CRC Press, Ch. 7.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
)

# ---------------------------------------------------------------------------
# Truck Cycle Time
# ---------------------------------------------------------------------------


def truck_cycle_time(
    load_time: float,
    haul_segments: list[dict],
    dump_time: float,
    return_segments: list[dict],
    spot_time: float = 0.5,
    queue_time: float = 0.0,
) -> dict:
    """Compute total truck cycle time from individual components.

    The cycle is decomposed into fixed times (load, dump, spot, queue) and
    variable travel times for haul and return segments.  Each segment is
    defined by a distance and an average speed.

    Parameters
    ----------
    load_time : float
        Time to load the truck, in minutes.  Must be positive.
    haul_segments : list of dict
        Haul route segments.  Each dict must contain ``"distance"`` (m) and
        ``"speed"`` (km/h).
    dump_time : float
        Time to dump the payload, in minutes.  Must be positive.
    return_segments : list of dict
        Return route segments.  Same format as *haul_segments*.
    spot_time : float, optional
        Time to spot (position) the truck at the loader, in minutes
        (default 0.5).  Must be non-negative.
    queue_time : float, optional
        Waiting time in queue at the loader, in minutes (default 0.0).
        Must be non-negative.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"total_time"`` : float -- Total cycle time in minutes.
        - ``"haul_time"`` : float -- Total haul travel time in minutes.
        - ``"return_time"`` : float -- Total return travel time in minutes.
        - ``"fixed_time"`` : float -- Sum of load, dump, spot, and queue times
          in minutes.

    Raises
    ------
    ValueError
        If any time or segment parameter is invalid.

    Examples
    --------
    >>> haul = [{"distance": 2000, "speed": 25}]
    >>> ret = [{"distance": 2000, "speed": 40}]
    >>> result = truck_cycle_time(3.0, haul, 1.5, ret, spot_time=0.5)
    >>> round(result["total_time"], 2)
    12.8

    References
    ----------
    .. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
    """
    validate_positive(load_time, "load_time")
    validate_positive(dump_time, "dump_time")
    validate_non_negative(spot_time, "spot_time")
    validate_non_negative(queue_time, "queue_time")

    haul_time = _sum_segment_times(haul_segments, "haul_segments")
    return_time = _sum_segment_times(return_segments, "return_segments")
    fixed_time = load_time + dump_time + spot_time + queue_time
    total_time = fixed_time + haul_time + return_time

    return {
        "total_time": total_time,
        "haul_time": haul_time,
        "return_time": return_time,
        "fixed_time": fixed_time,
    }


def _sum_segment_times(segments: list[dict], name: str) -> float:
    """Sum travel times across a list of route segments.

    Parameters
    ----------
    segments : list of dict
        Each dict must have ``"distance"`` (m) and ``"speed"`` (km/h).
    name : str
        Name used in error messages.

    Returns
    -------
    float
        Total travel time in minutes.
    """
    if not segments:
        raise ValueError(f"'{name}' must contain at least one segment.")

    total = 0.0
    for i, seg in enumerate(segments):
        if "distance" not in seg or "speed" not in seg:
            raise ValueError(f"Segment {i} of '{name}' must have 'distance' and 'speed' keys.")
        validate_positive(seg["distance"], f"{name}[{i}].distance")
        validate_positive(seg["speed"], f"{name}[{i}].speed")
        # distance in m, speed in km/h -> time in minutes
        # time = distance_m / (speed_kmh * 1000 / 60)
        total += seg["distance"] / (seg["speed"] * 1000.0 / 60.0)

    return total


# ---------------------------------------------------------------------------
# Rimpull Speed
# ---------------------------------------------------------------------------


def rimpull_speed(
    rimpull_available: float,
    grade: float,
    rolling_resistance: float,
    gross_weight: float,
) -> float:
    """Estimate achievable speed from available rimpull at given resistances.

    Total resistance is the sum of grade resistance and rolling resistance,
    expressed as percentages of gross vehicle weight.  The resulting speed is
    derived from the force balance:

    .. math::

        v = \\frac{F_{rimpull}}{W \\cdot R_{total} \\cdot g}

    where *v* is in m/s, *W* is gross weight in kg, and *g* = 9.81 m/s^2.
    The result is converted to km/h.

    Parameters
    ----------
    rimpull_available : float
        Available rimpull force in kN.  Must be positive.
    grade : float
        Road grade in percent (positive = uphill, negative = downhill).
    rolling_resistance : float
        Rolling resistance in percent of gross weight (typically 2--5 %).
        Must be non-negative.
    gross_weight : float
        Gross vehicle weight (truck + payload) in tonnes.  Must be positive.

    Returns
    -------
    float
        Achievable speed in km/h.

    Raises
    ------
    ValueError
        If total resistance is non-positive (truck would accelerate freely).

    Examples
    --------
    >>> round(rimpull_speed(300.0, 8.0, 2.0, 200.0), 1)
    5.5

    References
    ----------
    .. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.,
           Ch. on rimpull and speed estimation.
    """
    validate_positive(rimpull_available, "rimpull_available")
    validate_non_negative(rolling_resistance, "rolling_resistance")
    validate_positive(gross_weight, "gross_weight")

    total_resistance_pct = grade + rolling_resistance
    if total_resistance_pct <= 0:
        raise ValueError(
            "Total resistance (grade + rolling_resistance) must be positive "
            f"for rimpull speed calculation, got {total_resistance_pct}%."
        )

    total_resistance_fraction = total_resistance_pct / 100.0
    # F = m * a  =>  v = F / (m * g * resistance_fraction)
    # gross_weight in tonnes -> kg = gross_weight * 1000
    # rimpull in kN -> N = rimpull * 1000
    g = 9.81  # m/s^2
    speed_ms = (rimpull_available * 1000.0) / (
        gross_weight * 1000.0 * total_resistance_fraction * g
    )
    speed_kmh = speed_ms * 3.6

    return speed_kmh


# ---------------------------------------------------------------------------
# Travel Time
# ---------------------------------------------------------------------------


def travel_time(
    distance: float,
    max_speed: float,
    grade: float,
    rolling_resistance: float,
) -> float:
    """Estimate time to traverse a single haul road segment.

    The effective speed is the product of the maximum speed and a speed
    factor that accounts for total resistance.  For uphill segments the speed
    is reduced proportionally to total resistance; for downhill or flat
    segments the truck can achieve the maximum speed (factor capped at 1.0).

    .. math::

        v_{eff} = v_{max} \\times \\min\\!\\left(1,\\;
        \\frac{1}{1 + R_{total}/10}\\right)

    Parameters
    ----------
    distance : float
        Segment length in metres.  Must be positive.
    max_speed : float
        Maximum truck speed in km/h.  Must be positive.
    grade : float
        Road grade in percent (positive = uphill, negative = downhill).
    rolling_resistance : float
        Rolling resistance in percent (typically 2--5 %).  Must be
        non-negative.

    Returns
    -------
    float
        Travel time for the segment in minutes.

    Examples
    --------
    >>> round(travel_time(1500, 40, 5, 3), 2)
    4.05

    References
    ----------
    .. [1] Caterpillar Inc. (2019). *Caterpillar Performance Handbook*, 49th ed.
    """
    validate_positive(distance, "distance")
    validate_positive(max_speed, "max_speed")
    validate_non_negative(rolling_resistance, "rolling_resistance")

    total_resistance = grade + rolling_resistance

    # Speed factor: reduce speed when total resistance is positive (uphill)
    speed_factor = 1.0 / (1.0 + total_resistance / 10.0) if total_resistance > 0 else 1.0

    effective_speed = max_speed * speed_factor

    # distance in m, effective_speed in km/h -> time in minutes
    time_min = (distance / 1000.0) / effective_speed * 60.0

    return time_min
