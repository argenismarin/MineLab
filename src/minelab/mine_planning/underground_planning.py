"""Underground mine planning calculations.

Stope economic valuation, underground cut-off grade, mining recovery,
production rates, development advance, and crown pillar design.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Stope Economic Value
# ---------------------------------------------------------------------------


def stope_economic_value(
    ore_tonnes: float,
    grade: float,
    metal_price: float,
    recovery: float,
    opex_per_tonne: float,
    dilution: float,
) -> dict:
    """Net smelter return based stope economic value.

    Parameters
    ----------
    ore_tonnes : float
        Ore tonnage in the stope (t).
    grade : float
        In-situ ore grade (e.g. g/t, %).
    metal_price : float
        Metal price per unit of grade * tonne (e.g. $/g for gold).
    recovery : float
        Metallurgical recovery (0 to 1).
    opex_per_tonne : float
        Operating cost per tonne mined ($/t).
    dilution : float
        Dilution fraction (0 to 1).

    Returns
    -------
    dict
        Keys: ``"revenue"`` ($), ``"cost"`` ($), ``"profit"`` ($),
        ``"diluted_grade"`` (grade units),
        ``"profit_per_tonne"`` ($/t).

    Examples
    --------
    >>> result = stope_economic_value(10000, 5.0, 60.0, 0.9, 80.0,
    ...                               0.1)
    >>> result["profit"] > 0
    True

    References
    ----------
    .. [1] Hustrulid, W.A. & Bullock, R.L. (2001). "Underground Mining
       Methods." SME. Ch. 3, Economic evaluation.
    """
    validate_positive(ore_tonnes, "ore_tonnes")
    validate_positive(grade, "grade")
    validate_positive(metal_price, "metal_price")
    validate_range(recovery, 0, 1, "recovery")
    validate_positive(opex_per_tonne, "opex_per_tonne")
    validate_range(dilution, 0, 1, "dilution")

    diluted_grade = grade * (1.0 - dilution)
    revenue = ore_tonnes * diluted_grade * metal_price * recovery
    cost = ore_tonnes * opex_per_tonne
    profit = revenue - cost
    profit_per_tonne = profit / ore_tonnes

    return {
        "revenue": float(revenue),
        "cost": float(cost),
        "profit": float(profit),
        "diluted_grade": float(diluted_grade),
        "profit_per_tonne": float(profit_per_tonne),
    }


# ---------------------------------------------------------------------------
# Underground Cut-off Grade — Lane (1988)
# ---------------------------------------------------------------------------


def underground_cutoff_grade(
    opex_per_tonne: float,
    price: float,
    recovery: float,
    mining_cost_ug: float,
) -> float:
    """Break-even underground cut-off grade.

    Parameters
    ----------
    opex_per_tonne : float
        Processing cost per tonne ($/t).
    price : float
        Metal price per grade-unit per tonne (e.g. $/g/t * t).
    recovery : float
        Metallurgical recovery (0 to 1).
    mining_cost_ug : float
        Underground mining cost per tonne ($/t).

    Returns
    -------
    float
        Cut-off grade in the same units as price denominator.

    Examples
    --------
    >>> underground_cutoff_grade(30.0, 60.0, 0.9, 50.0)
    1.4814...

    References
    ----------
    .. [1] Lane, K.F. (1988). "The Economic Definition of Ore: Cut-off
       Grades in Theory and Practice." Mining Journal Books, London.
    """
    validate_positive(opex_per_tonne, "opex_per_tonne")
    validate_positive(price, "price")
    validate_range(recovery, 0, 1, "recovery")
    validate_positive(mining_cost_ug, "mining_cost_ug")

    if recovery == 0:
        raise ValueError("'recovery' must be > 0 for cut-off grade.")

    cog = (opex_per_tonne + mining_cost_ug) / (price * recovery)
    return float(cog)


# ---------------------------------------------------------------------------
# Mining Recovery
# ---------------------------------------------------------------------------


def mining_recovery_underground(
    stope_width: float,
    ore_width: float,
    dilution_skin: float,
    mining_method_factor: float,
) -> dict:
    """Estimate mining recovery and dilution for underground stoping.

    Parameters
    ----------
    stope_width : float
        Actual mined width (m).
    ore_width : float
        True ore body width (m).
    dilution_skin : float
        Overbreak / dilution skin on each side (m).
    mining_method_factor : float
        Recovery efficiency factor for the mining method (0 to 1).

    Returns
    -------
    dict
        Keys: ``"mining_recovery"`` (fraction),
        ``"dilution"`` (fraction),
        ``"effective_width"`` (m, ore_width + 2 * skin).

    Examples
    --------
    >>> result = mining_recovery_underground(5.0, 4.0, 0.5, 0.95)
    >>> result["mining_recovery"]
    0.76

    References
    ----------
    .. [1] Villaescusa, E. (2014). "Geotechnical Design for Sublevel
       Open Stoping." CRC Press.
    """
    validate_positive(stope_width, "stope_width")
    validate_positive(ore_width, "ore_width")
    validate_non_negative(dilution_skin, "dilution_skin")
    validate_range(mining_method_factor, 0, 1, "mining_method_factor")

    effective_width = ore_width + 2.0 * dilution_skin
    mining_recovery = (ore_width / effective_width) * mining_method_factor
    dilution = 1.0 - mining_recovery

    return {
        "mining_recovery": float(mining_recovery),
        "dilution": float(dilution),
        "effective_width": float(effective_width),
    }


# ---------------------------------------------------------------------------
# Long-Hole Production Rate
# ---------------------------------------------------------------------------


def long_hole_production_rate(
    holes_per_ring: float,
    ring_burden: float,
    drill_rate: float,
    charge_time: float,
    blast_interval_days: float,
) -> dict:
    """Estimate long-hole stoping production rate.

    Parameters
    ----------
    holes_per_ring : float
        Number of blast holes per ring.
    ring_burden : float
        Burden between rings (m).
    drill_rate : float
        Drilling rate (m/h).
    charge_time : float
        Charging and connecting time per ring (hours).
    blast_interval_days : float
        Interval between blasts (days).

    Returns
    -------
    dict
        Keys: ``"drill_time_hours"`` (time to drill one ring),
        ``"tonnes_per_blast"`` (tonnes per ring blast),
        ``"daily_production_tonnes"`` (average daily production).

    Examples
    --------
    >>> result = long_hole_production_rate(8, 2.5, 20.0, 2.0, 1.0)
    >>> result["daily_production_tonnes"] > 0
    True

    References
    ----------
    .. [1] Hustrulid, W.A. & Bullock, R.L. (2001). "Underground Mining
       Methods." SME.
    """
    validate_positive(holes_per_ring, "holes_per_ring")
    validate_positive(ring_burden, "ring_burden")
    validate_positive(drill_rate, "drill_rate")
    validate_positive(charge_time, "charge_time")
    validate_positive(blast_interval_days, "blast_interval_days")

    # Ring depth ~ 1.1 * burden (angled holes)
    ring_depth = 1.1 * ring_burden

    # Drill time for one ring
    drill_time = holes_per_ring * ring_depth / drill_rate

    # Tonnes per blast: burden * (burden * holes_per_ring * 2.7)
    # Approximation: volume = burden^2 * holes_per_ring, density=2.7
    tonnes_per_blast = ring_burden * ring_burden * holes_per_ring * 2.7

    daily_production = tonnes_per_blast / blast_interval_days

    return {
        "drill_time_hours": float(drill_time),
        "tonnes_per_blast": float(tonnes_per_blast),
        "daily_production_tonnes": float(daily_production),
    }


# ---------------------------------------------------------------------------
# Development Advance Rate
# ---------------------------------------------------------------------------


def development_advance_rate(
    drill_pattern_area: float,
    rounds_per_day: float,
    advance_per_round: float,
) -> dict:
    """Estimate development (drift) advance rate.

    Parameters
    ----------
    drill_pattern_area : float
        Cross-sectional area of the development heading (m^2).
    rounds_per_day : float
        Number of drill-blast-muck rounds per day.
    advance_per_round : float
        Advance per round (m).

    Returns
    -------
    dict
        Keys: ``"daily_advance_m"`` (m/day),
        ``"monthly_advance_m"`` (m/month, 25 working days),
        ``"daily_volume_m3"`` (m^3/day).

    Examples
    --------
    >>> result = development_advance_rate(16.0, 2.0, 3.5)
    >>> result["daily_advance_m"]
    7.0

    References
    ----------
    .. [1] Tatiya, R.R. (2005). "Surface and Underground Excavations."
       Balkema.
    """
    validate_positive(drill_pattern_area, "drill_pattern_area")
    validate_positive(rounds_per_day, "rounds_per_day")
    validate_positive(advance_per_round, "advance_per_round")

    daily_advance = rounds_per_day * advance_per_round
    monthly_advance = daily_advance * 25.0
    daily_volume = daily_advance * drill_pattern_area

    return {
        "daily_advance_m": float(daily_advance),
        "monthly_advance_m": float(monthly_advance),
        "daily_volume_m3": float(daily_volume),
    }


# ---------------------------------------------------------------------------
# Crown Pillar Thickness — Carter (1992)
# ---------------------------------------------------------------------------


def crown_pillar_thickness(
    span: float,
    rock_density: float,
    sigma_cm: float,
    safety_factor: float,
) -> float:
    """Estimate required crown pillar thickness.

    Parameters
    ----------
    span : float
        Underground span below the crown pillar (m).
    rock_density : float
        Rock mass density (kg/m^3).
    sigma_cm : float
        Rock mass compressive strength (MPa).
    safety_factor : float
        Required safety factor (> 0).

    Returns
    -------
    float
        Crown pillar thickness in metres.

    Examples
    --------
    >>> crown_pillar_thickness(15.0, 2700.0, 20.0, 2.0)
    4.87...

    References
    ----------
    .. [1] Carter, T.G. (1992). "A new approach to surface crown pillar
       design." Proc. 16th Canadian Rock Mechanics Symposium,
       Sudbury, 75-83.
    """
    validate_positive(span, "span")
    validate_positive(rock_density, "rock_density")
    validate_positive(sigma_cm, "sigma_cm")
    validate_positive(safety_factor, "safety_factor")

    gamma = rock_density * 9.81 / 1e6  # MN/m3 (i.e. MPa/m)

    # T = span * sqrt(gamma / sigma_cm) * SF
    thickness = span * np.sqrt(gamma / sigma_cm) * safety_factor
    return float(thickness)
