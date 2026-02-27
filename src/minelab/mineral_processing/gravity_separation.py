"""Gravity separation calculations.

Concentration criterion, Humphreys spiral, dense media separation,
and shaking table efficiency.
"""

from __future__ import annotations

from minelab.utilities.validators import validate_positive, validate_range


def concentration_criterion(
    sg_heavy: float,
    sg_light: float,
    sg_fluid: float = 1.0,
) -> dict:
    """Concentration criterion for gravity separation feasibility.

    Parameters
    ----------
    sg_heavy : float
        Specific gravity of the heavy (valuable) mineral.
    sg_light : float
        Specific gravity of the light (gangue) mineral.
    sg_fluid : float
        Specific gravity of the separating fluid. Default 1.0 (water).

    Returns
    -------
    dict
        Keys: ``"cc"`` (concentration criterion), ``"feasibility"`` (str).

    Examples
    --------
    >>> result = concentration_criterion(19.3, 2.65, 1.0)
    >>> round(result["cc"], 1)
    11.1

    References
    ----------
    .. [1] Taggart, A.F. (1945). Handbook of Mineral Dressing.
    .. [2] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.10.
    """
    validate_positive(sg_heavy, "sg_heavy")
    validate_positive(sg_light, "sg_light")

    # CC = (rho_h - rho_f) / (rho_l - rho_f)
    cc = (sg_heavy - sg_fluid) / (sg_light - sg_fluid)

    if cc > 2.5:
        feasibility = "Easy separation"
    elif cc > 1.75:
        feasibility = "Possible down to 75 micrometers"
    elif cc > 1.25:
        feasibility = "Possible down to 6 mm"
    else:
        feasibility = "Very difficult or impossible"

    return {"cc": float(cc), "feasibility": feasibility}


def humphreys_spiral_recovery(
    cc: float,
    feed_grade: float,
) -> dict:
    """Empirical Humphreys spiral recovery estimate.

    Parameters
    ----------
    cc : float
        Concentration criterion.
    feed_grade : float
        Feed grade of valuable mineral (fraction, 0-1).

    Returns
    -------
    dict
        Keys: ``"estimated_recovery"`` (fraction),
        ``"product_grade_factor"`` (upgrade ratio).

    Examples
    --------
    >>> result = humphreys_spiral_recovery(5.0, 0.05)
    >>> result["estimated_recovery"] > 0
    True

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed.
    """
    validate_positive(cc, "cc")
    validate_range(feed_grade, 0, 1, "feed_grade")

    # Empirical estimate: higher CC → higher recovery
    if cc > 2.5:
        recovery = min(0.95, 0.7 + 0.05 * cc)
    elif cc > 1.75:
        recovery = 0.5 + 0.1 * cc
    else:
        recovery = max(0.1, 0.3 * cc)

    # Upgrade ratio depends on recovery and feed grade
    upgrade = 1 / feed_grade * recovery if feed_grade > 0 else 1.0

    return {
        "estimated_recovery": float(recovery),
        "product_grade_factor": float(upgrade),
    }


def dms_cutpoint(
    medium_density: float,
    particle_sg: float,
) -> dict:
    """Dense media separation cut-point analysis.

    Parameters
    ----------
    medium_density : float
        Dense medium density (kg/m^3 or SG).
    particle_sg : float
        Particle specific gravity.

    Returns
    -------
    dict
        Keys: ``"reports_to"`` (str, "sink" or "float"),
        ``"density_difference"`` (SG units).

    Examples
    --------
    >>> result = dms_cutpoint(2.8, 3.5)
    >>> result["reports_to"]
    'sink'

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed.
    """
    validate_positive(medium_density, "medium_density")
    validate_positive(particle_sg, "particle_sg")

    diff = particle_sg - medium_density
    reports_to = "sink" if diff > 0 else "float"

    return {
        "reports_to": reports_to,
        "density_difference": float(diff),
    }


def shaking_table_efficiency(
    feed_grade: float,
    conc_grade: float,
    tail_grade: float,
) -> float:
    """Shaking table separation efficiency.

    Parameters
    ----------
    feed_grade : float
        Feed grade (fraction, 0-1).
    conc_grade : float
        Concentrate grade (fraction, 0-1).
    tail_grade : float
        Tailings grade (fraction, 0-1).

    Returns
    -------
    float
        Separation efficiency (percentage, 0-100).

    Examples
    --------
    >>> round(shaking_table_efficiency(0.05, 0.40, 0.01), 1)
    88.5

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed.
    """
    validate_range(feed_grade, 0, 1, "feed_grade")
    validate_range(conc_grade, 0, 1, "conc_grade")
    validate_range(tail_grade, 0, 1, "tail_grade")

    # E = c*(f-t) / (f*(c-t)) * 100
    if conc_grade == tail_grade or feed_grade == 0:
        return 0.0

    e = conc_grade * (feed_grade - tail_grade) / (feed_grade * (conc_grade - tail_grade)) * 100

    return float(e)
