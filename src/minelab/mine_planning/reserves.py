"""Mineral resource to reserve conversion and reconciliation.

Implements modifying-factor adjustments (dilution, ore loss, mining
recovery) to convert resources into mineable reserves, plus functions
for reconciling planned vs. actual production.

References
----------
.. [1] JORC Code (2012). Australasian Code for Reporting of Exploration
       Results, Mineral Resources and Ore Reserves. Joint Ore Reserves
       Committee.
"""

from __future__ import annotations

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)


def resource_to_reserve(
    resource_tonnes: float,
    resource_grade: float,
    dilution: float,
    ore_loss: float,
    mining_recovery: float = 0.95,
) -> dict:
    """Convert a mineral resource into a mineable reserve estimate.

    Applies modifying factors for mining recovery, dilution, and ore
    loss:

    .. math::

        T_{reserve} = T_{resource} \\cdot R_m \\cdot (1 + d)

    .. math::

        g_{reserve} = g_{resource} \\cdot \\frac{R_m}{1 + d}

    where *R_m* is mining recovery (fraction), *d* is dilution
    (fraction), and ore loss is accounted for via *R_m*.

    The actual mining recovery applied is ``mining_recovery * (1 - ore_loss)``.

    Parameters
    ----------
    resource_tonnes : float
        In-situ resource tonnage.  Must be positive.
    resource_grade : float
        In-situ resource grade (fraction or %).  Non-negative.
    dilution : float
        External dilution as a fraction (e.g. 0.10 for 10 %).
        Non-negative.
    ore_loss : float
        Ore loss as a fraction in [0, 1).  E.g. 0.05 for 5 %.
    mining_recovery : float, optional
        Mining recovery as a fraction in (0, 1].  Default 0.95.

    Returns
    -------
    dict
        ``"reserve_tonnes"`` : float
            Estimated mineable reserve tonnage.
        ``"reserve_grade"`` : float
            Diluted reserve grade.
        ``"metal_content"`` : float
            Total contained metal (``reserve_tonnes * reserve_grade``).
        ``"modifying_factors"`` : dict
            Summary of applied factors.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    >>> res = resource_to_reserve(1_000_000, 0.01, 0.10, 0.05, 0.95)
    >>> round(res["reserve_tonnes"], 0)
    992750.0
    >>> round(res["reserve_grade"], 6)
    0.008205

    References
    ----------
    .. [1] JORC Code (2012). Australasian Code for Reporting of
           Exploration Results, Mineral Resources and Ore Reserves.
    """
    validate_positive(resource_tonnes, "resource_tonnes")
    validate_non_negative(resource_grade, "resource_grade")
    validate_non_negative(dilution, "dilution")
    validate_range(ore_loss, 0.0, 0.999, "ore_loss")
    validate_range(mining_recovery, 0.001, 1.0, "mining_recovery")

    effective_recovery = mining_recovery * (1.0 - ore_loss)
    reserve_tonnes = resource_tonnes * effective_recovery * (1.0 + dilution)
    reserve_grade = resource_grade * effective_recovery / (1.0 + dilution)
    metal_content = reserve_tonnes * reserve_grade

    return {
        "reserve_tonnes": reserve_tonnes,
        "reserve_grade": reserve_grade,
        "metal_content": metal_content,
        "modifying_factors": {
            "mining_recovery": mining_recovery,
            "ore_loss": ore_loss,
            "dilution": dilution,
            "effective_recovery": effective_recovery,
        },
    }


def dilution_ore_loss(
    planned_tonnes: float,
    planned_grade: float,
    actual_tonnes: float,
    actual_grade: float,
) -> dict:
    """Reconcile planned versus actual production for dilution and ore loss.

    If actual tonnes exceed planned tonnes, the excess is classified as
    dilution.  If actual tonnes are less, the shortfall is ore loss.

    .. math::

        \\text{Dilution \\%} = \\frac{T_{actual} - T_{planned}}{T_{planned}}
        \\times 100 \\quad (T_{actual} > T_{planned})

    .. math::

        \\text{Ore loss \\%} = \\frac{T_{planned} - T_{actual}}{T_{planned}}
        \\times 100 \\quad (T_{actual} < T_{planned})

    Parameters
    ----------
    planned_tonnes : float
        Planned ore tonnage.  Must be positive.
    planned_grade : float
        Planned ore grade (fraction or %).  Non-negative.
    actual_tonnes : float
        Actual mined ore tonnage.  Must be positive.
    actual_grade : float
        Actual mined ore grade.  Non-negative.

    Returns
    -------
    dict
        ``"dilution_pct"`` : float
            Dilution as a percentage (0 if no dilution).
        ``"ore_loss_pct"`` : float
            Ore loss as a percentage (0 if no loss).
        ``"metal_variance"`` : float
            Difference between actual and planned metal content
            (actual - planned).  Positive means more metal recovered
            than planned.
        ``"planned_metal"`` : float
            Planned metal content.
        ``"actual_metal"`` : float
            Actual metal content.

    Raises
    ------
    ValueError
        If tonnages are non-positive or grades are negative.

    Examples
    --------
    Actual tonnes 10 % higher than planned (dilution case):

    >>> res = dilution_ore_loss(100_000, 0.01, 110_000, 0.008)
    >>> round(res["dilution_pct"], 1)
    10.0
    >>> res["ore_loss_pct"]
    0.0

    Actual tonnes 5 % lower than planned (ore loss case):

    >>> res = dilution_ore_loss(100_000, 0.01, 95_000, 0.011)
    >>> round(res["ore_loss_pct"], 1)
    5.0
    >>> res["dilution_pct"]
    0.0

    References
    ----------
    .. [1] Standard mining reconciliation practice. See also JORC
           Code (2012) Table 1, Section 4.
    """
    validate_positive(planned_tonnes, "planned_tonnes")
    validate_non_negative(planned_grade, "planned_grade")
    validate_positive(actual_tonnes, "actual_tonnes")
    validate_non_negative(actual_grade, "actual_grade")

    planned_metal = planned_tonnes * planned_grade
    actual_metal = actual_tonnes * actual_grade
    metal_variance = actual_metal - planned_metal

    if actual_tonnes > planned_tonnes:
        dilution_pct = (actual_tonnes - planned_tonnes) / planned_tonnes * 100.0
        ore_loss_pct = 0.0
    elif actual_tonnes < planned_tonnes:
        dilution_pct = 0.0
        ore_loss_pct = (planned_tonnes - actual_tonnes) / planned_tonnes * 100.0
    else:
        dilution_pct = 0.0
        ore_loss_pct = 0.0

    return {
        "dilution_pct": dilution_pct,
        "ore_loss_pct": ore_loss_pct,
        "metal_variance": metal_variance,
        "planned_metal": planned_metal,
        "actual_metal": actual_metal,
    }
