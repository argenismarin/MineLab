"""Mine-to-mill reconciliation: F-factors and variance analysis.

This module implements reconciliation factors (F1, F2, F3) as defined by
Morley (2003) for tracking discrepancies between the resource model,
mine production, and plant feed, as well as variance decomposition of
metal production.

References
----------
.. [1] Morley, C. (2003). Beyond reconciliation -- A proactive approach to
       using mining data. *Proceedings of the 5th Large Open Pit Mining
       Conference*, AusIMM, pp. 185-192.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# F-Factors
# ---------------------------------------------------------------------------


def f_factors(
    model_tonnes: float,
    model_grade: float,
    mined_tonnes: float,
    mined_grade: float,
    plant_tonnes: float,
    plant_grade: float,
) -> dict:
    """Compute reconciliation F-factors (F1, F2, F3).

    The F-factors quantify discrepancies between the resource model,
    mine production, and plant feed:

    - **F1** = mined / model (model accuracy).
    - **F2** = plant / mined (mining dilution and ore loss).
    - **F3** = plant / model (overall reconciliation = F1 * F2).

    Each factor is computed for both tonnes and grade, plus a combined
    metal factor (tonnes x grade).

    Parameters
    ----------
    model_tonnes : float
        Tonnage predicted by the resource model. Must be positive.
    model_grade : float
        Grade predicted by the resource model. Must be positive.
    mined_tonnes : float
        Tonnage actually mined. Must be positive.
    mined_grade : float
        Grade of mined material. Must be positive.
    plant_tonnes : float
        Tonnage delivered to the plant. Must be positive.
    plant_grade : float
        Grade of plant feed. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"F1_tonnes"`` : float -- mined_tonnes / model_tonnes.
        - ``"F1_grade"`` : float -- mined_grade / model_grade.
        - ``"F1_metal"`` : float -- F1_tonnes * F1_grade.
        - ``"F2_tonnes"`` : float -- plant_tonnes / mined_tonnes.
        - ``"F2_grade"`` : float -- plant_grade / mined_grade.
        - ``"F2_metal"`` : float -- F2_tonnes * F2_grade.
        - ``"F3_tonnes"`` : float -- plant_tonnes / model_tonnes.
        - ``"F3_grade"`` : float -- plant_grade / model_grade.
        - ``"F3_metal"`` : float -- F3_tonnes * F3_grade.

    Raises
    ------
    ValueError
        If any input is not positive.

    Examples
    --------
    >>> result = f_factors(1000, 1.5, 1050, 1.4, 980, 1.35)
    >>> round(result["F1_tonnes"], 3)
    1.05
    >>> round(result["F2_grade"], 4)
    0.9643

    References
    ----------
    .. [1] Morley (2003), Table 1.
    """
    validate_positive(model_tonnes, "model_tonnes")
    validate_positive(model_grade, "model_grade")
    validate_positive(mined_tonnes, "mined_tonnes")
    validate_positive(mined_grade, "mined_grade")
    validate_positive(plant_tonnes, "plant_tonnes")
    validate_positive(plant_grade, "plant_grade")

    f1_t = mined_tonnes / model_tonnes
    f1_g = mined_grade / model_grade
    f2_t = plant_tonnes / mined_tonnes
    f2_g = plant_grade / mined_grade
    f3_t = plant_tonnes / model_tonnes
    f3_g = plant_grade / model_grade

    return {
        "F1_tonnes": float(f1_t),
        "F1_grade": float(f1_g),
        "F1_metal": float(f1_t * f1_g),
        "F2_tonnes": float(f2_t),
        "F2_grade": float(f2_g),
        "F2_metal": float(f2_t * f2_g),
        "F3_tonnes": float(f3_t),
        "F3_grade": float(f3_g),
        "F3_metal": float(f3_t * f3_g),
    }


# ---------------------------------------------------------------------------
# Multi-period Reconciliation Report
# ---------------------------------------------------------------------------


def reconciliation_report(periods_data: list[dict]) -> dict:
    """Generate a multi-period reconciliation report with F-factors.

    Computes F1, F2, and F3 factors for each period and provides overall
    averages across all periods.

    Parameters
    ----------
    periods_data : list[dict]
        List of period data dictionaries. Each must have keys:

        - ``"model_tonnes"`` : float
        - ``"model_grade"`` : float
        - ``"mined_tonnes"`` : float
        - ``"mined_grade"`` : float
        - ``"plant_tonnes"`` : float
        - ``"plant_grade"`` : float

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"periods"`` : list[dict] -- F-factors for each period (output
          of :func:`f_factors`).
        - ``"averages"`` : dict -- mean of each F-factor across all periods.

    Raises
    ------
    ValueError
        If *periods_data* is empty or any period has invalid data.

    Examples
    --------
    >>> data = [
    ...     {"model_tonnes": 1000, "model_grade": 1.5,
    ...      "mined_tonnes": 1050, "mined_grade": 1.4,
    ...      "plant_tonnes": 980, "plant_grade": 1.35},
    ...     {"model_tonnes": 1200, "model_grade": 1.3,
    ...      "mined_tonnes": 1180, "mined_grade": 1.25,
    ...      "plant_tonnes": 1100, "plant_grade": 1.20},
    ... ]
    >>> report = reconciliation_report(data)
    >>> len(report["periods"])
    2

    References
    ----------
    .. [1] Morley (2003).
    """
    if not periods_data:
        raise ValueError("periods_data must not be empty.")

    periods: list[dict] = []
    for p in periods_data:
        factors = f_factors(
            model_tonnes=p["model_tonnes"],
            model_grade=p["model_grade"],
            mined_tonnes=p["mined_tonnes"],
            mined_grade=p["mined_grade"],
            plant_tonnes=p["plant_tonnes"],
            plant_grade=p["plant_grade"],
        )
        periods.append(factors)

    # Compute averages across all periods
    keys = periods[0].keys()
    averages = {k: float(np.mean([p[k] for p in periods])) for k in keys}

    return {
        "periods": periods,
        "averages": averages,
    }


# ---------------------------------------------------------------------------
# Variance Analysis
# ---------------------------------------------------------------------------


def variance_analysis(
    planned_tonnes: float,
    planned_grade: float,
    actual_tonnes: float,
    actual_grade: float,
) -> dict:
    """Decompose metal production variance into component effects.

    Metal production is the product of tonnes and grade. The total
    variance in metal is decomposed into three components:

    - **Tonnage effect**: change in tonnes at the planned grade.
    - **Grade effect**: change in grade at the planned tonnage.
    - **Combined effect**: interaction of tonnage and grade changes.

    .. math::

        \\Delta M = \\Delta T \\cdot g_p + t_p \\cdot \\Delta G
                  + \\Delta T \\cdot \\Delta G

    Parameters
    ----------
    planned_tonnes : float
        Planned tonnage. Must be positive.
    planned_grade : float
        Planned grade. Must be positive.
    actual_tonnes : float
        Actual tonnage. Must be positive.
    actual_grade : float
        Actual grade. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"planned_metal"`` : float -- planned_tonnes * planned_grade.
        - ``"actual_metal"`` : float -- actual_tonnes * actual_grade.
        - ``"tonnage_effect"`` : float -- delta_T * planned_grade.
        - ``"grade_effect"`` : float -- planned_tonnes * delta_G.
        - ``"combined_effect"`` : float -- delta_T * delta_G.
        - ``"total_variance"`` : float -- sum of all effects.

    Raises
    ------
    ValueError
        If any input is not positive.

    Examples
    --------
    >>> result = variance_analysis(1000, 1.5, 1050, 1.4)
    >>> result["tonnage_effect"]
    75.0
    >>> result["grade_effect"]
    -100.0
    >>> result["combined_effect"]
    -5.0
    >>> result["total_variance"]
    -30.0

    References
    ----------
    .. [1] Standard variance decomposition in mine reconciliation.
    """
    validate_positive(planned_tonnes, "planned_tonnes")
    validate_positive(planned_grade, "planned_grade")
    validate_positive(actual_tonnes, "actual_tonnes")
    validate_positive(actual_grade, "actual_grade")

    delta_t = actual_tonnes - planned_tonnes
    delta_g = actual_grade - planned_grade

    tonnage_effect = delta_t * planned_grade
    grade_effect = planned_tonnes * delta_g
    combined_effect = delta_t * delta_g
    total_variance = tonnage_effect + grade_effect + combined_effect

    return {
        "planned_metal": float(planned_tonnes * planned_grade),
        "actual_metal": float(actual_tonnes * actual_grade),
        "tonnage_effect": float(tonnage_effect),
        "grade_effect": float(grade_effect),
        "combined_effect": float(combined_effect),
        "total_variance": float(total_variance),
    }
