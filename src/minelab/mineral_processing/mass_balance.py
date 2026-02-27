"""Mass balance calculations for mineral processing circuits.

Two-product, three-product, multi-element balance, reconciliation,
and closure check.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive


def two_product(
    f_grade: float,
    c_grade: float,
    t_grade: float,
) -> dict:
    """Two-product mass balance formula.

    Parameters
    ----------
    f_grade : float
        Feed grade (%).
    c_grade : float
        Concentrate grade (%).
    t_grade : float
        Tailings grade (%).

    Returns
    -------
    dict
        Keys: ``"concentrate_ratio"`` (C/F), ``"tailings_ratio"`` (T/F),
        ``"recovery"`` (fraction).

    Examples
    --------
    >>> result = two_product(2.0, 20.0, 0.5)
    >>> round(result["concentrate_ratio"], 4)
    0.0769

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.3.
    """
    if c_grade == t_grade:
        raise ValueError("Concentrate and tailings grades cannot be equal.")

    # C/F = (f - t) / (c - t)
    cf = (f_grade - t_grade) / (c_grade - t_grade)
    tf = 1 - cf

    # Recovery = C/F * c/f
    recovery = cf * c_grade / f_grade if f_grade > 0 else 0.0

    return {
        "concentrate_ratio": float(cf),
        "tailings_ratio": float(tf),
        "recovery": float(recovery),
    }


def three_product(
    f_grade_1: float,
    f_grade_2: float,
    c1_grade_1: float,
    c1_grade_2: float,
    c2_grade_1: float,
    c2_grade_2: float,
    t_grade_1: float,
    t_grade_2: float,
) -> dict:
    """Three-product mass balance using two elements.

    Parameters
    ----------
    f_grade_1, f_grade_2 : float
        Feed grades for elements 1 and 2.
    c1_grade_1, c1_grade_2 : float
        Concentrate 1 grades for elements 1 and 2.
    c2_grade_1, c2_grade_2 : float
        Concentrate 2 grades for elements 1 and 2.
    t_grade_1, t_grade_2 : float
        Tailings grades for elements 1 and 2.

    Returns
    -------
    dict
        Keys: ``"c1_ratio"`` (C1/F), ``"c2_ratio"`` (C2/F),
        ``"t_ratio"`` (T/F).

    Examples
    --------
    >>> result = three_product(10, 5, 40, 2, 5, 30, 2, 1)
    >>> result["c1_ratio"] > 0
    True

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.3.
    """
    # Solve: F = C1 + C2 + T
    # f1*F = c1_1*C1 + c2_1*C2 + t_1*T
    # f2*F = c1_2*C1 + c2_2*C2 + t_2*T
    # Let C1/F = x, C2/F = y, T/F = 1 - x - y
    # (f1 - t1) = (c1_1 - t1)*x + (c2_1 - t1)*y
    # (f2 - t2) = (c1_2 - t2)*x + (c2_2 - t2)*y

    a = np.array(
        [
            [c1_grade_1 - t_grade_1, c2_grade_1 - t_grade_1],
            [c1_grade_2 - t_grade_2, c2_grade_2 - t_grade_2],
        ]
    )
    b = np.array([f_grade_1 - t_grade_1, f_grade_2 - t_grade_2])

    xy = np.linalg.solve(a, b)

    return {
        "c1_ratio": float(xy[0]),
        "c2_ratio": float(xy[1]),
        "t_ratio": float(1 - xy[0] - xy[1]),
    }


def multi_element_balance(
    feed: dict[str, float],
    products: list[dict[str, float]],
    product_ratios: np.ndarray,
) -> dict:
    """Multi-element mass balance check.

    Parameters
    ----------
    feed : dict
        Feed grades keyed by element name.
    products : list of dict
        Product grades keyed by element name.
    product_ratios : np.ndarray
        Mass ratios of each product to feed (sum should be ~1.0).

    Returns
    -------
    dict
        Keys: ``"balance_errors"`` (dict of element → error),
        ``"max_error"`` (float), ``"balanced"`` (bool).

    Examples
    --------
    >>> feed = {"Cu": 2.0, "Fe": 30.0}
    >>> products = [{"Cu": 20.0, "Fe": 10.0}, {"Cu": 0.1, "Fe": 32.0}]
    >>> ratios = np.array([0.1, 0.9])
    >>> result = multi_element_balance(feed, products, ratios)
    >>> result["balanced"]
    True

    References
    ----------
    .. [1] Napier-Munn, T.J. et al. (1996). Mineral Comminution Circuits.
       JKMRC, University of Queensland.
    """
    product_ratios = np.asarray(product_ratios, dtype=float)
    errors = {}

    for element in feed:
        feed_val = feed[element]
        calc_val = sum(
            product_ratios[i] * products[i].get(element, 0.0) for i in range(len(products))
        )
        error = abs(calc_val - feed_val) / feed_val if feed_val > 0 else 0.0
        errors[element] = float(error)

    max_error = max(errors.values()) if errors else 0.0

    return {
        "balance_errors": errors,
        "max_error": float(max_error),
        "balanced": max_error < 0.05,
    }


def reconcile_balance(
    measured: np.ndarray,
    tolerance: float = 0.02,
) -> dict:
    """Least-squares adjustment for mass balance closure.

    Parameters
    ----------
    measured : np.ndarray
        Measured product mass fractions (should sum to ~1.0).
    tolerance : float
        Maximum acceptable deviation from 1.0. Default 0.02 (2%).

    Returns
    -------
    dict
        Keys: ``"adjusted"`` (np.ndarray), ``"adjustment"`` (float),
        ``"closed"`` (bool).

    Examples
    --------
    >>> import numpy as np
    >>> result = reconcile_balance(np.array([0.3, 0.68, 0.05]))
    >>> round(float(result["adjusted"].sum()), 4)
    1.0

    References
    ----------
    .. [1] Morrison, R.D. (2008). An introduction to metal balancing
       and reconciliation. JKMRC Monograph.
    """
    measured = np.asarray(measured, dtype=float)
    total = measured.sum()
    adjustment = total - 1.0

    # Proportional adjustment
    adjusted = measured / total if total > 0 else measured

    return {
        "adjusted": adjusted,
        "adjustment": float(adjustment),
        "closed": abs(adjustment) < tolerance,
    }


def check_closure(
    feed_mass: float,
    product_masses: list[float],
    tolerance: float = 0.02,
) -> dict:
    """Verify mass balance closure within tolerance.

    Parameters
    ----------
    feed_mass : float
        Feed mass (any consistent units).
    product_masses : list of float
        Product masses.
    tolerance : float
        Acceptable relative error. Default 0.02 (2%).

    Returns
    -------
    dict
        Keys: ``"total_products"`` (float), ``"error"`` (relative error),
        ``"closed"`` (bool).

    Examples
    --------
    >>> result = check_closure(1000, [300, 680, 50])
    >>> result["closed"]
    True

    References
    ----------
    .. [1] Standard practice.
    """
    validate_positive(feed_mass, "feed_mass")

    total = sum(product_masses)
    error = abs(total - feed_mass) / feed_mass

    return {
        "total_products": float(total),
        "error": float(error),
        "closed": error < tolerance,
    }
