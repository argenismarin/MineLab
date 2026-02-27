"""JORC Code 2012 resource classification and reporting.

This module implements block-level resource classification following the
Joint Ore Reserves Committee (JORC) Code 2012, using kriging variance,
data spacing, and slope of regression as classification criteria.

References
----------
.. [1] JORC (2012). *Australasian Code for Reporting of Exploration
       Results, Mineral Resources and Ore Reserves*. Joint Ore Reserves
       Committee of The Australasian Institute of Mining and Metallurgy.
.. [2] Snowden, D.V. (2001). Practical interpretation of mineral resource
       and ore reserve classification guidelines. *Mineral Resource and
       Ore Reserve Estimation -- The AusIMM Guide to Good Practice*,
       AusIMM, pp. 643-652.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# JORC Classification
# ---------------------------------------------------------------------------


def jorc_classify(
    kriging_variance: np.ndarray,
    data_spacing: np.ndarray,
    slope_regression: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """Classify resource blocks according to JORC Code 2012 criteria.

    Each block is assigned a category based on kriging variance, data
    spacing, and slope of regression thresholds:

    - **Measured** (1): kriging variance <= kv_max AND data spacing <=
      spacing_max AND slope >= slope_min.
    - **Indicated** (2): same logic with relaxed thresholds.
    - **Inferred** (3): all remaining blocks.

    Parameters
    ----------
    kriging_variance : numpy.ndarray
        1-D array of kriging variance values per block (dimensionless or
        in grade-squared units).
    data_spacing : numpy.ndarray
        1-D array of average data spacing per block (metres).
    slope_regression : numpy.ndarray
        1-D array of conditional bias (slope of regression) values per
        block, typically in [0, 1].
    thresholds : dict
        Nested dictionary defining classification thresholds::

            {
                "measured": {
                    "kv_max": 0.2,
                    "spacing_max": 25,
                    "slope_min": 0.8
                },
                "indicated": {
                    "kv_max": 0.5,
                    "spacing_max": 50,
                    "slope_min": 0.5
                }
            }

    Returns
    -------
    numpy.ndarray
        1-D integer array of category codes: 1 = Measured, 2 = Indicated,
        3 = Inferred.

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty, or if
        required threshold keys are missing.

    Examples
    --------
    >>> import numpy as np
    >>> kv = np.array([0.1, 0.3, 0.8, 0.15, 0.6])
    >>> ds = np.array([20, 40, 80, 22, 55])
    >>> sr = np.array([0.9, 0.6, 0.3, 0.85, 0.4])
    >>> thresholds = {
    ...     "measured": {"kv_max": 0.2, "spacing_max": 25, "slope_min": 0.8},
    ...     "indicated": {"kv_max": 0.5, "spacing_max": 50, "slope_min": 0.5},
    ... }
    >>> jorc_classify(kv, ds, sr, thresholds)
    array([1, 2, 3, 1, 3])

    References
    ----------
    .. [1] JORC Code (2012), Table 1, Section 3.
    .. [2] Snowden (2001).
    """
    kv = np.asarray(kriging_variance, dtype=float).ravel()
    ds = np.asarray(data_spacing, dtype=float).ravel()
    sr = np.asarray(slope_regression, dtype=float).ravel()

    if kv.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (kv.size == ds.size == sr.size):
        raise ValueError(
            f"All input arrays must have the same length. Got "
            f"kriging_variance={kv.size}, data_spacing={ds.size}, "
            f"slope_regression={sr.size}."
        )

    for cat in ("measured", "indicated"):
        if cat not in thresholds:
            raise ValueError(f"thresholds must contain '{cat}' key.")
        for key in ("kv_max", "spacing_max", "slope_min"):
            if key not in thresholds[cat]:
                raise ValueError(f"thresholds['{cat}'] must contain '{key}' key.")

    # Start with all blocks as Inferred (3)
    classification = np.full(kv.size, 3, dtype=int)

    # Indicated (2): relaxed thresholds
    ind = thresholds["indicated"]
    indicated_mask = (kv <= ind["kv_max"]) & (ds <= ind["spacing_max"]) & (sr >= ind["slope_min"])
    classification[indicated_mask] = 2

    # Measured (1): strict thresholds (overrides Indicated)
    meas = thresholds["measured"]
    measured_mask = (
        (kv <= meas["kv_max"]) & (ds <= meas["spacing_max"]) & (sr >= meas["slope_min"])
    )
    classification[measured_mask] = 1

    return classification


# ---------------------------------------------------------------------------
# JORC Table 1 Summary
# ---------------------------------------------------------------------------


def jorc_table1(
    classification: np.ndarray,
    tonnages: np.ndarray,
    grades: np.ndarray,
    density: float = 2.7,
) -> dict:
    """Generate a JORC Table 1 resource summary.

    Aggregates tonnage, grade, and metal content by resource category
    (Measured, Indicated, Inferred).

    Parameters
    ----------
    classification : numpy.ndarray
        1-D integer array of category codes (1 = Measured, 2 = Indicated,
        3 = Inferred), as returned by :func:`jorc_classify`.
    tonnages : numpy.ndarray
        1-D array of block tonnages (in the same units, e.g. tonnes).
    grades : numpy.ndarray
        1-D array of block grades (e.g. % or g/t).
    density : float, optional
        Rock density in t/m^3 (default 2.7). Used for informational
        purposes; tonnages are assumed to already incorporate density.

    Returns
    -------
    dict
        Dictionary with keys ``"measured"``, ``"indicated"``,
        ``"inferred"``, and ``"total"``. Each contains:

        - ``"tonnes"`` : float -- total tonnage for the category.
        - ``"grade"`` : float -- tonnage-weighted average grade.
        - ``"metal"`` : float -- contained metal (tonnes * grade / 100
          if grade is in %, or tonnes * grade if grade is in fraction).

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty.

    Notes
    -----
    Metal is computed as ``tonnes * grade`` (without dividing by 100),
    allowing the caller to use whatever grade unit convention they prefer.

    Examples
    --------
    >>> import numpy as np
    >>> cls = np.array([1, 1, 2, 2, 3])
    >>> tonnes = np.array([1000, 1500, 800, 1200, 500])
    >>> grades = np.array([1.2, 1.0, 0.8, 0.7, 0.3])
    >>> result = jorc_table1(cls, tonnes, grades)
    >>> result["measured"]["tonnes"]
    2500.0

    References
    ----------
    .. [1] JORC Code (2012), Table 1.
    """
    cls = np.asarray(classification, dtype=int).ravel()
    t = np.asarray(tonnages, dtype=float).ravel()
    g = np.asarray(grades, dtype=float).ravel()

    if cls.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (cls.size == t.size == g.size):
        raise ValueError(
            f"All input arrays must have the same length. Got "
            f"classification={cls.size}, tonnages={t.size}, "
            f"grades={g.size}."
        )

    validate_positive(density, "density")

    categories = {
        "measured": 1,
        "indicated": 2,
        "inferred": 3,
    }

    result: dict = {}
    total_tonnes = 0.0
    total_metal = 0.0

    for name, code in categories.items():
        mask = cls == code
        cat_tonnes = float(t[mask].sum())
        cat_metal = float((t[mask] * g[mask]).sum())
        cat_grade = cat_metal / cat_tonnes if cat_tonnes > 0 else 0.0

        result[name] = {
            "tonnes": cat_tonnes,
            "grade": cat_grade,
            "metal": cat_metal,
        }
        total_tonnes += cat_tonnes
        total_metal += cat_metal

    result["total"] = {
        "tonnes": total_tonnes,
        "grade": total_metal / total_tonnes if total_tonnes > 0 else 0.0,
        "metal": total_metal,
    }

    return result
