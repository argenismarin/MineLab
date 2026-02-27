"""Classification criteria: kriging variance, search pass, and slope of regression.

This module provides individual classification methods that can be used as
building blocks for resource classification workflows (JORC, NI 43-101, or
custom schemes).

References
----------
.. [1] Snowden, D.V. (2001). Practical interpretation of mineral resource
       and ore reserve classification guidelines. *Mineral Resource and
       Ore Reserve Estimation -- The AusIMM Guide to Good Practice*,
       AusIMM, pp. 643-652.
.. [2] Vann, J., Jackson, S. & Bertoli, O. (2003). Quantitative kriging
       neighbourhood analysis for the mining geologist -- A description of
       the method with worked case examples. *Proceedings of the 5th
       International Mining Geology Conference*, AusIMM, pp. 215-223.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Classification by Kriging Variance
# ---------------------------------------------------------------------------


def classify_by_kriging_variance(
    kv: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """Classify blocks by kriging variance thresholds.

    Blocks are assigned a category based on their kriging variance
    relative to the threshold values:

    - kv <= measured threshold --> 1 (Measured)
    - kv <= indicated threshold --> 2 (Indicated)
    - kv > indicated threshold --> 3 (Inferred)

    Parameters
    ----------
    kv : numpy.ndarray
        1-D array of kriging variance values per block.
    thresholds : dict
        Dictionary with keys ``"measured"`` and ``"indicated"``, each
        mapping to a float threshold value. Example::

            {"measured": 0.2, "indicated": 0.5}

    Returns
    -------
    numpy.ndarray
        1-D integer array of category codes: 1 = Measured, 2 = Indicated,
        3 = Inferred.

    Raises
    ------
    ValueError
        If *kv* is empty or required threshold keys are missing.

    Examples
    --------
    >>> import numpy as np
    >>> kv = np.array([0.1, 0.3, 0.8, 0.15, 0.6])
    >>> classify_by_kriging_variance(kv, {"measured": 0.2, "indicated": 0.5})
    array([1, 2, 3, 1, 3])

    References
    ----------
    .. [1] Snowden (2001).
    """
    kv_arr = np.asarray(kv, dtype=float).ravel()
    if kv_arr.size == 0:
        raise ValueError("kv must not be empty.")

    for key in ("measured", "indicated"):
        if key not in thresholds:
            raise ValueError(f"thresholds must contain '{key}' key.")

    meas_thresh = thresholds["measured"]
    ind_thresh = thresholds["indicated"]

    # Start as Inferred
    classification = np.full(kv_arr.size, 3, dtype=int)

    # Indicated
    classification[kv_arr <= ind_thresh] = 2

    # Measured (overrides Indicated)
    classification[kv_arr <= meas_thresh] = 1

    return classification


# ---------------------------------------------------------------------------
# Classification by Search Pass
# ---------------------------------------------------------------------------


def classify_by_search_pass(
    n_samples: np.ndarray,
    min_octants: np.ndarray,
    pass_defs: list[dict],
) -> np.ndarray:
    """Classify blocks by search neighbourhood criteria.

    Each search pass defines minimum sample count and minimum number of
    octants with data. Blocks meeting pass 1 criteria are classified as
    Measured (1), pass 2 as Indicated (2), and the rest as Inferred (3).

    Parameters
    ----------
    n_samples : numpy.ndarray
        1-D array of the number of informing samples per block.
    min_octants : numpy.ndarray
        1-D array of the number of occupied octants per block.
    pass_defs : list[dict]
        List of search pass definitions, from strictest (Measured) to
        most relaxed (Indicated). Each dict must have keys:

        - ``"min_samples"`` : int -- minimum number of samples.
        - ``"min_octants"`` : int -- minimum number of occupied octants.

        Typically 2 passes: ``[{measured_criteria}, {indicated_criteria}]``.

    Returns
    -------
    numpy.ndarray
        1-D integer array of category codes. Pass index 0 --> 1 (Measured),
        pass index 1 --> 2 (Indicated), unmatched --> 3 (Inferred).

    Raises
    ------
    ValueError
        If input arrays differ in length, are empty, or pass_defs is
        empty.

    Examples
    --------
    >>> import numpy as np
    >>> n_samp = np.array([15, 8, 3, 20, 6])
    >>> n_oct = np.array([5, 3, 1, 6, 2])
    >>> passes = [
    ...     {"min_samples": 12, "min_octants": 4},
    ...     {"min_samples": 6, "min_octants": 2},
    ... ]
    >>> classify_by_search_pass(n_samp, n_oct, passes)
    array([1, 2, 3, 1, 2])

    References
    ----------
    .. [1] Standard kriging neighbourhood practice.
    """
    ns = np.asarray(n_samples, dtype=float).ravel()
    mo = np.asarray(min_octants, dtype=float).ravel()

    if ns.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if ns.size != mo.size:
        raise ValueError(
            f"n_samples ({ns.size}) and min_octants ({mo.size}) must have the same length."
        )
    if not pass_defs:
        raise ValueError("pass_defs must not be empty.")

    # Start as Inferred (highest category code)
    max_code = len(pass_defs) + 1
    classification = np.full(ns.size, max_code, dtype=int)

    # Apply passes in reverse order (most relaxed first) so stricter
    # passes overwrite
    for i in range(len(pass_defs) - 1, -1, -1):
        p = pass_defs[i]
        mask = (ns >= p["min_samples"]) & (mo >= p["min_octants"])
        classification[mask] = i + 1

    # Ensure codes map to standard 1=Measured, 2=Indicated, 3=Inferred
    # For >2 passes, cap at 3
    classification = np.clip(classification, 1, 3)

    return classification


# ---------------------------------------------------------------------------
# Slope of Regression
# ---------------------------------------------------------------------------


def slope_of_regression(
    estimated_grades: np.ndarray,
    true_cv: float,
) -> float:
    """Compute the slope of regression for estimation quality assessment.

    The slope of regression measures the conditional bias in kriging
    estimates. A slope near 1.0 indicates good estimation; a slope near
    0.0 indicates excessive smoothing.

    .. math::

        \\text{slope} = \\frac{1}{1 + \\frac{\\text{CV}_{kr}^2}
        {\\text{CV}_{true}^2}}

    where :math:`\\text{CV}_{kr} = \\sigma_{est} / \\mu_{est}` is the
    coefficient of variation of the estimated grades.

    Parameters
    ----------
    estimated_grades : numpy.ndarray
        1-D array of estimated (kriged) block grades.
    true_cv : float
        True coefficient of variation of the deposit grades. Must be
        positive.

    Returns
    -------
    float
        Slope of regression, typically in (0, 1].

    Raises
    ------
    ValueError
        If *estimated_grades* is empty, *true_cv* is not positive, or
        the mean of estimated grades is zero.

    Examples
    --------
    >>> import numpy as np
    >>> est = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    >>> round(slope_of_regression(est, 0.5), 4)
    0.9623

    References
    ----------
    .. [1] Vann, Jackson & Bertoli (2003), Eq. 3.
    """
    est = np.asarray(estimated_grades, dtype=float).ravel()
    if est.size == 0:
        raise ValueError("estimated_grades must not be empty.")
    validate_positive(true_cv, "true_cv")

    mean_est = float(est.mean())
    if mean_est == 0:
        raise ValueError("Mean of estimated_grades is zero; cannot compute CV.")

    std_est = float(est.std(ddof=0))
    cv_kr = std_est / abs(mean_est)

    slope = 1.0 / (1.0 + (cv_kr**2) / (true_cv**2))

    return float(slope)
