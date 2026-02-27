"""Descriptive and exploratory statistics for mining data.

Functions for computing summary statistics, log-transformed statistics,
contact analysis, capping (top-cut) analysis, and probability plots.

References
----------
.. [1] Isaaks, E.H. & Srivastava, R.M., *An Introduction to Applied
       Geostatistics*, Oxford University Press, 1989.
.. [2] Sinclair, A.J., *Applied Mineral Inventory Estimation*, Cambridge
       University Press, 2002.
.. [3] Rossi, M.E. & Deutsch, C.V., *Mineral Resource Estimation*,
       Springer, 2014.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from minelab.utilities.validators import validate_array, validate_positive

Number = int | float


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def descriptive_stats(data: Sequence[float]) -> dict[str, float]:
    """Compute descriptive statistics for a 1-D dataset.

    Parameters
    ----------
    data : array-like of float
        Input data values.

    Returns
    -------
    dict
        Keys: ``count``, ``mean``, ``var``, ``std``, ``cv``,
        ``skew``, ``kurtosis``, ``min``, ``max``, ``p25``, ``p50``,
        ``p75``.

    Examples
    --------
    >>> s = descriptive_stats([1, 2, 3, 4, 5])
    >>> s['mean']
    3.0
    >>> s['count']
    5.0

    References
    ----------
    .. [1] Isaaks & Srivastava, 1989, ch. 2.
    """
    arr = validate_array(data, "data", min_length=1)
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=1))
    std = float(np.std(arr, ddof=1))
    cv = std / mean if mean != 0 else float("inf")
    return {
        "count": float(arr.size),
        "mean": mean,
        "var": var,
        "std": std,
        "cv": cv,
        "skew": float(sp_stats.skew(arr, bias=False)),
        "kurtosis": float(sp_stats.kurtosis(arr, bias=False)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Log-transformed statistics
# ---------------------------------------------------------------------------


def log_stats(data: Sequence[float]) -> dict[str, float]:
    """Compute descriptive statistics on the natural-log transform.

    Only strictly positive values are retained; zeros and negatives are
    silently dropped.

    Parameters
    ----------
    data : array-like of float
        Input data values (positive values only).

    Returns
    -------
    dict
        Same keys as :func:`descriptive_stats`, computed on
        ``ln(data)``.  Additionally includes ``n_dropped`` indicating
        how many non-positive values were removed.

    Raises
    ------
    ValueError
        If no positive values remain after filtering.

    Examples
    --------
    >>> s = log_stats([1, 10, 100])
    >>> round(s['mean'], 4)
    3.0702

    References
    ----------
    .. [1] Sinclair, A.J., 2002, ch. 3.
    """
    arr = validate_array(data, "data", min_length=1)
    positive = arr[arr > 0]
    n_dropped = arr.size - positive.size
    if positive.size == 0:
        raise ValueError("No positive values in 'data' for log transform.")
    log_arr = np.log(positive)
    result = descriptive_stats(log_arr)
    result["n_dropped"] = float(n_dropped)
    return result


# ---------------------------------------------------------------------------
# Contact analysis
# ---------------------------------------------------------------------------


def contact_analysis(
    data: Sequence[float],
    coords: Sequence[float],
    direction: int,
    lag: float,
    n_lags: int,
) -> pd.DataFrame:
    """Compute a contact (transition) profile along a coordinate axis.

    The contact analysis evaluates mean grade in bins of increasing
    distance from a reference coordinate (the minimum coordinate in the
    dataset), which is useful for detecting grade trends near geological
    contacts.

    Parameters
    ----------
    data : array-like of float
        Grade or attribute values.
    coords : array-like of float
        Coordinates along the analysis direction (same length as
        *data*).  For 1-D analysis, pass the relevant axis directly.
    direction : int
        Column index — reserved for future multi-column support.
        Currently ignored (use 0).
    lag : float
        Bin width (distance increment).
    n_lags : int
        Number of lag bins.

    Returns
    -------
    pandas.DataFrame
        Columns: ``lag_start``, ``lag_end``, ``lag_center``, ``count``,
        ``mean``, ``variance``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> d = np.random.normal(5, 1, 100)
    >>> c = np.linspace(0, 100, 100)
    >>> df = contact_analysis(d, c, 0, 10.0, 10)
    >>> len(df)
    10

    References
    ----------
    .. [1] Rossi & Deutsch, 2014, ch. 4.
    """
    d = validate_array(data, "data")
    co = validate_array(coords, "coords")
    if d.size != co.size:
        raise ValueError(f"'data' and 'coords' must have the same length ({d.size} != {co.size}).")
    validate_positive(lag, "lag")
    validate_positive(n_lags, "n_lags")

    origin = co.min()
    rows = []
    for i in range(int(n_lags)):
        lo = origin + i * lag
        hi = lo + lag
        mask = (co >= lo) & (co < hi)
        subset = d[mask]
        count = int(subset.size)
        mean_val = float(np.mean(subset)) if count > 0 else float("nan")
        var_val = float(np.var(subset, ddof=1)) if count > 1 else float("nan")
        rows.append(
            {
                "lag_start": lo,
                "lag_end": hi,
                "lag_center": (lo + hi) / 2.0,
                "count": count,
                "mean": mean_val,
                "variance": var_val,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Capping analysis
# ---------------------------------------------------------------------------


def capping_analysis(
    data: Sequence[float],
    percentiles: Sequence[float],
) -> pd.DataFrame:
    """Evaluate the effect of top-cutting at various percentiles.

    For each percentile threshold, all values above that threshold are
    replaced (capped) by the threshold value, and the resulting mean
    and coefficient of variation (CV) are computed.

    Parameters
    ----------
    data : array-like of float
        Raw grade data.
    percentiles : array-like of float
        Percentiles at which to evaluate capping (e.g., ``[90, 95, 97.5,
        99]``).  Each value must be in [0, 100].

    Returns
    -------
    pandas.DataFrame
        Columns: ``percentile``, ``threshold``, ``capped_mean``,
        ``capped_cv``, ``n_capped``, ``pct_metal_removed``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> d = np.random.lognormal(0, 1, 1000)
    >>> df = capping_analysis(d, [90, 95, 99])
    >>> list(df.columns[:3])
    ['percentile', 'threshold', 'capped_mean']

    References
    ----------
    .. [1] Rossi & Deutsch, 2014, ch. 5.
    """
    arr = validate_array(data, "data", min_length=2)
    pcts = validate_array(percentiles, "percentiles")

    original_metal = float(np.sum(arr))

    rows = []
    for p in pcts:
        threshold = float(np.percentile(arr, p))
        capped = np.minimum(arr, threshold)
        capped_mean = float(np.mean(capped))
        capped_std = float(np.std(capped, ddof=1))
        capped_cv = capped_std / capped_mean if capped_mean != 0 else float("inf")
        n_capped = int(np.sum(arr > threshold))
        metal_removed = original_metal - float(np.sum(capped))
        pct_metal = metal_removed / original_metal * 100.0 if original_metal != 0 else 0.0
        rows.append(
            {
                "percentile": p,
                "threshold": threshold,
                "capped_mean": capped_mean,
                "capped_cv": capped_cv,
                "n_capped": n_capped,
                "pct_metal_removed": pct_metal,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probability plot (Q-Q)
# ---------------------------------------------------------------------------


def probability_plot(
    data: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sorted data and theoretical normal quantiles for a Q-Q plot.

    Parameters
    ----------
    data : array-like of float
        Input data values.

    Returns
    -------
    sorted_data : numpy.ndarray
        Sorted observed values.
    theoretical_quantiles : numpy.ndarray
        Expected quantiles from a standard normal distribution
        (Blom plotting positions).

    Examples
    --------
    >>> sd, tq = probability_plot([3, 1, 2])
    >>> list(sd)
    [1.0, 2.0, 3.0]

    References
    ----------
    .. [1] Blom, G., *Statistical Estimates and Transformed Beta
           Variables*, Wiley, 1958.
    .. [2] Isaaks & Srivastava, 1989, ch. 2.
    """
    arr = validate_array(data, "data", min_length=2)
    sorted_data = np.sort(arr)
    n = arr.size
    # Blom plotting positions: (i - 3/8) / (n + 1/4)
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_quantiles = sp_stats.norm.ppf(positions)
    return sorted_data, theoretical_quantiles
