"""Sensitivity analysis for mining project evaluation.

This module provides tornado and spider-plot analysis tools that help
identify the parameters with the greatest impact on project value.

References
----------
.. [1] Stermole, F.J. & Stermole, J.M. (2014). *Economic Evaluation and
       Investment Decision Methods*, 14th ed. Investment Evaluations Corp.
.. [2] Hustrulid, W. et al. (2013). *Open Pit Mine Planning and Design*,
       3rd ed. CRC Press, Ch. 3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tornado Analysis
# ---------------------------------------------------------------------------


def tornado_analysis(
    base_params: dict[str, float],
    variations: dict[str, tuple[float, float]],
    model_fn: Callable[..., float],
) -> list[dict[str, object]]:
    """Perform a tornado (one-at-a-time) sensitivity analysis.

    For each parameter listed in *variations*, the model is evaluated at
    the low and high values while all other parameters stay at their base
    values.  The results are returned sorted by descending *swing*
    (|high_result - low_result|).

    Parameters
    ----------
    base_params : dict
        ``{param_name: base_value}`` — the base-case parameter set.
    variations : dict
        ``{param_name: (low_value, high_value)}`` — the range to test
        for each parameter.  Every key must also be present in
        *base_params*.
    model_fn : callable
        A function ``(**params) -> float`` that returns the metric of
        interest.

    Returns
    -------
    list of dict
        Each dict contains:

        - ``"param"`` — parameter name
        - ``"low"`` — model result at the low variation
        - ``"high"`` — model result at the high variation
        - ``"base"`` — model result at base-case values
        - ``"swing"`` — absolute difference ``|high - low|``

        The list is sorted by descending *swing*.

    Raises
    ------
    KeyError
        If a key in *variations* is not found in *base_params*.

    Examples
    --------
    >>> def profit(price, cost):
    ...     return price - cost
    >>> base = {"price": 100, "cost": 50}
    >>> var = {"price": (80, 120), "cost": (40, 60)}
    >>> results = tornado_analysis(base, var, profit)
    >>> results[0]["swing"]
    40.0

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.14.
    """
    for key in variations:
        if key not in base_params:
            raise KeyError(f"Variation parameter '{key}' not found in base_params.")

    base_result = model_fn(**base_params)
    records: list[dict[str, object]] = []

    for param, (lo_val, hi_val) in variations.items():
        # Low case
        params_low = dict(base_params)
        params_low[param] = lo_val
        low_result = model_fn(**params_low)

        # High case
        params_high = dict(base_params)
        params_high[param] = hi_val
        high_result = model_fn(**params_high)

        records.append(
            {
                "param": param,
                "low": float(low_result),
                "high": float(high_result),
                "base": float(base_result),
                "swing": float(abs(high_result - low_result)),
            }
        )

    # Sort by descending swing
    records.sort(key=lambda r: r["swing"], reverse=True)  # type: ignore[arg-type]
    return records


# ---------------------------------------------------------------------------
# Spider-Plot Data
# ---------------------------------------------------------------------------


def spider_plot_data(
    base_params: dict[str, float],
    param_names: Sequence[str],
    range_pct: float,
    steps: int,
    model_fn: Callable[..., float],
) -> dict[str, tuple[list[float], list[float]]]:
    """Generate data for a spider (sensitivity) plot.

    Each selected parameter is varied from ``(1 - range_pct) * base`` to
    ``(1 + range_pct) * base`` in *steps* evenly spaced increments, while
    all other parameters remain at their base values.

    Parameters
    ----------
    base_params : dict
        ``{param_name: base_value}`` — the base-case parameter set.
    param_names : sequence of str
        Names of parameters to vary (must be keys in *base_params*).
    range_pct : float
        Fractional range to apply (e.g. 0.20 for +/- 20 %).  Must be > 0.
    steps : int
        Number of evaluation points along each parameter's range.
        Must be >= 2.
    model_fn : callable
        A function ``(**params) -> float``.

    Returns
    -------
    dict
        ``{param_name: (pct_changes, values)}`` where *pct_changes* is a
        list of percentage deviations from the base (e.g. ``[-20, 0, 20]``)
        and *values* is a list of corresponding model results.

    Raises
    ------
    KeyError
        If a name in *param_names* is not found in *base_params*.
    ValueError
        If *range_pct* <= 0 or *steps* < 2.

    Examples
    --------
    >>> def profit(price, cost):
    ...     return price - cost
    >>> base = {"price": 100, "cost": 50}
    >>> data = spider_plot_data(base, ["price", "cost"], 0.20, 5, profit)
    >>> len(data["price"][0])
    5

    References
    ----------
    .. [1] Stermole & Stermole (2014), Sec. 2.14.
    """
    if range_pct <= 0:
        raise ValueError("range_pct must be positive.")
    if steps < 2:
        raise ValueError("steps must be at least 2.")
    for name in param_names:
        if name not in base_params:
            raise KeyError(f"Parameter '{name}' not found in base_params.")

    multipliers = np.linspace(1.0 - range_pct, 1.0 + range_pct, steps)
    pct_changes = ((multipliers - 1.0) * 100.0).tolist()

    result: dict[str, tuple[list[float], list[float]]] = {}

    for name in param_names:
        values: list[float] = []
        base_val = base_params[name]
        for mult in multipliers:
            params = dict(base_params)
            params[name] = base_val * mult
            values.append(float(model_fn(**params)))
        result[name] = (pct_changes, values)

    return result
