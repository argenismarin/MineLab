"""Grade conversion and ore-reserve utilities for mining engineering.

Functions for converting between common grade units (ppm, %, g/t, oz/ton)
and for computing grade-tonnage curves, metal content, and equivalent
grades.

References
----------
.. [1] SME Mining Engineering Handbook, 3rd Edition, SME, 2011.
.. [2] Hustrulid, W., Kuchta, M. & Martin, R., *Open Pit Mine Planning
       and Design*, 3rd ed., CRC Press, 2013.
.. [3] 1 troy oz = 31.1035 g; 1 short ton = 907.185 kg.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from minelab.utilities.validators import (
    validate_array,
    validate_non_negative,
    validate_positive,
)

Number = int | float

# Troy ounce / short-ton  <->  grams / metric tonne
# 1 troy oz = 31.1035 g, 1 short ton = 0.907185 t
_TROY_OZ_G = 31.1035
_SHORT_TON_T = 0.90718474


# ---------------------------------------------------------------------------
# Simple grade unit conversions
# ---------------------------------------------------------------------------


def ppm_to_percent(value: Number) -> float:
    """Convert parts-per-million to weight percent.

    Parameters
    ----------
    value : int or float
        Grade in ppm.

    Returns
    -------
    float
        Grade in percent.

    Examples
    --------
    >>> ppm_to_percent(10000)
    1.0

    References
    ----------
    .. [1] 1 % = 10 000 ppm.
    """
    return float(value) / 10_000.0


def percent_to_ppm(value: Number) -> float:
    """Convert weight percent to parts-per-million.

    Parameters
    ----------
    value : int or float
        Grade in percent.

    Returns
    -------
    float
        Grade in ppm.

    Examples
    --------
    >>> percent_to_ppm(1.0)
    10000.0

    References
    ----------
    .. [1] 1 % = 10 000 ppm.
    """
    return float(value) * 10_000.0


def ppm_to_gpt(value: Number) -> float:
    """Convert ppm to grams per tonne (identity — 1 ppm = 1 g/t).

    Parameters
    ----------
    value : int or float
        Grade in ppm.

    Returns
    -------
    float
        Grade in g/t.

    Examples
    --------
    >>> ppm_to_gpt(5.0)
    5.0

    References
    ----------
    .. [1] 1 g/t = 1 mg/kg = 1 ppm (mass/mass).
    """
    return float(value)


def gpt_to_ppm(value: Number) -> float:
    """Convert grams per tonne to ppm (identity — 1 g/t = 1 ppm).

    Parameters
    ----------
    value : int or float
        Grade in g/t.

    Returns
    -------
    float
        Grade in ppm.

    Examples
    --------
    >>> gpt_to_ppm(5.0)
    5.0

    References
    ----------
    .. [1] 1 g/t = 1 mg/kg = 1 ppm (mass/mass).
    """
    return float(value)


def oz_per_ton_to_gpt(value: Number) -> float:
    """Convert troy ounces per short ton to grams per metric tonne.

    Parameters
    ----------
    value : int or float
        Grade in troy oz / short ton.

    Returns
    -------
    float
        Grade in g/t.

    Examples
    --------
    >>> round(oz_per_ton_to_gpt(1.0), 4)
    34.2857

    References
    ----------
    .. [1] 1 troy oz = 31.1035 g, 1 short ton = 0.907185 t.
           Factor = 31.1035 / 0.907185 = 34.2857 g/t per oz/ton.
    """
    return float(value) * _TROY_OZ_G / _SHORT_TON_T


def gpt_to_oz_per_ton(value: Number) -> float:
    """Convert grams per metric tonne to troy ounces per short ton.

    Parameters
    ----------
    value : int or float
        Grade in g/t.

    Returns
    -------
    float
        Grade in troy oz / short ton.

    Examples
    --------
    >>> round(gpt_to_oz_per_ton(34.2857), 4)
    1.0

    References
    ----------
    .. [1] 1 troy oz = 31.1035 g, 1 short ton = 0.907185 t.
    """
    return float(value) * _SHORT_TON_T / _TROY_OZ_G


# ---------------------------------------------------------------------------
# Grade-tonnage curve
# ---------------------------------------------------------------------------


def grade_tonnage_curve(
    grades: Sequence[float],
    tonnages: Sequence[float],
    cutoffs: Sequence[float],
) -> pd.DataFrame:
    """Compute a grade-tonnage table for a set of cutoff grades.

    For each cutoff, the function determines the tonnes above cutoff,
    the mean grade above cutoff, and the total contained metal above
    cutoff.

    Parameters
    ----------
    grades : array-like of float
        Individual block or sample grades.
    tonnages : array-like of float
        Tonnage associated with each grade entry (same length as
        *grades*).
    cutoffs : array-like of float
        Cutoff grades at which to evaluate the curve.

    Returns
    -------
    pandas.DataFrame
        Columns: ``cutoff``, ``tonnes_above``, ``mean_grade_above``,
        ``metal_above``.

    Raises
    ------
    ValueError
        If *grades* and *tonnages* have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> g = [0.5, 1.0, 1.5, 2.0]
    >>> t = [100, 100, 100, 100]
    >>> df = grade_tonnage_curve(g, t, [0.0, 1.0, 1.5])
    >>> list(df.columns)
    ['cutoff', 'tonnes_above', 'mean_grade_above', 'metal_above']

    References
    ----------
    .. [1] Hustrulid, W. et al., *Open Pit Mine Planning and Design*,
           3rd ed., CRC Press, 2013, ch. 6.
    """
    g = validate_array(grades, "grades")
    t = validate_array(tonnages, "tonnages")
    if g.size != t.size:
        raise ValueError(
            f"'grades' and 'tonnages' must have the same length ({g.size} != {t.size})."
        )
    c = validate_array(cutoffs, "cutoffs")

    rows = []
    for co in c:
        mask = g >= co
        tonnes_above = t[mask].sum()
        if tonnes_above > 0:
            mean_grade = np.average(g[mask], weights=t[mask])
            metal_above = (g[mask] * t[mask]).sum()
        else:
            mean_grade = 0.0
            metal_above = 0.0
        rows.append(
            {
                "cutoff": co,
                "tonnes_above": tonnes_above,
                "mean_grade_above": mean_grade,
                "metal_above": metal_above,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metal content
# ---------------------------------------------------------------------------


def metal_content(
    tonnage: Number,
    grade: Number,
    recovery: float = 1.0,
) -> float:
    """Compute contained (or recoverable) metal.

    Parameters
    ----------
    tonnage : int or float
        Ore tonnage (same mass unit as the result).
    grade : int or float
        Grade expressed as a fraction (e.g., 0.01 for 1 %).  For g/t
        grades, divide by 1e6 first so that the result is in the same
        mass unit as *tonnage*.
    recovery : float, optional
        Metallurgical recovery as a fraction in [0, 1] (default 1.0).

    Returns
    -------
    float
        Contained metal: ``tonnage * grade * recovery``.

    Examples
    --------
    >>> metal_content(1_000_000, 0.005, recovery=0.90)
    4500.0

    References
    ----------
    .. [1] SME Mining Engineering Handbook, 3rd ed., 2011, ch. 5.
    """
    validate_non_negative(tonnage, "tonnage")
    validate_non_negative(grade, "grade")
    validate_non_negative(recovery, "recovery")
    return float(tonnage) * float(grade) * float(recovery)


# ---------------------------------------------------------------------------
# Equivalent grade (multi-element)
# ---------------------------------------------------------------------------


def equivalent_grade(
    grades: Sequence[float],
    prices: Sequence[float],
    recoveries: Sequence[float] | None = None,
) -> float:
    """Compute a single equivalent grade from multiple elements.

    The first element is the *reference* element.  The equivalent grade
    is expressed in the same unit as the first element's grade.

    .. math::

        g_{eq} = g_1 + \\sum_{i=2}^{n}
                 \\frac{g_i \\, p_i \\, r_i}{p_1 \\, r_1}

    Parameters
    ----------
    grades : sequence of float
        Grades for each element (same units within each element).
    prices : sequence of float
        Prices per unit mass for each element (same currency).
    recoveries : sequence of float, optional
        Recovery fractions for each element (default all 1.0).

    Returns
    -------
    float
        Equivalent grade in the units of the first element.

    Raises
    ------
    ValueError
        If input sequences have mismatched lengths or fewer than 2
        elements.

    Examples
    --------
    >>> equivalent_grade([1.0, 20.0], [5000, 25], [0.90, 0.85])
    1.9444...

    References
    ----------
    .. [1] SME Mining Engineering Handbook, 3rd ed., 2011, ch. 28.
    """
    g = np.asarray(grades, dtype=float)
    p = np.asarray(prices, dtype=float)
    r = np.ones_like(g) if recoveries is None else np.asarray(recoveries, dtype=float)

    if g.size < 2 or p.size < 2:
        raise ValueError("At least 2 elements are required for equivalent grade.")
    if not (g.size == p.size == r.size):
        raise ValueError("grades, prices, and recoveries must have equal length.")

    ref_value = p[0] * r[0]
    validate_positive(ref_value, "reference price * recovery")

    eq = g[0]
    for i in range(1, g.size):
        eq += g[i] * p[i] * r[i] / ref_value
    return float(eq)
