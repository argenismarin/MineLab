"""Rock fragmentation prediction models for blast design.

This module implements the Kuz-Ram model, modified Kuz-Ram (Cunningham 2005),
uniformity index, and the Swebrec size distribution function used in modern
fragmentation analysis.

References
----------
.. [1] Cunningham, C.V.B. (1983). The Kuz-Ram model for prediction of
       fragmentation from blasting. *Proc. 1st Int. Symp. on Rock
       Fragmentation by Blasting*, 439-453.
.. [2] Cunningham, C.V.B. (1987). Fragmentation estimations and the
       Kuz-Ram model -- four years on. *Proc. 2nd Int. Symp. on Rock
       Fragmentation by Blasting*, 475-487.
.. [3] Cunningham, C.V.B. (2005). The Kuz-Ram fragmentation model -- 20
       years on. *Proc. Brighton Conf.*, 201-210.
.. [4] Ouchterlony, F. (2005). The Swebrec function: linking fragmentation
       by blasting and crushing. *Mining Technology*, 114(1), 29-44.
"""

from __future__ import annotations

import math

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Kuz-Ram Fragmentation Model
# ---------------------------------------------------------------------------


def kuz_ram(
    powder_factor: float,
    charge_per_hole: float,
    swe: float,
    rock_factor: float,
    n_rows: int = 1,
) -> dict:
    """Predict mean fragmentation size using the Kuz-Ram model.

    The mean fragment size *X50* is computed as:

    .. math::

        X_{50} = A \\, K^{-0.8} \\, Q^{1/6} \\,
                 \\left(\\frac{115}{\\text{SWE}}\\right)^{19/30}

    where *K* is the powder factor (kg/m³), *Q* is the charge per hole (kg),
    *SWE* is the relative weight strength compared to ANFO (= 100), and
    *A* is the rock factor.

    For multiple rows, a correction factor of ``0.95^(n_rows - 1)`` is
    applied following Cunningham (1987).

    Parameters
    ----------
    powder_factor : float
        Powder factor *K* in kg/m³. Must be positive.
    charge_per_hole : float
        Explosive charge per hole *Q* in kg. Must be positive.
    swe : float
        Relative weight strength of the explosive (ANFO = 100).
        Must be positive.
    rock_factor : float
        Rock factor *A* (typically 0.8 to 22). Must be positive.
    n_rows : int, optional
        Number of blast rows (default 1). Multi-row correction applied
        for n_rows > 1. Must be >= 1.

    Returns
    -------
    dict
        Dictionary with:

        - ``x50`` : float -- Mean fragment size in metres.
        - ``powder_factor`` : float -- Input powder factor (kg/m³).
        - ``rock_factor`` : float -- Input rock factor.

    Raises
    ------
    ValueError
        If any input is out of range.

    Examples
    --------
    >>> result = kuz_ram(0.5, 50, 100, 10)
    >>> round(result['x50'], 3)
    0.365

    References
    ----------
    .. [1] Cunningham (1983), Eq. 1.
    .. [2] Cunningham (1987), multi-row correction.
    """
    validate_positive(powder_factor, "powder_factor")
    validate_positive(charge_per_hole, "charge_per_hole")
    validate_positive(swe, "swe")
    validate_positive(rock_factor, "rock_factor")
    if n_rows < 1:
        raise ValueError(f"'n_rows' must be >= 1, got {n_rows}.")

    # Cunningham formula yields X50 in cm; convert to metres
    x50_cm = (
        rock_factor
        * powder_factor ** (-0.8)
        * charge_per_hole ** (1.0 / 6.0)
        * (115.0 / swe) ** (19.0 / 30.0)
    )

    # Multi-row correction (Cunningham 1987)
    if n_rows > 1:
        x50_cm *= 0.95 ** (n_rows - 1)

    x50 = x50_cm / 100.0  # cm -> m

    return {
        "x50": x50,
        "powder_factor": powder_factor,
        "rock_factor": rock_factor,
    }


# ---------------------------------------------------------------------------
# Uniformity Index (Cunningham 1987)
# ---------------------------------------------------------------------------


def uniformity_index(
    diameter: float,
    burden: float,
    spacing: float,
    bench_height: float,
    drill_accuracy: float,
    charge_length: float,
    bottom_charge_length: float,
) -> float:
    """Compute the Cunningham uniformity index *n*.

    The uniformity index controls the spread of the Rosin-Rammler
    distribution used in the Kuz-Ram model:

    .. math::

        n = (2.2 - 14\\,B/D) \\; (1 - W/B) \\;
            \\left(\\frac{|BCL - L|}{L} + 0.1\\right)^{0.1} \\;
            \\frac{L}{H}

    Parameters
    ----------
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    burden : float
        Burden *B* in metres. Must be positive.
    spacing : float
        Spacing *S* in metres. Must be positive.
    bench_height : float
        Bench height *H* in metres. Must be positive.
    drill_accuracy : float
        Standard deviation of drilling accuracy *W* in metres.
        Must be non-negative and less than *burden*.
    charge_length : float
        Total charge length *L* in metres. Must be positive.
    bottom_charge_length : float
        Bottom charge length *BCL* in metres. Must be non-negative.

    Returns
    -------
    float
        Uniformity index *n* (dimensionless). Typically 0.8 to 2.2.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    >>> round(uniformity_index(89, 2.4, 2.76, 10, 0.1, 8.0, 3.0), 2)
    1.35

    References
    ----------
    .. [1] Cunningham (1987), Eq. 3.
    """
    validate_positive(diameter, "diameter")
    validate_positive(burden, "burden")
    validate_positive(spacing, "spacing")
    validate_positive(bench_height, "bench_height")
    if drill_accuracy < 0:
        raise ValueError(f"'drill_accuracy' must be non-negative, got {drill_accuracy}.")
    if drill_accuracy >= burden:
        raise ValueError(
            f"'drill_accuracy' must be less than burden={burden}, got {drill_accuracy}."
        )
    validate_positive(charge_length, "charge_length")
    if bottom_charge_length < 0:
        raise ValueError(
            f"'bottom_charge_length' must be non-negative, got {bottom_charge_length}."
        )

    n = (
        (2.2 - 14.0 * burden / diameter)
        * (1.0 - drill_accuracy / burden)
        * (abs(bottom_charge_length - charge_length) / charge_length + 0.1) ** 0.1
        * (charge_length / bench_height)
    )

    return n


# ---------------------------------------------------------------------------
# Modified Kuz-Ram (Cunningham 2005) with Swebrec
# ---------------------------------------------------------------------------


def modified_kuz_ram(
    powder_factor: float,
    charge_per_hole: float,
    swe: float,
    rock_factor: float,
    diameter: float,
    n_rows: int = 1,
) -> dict:
    """Predict fragmentation using the Modified Kuz-Ram model.

    The Modified Kuz-Ram model (Cunningham 2005) uses the same *X50*
    calculation as the original Kuz-Ram but replaces the Rosin-Rammler
    distribution with the Swebrec function for better fines prediction.

    The maximum fragment size *Xmax* is estimated as the *in-situ block
    size*, approximated here as:

    .. math::

        X_{max} = D \\times \\frac{A}{0.06} \\times 0.01

    where *D* is the diameter in mm, *A* is the rock factor, and the
    result is in cm (same units as *X50*). A simplified Swebrec
    exponent is estimated from the ratio of *X50* to *Xmax*.

    Parameters
    ----------
    powder_factor : float
        Powder factor *K* in kg/m³. Must be positive.
    charge_per_hole : float
        Explosive charge per hole *Q* in kg. Must be positive.
    swe : float
        Relative weight strength (ANFO = 100). Must be positive.
    rock_factor : float
        Rock factor *A* (typically 0.8 to 22). Must be positive.
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    n_rows : int, optional
        Number of blast rows (default 1). Must be >= 1.

    Returns
    -------
    dict
        Dictionary with:

        - ``x50`` : float -- Mean fragment size (m).
        - ``xmax`` : float -- Maximum fragment size estimate (m).
        - ``n_swebrec`` : float -- Swebrec exponent.
        - ``powder_factor`` : float -- Input powder factor (kg/m³).
        - ``rock_factor`` : float -- Input rock factor.

    Examples
    --------
    >>> result = modified_kuz_ram(0.5, 50, 100, 10, 89)
    >>> round(result['x50'], 3)
    0.365

    References
    ----------
    .. [1] Cunningham (2005).
    .. [2] Ouchterlony (2005).
    """
    validate_positive(powder_factor, "powder_factor")
    validate_positive(charge_per_hole, "charge_per_hole")
    validate_positive(swe, "swe")
    validate_positive(rock_factor, "rock_factor")
    validate_positive(diameter, "diameter")
    if n_rows < 1:
        raise ValueError(f"'n_rows' must be >= 1, got {n_rows}.")

    # X50 from standard Kuz-Ram (already in metres)
    result = kuz_ram(powder_factor, charge_per_hole, swe, rock_factor, n_rows)
    x50 = result["x50"]

    # Estimate Xmax from in-situ block size approximation
    # rock_factor ~ 0.06 * BI, so BI ~ rock_factor / 0.06
    # Xmax in metres, estimated as function of hole diameter and blastability
    xmax = diameter * (rock_factor / 0.06) * 0.0001  # mm -> m with BI scaling

    # Swebrec exponent (simplified default)
    n_swebrec_val = 2.0 - math.log(x50 / xmax) if x50 < xmax else 1.0

    return {
        "x50": x50,
        "xmax": xmax,
        "n_swebrec": n_swebrec_val,
        "powder_factor": powder_factor,
        "rock_factor": rock_factor,
    }


# ---------------------------------------------------------------------------
# Swebrec Distribution Function
# ---------------------------------------------------------------------------


def swebrec_distribution(
    x50: float,
    xmax: float,
    n_swebrec: float,
    sizes: np.ndarray,
) -> np.ndarray:
    """Compute cumulative passing fraction using the Swebrec function.

    .. math::

        F(x) = \\frac{1}{1 + \\left(\\frac{\\ln(x_{max}/x)}
               {\\ln(x_{max}/x_{50})}\\right)^{n}}

    where *x* is the fragment size, *x50* the median size, *xmax* the
    maximum fragment size, and *n* the Swebrec exponent.

    Parameters
    ----------
    x50 : float
        Median fragment size (50 % passing). Must be positive and < *xmax*.
    xmax : float
        Maximum fragment size. Must be positive and > *x50*.
    n_swebrec : float
        Swebrec exponent. Must be positive.
    sizes : numpy.ndarray
        Array of fragment sizes at which to evaluate *F(x)*.
        All values must be positive and <= *xmax*.

    Returns
    -------
    numpy.ndarray
        Cumulative fraction passing (0 to 1) for each size.

    Raises
    ------
    ValueError
        If ``x50 >= xmax`` or any size exceeds *xmax*.

    Examples
    --------
    >>> import numpy as np
    >>> sizes = np.array([5, 10, 20, 40, 80])
    >>> passing = swebrec_distribution(20, 100, 2.0, sizes)
    >>> round(passing[2], 2)  # at x50 should be ~0.5
    0.5

    References
    ----------
    .. [1] Ouchterlony (2005), Eq. 5.
    """
    validate_positive(x50, "x50")
    validate_positive(xmax, "xmax")
    validate_positive(n_swebrec, "n_swebrec")

    if x50 >= xmax:
        raise ValueError(f"'x50' must be less than 'xmax'. Got x50={x50}, xmax={xmax}.")

    sizes = np.asarray(sizes, dtype=float)
    if np.any(sizes <= 0):
        raise ValueError("All sizes must be positive.")
    if np.any(sizes > xmax):
        raise ValueError(f"All sizes must be <= xmax={xmax}.")

    log_ratio_50 = math.log(xmax / x50)

    result = np.zeros_like(sizes)
    for i, x in enumerate(sizes):
        if x >= xmax:
            result[i] = 1.0
        else:
            log_ratio_x = math.log(xmax / x)
            result[i] = 1.0 / (1.0 + (log_ratio_x / log_ratio_50) ** n_swebrec)

    return result
