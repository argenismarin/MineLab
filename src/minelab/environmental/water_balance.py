"""Site water balance and dewatering calculations.

This module provides tools for managing mine-site water accounting:
monthly storage balance, pit dewatering estimates using Darcy's law, and
surface runoff via the rational method.

References
----------
.. [1] Younger, P.L., Banwart, S.A. & Hedin, R.S. (2002). *Mine Water:
       Hydrology, Pollution, Remediation*. Kluwer Academic.
.. [2] Freeze, R.A. & Cherry, J.A. (1979). *Groundwater*. Prentice-Hall.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Site Water Balance
# ---------------------------------------------------------------------------


def site_water_balance(
    precipitation: list[float],
    evaporation: list[float],
    inflows: list[float],
    outflows: list[float],
    initial_storage: float = 0.0,
) -> dict:
    """Compute a monthly site water balance.

    For each period *i*:

    .. math::

        \\Delta S_i = P_i - E_i + I_i - O_i

    Cumulative storage at period *i* is the running sum of net changes plus
    the initial storage.

    Parameters
    ----------
    precipitation : list[float]
        Precipitation volumes per period (m3).  All values must be >= 0.
    evaporation : list[float]
        Evaporation volumes per period (m3).  All values must be >= 0.
    inflows : list[float]
        External inflow volumes per period (m3).  All values must be >= 0.
    outflows : list[float]
        Managed discharge/outflow volumes per period (m3).  All values
        must be >= 0.
    initial_storage : float, optional
        Volume of water stored at the start of the first period (m3).
        Default is 0.0.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"net_change"`` : list[float] -- Net volume change per period.
        - ``"cumulative_storage"`` : list[float] -- Storage at the end of
          each period.
        - ``"final_storage"`` : float -- Storage after the last period.

    Raises
    ------
    ValueError
        If input lists have different lengths or contain negative values.

    Examples
    --------
    >>> result = site_water_balance(
    ...     precipitation=[100, 50],
    ...     evaporation=[30, 20],
    ...     inflows=[10, 10],
    ...     outflows=[20, 15],
    ...     initial_storage=0.0,
    ... )
    >>> result["net_change"]
    [60.0, 25.0]
    >>> result["cumulative_storage"]
    [60.0, 85.0]
    >>> result["final_storage"]
    85.0

    References
    ----------
    .. [1] Younger et al. (2002), Ch. 4 -- Mine water balance practice.
    """
    n = len(precipitation)
    if not (len(evaporation) == len(inflows) == len(outflows) == n):
        raise ValueError(
            "All input lists (precipitation, evaporation, inflows, outflows) "
            "must have the same length."
        )
    if n == 0:
        raise ValueError("Input lists must not be empty.")

    validate_non_negative(initial_storage, "initial_storage")

    p = np.asarray(precipitation, dtype=float)
    e = np.asarray(evaporation, dtype=float)
    i = np.asarray(inflows, dtype=float)
    o = np.asarray(outflows, dtype=float)

    if np.any(p < 0):
        raise ValueError("All precipitation values must be non-negative.")
    if np.any(e < 0):
        raise ValueError("All evaporation values must be non-negative.")
    if np.any(i < 0):
        raise ValueError("All inflow values must be non-negative.")
    if np.any(o < 0):
        raise ValueError("All outflow values must be non-negative.")

    net = p - e + i - o
    cumulative = np.cumsum(net) + initial_storage

    return {
        "net_change": net.tolist(),
        "cumulative_storage": cumulative.tolist(),
        "final_storage": float(cumulative[-1]),
    }


# ---------------------------------------------------------------------------
# Pit Dewatering Estimate
# ---------------------------------------------------------------------------


def pit_dewatering_estimate(
    permeability: float,
    head: float,
    area: float,
) -> float:
    """Estimate pit dewatering flow using Darcy's law (simplified).

    A simplified formulation assuming unit gradient:

    .. math::

        Q = K \\times h \\times A

    where *K* is hydraulic conductivity, *h* is hydraulic head, and *A*
    is the cross-sectional seepage area.

    Parameters
    ----------
    permeability : float
        Hydraulic conductivity *K* in m/s.  Must be positive.
    head : float
        Hydraulic head in metres.  Must be positive.
    area : float
        Seepage area in m2.  Must be positive.

    Returns
    -------
    float
        Estimated dewatering flow rate in m3/s.

    Examples
    --------
    >>> pit_dewatering_estimate(1e-5, 0.1, 1000)
    0.001

    References
    ----------
    .. [1] Freeze & Cherry (1979), *Groundwater*, Ch. 2 -- Darcy's law.
    """
    validate_positive(permeability, "permeability")
    validate_positive(head, "head")
    validate_positive(area, "area")
    return permeability * head * area


# ---------------------------------------------------------------------------
# Runoff Coefficient (Rational Method)
# ---------------------------------------------------------------------------


def runoff_coefficient(
    rainfall: float,
    area: float,
    coefficient: float,
) -> float:
    """Estimate surface runoff using the rational method.

    .. math::

        Q = \\frac{C \\times I \\times A}{3\\,600\\,000}

    where *C* is the dimensionless runoff coefficient, *I* is rainfall
    intensity in mm/h, and *A* is catchment area in m2.  The divisor
    converts mm * m2 / h to m3/s.

    Parameters
    ----------
    rainfall : float
        Rainfall intensity in mm/h.  Must be non-negative.
    area : float
        Catchment area in m2.  Must be positive.
    coefficient : float
        Dimensionless runoff coefficient (0--1).

    Returns
    -------
    float
        Peak runoff in m3/s.

    Examples
    --------
    >>> round(runoff_coefficient(10.0, 1000.0, 0.5), 7)
    0.0013889

    References
    ----------
    .. [1] Chow, V.T., Maidment, D.R. & Mays, L.W. (1988). *Applied
           Hydrology*. McGraw-Hill.
    """
    validate_non_negative(rainfall, "rainfall")
    validate_positive(area, "area")
    validate_range(coefficient, 0, 1, "coefficient")
    # mm/h * m2 -> m3/s : divide by 3_600_000
    return coefficient * (rainfall / 1000.0) * area / 3600.0
