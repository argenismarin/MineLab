"""Flyrock distance estimation and safety perimeter calculations.

This module provides empirical flyrock range prediction and safety distance
computation for blast design compliance.

References
----------
.. [1] Lundborg, N., Persson, A., Ladegaard-Pedersen, A. & Holmberg, R.
       (1975). Keeping the lid on flyrock in open-pit blasting.
       *Engineering and Mining Journal*, 176(5), 77-84.
.. [2] Richards, A.B. & Moore, A.J. (2004). Flyrock control -- by chance
       or design. *Proc. 30th ISEE Conf. on Explosives and Blasting
       Technique*, 335-348.
.. [3] IME (Institute of Makers of Explosives) (2011). *Safety Library
       Publication No. 20 -- Safety Guide for the Prevention of Flyrock*.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Flyrock Range -- Lundborg Empirical Model
# ---------------------------------------------------------------------------


def flyrock_range(
    diameter: float,
    burden: float,
    stemming: float,
    charge_conc: float,
) -> float:
    """Estimate maximum flyrock throw distance using the Lundborg model.

    The Lundborg et al. (1975) empirical formula estimates the maximum
    range of collar (vertical) flyrock:

    .. math::

        R = 260 \\, \\frac{(D/1000)^{2/3}}{T^{1/3}} \\,
            \\sqrt{\\frac{q_l}{B}}

    where *D* is the hole diameter (mm, converted to m), *T* is the
    stemming length (m), *q_l* is the linear charge concentration (kg/m),
    and *B* is the burden (m). The constant 260 is an empirical fit
    factor calibrated to SI units.

    Parameters
    ----------
    diameter : float
        Drill-hole diameter *D* in millimetres. Must be positive.
    burden : float
        Burden *B* in metres. Must be positive.
    stemming : float
        Stemming length *T* in metres. Must be positive.
    charge_conc : float
        Linear charge concentration *q_l* in kg/m. Must be positive.

    Returns
    -------
    float
        Estimated maximum flyrock range in metres.

    Examples
    --------
    >>> round(flyrock_range(89, 2.4, 1.68, 5.0), 1)
    62.9

    References
    ----------
    .. [1] Lundborg et al. (1975).
    .. [2] Richards & Moore (2004).
    """
    validate_positive(diameter, "diameter")
    validate_positive(burden, "burden")
    validate_positive(stemming, "stemming")
    validate_positive(charge_conc, "charge_conc")

    d_m = diameter / 1000.0  # mm -> m
    range_m = (
        260.0 * d_m ** (2.0 / 3.0) / stemming ** (1.0 / 3.0) * math.sqrt(charge_conc / burden)
    )

    return range_m


# ---------------------------------------------------------------------------
# Safety Distance
# ---------------------------------------------------------------------------


def safety_distance(flyrock_range_m: float, factor: float = 1.5) -> float:
    """Compute the minimum safety clearance distance around a blast.

    .. math::

        D_{safe} = R_{flyrock} \\times f_{safety}

    Parameters
    ----------
    flyrock_range_m : float
        Estimated maximum flyrock range in metres. Must be positive.
    factor : float, optional
        Safety factor (default 1.5). A factor >= 1.0 is recommended;
        typical practice uses 1.2 to 2.0. Must be >= 1.0.

    Returns
    -------
    float
        Minimum safety distance in metres.

    Examples
    --------
    >>> safety_distance(200.0)
    300.0
    >>> safety_distance(200.0, factor=2.0)
    400.0

    References
    ----------
    .. [1] IME (2011), Safety Library Publication No. 20.
    """
    validate_positive(flyrock_range_m, "flyrock_range_m")
    if factor < 1.0:
        raise ValueError(f"'factor' must be >= 1.0, got {factor}.")

    return flyrock_range_m * factor
