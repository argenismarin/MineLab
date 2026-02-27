"""Fan similarity (affinity) laws and specific speed calculations.

This module implements the fan affinity laws that relate changes in fan
speed and impeller diameter to the resulting changes in airflow, pressure,
and power, as well as the dimensionless specific speed used for fan type
selection.

References
----------
.. [1] McPherson, M.J. (1993). *Subsurface Ventilation and Environmental
       Engineering*, 1st ed. Chapman & Hall, Chapter 10.
"""

from __future__ import annotations

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Fan affinity laws
# ---------------------------------------------------------------------------


def fan_affinity_laws(  # noqa: N803
    Q1: float,  # noqa: N803
    P1: float,  # noqa: N803
    Power1: float,  # noqa: N803
    n1: float,
    n2: float,
    D1: float = 1.0,  # noqa: N803
    D2: float = 1.0,  # noqa: N803
) -> dict:
    """Apply the fan affinity (similarity) laws.

    The three affinity laws relate two operating states of geometrically
    similar fans:

    .. math::

        \\frac{Q_2}{Q_1} = \\frac{n_2}{n_1} \\left(\\frac{D_2}{D_1}\\right)^3

    .. math::

        \\frac{P_2}{P_1} = \\left(\\frac{n_2}{n_1}\\right)^2
                           \\left(\\frac{D_2}{D_1}\\right)^2

    .. math::

        \\frac{\\text{Power}_2}{\\text{Power}_1} =
        \\left(\\frac{n_2}{n_1}\\right)^3
        \\left(\\frac{D_2}{D_1}\\right)^5

    Parameters
    ----------
    Q1 : float
        Original airflow rate (m^3/s).  Must be positive.
    P1 : float
        Original fan pressure (Pa).  Must be positive.
    Power1 : float
        Original fan power (W).  Must be positive.
    n1 : float
        Original rotational speed (rpm).  Must be positive.
    n2 : float
        New rotational speed (rpm).  Must be positive.
    D1 : float, optional
        Original impeller diameter (m).  Must be positive (default 1.0).
    D2 : float, optional
        New impeller diameter (m).  Must be positive (default 1.0).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"Q2"`` : float -- New airflow rate (m^3/s).
        - ``"P2"`` : float -- New fan pressure (Pa).
        - ``"Power2"`` : float -- New fan power (W).

    Raises
    ------
    ValueError
        If any parameter is not positive.

    Examples
    --------
    Doubling the speed of a fan:

    >>> result = fan_affinity_laws(50, 1000, 5000, n1=600, n2=1200)
    >>> result["Q2"]
    100.0
    >>> result["P2"]
    4000.0
    >>> result["Power2"]
    40000.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 10, Sec. 10.4.
    """
    validate_positive(Q1, "Q1")
    validate_positive(P1, "P1")
    validate_positive(Power1, "Power1")
    validate_positive(n1, "n1")
    validate_positive(n2, "n2")
    validate_positive(D1, "D1")
    validate_positive(D2, "D2")

    speed_ratio = n2 / n1
    diam_ratio = D2 / D1

    q2 = Q1 * speed_ratio * diam_ratio**3
    p2 = P1 * speed_ratio**2 * diam_ratio**2
    power2 = Power1 * speed_ratio**3 * diam_ratio**5

    return {
        "Q2": float(q2),
        "P2": float(p2),
        "Power2": float(power2),
    }


# ---------------------------------------------------------------------------
# Specific speed
# ---------------------------------------------------------------------------


def specific_speed(rpm: float, Q: float, P: float) -> float:  # noqa: N803
    """Compute the dimensionless specific speed of a fan.

    .. math::

        N_s = N \\cdot \\frac{Q^{0.5}}{P^{0.75}}

    where *N* is the rotational speed in rev/s (rpm / 60).

    Specific speed is used for fan type selection:

    - Low Ns (< 1): centrifugal fans
    - Medium Ns (1--3): mixed-flow fans
    - High Ns (> 3): axial fans

    Parameters
    ----------
    rpm : float
        Rotational speed in revolutions per minute.  Must be positive.
    Q : float
        Volume airflow rate (m^3/s).  Must be positive.
    P : float
        Fan total pressure (Pa).  Must be positive.

    Returns
    -------
    float
        Dimensionless specific speed.

    Raises
    ------
    ValueError
        If any parameter is not positive.

    Examples
    --------
    >>> round(specific_speed(1200, 50, 2000), 4)
    0.4729

    References
    ----------
    .. [1] McPherson (1993), Ch. 10, Sec. 10.6.
    """
    validate_positive(rpm, "rpm")
    validate_positive(Q, "Q")
    validate_positive(P, "P")

    n_revs = rpm / 60.0  # Convert to rev/s
    return n_revs * Q**0.5 / P**0.75
