"""Gas and dust dilution airflow requirements for mine ventilation.

This module calculates the minimum airflow quantities needed to dilute
contaminants (diesel exhaust, blasting fumes, methane, and dust) to safe
concentrations in underground mine airways.

References
----------
.. [1] McPherson, M.J. (1993). *Subsurface Ventilation and Environmental
       Engineering*, 1st ed. Chapman & Hall, Chapter 9.
"""

from __future__ import annotations

import math  # noqa: I001

from minelab.utilities.validators import validate_non_negative, validate_positive

# ---------------------------------------------------------------------------
# Air for diesel equipment
# ---------------------------------------------------------------------------


def air_for_diesel(total_kw: float, altitude: float = 0.0) -> float:
    """Compute airflow required to dilute diesel exhaust emissions.

    At sea level the requirement is:

    .. math::

        Q_{\\text{sea}} = 0.06 \\times \\text{kW}

    An altitude correction is applied:

    .. math::

        Q_{\\text{actual}} = Q_{\\text{sea}} \\times \\exp\\!\\left(
        \\frac{\\text{altitude}}{8400}\\right)

    Parameters
    ----------
    total_kw : float
        Total rated power of diesel equipment (kW).  Must be positive.
    altitude : float, optional
        Mine altitude above sea level (m).  Must be non-negative
        (default 0.0).

    Returns
    -------
    float
        Required airflow in m^3/s.

    Raises
    ------
    ValueError
        If *total_kw* is not positive or *altitude* is negative.

    Examples
    --------
    >>> air_for_diesel(200)
    12.0
    >>> round(air_for_diesel(200, 3000), 2)
    17.15

    References
    ----------
    .. [1] McPherson (1993), Ch. 9, Sec. 9.3.
    """
    validate_positive(total_kw, "total_kw")
    validate_non_negative(altitude, "altitude")
    q_sea = 0.06 * total_kw
    return q_sea * math.exp(altitude / 8400.0)


# ---------------------------------------------------------------------------
# Air for blasting
# ---------------------------------------------------------------------------


def air_for_blasting(powder_mass: float, clearance_time: float) -> float:
    """Compute airflow required to clear blasting fumes.

    The gas volume produced is estimated as:

    .. math::

        V_{\\text{gas}} = 0.04 \\times m_{\\text{powder}}

    where *V_gas* is in m^3 at STP per kg of explosive, and the required
    airflow is:

    .. math::

        Q = \\frac{V_{\\text{gas}}}{t_{\\text{clearance}}}

    Parameters
    ----------
    powder_mass : float
        Mass of explosive used (kg).  Must be positive.
    clearance_time : float
        Time allowed for gas clearance (s).  Must be positive.

    Returns
    -------
    float
        Required airflow in m^3/s.

    Raises
    ------
    ValueError
        If *powder_mass* or *clearance_time* is not positive.

    Examples
    --------
    >>> round(air_for_blasting(100, 1800), 6)
    0.002222

    References
    ----------
    .. [1] McPherson (1993), Ch. 9, Sec. 9.5.
    """
    validate_positive(powder_mass, "powder_mass")
    validate_positive(clearance_time, "clearance_time")
    gas_volume = 0.04 * powder_mass
    return gas_volume / clearance_time


# ---------------------------------------------------------------------------
# Methane dilution
# ---------------------------------------------------------------------------


def methane_dilution(emission_rate: float, target_conc: float) -> float:
    """Compute airflow required to dilute methane to a target concentration.

    .. math::

        Q = \\frac{q_{\\text{CH}_4}}{C_{\\text{target}}}

    where *q_CH4* is the methane emission rate (m^3/s) and *C_target* is the
    maximum allowable concentration expressed as a fraction (e.g. 0.01 for
    1 %).

    Parameters
    ----------
    emission_rate : float
        Methane emission rate (m^3/s).  Must be positive.
    target_conc : float
        Target maximum methane concentration as a decimal fraction
        (e.g. 0.01 for 1 %).  Must be positive and < 1.

    Returns
    -------
    float
        Required airflow in m^3/s.

    Raises
    ------
    ValueError
        If *emission_rate* is not positive, or *target_conc* is not in
        (0, 1).

    Examples
    --------
    >>> methane_dilution(0.5, 0.01)
    50.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 9, Sec. 9.2.
    """
    validate_positive(emission_rate, "emission_rate")
    validate_positive(target_conc, "target_conc")
    if target_conc >= 1.0:
        raise ValueError(f"'target_conc' must be less than 1.0, got {target_conc}.")
    return emission_rate / target_conc


# ---------------------------------------------------------------------------
# Dust dilution
# ---------------------------------------------------------------------------


def dust_dilution(dust_rate: float, tlv: float) -> float:
    """Compute airflow required to dilute dust to the threshold limit value.

    .. math::

        Q = \\frac{\\dot{m}_{\\text{dust}}}{\\text{TLV}}

    Parameters
    ----------
    dust_rate : float
        Dust generation rate (mg/s).  Must be positive.
    tlv : float
        Threshold limit value for dust concentration (mg/m^3).  Must be
        positive.

    Returns
    -------
    float
        Required airflow in m^3/s.

    Raises
    ------
    ValueError
        If *dust_rate* or *tlv* is not positive.

    Examples
    --------
    >>> dust_dilution(10, 2)
    5.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 9, Sec. 9.7.
    """
    validate_positive(dust_rate, "dust_rate")
    validate_positive(tlv, "tlv")
    return dust_rate / tlv
