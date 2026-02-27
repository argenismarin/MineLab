"""Blast vibration prediction and regulatory compliance.

This module provides functions for computing peak particle velocity (PPV),
USBM scaled distance, and checking vibration levels against industry
standards (OSMRE, DIN 4150-3).

References
----------
.. [1] Siskind, D.E. et al. (1980). *Structure Response and Damage
       Produced by Ground Vibrations from Surface Blasting*. USBM
       RI 8507.
.. [2] Dowding, C.H. (1996). *Construction Vibrations*. Prentice Hall.
.. [3] DIN 4150-3 (1999). *Structural vibration -- Part 3: Effects of
       vibration on structures*.
.. [4] OSMRE (Office of Surface Mining Reclamation and Enforcement),
       30 CFR Part 816/817.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import validate_non_negative, validate_positive

# ---------------------------------------------------------------------------
# USBM Scaled Distance
# ---------------------------------------------------------------------------


def usbm_scaled_distance(distance: float, charge: float) -> float:
    """Compute the USBM square-root scaled distance.

    .. math::

        SD = \\frac{D}{\\sqrt{W}}

    where *D* is the distance from the blast (m) and *W* is the maximum
    charge weight per delay (kg).

    Parameters
    ----------
    distance : float
        Distance from the blast in metres. Must be positive.
    charge : float
        Maximum charge per delay in kg. Must be positive.

    Returns
    -------
    float
        Scaled distance (m/kg^0.5).

    Examples
    --------
    >>> round(usbm_scaled_distance(100, 50), 2)
    14.14

    References
    ----------
    .. [1] USBM RI 8507, Siskind et al. (1980), Eq. 1.
    """
    validate_positive(distance, "distance")
    validate_positive(charge, "charge")

    return distance / math.sqrt(charge)


# ---------------------------------------------------------------------------
# Peak Particle Velocity (PPV)
# ---------------------------------------------------------------------------


def ppv_scaled_distance(
    k_site: float,
    charge: float,
    distance: float,
    beta: float,
) -> dict:
    """Predict peak particle velocity using the scaled-distance law.

    .. math::

        PPV = K \\left(\\frac{D}{\\sqrt{W}}\\right)^{-\\beta}

    where *K* is a site-specific constant, *D* is distance (m), *W* is
    maximum charge per delay (kg), and *beta* is the attenuation
    exponent.

    Parameters
    ----------
    k_site : float
        Site constant *K* (intercept). Must be positive. Typical values
        range from 100 to 5000 depending on geology and units.
    charge : float
        Maximum charge per delay *W* in kg. Must be positive.
    distance : float
        Distance from the blast *D* in metres. Must be positive.
    beta : float
        Attenuation exponent (slope). Must be positive. Typical range
        1.0 to 2.0; commonly ~1.6 for hard rock.

    Returns
    -------
    dict
        Dictionary with:

        - ``ppv`` : float -- Peak particle velocity in mm/s.
        - ``scaled_distance`` : float -- USBM scaled distance (m/kg^0.5).

    Examples
    --------
    >>> result = ppv_scaled_distance(1140, 50, 100, 1.6)
    >>> round(result['scaled_distance'], 2)
    14.14
    >>> round(result['ppv'], 1)
    16.4

    References
    ----------
    .. [1] USBM RI 8507, Siskind et al. (1980).
    .. [2] Dowding (1996), Ch. 4.
    """
    validate_positive(k_site, "k_site")
    validate_positive(charge, "charge")
    validate_positive(distance, "distance")
    validate_positive(beta, "beta")

    sd = usbm_scaled_distance(distance, charge)
    ppv = k_site * sd ** (-beta)

    return {
        "ppv": ppv,
        "scaled_distance": sd,
    }


# ---------------------------------------------------------------------------
# Vibration Compliance Check
# ---------------------------------------------------------------------------


def vibration_compliance(
    ppv: float,
    frequency: float = 0.0,
    standard: str = "OSMRE",
) -> dict:
    """Check blast vibration against regulatory standards.

    Supported standards:

    - **OSMRE** (default): 25.4 mm/s (1 in/s) for low-frequency blasts
      (< 10 Hz), 50.8 mm/s for mid-frequency (10--40 Hz), and 50.8 mm/s
      for high-frequency (> 40 Hz). When *frequency* is 0 (unknown), the
      most conservative limit of 25.4 mm/s is used.
    - **DIN4150**: Frequency-dependent limits for residential buildings
      per DIN 4150-3 Table 3, Line 3 (most sensitive class):
      5 mm/s (< 10 Hz), 5--15 mm/s (10--50 Hz, linearly interpolated),
      15--20 mm/s (50--100 Hz).

    Parameters
    ----------
    ppv : float
        Measured or predicted peak particle velocity in mm/s.
        Must be non-negative.
    frequency : float, optional
        Dominant vibration frequency in Hz (default 0.0 = unknown).
        Must be non-negative.
    standard : str, optional
        Regulatory standard to check against. One of ``"OSMRE"`` or
        ``"DIN4150"`` (default ``"OSMRE"``).

    Returns
    -------
    dict
        Dictionary with:

        - ``compliant`` : bool -- Whether the PPV meets the standard.
        - ``ppv`` : float -- Input PPV (mm/s).
        - ``limit`` : float -- Applicable PPV limit (mm/s).
        - ``standard`` : str -- Standard used.
        - ``frequency`` : float -- Input frequency (Hz).

    Raises
    ------
    ValueError
        If *standard* is not recognized.

    Examples
    --------
    >>> result = vibration_compliance(20.0, standard="OSMRE")
    >>> result['compliant']
    True
    >>> result['limit']
    25.4

    >>> result = vibration_compliance(30.0, frequency=5.0, standard="OSMRE")
    >>> result['compliant']
    False

    References
    ----------
    .. [1] OSMRE, 30 CFR Part 816/817.
    .. [2] DIN 4150-3 (1999), Table 3.
    """
    validate_non_negative(ppv, "ppv")
    validate_non_negative(frequency, "frequency")

    standard_upper = standard.upper()

    if standard_upper == "OSMRE":
        if frequency <= 0:
            # Unknown frequency -- use most conservative limit
            limit = 25.4
        elif frequency < 10:
            limit = 25.4
        elif frequency <= 40:
            limit = 50.8
        else:
            limit = 50.8
    elif standard_upper == "DIN4150":
        if frequency <= 0:
            # Unknown frequency -- use most conservative limit
            limit = 5.0
        elif frequency < 10:
            limit = 5.0
        elif frequency <= 50:
            # Linear interpolation from 5 mm/s at 10 Hz to 15 mm/s at 50 Hz
            limit = 5.0 + (frequency - 10.0) * (15.0 - 5.0) / (50.0 - 10.0)
        elif frequency <= 100:
            # Linear interpolation from 15 mm/s at 50 Hz to 20 mm/s at 100 Hz
            limit = 15.0 + (frequency - 50.0) * (20.0 - 15.0) / (100.0 - 50.0)
        else:
            limit = 20.0
    else:
        raise ValueError(f"Unknown standard '{standard}'. Supported: 'OSMRE', 'DIN4150'.")

    return {
        "compliant": ppv <= limit,
        "ppv": ppv,
        "limit": limit,
        "standard": standard_upper,
        "frequency": frequency,
    }
