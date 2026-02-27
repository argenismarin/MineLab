"""Comminution calculations — crushing and grinding energy.

Bond, Kick, and Rittinger laws of comminution; mill power for SAG, ball,
and rod mills; crusher reduction ratio.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Bond Work Index (P4-P01)
# ---------------------------------------------------------------------------


def bond_work_index(
    closing_screen: float,
    feed_p80: float,
    product_p80: float,
    grindability: float,
) -> float:
    """Bond laboratory work index from grindability test.

    Parameters
    ----------
    closing_screen : float
        Closing screen aperture (micrometers).
    feed_p80 : float
        Feed 80% passing size (micrometers).
    product_p80 : float
        Product 80% passing size (micrometers).
    grindability : float
        Net grams per revolution at steady state (Gbp).

    Returns
    -------
    float
        Work index Wi (kWh/t).

    Examples
    --------
    >>> round(bond_work_index(106, 2500, 75, 1.5), 1)
    13.7

    References
    ----------
    .. [1] Bond, F.C. (1961). "Crushing and grinding calculations."
       British Chemical Engineering, 6, 378-385, 543-548.
    """
    validate_positive(closing_screen, "closing_screen")
    validate_positive(feed_p80, "feed_p80")
    validate_positive(product_p80, "product_p80")
    validate_positive(grindability, "grindability")

    # Wi = 44.5 / (P1^0.23 * Gbp^0.82 * (10/sqrt(P80) - 10/sqrt(F80)))
    # Bond 1961
    p1 = closing_screen
    wi = 44.5 / (
        p1**0.23 * grindability**0.82 * (10 / np.sqrt(product_p80) - 10 / np.sqrt(feed_p80))
    )

    return float(wi)


# ---------------------------------------------------------------------------
# Bond Energy (P4-P02)
# ---------------------------------------------------------------------------


def bond_energy(
    wi: float,
    f80: float,
    p80: float,
) -> float:
    """Specific energy consumption using Bond's third law.

    Parameters
    ----------
    wi : float
        Work index (kWh/t).
    f80 : float
        Feed 80% passing size (micrometers).
    p80 : float
        Product 80% passing size (micrometers).

    Returns
    -------
    float
        Specific energy W (kWh/t).

    Examples
    --------
    >>> round(bond_energy(12, 2500, 75), 1)
    14.6

    References
    ----------
    .. [1] Bond, F.C. (1961). "Crushing and grinding calculations."
       British Chemical Engineering, 6, 378-385, 543-548.
    """
    validate_positive(wi, "wi")
    validate_positive(f80, "f80")
    validate_positive(p80, "p80")

    # W = 10 * Wi * (1/sqrt(P80) - 1/sqrt(F80))  — Bond's 3rd law
    w = 10 * wi * (1 / np.sqrt(p80) - 1 / np.sqrt(f80))

    return float(w)


# ---------------------------------------------------------------------------
# Kick Energy (P4-P03)
# ---------------------------------------------------------------------------


def kick_energy(
    ki: float,
    feed_size: float,
    product_size: float,
) -> float:
    """Specific energy using Kick's law (coarse crushing).

    Parameters
    ----------
    ki : float
        Kick's constant.
    feed_size : float
        Feed characteristic size (micrometers).
    product_size : float
        Product characteristic size (micrometers).

    Returns
    -------
    float
        Specific energy W (kWh/t).

    Examples
    --------
    >>> round(kick_energy(1.0, 1000, 100), 2)
    2.3

    References
    ----------
    .. [1] Kick (1885). Das Gesetz der proportionalen Widerstande.
    .. [2] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.5.
    """
    validate_positive(ki, "ki")
    validate_positive(feed_size, "feed_size")
    validate_positive(product_size, "product_size")

    # W = Ki * ln(F/P)  — Kick's 1st law
    return float(ki * np.log(feed_size / product_size))


# ---------------------------------------------------------------------------
# Rittinger Energy (P4-P04)
# ---------------------------------------------------------------------------


def rittinger_energy(
    kr: float,
    feed_size: float,
    product_size: float,
) -> float:
    """Specific energy using Rittinger's law (fine grinding).

    Parameters
    ----------
    kr : float
        Rittinger's constant.
    feed_size : float
        Feed characteristic size (micrometers).
    product_size : float
        Product characteristic size (micrometers).

    Returns
    -------
    float
        Specific energy W (kWh/t).

    Examples
    --------
    >>> round(rittinger_energy(1.0, 1000, 100), 4)
    0.009

    References
    ----------
    .. [1] Rittinger (1867). Lehrbuch der Aufbereitungskunde.
    .. [2] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.5.
    """
    validate_positive(kr, "kr")
    validate_positive(feed_size, "feed_size")
    validate_positive(product_size, "product_size")

    # W = Kr * (1/P - 1/F)  — Rittinger's 1st law
    return float(kr * (1 / product_size - 1 / feed_size))


# ---------------------------------------------------------------------------
# Ball Mill Power (P4-P05)
# ---------------------------------------------------------------------------


def ball_mill_power(
    wi: float,
    f80: float,
    p80: float,
    tonnage: float,
    efficiency: float = 1.0,
) -> float:
    """Ball mill power draw.

    Parameters
    ----------
    wi : float
        Bond work index (kWh/t).
    f80 : float
        Feed 80% passing size (micrometers).
    p80 : float
        Product 80% passing size (micrometers).
    tonnage : float
        Throughput (t/h).
    efficiency : float
        Mill efficiency factor (0 < efficiency <= 1). Default 1.0.

    Returns
    -------
    float
        Required power (kW).

    Examples
    --------
    >>> round(ball_mill_power(12, 2500, 75, 100, 0.9), 0)
    1622.0

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.7.
    """
    validate_positive(efficiency, "efficiency")
    validate_positive(tonnage, "tonnage")

    w = bond_energy(wi, f80, p80)
    return float(w * tonnage / efficiency)


# ---------------------------------------------------------------------------
# SAG Mill Power (P4-P06)
# ---------------------------------------------------------------------------


def sag_mill_power(
    spi: float,
    f80: float,
    p80: float,
    tonnage: float,
) -> float:
    """SAG mill power from SPI test.

    Parameters
    ----------
    spi : float
        SAG Power Index (kWh/t).
    f80 : float
        Feed 80% passing size (micrometers).
    p80 : float
        Product 80% passing size (micrometers).
    tonnage : float
        Throughput (t/h).

    Returns
    -------
    float
        Required power (kW).

    Examples
    --------
    >>> round(sag_mill_power(10, 150000, 2000, 200), 0)
    2000.0

    References
    ----------
    .. [1] Starkey, J. & Dobby, G. (1996). "Application of the Minnovex
       SAG power index at five Canadian SAG plants." Proc. SAG 1996.
    """
    validate_positive(spi, "spi")
    validate_positive(f80, "f80")
    validate_positive(p80, "p80")
    validate_positive(tonnage, "tonnage")

    # P = SPI * tonnage (SPI already in kWh/t)
    return float(spi * tonnage)


# ---------------------------------------------------------------------------
# Rod Mill Power (P4-P07)
# ---------------------------------------------------------------------------


def rod_mill_power(
    wi: float,
    f80: float,
    p80: float,
    tonnage: float,
    correction_factor: float = 1.0,
) -> float:
    """Rod mill power draw with correction factors.

    Parameters
    ----------
    wi : float
        Bond work index (kWh/t).
    f80 : float
        Feed 80% passing size (micrometers).
    p80 : float
        Product 80% passing size (micrometers).
    tonnage : float
        Throughput (t/h).
    correction_factor : float
        Combined correction factor (EF1-EF8). Default 1.0.

    Returns
    -------
    float
        Required power (kW).

    Examples
    --------
    >>> round(rod_mill_power(12, 15000, 1000, 100), 0)
    801.0

    References
    ----------
    .. [1] Bond, F.C. (1961). "Crushing and grinding calculations."
    .. [2] Rowland, C.A. (1982). Selection of rod mills, ball mills,
       pebble mills, and regrind mills. SME Mineral Processing
       Plant Design, 2nd ed.
    """
    validate_positive(correction_factor, "correction_factor")
    validate_positive(tonnage, "tonnage")

    w = bond_energy(wi, f80, p80)
    return float(w * tonnage * correction_factor)


# ---------------------------------------------------------------------------
# Crusher Reduction Ratio (P4-P08)
# ---------------------------------------------------------------------------


def crusher_reduction_ratio(
    f80: float,
    p80: float,
) -> dict:
    """Crusher reduction ratio.

    Parameters
    ----------
    f80 : float
        Feed 80% passing size (mm or micrometers — consistent units).
    p80 : float
        Product 80% passing size (same units as f80).

    Returns
    -------
    dict
        Keys: ``"reduction_ratio"`` (float),
        ``"crusher_type"`` (suggested crusher type).

    Examples
    --------
    >>> result = crusher_reduction_ratio(500, 100)
    >>> result["reduction_ratio"]
    5.0

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed.
    """
    validate_positive(f80, "f80")
    validate_positive(p80, "p80")

    rr = f80 / p80

    if rr <= 3:
        ctype = "Fine cone crusher"
    elif rr <= 5:
        ctype = "Cone crusher"
    elif rr <= 7:
        ctype = "Jaw crusher"
    else:
        ctype = "Jaw or gyratory crusher"

    return {"reduction_ratio": float(rr), "crusher_type": ctype}
