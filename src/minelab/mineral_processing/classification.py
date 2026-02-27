"""Particle classification and size distribution models.

Plitt hydrocyclone model, screen efficiency, Lynch-Rao partition curve,
Tromp curve, Rosin-Rammler and Gates-Gaudin-Schuhmann distributions.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive


def plitt_model(
    dc: float,
    di: float,
    do: float,
    h: float,
    du: float,
    q: float,
    phi_f: float,
    rho_s: float,
    rho_l: float = 1000.0,
) -> dict:
    """Plitt hydrocyclone model for d50c estimation.

    Parameters
    ----------
    dc : float
        Cyclone diameter (m).
    di : float
        Inlet diameter (m).
    do : float
        Overflow (vortex finder) diameter (m).
    h : float
        Free vortex height (m).
    du : float
        Underflow (spigot) diameter (m).
    q : float
        Volumetric feed rate (m^3/h).
    phi_f : float
        Feed solids volume fraction (0-1).
    rho_s : float
        Solids density (kg/m^3).
    rho_l : float
        Liquid density (kg/m^3). Default 1000.

    Returns
    -------
    dict
        Keys: ``"d50c"`` (corrected cut size in micrometers),
        ``"pressure_drop"`` (kPa, estimated).

    Examples
    --------
    >>> result = plitt_model(0.25, 0.075, 0.1, 0.3, 0.05, 50, 0.1, 2700)
    >>> result["d50c"] > 0
    True

    References
    ----------
    .. [1] Plitt, L.R. (1976). "A mathematical model of the hydrocyclone
       classifier." CIM Bulletin, 69(776), 114-123.
    """
    validate_positive(dc, "dc")
    validate_positive(di, "di")
    validate_positive(do, "do")
    validate_positive(h, "h")
    validate_positive(du, "du")
    validate_positive(q, "q")
    validate_positive(rho_s, "rho_s")

    # Plitt d50c equation (simplified, meters → micrometers)
    # d50c = 14.8 * Dc^0.46 * Di^0.6 * Do^1.21 * exp(0.063*phi)
    #        / (Du^0.71 * h^0.38 * Q^0.45 * (rho_s - rho_l)^0.5)
    d50c = (
        14.8
        * dc**0.46
        * di**0.6
        * do**1.21
        * np.exp(0.063 * phi_f * 100)
        / (du**0.71 * h**0.38 * q**0.45 * (rho_s - rho_l) ** 0.5)
    )

    # Convert to micrometers (multiply by 1e6 since inputs are in meters)
    d50c_um = d50c * 1e6

    # Rough pressure drop estimate (kPa)
    pressure = 0.1 * q**2 / (dc**2 * di**2)

    return {"d50c": float(d50c_um), "pressure_drop": float(pressure)}


def screen_efficiency(
    feed_mass: float,
    oversize_mass: float,
    undersize_in_oversize: float,
    oversize_in_undersize: float,
) -> float:
    """Screen efficiency calculation.

    Parameters
    ----------
    feed_mass : float
        Total feed mass.
    oversize_mass : float
        Oversize product mass.
    undersize_in_oversize : float
        Fraction of undersize material in the oversize product.
    oversize_in_undersize : float
        Fraction of oversize material in the undersize product.

    Returns
    -------
    float
        Screen efficiency (fraction, 0-1).

    Examples
    --------
    >>> round(screen_efficiency(1000, 600, 0.05, 0.03), 3)
    0.923

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.8.
    """
    validate_positive(feed_mass, "feed_mass")
    validate_positive(oversize_mass, "oversize_mass")

    # Efficiency = (1 - undersize_in_oversize) * (1 - oversize_in_undersize)
    e_o = 1 - undersize_in_oversize
    e_u = 1 - oversize_in_undersize

    return float(e_o * e_u)


def lynch_rao_partition(
    d50c: float,
    alpha: float,
    sizes: np.ndarray,
) -> np.ndarray:
    """Lynch-Rao partition curve for hydrocyclone classification.

    Parameters
    ----------
    d50c : float
        Corrected cut size (micrometers).
    alpha : float
        Sharpness of separation parameter (higher = sharper).
    sizes : np.ndarray
        Particle sizes (micrometers).

    Returns
    -------
    np.ndarray
        Partition coefficients (fraction to underflow, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> y = lynch_rao_partition(75, 3.0, np.array([75]))
    >>> round(float(y[0]), 1)
    0.5

    References
    ----------
    .. [1] Lynch, A.J. & Rao, T.C. (1975). "Modelling and scale-up of
       hydrocyclone classifiers." Proc. 11th IMPC, Cagliari.
    """
    validate_positive(d50c, "d50c")
    validate_positive(alpha, "alpha")

    sizes = np.asarray(sizes, dtype=float)

    # Y = exp(alpha*d/d50c) / (exp(alpha*d/d50c) + exp(alpha))
    # Simplifies to: Y = 1 / (1 + exp(alpha * (1 - d/d50c)))
    y = 1 / (1 + np.exp(alpha * (1 - sizes / d50c)))

    return y


def tromp_curve(
    feed_psd: np.ndarray,
    overflow_psd: np.ndarray,
    underflow_psd: np.ndarray,
    feed_split: float = 0.5,
) -> np.ndarray:
    """Tromp partition curve from PSD data.

    Parameters
    ----------
    feed_psd : np.ndarray
        Feed particle size distribution (fractions per size class).
    overflow_psd : np.ndarray
        Overflow PSD (fractions per size class).
    underflow_psd : np.ndarray
        Underflow PSD (fractions per size class).
    feed_split : float
        Fraction of feed reporting to underflow (0-1). Default 0.5.

    Returns
    -------
    np.ndarray
        Partition coefficients per size class (fraction to underflow).

    Examples
    --------
    >>> import numpy as np
    >>> feed = np.array([0.1, 0.3, 0.4, 0.2])
    >>> uf = np.array([0.02, 0.1, 0.48, 0.4])
    >>> of = np.array([0.18, 0.5, 0.32, 0.0])
    >>> pc = tromp_curve(feed, of, uf, 0.5)
    >>> all(0 <= p <= 1 for p in pc)
    True

    References
    ----------
    .. [1] Tromp, K.F. (1937). Neue Wege fur die Beurteilung der
       Aufbereitung von Steinkohlen.
    """
    feed_psd = np.asarray(feed_psd, dtype=float)
    underflow_psd = np.asarray(underflow_psd, dtype=float)

    # Partition coefficient = (feed_split * u_i) / f_i
    feed_psd_safe = np.where(feed_psd > 0, feed_psd, 1e-10)
    partition = feed_split * underflow_psd / feed_psd_safe

    return np.clip(partition, 0, 1)


def rosin_rammler(
    sizes: np.ndarray,
    k: float,
    n: float,
) -> np.ndarray:
    """Rosin-Rammler particle size distribution.

    Parameters
    ----------
    sizes : np.ndarray
        Particle sizes (any consistent units).
    k : float
        Size parameter (characteristic size, same units as sizes).
    n : float
        Distribution modulus (shape parameter).

    Returns
    -------
    np.ndarray
        Cumulative fraction passing F(x).

    Examples
    --------
    >>> import numpy as np
    >>> f = rosin_rammler(np.array([100.0]), 100.0, 1.5)
    >>> round(float(f[0]), 3)
    0.632

    References
    ----------
    .. [1] Rosin, P. & Rammler, E. (1933). "Laws governing the fineness
       of powdered coal." J. Inst. Fuel, 7, 29-36.
    """
    validate_positive(k, "k")
    validate_positive(n, "n")

    sizes = np.asarray(sizes, dtype=float)

    # F(x) = 1 - exp(-(x/k)^n)
    return 1 - np.exp(-((sizes / k) ** n))


def gates_gaudin_schuhmann(
    sizes: np.ndarray,
    k: float,
    m: float,
) -> np.ndarray:
    """Gates-Gaudin-Schuhmann particle size distribution.

    Parameters
    ----------
    sizes : np.ndarray
        Particle sizes (any consistent units).
    k : float
        Maximum particle size (same units as sizes).
    m : float
        Distribution modulus.

    Returns
    -------
    np.ndarray
        Cumulative fraction passing F(x).

    Examples
    --------
    >>> import numpy as np
    >>> f = gates_gaudin_schuhmann(np.array([100.0]), 100.0, 0.5)
    >>> round(float(f[0]), 1)
    1.0

    References
    ----------
    .. [1] Gaudin, A.M. (1926); Schuhmann, R. (1940).
    """
    validate_positive(k, "k")
    validate_positive(m, "m")

    sizes = np.asarray(sizes, dtype=float)

    # F(x) = (x/k)^m for x <= k, else 1
    f = np.where(sizes <= k, (sizes / k) ** m, 1.0)

    return np.clip(f, 0, 1)
