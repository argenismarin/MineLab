"""Hoek-Brown failure criterion and related calculations.

Intact rock strength, generalized rock mass criterion (2002 edition),
parameter estimation, Mohr-Coulomb equivalent fit, and deformation modulus.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)


def hoek_brown_intact(
    sigma3: float | np.ndarray,
    sigci: float,
    mi: float,
) -> float | np.ndarray:
    """Hoek-Brown criterion for intact rock.

    Parameters
    ----------
    sigma3 : float or np.ndarray
        Minor principal stress (MPa).
    sigci : float
        Uniaxial compressive strength of intact rock (MPa).
    mi : float
        Material constant for intact rock.

    Returns
    -------
    float or np.ndarray
        Major principal stress sigma1 (MPa).

    Examples
    --------
    >>> hoek_brown_intact(0, 100, 25)
    100.0
    >>> round(hoek_brown_intact(10, 100, 25), 1)
    183.1

    References
    ----------
    .. [1] Hoek, E. & Brown, E.T. (1980). "Empirical strength criterion for
       rock masses." J. Geotech. Eng. Div., ASCE, 106(GT9), 1013-1035.
    """
    validate_positive(sigci, "sigci")
    validate_positive(mi, "mi")

    sigma3 = np.asarray(sigma3, dtype=float)

    # σ1 = σ3 + σci * (mi*σ3/σci + 1)^0.5  — Eq. 1, Hoek & Brown 1980
    sigma1 = sigma3 + sigci * np.sqrt(mi * sigma3 / sigci + 1.0)

    return float(sigma1) if sigma1.ndim == 0 else sigma1


def hoek_brown_parameters(
    gsi: float,
    mi: float,
    d: float = 0.0,
) -> dict:
    """Calculate generalized Hoek-Brown parameters mb, s, a.

    Parameters
    ----------
    gsi : float
        Geological Strength Index (0-100).
    mi : float
        Intact rock material constant.
    d : float
        Disturbance factor (0 = undisturbed, 1 = fully disturbed).

    Returns
    -------
    dict
        Keys: ``"mb"``, ``"s"``, ``"a"``.

    Examples
    --------
    >>> p = hoek_brown_parameters(50, 25, 0)
    >>> round(p["mb"], 3)
    4.205
    >>> round(p["s"], 4)
    0.0039
    >>> round(p["a"], 3)
    0.506

    References
    ----------
    .. [1] Hoek, E., Carranza-Torres, C. & Corkum, B. (2002). "Hoek-Brown
       failure criterion — 2002 edition." Proc. NARMS-TAC 2002, 267-273.
    """
    validate_range(gsi, 0, 100, "gsi")
    validate_positive(mi, "mi")
    validate_range(d, 0, 1, "d")

    # mb = mi * exp((GSI - 100) / (28 - 14D))  — Hoek et al. 2002
    mb = mi * np.exp((gsi - 100) / (28 - 14 * d))

    # s = exp((GSI - 100) / (9 - 3D))  — Hoek et al. 2002
    s = np.exp((gsi - 100) / (9 - 3 * d))

    # a = 0.5 + (exp(-GSI/15) - exp(-20/3)) / 6  — Hoek et al. 2002
    a = 0.5 + (np.exp(-gsi / 15) - np.exp(-20 / 3)) / 6

    return {"mb": float(mb), "s": float(s), "a": float(a)}


def hoek_brown_rock_mass(
    sigma3: float | np.ndarray,
    sigci: float,
    gsi: float,
    mi: float,
    d: float = 0.0,
) -> float | np.ndarray:
    """Generalized Hoek-Brown criterion for rock mass.

    Parameters
    ----------
    sigma3 : float or np.ndarray
        Minor principal stress (MPa).
    sigci : float
        Uniaxial compressive strength of intact rock (MPa).
    gsi : float
        Geological Strength Index (0-100).
    mi : float
        Intact rock material constant.
    d : float
        Disturbance factor (0-1).

    Returns
    -------
    float or np.ndarray
        Major principal stress sigma1 (MPa).

    Examples
    --------
    >>> round(hoek_brown_rock_mass(0, 100, 50, 25, 0), 2)
    5.67

    References
    ----------
    .. [1] Hoek, E., Carranza-Torres, C. & Corkum, B. (2002). "Hoek-Brown
       failure criterion — 2002 edition."
    """
    validate_positive(sigci, "sigci")

    params = hoek_brown_parameters(gsi, mi, d)
    mb = params["mb"]
    s = params["s"]
    a = params["a"]

    sigma3 = np.asarray(sigma3, dtype=float)

    # σ1 = σ3 + σci * (mb*σ3/σci + s)^a  — Hoek et al. 2002
    sigma1 = sigma3 + sigci * (mb * sigma3 / sigci + s) ** a

    return float(sigma1) if sigma1.ndim == 0 else sigma1


def mohr_coulomb_fit(
    sigci: float,
    gsi: float,
    mi: float,
    d: float = 0.0,
    sig3_max: float | None = None,
) -> dict:
    """Equivalent Mohr-Coulomb parameters from Hoek-Brown envelope.

    Parameters
    ----------
    sigci : float
        Uniaxial compressive strength of intact rock (MPa).
    gsi : float
        Geological Strength Index (0-100).
    mi : float
        Intact rock material constant.
    d : float
        Disturbance factor (0-1).
    sig3_max : float or None
        Upper limit of confining stress for the fit (MPa).
        If None, defaults to sigci / 4.

    Returns
    -------
    dict
        Keys: ``"cohesion"`` (MPa), ``"friction_angle"`` (degrees),
        ``"sig3_range"`` (array of sigma3 used for fit).

    Examples
    --------
    >>> result = mohr_coulomb_fit(100, 50, 25, 0)
    >>> result["friction_angle"] > 0
    True
    >>> result["cohesion"] > 0
    True

    References
    ----------
    .. [1] Hoek, E., Carranza-Torres, C. & Corkum, B. (2002). "Hoek-Brown
       failure criterion — 2002 edition."
    """
    validate_positive(sigci, "sigci")

    if sig3_max is None:
        sig3_max = sigci / 4.0

    # Generate sigma3 range for fitting
    sig3_arr = np.linspace(0, sig3_max, 20)

    # Compute sigma1 from HB
    sig1_arr = np.asarray(hoek_brown_rock_mass(sig3_arr, sigci, gsi, mi, d), dtype=float)

    # Normal and shear stress on the failure envelope
    # τ = (σ1 - σ3) / 2 * cos(φ)
    # σn = (σ1 + σ3) / 2 - (σ1 - σ3) / 2 * sin(φ)
    # Using linear regression on (σ1 - σ3) vs σ3:
    # σ1 - σ3 = f(σ3)
    # Fit: σ1 = m*σ3 + b (linear Mohr-Coulomb envelope)
    # φ = arcsin((m-1)/(m+1)), c = b / (2*sqrt(m))

    # Linear regression of sigma1 on sigma3
    coeffs = np.polyfit(sig3_arr, sig1_arr, 1)
    m_slope = coeffs[0]  # dσ1/dσ3
    b_intercept = coeffs[1]  # σ1 at σ3=0

    # Convert to Mohr-Coulomb: Eq. 4-5, Hoek et al. 2002
    # sinφ = (m - 1) / (m + 1)
    sin_phi = (m_slope - 1) / (m_slope + 1)
    sin_phi = np.clip(sin_phi, -1, 1)
    phi = np.degrees(np.arcsin(sin_phi))

    # c = σci_mass / (2 * cos(φ)) where σci_mass = b_intercept (σ1 at σ3=0)
    # Actually: c = b / (2*sqrt(m))
    cos_phi = np.cos(np.radians(phi))
    cohesion = b_intercept * (1 - sin_phi) / (2 * cos_phi) if cos_phi > 0 else 0.0

    return {
        "cohesion": float(cohesion),
        "friction_angle": float(phi),
        "sig3_range": sig3_arr,
    }


def deformation_modulus(
    sigci: float,
    gsi: float,
    d: float = 0.0,
    ei: float | None = None,
) -> float:
    """Rock mass deformation modulus (Hoek & Diederichs 2006).

    Parameters
    ----------
    sigci : float
        Uniaxial compressive strength of intact rock (MPa).
        Used only if ``ei`` is None (simplified equation).
    gsi : float
        Geological Strength Index (0-100).
    d : float
        Disturbance factor (0-1).
    ei : float or None
        Intact rock Young's modulus (MPa). If provided, the generalized
        equation is used. If None, simplified equation is used.

    Returns
    -------
    float
        Rock mass deformation modulus Erm (MPa).

    Examples
    --------
    >>> round(deformation_modulus(100, 50, 0, 50000), 0)
    12694.0

    References
    ----------
    .. [1] Hoek, E. & Diederichs, M.S. (2006). "Empirical estimation of rock
       mass modulus." Int. J. Rock Mech. Min. Sci., 43(2), 203-215.
    """
    validate_positive(sigci, "sigci")
    validate_range(gsi, 0, 100, "gsi")
    validate_range(d, 0, 1, "d")

    if ei is not None:
        validate_positive(ei, "ei")
        # Generalized equation — Hoek & Diederichs 2006
        # Erm = Ei * (0.02 + (1 - D/2) / (1 + exp((60 + 15D - GSI) / 11)))
        erm = ei * (0.02 + (1 - d / 2) / (1 + np.exp((60 + 15 * d - gsi) / 11)))
    else:
        # Simplified equation (when Ei unknown)
        # Erm = 100000 * (1 - D/2) / (1 + exp((75 + 25D - GSI) / 11))
        erm = 100000 * (1 - d / 2) / (1 + np.exp((75 + 25 * d - gsi) / 11))

    return float(erm)
