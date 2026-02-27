"""Underground support design calculations.

Pillar strength, tributary area stress, rock bolt design, shotcrete
thickness, and stand-up time estimation.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive, validate_range

# ---------------------------------------------------------------------------
# Pillar Strength — Bieniawski (P3-M17)
# ---------------------------------------------------------------------------


def pillar_strength_bieniawski(
    w: float,
    h: float,
    sigci: float,
    k: float = 1.0,
) -> dict:
    """Pillar strength using Bieniawski's formula.

    Parameters
    ----------
    w : float
        Pillar width (m).
    h : float
        Pillar height (m).
    sigci : float
        Uniaxial compressive strength of intact rock (MPa).
    k : float
        Rock mass strength reduction factor, typically
        ``k = sigcm / sigci`` where sigcm is the rock mass UCS.
        Default 1.0 (no reduction).

    Returns
    -------
    dict
        Keys: ``"strength"`` (pillar strength in MPa),
        ``"w_over_h"`` (width-to-height ratio).

    Examples
    --------
    >>> result = pillar_strength_bieniawski(10, 5, 100)
    >>> round(result["strength"], 1)
    136.0

    References
    ----------
    .. [1] Bieniawski, Z.T. (1968/1992). In situ strength and deformation
       characteristics of coal. Engineering Geology, 2(5), 325-340.
    """
    validate_positive(w, "w")
    validate_positive(h, "h")
    validate_positive(sigci, "sigci")
    validate_positive(k, "k")

    w_over_h = w / h

    # σp = k * σci * (0.64 + 0.36 * w/h) — Bieniawski 1968
    strength = k * sigci * (0.64 + 0.36 * w_over_h)

    return {"strength": float(strength), "w_over_h": float(w_over_h)}


# ---------------------------------------------------------------------------
# Pillar Strength — Lunder & Pakalnis (P3-M18)
# ---------------------------------------------------------------------------


def pillar_strength_lunder_pakalnis(
    w: float,
    h: float,
    ucs: float,
    c1: float = 0.68,
    c2: float = 0.52,
) -> dict:
    """Pillar strength using Lunder & Pakalnis (1997) formula.

    Parameters
    ----------
    w : float
        Pillar width (m).
    h : float
        Pillar height (m).
    ucs : float
        Uniaxial compressive strength (MPa).
    c1 : float
        Empirical constant (default 0.68).
    c2 : float
        Empirical constant (default 0.52).

    Returns
    -------
    dict
        Keys: ``"strength"`` (pillar strength in MPa),
        ``"kappa"`` (confinement parameter),
        ``"w_over_h"`` (width-to-height ratio).

    Examples
    --------
    >>> result = pillar_strength_lunder_pakalnis(10, 5, 100)
    >>> result["strength"] > 0
    True

    References
    ----------
    .. [1] Lunder, P.J. & Pakalnis, R.C. (1997). "Determination of the
       strength of hard-rock mine pillars." CIM Bulletin, 90(1013), 51-55.
    """
    validate_positive(w, "w")
    validate_positive(h, "h")
    validate_positive(ucs, "ucs")

    w_over_h = w / h

    # κ (kappa) = confinement parameter
    # κ = 1 - (w/h + 0.68) / (w/h + 0.32) ... simplified from Lunder 1997
    # Actually: Cpav = 0.46 * [log10(1 + κ)]^(2.7)
    # where κ is a function of w/h: κ = 1/(2*(w_h/(1+w_h)))... approximation
    # Simplified: use the empirical formula directly
    kappa = 0.46 * (np.log10(1 + w_over_h)) ** 2.7 if w_over_h > 0 else 0.0

    # σp = K * UCS * (C1 + C2 * κ) — Lunder & Pakalnis 1997
    # K is a sizing constant, typically 1.0 for hard rock
    strength = ucs * (c1 + c2 * kappa)

    return {
        "strength": float(strength),
        "kappa": float(kappa),
        "w_over_h": float(w_over_h),
    }


# ---------------------------------------------------------------------------
# Tributary Area Stress (P3-M19)
# ---------------------------------------------------------------------------


def tributary_area_stress(
    depth: float,
    extraction_ratio: float,
    density: float = 2700.0,
) -> dict:
    """Average pillar stress from tributary area theory.

    Parameters
    ----------
    depth : float
        Mining depth (m).
    extraction_ratio : float
        Extraction ratio e (0 < e < 1).
    density : float
        Rock mass density (kg/m^3). Default 2700.

    Returns
    -------
    dict
        Keys: ``"pillar_stress"`` (MPa), ``"vertical_stress"`` (MPa),
        ``"stress_concentration"`` (ratio).

    Examples
    --------
    >>> result = tributary_area_stress(100, 0.75, 2700)
    >>> round(result["pillar_stress"], 2)
    10.59

    References
    ----------
    .. [1] Brady, B.H.G. & Brown, E.T. (2006). Rock Mechanics for
       Underground Mining. 3rd ed., Springer.
    """
    validate_positive(depth, "depth")
    validate_range(extraction_ratio, 0.01, 0.99, "extraction_ratio")
    validate_positive(density, "density")

    g = 9.81  # m/s^2

    # Vertical (virgin) stress: σv = ρ * g * H
    sigma_v = density * g * depth / 1e6  # Pa to MPa

    # Pillar stress: σp = σv / (1 - e) — Brady & Brown 2006
    sigma_p = sigma_v / (1 - extraction_ratio)

    # Stress concentration factor
    scf = 1 / (1 - extraction_ratio)

    return {
        "pillar_stress": float(sigma_p),
        "vertical_stress": float(sigma_v),
        "stress_concentration": float(scf),
    }


# ---------------------------------------------------------------------------
# Rock Bolt Design (P3-M20)
# ---------------------------------------------------------------------------


def rock_bolt_design(
    q_value: float,
    span: float,
    esr: float = 1.0,
) -> dict:
    """Rock bolt length and spacing from Q-system.

    Parameters
    ----------
    q_value : float
        Barton Q-system value.
    span : float
        Excavation span (m).
    esr : float
        Excavation Support Ratio (default 1.0).

    Returns
    -------
    dict
        Keys: ``"bolt_length"`` (m), ``"spacing"`` (m),
        ``"equivalent_dimension"`` (m).

    Examples
    --------
    >>> result = rock_bolt_design(10, 10, 1.0)
    >>> result["bolt_length"] > 0
    True

    References
    ----------
    .. [1] Barton, N., Lien, R. & Lunde, J. (1974). "Engineering
       classification of rock masses for the design of tunnel support."
       Rock Mechanics, 6(4), 189-236.
    """
    validate_positive(span, "span")
    validate_positive(esr, "esr")

    # Equivalent dimension De = span / ESR
    de = span / esr

    # Bolt length: L = 2 + 0.15 * De — Barton et al. 1974
    bolt_length = 2 + 0.15 * de

    # Spacing from Q: S ≈ 2.0 + 0.1 * (De / (Q^0.33))
    # Simplified empirical: spacing decreases with lower Q
    spacing = max(1.0, 2.4 * q_value**0.1 - 0.2 * np.log10(de)) if q_value > 0 else 1.0

    # Clamp spacing
    spacing = np.clip(spacing, 1.0, 3.0)

    return {
        "bolt_length": float(bolt_length),
        "spacing": float(spacing),
        "equivalent_dimension": float(de),
    }


# ---------------------------------------------------------------------------
# Shotcrete Thickness (P3-M21)
# ---------------------------------------------------------------------------


def shotcrete_thickness(
    rmr: float,
    span: float,
) -> dict:
    """Empirical shotcrete thickness from RMR and span.

    Parameters
    ----------
    rmr : float
        Rock Mass Rating (0-100).
    span : float
        Excavation span (m).

    Returns
    -------
    dict
        Keys: ``"thickness"`` (mm), ``"recommendation"`` (str).

    Examples
    --------
    >>> result = shotcrete_thickness(40, 5)
    >>> result["thickness"] > 0
    True

    References
    ----------
    .. [1] Bieniawski, Z.T. (1989). Engineering Rock Mass Classifications.
       Wiley, Table 6.
    """
    validate_range(rmr, 0, 100, "rmr")
    validate_positive(span, "span")

    # Based on Bieniawski 1989 Table 6 support recommendations
    if rmr >= 81:
        thickness = 0.0
        rec = "Generally no support required"
    elif rmr >= 61:
        thickness = 50.0
        rec = "Locally, 50 mm in crown where required"
    elif rmr >= 41:
        thickness = 50 + (61 - rmr) * 2.5  # 50-100 mm
        rec = "Systematic bolts + 50-100 mm shotcrete in crown"
    elif rmr >= 21:
        thickness = 100 + (41 - rmr) * 2.5  # 100-150 mm
        rec = "Systematic bolts + 100-150 mm shotcrete"
    else:
        thickness = 150 + (21 - rmr) * 5  # 150-250+ mm
        rec = "Heavy sets or cast concrete lining with shotcrete"

    # Adjust for span (thicker for wider spans)
    span_factor = max(1.0, span / 5.0)
    thickness *= span_factor

    return {
        "thickness": float(round(thickness)),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Stand-up Time (P3-M22)
# ---------------------------------------------------------------------------


def stand_up_time(
    rmr: float,
    span: float,
) -> dict:
    """Estimate unsupported stand-up time from RMR and span.

    Parameters
    ----------
    rmr : float
        Rock Mass Rating (0-100).
    span : float
        Excavation span (m).

    Returns
    -------
    dict
        Keys: ``"time_hours"`` (stand-up time in hours),
        ``"time_days"`` (stand-up time in days).

    Examples
    --------
    >>> result = stand_up_time(60, 5)
    >>> result["time_hours"] > 0
    True

    References
    ----------
    .. [1] Bieniawski, Z.T. (1989). Engineering Rock Mass Classifications.
       Wiley. Stand-up time chart: log(T) = 1.28*RMR - 0.67*ln(span) - 4.
    """
    validate_range(rmr, 0, 100, "rmr")
    validate_positive(span, "span")

    # log10(T) = 1.28*RMR/100 * 6 - 0.67*ln(span) - 4
    # Simplified from Bieniawski chart:
    # ln(T_hours) = 1.28*RMR - 0.67*ln(span) - 4
    # T_hours = exp(1.28*RMR/10 - 0.67*ln(span))
    # Actually per Bieniawski 1989:
    # log10(stand_up_time_hours) ≈ (RMR - 0.67*ln(span)*14.5 - 30) / 14.5
    # Simplified empirical: ln(T) = 0.128*RMR - 0.67*ln(S) - 4

    ln_t = 0.128 * rmr - 0.67 * np.log(span)
    time_hours = max(0.01, np.exp(ln_t))
    time_days = time_hours / 24.0

    return {
        "time_hours": float(time_hours),
        "time_days": float(time_days),
    }
