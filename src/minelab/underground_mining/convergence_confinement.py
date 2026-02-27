"""Convergence-confinement method for tunnel support design.

Ground reaction curves, support reaction curves, longitudinal
deformation profiles, squeezing and rock-burst assessment.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# Ground Reaction Curve — Carranza-Torres & Fairhurst (2000)
# ---------------------------------------------------------------------------


def ground_reaction_curve(
    p_i: float,
    sigma_0: float,
    sigma_ci: float,
    mi: float,
    gsi: float,
    r_tunnel: float,
    e_rock: float,
) -> dict:
    """Ground reaction curve using simplified Hoek-Brown criterion.

    Parameters
    ----------
    p_i : float
        Internal support pressure (MPa).
    sigma_0 : float
        In-situ stress (MPa), assumed hydrostatic.
    sigma_ci : float
        Intact rock uniaxial compressive strength (MPa).
    mi : float
        Hoek-Brown material constant for intact rock.
    gsi : float
        Geological Strength Index (0-100).
    r_tunnel : float
        Tunnel radius (m).
    e_rock : float
        Young's modulus of rock mass (MPa).

    Returns
    -------
    dict
        Keys: ``"p_critical_mpa"`` (critical support pressure),
        ``"u_elastic_mm"`` (elastic wall displacement),
        ``"u_max_mm"`` (maximum displacement at given p_i),
        ``"r_plastic_m"`` (plastic zone radius, 0 if elastic),
        ``"convergence_pct"`` (percentage convergence).

    Examples
    --------
    >>> result = ground_reaction_curve(0.5, 10.0, 50.0, 10.0, 60.0,
    ...                                3.0, 5000.0)
    >>> result["p_critical_mpa"] >= 0
    True

    References
    ----------
    .. [1] Carranza-Torres, C. & Fairhurst, C. (2000). "Application of
       the convergence-confinement method of tunnel design to rock
       masses that satisfy the Hoek-Brown failure criterion."
       Tunnelling and Underground Space Technology, 15(2), 187-213.
    .. [2] Hoek, E. et al. (2002). "Hoek-Brown failure criterion —
       2002 edition." Proc. NARMS-TAC, Toronto.
    """
    validate_non_negative(p_i, "p_i")
    validate_positive(sigma_0, "sigma_0")
    validate_positive(sigma_ci, "sigma_ci")
    validate_positive(mi, "mi")
    validate_range(gsi, 0, 100, "gsi")
    validate_positive(r_tunnel, "r_tunnel")
    validate_positive(e_rock, "e_rock")

    # Hoek-Brown parameters (simplified for GSI > 25)
    _mb = mi * np.exp((gsi - 100.0) / 28.0)  # noqa: F841
    s = np.exp((gsi - 100.0) / 9.0)

    # Rock mass compressive strength (simplified)
    sigma_cm = sigma_ci * (s**0.5)

    # Critical internal pressure
    p_cr = max(0.0, sigma_0 - sigma_cm)

    nu = 0.25  # Poisson's ratio assumption

    # Elastic wall displacement at given p_i
    u_elastic = (1.0 + nu) / e_rock * (sigma_0 - p_i) * r_tunnel * 1000.0  # mm

    if p_i >= p_cr:
        # Elastic regime
        u_max = u_elastic
        r_plastic = 0.0
    else:
        # Plastic zone develops
        if sigma_cm > 0:
            ratio = np.sqrt(2.0 * (sigma_0 - p_cr) / sigma_cm)
            r_plastic = r_tunnel * max(1.0, ratio)
        else:
            r_plastic = r_tunnel * 2.0

        # Plastic displacement scaled by (r_p/r_t)^2
        u_elastic_at_pcr = (1.0 + nu) / e_rock * (sigma_0 - p_cr) * r_tunnel * 1000.0
        u_max = u_elastic_at_pcr * (r_plastic / r_tunnel) ** 2

    convergence_pct = (u_max / (r_tunnel * 1000.0)) * 100.0

    return {
        "p_critical_mpa": float(p_cr),
        "u_elastic_mm": float(u_elastic),
        "u_max_mm": float(u_max),
        "r_plastic_m": float(r_plastic),
        "convergence_pct": float(convergence_pct),
    }


# ---------------------------------------------------------------------------
# Support Reaction Curve
# ---------------------------------------------------------------------------


def support_reaction_curve(
    k_support: float,
    p_max: float,
    u_installation: float,
) -> dict:
    """Linear support reaction curve parameters.

    Parameters
    ----------
    k_support : float
        Support stiffness (MPa/mm).
    p_max : float
        Maximum support pressure capacity (MPa).
    u_installation : float
        Displacement at time of installation (mm).

    Returns
    -------
    dict
        Keys: ``"support_stiffness"`` (MPa/mm),
        ``"max_support_pressure"`` (MPa),
        ``"installation_displacement"`` (mm),
        ``"max_displacement_mm"`` (displacement at p_max).

    Examples
    --------
    >>> result = support_reaction_curve(0.01, 0.5, 5.0)
    >>> result["max_displacement_mm"]
    55.0

    References
    ----------
    .. [1] Hoek, E. (2007). "Practical Rock Engineering." Ch. 10,
       Support-interaction analysis.
    """
    validate_positive(k_support, "k_support")
    validate_positive(p_max, "p_max")
    validate_non_negative(u_installation, "u_installation")

    # Displacement at which support reaches max capacity
    u_at_pmax = u_installation + p_max / k_support

    return {
        "support_stiffness": float(k_support),
        "max_support_pressure": float(p_max),
        "installation_displacement": float(u_installation),
        "max_displacement_mm": float(u_at_pmax),
    }


# ---------------------------------------------------------------------------
# Longitudinal Deformation Profile — Vlachopoulos & Diederichs (2009)
# ---------------------------------------------------------------------------


def longitudinal_deformation_profile(
    x: float,
    r_tunnel: float,
    r_plastic: float,
    u_max: float,
) -> float:
    """Wall displacement at distance x from the tunnel face.

    Parameters
    ----------
    x : float
        Distance from tunnel face (m). Positive = behind face
        (excavated), negative = ahead of face (rock mass).
    r_tunnel : float
        Tunnel radius (m).
    r_plastic : float
        Plastic zone radius (m). Use r_tunnel if elastic.
    u_max : float
        Maximum radial displacement far behind face (mm).

    Returns
    -------
    float
        Radial wall displacement at position x (mm).

    Examples
    --------
    >>> longitudinal_deformation_profile(0.0, 3.0, 3.0, 10.0)  # at face
    3.0...

    References
    ----------
    .. [1] Vlachopoulos, N. & Diederichs, M.S. (2009). "Improved
       longitudinal displacement profiles for convergence confinement
       analysis of deep tunnels." Rock Mechanics and Rock Engineering,
       42(2), 131-146.
    .. [2] Panet, M. & Guenot, A. (1982). "Analysis of convergence
       behind the face of a tunnel." Proc. Tunnelling '82, London.
    """
    validate_positive(r_tunnel, "r_tunnel")
    validate_positive(r_plastic, "r_plastic")
    validate_non_negative(u_max, "u_max")

    if u_max == 0.0:
        return 0.0

    # Face displacement ratio from Vlachopoulos & Diederichs (2009)
    rp_rt = r_plastic / r_tunnel
    u_face_ratio = (1.0 / 3.0) * np.exp(-0.15 * (rp_rt - 1.0))

    if x >= 0:
        # Behind face: u/u_max approaches 1 exponentially
        u_ratio = 1.0 - (1.0 - u_face_ratio) * np.exp(-x / (1.5 * r_tunnel))
    else:
        # Ahead of face: displacement decays ahead
        u_ratio = u_face_ratio * np.exp(x / (1.5 * r_tunnel))

    return float(u_ratio * u_max)


# ---------------------------------------------------------------------------
# Rock-Support Interaction (Equilibrium)
# ---------------------------------------------------------------------------


def rock_support_interaction(
    sigma_0: float,
    p_i_array: list,
    u_grc_array: list,
    k_support: float,
    u_install: float,
    p_max_support: float,
) -> dict:
    """Find equilibrium between GRC and SRC by linear interpolation.

    Parameters
    ----------
    sigma_0 : float
        In-situ stress (MPa).
    p_i_array : list
        Internal pressures for GRC (MPa), descending.
    u_grc_array : list
        Corresponding wall displacements for GRC (mm), ascending.
    k_support : float
        Support stiffness (MPa/mm).
    u_install : float
        Displacement at installation (mm).
    p_max_support : float
        Maximum support capacity (MPa).

    Returns
    -------
    dict
        Keys: ``"equilibrium_pressure_mpa"``,
        ``"equilibrium_displacement_mm"``,
        ``"factor_of_safety"`` (p_max / p_equilibrium).

    Examples
    --------
    >>> p_i = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    >>> u_grc = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0]
    >>> result = rock_support_interaction(10.0, p_i, u_grc, 0.05,
    ...                                   2.0, 1.0)
    >>> result["equilibrium_pressure_mpa"] >= 0
    True

    References
    ----------
    .. [1] Carranza-Torres, C. & Fairhurst, C. (2000). "Application of
       the convergence-confinement method." Tunnelling and Underground
       Space Technology, 15(2), 187-213.
    .. [2] Hoek, E. (2007). "Practical Rock Engineering." Ch. 10.
    """
    validate_positive(sigma_0, "sigma_0")
    validate_positive(k_support, "k_support")
    validate_non_negative(u_install, "u_install")
    validate_positive(p_max_support, "p_max_support")

    p_i = np.asarray(p_i_array, dtype=float)
    u_grc = np.asarray(u_grc_array, dtype=float)

    if len(p_i) < 2 or len(u_grc) < 2:
        raise ValueError("'p_i_array' and 'u_grc_array' must have at least 2 elements each.")

    # Build SRC values at same displacements
    # p_support = k * (u - u_install) clamped to [0, p_max]
    p_src = np.clip(k_support * (u_grc - u_install), 0.0, p_max_support)

    # Equilibrium where GRC pressure == SRC pressure
    # Find sign change of (p_i - p_src)
    diff = p_i - p_src
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No intersection found; use last point
        eq_p = float(p_src[-1])
        eq_u = float(u_grc[-1])
    else:
        idx = sign_changes[0]
        # Linear interpolation between idx and idx+1
        d0 = diff[idx]
        d1 = diff[idx + 1]
        frac = d0 / (d0 - d1)
        eq_u = float(u_grc[idx] + frac * (u_grc[idx + 1] - u_grc[idx]))
        eq_p = float(p_i[idx] + frac * (p_i[idx + 1] - p_i[idx]))

    # Factor of safety
    fos = p_max_support / eq_p if eq_p > 0 else float("inf")

    return {
        "equilibrium_pressure_mpa": float(eq_p),
        "equilibrium_displacement_mm": float(eq_u),
        "factor_of_safety": float(fos),
    }


# ---------------------------------------------------------------------------
# Squeezing Index — Hoek & Marinos (2000)
# ---------------------------------------------------------------------------


def squeezing_index(
    sigma_0: float,
    sigma_cm: float,
) -> dict:
    """Squeezing potential from rock mass strength to stress ratio.

    Parameters
    ----------
    sigma_0 : float
        In-situ stress (MPa).
    sigma_cm : float
        Rock mass uniaxial compressive strength (MPa).

    Returns
    -------
    dict
        Keys: ``"ratio"`` (sigma_cm / sigma_0),
        ``"classification"`` (str),
        ``"strain_pct_estimate"`` (approximate tunnel strain %).

    Examples
    --------
    >>> result = squeezing_index(20.0, 5.0)
    >>> result["classification"]
    'extreme squeezing'

    References
    ----------
    .. [1] Hoek, E. & Marinos, P. (2000). "Predicting tunnel squeezing
       problems in weak heterogeneous rock masses." Tunnels and
       Tunnelling International, Part 1: Nov 2000, Part 2: Dec 2000.
    """
    validate_positive(sigma_0, "sigma_0")
    validate_positive(sigma_cm, "sigma_cm")

    ratio = sigma_cm / sigma_0

    if ratio < 0.2:
        classification = "extreme squeezing"
    elif ratio < 0.3:
        classification = "severe squeezing"
    elif ratio < 0.4:
        classification = "moderate squeezing"
    else:
        classification = "few support problems"

    # Approximate strain ~ 1 / ratio^2 (%)
    strain_pct = 1.0 / (ratio**2)

    return {
        "ratio": float(ratio),
        "classification": classification,
        "strain_pct_estimate": float(strain_pct),
    }


# ---------------------------------------------------------------------------
# Rock Burst Potential — Kaiser et al. (1996)
# ---------------------------------------------------------------------------


def rock_burst_potential(
    sigma_1: float,
    sigma_3: float,
    ucs: float,
    brittleness: float,
) -> dict:
    """Assess rock burst potential from stress and brittleness.

    Parameters
    ----------
    sigma_1 : float
        Major principal stress (MPa).
    sigma_3 : float
        Minor principal stress (MPa).
    ucs : float
        Uniaxial compressive strength (MPa).
    brittleness : float
        Brittleness index = UCS / tensile strength (dimensionless).

    Returns
    -------
    dict
        Keys: ``"bpi"`` (burst potential index sigma_1/ucs),
        ``"stress_ratio"`` (sigma_1/sigma_3),
        ``"potential"`` (``"none"``, ``"low"``, ``"moderate"``,
        or ``"high"``),
        ``"brittleness_class"`` (``"ductile"`` or ``"brittle"``).

    Examples
    --------
    >>> result = rock_burst_potential(80.0, 3.0, 100.0, 50.0)
    >>> result["potential"]
    'high'

    References
    ----------
    .. [1] Kaiser, P.K., McCreath, D.R. & Tannant, D.D. (1996).
       "Canadian Rockburst Support Handbook." Geomechanics Research
       Centre, Laurentian University.
    """
    validate_positive(sigma_1, "sigma_1")
    validate_non_negative(sigma_3, "sigma_3")
    validate_positive(ucs, "ucs")
    validate_positive(brittleness, "brittleness")

    bpi = sigma_1 / ucs
    stress_ratio = sigma_1 / sigma_3 if sigma_3 > 0 else float("inf")

    # Brittleness classification
    brittleness_class = "brittle" if brittleness > 40.0 else "ductile"

    # Rock burst potential classification
    if stress_ratio > 20.0 and bpi > 0.7:
        potential = "high"
    elif bpi > 0.5:
        potential = "moderate"
    elif bpi > 0.3:
        potential = "low"
    else:
        potential = "none"

    return {
        "bpi": float(bpi),
        "stress_ratio": float(stress_ratio),
        "potential": potential,
        "brittleness_class": brittleness_class,
    }


# ---------------------------------------------------------------------------
# Tunnel Deformation Strain — Hoek & Marinos (2000)
# ---------------------------------------------------------------------------


def tunnel_deformation_strain(
    p_i: float,
    sigma_0: float,
    sigma_cm: float,
    r_tunnel: float,
) -> float:
    """Estimate tunnel strain percentage.

    Parameters
    ----------
    p_i : float
        Internal support pressure (MPa).
    sigma_0 : float
        In-situ stress (MPa).
    sigma_cm : float
        Rock mass compressive strength (MPa).
    r_tunnel : float
        Tunnel radius (m).

    Returns
    -------
    float
        Estimated tunnel strain (%).

    Examples
    --------
    >>> tunnel_deformation_strain(0.0, 10.0, 3.0, 3.0)
    2.222...

    References
    ----------
    .. [1] Hoek, E. & Marinos, P. (2000). "Predicting tunnel squeezing
       problems in weak heterogeneous rock masses." Tunnels and
       Tunnelling International.
    """
    validate_non_negative(p_i, "p_i")
    validate_positive(sigma_0, "sigma_0")
    validate_positive(sigma_cm, "sigma_cm")
    validate_positive(r_tunnel, "r_tunnel")

    net_stress = sigma_0 - p_i
    if net_stress <= 0:
        return 0.0

    # strain_pct = 0.2 * (sigma_cm / net_stress)^(-2)
    ratio = sigma_cm / net_stress
    strain_pct = 0.2 * ratio ** (-2)
    return float(strain_pct)
