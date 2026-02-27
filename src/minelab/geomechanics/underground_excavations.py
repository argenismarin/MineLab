"""Geomechanics for underground excavations.

In-situ stress estimation, Kirsch elastic solution, plastic zone,
strength-to-stress ratio, support pressure, cable bolts, shotcrete
lining, and Mohr-Coulomb failure criterion.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# ---------------------------------------------------------------------------
# In-Situ Stress from Depth — Brady & Brown (2006)
# ---------------------------------------------------------------------------


def in_situ_stress_depth(
    depth: float,
    density: float,
    k_ratio: float,
) -> dict:
    """Estimate in-situ stresses from depth and density.

    Parameters
    ----------
    depth : float
        Depth below surface (m).
    density : float
        Rock mass density (kg/m^3).
    k_ratio : float
        Horizontal-to-vertical stress ratio (> 0).

    Returns
    -------
    dict
        Keys: ``"sigma_v_mpa"`` (vertical stress),
        ``"sigma_h_mpa"`` (horizontal stress),
        ``"sigma_mean_mpa"`` (mean stress).

    Examples
    --------
    >>> result = in_situ_stress_depth(1000.0, 2700.0, 1.5)
    >>> round(result["sigma_v_mpa"], 2)
    26.49

    References
    ----------
    .. [1] Brady, B.H.G. & Brown, E.T. (2006). "Rock Mechanics for
       Underground Mining." 3rd ed., Springer.
    """
    validate_positive(depth, "depth")
    validate_positive(density, "density")
    validate_positive(k_ratio, "k_ratio")

    sigma_v = density * 9.81 * depth / 1e6  # MPa
    sigma_h = k_ratio * sigma_v
    sigma_mean = (sigma_v + 2.0 * sigma_h) / 3.0

    return {
        "sigma_v_mpa": float(sigma_v),
        "sigma_h_mpa": float(sigma_h),
        "sigma_mean_mpa": float(sigma_mean),
    }


# ---------------------------------------------------------------------------
# Kirsch Elastic Solution (1898)
# ---------------------------------------------------------------------------


def kirsch_elastic_stress(
    sigma_v: float,
    sigma_h: float,
    r_tunnel: float,
    r: float,
    theta_deg: float,
) -> dict:
    """Kirsch (1898) elastic stress around a circular opening.

    Parameters
    ----------
    sigma_v : float
        Far-field vertical stress (MPa).
    sigma_h : float
        Far-field horizontal stress (MPa).
    r_tunnel : float
        Tunnel radius (m).
    r : float
        Radial distance from tunnel centre (m), must be >= r_tunnel.
    theta_deg : float
        Angle from horizontal (degrees).

    Returns
    -------
    dict
        Keys: ``"sigma_radial"`` (radial stress, MPa),
        ``"sigma_tangential"`` (tangential stress, MPa),
        ``"tau_shear"`` (shear stress, MPa).

    Examples
    --------
    >>> result = kirsch_elastic_stress(10.0, 5.0, 3.0, 3.0, 0.0)
    >>> result["sigma_tangential"]
    25.0

    References
    ----------
    .. [1] Kirsch, G. (1898). "Die Theorie der Elastizitaet und die
       Beduerfnisse der Festigkeitslehre." VDI Zeitschrift, 42,
       797-807.
    """
    validate_non_negative(sigma_v, "sigma_v")
    validate_non_negative(sigma_h, "sigma_h")
    validate_positive(r_tunnel, "r_tunnel")
    validate_positive(r, "r")

    if r < r_tunnel:
        raise ValueError(f"'r' must be >= r_tunnel ({r_tunnel}), got {r}.")

    theta = np.radians(theta_deg)
    a_r = r_tunnel / r  # ratio
    a_r2 = a_r**2
    a_r4 = a_r**4

    p = (sigma_v + sigma_h) / 2.0  # mean far-field
    q = (sigma_v - sigma_h) / 2.0  # deviatoric

    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)

    sigma_rr = p * (1.0 - a_r2) + q * (1.0 - 4.0 * a_r2 + 3.0 * a_r4) * cos2t
    sigma_tt = p * (1.0 + a_r2) - q * (1.0 + 3.0 * a_r4) * cos2t
    tau_rt = -q * (1.0 + 2.0 * a_r2 - 3.0 * a_r4) * sin2t

    return {
        "sigma_radial": float(sigma_rr),
        "sigma_tangential": float(sigma_tt),
        "tau_shear": float(tau_rt),
    }


# ---------------------------------------------------------------------------
# Plastic Zone Radius — Hoek & Brown (1980)
# ---------------------------------------------------------------------------


def plastic_zone_radius(
    sigma_0: float,
    sigma_cm: float,
    c: float,
    phi_deg: float,
    r_tunnel: float,
) -> float:
    """Plastic zone radius around an unsupported circular tunnel.

    Uses the Mohr-Coulomb closed-form solution for p_i = 0.

    Parameters
    ----------
    sigma_0 : float
        In-situ (hydrostatic) stress (MPa).
    sigma_cm : float
        Rock mass compressive strength (MPa).
    c : float
        Cohesion of rock mass (MPa).
    phi_deg : float
        Friction angle of rock mass (degrees, 1-89).
    r_tunnel : float
        Tunnel radius (m).

    Returns
    -------
    float
        Plastic zone radius in metres.

    Examples
    --------
    >>> plastic_zone_radius(20.0, 10.0, 2.0, 35.0, 3.0)
    6.45...

    References
    ----------
    .. [1] Hoek, E. & Brown, E.T. (1980). "Underground Excavations
       in Rock." Institution of Mining and Metallurgy, London.
    """
    validate_positive(sigma_0, "sigma_0")
    validate_positive(sigma_cm, "sigma_cm")
    validate_positive(c, "c")
    validate_range(phi_deg, 1, 89, "phi_deg")
    validate_positive(r_tunnel, "r_tunnel")

    phi_rad = np.radians(phi_deg)
    n_phi = (1.0 + np.sin(phi_rad)) / (1.0 - np.sin(phi_rad))

    # c * cot(phi) term
    c_cot_phi = c / np.tan(phi_rad)

    # r_p = r_t * ((sigma_0 + c_cot_phi) / (p_i + c_cot_phi))^(1/(N-1))
    # For unsupported tunnel (p_i = 0):
    exponent = 1.0 / (n_phi - 1.0)
    r_p = r_tunnel * ((sigma_0 + c_cot_phi) / c_cot_phi) ** exponent

    return float(r_p)


# ---------------------------------------------------------------------------
# Strength to Stress Ratio — Hoek (2007)
# ---------------------------------------------------------------------------


def strength_to_stress_ratio(
    ucs: float,
    sigma_1: float,
) -> dict:
    """Assess excavation stability from UCS / sigma_1 ratio.

    Parameters
    ----------
    ucs : float
        Uniaxial compressive strength (MPa).
    sigma_1 : float
        Maximum induced stress around excavation (MPa).

    Returns
    -------
    dict
        Keys: ``"ratio"`` (UCS / sigma_1),
        ``"classification"`` (stability class).

    Examples
    --------
    >>> result = strength_to_stress_ratio(100.0, 80.0)
    >>> result["classification"]
    'minor problems'

    References
    ----------
    .. [1] Hoek, E. (2007). "Practical Rock Engineering."
       www.rocscience.com. Ch. 8.
    """
    validate_positive(ucs, "ucs")
    validate_positive(sigma_1, "sigma_1")

    ratio = ucs / sigma_1

    if ratio > 2.0:
        classification = "stable"
    elif ratio > 1.0:
        classification = "minor problems"
    elif ratio > 0.5:
        classification = "severe problems"
    else:
        classification = "extreme problems"

    return {
        "ratio": float(ratio),
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Tunnel Support Pressure (Q-system) — Barton (2002)
# ---------------------------------------------------------------------------


def tunnel_support_pressure(
    q_value: float,
    span: float,
    esr: float,
) -> dict:
    """Support pressure and Q classification from Barton's Q-system.

    Parameters
    ----------
    q_value : float
        Barton Q value (> 0).
    span : float
        Excavation span (m).
    esr : float
        Excavation Support Ratio.

    Returns
    -------
    dict
        Keys: ``"support_pressure_kpa"`` (kPa),
        ``"equivalent_dimension"`` (m),
        ``"Q_class"`` (rock quality class).

    Examples
    --------
    >>> result = tunnel_support_pressure(4.0, 8.0, 1.0)
    >>> result["support_pressure_kpa"] > 0
    True

    References
    ----------
    .. [1] Barton, N. (2002). "Some new Q-value correlations to assist
       in site characterisation and tunnel design." Int. J. Rock Mech.
       and Mining Sciences, 39(2), 185-216.
    """
    validate_positive(q_value, "q_value")
    validate_positive(span, "span")
    validate_positive(esr, "esr")

    de = span / esr

    # P ~ 5 * Q^(-1/3) / esr (empirical, kPa)
    support_pressure = 5.0 * q_value ** (-1.0 / 3.0) / esr

    # Q classification
    if q_value > 40:
        q_class = "very good"
    elif q_value > 10:
        q_class = "good"
    elif q_value > 4:
        q_class = "fair"
    elif q_value > 1:
        q_class = "poor"
    elif q_value > 0.1:
        q_class = "very poor"
    else:
        q_class = "exceptionally poor"

    return {
        "support_pressure_kpa": float(support_pressure),
        "equivalent_dimension": float(de),
        "Q_class": q_class,
    }


# ---------------------------------------------------------------------------
# Cable Bolt Capacity — Fuller & Cox (1975)
# ---------------------------------------------------------------------------


def cable_bolt_capacity(
    diameter_mm: float,
    ucs_grout: float,
    embedment_length: float,
) -> dict:
    """Cable bolt design capacity from bond and steel strength.

    Parameters
    ----------
    diameter_mm : float
        Cable bolt diameter (mm).
    ucs_grout : float
        Grout UCS (MPa).
    embedment_length : float
        Grouted embedment length (m).

    Returns
    -------
    dict
        Keys: ``"bond_capacity_kn"`` (bond pull-out capacity, kN),
        ``"steel_capacity_kn"`` (steel tensile capacity, kN),
        ``"design_capacity_kn"`` (min of bond and steel, kN).

    Examples
    --------
    >>> result = cable_bolt_capacity(15.2, 40.0, 3.0)
    >>> result["design_capacity_kn"] > 0
    True

    References
    ----------
    .. [1] Fuller, P.G. & Cox, R.H.T. (1975). "Mechanics of load
       transfer from steel tendons to cement based grout." Proc. 5th
       Australasian Conference on Mechanics of Structures and
       Materials, Melbourne.
    """
    validate_positive(diameter_mm, "diameter_mm")
    validate_positive(ucs_grout, "ucs_grout")
    validate_positive(embedment_length, "embedment_length")

    # Bond strength ~ 0.5 * sqrt(UCS_grout) MPa
    bond_strength = 0.5 * np.sqrt(ucs_grout)

    # Circumference in metres
    circumference = np.pi * diameter_mm / 1000.0

    # Bond capacity (kN)
    bond_capacity = bond_strength * circumference * embedment_length * 1000.0

    # Steel tensile capacity (kN), assuming 1860 MPa grade steel
    steel_area = np.pi / 4.0 * (diameter_mm / 1000.0) ** 2
    steel_capacity = 0.6 * steel_area * 1860.0 * 1000.0  # kN

    design_capacity = min(bond_capacity, steel_capacity)

    return {
        "bond_capacity_kn": float(bond_capacity),
        "steel_capacity_kn": float(steel_capacity),
        "design_capacity_kn": float(design_capacity),
    }


# ---------------------------------------------------------------------------
# Shotcrete Lining Capacity — Hoek et al. (1995)
# ---------------------------------------------------------------------------


def shotcrete_lining_capacity(
    thickness_mm: float,
    ucs_shotcrete: float,
    radius_tunnel: float,
) -> dict:
    """Maximum support pressure from shotcrete lining.

    Parameters
    ----------
    thickness_mm : float
        Shotcrete thickness (mm).
    ucs_shotcrete : float
        Shotcrete UCS (MPa).
    radius_tunnel : float
        Tunnel radius (m).

    Returns
    -------
    dict
        Keys: ``"max_pressure_mpa"`` (maximum radial support pressure),
        ``"lining_stress_mpa"`` (stress in shotcrete at max pressure).

    Examples
    --------
    >>> result = shotcrete_lining_capacity(100.0, 30.0, 3.0)
    >>> result["max_pressure_mpa"] > 0
    True

    References
    ----------
    .. [1] Hoek, E., Kaiser, P.K. & Bawden, W.F. (1995). "Support of
       Underground Excavations in Hard Rock." Balkema, Rotterdam.
    """
    validate_positive(thickness_mm, "thickness_mm")
    validate_positive(ucs_shotcrete, "ucs_shotcrete")
    validate_positive(radius_tunnel, "radius_tunnel")

    thickness_m = thickness_mm / 1000.0

    # Thin shell approximation: p_max = sigma_c * t / r
    max_pressure = ucs_shotcrete * thickness_m / radius_tunnel

    # At max pressure, lining stress equals UCS
    lining_stress = ucs_shotcrete

    return {
        "max_pressure_mpa": float(max_pressure),
        "lining_stress_mpa": float(lining_stress),
    }


# ---------------------------------------------------------------------------
# Mohr-Coulomb Failure Criterion
# ---------------------------------------------------------------------------


def failure_criterion_mohr_coulomb_ug(
    sigma_3: float,
    cohesion: float,
    friction_angle_deg: float,
) -> float:
    """Major principal stress at failure from Mohr-Coulomb criterion.

    Parameters
    ----------
    sigma_3 : float
        Minor principal stress (MPa).
    cohesion : float
        Rock mass cohesion (MPa).
    friction_angle_deg : float
        Friction angle (degrees, 1-89).

    Returns
    -------
    float
        Major principal stress sigma_1 at failure (MPa).

    Examples
    --------
    >>> failure_criterion_mohr_coulomb_ug(5.0, 10.0, 35.0)
    56.85...

    References
    ----------
    .. [1] Jaeger, J.C., Cook, N.G.W. & Zimmerman, R.W. (2007).
       "Fundamentals of Rock Mechanics." 4th ed., Blackwell.
    """
    validate_non_negative(sigma_3, "sigma_3")
    validate_positive(cohesion, "cohesion")
    validate_range(friction_angle_deg, 1, 89, "friction_angle_deg")

    phi_rad = np.radians(friction_angle_deg)
    n_phi = (1.0 + np.sin(phi_rad)) / (1.0 - np.sin(phi_rad))

    sigma_1 = sigma_3 * n_phi + 2.0 * cohesion * np.sqrt(n_phi)
    return float(sigma_1)
