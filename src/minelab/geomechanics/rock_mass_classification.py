"""Rock mass classification systems.

RMR (Bieniawski 1989), Q-system (Barton 1974), GSI conversions, and
SMR (Romana 1985) for slope applications.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_range


def rmr_bieniawski(
    ucs_rating: float,
    rqd_rating: float,
    spacing_rating: float,
    condition_rating: float,
    groundwater_rating: float,
    orientation_adj: float = 0.0,
) -> dict:
    """Rock Mass Rating (RMR89) classification.

    Parameters
    ----------
    ucs_rating : float
        UCS rating (0-15).
    rqd_rating : float
        RQD rating (3-20).
    spacing_rating : float
        Discontinuity spacing rating (5-20).
    condition_rating : float
        Condition of discontinuities rating (0-30).
    groundwater_rating : float
        Groundwater rating (0-15).
    orientation_adj : float
        Orientation adjustment (-60 to 0). Default 0.

    Returns
    -------
    dict
        Keys: ``"rmr"`` (total RMR value), ``"class_number"`` (I-V),
        ``"description"`` (quality description).

    Examples
    --------
    >>> result = rmr_bieniawski(12, 17, 15, 20, 10, -5)
    >>> result["rmr"]
    69
    >>> result["class_number"]
    'II'

    References
    ----------
    .. [1] Bieniawski, Z.T. (1989). "Engineering Rock Mass Classifications."
       Wiley, Table 4.
    """
    validate_range(ucs_rating, 0, 15, "ucs_rating")
    validate_range(rqd_rating, 3, 20, "rqd_rating")
    validate_range(spacing_rating, 5, 20, "spacing_rating")
    validate_range(condition_rating, 0, 30, "condition_rating")
    validate_range(groundwater_rating, 0, 15, "groundwater_rating")
    validate_range(orientation_adj, -60, 0, "orientation_adj")

    # RMR = sum of 5 basic ratings + orientation adjustment
    basic = ucs_rating + rqd_rating + spacing_rating + condition_rating + groundwater_rating
    rmr = int(round(basic + orientation_adj))

    # Classification per Bieniawski 1989 Table 4
    if rmr >= 81:
        cls, desc = "I", "Very good rock"
    elif rmr >= 61:
        cls, desc = "II", "Good rock"
    elif rmr >= 41:
        cls, desc = "III", "Fair rock"
    elif rmr >= 21:
        cls, desc = "IV", "Poor rock"
    else:
        cls, desc = "V", "Very poor rock"

    return {"rmr": rmr, "class_number": cls, "description": desc}


def q_system(
    rqd: float,
    jn: float,
    jr: float,
    ja: float,
    jw: float,
    srf: float,
) -> dict:
    """Barton's Q-system rock mass classification.

    Parameters
    ----------
    rqd : float
        Rock Quality Designation (0-100).
    jn : float
        Joint set number (0.5-20).
    jr : float
        Joint roughness number (0.5-4).
    ja : float
        Joint alteration number (0.75-20).
    jw : float
        Joint water reduction factor (0.05-1).
    srf : float
        Stress Reduction Factor (0.5-400).

    Returns
    -------
    dict
        Keys: ``"Q"`` (Q value), ``"description"`` (quality class).

    Examples
    --------
    >>> result = q_system(90, 9, 3, 1, 1, 1)
    >>> result["Q"]
    30.0
    >>> result["description"]
    'Good'

    References
    ----------
    .. [1] Barton, N., Lien, R. & Lunde, J. (1974). "Engineering classification
       of rock masses for the design of tunnel support." Rock Mechanics,
       6(4), 189-236.
    """
    validate_range(rqd, 0, 100, "rqd")

    if jn <= 0 or ja <= 0 or srf <= 0:
        raise ValueError("jn, ja, srf must be positive.")

    # Q = (RQD/Jn) * (Jr/Ja) * (Jw/SRF)  — Eq. 1, Barton et al. 1974
    q_val = (rqd / jn) * (jr / ja) * (jw / srf)

    # Classification
    if q_val > 400:
        desc = "Exceptionally good"
    elif q_val > 100:
        desc = "Extremely good"
    elif q_val > 40:
        desc = "Very good"
    elif q_val > 10:
        desc = "Good"
    elif q_val > 4:
        desc = "Fair"
    elif q_val > 1:
        desc = "Poor"
    elif q_val > 0.1:
        desc = "Very poor"
    elif q_val > 0.01:
        desc = "Extremely poor"
    else:
        desc = "Exceptionally poor"

    return {"Q": float(q_val), "description": desc}


def gsi_from_rmr(rmr89: float) -> float:
    """Estimate GSI from RMR89.

    Parameters
    ----------
    rmr89 : float
        Rock Mass Rating (RMR89 basic, without groundwater adjustment set to
        dry = 15). Must be > 23.

    Returns
    -------
    float
        Geological Strength Index.

    Examples
    --------
    >>> gsi_from_rmr(60)
    55.0

    References
    ----------
    .. [1] Hoek, E., Kaiser, P.K. & Bawden, W.F. (1995). "Support of
       Underground Excavations in Hard Rock." Balkema.
    """
    if rmr89 <= 23:
        raise ValueError(
            f"RMR89 must be > 23 for GSI conversion, got {rmr89}. "
            "Use gsi_from_chart() for low RMR values."
        )

    # GSI ≈ RMR89 - 5  (Hoek, Kaiser & Bawden 1995)
    return float(rmr89 - 5)


def gsi_from_chart(
    structure_rating: float,
    surface_rating: float,
) -> float:
    """Estimate GSI from quantitative chart ratings.

    Parameters
    ----------
    structure_rating : float
        Rock structure rating (0-100). Higher = more intact/blocky.
        Typical: massive=85, blocky=65, very blocky=45, disturbed=25,
        disintegrated=10.
    surface_rating : float
        Discontinuity surface condition (0-100). Higher = better.
        Typical: very good=85, good=65, fair=45, poor=25, very poor=10.

    Returns
    -------
    float
        GSI value (0-100).

    Examples
    --------
    >>> gsi_from_chart(65, 65)
    65.0

    References
    ----------
    .. [1] Hoek, E. & Marinos, P. (2000). "Predicting tunnel squeezing
       problems in weak heterogeneous rock masses." Tunnels and Tunnelling
       International.
    """
    validate_range(structure_rating, 0, 100, "structure_rating")
    validate_range(surface_rating, 0, 100, "surface_rating")

    # Quantitative GSI: average of structure and surface ratings
    # Simplified from Hoek & Marinos (2000) chart
    gsi = 0.5 * (structure_rating + surface_rating)

    return float(np.clip(gsi, 0, 100))


def smr_romana(
    rmr: float,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
) -> dict:
    """Slope Mass Rating (Romana 1985).

    Parameters
    ----------
    rmr : float
        Basic RMR89 value.
    f1 : float
        Factor depending on parallelism between joint and slope strikes
        (0.15-1.0).
    f2 : float
        Factor depending on joint dip angle in the planar mode (0.15-1.0).
    f3 : float
        Factor reflecting relationship between slope and joint dips
        (-60 to 0).
    f4 : float
        Adjustment for excavation method (0-15).

    Returns
    -------
    dict
        Keys: ``"smr"`` (SMR value), ``"class_number"`` (I-V),
        ``"description"``, ``"stability"``.

    Examples
    --------
    >>> result = smr_romana(65, 0.7, 0.8, -25, 10)
    >>> result["smr"]
    61
    >>> result["class_number"]
    'II'

    References
    ----------
    .. [1] Romana, M. (1985). "New adjustment ratings for application of
       Bieniawski classification to slopes." Proc. Int. Symp. Role of Rock
       Mech., 49-53.
    """
    validate_range(f1, 0.15, 1.0, "f1")
    validate_range(f2, 0.15, 1.0, "f2")
    validate_range(f3, -60, 0, "f3")
    validate_range(f4, 0, 15, "f4")

    # SMR = RMR_basic + (F1 * F2 * F3) + F4  — Romana 1985
    # F3 is negative (0 to -60), so the product reduces SMR for unfavorable joints
    smr = int(round(rmr + (f1 * f2 * f3) + f4))

    if smr >= 81:
        cls, desc, stab = "I", "Very good", "Completely stable"
    elif smr >= 61:
        cls, desc, stab = "II", "Good", "Stable"
    elif smr >= 41:
        cls, desc, stab = "III", "Normal", "Partially stable"
    elif smr >= 21:
        cls, desc, stab = "IV", "Bad", "Unstable"
    else:
        cls, desc, stab = "V", "Very bad", "Completely unstable"

    return {
        "smr": smr,
        "class_number": cls,
        "description": desc,
        "stability": stab,
    }
