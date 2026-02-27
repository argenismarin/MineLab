"""Underground blast design and vibration control.

This module provides functions for tunnel blasting design, including cut
hole patterns, burn cut advance prediction, powder factor, controlled
blasting PPV, pre-split parameters, delay timing design, and vibration
limits per DIN 4150-3.

References
----------
.. [1] Langefors, U. & Kihlström, B. (1963). *The Modern Technique of
       Rock Blasting*. Almqvist & Wiksell.
.. [2] Persson, P.-A., Holmberg, R. & Lee, J. (1994). *Rock Blasting and
       Explosives Engineering*. CRC Press.
.. [3] Baillin, J.L. (1993). "Burn-cut blast design for underground
       mines." *Proceedings, 4th Int. Symp. Rock Fragmentation by
       Blasting*, Vienna, pp. 181--188.
.. [4] DIN 4150-3 (1999). *Structural vibration -- Part 3: Effects of
       vibration on structures*.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)

# Default ANFO bulk density (kg/m3)
_EXPLOSIVE_DENSITY: float = 1200.0

# DIN 4150-3 foundation vibration limits by structure type (mm/s)
_VIBRATION_LIMITS: dict[str, float] = {
    "commercial": 20.0,
    "residential": 5.0,
    "sensitive": 3.0,
}


# ---------------------------------------------------------------------------
# Cut Hole Design
# ---------------------------------------------------------------------------


def cut_hole_design(
    hole_diameter: float,
    uncharged_hole_diameter: float,
    burden: float,
) -> dict:
    """Design a four-hole box-cut pattern for tunnel blasting.

    The relief (uncharged) hole must be larger than the charged holes to
    provide a free face.  Spacing is computed as the diagonal of the
    burden square, and charge per hole assumes ANFO at 1200 kg/m3.

    .. math::

        S = B \\sqrt{2}

    .. math::

        q = \\frac{\\pi}{4} d^2 \\times B \\times \\rho_e

    where *d* is charged hole diameter (m), *B* is burden (m), and
    *rho_e* is explosive density (kg/m3).

    Parameters
    ----------
    hole_diameter : float
        Charged hole diameter in mm. Must be positive.
    uncharged_hole_diameter : float
        Relief (uncharged) hole diameter in mm. Must be greater than
        *hole_diameter*.
    burden : float
        Burden distance in metres. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"spacing_m"`` : float -- Pattern spacing (m).
        - ``"charge_per_hole_kg"`` : float -- ANFO charge per hole (kg).
        - ``"relief_hole_diameter_mm"`` : float -- Relief hole diameter.
        - ``"pattern_type"`` : str -- Always ``"four_hole_box_cut"``.

    Raises
    ------
    ValueError
        If *uncharged_hole_diameter* <= *hole_diameter*.

    Examples
    --------
    >>> result = cut_hole_design(45, 102, 0.5)
    >>> round(result["spacing_m"], 4)
    0.7071
    >>> result["pattern_type"]
    'four_hole_box_cut'

    References
    ----------
    .. [1] Baillin (1993). Burn-cut blast design for underground mines.
    .. [2] Langefors & Kihlström (1963). The Modern Technique of Rock
           Blasting, Ch. 7.
    """
    validate_positive(hole_diameter, "hole_diameter")
    validate_positive(uncharged_hole_diameter, "uncharged_hole_diameter")
    validate_positive(burden, "burden")

    if uncharged_hole_diameter <= hole_diameter:
        raise ValueError(
            "'uncharged_hole_diameter' must be greater than "
            f"'hole_diameter', got {uncharged_hole_diameter} <= "
            f"{hole_diameter}."
        )

    spacing = burden * math.sqrt(2)

    # Convert diameter from mm to m for volume calculation
    d_m = hole_diameter / 1000.0
    charge_per_hole = (math.pi / 4.0) * d_m**2 * burden * _EXPLOSIVE_DENSITY

    return {
        "spacing_m": float(spacing),
        "charge_per_hole_kg": float(charge_per_hole),
        "relief_hole_diameter_mm": float(uncharged_hole_diameter),
        "pattern_type": "four_hole_box_cut",
    }


# ---------------------------------------------------------------------------
# Burn Cut Advance
# ---------------------------------------------------------------------------


def burn_cut_advance(
    drill_length: float,
    charge_ratio: float,
    rock_factor: float,
) -> float:
    """Predict burn-cut advance per round.

    .. math::

        A = L \\times \\eta \\times f_r

    where *L* is drill-hole length (m), *eta* is the charge ratio
    (typically 0.85--0.95), and *f_r* is the rock factor (0.8--1.0).

    Parameters
    ----------
    drill_length : float
        Drill-hole length in metres. Must be positive.
    charge_ratio : float
        Ratio of charged length to drilled length, in [0.85, 0.95].
    rock_factor : float
        Rock quality factor, in [0.8, 1.0]. Lower values indicate
        harder or more massive rock.

    Returns
    -------
    float
        Expected advance per round in metres.

    Examples
    --------
    >>> burn_cut_advance(3.0, 0.90, 0.95)
    2.565

    References
    ----------
    .. [1] Langefors, U. & Kihlström, B. (1963). The Modern Technique
           of Rock Blasting, Ch. 9.
    """
    validate_positive(drill_length, "drill_length")
    validate_range(charge_ratio, 0.85, 0.95, "charge_ratio")
    validate_range(rock_factor, 0.8, 1.0, "rock_factor")

    return float(drill_length * charge_ratio * rock_factor)


# ---------------------------------------------------------------------------
# Tunnel Blast Powder Factor
# ---------------------------------------------------------------------------


def tunnel_blast_powder_factor(
    charge_per_blast_kg: float,
    volume_blasted_m3: float,
) -> float:
    """Compute powder factor for a tunnel blast round.

    .. math::

        PF = \\frac{Q}{V}

    Parameters
    ----------
    charge_per_blast_kg : float
        Total explosive charge in kg. Must be positive.
    volume_blasted_m3 : float
        Volume of rock broken in m3. Must be positive.

    Returns
    -------
    float
        Powder factor in kg/m3.

    Examples
    --------
    >>> tunnel_blast_powder_factor(50, 25)
    2.0

    References
    ----------
    .. [1] Langefors & Kihlström (1963). The Modern Technique of Rock
           Blasting, Ch. 3.
    """
    validate_positive(charge_per_blast_kg, "charge_per_blast_kg")
    validate_positive(volume_blasted_m3, "volume_blasted_m3")

    return float(charge_per_blast_kg / volume_blasted_m3)


# ---------------------------------------------------------------------------
# Controlled Blasting PPV
# ---------------------------------------------------------------------------


def controlled_blasting_ppv(
    charge_per_delay_kg: float,
    distance_m: float,
    k: float,
    alpha: float,
) -> float:
    """Predict peak particle velocity for controlled underground blasting.

    Uses the scaled-distance attenuation law:

    .. math::

        PPV = K \\left(\\frac{D}{\\sqrt{W}}\\right)^{-\\alpha}

    where *K* and *alpha* are site-calibrated constants, *D* is the
    distance, and *W* is the maximum charge per delay.

    Parameters
    ----------
    charge_per_delay_kg : float
        Maximum charge per delay in kg. Must be positive.
    distance_m : float
        Distance from the blast in metres. Must be positive.
    k : float
        Site constant (intercept). Must be positive.
    alpha : float
        Attenuation exponent. Must be positive. Typical range
        1.0--2.0 for underground hard rock.

    Returns
    -------
    float
        Predicted peak particle velocity in mm/s.

    Examples
    --------
    >>> round(controlled_blasting_ppv(10, 50, 700, 1.5), 2)
    5.6

    References
    ----------
    .. [1] Persson, P.-A. et al. (1994). Rock Blasting and Explosives
           Engineering, Ch. 12.
    """
    validate_positive(charge_per_delay_kg, "charge_per_delay_kg")
    validate_positive(distance_m, "distance_m")
    validate_positive(k, "k")
    validate_positive(alpha, "alpha")

    sd = distance_m / math.sqrt(charge_per_delay_kg)
    ppv = k * sd ** (-alpha)

    return float(ppv)


# ---------------------------------------------------------------------------
# Pre-split Parameters
# ---------------------------------------------------------------------------


def presplit_parameters(
    hole_diameter: float,
    rock_tensile_strength: float,
    hole_spacing: float,
) -> dict:
    """Compute pre-split blasting design parameters.

    Determines the linear charge density, recommended spacing, and
    spacing-to-diameter ratio for controlled pre-splitting.

    .. math::

        q_l = \\frac{\\pi}{4} (d \\times r_d)^2 \\times \\rho_e

    where *d* is hole diameter (m), *r_d* = 0.5 is the decoupling
    ratio, and *rho_e* = 1200 kg/m3.

    Parameters
    ----------
    hole_diameter : float
        Hole diameter in mm. Must be positive.
    rock_tensile_strength : float
        Uniaxial tensile strength in MPa. Must be positive.
    hole_spacing : float
        Actual hole spacing in metres. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"linear_charge_kg_per_m"`` : float -- Charge per metre of
          hole (kg/m).
        - ``"recommended_spacing_m"`` : float -- Recommended spacing
          based on 10--12 x diameter rule (midpoint = 11d).
        - ``"spacing_to_diameter_ratio"`` : float -- Actual spacing
          divided by hole diameter.

    Examples
    --------
    >>> result = presplit_parameters(76, 10, 0.8)
    >>> round(result["linear_charge_kg_per_m"], 4)
    0.8545
    >>> round(result["recommended_spacing_m"], 3)
    0.836

    References
    ----------
    .. [1] Persson et al. (1994). Rock Blasting and Explosives
           Engineering, Ch. 14 (Controlled Blasting).
    """
    validate_positive(hole_diameter, "hole_diameter")
    validate_positive(rock_tensile_strength, "rock_tensile_strength")
    validate_positive(hole_spacing, "hole_spacing")

    decoupling_ratio = 0.5
    d_m = hole_diameter / 1000.0  # mm -> m
    charge_diameter = d_m * decoupling_ratio

    # Linear charge density (kg per metre of hole)
    linear_charge = (math.pi / 4.0) * charge_diameter**2 * _EXPLOSIVE_DENSITY

    # Recommended spacing: midpoint of 10-12 x diameter rule
    recommended_spacing = 11.0 * hole_diameter / 1000.0

    # Spacing-to-diameter ratio (dimensionless, both in mm)
    s_to_d = hole_spacing * 1000.0 / hole_diameter

    return {
        "linear_charge_kg_per_m": float(linear_charge),
        "recommended_spacing_m": float(recommended_spacing),
        "spacing_to_diameter_ratio": float(s_to_d),
    }


# ---------------------------------------------------------------------------
# Delay Timing Design
# ---------------------------------------------------------------------------


def delay_timing_design(
    number_of_holes: int,
    ms_per_hole: float,
    detonation_sequence: list[int] | None = None,
) -> dict:
    """Design delay timing schedule for an underground blast.

    If a detonation sequence is provided (list of hole counts per delay
    group), timing is allocated per group.  Otherwise, sequential firing
    (one hole per delay) is assumed.

    Parameters
    ----------
    number_of_holes : int
        Total number of blast holes. Must be positive.
    ms_per_hole : float
        Delay interval in milliseconds between successive detonations.
        Must be positive.
    detonation_sequence : list of int, optional
        Number of holes firing simultaneously at each delay step.
        If ``None`` or empty, sequential (1 hole per delay) is used.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"total_blast_time_ms"`` : float -- Total duration of the
          blast in milliseconds.
        - ``"number_of_delays"`` : int -- Total number of delay groups.
        - ``"timing_schedule"`` : list of dict -- Each entry has
          ``"delay_number"``, ``"time_ms"``, and ``"holes_in_group"``.

    Raises
    ------
    ValueError
        If sum of detonation_sequence does not equal number_of_holes.

    Examples
    --------
    >>> result = delay_timing_design(4, 25)
    >>> result["total_blast_time_ms"]
    75.0
    >>> result["number_of_delays"]
    4

    >>> result = delay_timing_design(8, 50, [2, 2, 2, 2])
    >>> result["total_blast_time_ms"]
    150.0
    >>> result["number_of_delays"]
    4

    References
    ----------
    .. [1] Persson et al. (1994). Rock Blasting and Explosives
           Engineering, Ch. 10.
    """
    if number_of_holes <= 0:
        raise ValueError(f"'number_of_holes' must be positive, got {number_of_holes}.")
    validate_positive(ms_per_hole, "ms_per_hole")

    if detonation_sequence is None or len(detonation_sequence) == 0:
        # Sequential: one hole per delay
        seq = [1] * number_of_holes
    else:
        seq = list(detonation_sequence)
        total_in_seq = sum(seq)
        if total_in_seq != number_of_holes:
            raise ValueError(
                "Sum of 'detonation_sequence' must equal "
                f"'number_of_holes' ({number_of_holes}), "
                f"got {total_in_seq}."
            )

    group_count = len(seq)
    total_blast_time = (group_count - 1) * ms_per_hole

    timing_schedule = []
    for i, holes_in_group in enumerate(seq):
        timing_schedule.append(
            {
                "delay_number": i + 1,
                "time_ms": float(i * ms_per_hole),
                "holes_in_group": holes_in_group,
            }
        )

    return {
        "total_blast_time_ms": float(total_blast_time),
        "number_of_delays": group_count,
        "timing_schedule": timing_schedule,
    }


# ---------------------------------------------------------------------------
# Underground Blast Vibration Limit
# ---------------------------------------------------------------------------


def underground_blast_vibration_limit(
    depth_below_surface: float,
    structure_type: str,
) -> float:
    """Determine allowable PPV for underground blasting near structures.

    Uses DIN 4150-3 foundation vibration limits with a depth correction
    factor to account for increased attenuation at greater depths:

    .. math::

        PPV_{\\text{allow}} = L \\times \\sqrt{1 + \\frac{z}{100}}

    where *L* is the base limit for the structure type (mm/s) and *z*
    is the depth below surface (m).

    Parameters
    ----------
    depth_below_surface : float
        Depth of the blast below surface in metres. Must be positive.
    structure_type : str
        Type of surface structure. Recognised types and base limits:
        ``"commercial"`` (20 mm/s), ``"residential"`` (5 mm/s),
        ``"sensitive"`` (3 mm/s).

    Returns
    -------
    float
        Allowable peak particle velocity in mm/s.

    Raises
    ------
    ValueError
        If *structure_type* is not recognised.

    Examples
    --------
    >>> round(underground_blast_vibration_limit(100, "residential"), 2)
    7.07

    >>> round(underground_blast_vibration_limit(50, "commercial"), 2)
    24.49

    References
    ----------
    .. [1] DIN 4150-3 (1999). Structural vibration -- Part 3: Effects
           of vibration on structures, Table 3.
    """
    validate_positive(depth_below_surface, "depth_below_surface")

    key = structure_type.lower()
    if key not in _VIBRATION_LIMITS:
        allowed = ", ".join(sorted(_VIBRATION_LIMITS.keys()))
        raise ValueError(f"Unknown structure_type '{structure_type}'. Supported: {allowed}.")

    base_limit = _VIBRATION_LIMITS[key]
    depth_correction = math.sqrt(1.0 + depth_below_surface / 100.0)

    return float(base_limit * depth_correction)
