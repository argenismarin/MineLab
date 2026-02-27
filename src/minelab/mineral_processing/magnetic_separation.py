"""Magnetic separation calculations.

Magnetic susceptibility classification and Davis tube recovery.
"""

from __future__ import annotations

from minelab.utilities.validators import validate_positive

# Known magnetic susceptibilities (SI units, approximate)
_MINERAL_SUSCEPTIBILITY: dict[str, tuple[float, str]] = {
    "magnetite": (6.0, "ferromagnetic"),
    "pyrrhotite": (1.5, "ferromagnetic"),
    "ilmenite": (0.003, "paramagnetic"),
    "chromite": (0.003, "paramagnetic"),
    "hematite": (0.001, "paramagnetic"),
    "garnet": (0.0005, "paramagnetic"),
    "siderite": (0.001, "paramagnetic"),
    "quartz": (-1.5e-5, "diamagnetic"),
    "calcite": (-1.3e-5, "diamagnetic"),
    "feldspar": (-1.0e-5, "diamagnetic"),
}


def magnetic_susceptibility_classify(
    minerals: list[str],
) -> list[dict]:
    """Classify minerals by magnetic susceptibility.

    Parameters
    ----------
    minerals : list of str
        Mineral names (case-insensitive).

    Returns
    -------
    list of dict
        Each dict: ``"mineral"``, ``"class"`` (ferromagnetic, paramagnetic,
        or diamagnetic), ``"susceptibility"`` (SI units).

    Examples
    --------
    >>> result = magnetic_susceptibility_classify(["magnetite", "quartz"])
    >>> result[0]["class"]
    'ferromagnetic'
    >>> result[1]["class"]
    'diamagnetic'

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed., Ch.13.
    """
    results = []
    for mineral in minerals:
        key = mineral.lower().strip()
        susc, cls = _MINERAL_SUSCEPTIBILITY.get(key, (0.0, "unknown"))

        results.append(
            {
                "mineral": mineral,
                "class": cls,
                "susceptibility": float(susc),
            }
        )

    return results


def davis_tube_recovery(
    feed_weight: float,
    magnetic_weight: float,
    feed_grade: float = 0.0,
    magnetic_grade: float = 0.0,
) -> dict:
    """Davis tube test recovery and grade calculation.

    Parameters
    ----------
    feed_weight : float
        Feed sample weight (g).
    magnetic_weight : float
        Magnetic fraction weight (g).
    feed_grade : float
        Feed grade of element of interest (%). Default 0.
    magnetic_grade : float
        Magnetic fraction grade (%). Default 0.

    Returns
    -------
    dict
        Keys: ``"weight_recovery"`` (fraction), ``"grade_recovery"``
        (fraction, if grades provided), ``"upgrade_ratio"`` (float).

    Examples
    --------
    >>> result = davis_tube_recovery(100, 30, 20, 55)
    >>> round(result["weight_recovery"], 2)
    0.3

    References
    ----------
    .. [1] Wills, B.A. & Finch, J.A. (2016). Wills' Mineral Processing
       Technology. 8th ed.
    """
    validate_positive(feed_weight, "feed_weight")

    weight_recovery = magnetic_weight / feed_weight

    grade_recovery = 0.0
    upgrade_ratio = 1.0

    if feed_grade > 0 and magnetic_grade > 0:
        grade_recovery = (magnetic_weight * magnetic_grade) / (feed_weight * feed_grade)
        upgrade_ratio = magnetic_grade / feed_grade

    return {
        "weight_recovery": float(weight_recovery),
        "grade_recovery": float(grade_recovery),
        "upgrade_ratio": float(upgrade_ratio),
    }
