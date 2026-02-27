"""Resource reporting: statements and grade-tonnage curves by category.

This module provides functions for generating mineral resource statements
and grade-tonnage curves, grouped by classification category (Measured,
Indicated, Inferred).

References
----------
.. [1] JORC (2012). *Australasian Code for Reporting of Exploration
       Results, Mineral Resources and Ore Reserves*. Joint Ore Reserves
       Committee of The Australasian Institute of Mining and Metallurgy.
.. [2] CIM (2014). *CIM Definition Standards for Mineral Resources and
       Mineral Reserves*. Canadian Institute of Mining, Metallurgy and
       Petroleum.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Category name mapping
# ---------------------------------------------------------------------------

_CATEGORY_NAMES: dict[int, str] = {
    1: "measured",
    2: "indicated",
    3: "inferred",
}


# ---------------------------------------------------------------------------
# Resource Statement
# ---------------------------------------------------------------------------


def resource_statement(
    block_tonnages: np.ndarray,
    block_grades: np.ndarray,
    classification: np.ndarray,
    cutoff: float,
    density: float = 2.7,
) -> dict:
    """Generate a mineral resource statement above a cut-off grade.

    Summarises tonnes, grade, and contained metal by classification
    category for all blocks with grade >= *cutoff*.

    Parameters
    ----------
    block_tonnages : numpy.ndarray
        1-D array of block tonnages.
    block_grades : numpy.ndarray
        1-D array of block grades.
    classification : numpy.ndarray
        1-D integer array of category codes (1 = Measured, 2 = Indicated,
        3 = Inferred).
    cutoff : float
        Cut-off grade. Only blocks with grade >= *cutoff* are included.
    density : float, optional
        Rock density in t/m^3 (default 2.7). Informational; tonnages are
        assumed to already incorporate density.

    Returns
    -------
    dict
        Dictionary with keys ``"measured"``, ``"indicated"``,
        ``"inferred"``. Each contains:

        - ``"tonnes"`` : float -- total tonnage above cut-off.
        - ``"grade"`` : float -- tonnage-weighted average grade.
        - ``"metal"`` : float -- contained metal (tonnes * grade).

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty.

    Examples
    --------
    >>> import numpy as np
    >>> tonnes = np.array([1000, 1500, 800, 1200, 500])
    >>> grades = np.array([1.2, 1.0, 0.8, 0.3, 0.5])
    >>> cls = np.array([1, 1, 2, 2, 3])
    >>> result = resource_statement(tonnes, grades, cls, cutoff=0.5)
    >>> result["measured"]["tonnes"]
    2500.0

    References
    ----------
    .. [1] JORC Code (2012), Table 1.
    """
    t = np.asarray(block_tonnages, dtype=float).ravel()
    g = np.asarray(block_grades, dtype=float).ravel()
    cls = np.asarray(classification, dtype=int).ravel()

    if t.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (t.size == g.size == cls.size):
        raise ValueError(
            f"All input arrays must have the same length. Got "
            f"block_tonnages={t.size}, block_grades={g.size}, "
            f"classification={cls.size}."
        )

    validate_positive(density, "density")

    # Apply cut-off filter
    above_cutoff = g >= cutoff

    result: dict = {}

    for code, name in _CATEGORY_NAMES.items():
        mask = above_cutoff & (cls == code)
        cat_tonnes = float(t[mask].sum())
        cat_metal = float((t[mask] * g[mask]).sum())
        cat_grade = cat_metal / cat_tonnes if cat_tonnes > 0 else 0.0

        result[name] = {
            "tonnes": cat_tonnes,
            "grade": cat_grade,
            "metal": cat_metal,
        }

    return result


# ---------------------------------------------------------------------------
# Grade-Tonnage Curves by Category
# ---------------------------------------------------------------------------


def grade_tonnage_by_category(
    block_tonnages: np.ndarray,
    block_grades: np.ndarray,
    classification: np.ndarray,
    cutoffs: np.ndarray,
) -> dict:
    """Compute grade-tonnage curves for each classification category.

    For each cut-off grade in *cutoffs*, computes the total tonnage and
    average grade of blocks above that cut-off, separated by category.

    Parameters
    ----------
    block_tonnages : numpy.ndarray
        1-D array of block tonnages.
    block_grades : numpy.ndarray
        1-D array of block grades.
    classification : numpy.ndarray
        1-D integer array of category codes (1 = Measured, 2 = Indicated,
        3 = Inferred).
    cutoffs : numpy.ndarray
        1-D array of cut-off grade values to evaluate.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"cutoffs"`` : list[float] -- the cut-off values used.
        - ``"measured"`` : dict with ``"tonnes"`` and ``"grade"`` lists.
        - ``"indicated"`` : dict with ``"tonnes"`` and ``"grade"`` lists.
        - ``"inferred"`` : dict with ``"tonnes"`` and ``"grade"`` lists.

    Raises
    ------
    ValueError
        If input arrays have different lengths or are empty.

    Examples
    --------
    >>> import numpy as np
    >>> tonnes = np.array([1000, 1500, 800, 1200, 500])
    >>> grades = np.array([1.2, 1.0, 0.8, 0.3, 0.5])
    >>> cls = np.array([1, 1, 2, 2, 3])
    >>> cutoffs = np.array([0.0, 0.5, 1.0])
    >>> result = grade_tonnage_by_category(tonnes, grades, cls, cutoffs)
    >>> len(result["cutoffs"])
    3

    References
    ----------
    .. [1] JORC Code (2012), Table 1.
    """
    t = np.asarray(block_tonnages, dtype=float).ravel()
    g = np.asarray(block_grades, dtype=float).ravel()
    cls = np.asarray(classification, dtype=int).ravel()
    co = np.asarray(cutoffs, dtype=float).ravel()

    if t.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (t.size == g.size == cls.size):
        raise ValueError(
            f"All input arrays must have the same length. Got "
            f"block_tonnages={t.size}, block_grades={g.size}, "
            f"classification={cls.size}."
        )
    if co.size == 0:
        raise ValueError("cutoffs must not be empty.")

    result: dict = {
        "cutoffs": co.tolist(),
    }

    for code, name in _CATEGORY_NAMES.items():
        cat_mask = cls == code
        cat_t = t[cat_mask]
        cat_g = g[cat_mask]

        tonnes_list: list[float] = []
        grade_list: list[float] = []

        for c in co:
            above = cat_g >= c
            above_tonnes = float(cat_t[above].sum())
            above_metal = float((cat_t[above] * cat_g[above]).sum())
            above_grade = above_metal / above_tonnes if above_tonnes > 0 else 0.0

            tonnes_list.append(above_tonnes)
            grade_list.append(above_grade)

        result[name] = {
            "tonnes": tonnes_list,
            "grade": grade_list,
        }

    return result
