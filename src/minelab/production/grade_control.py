"""Grade control: SMU classification and information effect analysis.

This module provides functions for classifying selective mining unit (SMU)
blocks into ore and waste categories and for quantifying the information
effect (tonnage/grade changes caused by estimation smoothing).

References
----------
.. [1] Isaaks, E.H. & Srivastava, R.M. (1989). *An Introduction to Applied
       Geostatistics*. Oxford University Press, Ch. 18.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# SMU Classification
# ---------------------------------------------------------------------------


def smu_classification(block_grades: np.ndarray, cutoff: float) -> dict:
    """Classify selective mining unit blocks as ore or waste.

    Blocks with grade >= *cutoff* are classified as ore; all others are
    classified as waste.

    Parameters
    ----------
    block_grades : numpy.ndarray
        1-D array of block grades (e.g. % Cu).
    cutoff : float
        Cut-off grade. Blocks with grade >= *cutoff* are ore.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"ore_count"`` : int -- number of ore blocks.
        - ``"waste_count"`` : int -- number of waste blocks.
        - ``"ore_grade"`` : float -- mean grade of ore blocks (0.0 if none).
        - ``"waste_grade"`` : float -- mean grade of waste blocks
          (0.0 if none).

    Raises
    ------
    ValueError
        If *block_grades* is empty.

    Examples
    --------
    >>> import numpy as np
    >>> grades = np.array([0.3, 0.8, 1.2, 0.1, 0.5])
    >>> result = smu_classification(grades, 0.5)
    >>> result["ore_count"]
    3
    >>> result["waste_count"]
    2

    References
    ----------
    .. [1] Isaaks & Srivastava (1989), Ch. 18 -- Change of Support.
    """
    grades = np.asarray(block_grades, dtype=float).ravel()
    if grades.size == 0:
        raise ValueError("block_grades must not be empty.")

    ore_mask = grades >= cutoff
    waste_mask = ~ore_mask

    ore_grades = grades[ore_mask]
    waste_grades = grades[waste_mask]

    return {
        "ore_count": int(ore_mask.sum()),
        "waste_count": int(waste_mask.sum()),
        "ore_grade": float(ore_grades.mean()) if ore_grades.size > 0 else 0.0,
        "waste_grade": float(waste_grades.mean()) if waste_grades.size > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Information Effect
# ---------------------------------------------------------------------------


def information_effect(
    true_grades: np.ndarray,
    estimated_grades: np.ndarray,
    cutoff: float,
) -> dict:
    """Quantify tonnage and grade changes from estimation smoothing.

    Compares classification outcomes (ore/waste at *cutoff*) between true
    and estimated block grades. The information effect arises because
    estimation smooths the grade distribution, causing misclassification at
    the cutoff boundary.

    Parameters
    ----------
    true_grades : numpy.ndarray
        1-D array of true (simulated or actual) block grades.
    estimated_grades : numpy.ndarray
        1-D array of estimated (kriged) block grades. Must have the same
        length as *true_grades*.
    cutoff : float
        Cut-off grade used for ore/waste classification.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"true_ore_tonnage"`` : int -- ore blocks from true grades.
        - ``"est_ore_tonnage"`` : int -- ore blocks from estimated grades.
        - ``"tonnage_change_pct"`` : float -- percentage change in ore
          tonnage ((est - true) / true * 100).
        - ``"true_ore_grade"`` : float -- mean ore grade from true grades.
        - ``"est_ore_grade"`` : float -- mean ore grade from estimated grades.
        - ``"grade_change_pct"`` : float -- percentage change in ore grade.

    Raises
    ------
    ValueError
        If arrays have different lengths or are empty.

    Examples
    --------
    >>> import numpy as np
    >>> true = np.array([0.3, 0.8, 1.2, 0.1, 0.5, 1.0])
    >>> est = np.array([0.5, 0.7, 0.9, 0.3, 0.6, 0.8])
    >>> result = information_effect(true, est, 0.5)
    >>> result["true_ore_tonnage"]
    4
    >>> result["est_ore_tonnage"]
    5

    References
    ----------
    .. [1] Isaaks & Srivastava (1989), Ch. 18.
    """
    true_g = np.asarray(true_grades, dtype=float).ravel()
    est_g = np.asarray(estimated_grades, dtype=float).ravel()

    if true_g.size == 0:
        raise ValueError("true_grades must not be empty.")
    if true_g.size != est_g.size:
        raise ValueError(
            f"true_grades ({true_g.size}) and estimated_grades ({est_g.size}) "
            f"must have the same length."
        )

    true_ore_mask = true_g >= cutoff
    est_ore_mask = est_g >= cutoff

    true_ore_t = int(true_ore_mask.sum())
    est_ore_t = int(est_ore_mask.sum())

    true_ore_g = float(true_g[true_ore_mask].mean()) if true_ore_t > 0 else 0.0
    est_ore_g = float(est_g[est_ore_mask].mean()) if est_ore_t > 0 else 0.0

    tonnage_change_pct = (est_ore_t - true_ore_t) / true_ore_t * 100.0 if true_ore_t > 0 else 0.0
    grade_change_pct = (est_ore_g - true_ore_g) / true_ore_g * 100.0 if true_ore_g > 0 else 0.0

    return {
        "true_ore_tonnage": true_ore_t,
        "est_ore_tonnage": est_ore_t,
        "tonnage_change_pct": float(tonnage_change_pct),
        "true_ore_grade": true_ore_g,
        "est_ore_grade": est_ore_g,
        "grade_change_pct": float(grade_change_pct),
    }
