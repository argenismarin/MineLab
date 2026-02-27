"""Ore blending optimization and grade calculation.

This module provides functions for optimizing the blend of multiple ore
sources to meet grade constraints and target tonnages, using linear
programming (LP).

References
----------
.. [1] Hillier, F.S. & Lieberman, G.J. (2015). *Introduction to Operations
       Research*, 10th ed. McGraw-Hill.
.. [2] SciPy optimization reference: ``scipy.optimize.linprog``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Weighted Average Blend Grade
# ---------------------------------------------------------------------------


def blend_grade(source_tonnages: list[float], source_grades: list[float]) -> float:
    """Compute the weighted average grade of a blend.

    The blend grade is the tonnage-weighted average of individual source
    grades:

        blend_grade = sum(t_i * g_i) / sum(t_i)

    Parameters
    ----------
    source_tonnages : list[float]
        Tonnage from each source. All values must be non-negative and at
        least one must be positive.
    source_grades : list[float]
        Grade of each source. Must have the same length as
        *source_tonnages*.

    Returns
    -------
    float
        Weighted average grade of the blended material.

    Raises
    ------
    ValueError
        If inputs have different lengths, are empty, or total tonnage is
        zero.

    Examples
    --------
    >>> blend_grade([100, 200], [1.5, 0.8])
    1.0333333333333334

    References
    ----------
    .. [1] Standard blending practice.
    """
    t = np.asarray(source_tonnages, dtype=float)
    g = np.asarray(source_grades, dtype=float)

    if t.size == 0:
        raise ValueError("source_tonnages must not be empty.")
    if t.size != g.size:
        raise ValueError(
            f"source_tonnages ({t.size}) and source_grades ({g.size}) must have the same length."
        )
    if np.any(t < 0):
        raise ValueError("All source tonnages must be non-negative.")

    total_t = t.sum()
    if total_t == 0:
        raise ValueError("Total tonnage must be positive.")

    return float(np.dot(t, g) / total_t)


# ---------------------------------------------------------------------------
# LP Blend Optimization
# ---------------------------------------------------------------------------


def blend_optimize(
    sources: list[dict],
    grade_constraints: dict,
    tonnage_target: float,
) -> dict:
    """Optimize ore blending using linear programming.

    Determines the optimal tonnage to draw from each source so that the
    resulting blend meets grade constraints while matching a total tonnage
    target. The objective minimises the total tonnage drawn (equivalent to
    minimising cost when unit costs are equal).

    Parameters
    ----------
    sources : list[dict]
        Each source is a dictionary with keys:

        - ``"tonnage_available"`` : float -- maximum tonnage available.
        - ``"grades"`` : dict -- element grades, e.g.
          ``{"Cu": 1.2, "Fe": 30.0}``.

    grade_constraints : dict
        Grade bounds keyed by element name. Each value is a dict with
        optional ``"min"`` and/or ``"max"`` keys, e.g.
        ``{"Cu": {"min": 0.5, "max": 1.5}}``.
    tonnage_target : float
        Required total blend tonnage. Must be positive.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"tonnages"`` : list[float] -- optimal tonnage from each source.
        - ``"blend_grade"`` : dict -- resulting grade per element.
        - ``"feasible"`` : bool -- whether a feasible solution was found.

    Raises
    ------
    ValueError
        If *tonnage_target* is not positive or *sources* is empty.

    Notes
    -----
    The LP formulation is:

    - **Decision variables**: :math:`x_i` = tonnage from source *i*.
    - **Objective**: minimize :math:`\\sum x_i` (can be extended to costs).
    - **Constraints**:

      * :math:`\\sum x_i = T_{\\text{target}}`  (tonnage equality)
      * :math:`0 \\le x_i \\le T_{i,\\text{avail}}`  (bounds)
      * :math:`g_{\\min} \\le \\sum(x_i g_{ij}) / T_{\\text{target}}
        \\le g_{\\max}` per element *j*.

    The grade constraints are linearised by multiplying through by
    *tonnage_target*.

    Examples
    --------
    >>> sources = [
    ...     {"tonnage_available": 500, "grades": {"Cu": 1.5, "Fe": 30}},
    ...     {"tonnage_available": 800, "grades": {"Cu": 0.5, "Fe": 45}},
    ... ]
    >>> constraints = {"Cu": {"min": 0.8, "max": 1.2}}
    >>> result = blend_optimize(sources, constraints, 600)
    >>> result["feasible"]
    True

    References
    ----------
    .. [1] Hillier & Lieberman (2015), Ch. 3 -- Linear Programming.
    .. [2] ``scipy.optimize.linprog`` documentation.
    """
    validate_positive(tonnage_target, "tonnage_target")
    if not sources:
        raise ValueError("sources must not be empty.")

    n = len(sources)
    elements = sorted(grade_constraints.keys())

    # Objective: minimize sum(x_i)  -->  c = [1, 1, ..., 1]
    c = np.ones(n)

    # Inequality constraints: a_ub @ x <= b_ub
    ub_rows: list[list[float]] = []
    ub_vals: list[float] = []

    for elem in elements:
        bounds = grade_constraints[elem]
        grades_for_elem = [s["grades"].get(elem, 0.0) for s in sources]

        if "min" in bounds:
            # g_min * T <= sum(x_i * g_i)  -->  sum(x_i * (g_min - g_i)) <= 0
            g_min = bounds["min"]
            row = [g_min - g for g in grades_for_elem]
            ub_rows.append(row)
            ub_vals.append(0.0)

        if "max" in bounds:
            # sum(x_i * g_i) <= g_max * T  -->  sum(x_i * (g_i - g_max)) <= 0
            g_max = bounds["max"]
            row = [g - g_max for g in grades_for_elem]
            ub_rows.append(row)
            ub_vals.append(0.0)

    a_ub = np.array(ub_rows, dtype=float) if ub_rows else None
    b_ub = np.array(ub_vals, dtype=float) if ub_vals else None

    # Equality constraint: sum(x_i) = tonnage_target
    a_eq = np.ones((1, n))
    b_eq = np.array([tonnage_target])

    # Bounds: 0 <= x_i <= tonnage_available_i
    x_bounds = [(0.0, s["tonnage_available"]) for s in sources]

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=x_bounds,
        method="highs",
    )

    feasible = result.success
    tonnages = result.x.tolist() if feasible else [0.0] * n

    # Compute resulting blend grades
    blend_grades: dict[str, float] = {}
    if feasible:
        total_t = sum(tonnages)
        if total_t > 0:
            for elem in elements:
                grades_for_elem = [s["grades"].get(elem, 0.0) for s in sources]
                weighted = sum(t * g for t, g in zip(tonnages, grades_for_elem, strict=True))
                blend_grades[elem] = weighted / total_t
    else:
        for elem in elements:
            blend_grades[elem] = 0.0

    return {
        "tonnages": tonnages,
        "blend_grade": blend_grades,
        "feasible": feasible,
    }
