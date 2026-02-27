"""Pushback (phase) generation for open-pit mine planning.

Pushbacks are created from nested pit shells obtained by running pit
optimisation at different revenue factors (parametric analysis).
Adjacent shells can then be merged into practical mining phases.

References
----------
.. [1] Whittle, J. (1999). A decade of open pit mine planning and
       optimisation. *Proceedings of APCOM*, 515-522.
"""

from __future__ import annotations

import numpy as np

from minelab.mine_planning.pit_optimization import lerchs_grossmann_2d
from minelab.utilities.validators import validate_non_negative, validate_positive


def nested_pit_shells(
    block_values: np.ndarray,
    revenue_factors: list[float],
    slope_angles: tuple[float, float],
) -> dict:
    """Generate nested pit shells at varying revenue factors.

    Runs the 2-D Lerchs-Grossmann pit optimisation for each revenue
    factor (RF).  Block values are scaled by RF before optimisation, so
    lower RFs yield smaller pits and higher RFs yield larger pits.

    Parameters
    ----------
    block_values : numpy.ndarray
        2-D array of base-case economic block values.
    revenue_factors : list of float
        Revenue factors to evaluate.  Each must be positive.  Typical
        range is 0.3 to 1.2.  The list should be sorted ascending.
    slope_angles : tuple of float
        ``(left_angle_deg, right_angle_deg)`` passed to the pit
        optimiser.

    Returns
    -------
    dict
        ``"shells"`` : list of numpy.ndarray
            List of boolean pit masks, one per revenue factor.
        ``"revenue_factors"`` : list of float
            Corresponding revenue factors.
        ``"shell_values"`` : list of float
            Total value inside each shell at base-case prices.

    Raises
    ------
    ValueError
        If *revenue_factors* is empty or contains non-positive values.

    Examples
    --------
    >>> import numpy as np
    >>> bv = np.array([[-2, 10, -2],
    ...                [-3, 20, -3]])
    >>> res = nested_pit_shells(bv, [0.5, 1.0], (45.0, 45.0))
    >>> len(res["shells"])
    2
    >>> res["shells"][1].sum() >= res["shells"][0].sum()
    True

    References
    ----------
    .. [1] Whittle, J. (1999). A decade of open pit mine planning and
           optimisation. *Proceedings of APCOM*, 515-522.
    """
    block_values = np.asarray(block_values, dtype=float)
    if len(revenue_factors) == 0:
        raise ValueError("'revenue_factors' must not be empty.")

    shells: list[np.ndarray] = []
    shell_values: list[float] = []

    for rf in revenue_factors:
        validate_positive(rf, "revenue_factor")
        scaled = block_values * rf
        result = lerchs_grossmann_2d(scaled, slope_angles)
        pit_mask = result["pit_mask"]
        shells.append(pit_mask)
        # Compute value at base-case (unscaled) prices
        base_value = float(block_values[pit_mask].sum()) if pit_mask.any() else 0.0
        shell_values.append(base_value)

    return {
        "shells": shells,
        "revenue_factors": list(revenue_factors),
        "shell_values": shell_values,
    }


def design_pushbacks(
    shells: list[np.ndarray],
    min_width: float = 0.0,
    min_tonnage: float = 0.0,
) -> dict:
    """Merge adjacent nested shells into practical pushback phases.

    Starting from the innermost (smallest) shell, incremental material
    between successive shells forms a candidate pushback.  If a
    candidate is too thin (fewer blocks than implied by *min_width*) or
    too small (total blocks below the tonnage proxy *min_tonnage*), it
    is merged with the next shell outward.

    Parameters
    ----------
    shells : list of numpy.ndarray
        Nested boolean pit masks ordered from smallest (lowest revenue
        factor) to largest.
    min_width : float, optional
        Minimum number of block-widths for a pushback to be standalone.
        Default 0 (no minimum).
    min_tonnage : float, optional
        Minimum block count proxy for tonnage.  Pushbacks with fewer
        blocks are merged.  Default 0 (no minimum).

    Returns
    -------
    dict
        ``"pushbacks"`` : list of numpy.ndarray
            Boolean masks for each pushback (incremental, not
            cumulative).
        ``"tonnages"`` : list of int
            Block counts (tonnage proxy) for each pushback.

    Raises
    ------
    ValueError
        If *shells* is empty.

    Examples
    --------
    >>> import numpy as np
    >>> s1 = np.array([[False, True, False],
    ...                [False, True, False]])
    >>> s2 = np.array([[True, True, True],
    ...                [True, True, True]])
    >>> res = design_pushbacks([s1, s2])
    >>> len(res["pushbacks"])
    2
    >>> res["tonnages"][0] + res["tonnages"][1] == s2.sum()
    True

    References
    ----------
    .. [1] Whittle, J. (1999). A decade of open pit mine planning and
           optimisation. *Proceedings of APCOM*, 515-522.
    """
    if len(shells) == 0:
        raise ValueError("'shells' must not be empty.")

    validate_non_negative(min_width, "min_width")
    validate_non_negative(min_tonnage, "min_tonnage")

    # Compute incremental shells (material added by each successive shell)
    incremental: list[np.ndarray] = []
    prev = np.zeros_like(shells[0], dtype=bool)
    for shell in shells:
        shell = np.asarray(shell, dtype=bool)
        diff = shell & ~prev
        incremental.append(diff)
        prev = shell.copy()

    # Merge small pushbacks into the next one
    pushbacks: list[np.ndarray] = []
    tonnages: list[int] = []

    accumulator = np.zeros_like(shells[0], dtype=bool)
    acc_count = 0

    for inc in incremental:
        accumulator = accumulator | inc
        acc_count = int(accumulator.sum())

        # Check if the pushback meets minimum requirements
        if acc_count >= min_tonnage and acc_count >= min_width:
            pushbacks.append(accumulator.copy())
            tonnages.append(acc_count)
            accumulator = np.zeros_like(shells[0], dtype=bool)
            acc_count = 0

    # Flush any remaining accumulated material
    if acc_count > 0:
        if len(pushbacks) > 0:
            # Merge into the last pushback
            pushbacks[-1] = pushbacks[-1] | accumulator
            tonnages[-1] = int(pushbacks[-1].sum())
        else:
            pushbacks.append(accumulator.copy())
            tonnages.append(acc_count)

    return {"pushbacks": pushbacks, "tonnages": tonnages}
