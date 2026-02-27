"""Pit optimization algorithms for open-pit mine design.

Provides the Lerchs-Grossmann 2D algorithm, a simplified 3D pseudo-flow
approach, and block economic value calculation for ultimate pit limit
determination.

References
----------
.. [1] Lerchs, H. & Grossmann, I. F. (1965). Optimum design of open-pit
       mines. *CIM Bulletin*, 58, 47-54.
.. [2] Hochbaum, D. S. (2008). The pseudoflow algorithm: A new algorithm for
       the maximum-flow problem. *Operations Research*, 56(4), 992-1009.
.. [3] Whittle, J. (1999). A decade of open pit mine planning and
       optimisation. *Proceedings of APCOM*, 515-522.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)


def lerchs_grossmann_2d(
    block_values: np.ndarray,
    slope_angles: tuple[float, float],
) -> dict:
    """2D optimal pit by Lerchs-Grossmann graph closure (DP approach).

    Determines the ultimate pit limit for a 2-D cross-section of a block
    model using a dynamic-programming implementation of the
    Lerchs-Grossmann algorithm.  Blocks are arranged with rows
    representing levels (top to bottom) and columns representing
    horizontal position.

    Parameters
    ----------
    block_values : numpy.ndarray
        2-D array of shape ``(n_levels, n_columns)`` with the economic
        value of each block (positive = ore, negative = waste).  Row 0 is
        the topmost bench.
    slope_angles : tuple of float
        ``(left_angle_deg, right_angle_deg)`` overall slope angles in
        degrees measured from horizontal.  Must be in (0, 90].

    Returns
    -------
    dict
        ``"pit_mask"`` : numpy.ndarray of bool
            Same shape as *block_values*; ``True`` for blocks inside the
            optimal pit.
        ``"total_value"`` : float
            Sum of block values inside the optimal pit.

    Raises
    ------
    ValueError
        If *block_values* is not 2-D or slope angles are out of range.

    Examples
    --------
    >>> import numpy as np
    >>> bv = np.array([[-1, 5, -1],
    ...                [-2, 10, -2]])
    >>> result = lerchs_grossmann_2d(bv, (45.0, 45.0))
    >>> result["pit_mask"].any()
    True
    >>> result["total_value"] > 0
    True

    References
    ----------
    .. [1] Lerchs, H. & Grossmann, I. F. (1965). Optimum design of
           open-pit mines. *CIM Bulletin*, 58, 47-54.
    """
    block_values = np.asarray(block_values, dtype=float)
    if block_values.ndim != 2:
        raise ValueError(f"'block_values' must be 2-D, got {block_values.ndim}-D.")

    left_angle, right_angle = slope_angles
    validate_range(left_angle, 0.01, 90.0, "left_angle_deg")
    validate_range(right_angle, 0.01, 90.0, "right_angle_deg")

    n_levels, n_cols = block_values.shape

    # Compute the horizontal offset (in block-width units) per bench
    # going downward to honour each slope angle.
    left_offset = 1.0 / np.tan(np.radians(left_angle))
    right_offset = 1.0 / np.tan(np.radians(right_angle))

    # Dynamic programming: enumerate all possible top-bench column spans
    # [left_top, right_top] and expand downward according to slope.
    best_value = 0.0
    best_mask = np.zeros_like(block_values, dtype=bool)

    for left_top in range(n_cols):
        for right_top in range(left_top, n_cols):
            mask = np.zeros_like(block_values, dtype=bool)
            total = 0.0
            valid = True

            for level in range(n_levels):
                l_col = int(np.floor(left_top - level * left_offset))
                r_col = int(np.ceil(right_top + level * right_offset))
                l_col = max(l_col, 0)
                r_col = min(r_col, n_cols - 1)

                if l_col > r_col:
                    valid = False
                    break

                mask[level, l_col : r_col + 1] = True
                total += block_values[level, l_col : r_col + 1].sum()

            if valid and total > best_value:
                best_value = total
                best_mask = mask.copy()

    return {"pit_mask": best_mask, "total_value": float(best_value)}


def pseudoflow_3d(
    block_values: np.ndarray,
    slope_angles: tuple[float, float, float, float],
) -> dict:
    """Simplified 3D pit optimisation using iterative maximum-closure.

    Implements a greedy iterative approach inspired by the pseudoflow
    algorithm for small block models.  The four slope angles correspond
    to the north, south, east, and west wall orientations.

    Parameters
    ----------
    block_values : numpy.ndarray
        3-D array of shape ``(n_levels, n_rows, n_cols)`` with the
        economic value of each block.  Level 0 is the topmost bench.
    slope_angles : tuple of float
        ``(north_deg, south_deg, east_deg, west_deg)`` slope angles in
        degrees measured from horizontal, each in (0, 90].

    Returns
    -------
    dict
        ``"pit_mask"`` : numpy.ndarray of bool
            Same shape as *block_values*; ``True`` for blocks inside the
            optimal pit.
        ``"total_value"`` : float
            Sum of block values inside the optimal pit.

    Raises
    ------
    ValueError
        If *block_values* is not 3-D or slope angles are out of range.

    Examples
    --------
    >>> import numpy as np
    >>> bv = np.zeros((3, 3, 3))
    >>> bv[2, 1, 1] = 100  # deep ore block
    >>> bv[0, :, :] = -1   # surface waste
    >>> result = pseudoflow_3d(bv, (45.0, 45.0, 45.0, 45.0))
    >>> result["pit_mask"].any()
    True

    References
    ----------
    .. [1] Hochbaum, D. S. (2008). The pseudoflow algorithm: A new
           algorithm for the maximum-flow problem. *Operations Research*,
           56(4), 992-1009.
    """
    block_values = np.asarray(block_values, dtype=float)
    if block_values.ndim != 3:
        raise ValueError(f"'block_values' must be 3-D, got {block_values.ndim}-D.")

    north_deg, south_deg, east_deg, west_deg = slope_angles
    for angle, name in [
        (north_deg, "north_angle_deg"),
        (south_deg, "south_angle_deg"),
        (east_deg, "east_angle_deg"),
        (west_deg, "west_angle_deg"),
    ]:
        validate_range(angle, 0.01, 90.0, name)

    n_levels, n_rows, n_cols = block_values.shape

    # Offsets (block-width units per bench) for each direction.
    north_off = 1.0 / np.tan(np.radians(north_deg))
    south_off = 1.0 / np.tan(np.radians(south_deg))
    east_off = 1.0 / np.tan(np.radians(east_deg))
    west_off = 1.0 / np.tan(np.radians(west_deg))

    best_value = 0.0
    best_mask = np.zeros_like(block_values, dtype=bool)

    # Enumerate all possible top-bench rectangular footprints.
    for r1 in range(n_rows):
        for r2 in range(r1, n_rows):
            for c1 in range(n_cols):
                for c2 in range(c1, n_cols):
                    mask = np.zeros_like(block_values, dtype=bool)
                    total = 0.0
                    valid = True

                    for level in range(n_levels):
                        nr = int(np.floor(r1 - level * north_off))
                        sr = int(np.ceil(r2 + level * south_off))
                        wc = int(np.floor(c1 - level * west_off))
                        ec = int(np.ceil(c2 + level * east_off))

                        nr = max(nr, 0)
                        sr = min(sr, n_rows - 1)
                        wc = max(wc, 0)
                        ec = min(ec, n_cols - 1)

                        if nr > sr or wc > ec:
                            valid = False
                            break

                        mask[level, nr : sr + 1, wc : ec + 1] = True
                        total += block_values[level, nr : sr + 1, wc : ec + 1].sum()

                    if valid and total > best_value:
                        best_value = total
                        best_mask = mask.copy()

    return {"pit_mask": best_mask, "total_value": float(best_value)}


def block_economic_value(
    grade: float,
    tonnage: float,
    price: float,
    recovery: float,
    mining_cost: float,
    processing_cost: float,
) -> float:
    """Compute the economic value of a single mining block.

    .. math::

        V = T \\cdot (g \\cdot P \\cdot R - C_p) - C_m \\cdot T

    where *T* is tonnage, *g* is grade (fraction), *P* is commodity
    price, *R* is metallurgical recovery (fraction), *C_p* is processing
    cost per tonne, and *C_m* is mining cost per tonne.

    Parameters
    ----------
    grade : float
        Grade as a fraction (e.g. 0.005 for 0.5 %).
    tonnage : float
        Block tonnage in tonnes.  Must be positive.
    price : float
        Commodity price per unit of metal (e.g. $/t of metal).
    recovery : float
        Metallurgical recovery as a fraction in [0, 1].
    mining_cost : float
        Mining cost per tonne of material ($/t).  Non-negative.
    processing_cost : float
        Processing cost per tonne of ore ($/t).  Non-negative.

    Returns
    -------
    float
        Net economic value of the block in the same currency unit as
        *price*.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.

    Examples
    --------
    >>> block_economic_value(0.005, 10000, 8000, 0.9, 2.0, 15.0)
    190000.0

    References
    ----------
    .. [1] Whittle, J. (1999). A decade of open pit mine planning and
           optimisation. *Proceedings of APCOM*, 515-522.
    """
    validate_non_negative(grade, "grade")
    validate_positive(tonnage, "tonnage")
    validate_positive(price, "price")
    validate_range(recovery, 0.0, 1.0, "recovery")
    validate_non_negative(mining_cost, "mining_cost")
    validate_non_negative(processing_cost, "processing_cost")

    return tonnage * (grade * price * recovery - processing_cost) - mining_cost * tonnage
