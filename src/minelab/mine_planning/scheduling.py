"""Production scheduling for open-pit mines.

Implements greedy period assignment, NPV calculation for scheduled
cash flows, and precedence-constraint generation from slope geometry.

References
----------
.. [1] Newman, A. M., Rubio, E., Caro, R., Weintraub, A., & Eurek, K.
       (2010). A review of operations research in mine planning.
       *Interfaces*, 40(3), 222-245.
.. [2] Lerchs, H. & Grossmann, I. F. (1965). Optimum design of open-pit
       mines. *CIM Bulletin*, 58, 47-54.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import (
    validate_positive,
    validate_range,
)


def schedule_by_period(
    block_values: np.ndarray,
    pit_mask: np.ndarray,
    capacities: list[float],
    n_periods: int,
) -> dict:
    """Assign blocks to mining periods by greedy highest-value-first.

    Blocks within the pit mask are sorted by descending value and
    assigned to the earliest period that still has remaining capacity.
    This is a heuristic approach; optimal scheduling requires integer
    programming.

    Parameters
    ----------
    block_values : numpy.ndarray
        Array of economic values for every block (any shape).
    pit_mask : numpy.ndarray
        Boolean array of the same shape as *block_values*; ``True`` for
        blocks inside the pit.
    capacities : list of float
        Maximum total value (or tonnage proxy) that can be mined in each
        period.  Length must equal *n_periods*.
    n_periods : int
        Number of scheduling periods.  Must be positive.

    Returns
    -------
    dict
        ``"schedule"`` : numpy.ndarray
            Integer array of the same shape as *block_values*.  Each
            element is the assigned period (1-based) or 0 if the block
            is outside the pit or unscheduled.
        ``"period_values"`` : list of float
            Total economic value extracted in each period.

    Raises
    ------
    ValueError
        If shapes do not match, *n_periods* is non-positive, or
        ``len(capacities) != n_periods``.

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.array([10, 5, 8, 3, 1])
    >>> mask = np.array([True, True, True, True, False])
    >>> res = schedule_by_period(vals, mask, [15.0, 15.0], 2)
    >>> res["schedule"][4]
    0
    >>> len(res["period_values"])
    2

    References
    ----------
    .. [1] Newman, A. M. et al. (2010). A review of operations research
           in mine planning. *Interfaces*, 40(3), 222-245.
    """
    block_values = np.asarray(block_values, dtype=float)
    pit_mask = np.asarray(pit_mask, dtype=bool)

    if block_values.shape != pit_mask.shape:
        raise ValueError(
            "'block_values' and 'pit_mask' must have the same shape, "
            f"got {block_values.shape} vs {pit_mask.shape}."
        )
    if n_periods <= 0:
        raise ValueError(f"'n_periods' must be positive, got {n_periods}.")
    if len(capacities) != n_periods:
        raise ValueError(
            f"'capacities' length ({len(capacities)}) must equal 'n_periods' ({n_periods})."
        )

    schedule = np.zeros(block_values.shape, dtype=int)
    period_values = [0.0] * n_periods
    remaining = list(capacities)

    # Flatten and sort in-pit blocks by descending value
    flat_vals = block_values.ravel()
    flat_mask = pit_mask.ravel()
    in_pit_indices = np.where(flat_mask)[0]
    sorted_indices = in_pit_indices[np.argsort(-flat_vals[in_pit_indices])]

    flat_schedule = schedule.ravel()

    for idx in sorted_indices:
        val = flat_vals[idx]
        # Try to assign to the earliest period with remaining capacity
        for p in range(n_periods):
            if remaining[p] >= abs(val):
                flat_schedule[idx] = p + 1  # 1-based period
                remaining[p] -= abs(val)
                period_values[p] += val
                break
        # If no period has capacity, the block stays unscheduled (0)

    schedule = flat_schedule.reshape(block_values.shape)
    return {"schedule": schedule, "period_values": period_values}


def npv_schedule(
    period_values: list[float],
    discount_rate: float,
) -> float:
    """Net present value of a scheduled sequence of periodic cash flows.

    .. math::

        NPV = \\sum_{t=1}^{T} \\frac{V_t}{(1 + r)^t}

    Parameters
    ----------
    period_values : list of float
        Cash flow (value) for each period, starting at period 1.
    discount_rate : float
        Discount rate per period as a fraction (e.g. 0.10 for 10 %).
        Must be in [0, 1).

    Returns
    -------
    float
        Net present value.

    Raises
    ------
    ValueError
        If *discount_rate* is out of range or *period_values* is empty.

    Examples
    --------
    >>> npv_schedule([100, 100, 100], 0.10)
    248.68...

    References
    ----------
    .. [1] Newman, A. M. et al. (2010). A review of operations research
           in mine planning. *Interfaces*, 40(3), 222-245.
    """
    if len(period_values) == 0:
        raise ValueError("'period_values' must not be empty.")
    validate_range(discount_rate, 0.0, 0.99, "discount_rate")

    npv = 0.0
    for t, v in enumerate(period_values, start=1):
        npv += v / (1.0 + discount_rate) ** t
    return npv


def precedence_constraints(
    block_model_shape: tuple,
    slope_angle: float,
    bench_height: float,
    block_width: float,
) -> list[tuple]:
    """Build precedence pairs from slope-angle geometry.

    A block can only be mined if all blocks above it within the slope
    cone have already been mined.  This function returns the set of
    ``(parent, child)`` index pairs where *parent* must be mined before
    *child*.

    For a 2-D model (n_levels, n_cols), the cone is defined by the
    horizontal offset ``bench_height / tan(slope_angle)`` converted to
    column units via ``offset / block_width``.

    For a 3-D model (n_levels, n_rows, n_cols), the cone is applied
    symmetrically in both row and column directions.

    Parameters
    ----------
    block_model_shape : tuple
        Shape of the block model as ``(n_levels, n_cols)`` for 2-D or
        ``(n_levels, n_rows, n_cols)`` for 3-D.
    slope_angle : float
        Overall slope angle in degrees from horizontal, in (0, 90].
    bench_height : float
        Bench height in metres.  Must be positive.
    block_width : float
        Block width in metres.  Must be positive.

    Returns
    -------
    list of tuple
        Each tuple is ``(parent_index, child_index)`` where indices are
        flat (ravelled) positions.  *parent* must be mined before
        *child*.

    Raises
    ------
    ValueError
        If parameters are out of range or the shape is not 2-D or 3-D.

    Examples
    --------
    >>> pairs = precedence_constraints((3, 5), 45.0, 10.0, 10.0)
    >>> len(pairs) > 0
    True

    References
    ----------
    .. [1] Lerchs, H. & Grossmann, I. F. (1965). Optimum design of
           open-pit mines. *CIM Bulletin*, 58, 47-54.
    """
    validate_range(slope_angle, 0.01, 90.0, "slope_angle")
    validate_positive(bench_height, "bench_height")
    validate_positive(block_width, "block_width")

    ndim = len(block_model_shape)
    if ndim not in (2, 3):
        raise ValueError(f"'block_model_shape' must be 2-D or 3-D, got {ndim}-D.")

    # Horizontal offset in block units per single bench step
    h_offset = bench_height / np.tan(np.radians(slope_angle))
    col_offset = int(np.ceil(h_offset / block_width))

    pairs: list[tuple] = []

    if ndim == 2:
        n_levels, n_cols = block_model_shape

        for level in range(1, n_levels):
            for col in range(n_cols):
                child = np.ravel_multi_index((level, col), block_model_shape)
                # Parent: the block directly above
                parent_above = np.ravel_multi_index((level - 1, col), block_model_shape)
                pairs.append((int(parent_above), int(child)))

                # Parents within the slope cone on the level above
                for dc in range(1, col_offset + 1):
                    if col - dc >= 0:
                        parent = np.ravel_multi_index((level - 1, col - dc), block_model_shape)
                        pairs.append((int(parent), int(child)))
                    if col + dc < n_cols:
                        parent = np.ravel_multi_index((level - 1, col + dc), block_model_shape)
                        pairs.append((int(parent), int(child)))
    else:
        n_levels, n_rows, n_cols = block_model_shape
        row_offset = col_offset  # symmetric cone

        for level in range(1, n_levels):
            for row in range(n_rows):
                for col in range(n_cols):
                    child = np.ravel_multi_index((level, row, col), block_model_shape)
                    # All blocks in the cone on the level above
                    for dr in range(-row_offset, row_offset + 1):
                        for dc in range(-col_offset, col_offset + 1):
                            pr = row + dr
                            pc = col + dc
                            if 0 <= pr < n_rows and 0 <= pc < n_cols:
                                parent = np.ravel_multi_index(
                                    (level - 1, pr, pc),
                                    block_model_shape,
                                )
                                pairs.append((int(parent), int(child)))

    return pairs
