"""Stockpile management: FIFO and LIFO tracking.

This module provides functions for tracking material through stockpiles
using first-in-first-out (FIFO) and last-in-first-out (LIFO) strategies,
maintaining tonnage and grade information for each layer.

References
----------
.. [1] Standard stockpile management practice in mining operations.
"""

from __future__ import annotations

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# FIFO Stockpile
# ---------------------------------------------------------------------------


def stockpile_fifo(additions: list[dict], reclaims: list[float]) -> dict:
    """Track a stockpile using FIFO (first-in, first-out) strategy.

    Material is added in order and reclaimed starting from the oldest
    addition. Each reclaim operation removes material from the front of
    the queue, preserving grade tracking per layer.

    Parameters
    ----------
    additions : list[dict]
        Ordered list of material additions. Each dict must have keys:

        - ``"tonnes"`` : float -- tonnage added (must be positive).
        - ``"grade"`` : float -- grade of the added material.

    reclaims : list[float]
        Ordered list of reclaim tonnages. Each value must be positive.
        Reclaims are processed sequentially after all additions.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"reclaimed"`` : list[dict] -- list of reclaimed parcels, each
          with ``"tonnes"`` and ``"grade"``.
        - ``"remaining"`` : list[dict] -- list of remaining parcels in the
          stockpile, each with ``"tonnes"`` and ``"grade"``.

    Raises
    ------
    ValueError
        If a reclaim exceeds available stockpile tonnage, or if any
        addition tonnage is not positive.

    Examples
    --------
    >>> additions = [
    ...     {"tonnes": 100, "grade": 1.0},
    ...     {"tonnes": 200, "grade": 0.5},
    ...     {"tonnes": 150, "grade": 0.8},
    ... ]
    >>> result = stockpile_fifo(additions, [120, 80])
    >>> len(result["reclaimed"])
    2
    >>> result["reclaimed"][0]["tonnes"]
    120.0

    References
    ----------
    .. [1] Standard stockpile management practice.
    """
    # Validate and build queue (deep copy to avoid mutation)
    queue: list[dict] = []
    for i, a in enumerate(additions):
        validate_positive(a["tonnes"], f"additions[{i}].tonnes")
        queue.append({"tonnes": float(a["tonnes"]), "grade": float(a["grade"])})

    reclaimed_list: list[dict] = []

    for r_idx, reclaim_t in enumerate(reclaims):
        validate_positive(reclaim_t, f"reclaims[{r_idx}]")
        remaining_reclaim = float(reclaim_t)

        total_available = sum(layer["tonnes"] for layer in queue)
        if remaining_reclaim > total_available + 1e-9:
            raise ValueError(
                f"Reclaim {r_idx} requests {reclaim_t} t but only "
                f"{total_available:.2f} t available."
            )

        # Weighted reclaim from front of queue
        reclaim_tonnes = 0.0
        reclaim_metal = 0.0

        while remaining_reclaim > 1e-9 and queue:
            layer = queue[0]
            take = min(remaining_reclaim, layer["tonnes"])
            reclaim_tonnes += take
            reclaim_metal += take * layer["grade"]
            layer["tonnes"] -= take
            remaining_reclaim -= take

            if layer["tonnes"] < 1e-9:
                queue.pop(0)

        reclaim_grade = reclaim_metal / reclaim_tonnes if reclaim_tonnes > 0 else 0.0
        reclaimed_list.append(
            {
                "tonnes": round(reclaim_tonnes, 10),
                "grade": reclaim_grade,
            }
        )

    # Build remaining list, filtering near-zero layers
    remaining = [
        {"tonnes": layer["tonnes"], "grade": layer["grade"]}
        for layer in queue
        if layer["tonnes"] > 1e-9
    ]

    return {
        "reclaimed": reclaimed_list,
        "remaining": remaining,
    }


# ---------------------------------------------------------------------------
# LIFO Stockpile
# ---------------------------------------------------------------------------


def stockpile_lifo(additions: list[dict], reclaims: list[float]) -> dict:
    """Track a stockpile using LIFO (last-in, first-out) strategy.

    Material is added in order and reclaimed starting from the most
    recent addition. Each reclaim operation removes material from the top
    of the stack, preserving grade tracking per layer.

    Parameters
    ----------
    additions : list[dict]
        Ordered list of material additions. Each dict must have keys:

        - ``"tonnes"`` : float -- tonnage added (must be positive).
        - ``"grade"`` : float -- grade of the added material.

    reclaims : list[float]
        Ordered list of reclaim tonnages. Each value must be positive.
        Reclaims are processed sequentially after all additions.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"reclaimed"`` : list[dict] -- list of reclaimed parcels, each
          with ``"tonnes"`` and ``"grade"``.
        - ``"remaining"`` : list[dict] -- list of remaining parcels in the
          stockpile, each with ``"tonnes"`` and ``"grade"``.

    Raises
    ------
    ValueError
        If a reclaim exceeds available stockpile tonnage, or if any
        addition tonnage is not positive.

    Examples
    --------
    >>> additions = [
    ...     {"tonnes": 100, "grade": 1.0},
    ...     {"tonnes": 200, "grade": 0.5},
    ...     {"tonnes": 150, "grade": 0.8},
    ... ]
    >>> result = stockpile_lifo(additions, [120, 80])
    >>> result["reclaimed"][0]["grade"]  # First reclaim from top (grade=0.8)
    0.8

    References
    ----------
    .. [1] Standard stockpile management practice.
    """
    # Validate and build stack (deep copy to avoid mutation)
    stack: list[dict] = []
    for i, a in enumerate(additions):
        validate_positive(a["tonnes"], f"additions[{i}].tonnes")
        stack.append({"tonnes": float(a["tonnes"]), "grade": float(a["grade"])})

    reclaimed_list: list[dict] = []

    for r_idx, reclaim_t in enumerate(reclaims):
        validate_positive(reclaim_t, f"reclaims[{r_idx}]")
        remaining_reclaim = float(reclaim_t)

        total_available = sum(layer["tonnes"] for layer in stack)
        if remaining_reclaim > total_available + 1e-9:
            raise ValueError(
                f"Reclaim {r_idx} requests {reclaim_t} t but only "
                f"{total_available:.2f} t available."
            )

        # Weighted reclaim from top of stack
        reclaim_tonnes = 0.0
        reclaim_metal = 0.0

        while remaining_reclaim > 1e-9 and stack:
            layer = stack[-1]
            take = min(remaining_reclaim, layer["tonnes"])
            reclaim_tonnes += take
            reclaim_metal += take * layer["grade"]
            layer["tonnes"] -= take
            remaining_reclaim -= take

            if layer["tonnes"] < 1e-9:
                stack.pop()

        reclaim_grade = reclaim_metal / reclaim_tonnes if reclaim_tonnes > 0 else 0.0
        reclaimed_list.append(
            {
                "tonnes": round(reclaim_tonnes, 10),
                "grade": reclaim_grade,
            }
        )

    # Build remaining list, filtering near-zero layers
    remaining = [
        {"tonnes": layer["tonnes"], "grade": layer["grade"]}
        for layer in stack
        if layer["tonnes"] > 1e-9
    ]

    return {
        "reclaimed": reclaimed_list,
        "remaining": remaining,
    }
