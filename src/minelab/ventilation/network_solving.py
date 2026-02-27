"""Network solving methods for mine ventilation circuits.

This module implements the Hardy Cross iterative method for solving airflow
distribution in ventilation networks, as well as a direct solver for simple
series/parallel configurations.

References
----------
.. [1] McPherson, M.J. (1993). *Subsurface Ventilation and Environmental
       Engineering*, 1st ed. Chapman & Hall, Chapter 7.
"""

from __future__ import annotations

import math  # noqa: I001

import numpy as np

from minelab.utilities.validators import validate_non_negative, validate_positive

# ---------------------------------------------------------------------------
# Hardy Cross iterative solver
# ---------------------------------------------------------------------------


def hardy_cross(
    branches: list[dict],
    junctions: int,
    tol: float = 0.01,
    max_iter: int = 100,
) -> dict:
    """Solve mine ventilation network airflow using the Hardy Cross method.

    The Hardy Cross method is an iterative technique that corrects the
    assumed airflow in each mesh until Kirchhoff's pressure law is
    satisfied.  The mesh correction for each iteration is:

    .. math::

        \\Delta Q = \\frac{-\\sum_i (R_i Q_i |Q_i| \\pm P_{\\text{fan},i})}
                         {\\sum_i 2 R_i |Q_i|}

    Parameters
    ----------
    branches : list of dict
        Each branch dictionary must contain:

        - ``"from"`` : int -- Starting junction index (0-based).
        - ``"to"`` : int -- Ending junction index (0-based).
        - ``"resistance"`` : float -- Airway resistance (Ns^2/m^8).
        - ``"Q_init"`` or ``"initial_Q"`` : float -- Initial airflow
          estimate (m^3/s).  Positive flow goes from ``"from"`` to
          ``"to"``.
        - ``"fan_pressure"`` : float, optional -- Fan pressure in Pa
          (default 0.0).  Positive if assisting flow direction.
        - ``"mesh"`` : int, optional -- Mesh index this branch belongs
          to (0-based, default 0).  A branch may appear in multiple
          meshes by duplicating entries with different mesh indices.

    junctions : int
        Number of junctions in the network.
    tol : float, optional
        Convergence tolerance for the maximum mesh correction |DeltaQ|
        (default 0.01 m^3/s).
    max_iter : int, optional
        Maximum number of iterations (default 100).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"flows"`` : list of float -- Final airflow for each branch
          (m^3/s).
        - ``"pressure_drops"`` : list of float -- Pressure drop across
          each branch (Pa).
        - ``"iterations"`` : int -- Number of iterations performed.
        - ``"converged"`` : bool -- Whether the solver converged within
          *max_iter* iterations.
        - ``"max_correction"`` : float -- Final maximum mesh correction.

    Raises
    ------
    ValueError
        If *branches* is empty, *junctions* < 2, or branch data is invalid.

    Notes
    -----
    The implementation supports networks with multiple meshes.  Each branch
    must specify which mesh it belongs to via the ``"mesh"`` key.  Branches
    shared between meshes should be listed once per mesh.

    Examples
    --------
    Simple two-branch parallel network (one mesh):

    >>> branches = [
    ...     {"from": 0, "to": 1, "resistance": 4.0, "Q_init": 60.0,
    ...      "fan_pressure": 0.0, "mesh": 0},
    ...     {"from": 0, "to": 1, "resistance": 16.0, "Q_init": 40.0,
    ...      "fan_pressure": 0.0, "mesh": 0},
    ... ]
    >>> result = hardy_cross(branches, junctions=2, tol=0.001)
    >>> result["converged"]
    True

    References
    ----------
    .. [1] McPherson (1993), Ch. 7, Sec. 7.4.
    """
    if not branches:
        raise ValueError("'branches' must contain at least one element.")
    if junctions < 2:
        raise ValueError("'junctions' must be at least 2.")
    validate_positive(tol, "tol")
    validate_positive(max_iter, "max_iter")

    # Build branch list with working copies of flow
    n_branches = len(branches)
    flows = np.zeros(n_branches, dtype=float)
    res = np.zeros(n_branches, dtype=float)
    fan_pressures = np.zeros(n_branches, dtype=float)
    mesh_ids = np.zeros(n_branches, dtype=int)

    for i, br in enumerate(branches):
        if "resistance" not in br:
            raise ValueError(f"Branch {i} missing 'resistance' key.")
        # Accept either "Q_init" or "initial_Q" for initial airflow
        if "Q_init" in br:
            q_init = float(br["Q_init"])
        elif "initial_Q" in br:
            q_init = float(br["initial_Q"])
        else:
            raise ValueError(f"Branch {i} missing 'Q_init' (or 'initial_Q') key.")
        validate_non_negative(br["resistance"], f"branches[{i}].resistance")

        flows[i] = q_init
        res[i] = float(br["resistance"])
        fan_pressures[i] = float(br.get("fan_pressure", 0.0))
        mesh_ids[i] = int(br.get("mesh", 0))

    # Identify unique meshes
    unique_meshes = np.unique(mesh_ids)

    # Sign convention for each branch in its mesh:
    # +1 if branch direction aligns with mesh traversal direction
    # For simplicity, first branch in mesh is +1, second is -1
    # (suitable for simple parallel networks / small meshes)
    mesh_branch_indices: dict[int, list[int]] = {}
    mesh_branch_signs: dict[int, list[float]] = {}
    for m in unique_meshes:
        indices = [i for i in range(n_branches) if mesh_ids[i] == m]
        mesh_branch_indices[m] = indices
        # First branch: positive direction, subsequent: negative
        signs = [1.0] + [-1.0] * (len(indices) - 1)
        mesh_branch_signs[m] = signs

    converged = False
    max_correction = float("inf")
    iterations = 0

    for _iteration in range(max_iter):
        iterations += 1
        max_correction = 0.0

        for m in unique_meshes:
            indices = mesh_branch_indices[m]
            signs = mesh_branch_signs[m]

            # Numerator: sum of R_i * Q_i * |Q_i| * sign_i - fan_P * sign_i
            numerator = 0.0
            denominator = 0.0
            for idx, sign in zip(indices, signs, strict=True):
                q = flows[idx]
                r = res[idx]
                fp = fan_pressures[idx]
                numerator += sign * (r * q * abs(q) - fp)
                denominator += 2.0 * r * abs(q)

            if denominator == 0.0:
                continue

            delta_q = -numerator / denominator

            # Apply correction to all branches in this mesh
            for idx, sign in zip(indices, signs, strict=True):
                flows[idx] += sign * delta_q

            max_correction = max(max_correction, abs(delta_q))

        if max_correction <= tol:
            converged = True
            break

    # Compute final pressure drops
    p_drops = [float(res[i] * flows[i] * abs(flows[i])) for i in range(n_branches)]

    return {
        "flows": [float(q) for q in flows],
        "pressure_drops": p_drops,
        "iterations": iterations,
        "converged": converged,
        "max_correction": float(max_correction),
    }


# ---------------------------------------------------------------------------
# Simple network solver
# ---------------------------------------------------------------------------


def simple_network(resistances: list[float], topology: str = "series") -> float:
    """Compute equivalent resistance for a simple series or parallel network.

    Parameters
    ----------
    resistances : list of float
        Individual airway resistances (Ns^2/m^8).  All must be non-negative
        for series; all must be positive for parallel.
    topology : str, optional
        Network topology: ``"series"`` or ``"parallel"`` (default
        ``"series"``).

    Returns
    -------
    float
        Equivalent resistance in Ns^2/m^8.

    Raises
    ------
    ValueError
        If *resistances* is empty, values are invalid, or *topology* is
        unrecognised.

    Examples
    --------
    >>> simple_network([1.0, 2.0, 3.0], "series")
    6.0
    >>> simple_network([4.0, 4.0], "parallel")
    1.0

    References
    ----------
    .. [1] McPherson (1993), Ch. 7, Sec. 7.3.
    """
    if not resistances:
        raise ValueError("'resistances' must contain at least one element.")

    topology_lower = topology.lower()

    if topology_lower == "series":
        for i, r in enumerate(resistances):
            validate_non_negative(r, f"resistances[{i}]")
        return float(sum(resistances))

    if topology_lower == "parallel":
        for i, r in enumerate(resistances):
            validate_positive(r, f"resistances[{i}]")
        inv_sqrt_sum = sum(1.0 / math.sqrt(r) for r in resistances)
        return 1.0 / (inv_sqrt_sum**2)

    raise ValueError(f"'topology' must be 'series' or 'parallel', got '{topology}'.")
