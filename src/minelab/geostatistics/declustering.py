"""Spatial declustering methods.

Cell declustering, polygonal (Voronoi) declustering, and optimal cell size
selection for preferentially sampled data.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from minelab.utilities.validators import validate_array, validate_positive


def cell_declustering(
    coords: np.ndarray,
    values: np.ndarray,
    cell_sizes: np.ndarray | list[float],
) -> dict:
    """Cell-based declustering weights.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, 2) or (n, 3).
    values : np.ndarray
        Sample values, shape (n,).
    cell_sizes : array-like
        Cell dimensions, shape (d,). E.g., [50, 50] for 2D.

    Returns
    -------
    dict
        Keys: ``"weights"`` (declustering weights summing to n),
        ``"declustered_mean"`` (weighted mean of values).

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [10, 10]], dtype=float)
    >>> values = np.array([1, 2, 3, 10], dtype=float)
    >>> result = cell_declustering(coords, values, [5, 5])
    >>> abs(sum(result["weights"]) - 4) < 1e-10
    True

    References
    ----------
    .. [1] Deutsch, C.V. (1989). "DECLUS: A Fortran 77 program for determining
       optimum spatial declustering weights." Comput. & Geosci., 15(3), 325-332.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    cell_sizes = np.asarray(cell_sizes, dtype=float)

    n = len(values)
    ndim = coords.shape[1]

    if len(cell_sizes) != ndim:
        raise ValueError(f"cell_sizes must have {ndim} elements.")

    # Assign each point to a cell
    cell_indices = np.floor(coords / cell_sizes).astype(int)

    # Count points per cell
    unique_cells, inverse, counts = np.unique(
        cell_indices, axis=0, return_inverse=True, return_counts=True
    )
    n_occupied = len(unique_cells)

    # Weight = 1 / (n_in_cell * n_occupied_cells) * n_total
    # This ensures: sum(weights) = n, clustered points get lower weights
    weights = np.zeros(n)
    for i in range(n):
        cell_count = counts[inverse[i]]
        weights[i] = 1.0 / (cell_count * n_occupied) * n

    # Declustered mean
    decl_mean = float(np.sum(weights * values) / np.sum(weights))

    return {"weights": weights, "declustered_mean": decl_mean}


def polygonal_declustering(
    coords: np.ndarray,
    values: np.ndarray,
) -> dict:
    """Polygonal (Voronoi) declustering weights.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, 2). Only 2D is supported.
    values : np.ndarray
        Sample values, shape (n,).

    Returns
    -------
    dict
        Keys: ``"weights"`` (proportional to polygon area, sum = n),
        ``"declustered_mean"`` (weighted mean).

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [0.5, 1], [5, 5]], dtype=float)
    >>> values = np.array([1, 2, 3, 10], dtype=float)
    >>> result = polygonal_declustering(coords, values)
    >>> abs(sum(result["weights"]) - 4) < 0.01
    True

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")

    if coords.shape[1] != 2:
        raise ValueError("Polygonal declustering only supports 2D coordinates.")

    n = len(values)

    # Add bounding box points to handle unbounded Voronoi regions
    margin = 10 * np.max(np.ptp(coords, axis=0))
    cx, cy = np.mean(coords, axis=0)
    bounding = np.array(
        [
            [cx - margin, cy - margin],
            [cx + margin, cy - margin],
            [cx - margin, cy + margin],
            [cx + margin, cy + margin],
        ]
    )
    all_coords = np.vstack([coords, bounding])

    vor = Voronoi(all_coords)

    areas = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded region — use bounding box clip area
            areas[i] = margin**2 / n  # fallback area
        else:
            # Compute polygon area using shoelace formula
            vertices = vor.vertices[region]
            x = vertices[:, 0]
            y = vertices[:, 1]
            areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Normalize weights to sum to n
    total_area = np.sum(areas)
    weights = areas / total_area * n if total_area > 0 else np.ones(n)

    decl_mean = float(np.sum(weights * values) / np.sum(weights))

    return {"weights": weights, "declustered_mean": decl_mean}


def optimal_cell_size(
    coords: np.ndarray,
    values: np.ndarray,
    min_size: float,
    max_size: float,
    n_steps: int = 20,
) -> dict:
    """Search for optimal cell size that minimizes the declustered mean.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, 2) or (n, 3).
    values : np.ndarray
        Sample values, shape (n,).
    min_size : float
        Minimum cell size to test.
    max_size : float
        Maximum cell size to test.
    n_steps : int
        Number of sizes to evaluate.

    Returns
    -------
    dict
        Keys: ``"optimal_size"`` (best cell size), ``"cell_sizes"``
        (all tested sizes), ``"declustered_means"`` (mean at each size).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> coords = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(10, 1, (5, 2))])
    >>> values = np.concatenate([np.ones(50), np.full(5, 10.0)])
    >>> result = optimal_cell_size(coords, values, 0.5, 20.0, n_steps=10)
    >>> result["optimal_size"] > 0
    True

    References
    ----------
    .. [1] Deutsch, C.V. (1989). "DECLUS." Comput. & Geosci., 15(3), 325-332.
    """
    validate_positive(min_size, "min_size")
    validate_positive(max_size, "max_size")
    validate_positive(n_steps, "n_steps")

    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    ndim = coords.shape[1]

    sizes = np.linspace(min_size, max_size, n_steps)
    decl_means = np.zeros(n_steps)

    for j, sz in enumerate(sizes):
        cell_sz = np.full(ndim, sz)
        result = cell_declustering(coords, values, cell_sz)
        decl_means[j] = result["declustered_mean"]

    # Optimal size: the one that gives the minimum declustered mean
    best_idx = np.argmin(decl_means)

    return {
        "optimal_size": float(sizes[best_idx]),
        "cell_sizes": sizes,
        "declustered_means": decl_means,
    }
