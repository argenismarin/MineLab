"""Geostatistical simulation methods.

Sequential Gaussian simulation (SGS), sequential indicator simulation (SIS),
back-transform for simulations, and simulation summary statistics.
"""

from __future__ import annotations

import numpy as np

from minelab.geostatistics.kriging import simple_kriging
from minelab.geostatistics.transformations import (
    back_transform,
    normal_score_transform,
)
from minelab.geostatistics.variogram_fitting import VariogramModel
from minelab.utilities.validators import validate_array, validate_positive


def sequential_gaussian_simulation(
    coords: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    variogram_model: VariogramModel,
    n_realizations: int = 1,
    seed: int | None = None,
    search_radius: float | None = None,
    max_points: int = 16,
) -> np.ndarray:
    """Sequential Gaussian simulation with normal score transform.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, d).
    values : np.ndarray
        Sample values, shape (n,).
    grid : np.ndarray
        Grid node coordinates, shape (m, d).
    variogram_model : VariogramModel
        Variogram model for the normal-score data.
    n_realizations : int
        Number of realizations to generate.
    seed : int or None
        Random seed for reproducibility.
    search_radius : float or None
        Maximum search distance.
    max_points : int
        Maximum conditioning data per node.

    Returns
    -------
    np.ndarray
        Simulation results in original space, shape (n_realizations, m).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 1, 50)
    >>> coords = np.array([[10, 10], [40, 10], [10, 40], [40, 40]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> grid = np.array([[20, 20], [30, 30]], dtype=float)
    >>> sims = sequential_gaussian_simulation(coords, values, grid, model, 2, seed=42)
    >>> sims.shape
    (2, 2)

    References
    ----------
    .. [1] Deutsch, C.V. & Journel, A.G. (1998). "GSLIB." Oxford Univ. Press,
       Ch. IV.
    """
    validate_positive(n_realizations, "n_realizations")
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    grid = np.asarray(grid, dtype=float)

    if grid.ndim == 1:
        grid = grid.reshape(1, -1)

    # Normal score transform
    ns_result = normal_score_transform(values)
    ns_values = ns_result["transformed"]
    transform_table = ns_result["transform_table"]

    rng = np.random.default_rng(seed)
    m = grid.shape[0]
    realizations = np.zeros((n_realizations, m))

    for r in range(n_realizations):
        # Random path through grid nodes
        path = rng.permutation(m)

        # Start with conditioning data
        sim_coords = coords.copy()
        sim_values = ns_values.copy()

        sim_result = np.zeros(m)

        for node_idx in path:
            target = grid[node_idx : node_idx + 1]

            # Simple kriging with known mean=0 (standard normal)
            est, var = simple_kriging(
                sim_coords,
                sim_values,
                target,
                variogram_model,
                global_mean=0.0,
                search_radius=search_radius,
                max_points=max_points,
            )

            sk_est = est[0]
            sk_var = max(var[0], 1e-10)

            # Draw from conditional distribution N(sk_est, sk_var)
            sim_val = rng.normal(sk_est, np.sqrt(sk_var))
            sim_result[node_idx] = sim_val

            # Add to conditioning data
            sim_coords = np.vstack([sim_coords, target])
            sim_values = np.append(sim_values, sim_val)

        # Back-transform to original space
        realizations[r, :] = back_transform(sim_result, transform_table)

    return realizations


def sequential_indicator_simulation(
    coords: np.ndarray,
    indicators: np.ndarray,
    grid: np.ndarray,
    variogram_models: list[VariogramModel],
    cutoffs: np.ndarray | list[float],
    n_realizations: int = 1,
    seed: int | None = None,
    search_radius: float | None = None,
    max_points: int = 16,
) -> np.ndarray:
    """Sequential indicator simulation for categorical/threshold variables.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, d).
    indicators : np.ndarray
        Indicator matrix, shape (n, n_cutoffs).
    grid : np.ndarray
        Grid node coordinates, shape (m, d).
    variogram_models : list of VariogramModel
        One variogram per cutoff.
    cutoffs : array-like
        Cutoff thresholds (must be sorted ascending).
    n_realizations : int
        Number of realizations.
    seed : int or None
        Random seed.
    search_radius : float or None
        Maximum search distance.
    max_points : int
        Maximum conditioning data.

    Returns
    -------
    np.ndarray
        Simulated category indices, shape (n_realizations, m).
        Values are bin indices (0, 1, ..., n_cutoffs).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 0.25, 50)
    >>> coords = np.array([[10, 10], [40, 40]], dtype=float)
    >>> indicators = np.array([[1, 1], [0, 1]])
    >>> grid = np.array([[25, 25]], dtype=float)
    >>> sims = sequential_indicator_simulation(
    ...     coords, indicators, grid, [model, model], [1.0, 2.0], 1, seed=42
    ... )
    >>> sims.shape
    (1, 1)

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 7.
    """
    validate_positive(n_realizations, "n_realizations")
    coords = np.asarray(coords, dtype=float)
    indicators = np.asarray(indicators, dtype=float)
    grid = np.asarray(grid, dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=float)

    if grid.ndim == 1:
        grid = grid.reshape(1, -1)

    n_cutoffs = len(cutoffs)
    m = grid.shape[0]
    rng = np.random.default_rng(seed)

    realizations = np.zeros((n_realizations, m), dtype=int)

    for r in range(n_realizations):
        path = rng.permutation(m)

        sim_coords = coords.copy()
        sim_indicators = indicators.copy()

        sim_result = np.zeros(m, dtype=int)

        for node_idx in path:
            target = grid[node_idx : node_idx + 1]

            # Kriging each indicator
            probs = np.zeros(n_cutoffs)
            for k in range(n_cutoffs):
                from minelab.geostatistics.kriging import ordinary_kriging

                est, _ = ordinary_kriging(
                    sim_coords,
                    sim_indicators[:, k],
                    target,
                    variogram_models[k],
                    search_radius,
                    max_points,
                )
                probs[k] = np.clip(est[0], 0.0, 1.0)

            # Ensure monotonicity and build CDF
            for k in range(1, n_cutoffs):
                probs[k] = max(probs[k], probs[k - 1])

            # Convert CDF to PMF
            pmf = np.zeros(n_cutoffs + 1)
            pmf[0] = probs[0]
            for k in range(1, n_cutoffs):
                pmf[k] = probs[k] - probs[k - 1]
            pmf[n_cutoffs] = 1.0 - probs[-1]
            pmf = np.clip(pmf, 0, None)
            total = np.sum(pmf)
            if total > 0:
                pmf /= total
            else:
                pmf = np.ones(n_cutoffs + 1) / (n_cutoffs + 1)

            # Draw category
            category = rng.choice(n_cutoffs + 1, p=pmf)
            sim_result[node_idx] = category

            # Update indicators for conditioning
            new_ind = np.zeros(n_cutoffs)
            for k in range(n_cutoffs):
                new_ind[k] = 1.0 if category <= k else 0.0

            sim_coords = np.vstack([sim_coords, target])
            sim_indicators = np.vstack([sim_indicators, new_ind])

        realizations[r, :] = sim_result

    return realizations


def back_transform_simulation(
    sim_values: np.ndarray,
    original_data: np.ndarray,
    transform_table: np.ndarray,
) -> np.ndarray:
    """Back-transform simulation values from Gaussian to original space.

    Parameters
    ----------
    sim_values : np.ndarray
        Simulated values in normal score space, shape (n_realizations, m)
        or (m,).
    original_data : np.ndarray
        Original data used to build the transform table.
    transform_table : np.ndarray
        Two-column array (original, normal_score) from normal_score_transform.

    Returns
    -------
    np.ndarray
        Back-transformed values in original space.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.lognormal(0, 1, 100)
    >>> ns = normal_score_transform(data)
    >>> bt = back_transform_simulation(ns["transformed"], data, ns["transform_table"])
    >>> np.allclose(np.sort(bt), np.sort(data), rtol=1e-4)
    True

    References
    ----------
    .. [1] Deutsch, C.V. & Journel, A.G. (1998). "GSLIB." Oxford Univ. Press.
    """
    sim_values = np.asarray(sim_values, dtype=float)
    transform_table = np.asarray(transform_table, dtype=float)

    original_shape = sim_values.shape
    flat = sim_values.ravel()
    result = back_transform(flat, transform_table)

    return result.reshape(original_shape)


def simulation_statistics(
    realizations: np.ndarray,
) -> dict:
    """Compute summary statistics from multiple realizations.

    Parameters
    ----------
    realizations : np.ndarray
        Shape (n_realizations, m) — one row per realization.

    Returns
    -------
    dict
        Keys: ``"e_type"`` (mean across realizations), ``"variance"``
        (conditional variance), ``"p10"``, ``"p50"``, ``"p90"``
        (percentiles per cell).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> reals = rng.normal(5, 2, size=(100, 10))
    >>> stats = simulation_statistics(reals)
    >>> stats["e_type"].shape
    (10,)
    >>> np.allclose(stats["e_type"], 5.0, atol=1.0)
    True

    References
    ----------
    .. [1] Standard geostatistical practice.
    """
    realizations = np.asarray(realizations, dtype=float)
    if realizations.ndim == 1:
        realizations = realizations.reshape(1, -1)

    return {
        "e_type": np.mean(realizations, axis=0),
        "variance": np.var(realizations, axis=0, ddof=1),
        "p10": np.percentile(realizations, 10, axis=0),
        "p50": np.percentile(realizations, 50, axis=0),
        "p90": np.percentile(realizations, 90, axis=0),
    }
