"""Experimental variogram computation from spatial data.

Provides functions for omnidirectional, directional, cloud, and cross
variogram calculations.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_array, validate_positive


def experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 10,
    lag_dist: float | None = None,
    tol: float = 0.5,
) -> dict:
    """Compute omnidirectional experimental semivariogram.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n, 2) or (n, 3).
    values : np.ndarray
        Data values at each coordinate, shape (n,).
    n_lags : int
        Number of lag bins.
    lag_dist : float or None
        Lag spacing. If None, computed as max_dist / (n_lags + 1).
    tol : float
        Lag tolerance as fraction of lag_dist (default 0.5).

    Returns
    -------
    dict
        Keys: ``"lags"`` (bin centers), ``"semivariance"`` (γ values),
        ``"n_pairs"`` (pair counts per bin).

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    >>> values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    >>> result = experimental_variogram(coords, values, n_lags=4)
    >>> len(result["lags"])
    4

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 4.
    .. [2] Isaaks, E.H. & Srivastava, R.M. (1989). "An Introduction to
       Applied Geostatistics." Oxford University Press, Ch. 7.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = len(values)
    if coords.shape[0] != n:
        raise ValueError("coords and values must have the same number of points.")

    # Compute all pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    if lag_dist is None:
        max_dist = np.max(dists[np.triu_indices(n, k=1)])
        lag_dist = max_dist / (n_lags + 1)

    validate_positive(lag_dist, "lag_dist")

    # Squared differences for all pairs
    val_diff_sq = 0.5 * (values[:, np.newaxis] - values[np.newaxis, :]) ** 2

    lags = np.zeros(n_lags)
    semivariance = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        center = (k + 1) * lag_dist
        low = center - tol * lag_dist
        high = center + tol * lag_dist

        # Upper triangle only to avoid double-counting
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        in_bin = mask & (dists >= low) & (dists < high)

        count = np.sum(in_bin)
        if count > 0:
            semivariance[k] = np.sum(val_diff_sq[in_bin]) / count
            lags[k] = np.mean(dists[in_bin])
        else:
            semivariance[k] = np.nan
            lags[k] = center
        n_pairs[k] = count

    return {"lags": lags, "semivariance": semivariance, "n_pairs": n_pairs}


def directional_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    azimuth: float,
    tol_angle: float = 22.5,
    bandwidth: float | None = None,
    n_lags: int = 10,
    lag_dist: float | None = None,
) -> dict:
    """Compute directional experimental semivariogram.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n, 2) or (n, 3). Only first 2 columns
        are used for direction filtering.
    values : np.ndarray
        Data values at each coordinate, shape (n,).
    azimuth : float
        Direction in degrees clockwise from north (0=N, 90=E).
    tol_angle : float
        Angular tolerance in degrees (half-window).
    bandwidth : float or None
        Maximum perpendicular distance. If None, no bandwidth filter.
    n_lags : int
        Number of lag bins.
    lag_dist : float or None
        Lag spacing. If None, auto-computed.

    Returns
    -------
    dict
        Keys: ``"lags"``, ``"semivariance"``, ``"n_pairs"``.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]])
    >>> values = np.array([1.0, 2.0, 3.0, 1.5, 2.5])
    >>> result = directional_variogram(coords, values, azimuth=90)
    >>> len(result["lags"])
    10

    References
    ----------
    .. [1] Isaaks, E.H. & Srivastava, R.M. (1989). "An Introduction to
       Applied Geostatistics." Oxford University Press, Ch. 7.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = len(values)
    if coords.shape[0] != n:
        raise ValueError("coords and values must have the same number of points.")

    # Direction vector from azimuth (clockwise from north)
    az_rad = np.radians(azimuth)
    dir_vec = np.array([np.sin(az_rad), np.cos(az_rad)])

    # Pairwise vectors (2D only)
    dx = coords[:, np.newaxis, 0] - coords[np.newaxis, :, 0]
    dy = coords[:, np.newaxis, 1] - coords[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)

    # Angle of each pair vector
    # Compute dot product with direction to get angle
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angle = (dx * dir_vec[0] + dy * dir_vec[1]) / np.where(dists == 0, 1.0, dists)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(cos_angle)))

    # Direction filter: within angular tolerance
    dir_mask = angles <= tol_angle

    # Bandwidth filter
    if bandwidth is not None:
        perp_dist = np.abs(-dx * dir_vec[1] + dy * dir_vec[0])
        dir_mask = dir_mask & (perp_dist <= bandwidth)

    if lag_dist is None:
        upper = np.triu_indices(n, k=1)
        valid_dists = dists[upper][dir_mask[upper]]
        if len(valid_dists) == 0:
            lag_dist = np.max(dists[upper]) / (n_lags + 1)
        else:
            lag_dist = np.max(valid_dists) / (n_lags + 1)

    validate_positive(lag_dist, "lag_dist")

    val_diff_sq = 0.5 * (values[:, np.newaxis] - values[np.newaxis, :]) ** 2
    lag_tol = 0.5 * lag_dist

    lags = np.zeros(n_lags)
    semivariance = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        center = (k + 1) * lag_dist
        low = center - lag_tol
        high = center + lag_tol

        mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        in_bin = mask_upper & dir_mask & (dists >= low) & (dists < high)

        count = np.sum(in_bin)
        if count > 0:
            semivariance[k] = np.sum(val_diff_sq[in_bin]) / count
            lags[k] = np.mean(dists[in_bin])
        else:
            semivariance[k] = np.nan
            lags[k] = center
        n_pairs[k] = count

    return {"lags": lags, "semivariance": semivariance, "n_pairs": n_pairs}


def variogram_cloud(
    coords: np.ndarray,
    values: np.ndarray,
    max_dist: float | None = None,
) -> dict:
    """Compute variogram cloud (all-pairs).

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n, 2) or (n, 3).
    values : np.ndarray
        Data values, shape (n,).
    max_dist : float or None
        Maximum lag distance to include. If None, include all pairs.

    Returns
    -------
    dict
        Keys: ``"distances"`` (h values), ``"semivariance"`` (γ values),
        ``"n_pairs"`` (total number of pairs).

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0]])
    >>> values = np.array([1.0, 2.0, 4.0])
    >>> cloud = variogram_cloud(coords, values)
    >>> cloud["n_pairs"]
    3

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 4.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = len(values)
    if coords.shape[0] != n:
        raise ValueError("coords and values must have the same number of points.")

    # Upper triangle indices
    idx_i, idx_j = np.triu_indices(n, k=1)

    diffs = coords[idx_i] - coords[idx_j]
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    semivariances = 0.5 * (values[idx_i] - values[idx_j]) ** 2

    if max_dist is not None:
        validate_positive(max_dist, "max_dist")
        mask = distances <= max_dist
        distances = distances[mask]
        semivariances = semivariances[mask]

    return {
        "distances": distances,
        "semivariance": semivariances,
        "n_pairs": len(distances),
    }


def cross_variogram(
    coords: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    n_lags: int = 10,
    lag_dist: float | None = None,
) -> dict:
    """Compute cross-variogram for two co-located variables.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape (n, 2) or (n, 3).
    values1 : np.ndarray
        First variable values, shape (n,).
    values2 : np.ndarray
        Second variable values, shape (n,).
    n_lags : int
        Number of lag bins.
    lag_dist : float or None
        Lag spacing. If None, auto-computed.

    Returns
    -------
    dict
        Keys: ``"lags"``, ``"cross_semivariance"``, ``"n_pairs"``.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    >>> v1 = np.array([1.0, 2.0, 3.0, 4.0])
    >>> result = cross_variogram(coords, v1, v1, n_lags=3)
    >>> np.allclose(result["cross_semivariance"][:2],
    ...     experimental_variogram(coords, v1, n_lags=3)["semivariance"][:2],
    ...     rtol=0.1)
    True

    References
    ----------
    .. [1] Wackernagel, H. (2003). "Multivariate Geostatistics." 3rd ed.,
       Springer, Ch. 11.
    """
    coords = np.asarray(coords, dtype=float)
    values1 = validate_array(values1, "values1")
    values2 = validate_array(values2, "values2")

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = len(values1)
    if coords.shape[0] != n or len(values2) != n:
        raise ValueError("coords, values1, values2 must have the same length.")

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    if lag_dist is None:
        max_d = np.max(dists[np.triu_indices(n, k=1)])
        lag_dist = max_d / (n_lags + 1)

    validate_positive(lag_dist, "lag_dist")

    # Cross-semivariance: γ12(h) = (1/2N) Σ [z1(xi)-z1(xi+h)]*[z2(xi)-z2(xi+h)]
    cross_diff = 0.5 * (
        (values1[:, np.newaxis] - values1[np.newaxis, :])
        * (values2[:, np.newaxis] - values2[np.newaxis, :])
    )

    lag_tol = 0.5 * lag_dist
    lags = np.zeros(n_lags)
    cross_sv = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        center = (k + 1) * lag_dist
        low = center - lag_tol
        high = center + lag_tol

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        in_bin = mask & (dists >= low) & (dists < high)

        count = np.sum(in_bin)
        if count > 0:
            cross_sv[k] = np.sum(cross_diff[in_bin]) / count
            lags[k] = np.mean(dists[in_bin])
        else:
            cross_sv[k] = np.nan
            lags[k] = center
        n_pairs[k] = count

    return {"lags": lags, "cross_semivariance": cross_sv, "n_pairs": n_pairs}
