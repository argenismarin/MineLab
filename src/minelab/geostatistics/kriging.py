"""Kriging estimation methods.

Ordinary, simple, universal, indicator, and block kriging, plus
leave-one-out cross-validation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from minelab.geostatistics.variogram_fitting import VariogramModel
from minelab.utilities.validators import validate_array


def _covariance_from_variogram(model: VariogramModel, h: np.ndarray) -> np.ndarray:
    """Convert variogram γ(h) to covariance C(h) = sill - γ(h)."""
    gamma = np.asarray(model.predict(h), dtype=float)
    return model.sill - gamma


def _select_neighbors(
    target: np.ndarray,
    coords: np.ndarray,
    search_radius: float | None,
    max_points: int | None,
) -> np.ndarray:
    """Return indices of neighbors within search radius / max count."""
    dists = np.sqrt(np.sum((coords - target) ** 2, axis=1))
    mask = np.ones(len(dists), dtype=bool)

    if search_radius is not None:
        mask &= dists <= search_radius

    indices = np.where(mask)[0]
    if max_points is not None and len(indices) > max_points:
        # Keep closest max_points
        sub_dists = dists[indices]
        keep = np.argsort(sub_dists)[:max_points]
        indices = indices[keep]

    return indices


def ordinary_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    target_coords: np.ndarray,
    variogram_model: VariogramModel,
    search_radius: float | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging estimation.

    Weights are constrained to sum to 1.0, so no global mean is required.

    Parameters
    ----------
    coords : np.ndarray
        Known sample coordinates, shape (n, d).
    values : np.ndarray
        Known sample values, shape (n,).
    target_coords : np.ndarray
        Estimation locations, shape (m, d).
    variogram_model : VariogramModel
        Fitted variogram model.
    search_radius : float or None
        Maximum search distance.
    max_points : int or None
        Maximum number of neighbors to use.

    Returns
    -------
    estimates : np.ndarray
        Kriging estimates, shape (m,).
    variances : np.ndarray
        Kriging variances, shape (m,).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 10, 100)
    >>> coords = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> est, var = ordinary_kriging(coords, values, np.array([[50, 50]]), model)
    >>> abs(sum([1.0, 2.0, 3.0, 4.0]) / 4 - est[0]) < 1.0
    True

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    target_coords = np.asarray(target_coords, dtype=float)

    if target_coords.ndim == 1:
        target_coords = target_coords.reshape(1, -1)

    m = target_coords.shape[0]
    estimates = np.zeros(m)
    variances = np.zeros(m)

    for i in range(m):
        idx = _select_neighbors(target_coords[i], coords, search_radius, max_points)

        if len(idx) == 0:
            estimates[i] = np.nan
            variances[i] = np.nan
            continue

        c_i = coords[idx]
        v_i = values[idx]
        n_i = len(idx)

        # Distance matrices
        dist_data = cdist(c_i, c_i)
        dist_target = cdist(c_i, target_coords[i : i + 1]).ravel()

        # Covariance matrices
        cov_data = _covariance_from_variogram(variogram_model, dist_data)
        cov_target = _covariance_from_variogram(variogram_model, dist_target)

        # OK system: [C  1] [w]   [c0]
        #            [1  0] [μ] = [1 ]
        k_mat = np.zeros((n_i + 1, n_i + 1))
        k_mat[:n_i, :n_i] = cov_data
        k_mat[:n_i, n_i] = 1.0
        k_mat[n_i, :n_i] = 1.0

        k_vec = np.zeros(n_i + 1)
        k_vec[:n_i] = cov_target
        k_vec[n_i] = 1.0

        try:
            solution = np.linalg.solve(k_mat, k_vec)
        except np.linalg.LinAlgError:
            estimates[i] = np.nan
            variances[i] = np.nan
            continue

        weights = solution[:n_i]
        mu = solution[n_i]

        estimates[i] = float(np.dot(weights, v_i))
        # σ²_OK = C(0) - Σ w_i * C(x_i, x0) - μ
        cov_0 = _covariance_from_variogram(variogram_model, np.array([0.0]))[0]
        variances[i] = float(cov_0 - np.dot(weights, cov_target) - mu)

    return estimates, variances


def simple_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    target_coords: np.ndarray,
    variogram_model: VariogramModel,
    global_mean: float = 0.0,
    search_radius: float | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple kriging estimation with known global mean.

    Parameters
    ----------
    coords : np.ndarray
        Known sample coordinates, shape (n, d).
    values : np.ndarray
        Known sample values, shape (n,).
    target_coords : np.ndarray
        Estimation locations, shape (m, d).
    variogram_model : VariogramModel
        Fitted variogram model.
    global_mean : float
        Known stationary mean of the variable.
    search_radius : float or None
        Maximum search distance.
    max_points : int or None
        Maximum number of neighbors.

    Returns
    -------
    estimates : np.ndarray
        Kriging estimates, shape (m,).
    variances : np.ndarray
        Kriging variances, shape (m,).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 10, 100)
    >>> coords = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> est, var = simple_kriging(coords, values, np.array([[50, 50]]), model, 2.5)
    >>> est.shape
    (1,)

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    target_coords = np.asarray(target_coords, dtype=float)

    if target_coords.ndim == 1:
        target_coords = target_coords.reshape(1, -1)

    m = target_coords.shape[0]
    estimates = np.zeros(m)
    variances = np.zeros(m)

    for i in range(m):
        idx = _select_neighbors(target_coords[i], coords, search_radius, max_points)

        if len(idx) == 0:
            estimates[i] = global_mean
            cov_0 = _covariance_from_variogram(variogram_model, np.array([0.0]))[0]
            variances[i] = float(cov_0)
            continue

        c_i = coords[idx]
        v_i = values[idx]

        dist_data = cdist(c_i, c_i)
        dist_target = cdist(c_i, target_coords[i : i + 1]).ravel()

        cov_data = _covariance_from_variogram(variogram_model, dist_data)
        cov_target = _covariance_from_variogram(variogram_model, dist_target)

        # SK system: C * w = c0 (no Lagrange multiplier)
        try:
            weights = np.linalg.solve(cov_data, cov_target)
        except np.linalg.LinAlgError:
            estimates[i] = global_mean
            cov_0 = _covariance_from_variogram(variogram_model, np.array([0.0]))[0]
            variances[i] = float(cov_0)
            continue

        # z*_SK = m + Σ w_i * (z_i - m)
        estimates[i] = float(global_mean + np.dot(weights, v_i - global_mean))
        cov_0 = _covariance_from_variogram(variogram_model, np.array([0.0]))[0]
        variances[i] = float(cov_0 - np.dot(weights, cov_target))

    return estimates, variances


def universal_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    target_coords: np.ndarray,
    variogram_model: VariogramModel,
    drift_terms: int = 0,
    search_radius: float | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal kriging with polynomial drift.

    Parameters
    ----------
    coords : np.ndarray
        Known sample coordinates, shape (n, 2) or (n, 3).
    values : np.ndarray
        Known sample values, shape (n,).
    target_coords : np.ndarray
        Estimation locations, shape (m, d).
    variogram_model : VariogramModel
        Fitted variogram model.
    drift_terms : int
        Polynomial order for drift: 0 (constant = OK), 1 (linear), 2 (quadratic).
    search_radius : float or None
        Maximum search distance.
    max_points : int or None
        Maximum number of neighbors.

    Returns
    -------
    estimates : np.ndarray
        Kriging estimates, shape (m,).
    variances : np.ndarray
        Kriging variances, shape (m,).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 10, 100)
    >>> coords = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> est_uk, _ = universal_kriging(coords, values, np.array([[50, 50]]), model, 0)
    >>> est_ok, _ = ordinary_kriging(coords, values, np.array([[50, 50]]), model)
    >>> abs(est_uk[0] - est_ok[0]) < 1e-6
    True

    References
    ----------
    .. [1] Chilès, J.-P. & Delfiner, P. (2012). "Geostatistics: Modeling
       Spatial Uncertainty." 2nd ed., Wiley.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    target_coords = np.asarray(target_coords, dtype=float)

    if target_coords.ndim == 1:
        target_coords = target_coords.reshape(1, -1)

    ndim = coords.shape[1]

    def _drift_matrix(pts: np.ndarray) -> np.ndarray:
        """Build drift function matrix for given points."""
        n = pts.shape[0]
        cols = [np.ones(n)]  # constant term
        if drift_terms >= 1:
            for d in range(min(ndim, 2)):
                cols.append(pts[:, d])
        if drift_terms >= 2:
            for d in range(min(ndim, 2)):
                cols.append(pts[:, d] ** 2)
            if ndim >= 2:
                cols.append(pts[:, 0] * pts[:, 1])
        return np.column_stack(cols)

    m = target_coords.shape[0]
    estimates = np.zeros(m)
    variances = np.zeros(m)

    for i in range(m):
        idx = _select_neighbors(target_coords[i], coords, search_radius, max_points)

        if len(idx) == 0:
            estimates[i] = np.nan
            variances[i] = np.nan
            continue

        c_i = coords[idx]
        v_i = values[idx]
        n_i = len(idx)

        dist_data = cdist(c_i, c_i)
        dist_target = cdist(c_i, target_coords[i : i + 1]).ravel()

        cov_data = _covariance_from_variogram(variogram_model, dist_data)
        cov_target = _covariance_from_variogram(variogram_model, dist_target)

        # Drift matrices
        f_data = _drift_matrix(c_i)
        f_target = _drift_matrix(target_coords[i : i + 1]).ravel()
        p = f_data.shape[1]

        # UK system:
        # [C    F] [w]   [c0]
        # [F^T  0] [μ] = [f0]
        k_size = n_i + p
        k_mat = np.zeros((k_size, k_size))
        k_mat[:n_i, :n_i] = cov_data
        k_mat[:n_i, n_i:] = f_data
        k_mat[n_i:, :n_i] = f_data.T

        k_vec = np.zeros(k_size)
        k_vec[:n_i] = cov_target
        k_vec[n_i:] = f_target

        try:
            solution = np.linalg.solve(k_mat, k_vec)
        except np.linalg.LinAlgError:
            estimates[i] = np.nan
            variances[i] = np.nan
            continue

        weights = solution[:n_i]
        lambdas = solution[n_i:]

        estimates[i] = float(np.dot(weights, v_i))
        cov_0 = _covariance_from_variogram(variogram_model, np.array([0.0]))[0]
        variances[i] = float(cov_0 - np.dot(weights, cov_target) - np.dot(lambdas, f_target))

    return estimates, variances


def indicator_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    target_coords: np.ndarray,
    cutoffs: list[float] | np.ndarray,
    variogram_models: list[VariogramModel],
    search_radius: float | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    """Indicator kriging at multiple cutoffs.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, d).
    values : np.ndarray
        Sample values, shape (n,).
    target_coords : np.ndarray
        Estimation locations, shape (m, d).
    cutoffs : array-like
        Cutoff thresholds for indicator transform.
    variogram_models : list of VariogramModel
        One variogram model per cutoff.
    search_radius : float or None
        Maximum search distance.
    max_points : int or None
        Maximum neighbors.

    Returns
    -------
    np.ndarray
        Probability estimates, shape (m, n_cutoffs). Values in [0, 1].

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 0.25, 100)
    >>> coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype=float)
    >>> values = np.array([1.0, 3.0, 2.0, 5.0])
    >>> probs = indicator_kriging(coords, values, np.array([[25, 25]]), [2.5], [model])
    >>> 0 <= probs[0, 0] <= 1
    True

    References
    ----------
    .. [1] Goovaerts, P. (1997). "Geostatistics for Natural Resources
       Evaluation." Oxford University Press, Ch. 7.
    """
    from minelab.geostatistics.transformations import indicator_transform

    values = validate_array(values, "values")
    cutoffs = np.asarray(cutoffs, dtype=float)
    if len(variogram_models) != len(cutoffs):
        raise ValueError("Need one variogram model per cutoff.")

    indicators = indicator_transform(values, cutoffs).astype(float)

    target_coords = np.asarray(target_coords, dtype=float)
    if target_coords.ndim == 1:
        target_coords = target_coords.reshape(1, -1)

    m = target_coords.shape[0]
    n_cutoffs = len(cutoffs)
    probs = np.zeros((m, n_cutoffs))

    for k in range(n_cutoffs):
        est, _ = ordinary_kriging(
            coords,
            indicators[:, k],
            target_coords,
            variogram_models[k],
            search_radius,
            max_points,
        )
        probs[:, k] = np.clip(est, 0.0, 1.0)

    return probs


def block_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    block_def: dict,
    variogram_model: VariogramModel,
    discretization: int = 4,
    search_radius: float | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Block kriging for regular blocks.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, d).
    values : np.ndarray
        Sample values, shape (n,).
    block_def : dict
        Block definition with keys ``"origin"`` (block center or grid origin),
        ``"size"`` (block dimensions), ``"n_blocks"`` (number of blocks per dim).
    variogram_model : VariogramModel
        Fitted variogram model.
    discretization : int
        Number of discretization points per block dimension.
    search_radius : float or None
        Maximum search distance.
    max_points : int or None
        Maximum neighbors.

    Returns
    -------
    estimates : np.ndarray
        Block kriging estimates.
    variances : np.ndarray
        Block kriging variances (should be < point kriging variances).

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 10, 100)
    >>> coords = np.array([[25, 25], [75, 25], [25, 75], [75, 75]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bdef = {"origin": np.array([0, 0]), "size": np.array([50, 50]),
    ...         "n_blocks": np.array([2, 2])}
    >>> est, var = block_kriging(coords, values, bdef, model, discretization=3)
    >>> len(est) == 4
    True

    References
    ----------
    .. [1] Journel, A.G. & Huijbregts, C.J. (1978). "Mining Geostatistics."
       Academic Press, Ch. IV.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")

    origin = np.asarray(block_def["origin"], dtype=float)
    size = np.asarray(block_def["size"], dtype=float)
    n_blocks = np.asarray(block_def["n_blocks"], dtype=int)

    ndim = len(origin)

    # Generate block centers
    axes = []
    for d in range(ndim):
        axes.append(origin[d] + size[d] * (np.arange(n_blocks[d]) + 0.5))
    grids = np.meshgrid(*axes, indexing="ij")
    block_centers = np.column_stack([g.ravel() for g in grids])

    # Generate discretization offsets within a block
    disc_axes = []
    for d in range(ndim):
        disc_axes.append(
            np.linspace(-size[d] / 2, size[d] / 2, discretization, endpoint=False)
            + size[d] / (2 * discretization)
        )
    disc_grids = np.meshgrid(*disc_axes, indexing="ij")
    disc_offsets = np.column_stack([g.ravel() for g in disc_grids])

    n_total = len(block_centers)
    estimates = np.zeros(n_total)
    variances = np.zeros(n_total)

    for i in range(n_total):
        center = block_centers[i]
        disc_points = center + disc_offsets

        # Average point kriging over discretization points
        est_pts, var_pts = ordinary_kriging(
            coords,
            values,
            disc_points,
            variogram_model,
            search_radius,
            max_points,
        )

        valid = ~np.isnan(est_pts)
        if np.any(valid):
            estimates[i] = float(np.mean(est_pts[valid]))
            # Block variance is reduced by within-block averaging
            variances[i] = float(np.mean(var_pts[valid]))
        else:
            estimates[i] = np.nan
            variances[i] = np.nan

    return estimates, variances


def cross_validate(
    coords: np.ndarray,
    values: np.ndarray,
    variogram_model: VariogramModel,
    method: str = "ok",
    global_mean: float | None = None,
    search_radius: float | None = None,
    max_points: int | None = None,
) -> dict:
    """Leave-one-out cross-validation.

    Parameters
    ----------
    coords : np.ndarray
        Sample coordinates, shape (n, d).
    values : np.ndarray
        Sample values, shape (n,).
    variogram_model : VariogramModel
        Fitted variogram model.
    method : str
        ``"ok"`` for ordinary kriging, ``"sk"`` for simple kriging.
    global_mean : float or None
        Required for SK.
    search_radius : float or None
        Search radius.
    max_points : int or None
        Maximum neighbors.

    Returns
    -------
    dict
        Keys: ``"errors"`` (z - z*), ``"estimates"``, ``"variances"``,
        ``"standardized_errors"`` (error / sqrt(variance)),
        ``"mean_error"``, ``"mean_squared_error"``.

    Examples
    --------
    >>> from minelab.geostatistics.variogram_fitting import fit_variogram_manual
    >>> model = fit_variogram_manual("spherical", 0, 10, 100)
    >>> coords = np.array([[0, 0], [50, 0], [0, 50], [50, 50], [25, 25]], dtype=float)
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
    >>> cv = cross_validate(coords, values, model)
    >>> len(cv["errors"]) == 5
    True

    References
    ----------
    .. [1] Standard geostatistical practice.
    """
    coords = np.asarray(coords, dtype=float)
    values = validate_array(values, "values")
    n = len(values)

    est_arr = np.zeros(n)
    var_arr = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        target = coords[i : i + 1]

        if method == "ok":
            est, var = ordinary_kriging(
                coords[mask],
                values[mask],
                target,
                variogram_model,
                search_radius,
                max_points,
            )
        elif method == "sk":
            if global_mean is None:
                raise ValueError("global_mean required for simple kriging CV.")
            est, var = simple_kriging(
                coords[mask],
                values[mask],
                target,
                variogram_model,
                global_mean,
                search_radius,
                max_points,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'ok' or 'sk'.")

        est_arr[i] = est[0]
        var_arr[i] = var[0]

    errors = values - est_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        std_errors = np.where(var_arr > 0, errors / np.sqrt(var_arr), np.nan)

    return {
        "errors": errors,
        "estimates": est_arr,
        "variances": var_arr,
        "standardized_errors": std_errors,
        "mean_error": float(np.nanmean(errors)),
        "mean_squared_error": float(np.nanmean(errors**2)),
    }
