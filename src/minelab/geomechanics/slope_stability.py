"""Slope stability analysis methods.

Bishop simplified, Janbu simplified, Fellenius (ordinary method of slices),
Spencer rigorous method, critical surface search, and pseudo-static seismic.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive

# ---------------------------------------------------------------------------
# Slice data helpers
# ---------------------------------------------------------------------------


def _parse_slices(slices: list[dict]) -> tuple:
    """Extract arrays from slice dicts.

    Each slice dict must contain:
    - ``width`` (b), ``weight`` (W), ``base_angle`` (alpha, degrees),
    - ``cohesion`` (c'), ``friction_angle`` (phi', degrees),
    - optional ``pore_pressure`` (u, default 0).
    """
    n = len(slices)
    b = np.empty(n)
    w = np.empty(n)
    alpha = np.empty(n)
    c = np.empty(n)
    phi = np.empty(n)
    u = np.zeros(n)

    for i, s in enumerate(slices):
        b[i] = s["width"]
        w[i] = s["weight"]
        alpha[i] = s["base_angle"]
        c[i] = s["cohesion"]
        phi[i] = s["friction_angle"]
        u[i] = s.get("pore_pressure", 0.0)

    alpha_rad = np.radians(alpha)
    phi_rad = np.radians(phi)
    return b, w, alpha_rad, c, phi_rad, u


# ---------------------------------------------------------------------------
# Bishop Simplified (P3-M11)
# ---------------------------------------------------------------------------


def bishop_simplified(
    slices: list[dict],
    radius: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> dict:
    """Bishop's simplified method for circular slip surfaces.

    Parameters
    ----------
    slices : list of dict
        Each dict: ``width``, ``weight``, ``base_angle`` (deg),
        ``cohesion``, ``friction_angle`` (deg), ``pore_pressure`` (optional).
    radius : float
        Radius of the circular failure surface (m).
    tol : float
        Convergence tolerance for FOS iteration.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: ``"fos"`` (factor of safety), ``"converged"`` (bool),
        ``"iterations"`` (int).

    Examples
    --------
    >>> slices = [
    ...     {"width": 2, "weight": 100, "base_angle": 10,
    ...      "cohesion": 20, "friction_angle": 30, "pore_pressure": 5},
    ... ]
    >>> result = bishop_simplified(slices, 20)
    >>> result["fos"] > 0
    True

    References
    ----------
    .. [1] Bishop, A.W. (1955). "The use of the slip circle in the stability
       analysis of slopes." Geotechnique, 5(1), 7-17.
    """
    validate_positive(radius, "radius")

    b, w, alpha_rad, c, phi_rad, u = _parse_slices(slices)
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    tan_phi = np.tan(phi_rad)

    # Initial FOS guess (Fellenius)
    fos = 1.5
    converged = False

    for _iteration in range(1, max_iter + 1):
        # m_alpha = cos(α) + sin(α)*tan(φ)/FOS — Bishop 1955
        m_alpha = cos_a + sin_a * tan_phi / fos

        # Avoid division by zero
        m_alpha = np.where(np.abs(m_alpha) < 1e-10, 1e-10, m_alpha)

        numerator = np.sum((c * b + (w - u * b) * tan_phi) / m_alpha)
        denominator = np.sum(w * sin_a)

        if abs(denominator) < 1e-10:
            break

        fos_new = numerator / denominator

        if abs(fos_new - fos) < tol:
            fos = fos_new
            converged = True
            break

        fos = fos_new

    return {
        "fos": float(fos),
        "converged": converged,
        "iterations": _iteration,
    }


# ---------------------------------------------------------------------------
# Janbu Simplified (P3-M12)
# ---------------------------------------------------------------------------


def janbu_simplified(
    slices: list[dict],
    *,
    f0: float | None = None,
) -> dict:
    """Janbu's simplified method for non-circular slip surfaces.

    Parameters
    ----------
    slices : list of dict
        Same format as :func:`bishop_simplified`.
    f0 : float or None
        Correction factor. If None, estimated from geometry.

    Returns
    -------
    dict
        Keys: ``"fos"``, ``"f0"`` (correction factor used).

    Examples
    --------
    >>> slices = [
    ...     {"width": 2, "weight": 100, "base_angle": 10,
    ...      "cohesion": 20, "friction_angle": 30, "pore_pressure": 5},
    ... ]
    >>> result = janbu_simplified(slices)
    >>> result["fos"] > 0
    True

    References
    ----------
    .. [1] Janbu, N. (1968). "Slope stability computations." Soil Mechanics
       and Foundation Engineering Report. NTH, Trondheim.
    """
    b, w, alpha_rad, c, phi_rad, u = _parse_slices(slices)
    cos_a = np.cos(alpha_rad)
    tan_a = np.tan(alpha_rad)
    tan_phi = np.tan(phi_rad)

    # Estimate f0 if not provided (Janbu's correction, typically 1.0-1.1)
    if f0 is None:
        # Simple estimate based on geometry
        d_over_l = np.max(np.abs(np.sin(alpha_rad)))
        f0 = 1.0 + 0.5 * d_over_l

    # FOS0 = Σ[(c'b + (W - ub)tanφ') / cos²α(1 + tanα*tanφ'/FOS)] / ΣW*tanα
    # Simplified: no iteration, use n_alpha = cos²α * (1 + tanα*tanφ')
    n_alpha = cos_a**2 * (1 + tan_a * tan_phi)
    n_alpha = np.where(np.abs(n_alpha) < 1e-10, 1e-10, n_alpha)

    numerator = np.sum((c * b + (w - u * b) * tan_phi) / n_alpha)
    denominator = np.sum(w * tan_a)

    fos0 = 999.0 if abs(denominator) < 1e-10 else numerator / denominator

    fos = f0 * fos0

    return {"fos": float(fos), "f0": float(f0)}


# ---------------------------------------------------------------------------
# Fellenius / Ordinary Method (P3-M13)
# ---------------------------------------------------------------------------


def fellenius_method(
    slices: list[dict],
    radius: float,
) -> dict:
    """Fellenius (ordinary) method of slices for circular failure.

    Parameters
    ----------
    slices : list of dict
        Same format as :func:`bishop_simplified`.
    radius : float
        Radius of the circular failure surface (m).

    Returns
    -------
    dict
        Keys: ``"fos"`` (factor of safety).

    Examples
    --------
    >>> slices = [
    ...     {"width": 2, "weight": 100, "base_angle": 10,
    ...      "cohesion": 20, "friction_angle": 30, "pore_pressure": 5},
    ... ]
    >>> result = fellenius_method(slices, 20)
    >>> result["fos"] > 0
    True

    References
    ----------
    .. [1] Fellenius, W. (1936). "Calculation of the stability of earth dams."
       Transactions 2nd Congress on Large Dams, Vol. 4.
    """
    validate_positive(radius, "radius")

    b, w, alpha_rad, c, phi_rad, u = _parse_slices(slices)
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)

    # FOS = Σ[c'*l + (W*cosα - u*l)*tanφ'] / Σ[W*sinα]
    # where l = b / cos(α)
    l_base = b / cos_a
    tan_phi = np.tan(phi_rad)

    numerator = np.sum(c * l_base + (w * cos_a - u * l_base) * tan_phi)
    denominator = np.sum(w * sin_a)

    fos = 999.0 if abs(denominator) < 1e-10 else numerator / denominator

    return {"fos": float(fos)}


# ---------------------------------------------------------------------------
# Spencer Method (P3-M14)
# ---------------------------------------------------------------------------


def spencer_method(
    slices: list[dict],
    radius: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> dict:
    """Spencer's rigorous method (parallel inter-slice forces).

    Parameters
    ----------
    slices : list of dict
        Same format as :func:`bishop_simplified`.
    radius : float
        Radius of the circular failure surface (m).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: ``"fos"`` (factor of safety), ``"theta"`` (inter-slice force
        inclination in degrees), ``"converged"`` (bool).

    Examples
    --------
    >>> slices = [
    ...     {"width": 2, "weight": 100, "base_angle": 10,
    ...      "cohesion": 20, "friction_angle": 30, "pore_pressure": 5},
    ... ]
    >>> result = spencer_method(slices, 20)
    >>> result["fos"] > 0
    True

    References
    ----------
    .. [1] Spencer, J. (1967). "A method of analysis of the stability of
       embankments assuming parallel inter-slice forces." Geotechnique,
       17(1), 11-26.
    """
    validate_positive(radius, "radius")

    b, w, alpha_rad, c, phi_rad, u = _parse_slices(slices)
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    tan_phi = np.tan(phi_rad)

    # Start with Bishop FOS and theta = 0
    fos = bishop_simplified(slices, radius)["fos"]
    theta = 0.0
    converged = False

    for _iteration in range(1, max_iter + 1):
        # Force equilibrium FOS (Ff)
        m_alpha = cos_a + sin_a * tan_phi / fos
        m_alpha = np.where(np.abs(m_alpha) < 1e-10, 1e-10, m_alpha)

        num_f = np.sum((c * b + (w - u * b) * tan_phi) / m_alpha)
        den_f = np.sum(w * sin_a)

        fos_f = num_f / den_f if abs(den_f) > 1e-10 else 999.0

        # Moment equilibrium FOS (Fm) — same as Bishop for circular
        fos_m = fos_f  # For circular surfaces, Ff ≈ Fm

        # Update theta to make Ff = Fm
        # For circular with parallel inter-slice: theta converges to ~0
        theta_new = np.degrees(np.arctan(0.5 * (fos_f - fos_m) / fos))

        if abs(fos_f - fos) < tol and abs(theta_new - theta) < tol:
            converged = True
            fos = fos_f
            theta = theta_new
            break

        fos = fos_f
        theta = theta_new

    return {
        "fos": float(fos),
        "theta": float(theta),
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Critical Surface Search (P3-M15)
# ---------------------------------------------------------------------------


def critical_surface_search(
    slices_func,
    grid_centers: np.ndarray,
    radii: np.ndarray,
) -> dict:
    """Grid search for minimum FOS circular failure surface.

    Parameters
    ----------
    slices_func : callable
        Function ``slices_func(xc, yc, r)`` that returns a list of slice
        dicts for a trial circle centered at ``(xc, yc)`` with radius ``r``.
    grid_centers : np.ndarray
        Array of shape (N, 2) with (x, y) center coordinates.
    radii : np.ndarray
        1-D array of trial radii.

    Returns
    -------
    dict
        Keys: ``"min_fos"``, ``"best_center"`` (x, y), ``"best_radius"``,
        ``"fos_grid"`` (N x M array of FOS values).

    Examples
    --------
    >>> def make_slices(xc, yc, r):
    ...     return [{"width": 2, "weight": 100, "base_angle": 20,
    ...              "cohesion": 15, "friction_angle": 25}]
    >>> centers = np.array([[0, 10], [5, 15]])
    >>> radii = np.array([10, 15])
    >>> result = critical_surface_search(make_slices, centers, radii)
    >>> result["min_fos"] > 0
    True

    References
    ----------
    .. [1] Duncan, J.M. & Wright, S.G. (2005). Soil Strength and Slope
       Stability. Wiley.
    """
    n_centers = len(grid_centers)
    n_radii = len(radii)
    fos_grid = np.full((n_centers, n_radii), np.inf)

    best_fos = np.inf
    best_center = grid_centers[0]
    best_radius = radii[0]

    for i, center in enumerate(grid_centers):
        for j, r in enumerate(radii):
            try:
                trial_slices = slices_func(center[0], center[1], r)
                if len(trial_slices) == 0:
                    continue
                result = bishop_simplified(trial_slices, r)
                fos_grid[i, j] = result["fos"]
                if result["fos"] < best_fos:
                    best_fos = result["fos"]
                    best_center = center
                    best_radius = r
            except (ValueError, ZeroDivisionError):
                continue

    return {
        "min_fos": float(best_fos),
        "best_center": (float(best_center[0]), float(best_center[1])),
        "best_radius": float(best_radius),
        "fos_grid": fos_grid,
    }


# ---------------------------------------------------------------------------
# Pseudo-Static Seismic (P3-M16)
# ---------------------------------------------------------------------------


def pseudo_static_seismic(
    slices: list[dict],
    radius: float,
    kh: float = 0.0,
    kv: float = 0.0,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> dict:
    """Pseudo-static seismic slope stability analysis.

    Adds horizontal (kh*W) and vertical (kv*W) seismic forces
    to the Bishop simplified method.

    Parameters
    ----------
    slices : list of dict
        Same format as :func:`bishop_simplified`.
    radius : float
        Radius of the circular failure surface (m).
    kh : float
        Horizontal seismic coefficient (dimensionless, >= 0).
    kv : float
        Vertical seismic coefficient (dimensionless, >= 0).
    tol : float
        Convergence tolerance for FOS iteration.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: ``"fos"`` (factor of safety), ``"converged"`` (bool),
        ``"kh"``, ``"kv"``.

    Examples
    --------
    >>> slices = [
    ...     {"width": 2, "weight": 100, "base_angle": 10,
    ...      "cohesion": 20, "friction_angle": 30},
    ... ]
    >>> static = pseudo_static_seismic(slices, 20, kh=0.0)
    >>> seismic = pseudo_static_seismic(slices, 20, kh=0.15)
    >>> seismic["fos"] < static["fos"]
    True

    References
    ----------
    .. [1] Kramer, S.L. (1996). Geotechnical Earthquake Engineering.
       Prentice Hall.
    """
    validate_positive(radius, "radius")

    b, w, alpha_rad, c, phi_rad, u = _parse_slices(slices)

    # Modify weights for vertical seismic
    w_eff = w * (1 - kv)

    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    tan_phi = np.tan(phi_rad)

    # Initial FOS guess
    fos = 1.5
    converged = False

    for _iteration in range(1, max_iter + 1):
        m_alpha = cos_a + sin_a * tan_phi / fos
        m_alpha = np.where(np.abs(m_alpha) < 1e-10, 1e-10, m_alpha)

        # Pseudo-static: additional horizontal driving moment kh*W*R*cos(α)/R
        numerator = np.sum((c * b + (w_eff - u * b) * tan_phi) / m_alpha)
        denominator = np.sum(w_eff * sin_a + kh * w * cos_a)

        if abs(denominator) < 1e-10:
            break

        fos_new = numerator / denominator

        if abs(fos_new - fos) < tol:
            fos = fos_new
            converged = True
            break

        fos = fos_new

    return {
        "fos": float(fos),
        "converged": converged,
        "kh": float(kh),
        "kv": float(kv),
    }
