"""Stochastic simulation for mining project evaluation.

This module provides Monte Carlo simulation tools for risk analysis of
mining projects, including generic Monte Carlo engines and NPV-specific
simulations under uncertainty.

References
----------
.. [1] Gentry, D.W. & O'Neil, T.J. (1984). *Mine Investment Analysis*.
       AIME, Ch. 12.
.. [2] Hustrulid, W. et al. (2013). *Open Pit Mine Planning and Design*,
       3rd ed. CRC Press, Ch. 4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DistSpec = tuple[str, tuple[float, ...]]
"""A distribution specification: (dist_type, params).

Supported distributions:
- ``("triangular", (low, mode, high))``
- ``("uniform", (low, high))``
- ``("normal", (mean, std))``
- ``("lognormal", (mean, sigma))``  — parameters of the underlying normal
- ``("fixed", (value,))``
"""


def _sample_distribution(
    spec: _DistSpec,
    n: int,
    rng: Generator,
) -> np.ndarray:
    """Draw *n* samples from the distribution described by *spec*.

    Parameters
    ----------
    spec : tuple
        ``(dist_type, params)`` — see module-level ``_DistSpec``.
    n : int
        Number of samples.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n,)`` with sampled values.
    """
    dist_type, params = spec[0].lower(), spec[1]

    if dist_type == "triangular":
        low, mode, high = params
        return rng.triangular(low, mode, high, size=n)
    if dist_type == "uniform":
        low, high = params
        return rng.uniform(low, high, size=n)
    if dist_type == "normal":
        mean, std = params
        return rng.normal(mean, std, size=n)
    if dist_type == "lognormal":
        mean, sigma = params
        return rng.lognormal(mean, sigma, size=n)
    if dist_type == "fixed":
        return np.full(n, params[0])

    raise ValueError(f"Unsupported distribution type: '{dist_type}'")


# ---------------------------------------------------------------------------
# Triangular Sampling (convenience)
# ---------------------------------------------------------------------------


def triangular_sample(
    low: float,
    mode: float,
    high: float,
    n: int,
    rng: Generator | None = None,
) -> np.ndarray:
    """Sample from a triangular distribution.

    Parameters
    ----------
    low : float
        Minimum value.
    mode : float
        Most likely value.
    high : float
        Maximum value.
    n : int
        Number of samples.  Must be >= 1.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If ``None``, a new default generator is
        created.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n,)`` with sampled values.

    Raises
    ------
    ValueError
        If ``low > mode`` or ``mode > high`` or ``low == high``.

    Examples
    --------
    >>> samples = triangular_sample(1, 3, 5, 10_000, rng=np.random.default_rng(0))
    >>> 1.0 <= samples.min() and samples.max() <= 5.0
    True
    """
    if low > mode or mode > high:
        raise ValueError("Must satisfy low <= mode <= high.")
    if low == high:
        raise ValueError("low and high must differ.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if rng is None:
        rng = np.random.default_rng()
    return rng.triangular(low, mode, high, size=n)


# ---------------------------------------------------------------------------
# Generic Monte Carlo Engine
# ---------------------------------------------------------------------------


def run_monte_carlo(
    model_fn: Callable[..., float],
    param_distributions: dict[str, _DistSpec],
    n_iterations: int,
    rng: Generator | None = None,
) -> np.ndarray:
    """Run a generic Monte Carlo simulation.

    For each iteration, parameters are sampled from their specified
    distributions, passed to *model_fn* as keyword arguments, and the
    scalar result is stored.

    Parameters
    ----------
    model_fn : callable
        A function ``(**params) -> float`` that returns the metric of
        interest (e.g., NPV).
    param_distributions : dict
        ``{param_name: (dist_type, params)}`` mapping parameter names to
        their distribution specifications.  See ``_DistSpec`` for supported
        types.
    n_iterations : int
        Number of Monte Carlo iterations.  Must be >= 1.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_iterations,)`` with simulation results.

    Examples
    --------
    >>> def profit(price, cost):
    ...     return price - cost
    >>> dists = {
    ...     "price": ("uniform", (80, 120)),
    ...     "cost": ("fixed", (50,)),
    ... }
    >>> results = run_monte_carlo(profit, dists, 5000, rng=np.random.default_rng(42))
    >>> 20 < results.mean() < 80
    True

    References
    ----------
    .. [1] Gentry & O'Neil (1984), Ch. 12.
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")
    if rng is None:
        rng = np.random.default_rng()

    # Pre-sample all parameters at once for vectorisation where possible
    sampled: dict[str, np.ndarray] = {}
    for name, spec in param_distributions.items():
        sampled[name] = _sample_distribution(spec, n_iterations, rng)

    results = np.empty(n_iterations)
    for i in range(n_iterations):
        kwargs = {name: float(arr[i]) for name, arr in sampled.items()}
        results[i] = model_fn(**kwargs)

    return results


# ---------------------------------------------------------------------------
# Monte Carlo NPV
# ---------------------------------------------------------------------------


def mc_npv(
    rate: float,
    cashflow_distributions: Sequence[_DistSpec],
    n_iterations: int,
    rng: Generator | None = None,
) -> np.ndarray:
    """Monte Carlo simulation of Net Present Value under uncertainty.

    Each period's cash flow is drawn independently from its specified
    distribution.  The NPV is computed for every iteration.

    Parameters
    ----------
    rate : float
        Discount rate per period.  Must be > -1.
    cashflow_distributions : sequence of (dist_type, params)
        One distribution specification per period (starting at t = 0).
    n_iterations : int
        Number of Monte Carlo iterations.  Must be >= 1.
    rng : numpy.random.Generator or None, optional
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_iterations,)`` with simulated NPV values.

    Examples
    --------
    >>> dists = [
    ...     ("fixed", (-1000,)),
    ...     ("triangular", (200, 300, 400)),
    ...     ("triangular", (350, 420, 500)),
    ...     ("triangular", (500, 680, 800)),
    ... ]
    >>> results = mc_npv(0.10, dists, 10_000, rng=np.random.default_rng(42))
    >>> results.shape
    (10000,)

    References
    ----------
    .. [1] Gentry & O'Neil (1984), Ch. 12.
    """
    if rate <= -1:
        raise ValueError("Discount rate must be greater than -1.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")
    if rng is None:
        rng = np.random.default_rng()

    n_periods = len(cashflow_distributions)
    discount_factors = (1.0 + rate) ** (-np.arange(n_periods))

    # Build (n_iterations x n_periods) matrix of sampled cash flows
    cf_matrix = np.empty((n_iterations, n_periods))
    for t, spec in enumerate(cashflow_distributions):
        cf_matrix[:, t] = _sample_distribution(spec, n_iterations, rng)

    # Vectorised NPV computation
    return cf_matrix @ discount_factors


# ---------------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------------


def confidence_intervals(
    results: Any,
    levels: Sequence[float] = (10, 50, 90),
) -> dict[str, float]:
    """Compute percentile-based confidence intervals from simulation results.

    In mining convention, P10 is the value exceeded 90 % of the time
    (pessimistic), P50 is the median, and P90 is the value exceeded only
    10 % of the time (optimistic).

    Parameters
    ----------
    results : array_like
        1-D array of simulation outcomes.
    levels : sequence of float, optional
        Percentile levels to compute (default ``(10, 50, 90)``).

    Returns
    -------
    dict
        ``{f"P{level}": value}`` for each requested percentile.

    Examples
    --------
    >>> ci = confidence_intervals(np.arange(1, 101), levels=(10, 50, 90))
    >>> ci["P50"]
    50.5
    """
    arr = np.asarray(results, dtype=float)
    return {f"P{int(p)}": float(np.percentile(arr, p)) for p in levels}
