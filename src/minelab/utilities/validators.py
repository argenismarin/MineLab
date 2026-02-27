"""Input validation helpers for mining engineering calculations.

These validators provide consistent error messages and are used across
all MineLab modules to check function arguments before computation.

References
----------
.. [1] MineLab project coding conventions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

Number = int | float


def validate_positive(value: Number, name: str = "value") -> None:
    """Raise ``ValueError`` if *value* is not strictly positive.

    Parameters
    ----------
    value : int or float
        Value to check.
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If *value* <= 0.

    Examples
    --------
    >>> validate_positive(5.0, 'length')
    >>> validate_positive(-1, 'length')
    Traceback (most recent call last):
        ...
    ValueError: 'length' must be positive, got -1.

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    if value <= 0:
        raise ValueError(f"'{name}' must be positive, got {value}.")


def validate_non_negative(value: Number, name: str = "value") -> None:
    """Raise ``ValueError`` if *value* is negative.

    Parameters
    ----------
    value : int or float
        Value to check.
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If *value* < 0.

    Examples
    --------
    >>> validate_non_negative(0, 'recovery')
    >>> validate_non_negative(-0.1, 'recovery')
    Traceback (most recent call last):
        ...
    ValueError: 'recovery' must be non-negative, got -0.1.

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    if value < 0:
        raise ValueError(f"'{name}' must be non-negative, got {value}.")


def validate_range(
    value: Number,
    low: Number,
    high: Number,
    name: str = "value",
) -> None:
    """Raise ``ValueError`` if *value* is outside [*low*, *high*].

    Parameters
    ----------
    value : int or float
        Value to check.
    low : int or float
        Lower bound (inclusive).
    high : int or float
        Upper bound (inclusive).
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If *value* < *low* or *value* > *high*.

    Examples
    --------
    >>> validate_range(50, 0, 100, 'percentage')
    >>> validate_range(101, 0, 100, 'percentage')
    Traceback (most recent call last):
        ...
    ValueError: 'percentage' must be in [0, 100], got 101.

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    if value < low or value > high:
        raise ValueError(f"'{name}' must be in [{low}, {high}], got {value}.")


def validate_percentage(value: Number, name: str = "value") -> None:
    """Raise ``ValueError`` if *value* is not in [0, 100].

    Parameters
    ----------
    value : int or float
        Percentage value to check.
    name : str, optional
        Name used in the error message.

    Raises
    ------
    ValueError
        If *value* < 0 or *value* > 100.

    Examples
    --------
    >>> validate_percentage(85.5, 'recovery')
    >>> validate_percentage(101, 'recovery')
    Traceback (most recent call last):
        ...
    ValueError: 'recovery' must be in [0, 100], got 101.

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    validate_range(value, 0, 100, name)


def validate_array(
    arr: ArrayLike,
    name: str = "array",
    min_length: int = 1,
) -> np.ndarray:
    """Validate and convert *arr* to a 1-D NumPy array.

    Parameters
    ----------
    arr : array-like
        Input data (list, tuple, or ndarray).
    name : str, optional
        Name used in error messages.
    min_length : int, optional
        Minimum required number of elements (default 1).

    Returns
    -------
    numpy.ndarray
        Validated 1-D array.

    Raises
    ------
    ValueError
        If the resulting array has fewer than *min_length* elements.
    TypeError
        If *arr* cannot be converted to a numeric array.

    Examples
    --------
    >>> validate_array([1.0, 2.0, 3.0], 'grades').shape
    (3,)
    >>> validate_array([], 'grades')
    Traceback (most recent call last):
        ...
    ValueError: 'grades' must have at least 1 element(s), got 0.

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    try:
        result = np.asarray(arr, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{name}' must be convertible to a numeric array.") from exc
    if result.size < min_length:
        raise ValueError(
            f"'{name}' must have at least {min_length} element(s), got {result.size}."
        )
    return result


def validate_probabilities(
    probs: ArrayLike,
    name: str = "probabilities",
    tol: float = 1e-6,
) -> np.ndarray:
    """Validate that *probs* sums to approximately 1.0.

    Each element must be in [0, 1] and the total must be within *tol*
    of 1.0.

    Parameters
    ----------
    probs : array-like
        Probability values.
    name : str, optional
        Name used in error messages.
    tol : float, optional
        Tolerance for sum check (default 1e-6).

    Returns
    -------
    numpy.ndarray
        Validated 1-D probability array.

    Raises
    ------
    ValueError
        If any element is outside [0, 1] or the sum deviates from 1.0
        by more than *tol*.

    Examples
    --------
    >>> validate_probabilities([0.3, 0.7], 'p')
    array([0.3, 0.7])
    >>> validate_probabilities([0.5, 0.6], 'p')
    Traceback (most recent call last):
        ...
    ValueError: 'p' must sum to 1.0 (got 1.1).

    References
    ----------
    .. [1] MineLab project coding conventions.
    """
    result = validate_array(probs, name, min_length=1)
    if np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"All elements of '{name}' must be in [0, 1].")
    total = result.sum()
    if abs(total - 1.0) > tol:
        raise ValueError(f"'{name}' must sum to 1.0 (got {total}).")
    return result
