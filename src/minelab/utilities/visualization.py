"""Plotting helpers for mining engineering data visualization.

Convenience wrappers around Matplotlib for common mining plots including
histograms, scatter plots, variogram plots, grade-tonnage curves, and
boxplots.  All functions return ``(fig, ax)`` tuples for further
customization.

References
----------
.. [1] Rossi, M.E. & Deutsch, C.V., *Mineral Resource Estimation*,
       Springer, 2014.
.. [2] Isaaks, E.H. & Srivastava, R.M., *An Introduction to Applied
       Geostatistics*, Oxford University Press, 1989.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minelab.utilities.validators import validate_array


def histogram_plot(
    data: Sequence[float],
    bins: int = 30,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a histogram of the data.

    Parameters
    ----------
    data : array-like of float
        Data values to plot.
    bins : int, optional
        Number of histogram bins (default 30).
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import numpy as np
    >>> fig, ax = histogram_plot(np.random.normal(0, 1, 500))
    >>> ax.get_title()
    'Histogram'

    References
    ----------
    .. [1] Rossi & Deutsch, 2014, ch. 3.
    """
    arr = validate_array(data, "data")
    fig, ax = plt.subplots()
    ax.hist(arr, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    c: Sequence[float] | None = None,
    title: str = "Scatter",
    xlabel: str = "X",
    ylabel: str = "Y",
    colorbar_label: str = "Value",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a scatter plot with optional colour mapping.

    Parameters
    ----------
    x : array-like of float
        X coordinates.
    y : array-like of float
        Y coordinates.
    c : array-like of float, optional
        Colour values for each point.  If ``None``, a single colour is
        used.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    colorbar_label : str, optional
        Label for the colour bar (if *c* is provided).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import numpy as np
    >>> fig, ax = scatter_plot([1, 2, 3], [4, 5, 6])
    >>> ax.get_title()
    'Scatter'

    References
    ----------
    .. [1] Rossi & Deutsch, 2014, ch. 3.
    """
    xa = validate_array(x, "x")
    ya = validate_array(y, "y")
    fig, ax = plt.subplots()
    if c is not None:
        ca = validate_array(c, "c")
        sc = ax.scatter(xa, ya, c=ca, cmap="viridis", edgecolors="k", linewidths=0.3, s=20)
        fig.colorbar(sc, ax=ax, label=colorbar_label)
    else:
        ax.scatter(xa, ya, edgecolors="k", linewidths=0.3, s=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def variogram_plot(
    lags: Sequence[float],
    semivariances: Sequence[float],
    model_lags: Sequence[float] | None = None,
    model_sv: Sequence[float] | None = None,
    title: str = "Variogram",
    xlabel: str = "Lag distance",
    ylabel: str = "Semivariance",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an experimental variogram with optional fitted model.

    Parameters
    ----------
    lags : array-like of float
        Lag distances for experimental variogram points.
    semivariances : array-like of float
        Semivariance values at each lag.
    model_lags : array-like of float, optional
        Lag distances for the model curve.
    model_sv : array-like of float, optional
        Model semivariance values.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> fig, ax = variogram_plot([10, 20, 30], [0.5, 0.8, 1.0])
    >>> ax.get_title()
    'Variogram'

    References
    ----------
    .. [1] Isaaks & Srivastava, 1989, ch. 7.
    """
    la = validate_array(lags, "lags")
    sv = validate_array(semivariances, "semivariances")
    fig, ax = plt.subplots()
    ax.plot(la, sv, "ko", markersize=6, label="Experimental")
    if model_lags is not None and model_sv is not None:
        ml = validate_array(model_lags, "model_lags")
        msv = validate_array(model_sv, "model_sv")
        ax.plot(ml, msv, "b-", linewidth=1.5, label="Model")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def grade_tonnage_plot(
    gt_df: pd.DataFrame,
    title: str = "Grade-Tonnage Curve",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a dual-axis grade-tonnage curve.

    Expects a DataFrame produced by
    :func:`minelab.utilities.grades.grade_tonnage_curve` with columns
    ``cutoff``, ``tonnes_above``, ``mean_grade_above``.

    Parameters
    ----------
    gt_df : pandas.DataFrame
        Grade-tonnage table with at least ``cutoff``,
        ``tonnes_above``, and ``mean_grade_above`` columns.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The primary axes (tonnage).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'cutoff': [0.0, 0.5, 1.0],
    ...     'tonnes_above': [1000, 800, 500],
    ...     'mean_grade_above': [1.2, 1.5, 2.0],
    ...     'metal_above': [1200, 1200, 1000],
    ... })
    >>> fig, ax = grade_tonnage_plot(df)
    >>> ax.get_title()
    'Grade-Tonnage Curve'

    References
    ----------
    .. [1] Hustrulid, W. et al., *Open Pit Mine Planning and Design*,
           3rd ed., CRC Press, 2013, ch. 6.
    """
    fig, ax1 = plt.subplots()

    color_tonnes = "tab:blue"
    ax1.plot(
        gt_df["cutoff"],
        gt_df["tonnes_above"],
        "o-",
        color=color_tonnes,
        label="Tonnes above",
    )
    ax1.set_xlabel("Cutoff grade")
    ax1.set_ylabel("Tonnes above cutoff", color=color_tonnes)
    ax1.tick_params(axis="y", labelcolor=color_tonnes)

    ax2 = ax1.twinx()
    color_grade = "tab:red"
    ax2.plot(
        gt_df["cutoff"],
        gt_df["mean_grade_above"],
        "s-",
        color=color_grade,
        label="Mean grade",
    )
    ax2.set_ylabel("Mean grade above cutoff", color=color_grade)
    ax2.tick_params(axis="y", labelcolor=color_grade)

    ax1.set_title(title)
    fig.tight_layout()
    return fig, ax1


def boxplot(
    data_dict: dict[str, Sequence[float]],
    title: str = "Boxplot",
    ylabel: str = "Value",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a comparative boxplot from a dictionary of datasets.

    Parameters
    ----------
    data_dict : dict of {str: array-like}
        Keys are group labels, values are data arrays.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> fig, ax = boxplot({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> ax.get_title()
    'Boxplot'

    References
    ----------
    .. [1] Rossi & Deutsch, 2014, ch. 3.
    """
    fig, ax = plt.subplots()
    labels = list(data_dict.keys())
    datasets = [np.asarray(data_dict[k], dtype=float) for k in labels]
    ax.boxplot(datasets, tick_labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig, ax
