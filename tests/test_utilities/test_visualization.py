"""Tests for minelab.utilities.visualization."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from minelab.utilities.visualization import (
    boxplot,
    grade_tonnage_plot,
    histogram_plot,
    scatter_plot,
    variogram_plot,
)

# Use a non-interactive backend for CI/headless environments
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


class TestHistogramPlot:
    def test_returns_fig_ax(self):
        fig, ax = histogram_plot(np.random.normal(0, 1, 200))
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self):
        fig, ax = histogram_plot([1, 2, 3], title="My Histogram")
        assert ax.get_title() == "My Histogram"

    def test_default_title(self):
        fig, ax = histogram_plot([1, 2, 3])
        assert ax.get_title() == "Histogram"

    def test_custom_bins(self):
        fig, ax = histogram_plot(np.random.normal(0, 1, 500), bins=10)
        # Should have patches (bars)
        assert len(ax.patches) > 0


class TestScatterPlot:
    def test_returns_fig_ax(self):
        fig, ax = scatter_plot([1, 2, 3], [4, 5, 6])
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self):
        fig, ax = scatter_plot([1, 2], [3, 4], title="My Scatter")
        assert ax.get_title() == "My Scatter"

    def test_with_color(self):
        fig, ax = scatter_plot(
            [1, 2, 3], [4, 5, 6], c=[10, 20, 30]
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_default_title(self):
        fig, ax = scatter_plot([1, 2], [3, 4])
        assert ax.get_title() == "Scatter"


class TestVariogramPlot:
    def test_returns_fig_ax(self):
        fig, ax = variogram_plot([10, 20, 30], [0.5, 0.8, 1.0])
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self):
        fig, ax = variogram_plot(
            [10, 20], [0.5, 0.8], title="My Variogram"
        )
        assert ax.get_title() == "My Variogram"

    def test_with_model(self):
        lags = [10, 20, 30]
        sv = [0.5, 0.8, 1.0]
        ml = np.linspace(0, 35, 50)
        msv = 1.0 * (1 - np.exp(-ml / 15))
        fig, ax = variogram_plot(lags, sv, model_lags=ml, model_sv=msv)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Should have a legend
        assert ax.get_legend() is not None


class TestGradeTonnagePlot:
    def test_returns_fig_ax(self):
        df = pd.DataFrame({
            "cutoff": [0.0, 0.5, 1.0, 1.5],
            "tonnes_above": [1000, 800, 500, 200],
            "mean_grade_above": [1.2, 1.5, 2.0, 2.8],
            "metal_above": [1200, 1200, 1000, 560],
        })
        fig, ax = grade_tonnage_plot(df)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self):
        df = pd.DataFrame({
            "cutoff": [0.0, 1.0],
            "tonnes_above": [1000, 500],
            "mean_grade_above": [1.2, 2.0],
            "metal_above": [1200, 1000],
        })
        fig, ax = grade_tonnage_plot(df, title="Custom GT")
        assert ax.get_title() == "Custom GT"

    def test_default_title(self):
        df = pd.DataFrame({
            "cutoff": [0.0, 1.0],
            "tonnes_above": [1000, 500],
            "mean_grade_above": [1.2, 2.0],
            "metal_above": [1200, 1000],
        })
        fig, ax = grade_tonnage_plot(df)
        assert ax.get_title() == "Grade-Tonnage Curve"


class TestBoxplot:
    def test_returns_fig_ax(self):
        fig, ax = boxplot({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_title(self):
        fig, ax = boxplot({"X": [10, 20]}, title="My Boxplot")
        assert ax.get_title() == "My Boxplot"

    def test_default_title(self):
        fig, ax = boxplot({"X": [10, 20]})
        assert ax.get_title() == "Boxplot"

    def test_multiple_groups(self):
        fig, ax = boxplot({
            "Zone A": np.random.normal(5, 1, 50),
            "Zone B": np.random.normal(8, 2, 50),
            "Zone C": np.random.normal(3, 0.5, 50),
        })
        assert isinstance(fig, matplotlib.figure.Figure)
