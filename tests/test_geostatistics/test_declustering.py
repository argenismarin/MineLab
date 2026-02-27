"""Tests for minelab.geostatistics.declustering."""

import numpy as np
import pytest

from minelab.geostatistics.declustering import (
    cell_declustering,
    optimal_cell_size,
    polygonal_declustering,
)


class TestCellDeclustering:
    """Tests for cell declustering."""

    def test_weights_sum_to_n(self):
        """Weights should sum to number of points."""
        coords = np.array([[0, 0], [1, 0], [1, 1], [10, 10]], dtype=float)
        values = np.array([1, 2, 3, 10], dtype=float)
        result = cell_declustering(coords, values, [5, 5])
        assert sum(result["weights"]) == pytest.approx(4.0, rel=1e-10)

    def test_clustered_lower_weight(self):
        """Clustered points should get lower individual weights."""
        coords = np.array([
            [0, 0], [0.1, 0], [0.2, 0],  # cluster of 3
            [10, 10],                       # isolated
        ], dtype=float)
        values = np.array([1, 2, 3, 10], dtype=float)
        result = cell_declustering(coords, values, [5, 5])
        # Isolated point should have higher weight than clustered ones
        assert result["weights"][3] > result["weights"][0]

    def test_uniform_data_equal_weights(self):
        """Uniformly distributed data should get approximately equal weights."""
        coords = np.array([
            [5, 5], [15, 5], [5, 15], [15, 15],
        ], dtype=float)
        values = np.ones(4)
        result = cell_declustering(coords, values, [10, 10])
        np.testing.assert_allclose(result["weights"], 1.0, rtol=1e-10)

    def test_declustered_mean(self):
        """Declustered mean should differ from naive mean for clustered data."""
        coords = np.array([
            [0, 0], [0.1, 0], [0.2, 0],  # cluster, low values
            [10, 10],                       # isolated, high value
        ], dtype=float)
        values = np.array([1, 1, 1, 10], dtype=float)
        result = cell_declustering(coords, values, [5, 5])
        naive_mean = np.mean(values)  # 3.25
        # Declustered mean should give more weight to the isolated high value
        assert result["declustered_mean"] > naive_mean


class TestPolygonalDeclustering:
    """Tests for polygonal declustering."""

    def test_weights_sum_to_n(self):
        """Weights should sum to n."""
        coords = np.array([[0, 0], [1, 0], [0.5, 1], [5, 5]], dtype=float)
        values = np.array([1, 2, 3, 10], dtype=float)
        result = polygonal_declustering(coords, values)
        assert sum(result["weights"]) == pytest.approx(4.0, rel=0.01)

    def test_isolated_gets_higher_weight(self):
        """Isolated point should have higher weight."""
        coords = np.array([
            [0, 0], [0.1, 0], [0.2, 0],  # cluster
            [10, 10],                       # isolated
        ], dtype=float)
        values = np.array([1, 1, 1, 10], dtype=float)
        result = polygonal_declustering(coords, values)
        # Isolated point has larger Voronoi polygon
        assert result["weights"][3] > result["weights"][0]

    def test_3d_raises(self):
        """3D coordinates should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            polygonal_declustering(
                np.array([[0, 0, 0], [1, 0, 0]]),
                np.array([1.0, 2.0]),
            )


class TestOptimalCellSize:
    """Tests for optimal cell size search."""

    def test_returns_optimal(self):
        """Should return a valid optimal cell size."""
        rng = np.random.default_rng(42)
        # Clustered data: many low-value points near origin, few high-value far away
        coords = np.vstack([
            rng.normal(0, 1, (50, 2)),
            rng.normal(10, 1, (5, 2)),
        ])
        values = np.concatenate([np.ones(50) * 2, np.full(5, 20.0)])
        result = optimal_cell_size(coords, values, 0.5, 20.0, n_steps=10)
        assert result["optimal_size"] >= 0.5
        assert result["optimal_size"] <= 20.0

    def test_output_arrays(self):
        """Output should contain cell_sizes and declustered_means arrays."""
        coords = np.array([[i, j] for i in range(5) for j in range(5)], dtype=float)
        values = np.arange(25, dtype=float)
        result = optimal_cell_size(coords, values, 0.5, 5.0, n_steps=5)
        assert len(result["cell_sizes"]) == 5
        assert len(result["declustered_means"]) == 5

    def test_curve_not_constant(self):
        """Declustered means should vary with cell size for clustered data."""
        rng = np.random.default_rng(42)
        coords = np.vstack([
            rng.normal(0, 0.5, (40, 2)),
            rng.normal(10, 0.5, (5, 2)),
        ])
        values = np.concatenate([np.ones(40), np.full(5, 10.0)])
        result = optimal_cell_size(coords, values, 0.5, 15.0, n_steps=10)
        # Means should not all be equal
        assert np.std(result["declustered_means"]) > 0.01
