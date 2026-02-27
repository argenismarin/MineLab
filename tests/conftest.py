"""Shared fixtures and test configuration for MineLab test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_grades(rng):
    """Sample grade data (lognormal, typical gold deposit)."""
    return rng.lognormal(mean=0.5, sigma=1.0, size=200)


@pytest.fixture
def sample_coordinates(rng):
    """Sample 3D coordinates for 200 points."""
    return rng.uniform(0, 1000, size=(200, 3))


@pytest.fixture
def sample_drillhole_collar():
    """Sample drillhole collar DataFrame."""
    return pd.DataFrame({
        "hole_id": ["DH001", "DH002", "DH003"],
        "x": [1000.0, 1050.0, 1100.0],
        "y": [2000.0, 2000.0, 2000.0],
        "z": [500.0, 505.0, 510.0],
        "max_depth": [100.0, 120.0, 80.0],
    })


@pytest.fixture
def sample_drillhole_survey():
    """Sample drillhole survey DataFrame."""
    rows = []
    for hole_id, max_depth in [("DH001", 100), ("DH002", 120), ("DH003", 80)]:
        for depth in range(0, max_depth + 1, 20):
            rows.append({
                "hole_id": hole_id,
                "depth": float(depth),
                "azimuth": 0.0,
                "dip": -90.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_drillhole_assay():
    """Sample drillhole assay DataFrame with gold grades."""
    rng_local = np.random.default_rng(seed=123)
    rows = []
    for hole_id, max_depth in [("DH001", 100), ("DH002", 120), ("DH003", 80)]:
        for from_d in range(0, max_depth, 2):
            rows.append({
                "hole_id": hole_id,
                "from_depth": float(from_d),
                "to_depth": float(from_d + 2),
                "au_gpt": float(rng_local.lognormal(0.0, 1.0)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_cashflows():
    """Sample mining project cashflows (CAPEX + 5 years production)."""
    return [-50_000_000, 12_000_000, 15_000_000, 18_000_000, 14_000_000, 10_000_000]


@pytest.fixture
def walker_lake_small(rng):
    """Small synthetic dataset mimicking Walker Lake spatial structure."""
    n = 100
    x = rng.uniform(0, 300, n)
    y = rng.uniform(0, 300, n)
    z = np.sin(x / 50) * np.cos(y / 50) * 10 + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y, "v": z})
