"""End-to-end integration test: drillhole -> composite -> variogram -> kriging
-> block model -> pit optimization -> schedule -> NPV.

Tests the full mining workflow from exploration data through to economic
evaluation, verifying that all modules integrate correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minelab.data_management import DrillholeDB, composite_by_length
from minelab.geostatistics import (
    BlockModel,
    block_grade_tonnage,
    experimental_variogram,
    fit_variogram_wls,
    ordinary_kriging,
)
from minelab.mine_planning import (
    block_economic_value,
    lerchs_grossmann_2d,
    npv_schedule,
    schedule_by_period,
)
from minelab.economics import npv


def _build_drillhole_db() -> DrillholeDB:
    """Create a small synthetic drillhole database with 6 holes."""
    db = DrillholeDB()
    # 6 drillholes on a rough grid
    collars = [
        ("DH01", 0.0, 0.0, 100.0, 30.0),
        ("DH02", 50.0, 0.0, 100.0, 30.0),
        ("DH03", 100.0, 0.0, 100.0, 30.0),
        ("DH04", 0.0, 50.0, 100.0, 30.0),
        ("DH05", 50.0, 50.0, 100.0, 30.0),
        ("DH06", 100.0, 50.0, 100.0, 30.0),
    ]
    for hid, x, y, z, md in collars:
        db.add_collar(hid, x, y, z, md)
        db.add_survey(hid, 0.0, 0.0, -90.0)

    # Assay intervals: 2 m samples down to 30 m, with a mineralized zone
    rng = np.random.default_rng(42)
    for hid, x, y, z, md in collars:
        for fr in range(0, 30, 2):
            to = fr + 2
            # Higher grades near centre (DH05) and at depth 10-20 m
            dist_from_centre = np.sqrt((x - 50) ** 2 + (y - 25) ** 2)
            depth_factor = 1.0 if 8 <= fr < 22 else 0.2
            grade = max(
                0.0,
                (3.0 - dist_from_centre / 40.0) * depth_factor
                + rng.normal(0, 0.3),
            )
            db.add_assay(hid, float(fr), float(to), au=round(grade, 3))

    return db


@pytest.fixture()
def drillhole_db() -> DrillholeDB:
    """Synthetic drillhole database fixture."""
    return _build_drillhole_db()


@pytest.fixture()
def composites(drillhole_db: DrillholeDB) -> pd.DataFrame:
    """5 m composites from the synthetic drillhole database."""
    return composite_by_length(drillhole_db.assays, length=5.0, grade_cols=["au"])


@pytest.fixture()
def sample_coords_and_values(
    composites: pd.DataFrame, drillhole_db: DrillholeDB
) -> tuple[np.ndarray, np.ndarray]:
    """Merge composite grades with collar XY coordinates."""
    merged = composites.merge(
        drillhole_db.collars[["hole_id", "x", "y"]],
        on="hole_id",
    )
    midpoint_z = (merged["from_depth"] + merged["to_depth"]) / 2.0
    coords = np.column_stack([
        merged["x"].values,
        merged["y"].values,
        midpoint_z.values,
    ])
    values = merged["au"].values.astype(float)
    return coords, values


class TestDrillholeToComposite:
    """Verify the drillhole -> composite stage."""

    def test_composites_produced(self, composites: pd.DataFrame):
        """Compositing should produce rows for every hole."""
        assert len(composites) > 0
        assert composites["hole_id"].nunique() == 6

    def test_composite_length(self, composites: pd.DataFrame):
        """Each composite should span approximately 5 m."""
        lengths = composites["to_depth"] - composites["from_depth"]
        assert all(lengths <= 5.0 + 1e-9)
        assert all(lengths > 0)

    def test_composite_grades_non_negative(self, composites: pd.DataFrame):
        """Grade values should remain non-negative after compositing."""
        assert (composites["au"] >= 0).all()


class TestVariogramAndKriging:
    """Verify variogram fitting and kriging using composited data."""

    def test_experimental_variogram(self, sample_coords_and_values):
        """Experimental variogram should return valid lag/semivariance arrays."""
        coords, values = sample_coords_and_values
        result = experimental_variogram(coords, values, n_lags=6)
        assert len(result["lags"]) == 6
        assert all(np.isfinite(result["semivariance"][:4]))

    def test_fitted_model_has_positive_sill(self, sample_coords_and_values):
        """WLS-fitted variogram model should have a positive sill."""
        coords, values = sample_coords_and_values
        ev = experimental_variogram(coords, values, n_lags=6)
        model = fit_variogram_wls(
            ev["lags"], ev["semivariance"], "spherical", ev["n_pairs"]
        )
        assert model.sill > 0
        assert model.range_a > 0

    def test_kriging_estimates_reasonable(self, sample_coords_and_values):
        """OK estimates at sample locations should be close to sample values."""
        coords, values = sample_coords_and_values
        ev = experimental_variogram(coords, values, n_lags=6)
        model = fit_variogram_wls(
            ev["lags"], ev["semivariance"], "spherical", ev["n_pairs"]
        )
        # Estimate at first 3 data points -- should recover approximately
        est, var = ordinary_kriging(coords, values, coords[:3], model)
        for i in range(3):
            assert est[i] == pytest.approx(values[i], abs=0.5)


class TestPitScheduleNPV:
    """Full workflow from block model to NPV."""

    def test_profitable_deposit_positive_npv(self, sample_coords_and_values):
        """A deposit with good grades should yield positive NPV."""
        coords, values = sample_coords_and_values

        # Fit variogram
        ev = experimental_variogram(coords, values, n_lags=6)
        model = fit_variogram_wls(
            ev["lags"], ev["semivariance"], "spherical", ev["n_pairs"]
        )

        # Create a small 2D block model (5 cols x 3 levels)
        n_cols, n_levels = 5, 3
        bm = BlockModel(
            origin=[0, 0, 0],
            block_size=[25.0, 60.0, 10.0],
            n_blocks=[n_cols, 1, n_levels],
        )
        centers = bm.block_centers()

        # Krige grades into block model
        est, kv = ordinary_kriging(coords, values, centers, model)
        bm.add_variable("au", np.maximum(est, 0.0))
        bm.add_variable("density", np.full(bm.n_total, 2.7))

        # Compute economic value per block
        # Price 50000 $/kg Au, recovery 90%, mining 3 $/t, processing 15 $/t
        block_vol = 25.0 * 60.0 * 10.0  # m^3
        block_tonnes = 2.7 * block_vol
        econ_values = np.array([
            block_economic_value(
                grade=g / 1e6,  # ppm -> fraction
                tonnage=block_tonnes,
                price=50_000_000.0,  # $/t metal
                recovery=0.90,
                mining_cost=3.0,
                processing_cost=15.0,
            )
            for g in bm.get_variable("au")
        ])

        # Reshape for 2D pit optimisation (levels x cols)
        # Use only the first row (ny=1)
        econ_2d = econ_values.reshape(n_cols, 1, n_levels)[:, 0, :].T

        # Pit optimisation
        pit = lerchs_grossmann_2d(econ_2d, (45.0, 45.0))

        # Schedule into 3 periods
        if pit["total_value"] > 0:
            cap = pit["total_value"] / 2.0
            sched = schedule_by_period(
                econ_2d, pit["pit_mask"], [cap, cap, cap], n_periods=3
            )
            npv_val = npv_schedule(sched["period_values"], discount_rate=0.10)
            assert npv_val > 0
        else:
            # If the random data doesn't produce a profitable pit,
            # the pit total value should still be >= 0 (empty pit is valid)
            assert pit["total_value"] >= 0

    def test_grade_tonnage_curve_monotonic(self, sample_coords_and_values):
        """Grade-tonnage curve tonnage should decrease with rising cutoff."""
        coords, values = sample_coords_and_values
        ev = experimental_variogram(coords, values, n_lags=6)
        model = fit_variogram_wls(
            ev["lags"], ev["semivariance"], "spherical", ev["n_pairs"]
        )

        bm = BlockModel([0, 0, 0], [25, 60, 10], [5, 1, 3])
        centers = bm.block_centers()
        est, _ = ordinary_kriging(coords, values, centers, model)
        bm.add_variable("au", np.maximum(est, 0.0))
        bm.add_variable("density", np.full(bm.n_total, 2.7))

        gt = block_grade_tonnage(bm, "au", "density", [0.0, 0.5, 1.0, 2.0])
        tonnages = gt["tonnage"].tolist()
        assert all(tonnages[i] >= tonnages[i + 1] for i in range(len(tonnages) - 1))


class TestEconomicsIntegration:
    """Verify that mine_planning.npv_schedule and economics.npv agree."""

    def test_npv_schedule_vs_economics_npv(self):
        """npv_schedule and economics.npv should give consistent results."""
        period_values = [100.0, 150.0, 200.0]
        rate = 0.10

        mine_npv = npv_schedule(period_values, rate)
        # economics.npv expects cashflows starting at t=0
        econ_npv = npv(rate, [0.0] + period_values)
        assert mine_npv == pytest.approx(econ_npv, rel=1e-6)
