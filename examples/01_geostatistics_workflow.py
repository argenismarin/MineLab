"""Geostatistics workflow: drillhole data to resource estimate.

This example demonstrates a complete geostatistical resource estimation
pipeline using MineLab:

1. Build a synthetic drillhole database with collar, survey, and assay data.
2. Desurvey using the minimum-curvature method.
3. Composite assay intervals to uniform 5 m lengths.
4. Compute an experimental variogram and fit a spherical model.
5. Create a block model and estimate grades with ordinary kriging.
6. Classify blocks by kriging variance (Measured / Indicated / Inferred).
7. Generate a grade-tonnage curve and a resource statement.
"""

from __future__ import annotations

import numpy as np

from minelab.data_management import DrillholeDB, composite_by_length, minimum_curvature
from minelab.geostatistics import (
    BlockModel,
    block_grade_tonnage,
    experimental_variogram,
    fit_variogram_wls,
    ordinary_kriging,
)
from minelab.resource_classification import classify_by_kriging_variance, resource_statement


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a synthetic drillhole database
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    db = DrillholeDB()

    # 25 vertical drillholes on a 25 m grid within a 100x100 m area
    hole_idx = 0
    for east in [100.0, 125.0, 150.0, 175.0, 200.0]:
        for north in [100.0, 125.0, 150.0, 175.0, 200.0]:
            hid = f"DH{hole_idx:03d}"
            db.add_collar(hid, east, north, 500.0, max_depth=50.0)
            db.add_survey(hid, 0.0, 0.0, -90.0)  # collar: vertical
            db.add_survey(hid, 50.0, 0.0, -90.0)  # bottom: vertical

            # Assay every 2 m with a grade that varies spatially
            base_grade = 1.0 + 0.005 * east + 0.003 * north
            for d in range(0, 50, 2):
                grade = max(0.1, base_grade + rng.normal(0, 0.3))
                db.add_assay(hid, float(d), float(d + 2), au_gpt=round(grade, 2))
            hole_idx += 1

    issues = db.validate()
    print(f"Drillhole DB: {db}  |  Validation issues: {len(issues)}")

    # ------------------------------------------------------------------
    # 2. Desurvey with minimum curvature
    # ------------------------------------------------------------------
    desurveyed = minimum_curvature(db.surveys)
    print(f"Desurveyed records: {len(desurveyed)}")

    # ------------------------------------------------------------------
    # 3. Composite to 5 m intervals
    # ------------------------------------------------------------------
    merged = db.to_dataframe()
    composites = composite_by_length(merged, length=5.0, grade_cols=["au_gpt"])
    print(f"Composites: {len(composites)} intervals at 5 m")

    # Build coordinate arrays: use collar XY for each composite
    collar_map = {
        hid: (float(x), float(y))
        for hid, x, y in zip(
            db.collars["hole_id"],
            db.collars["x"],
            db.collars["y"],
            strict=False,
        )
    }
    xs, ys = [], []
    for _, row in composites.iterrows():
        cx, cy = collar_map[row["hole_id"]]
        xs.append(cx)
        ys.append(cy)
    coords = np.column_stack([xs, ys])
    values = composites["au_gpt"].values.astype(float)

    # De-duplicate by averaging composites at the same collar location
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    unique_values = np.array([values[inverse == i].mean() for i in range(len(unique_coords))])

    # ------------------------------------------------------------------
    # 4. Experimental variogram and model fitting
    # ------------------------------------------------------------------
    exp_vario = experimental_variogram(unique_coords, unique_values, n_lags=6, lag_dist=25.0)

    print("\nExperimental variogram:")
    for i, (lag, sv, nprs) in enumerate(
        zip(
            exp_vario["lags"],
            exp_vario["semivariance"],
            exp_vario["n_pairs"],
            strict=True,
        )
    ):
        print(f"  Lag {i + 1}: h={lag:6.1f} m, gamma={sv:.4f}, pairs={nprs}")

    # Fit a spherical model via weighted least squares
    model = fit_variogram_wls(
        exp_vario["lags"],
        exp_vario["semivariance"],
        model_type="spherical",
        n_pairs=exp_vario["n_pairs"],
    )
    print(f"\nFitted model: {model.model_type}")
    print(f"  Nugget = {model.nugget:.4f}")
    print(f"  Sill   = {model.sill:.4f}")
    print(f"  Range  = {model.range_a:.1f} m")
    print(f"  RMSE   = {model.rmse:.6f}")

    # ------------------------------------------------------------------
    # 5. Block model creation and ordinary kriging
    # ------------------------------------------------------------------
    bm = BlockModel(
        origin=[90.0, 90.0, 475.0],
        block_size=[10.0, 10.0, 25.0],
        n_blocks=[13, 13, 1],
    )
    centers = bm.block_centers()
    target_2d = centers[:, :2]  # kriging in 2D (plan view)

    estimates, variances = ordinary_kriging(
        unique_coords, unique_values, target_2d, model, max_points=16
    )

    # Assign grades and a constant density to the block model
    bm.add_variable("au_gpt", estimates)
    bm.add_variable("density", np.full(bm.n_total, 2.7))
    bm.add_variable("kv", variances)

    valid = ~np.isnan(estimates)
    print(f"\nBlock model: {bm.n_total} blocks ({bm.n_blocks})")
    print(f"  Estimated blocks: {valid.sum()}")
    if valid.any():
        print(f"  Mean grade: {np.nanmean(estimates):.3f} g/t Au")
        print(f"  Mean kriging variance: {np.nanmean(variances):.4f}")

    # ------------------------------------------------------------------
    # 6. Resource classification by kriging variance
    # ------------------------------------------------------------------
    # Set thresholds based on the kriging variance distribution
    valid_kv = variances[~np.isnan(variances)]
    kv_thresholds = {
        "measured": float(np.percentile(valid_kv, 30)),
        "indicated": float(np.percentile(valid_kv, 70)),
    }
    classification = classify_by_kriging_variance(variances, kv_thresholds)
    n_meas = int(np.sum(classification == 1))
    n_ind = int(np.sum(classification == 2))
    n_inf = int(np.sum(classification == 3))
    print("\nClassification (by kriging variance percentiles):")
    print(f"  Measured threshold:  {kv_thresholds['measured']:.4f}")
    print(f"  Indicated threshold: {kv_thresholds['indicated']:.4f}")
    print(f"  Measured:  {n_meas} blocks")
    print(f"  Indicated: {n_ind} blocks")
    print(f"  Inferred:  {n_inf} blocks")

    # ------------------------------------------------------------------
    # 7. Grade-tonnage curve and resource statement
    # ------------------------------------------------------------------
    cutoffs = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    gt = block_grade_tonnage(bm, "au_gpt", "density", cutoffs)

    print("\nGrade-tonnage curve:")
    print(f"  {'COG':>6}  {'Tonnage':>12}  {'Mean Grade':>10}  {'Metal':>12}")
    for _, row in gt.iterrows():
        print(
            f"  {row['cutoff']:6.2f}  {row['tonnage']:12,.0f}  "
            f"{row['mean_grade']:10.3f}  {row['metal']:12,.1f}"
        )

    # Resource statement at 0.5 g/t cutoff
    block_vol = float(np.prod(bm.block_size))
    block_tonnes = np.full(bm.n_total, 2.7 * block_vol)
    stmt = resource_statement(block_tonnes, estimates, classification, cutoff=0.5)
    print("\nResource statement (COG = 0.5 g/t Au):")
    for cat in ["measured", "indicated", "inferred"]:
        s = stmt[cat]
        print(
            f"  {cat.capitalize():10s}: {s['tonnes']:>12,.0f} t  "
            f"@ {s['grade']:.3f} g/t  |  Metal: {s['metal']:,.1f}"
        )


if __name__ == "__main__":
    main()
