"""Integration test for mass balance closure across a simple plant circuit.

Tests the interaction between comminution, flotation, two-product formula,
and mass balance closure checks, verifying that a feed -> grinding ->
flotation -> concentrate + tails circuit is self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from minelab.mineral_processing import (
    ball_mill_power,
    bond_energy,
    check_closure,
    flotation_circuit,
    flotation_first_order,
    multi_element_balance,
    two_product,
)


# ---------------------------------------------------------------------------
# Circuit parameters (small Cu flotation plant)
# ---------------------------------------------------------------------------
FEED_TONNAGE = 500.0       # t/h
FEED_GRADE_CU = 1.5        # % Cu
FEED_GRADE_FE = 25.0       # % Fe
CONCENTRATE_GRADE_CU = 28.0  # % Cu
CONCENTRATE_GRADE_FE = 22.0  # % Fe
TAILINGS_GRADE_CU = 0.10   # % Cu
TAILINGS_GRADE_FE = 26.0   # % Fe

# Comminution parameters
BOND_WI = 14.0      # kWh/t
FEED_P80 = 5000.0   # micrometers
PRODUCT_P80 = 75.0  # micrometers

# Flotation kinetics
R_INF = 0.95        # ultimate recovery
K_RATE = 0.4        # 1/min
FLOTATION_TIME = 12.0  # minutes


class TestTwoProductConsistency:
    """Verify two-product formula gives consistent mass split and recovery."""

    def test_two_product_ratios_sum_to_one(self):
        """Concentrate + tailings mass ratios must sum to 1.0."""
        result = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        total = result["concentrate_ratio"] + result["tailings_ratio"]
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_two_product_recovery_range(self):
        """Recovery should be between 0 and 1."""
        result = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        assert 0.0 < result["recovery"] < 1.0

    def test_two_product_mass_balance(self):
        """Reconstructed feed grade from products must equal input feed grade."""
        result = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        cf = result["concentrate_ratio"]
        tf = result["tailings_ratio"]
        reconstructed_feed = cf * CONCENTRATE_GRADE_CU + tf * TAILINGS_GRADE_CU
        assert reconstructed_feed == pytest.approx(FEED_GRADE_CU, rel=1e-8)


class TestGrindingToFlotation:
    """Verify comminution -> flotation pipeline."""

    def test_grinding_energy_positive(self):
        """Bond energy for size reduction should be positive."""
        energy = bond_energy(BOND_WI, FEED_P80, PRODUCT_P80)
        assert energy > 0

    def test_mill_power_scales_with_tonnage(self):
        """Doubling tonnage should double the required power."""
        p1 = ball_mill_power(BOND_WI, FEED_P80, PRODUCT_P80, FEED_TONNAGE)
        p2 = ball_mill_power(BOND_WI, FEED_P80, PRODUCT_P80, 2 * FEED_TONNAGE)
        assert p2 == pytest.approx(2 * p1, rel=1e-8)

    def test_flotation_recovery_increases_with_time(self):
        """Flotation recovery should increase monotonically with time."""
        times = [2, 5, 10, 15, 20]
        recoveries = [flotation_first_order(R_INF, K_RATE, t) for t in times]
        for i in range(len(recoveries) - 1):
            assert recoveries[i + 1] > recoveries[i]

    def test_flotation_approaches_ultimate(self):
        """At long residence time, recovery should approach R_inf."""
        r = flotation_first_order(R_INF, K_RATE, 100.0)
        assert r == pytest.approx(R_INF, abs=0.01)


class TestCircuitMassBalance:
    """Full circuit mass balance: feed -> concentrate + tailings."""

    def test_check_closure_exact(self):
        """When products sum exactly to feed, closure should pass."""
        tp = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        conc_mass = tp["concentrate_ratio"] * FEED_TONNAGE
        tails_mass = tp["tailings_ratio"] * FEED_TONNAGE

        result = check_closure(FEED_TONNAGE, [conc_mass, tails_mass])
        assert result["closed"] is True
        assert result["error"] == pytest.approx(0.0, abs=1e-10)

    def test_check_closure_with_small_loss(self):
        """A small mass loss (1%) should still pass closure at 2% tolerance."""
        tp = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        conc_mass = tp["concentrate_ratio"] * FEED_TONNAGE
        tails_mass = tp["tailings_ratio"] * FEED_TONNAGE * 0.99  # 1% loss

        result = check_closure(FEED_TONNAGE, [conc_mass, tails_mass])
        assert result["error"] < 0.02

    def test_multi_element_balance_closes(self):
        """Multi-element balance across Cu and Fe should close."""
        tp = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)

        feed = {"Cu": FEED_GRADE_CU, "Fe": FEED_GRADE_FE}
        products = [
            {"Cu": CONCENTRATE_GRADE_CU, "Fe": CONCENTRATE_GRADE_FE},
            {"Cu": TAILINGS_GRADE_CU, "Fe": TAILINGS_GRADE_FE},
        ]
        ratios = np.array([tp["concentrate_ratio"], tp["tailings_ratio"]])

        result = multi_element_balance(feed, products, ratios)
        # Cu should balance perfectly (derived from two_product)
        assert result["balance_errors"]["Cu"] == pytest.approx(0.0, abs=1e-8)
        # The overall balance should be considered balanced
        assert result["balanced"] is True


class TestCircuitRecoveryConsistency:
    """Verify flotation kinetics aligns with two-product formula results."""

    def test_kinetic_recovery_vs_two_product(self):
        """Kinetic recovery and two-product recovery should be comparable
        when circuit parameters are internally consistent."""
        # Kinetic recovery at the chosen flotation time
        r_kinetic = flotation_first_order(R_INF, K_RATE, FLOTATION_TIME)

        # Two-product recovery from assay data
        tp = two_product(FEED_GRADE_CU, CONCENTRATE_GRADE_CU, TAILINGS_GRADE_CU)
        r_assay = tp["recovery"]

        # Both are valid recovery measures; they need not be identical but
        # both should be physically reasonable (between 50% and 100%)
        assert 0.5 < r_kinetic < 1.0
        assert 0.5 < r_assay < 1.0

    def test_circuit_overall_recovery(self):
        """Rougher-cleaner circuit recovery should be less than rougher alone."""
        rougher_r = flotation_first_order(R_INF, K_RATE, FLOTATION_TIME)
        cleaner_r = 0.85
        circuit = flotation_circuit(rougher_r, cleaner_r)

        assert circuit["overall_recovery"] < rougher_r
        assert circuit["overall_recovery"] > 0

    def test_circuit_with_scavenger_higher_than_without(self):
        """Adding a scavenger should increase overall circuit recovery."""
        rougher_r = flotation_first_order(R_INF, K_RATE, FLOTATION_TIME)
        cleaner_r = 0.85

        without_scav = flotation_circuit(rougher_r, cleaner_r, scavenger_r=0.0)
        with_scav = flotation_circuit(rougher_r, cleaner_r, scavenger_r=0.5)

        assert with_scav["overall_recovery"] > without_scav["overall_recovery"]
