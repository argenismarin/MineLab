"""Tests for minelab.geomechanics.rock_mass_classification."""

import pytest

from minelab.geomechanics.rock_mass_classification import (
    gsi_from_chart,
    gsi_from_rmr,
    q_system,
    rmr_bieniawski,
    smr_romana,
)


class TestRMRBieniawski:
    """Tests for RMR89 classification."""

    def test_known_example(self):
        """Bieniawski 1989: ratings 12+17+15+20+10 - 5 = 69 → Class II."""
        result = rmr_bieniawski(12, 17, 15, 20, 10, -5)
        assert result["rmr"] == 69
        assert result["class_number"] == "II"
        assert result["description"] == "Good rock"

    def test_class_i(self):
        """RMR ≥ 81 → Class I Very good."""
        result = rmr_bieniawski(15, 20, 20, 30, 15, 0)
        assert result["class_number"] == "I"

    def test_class_v(self):
        """Very low ratings → Class V Very poor."""
        result = rmr_bieniawski(0, 3, 5, 0, 0, -10)
        assert result["class_number"] == "V"

    def test_no_adjustment(self):
        """Without orientation adjustment."""
        result = rmr_bieniawski(10, 15, 10, 15, 10)
        assert result["rmr"] == 60

    def test_invalid_ucs_rating(self):
        """UCS rating > 15 should raise."""
        with pytest.raises(ValueError):
            rmr_bieniawski(20, 17, 15, 20, 10)

    def test_invalid_orientation(self):
        """Positive orientation adjustment should raise."""
        with pytest.raises(ValueError):
            rmr_bieniawski(12, 17, 15, 20, 10, 5)


class TestQSystem:
    """Tests for Barton Q-system."""

    def test_known_example(self):
        """Q = (90/9)*(3/1)*(1/1) = 30 → Good."""
        result = q_system(90, 9, 3, 1, 1, 1)
        assert result["Q"] == 30.0
        assert result["description"] == "Good"

    def test_very_good(self):
        """High Q → Very good."""
        result = q_system(90, 6, 3, 1, 1, 0.5)
        assert result["Q"] == pytest.approx(90.0, rel=1e-3)
        assert result["description"] == "Very good"

    def test_poor(self):
        """Low Q → Poor."""
        result = q_system(25, 15, 1, 8, 0.5, 5)
        q = (25 / 15) * (1 / 8) * (0.5 / 5)
        assert result["Q"] == pytest.approx(q, rel=0.01)

    def test_zero_rqd(self):
        """RQD=0 should give Q=0."""
        result = q_system(0, 9, 3, 1, 1, 1)
        assert result["Q"] == 0.0

    def test_invalid_jn(self):
        """jn <= 0 should raise."""
        with pytest.raises(ValueError):
            q_system(90, 0, 3, 1, 1, 1)


class TestGSIFromRMR:
    """Tests for GSI from RMR conversion."""

    def test_known_conversion(self):
        """RMR=60 → GSI=55."""
        assert gsi_from_rmr(60) == 55.0

    def test_boundary(self):
        """RMR=24 → GSI=19."""
        assert gsi_from_rmr(24) == 19.0

    def test_low_rmr_raises(self):
        """RMR ≤ 23 should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 23"):
            gsi_from_rmr(23)


class TestGSIFromChart:
    """Tests for GSI from quantitative chart."""

    def test_blocky_good(self):
        """Blocky (65) + good surface (65) → GSI ≈ 65."""
        assert gsi_from_chart(65, 65) == 65.0

    def test_massive_very_good(self):
        """Massive + very good → high GSI."""
        gsi = gsi_from_chart(85, 85)
        assert gsi == 85.0

    def test_disintegrated_poor(self):
        """Disintegrated + poor → low GSI."""
        gsi = gsi_from_chart(10, 25)
        assert gsi == 17.5

    def test_clamp_to_100(self):
        """Max GSI should be 100."""
        gsi = gsi_from_chart(100, 100)
        assert gsi == 100.0

    def test_invalid_range(self):
        """Ratings out of 0-100 should raise."""
        with pytest.raises(ValueError):
            gsi_from_chart(110, 50)


class TestSMRRomana:
    """Tests for Slope Mass Rating."""

    def test_known_example(self):
        """SMR = RMR + (F1*F2*F3) + F4 = 65 + (0.7*0.8*(-25)) + 10 = 61."""
        result = smr_romana(65, 0.7, 0.8, -25, 10)
        expected = int(round(65 + (0.7 * 0.8 * (-25)) + 10))
        assert result["smr"] == expected
        assert result["class_number"] == "II"

    def test_class_ii(self):
        """SMR 61-80 → Class II Good."""
        result = smr_romana(60, 0.15, 0.15, 0, 5)
        assert result["class_number"] in ("I", "II")

    def test_unstable_slope(self):
        """Low SMR → unstable."""
        result = smr_romana(30, 1.0, 1.0, -50, 0)
        # SMR = 30 + (1.0*1.0*(-50)) + 0 = -20
        assert result["smr"] < 21

    def test_invalid_f1(self):
        """F1 out of range should raise."""
        with pytest.raises(ValueError):
            smr_romana(60, 0.1, 0.5, -10, 5)


# -------------------------------------------------------------------------
# Additional coverage tests
# -------------------------------------------------------------------------


class TestRMRBieniawskiClassIV:
    """Test RMR Class IV (Poor rock) classification."""

    def test_class_iv(self):
        """RMR 21-40 should classify as Class IV Poor rock."""
        # Ratings: 2+3+5+5+0 - 5 = 10... too low
        # Try: 4+5+8+10+3 = 30 → Class IV
        result = rmr_bieniawski(4, 5, 8, 10, 3, 0)
        assert result["rmr"] == 30
        assert result["class_number"] == "IV"
        assert result["description"] == "Poor rock"


class TestQSystemExtremeCategories:
    """Test Q-system extreme classification categories."""

    def test_exceptionally_good(self):
        """Q > 400 should be Exceptionally good."""
        # Q = (100/0.5) * (4/0.75) * (1/0.5) = 200 * 5.333 * 2 = 2133.3
        result = q_system(100, 0.5, 4, 0.75, 1, 0.5)
        assert result["Q"] > 400
        assert result["description"] == "Exceptionally good"

    def test_extremely_good(self):
        """Q between 100 and 400 should be Extremely good."""
        # Q = (90/3) * (3/1) * (1/0.5) = 30 * 3 * 2 = 180
        result = q_system(90, 3, 3, 1, 1, 0.5)
        assert 100 < result["Q"] <= 400
        assert result["description"] == "Extremely good"

    def test_fair(self):
        """Q between 4 and 10 should be Fair."""
        # Q = (50/9) * (2/2) * (0.5/0.5) = 5.556
        result = q_system(50, 9, 2, 2, 0.5, 0.5)
        assert 4 < result["Q"] <= 10
        assert result["description"] == "Fair"

    def test_very_poor(self):
        """Q between 0.1 and 1 should be Very poor."""
        # Q = (10/15) * (1/8) * (0.5/5) = 0.667 * 0.125 * 0.1 = 0.00833
        # Too low. Let's try: Q = (20/9) * (1/4) * (0.5/1) = 2.222 * 0.25 * 0.5 = 0.2778
        result = q_system(20, 9, 1, 4, 0.5, 1)
        assert 0.1 < result["Q"] <= 1
        assert result["description"] == "Very poor"

    def test_extremely_poor(self):
        """Q between 0.01 and 0.1 should be Extremely poor."""
        # Q = (10/15) * (0.5/8) * (0.05/5) = 0.667 * 0.0625 * 0.01 = 0.000417
        # Too low. Try: Q = (5/9) * (1/4) * (0.5/1) = 0.5556 * 0.25 * 0.5 = 0.0694
        result = q_system(5, 9, 1, 4, 0.5, 1)
        assert 0.01 < result["Q"] <= 0.1
        assert result["description"] == "Extremely poor"

    def test_exceptionally_poor(self):
        """Q <= 0.01 should be Exceptionally poor."""
        # Q = (1/20) * (0.5/20) * (0.05/400) = 0.05 * 0.025 * 0.000125 = 1.5625e-7
        result = q_system(1, 20, 0.5, 20, 0.05, 400)
        assert result["Q"] <= 0.01
        assert result["description"] == "Exceptionally poor"


class TestSMRRomanaAdditionalClasses:
    """Test SMR classifications for all classes."""

    def test_class_i_very_good(self):
        """SMR >= 81 should be Class I Very good, Completely stable."""
        # SMR = 80 + (0.15*0.15*0) + 15 = 95
        result = smr_romana(80, 0.15, 0.15, 0, 15)
        assert result["smr"] >= 81
        assert result["class_number"] == "I"
        assert result["stability"] == "Completely stable"

    def test_class_iii_normal(self):
        """SMR 41-60 should be Class III Normal, Partially stable."""
        # SMR = 50 + (0.15*0.15*(-5)) + 0 = 50 - 0.1125 + 0 ~ 50
        result = smr_romana(50, 0.15, 0.15, -5, 0)
        assert 41 <= result["smr"] <= 60
        assert result["class_number"] == "III"
        assert result["stability"] == "Partially stable"

    def test_class_iv_bad(self):
        """SMR 21-40 should be Class IV Bad, Unstable."""
        # SMR = 35 + (0.15*0.15*(-10)) + 0 = 35 - 0.225 ~ 35
        result = smr_romana(35, 0.15, 0.15, -10, 0)
        assert 21 <= result["smr"] <= 40
        assert result["class_number"] == "IV"
        assert result["stability"] == "Unstable"
