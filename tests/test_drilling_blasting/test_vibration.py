"""Tests for minelab.drilling_blasting.vibration."""

import pytest

from minelab.drilling_blasting.vibration import (
    ppv_scaled_distance,
    usbm_scaled_distance,
    vibration_compliance,
)


class TestPPVScaledDistance:
    """Tests for PPV prediction."""

    def test_positive_ppv(self):
        """PPV should be positive."""
        result = ppv_scaled_distance(1140, 50, 100, 1.6)
        assert result["ppv"] > 0

    def test_farther_lower_ppv(self):
        """Greater distance → lower PPV."""
        r_near = ppv_scaled_distance(1140, 50, 50, 1.6)
        r_far = ppv_scaled_distance(1140, 50, 200, 1.6)
        assert r_far["ppv"] < r_near["ppv"]


class TestUSBMScaledDistance:
    """Tests for USBM scaled distance."""

    def test_known_value(self):
        """D=100, W=50 → SD = 14.14."""
        sd = usbm_scaled_distance(100, 50)
        assert sd == pytest.approx(14.14, rel=0.01)

    def test_positive(self):
        """SD should be positive."""
        sd = usbm_scaled_distance(50, 25)
        assert sd > 0


class TestVibrationCompliance:
    """Tests for vibration compliance check."""

    def test_compliant(self):
        """Low PPV → compliant."""
        result = vibration_compliance(10.0)
        assert result["compliant"]

    def test_non_compliant(self):
        """High PPV → not compliant."""
        result = vibration_compliance(30.0)
        assert not result["compliant"]


class TestVibrationComplianceDIN4150:
    """Tests for DIN 4150-3 standard compliance."""

    def test_din4150_frequency_zero_unknown(self):
        """DIN4150 with unknown frequency (0) uses most conservative 5 mm/s."""
        result = vibration_compliance(4.0, frequency=0.0, standard="DIN4150")
        assert result["compliant"]
        assert result["limit"] == pytest.approx(5.0)
        assert result["standard"] == "DIN4150"

    def test_din4150_frequency_below_10(self):
        """DIN4150 with frequency < 10 Hz uses limit of 5 mm/s."""
        result = vibration_compliance(6.0, frequency=5.0, standard="DIN4150")
        assert not result["compliant"]
        assert result["limit"] == pytest.approx(5.0)

    def test_din4150_frequency_10_to_50_interpolation(self):
        """DIN4150 linearly interpolates 5-15 mm/s in 10-50 Hz range."""
        # At 30 Hz: limit = 5 + (30-10)*(15-5)/(50-10) = 5 + 20*10/40 = 10
        result = vibration_compliance(8.0, frequency=30.0, standard="DIN4150")
        assert result["compliant"]
        assert result["limit"] == pytest.approx(10.0)

    def test_din4150_frequency_50_to_100_interpolation(self):
        """DIN4150 linearly interpolates 15-20 mm/s in 50-100 Hz range."""
        # At 75 Hz: limit = 15 + (75-50)*(20-15)/(100-50) = 15 + 25*5/50 = 17.5
        result = vibration_compliance(16.0, frequency=75.0, standard="DIN4150")
        assert result["compliant"]
        assert result["limit"] == pytest.approx(17.5)

    def test_din4150_frequency_above_100(self):
        """DIN4150 with frequency > 100 Hz uses limit of 20 mm/s."""
        result = vibration_compliance(19.0, frequency=120.0, standard="DIN4150")
        assert result["compliant"]
        assert result["limit"] == pytest.approx(20.0)

    def test_din4150_non_compliant_high_ppv(self):
        """DIN4150 non-compliant when PPV exceeds limit."""
        result = vibration_compliance(25.0, frequency=120.0, standard="DIN4150")
        assert not result["compliant"]


class TestVibrationComplianceUnknownStandard:
    """Tests for unknown standard error handling."""

    def test_unknown_standard_raises(self):
        """Unknown standard should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown standard"):
            vibration_compliance(10.0, standard="ISO9999")
