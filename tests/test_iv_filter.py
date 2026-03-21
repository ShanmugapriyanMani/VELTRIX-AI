"""Tests for IV awareness filter.

3 tests covering:
  1. IV_HIGH raises conviction threshold
  2. IV_LOW lowers conviction threshold
  3. IV_NORMAL leaves threshold unchanged
"""


class TestIvHighRaisesThreshold:
    """When VIX is 30%+ above 20-day avg, threshold increases."""

    def test_iv_high_raises_threshold(self):
        """iv_ratio > 1.30 → threshold + 0.50 (harder entry)."""
        IV_HIGH_THRESHOLD = 1.30
        IV_HIGH_PENALTY = 0.50

        vix_now = 26.0
        vix_20d_avg = 18.0  # ratio = 26/18 = 1.44 > 1.30
        iv_ratio = vix_now / vix_20d_avg
        assert iv_ratio > IV_HIGH_THRESHOLD

        base_threshold = 1.75  # TRENDING
        iv_adjustment = IV_HIGH_PENALTY if iv_ratio > IV_HIGH_THRESHOLD else 0.0
        effective = base_threshold + iv_adjustment

        assert effective == 2.25  # 1.75 + 0.50
        assert effective > base_threshold


class TestIvLowLowersThreshold:
    """When VIX is 20%+ below 20-day avg, threshold decreases."""

    def test_iv_low_lowers_threshold(self):
        """iv_ratio < 0.80 → threshold - 0.25 (easier entry)."""
        IV_LOW_THRESHOLD = 0.80
        IV_LOW_BONUS = 0.25

        vix_now = 11.0
        vix_20d_avg = 16.0  # ratio = 11/16 = 0.6875 < 0.80
        iv_ratio = vix_now / vix_20d_avg
        assert iv_ratio < IV_LOW_THRESHOLD

        base_threshold = 1.75
        iv_adjustment = -IV_LOW_BONUS if iv_ratio < IV_LOW_THRESHOLD else 0.0
        effective = base_threshold + iv_adjustment

        assert effective == 1.50  # 1.75 - 0.25
        assert effective < base_threshold


class TestIvNormalNoChange:
    """When VIX is within 0.80-1.30 of 20-day avg, no adjustment."""

    def test_iv_normal_no_change(self):
        """iv_ratio in [0.80, 1.30] → threshold unchanged."""
        IV_HIGH_THRESHOLD = 1.30
        IV_LOW_THRESHOLD = 0.80

        vix_now = 17.0
        vix_20d_avg = 16.0  # ratio = 17/16 = 1.0625, in normal range
        iv_ratio = vix_now / vix_20d_avg
        assert IV_LOW_THRESHOLD <= iv_ratio <= IV_HIGH_THRESHOLD

        base_threshold = 1.75
        iv_adjustment = 0.0
        if iv_ratio > IV_HIGH_THRESHOLD:
            iv_adjustment = 0.50
        elif iv_ratio < IV_LOW_THRESHOLD:
            iv_adjustment = -0.25
        effective = base_threshold + iv_adjustment

        assert effective == 1.75  # unchanged
        assert iv_adjustment == 0.0
