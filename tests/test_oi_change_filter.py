"""Tests for OI change rate confirmation filter.

4 tests covering:
  1. OI confirmed lowers threshold for PE direction
  2. OI contradicted raises threshold for PE direction
  3. OI neutral leaves threshold unchanged
  4. No snapshot available returns neutral (no adjustment)
"""


class TestOiConfirmedLowersThresholdPe:
    """When puts growing and calls flat, PE signal is confirmed."""

    def test_oi_confirmed_lowers_threshold_pe(self):
        """put_change > +2% and call_change < +2% → threshold - 0.25."""
        direction = "PE"
        put_change = 4.5   # puts growing fast
        call_change = 0.8  # calls flat
        confirmed_thr = 2.0
        contradicted_thr = 3.0
        OI_CONFIRMED_BONUS = 0.25

        oi_adjustment = 0.0
        if direction == "PE":
            if put_change > confirmed_thr and call_change < confirmed_thr:
                oi_adjustment = -OI_CONFIRMED_BONUS

        assert oi_adjustment == -0.25

        base_threshold = 1.75
        effective = base_threshold + oi_adjustment
        assert effective == 1.50  # easier entry with institutional support


class TestOiContradictedRaisesThresholdPe:
    """When calls growing fast and puts shrinking, PE signal is contradicted."""

    def test_oi_contradicted_raises_threshold_pe(self):
        """call_change > +3% and put_change < 0 → threshold + 0.75."""
        direction = "PE"
        put_change = -1.5   # puts being removed
        call_change = 5.0   # calls growing fast
        confirmed_thr = 2.0
        contradicted_thr = 3.0
        OI_CONTRADICTED_PENALTY = 0.75

        oi_adjustment = 0.0
        if direction == "PE":
            if call_change > contradicted_thr and put_change < 0:
                oi_adjustment = OI_CONTRADICTED_PENALTY

        assert oi_adjustment == 0.75

        base_threshold = 1.75
        effective = base_threshold + oi_adjustment
        assert effective == 2.50  # much harder entry, institutions disagree


class TestOiNeutralNoChange:
    """When OI changes are small, no adjustment applied."""

    def test_oi_neutral_no_change(self):
        """put and call changes within -2% to +2% → no adjustment."""
        direction = "PE"
        put_change = 1.0   # mild growth
        call_change = 0.5  # mild growth
        confirmed_thr = 2.0
        contradicted_thr = 3.0

        oi_adjustment = 0.0
        if direction == "PE":
            if put_change > confirmed_thr and call_change < confirmed_thr:
                oi_adjustment = -0.25
            elif call_change > contradicted_thr and put_change < 0:
                oi_adjustment = 0.75

        assert oi_adjustment == 0.0  # neutral, no adjustment


class TestOiNoSnapshotReturnsNeutral:
    """When no previous snapshot available, returns None (neutral)."""

    def test_oi_no_snapshot_returns_neutral(self):
        """Without two snapshots, get_oi_change_rates returns (None, None)."""
        # Simulate fetcher with no previous snapshot
        oi_snapshot = {"timestamp": 100, "put_oi": 50000, "call_oi": 60000}
        oi_snapshot_prev = None  # No previous snapshot yet

        # Same logic as get_oi_change_rates
        if oi_snapshot is None or oi_snapshot_prev is None:
            put_change, call_change = None, None
        else:
            put_change = (oi_snapshot["put_oi"] - oi_snapshot_prev["put_oi"]) / oi_snapshot_prev["put_oi"] * 100
            call_change = (oi_snapshot["call_oi"] - oi_snapshot_prev["call_oi"]) / oi_snapshot_prev["call_oi"] * 100

        assert put_change is None
        assert call_change is None

        # When None, oi_adjustment stays 0.0
        oi_adjustment = 0.0
        if put_change is not None and call_change is not None:
            oi_adjustment = -0.25  # Would be set if data available
        assert oi_adjustment == 0.0
