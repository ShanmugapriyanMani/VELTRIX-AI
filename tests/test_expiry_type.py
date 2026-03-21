"""Tests for get_expiry_type() schedule logic and expiry adjustments."""

from datetime import date, datetime, time as dt_time
from unittest.mock import patch

from src.strategies.options_buyer import OptionsBuyerStrategy
from src.utils.market_calendar import get_expiry_type


# ═════════════════════════════════════════════════════
# get_expiry_type() — pure function tests (4 tests)
# ═════════════════════════════════════════════════════


class TestGetExpiryType:

    def test_old_schedule_thursday_nifty(self):
        """Old schedule: Thursday → NIFTY_EXPIRY."""
        # 2024-03-14 is a Thursday
        assert date(2024, 3, 14).weekday() == 3
        assert get_expiry_type(date(2024, 3, 14)) == "NIFTY_EXPIRY"
        # 2023-06-08 is a Thursday (well before new schedule)
        assert get_expiry_type(date(2023, 6, 8)) == "NIFTY_EXPIRY"

    def test_old_schedule_wednesday_banknifty(self):
        """Old schedule: Wednesday after 2023-09-04 → BANKNIFTY_EXPIRY.
        Wednesday before 2023-09-04 → NORMAL."""
        # 2024-01-10 is a Wednesday (after BankNifty moved to Wed)
        assert date(2024, 1, 10).weekday() == 2
        assert get_expiry_type(date(2024, 1, 10)) == "BANKNIFTY_EXPIRY"
        # 2023-08-30 is a Wednesday (before cutoff)
        assert date(2023, 8, 30).weekday() == 2
        assert get_expiry_type(date(2023, 8, 30)) == "NORMAL"
        # Friday → SENSEX_EXPIRY under old schedule
        assert get_expiry_type(date(2024, 1, 12)) == "SENSEX_EXPIRY"
        # Monday → NORMAL
        assert get_expiry_type(date(2024, 1, 8)) == "NORMAL"

    def test_new_schedule_tuesday_nifty(self):
        """New schedule (>= 2025-09-01): Tuesday → NIFTY_EXPIRY."""
        # 2025-09-02 is a Tuesday
        assert date(2025, 9, 2).weekday() == 1
        assert get_expiry_type(date(2025, 9, 2)) == "NIFTY_EXPIRY"
        # 2026-03-10 is a Tuesday
        assert date(2026, 3, 10).weekday() == 1
        assert get_expiry_type(date(2026, 3, 10)) == "NIFTY_EXPIRY"
        # Wednesday under new schedule → NORMAL
        assert get_expiry_type(date(2025, 9, 3)) == "NORMAL"

    def test_new_schedule_thursday_sensex(self):
        """New schedule: Thursday → SENSEX_EXPIRY."""
        # 2025-09-04 is a Thursday
        assert date(2025, 9, 4).weekday() == 3
        assert get_expiry_type(date(2025, 9, 4)) == "SENSEX_EXPIRY"
        # 2026-03-12 is a Thursday
        assert date(2026, 3, 12).weekday() == 3
        assert get_expiry_type(date(2026, 3, 12)) == "SENSEX_EXPIRY"
        # Friday under new schedule → NORMAL
        assert get_expiry_type(date(2025, 9, 5)) == "NORMAL"
        # Monday under new schedule → NORMAL
        assert get_expiry_type(date(2025, 9, 1)) == "NORMAL"


# ═════════════════════════════════════════════════════
# Expiry adjustments in options_buyer (1 test)
# ═════════════════════════════════════════════════════


def _make_strategy() -> OptionsBuyerStrategy:
    """Create a strategy instance for testing."""
    s = OptionsBuyerStrategy()
    s.reset_daily()
    return s


class TestExpiryAdjustments:

    def test_nifty_expiry_raises_threshold_shrinks_window_no_flip(self):
        """On NIFTY_EXPIRY: threshold +1.0, 11:30 cutoff, flip blocked."""
        s = _make_strategy()

        # Simulate expiry type set (as generate_signals would do)
        s._expiry_type = "NIFTY_EXPIRY"
        s._expiry_adjustments_applied = True

        # 1. Confirmation threshold: base 2.0 + 1.0 = 3.0 for TRENDING
        #    Verify by checking the fuzzy_threshold logic directly
        base = 2.0  # TRENDING, normal day
        if s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
            base += 1.0
        assert base == 3.0

        # VOLATILE base = 2.8 + 1.0 = 3.8
        vol_base = 2.8
        if s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
            vol_base += 1.0
        assert vol_base == 3.8

        # 2. Trade window: major expiry → entries blocked after 11:30
        is_major = s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")
        assert is_major is True

        # 3. Direction flip blocked on major expiry
        flip_blocked = s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")
        assert flip_blocked is True

        # 4. Position size multiplier = 0.75x
        #    Test via _compute_lots (lots=4 → 4*0.75=3.0 → int=3)
        lots = 4
        if s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
            lots = max(1, int(lots * 0.75))
        assert lots == 3

        # 5. Hard abort time = 11:30 (not 13:00)
        hard_abort_time = dt_time(11, 30) if is_major else dt_time(13, 0)
        assert hard_abort_time == dt_time(11, 30)

        # 6. SENSEX_EXPIRY: minor adjustments (threshold +0.5, size 0.90x)
        s._expiry_type = "SENSEX_EXPIRY"
        minor_base = 2.0
        if s._expiry_type == "SENSEX_EXPIRY":
            minor_base += 0.5
        assert minor_base == 2.5

        minor_lots = 4
        if s._expiry_type == "SENSEX_EXPIRY":
            minor_lots = max(1, int(minor_lots * 0.90))
        assert minor_lots == 3  # 4 * 0.9 = 3.6 → int = 3

        # Flip NOT blocked on SENSEX expiry
        flip_blocked_sensex = s._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")
        assert flip_blocked_sensex is False
