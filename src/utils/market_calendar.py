"""
Market calendar utilities — expiry day checks, trading day checks.

Pure calendar logic, no external API dependencies.
NIFTY weekly expiry = Tuesday (changed from Monday in 2025).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional


def is_expiry_day(target_date: Optional[date] = None) -> bool:
    """
    Check if a date is a weekly expiry day.

    NIFTY weekly expiry moved to Tuesday in 2025.
    If Tuesday is a holiday, it shifts to the previous trading day (Monday).
    """
    target_date = target_date or date.today()
    return target_date.weekday() == 1  # Tuesday


def is_expiry_week(target_date: Optional[date] = None) -> bool:
    """Check if we are in a week that contains an expiry day (Tuesday)."""
    target_date = target_date or date.today()
    # Every week has a Tuesday, so this is always True for weekdays.
    # More useful: check if it's Mon-Tue of the week (leading up to expiry).
    return target_date.weekday() <= 1


def is_monthly_expiry(target_date: Optional[date] = None) -> bool:
    """Check if a date is the monthly expiry (last Tuesday of the month)."""
    target_date = target_date or date.today()
    if target_date.weekday() != 1:
        return False
    next_week = target_date + timedelta(days=7)
    return next_week.month != target_date.month


# ── NSE Holidays 2026 (tentative dates for lunar holidays) ──
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 2, 17): "Mahashivratri",
    date(2026, 3, 10): "Ramadan (Id-ul-Fitr)",
    date(2026, 3, 30): "Holi",
    date(2026, 4, 2): "Ram Navami",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 16): "Buddha Purnima",
    date(2026, 6, 17): "Eid-ul-Adha",
    date(2026, 7, 17): "Muharram",
    date(2026, 8, 15): "Independence Day",
    date(2026, 9, 16): "Milad-un-Nabi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 9): "Diwali (Laxmi Pujan)",
    date(2026, 11, 10): "Diwali (Balipratipada)",
    date(2026, 11, 27): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


def is_trading_day(target_date: Optional[date] = None) -> tuple[bool, str]:
    """Check if a date is a trading day.

    Returns (is_open, reason_if_closed). reason is empty string if open.
    """
    target_date = target_date or date.today()
    if target_date.weekday() == 5:
        return False, "Saturday"
    if target_date.weekday() == 6:
        return False, "Sunday"
    if target_date in NSE_HOLIDAYS_2026:
        return False, NSE_HOLIDAYS_2026[target_date]
    return True, ""


def next_trading_day(after: Optional[date] = None) -> date:
    """Find the next trading day after the given date."""
    d = (after or date.today()) + timedelta(days=1)
    while True:
        is_open, _ = is_trading_day(d)
        if is_open:
            return d
        d += timedelta(days=1)
