"""
Market calendar utilities — expiry day checks.

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
