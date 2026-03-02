"""
Market calendar utilities — expiry day checks, trading day checks.

Holidays are fetched from Upstox Market Holidays API at startup,
with a hardcoded fallback for 2026 if the API is unavailable.

NIFTY weekly expiry = Tuesday (changed from Thursday in 2025).
If Tuesday is a holiday, expiry shifts to the previous trading day.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


# ── Hardcoded fallback: NSE Trading Holidays 2026 ──
# Used only when Upstox API is unavailable and no cached data exists.
_FALLBACK_HOLIDAYS_2026 = {
    date(2026, 1, 15): "Municipal Corporation Election (Maharashtra)",
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 3): "Holi",
    date(2026, 3, 26): "Shri Ram Navami",
    date(2026, 3, 31): "Shri Mahavir Jayanti",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 28): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali (Balipratipada)",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
    date(2026, 12, 25): "Christmas",
}

# Module-level cache — loaded once per process
_nse_holidays: dict[date, str] = {}
_holidays_loaded: bool = False

# Cache file path
_CACHE_DIR = Path("data")
_CACHE_FILE = _CACHE_DIR / "nse_holidays.json"


def _load_holidays_from_cache() -> dict[date, str]:
    """Load holidays from local JSON cache file."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(_CACHE_FILE.read_text())
        # Check if cache is for current year
        if data.get("year") != date.today().year:
            return {}
        holidays = {}
        for entry in data.get("holidays", []):
            d = date.fromisoformat(entry["date"])
            holidays[d] = entry["description"]
        return holidays
    except Exception:
        return {}


def _save_holidays_to_cache(holidays: dict[date, str]) -> None:
    """Save holidays to local JSON cache file."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "year": date.today().year,
            "fetched_at": datetime.now().isoformat(),
            "holidays": [
                {"date": d.isoformat(), "description": desc}
                for d, desc in sorted(holidays.items())
            ],
        }
        _CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.debug(f"Could not cache holidays: {e}")


def fetch_holidays_from_upstox(access_token: str) -> dict[date, str]:
    """
    Fetch NSE trading holidays from Upstox Market Holidays API.

    API: GET https://api.upstox.com/v2/market/holidays
    Returns: {date: description} for NSE trading holidays only.
    """
    import requests

    url = "https://api.upstox.com/v2/market/holidays"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "success":
            logger.warning(f"Upstox holidays API returned: {result.get('status')}")
            return {}

        holidays: dict[date, str] = {}
        for entry in result.get("data", []):
            # Only care about TRADING_HOLIDAY (not SETTLEMENT_HOLIDAY or SPECIAL_TIMING)
            if entry.get("holiday_type") != "TRADING_HOLIDAY":
                continue

            # Check if NSE is in closed_exchanges
            closed = entry.get("closed_exchanges", [])
            nse_closed = any(
                ex.get("exchange") == "NSE" for ex in closed
            ) if isinstance(closed, list) else False

            # If closed_exchanges is empty, it's a full market holiday
            if not closed or nse_closed:
                d = date.fromisoformat(entry["date"])
                holidays[d] = entry.get("description", "Market Holiday")

        if holidays:
            logger.info(f"Fetched {len(holidays)} NSE holidays from Upstox API")
            _save_holidays_to_cache(holidays)

        return holidays

    except Exception as e:
        logger.debug(f"Upstox holidays API failed: {e}")
        return {}


def load_holidays(access_token: Optional[str] = None) -> None:
    """
    Load NSE holidays. Called once at startup.

    Priority: Upstox API → local cache → hardcoded fallback.
    """
    global _nse_holidays, _holidays_loaded

    # 1. Try Upstox API (if token provided)
    if access_token:
        holidays = fetch_holidays_from_upstox(access_token)
        if holidays:
            _nse_holidays = holidays
            _holidays_loaded = True
            return

    # 2. Try local cache
    holidays = _load_holidays_from_cache()
    if holidays:
        logger.info(f"Loaded {len(holidays)} NSE holidays from cache")
        _nse_holidays = holidays
        _holidays_loaded = True
        return

    # 3. Fallback to hardcoded
    logger.info("Using hardcoded NSE holidays (no API/cache available)")
    _nse_holidays = dict(_FALLBACK_HOLIDAYS_2026)
    _holidays_loaded = True


def get_holidays() -> dict[date, str]:
    """Get the current holiday map. Auto-loads fallback if not initialized."""
    if not _holidays_loaded:
        load_holidays()
    return _nse_holidays


def is_trading_day(target_date: Optional[date] = None) -> tuple[bool, str]:
    """Check if a date is a trading day.

    Returns (is_open, reason_if_closed). reason is empty string if open.
    """
    target_date = target_date or date.today()
    if target_date.weekday() == 5:
        return False, "Saturday"
    if target_date.weekday() == 6:
        return False, "Sunday"
    holidays = get_holidays()
    if target_date in holidays:
        return False, holidays[target_date]
    return True, ""


def _prev_trading_day(d: date) -> date:
    """Find the previous trading day before the given date."""
    d = d - timedelta(days=1)
    while True:
        is_open, _ = is_trading_day(d)
        if is_open:
            return d
        d -= timedelta(days=1)


def is_expiry_day(target_date: Optional[date] = None) -> bool:
    """
    Check if a date is a NIFTY weekly expiry day.

    Normal expiry = Tuesday.
    If Tuesday is a holiday, expiry shifts to the previous trading day.
    e.g., Holi on Tue Mar 3 → expiry shifts to Mon Mar 2.
    """
    target_date = target_date or date.today()

    # If it IS Tuesday and it's a trading day → expiry
    if target_date.weekday() == 1:
        is_open, _ = is_trading_day(target_date)
        return is_open

    # If it's NOT Tuesday, check if this week's Tuesday is a holiday
    # and this date is the previous trading day before that Tuesday
    days_to_tuesday = (1 - target_date.weekday()) % 7
    if days_to_tuesday == 0:
        days_to_tuesday = 7  # Already past Tuesday, look at next week
    this_tuesday = target_date + timedelta(days=days_to_tuesday)

    # Only relevant if Tuesday is within the same week (Mon before Tue)
    if days_to_tuesday > 1:
        return False  # Wed/Thu/Fri — not close enough to be a shifted expiry

    # Tuesday is tomorrow (we're on Monday)
    is_tue_open, _ = is_trading_day(this_tuesday)
    if not is_tue_open:
        # Tuesday is a holiday — check if today is the previous trading day
        prev_td = _prev_trading_day(this_tuesday)
        return target_date == prev_td

    return False


def is_expiry_week(target_date: Optional[date] = None) -> bool:
    """Check if we are in a week leading up to expiry (Mon-Tue or shifted)."""
    target_date = target_date or date.today()
    # Check if today or any remaining day this week is expiry
    for offset in range(0, 3):  # Check today, tomorrow, day after
        check_date = target_date + timedelta(days=offset)
        if check_date.weekday() > 4:  # Skip weekend
            break
        if is_expiry_day(check_date):
            return True
    return False


def is_monthly_expiry(target_date: Optional[date] = None) -> bool:
    """Check if a date is the monthly expiry (last Tuesday of the month)."""
    target_date = target_date or date.today()
    if not is_expiry_day(target_date):
        return False
    # Find the actual Tuesday this expiry represents
    if target_date.weekday() == 1:
        tue = target_date
    else:
        # Shifted expiry — find the Tuesday it replaced
        days_to_tuesday = (1 - target_date.weekday()) % 7
        tue = target_date + timedelta(days=days_to_tuesday)
    next_week = tue + timedelta(days=7)
    return next_week.month != tue.month


def next_trading_day(after: Optional[date] = None) -> date:
    """Find the next trading day after the given date."""
    d = (after or date.today()) + timedelta(days=1)
    while True:
        is_open, _ = is_trading_day(d)
        if is_open:
            return d
        d += timedelta(days=1)
