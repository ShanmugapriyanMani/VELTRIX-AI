"""
Candle Backfill — Download 5-min NIFTY candles from Upstox V3 Historical API.

Upstox V3 limits: 1 month per request for 5-min candles.
Strategy: month-by-month iteration, skip months already in DB.

Usage: python src/main.py --mode ml_backfill [--from-date 2022-03-01] [--to-date 2026-03-12]
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.store import DataStore


class CandleBackfiller:
    """Download and store 5-min NIFTY candles."""

    NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
    NIFTY_SYMBOL = "NIFTY50"
    DEFAULT_START = "2022-03-01"  # 4 years back

    def __init__(self, store: DataStore, fetcher):
        self.store = store
        self.fetcher = fetcher

    def backfill(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict:
        """
        Month-by-month backfill of 5-min candles.

        Returns: {months_fetched, total_candles, skipped_months, errors}
        """
        start = date.fromisoformat(from_date) if from_date else date.fromisoformat(self.DEFAULT_START)
        end = date.fromisoformat(to_date) if to_date else date.today()

        month_ranges = self._get_month_ranges(start, end)
        logger.info(f"ML BACKFILL: {len(month_ranges)} months from {start} to {end}")

        stats = {"months_fetched": 0, "total_candles": 0, "skipped_months": 0, "errors": []}

        for i, (m_start, m_end) in enumerate(month_ranges):
            if self._is_month_complete(m_start, m_end):
                stats["skipped_months"] += 1
                continue

            # For current month with partial data, fetch only missing days
            fetch_start = m_start
            end_dt = date.fromisoformat(m_end)
            is_current_month = (end_dt.year == date.today().year and end_dt.month == date.today().month)
            if is_current_month:
                last_candle = self.store.get_ml_candle_coverage(self.NIFTY_SYMBOL)
                last_date_str = last_candle.get("to_date", "")
                if last_date_str:
                    try:
                        last_dt = date.fromisoformat(last_date_str[:10])
                        # Start from next day after last candle
                        incremental_start = (last_dt + timedelta(days=1)).isoformat()
                        if incremental_start > m_start:
                            fetch_start = incremental_start
                            logger.info(
                                f"  Current month incremental: {fetch_start} to {m_end} "
                                f"(last candle: {last_date_str[:10]})"
                            )
                    except (ValueError, TypeError):
                        pass

            try:
                df = self._fetch_month(fetch_start, m_end)
                if not df.empty:
                    saved = self.store.save_ml_candles(
                        self.NIFTY_SYMBOL, self.NIFTY_INSTRUMENT_KEY, df,
                    )
                    stats["months_fetched"] += 1
                    stats["total_candles"] += saved
                    logger.info(
                        f"  [{i + 1}/{len(month_ranges)}] {m_start} to {m_end}: {saved} candles"
                    )
                else:
                    stats["skipped_months"] += 1
                    logger.debug(f"  [{i + 1}/{len(month_ranges)}] {m_start} to {m_end}: no data")

                # Rate limit between months
                if i < len(month_ranges) - 1:
                    time.sleep(0.5)

            except Exception as e:
                stats["errors"].append(f"{m_start}: {e}")
                logger.warning(f"  [{i + 1}/{len(month_ranges)}] {m_start} error: {e}")

        logger.info(
            f"ML BACKFILL complete: {stats['months_fetched']} months, "
            f"{stats['total_candles']} candles, {stats['skipped_months']} skipped, "
            f"{len(stats['errors'])} errors"
        )
        return stats

    def _get_month_ranges(self, start: date, end: date) -> list[tuple[str, str]]:
        """Generate list of (month_start, month_end) date string pairs."""
        ranges = []
        current = start.replace(day=1)
        while current <= end:
            month_start = current
            # Last day of month
            if current.month == 12:
                month_end = date(current.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)

            # Clip to actual range
            month_start = max(month_start, start)
            month_end = min(month_end, end)

            ranges.append((month_start.isoformat(), month_end.isoformat()))

            # Advance to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return ranges

    def _fetch_month(self, from_dt: str, to_dt: str) -> pd.DataFrame:
        """
        Fetch one month of 5-min candles via fetcher._fetch_upstox_historical().

        Calls the low-level API method directly to avoid daily cache logic.
        """
        return self.fetcher._fetch_upstox_historical(
            self.NIFTY_INSTRUMENT_KEY, "5minute", from_dt, to_dt,
        )

    def _is_month_complete(self, month_start: str, month_end: str) -> bool:
        """Check if ml_candles_5min already has data for this month range.

        For the current month: only skip if last candle is within 1 day of end date.
        For past months: skip if any data exists (month is closed).
        """
        end_dt = date.fromisoformat(month_end)
        is_current_month = (end_dt.year == date.today().year and end_dt.month == date.today().month)

        df = self.store.get_ml_candles(
            self.NIFTY_SYMBOL,
            from_date=month_start,
            to_date=month_end + "T23:59:59",
            limit=1,
        )
        if df.empty:
            return False

        if not is_current_month:
            return True  # Past month with data — complete

        # Current month: check if last candle is recent enough
        last_candle = self.store.get_ml_candle_coverage(self.NIFTY_SYMBOL)
        last_date_str = last_candle.get("to_date", "")
        if not last_date_str:
            return False
        try:
            last_dt = date.fromisoformat(last_date_str[:10])
        except (ValueError, TypeError):
            return False
        # Stale if last candle is more than 1 day behind end date
        return (end_dt - last_dt).days <= 1

    def get_coverage_report(self) -> dict:
        """Return coverage stats."""
        return self.store.get_ml_candle_coverage(self.NIFTY_SYMBOL)
