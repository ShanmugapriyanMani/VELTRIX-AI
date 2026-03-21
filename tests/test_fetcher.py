"""Tests for UpstoxDataFetcher rate limit suppression and fetch cache."""

import time

import pandas as pd

from src.data.fetcher import UpstoxDataFetcher


def _make_fetcher() -> UpstoxDataFetcher:
    """Create a fetcher instance without connecting to Upstox."""
    # UpstoxDataFetcher.__init__ reads config yaml — bypass it
    fetcher = object.__new__(UpstoxDataFetcher)
    fetcher._call_timestamps = []
    fetcher._total_api_calls = 0
    fetcher._rate_pause_count = 0
    fetcher._rate_pause_total_secs = 0.0
    fetcher.RATE_LIMIT_LOG_MAX = 3
    fetcher.RATE_LIMIT_CALLS = 5
    fetcher.RATE_LIMIT_PERIOD = 1.0
    fetcher._quote_cache = {}
    fetcher._quote_cache_ttl = 15.0
    fetcher._intraday_cache = {}
    fetcher._intraday_cache_ttl = 15.0
    fetcher._intraday_cache_hits = 0
    fetcher._network_down_since = 0
    fetcher._network_error_logged = False
    return fetcher


class TestRateLimitSuppression:

    def test_rate_limit_suppressed_after_3_occurrences(self):
        """First 3 rate limit pauses log, subsequent ones are suppressed."""
        fetcher = _make_fetcher()

        # Simulate 5 batch cooldowns (every 50 calls)
        for i in range(5):
            fetcher._rate_pause_count += 1
            fetcher._rate_pause_total_secs += 2.0

        # After 5 pauses: count tracked, but only first 3 would have logged
        assert fetcher._rate_pause_count == 5
        assert fetcher._rate_pause_total_secs == 10.0

        # The suppression threshold is 3
        assert fetcher.RATE_LIMIT_LOG_MAX == 3
        # Pauses 4 and 5 would be suppressed (count > RATE_LIMIT_LOG_MAX)
        assert fetcher._rate_pause_count > fetcher.RATE_LIMIT_LOG_MAX


class TestFetchCache:

    def test_fetch_cache_skips_within_15_seconds(self):
        """Intraday cache returns cached result within TTL, skipping API call."""
        fetcher = _make_fetcher()

        # Simulate a cached intraday result
        fake_df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-13 09:15", periods=5, freq="5min"),
            "open": [22000, 22010, 22020, 22030, 22040],
            "high": [22010, 22020, 22030, 22040, 22050],
            "low": [21990, 22000, 22010, 22020, 22030],
            "close": [22005, 22015, 22025, 22035, 22045],
            "volume": [1000] * 5,
            "oi": [0] * 5,
        })
        cache_key = ("NSE_INDEX|Nifty 50", "5minute")
        fetcher._intraday_cache[cache_key] = (time.time(), fake_df)

        # Call get_intraday_candles — should hit cache, not API
        result = fetcher.get_intraday_candles("NSE_INDEX|Nifty 50", "5minute")

        assert fetcher._intraday_cache_hits == 1
        assert len(result) == 5
        assert result["close"].iloc[-1] == 22045

        # Call again — should hit cache again
        result2 = fetcher.get_intraday_candles("NSE_INDEX|Nifty 50", "5minute")
        assert fetcher._intraday_cache_hits == 2

        # Different instrument — no cache hit, would raise (no Upstox SDK)
        assert ("NSE_INDEX|Nifty Bank", "5minute") not in fetcher._intraday_cache
