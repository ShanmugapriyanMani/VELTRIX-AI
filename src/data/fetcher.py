"""
Upstox Data Fetcher — Handles OAuth2 auth, historical + intraday data, live streaming.
Uses upstox-python-sdk for all broker interactions (LIVE mode only).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import yaml
import requests
from loguru import logger

try:
    import upstox_client

    UPSTOX_AVAILABLE = True
except ImportError:
    UPSTOX_AVAILABLE = False
    logger.warning("upstox-python-sdk not installed.")


@dataclass
class UpstoxAuth:
    """Manages Upstox OAuth2 authentication lifecycle (LIVE mode only)."""

    api_key: str
    api_secret: str
    redirect_uri: str
    access_token_path: str
    api_base_url: str = "https://api.upstox.com"
    api_hft_url: str = "https://api-hft.upstox.com"
    _access_token: Optional[str] = field(default=None, repr=False)
    _token_expiry: Optional[datetime] = field(default=None, repr=False)

    def get_login_url(self) -> str:
        """Generate the OAuth2 login URL for user authorization."""
        base = f"{self.api_base_url}/v2/login/authorization/dialog"
        return (
            f"{base}?response_type=code&client_id={self.api_key}"
            f"&redirect_uri={self.redirect_uri}"
        )

    def exchange_code_for_token(self, auth_code: str) -> str:
        """Exchange authorization code for access token."""
        url = f"{self.api_base_url}/v2/login/authorization/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "code": auth_code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        self._access_token = result["access_token"]
        self._token_expiry = datetime.now() + timedelta(hours=23)
        self._save_token()
        logger.info("Upstox access token obtained successfully")
        return self._access_token

    def load_token(self) -> Optional[str]:
        """Load saved access token from disk."""
        token_path = Path(self.access_token_path)
        if not token_path.exists():
            logger.warning(f"No saved token at {token_path}")
            return None

        with open(token_path, "r") as f:
            data = json.load(f)

        expiry_str = data.get("expiry", "")
        if expiry_str:
            expiry = datetime.fromisoformat(expiry_str)
            if datetime.now() > expiry:
                logger.warning("Saved token is expired")
                return None
            self._token_expiry = expiry
        else:
            # LIVE tokens expire daily
            saved_date = data.get("date", "")
            if saved_date != date.today().isoformat():
                logger.warning("Saved token is expired (from a different day)")
                return None

        self._access_token = data["access_token"]
        logger.info("Loaded saved Upstox access token")
        return self._access_token

    def _save_token(self) -> None:
        """Persist access token to disk."""
        token_path = Path(self.access_token_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": self._access_token,
            "date": date.today().isoformat(),
            "expiry": self._token_expiry.isoformat() if self._token_expiry else "",
        }
        with open(token_path, "w") as f:
            json.dump(data, f)
        logger.debug(f"Token saved to {token_path}")

    @property
    def access_token(self) -> Optional[str]:
        if self._access_token is None:
            self.load_token()
        return self._access_token

    @property
    def is_valid(self) -> bool:
        if self._access_token is None:
            return False
        if self._token_expiry and datetime.now() > self._token_expiry:
            return False
        return True

    def get_configuration(self) -> "upstox_client.Configuration":
        """Get configured upstox_client.Configuration object."""
        if not UPSTOX_AVAILABLE:
            raise ImportError("upstox-python-sdk not installed")

        config = upstox_client.Configuration(sandbox=False)
        config.access_token = self.access_token
        return config

    def get_hft_configuration(self) -> "upstox_client.Configuration":
        """Get HFT-specific configuration for low-latency order execution."""
        if not UPSTOX_AVAILABLE:
            raise ImportError("upstox-python-sdk not installed")

        config = upstox_client.Configuration(sandbox=False)
        config.access_token = self.access_token
        if self.api_hft_url:
            config.host = self.api_hft_url
        return config


def build_auth_from_config(config: dict) -> UpstoxAuth:
    """Build UpstoxAuth from config + env vars (LIVE mode only)."""
    upstox_cfg = config["upstox"]
    mode_cfg = upstox_cfg.get("live", upstox_cfg)

    api_key = os.getenv("UPSTOX_LIVE_API_KEY", mode_cfg.get("api_key", ""))
    api_secret = os.getenv("UPSTOX_LIVE_API_SECRET", mode_cfg.get("api_secret", ""))
    redirect_uri = os.getenv("UPSTOX_LIVE_REDIRECT_URI", mode_cfg.get("redirect_uri", "http://127.0.0.1:5000/callback"))
    api_base_url = mode_cfg.get("api_base_url", "https://api.upstox.com")
    api_hft_url = mode_cfg.get("api_hft_url", "https://api-hft.upstox.com")

    logger.info(f"Upstox LIVE | Base: {api_base_url}")

    return UpstoxAuth(
        api_key=api_key,
        api_secret=api_secret,
        redirect_uri=redirect_uri,
        access_token_path=upstox_cfg.get("access_token_path", "config/.access_token"),
        api_base_url=api_base_url,
        api_hft_url=api_hft_url,
    )


class UpstoxDataFetcher:
    """
    Primary data fetcher using Upstox API V3 (LIVE mode only).
    Handles historical candles, intraday candles, live quotes, WebSocket streaming.
    """

    # Rate limiting: conservative 5 requests/sec to avoid 429s
    RATE_LIMIT_CALLS = 5
    RATE_LIMIT_PERIOD = 1.0  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0

    # Network health tracking
    NETWORK_COOLDOWN = 60  # seconds to wait before retrying after network failure
    _NETWORK_ERROR_PATTERNS = (
        "Failed to resolve",
        "Name or service not known",
        "nodename nor servname provided",
        "getaddrinfo failed",
        "ConnectionRefusedError",
        "Connection refused",
        "Network is unreachable",
        "No route to host",
        "Connection reset by peer",
        "ConnectionResetError",
        "RemoteDisconnected",
        "Connection aborted",
        "SSLError",
        "MaxRetryError",
        "NewConnectionError",
    )

    def __init__(self, config_path: str = "config/config.yaml", store: Optional[Any] = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.auth = build_auth_from_config(self.config)

        self.universe = self.config.get("universe", {})
        self._api_client: Optional[upstox_client.ApiClient] = None
        self._call_timestamps: list[float] = []
        self._streamer: Optional[Any] = None
        self._portfolio_streamer: Optional[Any] = None
        self._store = store  # Optional DataStore for DB-first caching

        # Network health state
        self._network_down_since: float = 0  # timestamp when network failure detected
        self._network_error_logged: bool = False  # avoid log spam

        # API call counter for batch cooldown
        self._total_api_calls: int = 0
        self._rate_pause_count: int = 0
        self._rate_pause_total_secs: float = 0.0

        # Rate limit log suppression: log first 3, suppress rest, EOD summary
        self.RATE_LIMIT_LOG_MAX = 3

        # Quote cache: {instrument_key: (timestamp, quote_dict)} — 15s TTL
        self._quote_cache: dict[str, tuple[float, dict]] = {}
        self._quote_cache_ttl: float = 15.0

        # PCR fallback cache: use last known good PCR when chain is empty
        self._last_known_pcr: float = 1.0  # neutral default

        # Intraday candle cache: {(instrument_key, interval): (timestamp, DataFrame)} — 15s TTL
        self._intraday_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._intraday_cache_ttl: float = 15.0
        self._intraday_cache_hits: int = 0

        # WebSocket LTP cache (thread-safe, updated by background WS thread)
        self._ws_ltp_cache: dict[str, float] = {}
        self._ws_ltp_lock = threading.Lock()
        self._ws_connected: bool = False

        # OI snapshot tracking for change rate filter
        self._oi_snapshot: Optional[dict] = None       # Current snapshot
        self._oi_snapshot_prev: Optional[dict] = None  # Previous snapshot (30 min ago)

        # Option chain failure counter per instrument (log noise reduction)
        self._oc_fail_count: dict[str, int] = {}

    def _get_api_client(self) -> "upstox_client.ApiClient":
        """Get or create authenticated API client (with default timeout)."""
        if self._api_client is None:
            configuration = self.auth.get_configuration()
            self._api_client = upstox_client.ApiClient(configuration)
            # Inject default timeout to prevent indefinite hangs
            from src.execution.upstox_broker import _inject_api_timeout
            from src.config.env_loader import get_config
            _inject_api_timeout(self._api_client, get_config().API_TIMEOUT_SECONDS)
        return self._api_client

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API calls.

        - Max 5 calls/sec (sliding window)
        - Min 0.2s between every call
        - 2s pause every 50 calls (batch cooldown)
        - Log first 3 pauses, suppress rest, EOD summary
        """
        # Min gap between calls
        if self._call_timestamps:
            elapsed = time.time() - self._call_timestamps[-1]
            if elapsed < 0.2:
                time.sleep(0.2 - elapsed)

        now = time.time()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < self.RATE_LIMIT_PERIOD
        ]
        if len(self._call_timestamps) >= self.RATE_LIMIT_CALLS:
            sleep_time = self.RATE_LIMIT_PERIOD - (now - self._call_timestamps[0])
            if sleep_time > 0:
                self._rate_pause_count += 1
                self._rate_pause_total_secs += sleep_time
                if self._rate_pause_count <= self.RATE_LIMIT_LOG_MAX:
                    logger.info(
                        f"RATE_LIMIT: pause {sleep_time:.1f}s "
                        f"({self._rate_pause_count}/session)"
                    )
                time.sleep(sleep_time)
        self._call_timestamps.append(time.time())

        # Batch cooldown every 50 calls
        self._total_api_calls += 1
        if self._total_api_calls % 50 == 0:
            self._rate_pause_count += 1
            self._rate_pause_total_secs += 2.0
            if self._rate_pause_count <= self.RATE_LIMIT_LOG_MAX:
                logger.info(
                    f"RATE_LIMIT: pause 2.0s "
                    f"({self._rate_pause_count}/session)"
                )
            time.sleep(2.0)

    def log_rate_limit_summary(self) -> None:
        """Log session-end summary of rate limiter pauses."""
        if self._rate_pause_count > 0 or self._total_api_calls > 0:
            logger.info(
                f"RATE_LIMIT_SUMMARY: {self._rate_pause_count} pauses | "
                f"{self._total_api_calls:,} calls | "
                f"{self._rate_pause_total_secs:.0f}s lost"
            )
        if self._intraday_cache_hits > 0:
            logger.info(
                f"FETCH_CACHE_SUMMARY: {self._intraday_cache_hits} cache hits "
                f"(API calls saved)"
            )

    def _is_network_error(self, error: Exception) -> bool:
        """Check if an exception is a network-level failure (DNS, connection refused, etc)."""
        err_str = str(error)
        # Check the full exception chain (cause / context)
        cause = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
        cause_str = str(cause) if cause else ""
        combined = f"{err_str} {cause_str}"
        return any(pat in combined for pat in self._NETWORK_ERROR_PATTERNS)

    @property
    def is_network_down(self) -> bool:
        """True if we're in a network-down cooldown period."""
        if self._network_down_since <= 0:
            return False
        return (time.time() - self._network_down_since) < self.NETWORK_COOLDOWN

    def _mark_network_down(self, error: Exception) -> None:
        """Record network failure and log once."""
        self._network_down_since = time.time()
        if not self._network_error_logged:
            logger.error(
                f"Network down — {error}. Suppressing API calls for "
                f"{self.NETWORK_COOLDOWN}s"
            )
            self._network_error_logged = True

    def _mark_network_up(self) -> None:
        """Clear network-down state after a successful API call."""
        if self._network_down_since > 0:
            downtime = time.time() - self._network_down_since
            logger.info(f"Network recovered after {downtime:.0f}s")
            self._network_down_since = 0
            self._network_error_logged = False

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry.

        Skips retries for 4xx client errors (permanent failures).
        Skips ALL retries during network-down cooldown.
        """
        # Fast-fail if network is down (avoid retry storms)
        if self.is_network_down:
            remaining = self.NETWORK_COOLDOWN - (time.time() - self._network_down_since)
            raise ConnectionError(
                f"Network down — skipping API call (retry in {remaining:.0f}s)"
            )

        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                # Successful call — clear network-down state
                self._mark_network_up()
                return result
            except Exception as e:
                last_exception = e
                # Don't retry 4xx client errors (except 429 rate limit)
                err_str = str(e)
                status_code = getattr(e, 'status', 0) or getattr(e, 'status_code', 0)
                if not status_code and hasattr(e, 'response'):
                    status_code = getattr(e.response, 'status_code', 0)
                is_client_error = (
                    status_code and 400 <= status_code < 500 and status_code != 429
                ) or (
                    "(400)" in err_str or "(401)" in err_str
                    or "(403)" in err_str or "(404)" in err_str
                )
                if is_client_error:
                    raise

                # Network-level failure — mark down and stop retrying immediately
                if self._is_network_error(e):
                    self._mark_network_down(e)
                    raise

                # HTTP 429 rate limit — wait 60s before retry
                is_429 = (status_code == 429) or "(429)" in err_str
                if is_429:
                    self._rate_pause_count += 1
                    self._rate_pause_total_secs += 60.0
                    if self._rate_pause_count <= self.RATE_LIMIT_LOG_MAX:
                        logger.warning(
                            f"RATE_LIMIT: 429 — waiting 60s "
                            f"({self._rate_pause_count}/session, "
                            f"attempt {attempt + 1}/{self.MAX_RETRIES})"
                        )
                    time.sleep(60)
                    continue

                wait = self.RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)
        raise last_exception  # type: ignore[misc]

    # ─────────────────────────────────────────
    # Authentication
    # ─────────────────────────────────────────

    def authenticate(self, auth_code: Optional[str] = None) -> bool:
        """
        Authenticate with Upstox.
        First tries loading saved token, then uses auth_code if provided.
        """
        if not UPSTOX_AVAILABLE:
            logger.error("upstox-python-sdk not installed. Install it: pip install upstox-python-sdk")
            return False

        token = self.auth.load_token()
        if token and self.auth.is_valid:
            logger.info("Using saved Upstox access token")
            return True

        if auth_code:
            self.auth.exchange_code_for_token(auth_code)
            return True

        logger.error(
            "No valid token. Run: python scripts/auth_upstox.py"
        )
        return False

    # ─────────────────────────────────────────
    # Historical Data (past days, excludes today)
    # ─────────────────────────────────────────

    def get_historical_candles(
        self,
        instrument_key: str,
        interval: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from Upstox.

        Uses DB cache first (if store is available), then Upstox API.
        NOTE: Historical API excludes today's data. Use get_intraday_candles() for today.

        Args:
            instrument_key: Upstox instrument key (e.g., "NSE_EQ|INE002A01018")
            interval: "1minute", "30minute", "day", "week", "month"
            from_date: Start date "YYYY-MM-DD"
            to_date: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: [datetime, open, high, low, close, volume, oi]
        """
        if to_date is None:
            to_date = date.today().isoformat()
        if from_date is None:
            from_date = (date.today() - timedelta(days=365)).isoformat()

        # Check local DB first
        if self._store is not None:
            symbol = self._resolve_symbol(instrument_key)
            if symbol:
                cached = self._store.get_candles(symbol, interval, from_date, to_date, limit=10000)
                if not cached.empty:
                    cached_max = str(cached["datetime"].max())[:10]
                    # Consider fresh if latest candle is within 2 days of requested end
                    if cached_max >= (pd.to_datetime(to_date) - timedelta(days=2)).strftime("%Y-%m-%d"):
                        logger.debug(f"Using cached data for {symbol} ({len(cached)} candles)")
                        return cached[["datetime", "open", "high", "low", "close", "volume", "oi"]]

        if not UPSTOX_AVAILABLE:
            raise ImportError("upstox-python-sdk is required. Install: pip install upstox-python-sdk")
        if not self.auth.is_valid:
            raise RuntimeError("Upstox auth not valid. Run: python scripts/auth_upstox.py")

        df = self._fetch_upstox_historical(instrument_key, interval, from_date, to_date)

        # Auto-save to DB
        if self._store is not None and not df.empty:
            symbol = self._resolve_symbol(instrument_key)
            if symbol:
                self._store.save_candles(symbol, instrument_key, df, interval)

        return df

    def _fetch_upstox_historical(
        self,
        instrument_key: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Fetch historical data using Upstox Historical Candle V3 API."""
        api_client = self._get_api_client()
        history_api = upstox_client.HistoryV3Api(api_client)

        # V3 uses unit/interval format: "day" → unit="days", interval="1"
        unit, iv = self._parse_interval_v3(interval)

        def fetch():
            return history_api.get_historical_candle_data1(
                instrument_key, unit, iv, to_date, from_date,
            )

        response = self._retry_with_backoff(fetch)

        candles = response.data.candles if response.data else []
        if not candles:
            logger.debug(
                f"No historical data for {instrument_key} ({from_date} to {to_date})"
            )
            return pd.DataFrame()

        return self._candles_to_df(candles)

    # ─────────────────────────────────────────
    # Intraday Data (today only)
    # ─────────────────────────────────────────

    def get_intraday_candles(
        self,
        instrument_key: str,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        """
        Fetch today's intraday candles from Upstox Intra-Day Candle V3 API.

        Unlike get_historical_candles() which excludes today, this returns
        ONLY today's data at the requested interval.

        Args:
            instrument_key: Upstox instrument key
            interval: "1minute", "5minute", "15minute", "30minute", "1hour", "day"

        Returns:
            DataFrame with columns: [datetime, open, high, low, close, volume, oi]
        """
        # Return cached result if fresh enough
        cache_key = (instrument_key, interval)
        cached = self._intraday_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._intraday_cache_ttl:
            self._intraday_cache_hits += 1
            logger.debug(
                f"FETCH_CACHE: skipping {instrument_key} "
                f"({time.time() - cached[0]:.0f}s ago)"
            )
            return cached[1]

        if not UPSTOX_AVAILABLE:
            raise ImportError("upstox-python-sdk is required")
        if not self.auth.is_valid:
            raise RuntimeError("Upstox auth not valid. Run: python scripts/auth_upstox.py")

        api_client = self._get_api_client()
        history_api = upstox_client.HistoryV3Api(api_client)

        unit, iv = self._parse_interval_v3(interval)

        def fetch():
            return history_api.get_intra_day_candle_data(
                instrument_key, unit, iv,
            )

        response = self._retry_with_backoff(fetch)

        candles = response.data.candles if response.data else []
        if not candles:
            return pd.DataFrame()

        df = self._candles_to_df(candles)
        self._intraday_cache[cache_key] = (time.time(), df)
        return df

    # ─────────────────────────────────────────
    # Common helpers
    # ─────────────────────────────────────────

    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        """Convert Upstox candle array to DataFrame."""
        df = pd.DataFrame(
            candles,
            columns=["datetime", "open", "high", "low", "close", "volume", "oi"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype(int)

        return df

    @staticmethod
    def _parse_interval_v3(interval: str) -> tuple[str, str]:
        """
        Convert legacy interval string to V3 (unit, interval) tuple.

        V3 format: unit = "minutes"|"hours"|"days"|"weeks"|"months", interval = "1"-"300"
        Legacy:    "1minute", "30minute", "day", "week", "month"
        """
        interval = interval.lower().strip()
        if interval == "day":
            return ("days", "1")
        if interval == "week":
            return ("weeks", "1")
        if interval == "month":
            return ("months", "1")
        if "minute" in interval:
            num = interval.replace("minute", "").replace("s", "").strip() or "1"
            return ("minutes", num)
        if "hour" in interval:
            num = interval.replace("hour", "").replace("s", "").strip() or "1"
            return ("hours", num)
        # Default to daily
        return ("days", "1")

    def _resolve_symbol(self, instrument_key: str) -> Optional[str]:
        """Resolve instrument_key to a symbol name for DB storage."""
        # Check equities
        sym = self.get_symbol_for_instrument(instrument_key)
        if sym:
            return sym
        # Check indices
        for idx_name, idx_info in self.universe.get("indices", {}).items():
            if idx_info.get("instrument_key") == instrument_key:
                return idx_name
        # F&O keys: use the key itself as identifier
        if instrument_key.startswith("NSE_FO|"):
            return instrument_key
        return None

    # ─────────────────────────────────────────
    # Live Quotes
    # ─────────────────────────────────────────

    def get_live_quote(self, instrument_key: str) -> dict[str, Any]:
        """Fetch current live quote for an instrument (15s cache)."""
        # Return cached quote if fresh enough
        cached = self._quote_cache.get(instrument_key)
        if cached and (time.time() - cached[0]) < self._quote_cache_ttl:
            return cached[1]

        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            logger.warning("Upstox not available for live quotes")
            return {}

        api_client = self._get_api_client()
        market_api = upstox_client.MarketQuoteApi(api_client)

        def fetch():
            return market_api.get_full_market_quote(
                symbol=instrument_key,
                api_version="2.0",
            )

        response = self._retry_with_backoff(fetch)

        if response.data:
            # Upstox SDK returns keys with ":" separator (e.g. "NSE_FO:NIFTY...")
            # but instrument_key uses "|" (e.g. "NSE_FO|54908"). Try both.
            quote = None
            if instrument_key in response.data:
                quote = response.data[instrument_key]
            else:
                # Try colon-separated or find by instrument_token field
                colon_key = instrument_key.replace("|", ":")
                if colon_key in response.data:
                    quote = response.data[colon_key]
                else:
                    # Last resort: find by instrument_token match
                    for k, v in response.data.items():
                        if getattr(v, "instrument_token", None) == instrument_key:
                            quote = v
                            break

            if quote is not None:
                result = {
                    "ltp": quote.last_price,
                    "open": quote.ohlc.open if quote.ohlc else None,
                    "high": quote.ohlc.high if quote.ohlc else None,
                    "low": quote.ohlc.low if quote.ohlc else None,
                    "close": quote.ohlc.close if quote.ohlc else None,
                    "volume": quote.volume,
                    "oi": quote.oi if hasattr(quote, "oi") else 0,
                    "bid": quote.depth.buy[0].price if quote.depth and quote.depth.buy else None,
                    "ask": quote.depth.sell[0].price if quote.depth and quote.depth.sell else None,
                    "timestamp": datetime.now().isoformat(),
                }
                self._quote_cache[instrument_key] = (time.time(), result)
                return result
        return {}

    def get_live_quotes_batch(
        self, instrument_keys: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch live quotes for multiple instruments (batched)."""
        results = {}
        batch_size = 50
        for i in range(0, len(instrument_keys), batch_size):
            batch = instrument_keys[i : i + batch_size]
            joined = ",".join(batch)

            if UPSTOX_AVAILABLE and self.auth.is_valid:
                api_client = self._get_api_client()
                market_api = upstox_client.MarketQuoteApi(api_client)

                def fetch():
                    return market_api.get_full_market_quote(symbol=joined, api_version="2.0")

                try:
                    response = self._retry_with_backoff(fetch)
                    if response.data:
                        for key in batch:
                            quote = response.data.get(key)
                            if quote is None:
                                quote = response.data.get(key.replace("|", ":"))
                            if quote is None:
                                for v in response.data.values():
                                    if getattr(v, "instrument_token", None) == key:
                                        quote = v
                                        break
                            if quote is not None:
                                results[key] = {
                                    "ltp": quote.last_price,
                                    "volume": quote.volume,
                                    "timestamp": datetime.now().isoformat(),
                                }
                except Exception as e:
                    logger.error(f"Batch quote fetch failed: {e}")

        return results

    def get_option_greeks(
        self, instrument_keys: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch option greeks (delta, gamma, theta, vega, IV) via V3 API.

        Args:
            instrument_keys: List of NSE_FO instrument keys (max 50 per batch)

        Returns:
            {instrument_key: {ltp, delta, gamma, theta, vega, iv, oi, volume}}
        """
        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            return {}

        results = {}
        batch_size = 50

        for i in range(0, len(instrument_keys), batch_size):
            batch = instrument_keys[i : i + batch_size]
            joined = ",".join(batch)

            try:
                api_client = self._get_api_client()
                v3_api = upstox_client.MarketQuoteV3Api(api_client)

                def fetch():
                    return v3_api.get_market_quote_option_greek(instrument_key=joined)

                response = self._retry_with_backoff(fetch)

                if response and response.data:
                    for key in batch:
                        quote = response.data.get(key)
                        if quote is None:
                            quote = response.data.get(key.replace("|", ":"))
                        if quote is None:
                            for v in response.data.values():
                                if getattr(v, "instrument_token", None) == key:
                                    quote = v
                                    break
                        if quote is not None:
                            results[key] = {
                                "ltp": getattr(quote, "last_price", 0) or 0,
                                "delta": getattr(quote, "delta", 0) or 0,
                                "gamma": getattr(quote, "gamma", 0) or 0,
                                "theta": getattr(quote, "theta", 0) or 0,
                                "vega": getattr(quote, "vega", 0) or 0,
                                "iv": getattr(quote, "iv", 0) or 0,
                                "oi": getattr(quote, "oi", 0) or 0,
                                "volume": getattr(quote, "volume", 0) or 0,
                            }
            except Exception as e:
                logger.error(f"Option greeks fetch failed: {e}")

        return results

    # ─────────────────────────────────────────
    # WebSocket Streaming
    # ─────────────────────────────────────────

    def start_market_stream(
        self,
        instrument_keys: list[str],
        mode: str = "ltpc",
    ) -> None:
        """
        Start MarketDataStreamerV3 for real-time LTP via WebSocket.

        Runs in a background thread (SDK-managed). Updates _ws_ltp_cache
        on every tick. Falls back silently if Upstox unavailable.

        Args:
            instrument_keys: Initial instrument keys to subscribe
            mode: "ltpc" (LTP + close) or "full" (LTP + depth + OHLC)
        """
        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            logger.warning("Cannot start market stream: Upstox not available")
            return

        if self._streamer:
            logger.warning("Market stream already running, skipping start")
            return

        try:
            configuration = self.auth.get_configuration()
            self._streamer = upstox_client.MarketDataStreamerV3(
                api_client=upstox_client.ApiClient(configuration),
                instrumentKeys=instrument_keys,
                mode=mode,
            )
            self._streamer.auto_reconnect(True, interval=5, retry_count=50)
            self._streamer.on("message", self._on_ws_message)
            self._streamer.on("open", self._on_ws_open)
            self._streamer.on("error", self._on_ws_error)
            self._streamer.on("close", self._on_ws_close)
            self._streamer.on(
                "reconnecting",
                lambda msg: logger.info(f"WS reconnecting: {msg}"),
            )
            self._streamer.connect()
            logger.info(
                f"WS stream started for {len(instrument_keys)} instruments ({mode})"
            )
        except Exception as e:
            logger.error(f"WS stream start failed: {e}")
            self._streamer = None

    def stop_market_stream(self) -> None:
        """Stop the market data stream and clear LTP cache."""
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception as e:
                logger.warning(f"WS disconnect error: {e}")
            self._streamer = None
        self._ws_connected = False
        with self._ws_ltp_lock:
            self._ws_ltp_cache.clear()
        logger.info("WS market stream stopped")

    # ── WebSocket callbacks ──

    def _on_ws_message(self, data_dict: dict) -> None:
        """Handle incoming WS market data — update LTP cache."""
        feeds = data_dict.get("feeds", {})
        if not feeds:
            return
        with self._ws_ltp_lock:
            for key, feed in feeds.items():
                ltpc = feed.get("ltpc", {})
                ltp = ltpc.get("ltp")
                if ltp and ltp > 0:
                    self._ws_ltp_cache[key] = float(ltp)

    def _on_ws_open(self) -> None:
        self._ws_connected = True
        logger.info("WS market feed connected")

    def _on_ws_error(self, error) -> None:
        logger.error(f"WS market feed error: {error}")

    def _on_ws_close(self, code=None, msg=None) -> None:
        self._ws_connected = False
        logger.warning(f"WS market feed closed (code={code})")

    # ── WebSocket LTP access ──

    def get_ws_ltp(self, instrument_key: str) -> float | None:
        """Get LTP from WebSocket cache. Returns None if not available."""
        with self._ws_ltp_lock:
            return self._ws_ltp_cache.get(instrument_key)

    @property
    def ws_connected(self) -> bool:
        """Whether WebSocket feed is currently connected."""
        return self._ws_connected

    def ws_subscribe(self, instrument_keys: list[str]) -> None:
        """Subscribe additional instruments to the live WS feed."""
        if self._streamer and self._ws_connected:
            try:
                self._streamer.subscribe(instrument_keys, "ltpc")
                logger.info(f"WS subscribed: {instrument_keys}")
            except Exception as e:
                logger.warning(f"WS subscribe failed: {e}")

    def ws_unsubscribe(self, instrument_keys: list[str]) -> None:
        """Unsubscribe instruments from the live WS feed."""
        if self._streamer and self._ws_connected:
            try:
                self._streamer.unsubscribe(instrument_keys)
                with self._ws_ltp_lock:
                    for key in instrument_keys:
                        self._ws_ltp_cache.pop(key, None)
                logger.info(f"WS unsubscribed: {instrument_keys}")
            except Exception as e:
                logger.warning(f"WS unsubscribe failed: {e}")

    def start_portfolio_stream(self, on_order_update: Optional[Any] = None) -> None:
        """Start PortfolioDataStreamer for real-time order/position updates."""
        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            logger.error("Cannot start portfolio stream: Upstox not available")
            return

        configuration = self.auth.get_configuration()

        self._portfolio_streamer = upstox_client.PortfolioDataStreamer(
            api_client=upstox_client.ApiClient(configuration)
        )

        if on_order_update:
            self._portfolio_streamer.on("order_update", on_order_update)

        self._portfolio_streamer.on(
            "open", lambda: logger.info("Portfolio stream connected")
        )
        self._portfolio_streamer.on(
            "error", lambda e: logger.error(f"Portfolio stream error: {e}")
        )

        self._portfolio_streamer.connect()
        logger.info("Portfolio data stream started")

    def stop_portfolio_stream(self) -> None:
        """Stop the portfolio data stream."""
        if self._portfolio_streamer:
            self._portfolio_streamer.disconnect()
            self._portfolio_streamer = None
            logger.info("Portfolio stream stopped")

    # ─────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────

    def get_instrument_keys(self, filter_nifty50: bool = True) -> list[str]:
        """Get list of instrument keys from config universe."""
        nifty50 = self.universe.get("nifty50", {})
        keys = [info["instrument_key"] for info in nifty50.values()]
        return keys

    def get_symbol_for_instrument(self, instrument_key: str) -> Optional[str]:
        """Reverse lookup: instrument_key -> symbol name."""
        nifty50 = self.universe.get("nifty50", {})
        for symbol, info in nifty50.items():
            if info.get("instrument_key") == instrument_key:
                return symbol
        return None

    def get_instrument_for_symbol(self, symbol: str) -> Optional[str]:
        """Lookup: symbol -> instrument_key."""
        nifty50 = self.universe.get("nifty50", {})
        if symbol in nifty50:
            return nifty50[symbol].get("instrument_key")
        return None

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Get sector classification for a symbol."""
        nifty50 = self.universe.get("nifty50", {})
        if symbol in nifty50:
            return nifty50[symbol].get("sector")
        return None

    def get_all_historical(
        self,
        symbols: Optional[list[str]] = None,
        interval: str = "day",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols in universe (DB-first, then API)."""
        if symbols is None:
            symbols = list(self.universe.get("nifty50", {}).keys())

        from_date = (date.today() - timedelta(days=days)).isoformat()
        to_date = date.today().isoformat()
        result = {}

        for sym in symbols:
            # Try DB first
            if self._store is not None:
                df = self._store.get_candles(sym, interval, from_date, to_date, limit=10000)
                if not df.empty and len(df) >= days * 0.3:
                    result[sym] = df[["datetime", "open", "high", "low", "close", "volume", "oi"]]
                    logger.debug(f"Loaded {sym} from DB ({len(df)} candles)")
                    continue

            # Fetch from API
            inst_key = self.get_instrument_for_symbol(sym)
            if inst_key:
                try:
                    df = self.get_historical_candles(
                        inst_key, interval, from_date, to_date
                    )
                    if not df.empty:
                        result[sym] = df
                        logger.debug(f"Fetched {len(df)} candles for {sym}")
                except Exception as e:
                    logger.error(f"Failed to fetch {sym}: {e}")

        logger.info(f"Fetched historical data for {len(result)}/{len(symbols)} symbols")
        return result

    # ─────────────────────────────────────────
    # Expired Instruments (historical F&O data)
    # ─────────────────────────────────────────

    def get_expired_expiries(self, underlying_key: str) -> list[str]:
        """
        Get list of available expired expiry dates for an underlying.

        Uses Upstox V2 API: GET /v2/expired-instruments/expiries
        Returns list of date strings in YYYY-MM-DD format (up to 6 months history).
        """
        if not self.auth.is_valid:
            raise RuntimeError("Upstox auth not valid. Run: python scripts/auth_upstox.py")

        url = f"{self.auth.api_base_url}/v2/expired-instruments/expiries"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth.access_token}",
        }
        params = {"instrument_key": underlying_key}

        def fetch():
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_with_backoff(fetch)
            if result.get("status") == "success" and result.get("data"):
                return result["data"]
        except Exception as e:
            logger.warning(f"Failed to get expiries for {underlying_key}: {e}")

        return []

    def get_expired_option_contracts(
        self,
        underlying_key: str,
        expiry_date: str,
    ) -> list[dict]:
        """
        Get expired option contracts for a given underlying and expiry date.

        Uses Upstox V2 API: GET /v2/expired-instruments/option/contract
        Returns list of contract dicts with instrument_key, strike_price, instrument_type, etc.
        """
        if not self.auth.is_valid:
            raise RuntimeError("Upstox auth not valid. Run: python scripts/auth_upstox.py")

        url = f"{self.auth.api_base_url}/v2/expired-instruments/option/contract"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth.access_token}",
        }
        params = {
            "instrument_key": underlying_key,
            "expiry_date": expiry_date,
        }

        def fetch():
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        result = self._retry_with_backoff(fetch)

        if result.get("status") == "success" and result.get("data"):
            return result["data"]

        logger.warning(f"No expired contracts for {underlying_key} expiry {expiry_date}")
        return []

    def get_expired_historical_candles(
        self,
        expired_instrument_key: str,
        interval: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical candles for EXPIRED F&O instruments.

        Uses Upstox V2 API: GET /v2/expired-instruments/historical-candle/{key}/{interval}/{to}/{from}
        This is separate from the regular historical API and works for expired contracts.

        Args:
            expired_instrument_key: Key from get_expired_option_contracts()
            interval: "1minute", "5minute", "15minute", "30minute", "day"
            from_date: Start date "YYYY-MM-DD"
            to_date: End date "YYYY-MM-DD"
        """
        if not self.auth.is_valid:
            raise RuntimeError("Upstox auth not valid. Run: python scripts/auth_upstox.py")

        if to_date is None:
            to_date = date.today().isoformat()
        if from_date is None:
            from_date = (date.today() - timedelta(days=60)).isoformat()

        # Check DB cache first
        if self._store is not None:
            cached = self._store.get_candles(
                expired_instrument_key, interval, from_date, to_date, limit=10000
            )
            if not cached.empty:
                logger.debug(f"Using cached expired data for {expired_instrument_key}")
                return cached[["datetime", "open", "high", "low", "close", "volume", "oi"]]

        import urllib.parse
        encoded_key = urllib.parse.quote(expired_instrument_key, safe="")

        url = (
            f"{self.auth.api_base_url}/v2/expired-instruments/historical-candle"
            f"/{encoded_key}/{interval}/{to_date}/{from_date}"
        )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth.access_token}",
        }

        def fetch():
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_with_backoff(fetch)
        except Exception as e:
            logger.warning(f"Expired candle fetch failed for {expired_instrument_key}: {e}")
            return pd.DataFrame()

        candles = result.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = self._candles_to_df(candles)

        # Auto-save to DB
        if self._store is not None and not df.empty:
            self._store.save_candles(
                expired_instrument_key, expired_instrument_key, df, interval
            )

        return df

    def get_vix_history(self, days: int = 365) -> dict[str, float]:
        """
        Fetch India VIX historical data using Upstox Historical API.

        Returns dict: {date_str: vix_close_value}
        """
        vix_key = "NSE_INDEX|India VIX"
        from_date = (date.today() - timedelta(days=days)).isoformat()
        to_date = date.today().isoformat()

        try:
            df = self.get_historical_candles(vix_key, "day", from_date, to_date)
            if df is not None and not df.empty:
                vix_map = {}
                for _, row in df.iterrows():
                    d = pd.to_datetime(row["datetime"]).date()
                    vix_map[d] = float(row["close"])
                logger.info(f"Fetched VIX history: {len(vix_map)} days")
                return vix_map
        except Exception as e:
            logger.warning(f"VIX history fetch failed: {e}")

        return {}

    def get_current_vix(self) -> dict[str, float]:
        """
        Get current India VIX value via Upstox live quote.

        Returns:
            {"vix": 14.5, "change": 0, "change_pct": 0}
        """
        vix_key = "NSE_INDEX|India VIX"
        try:
            quote = self.get_live_quote(vix_key)
            if quote:
                ltp = quote.get("ltp", 0)
                if ltp <= 0:
                    raise ValueError(f"VIX ltp={ltp} is invalid")
                prev_close = quote.get("close", ltp)
                change = ltp - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                return {
                    "vix": ltp,
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                }
        except Exception as e:
            logger.warning(f"Failed to get current VIX from Upstox: {e}")

        # Fallback: use latest from historical data in DB
        try:
            vix_hist = self.get_vix_history(days=5)
            if vix_hist:
                latest_date = max(vix_hist.keys())
                return {"vix": vix_hist[latest_date], "change": 0, "change_pct": 0}
        except Exception:
            pass

        # Safe default: 15 (India VIX long-term average)
        # VIX=0 would bypass VIX>28 filter and use wrong SL/TP thresholds
        logger.warning("VIX unavailable from API and DB — using safe default 15")
        return {"vix": 15, "change": 0, "change_pct": 0}

    # ─────────────────────────────────────────
    # Option Chain
    # ─────────────────────────────────────────

    def get_option_chain(self, instrument_key: str, expiry_date: str) -> dict[str, Any]:
        """
        Fetch option chain from Upstox and compute OI levels, PCR, max pain.

        Args:
            instrument_key: Underlying instrument key (e.g., "NSE_INDEX|Nifty 50")
            expiry_date: Expiry date in YYYY-MM-DD format

        Returns:
            {
                "oi_levels": {max_call_oi_strike, max_call_oi, max_call_oi_change,
                              max_put_oi_strike, max_put_oi, max_put_oi_change, underlying},
                "pcr": {pcr_oi, pcr_volume, pcr_change_oi},
                "max_pain": {max_pain_strike, distance_pct},
            }
        """
        fallback_pcr = self._last_known_pcr
        empty = {
            "oi_levels": {},
            "pcr": {"pcr_oi": fallback_pcr, "pcr_volume": 0, "pcr_change_oi": 0},
            "max_pain": {},
        }

        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            return empty

        try:
            api_client = self._get_api_client()
            options_api = upstox_client.OptionsApi(api_client)

            # Retry up to 3 times for empty responses
            response = None
            for attempt in range(3):
                try:
                    def fetch():
                        return options_api.get_put_call_option_chain(
                            instrument_key, expiry_date
                        )
                    response = self._retry_with_backoff(fetch)
                    if response and response.data:
                        self._oc_fail_count[instrument_key] = 0
                        break
                    if attempt < 2:
                        time.sleep(1)
                except Exception as e:
                    logger.warning(
                        f"Option chain attempt {attempt+1}/3 failed for "
                        f"{instrument_key}: {e}"
                    )
                    if attempt < 2:
                        time.sleep(2)

            if not response or not response.data:
                count = self._oc_fail_count.get(instrument_key, 0) + 1
                self._oc_fail_count[instrument_key] = count
                if count == 1 or count % 10 == 0:
                    logger.warning(
                        f"Option chain: empty response for {instrument_key} "
                        f"after 3 attempts (fail #{count})"
                    )
                return empty

            strikes_data = response.data
            spot = 0
            max_call_oi = 0
            max_call_oi_strike = 0
            max_call_oi_change = 0
            max_put_oi = 0
            max_put_oi_strike = 0
            max_put_oi_change = 0
            total_call_oi = 0
            total_put_oi = 0
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi_change = 0
            total_put_oi_change = 0

            # Collect per-strike data for max pain calculation
            strike_oi = []  # [(strike, call_oi, put_oi)]

            for s in strikes_data:
                strike = s.strike_price
                spot = s.underlying_spot_price or spot

                # Call side
                call_oi_val = 0
                call_vol = 0
                call_oi_chg = 0
                if s.call_options and s.call_options.market_data:
                    md = s.call_options.market_data
                    call_oi_val = int(md.oi or 0)
                    call_vol = int(md.volume or 0)
                    call_oi_chg = call_oi_val - int(md.prev_oi or 0)

                # Put side
                put_oi_val = 0
                put_vol = 0
                put_oi_chg = 0
                if s.put_options and s.put_options.market_data:
                    md = s.put_options.market_data
                    put_oi_val = int(md.oi or 0)
                    put_vol = int(md.volume or 0)
                    put_oi_chg = put_oi_val - int(md.prev_oi or 0)

                # Track max call OI strike
                if call_oi_val > max_call_oi:
                    max_call_oi = call_oi_val
                    max_call_oi_strike = strike
                    max_call_oi_change = call_oi_chg

                # Track max put OI strike
                if put_oi_val > max_put_oi:
                    max_put_oi = put_oi_val
                    max_put_oi_strike = strike
                    max_put_oi_change = put_oi_chg

                total_call_oi += call_oi_val
                total_put_oi += put_oi_val
                total_call_vol += call_vol
                total_put_vol += put_vol
                total_call_oi_change += call_oi_chg
                total_put_oi_change += put_oi_chg

                strike_oi.append((strike, call_oi_val, put_oi_val))

            # Compute PCR
            pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0
            pcr_volume = round(total_put_vol / total_call_vol, 3) if total_call_vol > 0 else 0
            pcr_change_oi = round(total_put_oi_change / total_call_oi_change, 3) if total_call_oi_change != 0 else 0

            # PCR fallback: use last known good value instead of 0
            if pcr_oi <= 0:
                pcr_oi = self._last_known_pcr
                logger.warning(f"PCR_FALLBACK: empty OI data, using last known PCR={pcr_oi:.2f}")
            else:
                self._last_known_pcr = pcr_oi

            # Compute max pain
            max_pain_strike = 0
            min_pain = float("inf")
            all_strikes = [s[0] for s in strike_oi]

            for candidate in all_strikes:
                total_pain = 0
                for strike, c_oi, p_oi in strike_oi:
                    if candidate > strike:
                        # Calls are ITM — writers pay (candidate - strike) * call_oi
                        total_pain += (candidate - strike) * c_oi
                    elif candidate < strike:
                        # Puts are ITM — writers pay (strike - candidate) * put_oi
                        total_pain += (strike - candidate) * p_oi
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = candidate

            distance_pct = round((max_pain_strike - spot) / spot * 100, 2) if spot > 0 else 0

            result = {
                "oi_levels": {
                    "max_call_oi_strike": max_call_oi_strike,
                    "max_call_oi": max_call_oi,
                    "max_call_oi_change": max_call_oi_change,
                    "max_put_oi_strike": max_put_oi_strike,
                    "max_put_oi": max_put_oi,
                    "max_put_oi_change": max_put_oi_change,
                    "underlying": spot,
                },
                "pcr": {
                    "pcr_oi": pcr_oi,
                    "pcr_volume": pcr_volume,
                    "pcr_change_oi": pcr_change_oi,
                },
                "max_pain": {
                    "max_pain_strike": max_pain_strike,
                    "distance_pct": distance_pct,
                },
            }

            # OI snapshot tracking for change rate filter
            from src.config.env_loader import get_config as _get_cfg
            _cfg = _get_cfg()
            snapshot_interval = getattr(_cfg, "OI_SNAPSHOT_INTERVAL_MINUTES", 30) * 60
            new_snapshot = {
                "timestamp": time.time(),
                "put_oi": total_put_oi,
                "call_oi": total_call_oi,
            }
            if self._oi_snapshot is None:
                self._oi_snapshot = new_snapshot
            elif time.time() - self._oi_snapshot["timestamp"] >= snapshot_interval:
                self._oi_snapshot_prev = self._oi_snapshot
                self._oi_snapshot = new_snapshot

            logger.info(
                f"Option chain ({instrument_key.split('|')[-1]}): "
                f"PCR={pcr_oi:.2f}, MaxCallOI@{max_call_oi_strike}, "
                f"MaxPutOI@{max_put_oi_strike}, MaxPain@{max_pain_strike}"
            )
            return result

        except Exception as e:
            logger.warning(f"Option chain fetch failed: {e}")
            return empty

    def get_oi_change_rates(self) -> tuple[Optional[float], Optional[float]]:
        """Compute OI change rates between current and previous snapshots.

        Returns (put_oi_change_pct, call_oi_change_pct) or (None, None) if
        not enough data yet (need 2 snapshots, 30 min apart).
        """
        if self._oi_snapshot is None or self._oi_snapshot_prev is None:
            return None, None
        prev_put = self._oi_snapshot_prev["put_oi"]
        prev_call = self._oi_snapshot_prev["call_oi"]
        if prev_put <= 0 or prev_call <= 0:
            return None, None
        put_change = (self._oi_snapshot["put_oi"] - prev_put) / prev_put * 100
        call_change = (self._oi_snapshot["call_oi"] - prev_call) / prev_call * 100
        return round(put_change, 2), round(call_change, 2)

    # ─────────────────────────────────────────
    # Fund & Margin
    # ─────────────────────────────────────────

    def get_fund_and_margin(self) -> dict[str, Any]:
        """Get user's fund and margin details from Upstox."""
        if not UPSTOX_AVAILABLE or not self.auth.is_valid:
            return {}

        api_client = self._get_api_client()
        user_api = upstox_client.UserApi(api_client)

        def fetch():
            return user_api.get_user_fund_margin(api_version="2.0")

        try:
            response = self._retry_with_backoff(fetch)
            if response.data:
                data = response.data
                # SDK returns data as dict(str, UserFundMarginData)
                if isinstance(data, dict):
                    equity = data.get("equity")
                elif hasattr(data, "equity"):
                    equity = data.equity
                else:
                    equity = None
                if equity is not None:
                    return {
                        "available_margin": getattr(equity, "available_margin", 0),
                        "used_margin": getattr(equity, "used_margin", 0),
                        "payin_amount": getattr(equity, "payin_amount", 0),
                    }
        except Exception as e:
            logger.error(f"Failed to get fund/margin: {e}")

        return {}
