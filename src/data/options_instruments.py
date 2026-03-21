"""
Options Instrument Resolver — Maps (NIFTY, strike, expiry, CE/PE) to instrument keys.

Supports both Upstox and Dhan brokers:
  - Upstox: Downloads instrument master CSV, resolves NSE_FO|xxxxx keys
  - Dhan: Downloads security list CSV, resolves integer security_ids → NSE_FO|{sid}
"""

from __future__ import annotations

import gzip
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

# Upstox instrument master URL (gzipped CSV)
_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
_CACHE_PATH = Path("data/instruments_nse.csv")

# Dhan instrument cache
_DHAN_CACHE_PATH = Path("data/instruments_dhan_fno.csv")

# Index-specific constants
INDEX_CONFIG = {
    "NIFTY": {"strike_gap": 50, "lot_size": 65, "symbol_prefix": "NIFTY"},
    "BANKNIFTY": {"strike_gap": 100, "lot_size": 30, "symbol_prefix": "BANKNIFTY"},
    "FINNIFTY": {"strike_gap": 50, "lot_size": 65, "symbol_prefix": "FINNIFTY"},
    "MIDCPNIFTY": {"strike_gap": 25, "lot_size": 120, "symbol_prefix": "MIDCPNIFTY"},
    "RELIANCE": {"strike_gap": 20, "lot_size": 250, "symbol_prefix": "RELIANCE"},
    "SENSEX": {"strike_gap": 100, "lot_size": 10, "symbol_prefix": "SENSEX"},
}


class OptionsInstrumentResolver:
    """
    Resolves Upstox instrument keys for NIFTY/BANKNIFTY options.

    Usage:
        resolver = OptionsInstrumentResolver()
        resolver.refresh()  # Download instrument master if stale
        key = resolver.get_instrument_key("NIFTY", 24500, date(2026, 2, 26), "CE")
        # → "NSE_FO|..."
    """

    def __init__(self, cache_path: str | Path = _CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None

    def refresh(self, force: bool = False) -> None:
        """Download instrument master CSV if stale (>1 day old) or forced."""
        if not force and self.cache_path.exists():
            mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
            if mtime.date() == date.today():
                logger.debug("Instrument master is fresh, loading from cache")
                self._load_cache()
                return

        logger.info("Downloading Upstox instrument master...")
        try:
            resp = requests.get(_INSTRUMENT_URL, timeout=30)
            resp.raise_for_status()

            # Decompress gzip and read CSV
            with gzip.open(io.BytesIO(resp.content), "rt") as gz:
                self._df = pd.read_csv(gz)

            # Filter to F&O options only
            self._df = self._df[
                (self._df["instrument_type"] == "OPTIDX")
                & (self._df["exchange"] == "NSE_FO")
            ].copy()

            # Save full CSV for cache
            self._df.to_csv(self.cache_path, index=False)
            logger.info(
                f"Instrument master cached: {len(self._df)} option instruments"
            )
        except Exception as e:
            logger.error(f"Failed to download instrument master: {e}")
            if self.cache_path.exists():
                logger.warning("Using stale cache as fallback")
                self._load_cache()
            else:
                self._df = pd.DataFrame()

    def _load_cache(self) -> None:
        """Load cached instrument CSV."""
        try:
            self._df = pd.read_csv(self.cache_path)
            logger.debug(f"Loaded {len(self._df)} instruments from cache")
        except Exception as e:
            logger.error(f"Failed to load instrument cache: {e}")
            self._df = pd.DataFrame()

    def _ensure_loaded(self) -> None:
        if self._df is None or self._df.empty:
            self.refresh()

    def get_instrument_key(
        self,
        symbol: str,
        strike: float,
        expiry_date: date,
        option_type: str,
    ) -> Optional[str]:
        """
        Get Upstox instrument key for a specific option contract.

        Args:
            symbol: "NIFTY" or "BANKNIFTY"
            strike: Strike price (e.g., 24500)
            expiry_date: Expiry date
            option_type: "CE" or "PE"

        Returns:
            Instrument key like "NSE_FO|..." or None if not found
        """
        self._ensure_loaded()
        if self._df is None or self._df.empty:
            return None

        expiry_str = expiry_date.strftime("%Y-%m-%d")

        # Filter by symbol, strike, expiry, option type
        mask = (
            (self._df["name"].str.upper() == symbol.upper())
            & (self._df["strike"] == strike)
            & (self._df["expiry"].str[:10] == expiry_str)
            & (self._df["option_type"] == option_type)
        )

        matches = self._df[mask]
        if matches.empty:
            logger.warning(
                f"No instrument found: {symbol} {strike} {option_type} {expiry_str}"
            )
            return None

        key = matches.iloc[0]["instrument_key"]
        logger.debug(f"Resolved: {symbol} {strike}{option_type} → {key}")
        return str(key)

    def get_atm_strike(self, symbol: str, spot_price: float) -> float:
        """Get ATM strike price (nearest valid strike to spot)."""
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        gap = cfg["strike_gap"]
        return round(spot_price / gap) * gap

    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for the given index (from instrument master if available)."""
        self._ensure_loaded()
        if self._df is not None and not self._df.empty:
            sym_mask = self._df["name"].str.upper() == symbol.upper()
            matches = self._df[sym_mask]
            if not matches.empty and "lot_size" in matches.columns:
                lot = int(matches.iloc[0]["lot_size"])
                if lot > 0:
                    return lot
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        return cfg["lot_size"]

    def get_weekly_expiry(self, symbol: str, ref_date: Optional[date] = None) -> date:
        """
        Get next weekly expiry date (Thursday) for the given index.

        If today is Thursday and market is still open, returns today.
        Otherwise returns next Thursday.
        """
        self._ensure_loaded()
        today = ref_date or date.today()

        if self._df is not None and not self._df.empty:
            # Use actual expiry dates from instrument master
            sym_mask = self._df["name"].str.upper() == symbol.upper()
            expiries = pd.to_datetime(
                self._df[sym_mask]["expiry"].str[:10]
            ).dt.date.unique()
            expiries = sorted(expiries)

            # Find next expiry >= today
            for exp in expiries:
                if exp >= today:
                    return exp

        # Fallback: next Monday (NIFTY weekly expiry changed to Monday in 2025)
        days_ahead = (0 - today.weekday()) % 7  # Monday = 0
        if days_ahead == 0 and datetime.now().hour >= 16:
            days_ahead = 7  # Past market close on expiry day
        if days_ahead == 0:
            return today
        return today + timedelta(days=days_ahead)

    def get_option_chain_keys(
        self,
        symbol: str,
        spot_price: float,
        num_strikes: int = 5,
        expiry_date: Optional[date] = None,
    ) -> list[dict]:
        """
        Get instrument keys for nearby strikes (ATM +/- num_strikes).

        Returns list of dicts:
            [{"strike": 24500, "type": "CE", "instrument_key": "NSE_FO|..."},
             {"strike": 24500, "type": "PE", "instrument_key": "NSE_FO|..."}, ...]
        """
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        gap = cfg["strike_gap"]
        atm = self.get_atm_strike(symbol, spot_price)
        expiry = expiry_date or self.get_weekly_expiry(symbol)

        results = []
        for offset in range(-num_strikes, num_strikes + 1):
            strike = atm + offset * gap
            for opt_type in ("CE", "PE"):
                key = self.get_instrument_key(symbol, strike, expiry, opt_type)
                if key:
                    results.append({
                        "strike": strike,
                        "type": opt_type,
                        "instrument_key": key,
                        "expiry": expiry,
                        "otm_distance": offset if opt_type == "CE" else -offset,
                    })

        return results

    def get_trading_symbol(
        self,
        symbol: str,
        strike: float,
        expiry_date: date,
        option_type: str,
    ) -> Optional[str]:
        """Get human-readable trading symbol (e.g., 'NIFTY 24500 CE')."""
        self._ensure_loaded()
        if self._df is None or self._df.empty:
            return None

        expiry_str = expiry_date.strftime("%Y-%m-%d")
        mask = (
            (self._df["name"].str.upper() == symbol.upper())
            & (self._df["strike"] == strike)
            & (self._df["expiry"].str[:10] == expiry_str)
            & (self._df["option_type"] == option_type)
        )

        matches = self._df[mask]
        if matches.empty:
            return f"{symbol} {int(strike)} {option_type}"

        return str(matches.iloc[0].get("trading_symbol", f"{symbol} {int(strike)} {option_type}"))


class DhanInstrumentResolver:
    """
    Resolves Dhan security IDs for NIFTY/BANKNIFTY options.

    Uses Dhan's security list CSV (fetched via dhanhq.fetch_security_list).
    Returns instrument keys as "NSE_FO|{security_id}" for compatibility with
    the existing order pipeline.

    Usage:
        resolver = DhanInstrumentResolver()
        resolver.refresh()
        key = resolver.get_instrument_key("NIFTY", 24500, date(2026, 3, 2), "CE")
        # → "NSE_FO|12345"
    """

    def __init__(self, cache_path: str | Path = _DHAN_CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None

    def refresh(self, force: bool = False) -> None:
        """Download Dhan security list if stale (>1 day old) or forced."""
        if not force and self.cache_path.exists():
            mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
            if mtime.date() == date.today():
                logger.debug("Dhan instrument master is fresh, loading from cache")
                self._load_cache()
                return

        logger.info("Downloading Dhan instrument master...")
        try:
            from dhanhq import dhanhq
            from src.config.env_loader import get_config

            cfg = get_config()
            token = cfg.DHAN_API_KEY
            client_id = cfg.DHAN_CLIENT_ID

            if not token or not client_id:
                logger.warning("Dhan credentials not set, trying cache fallback")
                if self.cache_path.exists():
                    self._load_cache()
                else:
                    self._df = pd.DataFrame()
                return

            dhan = dhanhq(client_id, token)
            dhan.fetch_security_list(mode="compact")

            csv_path = Path("security_id_list.csv")
            if not csv_path.exists():
                logger.warning("Dhan security CSV not found after fetch")
                if self.cache_path.exists():
                    self._load_cache()
                else:
                    self._df = pd.DataFrame()
                return

            df = pd.read_csv(csv_path)

            # Filter to NSE F&O options (NIFTY + BANKNIFTY)
            fno_mask = (
                (df["SEM_EXM_EXCH_ID"] == "NSE")
                & (df["SEM_INSTRUMENT_NAME"].isin(["OPTIDX"]))
                & (
                    df["SEM_TRADING_SYMBOL"].str.contains(
                        "NIFTY", case=False, na=False
                    )
                )
            )
            self._df = df[fno_mask].copy()

            # Normalize column names for consistent access
            self._df = self._df.rename(columns={
                "SEM_SMST_SECURITY_ID": "security_id",
                "SEM_TRADING_SYMBOL": "trading_symbol",
                "SEM_CUSTOM_SYMBOL": "custom_symbol",
                "SEM_EXPIRY_DATE": "expiry",
                "SEM_STRIKE_PRICE": "strike",
                "SEM_OPTION_TYPE": "option_type",
                "SEM_INSTRUMENT_NAME": "instrument_type",
                "SEM_LOT_UNITS": "lot_size",
            })

            # Parse expiry to date string for matching
            if "expiry" in self._df.columns:
                self._df["expiry_date"] = pd.to_datetime(
                    self._df["expiry"], format="mixed", dayfirst=False
                ).dt.strftime("%Y-%m-%d")

            # Extract underlying name (NIFTY or BANKNIFTY) from trading symbol
            self._df["name"] = self._df["trading_symbol"].apply(self._extract_name)

            # Add synthetic instrument_key column for compatibility with _fetch_fno_premiums
            self._df["instrument_key"] = "NSE_FO|" + self._df["security_id"].astype(str)

            # Save cache
            self._df.to_csv(self.cache_path, index=False)
            logger.info(
                f"Dhan instrument master cached: {len(self._df)} option instruments"
            )

            # Cleanup downloaded CSV
            csv_path.unlink(missing_ok=True)

        except ImportError:
            logger.warning("dhanhq not installed, trying cache fallback")
            if self.cache_path.exists():
                self._load_cache()
            else:
                self._df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to download Dhan instrument master: {e}")
            if self.cache_path.exists():
                logger.warning("Using stale Dhan cache as fallback")
                self._load_cache()
            else:
                self._df = pd.DataFrame()

    @staticmethod
    def _extract_name(trading_symbol: str) -> str:
        """Extract index name from trading symbol (e.g. 'BANKNIFTY...' → 'BANKNIFTY')."""
        ts = str(trading_symbol).upper()
        if ts.startswith("BANKNIFTY"):
            return "BANKNIFTY"
        if ts.startswith("NIFTY"):
            return "NIFTY"
        return ts.split("-")[0] if "-" in ts else ts[:5]

    def _load_cache(self) -> None:
        """Load cached Dhan instrument CSV."""
        try:
            self._df = pd.read_csv(self.cache_path)
            logger.debug(f"Loaded {len(self._df)} Dhan instruments from cache")
        except Exception as e:
            logger.error(f"Failed to load Dhan instrument cache: {e}")
            self._df = pd.DataFrame()

    def _ensure_loaded(self) -> None:
        if self._df is None or self._df.empty:
            self.refresh()

    def get_instrument_key(
        self,
        symbol: str,
        strike: float,
        expiry_date: date,
        option_type: str,
    ) -> Optional[str]:
        """
        Get Dhan instrument key (NSE_FO|{security_id}) for a specific option.

        Returns: "NSE_FO|12345" or None if not found
        """
        self._ensure_loaded()
        if self._df is None or self._df.empty:
            return None

        expiry_str = expiry_date.strftime("%Y-%m-%d")

        mask = (
            (self._df["name"].str.upper() == symbol.upper())
            & (self._df["strike"].astype(float) == float(strike))
            & (self._df["expiry_date"] == expiry_str)
            & (self._df["option_type"].str.upper() == option_type.upper())
        )

        matches = self._df[mask]
        if matches.empty:
            logger.warning(
                f"No Dhan instrument found: {symbol} {strike} {option_type} {expiry_str}"
            )
            return None

        sid = int(matches.iloc[0]["security_id"])
        key = f"NSE_FO|{sid}"
        logger.debug(f"Resolved (Dhan): {symbol} {strike}{option_type} → {key}")
        return key

    def get_atm_strike(self, symbol: str, spot_price: float) -> float:
        """Get ATM strike price (nearest valid strike to spot)."""
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        gap = cfg["strike_gap"]
        return round(spot_price / gap) * gap

    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for the given index."""
        self._ensure_loaded()
        if self._df is not None and not self._df.empty:
            sym_mask = self._df["name"].str.upper() == symbol.upper()
            matches = self._df[sym_mask]
            if not matches.empty and "lot_size" in matches.columns:
                try:
                    lot = int(matches.iloc[0]["lot_size"])
                    if lot > 0:
                        return lot
                except (ValueError, TypeError):
                    pass
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        return cfg["lot_size"]

    def get_weekly_expiry(self, symbol: str, ref_date: Optional[date] = None) -> date:
        """Get next weekly expiry date for the given index."""
        self._ensure_loaded()
        today = ref_date or date.today()

        if self._df is not None and not self._df.empty:
            sym_mask = self._df["name"].str.upper() == symbol.upper()
            filtered = self._df[sym_mask]
            if not filtered.empty and "expiry_date" in filtered.columns:
                expiries = pd.to_datetime(
                    filtered["expiry_date"]
                ).dt.date.unique()
                expiries = sorted(expiries)
                for exp in expiries:
                    if exp >= today:
                        return exp

        # Fallback: next Monday (NIFTY weekly expiry)
        days_ahead = (0 - today.weekday()) % 7
        if days_ahead == 0 and datetime.now().hour >= 16:
            days_ahead = 7
        if days_ahead == 0:
            return today
        return today + timedelta(days=days_ahead)

    def get_option_chain_keys(
        self,
        symbol: str,
        spot_price: float,
        num_strikes: int = 5,
        expiry_date: Optional[date] = None,
    ) -> list[dict]:
        """Get instrument keys for nearby strikes (ATM +/- num_strikes)."""
        cfg = INDEX_CONFIG.get(symbol.upper(), INDEX_CONFIG["NIFTY"])
        gap = cfg["strike_gap"]
        atm = self.get_atm_strike(symbol, spot_price)
        expiry = expiry_date or self.get_weekly_expiry(symbol)

        results = []
        for offset in range(-num_strikes, num_strikes + 1):
            strike = atm + offset * gap
            for opt_type in ("CE", "PE"):
                key = self.get_instrument_key(symbol, strike, expiry, opt_type)
                if key:
                    results.append({
                        "strike": strike,
                        "type": opt_type,
                        "instrument_key": key,
                        "expiry": expiry,
                        "otm_distance": offset if opt_type == "CE" else -offset,
                    })

        return results

    def get_trading_symbol(
        self,
        symbol: str,
        strike: float,
        expiry_date: date,
        option_type: str,
    ) -> Optional[str]:
        """Get human-readable trading symbol."""
        self._ensure_loaded()
        if self._df is None or self._df.empty:
            return f"{symbol} {int(strike)} {option_type}"

        expiry_str = expiry_date.strftime("%Y-%m-%d")
        mask = (
            (self._df["name"].str.upper() == symbol.upper())
            & (self._df["strike"].astype(float) == float(strike))
            & (self._df["expiry_date"] == expiry_str)
            & (self._df["option_type"].str.upper() == option_type.upper())
        )

        matches = self._df[mask]
        if matches.empty:
            return f"{symbol} {int(strike)} {option_type}"

        return str(matches.iloc[0].get(
            "trading_symbol", f"{symbol} {int(strike)} {option_type}"
        ))
