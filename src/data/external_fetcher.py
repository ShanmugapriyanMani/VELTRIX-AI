"""
External Data Fetcher — non-Upstox data sources.

Handles:
- Local NIFTY CSVs (5 years from NSE website downloads)
- FII/DII bulk CSV + daily via nsepython
- External markets via yfinance (S&P 500, NASDAQ, Crude, Gold, USD/INR)
- VIX extended history via nsepython

All methods are fail-safe: log warnings on failure, never crash.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path


import pandas as pd
from loguru import logger

from src.data.store import DataStore

# Optional dependencies — graceful degradation
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from nsepython import nse_fiidii, index_history

    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False


# yfinance ticker → our symbol mapping
EXTERNAL_TICKERS = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "CL=F": "CRUDE_OIL",
    "GC=F": "GOLD",
    "USDINR=X": "USDINR",
}


class ExternalDataFetcher:
    """Fetches data from non-Upstox sources: local CSVs, yfinance, nsepython."""

    def __init__(self, store: DataStore):
        self._store = store
        self._nse_last_call: float = 0
        self._nse_rate_limit_sec: float = 2.0

    # ─────────────────────────────────────────
    # 1. Local NIFTY CSVs
    # ─────────────────────────────────────────

    def load_local_nifty_csvs(self, csv_dir: str = "data/nifty_data") -> int:
        """
        Load local NIFTY 50 CSVs (downloaded from NSE website) into the candles table.

        CSV format (NSE):
          Date ,Open ,High ,Low ,Close ,Shares Traded ,Turnover (₹ Cr)
          21-FEB-2022,17192.25,17351.05,17070.7,17206.65,215183301,18725.57

        Returns total rows loaded.
        """
        csv_path = Path(csv_dir)
        if not csv_path.exists():
            logger.warning(f"NIFTY CSV directory not found: {csv_dir}")
            return 0

        csv_files = sorted(csv_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {csv_dir}")
            return 0

        total_rows = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Strip whitespace from column names (NSE CSVs have trailing spaces)
                df.columns = [c.strip() for c in df.columns]

                # Parse NSE date format: "21-FEB-2022"
                df["datetime"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
                df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

                # Map columns
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Shares Traded": "volume",
                    }
                )

                # Build candle DataFrame
                candle_df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
                candle_df["oi"] = 0

                # Remove any rows with NaN
                candle_df = candle_df.dropna(subset=["open", "high", "low", "close"])

                if candle_df.empty:
                    continue

                self._store.save_candles(
                    symbol="NIFTY50",
                    instrument_key="NSE_INDEX|Nifty 50",
                    df=candle_df,
                    interval="day",
                )
                total_rows += len(candle_df)
                logger.info(f"  Loaded {len(candle_df)} rows from {csv_file.name}")

            except Exception as e:
                logger.warning(f"  Failed to load {csv_file.name}: {e}")

        return total_rows

    # ─────────────────────────────────────────
    # 2. FII/DII Data
    # ─────────────────────────────────────────

    def load_fii_dii_csv(self, csv_path: str = "data/fii_dii_bulk.csv") -> int:
        """
        Load FII/DII data from a user-downloaded NSE bulk CSV.

        Expected format (header): date,fii_buy,fii_sell,fii_net,dii_buy,dii_sell,dii_net
        Also handles NSE website format with flexible column names.

        NOTE: dii_buy/dii_sell columns contain Index Options data (very large numbers).
        For ML, fii_net is the primary feature. dii_net is secondary (F&O data).

        Incremental: skips dates already in DB with real (non-zero) data.
        Returns NEW rows inserted (0 if file not found or all rows already exist).
        """
        path = Path(csv_path)
        if not path.exists():
            logger.info(f"  FII/DII CSV not found: {csv_path} (place NSE bulk download here)")
            return 0

        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]

            # Normalize column names to lowercase for flexible matching
            col_map = {c: c.lower().replace("/", "_").replace(" ", "_") for c in df.columns}
            df = df.rename(columns=col_map)
            cols = list(df.columns)

            # Find date column
            date_col = next((c for c in cols if "date" in c), None)
            if not date_col:
                logger.warning("FII/DII CSV: no 'date' column found")
                return 0

            # Find FII/DII columns
            fii_buy = next((c for c in cols if "fii" in c and "buy" in c), None)
            fii_sell = next((c for c in cols if "fii" in c and "sell" in c), None)
            fii_net = next((c for c in cols if "fii" in c and "net" in c), None)
            dii_buy = next((c for c in cols if "dii" in c and "buy" in c), None)
            dii_sell = next((c for c in cols if "dii" in c and "sell" in c), None)
            dii_net = next((c for c in cols if "dii" in c and "net" in c), None)

            if not fii_net:
                logger.warning("FII/DII CSV: could not find FII net column")
                return 0

            def _to_float(val) -> float:
                if pd.isna(val):
                    return 0.0
                s = str(val).replace(",", "").replace("(", "-").replace(")", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return 0.0

            # Check which dates already have real data in DB
            existing_cov = self._store.get_fii_dii_coverage()
            existing_dates: set[str] = set()
            if existing_cov["rows"] > 0:
                existing_df = self._store.get_fii_dii_history(days=9999)
                if not existing_df.empty:
                    # Only count dates with real (non-zero) data
                    real = existing_df[
                        (existing_df["fii_net_value"].abs() > 0)
                        | (existing_df["dii_net_value"].abs() > 0)
                    ]
                    existing_dates = set(
                        pd.to_datetime(real["date"]).dt.strftime("%Y-%m-%d")
                    )

            records = []
            skipped = 0
            for _, row in df.iterrows():
                try:
                    # Parse date (try multiple formats)
                    raw_date = str(row[date_col]).strip()
                    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y"):
                        try:
                            parsed = pd.to_datetime(raw_date, format=fmt)
                            break
                        except (ValueError, TypeError):
                            continue
                    else:
                        try:
                            parsed = pd.to_datetime(raw_date)
                        except Exception:
                            continue

                    date_str = parsed.strftime("%Y-%m-%d")

                    # Skip if already in DB with real data
                    if date_str in existing_dates:
                        skipped += 1
                        continue

                    records.append({
                        "date": date_str,
                        "fii_buy_value": _to_float(row.get(fii_buy, 0)),
                        "fii_sell_value": _to_float(row.get(fii_sell, 0)),
                        "fii_net_value": _to_float(row.get(fii_net, 0)),
                        "dii_buy_value": _to_float(row.get(dii_buy, 0)),
                        "dii_sell_value": _to_float(row.get(dii_sell, 0)),
                        "dii_net_value": _to_float(row.get(dii_net, 0)),
                    })
                except Exception:
                    continue

            if records:
                count = self._store.save_fii_dii_bulk(records)
                date_min = records[0]["date"]
                date_max = records[-1]["date"]
                logger.info(
                    f"  FII/DII CSV: loaded {count} rows "
                    f"({date_min} to {date_max})"
                )
                if skipped > 0:
                    logger.info(f"  FII/DII CSV: {skipped} rows skipped (already in DB)")
                return count
            elif skipped > 0:
                logger.info(f"  FII/DII CSV: all {skipped} rows already in DB, skipping")
                return 0
            else:
                logger.warning("FII/DII CSV: no valid records parsed")
                return 0

        except Exception as e:
            logger.error(f"  Failed to load FII/DII CSV: {e}")
            return 0

    def fetch_fii_dii_today(self) -> int:
        """
        Fetch latest FII/DII data via nsepython (nse_fiidii → nseindia.com/api/fiidiiTradeReact).

        NSE publishes previous trading day's FII/DII data by 6-8 PM.
        So a morning fetch at 8:30 AM gets yesterday's data.

        Uses the actual date from the API response (not today's date).
        Skips if that date's real data already exists in DB.

        Returns 1 if saved, 0 if unavailable/skipped/failed.
        """
        if not NSE_AVAILABLE:
            logger.debug("nsepython not installed — skipping FII/DII daily")
            return 0

        try:
            self._nse_rate_limit()
            data = nse_fiidii()

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                logger.debug("No FII/DII data returned from NSE")
                return 0

            # nse_fiidii() returns DataFrame with columns:
            # category, date, buyValue, sellValue, netValue
            if isinstance(data, pd.DataFrame):
                fii_row = data[data["category"].str.contains("FII|FPI", case=False, na=False)]
                dii_row = data[data["category"].str.contains("DII", case=False, na=False)]

                # Extract actual date from API response (e.g., "27-Feb-2026")
                api_date_str = None
                for _, row in data.iterrows():
                    raw_date = row.get("date", "")
                    if raw_date:
                        try:
                            api_date_str = pd.to_datetime(raw_date, dayfirst=True).strftime("%Y-%m-%d")
                        except Exception:
                            pass
                        break

                if not api_date_str:
                    logger.warning("FII/DII: could not parse date from API response")
                    return 0

                # Skip if real data for this date already exists
                if self._store.has_fii_dii_for_date(api_date_str):
                    logger.info(f"  FII/DII: {api_date_str} already in DB, skipping")
                    return 0

                def _safe_float(df_row: pd.DataFrame, col: str) -> float:
                    if df_row.empty:
                        return 0.0
                    val = df_row.iloc[0].get(col, 0)
                    try:
                        return float(str(val).replace(",", ""))
                    except (ValueError, TypeError):
                        return 0.0

                record = {
                    "date": api_date_str,
                    "fii_buy_value": _safe_float(fii_row, "buyValue"),
                    "fii_sell_value": _safe_float(fii_row, "sellValue"),
                    "fii_net_value": _safe_float(fii_row, "netValue"),
                    "dii_buy_value": _safe_float(dii_row, "buyValue"),
                    "dii_sell_value": _safe_float(dii_row, "sellValue"),
                    "dii_net_value": _safe_float(dii_row, "netValue"),
                }
                self._store.save_fii_dii(record)
                logger.info(
                    f"  FII/DII saved for {api_date_str}: "
                    f"FII={record['fii_net_value']:+,.0f}cr, "
                    f"DII={record['dii_net_value']:+,.0f}cr"
                )
                return 1

        except Exception as e:
            logger.warning(f"  Failed to fetch FII/DII: {e}")

        return 0

    # ─────────────────────────────────────────
    # 3. External Markets (yfinance)
    # ─────────────────────────────────────────

    def fetch_external_markets(self, days: int = 1825, force: bool = False) -> int:
        """
        Fetch external market data via yfinance: S&P 500, NASDAQ, Crude, Gold, USD/INR.

        Incremental: only downloads from last available date per symbol.
        Use force=True to re-fetch everything.
        Returns total rows saved.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not installed — skipping external market data")
            return 0

        total_rows = 0
        start_date = (date.today() - timedelta(days=days)).isoformat()

        for ticker, symbol in EXTERNAL_TICKERS.items():
            try:
                # Check existing coverage
                coverage = self._store.get_external_data_coverage(symbol)
                if not force and coverage["rows"] > 0 and coverage["to_date"]:
                    if coverage["to_date"] >= (date.today() - timedelta(days=2)).isoformat():
                        logger.info(f"  {symbol}: up-to-date ({coverage['to_date']}), skipping")
                        continue
                    # Incremental: only fetch from last date
                    fetch_start = coverage["to_date"]
                    logger.info(f"  {symbol}: incremental from {fetch_start}")
                else:
                    fetch_start = start_date
                    logger.info(f"  {symbol}: full fetch from {fetch_start}")

                # Download from yfinance
                data = yf.download(
                    ticker,
                    start=fetch_start,
                    end=date.today().isoformat(),
                    progress=False,
                    auto_adjust=True,
                )

                if data is None or data.empty:
                    logger.warning(f"  {symbol}: no data from yfinance")
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Build DataFrame for storage
                df = pd.DataFrame({
                    "date": data.index.strftime("%Y-%m-%d"),
                    "open": data["Open"].values,
                    "high": data["High"].values,
                    "low": data["Low"].values,
                    "close": data["Close"].values,
                    "volume": data["Volume"].fillna(0).astype(int).values,
                })

                rows = self._store.save_external_data(symbol, df)
                total_rows += rows
                logger.info(f"  {symbol}: {rows} rows saved")

            except Exception as e:
                logger.warning(f"  {symbol}: failed — {e}")

        return total_rows

    # ─────────────────────────────────────────
    # 4. VIX Extended History (nsepython)
    # ─────────────────────────────────────────

    def fetch_vix_history_nse(self, days: int = 1825, force: bool = False) -> int:
        """
        Fetch India VIX history from NSE for dates older than Upstox range.

        Uses nsepython.index_history("INDIA VIX", from_date, to_date).
        Only fetches dates NOT already in DB (Upstox data takes priority).
        Use force=True to re-fetch everything.

        Returns candle count saved.
        """
        if not NSE_AVAILABLE:
            logger.warning("nsepython not installed — skipping VIX extended history")
            return 0

        # Check existing VIX coverage
        coverage = self._store.get_data_coverage("INDIA_VIX")
        existing_from = coverage.get("from_date", "")
        existing_to = coverage.get("to_date", "")

        # Target: 5 years back
        target_from = (date.today() - timedelta(days=days)).strftime("%d-%m-%Y")
        target_to = date.today().strftime("%d-%m-%Y")

        # If we already have data going back far enough AND up-to-date, skip
        if not force and existing_from and existing_to:
            covers_back = existing_from <= (date.today() - timedelta(days=days - 30)).isoformat()
            covers_recent = existing_to >= (date.today() - timedelta(days=2)).isoformat()
            if covers_back and covers_recent:
                logger.info(
                    f"  VIX history: up-to-date ({existing_from} to {existing_to}), skipping"
                )
                return 0
            elif covers_back:
                logger.info(f"  VIX history already covers from {existing_from} — skipping")
                return 0

        # Fetch in yearly chunks to avoid NSE rate limits
        total_rows = 0
        current_end = date.today()
        chunk_days = 365

        for _ in range(5):  # Max 5 yearly chunks
            chunk_start = current_end - timedelta(days=chunk_days)
            if chunk_start < date.today() - timedelta(days=days):
                chunk_start = date.today() - timedelta(days=days)

            from_str = chunk_start.strftime("%d-%m-%Y")
            to_str = current_end.strftime("%d-%m-%Y")

            try:
                self._nse_rate_limit()
                vix_data = index_history("INDIA VIX", from_str, to_str)

                if vix_data is None or (isinstance(vix_data, pd.DataFrame) and vix_data.empty):
                    logger.debug(f"  VIX: no data for {from_str} to {to_str}")
                    current_end = chunk_start
                    continue

                if isinstance(vix_data, pd.DataFrame):
                    # NSE returns: Date, Open, High, Low, Close, Volume, Turnover
                    vix_data.columns = [c.strip() for c in vix_data.columns]

                    # Parse dates
                    date_col = next(
                        (c for c in vix_data.columns if "date" in c.lower()),
                        vix_data.columns[0],
                    )
                    vix_data["datetime"] = pd.to_datetime(vix_data[date_col]).dt.strftime("%Y-%m-%d")

                    col_map = {}
                    for c in vix_data.columns:
                        cl = c.lower().strip()
                        if cl == "open":
                            col_map[c] = "open"
                        elif cl == "high":
                            col_map[c] = "high"
                        elif cl == "low":
                            col_map[c] = "low"
                        elif cl == "close":
                            col_map[c] = "close"
                        elif "volume" in cl:
                            col_map[c] = "volume"
                    vix_data = vix_data.rename(columns=col_map)

                    candle_df = pd.DataFrame({
                        "datetime": vix_data["datetime"],
                        "open": pd.to_numeric(vix_data.get("open", 0), errors="coerce").fillna(0),
                        "high": pd.to_numeric(vix_data.get("high", 0), errors="coerce").fillna(0),
                        "low": pd.to_numeric(vix_data.get("low", 0), errors="coerce").fillna(0),
                        "close": pd.to_numeric(vix_data.get("close", 0), errors="coerce").fillna(0),
                        "volume": 0,
                        "oi": 0,
                    })
                    candle_df = candle_df.dropna(subset=["close"])
                    candle_df = candle_df[candle_df["close"] > 0]

                    if not candle_df.empty:
                        self._store.save_candles(
                            symbol="INDIA_VIX",
                            instrument_key="NSE_INDEX|India VIX",
                            df=candle_df,
                            interval="day",
                        )
                        total_rows += len(candle_df)
                        logger.info(f"  VIX: {len(candle_df)} rows for {from_str} to {to_str}")

            except Exception as e:
                logger.warning(f"  VIX fetch failed ({from_str} to {to_str}): {e}")

            current_end = chunk_start
            if chunk_start <= date.today() - timedelta(days=days):
                break

        return total_rows

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _nse_rate_limit(self) -> None:
        """Enforce 2-second gap between NSE API calls."""
        elapsed = time.time() - self._nse_last_call
        if elapsed < self._nse_rate_limit_sec:
            time.sleep(self._nse_rate_limit_sec - elapsed)
        self._nse_last_call = time.time()
