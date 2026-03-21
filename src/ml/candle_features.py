"""
ML Feature Engineering — 51 features from 5-min NIFTY candles + external data.

Aggregation: 5-min bars → daily session features (9:15-15:30 IST).
All features use ONLY data available at signal time (entry-contemporaneous).
Zero look-ahead: features for day T use data from days T-N to T-1 only.

Reuses: FeatureEngine.rsi(), .macd(), .bollinger_bands(), .atr(), .adx(),
        .ema(), .sma(), .volume_ratio(), .mfi(), .obv(), .vwap(), .bb_position()
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from src.utils.market_calendar import get_expiry_type
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.features import FeatureEngine
from src.data.store import DataStore

FEATURE_VERSION = 4  # Bumped: +8 PE-specific + 8 CE-specific direction features

FEATURE_NAMES = [
    # Group A: Daily Technical (16)
    "rsi_14", "rsi_7", "macd_line", "macd_histogram",
    "bb_position", "bb_width", "atr_14", "atr_pct",
    "adx_14", "ema_9_slope", "ema_21_slope", "ema_cross",
    "volume_ratio", "mfi_14", "obv_slope", "vwap_dist",
    # Group B: Returns & Momentum (6)
    "returns_1d", "returns_5d", "returns_20d",
    "volatility_5d", "volatility_20d", "range_5d_pct",
    # Group C: Intraday Session (8)
    "morning_momentum", "afternoon_strength", "intraday_range_pct",
    "bar_volatility", "volume_profile_skew", "up_bar_ratio",
    "max_drawdown_intraday", "close_vs_midrange",
    # Group D: Candlestick (4)
    "body_size", "upper_shadow", "lower_shadow", "gap_pct",
    # Group E: Market Context (6)
    "dist_from_day_open_pct", "days_to_expiry", "day_of_week",
    "week_of_month", "price_to_sma50", "price_to_ema21",
    # Group F: Normalised Context (6) — reduce overfitting
    "gap_vs_20d_avg", "first_candle_bullish", "first_candle_vol_ratio",
    "gap_direction_binary", "prev_day_return", "prev_day_range_pct",
    # Group G: External/Options (5) — option chain, VIX, FII tables
    "pcr_ratio", "vix_percentile_1y", "vix_change_1d",
    "maxpain_distance_pct", "fii_flow_direction",
]

assert len(FEATURE_NAMES) == 51

# 8 PE-specific features (fast drop detection signals)
PE_EXTRA_FEATURES = [
    "vix_spike_1d",           # VIX 1-day % change (large spike = crash likely)
    "rsi_drop_speed",         # RSI drop over 3 days (fast RSI decline)
    "volume_surge_ratio",     # Today's volume / 10-day avg (panic selling)
    "price_below_ema9",       # Close < EMA9 by % (breakdown signal)
    "red_candle_dominance",   # Fraction of bearish 5-min bars in session
    "fii_selling_streak",     # Consecutive days of net FII selling
    "dist_from_20d_high_pct", # Distance from 20-day high (% below peak)
    "intraday_reversal_down", # Morning up → afternoon down reversal strength
]

PE_FEATURE_NAMES = FEATURE_NAMES + PE_EXTRA_FEATURES

assert len(PE_FEATURE_NAMES) == 59

# 8 CE-specific features (bullish momentum / recovery signals)
CE_EXTRA_FEATURES = [
    "vix_drop_speed",         # VIX decline (fear reducing = bullish)
    "rsi_rise_speed",         # RSI 2-day rise (momentum building)
    "dii_buying_streak",      # Consecutive days of DII net buying (cap 5)
    "dist_from_20d_low_pct",  # Recovery from 20-day low (% above trough)
    "green_candle_dominance", # Fraction of bullish 5-min bars in session
    "fii_buying_streak",      # Consecutive days of FII net buying (cap 5)
    "gap_up_strength",        # Gap up opening strength (% above prev close)
    "intraday_reversal_up",   # Morning down → afternoon up reversal strength
]

CE_FEATURE_NAMES = FEATURE_NAMES + CE_EXTRA_FEATURES

assert len(CE_FEATURE_NAMES) == 59


class CandleFeatureBuilder:
    """Compute 51 ML features from 5-min candles + external data."""

    def __init__(self, store: DataStore):
        self.store = store

    def build_features(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Build full feature DataFrame (one row per trading day).

        Steps:
        1. Load 5-min candles from ml_candles_5min
        2. Aggregate to daily OHLCV session bars
        3. Compute 46 features using daily bars + intraday stats
        4. Cache results in ml_features_cache
        5. Return DataFrame (index=date, columns=46 features)
        """
        # Try cache first
        if use_cache:
            cached = self.store.get_ml_features(symbol, from_date, to_date, FEATURE_VERSION)
            if not cached.empty and len(cached) > 50:
                logger.debug(f"ML features cache hit: {len(cached)} rows for {symbol}")
                return cached

        # Load 5-min candles
        candles = self.store.get_ml_candles(symbol, from_date, to_date)
        if candles.empty or len(candles) < 100:
            logger.warning(f"Insufficient 5-min candles for {symbol}: {len(candles)}")
            return pd.DataFrame()

        candles["datetime"] = pd.to_datetime(
            candles["datetime"].astype(str).str.replace(r'\+\d{2}:\d{2}$', '', regex=True),
            format='%Y-%m-%d %H:%M:%S'
        )

        # Aggregate to daily bars
        daily = self._aggregate_daily(candles)
        if daily.empty or len(daily) < 60:
            logger.warning(f"Insufficient daily bars after aggregation: {len(daily)}")
            return pd.DataFrame()

        # Compute intraday features (per-day from raw 5min)
        intraday_feats = self._compute_intraday_features(candles)

        # Compute technical features from daily bars
        features_df = self._compute_all_features(daily)

        # Merge intraday features
        if not intraday_feats.empty:
            features_df = features_df.merge(intraday_feats, on="date", how="left")

        # Merge external features (PCR, VIX, FII, max pain)
        external_feats = self._compute_external_features(daily)
        if not external_feats.empty:
            features_df = features_df.merge(external_feats, on="date", how="left")

        # SHIFT: features for day T use data from day T-1
        # This ensures zero look-ahead
        feature_cols = [c for c in features_df.columns if c in FEATURE_NAMES]
        features_df[feature_cols] = features_df[feature_cols].shift(1)

        # Drop warmup rows (first 50 days)
        features_df = features_df.iloc[50:].reset_index(drop=True)

        # Drop rows with too many NaNs
        features_df = features_df.dropna(subset=feature_cols, thresh=len(feature_cols) - 5)

        # Cache results
        if use_cache and not features_df.empty:
            self._cache_features(symbol, features_df, feature_cols)

        logger.info(f"ML features built: {len(features_df)} rows, {len(feature_cols)} features for {symbol}")
        return features_df

    def build_features_single_day(
        self,
        symbol: str,
        target_date: str,
        lookback_days: int = 60,
    ) -> dict:
        """
        Compute features for a single day (for live prediction).

        Loads lookback_days of 5-min candles, aggregates, computes.
        Returns dict of 51 feature values (uses T-1 data).
        """
        end_dt = date.fromisoformat(target_date)
        start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))

        candles = self.store.get_ml_candles(
            symbol,
            from_date=start_dt.isoformat(),
            to_date=target_date + "T23:59:59",
        )
        if candles.empty or len(candles) < 100:
            return {}

        candles["datetime"] = pd.to_datetime(
            candles["datetime"].astype(str).str.replace(r'\+\d{2}:\d{2}$', '', regex=True),
            format='%Y-%m-%d %H:%M:%S'
        )

        daily = self._aggregate_daily(candles)
        if daily.empty or len(daily) < 30:
            return {}

        intraday_feats = self._compute_intraday_features(candles)
        features_df = self._compute_all_features(daily)

        if not intraday_feats.empty:
            features_df = features_df.merge(intraday_feats, on="date", how="left")

        # Merge external features (PCR, VIX, FII, max pain)
        external_feats = self._compute_external_features(daily)
        if not external_feats.empty:
            features_df = features_df.merge(external_feats, on="date", how="left")

        # Shift by 1 for entry-contemporaneous guarantee
        feature_cols = [c for c in features_df.columns if c in FEATURE_NAMES]
        features_df[feature_cols] = features_df[feature_cols].shift(1)

        # Get the last row (target_date)
        target_rows = features_df[features_df["date"] == target_date]
        if target_rows.empty:
            # Fall back to last available row
            if not features_df.empty:
                row = features_df.iloc[-1]
            else:
                return {}
        else:
            row = target_rows.iloc[0]

        result = {}
        for col in FEATURE_NAMES:
            val = row.get(col, np.nan)
            result[col] = float(val) if pd.notna(val) else 0.0
        return result

    def _aggregate_daily(self, candles_5min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 5-min bars to daily OHLCV + date column.

        Per day: first open, max high, min low, last close, sum volume.
        """
        df = candles_5min.copy()
        df["date"] = df["datetime"].dt.date.astype(str)

        daily = df.groupby("date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()

        daily = daily.sort_values("date").reset_index(drop=True)
        return daily

    def _compute_all_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Apply all Group A/B/D/E feature computations on daily bars."""
        df = daily.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        # ── Group A: Daily Technical (16) ──
        df["rsi_14"] = FeatureEngine.rsi(close, 14)
        df["rsi_7"] = FeatureEngine.rsi(close, 7)

        macd_line, _, macd_hist = FeatureEngine.macd(close)
        df["macd_line"] = macd_line
        df["macd_histogram"] = macd_hist

        df["bb_position"] = FeatureEngine.bb_position(close)
        upper, middle, lower = FeatureEngine.bollinger_bands(close)
        df["bb_width"] = (upper - lower) / middle.replace(0, np.nan)

        atr_14 = FeatureEngine.atr(high, low, close, 14)
        df["atr_14"] = atr_14
        df["atr_pct"] = atr_14 / close.replace(0, np.nan) * 100

        df["adx_14"] = FeatureEngine.adx(high, low, close, 14)

        ema_9 = FeatureEngine.ema(close, 9)
        ema_21 = FeatureEngine.ema(close, 21)
        df["ema_9_slope"] = (ema_9 - ema_9.shift(1)) / close.replace(0, np.nan) * 100
        df["ema_21_slope"] = (ema_21 - ema_21.shift(1)) / close.replace(0, np.nan) * 100
        df["ema_cross"] = np.sign(ema_9 - ema_21)

        df["volume_ratio"] = FeatureEngine.volume_ratio(volume, 20)
        df["mfi_14"] = FeatureEngine.mfi(high, low, close, volume, 14)

        obv = FeatureEngine.obv(close, volume)
        obv_shift = obv.shift(5).replace(0, np.nan)
        df["obv_slope"] = obv.diff(5) / obv_shift.abs()

        vwap = FeatureEngine.vwap(high, low, close, volume)
        df["vwap_dist"] = (close - vwap) / close.replace(0, np.nan) * 100

        # ── Group B: Returns & Momentum (6) ──
        df["returns_1d"] = close.pct_change(1)
        df["returns_5d"] = close.pct_change(5)
        df["returns_20d"] = close.pct_change(20)

        returns_1d = close.pct_change(1)
        df["volatility_5d"] = returns_1d.rolling(5).std()
        df["volatility_20d"] = returns_1d.rolling(20).std() * math.sqrt(252)

        rolling_5d_high = high.rolling(5).max()
        rolling_5d_low = low.rolling(5).min()
        df["range_5d_pct"] = (rolling_5d_high - rolling_5d_low) / close.replace(0, np.nan) * 100

        # ── Group D: Candlestick (4) ──
        df["body_size"] = (close - open_).abs() / open_.replace(0, np.nan) * 100
        df["upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_.replace(0, np.nan) * 100
        df["lower_shadow"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / open_.replace(0, np.nan) * 100
        df["gap_pct"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100

        # ── Group E: Market Context (6) ──
        df["dist_from_day_open_pct"] = (close - open_) / open_.replace(0, np.nan) * 100

        # Days to expiry (Thursday weekly)
        df["days_to_expiry"] = df["date"].apply(self._days_to_expiry)

        df["day_of_week"] = pd.to_datetime(df["date"]).dt.weekday
        df["week_of_month"] = pd.to_datetime(df["date"]).apply(lambda d: (d.day - 1) // 7)

        sma_50 = FeatureEngine.sma(close, 50)
        df["price_to_sma50"] = close / sma_50.replace(0, np.nan) - 1
        df["price_to_ema21"] = close / ema_21.replace(0, np.nan) - 1

        # ── Group F: Normalised Context (6) ──
        # gap_vs_20d_avg: normalised gap size
        abs_gap = df["gap_pct"].abs()
        avg_abs_gap = abs_gap.rolling(20).mean().replace(0, np.nan)
        df["gap_vs_20d_avg"] = (abs_gap / avg_abs_gap).clip(0, 5).fillna(1.0)

        # gap_direction_binary: +1 gap up, -1 gap down, 0 flat
        df["gap_direction_binary"] = np.where(
            df["gap_pct"] > 0.05, 1, np.where(df["gap_pct"] < -0.05, -1, 0)
        )

        # prev_day_return: yesterday's return as context
        df["prev_day_return"] = close.pct_change(1).shift(1) * 100

        # prev_day_range_pct: yesterday's high-low range
        day_range = (high - low) / open_.replace(0, np.nan) * 100
        df["prev_day_range_pct"] = day_range.shift(1)

        return df

    def _compute_intraday_features(self, candles_5min: pd.DataFrame) -> pd.DataFrame:
        """
        Intraday-specific features computed from raw 5-min bars.

        Per day: morning_momentum, afternoon_strength, intraday_range_pct,
        bar_volatility, volume_profile_skew, up_bar_ratio,
        max_drawdown_intraday, close_vs_midrange.
        """
        df = candles_5min.copy()
        df["date"] = df["datetime"].dt.date.astype(str)
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["time_decimal"] = df["hour"] + df["minute"] / 60.0
        df["bar_return"] = df.groupby("date")["close"].pct_change()

        records = []
        for dt, group in df.groupby("date"):
            if len(group) < 5:
                continue

            rec = {"date": dt}

            # Morning momentum: first hour return (9:15-10:15)
            morning = group[group["time_decimal"] < 10.25]
            if len(morning) >= 2:
                rec["morning_momentum"] = (morning["close"].iloc[-1] - morning["open"].iloc[0]) / morning["open"].iloc[0] * 100
            else:
                rec["morning_momentum"] = 0.0

            # Afternoon strength: last 2 hours return (13:30-15:30)
            afternoon = group[group["time_decimal"] >= 13.5]
            if len(afternoon) >= 2:
                rec["afternoon_strength"] = (afternoon["close"].iloc[-1] - afternoon["open"].iloc[0]) / afternoon["open"].iloc[0] * 100
            else:
                rec["afternoon_strength"] = 0.0

            # Intraday range %
            day_high = group["high"].max()
            day_low = group["low"].min()
            day_open = group["open"].iloc[0]
            rec["intraday_range_pct"] = (day_high - day_low) / day_open * 100 if day_open > 0 else 0.0

            # Bar volatility: std of 5-min returns
            bar_returns = group["bar_return"].dropna()
            rec["bar_volatility"] = float(bar_returns.std()) if len(bar_returns) > 1 else 0.0

            # Volume profile skew: morning / afternoon
            morning_vol = group[group["time_decimal"] < 12.0]["volume"].sum()
            afternoon_vol = group[group["time_decimal"] >= 12.0]["volume"].sum()
            rec["volume_profile_skew"] = morning_vol / max(afternoon_vol, 1)

            # Up bar ratio
            up_bars = (group["close"] > group["open"]).sum()
            rec["up_bar_ratio"] = up_bars / len(group)

            # Max drawdown intraday
            cum_high = group["high"].cummax()
            drawdowns = (group["low"] - cum_high) / cum_high * 100
            rec["max_drawdown_intraday"] = float(drawdowns.min())

            # Close vs midrange
            day_close = group["close"].iloc[-1]
            day_range = day_high - day_low
            if day_range > 0:
                rec["close_vs_midrange"] = (day_close - (day_high + day_low) / 2) / day_range
            else:
                rec["close_vs_midrange"] = 0.0

            # First candle bullish (1 if first 5-min close > open, else 0)
            rec["first_candle_bullish"] = 1.0 if group["close"].iloc[0] > group["open"].iloc[0] else 0.0

            # First candle volume ratio vs 20-day average (filled later via rolling)
            rec["first_candle_volume"] = float(group["volume"].iloc[0])

            records.append(rec)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)

        # Compute first_candle_vol_ratio as rolling 20-day average
        avg_first_vol = result["first_candle_volume"].rolling(20).mean().replace(0, np.nan)
        result["first_candle_vol_ratio"] = (result["first_candle_volume"] / avg_first_vol).clip(0, 5).fillna(1.0)
        result = result.drop(columns=["first_candle_volume"])

        return result

    def _compute_external_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Group G features from external data (option chain, VIX, FII).

        Returns DataFrame with columns: [date, pcr_ratio, vix_percentile_1y,
        vix_change_1d, maxpain_distance_pct, fii_flow_direction].
        All missing values filled with neutral defaults.
        """
        dates = daily[["date"]].copy()
        n_days = max(len(daily) + 60, 400)

        # ── PCR ratio from option chain ──
        try:
            oc_atm = self.store.get_option_chain_atm_history(days=n_days)
            if not oc_atm.empty and "pcr_oi" in oc_atm.columns:
                pcr_df = oc_atm[["date", "pcr_oi"]].rename(columns={"pcr_oi": "pcr_ratio"})
                pcr_df["date"] = pcr_df["date"].astype(str).str[:10]
                dates = dates.merge(pcr_df, on="date", how="left")
            else:
                dates["pcr_ratio"] = np.nan
        except Exception:
            dates["pcr_ratio"] = np.nan

        # ── VIX percentile (1y) + VIX change (1d) ──
        try:
            vix_df = self.store.get_external_data("INDIA_VIX")
            if not vix_df.empty and "close" in vix_df.columns:
                vix_df = vix_df[["date", "close"]].copy()
                vix_df["date"] = vix_df["date"].astype(str).str[:10]
                vix_df = vix_df.sort_values("date").drop_duplicates("date", keep="last")
                vix_df["vix_percentile_1y"] = vix_df["close"].rolling(252, min_periods=20).rank(pct=True)
                vix_df["vix_change_1d"] = vix_df["close"].pct_change(1)
                vix_merge = vix_df[["date", "vix_percentile_1y", "vix_change_1d"]]
                dates = dates.merge(vix_merge, on="date", how="left")
            else:
                dates["vix_percentile_1y"] = np.nan
                dates["vix_change_1d"] = np.nan
        except Exception:
            dates["vix_percentile_1y"] = np.nan
            dates["vix_change_1d"] = np.nan

        # ── Max pain distance from option chain strikes ──
        try:
            oc_raw = self.store.get_option_chain_history(days=n_days)
            if not oc_raw.empty:
                mp_rows = []
                for dt, group in oc_raw.groupby("date"):
                    if group.empty or group["underlying_value"].iloc[0] <= 0:
                        continue
                    underlying = group["underlying_value"].iloc[0]
                    strikes = group["strike_price"].values
                    ce_oi = group["ce_oi"].fillna(0).values
                    pe_oi = group["pe_oi"].fillna(0).values

                    # Max pain = strike that minimises total pain to option writers
                    min_pain = float("inf")
                    max_pain_strike = underlying
                    for j, s in enumerate(strikes):
                        # Pain to call writers + pain to put writers at expiry price s
                        pain = 0.0
                        for k in range(len(strikes)):
                            pain += max(0, s - strikes[k]) * float(ce_oi[k])
                            pain += max(0, strikes[k] - s) * float(pe_oi[k])
                        if pain < min_pain:
                            min_pain = pain
                            max_pain_strike = s

                    mp_dist = (underlying - max_pain_strike) / underlying
                    mp_rows.append({"date": str(dt)[:10], "maxpain_distance_pct": round(mp_dist, 6)})

                if mp_rows:
                    mp_df = pd.DataFrame(mp_rows)
                    dates = dates.merge(mp_df, on="date", how="left")
                else:
                    dates["maxpain_distance_pct"] = np.nan
            else:
                dates["maxpain_distance_pct"] = np.nan
        except Exception:
            dates["maxpain_distance_pct"] = np.nan

        # ── FII flow direction (5-day cumulative) ──
        try:
            fii_df = self.store.get_fii_dii_history(days=n_days)
            if not fii_df.empty and "fii_net_value" in fii_df.columns:
                fii_df = fii_df.copy()
                fii_df["date"] = fii_df["date"].astype(str).str[:10]
                fii_df = fii_df.sort_values("date").drop_duplicates("date", keep="last")
                fii_5d = fii_df["fii_net_value"].rolling(5, min_periods=1).sum()
                fii_df["fii_flow_direction"] = np.where(
                    fii_5d > 500, 1.0, np.where(fii_5d < -500, -1.0, 0.0)
                )
                fii_merge = fii_df[["date", "fii_flow_direction"]]
                dates = dates.merge(fii_merge, on="date", how="left")
            else:
                dates["fii_flow_direction"] = np.nan
        except Exception:
            dates["fii_flow_direction"] = np.nan

        # ── Fill defaults for missing data ──
        defaults = {
            "pcr_ratio": 1.0,
            "vix_percentile_1y": 0.5,
            "vix_change_1d": 0.0,
            "maxpain_distance_pct": 0.0,
            "fii_flow_direction": 0.0,
        }
        for col, default in defaults.items():
            if col in dates.columns:
                dates[col] = dates[col].fillna(default)
            else:
                dates[col] = default

        return dates[["date"] + list(defaults.keys())]

    def compute_pe_specific_features(
        self, candles_5min: pd.DataFrame, daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute 8 PE-specific features for crash/fast-drop prediction.

        All features use T-1 data (shifted by 1 day in build_features).
        Returns DataFrame with date + 8 PE feature columns.
        """
        df5 = candles_5min.copy()
        df5["date"] = df5["datetime"].dt.date.astype(str)
        dd = daily.copy()
        close = dd["close"]
        high = dd["high"]

        result = dd[["date"]].copy()

        # 1. VIX spike (1-day % change) — from external data
        try:
            vix_df = self.store.get_external_data("INDIA_VIX")
            if not vix_df.empty and "close" in vix_df.columns:
                vix = vix_df[["date", "close"]].copy()
                vix["date"] = vix["date"].astype(str).str[:10]
                vix = vix.sort_values("date").drop_duplicates("date", keep="last")
                vix["vix_spike_1d"] = vix["close"].pct_change(1)
                result = result.merge(vix[["date", "vix_spike_1d"]], on="date", how="left")
            else:
                result["vix_spike_1d"] = 0.0
        except Exception:
            result["vix_spike_1d"] = 0.0

        # 2. RSI drop speed (3-day RSI decline)
        rsi_14 = FeatureEngine.rsi(close, 14)
        result["rsi_drop_speed"] = rsi_14.diff(3)  # negative = dropping fast

        # 3. Volume surge ratio (today vs 10-day avg)
        vol = dd["volume"]
        vol_10d_avg = vol.rolling(10).mean().replace(0, np.nan)
        result["volume_surge_ratio"] = (vol / vol_10d_avg).clip(0, 10).fillna(1.0)

        # 4. Price below EMA9 (% below)
        ema_9 = FeatureEngine.ema(close, 9)
        result["price_below_ema9"] = ((close - ema_9) / ema_9.replace(0, np.nan) * 100).fillna(0.0)

        # 5. Red candle dominance (fraction of bearish 5-min bars per day)
        red_dom = {}
        for dt, group in df5.groupby("date"):
            n = len(group)
            if n > 0:
                red_bars = (group["close"] < group["open"]).sum()
                red_dom[dt] = red_bars / n
            else:
                red_dom[dt] = 0.5
        result["red_candle_dominance"] = result["date"].map(red_dom).fillna(0.5)

        # 6. FII selling streak (consecutive days of net selling)
        try:
            fii_df = self.store.get_fii_dii_history(days=max(len(dd) + 60, 400))
            if not fii_df.empty and "fii_net_value" in fii_df.columns:
                fii = fii_df[["date", "fii_net_value"]].copy()
                fii["date"] = fii["date"].astype(str).str[:10]
                fii = fii.sort_values("date").drop_duplicates("date", keep="last")
                # Compute selling streak
                streaks = []
                streak = 0
                for val in fii["fii_net_value"].values:
                    if val < 0:
                        streak += 1
                    else:
                        streak = 0
                    streaks.append(streak)
                fii["fii_selling_streak"] = streaks
                result = result.merge(fii[["date", "fii_selling_streak"]], on="date", how="left")
            else:
                result["fii_selling_streak"] = 0.0
        except Exception:
            result["fii_selling_streak"] = 0.0

        # 7. Distance from 20-day high (% below peak)
        high_20d = high.rolling(20).max()
        result["dist_from_20d_high_pct"] = ((close - high_20d) / high_20d.replace(0, np.nan) * 100).fillna(0.0)

        # 8. Intraday reversal down (morning up → afternoon down)
        reversal = {}
        for dt, group in df5.groupby("date"):
            td = group["datetime"].dt.hour + group["datetime"].dt.minute / 60.0
            morning = group[td < 10.25]
            afternoon = group[td >= 13.5]
            if len(morning) >= 2 and len(afternoon) >= 2:
                morn_ret = (morning["close"].iloc[-1] - morning["open"].iloc[0]) / morning["open"].iloc[0]
                aftn_ret = (afternoon["close"].iloc[-1] - afternoon["open"].iloc[0]) / afternoon["open"].iloc[0]
                # Reversal: morning up but afternoon down → positive value = bearish reversal
                reversal[dt] = max(0, morn_ret) * 100 + min(0, aftn_ret) * 100
            else:
                reversal[dt] = 0.0
        result["intraday_reversal_down"] = result["date"].map(reversal).fillna(0.0)

        # Fill remaining NaN
        for col in PE_EXTRA_FEATURES:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
            else:
                result[col] = 0.0

        return result[["date"] + PE_EXTRA_FEATURES]

    def compute_ce_specific_features(
        self, candles_5min: pd.DataFrame, daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute 8 CE-specific features for bullish momentum / recovery prediction.

        All features use T-1 data (shifted by 1 day in build_features).
        Returns DataFrame with date + 8 CE feature columns.
        """
        df5 = candles_5min.copy()
        df5["date"] = df5["datetime"].dt.date.astype(str)
        dd = daily.copy()
        close = dd["close"]
        low = dd["low"]
        open_ = dd["open"]

        result = dd[["date"]].copy()

        # 1. VIX drop speed (VIX falling = fear reducing = bullish)
        try:
            vix_df = self.store.get_external_data("INDIA_VIX")
            if not vix_df.empty and "close" in vix_df.columns:
                vix = vix_df[["date", "close"]].copy()
                vix["date"] = vix["date"].astype(str).str[:10]
                vix = vix.sort_values("date").drop_duplicates("date", keep="last")
                vix["vix_drop_speed"] = -vix["close"].pct_change(1)  # positive = VIX falling
                result = result.merge(vix[["date", "vix_drop_speed"]], on="date", how="left")
            else:
                result["vix_drop_speed"] = 0.0
        except Exception:
            result["vix_drop_speed"] = 0.0

        # 2. RSI rise speed (2-day RSI increase = momentum building)
        rsi_14 = FeatureEngine.rsi(close, 14)
        result["rsi_rise_speed"] = rsi_14.diff(2)  # positive = RSI rising

        # 3. DII buying streak (consecutive days of net DII buying, cap 5)
        try:
            fii_df = self.store.get_fii_dii_history(days=max(len(dd) + 60, 400))
            if not fii_df.empty and "dii_net_value" in fii_df.columns:
                dii = fii_df[["date", "dii_net_value"]].copy()
                dii["date"] = dii["date"].astype(str).str[:10]
                dii = dii.sort_values("date").drop_duplicates("date", keep="last")
                streaks = []
                streak = 0
                for val in dii["dii_net_value"].values:
                    if val > 0:
                        streak = min(streak + 1, 5)
                    else:
                        streak = 0
                    streaks.append(streak)
                dii["dii_buying_streak"] = streaks
                result = result.merge(dii[["date", "dii_buying_streak"]], on="date", how="left")
            else:
                result["dii_buying_streak"] = 0.0
        except Exception:
            result["dii_buying_streak"] = 0.0

        # 4. Distance from 20-day low (% above trough = recovery)
        low_20d = low.rolling(20).min()
        result["dist_from_20d_low_pct"] = ((close - low_20d) / low_20d.replace(0, np.nan) * 100).fillna(0.0)

        # 5. Green candle dominance (fraction of bullish 5-min bars per day)
        green_dom = {}
        for dt, group in df5.groupby("date"):
            n = len(group)
            if n > 0:
                green_bars = (group["close"] > group["open"]).sum()
                green_dom[dt] = green_bars / n
            else:
                green_dom[dt] = 0.5
        result["green_candle_dominance"] = result["date"].map(green_dom).fillna(0.5)

        # 6. FII buying streak (consecutive days of net FII buying, cap 5)
        try:
            fii_df2 = self.store.get_fii_dii_history(days=max(len(dd) + 60, 400))
            if not fii_df2.empty and "fii_net_value" in fii_df2.columns:
                fii = fii_df2[["date", "fii_net_value"]].copy()
                fii["date"] = fii["date"].astype(str).str[:10]
                fii = fii.sort_values("date").drop_duplicates("date", keep="last")
                streaks = []
                streak = 0
                for val in fii["fii_net_value"].values:
                    if val > 0:
                        streak = min(streak + 1, 5)
                    else:
                        streak = 0
                    streaks.append(streak)
                fii["fii_buying_streak"] = streaks
                result = result.merge(fii[["date", "fii_buying_streak"]], on="date", how="left")
            else:
                result["fii_buying_streak"] = 0.0
        except Exception:
            result["fii_buying_streak"] = 0.0

        # 7. Gap up strength (% above prev close, floored at 0)
        prev_close = close.shift(1)
        gap_up_raw = (open_ - prev_close) / prev_close.replace(0, np.nan) * 100
        result["gap_up_strength"] = gap_up_raw.clip(lower=0).fillna(0.0)

        # 8. Intraday reversal up (morning down → afternoon up)
        reversal = {}
        for dt, group in df5.groupby("date"):
            td = group["datetime"].dt.hour + group["datetime"].dt.minute / 60.0
            morning = group[td < 10.25]
            afternoon = group[td >= 13.5]
            if len(morning) >= 2 and len(afternoon) >= 2:
                morn_ret = (morning["close"].iloc[-1] - morning["open"].iloc[0]) / morning["open"].iloc[0]
                aftn_ret = (afternoon["close"].iloc[-1] - afternoon["open"].iloc[0]) / afternoon["open"].iloc[0]
                # Reversal: morning down but afternoon up → positive value = bullish reversal
                reversal[dt] = min(0, morn_ret) * -100 + max(0, aftn_ret) * 100
            else:
                reversal[dt] = 0.0
        result["intraday_reversal_up"] = result["date"].map(reversal).fillna(0.0)

        # Fill remaining NaN
        for col in CE_EXTRA_FEATURES:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
            else:
                result[col] = 0.0

        return result[["date"] + CE_EXTRA_FEATURES]

    @staticmethod
    def _days_to_expiry(date_str: str) -> int:
        """Compute calendar days until next NIFTY weekly expiry.

        Uses market_calendar.get_expiry_type() for dual-schedule logic:
          Pre Sep-2025: Thursday expiry
          Post Sep-2025: Tuesday expiry
        """
        dt = date.fromisoformat(date_str) if isinstance(date_str, str) else date_str
        for offset in range(0, 8):
            check = dt + timedelta(days=offset)
            if get_expiry_type(check) == "NIFTY_EXPIRY":
                return offset
        return 0

    def compute_direction_labels(self, daily_df: pd.DataFrame) -> pd.Series:
        """
        2-class label: CE (1), PE (0).

        CE: next_day_close >= next_day_open (NIFTY rose)
        PE: next_day_close <  next_day_open (NIFTY fell)

        Every trading day gets a label. No FLAT class.
        Random baseline = 50%.

        IMPORTANT: Label uses NEXT day data (shift -1).
        Features use CURRENT day and prior data only.
        """
        next_open = daily_df["open"].shift(-1)
        next_close = daily_df["close"].shift(-1)

        # CE=1 if rose, PE=0 if fell
        labels = (next_close >= next_open).astype(int)
        return labels

    def _cache_features(self, symbol: str, features_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Save features to cache."""
        for _, row in features_df.iterrows():
            feat_dict = {col: float(row[col]) if pd.notna(row.get(col)) else 0.0 for col in feature_cols}
            try:
                self.store.save_ml_features(symbol, row["date"], feat_dict, FEATURE_VERSION)
            except Exception as e:
                logger.warning(f"CACHE_WRITE_FAILED: {e}")

    def get_feature_names(self) -> list[str]:
        """Return ordered list of all 46 feature names."""
        return FEATURE_NAMES.copy()
