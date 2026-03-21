"""
Feature Engineering — Technical indicators + India-specific alternative data.

Combines traditional technical analysis with India-specific alpha signals
(FII flows, PCR, delivery %, VIX) for the ML prediction strategy.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class FeatureEngine:
    """
    Computes all features needed by the trading strategies:
    - Technical indicators (RSI, MACD, BB, ATR, ADX, VWAP, OBV, MFI)
    - Alternative data (FII flows, PCR, max pain, delivery %, VIX)
    - Cross-asset features (beta, relative strength, sector momentum)
    """

    def __init__(self, config_path: str = "config/strategies.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ml_features = self.config.get("ml_predictor", {}).get("features", [])

    # ═══════════════════════════════════════════
    # TECHNICAL INDICATORS
    # ═══════════════════════════════════════════

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands — upper, middle, lower."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def bb_position(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Position within Bollinger Bands (0 = lower, 1 = upper)."""
        upper, middle, lower = FeatureEngine.bollinger_bands(series, period, std_dev)
        width = upper - lower
        position = (series - lower) / width.replace(0, np.nan)
        return position.clip(0, 1)

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range — volatility measure."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average Directional Index — trend strength."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_val = FeatureEngine.atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.rolling(window=period).mean()
        return adx_val

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume."""
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        return (volume * direction).cumsum()

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Money Flow Index."""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        pos_sum = positive_flow.rolling(window=period).sum()
        neg_sum = negative_flow.rolling(window=period).sum()

        money_ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi_val = 100 - (100 / (1 + money_ratio))
        return mfi_val

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume relative to 20-day average."""
        avg_vol = volume.rolling(window=period).mean()
        return volume / avg_vol.replace(0, np.nan)

    # ═══════════════════════════════════════════
    # COMPUTE ALL TECHNICAL FEATURES
    # ═══════════════════════════════════════════

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicator columns to OHLCV DataFrame.

        Expects columns: [datetime, open, high, low, close, volume]
        """
        df = df.copy()

        # Core indicators
        df["rsi_14"] = self.rsi(df["close"], 14)
        df["rsi_7"] = self.rsi(df["close"], 7)

        macd_line, signal_line, histogram = self.macd(df["close"])
        df["macd_line"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        upper, middle, lower = self.bollinger_bands(df["close"])
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_position"] = self.bb_position(df["close"])
        df["bb_width"] = (upper - lower) / middle.replace(0, np.nan)

        df["atr_14"] = self.atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = df["atr_14"] / df["close"] * 100

        df["adx_14"] = self.adx(df["high"], df["low"], df["close"], 14)

        df["vwap"] = self.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["obv"] = self.obv(df["close"], df["volume"])
        df["mfi_14"] = self.mfi(df["high"], df["low"], df["close"], df["volume"], 14)

        # Moving averages
        df["ema_9"] = self.ema(df["close"], 9)
        df["ema_21"] = self.ema(df["close"], 21)
        df["ema_50"] = self.ema(df["close"], 50)
        df["sma_20"] = self.sma(df["close"], 20)
        df["sma_50"] = self.sma(df["close"], 50)
        df["sma_200"] = self.sma(df["close"], 200)

        # Derived
        df["volume_ratio"] = self.volume_ratio(df["volume"], 20)
        df["returns_1d"] = df["close"].pct_change(1)
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_20d"] = df["close"].pct_change(20)
        df["volatility_20d"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)

        # Price relative to MAs
        df["price_to_sma50"] = df["close"] / df["sma_50"].replace(0, np.nan) - 1
        df["price_to_sma200"] = df["close"] / df["sma_200"].replace(0, np.nan) - 1

        # Candlestick features
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"] * 100
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"] * 100
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"] * 100

        logger.debug(f"Added {len([c for c in df.columns if c not in ['datetime','open','high','low','close','volume','oi']])} technical features")
        return df

    # ═══════════════════════════════════════════
    # ALTERNATIVE DATA FEATURES
    # ═══════════════════════════════════════════

    def add_alternative_features(
        self,
        df: pd.DataFrame,
        fii_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[dict] = None,
        pcr_data: Optional[dict] = None,
        max_pain_data: Optional[dict] = None,
        delivery_data: Optional[pd.DataFrame] = None,
        futures_premium: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Add India-specific alternative data features.

        These are the primary alpha signals that differentiate this system
        from generic technical trading.
        """
        df = df.copy()

        # ── FII/DII Flow Features ──
        if fii_data is not None and not fii_data.empty:
            # Merge FII data by date
            fii_data = fii_data.copy()
            if "date" in fii_data.columns:
                fii_data["date"] = pd.to_datetime(fii_data["date"]).dt.date

            if "datetime" in df.columns:
                df["_date"] = pd.to_datetime(df["datetime"]).dt.date

                fii_merged = pd.merge(
                    df[["_date"]],
                    fii_data,
                    left_on="_date",
                    right_on="date",
                    how="left",
                )

                fii_net = fii_merged["fii_net_value"]
                df["fii_net_flow_1d"] = fii_net.values
                df["fii_net_flow_3d"] = (
                    fii_net.rolling(3, min_periods=1).sum().values
                )
                df["fii_net_flow_5d"] = (
                    fii_net.rolling(5, min_periods=1).sum().values
                )
                df["dii_net_flow_1d"] = fii_merged["dii_net_value"].values

                # FII flow momentum
                df["fii_flow_momentum"] = (
                    df["fii_net_flow_5d"] - df["fii_net_flow_3d"]
                )

                # FII direction: +1 buy, -1 sell, 0 missing
                df["fii_net_direction"] = 0
                df.loc[fii_net.values > 0, "fii_net_direction"] = 1
                df.loc[fii_net.values < 0, "fii_net_direction"] = -1

                # FII streak: consecutive days of same sign
                signs = fii_net.fillna(0).apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
                streak = []
                count = 0
                prev_sign = 0
                for s in signs:
                    if s == 0:
                        count = 0
                    elif s == prev_sign:
                        count += 1
                    else:
                        count = 1
                    prev_sign = s
                    streak.append(count * s)  # positive streak or negative streak
                df["fii_net_streak"] = streak

                df.drop("_date", axis=1, inplace=True)
        else:
            for col in ["fii_net_flow_1d", "fii_net_flow_3d", "fii_net_flow_5d",
                        "dii_net_flow_1d", "fii_flow_momentum",
                        "fii_net_direction", "fii_net_streak"]:
                df[col] = 0.0

        # ── India VIX Features ──
        if isinstance(vix_data, pd.DataFrame) and not vix_data.empty:
            # Historical VIX DataFrame (backtest mode with extended history)
            vix_df = vix_data.copy()
            if "datetime" in vix_df.columns:
                vix_df["_vdate"] = pd.to_datetime(vix_df["datetime"]).dt.date
            elif "date" in vix_df.columns:
                vix_df["_vdate"] = pd.to_datetime(vix_df["date"]).dt.date

            if "datetime" in df.columns:
                df["_date"] = pd.to_datetime(df["datetime"]).dt.date
                # Deduplicate VIX: keep last entry per date (avoids merge row inflation)
                vix_dedup = vix_df[["_vdate", "close"]].drop_duplicates(subset="_vdate", keep="last")
                vix_merged = pd.merge(
                    df[["_date"]],
                    vix_dedup.rename(columns={"close": "_vix_close", "_vdate": "_date"}),
                    on="_date",
                    how="left",
                )
                df["india_vix"] = vix_merged["_vix_close"].values
                df["india_vix"] = df["india_vix"].ffill().fillna(15.0)
                df["vix_change_pct"] = df["india_vix"].pct_change() * 100

                # Extended VIX features (benefit from long history)
                df["vix_5d_ma"] = df["india_vix"].rolling(5, min_periods=1).mean()
                df["vix_20d_ma"] = df["india_vix"].rolling(20, min_periods=1).mean()
                df["vix_percentile_252d"] = df["india_vix"].rolling(252, min_periods=20).apply(
                    lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0,
                    raw=False,
                )
                df.drop("_date", axis=1, inplace=True, errors="ignore")
            else:
                df["india_vix"] = 15.0
                df["vix_change_pct"] = 0.0
                df["vix_5d_ma"] = 15.0
                df["vix_20d_ma"] = 15.0
                df["vix_percentile_252d"] = 50.0
        elif isinstance(vix_data, dict) and vix_data:
            # Live mode: single VIX snapshot
            df["india_vix"] = vix_data.get("vix", 0)
            df["vix_change_pct"] = vix_data.get("change_pct", 0)
            df["vix_5d_ma"] = vix_data.get("vix", 0)
            df["vix_20d_ma"] = vix_data.get("vix", 0)
            df["vix_percentile_252d"] = 50.0
        else:
            df["india_vix"] = 15.0
            df["vix_change_pct"] = 0.0
            df["vix_5d_ma"] = 15.0
            df["vix_20d_ma"] = 15.0
            df["vix_percentile_252d"] = 50.0

        # ── Options Data Features ──
        if pcr_data:
            df["pcr_ratio"] = pcr_data.get("pcr_oi", 0)
            df["pcr_volume"] = pcr_data.get("pcr_volume", 0)
            df["pcr_change"] = pcr_data.get("pcr_change_oi", 0)
        else:
            df["pcr_ratio"] = 0.0
            df["pcr_volume"] = 0.0
            df["pcr_change"] = 0.0

        if max_pain_data:
            df["max_pain_distance"] = max_pain_data.get("distance_pct", 0)
        else:
            df["max_pain_distance"] = 0.0

        # ── Delivery Volume Features ──
        if delivery_data is not None and not delivery_data.empty:
            # Get latest delivery %
            latest = delivery_data.iloc[-1] if len(delivery_data) > 0 else {}
            df["delivery_pct"] = latest.get("delivery_pct", 0)

            # Delivery vs average
            if len(delivery_data) >= 5:
                avg_delivery = delivery_data["delivery_pct"].mean()
                df["delivery_vs_avg"] = df["delivery_pct"] - avg_delivery
            else:
                df["delivery_vs_avg"] = 0.0
        else:
            df["delivery_pct"] = 0.0
            df["delivery_vs_avg"] = 0.0

        # ── Futures Premium ──
        if futures_premium:
            df["futures_premium_pct"] = futures_premium.get("premium_pct", 0)
        else:
            df["futures_premium_pct"] = 0.0

        logger.debug("Added alternative data features")
        return df

    # ═══════════════════════════════════════════
    # EXTERNAL MARKET FEATURES
    # ═══════════════════════════════════════════

    def add_external_market_features(
        self,
        df: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add global market features from external data (yfinance).

        external_data: long-format DataFrame with columns [date, symbol, close].
        Symbols: SP500, NASDAQ, CRUDE_OIL, GOLD, USDINR.

        Features computed:
        - Lagged 1-day returns for each market (US close before Indian open → predictive)
        - Rolling 20-day correlations (S&P500 vs NIFTY, Crude vs NIFTY)
        - USD/INR 5-day momentum (INR weakening = bearish signal)
        - Global risk score (composite of negative signals)
        """
        ext_cols = [
            "sp500_prev_return", "nasdaq_prev_return", "crude_prev_return",
            "gold_prev_return", "usdinr_prev_return", "sp500_nifty_corr_20d",
            "crude_nifty_corr_20d", "dxy_momentum_5d", "global_risk_score",
        ]

        if external_data is None or external_data.empty:
            for col in ext_cols:
                df[col] = 0.0
            return df

        df = df.copy()

        # Pivot from long to wide: one column per symbol
        ext = external_data.copy()
        ext["date"] = pd.to_datetime(ext["date"]).dt.date
        wide = ext.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")

        # Compute daily returns for each external market
        ext_returns = wide.pct_change(fill_method=None)

        # Prepare merge key from df
        if "datetime" in df.columns:
            df["_date"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            df["_date"] = df.index

        # Map symbol names to feature names
        symbol_map = {
            "SP500": "sp500_prev_return",
            "NASDAQ": "nasdaq_prev_return",
            "CRUDE_OIL": "crude_prev_return",
            "GOLD": "gold_prev_return",
            "USDINR": "usdinr_prev_return",
        }

        # Create lagged returns (shift by 1 = previous day, since US closes before Indian open)
        lagged_returns = ext_returns.shift(1)

        # Merge lagged returns with df
        for symbol, feature_name in symbol_map.items():
            if symbol in lagged_returns.columns:
                ret_series = lagged_returns[symbol].reset_index()
                ret_series.columns = ["_date", feature_name]
                df = pd.merge(df, ret_series, on="_date", how="left")
                df[feature_name] = df[feature_name].fillna(0.0)
            else:
                df[feature_name] = 0.0

        # Rolling correlation: S&P500 vs NIFTY (using df's own returns)
        nifty_ret = df["close"].pct_change()
        if "sp500_prev_return" in df.columns:
            df["sp500_nifty_corr_20d"] = nifty_ret.rolling(20, min_periods=5).corr(
                df["sp500_prev_return"]
            ).fillna(0.0)
        else:
            df["sp500_nifty_corr_20d"] = 0.0

        # Rolling correlation: Crude vs NIFTY
        if "crude_prev_return" in df.columns:
            df["crude_nifty_corr_20d"] = nifty_ret.rolling(20, min_periods=5).corr(
                df["crude_prev_return"]
            ).fillna(0.0)
        else:
            df["crude_nifty_corr_20d"] = 0.0

        # USD/INR 5-day momentum (positive = INR weakening = bearish for Indian markets)
        if "USDINR" in wide.columns:
            usdinr_mom = wide["USDINR"].pct_change(5, fill_method=None).shift(1).reset_index()
            usdinr_mom.columns = ["_date", "dxy_momentum_5d"]
            df = pd.merge(df, usdinr_mom, on="_date", how="left")
            df["dxy_momentum_5d"] = df["dxy_momentum_5d"].fillna(0.0)
        else:
            df["dxy_momentum_5d"] = 0.0

        # Global risk score: composite of negative signals
        # Higher = more risk (negative US returns + crude drop + INR weakening)
        sp500_neg = df["sp500_prev_return"].clip(upper=0).abs()
        crude_neg = df["crude_prev_return"].clip(upper=0).abs()
        inr_weak = df["dxy_momentum_5d"].clip(lower=0)
        df["global_risk_score"] = (sp500_neg * 0.4 + crude_neg * 0.3 + inr_weak * 0.3) * 100
        df["global_risk_score"] = df["global_risk_score"].fillna(0.0)

        df.drop("_date", axis=1, inplace=True, errors="ignore")

        logger.debug(f"Added {len(ext_cols)} external market features")
        return df

    # ═══════════════════════════════════════════
    # CROSS-ASSET FEATURES
    # ═══════════════════════════════════════════

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add cross-asset and relative features.

        - Beta to NIFTY 50
        - Relative strength vs NIFTY
        - Sector momentum
        """
        df = df.copy()

        if nifty_df is not None and not nifty_df.empty:
            nifty_returns = nifty_df["close"].pct_change()
            stock_returns = df["close"].pct_change()

            # Rolling beta (60 day)
            rolling_cov = stock_returns.rolling(60).cov(nifty_returns)
            rolling_var = nifty_returns.rolling(60).var()
            df["beta_60d"] = rolling_cov / rolling_var.replace(0, np.nan)

            # Relative strength (stock return / nifty return over 20d)
            stock_ret_20d = df["close"].pct_change(20)
            nifty_ret_20d = nifty_df["close"].pct_change(20)
            if len(nifty_ret_20d) == len(stock_ret_20d):
                df["relative_strength_20d"] = stock_ret_20d - nifty_ret_20d.values
            else:
                df["relative_strength_20d"] = 0.0
        else:
            df["beta_60d"] = 1.0
            df["relative_strength_20d"] = 0.0

        # Sector relative strength
        if sector_df is not None and not sector_df.empty:
            sector_ret = sector_df["close"].pct_change(20)
            stock_ret = df["close"].pct_change(20)
            if len(sector_ret) == len(stock_ret):
                df["sector_relative_strength"] = stock_ret - sector_ret.values
            else:
                df["sector_relative_strength"] = 0.0
        else:
            df["sector_relative_strength"] = 0.0

        return df

    # ═══════════════════════════════════════════
    # OPTION CHAIN FEATURES (IV, Greeks, OI)
    # ═══════════════════════════════════════════

    def add_option_features(
        self,
        df: pd.DataFrame,
        option_chain_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add option-derived features from historical option chain data.

        option_chain_df: DataFrame from DataStore.get_option_chain_atm_history()
            with columns: date, atm_ce_iv, atm_pe_iv, atm_ce_delta, atm_pe_delta,
            atm_ce_theta, atm_pe_theta, atm_ce_gamma, atm_pe_gamma,
            atm_ce_vega, atm_pe_vega, atm_ce_ltp, atm_pe_ltp,
            atm_ce_oi, atm_pe_oi, total_ce_oi, total_pe_oi,
            total_ce_volume, total_pe_volume, pcr_oi, pcr_volume

        New features (30):
        - IV features (7): atm_iv_mean, iv_skew, iv_rank_20d, iv_percentile_252d,
                           iv_change_1d, iv_change_5d, iv_term_spread
        - Greeks features (8): atm_delta, atm_gamma, atm_theta, atm_vega,
                               gamma_risk, theta_decay_rate, vega_exposure, delta_skew
        - OI features (9): atm_oi_ratio, oi_buildup_ce, oi_buildup_pe,
                           oi_concentration, oi_change_ratio, pcr_oi_hist,
                           pcr_oi_change_5d, pcr_volume_hist, oi_momentum_5d
        - Premium features (6): atm_premium_pct, premium_ratio, premium_change_1d,
                                premium_change_5d, straddle_cost_pct, straddle_change_5d
        """
        option_cols = [
            # IV features
            "atm_iv_mean", "iv_skew", "iv_rank_20d", "iv_percentile_252d",
            "iv_change_1d", "iv_change_5d", "iv_term_spread",
            # Greeks features
            "atm_delta", "atm_gamma", "atm_theta", "atm_vega",
            "gamma_risk", "theta_decay_rate", "vega_exposure", "delta_skew",
            # OI features
            "atm_oi_ratio", "oi_buildup_ce", "oi_buildup_pe",
            "oi_concentration", "oi_change_ratio", "pcr_oi_hist",
            "pcr_oi_change_5d", "pcr_volume_hist", "oi_momentum_5d",
            # Premium features
            "atm_premium_pct", "premium_ratio", "premium_change_1d",
            "premium_change_5d", "straddle_cost_pct", "straddle_change_5d",
        ]

        if option_chain_df is None or option_chain_df.empty:
            for col in option_cols:
                df[col] = 0.0
            logger.debug("No option chain data — filled option features with 0")
            return df

        df = df.copy()
        oc = option_chain_df.copy()
        oc["date"] = pd.to_datetime(oc["date"]).dt.date

        # Merge key from df
        if "datetime" in df.columns:
            df["_date"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            for col in option_cols:
                df[col] = 0.0
            return df

        # ── Compute features on option chain DataFrame first ──

        # IV features
        oc["atm_iv_mean"] = (oc["atm_ce_iv"] + oc["atm_pe_iv"]) / 2
        oc["iv_skew"] = oc["atm_pe_iv"] - oc["atm_ce_iv"]  # put IV > call IV = fear
        oc["iv_change_1d"] = oc["atm_iv_mean"].pct_change(1) * 100
        oc["iv_change_5d"] = oc["atm_iv_mean"].pct_change(5) * 100
        oc["iv_rank_20d"] = oc["atm_iv_mean"].rolling(20, min_periods=5).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
            if (x.max() - x.min()) > 0 else 50.0,
            raw=False,
        )
        oc["iv_percentile_252d"] = oc["atm_iv_mean"].rolling(252, min_periods=20).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50.0,
            raw=False,
        )
        # IV term spread: not available from single expiry chain, use CE-PE IV gap as proxy
        oc["iv_term_spread"] = oc["iv_skew"].rolling(5, min_periods=1).mean()

        # Greeks features (from ATM strike)
        oc["atm_delta"] = oc["atm_ce_delta"]  # CE delta (0 to 1, ATM ≈ 0.5)
        oc["atm_gamma"] = oc["atm_ce_gamma"]
        oc["atm_theta"] = oc["atm_ce_theta"]
        oc["atm_vega"] = oc["atm_ce_vega"]
        # Gamma risk: high gamma near expiry = dangerous
        oc["gamma_risk"] = oc["atm_gamma"] * oc.get("underlying", oc["atm_ce_ltp"] + oc["atm_pe_ltp"])
        oc["gamma_risk"] = oc["gamma_risk"].clip(-1e6, 1e6).fillna(0)
        # Theta decay rate (theta / premium)
        atm_prem = (oc["atm_ce_ltp"] + oc["atm_pe_ltp"]).replace(0, np.nan)
        oc["theta_decay_rate"] = (oc["atm_ce_theta"] + oc["atm_pe_theta"]) / atm_prem
        oc["theta_decay_rate"] = oc["theta_decay_rate"].clip(-1, 0).fillna(0)
        # Vega exposure (vega * IV)
        oc["vega_exposure"] = oc["atm_vega"] * oc["atm_iv_mean"] / 100
        # Delta skew: CE delta + PE delta (should sum to ~0 at ATM; deviation = skew)
        oc["delta_skew"] = oc["atm_ce_delta"] + oc["atm_pe_delta"]

        # OI features
        atm_total_oi = (oc["atm_ce_oi"] + oc["atm_pe_oi"]).replace(0, np.nan)
        chain_total_oi = (oc["total_ce_oi"] + oc["total_pe_oi"]).replace(0, np.nan)
        oc["atm_oi_ratio"] = atm_total_oi / chain_total_oi  # ATM concentration
        oc["atm_oi_ratio"] = oc["atm_oi_ratio"].clip(0, 1).fillna(0)
        oc["oi_buildup_ce"] = oc["total_ce_oi"].pct_change(1) * 100
        oc["oi_buildup_pe"] = oc["total_pe_oi"].pct_change(1) * 100
        oc["oi_concentration"] = (
            (oc["atm_ce_oi"] + oc["atm_pe_oi"]) / chain_total_oi
        ).clip(0, 1).fillna(0)
        oc["oi_change_ratio"] = oc["oi_buildup_pe"] - oc["oi_buildup_ce"]
        oc["pcr_oi_hist"] = oc["pcr_oi"]  # Already computed in store
        oc["pcr_oi_change_5d"] = oc["pcr_oi"].pct_change(5) * 100
        oc["pcr_volume_hist"] = oc["pcr_volume"]
        oc["oi_momentum_5d"] = chain_total_oi.pct_change(5) * 100

        # Premium features
        underlying = oc.get("underlying", pd.Series(0, index=oc.index)).replace(0, np.nan)
        oc["atm_premium_pct"] = oc["atm_ce_ltp"] / underlying * 100
        oc["premium_ratio"] = oc["atm_ce_ltp"] / oc["atm_pe_ltp"].replace(0, np.nan)
        oc["premium_ratio"] = oc["premium_ratio"].clip(0.1, 10).fillna(1)
        oc["premium_change_1d"] = oc["atm_ce_ltp"].pct_change(1) * 100
        oc["premium_change_5d"] = oc["atm_ce_ltp"].pct_change(5) * 100
        straddle = oc["atm_ce_ltp"] + oc["atm_pe_ltp"]
        oc["straddle_cost_pct"] = straddle / underlying * 100
        oc["straddle_change_5d"] = straddle.pct_change(5) * 100

        # ── Merge with main DataFrame ──
        merge_cols = ["date"] + option_cols
        available_merge = [c for c in merge_cols if c in oc.columns]
        oc_merge = oc[available_merge].copy()

        df = pd.merge(df, oc_merge, left_on="_date", right_on="date", how="left")
        df.drop(["_date", "date"], axis=1, inplace=True, errors="ignore")

        # Fill any remaining NaN option features
        for col in option_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)
            else:
                df[col] = 0.0

        logger.debug(f"Added {len(option_cols)} option chain features")
        return df

    # ═══════════════════════════════════════════
    # ML DATASET PREPARATION
    # ═══════════════════════════════════════════

    def prepare_ml_dataset(
        self,
        df: pd.DataFrame,
        up_threshold: float = 0.005,
        down_threshold: float = -0.005,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and labels (y) for ML model training.

        Label:
            1 if next-day return > +0.5%
           -1 if next-day return < -0.5%
            0 otherwise

        Returns:
            (X, y) where X is the feature matrix and y is the target
        """
        df = df.copy()

        # Create target: next-day return classification
        df["next_return"] = df["close"].pct_change(1).shift(-1)
        df["target"] = 0
        df.loc[df["next_return"] > up_threshold, "target"] = 1
        df.loc[df["next_return"] < down_threshold, "target"] = -1

        # Select ML features (only those present in DataFrame)
        available_features = [f for f in self.ml_features if f in df.columns]
        missing_features = [f for f in self.ml_features if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing ML features: {missing_features}")

        # Add technical features that are in the ML feature list
        all_feature_cols = available_features.copy()

        # Also add any technical features that are common
        extra_tech = [
            "rsi_14", "macd_histogram", "bb_position", "atr_pct",
            "adx_14", "volume_ratio", "mfi_14", "volatility_20d",
        ]
        for col in extra_tech:
            if col in df.columns and col not in all_feature_cols:
                all_feature_cols.append(col)

        # Cap at max features
        max_features = self.config.get("ml_predictor", {}).get("max_features", 15)
        all_feature_cols = all_feature_cols[:max_features]

        # Drop NaN rows
        subset = df[all_feature_cols + ["target"]].dropna()

        X = subset[all_feature_cols]
        y = subset["target"]

        logger.info(
            f"ML dataset: {len(X)} samples, {len(all_feature_cols)} features, "
            f"class distribution: {dict(y.value_counts())}"
        )

        return X, y

    def prepare_premium_target(
        self,
        df: pd.DataFrame,
        option_chain_df: Optional[pd.DataFrame] = None,
        gain_threshold: float = 0.10,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Phase 2: Premium-based ML target — "will ATM CE premium gain >X% intraday?"

        Instead of predicting "NIFTY goes up/down", predict whether the ATM CE
        premium will gain more than `gain_threshold` (default 10%) during the day.
        This directly predicts what matters for options P&L.

        Uses real expired option premium data from DB when available,
        falls back to spot-based proxy when option history is sparse.

        Returns:
            (X, y) where y ∈ {1: CE premium gained >10%, 0: CE premium didn't,
                               -1: PE premium gained >10% (bearish)}
        """
        df = df.copy()

        if option_chain_df is not None and not option_chain_df.empty:
            oc = option_chain_df.copy()
            oc["date"] = pd.to_datetime(oc["date"]).dt.date

            # Compute actual ATM CE premium intraday change
            # Using day-over-day premium change as proxy for intraday
            oc["ce_prem_change"] = oc["atm_ce_ltp"].pct_change(1)
            oc["pe_prem_change"] = oc["atm_pe_ltp"].pct_change(1)

            if "datetime" in df.columns:
                df["_date"] = pd.to_datetime(df["datetime"]).dt.date
                prem_map = dict(zip(oc["date"], oc["ce_prem_change"]))
                pe_map = dict(zip(oc["date"], oc["pe_prem_change"]))

                # Shift by -1: we want NEXT day's premium change as target
                dates = df["_date"].tolist()
                ce_changes = []
                pe_changes = []
                for i in range(len(dates)):
                    next_date = dates[i + 1] if i + 1 < len(dates) else None
                    ce_changes.append(prem_map.get(next_date, np.nan))
                    pe_changes.append(pe_map.get(next_date, np.nan))

                df["_ce_prem_change"] = ce_changes
                df["_pe_prem_change"] = pe_changes
                df.drop("_date", axis=1, inplace=True, errors="ignore")

                # Target: 1 if CE gained >threshold, -1 if PE gained >threshold, 0 otherwise
                df["target"] = 0
                df.loc[df["_ce_prem_change"] > gain_threshold, "target"] = 1
                df.loc[df["_pe_prem_change"] > gain_threshold, "target"] = -1
                df.drop(["_ce_prem_change", "_pe_prem_change"], axis=1, inplace=True, errors="ignore")
            else:
                # Fallback: spot-based target
                df["target"] = self._spot_based_target(df, gain_threshold)
        else:
            # No option data: use spot-based proxy (scaled by typical premium leverage ~3x)
            df["target"] = self._spot_based_target(df, gain_threshold)

        # Select features
        available_features = [f for f in self.ml_features if f in df.columns]
        # Also include option features if present
        option_feat_names = [
            "atm_iv_mean", "iv_skew", "iv_rank_20d", "iv_percentile_252d",
            "iv_change_1d", "iv_change_5d", "iv_term_spread",
            "atm_delta", "atm_gamma", "atm_theta", "atm_vega",
            "gamma_risk", "theta_decay_rate", "vega_exposure", "delta_skew",
            "atm_oi_ratio", "oi_buildup_ce", "oi_buildup_pe",
            "oi_concentration", "oi_change_ratio", "pcr_oi_hist",
            "pcr_oi_change_5d", "pcr_volume_hist", "oi_momentum_5d",
            "atm_premium_pct", "premium_ratio", "premium_change_1d",
            "premium_change_5d", "straddle_cost_pct", "straddle_change_5d",
        ]
        for col in option_feat_names:
            if col in df.columns and col not in available_features:
                available_features.append(col)

        # Cap at max features
        max_features = self.config.get("ml_predictor", {}).get("max_features", 65)
        available_features = available_features[:max_features]

        subset = df[available_features + ["target"]].dropna()
        X = subset[available_features]
        y = subset["target"]

        logger.info(
            f"Premium ML dataset: {len(X)} samples, {len(available_features)} features, "
            f"class distribution: {dict(y.value_counts())}"
        )
        return X, y

    @staticmethod
    def _spot_based_target(df: pd.DataFrame, gain_threshold: float) -> pd.Series:
        """Fallback target using spot returns as proxy for premium returns."""
        # ATM option premium moves ~2-3x underlying for NIFTY
        # So a 10% premium gain ≈ 3-5% spot move, but we use a gentler mapping
        spot_threshold = gain_threshold / 3.0  # ~3.3% spot for 10% premium
        next_ret = df["close"].pct_change(1).shift(-1)
        target = pd.Series(0, index=df.index)
        target[next_ret > spot_threshold] = 1
        target[next_ret < -spot_threshold] = -1
        return target

    def compute_all_features(
        self,
        ohlcv_df: pd.DataFrame,
        fii_data: Optional[pd.DataFrame] = None,
        vix_data=None,
        pcr_data: Optional[dict] = None,
        max_pain_data: Optional[dict] = None,
        delivery_data: Optional[pd.DataFrame] = None,
        futures_premium: Optional[dict] = None,
        nifty_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
        external_data: Optional[pd.DataFrame] = None,
        option_chain_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        One-shot: compute all features for a stock.

        Steps:
        1. Technical indicators
        2. Alternative data (FII, VIX, PCR, delivery %)
        3. External market data (S&P500, NASDAQ, Crude, Gold, USD/INR)
        4. Cross-asset (beta, relative strength)
        5. Option chain features (IV, Greeks, OI) — Phase 1
        """
        df = self.add_technical_features(ohlcv_df)
        df = self.add_alternative_features(
            df, fii_data, vix_data, pcr_data, max_pain_data,
            delivery_data, futures_premium,
        )
        df = self.add_external_market_features(df, external_data)
        df = self.add_cross_asset_features(df, nifty_df, sector_df)
        df = self.add_option_features(df, option_chain_df)

        logger.info(f"Computed {len(df.columns)} total features for {len(df)} rows")
        return df
