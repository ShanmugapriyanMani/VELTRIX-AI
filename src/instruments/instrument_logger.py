"""
Passive Instrument Logger — scores 5 new instruments using the same
9-factor pipeline as NIFTY options_buyer, but only logs to DB. Never trades.

Instruments: BANKNIFTY, FINNIFTY, MIDCPNIFTY, RELIANCE, SENSEX
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.config.env_loader import get_config
from src.data.features import FeatureEngine
from src.data.store import DataStore


# ─────────────────────────────────────────
# Instrument Configuration
# ─────────────────────────────────────────

@dataclass
class InstrumentConfig:
    """Configuration for a tracked instrument."""

    name: str
    instrument_type: str  # "index" or "stock"
    exchange: str  # "NSE" or "BSE"
    upstox_symbol: str  # Upstox instrument key
    lot_size: int
    tick_size: float
    options_expiry: str  # "weekly" or "monthly"
    vix_multiplier: float  # Scales VIX effect on this instrument
    adx_threshold: float  # ADX threshold for TRENDING regime


TRACKED_INSTRUMENTS: list[InstrumentConfig] = [
    InstrumentConfig("BANKNIFTY", "index", "NSE", "NSE_INDEX|Nifty Bank", 30, 0.05, "weekly", 1.15, 22),
    InstrumentConfig("FINNIFTY", "index", "NSE", "NSE_INDEX|Nifty Fin Service", 65, 0.05, "weekly", 1.10, 22),
    InstrumentConfig("MIDCPNIFTY", "index", "NSE", "NSE_INDEX|NIFTY MID SELECT", 120, 0.05, "weekly", 1.25, 20),
    InstrumentConfig("RELIANCE", "stock", "NSE", "NSE_EQ|INE002A01018", 250, 0.05, "monthly", 1.20, 20),
    InstrumentConfig("SENSEX", "index", "BSE", "BSE_INDEX|SENSEX", 10, 0.01, "weekly", 1.05, 25),
]

def _conviction_thresholds() -> dict[str, float]:
    """Read conviction thresholds from EnvConfig (respects .env overrides)."""
    cfg = get_config()
    return {
        "TRENDING": cfg.TRENDING_THRESHOLD,
        "RANGEBOUND": cfg.RANGEBOUND_THRESHOLD,
        "VOLATILE": cfg.VOLATILE_THRESHOLD,
        "ELEVATED": cfg.ELEVATED_THRESHOLD,
    }


# ─────────────────────────────────────────
# Instrument Logger
# ─────────────────────────────────────────

class InstrumentLogger:
    """
    Passively scores and logs data for non-trading instruments.
    Runs alongside the main trading loop but never executes trades.
    """

    def __init__(
        self,
        store: DataStore,
        data_fetcher: Any,
        feature_engine: FeatureEngine,
        options_resolver: Any,
    ):
        self.store = store
        self.fetcher = data_fetcher
        self.fe = feature_engine
        self.resolver = options_resolver
        self.instruments = TRACKED_INSTRUMENTS

        # Per-instrument state
        self._prev_vix: dict[str, float] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_scores: dict[str, dict] = {}
        self._oi_cache: dict[str, tuple[float, dict, float]] = {}  # {name: (timestamp, oi_data, pcr)}
        self._OI_CACHE_TTL = 300  # 5 minutes

        # Register instruments in DB
        for inst in self.instruments:
            try:
                self.store.save_instrument_registry({
                    "name": inst.name,
                    "instrument_type": inst.instrument_type,
                    "exchange": inst.exchange,
                    "upstox_symbol": inst.upstox_symbol,
                    "lot_size": inst.lot_size,
                    "tick_size": inst.tick_size,
                    "options_expiry": inst.options_expiry,
                    "vix_multiplier": inst.vix_multiplier,
                    "adx_threshold": inst.adx_threshold,
                })
            except Exception:
                pass

    # ─────────────────────────────────────────
    # Pre-Market: Fetch + Feature Engineering
    # ─────────────────────────────────────────

    def prepare_daily_data(self, vix_data: dict) -> None:
        """Pre-market: fetch historical candles + compute features for all instruments."""
        for inst in self.instruments:
            try:
                df = self.fetcher.get_historical_candles(inst.upstox_symbol, "day")
                if df is None or df.empty or len(df) < 50:
                    logger.warning(
                        f"InstrumentLogger: {inst.name} insufficient data "
                        f"({0 if df is None else len(df)} bars)"
                    )
                    continue

                df = self.fe.add_technical_features(df)
                self._ohlcv_cache[inst.name] = df
                logger.info(f"InstrumentLogger: {inst.name} prepared ({len(df)} bars)")
            except Exception as e:
                logger.warning(f"InstrumentLogger: {inst.name} pre-market failed: {e}")

    # ─────────────────────────────────────────
    # Trading Loop: Score + Log
    # ─────────────────────────────────────────

    def score_and_log(
        self,
        vix_data: dict,
        fii_history: Optional[pd.DataFrame] = None,
        nifty_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, dict]:
        """Score all instruments and log signals if would_trade=True."""
        results = {}
        vix = vix_data.get("vix", 15)
        vix_change_pct = vix_data.get("change_pct", 0)

        for inst in self.instruments:
            try:
                df = self._ohlcv_cache.get(inst.name)
                if df is None or df.empty or len(df) < 50:
                    continue

                # Fetch option chain
                oi_data, pcr_val = self._fetch_option_chain(inst)

                # Compute regime
                regime = self._compute_regime(inst, df, vix)
                ema_weight, mr_weight = self._regime_weights(regime)

                # 9-factor scoring
                bull, bear, direction = self._score_instrument(
                    inst, df, vix, regime, ema_weight, mr_weight,
                    pcr_val, oi_data, nifty_df,
                )
                score_diff = abs(bull - bear)

                # Would this instrument have traded?
                would_trade, trade_type, blocking_reason = self._determine_would_trade(
                    inst, score_diff, regime, vix, direction,
                )

                result = {
                    "bull_score": bull,
                    "bear_score": bear,
                    "direction": direction,
                    "score_diff": score_diff,
                    "regime": regime,
                    "pcr": pcr_val,
                    "would_trade": would_trade,
                    "trade_type": trade_type,
                    "blocking_reason": blocking_reason,
                }
                results[inst.name] = result
                self._last_scores[inst.name] = result

                # Log signal if would_trade
                if would_trade and direction:
                    row = df.iloc[-1]
                    now = datetime.now()

                    self.store.save_instrument_signal_log({
                        "date": date.today().isoformat(),
                        "instrument": inst.name,
                        "signal_time": now.isoformat(),
                        "oi_score_diff": round(score_diff, 2),
                        "oi_bull_score": round(bull, 2),
                        "oi_bear_score": round(bear, 2),
                        "pcr": round(pcr_val, 3),
                        "vix_level": round(vix, 2),
                        "vix_change_pct": round(vix_change_pct, 2),
                        "rsi_14": round(float(row.get("rsi_14", 50)), 1),
                        "adx_14": round(float(row.get("adx_14", 0)), 1),
                        "regime": regime,
                        "direction": direction,
                        "entry_hour": now.hour,
                        "conviction": round(score_diff, 2),
                        "trade_type": trade_type,
                        "not_traded_reason": "PASSIVE_LOGGING",
                        "scored_at": now.isoformat(),
                    })

                    logger.info(
                        f"[INSTRUMENT] {inst.name} SIGNAL: {direction} "
                        f"bull={bull:.1f} bear={bear:.1f} diff={score_diff:.1f} "
                        f"regime={regime} type={trade_type}"
                    )

            except Exception as e:
                logger.warning(f"InstrumentLogger: {inst.name} scoring failed: {e}")

        return results

    # ─────────────────────────────────────────
    # Post-Market: Daily Log + EOD Outcomes
    # ─────────────────────────────────────────

    def save_daily_log(
        self,
        vix_data: dict,
        fii_history: Optional[pd.DataFrame] = None,
    ) -> None:
        """Post-market: save one daily summary row per instrument using cached scores."""
        vix = vix_data.get("vix", 15)
        vix_change_pct = vix_data.get("change_pct", 0)
        now = datetime.now()

        for inst in self.instruments:
            try:
                df = self._ohlcv_cache.get(inst.name)
                if df is None or df.empty or len(df) < 50:
                    continue

                row = df.iloc[-1]
                prev_row = df.iloc[-2]

                # Reuse scores from trading loop instead of re-scoring
                scores = self._last_scores.get(inst.name)
                oi_data = {}
                if scores:
                    bull = scores["bull_score"]
                    bear = scores["bear_score"]
                    direction = scores["direction"]
                    score_diff = scores["score_diff"]
                    regime = scores["regime"]
                    pcr_val = scores["pcr"]
                    would_trade = scores["would_trade"]
                    trade_type = scores["trade_type"]
                    blocking_reason = scores["blocking_reason"]
                    # Fetch OI data for daily log (not cached in scores)
                    oi_data, _ = self._fetch_option_chain(inst)
                else:
                    # Fallback: compute if score_and_log() never ran
                    oi_data, pcr_val = self._fetch_option_chain(inst)
                    regime = self._compute_regime(inst, df, vix)
                    ema_w, mr_w = self._regime_weights(regime)
                    bull, bear, direction = self._score_instrument(
                        inst, df, vix, regime, ema_w, mr_w, pcr_val, oi_data, None,
                    )
                    score_diff = abs(bull - bear)
                    would_trade, trade_type, blocking_reason = self._determine_would_trade(
                        inst, score_diff, regime, vix, direction,
                    )

                close = float(row.get("close", 0))
                prev_close = float(prev_row.get("close", 0))
                change_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0

                # ADX slope (current - previous)
                adx = float(row.get("adx_14", 0))
                prev_adx = float(prev_row.get("adx_14", 0))
                adx_slope = adx - prev_adx

                # BB width
                bb_upper = float(row.get("bb_upper", close * 1.02))
                bb_lower = float(row.get("bb_lower", close * 0.98))
                bb_width = (bb_upper - bb_lower) / close * 100 if close > 0 else 0

                # FII/DII
                fii_net, dii_net = 0.0, 0.0
                if fii_history is not None and not fii_history.empty:
                    last_fii = fii_history.iloc[-1]
                    fii_net = float(last_fii.get("fii_net_value", 0))
                    dii_net = float(last_fii.get("dii_net_value", 0))

                # Signal strength label
                if score_diff >= 4.0:
                    signal_strength = "STRONG"
                elif score_diff >= 2.5:
                    signal_strength = "MODERATE"
                elif score_diff >= 1.5:
                    signal_strength = "WEAK"
                else:
                    signal_strength = "FLAT"

                self.store.save_instrument_daily_log({
                    "date": date.today().isoformat(),
                    "instrument": inst.name,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": close,
                    "prev_close": prev_close,
                    "change_pct": round(change_pct, 2),
                    "regime": regime,
                    "adx": round(adx, 1),
                    "adx_slope": round(adx_slope, 2),
                    "bb_width": round(bb_width, 2),
                    "bull_score": round(bull, 2),
                    "bear_score": round(bear, 2),
                    "score_diff": round(score_diff, 2),
                    "direction": direction,
                    "conviction": round(score_diff, 2),
                    "pcr": round(pcr_val, 3),
                    "max_call_oi_strike": oi_data.get("max_call_oi_strike", 0),
                    "max_put_oi_strike": oi_data.get("max_put_oi_strike", 0),
                    "max_pain": oi_data.get("max_pain_strike", 0) if isinstance(oi_data, dict) else 0,
                    "vix_level": round(vix, 2),
                    "vix_change_pct": round(vix_change_pct, 2),
                    "rsi_14": round(float(row.get("rsi_14", 50)), 1),
                    "macd_signal": round(float(row.get("macd_signal", 0)), 2),
                    "ema9": round(float(row.get("ema_9", 0)), 2),
                    "ema21": round(float(row.get("ema_21", 0)), 2),
                    "ema50": round(float(row.get("ema_50", 0)), 2),
                    "would_trade": 1 if would_trade else 0,
                    "trade_type": trade_type,
                    "signal_strength": signal_strength,
                    "blocking_reason": blocking_reason,
                    "fii_net": fii_net,
                    "dii_net": dii_net,
                    "scored_at": now.isoformat(),
                })

                logger.info(
                    f"InstrumentLogger EOD: {inst.name} {direction or 'FLAT'} "
                    f"bull={bull:.1f} bear={bear:.1f} diff={score_diff:.1f} "
                    f"regime={regime} would_trade={'YES' if would_trade else 'NO'}"
                )

            except Exception as e:
                logger.warning(f"InstrumentLogger: {inst.name} daily log failed: {e}")

    def update_eod_outcomes(self, date_str: str) -> None:
        """Post-market: update signal log with EOD outcomes for would-have-traded signals."""
        signals_df = self.store.get_instrument_signal_log(limit=50)
        if signals_df.empty:
            return

        today_signals = signals_df[signals_df["date"] == date_str]
        if today_signals.empty:
            return

        # Batch-fetch all unique symbols
        symbols_to_fetch = [
            s for s in today_signals["would_buy_symbol"].dropna().unique() if s
        ]
        if not symbols_to_fetch:
            return

        batch_quotes = self.fetcher.get_live_quotes_batch(symbols_to_fetch)

        for _, signal in today_signals.iterrows():
            symbol = signal.get("would_buy_symbol", "")
            if not symbol:
                continue

            try:
                eod_premium = batch_quotes.get(symbol, {}).get("ltp", 0)
                entry_premium = signal.get("would_buy_premium", 0)

                if entry_premium > 0 and eod_premium > 0:
                    eod_pnl_pct = (eod_premium - entry_premium) / entry_premium * 100
                    eod_result = "WIN" if eod_pnl_pct > 0 else "LOSS"
                else:
                    eod_pnl_pct = 0
                    eod_result = "NO_DATA"

                self.store.update_instrument_signal_log(signal["id"], {
                    "eod_premium": eod_premium,
                    "eod_pnl_pct": round(eod_pnl_pct, 2),
                    "eod_result": eod_result,
                })
            except Exception as e:
                logger.debug(f"InstrumentLogger: EOD outcome update failed for {symbol}: {e}")

    # ─────────────────────────────────────────
    # Telegram Summary
    # ─────────────────────────────────────────

    def build_telegram_summary(self, nifty_status: str = "") -> str:
        """Build instrument summary text for Telegram daily report."""
        today_str = date.today().strftime("%Y-%m-%d")
        lines = [f"\n📊 INSTRUMENT SCAN — {today_str}"]

        for inst in self.instruments:
            scores = self._last_scores.get(inst.name)
            if not scores:
                lines.append(f"  {inst.name:<12} -- No data")
                continue

            regime = scores.get("regime", "--")[:8]
            direction = scores.get("direction", "--") or "--"
            would = scores.get("would_trade", False)
            diff = scores.get("score_diff", 0)

            if would:
                status = "📝 SIGNAL"
            elif diff >= 1.0:
                status = "📝 LOGGED"
            else:
                status = "⏭ FLAT  "

            reason = scores.get("blocking_reason", "")
            extra = f"  diff={diff:.1f}" if not would else ""
            if reason and not would:
                extra = f"  {reason}"

            lines.append(f"  {inst.name:<12} {regime:<10} {direction:<3} {status}{extra}")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # Internal: Scoring (9-Factor Pipeline)
    # ─────────────────────────────────────────

    def _score_instrument(
        self,
        inst: InstrumentConfig,
        df: pd.DataFrame,
        vix: float,
        regime: str,
        ema_weight: float,
        mr_weight: float,
        pcr_val: float,
        oi_data: dict,
        nifty_df: Optional[pd.DataFrame] = None,
    ) -> tuple[float, float, str]:
        """
        Stateless 9-factor scoring for any instrument.
        Mirrors OptionsBuyerStrategy._compute_direction_score() (lines 403-663).
        """
        row = df.iloc[-1]
        prev_row = df.iloc[-2]

        bull_score = 0.0
        bear_score = 0.0

        # Safe value extraction
        close = float(row.get("close", 0))
        ema_9 = float(row.get("ema_9", close))
        ema_21 = float(row.get("ema_21", close))
        ema_50 = float(row.get("ema_50", close))
        rsi = float(row.get("rsi_14", 50))
        prev_rsi = float(prev_row.get("rsi_14", 50))
        macd_hist = float(row.get("macd_histogram", 0))
        prev_macd_hist = float(prev_row.get("macd_histogram", 0))
        bb_upper = float(row.get("bb_upper", close * 1.02))
        bb_lower = float(row.get("bb_lower", close * 0.98))
        open_price = float(row.get("open", close))
        prev_close = float(prev_row.get("close", open_price))
        prev_high = float(prev_row.get("high", close))
        prev_low = float(prev_row.get("low", close))
        ret_5d = float(row.get("returns_5d", 0)) * 100 if "returns_5d" in row.index else 0
        adx = float(row.get("adx_14", 20))

        trend_up = ema_9 > ema_21 > ema_50
        trend_down = ema_9 < ema_21 < ema_50

        # === FACTOR 1: Trend alignment (regime-controlled weight) ===
        ema_base = ema_weight * 0.8
        ema_bonus = ema_weight * 0.2

        if trend_up:
            bull_score += ema_base
        elif trend_down:
            bear_score += ema_base

        if close > ema_21 * 1.005:
            bull_score += ema_bonus
        elif close < ema_21 * 0.995:
            bear_score += ema_bonus

        if adx > 30 and (trend_up or trend_down):
            if trend_up:
                bull_score += 0.5
            else:
                bear_score += 0.5

        if ret_5d > 0:
            bull_score += 0.3
        elif ret_5d < 0:
            bear_score += 0.3

        # === FACTOR 2: Momentum — RSI + MACD ===
        if rsi > 58 and rsi > prev_rsi:
            bull_score += 1.0
        elif rsi < 42 and rsi < prev_rsi:
            bear_score += 1.0

        if macd_hist > 0 and macd_hist > prev_macd_hist:
            bull_score += 1.0
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            bear_score += 1.0

        # === FACTOR 3: Price action — gap + breakout ===
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        if gap_pct > 0.4:
            bull_score += 0.75
        elif gap_pct < -0.4:
            bear_score += 0.75

        if close > prev_high:
            bull_score += 0.75
        elif close < prev_low:
            bear_score += 0.75

        if close > open_price:
            bull_score += 0.3
        elif close < open_price:
            bear_score += 0.3

        # === FACTOR 4: Mean reversion guard ===
        mr_score = mr_weight * 0.67
        mr_penalty = mr_weight * 0.33

        if ret_5d > 5.0:
            bear_score += mr_score + 1.0
            bull_score -= mr_penalty
        elif ret_5d > 3.5:
            bear_score += mr_score
            bull_score -= mr_penalty
        elif ret_5d < -5.0:
            bull_score += mr_score + 1.0
            bear_score -= mr_penalty
        elif ret_5d < -3.5:
            bull_score += mr_score
            bear_score -= mr_penalty

        # === FACTOR 5: Bollinger position ===
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        if bb_pos > 0.85:
            bull_score += 0.5
        elif bb_pos < 0.15:
            bear_score += 0.5

        prev_bb_upper = float(prev_row.get("bb_upper", prev_close * 1.02))
        prev_bb_lower = float(prev_row.get("bb_lower", prev_close * 0.98))
        bb_width = (bb_upper - bb_lower) / close if close > 0 else 0
        prev_bb_width = (prev_bb_upper - prev_bb_lower) / prev_close if prev_close > 0 else 0
        if prev_bb_width > 0 and bb_width > prev_bb_width * 1.20:
            if bull_score >= bear_score:
                bull_score += 0.25
            else:
                bear_score += 0.25

        # === FACTOR 6: VIX (scaled by instrument's vix_multiplier) ===
        effective_vix = vix * inst.vix_multiplier
        if effective_vix < 13:
            bull_score += 0.5
        elif effective_vix > 20:
            bear_score += 0.5

        # VIX momentum (per-instrument tracking)
        prev_vix = self._prev_vix.get(inst.name, 0)
        if prev_vix > 0:
            vix_delta = vix - prev_vix
            if vix > 20 and vix_delta < -1.0:
                bull_score += 0.3
            elif vix_delta > 1.0:
                bear_score += 0.3
        self._prev_vix[inst.name] = vix

        # === FACTOR 7: ML — DISABLED (v9) ===

        # === FACTOR 10: Global Macro ===
        if nifty_df is not None and not nifty_df.empty:
            last_row = nifty_df.iloc[-1]
            dxy_mom = float(last_row.get("dxy_momentum_5d", 0))
            sp_nifty_corr = float(last_row.get("sp500_nifty_corr_20d", 0.5))
            global_risk = float(last_row.get("global_risk_score", 0))
            sp500_ret = float(last_row.get("sp500_prev_return", 0))

            if dxy_mom > 0.5:
                bear_score += 0.5
            elif dxy_mom < -0.5:
                bull_score += 0.5
            if sp_nifty_corr > 0.5:
                if sp500_ret > 0.5:
                    bull_score += 0.5
                elif sp500_ret < -0.5:
                    bear_score += 0.5
            if global_risk < -1.0:
                bear_score += 0.5
            elif global_risk > 1.0:
                bull_score += 0.5

        # === FACTOR 8: OI/PCR consensus ===
        if pcr_val >= 1.3:
            bull_score += 1.0
        elif pcr_val >= 1.1:
            bull_score += 0.5
        elif pcr_val <= 0.7:
            bear_score += 1.0
        elif pcr_val <= 0.9:
            bear_score += 0.5

        # OI support/resistance from instrument's own option chain
        oi_resistance = oi_data.get("max_call_oi_strike", 0)
        oi_support = oi_data.get("max_put_oi_strike", 0)
        spot = close

        if oi_resistance > 0 and oi_support > 0 and spot > 0:
            dist_to_resistance = (oi_resistance - spot) / spot * 100
            dist_to_support = (spot - oi_support) / spot * 100

            if dist_to_support < 0.5:
                bull_score += 1.0
            elif dist_to_resistance < 0.5:
                bear_score += 1.0
            elif dist_to_support < 1.0:
                bull_score += 0.5
            elif dist_to_resistance < 1.0:
                bear_score += 0.5

        # === FACTOR 9: Volume Confirmation ===
        volume = float(row.get("volume", 0))
        vol_series = df.get("volume")
        vol_ma = 0.0
        if vol_series is not None and len(vol_series) >= 20:
            vol_ma = float(vol_series.iloc[-20:].mean())

        if vol_ma > 0 and volume > 0:
            vol_ratio = volume / vol_ma
            if vol_ratio > 1.3:
                if close > open_price:
                    bull_score += 1.0
                elif close < open_price:
                    bear_score += 1.0
            elif vol_ratio < 0.7:
                if close > open_price:
                    bull_score -= 0.3
                elif close < open_price:
                    bear_score -= 0.3

        # === Direction decision ===
        if bull_score > bear_score:
            direction = "CE"
        elif bear_score > bull_score:
            direction = "PE"
        else:
            direction = ""

        return bull_score, bear_score, direction

    # ─────────────────────────────────────────
    # Internal: Regime
    # ─────────────────────────────────────────

    @staticmethod
    def _compute_regime(inst: InstrumentConfig, df: pd.DataFrame, vix: float) -> str:
        """Simplified per-instrument regime using ADX + VIX × multiplier."""
        adx = float(df.iloc[-1].get("adx_14", 20))
        effective_vix = vix * inst.vix_multiplier

        if effective_vix > 22:
            return "VOLATILE"
        if adx > inst.adx_threshold:
            return "TRENDING"
        return "RANGEBOUND"

    @staticmethod
    def _regime_weights(regime: str) -> tuple[float, float]:
        """Return (ema_weight, mean_reversion_weight) for the regime."""
        if regime == "TRENDING":
            return 2.5, 1.5
        elif regime == "VOLATILE":
            return 0.5, 1.0
        return 1.0, 2.5  # RANGEBOUND

    # ─────────────────────────────────────────
    # Internal: Would-Trade Decision
    # ─────────────────────────────────────────

    def _determine_would_trade(
        self,
        inst: InstrumentConfig,
        score_diff: float,
        regime: str,
        vix: float,
        direction: str,
    ) -> tuple[bool, str, str]:
        """
        Apply same conviction thresholds as NIFTY.
        Returns: (would_trade, trade_type, blocking_reason)
        """
        if not direction:
            return False, "", "NO_DIRECTION"

        effective_vix = vix * inst.vix_multiplier
        if effective_vix > 35:
            return False, "", "VIX_TOO_HIGH"

        thresholds = _conviction_thresholds()
        threshold = thresholds.get(regime, 2.0)
        if score_diff < threshold:
            return False, "", f"BELOW_THRESHOLD_{regime}({score_diff:.1f}<{threshold})"

        # Trade type (mirrors options_buyer._determine_trade_type)
        if regime in ("VOLATILE", "ELEVATED"):
            if score_diff >= 2.0:
                trade_type = "CREDIT_SPREAD"
            else:
                return False, "", f"VOLATILE_SKIP({score_diff:.1f})"
        elif score_diff >= 3.0:
            trade_type = "NAKED_BUY"
        elif regime == "RANGEBOUND" and score_diff >= 2.5:
            trade_type = "NAKED_BUY"
        else:
            return False, "", f"NO_TRADE_TYPE({regime},{score_diff:.1f})"

        return True, trade_type, ""

    # ─────────────────────────────────────────
    # Internal: Option Chain Fetch
    # ─────────────────────────────────────────

    def _fetch_option_chain(self, inst: InstrumentConfig) -> tuple[dict, float]:
        """Fetch option chain data with 5-min cache. Returns (oi_levels_dict, pcr_value)."""
        import time as _time

        cached = self._oi_cache.get(inst.name)
        if cached and (_time.time() - cached[0]) < self._OI_CACHE_TTL:
            return cached[1], cached[2]

        try:
            expiry = self.resolver.get_weekly_expiry(inst.name)
            chain = self.fetcher.get_option_chain(inst.upstox_symbol, expiry.isoformat())
            oi_data = chain.get("oi_levels", {})
            pcr_val = chain.get("pcr", {}).get("pcr_oi", 1.0)
            self._oi_cache[inst.name] = (_time.time(), oi_data, pcr_val)
            return oi_data, pcr_val
        except Exception as e:
            logger.debug(f"InstrumentLogger: {inst.name} option chain failed: {e}")
            return {}, 1.0
