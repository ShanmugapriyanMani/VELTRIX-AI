"""
VELTRIX — Options Buyer Strategy V4.

9-factor scoring (all strategies unified) + intraday confirmation:
1. Compute 9-factor direction score:
   - EMA stack, RSI+MACD momentum, Price action, Mean reversion, BB, VIX
   - ML prediction (from ML strategy)
   - OI/PCR consensus (from Options OI strategy — support/resistance + PCR)
   - Volume confirmation (F9 — volume vs 20-day avg)
2. Regime conviction filter:
   - 9:30–11:00 → need 2/4 confirmations, score_diff >= regime min
   - 11:00–1:00 → need 1/4 confirmations, relaxed conviction
   - 1:00–2:30 → need 2/4 confirmations, conviction +0.5 (harder)
   - After 2:30 → NO new trades allowed
3. Wait for intraday confirmations (Price vs Open, RSI, morning breakout, PCR)
4. Buy ATM CE (bullish) or PE (bearish)
5. Max 2 trades/day, VIX-adaptive SL/TP, force exit 15:10
6. Dynamic lots: 1–7 lots based on ₹25K deploy / ₹10K risk caps

All strategies feed INTO this single scoring function — unified direction.
"""

from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.config.env_loader import get_config
from src.data.features import FeatureEngine
from src.strategies.base import BaseStrategy, Signal, SignalDirection


class OptionsBuyerStrategy(BaseStrategy):
    """
    Directional options buying with multi-factor scoring + intraday confirmation.

    Flow:
    1. Multi-factor daily scoring → determines CE or PE direction
    2. Capture morning range (9:15-9:30 high/low)
    3. Wait for 2/4 intraday confirmations matching scored direction
    4. Buy nearest weekly ATM CE/PE with delta-optimized strike selection
    5. VIX-adaptive SL/TP, force exit 15:10
    """

    # Delta targets by conviction × regime (V4)
    # Higher conviction → closer to ATM (higher delta)
    DELTA_TARGETS = {
        # conviction ≥ 4.0: aggressive, near ATM
        "high": {
            "TRENDING":   (0.50, 0.58),
            "RANGEBOUND": (0.42, 0.50),
            "VOLATILE":   (0.38, 0.45),
        },
        # conviction 2.0-4.0: balanced
        "medium": {
            "TRENDING":   (0.45, 0.52),
            "RANGEBOUND": (0.35, 0.45),
            "VOLATILE":   (0.30, 0.38),
        },
        # conviction 1.5-2.0: conservative, slight OTM
        "low": {
            "TRENDING":   (0.40, 0.48),
            "RANGEBOUND": (0.30, 0.40),
            "VOLATILE":   (0.25, 0.35),
        },
    }

    # Sweet-spot premium ranges for greeks-based scoring (not hard limits)
    PREMIUM_SWEET_SPOTS = {
        "TRENDING":   {"sweet_min": 90, "sweet_max": 120},
        "RANGEBOUND": {"sweet_min": 100, "sweet_max": 180},
        "VOLATILE":   {"sweet_min": 90, "sweet_max": 150},
    }

    MAX_DEPLOYABLE = 25_000    # ₹25K FIXED deploy cap per trade (never changes)
    MAX_RISK = 10_000          # ₹10K max loss per FULL trade
    MIN_PREMIUM = 80           # Universal minimum premium (avoid illiquid deep OTM)
    MIN_WALLET_BALANCE = 50_000  # Live mode: minimum wallet balance to trade
    BUFFER = 5_000             # ₹5K emergency reserve (never touched)

    # Active trading mode overrides (set via set_active_trading())
    ACTIVE_MAX_TRADES_PER_DAY = 5
    ACTIVE_COOLDOWN_MINUTES = 15
    ACTIVE_CONSEC_SL_HALT = 2
    ACTIVE_LAST_TRADE_TIME = dt_time(14, 45)

    def __init__(self, config_path: str = "config/strategies.yaml"):
        super().__init__("options_buyer", config_path)

        cfg = get_config()

        self.instruments = self.config.get("instruments", ["NIFTY", "BANKNIFTY"])
        self.strike_offset = self.config.get("strike_offset", 1)
        self.premium_sl_pct = cfg.SL_BASE_PCT / 100
        self.premium_tp_pct = cfg.TP_BASE_PCT / 100
        self.max_deployable = cfg.DEPLOY_CAP
        self.max_risk = cfg.RISK_PER_TRADE
        self.force_exit_time = cfg.TRADE_END
        self.min_confidence = self.config.get("min_confidence", 0.65)
        self.MIN_PREMIUM = cfg.MIN_PREMIUM
        self.MIN_WALLET_BALANCE = cfg.MIN_WALLET_BALANCE
        self.BUFFER = cfg.BUFFER

        # Resolver will be injected by main.py
        self._resolver = None
        self._data_fetcher = None

        # Intraday state (reset daily)
        self._direction: dict[str, str] = {}           # {symbol: "CE"/"PE"}
        self._direction_scores: dict[str, tuple] = {}  # {symbol: (bull, bear, diff)}
        self._morning_high: dict[str, float] = {}      # {symbol: float}
        self._morning_low: dict[str, float] = {}       # {symbol: float}
        self._traded_today: set[str] = set()            # symbols already traded
        self._trades_today: dict[str, int] = {}         # {symbol: trade_count}
        self._last_exit_reason: dict[str, str] = {}     # {symbol: "take_profit"/"stop_loss"/...}
        self._logged_today: set[str] = set()            # symbols whose direction was logged

        # Consecutive SL tracking (persists across days)
        self._consec_sl_count: int = 0
        self._consec_sl_direction: str = ""

        # Win/loss streak tracking (persists across days)
        self._streak: int = 0  # positive = wins, negative = losses
        # Whipsaw detection: last 5 outcomes (True=win, False=loss)
        self._recent_outcomes: list[bool] = []

        # Daily trade tracking (reset daily)
        self._full_trades_today: int = 0

        # V4: Direction flip tracking (reset daily)
        self._flips_today: int = 0           # Max 1 flip per day
        self._last_exit_time: dict[str, datetime] = {}  # {symbol: datetime}
        self._position_flat: dict[str, bool] = {}       # {symbol: True if no open position}

        # VIX momentum tracking
        self._prev_vix: float = 0.0

        # Active trading mode (off by default)
        self._active_trading: bool = False
        self._same_day_sl_count: int = 0  # Reset daily, halt at 2 SLs in active mode

        # PLUS spread tracking (reset daily)
        self._naked_trades_today: int = 0
        self._spread_trades_today: int = 0

        # Confirmation timeout: unlock stuck direction after 30 min of failed confirmations
        self._confirm_fail_since: dict[str, datetime] = {}  # {symbol: first_fail_time}
        self._direction_rescores_today: int = 0  # Max 3 re-scores per day

        # SL direction unlock: track per-direction SL count today
        self._sl_count_by_direction_today: dict[str, int] = {}  # {"CE": N, "PE": N}

    def set_active_trading(self, enabled: bool) -> None:
        """Enable/disable active trading mode."""
        self._active_trading = enabled
        if enabled:
            logger.info("OptionsBuyer: ACTIVE TRADING mode enabled")

    def set_resolver(self, resolver) -> None:
        """Inject the OptionsInstrumentResolver."""
        self._resolver = resolver

    def set_data_fetcher(self, data_fetcher) -> None:
        """Inject the data fetcher for live premium lookups."""
        self._data_fetcher = data_fetcher

    def reset_daily(self) -> None:
        """Reset all intraday state for a new trading day."""
        self._direction.clear()
        self._direction_scores.clear()
        self._morning_high.clear()
        self._morning_low.clear()
        self._traded_today.clear()
        self._trades_today.clear()
        self._last_exit_reason.clear()
        self._logged_today.clear()
        self._full_trades_today = 0
        self._flips_today = 0
        self._same_day_sl_count = 0
        self._naked_trades_today = 0
        self._spread_trades_today = 0
        self._last_exit_time.clear()
        self._position_flat.clear()
        self._confirm_fail_since.clear()
        self._direction_rescores_today = 0
        self._sl_count_by_direction_today.clear()

    def record_exit(self, symbol: str, exit_reason: str, direction: str) -> None:
        """Record a trade exit for consecutive SL and streak tracking.

        Also unlocks re-entry after TP/trail wins (max 2 trades/day/symbol).
        V4: tracks exit time for direction flip 90-min gap rule.
        """
        # Track exit reason for re-entry logic
        # Extract index symbol from option symbol (e.g. "NIFTY25500CE" → "NIFTY")
        index_sym = symbol.rstrip("0123456789").rstrip("CEPE") if symbol else ""
        if not index_sym:
            index_sym = symbol
        self._last_exit_reason[index_sym] = exit_reason
        self._last_exit_time[index_sym] = datetime.now()
        self._position_flat[index_sym] = True

        # Determine max trades per day based on mode
        max_trades_per_symbol = self.ACTIVE_MAX_TRADES_PER_DAY if self._active_trading else 2

        # Allow re-entry after TP or trail_stop
        if exit_reason in ("take_profit", "trail_stop"):
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                # Unlock symbol for re-entry — clear direction to re-score
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)
                self._logged_today.discard(index_sym)
                logger.info(
                    f"OptionsBuyer: {exit_reason} on {symbol} — "
                    f"unlocking {index_sym} for trade #{trades + 1}"
                )

        # Active mode: also allow re-entry after EOD exit (position was flat, market had more room)
        if self._active_trading and exit_reason == "eod_exit":
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)
                self._logged_today.discard(index_sym)

        # Active mode: also allow re-entry after SL (direction unlocked)
        if self._active_trading and exit_reason == "stop_loss":
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)  # Direction unlocked — free CE/PE
                self._logged_today.discard(index_sym)
                logger.info(
                    f"OptionsBuyer: ACTIVE SL exit on {symbol} — "
                    f"unlocking {index_sym} for re-entry (SL #{self._same_day_sl_count + 1})"
                )

        if exit_reason == "stop_loss":
            self._same_day_sl_count += 1
            # Track per-direction SL count today
            self._sl_count_by_direction_today[direction] = (
                self._sl_count_by_direction_today.get(direction, 0) + 1
            )
            if direction == self._consec_sl_direction:
                self._consec_sl_count += 1
            else:
                self._consec_sl_direction = direction
                self._consec_sl_count = 1

            # 2 SLs same direction today → unlock direction for re-score
            # (works regardless of ACTIVE_TRADING setting)
            if self._sl_count_by_direction_today[direction] >= 2:
                trades = self._trades_today.get(index_sym, 1)
                if trades < max_trades_per_symbol and self._direction_rescores_today < 3:
                    self._traded_today.discard(index_sym)
                    self._direction.pop(index_sym, None)
                    self._direction_scores.pop(index_sym, None)
                    self._logged_today.discard(index_sym)
                    self._confirm_fail_since.pop(index_sym, None)
                    self._direction_rescores_today += 1
                    logger.info(
                        f"OptionsBuyer: 2 SLs in {direction} today — "
                        f"unlocking {index_sym} for direction re-score "
                        f"(rescore #{self._direction_rescores_today}/3)"
                    )

            # Streak: decrement on loss
            self._streak = min(self._streak - 1, -1)
            self._recent_outcomes.append(False)
        else:
            self._consec_sl_count = 0
            self._consec_sl_direction = ""
            # Streak: increment on win (TP or trail_stop), reset on EOD
            if exit_reason in ("take_profit", "trail_stop"):
                self._streak = max(self._streak + 1, 1)
                self._recent_outcomes.append(True)
            else:
                # EOD exit — count as loss if negative
                self._recent_outcomes.append(False)
        # Keep last 5 outcomes only
        if len(self._recent_outcomes) > 5:
            self._recent_outcomes.pop(0)

    def confirm_execution(self, index_symbol: str) -> None:
        """Called by main.py after a trade is successfully executed."""
        self._trades_today[index_symbol] = self._trades_today.get(index_symbol, 0) + 1
        logger.info(f"OptionsBuyer: Confirmed execution for {index_symbol} (trade #{self._trades_today[index_symbol]})")

    def cancel_signal(self, index_symbol: str) -> None:
        """Called by main.py when execution fails — unblock symbol for retry."""
        self._traded_today.discard(index_symbol)
        logger.info(f"OptionsBuyer: Execution failed for {index_symbol} — unblocked for retry")

    def update(self, data: dict[str, Any]) -> None:
        """No internal state to update via this method."""
        pass

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """
        Generate option buying signals using multi-factor scoring + intraday
        confirmation.

        Expected data keys:
        - "nifty_df": pd.DataFrame (daily NIFTY with technical features)
        - "intraday_df": pd.DataFrame (5-min NIFTY candles)
        - "ml_direction_prob_up": float
        - "ml_direction_prob_down": float
        - "pcr": {symbol: float}
        - "nifty_price": float
        - "regime": str
        - "vix": float
        """
        if not self.enabled:
            return []

        now = datetime.now().time()
        exit_time = dt_time(*[int(x) for x in self.force_exit_time.split(":")])

        # No entry after 15:10
        if now >= exit_time:
            return []

        # Regime-based last-trade cutoff:
        # Conservative: TRENDING=14:30, RANGEBOUND=13:00, VOLATILE=12:00
        # Active:       TRENDING=14:45, RANGEBOUND=14:00, VOLATILE=13:00
        regime = data.get("regime", "")
        if self._active_trading:
            regime_cutoffs = {
                "TRENDING": self.ACTIVE_LAST_TRADE_TIME,  # 14:45
                "RANGEBOUND": dt_time(14, 0),
                "VOLATILE": dt_time(13, 0),
            }
        else:
            _nt_h, _nt_m = (int(x) for x in get_config().NO_NEW_TRADE_AFTER.split(":"))
            regime_cutoffs = {
                "TRENDING": dt_time(_nt_h, _nt_m),
                "RANGEBOUND": dt_time(13, 0),
                "VOLATILE": dt_time(12, 0),
            }
        last_trade_cutoff = regime_cutoffs.get(regime, dt_time(14, 30))
        if now >= last_trade_cutoff:
            return []

        # Active mode: halt if 2 same-day SLs
        if self._active_trading and self._same_day_sl_count >= self.ACTIVE_CONSEC_SL_HALT:
            return []

        # Active mode: cooldown — wait 15 min after last exit
        if self._active_trading:
            now_dt = datetime.now()
            for sym, exit_time in self._last_exit_time.items():
                mins_since = (now_dt - exit_time).total_seconds() / 60
                if mins_since < self.ACTIVE_COOLDOWN_MINUTES:
                    return []  # Still in cooldown

        # Need resolver for instrument keys
        if self._resolver is None:
            logger.warning("OptionsBuyer: No instrument resolver set")
            return []

        # Note: Consecutive SL direction pause is checked per-symbol
        # after direction scoring in _evaluate_symbol()

        signals = []
        regime = data.get("regime", "")

        # Max trades per day: active=5, conservative=2
        max_trades = self.ACTIVE_MAX_TRADES_PER_DAY if self._active_trading else 2

        for symbol in self.instruments:
            if symbol in self._traded_today:
                continue

            # Max trades per day per symbol
            if self._trades_today.get(symbol, 0) >= max_trades:
                continue

            signal = self._evaluate_symbol(symbol, data, regime)
            if signal:
                # Mark as having a pending signal (blocks duplicate signals this iteration)
                # Actual trade count incremented by confirm_execution() after success
                self._traded_today.add(symbol)
                signals.append(signal)

        return signals

    def _compute_direction_score(
        self,
        symbol: str,
        data: dict[str, Any],
        regime: str,
    ) -> tuple[float, float, str]:
        """
        Compute multi-factor bull/bear score using ALL strategy inputs.

        Returns: (bull_score, bear_score, direction "CE"/"PE" or "")
        8 factors: EMA, Momentum, Price Action, Mean Reversion, BB, VIX, ML, OI/PCR.
        All strategies feed into this single scoring — unified direction.
        """
        nifty_df = data.get("nifty_df")
        if nifty_df is None or nifty_df.empty or len(nifty_df) < 50:
            return self._ml_only_direction(data)

        row = nifty_df.iloc[-1]
        prev_row = nifty_df.iloc[-2]

        bull_score = 0.0
        bear_score = 0.0

        # Get values (with safe defaults)
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
        vix = data.get("vix", 15)

        trend_up = ema_9 > ema_21 > ema_50
        trend_down = ema_9 < ema_21 < ema_50

        adx = float(row.get("adx_14", 20))

        # === FACTOR 1: Trend alignment (regime-controlled weight) ===
        # Regime controls EMA weight: TRENDING=2.5, RANGEBOUND=1.0, VOLATILE=0.5
        ema_weight = data.get("ema_weight", 2.5)
        ema_base = ema_weight * 0.8  # Base EMA score scaled by regime
        ema_bonus = ema_weight * 0.2  # Bonus for close vs EMA21

        if trend_up:
            bull_score += ema_base
        elif trend_down:
            bear_score += ema_base

        if close > ema_21 * 1.005:
            bull_score += ema_bonus
        elif close < ema_21 * 0.995:
            bear_score += ema_bonus

        # ADX trend strength: strong trends are more reliable
        if adx > 30 and (trend_up or trend_down):
            if trend_up:
                bull_score += 0.5
            else:
                bear_score += 0.5

        # 5-day trend direction: +0.3 nudge
        if ret_5d > 0:
            bull_score += 0.3
        elif ret_5d < 0:
            bear_score += 0.3

        # === FACTOR 2: Momentum — RSI + MACD (weight: 2.0) ===
        if rsi > 58 and rsi > prev_rsi:
            bull_score += 1.0
        elif rsi < 42 and rsi < prev_rsi:
            bear_score += 1.0

        if macd_hist > 0 and macd_hist > prev_macd_hist:
            bull_score += 1.0
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            bear_score += 1.0

        # === FACTOR 3: Price action — gap + breakout (weight: 1.5) ===
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        if gap_pct > 0.4:
            bull_score += 0.75
        elif gap_pct < -0.4:
            bear_score += 0.75

        if close > prev_high:
            bull_score += 0.75
        elif close < prev_low:
            bear_score += 0.75

        # Candle body direction: +0.3 for strong body
        if close > open_price:
            bull_score += 0.3
        elif close < open_price:
            bear_score += 0.3

        # === FACTOR 4: Mean reversion guard (regime-controlled weight) ===
        # Regime controls mean rev weight: RANGEBOUND=2.5, TRENDING=1.5, VOLATILE=1.0
        mr_weight = data.get("mean_reversion_weight", 1.5)
        mr_score = mr_weight * 0.67  # Scale to ~1.0 at default weight
        mr_penalty = mr_weight * 0.33

        if ret_5d > 5.0:
            # Extreme overbought — strong mean reversion signal
            bear_score += mr_score + 1.0
            bull_score -= mr_penalty
        elif ret_5d > 3.5:
            bear_score += mr_score
            bull_score -= mr_penalty
        elif ret_5d < -5.0:
            # Extreme oversold — strong mean reversion signal
            bull_score += mr_score + 1.0
            bear_score -= mr_penalty
        elif ret_5d < -3.5:
            bull_score += mr_score
            bear_score -= mr_penalty

        # === FACTOR 5: Bollinger position (weight: 1.0) ===
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        if bb_pos > 0.85:
            bull_score += 0.5
        elif bb_pos < 0.15:
            bear_score += 0.5

        # BB width expanding >20% vs previous day → volatility breakout +0.25
        prev_bb_upper = float(prev_row.get("bb_upper", prev_close * 1.02))
        prev_bb_lower = float(prev_row.get("bb_lower", prev_close * 0.98))
        bb_width = (bb_upper - bb_lower) / close if close > 0 else 0
        prev_bb_width = (prev_bb_upper - prev_bb_lower) / prev_close if prev_close > 0 else 0
        if prev_bb_width > 0 and bb_width > prev_bb_width * 1.20:
            # Expanding BB → breakout conditions, boost current direction
            if bull_score >= bear_score:
                bull_score += 0.25
            else:
                bear_score += 0.25

        # === FACTOR 6: VIX (weight: 0.5) ===
        if vix < 13:
            bull_score += 0.5
        elif vix > 20:
            bear_score += 0.5

        # VIX momentum: falling VIX from >20 → bullish, rising VIX toward 20 → bearish
        if self._prev_vix > 0:
            vix_delta = vix - self._prev_vix
            if vix > 20 and vix_delta < -1.0:
                bull_score += 0.3  # VIX falling from elevated = fear easing
            elif vix_delta > 1.0:
                bear_score += 0.3  # VIX rising = fear increasing
        self._prev_vix = vix

        # === FACTOR 7: ML confidence (weight: 0.3 — informational only) ===
        # ML accuracy is ~48% (worse than coin flip). Only use as a minor
        # nudge when confidence is very high (>0.70), never as a driver.
        ml_prob_up = data.get("ml_direction_prob_up", 0.5)
        ml_prob_down = data.get("ml_direction_prob_down", 0.5)

        if ml_prob_up > 0.70:
            bull_score += 0.3
        elif ml_prob_down > 0.70:
            bear_score += 0.3

        # === FACTOR 8: OI/PCR consensus — Options OI strategy input (weight: 2.0) ===
        pcr_data = data.get("pcr", {})
        oi_data = data.get("oi_levels", {})
        nifty_oi = oi_data.get("NIFTY", {})
        pcr_val = pcr_data.get(symbol, pcr_data.get("NIFTY", 1.0))
        if isinstance(pcr_val, dict):
            pcr_val = pcr_val.get("pcr_oi", 1.0)

        # PCR signal: high PCR = bullish (puts being written), low PCR = bearish
        if pcr_val >= 1.3:
            bull_score += 1.0
        elif pcr_val >= 1.1:
            bull_score += 0.5
        elif pcr_val <= 0.7:
            bear_score += 1.0
        elif pcr_val <= 0.9:
            bear_score += 0.5

        # OI support/resistance: price near OI support = bullish, near resistance = bearish
        oi_resistance = nifty_oi.get("max_call_oi_strike", 0)
        oi_support = nifty_oi.get("max_put_oi_strike", 0)
        spot = data.get("nifty_price", close)

        if oi_resistance > 0 and oi_support > 0 and spot > 0:
            dist_to_resistance = (oi_resistance - spot) / spot * 100
            dist_to_support = (spot - oi_support) / spot * 100

            if dist_to_support < 0.5:
                # Near OI support — bounce expected → bullish
                bull_score += 1.0
            elif dist_to_resistance < 0.5:
                # Near OI resistance — rejection expected → bearish
                bear_score += 1.0
            elif dist_to_support < 1.0:
                bull_score += 0.5
            elif dist_to_resistance < 1.0:
                bear_score += 0.5

        # === FACTOR 9: Volume Confirmation (weight: 1.0) ===
        # Volume > 1.3× 20-day average confirms direction, divergence opposes
        volume = float(row.get("volume", 0))
        vol_ma = float(row.get("volume_ma_20", 0)) if "volume_ma_20" in row.index else 0
        if vol_ma <= 0:
            # Compute from DataFrame if not pre-computed
            vol_series = data.get("nifty_df", pd.DataFrame()).get("volume")
            if vol_series is not None and len(vol_series) >= 20:
                vol_ma = float(vol_series.iloc[-20:].mean())

        if vol_ma > 0 and volume > 0:
            vol_ratio = volume / vol_ma
            if vol_ratio > 1.3:
                # High volume confirms current direction
                if close > open_price:
                    bull_score += 1.0
                elif close < open_price:
                    bear_score += 1.0
            elif vol_ratio < 0.7:
                # Low volume weakens current direction (divergence)
                if close > open_price:
                    bull_score -= 0.3
                elif close < open_price:
                    bear_score -= 0.3

        # === Consecutive SL nudge: 3+ SLs → nudge opposite direction (relaxed from 2) ===
        if self._consec_sl_count >= 3:
            if self._consec_sl_direction == "CE":
                bear_score += 0.5
            elif self._consec_sl_direction == "PE":
                bull_score += 0.5

        # === Direction decision ===
        if bull_score > bear_score:
            direction = "CE"
        elif bear_score > bull_score:
            direction = "PE"
        else:
            direction = ""

        return bull_score, bear_score, direction

    def _ml_only_direction(
        self, data: dict[str, Any]
    ) -> tuple[float, float, str]:
        """Fallback: use ML prediction only when daily data is insufficient."""
        ml_prob_up = data.get("ml_direction_prob_up", 0.5)
        ml_prob_down = data.get("ml_direction_prob_down", 0.5)

        bull_score = ml_prob_up * 3.0
        bear_score = ml_prob_down * 3.0

        if bull_score > bear_score:
            return bull_score, bear_score, "CE"
        elif bear_score > bull_score:
            return bull_score, bear_score, "PE"
        return bull_score, bear_score, ""

    def _compute_position_size(
        self,
        symbol: str,
        regime: str,
        score_diff: float,
        vix: float,
        direction: str,
        sl_pct: float,
        daily_loss_pct: float = 0.0,
        circuit_breaker_warning: bool = False,
        wallet_balance: float = 0.0,
        mode: str = "paper",
        trade_type: str = "FULL",
    ) -> dict[str, Any] | None:
        """Determine lot quantity and max allowable premium.

        FIXED deploy/risk caps — no wallet tiers, no scaling:
        - MAX_DEPLOYABLE = ₹25,000 (ALWAYS, regardless of wallet)
        - MAX_RISK = ₹10,000 per trade
        - lots = min(floor(₹25K/(prem×65)), floor(risk/(prem×SL%×65)))
        - lots = max(1, lots)  — no max lots cap

        Live mode: wallet < ₹50K → BLOCKED + Telegram alert.
        """
        full_lot = self._resolver.get_lot_size(symbol) if self._resolver else 65

        # Live mode: simple wallet check (no tiers)
        if mode == "live" and wallet_balance > 0:
            if wallet_balance < self.MIN_WALLET_BALANCE:
                logger.warning(
                    f"OptionsBuyer: Wallet ₹{wallet_balance:,.0f} < "
                    f"min ₹{self.MIN_WALLET_BALANCE:,.0f} — BLOCKED"
                )
                return None

        # FIXED caps — all trades use FULL risk
        max_risk = self.max_risk
        reason = "full"

        # Max premium for strike selection (1-lot permissive)
        deploy_max_1lot = self.max_deployable / full_lot
        risk_max_1lot = max_risk / (full_lot * sl_pct) if sl_pct > 0 else deploy_max_1lot
        max_premium = min(deploy_max_1lot, risk_max_1lot)

        min_premium = self.MIN_PREMIUM  # ₹80

        if max_premium < min_premium:
            logger.warning(
                f"OptionsBuyer: No affordable premium — max ₹{max_premium:.0f} "
                f"< min ₹{min_premium}"
            )
            return None

        logger.info(
            f"OptionsBuyer sizing: {reason} lot={full_lot} | "
            f"max_premium=₹{max_premium:.0f} min=₹{min_premium} | "
            f"risk_cap=₹{max_risk:,.0f} deploy_cap=₹{self.max_deployable:,.0f} SL={sl_pct:.0%}"
        )

        return {
            "lot_qty": full_lot,
            "max_premium": round(max_premium, 1),
            "min_premium": min_premium,
            "reason": reason,
            "max_deployable": self.max_deployable,
            "max_risk": max_risk,
            "full_lot": full_lot,
            "trade_type": trade_type,
        }

    def _compute_lots(
        self, premium: float, full_lot: int, sl_pct: float,
        max_risk: float,
    ) -> int:
        """Dynamic lot calculation after premium is known.

        lots_by_deploy = floor(₹25K / (premium × 65))
        lots_by_risk   = floor(risk / (premium × SL% × 65))
        lots = min(lots_by_deploy, lots_by_risk)
        lots = max(1, lots)
        No max lots cap — deploy and risk naturally limit lots.
        """
        if premium <= 0:
            return full_lot  # Fallback 1 lot

        lots_by_deploy = int(self.max_deployable / (premium * full_lot))
        lots_by_risk = int(max_risk / (premium * sl_pct * full_lot)) if sl_pct > 0 else lots_by_deploy
        lots = min(lots_by_deploy, lots_by_risk)
        lots = max(1, lots)

        qty = lots * full_lot
        logger.info(
            f"OptionsBuyer lots: {lots} lot(s) = {qty} qty | "
            f"premium=₹{premium:.0f} × {full_lot} = ₹{premium * full_lot:,.0f}/lot | "
            f"deploy={lots_by_deploy}L risk={lots_by_risk}L"
        )
        return qty

    def _determine_trade_type(self, regime: str, score_diff: float) -> str:
        """PLUS decision tree for trade type selection.

        VOLATILE + conviction >= 2.0 → CREDIT_SPREAD
        VOLATILE + conviction < 2.0  → SKIP
        Any regime + conviction >= 3.0 → NAKED_BUY
        Any regime + conviction < 3.0 → DEBIT_SPREAD
        """
        if regime == "VOLATILE":
            return "CREDIT_SPREAD" if score_diff >= 2.0 else "SKIP"
        return "NAKED_BUY" if score_diff >= 3.0 else "DEBIT_SPREAD"

    def _build_spread_signal(
        self,
        symbol: str,
        direction: str,
        trade_type: str,
        score_diff: float,
        regime: str,
        data: dict,
    ) -> Optional[Signal]:
        """Build a two-leg spread signal for PLUS mode.

        For DEBIT_SPREAD: Buy ATM + Sell OTM (same option type).
        For CREDIT_SPREAD: Sell near-OTM + Buy far-OTM protection (opposite type).
        """
        cfg = get_config()
        spread_width = cfg.SPREAD_WIDTH
        spot = data.get("close", data.get("ltp", 0))
        if spot <= 0:
            return None

        # Get lot size from resolver
        full_lot = 65  # NIFTY default
        if self._resolver:
            full_lot = self._resolver.get_lot_size(symbol) or 65

        # ATM strike
        strike_gap = 50
        atm_strike = round(spot / strike_gap) * strike_gap

        if trade_type == "DEBIT_SPREAD":
            # Bull Call Spread (CE) or Bear Put Spread (PE)
            leg1_strike = atm_strike
            leg2_strike = (atm_strike + spread_width) if direction == "CE" else (atm_strike - spread_width)
            leg1_side = "BUY"
            leg2_side = "SELL"
            opt_type = direction  # Same option type for both legs
        else:  # CREDIT_SPREAD
            # Bull Put Spread (bullish) or Bear Call Spread (bearish)
            opt_type = "PE" if direction == "CE" else "CE"
            if opt_type == "PE":
                leg1_strike = atm_strike - 100  # Sell near-OTM PE
                leg2_strike = leg1_strike - spread_width  # Buy far-OTM PE
            else:
                leg1_strike = atm_strike + 100  # Sell near-OTM CE
                leg2_strike = leg1_strike + spread_width  # Buy far-OTM CE
            leg1_side = "SELL"
            leg2_side = "BUY"

        # Resolve instruments
        if not self._resolver:
            logger.warning("OptionsBuyer: No resolver for spread legs")
            return None

        expiry = self._resolver.get_weekly_expiry(symbol)
        leg1_key = self._resolver.get_instrument_key(symbol, leg1_strike, expiry, opt_type)
        leg2_key = self._resolver.get_instrument_key(symbol, leg2_strike, expiry, opt_type)
        if not leg1_key or not leg2_key:
            logger.warning(f"OptionsBuyer: Cannot resolve spread legs for {symbol}")
            return None

        # Get live premiums
        leg1_prem = 0.0
        leg2_prem = 0.0
        if self._data_fetcher:
            try:
                q1 = self._data_fetcher.get_live_quote(leg1_key)
                q2 = self._data_fetcher.get_live_quote(leg2_key)
                leg1_prem = q1.get("ltp", 0) if q1 else 0
                leg2_prem = q2.get("ltp", 0) if q2 else 0
            except Exception:
                return None

        if leg1_prem <= 0 or leg2_prem <= 0:
            return None

        # Spread economics
        if trade_type == "DEBIT_SPREAD":
            net_premium = leg1_prem - leg2_prem  # Net debit
            if net_premium <= 0:
                return None
            max_loss_per_unit = net_premium
            max_profit_per_unit = spread_width - net_premium
        else:  # CREDIT_SPREAD
            net_premium = leg1_prem - leg2_prem  # Net credit (sell - buy)
            if net_premium <= 0:
                return None
            max_loss_per_unit = spread_width - net_premium
            max_profit_per_unit = net_premium

        # Lot sizing
        if trade_type == "DEBIT_SPREAD":
            lots_by_deploy = int(cfg.DEPLOY_CAP / (net_premium * full_lot)) if net_premium > 0 else 1
            lots_by_risk = int(cfg.RISK_PER_TRADE / (net_premium * full_lot)) if net_premium > 0 else 1
            lots = max(1, min(lots_by_deploy, lots_by_risk))
        else:
            lots_by_risk = int(cfg.RISK_PER_TRADE / (max_loss_per_unit * full_lot)) if max_loss_per_unit > 0 else 1
            lots = max(1, lots_by_risk)
        qty = lots * full_lot

        # Risk check
        total_max_loss = max_loss_per_unit * qty
        if total_max_loss > cfg.RISK_PER_TRADE:
            logger.info(f"OptionsBuyer: Spread max loss ₹{total_max_loss:.0f} > risk cap ₹{cfg.RISK_PER_TRADE:.0f}")
            return None

        signal = Signal(
            strategy=self.name,
            symbol=f"{symbol}{int(leg1_strike)}{opt_type}",
            direction=SignalDirection.BUY if trade_type == "DEBIT_SPREAD" else SignalDirection.SELL,
            confidence=min(0.55 + score_diff * 0.06, 0.95),
            score=score_diff,
            price=net_premium,
            regime=regime,
            features={
                "is_options": True,
                "trade_type": trade_type,
                "is_spread": True,
                "leg1_instrument_key": leg1_key,
                "leg1_strike": leg1_strike,
                "leg1_side": leg1_side,
                "leg1_premium": leg1_prem,
                "leg2_instrument_key": leg2_key,
                "leg2_strike": leg2_strike,
                "leg2_side": leg2_side,
                "leg2_premium": leg2_prem,
                "option_type": opt_type,
                "spread_width": spread_width,
                "net_premium": net_premium,
                "max_profit": max_profit_per_unit * qty,
                "max_loss": total_max_loss,
                "lot_size": qty,
                "lots": lots,
                "index_symbol": symbol,
            },
        )

        logger.info(
            f"OptionsBuyer: {trade_type} signal for {symbol} | "
            f"{leg1_side} {leg1_strike}{opt_type}@₹{leg1_prem:.0f} + "
            f"{leg2_side} {leg2_strike}{opt_type}@₹{leg2_prem:.0f} | "
            f"net=₹{net_premium:.0f} qty={qty} max_loss=₹{total_max_loss:.0f}"
        )
        return signal

    def _evaluate_symbol(
        self,
        symbol: str,
        data: dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Evaluate one index using 8-factor scoring + intraday confirmation.

        Step 1: Compute direction from 8-factor scoring (all strategies feed in)
        Step 2: Capture morning range (9:15-9:30)
        Step 3: Check 4 intraday confirmations matching scored direction
        Step 4: Generate signal if confirmed

        Relaxed mode: After 11:00, lowers conviction threshold to ensure
        at least 1 trade per day.
        """
        now = datetime.now().time()

        # ── Step 1: Compute Direction from Multi-Factor Scoring ──
        # Re-compute scores periodically (every cycle) to pick up intraday changes
        # but only change direction once locked
        if symbol not in self._direction:
            bull_score, bear_score, direction = self._compute_direction_score(
                symbol, data, regime
            )
            score_diff = abs(bull_score - bear_score)

            # Store scores for logging
            self._direction_scores[symbol] = (bull_score, bear_score, score_diff)

            # Day-of-week adjustment:
            # Monday (30% WR historically) needs extra conviction
            # Friday (78% WR) gets a direction boost
            day_of_week = datetime.now().weekday()  # 0=Mon, 4=Fri
            is_monday = day_of_week == 0
            is_friday = day_of_week == 4

            if is_friday:
                # Friday boost: nudge the winning direction
                if bull_score > bear_score:
                    bull_score += 0.5
                elif bear_score > bull_score:
                    bear_score += 0.5
                score_diff = abs(bull_score - bear_score)

            # Consecutive SL nudge (V2.2): 3+ SLs same direction → +0.5 opposite (relaxed from 2)
            if self._consec_sl_count >= 3 and self._consec_sl_direction:
                if self._consec_sl_direction == "CE":
                    bear_score += 0.5  # Nudge toward PE
                else:
                    bull_score += 0.5  # Nudge toward CE
                score_diff = abs(bull_score - bear_score)
                logger.info(
                    f"OptionsBuyer: SL nudge — {self._consec_sl_count} consecutive "
                    f"{self._consec_sl_direction} SLs, +0.5 to opposite"
                )

            # Whipsaw detection: 3+ alternations in last 5 trades → boost conviction
            whipsaw_extra = 0.0
            if len(self._recent_outcomes) >= 5:
                alternations = sum(
                    1 for j in range(1, len(self._recent_outcomes))
                    if self._recent_outcomes[j] != self._recent_outcomes[j - 1]
                )
                if alternations >= 3:
                    whipsaw_extra = 0.5
                    logger.info(
                        f"OptionsBuyer: Whipsaw detected — {alternations} alternations "
                        f"in last 5 trades, +0.5 conviction needed"
                    )

            # Conviction threshold — V4 time gates:
            # 9:30–11:00 → standard conviction
            # 11:00–1:00 → relaxed (-0.5)
            # 1:00–2:30  → conviction +0.5 (harder to enter late)
            # After 2:30 → NO new trades (handled in generate_signals)
            monday_extra = 0.5 if is_monday else 0.0

            if now >= dt_time(13, 0):
                mode = f"{regime}_AFTERNOON"
            elif now >= dt_time(11, 0):
                mode = f"{regime}_RELAXED"
            else:
                mode = regime
            if is_monday:
                mode += "_MON"

            # ── Trade type selection ──
            cfg = get_config()
            if cfg.TRADING_STAGE == "PLUS":
                # PLUS decision tree
                trade_type = self._determine_trade_type(regime, score_diff)
                if trade_type == "SKIP":
                    if symbol not in self._logged_today:
                        logger.info(
                            f"OptionsBuyer: VOLATILE + low conviction → SKIP {symbol} "
                            f"(diff={score_diff:.1f})"
                        )
                        self._logged_today.add(symbol)
                    return None

                # Per-type daily limits: max 2 naked + 2 spreads
                if trade_type == "NAKED_BUY" and self._naked_trades_today >= 2:
                    return None
                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD") and self._spread_trades_today >= 2:
                    return None

                # For spreads, build signal and return early
                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD"):
                    self._direction[symbol] = direction
                    spread_signal = self._build_spread_signal(
                        symbol, direction, trade_type, score_diff, regime, data,
                    )
                    if spread_signal:
                        self._spread_trades_today += 1
                    return spread_signal

                # NAKED_BUY falls through to existing signal construction
            else:
                # BASIC: original logic (unchanged)
                # TRENDING ≥1.75, RANGEBOUND ≥2.0, VOLATILE ≥2.5
                if regime == "VOLATILE":
                    full_threshold = 2.5 + monday_extra + whipsaw_extra
                elif regime == "RANGEBOUND":
                    full_threshold = 2.0 + monday_extra + whipsaw_extra
                else:  # TRENDING
                    full_threshold = 1.75 + monday_extra + whipsaw_extra

                # Afternoon: FULL +0.5 (harder to enter)
                if now >= dt_time(13, 0):
                    full_threshold += 0.5

                if score_diff >= full_threshold and direction != "":
                    trade_type = "FULL"
                else:
                    if symbol not in self._logged_today:
                        logger.info(
                            f"OptionsBuyer: No conviction for {symbol} — "
                            f"bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} "
                            f"(need >= {full_threshold:.1f} in {mode}) | regime={regime}"
                        )
                        self._logged_today.add(symbol)
                    return None

            self._direction[symbol] = direction
            # Store trade type for use later in this method
            self._current_trade_type = trade_type

            if symbol not in self._logged_today:
                ml_prob_up = data.get("ml_direction_prob_up", 0.5)
                ml_prob_down = data.get("ml_direction_prob_down", 0.5)
                pcr_data = data.get("pcr", {})
                pcr_val = pcr_data.get(symbol, pcr_data.get("NIFTY", 1.0))
                if isinstance(pcr_val, dict):
                    pcr_val = pcr_val.get("pcr_oi", 1.0)
                logger.info(
                    f"OptionsBuyer: {symbol} direction={direction} ({mode} mode) | "
                    f"bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} | "
                    f"ML P(up)={ml_prob_up:.3f} P(down)={ml_prob_down:.3f} | "
                    f"PCR={pcr_val:.2f} | regime={regime}"
                )
                self._logged_today.add(symbol)

        direction = self._direction[symbol]

        # ── Consecutive SL protection (V2.2) ──
        # 3+ SLs in same direction → block THAT direction entirely
        if self._consec_sl_count >= 3 and direction == self._consec_sl_direction:
            if "SL_BLOCK" not in self._logged_today:
                logger.info(
                    f"OptionsBuyer: BLOCKED — {self._consec_sl_count} consecutive "
                    f"{self._consec_sl_direction} stop losses, blocking {direction}"
                )
                self._logged_today.add("SL_BLOCK")
            return None

        # ── Step 2: Get intraday data and capture morning range ──
        intraday_df = data.get("intraday_df")
        if intraday_df is None or intraday_df.empty or len(intraday_df) < 3:
            return None

        # Capture morning range from first 3 bars (9:15, 9:20, 9:25)
        if symbol not in self._morning_high:
            morning_bars = intraday_df.head(3)
            self._morning_high[symbol] = float(morning_bars["high"].max())
            self._morning_low[symbol] = float(morning_bars["low"].min())
            logger.info(
                f"OptionsBuyer: {symbol} morning range = "
                f"{self._morning_low[symbol]:.0f} - {self._morning_high[symbol]:.0f}"
            )

        # Don't trade before 10:00 (morning noise + gap fills settle by 9:45-10:00)
        if now < dt_time(10, 0):
            return None

        # ── Step 3: Check 4 Intraday Confirmations ──
        latest_close = float(intraday_df["close"].iloc[-1])
        triggers: list[str] = []

        # Condition 1: Price vs Day Open (replaces VWAP — index has no volume)
        # Simple and reliable: close above day's open = bullish momentum
        day_open = float(intraday_df["open"].iloc[0])
        if day_open > 0:
            if direction == "CE" and latest_close > day_open:
                triggers.append("ABOVE_DAY_OPEN")
            elif direction == "PE" and latest_close < day_open:
                triggers.append("BELOW_DAY_OPEN")

        # Condition 2: RSI on 5-min chart
        rsi = FeatureEngine.rsi(intraday_df["close"], period=14)
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50

        if direction == "CE" and current_rsi > 55:
            triggers.append("RSI_BULLISH")
        elif direction == "PE" and current_rsi < 45:
            triggers.append("RSI_BEARISH")

        # Condition 3: Morning range breakout
        morning_high = self._morning_high.get(symbol, 0)
        morning_low = self._morning_low.get(symbol, 0)

        if direction == "CE" and morning_high > 0 and latest_close > morning_high:
            triggers.append("MORNING_BREAKOUT_UP")
        elif direction == "PE" and morning_low > 0 and latest_close < morning_low:
            triggers.append("MORNING_BREAKOUT_DOWN")

        # Condition 4: PCR (V4 thresholds: <0.7 bullish, >1.2 bearish)
        pcr = data.get("pcr", {}).get(symbol, 1.0)

        if direction == "CE" and pcr < 0.7:
            triggers.append("PCR_BULLISH")
        elif direction == "PE" and pcr > 1.2:
            triggers.append("PCR_BEARISH")

        # V4 confirmation threshold — 3-tier time windows:
        # 9:30–11:00 → need 2/4 confirmations (standard entry)
        # 11:00–1:00 → need 1/4 confirmations (easier entry)
        # 1:00–2:30  → need 2/4 confirmations (harder — conviction +0.5 already applied)
        bull_score, bear_score, score_diff = self._direction_scores.get(
            symbol, (0, 0, 0)
        )

        if now >= dt_time(13, 0):
            min_triggers = 2  # Afternoon: harder to enter
        elif now >= dt_time(11, 0):
            min_triggers = 1  # Mid-day: easier
        else:
            min_triggers = 2  # Morning: standard

        if len(triggers) < min_triggers:
            # ── Confirmation timeout: unlock stuck direction after 30 min ──
            # Only when NO trade placed yet for this symbol
            now_dt = datetime.now()
            if symbol not in self._traded_today and self._trades_today.get(symbol, 0) == 0:
                if symbol not in self._confirm_fail_since:
                    self._confirm_fail_since[symbol] = now_dt
                elif (now_dt - self._confirm_fail_since[symbol]).total_seconds() >= 1800:
                    # 30 minutes of failed confirmations — re-score direction
                    if self._direction_rescores_today < 3:
                        old_dir = direction
                        self._direction.pop(symbol, None)
                        self._direction_scores.pop(symbol, None)
                        self._confirm_fail_since.pop(symbol, None)
                        self._direction_rescores_today += 1
                        # Clear skip logs so new direction logs fresh
                        self._logged_today = {
                            k for k in self._logged_today
                            if not k.startswith(f"SKIP_{symbol}")
                        }
                        self._logged_today.discard(symbol)
                        logger.info(
                            f"DIRECTION TIMEOUT: {symbol} {old_dir} — "
                            f"confirmations failed for 30 min, re-scoring "
                            f"(rescore #{self._direction_rescores_today}/3)"
                        )
                        return None  # Next loop will re-compute direction

            # Log skipped signals every ~5 minutes (not every 30s)
            skip_key = f"SKIP_{symbol}_{now.minute // 5}"
            if skip_key not in self._logged_today:
                logger.info(
                    f"SKIPPED: {symbol} {direction} | score_diff={score_diff:.1f} "
                    f"triggers={triggers} (need {min_triggers}) | "
                    f"vs_open={'above' if latest_close > day_open else 'below'} "
                    f"RSI={current_rsi:.1f} | close={latest_close:.0f} "
                    f"morning=[{self._morning_low.get(symbol, 0):.0f}-{self._morning_high.get(symbol, 0):.0f}]"
                )
                self._logged_today.add(skip_key)
            return None

        # Confirmations passed — clear timeout tracker
        self._confirm_fail_since.pop(symbol, None)

        bias = "BULLISH" if direction == "CE" else "BEARISH"

        logger.info(
            f"OptionsBuyer: {symbol} CONFIRMED {bias} — "
            f"triggers: {', '.join(triggers)} ({len(triggers)}/{min_triggers} needed) | "
            f"close={latest_close:.0f} open={day_open:.0f} RSI={current_rsi:.1f} "
            f"PCR={pcr:.2f} | scores: bull={bull_score:.1f} bear={bear_score:.1f}"
        )

        # ── Step 4: Build Signal ──
        # Get spot price
        spot_key = f"{symbol.lower()}_price"
        spot = data.get(spot_key, 0)
        if spot <= 0:
            oi_levels = data.get("oi_levels", {}).get(symbol, {})
            spot = oi_levels.get("underlying", 0)
        if spot <= 0:
            nifty_df = data.get("nifty_df")
            if nifty_df is not None and not nifty_df.empty and symbol == "NIFTY":
                spot = float(nifty_df["close"].iloc[-1])
        if spot <= 0:
            spot = latest_close

        if spot <= 0:
            logger.warning(f"OptionsBuyer: No spot price for {symbol}")
            return None

        # VIX filter: hard skip > 35
        vix = data.get("vix", 15)
        if vix > 35:
            logger.info(f"OptionsBuyer: VIX {vix:.1f} > 35, skipping {symbol}")
            return None

        # ── Determine trade type from stored value ──
        trade_type = getattr(self, "_current_trade_type", "FULL")

        # ── SL/TP: VIX-adaptive (lowered TP to reduce EOD exits) ──
        if vix < 13:
            adaptive_sl = 0.25
            adaptive_tp = 0.40
        elif vix < 18:
            adaptive_sl = 0.30
            adaptive_tp = 0.45
        elif vix < 22:
            adaptive_sl = 0.30
            adaptive_tp = 0.55
        elif vix < 28:
            adaptive_sl = 0.25
            adaptive_tp = 0.60
        elif vix <= 35:
            adaptive_sl = 0.20
            adaptive_tp = 0.45
        else:
            adaptive_sl = 0.20
            adaptive_tp = 0.40

        # High conviction: tighter SL (better R:R)
        if score_diff >= 4.0:
            adaptive_sl *= 0.85
            adaptive_tp *= 1.2
        elif score_diff >= 3.0:
            adaptive_tp *= 1.1

        # Apply regime SL/TP multipliers
        regime_sl_mult = data.get("sl_multiplier", 1.0)
        regime_tp_mult = data.get("tp_multiplier", 1.0)
        adaptive_sl *= regime_sl_mult
        adaptive_tp *= regime_tp_mult

        # Friday boost: wider TP on strongest day
        if datetime.now().weekday() == 4:
            adaptive_tp *= 1.15

        # 3+ SLs any direction → tighten SL × 0.90
        if self._consec_sl_count >= 3 or self._streak <= -3:
            adaptive_sl *= 0.90
            logger.info(
                f"OptionsBuyer: SL tighten — consec_sl={self._consec_sl_count} "
                f"streak={self._streak}, SL reduced to {adaptive_sl:.0%}"
            )

        # Dynamic SL by premium level (applied after live premium lookup below)
        # Saved here, applied later once premium is known
        _base_adaptive_sl = adaptive_sl

        # ── Position sizing: V4 dynamic lot system ──
        sizing = self._compute_position_size(
            symbol=symbol,
            regime=regime,
            score_diff=score_diff,
            vix=vix,
            direction=direction,
            sl_pct=adaptive_sl,
            daily_loss_pct=data.get("daily_loss_pct", 0),
            circuit_breaker_warning=data.get("circuit_breaker_warning", False),
            wallet_balance=data.get("wallet_balance", 0),
            mode=data.get("trading_mode", "paper"),
            trade_type=trade_type,
        )
        if sizing is None:
            logger.info(f"OptionsBuyer: {symbol} — no affordable position size, skipping")
            return []
        lot_qty = sizing["lot_qty"]
        effective_max_premium = sizing["max_premium"]
        effective_min_premium = sizing["min_premium"]
        sizing_max_risk = sizing["max_risk"]
        sizing_full_lot = sizing["full_lot"]

        # ── Resolve option instrument — delta-optimized strike selection ──
        atm_strike = self._resolver.get_atm_strike(symbol, spot)

        # On expiry day, use next week's expiry to avoid theta crush
        is_expiry = data.get("is_expiry_day", False)
        if is_expiry:
            expiry = self._resolver.get_weekly_expiry(
                symbol, ref_date=date.today() + timedelta(days=1)
            )
            logger.info(f"OptionsBuyer: Expiry day — using next expiry {expiry}")
        else:
            expiry = self._resolver.get_weekly_expiry(symbol)

        # Get candidate strikes: ATM ± 7 strikes
        chain_keys = self._resolver.get_option_chain_keys(
            symbol, spot, num_strikes=7, expiry_date=expiry
        )
        # Filter to direction only (CE or PE)
        candidates = [c for c in chain_keys if c["type"] == direction]

        strike = None
        instrument_key = None
        trading_symbol = None
        live_premium = 0
        selected_delta = None
        selected_iv = None
        selected_oi = None
        selection_method = "premium_only"

        # ── Path 1: Delta-optimized selection (live/paper with greeks API) ──
        greeks_data = {}
        if (
            self._data_fetcher is not None
            and hasattr(self._data_fetcher, "get_option_greeks")
            and candidates
        ):
            inst_keys = [c["instrument_key"] for c in candidates]
            try:
                greeks_data = self._data_fetcher.get_option_greeks(inst_keys)
            except Exception as e:
                logger.warning(f"OptionsBuyer: Greeks API failed: {e}")

        if greeks_data:
            # Conviction-based delta target (V2.2)
            if score_diff >= 4.0:
                conviction_tier = "high"
            elif score_diff >= 2.0:
                conviction_tier = "medium"
            else:
                conviction_tier = "low"
            tier_targets = self.DELTA_TARGETS.get(conviction_tier, self.DELTA_TARGETS["medium"])
            delta_min, delta_max = tier_targets.get(
                regime, tier_targets["TRENDING"]
            )
            sweet_range = self.PREMIUM_SWEET_SPOTS.get(
                regime, self.PREMIUM_SWEET_SPOTS["TRENDING"]
            )

            scored = []
            for c in candidates:
                g = greeks_data.get(c["instrument_key"])
                if not g:
                    continue
                ltp = g.get("ltp", 0)
                oi = g.get("oi", 0)
                delta = abs(g.get("delta", 0))

                # Filters: premium range and OI
                if ltp <= 0 or ltp < effective_min_premium or ltp > effective_max_premium:
                    continue
                if oi < 5000:
                    continue

                # Delta score: 1.0 if in target range, penalized by distance
                delta_mid = (delta_min + delta_max) / 2
                if delta_min <= delta <= delta_max:
                    delta_score = 1.0
                else:
                    delta_score = max(0, 1.0 - abs(delta - delta_mid) * 3)

                # Premium sweet spot score
                sweet_min = sweet_range["sweet_min"]
                sweet_max = sweet_range["sweet_max"]
                if sweet_min <= ltp <= sweet_max:
                    premium_score = 1.0
                elif effective_min_premium <= ltp <= effective_max_premium:
                    premium_score = 0.5
                else:
                    premium_score = 0

                # OI liquidity score (capped at 50K)
                oi_score = min(oi / 50000, 1.0)

                total_score = delta_score * 0.5 + premium_score * 0.3 + oi_score * 0.2

                scored.append({
                    **c,
                    "ltp": ltp,
                    "delta": delta,
                    "oi": oi,
                    "iv": g.get("iv", 0),
                    "volume": g.get("volume", 0),
                    "total_score": total_score,
                })

            scored.sort(key=lambda x: x["total_score"], reverse=True)

            # Pick best and verify spread
            for best in scored:
                # Spread check via live quote (bid/ask)
                try:
                    quote = self._data_fetcher.get_live_quote(best["instrument_key"])
                    bid = quote.get("bid", 0) if quote else 0
                    ask = quote.get("ask", 0) if quote else 0
                    if bid and ask and ask > 0:
                        spread_pct = (ask - bid) / ask * 100
                        if spread_pct > 3.0:
                            logger.debug(
                                f"OptionsBuyer: Skipped {best['strike']}{direction} "
                                f"spread={spread_pct:.1f}% > 3%"
                            )
                            continue
                except Exception:
                    pass  # Accept if spread check fails

                strike = best["strike"]
                instrument_key = best["instrument_key"]
                trading_symbol = self._resolver.get_trading_symbol(
                    symbol, strike, expiry, direction
                )
                live_premium = best["ltp"]
                selected_delta = best["delta"]
                selected_iv = best.get("iv")
                selected_oi = best.get("oi")
                selection_method = "greeks"
                logger.info(
                    f"OptionsBuyer: Delta-selected {symbol} {strike}{direction} | "
                    f"premium=₹{live_premium:.1f} delta={selected_delta:.3f} "
                    f"IV={selected_iv:.1%} OI={selected_oi:,.0f} | "
                    f"range ₹{effective_min_premium}-{effective_max_premium:.0f}"
                )
                break

        # ── Path 2: Fallback — premium-only selection ──
        if not instrument_key and self._data_fetcher is not None:
            for c in candidates:
                try:
                    quote = self._data_fetcher.get_live_quote(c["instrument_key"])
                    premium = quote.get("ltp", 0) if quote else 0
                except Exception:
                    premium = 0

                if premium > 0 and effective_min_premium <= premium <= effective_max_premium:
                    strike = c["strike"]
                    instrument_key = c["instrument_key"]
                    trading_symbol = self._resolver.get_trading_symbol(
                        symbol, strike, expiry, direction
                    )
                    live_premium = premium
                    selection_method = "premium_only"
                    logger.info(
                        f"OptionsBuyer: Premium-selected {symbol} {strike}{direction} | "
                        f"premium=₹{premium:.1f} (range ₹{effective_min_premium}-{effective_max_premium:.0f})"
                    )
                    break

        # ── Path 3: No data fetcher (backtest) — accept first available key ──
        if not instrument_key and self._data_fetcher is None:
            for c in candidates:
                strike = c["strike"]
                instrument_key = c["instrument_key"]
                trading_symbol = self._resolver.get_trading_symbol(
                    symbol, strike, expiry, direction
                )
                selection_method = "backtest"
                break

        # ── Next-week expiry fallback ──
        if not instrument_key and not is_expiry:
            next_expiry = self._resolver.get_weekly_expiry(
                symbol, ref_date=expiry + timedelta(days=1)
            )
            if next_expiry != expiry:
                logger.info(
                    f"OptionsBuyer: No valid strike for current expiry {expiry}, "
                    f"trying next week {next_expiry}"
                )
                next_chain = self._resolver.get_option_chain_keys(
                    symbol, spot, num_strikes=5, expiry_date=next_expiry
                )
                next_candidates = [c for c in next_chain if c["type"] == direction]
                for c in next_candidates:
                    premium = 0
                    if self._data_fetcher is not None:
                        try:
                            quote = self._data_fetcher.get_live_quote(c["instrument_key"])
                            premium = quote.get("ltp", 0) if quote else 0
                        except Exception:
                            pass
                    if premium > 0 and effective_min_premium <= premium <= effective_max_premium:
                        strike = c["strike"]
                        instrument_key = c["instrument_key"]
                        trading_symbol = self._resolver.get_trading_symbol(
                            symbol, strike, next_expiry, direction
                        )
                        live_premium = premium
                        expiry = next_expiry
                        selection_method = "premium_only_next_week"
                        break

        if not instrument_key:
            logger.warning(
                f"OptionsBuyer: No valid strike for {symbol} {direction} | "
                f"ATM={atm_strike} exp={expiry} | "
                f"range ₹{effective_min_premium}-{effective_max_premium:.0f} "
                f"lot={lot_qty}"
            )
            return None

        # ── Dynamic SL by premium level ──
        known_prem = live_premium if live_premium > 0 else effective_max_premium
        if known_prem < 100:
            adaptive_sl = max(_base_adaptive_sl, 0.30)  # ≥30% for cheap options
        elif known_prem > 200:
            adaptive_sl = min(_base_adaptive_sl, 0.20)  # ≤20% for expensive ones
        # else: keep VIX-adaptive default (_base_adaptive_sl already in adaptive_sl)

        # ── Compute actual lot count now that premium is known ──
        actual_qty = self._compute_lots(
            premium=known_prem,
            full_lot=sizing_full_lot,
            sl_pct=adaptive_sl,
            max_risk=sizing_max_risk,
        )
        if actual_qty <= 0:
            logger.info(
                f"OptionsBuyer: {symbol} premium ₹{live_premium:.0f} — "
                f"cannot size position, skipping"
            )
            return None

        # Track trade count for daily limits
        self._full_trades_today += 1
        if getattr(self, "_current_trade_type", "") == "NAKED_BUY":
            self._naked_trades_today += 1

        # Confidence based on score_diff + triggers
        confidence = min(0.55 + score_diff * 0.06 + len(triggers) * 0.05, 0.95)

        option_symbol = f"{symbol}{int(strike)}{direction}"
        ml_prob_up = data.get("ml_direction_prob_up", 0.5)
        ml_prob_down = data.get("ml_direction_prob_down", 0.5)

        signal = Signal(
            strategy=self.name,
            symbol=option_symbol,
            direction=SignalDirection.BUY,
            confidence=confidence,
            score=score_diff,
            price=live_premium,
            stop_loss=0,
            take_profit=0,
            hold_days=0,
            regime=regime,
            features={
                "instrument_key": instrument_key,
                "trading_symbol": trading_symbol or option_symbol,
                "index_symbol": symbol,
                "strike": strike,
                "option_type": direction,
                "expiry": expiry.isoformat(),
                "lot_size": actual_qty,
                "spot_price": spot,
                "atm_strike": atm_strike,
                "pcr": round(pcr, 2),
                "vix": round(vix, 1),
                "premium_sl_pct": adaptive_sl,
                "premium_tp_pct": adaptive_tp,
                "trail_trigger_pct": 0.08 if trade_type == "FULL" else 0,
                "trail_exit_pct": 0.03 if trade_type == "FULL" else 0,
                "max_deployable": self.max_deployable,
                "max_risk_per_trade": sizing["max_risk"],
                "min_premium": float(effective_min_premium),
                "max_premium": float(effective_max_premium),
                "is_options": True,
                "trade_type": trade_type,
                # Position sizing details
                "sizing_reason": sizing["reason"],
                "lots": actual_qty // sizing_full_lot,
                # Delta/greeks data (None if premium-only selection)
                "delta": round(selected_delta, 4) if selected_delta else None,
                "iv": round(selected_iv, 4) if selected_iv else None,
                "strike_oi": int(selected_oi) if selected_oi else None,
                "selection_method": selection_method,
                # Multi-factor scoring details
                "direction_source": "MULTI_FACTOR",
                "bull_score": round(bull_score, 1),
                "bear_score": round(bear_score, 1),
                "score_diff": round(score_diff, 1),
                "confirmation_triggers": triggers,
                "ml_prob_up": round(ml_prob_up, 3),
                "ml_prob_down": round(ml_prob_down, 3),
                "morning_bias": bias,
                "morning_high": morning_high,
                "morning_low": morning_low,
                "intraday_rsi": round(current_rsi, 1),
                "intraday_day_open": round(day_open, 1),
                "current_pcr": round(pcr, 2),
            },
            notes=(
                f"{trade_type} {direction} {symbol} {int(strike)} | {bias} | "
                f"Score: bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} | "
                f"Triggers: {'+'.join(triggers)} | PCR={pcr:.2f} | Regime={regime} | "
                f"Qty={actual_qty} ({actual_qty // sizing_full_lot}L) | {selection_method}"
            ),
        )

        logger.info(
            f"OptionsBuyer SIGNAL: {trade_type} BUY {option_symbol} | "
            f"direction={direction} conf={confidence:.2f} score_diff={score_diff:.1f} | "
            f"triggers={len(triggers)} PCR={pcr:.2f} spot={spot:.0f} strike={strike} | "
            f"qty={actual_qty} ({actual_qty // sizing_full_lot}L) premium=₹{live_premium:.1f} {selection_method}"
        )

        return signal

    def should_force_exit(self) -> bool:
        """Check if we should force-exit all option positions."""
        now = datetime.now().time()
        exit_time = dt_time(*[int(x) for x in self.force_exit_time.split(":")])
        return now >= exit_time
