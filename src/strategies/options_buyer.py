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

import math
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from src.config.env_loader import get_config, parse_time_config
from src.data.features import FeatureEngine
from src.risk.manager import clamp_sl_tp_by_premium
from src.strategies.base import BaseStrategy, Signal, SignalDirection
from src.utils.market_calendar import get_expiry_type


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
        self.fixed_r_sizing = cfg.FIXED_R_SIZING
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
        self._reentry_eval_count: dict[str, int] = {}    # {symbol: evals since TP unlock}
        self._last_skip_info: dict[str, dict] = {}          # {symbol: {reason, score_diff, threshold, ...}}
        self._skip_counts: dict[str, int] = {}               # {reason_code: count} for EOD summary

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
        self._vix_history: list[float] = []  # Rolling buffer for IV awareness filter

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

        # Trade type per symbol (isolated to prevent cross-symbol leakage)
        self._current_trade_type: dict[str, str] = {}  # {symbol: "NAKED_BUY"/"CREDIT_SPREAD"/...}

        # Fix 2: Intraday rescore state
        self._daily_scores: dict[str, tuple] = {}       # frozen daily (bull, bear, diff)
        self._intraday_scores: dict[str, tuple] = {}    # from 5min candles
        self._blended_scores: dict[str, tuple] = {}     # weighted blend
        self._rescore_weight: float = 0.0                # current intraday weight
        self._rescore_times_done: set[str] = set()       # {"10:30", "11:00", ...}

        # 30-min rescore schedule: (time, daily_weight, intraday_weight)
        self._rescore_schedule = [
            (dt_time(10, 30), 0.70, 0.30),
            (dt_time(11,  0), 0.50, 0.50),
            (dt_time(11, 30), 0.35, 0.65),
            (dt_time(12,  0), 0.25, 0.75),
            (dt_time(12, 30), 0.15, 0.85),
        ]

        # Peak score tracking for rescore exit
        self._peak_score_diff: dict[str, float] = {}

        # Fix 5: Rolling range state
        self._rolling_range_high: dict[str, float] = {}
        self._rolling_range_low: dict[str, float] = {}
        self._range_last_update: dict[str, datetime] = {}
        self._range_too_tight: dict[str, bool] = {}

        # Momentum mode: direction lock while position is open
        self._position_direction_lock: dict[str, str] = {}  # {symbol: "CE"/"PE"}

        # Fix 1: Direction contradiction/flip state
        self._signal_killed: dict[str, bool] = {}
        self._direction_flipped_today: bool = False

        # Fix 3: Abort mechanism state
        self._abort_stage: dict[str, str] = {}           # NONE / SOFT / HARD
        self._failed_confirm_count: dict[str, int] = {}
        self._abort_bypassed: dict[str, bool] = {}       # True after completed trade

        # Breakout re-entry: force range refresh after trade close
        self._breakout_pending_reentry: dict[str, bool] = {}  # watching for new breakout

        # Bidirectional reversal state (reset daily)
        self._reversal_eligible: bool = False
        self._reversal_pending: bool = False       # OPT 2: waiting for rescore confirmation
        self._reversal_pending_direction: str = "" # OPT 2: direction to confirm
        self._last_exit_direction: str = ""
        self._last_exit_pnl: float = 0.0
        self._is_reversal_trade: bool = False  # True if current signal is a reversal

        # Volatile dual mode state (reset daily)
        self._dual_mode_active: bool = False
        self._dual_mode_trades_today: int = 0
        self._is_dual_mode_trade: bool = False

        # Momentum decay: last RSI per symbol (updated during intraday scoring)
        self._last_rsi: dict[str, float] = {}

        # Safety guards: opposing spread, reversal force naked, one position type
        self._today_trade_direction: Optional[str] = None  # "CE" or "PE" — first trade's direction
        self._today_position_type: Optional[str] = None    # "NAKED_BUY" or "CREDIT_SPREAD"

        # Expiry type state (set once per day)
        self._expiry_type: str = "NORMAL"
        self._expiry_adjustments_applied: bool = False

        # Telegram alert callback (injected by main.py)
        self._alert_fn: Optional[Callable[[str], None]] = None

        # Counterfactual trade log: records blocked trades for EOD P&L analysis
        self._counterfactual_log: list[dict] = []

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

    def set_alert_fn(self, fn: Callable[[str], None]) -> None:
        """Inject a Telegram alert callback (main.py passes self.alerts.send_raw)."""
        self._alert_fn = fn

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
        self._reentry_eval_count.clear()
        self._last_skip_info.clear()
        self._skip_counts.clear()
        self._counterfactual_log.clear()
        # Fix 2: Intraday rescore
        self._daily_scores.clear()
        self._intraday_scores.clear()
        self._blended_scores.clear()
        self._rescore_weight = 0.0
        self._rescore_times_done.clear()
        self._peak_score_diff.clear()
        # Fix 5: Rolling range
        self._rolling_range_high.clear()
        self._rolling_range_low.clear()
        self._range_last_update.clear()
        self._range_too_tight.clear()
        # Fix 1: Direction contradiction/flip
        self._signal_killed.clear()
        self._direction_flipped_today = False
        # Fix 3: Abort mechanism
        self._abort_stage.clear()
        self._failed_confirm_count.clear()
        self._abort_bypassed.clear()
        # Breakout re-entry
        self._breakout_pending_reentry.clear()
        # Momentum mode: position direction lock
        self._position_direction_lock.clear()
        # Bidirectional reversal
        self._reversal_eligible = False
        self._reversal_pending = False
        self._reversal_pending_direction = ""
        self._last_exit_direction = ""
        self._last_exit_pnl = 0.0
        self._is_reversal_trade = False
        # Volatile dual mode
        self._dual_mode_active = False
        self._dual_mode_trades_today = 0
        self._is_dual_mode_trade = False
        # Trade type per symbol
        self._current_trade_type.clear()
        # Safety guards
        self._today_trade_direction = None
        self._today_position_type = None
        # Expiry state
        self._expiry_type = "NORMAL"
        self._expiry_adjustments_applied = False

    def record_skip(self, symbol: str) -> None:
        """Increment skip count for the current skip reason on a symbol."""
        info = self._last_skip_info.get(symbol)
        if info:
            reason = info.get("reason", "UNKNOWN")
            self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1

    @staticmethod
    def _pe_filter_passes(pe_prob: float, score_diff: float,
                          vix_now: float, vix_open: float) -> tuple[bool, str]:
        """Context-aware PE confidence filter with tolerance zone.

        Returns (passes, reason) where reason is "" if passes, else block reason.

        Tiers:
          >= 0.70  → always allow
          0.60-0.70 → allow if strong setup (VIX rising or high score_diff)
          < 0.60   → always block
        """
        cfg = get_config()
        threshold = cfg.PE_FILTER_THRESHOLD       # 0.70
        tol_low = cfg.PE_FILTER_TOLERANCE_LOW      # 0.60
        tol_score = cfg.PE_FILTER_TOLERANCE_SCORE  # 3.0
        tol_vix = cfg.PE_FILTER_TOLERANCE_VIX_RISE # 0.5

        if pe_prob >= threshold:
            return True, ""

        if pe_prob >= tol_low:
            # Tolerance zone: allow with strong supporting context
            vix_rising = vix_open > 0 and (vix_now - vix_open) >= tol_vix
            strong_score = abs(score_diff) >= tol_score
            if vix_rising or strong_score:
                return True, ""
            return False, "PE_TOLERANCE_BLOCK"

        return False, "PE_LOW_CONFIDENCE"

    @staticmethod
    def _ce_filter_passes(ce_prob: float, score_diff: float,
                          vix_now: float, vix_open: float) -> tuple[bool, str]:
        """Context-aware CE confidence filter with tolerance zone.

        Returns (passes, reason) where reason is "" if passes, else block reason.

        Tiers:
          >= 0.65  → always allow
          0.50-0.65 → allow if strong setup (VIX falling or high score_diff)
          < 0.50   → always block
        """
        cfg = get_config()
        threshold = cfg.CE_FILTER_THRESHOLD        # 0.65
        tol_low = cfg.CE_FILTER_TOLERANCE_LOW       # 0.50
        tol_score = cfg.CE_FILTER_TOLERANCE_SCORE   # 3.25
        tol_vix = cfg.CE_FILTER_TOLERANCE_VIX_FALL  # 0.5

        if ce_prob >= threshold:
            return True, ""

        if ce_prob >= tol_low:
            # Tolerance zone: allow with strong supporting context
            vix_falling = vix_open > 0 and (vix_open - vix_now) >= tol_vix
            strong_score = abs(score_diff) >= tol_score
            if vix_falling or strong_score:
                return True, ""
            return False, "CE_TOLERANCE_BLOCK"

        return False, "CE_LOW_CONFIDENCE"

    def get_skip_summary(self) -> dict[str, int]:
        """Return skip counts by reason for EOD summary."""
        return dict(self._skip_counts)

    def get_counterfactual_log(self) -> list[dict]:
        """Return accumulated counterfactual trades for EOD processing."""
        return list(self._counterfactual_log)

    def _record_counterfactual(
        self, symbol: str, direction: str, block_reason: str,
        spot: float, regime: str, score_diff: float,
        bull_score: float, bear_score: float, metadata: dict | None = None,
    ) -> None:
        """Record a blocked trade for counterfactual P&L analysis.

        Only records once per symbol+reason per day (deduplicates noisy blocks
        like CONFIRMATION_FAILED which fire every 30s).
        """
        # Deduplicate: only keep first block per symbol+reason
        existing = {(r["symbol"], r["block_reason"]) for r in self._counterfactual_log}
        if (symbol, block_reason) in existing:
            return

        self._counterfactual_log.append({
            "symbol": symbol,
            "direction": direction,
            "block_reason": block_reason,
            "block_time": datetime.now().strftime("%H:%M:%S"),
            "spot_at_block": round(spot, 2),
            "regime": regime,
            "score_diff": round(score_diff, 2),
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
            "metadata": metadata or {},
        })

    def record_exit(self, symbol: str, exit_reason: str, direction: str,
                     pnl: float = 0.0, entry_cost: float = 0.0) -> None:
        """Record a trade exit for consecutive SL and streak tracking.

        Also unlocks re-entry after TP/trail wins (max 2 trades/day/symbol).
        V4: tracks exit time for direction flip 90-min gap rule.
        V10: sets reversal eligibility on profitable exit.
        """
        # Track exit reason for re-entry logic
        # Extract index symbol from option symbol (e.g. "NIFTY25500CE" → "NIFTY")
        index_sym = symbol.rstrip("CEPE").rstrip("0123456789") if symbol else ""
        if not index_sym:
            index_sym = symbol
        self._last_exit_reason[index_sym] = exit_reason
        self._last_exit_time[index_sym] = datetime.now()
        self._position_flat[index_sym] = True
        # Momentum mode: unlock direction lock
        self._position_direction_lock.pop(index_sym, None)
        # Clear peak score diff for rescore exit tracking
        self.clear_peak_score(symbol)
        self.clear_peak_score(index_sym)
        # Clear trade type flags
        self._is_reversal_trade = False
        self._is_dual_mode_trade = False

        # Bidirectional reversal: set eligibility based on exit P&L
        cfg = get_config()
        if cfg.REVERSAL_ENABLED:
            self._last_exit_direction = direction
            self._last_exit_pnl = pnl
            # OPT 3: minimum exit profit gate
            profit_pct = (pnl / entry_cost) if entry_cost > 0 else 0.0
            min_profit = cfg.REVERSAL_MIN_EXIT_PROFIT
            if pnl > 0 and profit_pct >= min_profit:
                # OPT 2: set pending (not eligible) — needs rescore confirmation
                self._reversal_pending = True
                self._reversal_pending_direction = "PE" if direction == "CE" else "CE"
                self._reversal_eligible = False
                logger.info(
                    f"REVERSAL_PENDING: exited {direction} +₹{pnl:.0f} "
                    f"({profit_pct:.1%} >= {min_profit:.0%}) — awaiting rescore confirmation"
                )
            elif pnl > 0:
                self._reversal_pending = False
                self._reversal_eligible = False
                logger.info(
                    f"REVERSAL_BLOCKED_LOW_PROFIT: exited {direction} +₹{pnl:.0f} "
                    f"({profit_pct:.1%} < {min_profit:.0%}) — profit too low"
                )
            else:
                self._reversal_pending = False
                self._reversal_eligible = False
                logger.info(
                    f"REVERSAL_BLOCKED: exited {direction} at loss "
                    f"(₹{pnl:.0f}) — opposite entry not allowed"
                )

        # Determine max trades per day based on mode
        max_trades_per_symbol = self.ACTIVE_MAX_TRADES_PER_DAY if self._active_trading else 2

        # Allow re-entry after TP or trail_stop
        if exit_reason in ("take_profit", "trail_stop"):
            # Fix 3: Bypass abort after completed trade
            self._abort_bypassed[index_sym] = True
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                # Unlock symbol for re-entry — clear direction to re-score
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)
                self._logged_today.discard(index_sym)
                self._reentry_eval_count[index_sym] = 0
                # Force rolling range refresh for new breakout detection
                self._range_last_update.pop(index_sym, None)
                self._breakout_pending_reentry[index_sym] = True
                rh = self._rolling_range_high.get(index_sym, 0)
                rl = self._rolling_range_low.get(index_sym, 0)
                logger.info(
                    f"OptionsBuyer: {exit_reason} on {symbol} — "
                    f"unlocking {index_sym} for trade #{trades + 1}"
                )
                logger.info(
                    f"BREAKOUT_RESET: trade closed. "
                    f"Watching for new breakout above/below "
                    f"{rh:.0f}/{rl:.0f}"
                )

        # Reversal: unlock for opposite-direction re-entry after profitable exit
        # (extends TP/trail unlock to also cover rescore_flip, momentum_decay, etc.)
        if (cfg.REVERSAL_ENABLED and self._reversal_eligible
                and exit_reason not in ("take_profit", "trail_stop")):
            # TP/trail_stop already unlocked above — avoid double unlock
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                self._abort_bypassed[index_sym] = True
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)
                self._logged_today.discard(index_sym)
                self._reentry_eval_count[index_sym] = 0
                self._range_last_update.pop(index_sym, None)
                self._breakout_pending_reentry[index_sym] = True
                logger.info(
                    f"REVERSAL_UNLOCK: {exit_reason} on {symbol} profitable — "
                    f"unlocking {index_sym} for opposite direction"
                )

        # Active mode: also allow re-entry after EOD exit (position was flat, market had more room)
        if self._active_trading and exit_reason == "eod_exit":
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)
                self._logged_today.discard(index_sym)
                self._range_last_update.pop(index_sym, None)
                self._breakout_pending_reentry[index_sym] = True

        # Active mode: also allow re-entry after SL (direction unlocked)
        if self._active_trading and exit_reason == "stop_loss":
            trades = self._trades_today.get(index_sym, 1)
            if trades < max_trades_per_symbol:
                self._traded_today.discard(index_sym)
                self._direction.pop(index_sym, None)  # Direction unlocked — free CE/PE
                self._logged_today.discard(index_sym)
                self._range_last_update.pop(index_sym, None)
                self._breakout_pending_reentry[index_sym] = True
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
                    self._range_last_update.pop(index_sym, None)
                    self._breakout_pending_reentry[index_sym] = True
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

        # Momentum mode: lock direction while position is open
        cfg = get_config()
        if cfg.MOMENTUM_MODE_ENABLED:
            direction = self._direction.get(index_symbol, "")
            if direction:
                self._position_direction_lock[index_symbol] = direction

        # Safety guards: track direction and position type
        if self._today_trade_direction is None:
            self._today_trade_direction = self._direction.get(index_symbol, "")
        trade_type = self._current_trade_type.get(index_symbol, "NAKED_BUY")
        if self._today_position_type is None:
            self._today_position_type = trade_type

        # Clear reversal state after execution (one-shot: reversal consumed)
        if self._is_reversal_trade:
            self._reversal_eligible = False
            self._reversal_pending = False
            self._reversal_pending_direction = ""
            self._is_reversal_trade = False

        # Track dual mode trade
        if self._is_dual_mode_trade:
            self._dual_mode_trades_today += 1
            self._is_dual_mode_trade = False
            logger.info(f"DUAL_MODE_CONFIRMED: {index_symbol} dual mode trade #{self._dual_mode_trades_today}")

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
        _et_h, _et_m = parse_time_config(self.force_exit_time, 15, 10)
        exit_time = dt_time(_et_h, _et_m)

        # Set expiry type once per day
        expiry_type = data.get("expiry_type", "NORMAL")
        if not self._expiry_adjustments_applied:
            self._expiry_type = expiry_type
            self._expiry_adjustments_applied = True
            if expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
                logger.info(
                    f"EXPIRY_DAY: {expiry_type}. "
                    f"threshold=+1.0 window=10:00-11:30 size=0.75x"
                )
                if self._alert_fn:
                    idx = "NIFTY" if expiry_type == "NIFTY_EXPIRY" else "BANKNIFTY"
                    self._alert_fn(
                        f"\U0001f4c5 Expiry day ({idx}). Tighter rules active."
                    )
            elif expiry_type == "SENSEX_EXPIRY":
                logger.info(f"EXPIRY_DAY: SENSEX_EXPIRY. Minor adjustments.")

        # Major expiry: no entries after 11:30
        is_major_expiry = self._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")
        if is_major_expiry and now >= dt_time(11, 30):
            self._last_skip_info = {s: {"reason": "EXPIRY_CUTOFF"} for s in self.instruments}
            return []

        # No entry after 15:10
        if now >= exit_time:
            self._last_skip_info = {s: {"reason": "AFTER_CUTOFF"} for s in self.instruments}
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
            _nt_h, _nt_m = parse_time_config(get_config().NO_NEW_TRADE_AFTER, 14, 30)
            regime_cutoffs = {
                "TRENDING": dt_time(_nt_h, _nt_m),
                "RANGEBOUND": dt_time(13, 0),
                "VOLATILE": dt_time(12, 0),
            }
        last_trade_cutoff = regime_cutoffs.get(regime, dt_time(14, 30))
        if now >= last_trade_cutoff:
            self._last_skip_info = {s: {"reason": "AFTER_CUTOFF"} for s in self.instruments}
            return []

        # Active mode: halt if 2 same-day SLs
        if self._active_trading and self._same_day_sl_count >= self.ACTIVE_CONSEC_SL_HALT:
            self._last_skip_info = {s: {"reason": "CIRCUIT_BREAKER"} for s in self.instruments}
            return []

        # Active mode: cooldown — wait 15 min after last exit
        if self._active_trading:
            now_dt = datetime.now()
            for sym, exit_time in self._last_exit_time.items():
                mins_since = (now_dt - exit_time).total_seconds() / 60
                if mins_since < self.ACTIVE_COOLDOWN_MINUTES:
                    self._last_skip_info = {s: {"reason": "COOLDOWN"} for s in self.instruments}
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

        # Volatile dual mode activation
        if cfg.DUAL_MODE_ENABLED and regime == "VOLATILE" and not self._dual_mode_active:
            vix = data.get("vix", 0)
            self._dual_mode_active = True
            logger.info(
                f"DUAL_MODE: VOLATILE regime detected VIX={vix:.1f} "
                f"intraday threshold={cfg.DUAL_MODE_MIN_SCORE} "
                f"watching for both CE and PE swings"
            )

        cfg = get_config()

        # Reversal cutoff: no reversal entries after 13:00
        now_time = datetime.now().time()

        # OPT 2: Promote pending → eligible on rescore confirmation, timeout at 12:30
        if cfg.REVERSAL_ENABLED and self._reversal_pending:
            if now_time >= dt_time(12, 30):
                # Timeout: pending too long, cancel
                logger.info(
                    f"REVERSAL_TIMEOUT: pending {self._reversal_pending_direction} "
                    f"expired at 12:30 — cancelling"
                )
                self._reversal_pending = False
                self._reversal_pending_direction = ""
            else:
                # Check if rescore confirms the pending direction
                for sym in self.instruments:
                    bull, bear, diff = self._direction_scores.get(sym, (0, 0, 0))
                    if bull > 0 or bear > 0:
                        rescore_dir = "CE" if bull > bear else ("PE" if bear > bull else "")
                        if (rescore_dir == self._reversal_pending_direction
                                and diff >= cfg.REVERSAL_MIN_SCORE):
                            self._reversal_eligible = True
                            self._reversal_pending = False
                            logger.info(
                                f"REVERSAL_CONFIRMED: rescore confirms "
                                f"{self._reversal_pending_direction} "
                                f"diff={diff:.1f} >= {cfg.REVERSAL_MIN_SCORE:.1f} "
                                f"— reversal now eligible"
                            )
                            self._reversal_pending_direction = ""
                            break

        reversal_allowed = (
            cfg.REVERSAL_ENABLED
            and self._reversal_eligible
            and now_time < dt_time(13, 0)
        )

        for symbol in self.instruments:
            if symbol in self._traded_today:
                self._last_skip_info[symbol] = {"reason": "DIRECTION_LOCKED"}
                continue

            # Momentum mode: clear direction every loop when flat (no open position)
            if cfg.MOMENTUM_MODE_ENABLED and symbol not in self._position_direction_lock:
                self._direction.pop(symbol, None)

            # Max trades per day per symbol
            if self._trades_today.get(symbol, 0) >= max_trades:
                # Reversal blocked by max trades
                if reversal_allowed:
                    logger.info(
                        f"REVERSAL_MAX_TRADES: {self._trades_today.get(symbol, 0)}/{max_trades} "
                        f"daily limit reached — no reversal"
                    )
                self._last_skip_info[symbol] = {"reason": "MAX_TRADES_REACHED"}
                continue

            # Reversal strength check: set reversal trade flag when eligible
            if reversal_allowed and self._last_exit_direction:
                bull, bear, diff = self._direction_scores.get(symbol, (0, 0, 0))
                if bull > 0 or bear > 0:
                    rescore_dir = "CE" if bull > bear else ("PE" if bear > bull else "")
                    if rescore_dir and rescore_dir != self._last_exit_direction:
                        if diff >= cfg.REVERSAL_MIN_SCORE:
                            logger.info(
                                f"REVERSAL_TRADE: {self._last_exit_direction}→{rescore_dir} "
                                f"diff={diff:.1f} >= min {cfg.REVERSAL_MIN_SCORE:.1f}"
                            )
                            self._is_reversal_trade = True
                        else:
                            logger.info(
                                f"REVERSAL_WEAK: {self._last_exit_direction}→{rescore_dir} "
                                f"diff={diff:.1f} < min {cfg.REVERSAL_MIN_SCORE:.1f} — waiting"
                            )

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

        # Bucket accumulators for signal grouping (cap correlated factors)
        momentum_bull = 0.0  # F1 EMA + F2 RSI/MACD + F3 Price Action + F9 Volume
        momentum_bear = 0.0
        flow_bull = 0.0      # F8 OI/PCR + F10 Global Macro
        flow_bear = 0.0
        vol_bull = 0.0       # F5 Bollinger + F6 VIX
        vol_bear = 0.0
        mr_bull = 0.0        # F4 Mean Reversion
        mr_bear = 0.0

        # Bucket caps (99.0 = no practical cap; tuning showed all caps reduce CAGR)
        MOMENTUM_CAP = 99.0
        FLOW_CAP = 99.0
        VOLATILITY_CAP = 99.0
        MEAN_REVERSION_CAP = 99.0

        def _sg(series, key, default):
            """Safe get: handles missing keys AND NaN values."""
            val = series.get(key, default)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return float(val)

        # Get values (with safe defaults + NaN guard)
        close = _sg(row, "close", 0)
        ema_9 = _sg(row, "ema_9", close)
        ema_21 = _sg(row, "ema_21", close)
        ema_50 = _sg(row, "ema_50", close)
        rsi = _sg(row, "rsi_14", 50)
        prev_rsi = _sg(prev_row, "rsi_14", 50)
        macd_hist = _sg(row, "macd_histogram", 0)
        prev_macd_hist = _sg(prev_row, "macd_histogram", 0)
        bb_upper = _sg(row, "bb_upper", close * 1.02)
        bb_lower = _sg(row, "bb_lower", close * 0.98)
        open_price = _sg(row, "open", close)
        prev_close = _sg(prev_row, "close", open_price)
        prev_high = _sg(prev_row, "high", close)
        prev_low = _sg(prev_row, "low", close)
        ret_5d = _sg(row, "returns_5d", 0) * 100 if "returns_5d" in row.index else 0
        vix = data.get("vix", 15)

        trend_up = ema_9 > ema_21 > ema_50
        trend_down = ema_9 < ema_21 < ema_50

        adx = _sg(row, "adx_14", 20)

        # === BUCKET 1: MOMENTUM — F1 EMA + F2 RSI/MACD + F3 Price Action + F9 Volume ===

        # --- F1: Trend alignment (regime-controlled weight) — reduced ×0.6 (edge analysis) ---
        ema_weight = data.get("ema_weight", 2.5)
        ema_base = ema_weight * 0.8 * 0.6
        ema_bonus = ema_weight * 0.2 * 0.6

        if trend_up:
            momentum_bull += ema_base
        elif trend_down:
            momentum_bear += ema_base

        if close > ema_21 * 1.005:
            momentum_bull += ema_bonus
        elif close < ema_21 * 0.995:
            momentum_bear += ema_bonus

        if adx > 30 and (trend_up or trend_down):
            if trend_up:
                momentum_bull += 0.3
            else:
                momentum_bear += 0.3

        if ret_5d > 0:
            momentum_bull += 0.2
        elif ret_5d < 0:
            momentum_bear += 0.2

        # --- F2: Momentum — RSI + MACD (weight: 1.5, was 2.0 — redundant with F3/F5) ---
        if rsi > 58 and rsi > prev_rsi:
            momentum_bull += 0.75
        elif rsi < 42 and rsi < prev_rsi:
            momentum_bear += 0.75

        if macd_hist > 0 and macd_hist > prev_macd_hist:
            momentum_bull += 0.75
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            momentum_bear += 0.75

        # --- F3: Price action — gap + breakout (weight: 2.0, was 1.5 — strong aligned edge) ---
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        if gap_pct > 0.4:
            momentum_bull += 1.0
        elif gap_pct < -0.4:
            momentum_bear += 1.0

        if close > prev_high:
            momentum_bull += 0.7
        elif close < prev_low:
            momentum_bear += 0.7

        if close > open_price:
            momentum_bull += 0.3
        elif close < open_price:
            momentum_bear += 0.3

        # --- F9: Volume Confirmation (weight: 2.5, was 1.0 — strongest factor) ---
        volume = float(row.get("volume", 0))
        vol_ma = float(row.get("volume_ma_20", 0)) if "volume_ma_20" in row.index else 0
        if vol_ma <= 0:
            vol_series = data.get("nifty_df", pd.DataFrame()).get("volume")
            if vol_series is not None and len(vol_series) >= 20:
                vol_ma = float(vol_series.iloc[-20:].mean())

        if vol_ma > 0 and volume > 0:
            vol_ratio = volume / vol_ma
            if vol_ratio > 1.3:
                if close > open_price:
                    momentum_bull += 2.5
                elif close < open_price:
                    momentum_bear += 2.5
            elif vol_ratio < 0.7:
                if close > open_price:
                    momentum_bull -= 0.5
                elif close < open_price:
                    momentum_bear -= 0.5

        # === BUCKET 4: MEAN REVERSION — F4 INVERTED (confirms momentum) ===
        # Edge analysis: F4 always fires against direction but 80% WR
        # → inversion makes it a momentum confirmation signal
        # Extended up (ret_5d > 3.5) → confirms BULL momentum
        # Extended down (ret_5d < -3.5) → confirms BEAR momentum
        if ret_5d > 5.0:
            mr_bull += 1.5
        elif ret_5d > 3.5:
            mr_bull += 1.0
        elif ret_5d < -5.0:
            mr_bear += 1.5
        elif ret_5d < -3.5:
            mr_bear += 1.0

        # === BUCKET 3: VOLATILITY — F5 Bollinger + F6 VIX ===

        # --- F5: Bollinger position (weight: 1.5, was 0.75 — clean signal) ---
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        if bb_pos > 0.85:
            vol_bull += 1.0
        elif bb_pos < 0.15:
            vol_bear += 1.0

        prev_bb_upper = _sg(prev_row, "bb_upper", prev_close * 1.02)
        prev_bb_lower = _sg(prev_row, "bb_lower", prev_close * 0.98)
        bb_width = (bb_upper - bb_lower) / close if close > 0 else 0
        prev_bb_width = (prev_bb_upper - prev_bb_lower) / prev_close if prev_close > 0 else 0
        if prev_bb_width > 0 and bb_width > prev_bb_width * 1.20:
            # BB expanding — use pre-cap bucket sums for direction hint
            if momentum_bull >= momentum_bear:
                vol_bull += 0.5
            else:
                vol_bear += 0.5

        # --- F6: VIX (weight: 1.0, was 0.8 — slight increase) ---
        if vix < 13:
            vol_bull += 0.6
        elif vix > 20:
            vol_bear += 0.6

        if self._prev_vix > 0:
            vix_delta = vix - self._prev_vix
            if vix > 20 and vix_delta < -1.0:
                vol_bull += 0.4
            elif vix_delta > 1.0:
                vol_bear += 0.4
        self._prev_vix = vix

        # Track VIX history for IV awareness filter (daily granularity)
        if vix > 0:
            self._vix_history.append(vix)
            if len(self._vix_history) > 30:
                self._vix_history = self._vix_history[-30:]

        # === FACTOR 7: ML Direction Model (NOT BUCKETED — independent signal) ===
        ml_v2_ready = data.get("ml_v2_ready", False)
        ml_ce_deployed = data.get("ml_ce_ready", False)
        ml_pe_deployed = data.get("ml_pe_ready", False)
        ML_PE_WEIGHT = 1.5
        ML_CE_WEIGHT = 0.3
        ML_CONFIDENCE_THRESHOLD = 0.45
        ml_bull = 0.0
        ml_bear = 0.0

        if ml_v2_ready:
            # V2: both binary models deployed — use their independent probabilities
            v2_pe_prob = data.get("ml_v2_pe_prob", 0.5)
            v2_ce_prob = data.get("ml_v2_ce_prob", 0.5)
            if v2_pe_prob > 0.55:
                ml_bear += ML_PE_WEIGHT * (v2_pe_prob - 0.5) / 0.5
            if v2_ce_prob > 0.55:
                ml_bull += ML_CE_WEIGHT * (v2_ce_prob - 0.5) / 0.5
        else:
            # Hybrid: use deployed binary models where available, V1 fallback otherwise
            ml_stage1_ce = data.get("ml_stage1_prob_ce", 0.33)
            ml_stage1_pe = data.get("ml_stage1_prob_pe", 0.33)

            # CE signal: prefer CE binary model (58.7%) over V1 CE (37.1%)
            if ml_ce_deployed:
                ce_binary_prob = data.get("ml_ce_binary_prob", 0.5)
                if ce_binary_prob > 0.55:
                    ml_bull += ML_CE_WEIGHT * (ce_binary_prob - 0.5) / 0.5
            elif ml_stage1_ce > ML_CONFIDENCE_THRESHOLD:
                ml_bull += ML_CE_WEIGHT * (ml_stage1_ce - 0.33) / 0.67

            # PE signal: prefer PE binary model if deployed, else V1
            if ml_pe_deployed:
                pe_binary_prob = data.get("ml_pe_binary_prob", 0.5)
                if pe_binary_prob > 0.55:
                    ml_bear += ML_PE_WEIGHT * (pe_binary_prob - 0.5) / 0.5
            elif ml_stage1_pe > ML_CONFIDENCE_THRESHOLD:
                ml_bear += ML_PE_WEIGHT * (ml_stage1_pe - 0.33) / 0.67

        # === BUCKET 2: FLOW — F8 OI/PCR + F10 Global Macro ===

        # --- F10: Global Macro (weight: 0.5, was 1.5 — negative edge) ---
        dxy_mom = data.get("dxy_momentum_5d", 0)
        sp_nifty_corr = data.get("sp500_nifty_corr_20d", 0.5)
        global_risk = data.get("global_risk_score", 0)
        sp500_ret = data.get("sp500_prev_return", 0)

        if dxy_mom > 0.5:
            flow_bear += 0.17
        elif dxy_mom < -0.5:
            flow_bull += 0.17
        if sp_nifty_corr > 0.5:
            if sp500_ret > 0.5:
                flow_bull += 0.17
            elif sp500_ret < -0.5:
                flow_bear += 0.17
        if global_risk < -1.0:
            flow_bear += 0.16
        elif global_risk > 1.0:
            flow_bull += 0.16

        # --- F8: OI/PCR consensus (weight: 2.0) ---
        pcr_data = data.get("pcr", {})
        oi_data = data.get("oi_levels", {})
        nifty_oi = oi_data.get("NIFTY", {})
        pcr_val = pcr_data.get(symbol, pcr_data.get("NIFTY", 1.0))
        if isinstance(pcr_val, dict):
            pcr_val = pcr_val.get("pcr_oi", 1.0)

        if pcr_val >= 1.3:
            flow_bull += 1.0
        elif pcr_val >= 1.1:
            flow_bull += 0.5
        elif pcr_val <= 0.7:
            flow_bear += 1.0
        elif pcr_val <= 0.9:
            flow_bear += 0.5

        oi_resistance = nifty_oi.get("max_call_oi_strike", 0)
        oi_support = nifty_oi.get("max_put_oi_strike", 0)
        spot = data.get("nifty_price", close)

        if oi_resistance > 0 and oi_support > 0 and spot > 0:
            dist_to_resistance = (oi_resistance - spot) / spot * 100
            dist_to_support = (spot - oi_support) / spot * 100

            if dist_to_support < 0.5:
                flow_bull += 1.0
            elif dist_to_resistance < 0.5:
                flow_bear += 1.0
            elif dist_to_support < 1.0:
                flow_bull += 0.5
            elif dist_to_resistance < 1.0:
                flow_bear += 0.5

        # === Apply bucket caps and sum into final scores ===
        raw_momentum_bull, raw_momentum_bear = momentum_bull, momentum_bear
        raw_flow_bull, raw_flow_bear = flow_bull, flow_bear
        raw_vol_bull, raw_vol_bear = vol_bull, vol_bear
        raw_mr_bull, raw_mr_bear = mr_bull, mr_bear

        momentum_bull = max(min(momentum_bull, MOMENTUM_CAP), -MOMENTUM_CAP)
        momentum_bear = max(min(momentum_bear, MOMENTUM_CAP), -MOMENTUM_CAP)
        flow_bull = max(min(flow_bull, FLOW_CAP), -FLOW_CAP)
        flow_bear = max(min(flow_bear, FLOW_CAP), -FLOW_CAP)
        vol_bull = max(min(vol_bull, VOLATILITY_CAP), -VOLATILITY_CAP)
        vol_bear = max(min(vol_bear, VOLATILITY_CAP), -VOLATILITY_CAP)
        mr_bull = max(min(mr_bull, MEAN_REVERSION_CAP), -MEAN_REVERSION_CAP)
        mr_bear = max(min(mr_bear, MEAN_REVERSION_CAP), -MEAN_REVERSION_CAP)

        bull_score = momentum_bull + flow_bull + vol_bull + mr_bull + ml_bull
        bear_score = momentum_bear + flow_bear + vol_bear + mr_bear + ml_bear

        # Bucket breakdown logging
        caps_hit = []
        if raw_momentum_bull != momentum_bull or raw_momentum_bear != momentum_bear:
            caps_hit.append("MOMENTUM")
        if raw_flow_bull != flow_bull or raw_flow_bear != flow_bear:
            caps_hit.append("FLOW")
        if raw_vol_bull != vol_bull or raw_vol_bear != vol_bear:
            caps_hit.append("VOLATILITY")
        if raw_mr_bull != mr_bull or raw_mr_bear != mr_bear:
            caps_hit.append("MEAN_REV")
        cap_str = f" [CAPPED: {','.join(caps_hit)}]" if caps_hit else ""
        logger.info(
            f"Buckets: MOM={momentum_bull:.1f}/{momentum_bear:.1f} "
            f"FLOW={flow_bull:.1f}/{flow_bear:.1f} "
            f"VOL={vol_bull:.1f}/{vol_bear:.1f} "
            f"MR={mr_bull:.1f}/{mr_bear:.1f} "
            f"ML={ml_bull:.1f}/{ml_bear:.1f}{cap_str}"
        )

        # === Consecutive SL nudge: 3+ SLs → nudge opposite direction (not bucketed) ===
        if self._consec_sl_count >= 3:
            if self._consec_sl_direction == "CE":
                bear_score += 0.5
            elif self._consec_sl_direction == "PE":
                bull_score += 0.5

        # === H7: Final NaN guard on scores ===
        if math.isnan(bull_score) or math.isnan(bear_score):
            logger.warning(
                f"NaN in direction scores: bull={bull_score}, bear={bear_score} — skipping signal"
            )
            return None

        # === Direction decision ===
        if bull_score > bear_score:
            direction = "CE"
        elif bear_score > bull_score:
            direction = "PE"
        else:
            direction = ""

        # Factor dominance tracker — log top 2 contributing buckets
        if direction:
            winning = bull_score if direction == "CE" else bear_score
            buckets = {
                "MOM": momentum_bull if direction == "CE" else momentum_bear,
                "FLOW": flow_bull if direction == "CE" else flow_bear,
                "VOL": vol_bull if direction == "CE" else vol_bear,
                "MR": mr_bull if direction == "CE" else mr_bear,
                "ML": ml_bull if direction == "CE" else ml_bear,
            }
            sorted_b = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_b[:2]
            pct1 = top2[0][1] / winning * 100 if winning > 0 else 0
            pct2 = top2[1][1] / winning * 100 if winning > 0 and len(top2) > 1 else 0
            logger.info(
                f"FACTOR_DOMINANCE: {symbol} {direction} | "
                f"top={top2[0][0]}({top2[0][1]:.1f},{pct1:.0f}%) "
                f"2nd={top2[1][0]}({top2[1][1]:.1f},{pct2:.0f}%) | "
                f"total={winning:.1f}"
            )

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

    # ─────────────────────────────────────────────────────
    # Fix 2: Intraday Score Recalculation
    # ─────────────────────────────────────────────────────

    def _compute_intraday_score(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> tuple[float, float, str]:
        """Compute direction score from 5-min intraday candles.

        Recomputes F1 (EMA), F2 (RSI+MACD), F3 (price action),
        F6 (VIX — live), F8 (OI/PCR), F9 (volume).
        F4/F5/F10 kept from daily scores.
        """
        intraday_df = data.get("intraday_df")
        if intraday_df is None or intraday_df.empty or len(intraday_df) < 12:
            return 0.0, 0.0, ""

        bull_score = 0.0
        bear_score = 0.0

        closes = intraday_df["close"]
        close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else close
        open_price = float(intraday_df["open"].iloc[-1])
        # === F1: EMA trend from intraday candles ===
        ema_9 = float(FeatureEngine.ema(closes, 9).iloc[-1])
        ema_21 = float(FeatureEngine.ema(closes, 21).iloc[-1])
        ema_50_s = FeatureEngine.ema(closes, 50)
        ema_50 = float(ema_50_s.iloc[-1]) if not pd.isna(ema_50_s.iloc[-1]) else close

        ema_weight = data.get("ema_weight", 2.5)
        ema_base = ema_weight * 0.8 * 0.6
        ema_bonus = ema_weight * 0.2 * 0.6

        trend_up = ema_9 > ema_21 > ema_50
        trend_down = ema_9 < ema_21 < ema_50

        if trend_up:
            bull_score += ema_base
        elif trend_down:
            bear_score += ema_base

        if close > ema_21 * 1.005:
            bull_score += ema_bonus
        elif close < ema_21 * 0.995:
            bear_score += ema_bonus

        # === F2: RSI + MACD from intraday candles ===
        rsi_series = FeatureEngine.rsi(closes, 14)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        prev_rsi = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 and not pd.isna(rsi_series.iloc[-2]) else 50
        self._last_rsi[symbol] = rsi  # Cache for momentum decay check

        _, _, macd_hist_series = FeatureEngine.macd(closes)
        macd_hist = float(macd_hist_series.iloc[-1]) if not pd.isna(macd_hist_series.iloc[-1]) else 0
        prev_macd = float(macd_hist_series.iloc[-2]) if len(macd_hist_series) > 1 and not pd.isna(macd_hist_series.iloc[-2]) else 0

        if rsi > 58 and rsi > prev_rsi:
            bull_score += 0.75
        elif rsi < 42 and rsi < prev_rsi:
            bear_score += 0.75

        if macd_hist > 0 and macd_hist > prev_macd:
            bull_score += 0.75
        elif macd_hist < 0 and macd_hist < prev_macd:
            bear_score += 0.75

        # === F3: Price action from intraday candles ===
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close > 0 else 0
        if gap_pct > 0.1:
            bull_score += 0.75
        elif gap_pct < -0.1:
            bear_score += 0.75

        prev_high = float(intraday_df["high"].iloc[-2]) if len(intraday_df) > 1 else close
        prev_low = float(intraday_df["low"].iloc[-2]) if len(intraday_df) > 1 else close
        if close > prev_high:
            bull_score += 0.75
        elif close < prev_low:
            bear_score += 0.75

        if close > open_price:
            bull_score += 0.3
        elif close < open_price:
            bear_score += 0.3

        # === F6: VIX (live) ===
        vix = data.get("vix", 15)
        if vix < 13:
            bull_score += 0.5
        elif vix > 20:
            bear_score += 0.5

        if self._prev_vix > 0:
            vix_delta = vix - self._prev_vix
            if vix > 20 and vix_delta < -1.0:
                bull_score += 0.3
            elif vix_delta > 1.0:
                bear_score += 0.3

        # === F8: OI/PCR consensus (live) ===
        pcr_data = data.get("pcr", {})
        pcr_val = pcr_data.get(symbol, pcr_data.get("NIFTY", 1.0))
        if isinstance(pcr_val, dict):
            pcr_val = pcr_val.get("pcr_oi", 1.0)

        if pcr_val >= 1.3:
            bull_score += 1.0
        elif pcr_val >= 1.1:
            bull_score += 0.5
        elif pcr_val <= 0.7:
            bear_score += 1.0
        elif pcr_val <= 0.9:
            bear_score += 0.5

        # === F9: Volume from intraday candles ===
        if "volume" in intraday_df.columns:
            volume = float(intraday_df["volume"].iloc[-1])
            vol_ma = float(intraday_df["volume"].iloc[-20:].mean()) if len(intraday_df) >= 20 else float(intraday_df["volume"].mean())
            if vol_ma > 0 and volume > 0:
                vol_ratio = volume / vol_ma
                if vol_ratio > 1.3:
                    if close > open_price:
                        bull_score += 2.5
                    elif close < open_price:
                        bear_score += 2.5
                elif vol_ratio < 0.7:
                    if close > open_price:
                        bull_score -= 0.5
                    elif close < open_price:
                        bear_score -= 0.5

        # Direction decision
        if bull_score > bear_score:
            direction = "CE"
        elif bear_score > bull_score:
            direction = "PE"
        else:
            direction = ""

        return bull_score, bear_score, direction

    def _compute_momentum_direction(
        self, symbol: str, data: dict[str, Any],
    ) -> tuple[str, float, float, float]:
        """Per-loop momentum direction using blended daily+intraday scores + ML gates.

        Returns: (direction, bull_score, bear_score, score_diff)
        Direction is "" if no clear momentum or ML gates fail.
        """
        cfg = get_config()

        # Get blended scores (set by intraday_rescore) or fall back to daily
        bull, bear, diff = self._direction_scores.get(symbol, (0, 0, 0))
        if bull == 0 and bear == 0:
            # No scores yet — compute fresh daily
            bull, bear, _ = self._compute_direction_score(
                symbol, data, data.get("regime", "")
            )
            diff = abs(bull - bear)

        if diff < cfg.MOMENTUM_MIN_SCORE_DIFF:
            return "", bull, bear, diff

        if bull > bear:
            direction = "CE"
        elif bear > bull:
            direction = "PE"
        else:
            direction = ""

        if not direction:
            return "", bull, bear, diff

        # ML confidence gates
        pe_prob = data.get("ml_direction_prob_down", 0.5)
        ce_prob = data.get("ml_direction_prob_up", 0.5)

        if direction == "PE" and pe_prob < cfg.MOMENTUM_PE_MIN_PROB:
            return "", bull, bear, diff
        if direction == "CE" and ce_prob < cfg.MOMENTUM_CE_MIN_PROB:
            return "", bull, bear, diff

        return direction, bull, bear, diff

    def intraday_rescore(self, symbol: str, data: dict[str, Any]) -> None:
        """Blend daily + intraday scores using 30-min progressive weight schedule.

        Schedule (progressive intraday ramp):
          10:30 → daily 70% / intraday 30%
          11:00 → daily 50% / intraday 50%
          11:30 → daily 35% / intraday 65%
          12:00 → daily 25% / intraday 75%
          12:30 → daily 15% / intraday 85%

        Before 10:30: daily 100% (no rescore).
        After 12:30: keep last blended score (no more rescores).
        Updates _direction_scores with blended values. Does NOT change locked _direction.
        Skipped if: hard abort fired, or blend disabled.
        """
        cfg = get_config()
        if not cfg.INTRADAY_BLEND_ENABLED:
            return

        # Skip if hard abort already fired
        if self._abort_stage.get(symbol) == "HARD" and not self._abort_bypassed.get(symbol, False):
            return

        now = datetime.now().time()

        # Before 10:30: no rescore yet
        if now < dt_time(10, 30):
            return

        # Find the most recent schedule slot that has passed
        daily_weight = 1.0
        intraday_weight = 0.0
        matched_time_key = None

        for sched_time, d_w, i_w in reversed(self._rescore_schedule):
            if now >= sched_time:
                daily_weight = d_w
                intraday_weight = i_w
                matched_time_key = sched_time.strftime("%H:%M")
                break

        if intraday_weight == 0:
            return  # Before first schedule slot

        # Save daily scores on first blend (freeze them)
        if symbol not in self._daily_scores and symbol in self._direction_scores:
            self._daily_scores[symbol] = self._direction_scores[symbol]

        daily_bull, daily_bear, daily_diff = self._daily_scores.get(symbol, (0, 0, 0))
        if daily_bull == 0 and daily_bear == 0:
            return  # No daily scores yet

        # Compute intraday scores from 5-min candles
        intraday_bull, intraday_bear, intraday_dir = self._compute_intraday_score(symbol, data)
        if intraday_bull == 0 and intraday_bear == 0:
            return  # Insufficient intraday data

        self._intraday_scores[symbol] = (intraday_bull, intraday_bear, abs(intraday_bull - intraday_bear))

        # Blend: final = daily × daily_weight + intraday × intraday_weight
        blended_bull = daily_bull * daily_weight + intraday_bull * intraday_weight
        blended_bear = daily_bear * daily_weight + intraday_bear * intraday_weight
        blended_diff = abs(blended_bull - blended_bear)

        self._blended_scores[symbol] = (blended_bull, blended_bear, blended_diff)
        self._direction_scores[symbol] = (blended_bull, blended_bear, blended_diff)
        self._rescore_weight = intraday_weight

        # Update peak score diff tracking
        if symbol in self._peak_score_diff:
            if blended_diff > self._peak_score_diff[symbol]:
                self._peak_score_diff[symbol] = blended_diff

        # Log at each 30-min rescore window (once per slot)
        if matched_time_key and matched_time_key not in self._rescore_times_done:
            self._rescore_times_done.add(matched_time_key)
            blended_dir = "CE" if blended_bull > blended_bear else "PE"
            logger.info(
                f"RESCORE_30M {matched_time_key}: "
                f"daily={daily_diff:.1f} intraday={abs(intraday_bull - intraday_bear):.1f} "
                f"blended={blended_diff:.1f} "
                f"weight=daily{daily_weight:.0%}/intra{intraday_weight:.0%} "
                f"direction={blended_dir}"
            )

    # ─────────────────────────────────────────────────────
    # Rescore Exit (position-open exit at 30-min rescore)
    # ─────────────────────────────────────────────────────

    def init_peak_score(self, symbol: str, score_diff: float) -> None:
        """Set initial peak score_diff on trade entry."""
        self._peak_score_diff[symbol] = score_diff

    def clear_peak_score(self, symbol: str) -> None:
        """Clear peak score_diff on trade exit."""
        self._peak_score_diff.pop(symbol, None)

    def rescore_exit_check(
        self, symbol: str, position: Any, trade_direction: str,
    ) -> tuple[bool, str]:
        """Check if an open position should exit based on rescore results.

        Called at each 30-min rescore while position is open.

        Returns: (should_exit, reason)
          reason is "rescore_flip" or "rescore_decay" or ""
        """
        cfg = get_config()
        if not cfg.RESCORE_EXIT_ENABLED:
            return False, ""

        entry_price = position.entry_price
        if entry_price <= 0:
            return False, ""

        current_price = position.current_price
        profit_pct = (current_price - entry_price) / entry_price

        # Step 1: Below min profit → hold unconditionally
        if profit_pct < cfg.RESCORE_EXIT_MIN_PROFIT:
            logger.info(
                f"RESCORE_HOLD_LOW_PROFIT: {symbol} "
                f"profit={profit_pct:.1%} < {cfg.RESCORE_EXIT_MIN_PROFIT:.0%} threshold "
                f"skipping rescore exit check"
            )
            return False, ""

        # Get current blended direction from rescore
        bull, bear, diff = self._direction_scores.get(symbol, (0, 0, 0))
        if bull == 0 and bear == 0:
            return False, ""

        if bull > bear:
            rescore_direction = "CE"
        elif bear > bull:
            rescore_direction = "PE"
        else:
            rescore_direction = ""

        # Step 2: Direction flip → exit
        if rescore_direction and rescore_direction != trade_direction:
            logger.info(
                f"RESCORE_FLIP_EXIT: {symbol} "
                f"trade={trade_direction} rescore={rescore_direction} "
                f"profit={profit_pct:.1%} blended_diff={diff:.1f} "
                f"closing position early"
            )
            return True, "rescore_flip"

        # Step 3: Score decay (same direction)
        peak_sd = self._peak_score_diff.get(symbol, 0)
        if peak_sd > 0:
            score_decay = (peak_sd - diff) / peak_sd
            if (score_decay >= cfg.RESCORE_EXIT_DECAY_THRESHOLD
                    and profit_pct >= cfg.RESCORE_EXIT_DECAY_MIN_PROFIT):
                logger.info(
                    f"RESCORE_DECAY_EXIT: {symbol} "
                    f"peak_diff={peak_sd:.1f} current_diff={diff:.1f} "
                    f"decay={score_decay:.0%} profit={profit_pct:.1%} "
                    f"momentum fading, closing"
                )
                return True, "rescore_decay"

        # Hold — signal still valid
        logger.info(
            f"RESCORE_HOLD: {symbol} direction={rescore_direction or trade_direction} unchanged "
            f"diff={diff:.1f} profit={profit_pct:.1%} continuing"
        )
        return False, ""

    # ─────────────────────────────────────────────────────
    # Momentum Decay Exit
    # ─────────────────────────────────────────────────────

    def momentum_decay_check(self, symbol: str, position: Any) -> bool:
        """Check if an open position should exit due to momentum decay.

        Fires at rescore times (11:00, 12:30) when ALL of:
          1. Profit > MOMENTUM_DECAY_MIN_PROFIT (default 10%)
          2. Current score_diff < MOMENTUM_DECAY_FACTOR × peak_score_diff
          3. Current RSI dropped >= MOMENTUM_DECAY_RSI_DROP from peak_rsi

        Returns True if position should be closed for MOMENTUM_DECAY.
        """
        cfg = get_config()
        if not cfg.MOMENTUM_DECAY_ENABLED:
            return False

        entry_price = position.entry_price
        if entry_price <= 0:
            return False

        profit_pct = (position.current_price - entry_price) / entry_price
        if profit_pct < cfg.MOMENTUM_DECAY_MIN_PROFIT:
            return False

        peak_sd = position.peak_score_diff
        if peak_sd <= 0:
            return False

        # Get current score_diff from direction_scores
        current_sd = self._direction_scores.get(symbol, (0, 0, 0))[2]
        if current_sd >= cfg.MOMENTUM_DECAY_FACTOR * peak_sd:
            return False

        # RSI drop check
        current_rsi = self._get_current_rsi(symbol)
        rsi_drop = position.peak_rsi - current_rsi
        if rsi_drop < cfg.MOMENTUM_DECAY_RSI_DROP:
            return False

        logger.info(
            f"MOMENTUM_DECAY: {symbol} profit={profit_pct:.1%} "
            f"score_diff={current_sd:.1f}/{peak_sd:.1f} "
            f"rsi={current_rsi:.1f}/{position.peak_rsi:.1f} (drop={rsi_drop:.1f})"
        )
        return True

    def late_weak_exit_check(self, position: Any) -> bool:
        """Check if an open position should exit as a late weak position at 14:45.

        Fires when profit is between -MAX_PROFIT and +MAX_PROFIT (default ±5%).
        These are stale positions that will likely drift to EOD — exit early.

        Returns True if position should be closed for LATE_WEAK_EXIT.
        """
        cfg = get_config()
        if not cfg.LATE_WEAK_EXIT_ENABLED:
            return False

        entry_price = position.entry_price
        if entry_price <= 0:
            return False

        profit_pct = (position.current_price - entry_price) / entry_price
        if abs(profit_pct) >= cfg.LATE_WEAK_EXIT_MAX_PROFIT:
            return False

        logger.info(
            f"LATE_WEAK_EXIT: {position.symbol} profit={profit_pct:.1%} "
            f"(within ±{cfg.LATE_WEAK_EXIT_MAX_PROFIT:.0%} at 14:45)"
        )
        return True

    def update_peak_scoring(self, symbol: str, position: Any) -> None:
        """Update peak_score_diff and peak_rsi on position. Called at rescore times."""
        current_sd = self._direction_scores.get(symbol, (0, 0, 0))[2]
        current_rsi = self._get_current_rsi(symbol)

        if current_sd > position.peak_score_diff:
            position.peak_score_diff = current_sd
        if current_rsi > position.peak_rsi:
            position.peak_rsi = current_rsi

    def _get_current_rsi(self, symbol: str) -> float:
        """Get current intraday RSI for a symbol from cached 5-min data."""
        # Use intraday scores if available (set during rescore)
        # Fall back to last known RSI from direction scoring
        return getattr(self, "_last_rsi", {}).get(symbol, 50.0)

    # ─────────────────────────────────────────────────────
    # Fix 1: Direction Flip / Contradiction Handler
    # ─────────────────────────────────────────────────────

    def check_direction_contradiction(
        self,
        symbol: str,
        data: dict[str, Any],
        alerts: Any = None,
    ) -> str:
        """After intraday_rescore(), check for direction contradiction/flip.

        Returns: "AGREEMENT" | "WEAK_CONTRADICTION" | "CONTRADICTION" | "FLIP"
        """
        cfg = get_config()
        if cfg.MOMENTUM_MODE_ENABLED:
            return "AGREEMENT"  # Momentum mode handles direction changes natively

        daily_bull, daily_bear, _ = self._daily_scores.get(symbol, (0, 0, 0))
        intraday_bull, intraday_bear, _ = self._intraday_scores.get(symbol, (0, 0, 0))

        if (daily_bull == 0 and daily_bear == 0) or (intraday_bull == 0 and intraday_bear == 0):
            return "AGREEMENT"

        daily_dir = "CE" if daily_bull > daily_bear else "PE"
        intraday_dir = "CE" if intraday_bull > intraday_bear else "PE"
        intraday_diff = abs(intraday_bull - intraday_bear)
        time_key = max(self._rescore_times_done) if self._rescore_times_done else "?"

        if daily_dir == intraday_dir:
            blended = self._blended_scores.get(symbol, (0, 0, 0))
            logger.info(
                f"RESCORE {time_key}: {symbol} {daily_dir}\u2192{intraday_dir} "
                f"blended={blended[2]:.1f} "
                f"(daily={abs(daily_bull - daily_bear):.1f} "
                f"intra={intraday_diff:.1f})"
            )
            return "AGREEMENT"

        # Directions differ — check severity
        now = datetime.now().time()

        # Weak contradiction: diff < 1.5 → keep direction, reduce score 20%
        if intraday_diff < 1.5:
            old_diff = self._direction_scores.get(symbol, (0, 0, 0))[2]
            blended = self._blended_scores.get(symbol, self._direction_scores.get(symbol, (0, 0, 0)))
            reduced_bull = blended[0] * 0.8
            reduced_bear = blended[1] * 0.8
            reduced_diff = abs(reduced_bull - reduced_bear)
            self._direction_scores[symbol] = (reduced_bull, reduced_bear, reduced_diff)
            logger.info(
                f"RESCORE {time_key}: WEAK_CONTRADICTION {symbol} "
                f"{daily_dir} vs {intraday_dir}. "
                f"Score reduced {old_diff:.1f}\u2192{reduced_diff:.1f}"
            )
            return "WEAK_CONTRADICTION"

        # FLIP conditions: strong intraday signal, early enough, no prior trade/flip
        # Block flips on major expiry days (NIFTY_EXPIRY, BANKNIFTY_EXPIRY)
        flip_blocked_by_expiry = self._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")
        if (
            intraday_diff >= 2.5
            and now < dt_time(12, 0)
            and self._trades_today.get(symbol, 0) == 0
            and not self._direction_flipped_today
            and not flip_blocked_by_expiry
            and self._same_day_sl_count < 2
        ):
            old_dir = self._direction.get(symbol, daily_dir)
            self._direction.pop(symbol, None)
            self._direction[symbol] = intraday_dir
            self._direction_flipped_today = True
            self._logged_today.discard(symbol)
            logger.info(
                f"RESCORE {time_key}: DIRECTION_FLIP {symbol} "
                f"{old_dir}\u2192{intraday_dir} diff={intraday_diff:.1f}. "
                f"Intraday signal stronger."
            )
            if alerts and hasattr(alerts, "alert_direction_flip"):
                alerts.alert_direction_flip(symbol, old_dir, intraday_dir, intraday_diff)
            return "FLIP"

        # Strong CONTRADICTION: diff >= 1.5 but flip conditions not met
        self._signal_killed[symbol] = True
        logger.info(
            f"RESCORE {time_key}: STRONG_CONTRADICTION {symbol} "
            f"{daily_dir} vs {intraday_dir} diff={intraday_diff:.1f}. "
            f"Signal killed."
        )
        if alerts and hasattr(alerts, "alert_direction_contradiction"):
            alerts.alert_direction_contradiction(symbol, daily_dir, intraday_dir, intraday_diff)
        return "CONTRADICTION"

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
                if self._alert_fn:
                    self._alert_fn(
                        f"⚠️ Trade blocked: insufficient balance. "
                        f"Required ₹{self.MIN_WALLET_BALANCE:,.0f} "
                        f"Available ₹{wallet_balance:,.0f}"
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
        max_risk: float, cb_size_multiplier: float = 1.0,
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

        # Expiry position size multiplier (skip in Fixed-R mode to keep risk constant)
        if not self.fixed_r_sizing:
            if self._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
                lots = max(1, int(lots * 0.75))
            elif self._expiry_type == "SENSEX_EXPIRY":
                lots = max(1, int(lots * 0.90))

        # Circuit breaker loss-based size reduction
        if cb_size_multiplier < 1.0:
            lots = max(1, int(lots * cb_size_multiplier))
            logger.info(
                f"SIZE_REDUCED: CB multiplier {cb_size_multiplier:.0%} → {lots}L"
            )

        qty = lots * full_lot
        sl_distance = premium * sl_pct
        actual_risk = sl_distance * qty
        sizing_mode = "FIXED_R" if self.fixed_r_sizing else "DEPLOY"
        logger.info(
            f"SIZING_{sizing_mode}: {lots}L={qty}q | "
            f"prem=₹{premium:.0f} sl={sl_pct:.0%} | "
            f"risk_qty={lots_by_risk}L deploy_qty={lots_by_deploy}L | "
            f"risk=₹{actual_risk:,.0f}"
        )
        return qty

    def _determine_trade_type(self, regime: str, score_diff: float, iv_adjustment: float = 0.0) -> str:
        """V9 PLUS decision tree for trade type selection.

        VOLATILE/ELEVATED + conviction >= threshold → CREDIT_SPREAD
        High conviction (>= 3.0) → NAKED_BUY
        RANGEBOUND + conviction >= 2.5 → NAKED_BUY
        Everything else → SKIP

        Dual mode: VOLATILE + intraday score >= DUAL_MODE_MIN_SCORE → NAKED_BUY

        iv_adjustment: added to all thresholds (positive = harder entry)
        """
        cfg = get_config()
        if regime in ("VOLATILE", "ELEVATED"):
            # Dual mode: VOLATILE naked buy at lower threshold (replaces credit spread)
            cutoff_h, cutoff_m = (int(x) for x in cfg.DUAL_MODE_ENTRY_CUTOFF.split(":"))
            now_t = datetime.now().time()
            if (cfg.DUAL_MODE_ENABLED and self._dual_mode_active
                    and regime == "VOLATILE"
                    and self._dual_mode_trades_today < 1
                    and now_t >= dt_time(10, 30)
                    and now_t < dt_time(cutoff_h, cutoff_m)
                    and score_diff >= cfg.DUAL_MODE_MIN_SCORE + iv_adjustment):
                self._is_dual_mode_trade = True
                return "NAKED_BUY"
            if score_diff >= 2.0 + iv_adjustment:
                return "CREDIT_SPREAD"
            return "SKIP"
        if score_diff >= 3.0 + iv_adjustment:
            return "NAKED_BUY"
        if regime == "RANGEBOUND" and score_diff >= 2.5 + iv_adjustment:
            return "NAKED_BUY"
        return "SKIP"

    def _apply_safety_guards(self, trade_type: str, direction: str, symbol: str) -> str:
        """Apply safety guards to trade type. Returns modified trade_type or "GUARD_BLOCK"."""
        is_reversal = self._is_reversal_trade

        # Guard 2: Reversal always naked buy, never spread
        if is_reversal and trade_type in ("CREDIT_SPREAD", "DEBIT_SPREAD"):
            logger.info(f"GUARD_REVERSAL_NAKED: {symbol} reversal forced NAKED_BUY (was {trade_type})")
            return "NAKED_BUY"

        # Guard 3: One position type per day (reversal exempt)
        if self._today_position_type and not is_reversal:
            if self._today_position_type == "NAKED_BUY" and trade_type in ("CREDIT_SPREAD", "DEBIT_SPREAD"):
                logger.info(f"GUARD_ONE_TYPE: {symbol} spread blocked — already took NAKED_BUY today")
                return "GUARD_BLOCK"
            if self._today_position_type == "CREDIT_SPREAD":
                logger.info(f"GUARD_ONE_TYPE: {symbol} {trade_type} blocked — already took CREDIT_SPREAD today")
                return "GUARD_BLOCK"

        # Guard 1: No opposing spread after existing trade
        if trade_type in ("CREDIT_SPREAD", "DEBIT_SPREAD") and self._today_trade_direction:
            # Credit spread sells opposite direction: CE signal → PE Sell spread
            credit_dir = "PE" if direction == "CE" else "CE"
            if self._today_trade_direction == "CE" and credit_dir == "CE":
                # Today was bullish CE, now trying CE Sell (bearish) — block
                logger.info(f"GUARD_OPPOSING: {symbol} {credit_dir} Sell blocked — today was {self._today_trade_direction}")
                return "GUARD_BLOCK"
            if self._today_trade_direction == "PE" and credit_dir == "PE":
                # Today was bearish PE, now trying PE Sell (bullish) — block
                logger.info(f"GUARD_OPPOSING: {symbol} {credit_dir} Sell blocked — today was {self._today_trade_direction}")
                return "GUARD_BLOCK"

        return trade_type

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
                batch = self._data_fetcher.get_live_quotes_batch([leg1_key, leg2_key])
                leg1_prem = batch.get(leg1_key, {}).get("ltp", 0)
                leg2_prem = batch.get(leg2_key, {}).get("ltp", 0)
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

        # Lot sizing (Kelly-adjusted risk)
        spread_kelly = data.get("kelly_mult", 1.0)
        spread_risk = cfg.RISK_PER_TRADE * spread_kelly
        if trade_type == "DEBIT_SPREAD":
            lots_by_deploy = int(cfg.DEPLOY_CAP / (net_premium * full_lot)) if net_premium > 0 else 1
            lots_by_risk = int(spread_risk / (net_premium * full_lot)) if net_premium > 0 else 1
            lots = max(1, min(lots_by_deploy, lots_by_risk))
        else:
            lots_by_risk = int(spread_risk / (max_loss_per_unit * full_lot)) if max_loss_per_unit > 0 else 1
            lots = max(1, lots_by_risk)

        # Combined size reduction (CB + equity curve)
        cb_mult = min(data.get("cb_size_multiplier", 1.0), data.get("equity_size_multiplier", 1.0))
        if cb_mult < 1.0:
            lots = max(1, int(lots * cb_mult))
            logger.info(f"SIZE_REDUCED: Combined multiplier {cb_mult:.0%} → {lots}L")

        qty = lots * full_lot

        # Risk check
        total_max_loss = max_loss_per_unit * qty
        if total_max_loss > spread_risk:
            logger.info(f"OptionsBuyer: Spread max loss ₹{total_max_loss:.0f} > risk cap ₹{spread_risk:,.0f}")
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

    # ─────────────────────────────────────────────────────
    # Fix 4: Fuzzy Confirmation Thresholds
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_fuzzy_triggers(
        direction: str,
        latest_close: float,
        day_open: float,
        current_rsi: float,
        range_high: float,
        range_low: float,
        pcr: float,
    ) -> tuple[float, float, float, float, float]:
        """Compute fuzzy confirmation triggers (each 0.0-1.0).

        Returns: (t1, t2, t3, t4, trigger_sum)
        """
        # T1: PRICE_VS_OPEN — gradient based on distance from open
        # 0% diff=0.0, 0.5% diff=1.0, linear
        if day_open <= 0:
            t1 = 0.0
        else:
            dist_pct = (latest_close - day_open) / day_open * 100
            if direction == "CE":
                t1 = max(0.0, min(1.0, dist_pct / 0.5))
            else:  # PE
                t1 = max(0.0, min(1.0, -dist_pct / 0.5))

        # T2: RSI_SIGNAL — gradient based on RSI level
        # CE: RSI=45→0.0, RSI=55→0.5, RSI=65→1.0
        # PE: RSI=55→0.0, RSI=45→0.5, RSI=35→1.0
        if direction == "CE":
            t2 = max(0.0, min(1.0, (current_rsi - 45) / 20))
        else:  # PE
            t2 = max(0.0, min(1.0, (55 - current_rsi) / 20))

        # T3: BREAKOUT — gradient based on distance from range boundary
        if range_high <= 0 or range_low <= 0 or range_high <= range_low:
            t3 = 0.0
        else:
            range_width = range_high - range_low
            if direction == "CE":
                dist = latest_close - range_high
                if dist <= 0:
                    t3 = max(0.0, 0.5 * (latest_close - range_low) / range_width)
                else:
                    t3 = min(1.0, 0.5 + 0.5 * dist / (range_width * 0.5)) if range_width > 0 else 0.5
            else:  # PE
                dist = range_low - latest_close
                if dist <= 0:
                    t3 = max(0.0, 0.5 * (range_high - latest_close) / range_width)
                else:
                    t3 = min(1.0, 0.5 + 0.5 * dist / (range_width * 0.5)) if range_width > 0 else 0.5

        # T4: PCR_SIGNAL — gradient based on PCR level
        # CE: PCR=1.2→0.0, PCR=0.8→0.5, PCR=0.4→1.0
        # PE: PCR=0.8→0.0, PCR=1.2→0.5, PCR=1.6→1.0
        if direction == "CE":
            t4 = max(0.0, min(1.0, (1.2 - pcr) / 0.8))
        else:  # PE
            t4 = max(0.0, min(1.0, (pcr - 0.8) / 0.8))

        trigger_sum = t1 + t2 + t3 + t4
        return (round(t1, 2), round(t2, 2), round(t3, 2), round(t4, 2), round(trigger_sum, 2))

    # ─────────────────────────────────────────────────────
    # Fix 5: Adaptive Rolling Range
    # ─────────────────────────────────────────────────────

    def _update_rolling_range(
        self, symbol: str, intraday_df: pd.DataFrame,
    ) -> tuple[float, float, float, bool]:
        """Update rolling 60-minute range from 5-min candles.

        Updates every 30 minutes. Returns: (range_high, range_low, range_width_pct, too_tight)
        """
        now = datetime.now()

        if intraday_df is None or intraday_df.empty or len(intraday_df) < 3:
            rh = self._rolling_range_high.get(symbol, 0)
            rl = self._rolling_range_low.get(symbol, 0)
            mid = (rh + rl) / 2 if (rh + rl) > 0 else 1
            return rh, rl, (rh - rl) / mid * 100 if mid > 0 else 0, False

        # Check if update is needed (every 30 min)
        last_update = self._range_last_update.get(symbol)
        if last_update and (now - last_update).total_seconds() < 1740:
            rh = self._rolling_range_high.get(symbol, 0)
            rl = self._rolling_range_low.get(symbol, 0)
            mid = (rh + rl) / 2 if (rh + rl) > 0 else 1
            return rh, rl, (rh - rl) / mid * 100 if mid > 0 else 0, self._range_too_tight.get(symbol, False)

        # Last 12 bars (60 min of 5-min candles)
        window = intraday_df.tail(min(12, len(intraday_df)))
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        mid = (range_high + range_low) / 2 if (range_high + range_low) > 0 else 1
        range_width_pct = (range_high - range_low) / mid * 100
        too_tight = range_width_pct < 0.2

        self._rolling_range_high[symbol] = range_high
        self._rolling_range_low[symbol] = range_low
        self._range_last_update[symbol] = now
        self._range_too_tight[symbol] = too_tight

        logger.info(
            f"RANGE_UPDATE: {symbol} high={range_high:.0f} low={range_low:.0f} "
            f"width={range_width_pct:.2f}%{' TIGHT' if too_tight else ''}"
        )
        return range_high, range_low, range_width_pct, too_tight

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
        is_major_expiry = self._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY")

        # ── Step 1: Compute Direction from Multi-Factor Scoring ──
        cfg = get_config()

        if cfg.MOMENTUM_MODE_ENABLED:
            # ── Momentum mode: re-evaluate direction every loop ──
            if symbol in self._position_direction_lock:
                # Position open — use locked direction
                direction = self._position_direction_lock[symbol]
                bull_score, bear_score, score_diff = self._direction_scores.get(symbol, (0, 0, 0))
            else:
                # Position flat — compute fresh momentum direction
                direction, bull_score, bear_score, score_diff = self._compute_momentum_direction(
                    symbol, data
                )
                if not direction:
                    self._last_skip_info[symbol] = {
                        "reason": "MOMENTUM_NO_DIRECTION",
                        "score_diff": score_diff,
                    }
                    return None

                self._direction_scores[symbol] = (bull_score, bear_score, score_diff)

                # Save daily scores for blending (first computation only)
                if symbol not in self._daily_scores:
                    self._daily_scores[symbol] = (bull_score, bear_score, score_diff)

                # Log direction change
                prev_dir = self._direction.get(symbol)
                if prev_dir and prev_dir != direction:
                    logger.info(
                        f"MOMENTUM_FLIP: {symbol} {prev_dir}\u2192{direction} "
                        f"bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f}"
                    )

            self._direction[symbol] = direction
            trade_type = "NAKED_BUY"

            if cfg.TRADING_STAGE == "PLUS":
                trade_type = self._determine_trade_type(regime, score_diff, 0)
                if trade_type == "SKIP":
                    self._last_skip_info[symbol] = {
                        "reason": "CONVICTION_BELOW_THRESHOLD",
                        "score_diff": score_diff, "threshold": 0, "direction": direction,
                    }
                    return None

                # Safety guards
                trade_type = self._apply_safety_guards(trade_type, direction, symbol)
                if trade_type == "GUARD_BLOCK":
                    return None

                if trade_type == "NAKED_BUY" and self._naked_trades_today >= 2:
                    return None
                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD") and self._spread_trades_today >= 2:
                    return None

                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD"):
                    self._current_trade_type[symbol] = trade_type
                    spread_signal = self._build_spread_signal(
                        symbol, direction, trade_type, score_diff, regime, data,
                    )
                    if spread_signal:
                        self._spread_trades_today += 1
                    return spread_signal

            self._current_trade_type[symbol] = trade_type

            if symbol not in self._logged_today:
                ml_prob_up = data.get("ml_direction_prob_up", 0.5)
                ml_prob_down = data.get("ml_direction_prob_down", 0.5)
                logger.info(
                    f"MOMENTUM: {symbol} direction={direction} | "
                    f"bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} | "
                    f"ML P(up)={ml_prob_up:.3f} P(down)={ml_prob_down:.3f} | regime={regime}"
                )
                self._logged_today.add(symbol)

        elif symbol not in self._direction:
            bull_score, bear_score, direction = self._compute_direction_score(
                symbol, data, regime
            )
            score_diff = abs(bull_score - bear_score)

            # Store scores for logging
            self._direction_scores[symbol] = (bull_score, bear_score, score_diff)

            # Fix 2: Save daily scores for intraday rescore blending
            if symbol not in self._daily_scores:
                self._daily_scores[symbol] = (bull_score, bear_score, score_diff)

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

            # ── IV Awareness Filter ──
            cfg = get_config()
            iv_adjustment = 0.0
            is_expiry_day = data.get("is_expiry_day", False)
            if cfg.IV_FILTER_ENABLED and not is_expiry_day and len(self._vix_history) >= 20:
                vix_now = data.get("vix", 15)
                vix_20d_avg = sum(self._vix_history[-20:]) / 20
                if vix_20d_avg > 0:
                    iv_ratio = vix_now / vix_20d_avg
                    if iv_ratio > cfg.IV_HIGH_THRESHOLD:
                        iv_adjustment = cfg.IV_HIGH_PENALTY
                    elif iv_ratio < cfg.IV_LOW_THRESHOLD:
                        iv_adjustment = -cfg.IV_LOW_BONUS
                    logger.info(
                        f"IV_FILTER: VIX={vix_now:.1f} avg={vix_20d_avg:.1f} "
                        f"ratio={iv_ratio:.2f} regime={'IV_HIGH' if iv_ratio > cfg.IV_HIGH_THRESHOLD else 'IV_LOW' if iv_ratio < cfg.IV_LOW_THRESHOLD else 'IV_NORMAL'} "
                        f"adj={iv_adjustment:+.2f}"
                    )

            # ── OI Change Rate Confirmation Filter ──
            oi_adjustment = 0.0
            if cfg.OI_CHANGE_FILTER_ENABLED and self._data_fetcher and hasattr(self._data_fetcher, "get_oi_change_rates"):
                put_change, call_change = self._data_fetcher.get_oi_change_rates()
                if put_change is not None and call_change is not None:
                    oi_status = "NEUTRAL"
                    confirmed_thr = cfg.OI_CHANGE_CONFIRMED_THRESHOLD
                    contradicted_thr = cfg.OI_CHANGE_CONTRADICTED_THRESHOLD
                    if direction == "PE":
                        if put_change > confirmed_thr and call_change < confirmed_thr:
                            oi_adjustment = -cfg.OI_CONFIRMED_BONUS
                            oi_status = "CONFIRMED"
                        elif call_change > contradicted_thr and put_change < 0:
                            oi_adjustment = cfg.OI_CONTRADICTED_PENALTY
                            oi_status = "CONTRADICTED"
                    elif direction == "CE":
                        if call_change > confirmed_thr and put_change < confirmed_thr:
                            oi_adjustment = -cfg.OI_CONFIRMED_BONUS
                            oi_status = "CONFIRMED"
                        elif put_change > contradicted_thr and call_change < 0:
                            oi_adjustment = cfg.OI_CONTRADICTED_PENALTY
                            oi_status = "CONTRADICTED"
                    logger.info(
                        f"OI_CHANGE: put={put_change:+.1f}% call={call_change:+.1f}% "
                        f"status={oi_status} adj={oi_adjustment:+.2f}"
                    )

            total_adjustment = iv_adjustment + oi_adjustment

            # ── Trade type selection ──
            if cfg.TRADING_STAGE == "PLUS":
                # PLUS decision tree
                trade_type = self._determine_trade_type(regime, score_diff, total_adjustment)
                if trade_type == "SKIP":
                    self._last_skip_info[symbol] = {
                        "reason": "CONVICTION_BELOW_THRESHOLD",
                        "score_diff": score_diff, "threshold": 0, "direction": direction,
                    }
                    self._record_counterfactual(
                        symbol, direction, "CONVICTION_BELOW_THRESHOLD",
                        data.get("nifty_price", 0), regime, score_diff,
                        bull_score, bear_score,
                    )
                    if symbol not in self._logged_today:
                        logger.info(
                            f"OptionsBuyer: VOLATILE + low conviction → SKIP {symbol} "
                            f"(diff={score_diff:.1f})"
                        )
                        self._logged_today.add(symbol)
                    return None

                # Safety guards
                trade_type = self._apply_safety_guards(trade_type, direction, symbol)
                if trade_type == "GUARD_BLOCK":
                    return None

                # Per-type daily limits: max 2 naked + 2 spreads
                if trade_type == "NAKED_BUY" and self._naked_trades_today >= 2:
                    return None
                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD") and self._spread_trades_today >= 2:
                    return None

                # For spreads, build signal and return early
                if trade_type in ("DEBIT_SPREAD", "CREDIT_SPREAD"):
                    self._direction[symbol] = direction
                    self._current_trade_type[symbol] = trade_type
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

                # IV awareness + OI change rate filter
                full_threshold += total_adjustment

                if score_diff >= full_threshold and direction != "":
                    trade_type = "FULL"
                elif (cfg.DUAL_MODE_ENABLED and self._dual_mode_active
                      and regime == "VOLATILE"
                      and self._dual_mode_trades_today < 1
                      and now >= dt_time(10, 30)
                      and now < dt_time(12, 0)
                      and score_diff >= cfg.DUAL_MODE_MIN_SCORE
                      and direction != ""):
                    trade_type = "FULL"
                    self._is_dual_mode_trade = True
                else:
                    self._last_skip_info[symbol] = {
                        "reason": "CONVICTION_BELOW_THRESHOLD",
                        "score_diff": score_diff, "threshold": full_threshold,
                        "direction": direction,
                    }
                    self._record_counterfactual(
                        symbol, direction, "CONVICTION_BELOW_THRESHOLD",
                        data.get("nifty_price", 0), regime, score_diff,
                        bull_score, bear_score,
                        {"threshold": full_threshold},
                    )
                    reentry_n = self._reentry_eval_count.get(symbol)
                    if symbol not in self._logged_today or (reentry_n is not None and reentry_n < 5):
                        logger.info(
                            f"OptionsBuyer: No conviction for {symbol} — "
                            f"bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} "
                            f"(need >= {full_threshold:.1f} in {mode}) | regime={regime}"
                            f"{f' [re-entry eval #{reentry_n + 1}]' if reentry_n is not None else ''}"
                        )
                        self._logged_today.add(symbol)
                        if reentry_n is not None:
                            self._reentry_eval_count[symbol] = reentry_n + 1
                    return None

            self._direction[symbol] = direction
            # Store trade type per symbol (isolated to prevent cross-symbol leakage)
            self._current_trade_type[symbol] = trade_type

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
        # Exception: reversal trade in opposite direction is allowed
        if self._consec_sl_count >= 3 and direction == self._consec_sl_direction:
            cfg_rev = get_config()
            if (cfg_rev.REVERSAL_ENABLED and self._reversal_eligible
                    and direction != self._last_exit_direction):
                logger.info(
                    f"COOLDOWN_BYPASSED: reversal eligible "
                    f"{self._last_exit_direction}→{direction} entry allowed"
                )
            else:
                self._last_skip_info[symbol] = {"reason": "DIRECTION_LOCKED", "direction": direction}
                if "SL_BLOCK" not in self._logged_today:
                    logger.info(
                        f"OptionsBuyer: BLOCKED — {self._consec_sl_count} consecutive "
                        f"{self._consec_sl_direction} stop losses, blocking {direction}"
                    )
                    self._logged_today.add("SL_BLOCK")
                return None

        # Fix 1: Direction killed by contradiction
        if self._signal_killed.get(symbol, False):
            self._last_skip_info[symbol] = {"reason": "CONTRADICTION_KILLED", "direction": direction}
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
            # Initialize rolling range with morning range
            self._rolling_range_high[symbol] = self._morning_high[symbol]
            self._rolling_range_low[symbol] = self._morning_low[symbol]
            self._range_last_update[symbol] = datetime.now()
            logger.info(
                f"OptionsBuyer: {symbol} morning range = "
                f"{self._morning_low[symbol]:.0f} - {self._morning_high[symbol]:.0f}"
            )

        # Don't trade before 10:00 (morning noise + gap fills settle by 9:45-10:00)
        if now < dt_time(10, 0):
            return None

        # Fix 5: Update rolling range (every 30 min, or immediately after trade close)
        range_high, range_low, range_width_pct, range_too_tight = self._update_rolling_range(
            symbol, intraday_df
        )

        # Log new range after breakout reset (re-entry mode)
        if self._breakout_pending_reentry.get(symbol, False):
            latest = float(intraday_df["close"].iloc[-1])
            logger.info(
                f"BREAKOUT_REENTRY: new breakout range. "
                f"Range high={range_high:.0f} low={range_low:.0f} current={latest:.0f}"
            )
            self._breakout_pending_reentry[symbol] = False

        # Range too tight filter — ADX < 15 territory, no reliable breakout
        if range_too_tight:
            self._last_skip_info[symbol] = {
                "reason": "RANGE_TOO_TIGHT", "width_pct": round(range_width_pct, 2),
            }
            return None

        # Fix 3: Time-based abort mechanism
        # Skip in momentum mode — direction re-evaluates every loop
        abort_bypassed = self._abort_bypassed.get(symbol, False)
        if not cfg.MOMENTUM_MODE_ENABLED and abort_bypassed and "ABORT_BYPASS_LOGGED" not in self._logged_today:
            logger.info(f"ABORT_BYPASS: {symbol} — re-entry mode after closed trade")
            self._logged_today.add("ABORT_BYPASS_LOGGED")
        if not cfg.MOMENTUM_MODE_ENABLED and not abort_bypassed:
            # Determine hard abort time: 11:30 on major expiry, 13:00 otherwise
            hard_abort_time = dt_time(11, 30) if is_major_expiry else dt_time(13, 0)

            if now >= hard_abort_time and self._abort_stage.get(symbol) != "HARD":
                if self._trades_today.get(symbol, 0) == 0:
                    self._abort_stage[symbol] = "HARD"
                    dir_str = self._direction.get(symbol, "?")
                    diff_val = self._direction_scores.get(symbol, (0, 0, 0))[2]
                    failed = self._failed_confirm_count.get(symbol, 0)
                    if is_major_expiry:
                        logger.info(
                            f"EXPIRY_HARD_ABORT 11:30: {symbol} — major expiry day. "
                            f"No entry in window. Sitting out."
                        )
                        if self._alert_fn:
                            self._alert_fn(
                                f"\U0001f6d1 Expiry day — no entry by 11:30. Done."
                            )
                    else:
                        logger.info(
                            f"HARD_ABORT 13:00: {symbol} — no entry in 3hrs. "
                            f"Daily direction={dir_str} diff={diff_val:.1f}. Sitting out."
                        )
                        if self._alert_fn:
                            self._alert_fn(
                                f"\U0001f6d1 SIGNAL ABORTED — {symbol}\n"
                                f"3hrs no entry. direction={dir_str} diff={diff_val:.1f}\n"
                                f"Failed confirmations: {failed}. Sitting out."
                            )
                    # Clear direction — signal is dead
                    self._direction.pop(symbol, None)
            elif not is_major_expiry and now >= dt_time(11, 30) and self._abort_stage.get(symbol, "NONE") == "NONE":
                if self._trades_today.get(symbol, 0) == 0:
                    self._abort_stage[symbol] = "SOFT"
                    failed = self._failed_confirm_count.get(symbol, 0)
                    old_thresh = 2.8 if data.get("regime") == "VOLATILE" else 2.0
                    new_thresh = old_thresh + 0.5
                    logger.info(
                        f"SOFT_ABORT 11:30: {symbol} — threshold raised "
                        f"{old_thresh:.1f}\u2192{new_thresh:.1f}. "
                        f"{failed} failed confirmations so far."
                    )
                    if self._alert_fn:
                        self._alert_fn(
                            f"\u26a0\ufe0f No entry by 11:30 — {symbol}\n"
                            f"Raising bar (threshold {new_thresh:.1f}). "
                            f"{failed} failed confirmations."
                        )

        # Hard abort: block all entries (skip in momentum mode)
        if not cfg.MOMENTUM_MODE_ENABLED and self._abort_stage.get(symbol) == "HARD" and not abort_bypassed:
            self._last_skip_info[symbol] = {
                "reason": "HARD_ABORT",
                "failed_confirms": self._failed_confirm_count.get(symbol, 0),
            }
            return None

        # ── Step 3: Fuzzy Intraday Confirmations (Fix 4) ──
        latest_close = float(intraday_df["close"].iloc[-1])
        day_open = float(intraday_df["open"].iloc[0])

        # RSI on 5-min chart
        rsi_series = FeatureEngine.rsi(intraday_df["close"], period=14)
        current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50
        self._last_rsi[symbol] = current_rsi  # Cache for momentum decay check

        # Use rolling range for breakout detection (Fix 5)
        pcr = data.get("pcr", {}).get(symbol, 1.0)

        t1, t2, t3, t4, trigger_sum = self._compute_fuzzy_triggers(
            direction, latest_close, day_open, current_rsi, range_high, range_low, pcr
        )
        triggers = f"T1={t1:.1f}+T2={t2:.1f}+T3={t3:.1f}+T4={t4:.1f}={trigger_sum:.1f}"

        # ── Price contradiction check: block entry when price opposes direction ──
        cfg = get_config()
        if cfg.PRICE_CONTRADICTION_ENABLED and self._expiry_type == "NORMAL" and now < dt_time(11, 30):
            dist_from_open = (latest_close - day_open) / day_open if day_open > 0 else 0
            pc_threshold = cfg.PRICE_CONTRADICTION_THRESHOLD
            if (direction == "PE" and dist_from_open > pc_threshold and current_rsi > 55) or \
               (direction == "CE" and dist_from_open < -pc_threshold and current_rsi < 45):
                logger.info(
                    f"PRICE_CONTRADICTION: {symbol} {direction} signal but "
                    f"NIFTY {dist_from_open:+.1%} from open RSI={current_rsi:.1f} "
                    f"Waiting for price to confirm direction"
                )
                self._last_skip_info[symbol] = {
                    "reason": "PRICE_CONTRADICTS_SIGNAL",
                    "direction": direction,
                    "dist_from_open": round(dist_from_open, 4),
                    "rsi": round(current_rsi, 1),
                }
                _bs, _brs, _sd = self._direction_scores.get(symbol, (0, 0, 0))
                self._record_counterfactual(
                    symbol, direction, "PRICE_CONTRADICTS_SIGNAL",
                    latest_close, regime, _sd, _bs, _brs,
                    {"dist_from_open": round(dist_from_open, 4), "rsi": round(current_rsi, 1)},
                )
                return None

        # ── ML disagreement check: block when ML strongly opposes direction ──
        if cfg.ML_DISAGREEMENT_ENABLED and now < dt_time(11, 30):
            ml_v2_on = data.get("ml_v2_ready", False)
            ml_ce_on = data.get("ml_ce_ready", False)
            ml_pe_on = data.get("ml_pe_ready", False)
            ml_disagrees = False
            ml_dir = "FLAT"
            ml_conf = 0.5

            if ml_v2_on:
                # V2: both binary models deployed — use combined direction
                ml_dir = data.get("ml_v2_direction", "FLAT")
                v2_pe = data.get("ml_v2_pe_prob", 0.5)
                v2_ce = data.get("ml_v2_ce_prob", 0.5)
                ml_conf = max(v2_pe, v2_ce)
                # Only disagree when ML direction actively opposes score direction
                ml_disagrees = ml_dir != "FLAT" and ml_dir != direction
            else:
                # Hybrid: use deployed binary model for disagreement if available
                if direction == "PE" and ml_ce_on:
                    # CE binary model says bullish? Block PE entry
                    ce_bp = data.get("ml_ce_binary_prob", 0.5)
                    ml_disagrees = ce_bp > 0.60
                    ml_dir = "CE"
                    ml_conf = ce_bp
                elif direction == "CE" and ml_pe_on:
                    # PE binary model says bearish? Block CE entry
                    pe_bp = data.get("ml_pe_binary_prob", 0.5)
                    ml_disagrees = pe_bp > 0.60
                    ml_dir = "PE"
                    ml_conf = pe_bp
                else:
                    # V1 fallback
                    ml_prob_ce = data.get("ml_stage1_prob_ce", 0.33)
                    ml_prob_pe = data.get("ml_stage1_prob_pe", 0.33)
                    ml_threshold = cfg.ML_DISAGREEMENT_THRESHOLD
                    ml_disagrees = (
                        (direction == "PE" and ml_prob_ce > ml_threshold) or
                        (direction == "CE" and ml_prob_pe > ml_threshold)
                    )
                    ml_dir = "CE" if ml_prob_ce > ml_prob_pe else "PE"
                    ml_conf = max(ml_prob_ce, ml_prob_pe)

                # Guard: never block when ML agrees with score direction
                if ml_dir == direction:
                    ml_disagrees = False

            if ml_disagrees:
                logger.info(
                    f"ML_DISAGREEMENT: {symbol} score={direction} but "
                    f"ML predicts {ml_dir} ({ml_conf:.1%}) "
                    f"Blocking entry until alignment"
                )
                self._last_skip_info[symbol] = {
                    "reason": "ML_DISAGREES_WITH_DIRECTION",
                    "direction": direction,
                    "ml_direction": ml_dir,
                    "ml_confidence": round(ml_conf, 3),
                }
                _bs, _brs, _sd = self._direction_scores.get(symbol, (0, 0, 0))
                self._record_counterfactual(
                    symbol, direction, "ML_DISAGREES_WITH_DIRECTION",
                    latest_close, regime, _sd, _bs, _brs,
                    {"ml_direction": ml_dir, "ml_confidence": round(ml_conf, 3)},
                )
                return None
            elif ml_dir == direction and ml_dir != "FLAT":
                if f"ML_AGREE_{symbol}" not in self._logged_today:
                    logger.info(
                        f"ML_AGREE: {symbol} both score and ML say {direction} "
                        f"({ml_conf:.1%}) — entry allowed"
                    )
                    self._logged_today.add(f"ML_AGREE_{symbol}")

        # ── PE confidence filter: context-aware tolerance zone ──
        if direction == "PE" and cfg.PE_FILTER_ENABLED:
            pe_prob = data.get("ml_pe_binary_prob", 0.0)
            pe_v2 = data.get("ml_v2_pe_prob", 0.0)
            pe_conf = pe_v2 if data.get("ml_v2_ready", False) else pe_prob
            _bs, _brs, _sd = self._direction_scores.get(symbol, (0, 0, 0))
            vix_now = data.get("vix", 0)
            vix_open = data.get("vix_open", 0)
            passes, block_reason = self._pe_filter_passes(
                pe_conf, _sd, vix_now, vix_open
            )
            if not passes and pe_conf > 0:
                if block_reason == "PE_TOLERANCE_BLOCK":
                    logger.info(
                        f"PE_TOLERANCE_BLOCK: {symbol} pe_prob={pe_conf:.3f} "
                        f"in tolerance zone but weak setup "
                        f"(score_diff={_sd:.1f}, vix_delta={vix_now - vix_open:+.1f})"
                    )
                else:
                    logger.info(
                        f"PE_FILTER: {symbol} PE blocked — pe_prob={pe_conf:.1%} "
                        f"< {cfg.PE_FILTER_TOLERANCE_LOW:.0%}"
                    )
                self._last_skip_info[symbol] = {
                    "reason": block_reason,
                    "pe_prob": round(pe_conf, 3),
                    "threshold": cfg.PE_FILTER_THRESHOLD,
                }
                self._record_counterfactual(
                    symbol, direction, block_reason,
                    latest_close, regime, _sd, _bs, _brs,
                    {"pe_prob": round(pe_conf, 3),
                     "threshold": cfg.PE_FILTER_THRESHOLD,
                     "tolerance_low": cfg.PE_FILTER_TOLERANCE_LOW},
                )
                return None
            elif passes and pe_conf > 0 and pe_conf < cfg.PE_FILTER_THRESHOLD:
                # Tolerance zone pass — log for analysis
                if f"PE_TOL_PASS_{symbol}" not in self._logged_today:
                    logger.info(
                        f"PE_TOLERANCE_PASS: {symbol} pe_prob={pe_conf:.3f} "
                        f"allowed — strong setup (score_diff={_sd:.1f}, "
                        f"vix_delta={vix_now - vix_open:+.1f})"
                    )
                    self._logged_today.add(f"PE_TOL_PASS_{symbol}")

        # ── CE confidence filter: context-aware tolerance zone ──
        if direction == "CE" and cfg.CE_FILTER_ENABLED:
            ce_prob = data.get("ml_ce_binary_prob", 0.0)
            ce_v2 = data.get("ml_v2_ce_prob", 0.0)
            ce_conf = ce_v2 if data.get("ml_v2_ready", False) else ce_prob
            _bs, _brs, _sd = self._direction_scores.get(symbol, (0, 0, 0))
            vix_now = data.get("vix", 0)
            vix_open = data.get("vix_open", 0)
            passes, block_reason = self._ce_filter_passes(
                ce_conf, _sd, vix_now, vix_open
            )
            if not passes and ce_conf > 0:
                if block_reason == "CE_TOLERANCE_BLOCK":
                    logger.info(
                        f"CE_TOLERANCE_BLOCK: {symbol} ce_prob={ce_conf:.3f} "
                        f"in tolerance zone but weak setup "
                        f"(score_diff={_sd:.1f}, vix_delta={vix_open - vix_now:+.1f})"
                    )
                else:
                    logger.info(
                        f"CE_FILTER: {symbol} CE blocked — ce_prob={ce_conf:.1%} "
                        f"< {cfg.CE_FILTER_TOLERANCE_LOW:.0%}"
                    )
                self._last_skip_info[symbol] = {
                    "reason": block_reason,
                    "ce_prob": round(ce_conf, 3),
                    "threshold": cfg.CE_FILTER_THRESHOLD,
                }
                self._record_counterfactual(
                    symbol, direction, block_reason,
                    latest_close, regime, _sd, _bs, _brs,
                    {"ce_prob": round(ce_conf, 3),
                     "threshold": cfg.CE_FILTER_THRESHOLD,
                     "tolerance_low": cfg.CE_FILTER_TOLERANCE_LOW},
                )
                return None
            elif passes and ce_conf > 0 and ce_conf < cfg.CE_FILTER_THRESHOLD:
                if f"CE_TOL_PASS_{symbol}" not in self._logged_today:
                    logger.info(
                        f"CE_TOLERANCE_PASS: {symbol} ce_prob={ce_conf:.3f} "
                        f"allowed — strong setup (score_diff={_sd:.1f}, "
                        f"vix_delta={vix_open - vix_now:+.1f})"
                    )
                    self._logged_today.add(f"CE_TOL_PASS_{symbol}")

        bull_score, bear_score, score_diff = self._direction_scores.get(
            symbol, (0, 0, 0)
        )

        # Fuzzy threshold: 2.0 (normal), 2.8 (volatile)
        regime = data.get("regime", regime)
        if regime == "VOLATILE":
            fuzzy_threshold = 2.8
        else:
            fuzzy_threshold = 2.0

        # Adaptive fuzzy: lower threshold for strong signals (before cutoff hour)
        cfg = get_config()
        _adaptive_applied = ""
        if cfg.ADAPTIVE_FUZZY_ENABLED and datetime.now().hour < cfg.ADAPTIVE_FUZZY_CUTOFF_HOUR:
            if score_diff >= cfg.ADAPTIVE_FUZZY_STRONG_SCORE:
                fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_STRONG_THRESHOLD)
                _adaptive_applied = "STRONG"
            elif score_diff >= cfg.ADAPTIVE_FUZZY_MID_SCORE:
                fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_MID_THRESHOLD)
                _adaptive_applied = "MID"

        # Expiry type adjustments to confirmation threshold
        if self._expiry_type in ("NIFTY_EXPIRY", "BANKNIFTY_EXPIRY"):
            fuzzy_threshold += 1.0  # Major expiry: +1.0
        elif self._expiry_type == "SENSEX_EXPIRY":
            fuzzy_threshold += 0.5  # Minor expiry: +0.5

        # Fix 3: Soft abort raises threshold by 0.5
        if self._abort_stage.get(symbol) == "SOFT" and not abort_bypassed:
            fuzzy_threshold += 0.5

        if trigger_sum < fuzzy_threshold:
            # Fix 3: Increment failed confirmation counter
            self._failed_confirm_count[symbol] = self._failed_confirm_count.get(symbol, 0) + 1
            if self._failed_confirm_count[symbol] % 10 == 0:
                logger.info(
                    f"CONFIRM_FAIL_COUNT: {symbol} — {self._failed_confirm_count[symbol]} failures | "
                    f"abort_stage={self._abort_stage.get(symbol, 'NONE')}"
                )

            self._last_skip_info[symbol] = {
                "reason": "CONFIRMATION_FAILED",
                "score_diff": score_diff, "threshold": fuzzy_threshold,
                "direction": direction,
                "triggers": triggers,
            }
            self._record_counterfactual(
                symbol, direction, "CONFIRMATION_FAILED",
                latest_close, regime, score_diff, bull_score, bear_score,
                {"triggers": triggers, "threshold": fuzzy_threshold},
            )
            # ── Confirmation timeout: unlock stuck direction after 30 min ──
            now_dt = datetime.now()
            if symbol not in self._traded_today and self._trades_today.get(symbol, 0) == 0:
                if self._abort_stage.get(symbol) != "HARD":
                    if symbol not in self._confirm_fail_since:
                        self._confirm_fail_since[symbol] = now_dt
                    elif (now_dt - self._confirm_fail_since[symbol]).total_seconds() >= 1800:
                        if self._direction_rescores_today < 3:
                            old_dir = direction
                            self._direction.pop(symbol, None)
                            self._direction_scores.pop(symbol, None)
                            self._confirm_fail_since.pop(symbol, None)
                            self._direction_rescores_today += 1
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
                            return None

            # Log skipped signals every ~5 minutes (not every 30s)
            skip_key = f"SKIP_{symbol}_{now.minute // 5}"
            if skip_key not in self._logged_today:
                logger.info(
                    f"SKIPPED: {symbol} {direction} | score_diff={score_diff:.1f} "
                    f"{triggers} (need {fuzzy_threshold:.1f}) | "
                    f"vs_open={'above' if latest_close > day_open else 'below'} "
                    f"RSI={current_rsi:.1f} | close={latest_close:.0f} "
                    f"range=[{range_low:.0f}-{range_high:.0f}]"
                )
                self._logged_today.add(skip_key)
            return None

        # Confirmations passed — clear timeout tracker
        self._confirm_fail_since.pop(symbol, None)

        # ── Entry distance filter: block chasing moves already exhausted ──
        cfg = get_config()
        max_dist = cfg.MAX_ENTRY_DIST_FROM_OPEN
        if cfg.ENTRY_DIST_FILTER_ENABLED and max_dist > 0 and day_open > 0:
            dist_from_open = (latest_close - day_open) / day_open
            # Regime-specific limits
            if regime == "TRENDING":
                effective_limit = max_dist * 1.5   # 1.2% — trends can run
            elif regime == "VOLATILE":
                effective_limit = max_dist * 0.6   # 0.48% — volatility = exhaustion
            elif regime == "RANGEBOUND":
                effective_limit = max_dist * 0.5   # 0.4% — tight range
            else:  # ELEVATED
                effective_limit = max_dist * 0.75  # 0.6% — cautious

            blocked = False
            if direction == "PE" and dist_from_open < -effective_limit:
                blocked = True
            elif direction == "CE" and dist_from_open > effective_limit:
                blocked = True

            if blocked:
                dist_key = f"ENTRY_DIST_{symbol}_{now.hour}"
                if dist_key not in self._logged_today:
                    logger.info(
                        f"ENTRY_DIST_BLOCKED: {symbol} {direction} | "
                        f"dist={dist_from_open:+.2%} limit={effective_limit:.2%} ({regime}) | "
                        f"waiting for pullback"
                    )
                    self._logged_today.add(dist_key)
                self._last_skip_info[symbol] = {
                    "reason": "ENTRY_DIST_BLOCKED",
                    "score_diff": score_diff, "direction": direction,
                    "dist_from_open": dist_from_open,
                    "effective_limit": effective_limit,
                }
                self._record_counterfactual(
                    symbol, direction, "ENTRY_DIST_BLOCKED",
                    latest_close, regime, score_diff, bull_score, bear_score,
                    {"dist_from_open": round(dist_from_open, 4), "limit": round(effective_limit, 4)},
                )
                return None  # Block entry, keep direction signal alive

        bias = "BULLISH" if direction == "CE" else "BEARISH"

        logger.info(
            f"OptionsBuyer: {symbol} CONFIRMED {bias} — "
            f"{triggers} (need {fuzzy_threshold:.1f}) | "
            f"close={latest_close:.0f} open={day_open:.0f} RSI={current_rsi:.1f} "
            f"PCR={pcr:.2f} | scores: bull={bull_score:.1f} bear={bear_score:.1f}"
        )
        logger.info(
            f"FUZZY_AT_ENTRY: {symbol} | "
            f"trigger_sum={trigger_sum:.2f} threshold={fuzzy_threshold:.2f} "
            f"score_diff={score_diff:.1f} adaptive={_adaptive_applied or 'NONE'}"
        )

        # ── Stage 2: ML Quality Gate ──
        ml_quality_fn = data.get("ml_quality_predict")
        if ml_quality_fn and data.get("ml_quality_ready", False):
            quality_features = {
                "score_diff": score_diff,
                "conviction": score_diff * 0.06 + trigger_sum * 0.04,
                "vix_at_entry": data.get("vix", 15),
                "rsi_at_entry": current_rsi,
                "adx_at_entry": 0,
                "pcr_at_entry": pcr,
                "ml_prob_ce": data.get("ml_stage1_prob_ce", 0.33),
                "ml_prob_pe": data.get("ml_stage1_prob_pe", 0.33),
                "trigger_count": trigger_sum,
                "regime_encoded": {"TRENDING": 0, "RANGEBOUND": 1, "VOLATILE": 2, "ELEVATED": 3}.get(regime, 1),
                "direction_encoded": 1 if direction == "CE" else 0,
                "days_to_expiry": data.get("days_to_expiry", 3),
            }
            try:
                quality_result = ml_quality_fn(quality_features)
                if quality_result.get("win_prob", 0.5) < 0.45:
                    logger.info(
                        f"ML_QUALITY_GATE BLOCKED: {symbol} win_prob={quality_result['win_prob']:.3f} < 0.45"
                    )
                    return None
            except Exception as e:
                logger.warning(f"ML_QUALITY_GATE_ERROR: {symbol} {e}")
                # Quality gate is optional — continue without blocking

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

        # ── Determine trade type from stored value (per-symbol) ──
        trade_type = self._current_trade_type.get(symbol, "FULL")

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

        # Dual mode: override with tighter SL/TP for VOLATILE naked buys
        if self._is_dual_mode_trade:
            cfg_dm = get_config()
            adaptive_sl = cfg_dm.DUAL_MODE_SL_PCT
            adaptive_tp = cfg_dm.DUAL_MODE_TP_PCT
            logger.info(
                f"DUAL_MODE_SL_TP: SL={adaptive_sl:.0%} TP={adaptive_tp:.0%} "
                f"(tighter than normal VOLATILE)"
            )

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
        # Kelly sizing: scale risk by recent performance
        kelly_mult = data.get("kelly_mult", 1.0)
        if kelly_mult != 1.0:
            sizing_max_risk = sizing_max_risk * kelly_mult
            logger.info(f"KELLY_SIZING: risk ₹{sizing['max_risk']:,.0f} × {kelly_mult:.2f} = ₹{sizing_max_risk:,.0f}")
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
            if regime not in tier_targets:
                logger.warning(f"OptionsBuyer: {regime} regime not in DELTA_TARGETS — falling back to TRENDING")
            delta_min, delta_max = tier_targets.get(
                regime, tier_targets["TRENDING"]
            )
            if regime not in self.PREMIUM_SWEET_SPOTS:
                logger.warning(f"OptionsBuyer: {regime} regime not in PREMIUM_SWEET_SPOTS — falling back to TRENDING")
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
                            logger.warning(
                                f"SPREAD_TOO_WIDE: bid={bid} ask={ask} "
                                f"spread={spread_pct:.1f}% > max allowed "
                                f"({best['strike']}{direction})"
                            )
                            continue
                except Exception as e:
                    logger.warning(f"SPREAD_CHECK_FAILED: {best['strike']}{direction} — {e}")

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

        # ── Path 2: Fallback — premium-only selection (batched) ──
        if not instrument_key and self._data_fetcher is not None:
            cand_keys = [c["instrument_key"] for c in candidates]
            cand_quotes = self._data_fetcher.get_live_quotes_batch(cand_keys)
            for c in candidates:
                premium = cand_quotes.get(c["instrument_key"], {}).get("ltp", 0)

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
                next_keys = [c["instrument_key"] for c in next_candidates]
                next_quotes = (
                    self._data_fetcher.get_live_quotes_batch(next_keys)
                    if self._data_fetcher and next_keys else {}
                )
                for c in next_candidates:
                    premium = next_quotes.get(c["instrument_key"], {}).get("ltp", 0)
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

        # ── Dynamic SL/TP by premium level ──
        known_prem = live_premium if live_premium > 0 else effective_max_premium
        adaptive_sl, adaptive_tp = clamp_sl_tp_by_premium(known_prem, _base_adaptive_sl, adaptive_tp)

        # ── Compute actual lot count now that premium is known ──
        actual_qty = self._compute_lots(
            premium=known_prem,
            full_lot=sizing_full_lot,
            sl_pct=adaptive_sl,
            max_risk=sizing_max_risk,
            cb_size_multiplier=min(data.get("cb_size_multiplier", 1.0), data.get("equity_size_multiplier", 1.0)),
        )
        # Reversal sizing: reduce to 0.75× for less-convicted reversal trades
        if self._is_reversal_trade:
            reversal_mult = get_config().REVERSAL_SIZE_MULT
            original_qty = actual_qty
            actual_qty = max(sizing_full_lot, int(actual_qty * reversal_mult / sizing_full_lot) * sizing_full_lot)
            logger.info(
                f"REVERSAL_SIZING: {direction} entry size={reversal_mult:.0%} "
                f"({original_qty}→{actual_qty}) risk=₹{known_prem * actual_qty * adaptive_sl:.0f}"
            )
        # Dual mode sizing: reduce to 0.60× for VOLATILE naked buys
        if self._is_dual_mode_trade:
            dual_mult = get_config().DUAL_MODE_SIZE_MULT
            original_qty = actual_qty
            actual_qty = max(sizing_full_lot, int(actual_qty * dual_mult / sizing_full_lot) * sizing_full_lot)
            logger.info(
                f"DUAL_MODE_SIZING: {direction} entry size={dual_mult:.0%} "
                f"({original_qty}→{actual_qty}) risk=₹{known_prem * actual_qty * adaptive_sl:.0f}"
            )
        if actual_qty <= 0:
            logger.info(
                f"OptionsBuyer: {symbol} premium ₹{live_premium:.0f} — "
                f"cannot size position, skipping"
            )
            return None

        # Track trade count for daily limits
        self._full_trades_today += 1
        if self._current_trade_type.get(symbol) == "NAKED_BUY":
            self._naked_trades_today += 1

        # Confidence based on score_diff + triggers
        confidence = min(0.55 + score_diff * 0.06 + trigger_sum * 0.04, 0.95)

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
                "tp1_pct": adaptive_tp * get_config().PARTIAL_TP1_RATIO if get_config().PARTIAL_EXIT_ENABLED else 0,
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
                "confirmation_triggers": {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "sum": trigger_sum, "threshold": fuzzy_threshold},
                # Fix 4: Individual fuzzy trigger scores
                "trigger_t1": t1,
                "trigger_t2": t2,
                "trigger_t3": t3,
                "trigger_t4": t4,
                "trigger_sum": trigger_sum,
                "trigger_threshold": fuzzy_threshold,
                # Fix 5: Rolling range fields
                "range_high_at_entry": round(range_high, 1),
                "range_low_at_entry": round(range_low, 1),
                "range_width_pct": round(range_width_pct, 2),
                "range_age_minutes": round((datetime.now() - self._range_last_update.get(symbol, datetime.now())).total_seconds() / 60, 0),
                "ml_prob_up": round(ml_prob_up, 3),
                "ml_prob_down": round(ml_prob_down, 3),
                "morning_bias": bias,
                "morning_high": self._morning_high.get(symbol, 0),
                "morning_low": self._morning_low.get(symbol, 0),
                "intraday_rsi": round(current_rsi, 1),
                "intraday_day_open": round(day_open, 1),
                "current_pcr": round(pcr, 2),
                # Fix 2: Rescore fields
                "rescore_weight": self._rescore_weight,
                "daily_bull_score": round(self._daily_scores.get(symbol, (0, 0, 0))[0], 1),
                "daily_bear_score": round(self._daily_scores.get(symbol, (0, 0, 0))[1], 1),
                "intraday_bull_score": round(self._intraday_scores.get(symbol, (0, 0, 0))[0], 1),
                "intraday_bear_score": round(self._intraday_scores.get(symbol, (0, 0, 0))[1], 1),
                # Fix 1: Direction flip fields
                "direction_flipped": self._direction_flipped_today,
                "daily_direction": "CE" if self._daily_scores.get(symbol, (1, 0, 0))[0] > self._daily_scores.get(symbol, (0, 1, 0))[1] else "PE",
                "final_direction": direction,
                # Fix 3: Abort fields
                "abort_stage": self._abort_stage.get(symbol, "NONE"),
                "failed_confirmation_count": self._failed_confirm_count.get(symbol, 0),
                "is_reversal": self._is_reversal_trade,
                "is_dual_mode": self._is_dual_mode_trade,
            },
            notes=(
                f"{'DUAL_MODE ' if self._is_dual_mode_trade else ''}"
                f"{'REVERSAL ' if self._is_reversal_trade else ''}"
                f"{trade_type} {direction} {symbol} {int(strike)} | {bias} | "
                f"Score: bull={bull_score:.1f} bear={bear_score:.1f} diff={score_diff:.1f} | "
                f"Triggers: {triggers} | PCR={pcr:.2f} | Regime={regime} | "
                f"Qty={actual_qty} ({actual_qty // sizing_full_lot}L) | {selection_method}"
            ),
        )

        logger.info(
            f"OptionsBuyer SIGNAL: {trade_type} BUY {option_symbol} | "
            f"direction={direction} conf={confidence:.2f} score_diff={score_diff:.1f} | "
            f"trig_sum={trigger_sum:.1f}/{fuzzy_threshold:.1f} PCR={pcr:.2f} spot={spot:.0f} strike={strike} | "
            f"qty={actual_qty} ({actual_qty // sizing_full_lot}L) premium=₹{live_premium:.1f} {selection_method}"
        )

        return signal

    def should_force_exit(self) -> bool:
        """Check if we should force-exit all option positions."""
        now = datetime.now().time()
        _et_h, _et_m = parse_time_config(self.force_exit_time, 15, 10)
        exit_time = dt_time(_et_h, _et_m)
        return now >= exit_time
