"""
Circuit Breaker System — Simplified V3.

Only 2 rules:
1. 2 consecutive SL exits → halt rest of day
2. Daily loss > ₹20,000 → halt rest of day

State machine:
  NORMAL → trading allowed
  HALTED → no new entries today (exits/monitoring still run)

Daily reset every morning clears everything — no weekly/monthly carry-over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from src.config.env_loader import get_config


class BreakerState(str, Enum):
    NORMAL = "NORMAL"
    HALTED = "HALTED"


@dataclass
class BreakerStatus:
    """Current state of circuit breaker."""

    state: BreakerState = BreakerState.NORMAL
    consecutive_sl: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    halt_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_sl": self.consecutive_sl,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "halt_reason": self.halt_reason,
            "can_trade": self.state == BreakerState.NORMAL,
        }


class CircuitBreaker:
    """
    Simplified circuit breaker — 2 rules only.

    Rule 1: 2 consecutive SL exits → HALTED rest of day
    Rule 2: Daily loss > ₹20,000 → HALTED rest of day

    Daily reset every morning clears all state.
    """

    def __init__(self, config_path: str = "config/risk.yaml"):
        cfg = get_config()
        self._daily_loss_limit = cfg.DAILY_LOSS_HALT  # ₹20,000 from .env.plus
        self._consec_sl_limit = 2

        # State
        self._state = BreakerState.NORMAL
        self._halt_reason = ""
        self._consecutive_sl = 0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date: Optional[date] = None
        self._alert_fn: Optional[Callable[[str], None]] = None

        # Equity curve sizing (carries across days — never resets)
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

        # Persistence (daily only — for crash recovery within same day)
        self._state_file = Path(__file__).resolve().parent.parent.parent / "data" / "circuit_breaker_state.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def set_alert_fn(self, fn: Callable[[str], None]) -> None:
        """Inject Telegram alert function."""
        self._alert_fn = fn

    def _send_alert(self, message: str) -> None:
        """Send alert via injected function if available."""
        if self._alert_fn:
            try:
                self._alert_fn(message)
            except Exception:
                pass

    def check(
        self,
        daily_loss_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        open_positions: int = 0,
    ) -> BreakerStatus:
        """
        Run circuit breaker checks. Accepts legacy args for compatibility.

        The actual checks happen in record_trade() — this just ensures
        daily reset and returns current status.
        """
        self._reset_daily_if_new_day()
        return self._get_status()

    def record_trade(self, pnl: float, capital: float = 0) -> None:
        """Record a trade result. Checks both rules after each trade."""
        self._daily_trades += 1
        self._daily_pnl += pnl

        # Track consecutive SL: loss = SL hit, win = reset
        if pnl < 0:
            self._consecutive_sl += 1
        else:
            self._consecutive_sl = 0

        # Rule 1: 2 consecutive SL hits → halt
        if self._state == BreakerState.NORMAL and self._consecutive_sl >= self._consec_sl_limit:
            self._state = BreakerState.HALTED
            self._halt_reason = "consecutive_losses"
            logger.critical(
                f"CB_HALT: {self._consec_sl_limit} consecutive SL hits. "
                f"Halting rest of day."
            )
            self._send_alert(
                f"\U0001f6d1 {self._consec_sl_limit} consecutive losses.\n"
                f"Halting trading for today.\n"
                f"Resume tomorrow."
            )

        # Rule 2: Daily loss > limit → halt
        daily_loss = abs(min(0, self._daily_pnl))
        if self._state == BreakerState.NORMAL and daily_loss >= self._daily_loss_limit:
            self._state = BreakerState.HALTED
            self._halt_reason = "daily_loss"
            logger.critical(
                f"CB_HALT: Daily loss \u20b9{daily_loss:,.0f} "
                f">= \u20b9{self._daily_loss_limit:,.0f}. Halting rest of day."
            )
            self._send_alert(
                f"\U0001f6d1 Daily loss \u20b9{daily_loss:,.0f}.\n"
                f"Halting trading for today.\n"
                f"Resume tomorrow."
            )

        self.save_state()

    def record_order(self) -> bool:
        """Record an order attempt. Returns True (always allowed — no order rate limits)."""
        return True

    def can_trade(self) -> bool:
        """Check if trading is currently allowed."""
        self._reset_daily_if_new_day()
        return self._state == BreakerState.NORMAL

    def get_size_multiplier(self) -> float:
        """Loss-based position size reduction.

        0 consecutive SL → 1.0 (full size)
        1 consecutive SL → 0.75 (75% size)
        2+ consecutive SL → 0.50 (but halt fires at 2, so effectively 0.0)
        Resets to 1.0 automatically when consecutive SL resets on any win.
        """
        if self._state == BreakerState.HALTED:
            return 0.0
        if self._consecutive_sl == 0:
            return 1.0
        elif self._consecutive_sl == 1:
            return 0.75
        else:
            return 0.50

    def update_equity(self, current_equity: float) -> None:
        """Update current equity for equity curve sizing."""
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    @property
    def equity_size_multiplier(self) -> float:
        """Equity curve position size reduction.

        < 5% from peak  → 1.0 (full size)
        5-10% from peak  → 0.85
        10-15% from peak → 0.70
        15%+ from peak   → 0.50

        Never resets — peak equity carries across days.
        """
        if self._peak_equity <= 0:
            return 1.0
        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
        if drawdown < 0.05:
            return 1.0
        elif drawdown < 0.10:
            return 0.85
        elif drawdown < 0.15:
            return 0.70
        else:
            return 0.50

    def get_conviction_boost(self) -> float:
        """No conviction boost — removed with WARNING state."""
        return 0.0

    def activate_kill_switch(self) -> dict[str, Any]:
        """Emergency kill switch — halt and flatten."""
        self._state = BreakerState.HALTED
        self._halt_reason = "KILL SWITCH ACTIVATED"
        logger.critical("KILL SWITCH ACTIVATED — Halting all trading")
        self._send_alert("\U0001f480 Kill switch activated. Trading halted. Manual reset required.")
        return {
            "action": "cancel_all_flatten",
            "cancel_pending": True,
            "flatten_positions": True,
            "reason": "Kill switch activated",
            "timestamp": datetime.now().isoformat(),
        }

    def reset(self, force: bool = False) -> None:
        """Reset circuit breaker to normal state."""
        prev_state = self._state
        self._state = BreakerState.NORMAL
        self._halt_reason = ""
        logger.info(f"CIRCUIT BREAKER: Reset from {prev_state.value} to NORMAL")

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day. Always resets — no carry-over.

        Note: peak_equity and current_equity are NOT reset — they carry across days.
        """
        self._state = BreakerState.NORMAL
        self._halt_reason = ""
        self._consecutive_sl = 0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date = date.today()
        self.save_state()
        logger.info("CB_RESET: Daily reset — all clear")

    def _reset_daily_if_new_day(self) -> None:
        """Auto-reset daily counters on new trading day."""
        today = date.today()
        if self._last_reset_date != today:
            self.reset_daily()

    def _get_status(self) -> BreakerStatus:
        return BreakerStatus(
            state=self._state,
            consecutive_sl=self._consecutive_sl,
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            halt_reason=self._halt_reason,
        )

    @property
    def status(self) -> BreakerStatus:
        return self._get_status()

    # ── State persistence (same-day crash recovery only) ──

    def _load_state(self) -> None:
        """Load state from disk on startup (same day only)."""
        try:
            data = json.loads(self._state_file.read_text())
            saved_date = date.fromisoformat(data.get("date", "2000-01-01"))
            # Always restore peak equity (carries across days)
            self._peak_equity = data.get("peak_equity", 0.0)
            self._current_equity = data.get("current_equity", 0.0)

            if saved_date == date.today():
                self._consecutive_sl = data.get("consecutive_sl", 0)
                self._daily_pnl = data.get("daily_pnl", 0.0)
                self._daily_trades = data.get("daily_trades", 0)
                self._last_reset_date = saved_date
                # Restore halt state if was halted
                if data.get("state") == "HALTED":
                    self._state = BreakerState.HALTED
                    self._halt_reason = data.get("halt_reason", "")
                logger.info(f"CIRCUIT BREAKER: State loaded from {self._state_file}")
            else:
                # Different day — fresh start (but peak equity preserved above)
                self._last_reset_date = date.today()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"CIRCUIT BREAKER: Could not load state: {e}")

    def save_state(self) -> None:
        """Persist state to disk for crash recovery."""
        try:
            data = {
                "date": date.today().isoformat(),
                "state": self._state.value,
                "halt_reason": self._halt_reason,
                "consecutive_sl": self._consecutive_sl,
                "daily_pnl": round(self._daily_pnl, 2),
                "daily_trades": self._daily_trades,
                "peak_equity": round(self._peak_equity, 2),
                "current_equity": round(self._current_equity, 2),
            }
            self._state_file.write_text(json.dumps(data, indent=2))
        except PermissionError as e:
            logger.warning(f"CIRCUIT BREAKER: Read-only filesystem, cannot save state: {e}")
        except Exception as e:
            logger.warning(f"CIRCUIT BREAKER: Could not save state: {e}")
