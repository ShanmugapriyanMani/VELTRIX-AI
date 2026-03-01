"""
Circuit Breaker System — Emergency risk controls (V2.2).

Levels:
1. NORMAL  — all trading active
2. WARNING — daily loss >3% OR drawdown >15% → conviction +1.0 (stricter entry)
3. PAUSED  — 4 consecutive losses → pause 1 hour
4. HALTED  — daily loss >5% OR drawdown >18% OR 7 consec losses → no new trades
5. KILLED  — emergency kill switch, cancel all + flatten

Weekly/monthly tiers:
- Weekly loss ≥8% → half lot rest of week
- Monthly loss ≥12% → pause 3 days

Also enforces:
- Max 10 orders/minute (runaway detection → emergency halt)
- Max 2 trades/day, 20 orders/day
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Optional

import yaml
from loguru import logger

from src.config.env_loader import get_config, _env_is_set


class BreakerState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    KILLED = "KILLED"


@dataclass
class BreakerStatus:
    """Current state of all circuit breakers."""

    state: BreakerState = BreakerState.NORMAL
    daily_loss_pct: float = 0.0
    drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    daily_trades: int = 0
    daily_orders: int = 0
    open_positions: int = 0
    pause_until: Optional[datetime] = None
    halt_reason: str = ""
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    weekly_size_reduced: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "daily_loss_pct": round(self.daily_loss_pct, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades,
            "daily_orders": self.daily_orders,
            "open_positions": self.open_positions,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "halt_reason": self.halt_reason,
            "weekly_pnl": round(self.weekly_pnl, 2),
            "monthly_pnl": round(self.monthly_pnl, 2),
            "weekly_size_reduced": self.weekly_size_reduced,
            "can_trade": self.state in (BreakerState.NORMAL, BreakerState.WARNING),
        }


class CircuitBreaker:
    """
    Multi-level circuit breaker system (V2.2).

    Levels:
    1. NORMAL  — all trading active
    2. WARNING — daily loss >3% OR drawdown >15%, conviction +1.0
    3. PAUSED  — 4 consecutive losses, pause 1 hour
    4. HALTED  — daily loss >5% OR drawdown >18%, no new trades
    5. KILLED  — emergency kill switch, cancel all + flatten
    """

    def __init__(self, config_path: str = "config/risk.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cb_cfg = config.get("circuit_breakers", {})

        # Daily loss — two-tier: warning (reduce size) then halt
        daily_cfg = cb_cfg.get("daily_loss", {})
        self.daily_loss_warning = daily_cfg.get("warning_pct", 3.0)
        self.daily_loss_threshold = daily_cfg.get("threshold_pct", 5.0)

        # Drawdown
        dd_cfg = cb_cfg.get("drawdown", {})
        self.drawdown_warning = dd_cfg.get("warning_pct", 15.0)
        self.drawdown_critical = dd_cfg.get("critical_pct", 22.0)

        # Consecutive losses
        cl_cfg = cb_cfg.get("consecutive_losses", {})
        self.pause_threshold = cl_cfg.get("pause_threshold", 6)
        self.halt_threshold = cl_cfg.get("halt_threshold", 10)
        self.pause_duration_min = cl_cfg.get("pause_duration_minutes", 60)

        # Limits
        self.max_daily_trades = cb_cfg.get("max_daily_trades", 20)
        self.max_daily_orders = cb_cfg.get("max_daily_orders", 50)
        self.max_open_positions = cb_cfg.get("max_open_positions", 10)
        self.max_orders_per_minute = cb_cfg.get("max_orders_per_minute", 10)

        # EnvConfig overlay
        cfg = get_config()
        if _env_is_set("MAX_TRADES_PER_DAY"):
            self.max_daily_trades = cfg.MAX_TRADES_PER_DAY

        # Weekly/monthly loss tiers
        wl_cfg = cb_cfg.get("weekly_loss", {})
        self.weekly_loss_warning = wl_cfg.get("warning_pct", 8.0)
        ml_cfg = cb_cfg.get("monthly_loss", {})
        self.monthly_loss_warning = ml_cfg.get("warning_pct", 12.0)
        self.monthly_pause_days = ml_cfg.get("pause_days", 3)

        # Kill switch
        ks_cfg = cb_cfg.get("kill_switch", {})
        self.kill_switch_enabled = ks_cfg.get("enabled", True)

        # State tracking
        self._state = BreakerState.NORMAL
        self._consecutive_losses = 0
        self._daily_trades = 0
        self._daily_orders = 0
        self._pause_until: Optional[datetime] = None
        self._halt_reason = ""
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._last_reset_date: Optional[date] = None
        self._week_start_date: Optional[date] = None
        self._month_start_date: Optional[date] = None
        self._weekly_size_reduced = False
        self._monthly_paused_until: Optional[date] = None
        self._alerts_sent: set[str] = set()

        # Per-minute order tracking (sliding window)
        self._order_timestamps: deque[float] = deque()

    def check(
        self,
        daily_loss_pct: float,
        drawdown_pct: float,
        open_positions: int,
    ) -> BreakerStatus:
        """
        Run all circuit breaker checks.

        Args:
            daily_loss_pct: Today's loss as % of capital (positive = loss)
            drawdown_pct: Current drawdown from peak (positive)
            open_positions: Number of open positions

        Returns:
            BreakerStatus with current state
        """
        self._reset_daily_if_new_day()
        self._reset_weekly_if_new_week()
        self._reset_monthly_if_new_month()

        now = datetime.now()
        today = now.date()

        # ── Monthly pause check ──
        if self._monthly_paused_until and today < self._monthly_paused_until:
            if self._state not in (BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.PAUSED
                self._halt_reason = (
                    f"Monthly loss pause until {self._monthly_paused_until.isoformat()}"
                )
                return self._get_status(daily_loss_pct, drawdown_pct, open_positions)
        elif self._monthly_paused_until and today >= self._monthly_paused_until:
            self._monthly_paused_until = None
            if self._state == BreakerState.PAUSED:
                self._state = BreakerState.NORMAL
                logger.info("CIRCUIT BREAKER: Monthly pause expired, resuming NORMAL")

        # ── Check pause expiry ──
        if self._state == BreakerState.PAUSED and self._pause_until:
            if now >= self._pause_until:
                self._state = BreakerState.NORMAL
                self._pause_until = None
                logger.info("CIRCUIT BREAKER: Pause expired, resuming NORMAL state")

        # If KILLED, stay killed until manual reset
        if self._state == BreakerState.KILLED:
            return self._get_status(daily_loss_pct, drawdown_pct, open_positions)

        # ── Check 1: Daily loss HALT (5%) ──
        if daily_loss_pct >= self.daily_loss_threshold:
            if self._state != BreakerState.HALTED:
                self._state = BreakerState.HALTED
                self._halt_reason = f"Daily loss {daily_loss_pct:.1f}% >= {self.daily_loss_threshold}%"
                logger.critical(f"CIRCUIT BREAKER: HALTED — {self._halt_reason}")

        # ── Check 2: Drawdown HALT (22%) ──
        elif drawdown_pct >= self.drawdown_critical:
            if self._state not in (BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.HALTED
                self._halt_reason = f"Drawdown {drawdown_pct:.1f}% >= {self.drawdown_critical}%"
                logger.critical(f"CIRCUIT BREAKER: HALTED — {self._halt_reason}")

        # ── Check 3: Daily loss WARNING (3%) or Drawdown WARNING (15%) ──
        elif daily_loss_pct >= self.daily_loss_warning or drawdown_pct >= self.drawdown_warning:
            if self._state == BreakerState.NORMAL:
                self._state = BreakerState.WARNING
                reasons = []
                if daily_loss_pct >= self.daily_loss_warning:
                    reasons.append(f"Daily loss {daily_loss_pct:.1f}% >= {self.daily_loss_warning}%")
                if drawdown_pct >= self.drawdown_warning:
                    reasons.append(f"Drawdown {drawdown_pct:.1f}% >= {self.drawdown_warning}%")
                logger.warning(
                    f"CIRCUIT BREAKER: WARNING — {'; '.join(reasons)}. Conviction +1.0 required."
                )

        # ── Check 4: Consecutive losses ──
        elif self._consecutive_losses >= self.halt_threshold:
            if self._state not in (BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.HALTED
                self._halt_reason = f"{self._consecutive_losses} consecutive losses"
                logger.critical(f"CIRCUIT BREAKER: HALTED — {self._halt_reason}")

        elif self._consecutive_losses >= self.pause_threshold:
            if self._state not in (BreakerState.PAUSED, BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.PAUSED
                self._pause_until = now + timedelta(minutes=self.pause_duration_min)
                logger.warning(
                    f"CIRCUIT BREAKER: PAUSED for {self.pause_duration_min}min — "
                    f"{self._consecutive_losses} consecutive losses"
                )

        # ── Check 5: Daily trade limit ──
        elif self._daily_trades >= self.max_daily_trades:
            if self._state not in (BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.HALTED
                self._halt_reason = f"Max daily trades ({self.max_daily_trades}) reached"
                logger.warning(f"CIRCUIT BREAKER: HALTED — {self._halt_reason}")

        else:
            # Clear warnings if conditions improve
            if self._state == BreakerState.WARNING:
                if daily_loss_pct < self.daily_loss_warning and drawdown_pct < self.drawdown_warning:
                    self._state = BreakerState.NORMAL
                    logger.info("CIRCUIT BREAKER: Conditions improved, back to NORMAL")

        return self._get_status(daily_loss_pct, drawdown_pct, open_positions)

    def record_trade(self, pnl: float, capital: float = 0) -> None:
        """Record a trade result for consecutive loss and weekly/monthly tracking."""
        if capital <= 0:
            capital = get_config().TRADING_CAPITAL
        self._daily_trades += 1

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on win

        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl

        # ── Weekly loss tier: ≥8% → half lot rest of week ──
        if capital > 0 and not self._weekly_size_reduced:
            weekly_loss_pct = abs(min(0, self._weekly_pnl)) / capital * 100
            if weekly_loss_pct >= self.weekly_loss_warning:
                self._weekly_size_reduced = True
                logger.warning(
                    f"CIRCUIT BREAKER: Weekly loss {weekly_loss_pct:.1f}% >= "
                    f"{self.weekly_loss_warning}% — half lot rest of week"
                )

        # ── Monthly loss tier: ≥12% → pause N days ──
        if capital > 0 and self._monthly_paused_until is None:
            monthly_loss_pct = abs(min(0, self._monthly_pnl)) / capital * 100
            if monthly_loss_pct >= self.monthly_loss_warning:
                self._monthly_paused_until = date.today() + timedelta(days=self.monthly_pause_days)
                self._state = BreakerState.PAUSED
                self._halt_reason = (
                    f"Monthly loss {monthly_loss_pct:.1f}% >= {self.monthly_loss_warning}% "
                    f"— paused until {self._monthly_paused_until.isoformat()}"
                )
                logger.critical(f"CIRCUIT BREAKER: {self._halt_reason}")

    def record_order(self) -> bool:
        """
        Record an order attempt. Returns False if limit reached.

        Also checks per-minute rate: >10 orders/minute triggers emergency halt.
        """
        now_ts = datetime.now().timestamp()
        self._daily_orders += 1

        # Track per-minute sliding window
        self._order_timestamps.append(now_ts)
        cutoff = now_ts - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()

        # Runaway detection: too many orders in 1 minute
        if len(self._order_timestamps) > self.max_orders_per_minute:
            if self._state not in (BreakerState.HALTED, BreakerState.KILLED):
                self._state = BreakerState.HALTED
                self._halt_reason = (
                    f"Runaway detection: {len(self._order_timestamps)} orders in 1 minute "
                    f"> limit {self.max_orders_per_minute}"
                )
                logger.critical(f"CIRCUIT BREAKER: HALTED — {self._halt_reason}")
            return False

        if self._daily_orders > self.max_daily_orders:
            return False

        return True

    def can_trade(self) -> bool:
        """Check if trading is currently allowed."""
        if self._state in (BreakerState.HALTED, BreakerState.KILLED):
            return False
        if self._state == BreakerState.PAUSED:
            if self._pause_until and datetime.now() < self._pause_until:
                return False
        if self._daily_trades >= self.max_daily_trades:
            return False
        if self._daily_orders >= self.max_daily_orders:
            return False
        return True

    def get_size_multiplier(self) -> float:
        """Get position size multiplier based on breaker state.

        V2.2: WARNING no longer reduces size — it adds conviction +1.0 instead.
        Use get_conviction_boost() for the conviction penalty.
        """
        if self._state in (BreakerState.PAUSED, BreakerState.HALTED, BreakerState.KILLED):
            return 0.0
        # Weekly loss tier: half lot rest of week (still applies)
        if self._weekly_size_reduced:
            return 0.5
        return 1.0

    def get_conviction_boost(self) -> float:
        """Get conviction requirement boost based on breaker state.

        V2.2: WARNING state adds +1.0 to conviction threshold instead of
        reducing position size. This makes the system more selective
        rather than trading smaller.
        """
        if self._state == BreakerState.WARNING:
            return 1.0
        return 0.0

    def activate_kill_switch(self) -> dict[str, Any]:
        """
        Emergency kill switch — cancel all orders and flatten all positions.

        Returns instructions for the execution engine.
        """
        if not self.kill_switch_enabled:
            logger.warning("Kill switch is disabled in config")
            return {"action": "none", "reason": "kill switch disabled"}

        self._state = BreakerState.KILLED
        self._halt_reason = "KILL SWITCH ACTIVATED"

        logger.critical(
            "KILL SWITCH ACTIVATED — Cancelling all orders and flattening positions"
        )

        return {
            "action": "cancel_all_flatten",
            "cancel_pending": True,
            "flatten_positions": True,
            "reason": "Kill switch activated",
            "timestamp": datetime.now().isoformat(),
        }

    def reset(self, force: bool = False) -> None:
        """Reset circuit breaker to normal state."""
        if self._state == BreakerState.KILLED and not force:
            logger.warning("Cannot reset KILLED state without force=True")
            return

        prev_state = self._state
        self._state = BreakerState.NORMAL
        self._halt_reason = ""
        self._pause_until = None
        self._alerts_sent.clear()

        logger.info(f"CIRCUIT BREAKER: Reset from {prev_state.value} to NORMAL")

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self._daily_trades = 0
        self._daily_orders = 0
        self._daily_pnl = 0.0
        self._last_reset_date = date.today()
        self._order_timestamps.clear()

        # Reset halt if not killed
        if self._state in (BreakerState.HALTED, BreakerState.PAUSED):
            self._state = BreakerState.NORMAL
            self._halt_reason = ""
            self._pause_until = None

        self._alerts_sent.clear()
        logger.info("CIRCUIT BREAKER: Daily counters reset")

    def _reset_daily_if_new_day(self) -> None:
        """Auto-reset daily counters on new trading day."""
        today = date.today()
        if self._last_reset_date != today:
            self.reset_daily()

    def _reset_weekly_if_new_week(self) -> None:
        """Auto-reset weekly counters on new week (Monday)."""
        today = date.today()
        if self._week_start_date is None:
            self._week_start_date = today - timedelta(days=today.weekday())
        week_start = today - timedelta(days=today.weekday())
        if week_start > self._week_start_date:
            self._weekly_pnl = 0.0
            self._weekly_size_reduced = False
            self._week_start_date = week_start
            logger.info("CIRCUIT BREAKER: Weekly counters reset")

    def _reset_monthly_if_new_month(self) -> None:
        """Auto-reset monthly counters on new month."""
        today = date.today()
        if self._month_start_date is None:
            self._month_start_date = today.replace(day=1)
        month_start = today.replace(day=1)
        if month_start > self._month_start_date:
            self._monthly_pnl = 0.0
            self._month_start_date = month_start
            # Don't clear monthly_paused_until — let it expire naturally
            logger.info("CIRCUIT BREAKER: Monthly counters reset")

    def _get_status(
        self,
        daily_loss_pct: float,
        drawdown_pct: float,
        open_positions: int,
    ) -> BreakerStatus:
        return BreakerStatus(
            state=self._state,
            daily_loss_pct=daily_loss_pct,
            drawdown_pct=drawdown_pct,
            consecutive_losses=self._consecutive_losses,
            daily_trades=self._daily_trades,
            daily_orders=self._daily_orders,
            open_positions=open_positions,
            pause_until=self._pause_until,
            halt_reason=self._halt_reason,
            weekly_pnl=self._weekly_pnl,
            monthly_pnl=self._monthly_pnl,
            weekly_size_reduced=self._weekly_size_reduced,
        )

    @property
    def status(self) -> BreakerStatus:
        return BreakerStatus(
            state=self._state,
            consecutive_losses=self._consecutive_losses,
            daily_trades=self._daily_trades,
            daily_orders=self._daily_orders,
            halt_reason=self._halt_reason,
            pause_until=self._pause_until,
            weekly_pnl=self._weekly_pnl,
            monthly_pnl=self._monthly_pnl,
            weekly_size_reduced=self._weekly_size_reduced,
        )
