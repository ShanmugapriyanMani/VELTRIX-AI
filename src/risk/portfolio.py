"""
Portfolio Manager — Position tracking, P&L, correlation, VaR, drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    instrument_key: str = ""
    side: str = "BUY"
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    strategy: str = ""
    sector: str = ""
    entry_time: datetime = field(default_factory=datetime.now)
    trade_id: str = ""
    order_id: str = ""
    tp_reduced: bool = False
    original_quantity: int = 0
    partial_exit_done: bool = False
    # Momentum decay exit metadata
    entry_score_diff: float = 0.0
    entry_rsi: float = 50.0
    peak_score_diff: float = 0.0
    peak_rsi: float = 50.0

    @property
    def value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.quantity
        return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100

    @property
    def hold_hours(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "instrument_key": self.instrument_key,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "value": round(self.value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy": self.strategy,
            "sector": self.sector,
            "entry_time": self.entry_time.isoformat(),
            "hold_hours": round(self.hold_hours, 1),
            "trade_id": self.trade_id,
            "original_quantity": self.original_quantity,
            "partial_exit_done": self.partial_exit_done,
        }


@dataclass
class IronCondorPosition:
    """Represents an open Iron Condor position (4 legs)."""

    position_id: str = ""
    entry_time: datetime = field(default_factory=datetime.now)
    regime: str = "RANGEBOUND"
    spot_at_entry: float = 0.0
    quantity: int = 0
    lots: int = 0

    # Call side (Bear Call Spread): SELL near-OTM CE + BUY far-OTM CE
    sell_ce_strike: float = 0.0
    sell_ce_instrument_key: str = ""
    sell_ce_premium: float = 0.0
    buy_ce_strike: float = 0.0
    buy_ce_instrument_key: str = ""
    buy_ce_premium: float = 0.0

    # Put side (Bull Put Spread): SELL near-OTM PE + BUY far-OTM PE
    sell_pe_strike: float = 0.0
    sell_pe_instrument_key: str = ""
    sell_pe_premium: float = 0.0
    buy_pe_strike: float = 0.0
    buy_pe_instrument_key: str = ""
    buy_pe_premium: float = 0.0

    # Economics
    net_credit: float = 0.0
    spread_width: int = 200
    max_profit: float = 0.0
    max_loss: float = 0.0
    tp_threshold: float = 0.0
    sl_threshold: float = 0.0

    # Current state
    current_pnl: float = 0.0
    status: str = "open"
    exit_reason: str = ""
    exit_time: Optional[datetime] = None
    expiry_type: str = ""
    trade_type: str = "IRON_CONDOR"

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "entry_time": self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else str(self.entry_time),
            "regime": self.regime,
            "spot_at_entry": self.spot_at_entry,
            "quantity": self.quantity,
            "lots": self.lots,
            "sell_ce_strike": self.sell_ce_strike,
            "sell_ce_premium": self.sell_ce_premium,
            "buy_ce_strike": self.buy_ce_strike,
            "buy_ce_premium": self.buy_ce_premium,
            "sell_pe_strike": self.sell_pe_strike,
            "sell_pe_premium": self.sell_pe_premium,
            "buy_pe_strike": self.buy_pe_strike,
            "buy_pe_premium": self.buy_pe_premium,
            "net_credit": self.net_credit,
            "spread_width": self.spread_width,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "tp_threshold": self.tp_threshold,
            "sl_threshold": self.sl_threshold,
            "current_pnl": self.current_pnl,
            "status": self.status,
            "exit_reason": self.exit_reason,
            "expiry_type": self.expiry_type,
            "trade_type": self.trade_type,
        }


def compute_kelly_fraction(
    pnl_list: list[float],
    window: int = 20,
    min_trades: int = 10,
    min_mult: float = 0.50,
    max_mult: float = 1.50,
) -> float:
    """Compute Half-Kelly sizing multiplier from recent trade PnLs.

    Returns a multiplier (0.5× to 1.5×) to scale RISK_PER_TRADE.
    Uses rolling window of completed trades. If insufficient data, returns 1.0.
    """
    if len(pnl_list) < min_trades:
        return 1.0

    recent = pnl_list[-window:]
    wins = [p for p in recent if p > 0]
    losses = [p for p in recent if p <= 0]

    if not wins or not losses:
        return 1.0

    win_rate = len(wins) / len(recent)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))

    if avg_loss == 0:
        return 1.0

    payoff_ratio = avg_win / avg_loss

    # Full Kelly: f* = p - q/b  where p=win_rate, q=1-p, b=payoff_ratio
    kelly = win_rate - (1 - win_rate) / payoff_ratio

    # Half-Kelly for safety
    half_kelly = kelly / 2

    # Normalize: baseline half-Kelly ≈ 0.30 maps to 1.0×
    multiplier = half_kelly / 0.30

    return max(min_mult, min(max_mult, multiplier))


class PortfolioManager:
    """
    Tracks all positions, calculates P&L, exposure, correlation, VaR, drawdown.
    """

    def __init__(self, config_path: str = "config/risk.yaml", initial_capital: float = 500000):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        port_cfg = config.get("portfolio", {})
        self.max_correlation = port_cfg.get("max_correlation", 0.7)
        self.correlation_lookback = port_cfg.get("correlation_lookback", 60)
        self.var_confidence = port_cfg.get("var_confidence", 0.95)
        self.var_method = port_cfg.get("var_method", "historical")
        self.max_var_pct = port_cfg.get("max_var_pct", 3.0)
        self.beta_target = port_cfg.get("beta_target", 1.0)
        self.beta_tolerance = port_cfg.get("beta_tolerance", 0.3)

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.ic_positions: dict[str, IronCondorPosition] = {}
        self.closed_trades: list[dict[str, Any]] = []
        self.daily_pnl_history: list[dict] = []

        # High water mark for drawdown
        self._peak_value = initial_capital

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day."""
        self.closed_trades = []
        self.daily_pnl_history = []
        self.ic_positions = {}
        self._peak_value = max(self._peak_value, self.cash)
        logger.info("PORTFOLIO_RESET: daily state cleared")

    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        position.original_quantity = position.quantity
        self.positions[position.symbol] = position
        self.cash -= position.cost_basis
        logger.info(
            f"Position opened: {position.side} {position.quantity} {position.symbol} "
            f"@ ₹{position.entry_price:.2f} (₹{position.cost_basis:.2f})"
        )

    def restore_position(self, trade: dict) -> None:
        """Restore a position from DB trade record (crash recovery).

        Does NOT deduct cash — the original add_position already did that
        before the crash. We only re-create the Position object in memory.
        """
        entry_price = trade.get("fill_price") or trade.get("price", 0)
        entry_time_str = trade.get("entry_time", "")
        try:
            entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else datetime.now()
        except (ValueError, TypeError):
            entry_time = datetime.now()

        pos = Position(
            symbol=trade.get("symbol", ""),
            instrument_key=trade.get("instrument_key", ""),
            side=trade.get("side", "BUY"),
            quantity=int(trade.get("fill_quantity") or trade.get("quantity", 0)),
            entry_price=float(entry_price),
            current_price=float(entry_price),  # Will be updated by LTP feed
            stop_loss=float(trade.get("stop_loss", 0)),
            take_profit=float(trade.get("take_profit", 0)),
            strategy=trade.get("strategy", ""),
            sector=trade.get("sector", ""),
            entry_time=entry_time,
            trade_id=trade.get("trade_id", ""),
        )
        pos.original_quantity = pos.quantity
        self.positions[pos.symbol] = pos
        # Deduct cash for the restored position
        self.cash -= pos.cost_basis
        logger.info(
            f"CRASH_RECOVERY: restored {pos.symbol} {pos.side} {pos.quantity} "
            f"@ ₹{pos.entry_price:.2f} SL=₹{pos.stop_loss:.0f} TP=₹{pos.take_profit:.0f}"
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "",
        charges: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """Close a position and record the trade."""
        if symbol not in self.positions:
            logger.warning(f"No position for {symbol} to close")
            return None

        pos = self.positions[symbol]
        if pos.side == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl -= charges

        trade_record = {
            "symbol": symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            # DB-compatible aliases
            "price": pos.entry_price,
            "fill_price": exit_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "instrument_key": pos.instrument_key,
            "order_id": pos.order_id,
            "hold_duration_hours": round(pos.hold_hours, 1),
            "notes": reason,
            "status": "closed",
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / pos.cost_basis * 100, 2) if pos.cost_basis > 0 else 0,
            "charges": charges,
            "total_charges": charges,
            "strategy": pos.strategy,
            "sector": pos.sector,
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "hold_hours": round(pos.hold_hours, 1),
            "reason": reason,
            "trade_id": pos.trade_id,
        }

        self.closed_trades.append(trade_record)
        self.cash += exit_price * pos.quantity
        del self.positions[symbol]

        logger.info(
            f"Position closed: {symbol} PnL=₹{pnl:.2f} ({trade_record['pnl_pct']:.1f}%) "
            f"Reason: {reason}"
        )
        return trade_record

    def partial_close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_qty: int,
        reason: str = "",
    ) -> Optional[dict[str, Any]]:
        """Close part of a position. Keeps position open with reduced quantity."""
        if symbol not in self.positions:
            logger.warning(f"No position for {symbol} to partial close")
            return None

        pos = self.positions[symbol]
        if exit_qty >= pos.quantity:
            return self.close_position(symbol, exit_price, reason)

        if pos.side == "BUY":
            pnl = (exit_price - pos.entry_price) * exit_qty
        else:
            pnl = (pos.entry_price - exit_price) * exit_qty

        original_cost = pos.entry_price * exit_qty
        trade_record = {
            "symbol": symbol,
            "side": pos.side,
            "quantity": exit_qty,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            # DB-compatible aliases
            "price": pos.entry_price,
            "fill_price": exit_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "instrument_key": pos.instrument_key,
            "order_id": pos.order_id,
            "hold_duration_hours": round(pos.hold_hours, 1),
            "notes": reason,
            "status": "closed",
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / original_cost * 100, 2) if original_cost > 0 else 0,
            "charges": 0,
            "total_charges": 0,
            "strategy": pos.strategy,
            "sector": pos.sector,
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "hold_hours": round(pos.hold_hours, 1),
            "reason": reason,
            "trade_id": pos.trade_id,
        }

        pos.quantity -= exit_qty
        pos.partial_exit_done = True
        pos.stop_loss = pos.entry_price  # Move SL to breakeven
        pos.take_profit = 0  # Remove TP — runner trails only
        self.cash += exit_price * exit_qty
        self.closed_trades.append(trade_record)

        logger.info(
            f"PARTIAL EXIT: {symbol} {exit_qty}qty @ ₹{exit_price:.2f} "
            f"PnL=₹{pnl:.2f} | Runner: {pos.quantity}qty SL→breakeven ₹{pos.entry_price:.2f}"
        )
        return trade_record

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    # ─────────────────────────────────────────
    # Iron Condor Position Management
    # ─────────────────────────────────────────

    def open_ic_position(self, ic_pos: IronCondorPosition) -> None:
        """Register an open IC position."""
        self.ic_positions[ic_pos.position_id] = ic_pos
        logger.info(
            f"IC OPENED: {ic_pos.position_id} | "
            f"CE {ic_pos.sell_ce_strike}/{ic_pos.buy_ce_strike} "
            f"PE {ic_pos.sell_pe_strike}/{ic_pos.buy_pe_strike} | "
            f"credit=₹{ic_pos.net_credit:.0f} qty={ic_pos.quantity}"
        )

    def close_ic_position(
        self,
        position_id: str,
        exit_reason: str,
        pnl: float,
        charges: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """Close an IC position and return trade record."""
        if position_id not in self.ic_positions:
            logger.warning(f"No IC position {position_id} to close")
            return None

        ic = self.ic_positions[position_id]
        ic.status = "closed"
        ic.exit_reason = exit_reason
        ic.exit_time = datetime.now()
        ic.current_pnl = pnl - charges

        trade_record = ic.to_dict()
        trade_record["pnl"] = round(ic.current_pnl, 2)
        trade_record["charges"] = round(charges, 2)
        trade_record["exit_reason"] = exit_reason
        trade_record["exit_time"] = datetime.now().isoformat()

        self.closed_trades.append(trade_record)
        self.cash += ic.current_pnl
        del self.ic_positions[position_id]

        logger.info(
            f"IC CLOSED: {position_id} | reason={exit_reason} | "
            f"P&L=₹{ic.current_pnl:,.0f}"
        )
        return trade_record

    def has_ic_position(self) -> bool:
        """Check if any IC position is open."""
        return len(self.ic_positions) > 0

    def get_ic_pnl(
        self,
        position_id: str,
        ltp_sell_ce: float,
        ltp_buy_ce: float,
        ltp_sell_pe: float,
        ltp_buy_pe: float,
    ) -> float:
        """Calculate current unrealized P&L for an IC position.

        IC P&L = (credit received - cost to close) * quantity
        Cost to close = (sell_ce_ltp - buy_ce_ltp) + (sell_pe_ltp - buy_pe_ltp)
        """
        if position_id not in self.ic_positions:
            return 0.0

        ic = self.ic_positions[position_id]
        close_cost_ce = ltp_sell_ce - ltp_buy_ce
        close_cost_pe = ltp_sell_pe - ltp_buy_pe
        total_close_cost = close_cost_ce + close_cost_pe
        pnl = (ic.net_credit - total_close_cost) * ic.quantity
        ic.current_pnl = pnl
        return pnl

    def check_ic_stops(
        self,
        ltp_dict: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Check IC positions against SL/TP thresholds.

        Args:
            ltp_dict: {instrument_key: ltp} for all 4 legs
        """
        triggers = []
        for pos_id, ic in list(self.ic_positions.items()):
            ltp_sell_ce = ltp_dict.get(ic.sell_ce_instrument_key, 0)
            ltp_buy_ce = ltp_dict.get(ic.buy_ce_instrument_key, 0)
            ltp_sell_pe = ltp_dict.get(ic.sell_pe_instrument_key, 0)
            ltp_buy_pe = ltp_dict.get(ic.buy_pe_instrument_key, 0)

            if any(v <= 0 for v in [ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe]):
                zero_legs = [k for k, v in {
                    "sell_ce": ltp_sell_ce, "buy_ce": ltp_buy_ce,
                    "sell_pe": ltp_sell_pe, "buy_pe": ltp_buy_pe,
                }.items() if v <= 0]
                logger.warning(
                    f"IC_SL_SKIP: LTP=0 for {pos_id} legs={zero_legs} — "
                    f"skipping stop check (stale data)"
                )
                continue

            pnl = self.get_ic_pnl(pos_id, ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe)

            if pnl <= ic.sl_threshold:
                triggers.append({
                    "position_id": pos_id,
                    "type": "ic_stop_loss",
                    "pnl": pnl,
                    "threshold": ic.sl_threshold,
                })
            elif pnl >= ic.tp_threshold:
                triggers.append({
                    "position_id": pos_id,
                    "type": "ic_take_profit",
                    "pnl": pnl,
                    "threshold": ic.tp_threshold,
                })

        return triggers

    # ─────────────────────────────────────────
    # Portfolio Metrics
    # ─────────────────────────────────────────

    @property
    def total_value(self) -> float:
        positions_value = sum(p.value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def invested_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        if self.total_value == 0:
            return 0
        return self.invested_value / self.total_value * 100

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(t["pnl"] for t in self.closed_trades)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0
        return self.total_pnl / self.initial_capital * 100

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak portfolio value."""
        current = self.total_value
        self._peak_value = max(self._peak_value, current)
        if self._peak_value == 0:
            return 0
        return (self._peak_value - current) / self._peak_value * 100

    def get_day_pnl(self) -> float:
        """P&L for today's trades."""
        today = date.today().isoformat()
        day_pnl = sum(
            t["pnl"] for t in self.closed_trades
            if t["exit_time"].startswith(today)
        )
        day_pnl += self.unrealized_pnl
        return round(day_pnl, 2)

    def get_sector_exposure(self) -> dict[str, float]:
        """Get exposure breakdown by sector."""
        sectors: dict[str, float] = {}
        for pos in self.positions.values():
            sector = pos.sector or "UNKNOWN"
            sectors[sector] = sectors.get(sector, 0) + pos.value
        return sectors

    def get_strategy_exposure(self) -> dict[str, float]:
        """Get exposure breakdown by strategy."""
        strategies: dict[str, float] = {}
        for pos in self.positions.values():
            strat = pos.strategy or "UNKNOWN"
            strategies[strat] = strategies.get(strat, 0) + pos.value
        return strategies

    # ─────────────────────────────────────────
    # Risk Metrics
    # ─────────────────────────────────────────

    def calculate_var(
        self,
        returns_history: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Calculate Value at Risk (Historical method).

        Returns VaR as a positive number (potential loss).
        """
        if returns_history is None or returns_history.empty:
            return 0.0

        portfolio_returns = returns_history.mean(axis=1) if returns_history.ndim > 1 else returns_history
        var_pct = abs(float(np.percentile(portfolio_returns, (1 - self.var_confidence) * 100)))
        var_value = var_pct * self.total_value

        return round(var_value, 2)

    def check_correlation(
        self,
        symbol: str,
        returns_history: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Check correlation of a new symbol with existing positions.

        Returns correlation with each existing position.
        """
        correlations = {}

        if symbol not in returns_history.columns:
            return correlations

        for pos_symbol in self.positions:
            if pos_symbol in returns_history.columns:
                corr = returns_history[symbol].corr(returns_history[pos_symbol])
                if not np.isnan(corr):
                    correlations[pos_symbol] = round(float(corr), 3)

        return correlations

    def is_highly_correlated(
        self,
        symbol: str,
        returns_history: pd.DataFrame,
    ) -> bool:
        """Check if adding this symbol would create too-high correlation."""
        corrs = self.check_correlation(symbol, returns_history)
        return any(abs(c) > self.max_correlation for c in corrs.values())

    def get_positions_df(self) -> pd.DataFrame:
        """Get all positions as DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.positions.values()])

    def get_snapshot(self) -> dict[str, Any]:
        """Get complete portfolio snapshot."""
        snapshot = {
            "datetime": datetime.now().isoformat(),
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "invested": round(self.invested_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "day_pnl": self.get_day_pnl(),
            "positions_count": len(self.positions),
            "ic_positions_count": len(self.ic_positions),
            "exposure_pct": round(self.exposure_pct, 2),
            "drawdown_pct": round(self.drawdown, 2),
            "sector_exposure": self.get_sector_exposure(),
            "strategy_exposure": self.get_strategy_exposure(),
        }
        return snapshot

    def check_stops(self) -> list[dict[str, Any]]:
        """Check all positions against their stop loss and take profit."""
        triggers = []

        for symbol, pos in self.positions.items():
            price = pos.current_price
            if price is None or price <= 0:
                logger.warning(f"STOP_CHECK_SKIP: invalid price {price} for {symbol}")
                continue

            if pos.side == "BUY":
                if price <= pos.stop_loss and pos.stop_loss > 0:
                    triggers.append({
                        "symbol": symbol, "type": "stop_loss",
                        "price": price, "trigger": pos.stop_loss,
                    })
                elif price >= pos.take_profit and pos.take_profit > 0:
                    triggers.append({
                        "symbol": symbol, "type": "take_profit",
                        "price": price, "trigger": pos.take_profit,
                    })
            else:
                if price >= pos.stop_loss and pos.stop_loss > 0:
                    triggers.append({
                        "symbol": symbol, "type": "stop_loss",
                        "price": price, "trigger": pos.stop_loss,
                    })
                elif price <= pos.take_profit and pos.take_profit > 0:
                    triggers.append({
                        "symbol": symbol, "type": "take_profit",
                        "price": price, "trigger": pos.take_profit,
                    })

            # Time stop
            if pos.hold_hours > self.time_stop_days * 6.25:  # 6.25 trading hours/day
                triggers.append({
                    "symbol": symbol, "type": "time_stop",
                    "hold_hours": pos.hold_hours,
                })

        return triggers

    @property
    def time_stop_days(self):
        return 5  # Default from config
