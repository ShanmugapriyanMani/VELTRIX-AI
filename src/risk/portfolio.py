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
        }


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
        self.closed_trades: list[dict[str, Any]] = []
        self.daily_pnl_history: list[dict] = []

        # High water mark for drawdown
        self._peak_value = initial_capital

    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        self.positions[position.symbol] = position
        self.cash -= position.cost_basis
        logger.info(
            f"Position opened: {position.side} {position.quantity} {position.symbol} "
            f"@ ₹{position.entry_price:.2f} (₹{position.cost_basis:.2f})"
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
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / pos.cost_basis * 100, 2) if pos.cost_basis > 0 else 0,
            "charges": charges,
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

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

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
            if price <= 0:
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
