"""
Backtest Metrics — Comprehensive performance analysis.

Sharpe, Sortino, Alpha/Beta vs NIFTY, per-strategy attribution, charts.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


class BacktestMetrics:
    """
    Calculates comprehensive backtest performance metrics.
    """

    RISK_FREE_RATE = 0.065  # 6.5% Indian risk-free (10Y G-Sec)
    TRADING_DAYS = 252

    def __init__(
        self,
        trades: list,
        equity_curve: list[dict],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        self.trades_raw = trades
        self.equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns

        # Convert trades to DataFrame
        if trades:
            self.trades_df = pd.DataFrame([
                {
                    "symbol": t.symbol, "side": t.side, "quantity": t.quantity,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "entry_date": t.entry_date, "exit_date": t.exit_date,
                    "strategy": t.strategy, "regime": t.regime,
                    "charges": t.charges, "slippage": t.slippage,
                    "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                    "hold_days": t.hold_days, "exit_reason": t.exit_reason,
                }
                for t in trades
            ])
        else:
            self.trades_df = pd.DataFrame()

    def summary(self) -> dict[str, Any]:
        """Complete backtest summary."""
        return {
            "overview": self._overview(),
            "returns": self._return_metrics(),
            "risk": self._risk_metrics(),
            "trades": self._trade_metrics(),
            "strategy_attribution": self._strategy_attribution(),
            "regime_performance": self._regime_performance(),
            "monthly_returns": self._monthly_returns(),
            "cost_analysis": self._cost_analysis(),
        }

    def _overview(self) -> dict[str, Any]:
        if self.equity_df.empty:
            return {}

        final_equity = self.equity_df["equity"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        n_days = len(self.equity_df)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "total_pnl": round(final_equity - self.initial_capital, 2),
            "trading_days": n_days,
            "total_trades": len(self.trades_df),
        }

    def _return_metrics(self) -> dict[str, Any]:
        if self.equity_df.empty:
            return {}

        returns = self.equity_df["daily_return"].values / 100
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return {}

        # Annualized return
        total_ret = self.equity_df["equity"].iloc[-1] / self.initial_capital - 1
        n_years = len(returns) / self.TRADING_DAYS
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe Ratio (annualized)
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-10
        sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(self.TRADING_DAYS) if std_ret > 1e-10 else 0

        # Sortino Ratio (downside deviation only)
        downside = returns[returns < daily_rf] - daily_rf
        downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-10
        sortino = (mean_ret - daily_rf) / downside_std * np.sqrt(self.TRADING_DAYS) if downside_std > 1e-10 else 0

        # Calmar Ratio
        max_dd = self._max_drawdown()
        calmar = cagr / (max_dd / 100) if max_dd > 0 else 0

        metrics = {
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "avg_daily_return_pct": round(np.mean(returns) * 100, 4),
            "daily_volatility_pct": round(np.std(returns) * 100, 4),
            "annual_volatility_pct": round(np.std(returns) * np.sqrt(self.TRADING_DAYS) * 100, 2),
        }

        # Alpha / Beta vs benchmark
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            bench = self.benchmark_returns.values[:len(returns)]
            if len(bench) == len(returns):
                cov = np.cov(returns, bench)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1
                alpha = (np.mean(returns) - self.RISK_FREE_RATE / self.TRADING_DAYS - beta * (np.mean(bench) - self.RISK_FREE_RATE / self.TRADING_DAYS)) * self.TRADING_DAYS
                metrics["alpha"] = round(alpha * 100, 3)
                metrics["beta"] = round(beta, 3)

        return metrics

    def _risk_metrics(self) -> dict[str, Any]:
        if self.equity_df.empty:
            return {}

        returns = self.equity_df["daily_return"].values / 100
        returns = returns[~np.isnan(returns)]

        max_dd = self._max_drawdown()

        # VaR
        var_95 = abs(np.percentile(returns, 5)) * 100 if len(returns) > 0 else 0
        var_99 = abs(np.percentile(returns, 1)) * 100 if len(returns) > 0 else 0

        # Win/Loss streaks
        win_streak, loss_streak = self._max_streaks()

        return {
            "max_drawdown_pct": round(max_dd, 2),
            "var_95_pct": round(var_95, 2),
            "var_99_pct": round(var_99, 2),
            "max_win_streak": win_streak,
            "max_loss_streak": loss_streak,
            "positive_days_pct": round(np.mean(returns > 0) * 100, 1) if len(returns) > 0 else 0,
            "worst_day_pct": round(np.min(returns) * 100, 2) if len(returns) > 0 else 0,
            "best_day_pct": round(np.max(returns) * 100, 2) if len(returns) > 0 else 0,
        }

    def _trade_metrics(self) -> dict[str, Any]:
        if self.trades_df.empty:
            return {}

        df = self.trades_df
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1
        profit_factor = round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if len(losses) > 0 and losses["pnl"].sum() != 0 else 99.99

        return {
            "total_trades": len(df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_loss_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
            "profit_factor": round(profit_factor, 2),
            "largest_win": round(df["pnl"].max(), 2),
            "largest_loss": round(df["pnl"].min(), 2),
            "avg_hold_days": round(df["hold_days"].mean(), 1),
            "total_charges": round(df["charges"].sum(), 2),
            "total_slippage": round(df["slippage"].sum(), 2),
            "exit_reasons": dict(df["exit_reason"].value_counts()),
        }

    def _strategy_attribution(self) -> dict[str, Any]:
        """Per-strategy performance breakdown."""
        if self.trades_df.empty:
            return {}

        result = {}
        for strategy in self.trades_df["strategy"].unique():
            strat_trades = self.trades_df[self.trades_df["strategy"] == strategy]
            wins = strat_trades[strat_trades["pnl"] > 0]

            result[strategy] = {
                "trades": len(strat_trades),
                "total_pnl": round(strat_trades["pnl"].sum(), 2),
                "win_rate": round(len(wins) / len(strat_trades) * 100, 1) if len(strat_trades) > 0 else 0,
                "avg_pnl": round(strat_trades["pnl"].mean(), 2),
                "sharpe": self._strategy_sharpe(strat_trades),
            }

        return result

    def _regime_performance(self) -> dict[str, Any]:
        """Performance breakdown by market regime."""
        if self.trades_df.empty or "regime" not in self.trades_df.columns:
            return {}

        result = {}
        for regime in self.trades_df["regime"].unique():
            regime_trades = self.trades_df[self.trades_df["regime"] == regime]
            if regime_trades.empty:
                continue
            wins = regime_trades[regime_trades["pnl"] > 0]

            result[regime] = {
                "trades": len(regime_trades),
                "total_pnl": round(regime_trades["pnl"].sum(), 2),
                "win_rate": round(len(wins) / len(regime_trades) * 100, 1),
                "avg_pnl": round(regime_trades["pnl"].mean(), 2),
            }

        return result

    def _monthly_returns(self) -> dict[str, float]:
        """Monthly return series."""
        if self.equity_df.empty:
            return {}

        df = self.equity_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")

        monthly = {}
        for month, group in df.groupby("month"):
            start_eq = group["equity"].iloc[0]
            end_eq = group["equity"].iloc[-1]
            ret = (end_eq - start_eq) / start_eq * 100
            monthly[str(month)] = round(ret, 2)

        return monthly

    def _cost_analysis(self) -> dict[str, Any]:
        """Breakdown of trading costs."""
        if self.trades_df.empty:
            return {}

        total_turnover = (
            self.trades_df["entry_price"] * self.trades_df["quantity"]
        ).sum() * 2  # Both sides

        return {
            "total_charges": round(self.trades_df["charges"].sum(), 2),
            "total_slippage": round(self.trades_df["slippage"].sum(), 2),
            "total_turnover": round(total_turnover, 2),
            "charges_pct_of_turnover": round(
                self.trades_df["charges"].sum() / total_turnover * 100, 3
            ) if total_turnover > 0 else 0,
            "avg_charge_per_trade": round(self.trades_df["charges"].mean(), 2),
            "charges_pct_of_pnl": round(
                self.trades_df["charges"].sum() / abs(self.trades_df["pnl"].sum()) * 100, 1
            ) if self.trades_df["pnl"].sum() != 0 else 0,
        }

    def _max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if self.equity_df.empty:
            return 0.0

        equity = self.equity_df["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        return float(np.max(drawdown))

    def _max_streaks(self) -> tuple[int, int]:
        """Calculate max winning and losing streaks."""
        if self.trades_df.empty:
            return 0, 0

        pnls = self.trades_df["pnl"].values
        max_win = max_loss = current_win = current_loss = 0

        for pnl in pnls:
            if pnl > 0:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        return max_win, max_loss

    def _strategy_sharpe(self, trades: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for a subset of trades."""
        if len(trades) < 5:
            return 0.0
        pnls = trades["pnl"].values
        if np.std(pnls) == 0:
            return 0.0
        return round(float(np.mean(pnls) / np.std(pnls) * np.sqrt(self.TRADING_DAYS)), 3)
