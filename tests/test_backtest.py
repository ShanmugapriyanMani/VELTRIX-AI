"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.backtest.engine import BacktestEngine, BacktestTrade
from src.backtest.metrics import BacktestMetrics
from src.backtest.optimizer import WalkForwardOptimizer


def make_sample_data(n_days: int = 252, symbol: str = "TEST") -> dict[str, pd.DataFrame]:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    price = 1000
    rows = []

    for dt in dates:
        change = np.random.randn() * 15
        open_p = price
        close_p = price + change
        high_p = max(open_p, close_p) + abs(np.random.randn() * 5)
        low_p = min(open_p, close_p) - abs(np.random.randn() * 5)
        vol = int(np.random.uniform(500000, 2000000))

        rows.append({
            "datetime": dt,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
        price = close_p

    df = pd.DataFrame(rows)
    # Add basic features
    df["rsi_14"] = 50 + np.random.randn(n_days) * 10
    df["macd_histogram"] = np.random.randn(n_days) * 2
    df["atr_14"] = df["close"].rolling(14).apply(lambda x: np.std(x) * 1.5).fillna(15)

    return {symbol: df}


class TestBacktestEngine:
    def test_basic_backtest(self):
        data = make_sample_data(100)

        def signal_gen(date_str, bars, context):
            for sym, bar in bars.items():
                if bar.get("rsi_14", 50) < 40:
                    return {
                        "symbol": sym,
                        "direction": "BUY",
                        "confidence": 0.6,
                        "stop_loss": bar["close"] * 0.97,
                        "take_profit": bar["close"] * 1.06,
                        "strategy": "test",
                        "hold_days": 5,
                    }
            return None

        engine = BacktestEngine(initial_capital=500000)
        results = engine.run(data, signal_gen)

        assert "overview" in results
        assert results["overview"]["initial_capital"] == 500000
        assert results["overview"]["trading_days"] > 0

    def test_no_signals_backtest(self):
        data = make_sample_data(50)

        def no_signal(date_str, bars, context):
            return None

        engine = BacktestEngine(initial_capital=500000)
        results = engine.run(data, no_signal)

        assert results["overview"]["total_trades"] == 0
        assert results["overview"]["final_equity"] == 500000

    def test_costs_are_calculated(self):
        data = make_sample_data(100)

        def always_buy(date_str, bars, context):
            if context["n_positions"] > 0:
                return None
            for sym, bar in bars.items():
                return {
                    "symbol": sym, "direction": "BUY",
                    "confidence": 0.6,
                    "stop_loss": bar["close"] * 0.95,
                    "take_profit": bar["close"] * 1.10,
                    "strategy": "test", "hold_days": 3,
                }
            return None

        engine = BacktestEngine(initial_capital=500000)
        results = engine.run(data, always_buy)

        if results["trades"]["total_trades"] > 0:
            assert results["trades"]["total_charges"] > 0


class TestBacktestMetrics:
    def test_metrics_with_trades(self):
        trades = [
            BacktestTrade(pnl=100, pnl_pct=1.0, hold_days=3, strategy="test",
                          exit_reason="take_profit", charges=10, slippage=5,
                          entry_price=2500, quantity=4),
            BacktestTrade(pnl=-50, pnl_pct=-0.5, hold_days=2, strategy="test",
                          exit_reason="stop_loss", charges=10, slippage=5,
                          entry_price=2500, quantity=4),
            BacktestTrade(pnl=200, pnl_pct=2.0, hold_days=5, strategy="test",
                          exit_reason="take_profit", charges=10, slippage=5,
                          entry_price=2500, quantity=4),
        ]

        equity = [
            {"date": "2024-01-01", "equity": 500000, "daily_return": 0},
            {"date": "2024-01-02", "equity": 500100, "daily_return": 0.02},
            {"date": "2024-01-03", "equity": 500050, "daily_return": -0.01},
            {"date": "2024-01-04", "equity": 500250, "daily_return": 0.04},
        ]

        metrics = BacktestMetrics(trades, equity, 500000)
        summary = metrics.summary()

        assert summary["trades"]["total_trades"] == 3
        assert summary["trades"]["winning_trades"] == 2
        assert summary["trades"]["losing_trades"] == 1
        assert summary["trades"]["win_rate_pct"] == pytest.approx(66.7, rel=0.1)


class TestMonteCarlo:
    def test_monte_carlo(self):
        optimizer = WalkForwardOptimizer()

        trades = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30},
            {"pnl": 150}, {"pnl": -80}, {"pnl": 120}, {"pnl": -40},
            {"pnl": 180}, {"pnl": -60},
        ]

        result = optimizer.monte_carlo_test(trades, n_simulations=100)

        assert result["n_simulations"] == 100
        assert result["n_trades"] == 10
        assert "probability_profit" in result
        assert "mc_median_equity" in result
