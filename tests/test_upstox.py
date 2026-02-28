"""Tests for Upstox integration (using sandbox/paper trader)."""

import pytest
from src.execution.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager
from src.risk.manager import RiskManager
from src.risk.circuit_breaker import CircuitBreaker


class TestPaperTrader:
    def setup_method(self):
        self.trader = PaperTrader(initial_capital=500000)

    def test_connect(self):
        assert self.trader.connect() is True

    def test_place_buy_order(self):
        result = self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        assert result["status"] == "success"
        assert result["order_id"] is not None

    def test_place_sell_order(self):
        # First buy
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        # Then sell
        result = self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="SELL",
            price=2550,
        )
        assert result["status"] == "success"

    def test_insufficient_funds(self):
        result = self.trader.place_order(
            symbol="TEST",
            instrument_key="NSE_EQ|TEST",
            quantity=1000,
            side="BUY",
            price=10000,  # ₹1cr > ₹5L
        )
        assert result["status"] == "rejected"

    def test_get_positions(self):
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        positions = self.trader.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "RELIANCE"

    def test_get_funds(self):
        funds = self.trader.get_funds()
        assert funds["available_margin"] == 500000

    def test_square_off_all(self):
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        result = self.trader.square_off_all()
        assert result["squared"] == 1
        assert len(self.trader.get_positions()) == 0

    def test_gtt_order(self):
        result = self.trader.place_gtt_order(
            instrument_key="NSE_EQ|INE002A01018",
            trigger_price=2400,
            limit_price=2395,
            quantity=10,
            side="SELL",
        )
        assert result["status"] == "success"

    def test_slippage_applied(self):
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        positions = self.trader.get_positions()
        # With 0.05% slippage, fill price should be slightly higher
        assert positions[0]["average_price"] >= 2500

    def test_update_prices(self):
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        self.trader.update_prices({"RELIANCE": 2600})
        positions = self.trader.get_positions()
        assert positions[0]["pnl"] > 0


class TestOrderManager:
    def setup_method(self):
        self.trader = PaperTrader(initial_capital=25000)
        self.trader.connect()
        self.rm = RiskManager()
        self.cb = CircuitBreaker()
        self.om = OrderManager(self.trader, self.rm, self.cb)

    def test_execute_signal(self):
        import pandas as pd
        signal = {
            "symbol": "ITC",
            "instrument_key": "NSE_EQ|INE154A01025",
            "direction": "BUY",
            "price": 450,
            "confidence": 0.7,
            "stop_loss": 430,
            "take_profit": 490,
            "strategy": "test",
            "regime": "BULL_TRENDING",
            "size_multiplier": 1.0,
            "atr": 10,
            "sector": "FMCG",
        }

        result = self.om.execute_signal(signal, 25000, pd.DataFrame())
        assert result["status"] == "success"
        assert result["quantity"] > 0

    def test_circuit_breaker_blocks_trade(self):
        import pandas as pd
        # Activate halt (6% > 5% threshold)
        self.cb.check(daily_loss_pct=6.0, drawdown_pct=5.0, open_positions=0)

        signal = {
            "symbol": "ITC",
            "instrument_key": "NSE_EQ|INE154A01025",
            "direction": "BUY",
            "price": 450,
            "confidence": 0.7,
            "atr": 10,
        }

        result = self.om.execute_signal(signal, 25000, pd.DataFrame())
        assert result["status"] == "blocked"
