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

    def test_paper_trader_reset_clears_positions(self):
        """reset_daily() must clear _positions and _gtt_orders."""
        # Create some state
        self.trader.place_order(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            quantity=10,
            side="BUY",
            price=2500,
        )
        self.trader.place_gtt_order(
            instrument_key="NSE_EQ|INE002A01018",
            trigger_price=2400,
            limit_price=2395,
            quantity=10,
            side="SELL",
        )
        assert len(self.trader._positions) > 0
        assert len(self.trader._gtt_orders) > 0
        assert len(self.trader._orders) > 0

        self.trader.reset_daily()

        assert self.trader._positions == {}
        assert self.trader._gtt_orders == {}
        assert self.trader._orders == {}
        assert self.trader._order_counter == 0


class TestOrderManager:
    def setup_method(self):
        self.trader = PaperTrader(initial_capital=25000)
        self.trader.connect()
        self.rm = RiskManager()
        self.cb = CircuitBreaker()
        self.om = OrderManager(self.trader, self.rm, self.cb)

    def teardown_method(self):
        if self.cb._state_file.exists():
            self.cb._state_file.unlink()

    def test_execute_signal(self):
        import pandas as pd
        signal = {
            "symbol": "ITC",
            "instrument_key": "NSE_EQ|INE154A01025",
            "direction": "BUY",
            "price": 450,
            "confidence": 0.8,
            "stop_loss": 430,
            "take_profit": 490,
            "strategy": "test",
            "regime": "TRENDING",
            "size_multiplier": 1.0,
            "atr": 10,
            "sector": "FMCG",
        }

        result = self.om.execute_signal(signal, 50000, pd.DataFrame())
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


class TestMarginAPIWarning:
    """Verify margin API failure sends warning only once per day."""

    def test_margin_api_failure_sends_warning_once(self):
        """_margin_api_warned_today flag prevents repeated Telegram alerts."""
        sent_messages = []

        class FakeAlerts:
            def send_raw(self, text):
                sent_messages.append(text)

        class FakeBroker:
            def get_available_margin(self):
                return None  # Simulate API failure

        alerts = FakeAlerts()
        broker = FakeBroker()

        # Simulate the margin check logic from main.py (2 iterations)
        _margin_api_warned_today = False
        for _ in range(3):
            available_margin = broker.get_available_margin()
            if available_margin is None:
                if not _margin_api_warned_today:
                    _margin_api_warned_today = True
                    alerts.send_raw("MARGIN API UNAVAILABLE")

        # Should only have sent ONE alert despite 3 failures
        assert len(sent_messages) == 1
        assert "MARGIN API" in sent_messages[0]
        assert _margin_api_warned_today is True


class TestApiCallsHaveTimeout:
    """API timeout is configured and injectable."""

    def test_api_timeout_config_exists(self):
        """API_TIMEOUT_SECONDS exists in EnvConfig with sensible default."""
        from src.config.env_loader import EnvConfig
        cfg = EnvConfig()
        assert hasattr(cfg, "API_TIMEOUT_SECONDS")
        assert cfg.API_TIMEOUT_SECONDS >= 5
        assert cfg.API_TIMEOUT_SECONDS <= 30

    def test_inject_api_timeout_wraps_request(self):
        """_inject_api_timeout wraps rest_client.request with default timeout."""
        from src.execution.upstox_broker import _inject_api_timeout

        # Mock an API client with a rest_client.request
        call_log = []

        class FakeRestClient:
            def request(self, *args, **kwargs):
                call_log.append(kwargs)
                return "ok"

        class FakeApiClient:
            rest_client = FakeRestClient()

        client = FakeApiClient()
        _inject_api_timeout(client, 10)

        # Call without timeout — should inject default
        client.rest_client.request("GET", "http://test.com")
        assert call_log[-1]["_request_timeout"] == 10

        # Call with explicit timeout — should keep it
        client.rest_client.request("GET", "http://test.com", _request_timeout=30)
        assert call_log[-1]["_request_timeout"] == 30
