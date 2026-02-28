"""Tests for risk management system."""

import pandas as pd

from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioManager, Position
from src.risk.circuit_breaker import CircuitBreaker, BreakerState


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()

    def test_position_sizing_basic(self):
        result = self.rm.calculate_position_size(
            capital=25000,
            price=500,
            confidence=0.7,
            atr=10,
            win_rate=0.55,
        )
        assert result["quantity"] > 0
        assert result["value"] <= 25000 * 0.04  # Max 4% per trade = ₹1,000

    def test_position_sizing_zero_price(self):
        result = self.rm.calculate_position_size(
            capital=25000, price=0, confidence=0.7, atr=50,
        )
        assert result["quantity"] == 0

    def test_position_sizing_exposure_limit(self):
        result = self.rm.calculate_position_size(
            capital=25000, price=500, confidence=0.7, atr=10,
            current_exposure=19500,  # Near 80% limit (₹20,000)
        )
        # Deployable = min(80%, 80%) of 25K = 20K, remaining = 500
        assert result["value"] <= 25000 * 0.80 - 19500 + 100

    def test_position_sizing_cash_reserve(self):
        """20% cash reserve means max deployable is 80% of capital."""
        result = self.rm.calculate_position_size(
            capital=25000, price=500, confidence=0.9, atr=10,
            current_exposure=19000,
        )
        # With 20% cash reserve: deployable = 25000 * 0.80 = 20000
        # remaining = 20000 - 19000 = 1000
        assert result["value"] <= 1000 + 1

    def test_stop_loss_calculation(self):
        stops = self.rm.calculate_stops(
            entry_price=2500, atr=50, direction="BUY"
        )
        assert stops["stop_loss"] == 2500 - 2.0 * 50  # 2400
        assert stops["take_profit"] == 2500 + 4.0 * 50  # 2700
        assert stops["reward_risk_ratio"] == 2.0

    def test_trade_cost_calculation(self):
        costs = self.rm.calculate_trade_costs(
            price=2500, quantity=10, side="BUY", product="I"
        )
        assert costs["brokerage"] <= 20  # Max ₹20 per order
        assert costs["total_charges"] > 0

    def test_round_trip_cost(self):
        rt = self.rm.calculate_round_trip_cost(price=2500, quantity=10, product="I")
        assert rt > 0

    def test_pre_trade_check_passes(self):
        result = self.rm.pre_trade_check(
            symbol="RELIANCE",
            price=500,
            quantity=2,  # ₹1,000 < 8% of ₹25K (₹2,000)
            direction="BUY",
            capital=25000,
            current_positions=pd.DataFrame(),
            sector="OIL_GAS",
        )
        assert result["passed"] is True

    def test_pre_trade_check_exceeds_stock_limit(self):
        result = self.rm.pre_trade_check(
            symbol="RELIANCE",
            price=2500,
            quantity=10,  # ₹25,000 > 8% of ₹25K (₹2,000)
            direction="BUY",
            capital=25000,
            current_positions=pd.DataFrame(),
        )
        assert result["passed"] is False


class TestPortfolioManager:
    def setup_method(self):
        self.pm = PortfolioManager(initial_capital=25000)

    def test_add_position(self):
        pos = Position(
            symbol="SBIN", side="BUY", quantity=2,
            entry_price=800, current_price=800,
        )
        self.pm.add_position(pos)
        assert len(self.pm.positions) == 1
        assert self.pm.cash == 25000 - 1600

    def test_close_position_profit(self):
        pos = Position(
            symbol="SBIN", side="BUY", quantity=2,
            entry_price=800, current_price=850,
        )
        self.pm.add_position(pos)
        result = self.pm.close_position("SBIN", 850, "take_profit")

        assert result is not None
        assert result["pnl"] > 0
        assert len(self.pm.positions) == 0

    def test_close_position_loss(self):
        pos = Position(
            symbol="SBIN", side="BUY", quantity=2,
            entry_price=800, current_price=750,
        )
        self.pm.add_position(pos)
        result = self.pm.close_position("SBIN", 750, "stop_loss")

        assert result is not None
        assert result["pnl"] < 0

    def test_drawdown_calculation(self):
        assert self.pm.drawdown == 0

        pos = Position(
            symbol="ITC", side="BUY", quantity=5,
            entry_price=450, current_price=430,
        )
        self.pm.add_position(pos)
        self.pm.update_prices({"ITC": 430})
        assert self.pm.drawdown >= 0

    def test_sector_exposure(self):
        pos1 = Position(symbol="HDFCBANK", side="BUY", quantity=1,
                        entry_price=1700, current_price=1700, sector="BANKING")
        pos2 = Position(symbol="ICICIBANK", side="BUY", quantity=1,
                        entry_price=1200, current_price=1200, sector="BANKING")
        self.pm.add_position(pos1)
        self.pm.add_position(pos2)

        sectors = self.pm.get_sector_exposure()
        assert "BANKING" in sectors
        assert sectors["BANKING"] == 1700 + 1200

    def test_snapshot(self):
        snapshot = self.pm.get_snapshot()
        assert "total_value" in snapshot
        assert "exposure_pct" in snapshot
        assert snapshot["total_value"] == 25000


class TestCircuitBreaker:
    def setup_method(self):
        self.cb = CircuitBreaker()

    def test_normal_state(self):
        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.NORMAL
        assert self.cb.can_trade()

    def test_daily_loss_warning(self):
        """3% daily loss → WARNING (reduce sizes 50%)."""
        status = self.cb.check(daily_loss_pct=3.5, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.WARNING
        assert self.cb.get_size_multiplier() == 0.5

    def test_daily_loss_halt(self):
        """5% daily loss → HALT trading."""
        status = self.cb.check(daily_loss_pct=6.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.HALTED
        assert not self.cb.can_trade()

    def test_drawdown_warning(self):
        """15% drawdown → WARNING (reduce sizes 50%)."""
        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=16.0, open_positions=3)
        assert status.state == BreakerState.WARNING
        assert self.cb.get_size_multiplier() == 0.5

    def test_drawdown_critical_halt(self):
        """22% drawdown → HALT."""
        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=23.0, open_positions=3)
        assert status.state == BreakerState.HALTED

    def test_consecutive_losses_pause(self):
        """6 consecutive losses → PAUSE 1 hour."""
        for _ in range(6):
            self.cb.record_trade(-100)

        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.PAUSED

    def test_consecutive_losses_halt(self):
        """10 consecutive losses → HALT."""
        for _ in range(10):
            self.cb.record_trade(-100)

        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.HALTED

    def test_consecutive_losses_reset_on_win(self):
        for _ in range(3):
            self.cb.record_trade(-100)
        self.cb.record_trade(200)  # Win resets streak

        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.NORMAL

    def test_kill_switch(self):
        result = self.cb.activate_kill_switch()
        assert result["action"] == "cancel_all_flatten"
        assert not self.cb.can_trade()

    def test_daily_reset(self):
        self.cb.check(daily_loss_pct=6.0, drawdown_pct=5.0, open_positions=0)
        assert not self.cb.can_trade()

        self.cb.reset_daily()
        assert self.cb.can_trade()

    def test_runaway_order_detection(self):
        """More than 10 orders in 1 minute → HALT."""
        for _ in range(11):
            self.cb.record_order()

        assert not self.cb.can_trade()
        assert self.cb._state == BreakerState.HALTED
        assert "Runaway" in self.cb._halt_reason
