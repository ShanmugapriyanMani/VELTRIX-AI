"""Tests for risk management system."""

import pandas as pd

from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioManager, Position, compute_kelly_fraction
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
        # Max 40% per trade (from config), capped by min_position_value/max_position_value
        assert result["value"] <= 25000 * 0.40

    def test_position_sizing_zero_price(self):
        result = self.rm.calculate_position_size(
            capital=25000, price=0, confidence=0.7, atr=50,
        )
        assert result["quantity"] == 0

    def test_position_sizing_exposure_limit(self):
        result = self.rm.calculate_position_size(
            capital=25000, price=500, confidence=0.7, atr=10,
            current_exposure=21500,  # Near 90% limit (₹22,500)
        )
        # Deployable = min(90%, 90%) of 25K = 22500, remaining = 1000
        # Result should be capped by remaining exposure
        assert result["value"] <= 25000 * 0.90 - 21500 + 100

    def test_position_sizing_cash_reserve(self):
        """10% cash reserve means max deployable is 90% of capital."""
        result = self.rm.calculate_position_size(
            capital=25000, price=500, confidence=0.9, atr=10,
            current_exposure=21500,
        )
        # With 10% cash reserve: deployable = 25000 * 0.90 = 22500
        # remaining = 22500 - 21500 = 1000
        assert result["value"] <= 1000 + 1

    def test_stop_loss_calculation(self):
        stops = self.rm.calculate_stops(
            entry_price=2500, atr=50, direction="BUY"
        )
        # Config: sl_atr_mult=1.5, tp_atr_mult=2.0
        assert stops["stop_loss"] == 2500 - 1.5 * 50  # 2425
        assert stops["take_profit"] == 2500 + 2.0 * 50  # 2600
        # risk = 75, reward = 100, ratio = 1.33
        assert stops["reward_risk_ratio"] == round(100 / 75, 2)

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
            price=2500,
            quantity=4,  # ₹10,000 > min_position_value (₹5,000) and < 50% of ₹25K
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
            quantity=10,  # ₹25,000 > 50% of ₹25K (₹12,500)
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


class TestTradeRecordDBFields:
    """Verify close_position returns all fields needed by store.save_trade()."""

    def setup_method(self):
        self.pm = PortfolioManager(initial_capital=100000)

    def test_save_trade_populates_all_price_fields(self):
        """close_position must return DB-compatible keys so save_trade writes non-zero values."""
        pos = Position(
            symbol="NIFTY2560524200CE",
            instrument_key="NSE_FO|NIFTY2560524200CE",
            side="BUY",
            quantity=65,
            entry_price=120.0,
            current_price=150.0,
            stop_loss=90.0,
            take_profit=180.0,
            strategy="options_buyer",
            trade_id="T_20260317_001",
            order_id="ORD_123",
        )
        self.pm.add_position(pos)
        result = self.pm.close_position("NIFTY2560524200CE", 150.0, "take_profit")

        assert result is not None
        # DB-compatible keys must be present and non-zero
        assert result["price"] == 120.0, "price (entry) must match entry_price"
        assert result["fill_price"] == 150.0, "fill_price (exit) must match exit_price"
        assert result["stop_loss"] == 90.0, "stop_loss must be preserved"
        assert result["take_profit"] == 180.0, "take_profit must be preserved"
        assert result["instrument_key"] == "NSE_FO|NIFTY2560524200CE"
        assert result["order_id"] == "ORD_123"
        assert result["status"] == "closed"
        assert result["hold_duration_hours"] >= 0
        assert result["notes"] == "take_profit"
        assert result["total_charges"] == 0
        # Original keys still present for backward compat
        assert result["entry_price"] == 120.0
        assert result["exit_price"] == 150.0
        assert result["pnl"] == (150.0 - 120.0) * 65

    def test_partial_close_populates_all_price_fields(self):
        """partial_close_position must also return DB-compatible keys."""
        pos = Position(
            symbol="NIFTY2560524200PE",
            instrument_key="NSE_FO|NIFTY2560524200PE",
            side="BUY",
            quantity=130,
            entry_price=80.0,
            current_price=100.0,
            stop_loss=60.0,
            take_profit=120.0,
            strategy="options_buyer",
            trade_id="T_20260317_002",
            order_id="ORD_456",
        )
        self.pm.add_position(pos)
        result = self.pm.partial_close_position(
            "NIFTY2560524200PE", 100.0, 65, "tp1_partial"
        )

        assert result is not None
        assert result["price"] == 80.0
        assert result["fill_price"] == 100.0
        assert result["stop_loss"] == 60.0
        assert result["take_profit"] == 120.0
        assert result["instrument_key"] == "NSE_FO|NIFTY2560524200PE"
        assert result["order_id"] == "ORD_456"
        assert result["status"] == "closed"
        assert result["notes"] == "tp1_partial"
        assert result["quantity"] == 65
        assert result["pnl"] == (100.0 - 80.0) * 65


class TestCircuitBreaker:

    def setup_method(self):
        self.cb = CircuitBreaker()
        # Ensure clean state (state file may have stale data from previous test)
        self.cb.reset_daily()

    def teardown_method(self):
        # Clean up state file created by persistence
        if self.cb._state_file.exists():
            self.cb._state_file.unlink()

    def test_normal_state(self):
        status = self.cb.check(daily_loss_pct=1.0, drawdown_pct=5.0, open_positions=3)
        assert status.state == BreakerState.NORMAL
        assert self.cb.can_trade()

    def test_two_consecutive_sl_halts_day(self):
        """Rule 1: 2 consecutive SL exits halt trading for the day."""
        self.cb.record_trade(-5000)  # SL 1
        assert self.cb.can_trade()  # Still OK after 1 SL
        assert self.cb._consecutive_sl == 1

        self.cb.record_trade(-5000)  # SL 2
        assert not self.cb.can_trade()  # Halted
        assert self.cb._state == BreakerState.HALTED
        assert self.cb._halt_reason == "consecutive_losses"

    def test_daily_loss_limit_halts_day(self):
        """Rule 2: Daily loss > ₹20,000 halts trading for the day."""
        # Use win-loss-win-loss pattern to avoid triggering Rule 1 (consecutive SL)
        self.cb.record_trade(-15000)  # Loss 1
        assert self.cb.can_trade()
        self.cb.record_trade(100)     # Win — resets consecutive SL counter
        self.cb.record_trade(-6000)   # Loss 2, total: -₹20,900 > ₹20K
        assert not self.cb.can_trade()
        assert self.cb._state == BreakerState.HALTED
        assert self.cb._halt_reason == "daily_loss"

    def test_daily_reset_clears_halt(self):
        """Daily reset always clears halt — no carry-over."""
        # Trigger halt via consecutive SL
        self.cb.record_trade(-5000)
        self.cb.record_trade(-5000)
        assert not self.cb.can_trade()

        # Daily reset clears everything
        self.cb.reset_daily()
        assert self.cb.can_trade()
        assert self.cb._state == BreakerState.NORMAL
        assert self.cb._consecutive_sl == 0
        assert self.cb._daily_pnl == 0.0
        assert self.cb._halt_reason == ""

    def test_consecutive_sl_resets_on_win(self):
        """A win resets the consecutive SL counter."""
        self.cb.record_trade(-5000)  # SL 1
        assert self.cb._consecutive_sl == 1

        self.cb.record_trade(3000)  # Win
        assert self.cb._consecutive_sl == 0
        assert self.cb.can_trade()

        # Need 2 more SLs to halt again
        self.cb.record_trade(-5000)  # SL 1 (fresh)
        assert self.cb.can_trade()

    def test_size_multiplier_reduces_after_sl(self):
        """Size multiplier reduces after consecutive SL hits."""
        # 0 losses → full size
        assert self.cb.get_size_multiplier() == 1.0

        # 1 loss → 75% size
        self.cb.record_trade(-5000)
        assert self.cb.get_size_multiplier() == 0.75

        # 2 losses → halted → 0.0
        self.cb.record_trade(-5000)
        assert self.cb.get_size_multiplier() == 0.0

    def test_size_multiplier_resets_after_win(self):
        """Size multiplier resets to 1.0 after a win."""
        self.cb.record_trade(-5000)  # 1 SL
        assert self.cb.get_size_multiplier() == 0.75

        self.cb.record_trade(3000)  # Win → reset
        assert self.cb.get_size_multiplier() == 1.0

    def test_equity_mult_full_size_near_peak(self):
        """Equity multiplier stays 1.0 when near peak."""
        self.cb.update_equity(150000)  # Set peak
        self.cb.update_equity(148000)  # 1.3% drawdown (<5%)
        assert self.cb.equity_size_multiplier == 1.0

    def test_equity_mult_reduces_at_10pct_drawdown(self):
        """Equity multiplier drops through tiers as drawdown increases."""
        self.cb.update_equity(150000)  # Set peak
        # 7% drawdown → 0.85
        self.cb.update_equity(139500)
        assert self.cb.equity_size_multiplier == 0.85
        # 12% drawdown → 0.70
        self.cb.update_equity(132000)
        assert self.cb.equity_size_multiplier == 0.70
        # 20% drawdown → 0.50
        self.cb.update_equity(120000)
        assert self.cb.equity_size_multiplier == 0.50

    def test_equity_mult_combines_with_cb_mult(self):
        """Combined multiplier is min of CB and equity multipliers."""
        # CB at 0.75 (1 SL), equity at 1.0 → combined = 0.75
        self.cb.record_trade(-5000)
        self.cb.update_equity(150000)
        assert min(self.cb.get_size_multiplier(), self.cb.equity_size_multiplier) == 0.75

        # CB at 0.75 (1 SL), equity at 0.50 (20% DD) → combined = 0.50
        self.cb.update_equity(120000)
        assert min(self.cb.get_size_multiplier(), self.cb.equity_size_multiplier) == 0.50

    def test_conviction_boost_always_zero(self):
        """No conviction boost — removed with simplified CB."""
        assert self.cb.get_conviction_boost() == 0.0

    def test_kill_switch(self):
        result = self.cb.activate_kill_switch()
        assert result["action"] == "cancel_all_flatten"
        assert not self.cb.can_trade()

    def test_record_order_always_true(self):
        """No order rate limiting — always returns True."""
        for _ in range(20):
            assert self.cb.record_order() is True


class TestKellyFraction:
    """Quant Phase 2: Dynamic Kelly Sizing."""

    def test_kelly_fraction_computed_correctly(self):
        """Known trade history → correct half-Kelly multiplier."""
        # 15 wins of +3000, 5 losses of -5000
        # WR = 0.75, avg_win = 3000, avg_loss = 5000
        # payoff = 3000/5000 = 0.6
        # kelly = 0.75 - 0.25/0.6 = 0.75 - 0.4167 = 0.3333
        # half_kelly = 0.1667
        # multiplier = 0.1667 / 0.30 = 0.556
        pnl = [3000] * 15 + [-5000] * 5
        result = compute_kelly_fraction(pnl, window=20, min_trades=10)
        assert 0.50 <= result <= 0.60, f"Expected ~0.56, got {result:.3f}"

        # 18 wins of +5000, 2 losses of -3000
        # WR = 0.90, avg_win = 5000, avg_loss = 3000
        # payoff = 5000/3000 = 1.667
        # kelly = 0.90 - 0.10/1.667 = 0.90 - 0.060 = 0.840
        # half_kelly = 0.420
        # multiplier = 0.420 / 0.30 = 1.40
        pnl2 = [5000] * 18 + [-3000] * 2
        result2 = compute_kelly_fraction(pnl2, window=20, min_trades=10)
        assert 1.35 <= result2 <= 1.45, f"Expected ~1.40, got {result2:.3f}"

        # Capped at max_mult
        result3 = compute_kelly_fraction(pnl2, window=20, min_trades=10, max_mult=1.20)
        assert result3 == 1.20

    def test_kelly_returns_full_size_insufficient_data(self):
        """With <min_trades data, Kelly returns 1.0 (full size)."""
        # Less than 10 trades → 1.0
        assert compute_kelly_fraction([1000, -500, 2000], min_trades=10) == 1.0

        # All wins → 1.0
        assert compute_kelly_fraction([1000] * 20) == 1.0

        # All losses → 1.0
        assert compute_kelly_fraction([-1000] * 20) == 1.0

        # Empty → 1.0
        assert compute_kelly_fraction([]) == 1.0
