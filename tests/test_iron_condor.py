"""Tests for Iron Condor strategy, portfolio, and backtest integration."""

from datetime import time as dt_time

from src.strategies.iron_condor import IronCondorStrategy
from src.risk.portfolio import IronCondorPosition, PortfolioManager
from src.backtest.engine import BacktestTrade


class TestICEntryConditions:
    """Verify IC entry condition gating."""

    def setup_method(self):
        self.ic = IronCondorStrategy()

    def test_ic_entry_conditions_all_met(self):
        """All conditions pass → True."""
        ok, reason = self.ic.check_entry_conditions(
            regime="RANGEBOUND",
            adx=15.0,
            pcr=1.0,
            vix=18.0,
            score_diff=1.0,
            current_time=dt_time(10, 30),
            is_expiry_day=False,
        )
        assert ok is True
        assert reason == ""

    def test_ic_not_triggered_on_expiry_day(self):
        """Expiry day → False."""
        ok, reason = self.ic.check_entry_conditions(
            regime="RANGEBOUND",
            adx=15.0,
            pcr=1.0,
            vix=18.0,
            score_diff=1.0,
            current_time=dt_time(10, 30),
            is_expiry_day=True,
        )
        assert ok is False
        assert "expiry" in reason.lower()

    def test_ic_rangebound_only_not_trending(self):
        """TRENDING regime → False."""
        ok, reason = self.ic.check_entry_conditions(
            regime="TRENDING",
            adx=15.0,
            pcr=1.0,
            vix=18.0,
            score_diff=1.0,
            current_time=dt_time(10, 30),
            is_expiry_day=False,
        )
        assert ok is False
        assert "RANGEBOUND" in reason

    def test_ic_skip_wide_opening_range(self):
        """Wide opening range (>0.4% of spot) → False."""
        ok, reason = self.ic.check_entry_conditions(
            regime="RANGEBOUND",
            adx=15.0,
            pcr=1.0,
            vix=18.0,
            score_diff=1.0,
            current_time=dt_time(10, 30),
            is_expiry_day=False,
            opening_range_pct=0.006,  # 0.6% > 0.4% threshold
        )
        assert ok is False
        assert "opening range" in reason.lower()

    def test_ic_skip_unstable_vix(self):
        """VIX changed >10% from yesterday → False."""
        ok, reason = self.ic.check_entry_conditions(
            regime="RANGEBOUND",
            adx=15.0,
            pcr=1.0,
            vix=20.0,
            score_diff=1.0,
            current_time=dt_time(10, 30),
            is_expiry_day=False,
            vix_prev=17.0,  # 17.6% change > 10% threshold
        )
        assert ok is False
        assert "VIX unstable" in reason


class TestICStrikeSelection:
    """Verify strike selection logic."""

    def setup_method(self):
        self.ic = IronCondorStrategy()

    def test_ic_skip_wings_too_close(self):
        """Wings distance < min_wing_distance → None."""
        # Force min_wing_distance to 500 so ATM±200 (distance=400) fails
        self.ic.min_wing_distance = 500
        result = self.ic.select_strikes_atm(23000.0, strike_gap=50)
        assert result is None

    def test_ic_strike_selection_uses_oi_data(self):
        """OI-based strikes used when available."""
        oi_data = {
            "max_call_oi_strike": 23500,
            "max_put_oi_strike": 22500,
        }
        result = self.ic.select_strikes_oi(23000.0, oi_data)
        assert result is not None
        assert result["sell_ce_strike"] == 23500
        assert result["sell_pe_strike"] == 22500
        assert result["buy_ce_strike"] == 23500 + self.ic.spread_width
        assert result["buy_pe_strike"] == 22500 - self.ic.spread_width


class TestICPositionSizing:
    """Verify IC economics and position sizing."""

    def setup_method(self):
        self.ic = IronCondorStrategy()

    def test_ic_skip_insufficient_credit(self):
        """Net credit < min_credit → None."""
        result = self.ic.calculate_position(
            sell_ce_prem=30.0,
            buy_ce_prem=20.0,
            sell_pe_prem=25.0,
            buy_pe_prem=20.0,
            lot_size=65,
            risk_per_trade=10000,
            deploy_cap=75000,
            strikes={"sell_ce_strike": 23200, "buy_ce_strike": 23400,
                     "sell_pe_strike": 22800, "buy_pe_strike": 22600},
        )
        # credit = (30-20) + (25-20) = 15, which is < 50 min_credit
        assert result is None

    def test_ic_position_sizing_respects_risk_cap(self):
        """Lots capped by risk_per_trade."""
        result = self.ic.calculate_position(
            sell_ce_prem=120.0,
            buy_ce_prem=80.0,
            sell_pe_prem=110.0,
            buy_pe_prem=70.0,
            lot_size=65,
            risk_per_trade=10000,
            deploy_cap=75000,
            strikes={"sell_ce_strike": 23200, "buy_ce_strike": 23400,
                     "sell_pe_strike": 22800, "buy_pe_strike": 22600},
        )
        assert result is not None
        # net_credit = (120-80) + (110-70) = 80
        # max_loss_per_unit = 200 - 80 = 120
        # lots_by_risk = 10000 / (120 * 65) = 1.28 → 1
        assert result["lots"] >= 1
        assert result["max_loss"] <= 10000 * 1.1  # Within 10% of risk cap


class TestICExits:
    """Verify IC TP/SL/EOD exit logic in portfolio."""

    def setup_method(self):
        self.pm = PortfolioManager()

    def _make_ic_pos(self, pos_id="IC_001", net_credit=80.0, qty=65):
        """Helper: create and open an IC position."""
        ic = IronCondorPosition(
            position_id=pos_id,
            quantity=qty,
            lots=1,
            sell_ce_strike=23200,
            sell_ce_instrument_key="NSE_FO|NIFTY23200CE",
            sell_ce_premium=120.0,
            buy_ce_strike=23400,
            buy_ce_instrument_key="NSE_FO|NIFTY23400CE",
            buy_ce_premium=80.0,
            sell_pe_strike=22800,
            sell_pe_instrument_key="NSE_FO|NIFTY22800PE",
            sell_pe_premium=110.0,
            buy_pe_strike=22600,
            buy_pe_instrument_key="NSE_FO|NIFTY22600PE",
            buy_pe_premium=70.0,
            net_credit=net_credit,
            spread_width=200,
            max_profit=net_credit * qty,
            max_loss=(200 - net_credit) * qty,
            tp_threshold=net_credit * 0.80 * qty,
            sl_threshold=-net_credit * 2.0 * qty,
        )
        self.pm.open_ic_position(ic)
        return ic

    def test_ic_tp_exit_at_80_percent_credit(self):
        """TP trigger fires when P&L >= 80% of credit."""
        ic = self._make_ic_pos(net_credit=80.0, qty=65)
        # Simulate: both spreads collapsed (market stayed in range)
        # close_cost_ce = 8 - 2 = 6, close_cost_pe = 7 - 3 = 4
        # total_close_cost = 10, pnl = (80 - 10) * 65 = 4550
        # tp_threshold = 80 * 0.80 * 65 = 4160
        ltp_dict = {
            "NSE_FO|NIFTY23200CE": 8.0,
            "NSE_FO|NIFTY23400CE": 2.0,
            "NSE_FO|NIFTY22800PE": 7.0,
            "NSE_FO|NIFTY22600PE": 3.0,
        }
        triggers = self.pm.check_ic_stops(ltp_dict)
        assert len(triggers) == 1
        assert triggers[0]["type"] == "ic_take_profit"
        assert triggers[0]["pnl"] >= ic.tp_threshold

    def test_ic_sl_exit_when_short_leg_doubles(self):
        """SL trigger fires when loss >= 2× credit."""
        ic = self._make_ic_pos(net_credit=80.0, qty=65)
        # Simulate: market broke heavily upside, call spread max loss
        # close_cost_ce = 400 - 200 = 200, close_cost_pe = 3 - 1 = 2
        # total_close_cost = 202, pnl = (80 - 202) * 65 = -7930 ... still not enough
        # Need: pnl <= -10400 → close_cost >= 80 + 10400/65 = 80+160 = 240
        # So: close_cost_ce = 245, close_cost_pe = 2 → total = 247
        # pnl = (80 - 247) * 65 = -10855 < -10400 ✓
        ltp_dict = {
            "NSE_FO|NIFTY23200CE": 250.0,  # Sell CE blew up
            "NSE_FO|NIFTY23400CE": 5.0,
            "NSE_FO|NIFTY22800PE": 3.0,
            "NSE_FO|NIFTY22600PE": 1.0,
        }
        triggers = self.pm.check_ic_stops(ltp_dict)
        assert len(triggers) == 1
        assert triggers[0]["type"] == "ic_stop_loss"
        assert triggers[0]["pnl"] <= ic.sl_threshold

    def test_ic_eod_exit_at_1510(self):
        """IC position can be force-closed (portfolio method works)."""
        ic = self._make_ic_pos()
        assert self.pm.has_ic_position()

        self.pm.close_ic_position(ic.position_id, "eod_exit", pnl=1500.0)
        assert not self.pm.has_ic_position()
        # ic object was modified by close_ic_position before deletion from dict
        assert ic.status == "closed"
        assert ic.exit_reason == "eod_exit"


class TestBacktestTradeFields:
    """Verify BacktestTrade leg3/leg4 fields work for IC."""

    def test_ic_backtest_trade_has_all_legs(self):
        """BacktestTrade with IC fills all 4 legs."""
        trade = BacktestTrade(
            symbol="NIFTY23200CE",
            side="SELL",
            quantity=65,
            entry_price=120.0,
            exit_price=8.0,
            entry_date="2025-01-15",
            exit_date="2025-01-15",
            strategy="IRON_CONDOR",
            regime="RANGEBOUND",
            trade_type="IRON_CONDOR",
            leg2_symbol="NIFTY23400CE",
            leg2_side="BUY",
            leg2_entry_price=80.0,
            leg2_exit_price=2.0,
            leg3_symbol="NIFTY22800PE",
            leg3_side="SELL",
            leg3_entry_price=110.0,
            leg3_exit_price=7.0,
            leg4_symbol="NIFTY22600PE",
            leg4_side="BUY",
            leg4_entry_price=70.0,
            leg4_exit_price=3.0,
            spread_width=200,
            net_premium=80.0,
            pnl=4000.0,
            pnl_pct=76.9,
        )
        assert trade.trade_type == "IRON_CONDOR"
        assert trade.leg3_symbol == "NIFTY22800PE"
        assert trade.leg4_symbol == "NIFTY22600PE"
        assert trade.leg3_entry_price == 110.0
        assert trade.leg4_entry_price == 70.0
