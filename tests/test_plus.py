"""Tests for PLUS stage — trade type decision, spread risk, spread execution."""

from src.backtest.engine import BacktestTrade
from src.risk.manager import RiskManager


class TestTradeTypeDecision:
    """Verify PLUS decision tree routing."""

    def setup_method(self):
        # Import here to avoid env_loader side effects
        from src.strategies.options_buyer import OptionsBuyerStrategy
        self.strategy = OptionsBuyerStrategy()

    def test_volatile_high_conviction_credit_spread(self):
        result = self.strategy._determine_trade_type("VOLATILE", 2.5)
        assert result == "CREDIT_SPREAD"

    def test_volatile_low_conviction_skip(self):
        result = self.strategy._determine_trade_type("VOLATILE", 1.5)
        assert result == "SKIP"

    def test_volatile_exact_threshold_credit_spread(self):
        result = self.strategy._determine_trade_type("VOLATILE", 2.0)
        assert result == "CREDIT_SPREAD"

    def test_high_conviction_naked_buy(self):
        result = self.strategy._determine_trade_type("TRENDING", 3.5)
        assert result == "NAKED_BUY"

    def test_exact_3_naked_buy(self):
        result = self.strategy._determine_trade_type("TRENDING", 3.0)
        assert result == "NAKED_BUY"

    def test_medium_conviction_trending_skip(self):
        # V9 P1: TRENDING + low conviction → SKIP (was DEBIT_SPREAD)
        result = self.strategy._determine_trade_type("TRENDING", 2.0)
        assert result == "SKIP"

    def test_rangebound_medium_conviction_naked(self):
        result = self.strategy._determine_trade_type("RANGEBOUND", 2.5)
        assert result == "NAKED_BUY"

    def test_rangebound_high_conviction_naked(self):
        result = self.strategy._determine_trade_type("RANGEBOUND", 4.0)
        assert result == "NAKED_BUY"

    def test_elevated_credit_spread(self):
        # V9 P3: ELEVATED regime → CREDIT_SPREAD
        result = self.strategy._determine_trade_type("ELEVATED", 2.5)
        assert result == "CREDIT_SPREAD"

    def test_elevated_low_conviction_skip(self):
        result = self.strategy._determine_trade_type("ELEVATED", 1.5)
        assert result == "SKIP"

    def test_rangebound_below_25_skip(self):
        # V9 P1: RANGEBOUND needs conv >= 2.5 for debit spread
        result = self.strategy._determine_trade_type("RANGEBOUND", 2.0)
        assert result == "SKIP"


class TestSpreadRisk:
    """Test spread risk validation."""

    def setup_method(self):
        self.rm = RiskManager()

    def test_debit_spread_within_risk(self):
        result = self.rm.validate_spread_risk(
            trade_type="DEBIT_SPREAD",
            net_premium=50,
            spread_width=200,
            quantity=195,
        )
        # max_loss = 50 * 195 = 9750 ≤ 10000
        assert result["passed"] is True
        assert result["max_loss"] == 9750.0

    def test_debit_spread_exceeds_risk(self):
        result = self.rm.validate_spread_risk(
            trade_type="DEBIT_SPREAD",
            net_premium=100,
            spread_width=200,
            quantity=195,
        )
        # max_loss = 100 * 195 = 19500 > 10000
        assert result["passed"] is False

    def test_credit_spread_within_risk(self):
        result = self.rm.validate_spread_risk(
            trade_type="CREDIT_SPREAD",
            net_premium=55,
            spread_width=200,
            quantity=65,
        )
        # max_loss = (200 - 55) * 65 = 9425 ≤ 10000
        assert result["passed"] is True
        assert result["max_loss"] == 9425.0

    def test_credit_spread_exceeds_risk(self):
        result = self.rm.validate_spread_risk(
            trade_type="CREDIT_SPREAD",
            net_premium=30,
            spread_width=200,
            quantity=130,
        )
        # max_loss = (200 - 30) * 130 = 22100 > 10000
        assert result["passed"] is False

    def test_unknown_type_fails(self):
        result = self.rm.validate_spread_risk(
            trade_type="UNKNOWN",
            net_premium=50,
            spread_width=200,
            quantity=65,
        )
        assert result["passed"] is False


class TestBacktestTradeSpreadFields:
    """Test BacktestTrade dataclass has correct spread defaults."""

    def test_default_fields(self):
        trade = BacktestTrade()
        assert trade.trade_type == "NAKED_BUY"
        assert trade.leg2_symbol == ""
        assert trade.leg2_side == ""
        assert trade.leg2_entry_price == 0.0
        assert trade.leg2_exit_price == 0.0
        assert trade.spread_width == 0
        assert trade.net_premium == 0.0
        assert trade.max_profit == 0.0
        assert trade.max_loss == 0.0

    def test_debit_spread_fields(self):
        trade = BacktestTrade(
            symbol="NIFTY25500CE",
            side="BUY",
            trade_type="DEBIT_SPREAD",
            leg2_symbol="NIFTY25700CE",
            leg2_side="SELL",
            leg2_entry_price=55.0,
            leg2_exit_price=40.0,
            spread_width=200,
            net_premium=65.0,
            max_profit=26325.0,
            max_loss=12675.0,
        )
        assert trade.trade_type == "DEBIT_SPREAD"
        assert trade.leg2_symbol == "NIFTY25700CE"
        assert trade.leg2_side == "SELL"
        assert trade.spread_width == 200

    def test_credit_spread_fields(self):
        trade = BacktestTrade(
            symbol="NIFTY25400PE",
            side="SELL",
            trade_type="CREDIT_SPREAD",
            leg2_symbol="NIFTY25200PE",
            leg2_side="BUY",
            leg2_entry_price=55.0,
            leg2_exit_price=60.0,
            spread_width=200,
            net_premium=55.0,
            max_profit=3575.0,
            max_loss=9425.0,
        )
        assert trade.trade_type == "CREDIT_SPREAD"
        assert trade.side == "SELL"
        assert trade.leg2_side == "BUY"

    def test_basic_trade_unchanged(self):
        """BASIC trades should work without spread fields."""
        trade = BacktestTrade(
            symbol="NIFTY25500CE",
            side="BUY",
            quantity=130,
            entry_price=120.0,
            exit_price=150.0,
            strategy="FULL",
            pnl=3900.0,
        )
        assert trade.trade_type == "NAKED_BUY"
        assert trade.leg2_symbol == ""
        assert trade.spread_width == 0


class TestPaperTraderSpread:
    """Test paper trader spread order execution."""

    def setup_method(self):
        from src.execution.paper_trader import PaperTrader
        self.pt = PaperTrader(initial_capital=150000, slippage_pct=0)

    def test_spread_order_both_legs_fill(self):
        result = self.pt.place_spread_order(
            leg1_symbol="NIFTY25500CE", leg1_key="NSE_FO|LEG1",
            leg1_qty=65, leg1_side="BUY", leg1_price=120.0,
            leg2_symbol="NIFTY25700CE", leg2_key="NSE_FO|LEG2",
            leg2_qty=65, leg2_side="SELL", leg2_price=55.0,
        )
        assert result["status"] == "success"
        assert "leg1_order_id" in result
        assert "leg2_order_id" in result

    def test_spread_leg2_failure_rolls_back(self):
        # SELL leg adds proceeds to cash, so we need BUY cost > (cash + sell proceeds)
        # Set cash so low that even after sell proceeds, BUY still fails
        self.pt.available_cash = 500  # After SELL: 500 + 110*65=7650, but BUY needs 200*65=13000
        result = self.pt.place_spread_order(
            leg1_symbol="NIFTY25600CE", leg1_key="NSE_FO|L1",
            leg1_qty=65, leg1_side="SELL", leg1_price=110.0,
            leg2_symbol="NIFTY25800CE", leg2_key="NSE_FO|L2",
            leg2_qty=65, leg2_side="BUY", leg2_price=200.0,  # 200 * 65 = 13000 > 7650
        )
        assert result["status"] == "error"
        assert "rolled back" in result["reason"]


class TestEnvConfigSpread:
    """Test spread config vars are loaded correctly."""

    def test_spread_defaults(self):
        from src.config.env_loader import EnvConfig
        cfg = EnvConfig()
        assert cfg.SPREAD_WIDTH == 200
        assert cfg.DEBIT_SPREAD_SL_PCT == 50
        assert cfg.DEBIT_SPREAD_TP_PCT == 70
        assert cfg.CREDIT_SPREAD_SL_MULTIPLIER == 2.0
        assert cfg.CREDIT_SPREAD_TP_PCT == 80
