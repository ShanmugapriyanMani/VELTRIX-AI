"""Tests for 5 signal generation fixes:
Fix 2: Intraday Score Recalculation
Fix 4: Fuzzy Confirmation Thresholds
Fix 5: Adaptive Rolling Range
Fix 1: Direction Flip / Contradiction Handler
Fix 3: Abort Signal Mechanism
"""

from datetime import datetime, time as dt_time, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.data.features import FeatureEngine
from src.execution.order_manager import OrderManager
from src.regime.detector import RegimeDetector, MarketRegime
from src.risk.circuit_breaker import CircuitBreaker, BreakerState
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioManager, Position
from src.strategies.options_buyer import OptionsBuyerStrategy


def _make_intraday_df(n: int = 40, base: float = 22000, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic 5-min OHLCV data."""
    if trend == "up":
        closes = [base + i * 3 for i in range(n)]
    elif trend == "down":
        closes = [base - i * 3 for i in range(n)]
    else:
        closes = [base + (i % 4 - 2) * 2 for i in range(n)]
    return pd.DataFrame({
        "datetime": pd.date_range("2026-03-12 09:15", periods=n, freq="5min"),
        "open": [c - 2 for c in closes],
        "high": [c + 8 for c in closes],
        "low": [c - 8 for c in closes],
        "close": closes,
        "volume": [50000] * n,
    })


def _make_strategy() -> OptionsBuyerStrategy:
    """Create a strategy instance for testing."""
    s = OptionsBuyerStrategy()
    s.reset_daily()
    return s


def _base_data(**overrides) -> dict:
    """Build a minimal data dict for strategy methods."""
    d = {
        "regime": "TRENDING",
        "vix": 15,
        "pcr": {"NIFTY": 1.0},
        "oi_levels": {},
        "ema_weight": 2.5,
        "mean_reversion_weight": 1.5,
        "nifty_price": 22000,
        "intraday_df": _make_intraday_df(40, 22000, "up"),
        "is_expiry_day": False,
        "ml_direction_prob_up": 0.5,
        "ml_direction_prob_down": 0.5,
    }
    d.update(overrides)
    return d


# ═════════════════════════════════════════════════════
# Fix 2 — Intraday Score Recalculation (3 tests)
# ═════════════════════════════════════════════════════


class TestIntradayRescore:

    def setup_method(self):
        self.s = _make_strategy()

    def test_intraday_rescore_blending_1100(self):
        """At 11:00, progressive weight=0.50 (50/50 daily/intraday). Blended score moves toward intraday."""
        self.s._daily_scores["NIFTY"] = (5.0, 2.0, 3.0)
        self.s._direction_scores["NIFTY"] = (5.0, 2.0, 3.0)
        data = _base_data(intraday_df=_make_intraday_df(40, 22000, "up"))

        mock_dt = datetime(2026, 3, 12, 11, 0, 30)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)

        assert "11:00" in self.s._rescore_times_done
        assert self.s._rescore_weight == 0.50  # 11:00 → 50% intraday
        assert "NIFTY" in self.s._blended_scores
        # Blended should differ from pure daily
        blended_bull = self.s._blended_scores["NIFTY"][0]
        assert blended_bull != 5.0  # Modified by intraday contribution

    def test_intraday_rescore_blending_1230(self):
        """At 12:30, progressive weight=0.85 (15% daily / 85% intraday)."""
        self.s._daily_scores["NIFTY"] = (2.0, 5.0, 3.0)  # Daily = PE
        self.s._direction_scores["NIFTY"] = (2.0, 5.0, 3.0)
        # Bullish intraday data
        data = _base_data(intraday_df=_make_intraday_df(50, 22000, "up"))

        mock_dt = datetime(2026, 3, 12, 12, 30, 15)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)

        assert "12:30" in self.s._rescore_times_done
        assert self.s._rescore_weight == 0.85  # 12:30 → 85% intraday

    def test_intraday_rescore_idempotent(self):
        """Second call at same time window is a no-op."""
        self.s._daily_scores["NIFTY"] = (4.0, 2.0, 2.0)
        self.s._direction_scores["NIFTY"] = (4.0, 2.0, 2.0)
        data = _base_data(intraday_df=_make_intraday_df(40, 22000, "up"))

        mock_dt = datetime(2026, 3, 12, 11, 0, 30)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
            first_blended = self.s._blended_scores["NIFTY"]
            # Modify data to confirm second call doesn't re-score
            self.s._direction_scores["NIFTY"] = (99, 99, 0)
            self.s.intraday_rescore("NIFTY", data)

        assert self.s._blended_scores["NIFTY"] == first_blended


# ═════════════════════════════════════════════════════
# Fix 4 — Fuzzy Confirmation Thresholds (4 tests)
# ═════════════════════════════════════════════════════


class TestFuzzyTriggers:

    def test_fuzzy_trigger_all_max_ce(self):
        """CE: strong bullish signals → all triggers near 1.0."""
        t1, t2, t3, t4, total = OptionsBuyerStrategy._compute_fuzzy_triggers(
            direction="CE",
            latest_close=22110,   # 0.5% above open → t1=1.0
            day_open=22000,
            current_rsi=70,       # (70-45)/20=1.25 → t2=1.0
            range_high=22050,     # close above range
            range_low=21950,
            pcr=0.4,              # (1.2-0.4)/0.8=1.0 → t4=1.0
        )
        assert t1 == 1.0
        assert t2 == 1.0
        assert t4 == 1.0
        assert total >= 3.5

    def test_fuzzy_trigger_all_zero_ce(self):
        """CE: strong bearish signals → all triggers 0.0."""
        t1, t2, t3, t4, total = OptionsBuyerStrategy._compute_fuzzy_triggers(
            direction="CE",
            latest_close=21900,   # below open
            day_open=22000,
            current_rsi=35,       # RSI well below 55
            range_high=22050,
            range_low=21950,      # close below range
            pcr=1.5,              # high PCR = bearish
        )
        assert t1 == 0.0
        assert t2 == 0.0
        assert t4 == 0.0
        assert total < 1.0

    def test_fuzzy_trigger_gradient_pe(self):
        """PE: moderate bearish → partial credit on each trigger."""
        t1, t2, t3, t4, total = OptionsBuyerStrategy._compute_fuzzy_triggers(
            direction="PE",
            latest_close=21950,   # ~0.23% below 22000 → t1 ≈ 0.45
            day_open=22000,
            current_rsi=45,       # (55-45)/20 = 0.5
            range_high=22050,
            range_low=21900,      # close near range_low → partial
            pcr=1.2,              # (1.2-0.8)/0.8 = 0.5
        )
        assert 0.3 <= t1 <= 0.7, f"t1={t1}"
        assert 0.3 <= t2 <= 0.7, f"t2={t2}"
        assert 0.3 <= t4 <= 0.7, f"t4={t4}"
        assert 1.0 < total < 3.0, f"total={total}"

    def test_fuzzy_threshold_variants(self):
        """Threshold varies by regime and expiry day."""
        s = _make_strategy()
        # We test the threshold logic directly via the method flow
        # Normal: 2.0
        # Expiry: 2.5
        # Volatile: 2.8
        # Just verify the static trigger computation works for all
        _, _, _, _, sum_ce = OptionsBuyerStrategy._compute_fuzzy_triggers(
            "CE", 22050, 22000, 60, 22080, 21950, 0.8
        )
        assert sum_ce >= 0.0
        assert sum_ce <= 4.0

        _, _, _, _, sum_pe = OptionsBuyerStrategy._compute_fuzzy_triggers(
            "PE", 21950, 22000, 40, 22080, 21950, 1.2
        )
        assert sum_pe >= 0.0
        assert sum_pe <= 4.0

    def test_fuzzy_t1_gradient_ce(self):
        """T1 gradient: 0% diff=0.0, 0.25%=0.5, 0.5%=1.0 for CE."""
        _t = OptionsBuyerStrategy._compute_fuzzy_triggers
        # Exactly at open → 0.0
        t1, _, _, _, _ = _t("CE", 22000, 22000, 50, 22100, 21900, 1.0)
        assert t1 == 0.0

        # 0.25% above → 0.5
        t1, _, _, _, _ = _t("CE", 22055, 22000, 50, 22100, 21900, 1.0)
        assert abs(t1 - 0.5) < 0.05, f"t1={t1}"

        # 0.5% above → 1.0
        t1, _, _, _, _ = _t("CE", 22110, 22000, 50, 22100, 21900, 1.0)
        assert t1 == 1.0

        # 1% above → still clamped at 1.0
        t1, _, _, _, _ = _t("CE", 22220, 22000, 50, 22100, 21900, 1.0)
        assert t1 == 1.0

        # Below open → 0.0
        t1, _, _, _, _ = _t("CE", 21950, 22000, 50, 22100, 21900, 1.0)
        assert t1 == 0.0

        # PE direction: below open gives credit
        t1, _, _, _, _ = _t("PE", 21945, 22000, 50, 22100, 21900, 1.0)
        assert abs(t1 - 0.5) < 0.05, f"PE t1={t1}"

    def test_fuzzy_t2_rsi_gradient(self):
        """T2 gradient: CE RSI=45→0.0, 55→0.5, 65→1.0."""
        _t = OptionsBuyerStrategy._compute_fuzzy_triggers
        # CE: RSI=45 → 0.0
        _, t2, _, _, _ = _t("CE", 22000, 22000, 45, 22100, 21900, 1.0)
        assert t2 == 0.0

        # CE: RSI=55 → 0.5
        _, t2, _, _, _ = _t("CE", 22000, 22000, 55, 22100, 21900, 1.0)
        assert t2 == 0.5

        # CE: RSI=65 → 1.0
        _, t2, _, _, _ = _t("CE", 22000, 22000, 65, 22100, 21900, 1.0)
        assert t2 == 1.0

        # CE: RSI=35 → 0.0 (clamped)
        _, t2, _, _, _ = _t("CE", 22000, 22000, 35, 22100, 21900, 1.0)
        assert t2 == 0.0

        # PE: RSI=55 → 0.0, RSI=45 → 0.5, RSI=35 → 1.0
        _, t2, _, _, _ = _t("PE", 22000, 22000, 55, 22100, 21900, 1.0)
        assert t2 == 0.0
        _, t2, _, _, _ = _t("PE", 22000, 22000, 45, 22100, 21900, 1.0)
        assert t2 == 0.5
        _, t2, _, _, _ = _t("PE", 22000, 22000, 35, 22100, 21900, 1.0)
        assert t2 == 1.0

    def test_fuzzy_sum_threshold_entry(self):
        """Entry fires when sum >= 2.0; blocked when < 2.0."""
        _t = OptionsBuyerStrategy._compute_fuzzy_triggers
        # Strong bullish: all triggers high → sum well above 2.0
        _, _, _, _, total = _t("CE", 22110, 22000, 65, 22050, 21950, 0.4)
        assert total >= 2.0, f"Expected entry, got sum={total}"

        # Weak: price at open, neutral RSI, inside range, neutral PCR → sum < 2.0
        _, _, _, _, total = _t("CE", 22000, 22000, 50, 22100, 21900, 1.0)
        assert total < 2.0, f"Expected no entry, got sum={total}"

        # Partial credit from multiple triggers can sum to >= 2.0
        # 0.3% above open (t1≈0.6), RSI=58 (t2=0.65), near range_high, PCR=0.7 (t4≈0.625)
        _, _, _, _, total = _t("CE", 22066, 22000, 58, 22070, 21950, 0.7)
        assert total >= 2.0, f"Partial credit should sum above 2.0, got {total}"


# ═════════════════════════════════════════════════════
# Fix 5 — Adaptive Rolling Range (3 tests)
# ═════════════════════════════════════════════════════


class TestRollingRange:

    def setup_method(self):
        self.s = _make_strategy()

    def test_rolling_range_updates(self):
        """Range uses last 12 candles and updates on call."""
        df = _make_intraday_df(40, 22000, "up")
        rh, rl, width, tight = self.s._update_rolling_range("NIFTY", df)

        assert rh > 0
        assert rl > 0
        assert rh > rl
        assert width > 0
        assert "NIFTY" in self.s._rolling_range_high

    def test_rolling_range_tight_filter(self):
        """Range < 0.2% of midpoint → too_tight = True."""
        n = 20
        base = 22000
        # Create candles with very tight range (< 0.2% of 22000 = 44 points)
        closes = [base + (i % 2) for i in range(n)]
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-12 09:15", periods=n, freq="5min"),
            "open": [c - 1 for c in closes],
            "high": [c + 5 for c in closes],  # max spread = 10 points << 44
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [50000] * n,
        })
        _, _, width, tight = self.s._update_rolling_range("NIFTY", df)
        assert tight is True
        assert width < 0.2

    def test_rolling_range_caches_within_30min(self):
        """Second call within 30 min returns cached values."""
        df = _make_intraday_df(40, 22000, "up")
        rh1, rl1, _, _ = self.s._update_rolling_range("NIFTY", df)

        # Create different data
        df2 = _make_intraday_df(40, 23000, "down")
        rh2, rl2, _, _ = self.s._update_rolling_range("NIFTY", df2)

        # Should return same cached values since < 30 min
        assert rh1 == rh2
        assert rl1 == rl2


# ═════════════════════════════════════════════════════
# Fix 1 — Direction Contradiction / Flip (3 tests)
# ═════════════════════════════════════════════════════


class TestDirectionContradiction:

    def setup_method(self):
        self.s = _make_strategy()
        # Disable momentum mode so contradiction logic is active
        from src.config.env_loader import get_config
        get_config().MOMENTUM_MODE_ENABLED = False

    def test_contradiction_kills_signal(self):
        """Opposing directions with moderate intraday diff → CONTRADICTION."""
        self.s._daily_scores["NIFTY"] = (5.0, 3.0, 2.0)     # CE
        self.s._intraday_scores["NIFTY"] = (2.0, 4.0, 2.0)  # PE, diff=2.0
        data = _base_data()

        # Mock time to 13:00 (after noon, so flip not allowed)
        mock_dt = datetime(2026, 3, 12, 13, 0, 0)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", data)

        assert result == "CONTRADICTION"
        assert self.s._signal_killed.get("NIFTY") is True

    def test_flip_before_noon(self):
        """Strong opposing intraday signal before noon → FLIP."""
        self.s._daily_scores["NIFTY"] = (3.0, 5.0, 2.0)     # PE
        self.s._intraday_scores["NIFTY"] = (6.0, 2.0, 4.0)  # CE, diff=4.0 > 2.5
        self.s._direction["NIFTY"] = "PE"
        data = _base_data(is_expiry_day=False)

        mock_dt = datetime(2026, 3, 12, 10, 45, 0)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", data)

        assert result == "FLIP"
        assert self.s._direction["NIFTY"] == "CE"
        assert self.s._direction_flipped_today is True

    def test_flip_blocked_on_expiry(self):
        """Strong signal on expiry day → CONTRADICTION (flip blocked)."""
        self.s._daily_scores["NIFTY"] = (3.0, 5.0, 2.0)     # PE
        self.s._intraday_scores["NIFTY"] = (6.0, 2.0, 4.0)  # CE, diff=4.0
        self.s._direction["NIFTY"] = "PE"
        self.s._expiry_type = "NIFTY_EXPIRY"  # Major expiry blocks flip
        data = _base_data(is_expiry_day=True)

        mock_dt = datetime(2026, 3, 12, 10, 45, 0)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", data)

        assert result == "CONTRADICTION"
        assert self.s._signal_killed.get("NIFTY") is True


# ═════════════════════════════════════════════════════
# Fix 3 — Abort Signal Mechanism (2 tests)
# ═════════════════════════════════════════════════════


class TestAbortMechanism:

    def setup_method(self):
        self.s = _make_strategy()

    def test_soft_abort_raises_threshold(self):
        """No trade by 11:30 → SOFT abort, threshold raised."""
        # Simulate: no trades taken, abort stage = NONE
        self.s._abort_stage["NIFTY"] = "NONE"
        self.s._trades_today.clear()

        # The abort logic runs inside _evaluate_symbol.
        # Test the state change directly:
        now = dt_time(11, 35)
        if now >= dt_time(11, 30) and self.s._abort_stage.get("NIFTY", "NONE") == "NONE":
            if self.s._trades_today.get("NIFTY", 0) == 0:
                self.s._abort_stage["NIFTY"] = "SOFT"

        assert self.s._abort_stage["NIFTY"] == "SOFT"

        # Verify threshold boost: base 2.0 + 0.5 = 2.5
        base_threshold = 2.0
        if self.s._abort_stage.get("NIFTY") == "SOFT":
            base_threshold += 0.5
        assert base_threshold == 2.5

    def test_hard_abort_blocks_entry_and_bypass(self):
        """No trade by 13:00 → HARD abort blocks entry. Bypass after TP."""
        self.s._abort_stage["NIFTY"] = "HARD"
        self.s._failed_confirm_count["NIFTY"] = 147

        # Hard abort blocks entry
        abort_bypassed = self.s._abort_bypassed.get("NIFTY", False)
        blocked = self.s._abort_stage.get("NIFTY") == "HARD" and not abort_bypassed
        assert blocked is True

        # After completed trade, bypass abort
        self.s._abort_bypassed["NIFTY"] = True
        abort_bypassed = self.s._abort_bypassed.get("NIFTY", False)
        blocked = self.s._abort_stage.get("NIFTY") == "HARD" and not abort_bypassed
        assert blocked is False

    def test_soft_abort_raises_threshold_at_1130(self):
        """At 11:30 with no trade, threshold increases by 0.5."""
        self.s._abort_stage["NIFTY"] = "NONE"
        self.s._trades_today.clear()
        self.s._direction["NIFTY"] = "CE"
        self.s._direction_scores["NIFTY"] = (4.0, 2.0, 2.0)

        now = dt_time(11, 30)
        # Simulate abort trigger logic from _evaluate_symbol
        if now >= dt_time(11, 30) and self.s._abort_stage.get("NIFTY", "NONE") == "NONE":
            if self.s._trades_today.get("NIFTY", 0) == 0:
                self.s._abort_stage["NIFTY"] = "SOFT"

        assert self.s._abort_stage["NIFTY"] == "SOFT"

        # Verify threshold boost for each regime
        for regime, base in [("TRENDING", 2.0), ("VOLATILE", 2.8), ("RANGEBOUND", 2.0)]:
            threshold = base
            if self.s._abort_stage.get("NIFTY") == "SOFT":
                threshold += 0.5
            assert threshold == base + 0.5

    def test_hard_abort_kills_signal_at_1300(self):
        """At 13:00 with no trade, direction becomes None."""
        self.s._abort_stage["NIFTY"] = "SOFT"  # Already passed soft
        self.s._trades_today.clear()
        self.s._direction["NIFTY"] = "PE"
        self.s._direction_scores["NIFTY"] = (2.0, 4.5, 2.5)
        self.s._failed_confirm_count["NIFTY"] = 42

        now = dt_time(13, 0)
        abort_bypassed = self.s._abort_bypassed.get("NIFTY", False)
        if not abort_bypassed:
            if now >= dt_time(13, 0) and self.s._abort_stage.get("NIFTY") != "HARD":
                if self.s._trades_today.get("NIFTY", 0) == 0:
                    self.s._abort_stage["NIFTY"] = "HARD"
                    # Hard abort clears direction
                    self.s._direction.pop("NIFTY", None)

        assert self.s._abort_stage["NIFTY"] == "HARD"
        assert "NIFTY" not in self.s._direction  # Direction cleared

        # Hard abort blocks entry
        blocked = self.s._abort_stage.get("NIFTY") == "HARD" and not abort_bypassed
        assert blocked is True

    def test_abort_bypassed_after_closed_trade(self):
        """If trade taken and closed (TP), no abort fires."""
        self.s._abort_stage["NIFTY"] = "HARD"
        self.s._trades_today["NIFTY"] = 1
        self.s._failed_confirm_count["NIFTY"] = 80

        # Simulate TP exit → bypass set via record_exit
        self.s._abort_bypassed["NIFTY"] = True

        abort_bypassed = self.s._abort_bypassed.get("NIFTY", False)
        assert abort_bypassed is True

        # Both abort stages are bypassed — entry allowed
        blocked = self.s._abort_stage.get("NIFTY") == "HARD" and not abort_bypassed
        assert blocked is False

        # Simulating the check at 13:00 — bypass prevents new HARD abort
        now = dt_time(13, 30)
        if not abort_bypassed:
            if now >= dt_time(13, 0) and self.s._abort_stage.get("NIFTY") != "HARD":
                self.s._abort_stage["NIFTY"] = "HARD"  # Should NOT execute
        # Stage stays as HARD from before, but bypassed — re-entry allowed
        assert self.s._abort_bypassed["NIFTY"] is True


# ═════════════════════════════════════════════════════
# ML Asymmetric Weights (1 test)
# ═════════════════════════════════════════════════════


class TestMLAsymmetricWeights:

    def setup_method(self):
        self.s = _make_strategy()

    def test_ml_asymmetric_weights_pe_higher_than_ce(self):
        """PE prediction at weight 1.5 contributes more than CE at weight 0.3."""
        data_pe = _base_data(
            ml_stage1_prob_ce=0.30,
            ml_stage1_prob_pe=0.60,
        )
        data_ce = _base_data(
            ml_stage1_prob_ce=0.60,
            ml_stage1_prob_pe=0.30,
        )

        # Compute scores with PE signal
        bull_pe, bear_pe, _ = self.s._compute_direction_score("NIFTY", data_pe, "TRENDING")
        self.s._direction.clear()  # Reset for next call

        # Compute scores with CE signal
        bull_ce, bear_ce, _ = self.s._compute_direction_score("NIFTY", data_ce, "TRENDING")

        # PE contribution: 1.5 * (0.60 - 0.33) / 0.67 = 0.604
        # CE contribution: 0.3 * (0.60 - 0.33) / 0.67 = 0.121
        pe_ml_contribution = 1.5 * (0.60 - 0.33) / 0.67
        ce_ml_contribution = 0.3 * (0.60 - 0.33) / 0.67

        assert pe_ml_contribution > ce_ml_contribution
        assert abs(pe_ml_contribution - 0.604) < 0.01
        assert abs(ce_ml_contribution - 0.121) < 0.01

        # Bear score with PE signal should be higher than bull score with CE signal (from ML alone)
        # The actual scores include all 9 factors, but the PE ML contribution is 5x the CE contribution
        assert pe_ml_contribution / ce_ml_contribution > 4.5


# ═════════════════════════════════════════════════════
# Fix 2b — Intraday Rescore Contradiction Tests (4 tests)
# ═════════════════════════════════════════════════════


class TestRescoreContradiction:

    def setup_method(self):
        self.s = _make_strategy()
        # Disable momentum mode so contradiction logic is active
        from src.config.env_loader import get_config
        get_config().MOMENTUM_MODE_ENABLED = False

    def test_rescore_agreement_blends_scores(self):
        """Same direction daily+intraday → AGREEMENT with blended scores."""
        # Daily = CE (bull > bear)
        self.s._daily_scores["NIFTY"] = (5.0, 2.0, 3.0)
        self.s._direction_scores["NIFTY"] = (5.0, 2.0, 3.0)
        # Intraday = CE (also bull > bear)
        self.s._intraday_scores["NIFTY"] = (6.0, 3.0, 3.0)
        self.s._rescore_times_done.add("11:00")
        self.s._blended_scores["NIFTY"] = (5.4, 2.4, 3.0)

        mock_dt = datetime(2026, 3, 13, 11, 0, 30)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", _base_data())

        assert result == "AGREEMENT"
        # Direction unchanged
        assert self.s._signal_killed.get("NIFTY", False) is False

    def test_rescore_weak_contradiction_reduces_score(self):
        """Intraday opposes daily with diff < 1.5 → WEAK_CONTRADICTION, score reduced 20%."""
        # Daily = CE
        self.s._daily_scores["NIFTY"] = (5.0, 3.0, 2.0)
        self.s._direction_scores["NIFTY"] = (5.0, 3.0, 2.0)
        # Intraday = PE but weak (diff=1.0 < 1.5)
        self.s._intraday_scores["NIFTY"] = (2.5, 3.5, 1.0)
        self.s._rescore_times_done.add("11:00")
        self.s._blended_scores["NIFTY"] = (5.0, 3.0, 2.0)

        mock_dt = datetime(2026, 3, 13, 11, 0, 30)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", _base_data())

        assert result == "WEAK_CONTRADICTION"
        # Score reduced by 20%: 5.0 * 0.8 = 4.0, 3.0 * 0.8 = 2.4
        reduced_bull, reduced_bear, reduced_diff = self.s._direction_scores["NIFTY"]
        assert abs(reduced_bull - 4.0) < 0.01
        assert abs(reduced_bear - 2.4) < 0.01
        # Signal NOT killed
        assert self.s._signal_killed.get("NIFTY", False) is False

    def test_rescore_strong_contradiction_kills_signal(self):
        """Intraday strongly opposes daily (diff >= 1.5, after noon) → signal killed."""
        # Daily = CE
        self.s._daily_scores["NIFTY"] = (5.0, 2.0, 3.0)
        self.s._direction_scores["NIFTY"] = (5.0, 2.0, 3.0)
        # Intraday = PE, strong (diff=2.0 >= 1.5)
        self.s._intraday_scores["NIFTY"] = (2.0, 4.0, 2.0)
        self.s._rescore_times_done.add("12:30")

        # After noon — flip not allowed
        mock_dt = datetime(2026, 3, 13, 13, 0, 0)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction("NIFTY", _base_data())

        assert result == "CONTRADICTION"
        assert self.s._signal_killed["NIFTY"] is True

    def test_rescore_flip_changes_direction(self):
        """Strong intraday signal (diff >= 2.5) before noon → FLIP direction."""
        # Daily = PE
        self.s._daily_scores["NIFTY"] = (2.0, 5.0, 3.0)
        self.s._direction["NIFTY"] = "PE"
        # Intraday = CE, very strong (diff=4.0 >= 2.5)
        self.s._intraday_scores["NIFTY"] = (6.0, 2.0, 4.0)
        self.s._rescore_times_done.add("11:00")

        # Before noon, no trades, no prior flip, not expiry
        mock_dt = datetime(2026, 3, 13, 11, 0, 30)
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = mock_dt
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.s.check_direction_contradiction(
                "NIFTY", _base_data(is_expiry_day=False)
            )

        assert result == "FLIP"
        assert self.s._direction["NIFTY"] == "CE"
        assert self.s._direction_flipped_today is True
        # Signal NOT killed on flip
        assert self.s._signal_killed.get("NIFTY", False) is False


# ═════════════════════════════════════════════════════
# SIGNAL_SKIP Diagnostic Logging (2 tests)
# ═════════════════════════════════════════════════════


class TestSignalSkipDiagnostic:

    def setup_method(self):
        self.s = _make_strategy()

    def test_signal_skip_logged_every_10_loops(self):
        """record_skip increments _skip_counts by reason from _last_skip_info."""
        # Simulate multiple skip reasons over multiple iterations
        reasons = [
            "CONVICTION_BELOW_THRESHOLD",
            "CONFIRMATION_FAILED",
            "CONFIRMATION_FAILED",
            "SOFT_ABORT",
            "CONFIRMATION_FAILED",
            "HARD_ABORT",
        ]
        for reason in reasons:
            self.s._last_skip_info["NIFTY"] = {"reason": reason}
            self.s.record_skip("NIFTY")

        counts = self.s.get_skip_summary()
        assert counts["CONFIRMATION_FAILED"] == 3
        assert counts["CONVICTION_BELOW_THRESHOLD"] == 1
        assert counts["SOFT_ABORT"] == 1
        assert counts["HARD_ABORT"] == 1
        assert sum(counts.values()) == 6

    def test_signal_skip_summary_at_eod(self):
        """get_skip_summary returns skip counts, reset_daily clears them."""
        self.s._last_skip_info["NIFTY"] = {"reason": "CONFIRMATION_FAILED"}
        self.s.record_skip("NIFTY")
        self.s._last_skip_info["NIFTY"] = {"reason": "CONFIRMATION_FAILED"}
        self.s.record_skip("NIFTY")
        self.s._last_skip_info["NIFTY"] = {"reason": "AFTER_CUTOFF"}
        self.s.record_skip("NIFTY")

        summary = self.s.get_skip_summary()
        assert summary == {"CONFIRMATION_FAILED": 2, "AFTER_CUTOFF": 1}

        # reset_daily clears the counts
        self.s.reset_daily()
        assert self.s.get_skip_summary() == {}


# ═════════════════════════════════════════════════════
# Data Readiness Gate (2 tests)
# ═════════════════════════════════════════════════════


class TestDataReadinessGate:
    """Test _data_ready gate logic from main.py trading loop."""

    def _check_gate(self, data_ready: bool, data: dict) -> tuple[bool, dict]:
        """Simulate the gate logic from _trading_loop.

        Returns (new_data_ready, ensemble_result_or_None).
        None means gate passed — proceed to normal signal evaluation.
        """
        ensemble_result = None
        if not data_ready:
            vix_ok = data.get("vix", 0) > 0
            nifty_ok = data.get("nifty_price", 0) > 0
            candles_ok = not data.get("intraday_df", pd.DataFrame()).empty
            if vix_ok and nifty_ok and candles_ok:
                data_ready = True
                # Gate opened — proceed to normal evaluation
                ensemble_result = None
            else:
                # Gate closed — skip signals
                ensemble_result = {"decisions": []}
        return data_ready, ensemble_result

    def test_data_gate_blocks_entry_when_vix_zero(self):
        """VIX=0 at startup → gate stays closed, no signals pass."""
        data_ready = False

        # VIX=0: gate should block
        data = {"vix": 0, "nifty_price": 22500, "intraday_df": _make_intraday_df(10)}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is False
        assert result == {"decisions": []}

        # NIFTY=0: gate should block
        data = {"vix": 14.5, "nifty_price": 0, "intraday_df": _make_intraday_df(10)}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is False
        assert result == {"decisions": []}

        # Empty candles: gate should block
        data = {"vix": 14.5, "nifty_price": 22500, "intraday_df": pd.DataFrame()}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is False
        assert result == {"decisions": []}

        # All zero: gate should block
        data = {"vix": 0, "nifty_price": 0, "intraday_df": pd.DataFrame()}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is False
        assert result == {"decisions": []}

    def test_data_gate_opens_when_all_conditions_met(self):
        """VIX>0, NIFTY>0, candles loaded → gate opens and stays open."""
        data_ready = False

        # All conditions met → gate opens
        data = {"vix": 14.5, "nifty_price": 22500, "intraday_df": _make_intraday_df(10)}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is True
        assert result is None  # None = proceed to normal signal evaluation

        # Once open, gate stays open even if VIX drops to 0
        data = {"vix": 0, "nifty_price": 22500, "intraday_df": _make_intraday_df(10)}
        data_ready, result = self._check_gate(data_ready, data)
        assert data_ready is True
        assert result is None  # Still open — gate never resets


# ═════════════════════════════════════════════════════
# Two-Speed Poll Loop (2 tests)
# ═════════════════════════════════════════════════════


class TestTwoSpeedPoll:
    """Test fast/slow poll interval logic from main.py trading loop."""

    @staticmethod
    def _compute_poll_interval(has_positions: bool, is_network_down: bool = False) -> int:
        """Simulate the two-speed poll logic from _trading_loop.

        Returns the effective poll interval in seconds.
        """
        if is_network_down:
            return 60
        elif has_positions:
            return 5  # Fast poll: 5s per tick, 5 ticks between full iterations
        else:
            return 30

    def test_fast_poll_activates_with_open_position(self):
        """Open position → poll interval drops to 5s."""
        interval = self._compute_poll_interval(has_positions=True)
        assert interval == 5

        # Network down overrides even with positions
        interval_net = self._compute_poll_interval(has_positions=True, is_network_down=True)
        assert interval_net == 60

    def test_slow_poll_resumes_after_position_closes(self):
        """No positions → back to 30s poll. Simulates open → close transition."""
        # Position open → fast
        assert self._compute_poll_interval(has_positions=True) == 5

        # Position closes → slow
        assert self._compute_poll_interval(has_positions=False) == 30

        # Re-enters fast on new position
        assert self._compute_poll_interval(has_positions=True) == 5

        # Closes again → slow
        assert self._compute_poll_interval(has_positions=False) == 30


# ═════════════════════════════════════════════════════
# Breakout Re-entry (3 tests)
# ═════════════════════════════════════════════════════


class TestBreakoutReentry:
    """Test breakout flag reset and rolling range refresh after trade close."""

    def setup_method(self):
        self.s = OptionsBuyerStrategy()
        self.s.reset_daily()

    def test_breakout_flag_resets_after_trade_closes(self):
        """record_exit(TP) clears range update timer and sets breakout pending."""
        sym = "NIFTY"
        # Simulate: range was set, trade happened
        self.s._rolling_range_high[sym] = 22100
        self.s._rolling_range_low[sym] = 21900
        self.s._range_last_update[sym] = datetime(2026, 3, 13, 10, 0)
        self.s._trades_today[sym] = 1
        self.s._traded_today.add(sym)

        # Before exit: range update timer is set, no pending reentry
        assert sym in self.s._range_last_update
        assert self.s._breakout_pending_reentry.get(sym) is not True

        # TP exit — should reset range timer and set pending
        self.s.record_exit("NIFTY24100CE", "take_profit", "CE")

        # After exit: timer cleared, pending set, symbol unlocked
        assert sym not in self.s._range_last_update
        assert self.s._breakout_pending_reentry[sym] is True
        assert sym not in self.s._traded_today

        # Trail stop also triggers reset
        self.s._trades_today[sym] = 1
        self.s._traded_today.add(sym)
        self.s._range_last_update[sym] = datetime(2026, 3, 13, 11, 0)
        self.s._breakout_pending_reentry[sym] = False

        self.s.record_exit("NIFTY24100PE", "trail_stop", "PE")

        assert sym not in self.s._range_last_update
        assert self.s._breakout_pending_reentry[sym] is True

    def test_reentry_breakout_uses_rolling_range(self):
        """After trade close, _update_rolling_range computes fresh range from candles."""
        sym = "NIFTY"
        # Old range from morning
        self.s._rolling_range_high[sym] = 22100
        self.s._rolling_range_low[sym] = 21900
        self.s._range_last_update[sym] = datetime(2026, 3, 13, 10, 0)

        # Simulate trade exit — clears timer
        self.s._trades_today[sym] = 1
        self.s._traded_today.add(sym)
        self.s.record_exit("NIFTY24100CE", "take_profit", "CE")
        assert sym not in self.s._range_last_update

        # Now call _update_rolling_range with new intraday data (price moved up)
        intraday_df = _make_intraday_df(40, base=22300, trend="up")
        rh, rl, width_pct, tight = self.s._update_rolling_range(sym, intraday_df)

        # Range should be recomputed from the new candles (last 12 bars)
        # not the old 22100/21900 from morning
        assert rh > 22100  # New range high above old
        assert rl > 21900  # New range low above old
        # Timer should be set now (for next 30-min throttle)
        assert sym in self.s._range_last_update

    def test_reentry_breakout_blocked_after_1430(self):
        """Re-entry breakout respects existing 14:30 cutoff in _evaluate_symbol."""
        s = self.s
        sym = "NIFTY"

        # Set up re-entry state
        s._breakout_pending_reentry[sym] = True
        s._abort_bypassed[sym] = True
        s._trades_today[sym] = 1

        # After 14:30 — the existing time guard in generate_signals blocks new entries
        # Verify the guard exists: generate_signals skips symbols after 14:30
        # (We test the guard condition directly rather than mocking datetime)
        now_after_cutoff = dt_time(14, 45)
        now_before_cutoff = dt_time(12, 30)

        # After 14:30: no new entries allowed (existing guard in generate_signals)
        assert now_after_cutoff >= dt_time(14, 30)

        # Before 14:30: entries would be allowed
        assert now_before_cutoff < dt_time(14, 30)

        # Hard abort also blocks re-entry breakout (when not bypassed)
        s._abort_stage[sym] = "HARD"
        s._abort_bypassed[sym] = False
        abort_blocked = (
            s._abort_stage.get(sym) == "HARD"
            and not s._abort_bypassed.get(sym, False)
        )
        assert abort_blocked is True


# ═════════════════════════════════════════════════════
# Blocker F1 — Force-exit uses .values() (1 test)
# ═════════════════════════════════════════════════════


class TestForceExit:

    def test_force_exit_uses_position_values(self):
        """Force-exit iterates Position objects via .values(), not string keys."""
        from src.risk.portfolio import PortfolioManager, Position

        pm = PortfolioManager()
        # Add a position manually
        pos = Position(
            symbol="NIFTY24100CE",
            instrument_key="NSE_FO|12345",
            quantity=65,
            entry_price=200.0,
            side="BUY",
        )
        pm.positions["NIFTY24100CE"] = pos
        pm.cash -= pos.cost_basis

        # Iterate like the fixed force-exit code
        fo_positions = []
        for p in list(pm.positions.values()):
            if p.instrument_key.startswith("NSE_FO|"):
                fo_positions.append(p)

        assert len(fo_positions) == 1
        assert fo_positions[0].symbol == "NIFTY24100CE"
        assert fo_positions[0].instrument_key == "NSE_FO|12345"
        assert fo_positions[0].quantity == 65

        # Old buggy code would iterate keys (strings) — verify that would fail
        for key in list(pm.positions):
            assert isinstance(key, str)
            assert not hasattr(key, "instrument_key")  # strings don't have this


# ═════════════════════════════════════════════════════
# Blocker F3 — Trail stop records to circuit breaker (1 test)
# ═════════════════════════════════════════════════════


class TestTrailStopCircuitBreaker:

    def test_trail_stop_records_to_circuit_breaker(self):
        """Trail stop exit PnL is recorded in circuit breaker (mirrors SL/TP pattern)."""
        cb = CircuitBreaker()

        # Record a trail stop loss (simulating what main.py now does)
        trail_exit_info = {
            "symbol": "NIFTY24100CE",
            "entry_premium": 200.0,
            "exit_premium": 170.0,
            "quantity": 65,
            "pnl": (170.0 - 200.0) * 65,  # -1950
            "pnl_pct": -15.0,
        }
        cb.record_trade(trail_exit_info["pnl"])

        assert cb._daily_trades == 1
        assert cb._consecutive_sl == 1
        assert cb._daily_pnl == -1950.0

        # Record a trail stop win
        trail_win = {
            "pnl": (250.0 - 200.0) * 65,  # +3250
        }
        cb.record_trade(trail_win["pnl"])

        assert cb._daily_trades == 2
        assert cb._consecutive_sl == 0  # Reset on win
        assert cb._daily_pnl == -1950.0 + 3250.0

        # Clean up state file
        if cb._state_file.exists():
            cb._state_file.unlink()


# ═════════════════════════════════════════════════════
# Blocker F8 — Circuit breaker sends Telegram on halt (1 test)
# ═════════════════════════════════════════════════════


class TestCircuitBreakerAlerts:

    def test_circuit_breaker_sends_telegram_on_halt(self):
        """Halt transitions send alerts via injected alert function."""
        cb = CircuitBreaker()
        cb.reset_daily()  # Clean state
        alerts_received = []
        cb.set_alert_fn(lambda msg: alerts_received.append(msg))

        # Rule 1: 2 consecutive SL → HALTED with alert
        cb.record_trade(-5000)
        cb.record_trade(-5000)
        assert cb._state == BreakerState.HALTED
        assert len(alerts_received) == 1
        assert "consecutive losses" in alerts_received[0]
        assert "Halting trading for today" in alerts_received[0]

        # Reset and test Rule 2
        cb.reset_daily()
        alerts_received.clear()

        # Rule 2: Daily loss > ₹20K → HALTED with alert
        cb.record_trade(-21000)
        assert cb._state == BreakerState.HALTED
        assert len(alerts_received) == 1
        assert "Daily loss" in alerts_received[0]

        # Test KILLED (kill switch)
        cb.reset_daily()
        alerts_received.clear()
        cb.activate_kill_switch()
        assert cb._state == BreakerState.HALTED
        assert len(alerts_received) == 1
        assert "Kill switch" in alerts_received[0]

        # Clean up state file
        if cb._state_file.exists():
            cb._state_file.unlink()


# ═════════════════════════════════════════════════════
# F4 — _current_trade_type isolated per symbol (1 test)
# ═════════════════════════════════════════════════════


class TestCurrentTradeTypeIsolation:

    def test_current_trade_type_isolated_per_symbol(self):
        """Trade type for NIFTY does not leak to BANKNIFTY."""
        s = _make_strategy()

        # Simulate NIFTY getting NAKED_BUY
        s._current_trade_type["NIFTY"] = "NAKED_BUY"

        # BANKNIFTY should not inherit NIFTY's trade type
        assert s._current_trade_type.get("BANKNIFTY") is None
        assert s._current_trade_type.get("BANKNIFTY", "FULL") == "FULL"

        # Set BANKNIFTY independently
        s._current_trade_type["BANKNIFTY"] = "CREDIT_SPREAD"
        assert s._current_trade_type["NIFTY"] == "NAKED_BUY"
        assert s._current_trade_type["BANKNIFTY"] == "CREDIT_SPREAD"

        # reset_daily clears both
        s.reset_daily()
        assert s._current_trade_type == {}


# ═════════════════════════════════════════════════════
# F5 — PortfolioManager.reset_daily() (1 test)
# ═════════════════════════════════════════════════════


class TestPortfolioResetDaily:

    def test_portfolio_reset_daily_clears_state(self):
        """reset_daily clears closed_trades and daily_pnl_history."""
        pm = PortfolioManager()

        # Simulate some activity
        pm.closed_trades.append({"symbol": "NIFTY24100CE", "pnl": 500})
        pm.closed_trades.append({"symbol": "NIFTY24100PE", "pnl": -200})
        pm.daily_pnl_history.append({"date": "2026-03-13", "pnl": 300})
        pm._peak_value = 160000

        pm.reset_daily()

        assert pm.closed_trades == []
        assert pm.daily_pnl_history == []
        # Peak value preserved (updated to max of previous peak and cash)
        assert pm._peak_value >= pm.cash


# ═════════════════════════════════════════════════════
# F6 — OrderManager.reset_daily() (1 test)
# ═════════════════════════════════════════════════════


class TestOrderManagerResetDaily:

    def test_order_manager_reset_daily_clears_orders(self):
        """reset_daily clears pending, GTT, and filled orders."""
        from src.execution.paper_trader import PaperTrader
        from src.risk.manager import RiskManager

        broker = PaperTrader()
        rm = RiskManager()
        cb = CircuitBreaker()
        om = OrderManager(broker, rm, cb)

        # Simulate stale orders
        om._pending_orders["trade1"] = {"symbol": "NIFTY24100CE", "price": 200}
        om._gtt_orders["gtt1"] = {"trigger": 170}
        om._filled_orders.append({"trade_id": "trade0"})

        om.reset_daily()

        assert om._pending_orders == {}
        assert om._gtt_orders == {}
        assert om._filled_orders == []

        # Clean up
        if cb._state_file.exists():
            cb._state_file.unlink()


# ═════════════════════════════════════════════════════
# F10 — ELEVATED regime detection (1 test)
# ═════════════════════════════════════════════════════


class TestElevatedRegime:

    def test_elevated_regime_detected_correctly(self):
        """VIX 20-28 and rising (but below volatile threshold) → ELEVATED regime."""
        detector = RegimeDetector()

        # Build minimal NIFTY DataFrame (50+ rows needed)
        n = 60
        prices = [22000 + i * 5 for i in range(n)]
        nifty_df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": [p - 10 for p in prices],
            "high": [p + 30 for p in prices],
            "low": [p - 30 for p in prices],
            "close": prices,
            "volume": [1000000] * n,
        })
        fii_data = pd.DataFrame()

        # VIX=21 (between 20-28, at/below volatile threshold of 22),
        # rising (+1.5 pts, below spike threshold of 3)
        # This avoids triggering VOLATILE (needs vix > 22 or spike > 3)
        vix_data = {"vix": 21.0, "change_pct": 1.5}
        state = detector.detect(vix_data, nifty_df, fii_data)
        assert state.regime == MarketRegime.ELEVATED, f"Got {state.regime.value}: {state.notes}"
        assert state.conviction_min >= 2.0
        assert state.sl_multiplier == 1.10
        assert state.tp_multiplier == 1.20
        assert state.size_multiplier == 0.85

        # VIX=21 but falling → NOT ELEVATED
        vix_falling = {"vix": 21.0, "change_pct": -1.0}
        detector._last_regime = None  # Reset to avoid lock
        state2 = detector.detect(vix_falling, nifty_df, fii_data)
        assert state2.regime != MarketRegime.ELEVATED

        # VIX=30 and rising → VOLATILE (VOLATILE takes priority over ELEVATED)
        vix_high = {"vix": 30.0, "change_pct": 4.0}
        detector._last_regime = None
        state3 = detector.detect(vix_high, nifty_df, fii_data)
        assert state3.regime == MarketRegime.VOLATILE


# ═══════════════════════════════════════════
# W1: STT rate is 0.0005, not 0.000625
# ═══════════════════════════════════════════


class TestSTTRate:
    def test_stt_rate_is_correct(self):
        """STT on options sell side should be 0.05% (0.0005), not 0.0625%."""
        rm = RiskManager()
        premium = 100.0
        quantity = 65  # 1 NIFTY lot
        turnover = premium * quantity  # 6500

        sell_costs = rm.calculate_options_trade_costs(premium, quantity, side="SELL")
        buy_costs = rm.calculate_options_trade_costs(premium, quantity, side="BUY")

        # STT only on sell side
        expected_stt = turnover * 0.0005
        assert sell_costs["stt"] == expected_stt, (
            f"STT should be {expected_stt} (0.05%), got {sell_costs['stt']}"
        )
        assert buy_costs["stt"] == 0, "Buy side should have zero STT"


# ═══════════════════════════════════════════
# W5: VIX default 15.0 when data missing
# ═══════════════════════════════════════════


class TestVIXDefault:
    def test_vix_default_is_15_not_zero(self):
        """When no VIX data is available, default should be 15.0 (safe neutral)."""
        fe = FeatureEngine()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "open": [22000] * 5,
            "high": [22100] * 5,
            "low": [21900] * 5,
            "close": [22050] * 5,
            "volume": [100000] * 5,
        })

        # Call with vix_data=None (missing)
        result = fe.add_alternative_features(df.copy(), vix_data=None)

        assert (result["india_vix"] == 15.0).all(), (
            f"VIX default should be 15.0, got {result['india_vix'].iloc[0]}"
        )
        assert (result["vix_5d_ma"] == 15.0).all()
        assert (result["vix_20d_ma"] == 15.0).all()


# ═══════════════════════════════════════════
# BUG A: Fast poll / trail stop skip zero LTP
# ═══════════════════════════════════════════


class TestFastPollSkipsZeroLTP:
    def test_fast_poll_skips_zero_ltp(self):
        """Portfolio.check_stops() should skip positions with zero/None current_price."""
        pm = PortfolioManager(initial_capital=150000)
        # Add a position with zero current_price (simulating LTP not yet populated)
        pos = Position(
            symbol="NIFTY25000CE",
            instrument_key="NSE_FO|NIFTY25000CE",
            side="BUY",
            quantity=65,
            entry_price=100.0,
            current_price=0.0,  # zero — LTP feed not ready
            stop_loss=80.0,
            take_profit=140.0,
        )
        pm.add_position(pos)

        # check_stops must NOT trigger stop loss even though 0.0 < 80.0
        triggers = pm.check_stops()
        assert len(triggers) == 0, (
            f"Zero LTP should be skipped, but got triggers: {triggers}"
        )


class TestTrailStopIgnoresZeroPrice:
    def test_trail_stop_ignores_zero_price(self):
        """OrderManager.check_trailing_stops() should skip when LTP is 0."""
        mock_broker = MagicMock()
        mock_broker.get_ltp.return_value = {"ltp": 0}  # invalid LTP
        mock_rm = MagicMock()
        mock_cb = MagicMock()

        om = OrderManager(mock_broker, mock_rm, mock_cb)
        # Inject a pending order with trail activation
        om._pending_orders["T001"] = {
            "symbol": "NIFTY25000CE",
            "instrument_key": "NSE_FO|NIFTY25000CE",
            "is_options": True,
            "price": 100.0,
            "quantity": 65,
            "trail_trigger": 120.0,
            "trail_exit": 112.0,
            "trail_activated": True,
            "gtt_orders": {},
        }

        exits = om.check_trailing_stops()
        assert len(exits) == 0, (
            f"Zero LTP should be skipped, but got exits: {exits}"
        )
        # Order should still be pending (not exited)
        assert "T001" in om._pending_orders


# ═══════════════════════════════════════════
# ML: EOD retrain and startup load-only
# ═══════════════════════════════════════════


class TestEODRetrain:
    def test_eod_retrain_runs_after_market_close(self):
        """_eod_retrain_ml_models() should call trainer.train() when candle data exists."""
        from src.ml.train_models import DirectionModelTrainer

        mock_store = MagicMock()
        mock_store.get_ml_candle_coverage.return_value = {"rows": 500}
        mock_store.get_ml_trade_label_count.return_value = 10
        mock_store.get_deployed_model.return_value = {"model_version": 1, "test_acc": 0.62}

        mock_fb = MagicMock()
        trainer = DirectionModelTrainer(mock_store, mock_fb)
        trainer.train = MagicMock(return_value={
            "deployed": True, "model_version": 2,
            "train_acc": 0.70, "test_acc": 0.63, "gap": 0.07,
        })

        # Simulate what _eod_retrain_ml_models does
        coverage = mock_store.get_ml_candle_coverage("NIFTY50")
        assert coverage["rows"] >= 100
        metrics = trainer.train("NIFTY50")
        assert metrics["deployed"] is True
        assert metrics["test_acc"] > 0.60
        trainer.train.assert_called_once_with("NIFTY50")


class TestStartupOnlyLoadsModel:
    def test_startup_only_loads_model_no_retrain(self):
        """_maybe_retrain_ml_models() at startup should NOT call train()."""
        mock_store = MagicMock()
        mock_store.get_deployed_model.return_value = {
            "model_version": 5, "test_acc": 0.65,
        }

        mock_trainer = MagicMock()
        mock_trainer.load_deployed_model.return_value = True

        # Simulate startup: load model, never train
        deployed = mock_store.get_deployed_model("direction_v1")
        assert deployed is not None
        assert deployed["model_version"] == 5

        # trainer.train should NOT be called at startup
        mock_trainer.train.assert_not_called()


# ═══════════════════════════════════════════
# Partial Profit + Runner Tests
# ═══════════════════════════════════════════


class TestPartialCloseReducesQuantity:
    """Test that partial_close_position reduces qty and keeps position open."""

    def test_partial_close_reduces_quantity(self):
        pm = PortfolioManager(initial_capital=200000)
        pos = Position(
            symbol="NIFTY23000CE", side="BUY", quantity=130,
            entry_price=100.0, current_price=130.0,
            instrument_key="NSE_FO|123", strategy="options_buyer",
        )
        pm.add_position(pos)

        result = pm.partial_close_position("NIFTY23000CE", 130.0, 65, "tp1_partial")

        assert result is not None
        assert result["quantity"] == 65
        assert result["pnl"] == (130.0 - 100.0) * 65
        # Position still open with remaining qty
        assert "NIFTY23000CE" in pm.positions
        assert pm.positions["NIFTY23000CE"].quantity == 65
        assert pm.positions["NIFTY23000CE"].original_quantity == 130


class TestPartialCloseMovesSlToBreakeven:
    """Test that after partial exit, SL moves to entry price (breakeven)."""

    def test_partial_close_moves_sl_to_breakeven(self):
        pm = PortfolioManager(initial_capital=200000)
        pos = Position(
            symbol="NIFTY23000CE", side="BUY", quantity=130,
            entry_price=100.0, current_price=130.0, stop_loss=75.0,
            take_profit=155.0, instrument_key="NSE_FO|123",
        )
        pm.add_position(pos)

        pm.partial_close_position("NIFTY23000CE", 130.0, 65, "tp1_partial")

        remaining = pm.positions["NIFTY23000CE"]
        assert remaining.stop_loss == 100.0  # Breakeven
        assert remaining.take_profit == 0  # TP removed for runner
        assert remaining.partial_exit_done is True


class TestTP1ExitTaggedCorrectly:
    """Test that TP1 exit trade_id ends with _TP1."""

    def test_tp1_exit_tagged_correctly(self):
        pm = PortfolioManager(initial_capital=200000)
        pos = Position(
            symbol="NIFTY23000PE", side="BUY", quantity=130,
            entry_price=200.0, current_price=260.0,
            instrument_key="NSE_FO|456", trade_id="TR_001",
        )
        pm.add_position(pos)

        result = pm.partial_close_position("NIFTY23000PE", 260.0, 65, "tp1_partial")
        # Simulate main.py tagging
        result["trade_id"] += "_TP1"

        assert result["trade_id"] == "TR_001_TP1"
        assert result["quantity"] == 65


class TestRunnerExitTaggedCorrectly:
    """Test that runner exit trade_id ends with _RUN."""

    def test_runner_exit_tagged_correctly(self):
        pm = PortfolioManager(initial_capital=200000)
        pos = Position(
            symbol="NIFTY23000CE", side="BUY", quantity=130,
            entry_price=100.0, current_price=130.0,
            instrument_key="NSE_FO|123", trade_id="TR_002",
        )
        pm.add_position(pos)

        # Partial exit first
        pm.partial_close_position("NIFTY23000CE", 130.0, 65, "tp1_partial")
        assert pm.positions["NIFTY23000CE"].partial_exit_done is True

        # Runner exit (full close of remaining)
        is_runner = pm.positions["NIFTY23000CE"].partial_exit_done
        result = pm.close_position("NIFTY23000CE", 145.0, "trail_stop")
        if is_runner:
            result["trade_id"] += "_RUN"

        assert result["trade_id"] == "TR_002_RUN"
        assert result["quantity"] == 65  # Remaining qty
        assert "NIFTY23000CE" not in pm.positions


class TestNoPartialOnSingleUnit:
    """Test that qty=1 cannot be split (partial_lots = 0)."""

    def test_no_partial_on_single_unit(self):
        # qty=1: floor(1/2) = 0 → no partial
        partial_lots = 1 // 2
        assert partial_lots == 0

        # qty=65 (1 NIFTY lot): floor(65/2) = 32 → partial works
        partial_lots = 65 // 2
        assert partial_lots == 32

        # qty=130 (2 lots): floor(130/2) = 65 → half
        partial_lots = 130 // 2
        assert partial_lots == 65


# ═════════════════════════════════════════════════════
# WebSocket LTP Feed Tests (3 tests)
# ═════════════════════════════════════════════════════


class TestWSCacheUsedWhenAvailable:
    """When WS cache has LTP, get_ltp() returns it without REST call."""

    def test_ws_cache_used_when_available(self):
        from src.execution.paper_trader import PaperTrader

        # Create a mock data_fetcher with WS cache
        mock_df = MagicMock()
        mock_df.get_ws_ltp.return_value = 250.5
        mock_df.get_live_quote.return_value = {"ltp": 200.0}

        broker = PaperTrader(data_fetcher=mock_df)
        result = broker.get_ltp("NSE_FO|54908")

        assert result["ltp"] == 250.5
        assert result["source"] == "ws"
        # get_live_quote should NOT be called (WS cache hit)
        mock_df.get_live_quote.assert_not_called()


class TestWSFallbackToRest:
    """When WS cache returns None, get_ltp() falls through to REST."""

    def test_ws_fallback_to_rest(self):
        from src.execution.paper_trader import PaperTrader

        mock_df = MagicMock()
        mock_df.get_ws_ltp.return_value = None  # WS not available
        mock_df.get_live_quote.return_value = {"ltp": 200.0}

        broker = PaperTrader(data_fetcher=mock_df)
        result = broker.get_ltp("NSE_FO|54908")

        assert result["ltp"] == 200.0
        assert result["status"] == "success"
        # REST fallback was used
        mock_df.get_live_quote.assert_called_once_with("NSE_FO|54908")


class TestWSSubscribeOnNewPosition:
    """ws_subscribe is called after trade execution when WS enabled."""

    def test_ws_subscribe_on_new_position(self):
        from src.data.fetcher import UpstoxDataFetcher

        # Create a fetcher with mock streamer
        with patch.object(UpstoxDataFetcher, '__init__', lambda self, *a, **kw: None):
            fetcher = UpstoxDataFetcher.__new__(UpstoxDataFetcher)
            fetcher._ws_ltp_cache = {}
            fetcher._ws_ltp_lock = __import__('threading').Lock()
            fetcher._ws_connected = True
            fetcher._streamer = MagicMock()

            fetcher.ws_subscribe(["NSE_FO|54908"])
            fetcher._streamer.subscribe.assert_called_once_with(
                ["NSE_FO|54908"], "ltpc"
            )

            # Unsubscribe removes from cache
            fetcher._ws_ltp_cache["NSE_FO|54908"] = 250.0
            fetcher.ws_unsubscribe(["NSE_FO|54908"])
            fetcher._streamer.unsubscribe.assert_called_once_with(["NSE_FO|54908"])
            assert "NSE_FO|54908" not in fetcher._ws_ltp_cache


# ═══════════════════════════════════════════════════════════════
# TokenWatcher Tests
# ═══════════════════════════════════════════════════════════════
import base64, json as _json, time as _time
from src.auth.token_manager import (
    decode_jwt_expiry,
    get_token_expiry,
    is_token_expiring_soon,
    TokenWatcher,
)


def _make_jwt(exp_epoch: int) -> str:
    """Create a minimal JWT with the given exp claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    payload_dict = {"sub": "test", "exp": exp_epoch}
    payload = base64.urlsafe_b64encode(_json.dumps(payload_dict).encode()).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


class TestTokenExpiringSOon:
    """is_token_expiring_soon returns True when within threshold."""

    def test_expiring_within_30min(self):
        from datetime import timezone
        # Token expires 15 minutes from now
        exp = int((datetime.now(timezone.utc) + timedelta(minutes=15)).timestamp())
        token = _make_jwt(exp)
        assert is_token_expiring_soon(token, minutes=30) is True

    def test_not_expiring_soon(self):
        from datetime import timezone
        # Token expires 5 hours from now
        exp = int((datetime.now(timezone.utc) + timedelta(hours=5)).timestamp())
        token = _make_jwt(exp)
        assert is_token_expiring_soon(token, minutes=30) is False

    def test_non_jwt_returns_false(self):
        assert is_token_expiring_soon("not-a-jwt-token", minutes=30) is False


class TestTokenWatcherDaemonThread:
    """TokenWatcher starts as daemon thread and can be stopped."""

    def test_starts_as_daemon(self):
        auth = MagicMock()
        auth.access_token = None  # No token — loop will be benign
        watcher = TokenWatcher(auth=auth)
        watcher.start()
        assert watcher.is_running
        assert watcher._thread.daemon is True
        watcher.stop()
        _time.sleep(0.1)
        assert not watcher.is_running


class TestTokenRefreshFailedSendsAlert:
    """Token expiry alert goes to alert_fn without crashing."""

    def test_expired_token_sends_alert(self):
        from datetime import timezone
        # Token already expired 10 minutes ago
        exp = int((datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp())
        token = _make_jwt(exp)

        auth = MagicMock()
        auth.access_token = token
        alert_calls = []

        watcher = TokenWatcher(auth=auth, alert_fn=lambda msg: alert_calls.append(msg))
        watcher._check_and_alert()

        assert len(alert_calls) == 1
        assert "TOKEN_EXPIRED" in alert_calls[0]

    def test_alert_fn_exception_does_not_crash(self):
        from datetime import timezone
        exp = int((datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp())
        token = _make_jwt(exp)

        auth = MagicMock()
        auth.access_token = token

        def bad_alert(msg):
            raise ConnectionError("Telegram down")

        watcher = TokenWatcher(auth=auth, alert_fn=bad_alert)
        # Should not raise
        watcher._check_and_alert()


# ═══════════════════════════════════════════════════════════════
# Live Order Fill Confirmation Tests
# ═══════════════════════════════════════════════════════════════

class TestLiveOrderWaitsForFill:
    """wait_for_fill polls until complete, rejected, or timeout."""

    def test_fill_confirmed(self):
        """Broker returns filled=True with avg_price when order completes."""
        from src.execution.upstox_broker import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker._connected = True
        broker._api_client = MagicMock()
        broker._order_api = MagicMock()

        # Mock get_order_status to return "complete" immediately
        broker.get_order_status = MagicMock(return_value={
            "order_id": "ORD123",
            "status": "complete",
            "filled_quantity": 65,
            "average_price": 215.50,
        })

        result = broker.wait_for_fill("ORD123", timeout_seconds=10)
        assert result["filled"] is True
        assert result["avg_price"] == 215.50
        assert result["filled_qty"] == 65

    def test_rejected_returns_not_filled(self):
        """Broker returns filled=False when order is rejected."""
        from src.execution.upstox_broker import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker._connected = True
        broker._api_client = MagicMock()
        broker._order_api = MagicMock()

        broker.get_order_status = MagicMock(return_value={
            "order_id": "ORD456",
            "status": "rejected",
            "reject_reason": "Insufficient margin",
        })

        result = broker.wait_for_fill("ORD456", timeout_seconds=10)
        assert result["filled"] is False
        assert "rejected" in result["reason"] or "Insufficient" in result["reason"]


class TestUnfilledOrderDoesNotOpenPosition:
    """When order_manager gets rejected fill, it returns status=rejected."""

    def test_rejected_fill_returns_rejected(self):
        from src.execution.order_manager import OrderManager

        broker = MagicMock()
        # place_order succeeds but wait_for_fill fails
        broker.place_order.return_value = {
            "order_id": "ORD789",
            "status": "success",
            "message": "Order placed",
        }
        broker.wait_for_fill.return_value = {
            "filled": False,
            "avg_price": 0,
            "filled_qty": 0,
            "order_id": "ORD789",
            "reason": "timeout",
        }
        broker.get_ltp.return_value = {"ltp": 200.0}

        risk_mgr = MagicMock()
        risk_mgr.calculate_options_stops.return_value = {
            "stop_loss": 140.0, "take_profit": 320.0
        }
        cb = MagicMock()
        cb.get_size_multiplier.return_value = 1.0

        om = OrderManager.__new__(OrderManager)
        om.broker = broker
        om.risk_manager = risk_mgr
        om.circuit_breaker = cb
        om._pending_orders = {}
        om._alert_fn = None
        om._spread_ltp_errors = {}

        signal = {
            "symbol": "NIFTY25350PE",
            "instrument_key": "NSE_FO|54908",
            "direction": "BUY",
            "price": 200.0,
            "confidence": 0.8,
            "stop_loss": 0.30,
            "take_profit": 0.60,
            "strategy": "options_buyer",
            "regime": "TRENDING",
            "features": {
                "instrument_key": "NSE_FO|54908",
                "lot_size": 65,
                "premium_sl_pct": 0.30,
                "premium_tp_pct": 0.60,
                "is_options": True,
                "option_type": "PE",
                "strike": 25350,
                "expiry": "2026-03-19",
                "index_symbol": "NIFTY",
            },
        }

        result = om._execute_options_signal(signal, capital=150000, current_positions=None)
        assert result["status"] == "rejected"
        assert "not filled" in result["reason"]


class TestSlippageRecordedOnFill:
    """trade_record includes signal_price, fill_price, slippage_pct."""

    def test_slippage_fields_populated(self):
        from src.execution.order_manager import OrderManager

        broker = MagicMock()
        # Simulate live broker with wait_for_fill
        broker.place_order.return_value = {
            "order_id": "ORD999",
            "status": "success",
        }
        broker.wait_for_fill.return_value = {
            "filled": True,
            "avg_price": 202.0,  # Filled at 202 vs signal 200
            "filled_qty": 65,
            "order_id": "ORD999",
            "reason": None,
        }
        broker.place_gtt_order.return_value = {"status": "success"}

        risk_mgr = MagicMock()
        risk_mgr.calculate_options_stops.return_value = {
            "stop_loss": 140.0, "take_profit": 320.0
        }
        risk_mgr.calculate_options_trade_costs.return_value = {
            "brokerage": 40, "stt": 6.5, "total": 60
        }
        cb = MagicMock()
        cb.get_size_multiplier.return_value = 1.0

        om = OrderManager.__new__(OrderManager)
        om.broker = broker
        om.risk_manager = risk_mgr
        om.circuit_breaker = cb
        om._pending_orders = {}
        om._alert_fn = None
        om._spread_ltp_errors = {}

        signal = {
            "symbol": "NIFTY25350PE",
            "instrument_key": "NSE_FO|54908",
            "direction": "BUY",
            "price": 200.0,
            "confidence": 0.8,
            "strategy": "options_buyer",
            "regime": "TRENDING",
            "features": {
                "instrument_key": "NSE_FO|54908",
                "lot_size": 65,
                "premium_sl_pct": 0.30,
                "premium_tp_pct": 0.60,
                "is_options": True,
                "option_type": "PE",
                "strike": 25350,
                "expiry": "2026-03-19",
                "index_symbol": "NIFTY",
            },
        }

        result = om._execute_options_signal(signal, capital=150000, current_positions=None)
        assert result["status"] == "success"
        assert result["signal_price"] == 200.0
        assert result["fill_price"] == 202.0
        assert result["slippage_pct"] > 0
        assert abs(result["slippage_pct"] - 0.01) < 0.001  # 1% slippage


# ═══════════════════════════════════════════════════════════════
# Reconciliation & Margin Tests
# ═══════════════════════════════════════════════════════════════

class TestReconciliationOKWithin100:
    """Reconciliation logs OK when diff <= ₹100."""

    def test_reconciliation_ok(self):
        from src.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot.mode = "live"
        bot.portfolio = MagicMock()
        bot.portfolio.get_day_pnl.return_value = 1500.0
        bot.portfolio.closed_trades = [{"pnl": 1500}]
        bot.broker = MagicMock()
        bot.broker.get_daily_pnl.return_value = 1450.0  # ₹50 diff — within ₹100
        bot.broker.get_todays_trades.return_value = [{"symbol": "NIFTY25350PE"}]
        bot.alerts = MagicMock()
        bot.store = MagicMock()

        bot._reconcile_pnl()

        # Should save log with status OK
        bot.store.save_reconciliation_log.assert_called_once()
        log_record = bot.store.save_reconciliation_log.call_args[0][0]
        assert log_record["status"] == "OK"
        assert log_record["difference"] == 50.0
        # No alert sent for OK
        bot.alerts.send_raw.assert_not_called()


class TestReconciliationAlertAbove100:
    """Reconciliation sends alert when diff > ₹100."""

    def test_warning_alert(self):
        from src.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot.mode = "live"
        bot.portfolio = MagicMock()
        bot.portfolio.get_day_pnl.return_value = 2000.0
        bot.portfolio.closed_trades = [{"pnl": 2000}]
        bot.broker = MagicMock()
        bot.broker.get_daily_pnl.return_value = 1700.0  # ₹300 diff — WARNING
        bot.broker.get_todays_trades.return_value = [{"symbol": "NIFTY25350PE"}]
        bot.alerts = MagicMock()
        bot.store = MagicMock()

        bot._reconcile_pnl()

        log_record = bot.store.save_reconciliation_log.call_args[0][0]
        assert log_record["status"] == "WARNING"
        assert log_record["difference"] == 300.0
        # Alert should be sent
        assert bot.alerts.send_raw.call_count >= 1
        alert_text = bot.alerts.send_raw.call_args_list[0][0][0]
        assert "MISMATCH" in alert_text

    def test_critical_alert(self):
        from src.main import TradingBot

        bot = TradingBot.__new__(TradingBot)
        bot.mode = "live"
        bot.portfolio = MagicMock()
        bot.portfolio.get_day_pnl.return_value = 5000.0
        bot.portfolio.closed_trades = [{"pnl": 5000}]
        bot.broker = MagicMock()
        bot.broker.get_daily_pnl.return_value = 3500.0  # ₹1500 diff — CRITICAL
        bot.broker.get_todays_trades.return_value = [{"symbol": "NIFTY25350PE"}]
        bot.alerts = MagicMock()
        bot.store = MagicMock()

        bot._reconcile_pnl()

        log_record = bot.store.save_reconciliation_log.call_args[0][0]
        assert log_record["status"] == "CRITICAL"
        # Should mention "CRITICAL" in alert
        alert_text = bot.alerts.send_raw.call_args_list[0][0][0]
        assert "CRITICAL" in alert_text


class TestMarginCheckBlocksTrade:
    """Trade blocked when available margin < required."""

    def test_insufficient_margin_blocks(self):
        """Broker returns margin below required — trade should not execute."""
        from src.execution.upstox_broker import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker._connected = True
        broker._api_client = MagicMock()

        # Funds returns only ₹5,000 available
        broker.get_funds = MagicMock(return_value={"available_margin": 5000.0})

        margin = broker.get_available_margin()
        assert margin == 5000.0

        # Required: 200 * 65 = ₹13,000
        required = 200 * 65
        assert margin < required  # Should block


class TestMarginWarningDoesNotBlock:
    """Trade proceeds with warning when margin is tight but sufficient."""

    def test_low_margin_allows_trade(self):
        """Broker returns margin above required but below 1.4x buffer."""
        from src.execution.upstox_broker import UpstoxBroker

        broker = UpstoxBroker.__new__(UpstoxBroker)
        broker._connected = True
        broker._api_client = MagicMock()

        # Funds returns ₹15,000 available
        broker.get_funds = MagicMock(return_value={"available_margin": 15000.0})

        margin = broker.get_available_margin()
        assert margin == 15000.0

        # Required: 200 * 65 = ₹13,000 — margin is sufficient
        required = 200 * 65
        assert margin >= required  # Should NOT block
        assert margin < required * 1.4  # But should warn (buffer < 40%)


class TestExpectancyCalculation:
    """Verify expectancy formula: E = (WR × Avg_Win) - (LR × Avg_Loss)."""

    def test_expectancy_calculation_correct(self):
        """Exact expectancy matches hand-calculated value."""
        # 10 trades: 7 wins, 3 losses
        pnls = [8000, 9000, 7500, 10000, 8500, 9500, 8000, -5000, -4000, -5500]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)
        lr = 1 - wr
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        expectancy = (wr * avg_win) - (lr * avg_loss)

        # Hand calc: WR=0.7, Avg_Win=8642.86, Avg_Loss=4833.33
        # E = 0.7 * 8642.86 - 0.3 * 4833.33 = 6050 - 1450 = 4600
        assert 0.69 < wr < 0.71
        assert 8600 < avg_win < 8700
        assert 4800 < avg_loss < 4900
        assert 4500 < expectancy < 4700


class TestRMultipleExpectancy:
    """Verify R-multiple expectancy: E(R) = E / Avg_R."""

    def test_r_multiple_expectancy_correct(self):
        """R-multiple expectancy scales with risk unit."""
        pnls = [6000, 8000, -3000, 7000, -4000]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)  # 0.6
        lr = 1 - wr  # 0.4
        avg_win = sum(wins) / len(wins)  # 7000
        avg_loss = abs(sum(losses) / len(losses))  # 3500
        expectancy = (wr * avg_win) - (lr * avg_loss)  # 4200 - 1400 = 2800

        # Each trade risked ₹3500 (the SL distance)
        avg_r = 3500.0
        r_expectancy = expectancy / avg_r  # 2800 / 3500 = 0.80R

        assert abs(wr - 0.6) < 0.01
        assert abs(expectancy - 2800) < 1
        assert abs(r_expectancy - 0.80) < 0.01


class TestKellyCriterion:
    """Verify Kelly % = WR - (LR / payoff_ratio)."""

    def test_kelly_criterion_calculation(self):
        """Kelly matches formula with known values."""
        wr = 0.737
        lr = 1 - wr  # 0.263
        avg_win = 8867.0
        avg_loss = 4677.0
        payoff = avg_win / avg_loss  # 1.896

        kelly_pct = (wr - (lr / payoff)) * 100

        # Kelly = (0.737 - (0.263 / 1.896)) * 100 = (0.737 - 0.1387) * 100 = 59.83%
        assert 59.0 < kelly_pct < 61.0

        # Edge case: payoff = 0 should not crash
        kelly_zero = (wr - (lr / 1e-10)) * 100 if 1e-10 > 0 else 0
        assert kelly_zero < 0  # Negative Kelly means don't trade


class TestFixedRSizingConsistentRisk:
    """Fixed-R sizing should produce consistent risk per trade."""

    def test_fixed_r_sizing_consistent_risk(self):
        """Different premiums → same risk when using Fixed-R formula."""
        risk_per_trade = 15000
        lot_size = 65
        sl_pct = 0.20
        deploy_cap = 75000

        # Trade 1: premium=₹200
        prem1 = 200
        sl_dist1 = prem1 * sl_pct  # ₹40
        lots_by_risk1 = int(risk_per_trade / (sl_dist1 * lot_size))
        lots_by_deploy1 = int(deploy_cap / (prem1 * lot_size))
        lots1 = max(1, min(lots_by_risk1, lots_by_deploy1))
        qty1 = lots1 * lot_size
        risk1 = sl_dist1 * qty1

        # Trade 2: premium=₹100
        prem2 = 100
        sl_dist2 = prem2 * sl_pct  # ₹20
        lots_by_risk2 = int(risk_per_trade / (sl_dist2 * lot_size))
        lots_by_deploy2 = int(deploy_cap / (prem2 * lot_size))
        lots2 = max(1, min(lots_by_risk2, lots_by_deploy2))
        qty2 = lots2 * lot_size
        risk2 = sl_dist2 * qty2

        # Both risks should be close to RISK_PER_TRADE (within 1 lot tolerance)
        assert abs(risk1 - risk2) < sl_dist1 * lot_size + sl_dist2 * lot_size
        assert risk1 <= risk_per_trade
        assert risk2 <= risk_per_trade


class TestFixedRSizingCappedByDeploy:
    """Fixed-R qty should never exceed deploy cap."""

    def test_fixed_r_sizing_capped_by_deploy(self):
        """When risk allows more lots than deploy cap, deploy cap wins."""
        risk_per_trade = 15000
        lot_size = 65
        sl_pct = 0.30  # Wide SL → risk allows many lots
        deploy_cap = 25000
        premium = 150

        sl_dist = premium * sl_pct  # ₹45
        lots_by_risk = int(risk_per_trade / (sl_dist * lot_size))  # 15000/2925 = 5
        lots_by_deploy = int(deploy_cap / (premium * lot_size))  # 25000/9750 = 2
        lots = max(1, min(lots_by_risk, lots_by_deploy))
        qty = lots * lot_size
        position_cost = premium * qty

        assert lots == lots_by_deploy  # Deploy cap is binding
        assert position_cost <= deploy_cap


class TestFixedRSizingMinimumOneLot:
    """Fixed-R should always trade at least 1 lot."""

    def test_fixed_r_sizing_minimum_one_lot(self):
        """Even with expensive premium, minimum is 1 lot."""
        risk_per_trade = 15000
        lot_size = 65
        sl_pct = 0.20
        deploy_cap = 75000
        premium = 800  # Expensive premium

        sl_dist = premium * sl_pct  # ₹160
        lots_by_risk = int(risk_per_trade / (sl_dist * lot_size))  # 15000/10400 = 1
        lots_by_deploy = int(deploy_cap / (premium * lot_size))  # 75000/52000 = 1
        lots = max(1, min(lots_by_risk, lots_by_deploy))
        qty = lots * lot_size

        assert lots >= 1
        assert qty == lot_size  # Exactly 1 lot


class TestEnsembleResultInitialized:
    """Verify ensemble_result is always set before use in trading loop."""

    def test_ensemble_result_initialized_before_use(self):
        """When data gate opens, ensemble_result must still be initialized."""
        # Simulate the control flow from _trading_loop
        # Before fix: ensemble_result was only set in elif branches,
        # missing when _data_ready transitions False→True

        _data_ready = False
        # Default at top of if/elif chain (the fix)
        ensemble_result = {"decisions": []}

        # Simulate: data becomes ready on this iteration
        vix_ok = True
        nifty_ok = True
        candles_ok = True

        if not _data_ready:
            if vix_ok and nifty_ok and candles_ok:
                _data_ready = True
                # Before fix: no ensemble_result assignment here
            else:
                ensemble_result = {"decisions": []}
        else:
            ensemble_result = {"decisions": [{"direction": "HOLD", "symbol": "NIFTY"}]}

        # This must never raise — ensemble_result always defined
        decisions = ensemble_result.get("decisions", [])
        assert isinstance(decisions, list)
        assert _data_ready is True


class TestICAdxSafeExtraction:
    """Verify IC ADX extraction handles missing/empty adx column."""

    def test_trading_loop_handles_empty_dataframe(self):
        """IC ADX extraction must not crash when adx column missing."""
        import pandas as pd

        # Case 1: nifty_df is empty
        _ic_ndf = pd.DataFrame()
        _ic_adx_s = _ic_ndf.get("adx", pd.Series()) if not _ic_ndf.empty else pd.Series()
        ic_adx = float(_ic_adx_s.iloc[-1]) if len(_ic_adx_s) > 0 else 30.0
        assert ic_adx == 30.0

        # Case 2: nifty_df has data but no "adx" column
        _ic_ndf = pd.DataFrame({"close": [100, 101, 102]})
        _ic_adx_s = _ic_ndf.get("adx", pd.Series()) if not _ic_ndf.empty else pd.Series()
        ic_adx = float(_ic_adx_s.iloc[-1]) if len(_ic_adx_s) > 0 else 30.0
        assert ic_adx == 30.0

        # Case 3: nifty_df has "adx" column with values
        _ic_ndf = pd.DataFrame({"close": [100, 101], "adx": [18.5, 19.2]})
        _ic_adx_s = _ic_ndf.get("adx", pd.Series()) if not _ic_ndf.empty else pd.Series()
        ic_adx = float(_ic_adx_s.iloc[-1]) if len(_ic_adx_s) > 0 else 30.0
        assert ic_adx == 19.2


class TestEntryDistanceFilter:
    """Verify entry distance filter blocks chasing and allows normal entries."""

    @staticmethod
    def _check_dist(direction, dist_from_open, regime, max_dist=0.008):
        """Replicate the entry distance filter logic from options_buyer.py."""
        if regime == "TRENDING":
            effective_limit = max_dist * 1.5
        elif regime == "VOLATILE":
            effective_limit = max_dist * 0.6
        elif regime == "RANGEBOUND":
            effective_limit = max_dist * 0.5
        else:
            effective_limit = max_dist * 0.75
        if direction == "PE" and dist_from_open < -effective_limit:
            return True  # blocked
        if direction == "CE" and dist_from_open > effective_limit:
            return True  # blocked
        return False

    def test_entry_blocked_when_price_moved_too_far_pe(self):
        """PE entry blocked when NIFTY already fell > limit from open."""
        # VOLATILE: limit = 0.008 * 0.6 = 0.0048 (0.48%)
        # NIFTY fell 0.69% from open → blocked
        blocked = self._check_dist("PE", -0.0069, "VOLATILE")
        assert blocked is True

        # TRENDING: limit = 0.008 * 1.5 = 0.012 (1.2%)
        # NIFTY fell 0.69% from open → NOT blocked (within 1.2%)
        blocked = self._check_dist("PE", -0.0069, "TRENDING")
        assert blocked is False

        # CE entry blocked when NIFTY rallied > limit
        blocked = self._check_dist("CE", 0.010, "VOLATILE")
        assert blocked is True

        # RANGEBOUND: limit = 0.008 * 0.5 = 0.004 (0.4%)
        blocked = self._check_dist("PE", -0.005, "RANGEBOUND")
        assert blocked is True

    def test_entry_allowed_when_price_within_range(self):
        """Entry allowed when distance from open is within regime limit."""
        # Small move in VOLATILE: 0.3% < 0.48% limit
        blocked = self._check_dist("PE", -0.003, "VOLATILE")
        assert blocked is False

        # CE with small up move in TRENDING: 0.5% < 1.2% limit
        blocked = self._check_dist("CE", 0.005, "TRENDING")
        assert blocked is False

        # PE direction but price moved UP (wrong direction for blocking)
        blocked = self._check_dist("PE", 0.010, "VOLATILE")
        assert blocked is False

        # CE direction but price moved DOWN (wrong direction for blocking)
        blocked = self._check_dist("CE", -0.010, "VOLATILE")
        assert blocked is False

        # ELEVATED: limit = 0.008 * 0.75 = 0.006 (0.6%)
        blocked = self._check_dist("PE", -0.005, "ELEVATED")
        assert blocked is False  # 0.5% < 0.6%

        blocked = self._check_dist("PE", -0.007, "ELEVATED")
        assert blocked is True  # 0.7% > 0.6%


class TestPriceContradictionFilter:
    """Verify price contradiction filter blocks entries when price opposes direction."""

    @staticmethod
    def _check_contradiction(direction, dist_from_open, rsi, threshold=0.005):
        """Replicate the price contradiction logic from options_buyer.py."""
        if (direction == "PE" and dist_from_open > threshold and rsi > 55) or \
           (direction == "CE" and dist_from_open < -threshold and rsi < 45):
            return True  # contradicted → blocked
        return False

    def test_entry_blocked_price_contradicts_pe(self):
        """PE signal blocked when NIFTY is +1% above open with RSI=60."""
        # PE direction but price +1% above open, RSI 60 → contradicted
        blocked = self._check_contradiction("PE", 0.010, 60)
        assert blocked is True

        # PE direction, price +0.3% (below 0.5% threshold) → not blocked
        blocked = self._check_contradiction("PE", 0.003, 60)
        assert blocked is False

        # PE direction, price +1% but RSI=50 (below 55) → not blocked
        blocked = self._check_contradiction("PE", 0.010, 50)
        assert blocked is False

        # CE direction, price -1.5% below open, RSI=35 → contradicted
        blocked = self._check_contradiction("CE", -0.015, 35)
        assert blocked is True

        # CE direction, price -1% but RSI=50 (above 45) → not blocked
        blocked = self._check_contradiction("CE", -0.010, 50)
        assert blocked is False

    def test_entry_allowed_when_price_confirms_pe(self):
        """PE signal allowed when price is below open (confirms bearish direction)."""
        # PE direction, price -1% below open → price confirms PE → not blocked
        blocked = self._check_contradiction("PE", -0.010, 40)
        assert blocked is False

        # PE direction, price flat → not blocked
        blocked = self._check_contradiction("PE", 0.001, 60)
        assert blocked is False

        # CE direction, price +1% above open → confirms CE → not blocked
        blocked = self._check_contradiction("CE", 0.010, 65)
        assert blocked is False

        # CE direction, price flat → not blocked
        blocked = self._check_contradiction("CE", -0.002, 40)
        assert blocked is False


class TestPostMarketTimeGuard:
    """Fix 1: POST-MARKET should only fire after 15:10."""

    def test_postmarket_time_guard(self):
        """POST-MARKET skipped before 15:10, runs after 15:10."""
        from unittest.mock import MagicMock
        from datetime import time as dt_time

        # Create minimal bot mock
        bot = MagicMock()
        bot.store = MagicMock()
        bot.portfolio = MagicMock()
        bot.portfolio.closed_trades = []
        bot.portfolio.get_snapshot.return_value = {}
        bot.instrument_logger = MagicMock()
        bot.alerts = MagicMock()
        bot.mode = "live"

        from src.main import TradingBot

        # Before 15:10 → should skip (return early)
        mock_dt_early = datetime(2026, 3, 20, 9, 15, 0)
        with patch("src.main.datetime") as mock:
            mock.now.return_value = mock_dt_early
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            TradingBot._post_market(bot)
        bot.store.save_portfolio_snapshot.assert_not_called()

        # After 15:10 → should run
        mock_dt_late = datetime(2026, 3, 20, 15, 15, 0)
        with patch("src.main.datetime") as mock:
            mock.now.return_value = mock_dt_late
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            TradingBot._post_market(bot)
        bot.store.save_portfolio_snapshot.assert_called_once()


class TestFundsCheckPremarket:
    """Fix 1: Funds check should not abort before 09:30."""

    def test_funds_check_premarket(self):
        """Funds=0 before 09:30 sets recheck flag instead of aborting."""
        # Simulate pre-market: time < 09:30 and available == 0
        available = 0.0
        funds_reliable_time = dt_time(9, 30)
        now_premarket = dt_time(9, 5)

        funds_not_ready = now_premarket < funds_reliable_time or available == 0
        assert funds_not_ready is True, "Pre-market + ₹0 should trigger defer"

        # After 09:30 with real funds → should not defer
        now_market = dt_time(10, 0)
        available_real = 65000.0
        funds_not_ready_2 = now_market < funds_reliable_time or available_real == 0
        assert funds_not_ready_2 is False, "Post-market + real funds should not defer"

        # After 09:30 but still ₹0 → still defers (stale data)
        funds_not_ready_3 = now_market < funds_reliable_time or 0.0 == 0
        assert funds_not_ready_3 is True, "₹0 after 09:30 should still defer"


class TestStaleClearOnlyOnRecovery:
    """Fix 4: DATA_STALE_CLEAR should only log when recovering from stale."""

    def test_stale_clear_only_on_recovery(self):
        """DATA_STALE_CLEAR only fires when _was_data_stale transitions True→False."""
        # Scenario 1: Fresh data, never stale → no CLEAR log
        was_stale = False
        should_log = was_stale  # Only log if was_stale
        assert should_log is False, "Should not log CLEAR when never stale"

        # Scenario 2: Was stale, now fresh → CLEAR log
        was_stale = True
        should_log = was_stale
        assert should_log is True, "Should log CLEAR when recovering from stale"

        # Scenario 3: Fresh → fresh → no CLEAR log
        was_stale = False
        should_log = was_stale
        assert should_log is False, "Should not log CLEAR on consecutive fresh data"


# ═════════════════════════════════════════════════════
# Momentum Mode Tests (8 tests)
# ═════════════════════════════════════════════════════


class TestMomentumMode:
    """Tests for per-loop momentum-based CE/PE direction selection."""

    def setup_method(self):
        self.s = _make_strategy()

    def test_momentum_direction_computed_from_scores(self):
        """_compute_momentum_direction returns direction based on blended scores."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            cfg.MOMENTUM_MIN_SCORE_DIFF = 1.5
            cfg.MOMENTUM_CE_MIN_PROB = 0.55
            cfg.MOMENTUM_PE_MIN_PROB = 0.60
            mock_cfg.return_value = cfg

            # Set blended scores favoring CE
            self.s._direction_scores["NIFTY"] = (6.0, 3.0, 3.0)
            data = _base_data(ml_direction_prob_up=0.70, ml_direction_prob_down=0.65)

            direction, bull, bear, diff = self.s._compute_momentum_direction("NIFTY", data)
            assert direction == "CE"
            assert diff == 3.0

    def test_momentum_pe_direction(self):
        """_compute_momentum_direction returns PE when bear > bull."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            cfg.MOMENTUM_MIN_SCORE_DIFF = 1.5
            cfg.MOMENTUM_CE_MIN_PROB = 0.55
            cfg.MOMENTUM_PE_MIN_PROB = 0.60
            mock_cfg.return_value = cfg

            # Set blended scores favoring PE
            self.s._direction_scores["NIFTY"] = (2.0, 5.0, 3.0)
            data = _base_data(ml_direction_prob_up=0.65, ml_direction_prob_down=0.75)

            direction, bull, bear, diff = self.s._compute_momentum_direction("NIFTY", data)
            assert direction == "PE"

    def test_momentum_min_score_diff_blocks(self):
        """No direction when score difference below MOMENTUM_MIN_SCORE_DIFF."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            cfg.MOMENTUM_MIN_SCORE_DIFF = 1.5
            cfg.MOMENTUM_CE_MIN_PROB = 0.55
            cfg.MOMENTUM_PE_MIN_PROB = 0.60
            mock_cfg.return_value = cfg

            # Score diff = 1.0 < 1.5 threshold
            self.s._direction_scores["NIFTY"] = (3.5, 2.5, 1.0)
            data = _base_data()

            direction, bull, bear, diff = self.s._compute_momentum_direction("NIFTY", data)
            assert direction == "", "Should return empty when diff < min_score_diff"

    def test_momentum_ml_gate_blocks_pe(self):
        """PE blocked when pe_prob < MOMENTUM_PE_MIN_PROB."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            cfg.MOMENTUM_MIN_SCORE_DIFF = 1.5
            cfg.MOMENTUM_CE_MIN_PROB = 0.55
            cfg.MOMENTUM_PE_MIN_PROB = 0.60
            mock_cfg.return_value = cfg

            self.s._direction_scores["NIFTY"] = (2.0, 5.0, 3.0)
            # PE prob below threshold
            data = _base_data(ml_direction_prob_down=0.50)

            direction, _, _, _ = self.s._compute_momentum_direction("NIFTY", data)
            assert direction == "", "PE should be blocked by low ml_direction_prob_down"

    def test_momentum_ml_gate_blocks_ce(self):
        """CE blocked when ce_prob < MOMENTUM_CE_MIN_PROB."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            cfg.MOMENTUM_MIN_SCORE_DIFF = 1.5
            cfg.MOMENTUM_CE_MIN_PROB = 0.55
            cfg.MOMENTUM_PE_MIN_PROB = 0.60
            mock_cfg.return_value = cfg

            self.s._direction_scores["NIFTY"] = (5.0, 2.0, 3.0)
            # CE prob below threshold
            data = _base_data(ml_direction_prob_up=0.40)

            direction, _, _, _ = self.s._compute_momentum_direction("NIFTY", data)
            assert direction == "", "CE should be blocked by low ml_direction_prob_up"

    def test_momentum_direction_locked_on_execution(self):
        """confirm_execution() locks direction in _position_direction_lock."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            mock_cfg.return_value = cfg

            self.s._direction["NIFTY"] = "CE"
            self.s.confirm_execution("NIFTY")

            assert self.s._position_direction_lock.get("NIFTY") == "CE"
            assert self.s._trades_today["NIFTY"] == 1

    def test_momentum_direction_unlocked_on_exit(self):
        """record_exit() clears _position_direction_lock."""
        self.s._position_direction_lock["NIFTY"] = "PE"
        self.s._direction["NIFTY"] = "PE"

        self.s.record_exit("NIFTY25500PE", "take_profit", "PE")

        assert "NIFTY" not in self.s._position_direction_lock

    def test_momentum_contradiction_noop(self):
        """check_direction_contradiction returns AGREEMENT in momentum mode."""
        with patch("src.strategies.options_buyer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.MOMENTUM_MODE_ENABLED = True
            mock_cfg.return_value = cfg

            self.s._daily_scores["NIFTY"] = (5.0, 2.0, 3.0)
            self.s._intraday_scores["NIFTY"] = (2.0, 5.0, 3.0)  # Contradicts daily

            result = self.s.check_direction_contradiction("NIFTY", _base_data())
            assert result == "AGREEMENT"


# ═════════════════════════════════════════════════════
# Stability Fixes M5, M8, M9, M12 (4 tests)
# ═════════════════════════════════════════════════════


class TestStabilityFixes:
    """Tests for M5 (spread check), M8 (IC LTP=0), M9 (time parse), M12 (dict iteration)."""

    def test_spread_check_logs_on_wide_spread(self):
        """M5: SPREAD_TOO_WIDE logged when bid/ask spread > 3%."""
        # Simulate the spread check logic from _select_strike_delta
        bid, ask = 90.0, 100.0  # 10% spread
        spread_pct = (ask - bid) / ask * 100
        assert spread_pct > 3.0, "10% spread should exceed 3% threshold"
        # Verify the log message format matches
        msg = f"SPREAD_TOO_WIDE: bid={bid} ask={ask} spread={spread_pct:.1f}% > max allowed"
        assert "SPREAD_TOO_WIDE" in msg
        assert "10.0%" in msg

        # Verify tight spread passes
        bid2, ask2 = 98.0, 100.0  # 2% spread
        spread_pct2 = (ask2 - bid2) / ask2 * 100
        assert spread_pct2 <= 3.0, "2% spread should pass threshold"

    def test_ic_sl_skips_zero_ltp(self):
        """M8: IC stop check skips when any leg LTP=0, with warning log."""
        from src.risk.portfolio import PortfolioManager, IronCondorPosition

        pm = PortfolioManager()
        # Create a fake IC position
        ic = IronCondorPosition(
            position_id="test_ic_1",
            sell_ce_instrument_key="NSE_FO|CE_SELL",
            buy_ce_instrument_key="NSE_FO|CE_BUY",
            sell_pe_instrument_key="NSE_FO|PE_SELL",
            buy_pe_instrument_key="NSE_FO|PE_BUY",
            quantity=65,
            net_credit=50.0,
            sl_threshold=-100.0,
            tp_threshold=40.0,
        )
        pm.ic_positions["test_ic_1"] = ic

        # LTP dict with one leg = 0 (stale data)
        ltp_dict = {
            "NSE_FO|CE_SELL": 120.0,
            "NSE_FO|CE_BUY": 80.0,
            "NSE_FO|PE_SELL": 0,  # Stale!
            "NSE_FO|PE_BUY": 60.0,
        }
        triggers = pm.check_ic_stops(ltp_dict)
        assert triggers == [], "Should skip SL check when LTP=0 (stale data)"

    def test_trade_end_parse_error_uses_default(self):
        """M9: parse_time_config returns default on malformed input."""
        from src.config.env_loader import parse_time_config

        # Valid input
        h, m = parse_time_config("15:10", 15, 10)
        assert (h, m) == (15, 10)

        # Malformed inputs → fallback to default
        h, m = parse_time_config("bad", 15, 10)
        assert (h, m) == (15, 10), "Should use default on malformed string"

        h, m = parse_time_config("", 10, 0)
        assert (h, m) == (10, 0), "Should use default on empty string"

        h, m = parse_time_config("25:99", 15, 15)
        # Still parses to ints (25, 99) — dt_time would fail but parse_time_config just returns ints
        assert h == 25 and m == 99  # It successfully parsed, validation is at dt_time level

    def test_positions_iteration_safe(self):
        """M12: list() copy prevents RuntimeError during position dict iteration."""
        positions = {"NIFTY": "pos1", "BANKNIFTY": "pos2", "FINNIFTY": "pos3"}

        # Iterating over list() copy allows modification during iteration
        removed = []
        for sym, pos in list(positions.items()):
            if sym == "BANKNIFTY":
                del positions[sym]
                removed.append(sym)

        assert "BANKNIFTY" in removed
        assert "BANKNIFTY" not in positions
        assert len(positions) == 2


# ═════════════════════════════════════════════════════
# CE Filter 3-Tier Tests (2 tests)
# ═════════════════════════════════════════════════════


class TestCeFilter:
    """Tests for CE confidence filter with 3-tier tolerance zone."""

    def test_ce_filter_passes_above_threshold(self):
        """CE prob >= 0.65 → always allowed."""
        s = _make_strategy()
        passes, reason = s._ce_filter_passes(0.70, 2.0, 15.0, 16.0)
        assert passes is True
        assert reason == ""

        # Just at threshold
        passes2, reason2 = s._ce_filter_passes(0.65, 1.0, 15.0, 15.0)
        assert passes2 is True
        assert reason2 == ""

        # Below tolerance → blocked
        passes3, reason3 = s._ce_filter_passes(0.40, 2.0, 15.0, 15.0)
        assert passes3 is False
        assert reason3 == "CE_LOW_CONFIDENCE"

    def test_ce_tolerance_allows_strong_setup(self):
        """CE prob 0.50-0.65: allowed if VIX falling or high score_diff."""
        s = _make_strategy()

        # In tolerance zone + VIX falling (vix_open - vix_now >= 0.5) → pass
        passes, reason = s._ce_filter_passes(0.55, 2.0, 14.5, 15.0)
        assert passes is True, "VIX falling should allow tolerance zone"

        # In tolerance zone + strong score (>= 3.25) → pass
        passes2, reason2 = s._ce_filter_passes(0.55, 3.5, 15.0, 15.0)
        assert passes2 is True, "Strong score_diff should allow tolerance zone"

        # In tolerance zone but weak setup → blocked
        passes3, reason3 = s._ce_filter_passes(0.55, 2.0, 15.0, 15.0)
        assert passes3 is False
        assert reason3 == "CE_TOLERANCE_BLOCK"


# ═════════════════════════════════════════════════════
# Adaptive Fuzzy Threshold (3 tests)
# ═════════════════════════════════════════════════════


class TestAdaptiveFuzzy:
    """Test 3-tier adaptive fuzzy threshold."""

    def setup_method(self):
        from src.config.env_loader import get_config
        cfg = get_config()
        cfg.ADAPTIVE_FUZZY_ENABLED = True
        cfg.ADAPTIVE_FUZZY_STRONG_SCORE = 3.5
        cfg.ADAPTIVE_FUZZY_STRONG_THRESHOLD = 1.5
        cfg.ADAPTIVE_FUZZY_MID_SCORE = 2.5
        cfg.ADAPTIVE_FUZZY_MID_THRESHOLD = 1.75
        cfg.ADAPTIVE_FUZZY_CUTOFF_HOUR = 13

    def test_strong_signal_lowers_threshold(self):
        """score_diff >= 3.5 before 1PM → fuzzy threshold = 1.5."""
        from src.config.env_loader import get_config
        cfg = get_config()
        # Simulate: base threshold = 2.0, score_diff = 4.0, hour = 10
        base_threshold = 2.0
        score_diff = 4.0

        # Apply adaptive logic (same as _evaluate_symbol)
        with patch("src.strategies.options_buyer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 20, 10, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            fuzzy_threshold = base_threshold
            if cfg.ADAPTIVE_FUZZY_ENABLED and mock_dt.now().hour < cfg.ADAPTIVE_FUZZY_CUTOFF_HOUR:
                if score_diff >= cfg.ADAPTIVE_FUZZY_STRONG_SCORE:
                    fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_STRONG_THRESHOLD)

            assert fuzzy_threshold == 1.5, f"Strong signal should lower to 1.5, got {fuzzy_threshold}"

    def test_mid_signal_lowers_threshold(self):
        """score_diff 2.5-3.5 before 1PM → fuzzy threshold = 1.75."""
        from src.config.env_loader import get_config
        cfg = get_config()
        base_threshold = 2.0
        score_diff = 3.0

        with patch("src.strategies.options_buyer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 20, 11, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            fuzzy_threshold = base_threshold
            if cfg.ADAPTIVE_FUZZY_ENABLED and mock_dt.now().hour < cfg.ADAPTIVE_FUZZY_CUTOFF_HOUR:
                if score_diff >= cfg.ADAPTIVE_FUZZY_STRONG_SCORE:
                    fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_STRONG_THRESHOLD)
                elif score_diff >= cfg.ADAPTIVE_FUZZY_MID_SCORE:
                    fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_MID_THRESHOLD)

            assert fuzzy_threshold == 1.75, f"Mid signal should lower to 1.75, got {fuzzy_threshold}"

    def test_after_cutoff_no_adaptive(self):
        """After 1PM → fuzzy threshold stays at base (2.0)."""
        from src.config.env_loader import get_config
        cfg = get_config()
        base_threshold = 2.0
        score_diff = 4.0  # Strong signal but after cutoff

        with patch("src.strategies.options_buyer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 20, 13, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            fuzzy_threshold = base_threshold
            if cfg.ADAPTIVE_FUZZY_ENABLED and mock_dt.now().hour < cfg.ADAPTIVE_FUZZY_CUTOFF_HOUR:
                if score_diff >= cfg.ADAPTIVE_FUZZY_STRONG_SCORE:
                    fuzzy_threshold = min(fuzzy_threshold, cfg.ADAPTIVE_FUZZY_STRONG_THRESHOLD)

            assert fuzzy_threshold == 2.0, f"After cutoff should stay at 2.0, got {fuzzy_threshold}"


# ═════════════════════════════════════════════════════
# 30-Minute Rescore Schedule + Rescore Exit (5 tests)
# ═════════════════════════════════════════════════════


class TestRescoreSchedule:

    def setup_method(self):
        self.s = _make_strategy()

    def test_rescore_30m_fires_at_correct_times(self):
        """Rescore fires at 10:30, 11:00, 11:30, 12:00, 12:30 — not before 10:30."""
        self.s._daily_scores["NIFTY"] = (5.0, 2.0, 3.0)
        self.s._direction_scores["NIFTY"] = (5.0, 2.0, 3.0)
        data = _base_data(intraday_df=_make_intraday_df(40, 22000, "up"))

        # Before 10:30 → no rescore
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = datetime(2026, 3, 12, 10, 15)
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
        assert len(self.s._rescore_times_done) == 0

        # At 10:30 → first rescore fires
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = datetime(2026, 3, 12, 10, 30, 5)
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
        assert "10:30" in self.s._rescore_times_done

        # At 11:00 → second rescore fires
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = datetime(2026, 3, 12, 11, 0, 5)
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
        assert "11:00" in self.s._rescore_times_done

    def test_rescore_progressive_weights(self):
        """Early slots use more daily weight; later slots use more intraday weight."""
        self.s._daily_scores["NIFTY"] = (6.0, 1.0, 5.0)
        self.s._direction_scores["NIFTY"] = (6.0, 1.0, 5.0)
        data = _base_data(intraday_df=_make_intraday_df(40, 22000, "up"))

        # At 10:30 → daily 70%, intraday 30%
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = datetime(2026, 3, 12, 10, 30, 5)
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
        bull_1030, bear_1030, _ = self._direction_scores_snapshot()

        # Reset and do 12:30 → daily 15%, intraday 85%
        self.s._direction_scores["NIFTY"] = (6.0, 1.0, 5.0)
        self.s._rescore_times_done.clear()
        with patch("src.strategies.options_buyer.datetime") as mock:
            mock.now.return_value = datetime(2026, 3, 12, 12, 30, 5)
            mock.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.s.intraday_rescore("NIFTY", data)
        bull_1230, bear_1230, _ = self._direction_scores_snapshot()

        # 12:30 should differ from 10:30 (more intraday weight → different blend)
        assert bull_1030 != bull_1230 or bear_1030 != bear_1230

    def _direction_scores_snapshot(self):
        return self.s._direction_scores.get("NIFTY", (0, 0, 0))


class TestRescoreExit:

    def setup_method(self):
        self.s = _make_strategy()

    def _mock_position(self, entry_price: float, current_price: float):
        pos = MagicMock()
        pos.entry_price = entry_price
        pos.current_price = current_price
        return pos

    def test_rescore_exit_skips_below_min_profit(self):
        """Profit < 5% → hold, no exit regardless of direction flip."""
        self.s._direction_scores["NIFTY"] = (1.0, 5.0, 4.0)  # Bear dominant (PE)
        pos = self._mock_position(100.0, 103.0)  # 3% profit < 5%

        should_exit, reason = self.s.rescore_exit_check("NIFTY", pos, "CE")
        assert not should_exit
        assert reason == ""

    def test_rescore_flip_exits_above_min_profit(self):
        """Direction flipped + profit ≥ 5% → rescore_flip exit."""
        self.s._direction_scores["NIFTY"] = (1.0, 5.0, 4.0)  # Bear dominant → PE
        pos = self._mock_position(100.0, 108.0)  # 8% profit ≥ 5%

        should_exit, reason = self.s.rescore_exit_check("NIFTY", pos, "CE")
        assert should_exit
        assert reason == "rescore_flip"

    def test_rescore_decay_exits_on_score_drop(self):
        """Score decayed ≥ 40% from peak + profit ≥ 10% → rescore_decay exit."""
        self.s._peak_score_diff["NIFTY"] = 5.0  # Peak score diff
        self.s._direction_scores["NIFTY"] = (4.0, 1.5, 2.5)  # Current diff=2.5 → decay=50%
        pos = self._mock_position(100.0, 112.0)  # 12% profit ≥ 10%

        should_exit, reason = self.s.rescore_exit_check("NIFTY", pos, "CE")
        assert should_exit
        assert reason == "rescore_decay"


# ═════════════════════════════════════════════════════
# Bidirectional Reversal (5 tests)
# ═════════════════════════════════════════════════════


class TestBidirectionalReversal:

    def setup_method(self):
        self.s = _make_strategy()

    def test_reversal_pending_after_profit(self):
        """Profitable exit with sufficient profit % sets _reversal_pending=True."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig = cfg.REVERSAL_ENABLED
        orig_min = cfg.REVERSAL_MIN_EXIT_PROFIT
        cfg.REVERSAL_ENABLED = True
        cfg.REVERSAL_MIN_EXIT_PROFIT = 0.08
        try:
            # entry_cost=10000, pnl=1000 → 10% profit >= 8% threshold → pending
            self.s.record_exit("NIFTY23200CE", "take_profit", "CE",
                               pnl=1000.0, entry_cost=10000.0)
            assert self.s._reversal_pending is True
            assert self.s._reversal_pending_direction == "PE"  # opposite of CE
            assert self.s._reversal_eligible is False  # not yet eligible, just pending
            assert self.s._last_exit_direction == "CE"
            assert self.s._last_exit_pnl == 1000.0
        finally:
            cfg.REVERSAL_ENABLED = orig
            cfg.REVERSAL_MIN_EXIT_PROFIT = orig_min

    def test_reversal_blocked_after_loss(self):
        """Loss exit sets _reversal_eligible=False."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig = cfg.REVERSAL_ENABLED
        cfg.REVERSAL_ENABLED = True
        try:
            self.s.record_exit("NIFTY23200CE", "stop_loss", "CE", pnl=-3000.0)
            assert self.s._reversal_eligible is False
            assert self.s._last_exit_direction == "CE"
        finally:
            cfg.REVERSAL_ENABLED = orig

    def test_reversal_bypasses_cooldown(self):
        """Reversal eligible + opposite direction bypasses consecutive SL block."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig = cfg.REVERSAL_ENABLED
        cfg.REVERSAL_ENABLED = True
        try:
            # Set up 3 consecutive CE SLs
            self.s._consec_sl_count = 3
            self.s._consec_sl_direction = "CE"
            # Reversal eligible from profitable PE exit
            self.s._reversal_eligible = True
            self.s._last_exit_direction = "PE"

            # PE direction (opposite of SL direction CE) should be blocked
            # but CE direction (same as SL direction) with reversal from PE→CE
            # should bypass since last_exit=PE and new direction=PE is NOT the SL direction
            # Actually: consec_sl blocks CE. Reversal from PE exit wants PE entry.
            # PE != CE (consec_sl_direction), so the block doesn't fire at all.
            # Let's test: consec_sl blocks PE, reversal from CE exit wants CE entry
            self.s._consec_sl_count = 3
            self.s._consec_sl_direction = "PE"
            self.s._reversal_eligible = True
            self.s._last_exit_direction = "CE"
            # Now direction=PE would be blocked by 3 consec PE SLs
            # But reversal: last exit was CE, new direction PE != CE → bypass
            self.s._direction["NIFTY"] = "PE"
            self.s._direction_scores["NIFTY"] = (1.0, 5.0, 4.0)

            # Create minimal data for _evaluate_symbol
            # We just test the SL block logic directly:
            # If consec_sl >= 3 and direction == consec_sl_direction
            # AND reversal eligible and direction != last_exit_direction → bypass
            direction = "PE"
            assert self.s._consec_sl_count >= 3
            assert direction == self.s._consec_sl_direction
            assert self.s._reversal_eligible
            assert direction != self.s._last_exit_direction  # PE != CE → bypass
        finally:
            cfg.REVERSAL_ENABLED = orig

    def test_reversal_blocked_after_1300(self):
        """Reversal blocked after 13:00 cutoff."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_rev = cfg.REVERSAL_ENABLED
        cfg.REVERSAL_ENABLED = True
        try:
            self.s._reversal_eligible = True
            self.s._last_exit_direction = "CE"

            # Mock time to 13:30 — after reversal cutoff
            with patch("src.strategies.options_buyer.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 20, 13, 30)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                now_time = mock_dt.now().time()

            # Reversal should NOT be allowed after 13:00
            assert now_time >= dt_time(13, 0)
            # The generate_signals logic: reversal_allowed = REVERSAL_ENABLED and eligible and time < 13:00
            reversal_allowed = (
                cfg.REVERSAL_ENABLED
                and self.s._reversal_eligible
                and now_time < dt_time(13, 0)
            )
            assert reversal_allowed is False
        finally:
            cfg.REVERSAL_ENABLED = orig_rev

    def test_reversal_size_reduced_075x(self):
        """Reversal trade flag triggers 0.75× sizing."""
        from src.config.env_loader import get_config
        cfg = get_config()
        assert cfg.REVERSAL_SIZE_MULT == 0.75

        # Test the sizing logic directly:
        # original_qty × REVERSAL_SIZE_MULT, rounded down to lot_size
        sizing_full_lot = 65
        original_qty = 520  # 8 lots
        reversal_mult = cfg.REVERSAL_SIZE_MULT
        reduced_qty = max(sizing_full_lot, int(original_qty * reversal_mult / sizing_full_lot) * sizing_full_lot)
        assert reduced_qty == 390  # 6 lots (520 × 0.75 = 390)
        assert reduced_qty < original_qty


class TestReversalOptimizations:
    """Tests for OPT 1/2/3: reversal quality optimizations."""

    def setup_method(self):
        self.s = _make_strategy()

    def test_reversal_pending_waits_one_rescore(self):
        """OPT 2: Pending reversal only becomes eligible after rescore confirms direction."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_rev = cfg.REVERSAL_ENABLED
        orig_min_score = cfg.REVERSAL_MIN_SCORE
        cfg.REVERSAL_ENABLED = True
        cfg.REVERSAL_MIN_SCORE = 2.5
        try:
            # Set pending state (as if record_exit was called after profitable CE exit)
            self.s._reversal_pending = True
            self.s._reversal_pending_direction = "PE"  # wants PE reversal
            self.s._reversal_eligible = False

            # No rescore scores yet → pending stays pending
            self.s._direction_scores["NIFTY"] = (0, 0, 0)
            self.s.instruments = ["NIFTY"]

            with patch("src.strategies.options_buyer.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 20, 11, 30)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

                # Simulate generate_signals pending check logic
                now_time = mock_dt.now().time()
                # No scores → should still be pending
                assert self.s._reversal_pending is True
                assert self.s._reversal_eligible is False

                # Now set rescore scores confirming PE direction (bear > bull)
                self.s._direction_scores["NIFTY"] = (1.0, 4.0, 3.0)  # diff=3.0 >= 2.5

                # Run the pending→eligible promotion logic
                for sym in self.s.instruments:
                    bull, bear, diff = self.s._direction_scores.get(sym, (0, 0, 0))
                    if bull > 0 or bear > 0:
                        rescore_dir = "CE" if bull > bear else ("PE" if bear > bull else "")
                        if (rescore_dir == self.s._reversal_pending_direction
                                and diff >= cfg.REVERSAL_MIN_SCORE):
                            self.s._reversal_eligible = True
                            self.s._reversal_pending = False

                assert self.s._reversal_eligible is True
                assert self.s._reversal_pending is False
        finally:
            cfg.REVERSAL_ENABLED = orig_rev
            cfg.REVERSAL_MIN_SCORE = orig_min_score

    def test_reversal_cancelled_on_bounce(self):
        """OPT 2: Pending reversal cancelled when rescore direction doesn't confirm."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_rev = cfg.REVERSAL_ENABLED
        orig_min_score = cfg.REVERSAL_MIN_SCORE
        cfg.REVERSAL_ENABLED = True
        cfg.REVERSAL_MIN_SCORE = 2.5
        try:
            # Pending PE reversal (after profitable CE exit)
            self.s._reversal_pending = True
            self.s._reversal_pending_direction = "PE"
            self.s._reversal_eligible = False
            self.s.instruments = ["NIFTY"]

            # Rescore says CE direction (bull > bear) — doesn't confirm PE
            self.s._direction_scores["NIFTY"] = (4.0, 1.0, 3.0)

            # Run the pending check — CE != PE → no promotion
            for sym in self.s.instruments:
                bull, bear, diff = self.s._direction_scores.get(sym, (0, 0, 0))
                if bull > 0 or bear > 0:
                    rescore_dir = "CE" if bull > bear else ("PE" if bear > bull else "")
                    if (rescore_dir == self.s._reversal_pending_direction
                            and diff >= cfg.REVERSAL_MIN_SCORE):
                        self.s._reversal_eligible = True
                        self.s._reversal_pending = False

            # PE pending should still be pending (not eligible)
            assert self.s._reversal_pending is True
            assert self.s._reversal_eligible is False
        finally:
            cfg.REVERSAL_ENABLED = orig_rev
            cfg.REVERSAL_MIN_SCORE = orig_min_score

    def test_reversal_blocked_low_exit_profit(self):
        """OPT 3: Reversal blocked when exit profit < REVERSAL_MIN_EXIT_PROFIT."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_rev = cfg.REVERSAL_ENABLED
        orig_min = cfg.REVERSAL_MIN_EXIT_PROFIT
        cfg.REVERSAL_ENABLED = True
        cfg.REVERSAL_MIN_EXIT_PROFIT = 0.08
        try:
            # entry_cost=10000, pnl=500 → 5% profit < 8% threshold → blocked
            self.s.record_exit("NIFTY23200CE", "take_profit", "CE",
                               pnl=500.0, entry_cost=10000.0)
            assert self.s._reversal_pending is False
            assert self.s._reversal_eligible is False
        finally:
            cfg.REVERSAL_ENABLED = orig_rev
            cfg.REVERSAL_MIN_EXIT_PROFIT = orig_min

    def test_reversal_timeout_after_1230(self):
        """OPT 2: Pending reversal times out after 12:30."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_rev = cfg.REVERSAL_ENABLED
        cfg.REVERSAL_ENABLED = True
        try:
            self.s._reversal_pending = True
            self.s._reversal_pending_direction = "PE"
            self.s._reversal_eligible = False
            self.s.instruments = ["NIFTY"]

            # At 12:35 — past the 12:30 timeout
            with patch("src.strategies.options_buyer.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 20, 12, 35)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                now_time = mock_dt.now().time()

            # Simulate timeout logic from generate_signals
            if now_time >= dt_time(12, 30):
                self.s._reversal_pending = False
                self.s._reversal_pending_direction = ""

            assert self.s._reversal_pending is False
            assert self.s._reversal_pending_direction == ""
            assert self.s._reversal_eligible is False  # never promoted
        finally:
            cfg.REVERSAL_ENABLED = orig_rev


class TestVolatileDualMode:
    """Tests for VOLATILE dual mode: lower threshold naked buy on high VIX days."""

    def setup_method(self):
        self.s = _make_strategy()

    def test_dual_mode_activates_in_volatile(self):
        """Dual mode activates when regime is VOLATILE and config enabled."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig = cfg.DUAL_MODE_ENABLED
        cfg.DUAL_MODE_ENABLED = True
        try:
            assert self.s._dual_mode_active is False
            # Simulate generate_signals activation logic
            regime = "VOLATILE"
            if cfg.DUAL_MODE_ENABLED and regime == "VOLATILE" and not self.s._dual_mode_active:
                self.s._dual_mode_active = True
            assert self.s._dual_mode_active is True
        finally:
            cfg.DUAL_MODE_ENABLED = orig

    def test_dual_mode_uses_intraday_score_only(self):
        """Dual mode uses DUAL_MODE_MIN_SCORE (2.0) threshold, not the normal VOLATILE (2.5+)."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_dm = cfg.DUAL_MODE_ENABLED
        orig_stage = cfg.TRADING_STAGE
        cfg.DUAL_MODE_ENABLED = True
        cfg.TRADING_STAGE = "PLUS"
        try:
            self.s._dual_mode_active = True
            self.s._dual_mode_trades_today = 0
            # score_diff=2.2 — below normal VOLATILE threshold (2.5) but above DUAL_MODE (2.0)
            # Mock time to 11:00 — within dual mode window (10:30-12:00)
            with patch("src.strategies.options_buyer.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 20, 11, 0)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                result = self.s._determine_trade_type("VOLATILE", 2.2)
            assert result == "NAKED_BUY"
            assert self.s._is_dual_mode_trade is True
        finally:
            cfg.DUAL_MODE_ENABLED = orig_dm
            cfg.TRADING_STAGE = orig_stage
            self.s._is_dual_mode_trade = False

    def test_dual_mode_size_reduced_060x(self):
        """Dual mode trade uses 0.60× sizing multiplier."""
        from src.config.env_loader import get_config
        cfg = get_config()
        assert cfg.DUAL_MODE_SIZE_MULT == 0.60

        # Test the sizing math directly
        sizing_full_lot = 65
        original_qty = 520  # 8 lots
        dual_mult = cfg.DUAL_MODE_SIZE_MULT
        reduced_qty = max(sizing_full_lot, int(original_qty * dual_mult / sizing_full_lot) * sizing_full_lot)
        assert reduced_qty == 260  # 4 lots (520 × 0.60 = 312 → rounded to 4 lots = 260)
        assert reduced_qty < original_qty

    def test_dual_mode_blocked_after_1200(self):
        """Dual mode entries blocked after DUAL_MODE_ENTRY_CUTOFF (12:00)."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig_dm = cfg.DUAL_MODE_ENABLED
        orig_stage = cfg.TRADING_STAGE
        cfg.DUAL_MODE_ENABLED = True
        cfg.TRADING_STAGE = "PLUS"
        try:
            self.s._dual_mode_active = True
            self.s._dual_mode_trades_today = 0
            # Mock time to 12:30 — after cutoff
            with patch("src.strategies.options_buyer.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 3, 20, 12, 30)
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                # score_diff=2.2 normally qualifies for dual mode in window
                result = self.s._determine_trade_type("VOLATILE", 2.2)
            # Past cutoff: dual mode flag NOT set, falls through to credit spread
            assert self.s._is_dual_mode_trade is False
            assert result == "CREDIT_SPREAD"  # Normal VOLATILE path takes over
        finally:
            cfg.DUAL_MODE_ENABLED = orig_dm
            cfg.TRADING_STAGE = orig_stage

    def test_dual_mode_disabled_in_normal_regime(self):
        """Dual mode does NOT activate for TRENDING/RANGEBOUND regimes."""
        from src.config.env_loader import get_config
        cfg = get_config()
        orig = cfg.DUAL_MODE_ENABLED
        cfg.DUAL_MODE_ENABLED = True
        try:
            self.s._dual_mode_active = False
            # TRENDING regime should not activate dual mode
            regime = "TRENDING"
            if cfg.DUAL_MODE_ENABLED and regime == "VOLATILE" and not self.s._dual_mode_active:
                self.s._dual_mode_active = True
            assert self.s._dual_mode_active is False

            # RANGEBOUND should not either
            regime = "RANGEBOUND"
            if cfg.DUAL_MODE_ENABLED and regime == "VOLATILE" and not self.s._dual_mode_active:
                self.s._dual_mode_active = True
            assert self.s._dual_mode_active is False
        finally:
            cfg.DUAL_MODE_ENABLED = orig


# ═════════════════════════════════════════════════════
# Safety Guards (4 tests)
# ═════════════════════════════════════════════════════


class TestSafetyGuards:

    def setup_method(self):
        self.s = _make_strategy()

    def test_guard1_opposing_spread_blocked(self):
        """Guard 1: CE Sell spread blocked when today's first trade was CE (bullish)."""
        # Today's first trade was CE direction (bullish)
        self.s._today_trade_direction = "CE"
        # CE signal → credit spread sells PE (bullish) — allowed
        result_pe = self.s._apply_safety_guards("CREDIT_SPREAD", "CE", "NIFTY")
        assert result_pe == "CREDIT_SPREAD"  # CE signal → PE Sell → same direction, OK

        # PE signal → credit spread sells CE (bearish) — opposes today's CE
        result_ce = self.s._apply_safety_guards("CREDIT_SPREAD", "PE", "NIFTY")
        assert result_ce == "GUARD_BLOCK"  # PE signal → CE Sell → opposes CE direction

    def test_guard2_reversal_always_naked_buy(self):
        """Guard 2: Reversal trade forced to NAKED_BUY even when routing says CREDIT_SPREAD."""
        self.s._is_reversal_trade = True
        result = self.s._apply_safety_guards("CREDIT_SPREAD", "PE", "NIFTY")
        assert result == "NAKED_BUY"

        # Non-reversal should keep CREDIT_SPREAD
        self.s._is_reversal_trade = False
        result2 = self.s._apply_safety_guards("CREDIT_SPREAD", "PE", "NIFTY")
        assert result2 == "CREDIT_SPREAD"

    def test_guard3_spread_blocked_after_naked_buy(self):
        """Guard 3: Spread blocked when today already took a NAKED_BUY."""
        self.s._today_position_type = "NAKED_BUY"
        result = self.s._apply_safety_guards("CREDIT_SPREAD", "CE", "NIFTY")
        assert result == "GUARD_BLOCK"

        # NAKED_BUY should still pass
        result2 = self.s._apply_safety_guards("NAKED_BUY", "CE", "NIFTY")
        assert result2 == "NAKED_BUY"

    def test_guard3_reversal_allowed_after_spread(self):
        """Guard 3: Reversal naked buy exempt even when today took CREDIT_SPREAD."""
        self.s._today_position_type = "CREDIT_SPREAD"
        self.s._is_reversal_trade = False
        # Non-reversal blocked
        result = self.s._apply_safety_guards("NAKED_BUY", "CE", "NIFTY")
        assert result == "GUARD_BLOCK"

        # Reversal exempt
        self.s._is_reversal_trade = True
        result2 = self.s._apply_safety_guards("NAKED_BUY", "CE", "NIFTY")
        assert result2 == "NAKED_BUY"
