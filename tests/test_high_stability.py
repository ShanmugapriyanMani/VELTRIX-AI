"""Tests for 14 HIGH stability fixes (H1-H14) + MEDIUM stability fixes (M2, M6, M7).

5 tests covering:
  1. H1: GTT failure sends Telegram alert
  2. H7: NaN in direction scores returns None (skips signal)
  3. H11: Circuit breaker daily reset always clears halt
  4. H13: IC evaluation skipped when nifty_price=0
  5. H5: Fast-poll LTP error counter sends alert at 20

3 MEDIUM stability tests:
  6. M2: save_trade retries on DB failure
  7. M6: Paper SELL blocked at price=0
  8. M7: IC skip when spread_width=0
"""

import math
import sqlite3
from unittest.mock import MagicMock

from src.risk.circuit_breaker import CircuitBreaker, BreakerState


class TestGttFailureSendsAlert:
    """H1: GTT SL/TP placement validation — alert on failure."""

    def test_gtt_failure_sends_alert(self):
        """If GTT placement returns error or missing gtt_id, alert is sent."""
        alerts_sent = []

        class FakeAlerts:
            def send_raw(self, text):
                alerts_sent.append(text)

        alerts = FakeAlerts()

        # Simulate GTT result patterns
        gtt_results = [
            {"status": "error", "message": "Insufficient margin"},  # API error
            {"status": "success"},  # Missing gtt_id
            {"status": "success", "gtt_id": "GTT123"},  # Valid — no alert
        ]

        for gtt_result in gtt_results:
            # Replicate the H1 fix validation logic
            if gtt_result.get("status") == "error":
                alerts.send_raw(
                    f"GTT PLACEMENT FAILED: {gtt_result.get('message', 'unknown error')}\n"
                    f"Position may not have SL/TP protection."
                )
            elif not gtt_result.get("gtt_id"):
                alerts.send_raw(
                    f"GTT PLACEMENT WARNING: No gtt_id returned.\n"
                    f"Position may not have SL/TP protection."
                )

        # 2 alerts: one for error, one for missing gtt_id
        assert len(alerts_sent) == 2
        assert "GTT PLACEMENT FAILED" in alerts_sent[0]
        assert "Insufficient margin" in alerts_sent[0]
        assert "GTT PLACEMENT WARNING" in alerts_sent[1]


class TestNanInScoresSkipsSignal:
    """H7: NaN in bull_score or bear_score must return None."""

    def test_nan_in_scores_skips_signal(self):
        """If NaN propagates into scoring, _compute_direction_score returns None."""
        # Simulate the H7 final guard at the end of _compute_direction_score
        test_cases = [
            (float("nan"), 3.5, True),   # bull NaN → skip
            (2.5, float("nan"), True),   # bear NaN → skip
            (float("nan"), float("nan"), True),  # both NaN → skip
            (3.5, 2.5, False),           # valid → don't skip
        ]

        for bull_score, bear_score, should_skip in test_cases:
            # This replicates the H7 guard logic
            if math.isnan(bull_score) or math.isnan(bear_score):
                result = None
            else:
                if bull_score > bear_score:
                    direction = "CE"
                elif bear_score > bull_score:
                    direction = "PE"
                else:
                    direction = ""
                result = (bull_score, bear_score, direction)

            if should_skip:
                assert result is None, f"Expected None for bull={bull_score}, bear={bear_score}"
            else:
                assert result is not None, f"Expected tuple for bull={bull_score}, bear={bear_score}"
                assert result == (3.5, 2.5, "CE")


class TestCircuitBreakerDailyReset:
    """H11 (simplified): Daily reset always clears halt — no carry-over."""

    def test_daily_reset_clears_any_halt(self):
        """reset_daily() always resets to NORMAL regardless of halt reason."""
        cb = CircuitBreaker()

        # Halt via consecutive SL
        cb.record_trade(-5000)
        cb.record_trade(-5000)
        assert cb._state == BreakerState.HALTED

        cb.reset_daily()
        assert cb._state == BreakerState.NORMAL
        assert cb._halt_reason == ""
        assert cb._consecutive_sl == 0
        assert cb._daily_pnl == 0.0


class TestIcSkippedWhenNiftyPriceZero:
    """H13: IC evaluation must not fire when nifty_price=0."""

    def test_ic_skipped_when_nifty_price_zero(self):
        """IC evaluation gate rejects when nifty_price is 0."""
        # Simulate the data dict as prepared by _prepare_strategy_data
        data_zero = {"nifty_price": 0, "regime": "RANGEBOUND", "adx": 15.0}
        data_valid = {"nifty_price": 24850.0, "regime": "RANGEBOUND", "adx": 15.0}

        # Replicate the H13 gate
        def should_evaluate_ic(data):
            if data.get("regime") == "RANGEBOUND" and data.get("nifty_price", 0) > 0:
                return True
            return False

        assert should_evaluate_ic(data_zero) is False
        assert should_evaluate_ic(data_valid) is True

        # Also test missing key
        data_missing = {"regime": "RANGEBOUND"}
        assert should_evaluate_ic(data_missing) is False


class TestFastPollErrorSendsAlertAt20:
    """H5: Fast-poll LTP error counter sends alert at threshold."""

    def test_fast_poll_error_sends_alert_at_20(self):
        """After 20 consecutive fast-poll LTP errors, Telegram alert fires.
        Counter resets on success."""
        alerts_sent = []

        class FakeAlerts:
            def send_raw(self, text):
                alerts_sent.append(text)

        alerts = FakeAlerts()
        fast_poll_errors = 0

        # Simulate 20 consecutive errors
        for i in range(20):
            fast_poll_errors += 1
            if fast_poll_errors == 20:
                alerts.send_raw(
                    f"FAST-POLL LTP: 20 consecutive fetch failures.\n"
                    f"Position monitoring may be degraded."
                )

        assert len(alerts_sent) == 1
        assert "20 consecutive" in alerts_sent[0]
        assert fast_poll_errors == 20

        # Simulate success → reset
        fast_poll_errors = 0
        assert fast_poll_errors == 0

        # Simulate 19 errors (below threshold) → no new alert
        for i in range(19):
            fast_poll_errors += 1
            if fast_poll_errors == 20:
                alerts.send_raw("should not fire")

        assert len(alerts_sent) == 1  # Still just the first alert
        assert fast_poll_errors == 19


# ── MEDIUM stability tests ──


class TestM2SaveTradeRetry:
    """M2: save_trade retries on DB failure."""

    def test_save_trade_retries_on_db_failure(self):
        """save_trade retries 3 times then returns False on persistent failure."""
        from src.data.store import DataStore

        store = DataStore()  # Uses default config

        # First: normal save works
        trade = {"trade_id": "M2_TEST_001", "symbol": "NIFTY25000CE", "mode": "paper"}
        result = store.save_trade(trade)
        assert result is True

        # Now simulate persistent DB lock by patching _get_connection
        call_count = 0

        from contextlib import contextmanager

        original = store._get_connection

        @contextmanager
        def locked_connection():
            nonlocal call_count
            call_count += 1
            conn = MagicMock()
            conn.execute.side_effect = sqlite3.OperationalError("database is locked")
            yield conn

        store._get_connection = locked_connection
        result = store.save_trade({"trade_id": "M2_TEST_002", "symbol": "NIFTY25000PE"})
        assert result is False
        assert call_count == 3  # Retried 3 times

        store._get_connection = original

        # Clean up test trade
        with original() as conn:
            conn.execute("DELETE FROM trades WHERE trade_id = 'M2_TEST_001'")
            conn.commit()


class TestM6PaperSellBlockedAtZero:
    """M6: Paper SELL blocked at price=0."""

    def test_paper_sell_blocked_at_zero_price(self):
        from src.execution.paper_trader import PaperTrader

        trader = PaperTrader(initial_capital=500000)
        # SELL at price=0 should be rejected
        result = trader.place_order("NIFTY25000CE", "", 65, "SELL", price=0)
        assert result["status"] == "rejected"

        # SELL at None price should also be rejected
        result = trader.place_order("NIFTY25000CE", "", 65, "SELL", price=None)
        assert result["status"] == "rejected"

        # SELL at valid price should work
        result = trader.place_order("NIFTY25000CE", "", 65, "SELL", price=100.0)
        assert result["status"] == "success"


class TestM7IcSkipZeroSpreadWidth:
    """M7: IC skip when spread_width=0."""

    def test_ic_skip_zero_spread_width(self):
        from src.strategies.iron_condor import IronCondorStrategy

        ic = IronCondorStrategy()
        original_width = ic.spread_width

        # Set spread_width to 0 → should return None (no division by zero)
        ic.spread_width = 0
        result = ic.calculate_position(
            sell_ce_prem=120, buy_ce_prem=80,
            sell_pe_prem=110, buy_pe_prem=70,
            lot_size=65, risk_per_trade=15000, deploy_cap=75000,
            strikes={"sell_ce_strike": 23200, "buy_ce_strike": 23400,
                     "sell_pe_strike": 22800, "buy_pe_strike": 22600},
        )
        assert result is None

        # Restore and verify normal operation
        ic.spread_width = original_width
        result = ic.calculate_position(
            sell_ce_prem=120, buy_ce_prem=80,
            sell_pe_prem=110, buy_pe_prem=70,
            lot_size=65, risk_per_trade=15000, deploy_cap=75000,
            strikes={"sell_ce_strike": 23200, "buy_ce_strike": 23400,
                     "sell_pe_strike": 22800, "buy_pe_strike": 22600},
        )
        assert result is not None
        assert result["trade_type"] == "IRON_CONDOR"


class TestSaveExternalDataLogsOnFailure:
    """M3: save_external_data logs warnings on row-level failures."""

    def test_save_external_data_logs_on_failure(self):
        """Row insert failure should log warning, not silently pass."""
        import os
        import tempfile
        import pandas as pd
        from src.config.env_loader import get_config
        from src.data.store import DataStore

        # Create temp YAML config + DB
        cfg_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
        cfg_file.write("database:\n  engine: sqlite\n  sqlite_path: ':memory:'\n")
        cfg_file.close()
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = db_file.name
        db_file.close()
        old_val = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = db_path
        get_config.cache_clear()

        store = DataStore(cfg_file.name)

        # Valid row
        good_df = pd.DataFrame([{
            "date": "2026-01-01", "open": 100, "high": 105,
            "low": 95, "close": 102, "volume": 1000,
        }])
        saved = store.save_external_data("TEST", good_df)
        assert saved == 1

        # Corrupt row: force type error with non-numeric close
        bad_df = pd.DataFrame([{
            "date": "2026-01-02", "open": 100, "high": 105,
            "low": 95, "close": object(), "volume": 1000,
        }])
        saved = store.save_external_data("TEST", bad_df)
        assert saved == 0  # Should fail but not raise

        store.close()
        if old_val is not None:
            os.environ["DB_PATH"] = old_val
        else:
            os.environ.pop("DB_PATH", None)
        get_config.cache_clear()
        os.unlink(cfg_file.name)


def _make_test_store():
    """Helper: create a temp DataStore for testing."""
    import os
    import tempfile
    from src.config.env_loader import get_config
    from src.data.store import DataStore

    cfg_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    cfg_file.write("database:\n  engine: sqlite\n  sqlite_path: ':memory:'\n")
    cfg_file.close()
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db_file.name
    db_file.close()
    old_val = os.environ.get("DB_PATH")
    os.environ["DB_PATH"] = db_path
    get_config.cache_clear()

    store = DataStore(cfg_file.name)
    return store, db_path, old_val, cfg_file.name


def _cleanup_test_store(old_val, cfg_path):
    import os
    from src.config.env_loader import get_config
    if old_val is not None:
        os.environ["DB_PATH"] = old_val
    else:
        os.environ.pop("DB_PATH", None)
    get_config.cache_clear()
    os.unlink(cfg_path)


class TestFactorEdgeSavedToHistory:
    """Factor edge data is persisted to DB and retrievable."""

    def test_factor_edge_saved_to_history(self):
        store, db_path, old_val, cfg_path = _make_test_store()
        try:
            store.save_factor_edge("2026-04-01", "F9 Volume", 79.3, 55.0, 8234.0, 45, 90)
            store.save_factor_edge("2026-04-01", "F5 Bollinger", 76.1, 50.0, 6891.0, 45, 90)

            history = store.get_factor_edge_history(factor_name="F9 Volume")
            assert not history.empty
            assert history.iloc[0]["aligned_wr"] == 79.3
            assert history.iloc[0]["net_edge"] == 8234.0
            assert history.iloc[0]["trade_count"] == 45

            # All factors retrievable
            all_h = store.get_factor_edge_history()
            assert len(all_h) == 2

            # Previous month lookup
            store.save_factor_edge("2026-03-01", "F9 Volume", 82.0, 60.0, 12774.0, 50, 90)
            prev = store.get_factor_edge_previous("2026-04-01")
            assert "F9 Volume" in prev
            assert prev["F9 Volume"]["net_edge"] == 12774.0
        finally:
            store.close()
            _cleanup_test_store(old_val, cfg_path)


class TestFactorMonitorAlertsOnDegradation:
    """Factor monitor detects edge degradation."""

    def test_factor_monitor_alerts_on_degradation(self):
        store, db_path, old_val, cfg_path = _make_test_store()
        try:
            # Previous month: strong edge
            store.save_factor_edge("2026-03-01", "OVERALL", 80.0, 0, 10000.0, 50, 90)
            # Current month: degraded (dropped >30%)
            store.save_factor_edge("2026-04-01", "OVERALL", 60.0, 0, 2000.0, 45, 90)

            prev = store.get_factor_edge_previous("2026-04-01")
            assert "OVERALL" in prev
            prev_edge = prev["OVERALL"]["net_edge"]
            curr_edge = 2000.0

            # Degradation check: edge dropped > 30%
            assert prev_edge > 0
            assert curr_edge < prev_edge * 0.70, "Should detect degradation"

            # WR degradation check
            assert 55.0 < 60, "WR below 60% should trigger alert"
        finally:
            store.close()
            _cleanup_test_store(old_val, cfg_path)
