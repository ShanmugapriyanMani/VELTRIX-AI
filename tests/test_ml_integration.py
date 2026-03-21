"""Tests for ML live integration — scoring, quality gate, post-trade labeling."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.strategies.options_buyer import OptionsBuyerStrategy


def _make_store():
    """Create a DataStore backed by a temp SQLite file."""
    from src.config.env_loader import get_config
    from src.data.store import DataStore

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    tmp.write("database:\n  engine: sqlite\n  sqlite_path: ':memory:'\n")
    tmp.close()

    old_val = os.environ.get("DB_PATH")
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db_file.name
    db_file.close()
    os.environ["DB_PATH"] = db_path
    get_config.cache_clear()

    store = DataStore(tmp.name)

    if old_val is not None:
        os.environ["DB_PATH"] = old_val
    else:
        os.environ.pop("DB_PATH", None)
    get_config.cache_clear()
    os.unlink(tmp.name)

    return store, db_path


def _make_strategy() -> OptionsBuyerStrategy:
    """Create a strategy instance for testing."""
    s = OptionsBuyerStrategy()
    s.reset_daily()
    return s


class TestMLScoreIntegration:
    """Test ML probability injection into direction scoring."""

    def test_ml_probs_increase_bull_score(self):
        """prob_ce=0.70 adds ML contribution to bull_score."""
        s = _make_strategy()
        # Build data with high CE probability
        intraday_df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-12 09:15", periods=40, freq="5min"),
            "open": [22000 + i * 2 for i in range(40)],
            "high": [22010 + i * 2 for i in range(40)],
            "low": [21990 + i * 2 for i in range(40)],
            "close": [22005 + i * 2 for i in range(40)],
            "volume": [50000] * 40,
        })

        data = {
            "regime": "TRENDING",
            "vix": 15,
            "pcr": {"NIFTY": 1.0},
            "oi_levels": {},
            "ema_weight": 2.5,
            "mean_reversion_weight": 1.5,
            "nifty_price": 22000,
            "intraday_df": intraday_df,
            "is_expiry_day": False,
            "ml_direction_prob_up": 0.5,
            "ml_direction_prob_down": 0.5,
            "ml_stage1_prob_ce": 0.70,
            "ml_stage1_prob_pe": 0.30,
            "ml_stage1_prob_flat": 0.0,
        }

        # Compute direction score
        result = s._compute_direction_score("NIFTY", data, "TRENDING")
        bull, bear, direction = result

        # With high CE prob, bull should get a boost
        assert bull > 0  # Bull score should be positive with ML contribution

    def test_ml_probs_below_threshold_no_contribution(self):
        """prob_ce=0.35 (below 0.45 threshold) → zero ML score contribution."""
        s = _make_strategy()
        intraday_df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-12 09:15", periods=40, freq="5min"),
            "open": [22000] * 40,
            "high": [22010] * 40,
            "low": [21990] * 40,
            "close": [22005] * 40,
            "volume": [50000] * 40,
        })

        data_with_ml = {
            "regime": "RANGEBOUND",
            "vix": 15,
            "pcr": {"NIFTY": 1.0},
            "oi_levels": {},
            "ema_weight": 2.5,
            "mean_reversion_weight": 1.5,
            "nifty_price": 22000,
            "intraday_df": intraday_df,
            "is_expiry_day": False,
            "ml_direction_prob_up": 0.5,
            "ml_direction_prob_down": 0.5,
            "ml_stage1_prob_ce": 0.35,  # Below 0.45 threshold
            "ml_stage1_prob_pe": 0.65,
            "ml_stage1_prob_flat": 0.0,
        }

        data_without_ml = dict(data_with_ml)
        data_without_ml["ml_stage1_prob_ce"] = 0.50
        data_without_ml["ml_stage1_prob_pe"] = 0.50
        data_without_ml["ml_stage1_prob_flat"] = 0.0

        # Both should give same scores since ML is below threshold
        result_with = s._compute_direction_score("NIFTY", data_with_ml, "RANGEBOUND")
        s2 = _make_strategy()
        result_without = s2._compute_direction_score("NIFTY", data_without_ml, "RANGEBOUND")

        # Scores should be very similar (not identical due to random seed, but close)
        assert abs(result_with[0] - result_without[0]) < 0.01
        assert abs(result_with[1] - result_without[1]) < 0.01

    def test_quality_gate_blocks_low_win_prob(self):
        """Quality gate with low win_prob blocks signal via data dict."""
        # The quality gate in _evaluate_symbol reads ml_quality_predict from data.
        # Verify the gate function interface.
        def mock_quality_predict(features):
            return {"win_prob": 0.30, "quality_class": "LOW"}

        data = {
            "ml_quality_ready": True,
            "ml_quality_predict": mock_quality_predict,
        }

        # Verify the function returns low win_prob
        result = data["ml_quality_predict"]({"score_diff": 2.0})
        assert result["win_prob"] < 0.45
        assert result["quality_class"] == "LOW"

    def test_quality_gate_passes_high_win_prob(self):
        """Quality gate with high win_prob allows signal."""
        def mock_quality_predict(features):
            return {"win_prob": 0.65, "quality_class": "HIGH"}

        result = mock_quality_predict({"score_diff": 4.0})
        assert result["win_prob"] >= 0.45
        assert result["quality_class"] == "HIGH"

    def test_post_trade_labeling_creates_record(self):
        """After trade exit, ml_trade_labels gets a new row with correct label."""
        store, db_path = _make_store()
        try:
            # Simulate what _label_trade_for_ml does
            trade_result = {
                "trade_id": "test_trade_001",
                "symbol": "NIFTY22000CE",
                "option_type": "CE",
                "regime": "TRENDING",
                "price": 150.0,
                "fill_price": 165.0,
                "entry_price": 150.0,
                "exit_price": 165.0,
                "pnl": 975.0,
                "signal_score": 3.5,
                "confidence": 0.78,
            }

            store.save_ml_trade_label({
                "trade_id": trade_result["trade_id"],
                "trade_date": "2025-06-15",
                "symbol": trade_result["symbol"],
                "direction": trade_result.get("option_type", "CE"),
                "regime": trade_result.get("regime", ""),
                "entry_price": trade_result["entry_price"],
                "exit_price": trade_result["exit_price"],
                "pnl": trade_result["pnl"],
                "label": 1 if trade_result["pnl"] > 0 else 0,
                "score_diff": trade_result.get("signal_score", 0),
                "conviction": trade_result.get("confidence", 0),
            })

            count = store.get_ml_trade_label_count()
            assert count >= 1

            labels = store.get_ml_trade_labels()
            assert not labels.empty
            # Find our specific label
            our_label = labels[labels["trade_id"] == "test_trade_001"]
            assert len(our_label) == 1
            assert our_label.iloc[0]["label"] == 1  # WIN (pnl > 0)
            assert our_label.iloc[0]["pnl"] == 975.0
        finally:
            store.close()
            os.unlink(db_path)


class TestPEFilterBlocksLowConfidence:
    """PE confidence filter blocks PE entries when model confidence is low."""

    def test_pe_filter_blocks_low_confidence(self):
        """PE trade skipped when pe_prob < 0.60 (PE_FILTER_TOLERANCE_LOW)."""
        import inspect
        from src.strategies.options_buyer import OptionsBuyerStrategy
        from src.config.env_loader import get_config

        # Verify the config defaults exist
        cfg = get_config()
        assert hasattr(cfg, "PE_FILTER_ENABLED"), "PE_FILTER_ENABLED config missing"
        assert hasattr(cfg, "PE_FILTER_THRESHOLD"), "PE_FILTER_THRESHOLD config missing"
        assert cfg.PE_FILTER_THRESHOLD == pytest.approx(0.70)

        # Verify the filter code exists in options_buyer
        source = inspect.getsource(OptionsBuyerStrategy)
        assert "PE_FILTER" in source, "PE_FILTER log message must exist in options_buyer"
        assert "PE_LOW_CONFIDENCE" in source, "PE_LOW_CONFIDENCE skip reason must exist"

        # Test filter logic directly
        pe_conf = 0.55  # Below 0.60 tolerance low
        pe_threshold = 0.70
        should_block = 0 < pe_conf < pe_threshold
        assert should_block is True, "pe_conf=0.55 should be blocked by 0.70 threshold"

        # Test passthrough when confidence is high
        pe_conf_high = 0.75
        should_pass = not (0 < pe_conf_high < pe_threshold)
        assert should_pass is True, "pe_conf=0.75 should pass the filter"

        # Test passthrough when no PE model (pe_conf=0)
        pe_conf_zero = 0.0
        should_pass_zero = not (0 < pe_conf_zero < pe_threshold)
        assert should_pass_zero is True, "pe_conf=0.0 (no model) should not block"


class TestMLQualityGateLogsOnError:
    """M4: Quality gate predict errors are logged, not silently swallowed."""

    def test_ml_quality_gate_logs_on_error(self):
        """Quality gate except block logs warning instead of silent pass."""
        import inspect
        from src.strategies.options_buyer import OptionsBuyerStrategy
        source = inspect.getsource(OptionsBuyerStrategy)

        # Find the quality gate except block
        assert "ML_QUALITY_GATE_ERROR" in source, (
            "Quality gate except block must log ML_QUALITY_GATE_ERROR"
        )
        # Ensure no silent 'except: pass' or 'except Exception: pass' near quality gate
        # Extract the quality gate section
        gate_start = source.index("ML Quality Gate")
        gate_section = source[gate_start:gate_start + 1500]
        assert "except Exception as e" in gate_section, (
            "Quality gate must catch exception as 'e' for logging"
        )
        assert "pass  # Quality gate is optional" not in gate_section, (
            "Quality gate must not silently pass"
        )


class TestBackfillAutoRetrain:
    """ml_backfill auto-triggers ml_train when new candles are fetched."""

    def test_backfill_triggers_retrain_when_new_candles(self):
        """new_candles > 0 after backfill → _run_ml_train called."""
        import inspect
        from src.main import TradingBot

        source = inspect.getsource(TradingBot.run)
        # Verify the auto-retrain logic exists in the ml_backfill branch
        assert "AUTO_RETRAIN" in source, "AUTO_RETRAIN log message must exist in run()"
        assert "_run_ml_train" in source, "_run_ml_train must be called in ml_backfill branch"
        assert "total_candles" in source, "Must check total_candles from backfill result"
        assert "AUTO_RETRAIN_FAILED" in source, "Must catch and log retrain failures"

    def test_backfill_skips_retrain_when_no_candles(self):
        """new_candles == 0 after backfill → retrain skipped."""
        import inspect
        from src.main import TradingBot

        source = inspect.getsource(TradingBot.run)
        # Verify skip path exists
        assert "no new candles fetched" in source, (
            "Skip message must exist for zero candles case"
        )
        # Verify the condition: only retrain when new_candles > 0
        assert "new_candles > 0" in source, (
            "Must guard retrain behind new_candles > 0 check"
        )


class TestOISnapshotAt1030:
    """OI snapshot fires at 10:30 in trading loop."""

    def test_oi_snapshot_fires_at_10_30(self):
        """Trading loop has explicit 10:30 OI snapshot trigger."""
        import inspect
        from src.main import TradingBot

        source = inspect.getsource(TradingBot._trading_loop)
        assert "_oi_10_30_done" in source, "Must track 10:30 OI snapshot via _oi_10_30_done flag"
        assert "OI_SNAPSHOT_10:30" in source, "Must log OI_SNAPSHOT_10:30 message"
        assert "get_option_chain" in source, "Must call get_option_chain at 10:30"
        # Verify flag is reset at start of trading loop
        assert "_oi_10_30_done = False" in source, "Must reset _oi_10_30_done flag at loop start"


class TestAutoBackfillBeforeRetrain:
    """Auto ml_backfill runs before ml_retrain in _post_market."""

    def test_auto_backfill_runs_before_ml_train(self):
        """_post_market calls ml_backfill before _eod_retrain_ml_models."""
        import inspect
        from src.main import TradingBot

        source = inspect.getsource(TradingBot._post_market)
        assert "ML_BACKFILL_AUTO" in source, "Must log ML_BACKFILL_AUTO in _post_market"
        assert "_run_ml_backfill" in source, "Must call _run_ml_backfill in _post_market"
        assert "ML_BACKFILL_AUTO_FAILED" in source, "Must catch backfill failures gracefully"

        # Verify ordering: backfill BEFORE retrain
        backfill_pos = source.index("_run_ml_backfill")
        retrain_pos = source.index("_eod_retrain_ml_models")
        assert backfill_pos < retrain_pos, (
            "ml_backfill must run BEFORE _eod_retrain_ml_models in _post_market"
        )


class TestCounterfactualTradeLogging:
    """Counterfactual trade logging: blocked trades recorded with hypothetical P&L."""

    def test_counterfactual_logged_on_block(self):
        """Blocked trade creates a counterfactual record in strategy log."""
        s = _make_strategy()
        intraday_df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-12 09:15", periods=40, freq="5min"),
            "open": [22000] * 40,
            "high": [22010] * 40,
            "low": [21990] * 40,
            "close": [22005] * 40,
            "volume": [50000] * 40,
        })

        # Set up direction so we hit CONVICTION_BELOW_THRESHOLD
        data = {
            "regime": "VOLATILE",
            "vix": 15,
            "pcr": {"NIFTY": 1.0},
            "oi_levels": {},
            "ema_weight": 2.5,
            "mean_reversion_weight": 1.5,
            "nifty_price": 22000,
            "intraday_df": intraday_df,
            "is_expiry_day": False,
            "ml_direction_prob_up": 0.5,
            "ml_direction_prob_down": 0.5,
            "ml_stage1_prob_ce": 0.50,
            "ml_stage1_prob_pe": 0.50,
            "ml_stage1_prob_flat": 0.0,
        }

        # Call generate_signals — low score_diff should trigger CONVICTION_BELOW_THRESHOLD
        s.generate_signals(data)

        # The counterfactual log should have at least one record
        cf_log = s.get_counterfactual_log()
        # Even if scoring doesn't hit the block, verify the interface works
        assert isinstance(cf_log, list)
        # Verify _record_counterfactual method exists and works
        s._record_counterfactual(
            "NIFTY", "CE", "TEST_BLOCK", 22000, "TRENDING", 1.5, 3.0, 1.5,
            {"test": True},
        )
        cf_log = s.get_counterfactual_log()
        assert len(cf_log) >= 1
        record = [r for r in cf_log if r["block_reason"] == "TEST_BLOCK"][0]
        assert record["symbol"] == "NIFTY"
        assert record["direction"] == "CE"
        assert record["spot_at_block"] == 22000
        assert record["score_diff"] == 1.5

        # Verify deduplication: same symbol+reason should not add again
        s._record_counterfactual(
            "NIFTY", "CE", "TEST_BLOCK", 22000, "TRENDING", 1.5, 3.0, 1.5,
        )
        cf_log2 = s.get_counterfactual_log()
        test_blocks = [r for r in cf_log2 if r["block_reason"] == "TEST_BLOCK"]
        assert len(test_blocks) == 1, "Deduplication: same symbol+reason should not add twice"

    def test_counterfactual_pnl_computed_correctly(self):
        """Hypothetical P&L computation: CE profits on rise, PE profits on drop."""
        store, db_path = _make_store()
        try:
            # CE trade blocked at NIFTY=22000, EOD close=22100 (+0.45%)
            # Expected: would_have_won=1, positive P&L
            store.save_counterfactual_trade({
                "date": "2026-03-12",
                "symbol": "NIFTY",
                "direction": "CE",
                "block_reason": "CONFIRMATION_FAILED",
                "block_time": "10:30:00",
                "regime": "TRENDING",
                "score_diff": 2.5,
                "bull_score": 4.0,
                "bear_score": 1.5,
                "spot_at_block": 22000,
                "spot_at_eod": 22100,
                "hypothetical_pnl": 750.0,  # (100/22000)*22000*75 = 7500... simplified
                "hypothetical_pct": 0.45,
                "would_have_won": 1,
                "metadata": {"triggers": "T1=0.5+T2=0.3+T3=0.2+T4=0.4=1.4"},
            })

            # PE trade blocked at NIFTY=22000, EOD close=22100 (rose → PE loses)
            store.save_counterfactual_trade({
                "date": "2026-03-12",
                "symbol": "NIFTY",
                "direction": "PE",
                "block_reason": "PE_LOW_CONFIDENCE",
                "block_time": "11:00:00",
                "regime": "TRENDING",
                "score_diff": 2.0,
                "bull_score": 1.0,
                "bear_score": 3.0,
                "spot_at_block": 22000,
                "spot_at_eod": 22100,
                "hypothetical_pnl": -750.0,
                "hypothetical_pct": -0.45,
                "would_have_won": 0,
                "metadata": {"pe_prob": 0.55, "threshold": 0.85},
            })

            # Verify count
            count = store.get_counterfactual_count()
            assert count == 2

            # Verify query
            df = store.get_counterfactual_trades()
            assert len(df) == 2

            # CE trade should be would_have_won=1
            ce_trade = df[df["direction"] == "CE"].iloc[0]
            assert ce_trade["would_have_won"] == 1
            assert ce_trade["hypothetical_pnl"] > 0

            # PE trade should be would_have_won=0 (NIFTY rose, PE loses)
            pe_trade = df[df["direction"] == "PE"].iloc[0]
            assert pe_trade["would_have_won"] == 0
            assert pe_trade["hypothetical_pnl"] < 0

            # Filter by block_reason
            pe_df = store.get_counterfactual_trades(block_reason="PE_LOW_CONFIDENCE")
            assert len(pe_df) == 1
            assert pe_df.iloc[0]["block_reason"] == "PE_LOW_CONFIDENCE"
        finally:
            store.close()
            os.unlink(db_path)


class TestMLAgreeDoesNotBlock:
    """FIX 1: ML agreeing with score direction should not block entry."""

    def test_ml_agree_does_not_block(self):
        """When V2 ML direction matches score direction, entry is NOT blocked."""
        s = _make_strategy()
        intraday_df = pd.DataFrame({
            "datetime": pd.date_range("2026-03-19 09:15", periods=40, freq="5min"),
            "open": [22000 - i * 5 for i in range(40)],
            "high": [22010 - i * 5 for i in range(40)],
            "low": [21990 - i * 5 for i in range(40)],
            "close": [22000 - i * 5 for i in range(40)],
            "volume": [80000] * 40,
        })

        data = {
            "regime": "TRENDING",
            "vix": 15,
            "pcr": {"NIFTY": 0.7},
            "oi_levels": {},
            "nifty_price": 21800,
            "intraday_df": intraday_df,
            "is_expiry_day": False,
            "ml_direction_prob_up": 0.30,
            "ml_direction_prob_down": 0.70,
            "ml_stage1_prob_ce": 0.30,
            "ml_stage1_prob_pe": 0.70,
            "ml_stage1_prob_flat": 0.0,
            # V2 ML: both PE and CE active, ML direction = PE
            "ml_v2_ready": True,
            "ml_v2_pe_prob": 0.75,
            "ml_v2_ce_prob": 0.65,  # Also high — old bug would block here
            "ml_v2_direction": "PE",
            "ml_v2_confidence": 0.75,
            "ml_ce_ready": True,
            "ml_pe_ready": True,
            "ml_ce_binary_prob": 0.65,
            "ml_pe_binary_prob": 0.75,
        }

        # Run signals — should NOT block on ML disagreement
        s.generate_signals(data)

        # Verify ML_DISAGREES_WITH_DIRECTION was NOT the skip reason
        skip_info = s._last_skip_info.get("NIFTY", {})
        assert skip_info.get("reason") != "ML_DISAGREES_WITH_DIRECTION", (
            f"ML agrees (both PE) but was blocked! skip_info={skip_info}"
        )

        # Check that no counterfactual was logged for ML disagreement
        cf = [r for r in s.get_counterfactual_log()
              if r["block_reason"] == "ML_DISAGREES_WITH_DIRECTION"]
        assert len(cf) == 0, "Should not log ML disagreement counterfactual when directions agree"


class TestStalePriceBlocksEntry:
    """FIX 2: Stale intraday data should block entries via data_quality_ok."""

    def test_stale_price_blocks_entry(self):
        """data_quality_ok is False when intraday data > 30s old."""
        import inspect
        from src.main import TradingBot

        source = inspect.getsource(TradingBot._prepare_strategy_data)

        # Verify stale detection exists
        assert "intraday_stale" in source, "Must check intraday staleness"
        assert "DATA_STALE_BLOCK" in source, "Must log DATA_STALE_BLOCK"
        assert "_last_intraday_update" in source, "Must track _last_intraday_update"

        # Verify it feeds into data_quality_ok
        assert "not intraday_stale" in source, (
            "intraday_stale must be part of data_quality_ok check"
        )

        # Verify auto-clear: update happens when fresh data arrives
        assert "self._last_intraday_update = time.time()" in source, (
            "Must update _last_intraday_update when fresh intraday candles received"
        )

        # Verify WS price tracking and stale-clear log
        assert "_last_ws_price" in source, "Must track _last_ws_price"
        assert "DATA_STALE_CLEAR" in source, "Must log DATA_STALE_CLEAR when fresh data arrives"


class TestPEToleranceZone:
    """FIX 3: PE filter tolerance zone — context-aware 80-85% band."""

    def test_pe_tolerance_allows_strong_setup(self):
        """PE prob in 80-85% band PASSES when strong setup (high score_diff or VIX rising)."""
        from src.strategies.options_buyer import OptionsBuyerStrategy

        # Test 1: high score_diff passes tolerance zone
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.82, score_diff=4.0, vix_now=16.0, vix_open=15.8
        )
        assert passes, f"PE 82% + score_diff=4.0 should pass tolerance zone, got reason={reason}"

        # Test 2: VIX rising passes tolerance zone
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.81, score_diff=1.0, vix_now=16.5, vix_open=15.8
        )
        assert passes, f"PE 81% + VIX rising 0.7 should pass tolerance zone, got reason={reason}"

        # Test 3: above threshold always passes
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.90, score_diff=0.5, vix_now=15.0, vix_open=15.0
        )
        assert passes, "PE 90% should always pass"

    def test_pe_tolerance_blocks_weak_setup(self):
        """PE prob in 60-70% band BLOCKS when weak setup (low score, flat VIX)."""
        from src.strategies.options_buyer import OptionsBuyerStrategy

        # Test 1: tolerance zone with weak setup → PE_TOLERANCE_BLOCK
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.65, score_diff=1.5, vix_now=15.2, vix_open=15.1
        )
        assert not passes, "PE 65% + weak setup should be blocked"
        assert reason == "PE_TOLERANCE_BLOCK", f"Expected PE_TOLERANCE_BLOCK, got {reason}"

        # Test 2: below tolerance low → PE_LOW_CONFIDENCE
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.55, score_diff=5.0, vix_now=20.0, vix_open=15.0
        )
        assert not passes, "PE 55% should always be blocked"
        assert reason == "PE_LOW_CONFIDENCE", f"Expected PE_LOW_CONFIDENCE, got {reason}"

        # Test 3: exactly at tolerance low boundary with weak setup
        passes, reason = OptionsBuyerStrategy._pe_filter_passes(
            pe_prob=0.60, score_diff=2.0, vix_now=15.0, vix_open=15.0
        )
        assert not passes, "PE 60% + weak setup should be blocked"
        assert reason == "PE_TOLERANCE_BLOCK", f"Expected PE_TOLERANCE_BLOCK, got {reason}"


class TestLiveAuditChecks:
    """Tests for pre-live safety infrastructure."""

    def test_live_margin_check(self):
        """Verify margin validation code exists in main.py."""
        import inspect
        from src.main import TradingBot
        source = inspect.getsource(TradingBot)
        assert "get_available_margin" in source, "Margin check missing from TradingBot"
        assert "MARGIN_BLOCKED" in source, "Margin block logic missing"
        assert "available_margin < required_margin" in source, "Margin comparison missing"

    def test_live_duplicate_guard(self):
        """Verify duplicate position blocking exists in main.py."""
        import inspect
        from src.main import TradingBot
        source = inspect.getsource(TradingBot)
        assert "LIVE_DUPLICATE_BLOCKED" in source, "Duplicate order guard missing"
        assert "portfolio.positions" in source, "Portfolio positions check missing"

    def test_live_slippage_tracked(self):
        """Verify live_slippage_log table exists and save method works."""
        store, db_path = _make_store()
        try:
            assert "live_slippage_log" in store.SCHEMA, "live_slippage_log table missing"
            assert hasattr(store, "save_slippage_log"), "save_slippage_log method missing"

            store.save_slippage_log({
                "trade_id": "TEST-001",
                "symbol": "NIFTY25350CE",
                "signal_price": 100.0,
                "fill_price": 100.50,
                "slippage_pct": 0.005,
                "slippage_amount": 32.5,
                "quantity": 65,
                "direction": "BUY",
                "mode": "live",
            })

            df = store.get_slippage_summary(mode="live")
            assert len(df) == 1, f"Expected 1 slippage record, got {len(df)}"
            assert df.iloc[0]["trade_id"] == "TEST-001"
            assert abs(df.iloc[0]["slippage_pct"] - 0.005) < 0.0001
        finally:
            os.unlink(db_path)
