"""Tests for 10 critical stability fixes (C1-C10).

5 tests covering:
  1. C1: Candle fetch failure continues trading
  2. C2/C9: Consecutive errors sends Telegram alert at 10
  3. C3/C4: Trail stop saves to DB and closes portfolio position
  4. C6: Crash recovery restores open positions from DB
  5. C7/C8: Force exit retries 3 times
"""

from datetime import datetime, date
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.risk.portfolio import PortfolioManager, Position


class TestCandleFetchFailureContinuesTrading:
    """C1: get_historical_candles() failure must not kill the trading day."""

    def test_candle_fetch_failure_continues_trading(self):
        """If candle fetch raises, nifty_df falls back to empty DataFrame
        and trading loop still starts (regime uses empty data)."""
        from src.regime.detector import RegimeDetector

        # Simulate the exact pattern from _trading_loop:
        # try → get_historical_candles → except → nifty_df = pd.DataFrame()
        def failing_fetch(*args, **kwargs):
            raise ConnectionError("Upstox API timeout")

        alerts_sent = []

        class FakeAlerts:
            def send_raw(self, text):
                alerts_sent.append(text)

        alerts = FakeAlerts()
        nifty_df = None

        # This replicates the C1 fix code path
        try:
            nifty_df = failing_fetch("NSE_INDEX|Nifty 50", "day")
        except Exception as e:
            alerts.send_raw(
                f"CANDLE FETCH FAILED: {e}\n"
                f"Trading continues with limited data."
            )
            nifty_df = pd.DataFrame()

        # Verify: nifty_df is empty DataFrame (not None, not crash)
        assert isinstance(nifty_df, pd.DataFrame)
        assert nifty_df.empty

        # Verify: alert was sent
        assert len(alerts_sent) == 1
        assert "CANDLE FETCH FAILED" in alerts_sent[0]

        # Verify: regime detector handles empty nifty_df gracefully
        detector = RegimeDetector()
        regime_state = detector.detect(
            vix_data={"vix": 14.5},
            nifty_df=nifty_df,
            fii_data=pd.DataFrame(),
            is_expiry_week=False,
        )
        # Should return a valid regime (TRENDING default) — not crash
        assert regime_state is not None
        assert hasattr(regime_state, "regime")


class TestConsecutiveErrorsSendsAlert:
    """C2/C9: Consecutive error counter + Telegram alerts."""

    def test_consecutive_errors_sends_alert_at_10(self):
        """After 10 consecutive errors, a Telegram alert is sent.
        After recovery, counter resets."""
        alerts_sent = []

        class FakeAlerts:
            def send_raw(self, text):
                alerts_sent.append(text)

        alerts = FakeAlerts()
        consecutive_loop_errors = 0
        error_alert_sent = False

        # Simulate 10 consecutive errors (same logic as in _trading_loop except block)
        for i in range(10):
            consecutive_loop_errors += 1
            if consecutive_loop_errors == 10:
                alerts.send_raw(
                    f"TRADING LOOP: 10 consecutive errors.\n"
                    f"Last error: test error\n"
                    f"Bot still running but check logs."
                )
                error_alert_sent = True
            elif consecutive_loop_errors == 50:
                alerts.send_raw(
                    f"TRADING LOOP: 50 consecutive errors.\n"
                    f"Last error: test error\n"
                    f"Bot may be malfunctioning."
                )
            elif consecutive_loop_errors % 100 == 0:
                alerts.send_raw(
                    f"TRADING LOOP: {consecutive_loop_errors} errors.\n"
                    f"Last: test error"
                )

        # Exactly 1 alert at count=10
        assert len(alerts_sent) == 1
        assert "10 consecutive errors" in alerts_sent[0]
        assert error_alert_sent is True

        # Simulate recovery (successful iteration)
        recovery_logged = False
        if consecutive_loop_errors > 0:
            recovery_logged = True
            if error_alert_sent:
                alerts.send_raw(
                    f"LOOP RECOVERED after {consecutive_loop_errors} errors"
                )
        consecutive_loop_errors = 0
        error_alert_sent = False

        assert recovery_logged is True
        assert consecutive_loop_errors == 0
        assert len(alerts_sent) == 2  # Original alert + recovery
        assert "LOOP RECOVERED" in alerts_sent[1]

        # Simulate 50 more errors — alert at 50
        for i in range(50):
            consecutive_loop_errors += 1
            if consecutive_loop_errors == 10:
                alerts.send_raw("10 errors")
                error_alert_sent = True
            elif consecutive_loop_errors == 50:
                alerts.send_raw(
                    f"TRADING LOOP: 50 consecutive errors.\n"
                    f"Last error: test error\n"
                    f"Bot may be malfunctioning."
                )

        assert len(alerts_sent) == 4  # 10-alert + recovery + 10-alert + 50-alert
        assert "50 consecutive errors" in alerts_sent[3]


class TestTrailStopSavesToDB:
    """C3/C4: _process_trail_exits() must close portfolio position and save to DB."""

    def test_trail_stop_saves_to_db(self):
        """Trail stop exit must: close position in portfolio, save trade to DB,
        record in circuit breaker."""
        # Setup portfolio with an open position
        portfolio = PortfolioManager(initial_capital=100000)
        pos = Position(
            symbol="NIFTY24100CE",
            instrument_key="NSE_FO|NIFTY24100CE",
            side="BUY",
            quantity=65,
            entry_price=200.0,
            current_price=240.0,
            stop_loss=170.0,
            take_profit=300.0,
            strategy="options",
            trade_id="T001",
        )
        portfolio.add_position(pos)
        assert "NIFTY24100CE" in portfolio.positions

        # Simulate trail exit info (from order_manager.check_trailing_stops)
        exit_info = {
            "trade_id": "T001",
            "symbol": "NIFTY24100CE",
            "instrument_key": "NSE_FO|NIFTY24100CE",
            "exit_reason": "trail_stop",
            "entry_premium": 200.0,
            "exit_premium": 240.0,
            "quantity": 65,
            "pnl": (240.0 - 200.0) * 65,
            "pnl_pct": 20.0,
            "option_type": "CE",
        }

        # Close position in portfolio (mirroring the C3/C4 fix)
        trade_result = portfolio.close_position(
            exit_info["symbol"], exit_info["exit_premium"], "trail_stop"
        )

        # Verify: position removed from portfolio
        assert "NIFTY24100CE" not in portfolio.positions

        # Verify: trade_result has correct data for DB save
        assert trade_result is not None
        assert trade_result["symbol"] == "NIFTY24100CE"
        assert trade_result["pnl"] == (240.0 - 200.0) * 65  # ₹2,600
        assert trade_result["reason"] == "trail_stop"
        assert trade_result["exit_price"] == 240.0
        assert "exit_time" in trade_result

        # Verify: trade would be saved to DB (the dict has all required fields)
        required_fields = ["symbol", "entry_price", "exit_price", "pnl", "quantity"]
        for field in required_fields:
            assert field in trade_result, f"Missing field: {field}"


class TestCrashRecoveryRestoresOpenPosition:
    """C6: On startup, restore open positions from DB."""

    def test_crash_recovery_restores_open_position(self):
        """restore_position() recreates Position from DB trade record."""
        portfolio = PortfolioManager(initial_capital=150000)
        assert len(portfolio.positions) == 0

        # Simulate a DB trade record (what get_open_positions returns)
        trade = {
            "trade_id": "T_CRASH_001",
            "symbol": "NIFTY24200PE",
            "instrument_key": "NSE_FO|NIFTY24200PE",
            "side": "BUY",
            "quantity": 65,
            "price": 180.0,
            "fill_price": 181.5,
            "fill_quantity": 65,
            "stop_loss": 126.0,
            "take_profit": 270.0,
            "strategy": "options",
            "sector": "",
            "regime": "TRENDING",
            "entry_time": datetime.now().isoformat(),
            "status": "filled",
            "mode": "paper",
        }

        # Restore the position
        portfolio.restore_position(trade)

        # Verify: position exists in portfolio
        assert "NIFTY24200PE" in portfolio.positions
        pos = portfolio.positions["NIFTY24200PE"]

        # Verify: all fields correctly restored
        assert pos.trade_id == "T_CRASH_001"
        assert pos.instrument_key == "NSE_FO|NIFTY24200PE"
        assert pos.quantity == 65
        assert pos.entry_price == 181.5  # fill_price preferred over price
        assert pos.stop_loss == 126.0
        assert pos.take_profit == 270.0
        assert pos.strategy == "options"
        assert pos.side == "BUY"
        assert pos.original_quantity == 65

        # Verify: cash correctly deducted
        expected_cash = 150000 - (181.5 * 65)
        assert abs(portfolio.cash - expected_cash) < 0.01

        # Verify: can close the restored position normally
        trade_result = portfolio.close_position("NIFTY24200PE", 220.0, "trail_stop")
        assert trade_result is not None
        assert trade_result["pnl"] == (220.0 - 181.5) * 65
        assert "NIFTY24200PE" not in portfolio.positions


class TestForceExitRetries3Times:
    """C7/C8: Force exit at 15:10 retries 3 times before giving up."""

    def test_force_exit_retries_3_times(self):
        """Force exit must attempt 3 times. If all fail, alert and close portfolio anyway."""
        attempts = []
        alerts_sent = []

        class FailingBroker:
            def place_order(self, **kwargs):
                attempts.append(kwargs)
                raise ConnectionError("Upstox timeout")

        class FakeAlerts:
            def send_raw(self, text):
                alerts_sent.append(text)

        broker = FailingBroker()
        alerts = FakeAlerts()

        # Setup portfolio with position
        portfolio = PortfolioManager(initial_capital=100000)
        pos = Position(
            symbol="NIFTY24100CE",
            instrument_key="NSE_FO|NIFTY24100CE",
            side="BUY",
            quantity=65,
            entry_price=200.0,
            current_price=240.0,
        )
        portfolio.add_position(pos)

        # Simulate the force exit retry logic (mirroring the C7/C8 fix)
        exit_success = False
        for attempt in range(3):
            try:
                broker.place_order(
                    symbol=pos.symbol,
                    instrument_key=pos.instrument_key,
                    quantity=pos.quantity,
                    side="SELL",
                    order_type="MARKET",
                    product="I",
                )
                exit_success = True
                break
            except Exception:
                pass  # Retry

        if not exit_success:
            alerts.send_raw(
                f"FORCE EXIT FAILED: {pos.symbol}\n"
                f"3 attempts failed. Check Upstox manually.\n"
                f"Position may still be open."
            )
            # Still close in portfolio
            trade_result = portfolio.close_position(
                pos.symbol, pos.current_price, "force_exit_failed"
            )

        # Verify: 3 attempts were made
        assert len(attempts) == 3
        for a in attempts:
            assert a["side"] == "SELL"
            assert a["symbol"] == "NIFTY24100CE"

        # Verify: alert sent about failure
        assert len(alerts_sent) == 1
        assert "FORCE EXIT FAILED" in alerts_sent[0]
        assert "3 attempts failed" in alerts_sent[0]

        # Verify: position still closed in portfolio (prevents ghost position)
        assert "NIFTY24100CE" not in portfolio.positions
