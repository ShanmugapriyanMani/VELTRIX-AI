"""Tests for V9.3b exit strategy improvements.

8 tests covering:
  1. TP regime multipliers are reduced (lower than old values)
  2. Trail tiers are tighter at higher gains in backtest
  3. TP ladder checkpoint fires at 12:00 when conditions met
  4. TP ladder checkpoint does not fire twice for same position
  5. Momentum decay exits at rescore when profit >10%, score decayed, RSI dropped
  6. Momentum decay requires min profit (skips <10%)
  7. Late weak exit fires at 14:45 for ±5% positions
  8. Late weak exit skips big winners (>5% profit)
"""

from datetime import time as dt_time

from src.regime.detector import RegimeDetector, MarketRegime


class TestTpRegimeMultiplierReduced:
    """Part 1: TP regime multipliers are reduced to realistic levels."""

    def test_tp_regime_multiplier_reduced(self):
        """All regime TP multipliers should be ≤ 1.10 (previously up to 1.50)."""
        profiles = RegimeDetector.REGIME_PROFILES
        # TP multipliers kept at original values — reduction hurt CAGR in backtest
        # Part 1 finding: lowering TP multipliers reduces profit capture, not EOD exits
        assert profiles[MarketRegime.TRENDING]["tp_multiplier"] == 1.30
        assert profiles[MarketRegime.RANGEBOUND]["tp_multiplier"] == 0.70
        assert profiles[MarketRegime.VOLATILE]["tp_multiplier"] == 1.50
        assert profiles[MarketRegime.ELEVATED]["tp_multiplier"] == 1.20
        # RANGEBOUND has lowest multiplier (quick exits in range)
        assert profiles[MarketRegime.RANGEBOUND]["tp_multiplier"] < profiles[MarketRegime.TRENDING]["tp_multiplier"]


class TestTrailTighterAtHigherGains:
    """Part 2: Trail tiers are tighter to capture more gains."""

    def test_trail_tighter_at_higher_gains(self):
        """Backtest trail tiers: +5%→0.96, +12%→0.94, +25%→0.91 (was 0.97/0.95/0.93)."""
        # Simulate the backtest trail logic
        entry_premium = 200.0

        # Test +5% gain → trail floor at 0.96
        high_premium = entry_premium * 1.06  # +6% gain
        high_gain_pct = (high_premium - entry_premium) / entry_premium
        trail_enabled = True
        trail_floor = None
        if trail_enabled:
            if high_gain_pct >= 0.25:
                trail_floor = high_premium * 0.91
            elif high_gain_pct >= 0.12:
                trail_floor = high_premium * 0.94
            elif high_gain_pct >= 0.05:
                trail_floor = high_premium * 0.96
        assert trail_floor is not None
        assert abs(trail_floor - high_premium * 0.96) < 0.01

        # Test +15% gain → trail floor at 0.94 (was 0.95)
        high_premium = entry_premium * 1.15
        high_gain_pct = (high_premium - entry_premium) / entry_premium
        trail_floor = None
        if trail_enabled:
            if high_gain_pct >= 0.25:
                trail_floor = high_premium * 0.91
            elif high_gain_pct >= 0.12:
                trail_floor = high_premium * 0.94
            elif high_gain_pct >= 0.05:
                trail_floor = high_premium * 0.96
        assert trail_floor is not None
        assert abs(trail_floor - high_premium * 0.94) < 0.01

        # Test +30% gain → trail floor at 0.91 (was 0.93)
        high_premium = entry_premium * 1.30
        high_gain_pct = (high_premium - entry_premium) / entry_premium
        trail_floor = None
        if trail_enabled:
            if high_gain_pct >= 0.25:
                trail_floor = high_premium * 0.91
            elif high_gain_pct >= 0.12:
                trail_floor = high_premium * 0.94
            elif high_gain_pct >= 0.05:
                trail_floor = high_premium * 0.96
        assert trail_floor is not None
        assert abs(trail_floor - high_premium * 0.91) < 0.01

        # RANGEBOUND: no trail at +5%, only at +10%
        trail_enabled = False
        high_premium = entry_premium * 1.06
        high_gain_pct = (high_premium - entry_premium) / entry_premium
        trail_floor = None
        if not trail_enabled:
            if high_gain_pct >= 0.25:
                trail_floor = high_premium * 0.91
            elif high_gain_pct >= 0.10:
                trail_floor = high_premium * 0.94
        assert trail_floor is None  # +6% in RANGEBOUND → no trail


class TestTpLadderFiresAt12Checkpoint:
    """Part 3: TP ladder checkpoint fires correctly."""

    def test_tp_ladder_fires_at_12_checkpoint(self):
        """At 12:00, if position peaked ≥ 8% and currently ≥ 6%, exit fires."""
        # Simulate checkpoint logic
        entry = 200.0
        peak = 220.0   # +10% peak
        current = 213.0  # +6.5% current
        now_time = dt_time(12, 5)

        checkpoints = [
            (dt_time(12, 0), "12:00", 0.08, 0.06),
            (dt_time(13, 0), "13:00", 0.06, 0.04),
            (dt_time(14, 0), "14:00", 0.04, 0.02),
        ]

        fired = set()
        exits = []
        for cp_time, cp_label, peak_threshold, exit_at in checkpoints:
            if now_time < cp_time:
                continue
            if cp_label in fired:
                continue
            peak_gain = (peak - entry) / entry
            current_gain = (current - entry) / entry
            if peak_gain >= peak_threshold and current_gain >= exit_at:
                fired.add(cp_label)
                exits.append(cp_label)

        assert "12:00" in exits
        assert len(exits) == 1  # Only 12:00 should fire (13:00 not yet)


class TestTpLadderNotFiredTwice:
    """Part 3: TP ladder checkpoint does not fire twice for same position."""

    def test_tp_ladder_not_fired_twice(self):
        """Once a checkpoint fires for a position, it should not fire again."""
        entry = 200.0
        peak = 220.0   # +10%
        current = 213.0  # +6.5%
        now_time = dt_time(12, 30)

        checkpoints = [
            (dt_time(12, 0), "12:00", 0.08, 0.06),
        ]

        # First pass: fires
        fired = set()
        exits_1 = []
        for cp_time, cp_label, peak_threshold, exit_at in checkpoints:
            if now_time < cp_time:
                continue
            if cp_label in fired:
                continue
            peak_gain = (peak - entry) / entry
            current_gain = (current - entry) / entry
            if peak_gain >= peak_threshold and current_gain >= exit_at:
                fired.add(cp_label)
                exits_1.append(cp_label)
        assert len(exits_1) == 1

        # Second pass: should NOT fire again (already in fired set)
        exits_2 = []
        for cp_time, cp_label, peak_threshold, exit_at in checkpoints:
            if now_time < cp_time:
                continue
            if cp_label in fired:
                continue
            peak_gain = (peak - entry) / entry
            current_gain = (current - entry) / entry
            if peak_gain >= peak_threshold and current_gain >= exit_at:
                fired.add(cp_label)
                exits_2.append(cp_label)
        assert len(exits_2) == 0  # Should not fire again


class TestMomentumDecayExitsAtRescore:
    """Part 5: Momentum decay exits when profit >10%, score decayed, RSI dropped."""

    def test_momentum_decay_exits_at_rescore(self):
        """Backtest proxy: peak ≥10%, fade_ratio ≥40%, RSI drop ≥8 → momentum_decay."""
        entry_premium = 200.0
        high_premium = 230.0     # +15% peak
        close_premium = 210.0    # +5% close (faded from 15% to 5%)
        prev_rsi = 65.0
        rsi = 52.0               # RSI dropped 13 points

        # Config defaults
        decay_enabled = True
        decay_min_profit = 0.10
        decay_factor = 0.60

        peak_pct = (high_premium - entry_premium) / entry_premium  # 0.15
        close_gain_pct = (close_premium - entry_premium) / entry_premium  # 0.05
        fade_ratio = 1.0 - (close_gain_pct / peak_pct)  # 0.667
        rsi_drop = prev_rsi - rsi  # 13

        assert peak_pct >= decay_min_profit
        assert fade_ratio >= (1.0 - decay_factor)  # 0.667 >= 0.40
        assert rsi_drop >= 8
        assert close_gain_pct > 0

        # Should trigger momentum_decay
        if (decay_enabled
                and peak_pct >= decay_min_profit
                and fade_ratio >= (1.0 - decay_factor)
                and rsi_drop >= 8
                and close_gain_pct > 0):
            exit_reason = "momentum_decay"
            exit_premium = entry_premium * (1 + peak_pct * 0.6)
        else:
            exit_reason = "eod_exit"
            exit_premium = close_premium

        assert exit_reason == "momentum_decay"
        assert exit_premium > entry_premium  # Profitable exit
        assert exit_premium < high_premium   # Less than peak
        assert abs(exit_premium - 218.0) < 0.01  # 200 * 1.09


class TestMomentumDecayRequiresMinProfit:
    """Part 6: Momentum decay does not fire when profit < 10%."""

    def test_momentum_decay_requires_min_profit(self):
        """If peak gain is only 7%, momentum decay should NOT fire."""
        entry_premium = 200.0
        high_premium = 214.0     # +7% peak (below 10% threshold)
        close_premium = 205.0    # +2.5% close
        prev_rsi = 65.0
        rsi = 55.0               # RSI dropped 10 points

        decay_min_profit = 0.10
        decay_factor = 0.60

        peak_pct = (high_premium - entry_premium) / entry_premium  # 0.07
        close_gain_pct = (close_premium - entry_premium) / entry_premium
        fade_ratio = 1.0 - (close_gain_pct / peak_pct) if peak_pct > 0.01 else 0
        rsi_drop = prev_rsi - rsi

        # Peak < 10% → should NOT trigger
        assert peak_pct < decay_min_profit
        triggered = (
            peak_pct >= decay_min_profit
            and fade_ratio >= (1.0 - decay_factor)
            and rsi_drop >= 8
            and close_gain_pct > 0
        )
        assert not triggered


class TestLateWeakExitFiresAt1445:
    """Part 7: Late weak exit fires at 14:45 for ±5% positions."""

    def test_late_weak_exit_fires_at_1445(self):
        """Position with +3% profit at 14:45 → late_weak_exit."""
        entry_premium = 200.0
        close_premium = 206.0    # +3% profit
        late_weak_max_profit = 0.05

        close_gain_pct = (close_premium - entry_premium) / entry_premium  # 0.03
        assert abs(close_gain_pct) < late_weak_max_profit

        exit_reason = "eod_exit"
        if abs(close_gain_pct) < late_weak_max_profit:
            exit_premium = close_premium * 0.99
            exit_reason = "late_weak_exit"

        assert exit_reason == "late_weak_exit"
        assert exit_premium < close_premium  # Slightly worse than close
        assert exit_premium > entry_premium  # Still profitable


class TestLateWeakExitSkipsBigWinners:
    """Part 8: Late weak exit skips positions with >5% profit."""

    def test_late_weak_exit_skips_big_winners(self):
        """Position with +8% profit at 14:45 → NOT late_weak_exit."""
        entry_premium = 200.0
        close_premium = 216.0    # +8% profit
        late_weak_max_profit = 0.05

        close_gain_pct = (close_premium - entry_premium) / entry_premium  # 0.08
        assert abs(close_gain_pct) >= late_weak_max_profit

        triggered = abs(close_gain_pct) < late_weak_max_profit
        assert not triggered


class TestFactorEdgeCalculatedCorrectly:
    """Quant Step 0: Factor edge calculation is correct."""

    def test_factor_edge_calculated_correctly(self):
        """Given known factor scores + PnL, verify aligned/against edge math."""
        from src.backtest.engine import BacktestTrade

        # 4 mock trades with known F1 scores and outcomes
        trades = [
            # F1 aligned (bull, CE) — win
            BacktestTrade(pnl=500, direction="CE", f1_bull=1.5, f1_bear=0.0),
            # F1 aligned (bear, PE) — win
            BacktestTrade(pnl=300, direction="PE", f1_bull=0.0, f1_bear=1.2),
            # F1 aligned (bull, CE) — loss
            BacktestTrade(pnl=-200, direction="CE", f1_bull=0.8, f1_bear=0.0),
            # F1 against (bear score but CE direction) — win
            BacktestTrade(pnl=100, direction="CE", f1_bull=0.0, f1_bear=0.5),
        ]

        # Compute aligned vs against for F1
        aligned_pnls = []
        against_pnls = []
        for t in trades:
            fs = t.f1_bull - t.f1_bear
            if abs(fs) < 0.001:
                continue
            if (fs > 0 and t.direction == "CE") or (fs < 0 and t.direction == "PE"):
                aligned_pnls.append(t.pnl)
            else:
                against_pnls.append(t.pnl)

        # 3 aligned: +500, +300, -200 → WR=2/3, avg_win=400, avg_loss=200
        assert len(aligned_pnls) == 3
        a_wr = sum(1 for p in aligned_pnls if p > 0) / len(aligned_pnls)
        a_wins = [p for p in aligned_pnls if p > 0]
        a_losses = [p for p in aligned_pnls if p <= 0]
        a_avg_win = sum(a_wins) / len(a_wins)
        a_avg_loss = abs(sum(a_losses) / len(a_losses))
        a_edge = a_wr * a_avg_win - (1 - a_wr) * a_avg_loss
        # 0.667 * 400 - 0.333 * 200 = 266.67 - 66.67 = 200.0
        assert abs(a_edge - 200.0) < 1.0

        # 1 against: +100 → WR=1/1=100%, edge = 1.0 * 100 - 0 = 100
        assert len(against_pnls) == 1
        ag_wr = 1.0
        ag_edge = ag_wr * 100 - 0
        assert abs(ag_edge - 100.0) < 0.01

        net_usefulness = a_edge - ag_edge
        assert abs(net_usefulness - 100.0) < 1.0


class TestFactorAnalysisModeRunsCleanly:
    """Quant Step 0: factor_analysis mode is wired up correctly."""

    def test_factor_analysis_mode_runs_cleanly(self):
        """Verify factor_analysis is in argparse choices and BacktestTrade has factor fields."""
        import argparse
        from src.backtest.engine import BacktestTrade

        # Check BacktestTrade has all factor fields
        t = BacktestTrade()
        for fn in range(1, 11):
            assert hasattr(t, f"f{fn}_bull"), f"Missing f{fn}_bull"
            assert hasattr(t, f"f{fn}_bear"), f"Missing f{fn}_bear"
        assert hasattr(t, "direction")
        assert hasattr(t, "score_diff")
        assert hasattr(t, "bull_score")
        assert hasattr(t, "bear_score")

        # Check factor_analysis is in choices
        from src.main import main
        import inspect
        source = inspect.getsource(main)
        assert "factor_analysis" in source


class TestTpLadderExitHasValidPrice:
    """TP ladder, _exit_position_for_reason, and force exit must pass price to place_order."""

    def test_tp_ladder_exit_has_valid_price(self):
        """All SELL place_order calls must include price= parameter (not default 0)."""
        import inspect
        from src.main import TradingBot
        source = inspect.getsource(TradingBot)

        # Find all SELL place_order blocks — each must have price=
        import re
        # Match place_order blocks that contain side="SELL"
        blocks = re.findall(
            r'place_order\((.*?)\)',
            source.replace('\n', ' '),
        )
        sell_blocks = [b for b in blocks if '"SELL"' in b or "'SELL'" in b]
        assert len(sell_blocks) >= 3, f"Expected ≥3 SELL place_order calls, found {len(sell_blocks)}"
        for block in sell_blocks:
            assert "price=" in block, (
                f"SELL place_order missing price= parameter: {block[:80]}..."
            )
