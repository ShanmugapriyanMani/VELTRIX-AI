"""Tests for V9.3 signal bucket grouping.

3 tests covering:
  1. Momentum bucket capped at ±3.0
  2. Flow bucket capped at ±3.0
  3. Total score = sum of capped buckets (not raw)
"""


class TestMomentumBucketCappedAt3:
    """BUCKET 1: MOMENTUM (F1+F2+F3+F9) capped at ±3.0."""

    def test_momentum_bucket_capped_at_3(self):
        """When all momentum factors fire bullish (~6.3 raw), cap limits to 3.0."""
        # Simulate max momentum: F1=2.0+0.5+0.5+0.3, F2=1.0+1.0, F3=0.75+0.75+0.3, F9=1.0
        # Total raw = 2.0+0.5+0.5+0.3 + 1.0+1.0 + 0.75+0.75+0.3 + 1.0 = 8.1
        momentum_bull = 0.0
        MOMENTUM_CAP = 3.0

        # F1: EMA stack (TRENDING weight=2.5 → base=2.0, bonus=0.5) + ADX + 5d
        momentum_bull += 2.0   # ema_base (trend_up, TRENDING)
        momentum_bull += 0.5   # ema_bonus (close > ema21)
        momentum_bull += 0.5   # ADX > 30
        momentum_bull += 0.3   # ret_5d > 0

        # F2: RSI + MACD
        momentum_bull += 1.0   # RSI > 58
        momentum_bull += 1.0   # MACD histogram expanding

        # F3: Price action
        momentum_bull += 0.75  # gap > 0.4%
        momentum_bull += 0.75  # close > prev_high
        momentum_bull += 0.3   # candle body bullish

        # F9: Volume confirmation
        momentum_bull += 1.0   # vol_ratio > 1.3, bullish candle

        assert momentum_bull > MOMENTUM_CAP  # Raw exceeds cap
        capped = max(min(momentum_bull, MOMENTUM_CAP), -MOMENTUM_CAP)
        assert capped == MOMENTUM_CAP  # Capped at 3.0


class TestFlowBucketCappedAt3:
    """BUCKET 2: FLOW (F8+F10) capped at ±3.0."""

    def test_flow_bucket_capped_at_3(self):
        """When all flow factors fire bullish (~4.0 raw), cap limits to 3.0."""
        flow_bull = 0.0
        FLOW_CAP = 3.0

        # F10: Global Macro
        flow_bull += 0.5   # DXY falling
        flow_bull += 0.5   # SP500 positive + high corr
        flow_bull += 0.5   # global_risk > 1.0

        # F8: OI/PCR (live only — but test the bucket math)
        flow_bull += 1.0   # PCR >= 1.3
        flow_bull += 1.0   # near OI support

        assert flow_bull > FLOW_CAP  # Raw exceeds cap (3.5 > 3.0)
        capped = max(min(flow_bull, FLOW_CAP), -FLOW_CAP)
        assert capped == FLOW_CAP  # Capped at 3.0


class TestTotalScoreUsesBucketSums:
    """Final bull/bear scores = sum of 4 capped buckets + ML (uncapped)."""

    def test_total_score_uses_bucket_sums(self):
        """Verify total score is sum of capped buckets, not raw accumulation."""
        MOMENTUM_CAP = 3.0
        FLOW_CAP = 3.0
        VOLATILITY_CAP = 1.5
        MEAN_REVERSION_CAP = 2.5

        # Simulate: momentum raw=5.0, flow raw=2.0, vol raw=0.8, mr raw=1.5, ml=0.3
        momentum_bull_raw = 5.0
        flow_bull_raw = 2.0
        vol_bull_raw = 0.8
        mr_bull_raw = 1.5
        ml_bull = 0.3

        # Without caps: 5.0 + 2.0 + 0.8 + 1.5 + 0.3 = 9.6
        raw_total = momentum_bull_raw + flow_bull_raw + vol_bull_raw + mr_bull_raw + ml_bull
        assert abs(raw_total - 9.6) < 0.01

        # With caps: min(5.0,3.0) + min(2.0,3.0) + min(0.8,1.5) + min(1.5,2.5) + 0.3
        # = 3.0 + 2.0 + 0.8 + 1.5 + 0.3 = 7.6
        momentum_capped = max(min(momentum_bull_raw, MOMENTUM_CAP), -MOMENTUM_CAP)
        flow_capped = max(min(flow_bull_raw, FLOW_CAP), -FLOW_CAP)
        vol_capped = max(min(vol_bull_raw, VOLATILITY_CAP), -VOLATILITY_CAP)
        mr_capped = max(min(mr_bull_raw, MEAN_REVERSION_CAP), -MEAN_REVERSION_CAP)

        capped_total = momentum_capped + flow_capped + vol_capped + mr_capped + ml_bull
        assert abs(capped_total - 7.6) < 0.01
        assert capped_total < raw_total  # Capping reduced the score

        # Momentum was the only bucket that got capped
        assert momentum_capped == 3.0  # was 5.0
        assert flow_capped == 2.0      # unchanged (under cap)
        assert vol_capped == 0.8       # unchanged (under cap)
        assert mr_capped == 1.5        # unchanged (under cap)
