"""Tests for passive instrument logger."""

import numpy as np
import pandas as pd

from src.data.features import FeatureEngine
from src.instruments.instrument_logger import (
    InstrumentConfig,
    InstrumentLogger,
    TRACKED_INSTRUMENTS,
)


def _make_ohlcv(n: int = 60, base_close: float = 50000.0) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame with technical features."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-03-12", periods=n)
    closes = [base_close + i * 50 + np.random.randn() * 100 for i in range(n)]
    df = pd.DataFrame({
        "datetime": dates,
        "open": [c - 20 for c in closes],
        "high": [c + 50 for c in closes],
        "low": [c - 50 for c in closes],
        "close": closes,
        "volume": [1000000 + np.random.randint(-200000, 200000) for _ in range(n)],
    })
    fe = FeatureEngine()
    return fe.add_technical_features(df)


def _make_inst(name: str = "TEST", adx_threshold: float = 22, vix_mult: float = 1.0) -> InstrumentConfig:
    return InstrumentConfig(
        name=name, instrument_type="index", exchange="NSE",
        upstox_symbol="NSE_INDEX|Test", lot_size=50, tick_size=0.05,
        options_expiry="weekly", vix_multiplier=vix_mult, adx_threshold=adx_threshold,
    )


class TestInstrumentConfig:
    def test_tracked_instruments_count(self):
        assert len(TRACKED_INSTRUMENTS) == 5

    def test_instrument_config_fields(self):
        bn = next(i for i in TRACKED_INSTRUMENTS if i.name == "BANKNIFTY")
        assert bn.instrument_type == "index"
        assert bn.exchange == "NSE"
        assert bn.lot_size == 30
        assert bn.vix_multiplier == 1.15
        assert bn.adx_threshold == 22

    def test_sensex_is_bse(self):
        sensex = next(i for i in TRACKED_INSTRUMENTS if i.name == "SENSEX")
        assert sensex.exchange == "BSE"
        assert sensex.upstox_symbol == "BSE_INDEX|SENSEX"


class TestRegimeComputation:
    def test_compute_regime_volatile(self):
        inst = _make_inst(vix_mult=1.0)
        df = _make_ohlcv(60)
        # VIX 25 * 1.0 = 25 > 22 → VOLATILE
        assert InstrumentLogger._compute_regime(inst, df, 25) == "VOLATILE"

    def test_compute_regime_trending(self):
        inst = _make_inst(adx_threshold=22)
        df = _make_ohlcv(60)
        df.loc[df.index[-1], "adx_14"] = 30
        # VIX 12 * 1.0 = 12 < 22, ADX 30 > 22 → TRENDING
        assert InstrumentLogger._compute_regime(inst, df, 12) == "TRENDING"

    def test_compute_regime_rangebound(self):
        inst = _make_inst(adx_threshold=22)
        df = _make_ohlcv(60)
        df.loc[df.index[-1], "adx_14"] = 15
        # VIX 12, ADX 15 < 22 → RANGEBOUND
        assert InstrumentLogger._compute_regime(inst, df, 12) == "RANGEBOUND"


class TestRegimeWeights:
    def test_trending_weights(self):
        ema_w, mr_w = InstrumentLogger._regime_weights("TRENDING")
        assert ema_w == 2.5
        assert mr_w == 1.5

    def test_rangebound_weights(self):
        ema_w, mr_w = InstrumentLogger._regime_weights("RANGEBOUND")
        assert ema_w == 1.0
        assert mr_w == 2.5


class TestScoring:
    def setup_method(self):
        self.inst = _make_inst()
        # Create logger without full dependencies — only need _prev_vix for scoring
        self.logger = InstrumentLogger.__new__(InstrumentLogger)
        self.logger._prev_vix = {}

    def test_score_returns_tuple(self):
        df = _make_ohlcv(60, base_close=50000)
        bull, bear, direction = self.logger._score_instrument(
            self.inst, df, 15, "TRENDING", 2.5, 1.5, 1.0, {}, None,
        )
        assert isinstance(bull, float)
        assert isinstance(bear, float)
        assert direction in ("CE", "PE", "")

    def test_score_uptrend_produces_ce(self):
        df = _make_ohlcv(60, base_close=50000)
        # Force strong uptrend indicators
        df.loc[df.index[-1], "ema_9"] = 53100
        df.loc[df.index[-1], "ema_21"] = 53000
        df.loc[df.index[-1], "ema_50"] = 52500
        df.loc[df.index[-1], "close"] = 53200
        df.loc[df.index[-1], "open"] = 52800
        df.loc[df.index[-1], "rsi_14"] = 65
        df.loc[df.index[-2], "rsi_14"] = 60
        df.loc[df.index[-1], "macd_histogram"] = 100
        df.loc[df.index[-2], "macd_histogram"] = 50

        bull, bear, direction = self.logger._score_instrument(
            self.inst, df, 12, "TRENDING", 2.5, 1.5, 1.2, {}, None,
        )
        assert direction == "CE"
        assert bull > bear

    def test_score_downtrend_produces_pe(self):
        df = _make_ohlcv(60, base_close=50000)
        # Force strong downtrend indicators
        df.loc[df.index[-1], "ema_9"] = 47000
        df.loc[df.index[-1], "ema_21"] = 47500
        df.loc[df.index[-1], "ema_50"] = 48000
        df.loc[df.index[-1], "close"] = 46800
        df.loc[df.index[-1], "open"] = 47500
        df.loc[df.index[-1], "rsi_14"] = 35
        df.loc[df.index[-2], "rsi_14"] = 40
        df.loc[df.index[-1], "macd_histogram"] = -100
        df.loc[df.index[-2], "macd_histogram"] = -50

        bull, bear, direction = self.logger._score_instrument(
            self.inst, df, 25, "VOLATILE", 0.5, 1.0, 0.7, {}, None,
        )
        assert direction == "PE"
        assert bear > bull
