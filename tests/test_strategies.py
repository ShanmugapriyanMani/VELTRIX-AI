"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from src.strategies.base import Signal, SignalDirection
from src.strategies.fii_flow import FIIFlowStrategy
from src.strategies.options_oi import OptionsOIStrategy
from src.strategies.delivery_volume import DeliveryVolumeStrategy
from src.strategies.ml_predictor import MLPredictorStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.regime.detector import RegimeDetector, MarketRegime, RegimeState


class TestFIIFlowStrategy:
    def setup_method(self):
        self.strategy = FIIFlowStrategy()

    def test_no_signal_when_insufficient_consecutive_days(self):
        data = {
            "regime": "BULL_TRENDING",
            "fii_consecutive": {"direction": "buy", "consecutive_days": 1, "total_flow_cr": 500},
            "stock_universe": {},
        }
        signals = self.strategy.generate_signals(data)
        assert len(signals) == 0

    def test_buy_signal_on_consecutive_fii_buying(self):
        data = {
            "regime": "BULL_TRENDING",
            "fii_consecutive": {"direction": "buy", "consecutive_days": 4, "total_flow_cr": 5000},
            "stock_universe": {
                "RELIANCE": {
                    "price": 2500, "atr": 50, "sector": "OIL_GAS",
                    "fii_holding_pct": 25, "delivery_pct": 55, "rsi": 45,
                },
                "TCS": {
                    "price": 3800, "atr": 70, "sector": "IT",
                    "fii_holding_pct": 30, "delivery_pct": 40, "rsi": 50,
                },
            },
        }
        signals = self.strategy.generate_signals(data)
        assert len(signals) > 0
        assert all(s.direction == SignalDirection.BUY for s in signals)

    def test_sell_signal_on_fii_selling(self):
        data = {
            "regime": "BEAR_TRENDING",
            "fii_consecutive": {"direction": "sell", "consecutive_days": 4, "total_flow_cr": -5000},
            "stock_universe": {"RELIANCE": {"price": 2500}},
        }
        signals = self.strategy.generate_signals(data)
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.SELL

    def test_inactive_in_wrong_regime(self):
        data = {
            "regime": "HIGH_VOLATILITY",
            "fii_consecutive": {"direction": "buy", "consecutive_days": 5, "total_flow_cr": 8000},
            "stock_universe": {"RELIANCE": {"price": 2500}},
        }
        signals = self.strategy.generate_signals(data)
        assert len(signals) == 0


class TestOptionsOIStrategy:
    def setup_method(self):
        self.strategy = OptionsOIStrategy()

    def test_pcr_bullish_signal(self):
        data = {
            "regime": "BULL_TRENDING",
            "oi_levels": {
                "max_call_oi_strike": 22500,
                "max_call_oi": 1000000,
                "max_call_oi_change": -50000,
                "max_put_oi_strike": 22000,
                "max_put_oi": 1200000,
                "max_put_oi_change": 100000,
                "underlying": 22250,
            },
            "pcr": {"pcr_oi": 1.4, "pcr_volume": 1.1, "pcr_change_oi": 1.2},
            "max_pain": {"max_pain_strike": 22200, "distance_pct": 0.22},
            "is_expiry_day": False,
            "nifty_price": 22250,
            "stock_universe": {},
        }
        signals = self.strategy.generate_signals(data)
        # Should generate PCR bullish signal
        buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
        assert len(buy_signals) > 0

    def test_no_signal_in_wrong_regime(self):
        data = {
            "regime": "HIGH_VOLATILITY",
            "oi_levels": {},
            "pcr": {},
            "max_pain": {},
            "nifty_price": 0,
        }
        signals = self.strategy.generate_signals(data)
        assert len(signals) == 0


class TestDeliveryVolumeStrategy:
    def setup_method(self):
        self.strategy = DeliveryVolumeStrategy()

    def test_accumulation_signal(self):
        data = {
            "regime": "BULL_TRENDING",
            "delivery_divergences": {
                "accumulation": [
                    {
                        "symbol": "RELIANCE",
                        "close": 2500,
                        "change_pct": -2.5,
                        "delivery_pct": 72,
                        "traded_value_cr": 150,
                    }
                ],
                "distribution": [],
            },
            "stock_prices": {
                "RELIANCE": {"price": 2500, "atr": 50, "rsi": 35},
            },
        }
        # Provide delivery history for spike detection
        self.strategy._delivery_history["RELIANCE"] = pd.DataFrame({
            "delivery_pct": [40, 42, 38, 45, 41] * 4,
        })

        signals = self.strategy.generate_signals(data)
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].symbol == "RELIANCE"

    def test_distribution_signal(self):
        data = {
            "regime": "BULL_TRENDING",
            "delivery_divergences": {
                "accumulation": [],
                "distribution": [
                    {
                        "symbol": "INFY",
                        "close": 1800,
                        "change_pct": 3.0,
                        "delivery_pct": 22,
                        "traded_value_cr": 100,
                    }
                ],
            },
            "stock_prices": {
                "INFY": {"price": 1800, "atr": 30, "rsi": 75},
            },
        }
        self.strategy._delivery_history["INFY"] = pd.DataFrame({
            "delivery_pct": [45, 48, 42, 50, 44] * 4,
        })

        signals = self.strategy.generate_signals(data)
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.SELL


class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector()

    def _make_nifty_df(self, trend_up: bool = True, n: int = 100):
        """Create synthetic NIFTY OHLCV data."""
        base = 22000
        if trend_up:
            closes = [base + i * 10 + np.random.randn() * 20 for i in range(n)]
        else:
            closes = [base - i * 10 + np.random.randn() * 20 for i in range(n)]

        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": [c - 5 for c in closes],
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
            "volume": [10000000] * n,
        })

    def test_high_vix_regime(self):
        vix_data = {"vix": 35, "change_pct": 5}
        nifty_df = self._make_nifty_df()
        fii_data = pd.DataFrame({"fii_net_value": [100, 200, -50, 300, 150]})

        state = self.detector.detect(vix_data, nifty_df, fii_data)
        assert state.regime == MarketRegime.HIGH_VOLATILITY
        assert state.size_multiplier == 0.0

    def test_bull_trending(self):
        vix_data = {"vix": 14, "change_pct": -1}
        nifty_df = self._make_nifty_df(trend_up=True)
        fii_data = pd.DataFrame({"fii_net_value": [500, 800, 600, 700, 900]})

        state = self.detector.detect(vix_data, nifty_df, fii_data)
        assert "fii_flow" in state.active_strategies

    def test_active_strategies_for_sideways(self):
        active = self.detector.get_active_strategies(MarketRegime.SIDEWAYS_LOW_VOL)
        assert "options_oi" in active
        assert "delivery_volume" in active


class TestEnsemble:
    def test_register_strategies(self):
        ensemble = EnsembleStrategy()
        strategy = FIIFlowStrategy()
        ensemble.register_strategy(strategy)
        assert "fii_flow" in ensemble._strategies
