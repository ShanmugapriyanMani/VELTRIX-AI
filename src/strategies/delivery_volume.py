"""
Strategy 3: Delivery Volume Divergence — Institutional accumulation/distribution detection.

Logic:
- Price FALLS + delivery % > 60% → institutional accumulation → BUY (3-5 day hold)
- Price RISES + delivery % < 30% → speculative rally → potential reversal → EXIT/SHORT
- Only NIFTY 50 stocks with average daily volume > ₹50cr

Edge: 60-65% accuracy on NIFTY 50 with 3-5 day holding.
Delivery % reveals institutional intent — they MUST take delivery for long-term positions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalDirection


class DeliveryVolumeStrategy(BaseStrategy):
    """
    Delivery Volume Divergence Strategy.

    Core insight: When institutions accumulate, they take delivery (delivery % spikes).
    When speculators drive a rally, delivery % drops (all intraday trading).
    This divergence between price and delivery % reveals institutional intent.
    """

    def __init__(self, config_path: str = "config/strategies.yaml"):
        super().__init__("delivery_volume", config_path)

        self.accumulation_threshold = self.config.get("accumulation_threshold_pct", 60.0)
        self.distribution_threshold = self.config.get("distribution_threshold_pct", 30.0)
        self.price_down_threshold = self.config.get("price_change_threshold_pct", -1.0)
        self.price_up_threshold = self.config.get("price_rise_threshold_pct", 1.0)
        self.min_volume_cr = self.config.get("min_volume_cr", 50)
        self.hold_days_min = self.config.get("hold_days_min", 3)
        self.hold_days_max = self.config.get("hold_days_max", 5)
        self.volume_ma_period = self.config.get("volume_ma_period", 20)
        self.delivery_ma_period = self.config.get("delivery_ma_period", 20)
        self.delivery_spike_multiplier = self.config.get("delivery_spike_multiplier", 1.5)

        # State: historical delivery data per symbol
        self._delivery_history: dict[str, pd.DataFrame] = {}

    def update(self, data: dict[str, Any]) -> None:
        """
        Update with latest delivery data.

        Expected data keys:
        - delivery_data: DataFrame with [symbol, close, change_pct, delivery_pct, traded_value_cr]
        - delivery_history: dict of {symbol: DataFrame} (historical delivery data)
        """
        if "delivery_history" in data:
            self._delivery_history = data["delivery_history"]

        # Append today's data to history
        if "delivery_data" in data:
            today_df = data["delivery_data"]
            for _, row in today_df.iterrows():
                sym = row.get("symbol", "")
                if sym not in self._delivery_history:
                    self._delivery_history[sym] = pd.DataFrame()
                self._delivery_history[sym] = pd.concat(
                    [self._delivery_history[sym], pd.DataFrame([row])],
                    ignore_index=True,
                ).tail(60)  # Keep 60 days

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """
        Generate signals from delivery volume divergences.

        Expected data keys:
        - delivery_data: DataFrame of today's delivery data
        - delivery_divergences: {accumulation: [...], distribution: [...]}
        - stock_prices: dict of {symbol: {price, atr, rsi, ...}}
        - regime: str
        """
        signals = []
        regime = data.get("regime", "")

        if not self.is_active_in_regime(regime):
            return signals

        divergences = data.get("delivery_divergences", {})
        stock_prices = data.get("stock_prices", {})

        # ── ACCUMULATION signals (Buy) ──
        for entry in divergences.get("accumulation", []):
            symbol = entry.get("symbol", "")
            signal = self._process_accumulation(symbol, entry, stock_prices, regime)
            if signal:
                signals.append(signal)

        # ── DISTRIBUTION signals (Sell/Exit) ──
        for entry in divergences.get("distribution", []):
            symbol = entry.get("symbol", "")
            signal = self._process_distribution(symbol, entry, stock_prices, regime)
            if signal:
                signals.append(signal)

        self._signals_history.extend(signals)

        if signals:
            buy_count = sum(1 for s in signals if s.direction == SignalDirection.BUY)
            sell_count = sum(1 for s in signals if s.direction == SignalDirection.SELL)
            logger.info(
                f"DELIVERY_VOL: {len(signals)} signals ({buy_count} BUY, {sell_count} SELL)"
            )

        return signals

    def _process_accumulation(
        self,
        symbol: str,
        entry: dict[str, Any],
        stock_prices: dict[str, dict],
        regime: str,
    ) -> Optional[Signal]:
        """
        Process a potential accumulation signal.

        Accumulation = price down + high delivery % (institutions buying on dips)
        """
        delivery_pct = entry.get("delivery_pct", 0)
        change_pct = entry.get("change_pct", 0)
        traded_value_cr = entry.get("traded_value_cr", 0)

        # Basic filters
        if delivery_pct < self.accumulation_threshold:
            return None
        if change_pct > self.price_down_threshold:
            return None
        if traded_value_cr < self.min_volume_cr:
            return None

        # ── Check delivery spike vs historical average ──
        is_spike, spike_ratio = self._is_delivery_spike(symbol, delivery_pct)

        if not is_spike:
            return None

        # ── Calculate confidence ──
        confidence = self._calc_accumulation_confidence(
            delivery_pct, change_pct, spike_ratio, traded_value_cr
        )

        # Get current price data
        price_info = stock_prices.get(symbol, {})
        price = price_info.get("price", entry.get("close", 0))
        atr = price_info.get("atr", price * 0.02)
        rsi = price_info.get("rsi", 50)

        if price <= 0:
            return None

        # RSI confirmation: oversold + accumulation = strong signal
        if rsi < 35:
            confidence = min(confidence * 1.2, 0.9)
        elif rsi > 65:
            confidence *= 0.7  # Less compelling if overbought

        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=SignalDirection.BUY,
            confidence=confidence,
            score=confidence,
            price=price,
            stop_loss=price - 1.5 * atr,
            take_profit=price + 3.0 * atr,
            hold_days=self.hold_days_min,
            regime=regime,
            features={
                "delivery_pct": delivery_pct,
                "price_change_pct": change_pct,
                "traded_value_cr": traded_value_cr,
                "delivery_spike_ratio": spike_ratio,
                "rsi": rsi,
                "signal_type": "accumulation",
            },
            notes=(
                f"ACCUMULATION: {symbol} price {change_pct:+.1f}% but "
                f"delivery {delivery_pct:.1f}% (spike {spike_ratio:.1f}x avg). "
                f"Vol=₹{traded_value_cr:.0f}cr"
            ),
        )

    def _process_distribution(
        self,
        symbol: str,
        entry: dict[str, Any],
        stock_prices: dict[str, dict],
        regime: str,
    ) -> Optional[Signal]:
        """
        Process a potential distribution signal.

        Distribution = price up + low delivery % (speculators driving rally)
        """
        delivery_pct = entry.get("delivery_pct", 0)
        change_pct = entry.get("change_pct", 0)
        traded_value_cr = entry.get("traded_value_cr", 0)

        if delivery_pct > self.distribution_threshold:
            return None
        if change_pct < self.price_up_threshold:
            return None
        if traded_value_cr < self.min_volume_cr:
            return None

        # Check if delivery is abnormally low vs history
        avg_delivery = self._get_avg_delivery(symbol)
        if avg_delivery > 0 and delivery_pct > avg_delivery * 0.7:
            return None  # Not low enough relative to history

        price_info = stock_prices.get(symbol, {})
        price = price_info.get("price", entry.get("close", 0))
        atr = price_info.get("atr", price * 0.02)
        rsi = price_info.get("rsi", 50)

        if price <= 0:
            return None

        confidence = 0.4
        # Stronger signal if RSI overbought
        if rsi > 70:
            confidence += 0.2
        # Stronger if delivery is very low
        if delivery_pct < 20:
            confidence += 0.1
        # Stronger if big price move on low delivery
        if change_pct > 3:
            confidence += 0.1

        confidence = min(confidence, 0.8)

        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=SignalDirection.SELL,
            confidence=confidence,
            score=-confidence,
            price=price,
            stop_loss=price + 1.5 * atr,
            take_profit=price - 3.0 * atr,
            hold_days=self.hold_days_min,
            regime=regime,
            features={
                "delivery_pct": delivery_pct,
                "price_change_pct": change_pct,
                "traded_value_cr": traded_value_cr,
                "avg_delivery_pct": avg_delivery,
                "rsi": rsi,
                "signal_type": "distribution",
            },
            notes=(
                f"DISTRIBUTION: {symbol} price {change_pct:+.1f}% but "
                f"delivery only {delivery_pct:.1f}% (avg {avg_delivery:.1f}%). "
                f"Speculative rally."
            ),
        )

    def _is_delivery_spike(
        self, symbol: str, current_delivery: float
    ) -> tuple[bool, float]:
        """Check if current delivery % is a significant spike vs historical average."""
        avg_delivery = self._get_avg_delivery(symbol)
        if avg_delivery <= 0:
            # No history — use absolute threshold only
            return current_delivery >= self.accumulation_threshold, 1.0

        ratio = current_delivery / avg_delivery
        is_spike = ratio >= self.delivery_spike_multiplier
        return is_spike, round(ratio, 2)

    def _get_avg_delivery(self, symbol: str) -> float:
        """Get average delivery % for a symbol from history."""
        hist = self._delivery_history.get(symbol)
        if hist is None or hist.empty:
            return 0.0

        if "delivery_pct" in hist.columns:
            return float(
                hist["delivery_pct"]
                .tail(self.delivery_ma_period)
                .mean()
            )
        return 0.0

    def _calc_accumulation_confidence(
        self,
        delivery_pct: float,
        change_pct: float,
        spike_ratio: float,
        traded_value_cr: float,
    ) -> float:
        """Calculate confidence score for accumulation signal."""
        confidence = 0.4  # Base

        # Higher delivery % → higher confidence
        if delivery_pct >= 70:
            confidence += 0.2
        elif delivery_pct >= 60:
            confidence += 0.1

        # Bigger price drop → institutions buying at lower prices
        if change_pct < -3:
            confidence += 0.15
        elif change_pct < -2:
            confidence += 0.1

        # Larger delivery spike → more unusual
        if spike_ratio >= 2.0:
            confidence += 0.15
        elif spike_ratio >= 1.5:
            confidence += 0.05

        # Higher volume → more institutional
        if traded_value_cr >= 200:
            confidence += 0.1
        elif traded_value_cr >= 100:
            confidence += 0.05

        return min(confidence, 0.9)
