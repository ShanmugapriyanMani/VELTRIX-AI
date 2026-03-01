"""
Strategy 2: Options OI Intelligence — Institutional level/breakout detection.

Logic:
- Max Call OI strike = resistance, Max Put OI strike = support
- If price approaches max Call OI with declining OI → breakout imminent → BUY
- If price stuck between max Call and Put OI → range-bound → mean reversion
- PCR > 1.3 → Bullish | PCR < 0.7 → Bearish
- Expiry day (Thursday): OI unwinding for pinning effect

Edge: Institutional options writers defend these levels — structural, not speculative.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalDirection


class OptionsOIStrategy(BaseStrategy):
    """
    Options OI Intelligence Strategy.

    Uses options chain data to identify institutional support/resistance levels
    and detect breakout conditions based on OI changes.
    """

    def __init__(self, config_path: str = "config/strategies.yaml"):
        super().__init__("options_oi", config_path)

        self.pcr_bullish = self.config.get("pcr_bullish", 1.3)
        self.pcr_bearish = self.config.get("pcr_bearish", 0.7)
        self.oi_change_threshold = self.config.get("oi_change_threshold_pct", 10)
        self.breakout_oi_decline = self.config.get("breakout_oi_decline_pct", 15)
        self.max_pain_distance_pct = self.config.get("max_pain_distance_pct", 2.0)
        self.expiry_day_special = self.config.get("expiry_day_special", True)
        self.expiry_oi_unwind = self.config.get("expiry_oi_unwind_threshold", 20)
        self.strike_range = self.config.get("strike_range", 10)

        # State
        self._prev_oi_levels: Optional[dict] = None
        self._prev_pcr: Optional[dict] = None

    def update(self, data: dict[str, Any]) -> None:
        """Update with latest OI data for change tracking."""
        if "oi_levels" in data and self._prev_oi_levels is None:
            self._prev_oi_levels = data["oi_levels"]
        if "pcr" in data and self._prev_pcr is None:
            self._prev_pcr = data["pcr"]

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """
        Generate signals from options OI analysis.

        Expected data keys:
        - oi_levels: {max_call_oi_strike, max_put_oi_strike, underlying, ...}
        - pcr: {pcr_oi, pcr_volume, ...}
        - max_pain: {max_pain_strike, distance_pct, ...}
        - option_chain: DataFrame with full chain
        - is_expiry_day: bool
        - is_expiry_week: bool
        - nifty_price: float
        - regime: str
        - stock_universe: dict (for individual stock signals)
        """
        signals = []
        regime = data.get("regime", "")

        if not self.is_active_in_regime(regime):
            return signals

        oi_levels = data.get("oi_levels", {})
        pcr = data.get("pcr", {})
        max_pain = data.get("max_pain", {})
        is_expiry_day = data.get("is_expiry_day", False)
        nifty_price = data.get("nifty_price", oi_levels.get("underlying", 0))

        if not oi_levels or nifty_price <= 0:
            return signals

        resistance = oi_levels.get("max_call_oi_strike", 0)
        support = oi_levels.get("max_put_oi_strike", 0)
        call_oi_change = oi_levels.get("max_call_oi_change", 0)
        max_call_oi = oi_levels.get("max_call_oi", 1)
        max_put_oi = oi_levels.get("max_put_oi", 1)
        pcr_oi = pcr.get("pcr_oi", 1.0)

        if resistance <= 0 or support <= 0:
            return signals

        # Calculate key distances
        dist_to_resistance = (resistance - nifty_price) / nifty_price * 100
        dist_to_support = (nifty_price - support) / nifty_price * 100
        range_width = (resistance - support) / nifty_price * 100

        # ── Analysis 1: Breakout Detection ──
        breakout_signal = self._check_breakout(
            nifty_price, resistance, support,
            call_oi_change, max_call_oi, max_put_oi,
            dist_to_resistance, dist_to_support, pcr_oi,
        )

        if breakout_signal:
            signals.append(self._create_index_signal(
                breakout_signal, nifty_price, regime, oi_levels, pcr
            ))

        # ── Analysis 2: Range-bound / Mean Reversion ──
        range_signal = self._check_range_bound(
            nifty_price, resistance, support,
            dist_to_resistance, dist_to_support, range_width,
            pcr_oi,
        )

        if range_signal and not breakout_signal:
            signals.append(self._create_index_signal(
                range_signal, nifty_price, regime, oi_levels, pcr
            ))

        # ── Analysis 3: PCR Sentiment ──
        pcr_signal = self._analyze_pcr(pcr_oi, nifty_price)
        if pcr_signal and not breakout_signal:
            signals.append(self._create_index_signal(
                pcr_signal, nifty_price, regime, oi_levels, pcr
            ))

        # ── Analysis 4: Expiry Day Special ──
        if is_expiry_day and self.expiry_day_special:
            expiry_signal = self._expiry_day_analysis(
                nifty_price, max_pain, oi_levels, data.get("option_chain")
            )
            if expiry_signal:
                signals.append(self._create_index_signal(
                    expiry_signal, nifty_price, regime, oi_levels, pcr
                ))

        # ── Generate stock-level signals based on index view ──
        if signals:
            stock_signals = self._propagate_to_stocks(
                signals[0], data.get("stock_universe", {}), regime
            )
            signals.extend(stock_signals)

        # Update state for next iteration
        self._prev_oi_levels = oi_levels
        self._prev_pcr = pcr
        self._signals_history.extend(signals)

        if signals:
            logger.info(
                f"OPTIONS_OI: {len(signals)} signals | NIFTY={nifty_price:.0f}, "
                f"Support={support:.0f}, Resistance={resistance:.0f}, PCR={pcr_oi:.2f}"
            )

        return signals

    def _check_breakout(
        self,
        price: float,
        resistance: float,
        support: float,
        call_oi_change: int,
        max_call_oi: int,
        max_put_oi: int,
        dist_to_resistance: float,
        dist_to_support: float,
        pcr_oi: float,
    ) -> Optional[dict[str, Any]]:
        """
        Detect breakout conditions.

        Bullish breakout: Price near resistance + Call OI declining
        (writers closing positions = they expect price to go through)
        """
        # ── Bullish breakout: near resistance + OI unwinding ──
        if dist_to_resistance < 1.0 and dist_to_resistance > 0:
            oi_decline_pct = 0
            if max_call_oi > 0:
                oi_decline_pct = abs(min(call_oi_change, 0)) / max_call_oi * 100

            if oi_decline_pct >= self.breakout_oi_decline:
                confidence = min(0.5 + oi_decline_pct / 100, 0.85)
                if pcr_oi > 1.0:  # PCR confirming bullish
                    confidence += 0.1
                return {
                    "direction": SignalDirection.BUY,
                    "confidence": confidence,
                    "type": "breakout_up",
                    "notes": (
                        f"Bullish breakout: Price within {dist_to_resistance:.1f}% of "
                        f"resistance {resistance:.0f}, Call OI declining {oi_decline_pct:.1f}%"
                    ),
                }

        # ── Bearish breakdown: near support + Put OI declining ──
        if dist_to_support < 1.0 and dist_to_support > 0:
            # Check if put OI at support is declining
            if self._prev_oi_levels:
                prev_put_oi = self._prev_oi_levels.get("max_put_oi", 0)
                curr_put_oi = max_put_oi
                if prev_put_oi > 0:
                    put_decline = (prev_put_oi - curr_put_oi) / prev_put_oi * 100
                    if put_decline > self.breakout_oi_decline:
                        confidence = min(0.5 + put_decline / 100, 0.8)
                        return {
                            "direction": SignalDirection.SELL,
                            "confidence": confidence,
                            "type": "breakdown",
                            "notes": f"Bearish breakdown below support {support:.0f}",
                        }

        return None

    def _check_range_bound(
        self,
        price: float,
        resistance: float,
        support: float,
        dist_to_resistance: float,
        dist_to_support: float,
        range_width: float,
        pcr_oi: float,
    ) -> Optional[dict[str, Any]]:
        """
        Mean reversion signal when price is range-bound between OI levels.
        """
        # Only if price is well within the range (not near edges)
        if dist_to_resistance > 1.5 and dist_to_support > 1.5:
            return None

        # Near support in range → buy
        if dist_to_support < 1.0 and range_width > 2.0:
            confidence = 0.5 + (1.0 - dist_to_support) * 0.3
            if pcr_oi > 1.0:
                confidence += 0.1
            return {
                "direction": SignalDirection.BUY,
                "confidence": min(confidence, 0.75),
                "type": "range_support",
                "notes": (
                    f"Range-bound: Price near Put OI support {support:.0f} "
                    f"(range: {support:.0f}-{resistance:.0f})"
                ),
            }

        # Near resistance in range → sell
        if dist_to_resistance < 1.0 and range_width > 2.0:
            confidence = 0.5 + (1.0 - dist_to_resistance) * 0.3
            if pcr_oi < 1.0:
                confidence += 0.1
            return {
                "direction": SignalDirection.SELL,
                "confidence": min(confidence, 0.75),
                "type": "range_resistance",
                "notes": (
                    f"Range-bound: Price near Call OI resistance {resistance:.0f} "
                    f"(range: {support:.0f}-{resistance:.0f})"
                ),
            }

        return None

    def _analyze_pcr(self, pcr_oi: float, price: float) -> Optional[dict[str, Any]]:
        """PCR-based sentiment signal."""
        if pcr_oi >= self.pcr_bullish:
            # High PCR = more puts being written = bullish (writers are bullish)
            confidence = min(0.4 + (pcr_oi - self.pcr_bullish) * 0.5, 0.7)
            return {
                "direction": SignalDirection.BUY,
                "confidence": confidence,
                "type": "pcr_bullish",
                "notes": f"PCR={pcr_oi:.2f} > {self.pcr_bullish} → Bullish sentiment",
            }

        elif pcr_oi <= self.pcr_bearish:
            # Low PCR = more calls being bought = bearish
            confidence = min(0.4 + (self.pcr_bearish - pcr_oi) * 0.5, 0.7)
            return {
                "direction": SignalDirection.SELL,
                "confidence": confidence,
                "type": "pcr_bearish",
                "notes": f"PCR={pcr_oi:.2f} < {self.pcr_bearish} → Bearish sentiment",
            }

        return None

    def _expiry_day_analysis(
        self,
        price: float,
        max_pain: dict,
        oi_levels: dict,
        option_chain: Optional[pd.DataFrame],
    ) -> Optional[dict[str, Any]]:
        """
        Expiry day special logic: Pinning to max pain + OI unwinding.

        On expiry, price tends to gravitate toward max pain as option writers
        (who are net short gamma) push price toward their break-even.
        """
        max_pain_strike = max_pain.get("max_pain_strike", 0)
        distance_pct = max_pain.get("distance_pct", 0)

        if max_pain_strike <= 0:
            return None

        # If price is far from max pain, expect pull toward it
        if abs(distance_pct) > self.max_pain_distance_pct:
            if price > max_pain_strike:
                # Price above max pain → expect downward pull
                return {
                    "direction": SignalDirection.SELL,
                    "confidence": 0.6,
                    "type": "expiry_max_pain_pull",
                    "notes": (
                        f"Expiry day: Price {price:.0f} above max pain {max_pain_strike:.0f} "
                        f"({distance_pct:.1f}%). Expecting gravity pull."
                    ),
                }
            else:
                # Price below max pain → expect upward pull
                return {
                    "direction": SignalDirection.BUY,
                    "confidence": 0.6,
                    "type": "expiry_max_pain_pull",
                    "notes": (
                        f"Expiry day: Price {price:.0f} below max pain {max_pain_strike:.0f} "
                        f"({distance_pct:.1f}%). Expecting gravity pull."
                    ),
                }

        return None

    def _propagate_to_stocks(
        self,
        index_signal: Signal,
        stock_universe: dict[str, dict],
        regime: str,
    ) -> list[Signal]:
        """Convert index-level OI signal to stock-level signals."""
        stock_signals = []

        if not stock_universe:
            return stock_signals

        # For bullish index signal → buy high-beta NIFTY stocks
        # For bearish → sell/avoid high-beta stocks
        for symbol, info in stock_universe.items():
            beta = info.get("beta", 1.0)
            price = info.get("price", 0)
            atr = info.get("atr", price * 0.02)

            # Only propagate to high-beta stocks that amplify the move
            if beta < 1.0:
                continue

            adjusted_confidence = index_signal.confidence * min(beta, 1.5) * 0.7
            adjusted_confidence = min(adjusted_confidence, 0.8)

            if adjusted_confidence < 0.3:
                continue

            signal = Signal(
                strategy=self.name,
                symbol=symbol,
                direction=index_signal.direction,
                confidence=adjusted_confidence,
                score=index_signal.score * beta,
                price=price,
                stop_loss=price - 1.5 * atr if index_signal.direction == SignalDirection.BUY else price + 1.5 * atr,
                take_profit=price + 3.0 * atr if index_signal.direction == SignalDirection.BUY else price - 3.0 * atr,
                hold_days=1 if "expiry" in index_signal.notes.lower() else 3,
                regime=regime,
                features={
                    "source": "index_oi_propagation",
                    "index_signal_type": index_signal.features.get("type", ""),
                    "beta": beta,
                },
                notes=f"Propagated from index OI signal: {index_signal.notes[:100]}",
            )
            stock_signals.append(signal)

        return stock_signals[:self.max_stocks if hasattr(self, 'max_stocks') else 5]

    def _create_index_signal(
        self,
        analysis: dict[str, Any],
        price: float,
        regime: str,
        oi_levels: dict,
        pcr: dict,
    ) -> Signal:
        """Create a Signal object from analysis result."""
        direction = analysis["direction"]
        atr_estimate = price * 0.015  # ~1.5% for NIFTY

        return Signal(
            strategy=self.name,
            symbol="NIFTY",
            direction=direction,
            confidence=analysis["confidence"],
            score=analysis["confidence"] if direction == SignalDirection.BUY else -analysis["confidence"],
            price=price,
            stop_loss=price - 1.5 * atr_estimate if direction == SignalDirection.BUY else price + 1.5 * atr_estimate,
            take_profit=price + 3.0 * atr_estimate if direction == SignalDirection.BUY else price - 3.0 * atr_estimate,
            hold_days=1 if "expiry" in analysis.get("type", "") else 3,
            regime=regime,
            features={
                "type": analysis["type"],
                "pcr_oi": pcr.get("pcr_oi", 0),
                "max_call_oi_strike": oi_levels.get("max_call_oi_strike", 0),
                "max_put_oi_strike": oi_levels.get("max_put_oi_strike", 0),
            },
            notes=analysis["notes"],
        )
