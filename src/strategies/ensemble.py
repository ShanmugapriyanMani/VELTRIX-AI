"""
Regime-Aware Ensemble — Layer 3 of the alpha system.

Combines signals from all active strategies using Sharpe-weighted averaging.
Regime filter controls which strategies contribute.
Score > +0.5 → BUY, Score < -0.5 → SELL, otherwise HOLD.
Strategy disagreement → reduce position size 50%.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.strategies.base import BaseStrategy, Signal, SignalDirection
from src.regime.detector import RegimeDetector, RegimeState, MarketRegime


class EnsembleStrategy:
    """
    Regime-aware ensemble that combines signals from all active strategies.

    Process:
    1. Regime detector determines which strategies are active
    2. Each active strategy generates signals
    3. Strategies are weighted by their rolling 63-day Sharpe ratio
    4. Ensemble score = Σ(weight_i × signal_i × confidence_i)
    5. Final decision with disagreement handling
    """

    def __init__(self, config_path: str = "config/strategies.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ensemble_cfg = config.get("ensemble", {})
        self.sharpe_lookback = ensemble_cfg.get("sharpe_lookback_days", 63)
        self.min_weight = ensemble_cfg.get("min_sharpe_weight", 0.05)
        self.max_weight = ensemble_cfg.get("max_sharpe_weight", 0.50)
        self.buy_threshold = ensemble_cfg.get("buy_threshold", 0.5)
        self.sell_threshold = ensemble_cfg.get("sell_threshold", -0.5)
        self.disagreement_reduction = ensemble_cfg.get("disagreement_size_reduction", 0.5)
        self.regime_override = ensemble_cfg.get("regime_override", True)
        self.min_active_strategies = ensemble_cfg.get("min_active_strategies", 2)
        self.signal_expiry_minutes = ensemble_cfg.get("signal_expiry_minutes", 30)

        self._strategies: dict[str, BaseStrategy] = {}
        self._regime_detector: Optional[RegimeDetector] = None
        self._signal_history: list[dict] = []

    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy with the ensemble."""
        self._strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name} (weight={strategy.weight})")

    def set_regime_detector(self, detector: RegimeDetector) -> None:
        """Set the regime detector."""
        self._regime_detector = detector

    def generate_ensemble_signals(
        self,
        data: dict[str, Any],
        regime_state: RegimeState,
    ) -> dict[str, Any]:
        """
        Generate the final ensemble decision.

        Args:
            data: All data needed by individual strategies
            regime_state: Current regime from RegimeDetector

        Returns:
            {
                "symbol": "RELIANCE",
                "direction": "BUY" | "SELL" | "HOLD",
                "ensemble_score": 0.65,
                "confidence": 0.72,
                "size_multiplier": 1.0,
                "active_strategies": [...],
                "strategy_signals": [...],
                "regime": "TRENDING",
                "notes": "...",
            }
        """
        active_names = regime_state.active_strategies
        regime_mult = regime_state.size_multiplier

        # Collect signals from active strategies
        all_signals: dict[str, list[Signal]] = {}  # {strategy_name: [signals]}

        for strat_name, strategy in self._strategies.items():
            if strat_name not in active_names:
                continue
            if not strategy.enabled:
                continue

            data["regime"] = regime_state.regime.value
            try:
                signals = strategy.generate_signals(data)
                if signals:
                    all_signals[strat_name] = signals
            except Exception as e:
                logger.error(f"Strategy {strat_name} failed: {e}")

        if len(all_signals) < self.min_active_strategies:
            logger.debug(
                f"Only {len(all_signals)} active strategies "
                f"(need {self.min_active_strategies}). Holding."
            )
            return self._hold_result(regime_state, all_signals)

        # Group signals by symbol
        symbol_signals = self._group_by_symbol(all_signals)

        # Generate ensemble decision per symbol
        results = []
        for symbol, strat_signals in symbol_signals.items():
            result = self._combine_signals(
                symbol, strat_signals, regime_state, regime_mult
            )
            results.append(result)

        return {
            "regime": regime_state.regime.value,
            "regime_confidence": regime_state.confidence,
            "active_strategies": active_names,
            "total_signals": sum(len(s) for s in all_signals.values()),
            "decisions": results,
            "timestamp": datetime.now().isoformat(),
        }

    def _combine_signals(
        self,
        symbol: str,
        strat_signals: dict[str, list[Signal]],
        regime_state: RegimeState,
        regime_mult: float,
    ) -> dict[str, Any]:
        """
        Combine signals for a single symbol using Sharpe-weighted averaging.
        """
        # Calculate weights based on rolling Sharpe
        weights = self._compute_weights(strat_signals.keys())

        # Compute weighted ensemble score
        total_score = 0.0
        total_weight = 0.0
        strategy_details = []
        buy_count = 0
        sell_count = 0

        for strat_name, signals in strat_signals.items():
            w = weights.get(strat_name, 0.25)

            # Aggregate signals from same strategy (take strongest)
            best_signal = max(signals, key=lambda s: abs(s.score))

            if best_signal.direction == SignalDirection.BUY:
                signal_val = best_signal.confidence
                buy_count += 1
            elif best_signal.direction == SignalDirection.SELL:
                signal_val = -best_signal.confidence
                sell_count += 1
            else:
                signal_val = 0
                continue

            weighted_score = w * signal_val
            total_score += weighted_score
            total_weight += w

            strategy_details.append({
                "strategy": strat_name,
                "direction": best_signal.direction.value,
                "confidence": best_signal.confidence,
                "score": best_signal.score,
                "weight": w,
                "weighted_score": round(weighted_score, 4),
                "notes": best_signal.notes[:100],
            })

        # Normalize score
        if total_weight > 0:
            ensemble_score = total_score / total_weight
        else:
            ensemble_score = 0

        # ── Determine direction ──
        if ensemble_score > self.buy_threshold:
            direction = "BUY"
        elif ensemble_score < self.sell_threshold:
            direction = "SELL"
        else:
            direction = "HOLD"

        # ── Disagreement check ──
        size_multiplier = regime_mult
        disagreement = False

        total_strategies = buy_count + sell_count
        if total_strategies >= 2 and buy_count >= 1 and sell_count >= 1:
            disagreement = True
            size_multiplier *= self.disagreement_reduction
            logger.info(
                f"ENSEMBLE DISAGREEMENT for {symbol}: "
                f"{buy_count} buy, {sell_count} sell → size reduced 50%"
            )

        # Confidence = absolute ensemble score normalized
        confidence = min(abs(ensemble_score), 1.0)

        # Get best stop/take profit from strongest signal
        all_signals = [s for sigs in strat_signals.values() for s in sigs]
        best = max(all_signals, key=lambda s: s.confidence)

        result = {
            "symbol": symbol,
            "direction": direction,
            "ensemble_score": round(ensemble_score, 4),
            "confidence": round(confidence, 4),
            "size_multiplier": round(size_multiplier, 3),
            "regime": regime_state.regime.value,
            "stop_loss": best.stop_loss,
            "take_profit": best.take_profit,
            "hold_days": best.hold_days,
            "price": best.price,
            "disagreement": disagreement,
            "buy_strategies": buy_count,
            "sell_strategies": sell_count,
            "strategy_details": strategy_details,
            "features": best.features,
            "strategy": best.strategy,
            "timestamp": datetime.now().isoformat(),
        }

        # Log the decision
        self._signal_history.append(result)
        if direction != "HOLD":
            logger.info(
                f"ENSEMBLE → {symbol}: {direction} (score={ensemble_score:.3f}, "
                f"conf={confidence:.3f}, size_mult={size_multiplier:.2f}) | "
                f"Strategies: {[d['strategy'] for d in strategy_details]}"
            )

        return result

    def _compute_weights(self, strategy_names) -> dict[str, float]:
        """
        Compute Sharpe-weighted strategy weights.

        Strategies with higher rolling Sharpe get more weight.
        """
        raw_weights = {}

        for name in strategy_names:
            strategy = self._strategies.get(name)
            if not strategy:
                continue

            sharpe = max(strategy.rolling_sharpe, 0.01)  # Floor at 0.01
            base_weight = strategy.weight
            raw_weights[name] = sharpe * base_weight

        # Normalize
        total = sum(raw_weights.values())
        if total <= 0:
            # Equal weights as fallback
            n = len(raw_weights)
            return {name: 1.0 / n for name in raw_weights} if n > 0 else {}

        weights = {}
        for name, raw in raw_weights.items():
            w = raw / total
            w = max(self.min_weight, min(w, self.max_weight))
            weights[name] = w

        # Re-normalize after clamping
        total_clamped = sum(weights.values())
        if total_clamped > 0:
            weights = {k: v / total_clamped for k, v in weights.items()}

        return weights

    def _group_by_symbol(
        self, all_signals: dict[str, list[Signal]]
    ) -> dict[str, dict[str, list[Signal]]]:
        """Group signals by symbol across strategies."""
        symbol_signals: dict[str, dict[str, list[Signal]]] = {}

        for strat_name, signals in all_signals.items():
            for signal in signals:
                sym = signal.symbol
                if sym not in symbol_signals:
                    symbol_signals[sym] = {}
                if strat_name not in symbol_signals[sym]:
                    symbol_signals[sym][strat_name] = []
                symbol_signals[sym][strat_name].append(signal)

        return symbol_signals

    def _hold_result(
        self,
        regime_state: RegimeState,
        all_signals: dict[str, list[Signal]],
    ) -> dict[str, Any]:
        """Return a HOLD result when not enough strategies are active."""
        return {
            "regime": regime_state.regime.value,
            "regime_confidence": regime_state.confidence,
            "active_strategies": regime_state.active_strategies,
            "total_signals": sum(len(s) for s in all_signals.values()),
            "decisions": [],
            "timestamp": datetime.now().isoformat(),
            "notes": "HOLD: insufficient active strategies",
        }

    @property
    def strategy_stats(self) -> list[dict[str, Any]]:
        """Get performance stats for all registered strategies."""
        return [s.stats for s in self._strategies.values()]

    @property
    def recent_decisions(self) -> list[dict]:
        """Get recent ensemble decisions."""
        return self._signal_history[-50:]
