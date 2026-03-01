"""
Regime Detection — Layer 1 of the alpha system.

Determines the current market ENVIRONMENT (not direction).
Direction comes from scoring. Regime controls HOW to trade.

3 Regimes:
- TRENDING:   Clear directional move, strong momentum (ADX > 25)
- RANGEBOUND: No clear direction, choppy (ADX < 20, VIX < 22)
- VOLATILE:   High uncertainty, big swings (VIX > 22 or sudden spike)

Each regime deeply controls: conviction threshold, SL/TP multipliers,
position sizing, trailing stop behavior, and factor weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import copy

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.config.env_loader import get_config, _env_is_set


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGEBOUND = "RANGEBOUND"
    VOLATILE = "VOLATILE"


@dataclass
class RegimeState:
    """Current market regime with all supporting data."""

    regime: MarketRegime
    timestamp: datetime
    vix: float = 0.0
    vix_5d_change: float = 0.0
    nifty_price: float = 0.0
    nifty_ma50: float = 0.0
    adx: float = 0.0
    adx_slope: float = 0.0        # ADX direction: positive = strengthening
    bb_width: float = 0.0         # Bollinger Band width (volatility proxy)
    intraday_range_pct: float = 0.0  # Today's high-low / close
    fii_net_5d: float = 0.0
    is_expiry_week: bool = False
    active_strategies: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0
    confidence: float = 0.0
    notes: str = ""

    # Regime behavior parameters — controls HOW strategies trade
    conviction_min: float = 1.5      # Minimum score_diff to trade
    sl_multiplier: float = 1.0       # Multiply adaptive SL by this
    tp_multiplier: float = 1.0       # Multiply adaptive TP by this
    trailing_stop_enabled: bool = True
    ema_weight: float = 2.0          # EMA factor weight in scoring
    mean_reversion_weight: float = 1.5  # Mean reversion factor weight
    max_trades_per_day: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "datetime": self.timestamp.isoformat(),
            "vix_value": self.vix,
            "vix_5d_change": self.vix_5d_change,
            "nifty_value": self.nifty_price,
            "nifty_ma50": self.nifty_ma50,
            "adx_value": self.adx,
            "adx_slope": self.adx_slope,
            "bb_width": self.bb_width,
            "intraday_range_pct": self.intraday_range_pct,
            "fii_net_value": self.fii_net_5d,
            "is_expiry_week": self.is_expiry_week,
            "active_strategies": self.active_strategies,
            "size_multiplier": self.size_multiplier,
            "confidence": self.confidence,
            "notes": self.notes,
            "conviction_min": self.conviction_min,
            "sl_multiplier": self.sl_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "ema_weight": self.ema_weight,
            "mean_reversion_weight": self.mean_reversion_weight,
            "max_trades_per_day": self.max_trades_per_day,
        }


class RegimeDetector:
    """
    3-regime market environment classifier.

    TRENDING:   ADX > 25 or (ADX > 20 and rising) — ride the wave
    RANGEBOUND: ADX < 20 and VIX < 22 — quick in/out, take small profits
    VOLATILE:   VIX > 22 or VIX spike > 3pts/5d — protect capital

    Direction (bull/bear) is NOT part of regime.
    Regime describes the environment. Scoring describes the direction.
    """

    # All regimes allow options_buyer (it has its own scoring filters)
    REGIME_STRATEGIES: dict[MarketRegime, list[str]] = {
        MarketRegime.TRENDING: [
            "options_oi", "ml_predictor", "options_buyer"
        ],
        MarketRegime.RANGEBOUND: [
            "options_oi", "ml_predictor", "options_buyer"
        ],
        MarketRegime.VOLATILE: [
            "options_oi", "options_buyer"  # ML unreliable in chaos
        ],
    }

    # Regime behavior profiles
    REGIME_PROFILES: dict[MarketRegime, dict[str, Any]] = {
        MarketRegime.TRENDING: {
            # "Ride the wave, let winners run"
            "size_multiplier": 1.0,
            "conviction_min": 1.75,         # V8: lowered from 2.0 for more trades
            "sl_multiplier": 1.0,           # Standard SL
            "tp_multiplier": 1.3,           # Let winners run (+30% wider TP)
            "trailing_stop_enabled": True,   # Trail profits
            "ema_weight": 2.5,              # Trust trend signals
            "mean_reversion_weight": 1.5,   # Standard
            "max_trades_per_day": 2,
        },
        MarketRegime.RANGEBOUND: {
            # "Quick in, quick out, take small profits"
            "size_multiplier": 0.5,         # Half position size
            "conviction_min": 2.0,          # Higher bar (fewer trades in noise)
            "sl_multiplier": 0.85,          # Tighter SL (cut losses faster)
            "tp_multiplier": 0.70,          # Lower TP (take quick wins)
            "trailing_stop_enabled": False,  # Moves reverse too fast
            "ema_weight": 1.0,              # Halved (trend signals unreliable)
            "mean_reversion_weight": 2.5,   # Mean reversion is king
            "max_trades_per_day": 1,
        },
        MarketRegime.VOLATILE: {
            # "Protect capital, only trade A+ setups"
            "size_multiplier": 0.5,         # Half position always
            "conviction_min": 2.5,          # V8: lowered from 3.0
            "sl_multiplier": 1.20,          # Wider SL (give room for swings)
            "tp_multiplier": 1.50,          # Much higher TP (big moves possible)
            "trailing_stop_enabled": True,   # Trail the big moves
            "ema_weight": 0.5,              # EMAs lag in chaos
            "mean_reversion_weight": 1.0,   # Standard
            "max_trades_per_day": 1,
        },
    }

    def __init__(self, config_path: str = "config/strategies.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        regime_cfg = config.get("regime", {})
        vix_cfg = regime_cfg.get("vix", {})
        adx_cfg = regime_cfg.get("adx", {})

        self.vix_volatile = vix_cfg.get("high_threshold", 22.0)
        self.vix_extreme = vix_cfg.get("extreme_threshold", 30.0)
        self.vix_spike_threshold = 3.0  # 5-day VIX change > 3 points = spike

        self.adx_trending = adx_cfg.get("trending_threshold", 25)
        self.adx_sideways = adx_cfg.get("sideways_threshold", 20)

        self.cooldown_minutes = regime_cfg.get("regime_change_cooldown_minutes", 60)

        # Instance-level copy so class constant is never mutated
        self._profiles = copy.deepcopy(self.REGIME_PROFILES)

        # Override conviction thresholds from env if explicitly set
        cfg = get_config()
        if _env_is_set("TRENDING_THRESHOLD"):
            self._profiles[MarketRegime.TRENDING]["conviction_min"] = cfg.TRENDING_THRESHOLD
        if _env_is_set("RANGEBOUND_THRESHOLD"):
            self._profiles[MarketRegime.RANGEBOUND]["conviction_min"] = cfg.RANGEBOUND_THRESHOLD
        if _env_is_set("VOLATILE_THRESHOLD"):
            self._profiles[MarketRegime.VOLATILE]["conviction_min"] = cfg.VOLATILE_THRESHOLD

        self._last_regime: Optional[RegimeState] = None
        self._last_change_time: Optional[datetime] = None

    def detect(
        self,
        vix_data: dict[str, float],
        nifty_df: pd.DataFrame,
        fii_data: pd.DataFrame,
        is_expiry_week: bool = False,
        intraday_df: Optional[pd.DataFrame] = None,
    ) -> RegimeState:
        """
        Determine current market regime.

        Args:
            vix_data: {"vix": 14.5, "change_pct": -2.0}
            nifty_df: NIFTY 50 OHLCV DataFrame (needs at least 50 bars)
            fii_data: FII/DII flow history DataFrame
            is_expiry_week: Whether this is an expiry week
            intraday_df: Optional 5-min intraday data for intraday range calc

        Returns:
            RegimeState with the detected regime and behavior parameters
        """
        now = datetime.now()
        vix = vix_data.get("vix", 0)

        # ── Compute indicators ──
        nifty_price = 0.0
        nifty_ma50 = 0.0
        adx_value = 0.0
        adx_slope = 0.0
        bb_width = 0.0

        if not nifty_df.empty and len(nifty_df) >= 50:
            nifty_price = float(nifty_df["close"].iloc[-1])
            ma = nifty_df["close"].ewm(span=50, adjust=False).mean()
            nifty_ma50 = float(ma.iloc[-1])

            adx_value = self._compute_adx(nifty_df)

            # ADX slope: is trend strengthening or weakening?
            adx_series = self._compute_adx_series(nifty_df)
            if len(adx_series) >= 5:
                adx_slope = float(adx_series.iloc[-1] - adx_series.iloc[-5])

            # Bollinger Band width
            bb_width = self._compute_bb_width(nifty_df)

        # VIX 5-day change
        vix_5d_change = vix_data.get("change_pct", 0)

        # Intraday range
        intraday_range_pct = 0.0
        if intraday_df is not None and not intraday_df.empty:
            day_high = float(intraday_df["high"].max())
            day_low = float(intraday_df["low"].min())
            day_close = float(intraday_df["close"].iloc[-1])
            if day_close > 0:
                intraday_range_pct = (day_high - day_low) / day_close * 100

        # FII flow (informational only — often zero)
        fii_net_5d = 0.0
        if isinstance(fii_data, pd.DataFrame) and not fii_data.empty and "fii_net_value" in fii_data.columns:
            recent_fii = fii_data.tail(5)
            fii_net_5d = float(recent_fii["fii_net_value"].sum())

        # ── Classify ──
        regime, confidence, notes = self._classify(
            vix, vix_5d_change, adx_value, adx_slope,
            intraday_range_pct, bb_width,
        )

        # ── Upgrade-only enforcement after morning session ──
        # After 11:00, regime can only UPGRADE to VOLATILE (never downgrade).
        # This prevents whipsawing between TRENDING/RANGEBOUND during the day.
        if self._last_regime and now.hour >= 11:
            prev = self._last_regime.regime
            if prev == MarketRegime.VOLATILE and regime != MarketRegime.VOLATILE:
                # Once VOLATILE, stay VOLATILE for the day
                regime = MarketRegime.VOLATILE
                notes += " [VOLATILE locked]"
            elif prev != MarketRegime.VOLATILE and regime != MarketRegime.VOLATILE and regime != prev:
                # After 11:00, don't switch between TRENDING↔RANGEBOUND
                regime = prev
                notes += " [regime locked after 11:00]"

        # ── Cooldown check ──
        if self._last_regime and self._last_change_time:
            elapsed = (now - self._last_change_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes and regime != self._last_regime.regime:
                logger.info(
                    f"Regime change cooldown: {elapsed:.0f}/{self.cooldown_minutes}min. "
                    f"Keeping {self._last_regime.regime.value}"
                )
                regime = self._last_regime.regime
                notes += " [cooldown active]"

        # ── Build result with behavior profile ──
        active_strategies = list(self.REGIME_STRATEGIES.get(regime, []))
        profile = self._profiles.get(regime, self._profiles[MarketRegime.TRENDING])

        state = RegimeState(
            regime=regime,
            timestamp=now,
            vix=vix,
            vix_5d_change=vix_5d_change,
            nifty_price=nifty_price,
            nifty_ma50=nifty_ma50,
            adx=adx_value,
            adx_slope=adx_slope,
            bb_width=bb_width,
            intraday_range_pct=intraday_range_pct,
            fii_net_5d=fii_net_5d,
            is_expiry_week=is_expiry_week,
            active_strategies=active_strategies,
            size_multiplier=profile["size_multiplier"],
            confidence=confidence,
            notes=notes,
            conviction_min=profile["conviction_min"],
            sl_multiplier=profile["sl_multiplier"],
            tp_multiplier=profile["tp_multiplier"],
            trailing_stop_enabled=profile["trailing_stop_enabled"],
            ema_weight=profile["ema_weight"],
            mean_reversion_weight=profile["mean_reversion_weight"],
            max_trades_per_day=profile["max_trades_per_day"],
        )

        # Track regime changes
        if self._last_regime is None or regime != self._last_regime.regime:
            self._last_change_time = now
            logger.info(
                f"REGIME: {self._last_regime.regime.value if self._last_regime else 'INIT'} "
                f"→ {regime.value} | VIX={vix:.1f} ADX={adx_value:.1f} "
                f"ADX_slope={adx_slope:+.1f} BB_width={bb_width:.3f} "
                f"intraday_range={intraday_range_pct:.1f}% | "
                f"conviction>={profile['conviction_min']:.1f} "
                f"SL×{profile['sl_multiplier']:.2f} TP×{profile['tp_multiplier']:.2f} "
                f"size×{profile['size_multiplier']:.1f} "
                f"trailing={'ON' if profile['trailing_stop_enabled'] else 'OFF'}"
            )

        self._last_regime = state
        return state

    def _classify(
        self,
        vix: float,
        vix_5d_change: float,
        adx: float,
        adx_slope: float,
        intraday_range_pct: float,
        bb_width: float,
    ) -> tuple[MarketRegime, float, str]:
        """
        Core 3-regime classification.

        Priority: VOLATILE > TRENDING > RANGEBOUND
        """
        notes_parts: list[str] = []

        # ── VOLATILE — takes priority (risk-off environment) ──
        # VIX extreme: immediate volatile
        if vix >= self.vix_extreme:
            notes_parts.append(f"VIX={vix:.1f} EXTREME (>={self.vix_extreme})")
            return MarketRegime.VOLATILE, 0.95, " | ".join(notes_parts)

        # VIX elevated OR sudden spike OR wide intraday range
        volatile_score = 0
        if vix > self.vix_volatile:
            volatile_score += 2
            notes_parts.append(f"VIX={vix:.1f} HIGH (>{self.vix_volatile})")
        if vix_5d_change > self.vix_spike_threshold:
            volatile_score += 2
            notes_parts.append(f"VIX spike +{vix_5d_change:.1f}pts/5d")
        if intraday_range_pct > 2.0:
            volatile_score += 1
            notes_parts.append(f"Wide range {intraday_range_pct:.1f}%")

        if volatile_score >= 2:
            notes_parts.append(f"ADX={adx:.1f}")
            conf = min(0.5 + volatile_score * 0.15, 0.95)
            return MarketRegime.VOLATILE, conf, " | ".join(notes_parts)

        # ── TRENDING — clear directional move ──
        trending_score = 0
        if adx > self.adx_trending:
            trending_score += 2
            notes_parts.append(f"ADX={adx:.1f} TRENDING (>{self.adx_trending})")
        elif adx > self.adx_sideways and adx_slope > 0:
            trending_score += 1
            notes_parts.append(f"ADX={adx:.1f} strengthening (slope={adx_slope:+.1f})")

        if bb_width > 0.04:  # Expanding Bollinger = trending/volatile
            trending_score += 1
            notes_parts.append(f"BB expanding ({bb_width:.3f})")

        if trending_score >= 2:
            conf = min(0.5 + trending_score * 0.15, 0.90)
            return MarketRegime.TRENDING, conf, " | ".join(notes_parts)

        # ── RANGEBOUND — everything else ──
        notes_parts.append(f"VIX={vix:.1f} ADX={adx:.1f}")
        if adx <= self.adx_sideways:
            notes_parts.append("ADX SIDEWAYS")
        if bb_width <= 0.03:
            notes_parts.append(f"BB narrow ({bb_width:.3f})")

        return MarketRegime.RANGEBOUND, 0.70, " | ".join(notes_parts)

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute current ADX value from OHLCV data."""
        adx_series = self._compute_adx_series(df)
        if adx_series.empty:
            return 0.0
        latest = adx_series.iloc[-1]
        return float(latest) if not np.isnan(latest) else 0.0

    def _compute_adx_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute full ADX series for slope calculation."""
        if len(df) < period * 2:
            return pd.Series(dtype=float)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    def _compute_bb_width(self, df: pd.DataFrame, period: int = 20) -> float:
        """Compute Bollinger Band width (normalized by price)."""
        if len(df) < period:
            return 0.0
        close = df["close"]
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        width = (upper - lower) / sma
        latest = width.iloc[-1]
        return float(latest) if not np.isnan(latest) else 0.0

    def get_active_strategies(self, regime: Optional[MarketRegime] = None) -> list[str]:
        """Get list of active strategy names for given (or current) regime."""
        if regime is None:
            if self._last_regime:
                regime = self._last_regime.regime
            else:
                return []
        return list(self.REGIME_STRATEGIES.get(regime, []))

    def get_size_multiplier(self, regime: Optional[MarketRegime] = None) -> float:
        """Get position size multiplier for given (or current) regime."""
        if regime is None:
            if self._last_regime:
                return self._last_regime.size_multiplier
            return 1.0
        profile = self._profiles.get(regime, {})
        return profile.get("size_multiplier", 1.0)

    @property
    def current_regime(self) -> Optional[RegimeState]:
        return self._last_regime
