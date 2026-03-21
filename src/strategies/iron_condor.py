"""
VELTRIX -- Iron Condor Strategy for RANGEBOUND regime.

Entry conditions (ALL must be true):
1. Regime == RANGEBOUND
2. ADX < 20 (confirmed range)
3. PCR between 0.80 and 1.20 (balanced market)
4. VIX between 15 and 22 (moderate IV)
5. abs(score_diff) < 2.0 (no strong directional bias)
6. Time window: 10:00 - 11:30 IST
7. NOT expiry day

Strike selection:
- Live/Paper: OI-based (sell at max OI strikes, buy protection SPREAD_WIDTH away)
- Backtest: ATM-based (sell_CE at ATM+200, sell_PE at ATM-200, buy wings further out)
"""

from __future__ import annotations

from datetime import time as dt_time
from typing import Any, Optional

from loguru import logger

from src.config.env_loader import get_config


class IronCondorStrategy:
    """Evaluates Iron Condor entry conditions and builds signal dicts."""

    def __init__(self) -> None:
        cfg = get_config()
        self.spread_width = cfg.IC_SPREAD_WIDTH
        self.min_credit = cfg.IC_MIN_CREDIT
        self.tp_pct = cfg.IC_TP_PCT / 100
        self.sl_multiplier = cfg.IC_SL_MULTIPLIER
        self.min_wing_distance = cfg.IC_MIN_WING_DISTANCE
        self.max_trades_per_day = cfg.IC_MAX_TRADES_PER_DAY
        self.adx_max = cfg.IC_ADX_MAX
        self.vix_min = cfg.IC_VIX_MIN
        self.vix_max = cfg.IC_VIX_MAX
        self.pcr_min = cfg.IC_PCR_MIN
        self.pcr_max = cfg.IC_PCR_MAX
        self.score_diff_max = cfg.IC_SCORE_DIFF_MAX
        self.trade_start = dt_time(*[int(x) for x in cfg.IC_TRADE_START.split(":")])
        self.trade_end = dt_time(*[int(x) for x in cfg.IC_TRADE_END.split(":")])
        self.max_opening_range_pct = cfg.IC_MAX_OPENING_RANGE_PCT
        self.max_vix_change_pct = cfg.IC_MAX_VIX_CHANGE_PCT
        self._trades_today = 0

    def reset_daily(self) -> None:
        """Reset daily trade counter."""
        self._trades_today = 0

    def check_entry_conditions(
        self,
        regime: str,
        adx: float,
        pcr: float,
        vix: float,
        score_diff: float,
        current_time: dt_time,
        is_expiry_day: bool,
        opening_range_pct: float = 0.0,
        vix_prev: float = 0.0,
    ) -> tuple[bool, str]:
        """Check all entry conditions. Returns (passed, reason_if_failed)."""
        if regime != "RANGEBOUND":
            return False, f"regime={regime} (need RANGEBOUND)"
        if adx >= self.adx_max:
            return False, f"ADX={adx:.1f} >= {self.adx_max}"
        if not (self.pcr_min <= pcr <= self.pcr_max):
            return False, f"PCR={pcr:.2f} outside [{self.pcr_min}, {self.pcr_max}]"
        if not (self.vix_min <= vix <= self.vix_max):
            return False, f"VIX={vix:.1f} outside [{self.vix_min}, {self.vix_max}]"
        if abs(score_diff) >= self.score_diff_max:
            return False, f"|score_diff|={abs(score_diff):.1f} >= {self.score_diff_max}"
        if not (self.trade_start <= current_time <= self.trade_end):
            return False, f"time={current_time} outside window"
        if is_expiry_day:
            return False, "expiry day -- no IC"
        if opening_range_pct > self.max_opening_range_pct > 0:
            return False, f"IC_SKIP: wide opening range {opening_range_pct:.2%}"
        if vix_prev > 0:
            vix_change_pct = abs(vix - vix_prev) / vix_prev
            if vix_change_pct > self.max_vix_change_pct:
                return False, f"IC_SKIP: VIX unstable change={vix_change_pct:.1%}"
        if self._trades_today >= self.max_trades_per_day:
            return False, f"IC trades today={self._trades_today} >= max"
        return True, ""

    def select_strikes_oi(
        self,
        spot_price: float,
        oi_data: dict[str, Any],
        strike_gap: int = 50,
    ) -> Optional[dict[str, float]]:
        """Select strikes using OI data (live/paper mode).

        Sell strikes at max OI positions, buy wings SPREAD_WIDTH further out.
        """
        max_call_oi_strike = oi_data.get("max_call_oi_strike", 0)
        max_put_oi_strike = oi_data.get("max_put_oi_strike", 0)

        if max_call_oi_strike <= 0 or max_put_oi_strike <= 0:
            return self.select_strikes_atm(spot_price, strike_gap)

        sell_ce = max_call_oi_strike
        sell_pe = max_put_oi_strike
        buy_ce = sell_ce + self.spread_width
        buy_pe = sell_pe - self.spread_width

        wing_distance = sell_ce - sell_pe
        if wing_distance < self.min_wing_distance:
            logger.info(
                f"IC_SKIP: wings too close (CE={sell_ce} PE={sell_pe} "
                f"dist={wing_distance} < {self.min_wing_distance})"
            )
            return None

        return {
            "sell_ce_strike": float(sell_ce),
            "buy_ce_strike": float(buy_ce),
            "sell_pe_strike": float(sell_pe),
            "buy_pe_strike": float(buy_pe),
        }

    def select_strikes_atm(
        self,
        spot_price: float,
        strike_gap: int = 50,
    ) -> Optional[dict[str, float]]:
        """Select strikes using ATM-based logic (backtest mode)."""
        atm = round(spot_price / strike_gap) * strike_gap
        sell_ce = atm + 200
        sell_pe = atm - 200
        buy_ce = sell_ce + self.spread_width
        buy_pe = sell_pe - self.spread_width

        wing_distance = sell_ce - sell_pe
        if wing_distance < self.min_wing_distance:
            logger.info(
                f"IC_SKIP: wings too close (CE={sell_ce} PE={sell_pe} "
                f"dist={wing_distance} < {self.min_wing_distance})"
            )
            return None

        return {
            "sell_ce_strike": float(sell_ce),
            "buy_ce_strike": float(buy_ce),
            "sell_pe_strike": float(sell_pe),
            "buy_pe_strike": float(buy_pe),
        }

    def calculate_position(
        self,
        sell_ce_prem: float,
        buy_ce_prem: float,
        sell_pe_prem: float,
        buy_pe_prem: float,
        lot_size: int,
        risk_per_trade: float,
        deploy_cap: float,
        strikes: dict[str, float],
    ) -> Optional[dict[str, Any]]:
        """Calculate IC economics and position size.

        Returns signal dict or None if invalid.
        """
        ce_credit = sell_ce_prem - buy_ce_prem
        pe_credit = sell_pe_prem - buy_pe_prem
        net_credit = ce_credit + pe_credit

        if net_credit < self.min_credit:
            logger.info(f"IC_SKIP: insufficient credit ₹{net_credit:.0f}")
            return None

        if ce_credit <= 0 or pe_credit <= 0:
            logger.info("IC_SKIP: one side has non-positive credit")
            return None

        # Max loss per unit (only one side can be ITM)
        if self.spread_width <= 0:
            logger.error(f"IC_SIZING_ERROR: spread_width={self.spread_width} must be > 0")
            return None

        max_loss_per_unit = self.spread_width - net_credit
        if max_loss_per_unit <= 0:
            max_loss_per_unit = self.spread_width

        # Position sizing
        if lot_size <= 0 or max_loss_per_unit <= 0:
            logger.error(f"IC_SIZING_ERROR: lot_size={lot_size} max_loss={max_loss_per_unit}")
            return None
        lots_by_risk = int(risk_per_trade / (max_loss_per_unit * lot_size))
        lots_by_deploy = int(deploy_cap / (max_loss_per_unit * lot_size))
        lots = max(1, min(lots_by_risk, lots_by_deploy))
        qty = lots * lot_size

        # Final risk check
        total_max_loss = max_loss_per_unit * qty
        if total_max_loss > risk_per_trade * 1.1:
            lots = 1
            qty = lot_size
            total_max_loss = max_loss_per_unit * qty

        max_profit = net_credit * qty
        tp_threshold = max_profit * self.tp_pct
        sl_threshold = -net_credit * self.sl_multiplier * qty

        return {
            "is_options": True,
            "is_iron_condor": True,
            "trade_type": "IRON_CONDOR",
            "direction": "NEUTRAL",
            "sell_ce_strike": strikes["sell_ce_strike"],
            "buy_ce_strike": strikes["buy_ce_strike"],
            "sell_pe_strike": strikes["sell_pe_strike"],
            "buy_pe_strike": strikes["buy_pe_strike"],
            "sell_ce_premium": sell_ce_prem,
            "buy_ce_premium": buy_ce_prem,
            "sell_pe_premium": sell_pe_prem,
            "buy_pe_premium": buy_pe_prem,
            "ce_credit": ce_credit,
            "pe_credit": pe_credit,
            "net_credit": net_credit,
            "max_loss_per_unit": max_loss_per_unit,
            "max_profit": round(max_profit, 2),
            "max_loss": round(total_max_loss, 2),
            "spread_width": self.spread_width,
            "quantity": qty,
            "lots": lots,
            "lot_size": lot_size,
            "tp_threshold": round(tp_threshold, 2),
            "sl_threshold": round(sl_threshold, 2),
        }

    def evaluate(
        self,
        regime: str,
        adx: float,
        pcr: float,
        vix: float,
        score_diff: float,
        current_time: dt_time,
        is_expiry_day: bool,
        spot_price: float,
        lot_size: int,
        risk_per_trade: float,
        deploy_cap: float,
        oi_data: Optional[dict[str, Any]] = None,
        premiums: Optional[dict[str, float]] = None,
        expiry_type: str = "",
        opening_range_pct: float = 0.0,
        vix_prev: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """Full IC evaluation: conditions -> strikes -> economics -> signal.

        In backtest mode: premiums are passed in directly.
        In live mode: premiums will be fetched by the caller after strike selection.
        Returns signal dict or None.
        """
        ok, reason = self.check_entry_conditions(
            regime, adx, pcr, vix, score_diff, current_time, is_expiry_day,
            opening_range_pct=opening_range_pct, vix_prev=vix_prev,
        )
        if not ok:
            return None

        # Strike selection
        if oi_data and oi_data.get("max_call_oi_strike", 0) > 0:
            strikes = self.select_strikes_oi(spot_price, oi_data)
        else:
            strikes = self.select_strikes_atm(spot_price)

        if strikes is None:
            return None

        # If premiums not provided, return strikes for caller to fetch
        if premiums is None:
            return {"strikes": strikes, "need_premiums": True}

        # Calculate position
        signal = self.calculate_position(
            sell_ce_prem=premiums["sell_ce"],
            buy_ce_prem=premiums["buy_ce"],
            sell_pe_prem=premiums["sell_pe"],
            buy_pe_prem=premiums["buy_pe"],
            lot_size=lot_size,
            risk_per_trade=risk_per_trade,
            deploy_cap=deploy_cap,
            strikes=strikes,
        )
        if signal:
            signal["regime"] = regime
            signal["expiry_type"] = expiry_type
            signal["conviction"] = round(abs(score_diff), 2)
            self._trades_today += 1
        return signal
