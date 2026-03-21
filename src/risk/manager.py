"""
Risk Manager — Position sizing (Half-Kelly), exposure limits, ATR-based stops.

Pre-trade risk checks with Upstox ChargeApi for exact cost calculation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml
from loguru import logger

from src.config.env_loader import get_config, _env_is_set


def clamp_sl_tp_by_premium(
    entry_premium: float, base_sl: float, base_tp: float
) -> tuple[float, float]:
    """Adjust SL/TP percentages based on premium level.

    Expensive premiums need massive NIFTY moves for same % TP.
    At ₹300 delta~0.5: 60% TP needs 360pts (impossible intraday).
    Cap TP inversely with premium to keep targets achievable.

    Returns (clamped_sl, clamped_tp).
    """
    if entry_premium < 100:
        return max(base_sl, 0.30), base_tp
    if entry_premium > 300:
        return min(base_sl, 0.20), min(base_tp, 0.25)
    if entry_premium > 200:
        return min(base_sl, 0.20), min(base_tp, 0.35)
    if entry_premium > 150:
        return base_sl, min(base_tp, 0.45)
    return base_sl, base_tp


class RiskManager:
    """
    Controls all risk parameters for the trading system.

    Responsibilities:
    - Position sizing (Half-Kelly, scaled by confidence + ATR)
    - Per-stock, per-sector, and total exposure limits
    - Pre-trade cost calculation (Upstox brokerage model)
    - Stop loss / take profit calculation
    - Pre-trade risk validation
    """

    def __init__(self, config_path: str = "config/risk.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        pos_cfg = self.config.get("position_sizing", {})
        self.kelly_fraction = pos_cfg.get("kelly_fraction", 0.6)
        self.max_per_trade_pct = pos_cfg.get("max_per_trade_pct", 4.0) / 100
        self.max_per_stock_pct = pos_cfg.get("max_per_stock_pct", 8.0) / 100
        self.max_per_sector_pct = pos_cfg.get("max_per_sector_pct", 30.0) / 100
        self.max_total_exposure_pct = pos_cfg.get("max_total_exposure_pct", 80.0) / 100
        self.min_position_value = pos_cfg.get("min_position_value", 5000)
        self.max_position_value = pos_cfg.get("max_position_value", 50000)
        self.confidence_scaling = pos_cfg.get("confidence_scaling", True)
        self.atr_scaling = pos_cfg.get("atr_scaling", True)
        self.atr_period = pos_cfg.get("atr_period", 14)
        self.min_cash_reserve_pct = pos_cfg.get("min_cash_reserve_pct", 20.0) / 100

        stops_cfg = self.config.get("stops", {})
        self.sl_atr_mult = stops_cfg.get("stop_loss_atr_multiplier", 1.5)
        self.tp_atr_mult = stops_cfg.get("take_profit_atr_multiplier", 3.0)
        self.trailing_stop = stops_cfg.get("trailing_stop", True)
        self.trailing_atr_mult = stops_cfg.get("trailing_stop_atr_multiplier", 2.0)
        self.trailing_activation_pct = stops_cfg.get("trailing_activation_pct", 1.0)
        self.time_stop_days = stops_cfg.get("time_stop_days", 5)
        self.max_slippage_pct = stops_cfg.get("max_slippage_pct", 0.5)
        self.use_gtt = stops_cfg.get("use_gtt_orders", True)

        # Brokerage model
        self.brokerage = self.config.get("brokerage", {})

        # Options risk config
        opts_cfg = self.config.get("options_risk", {})
        self.max_premium_per_trade = opts_cfg.get("max_premium_per_trade", 12000)
        self.max_lots_per_trade = opts_cfg.get("max_lots_per_trade", 1)
        self.options_sl_pct = opts_cfg.get("premium_stop_loss_pct", 30) / 100
        self.options_tp_pct = opts_cfg.get("premium_take_profit_pct", 60) / 100
        self.max_daily_options_loss = opts_cfg.get("max_daily_options_loss", 5000)
        self.options_min_premium = opts_cfg.get("min_premium", 50)
        self.options_max_premium = opts_cfg.get("max_premium", 500)

        # EnvConfig overlay for SL/TP
        cfg = get_config()
        if _env_is_set("SL_BASE_PCT"):
            self.options_sl_pct = cfg.SL_BASE_PCT / 100
        if _env_is_set("TP_BASE_PCT"):
            self.options_tp_pct = cfg.TP_BASE_PCT / 100

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        confidence: float,
        atr: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 2.0,
        current_exposure: float = 0.0,
        sector_exposure: float = 0.0,
        regime_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """
        Calculate position size using Half-Kelly criterion.

        Kelly: f* = (p * b - q) / b
        where p = win_rate, q = 1-p, b = avg_win/avg_loss

        Half-Kelly: f = f*/2 (more conservative)

        Further scaled by:
        - Confidence (higher confidence → bigger size)
        - ATR (higher volatility → smaller size)
        - Regime multiplier
        """
        if price <= 0 or capital <= 0:
            return {"quantity": 0, "value": 0, "reason": "invalid price/capital"}

        # ── Kelly Criterion ──
        p = max(0.01, min(win_rate, 0.99))
        q = 1 - p
        b = max(0.01, avg_win_loss_ratio)

        kelly_pct = (p * b - q) / b
        kelly_pct = max(0, kelly_pct)  # Never negative

        half_kelly_pct = kelly_pct * self.kelly_fraction

        # ── Apply limits ──
        position_pct = min(half_kelly_pct, self.max_per_trade_pct)

        # ── Confidence scaling ──
        if self.confidence_scaling:
            position_pct *= confidence

        # ── ATR (volatility) scaling ──
        if self.atr_scaling and atr > 0:
            # Inverse ATR: higher volatility → smaller position
            atr_pct = atr / price
            avg_atr_pct = 0.02  # Assume 2% as average ATR
            vol_scale = min(avg_atr_pct / max(atr_pct, 0.005), 2.0)
            position_pct *= vol_scale

        # ── Regime multiplier ──
        position_pct *= regime_multiplier

        # ── Calculate value and quantity ──
        position_value = capital * position_pct
        position_value = max(self.min_position_value, min(position_value, self.max_position_value))

        # ── Check exposure limits (respect cash reserve) ──
        deployable_capital = capital * (1 - self.min_cash_reserve_pct)
        max_exposure = min(self.max_total_exposure_pct * capital, deployable_capital)
        remaining_exposure = max_exposure - current_exposure
        if position_value > remaining_exposure:
            position_value = max(0, remaining_exposure)

        remaining_sector = (self.max_per_sector_pct * capital) - sector_exposure
        if position_value > remaining_sector:
            position_value = max(0, remaining_sector)

        # Final quantity (round to whole shares)
        quantity = int(position_value / price) if price > 0 else 0

        if quantity <= 0:
            return {
                "quantity": 0,
                "value": 0,
                "reason": "position too small after limits",
                "kelly_pct": round(kelly_pct * 100, 3),
                "half_kelly_pct": round(half_kelly_pct * 100, 3),
            }

        actual_value = quantity * price

        return {
            "quantity": quantity,
            "value": round(actual_value, 2),
            "position_pct": round(actual_value / capital * 100, 3),
            "kelly_pct": round(kelly_pct * 100, 3),
            "half_kelly_pct": round(half_kelly_pct * 100, 3),
            "confidence_scale": round(confidence, 3),
            "regime_scale": round(regime_multiplier, 3),
            "remaining_exposure": round(remaining_exposure, 2),
        }

    def calculate_stops(
        self,
        entry_price: float,
        atr: float,
        direction: str = "BUY",
    ) -> dict[str, float]:
        """
        Calculate stop loss and take profit using ATR multiples.

        Returns:
            {stop_loss, take_profit, trailing_trigger, trailing_stop, risk_per_share}
        """
        if direction == "BUY":
            sl = entry_price - self.sl_atr_mult * atr
            tp = entry_price + self.tp_atr_mult * atr
            trail_trigger = entry_price * (1 + self.trailing_activation_pct / 100)
            trail_stop = entry_price + self.trailing_atr_mult * atr  # Initial trail
        else:
            sl = entry_price + self.sl_atr_mult * atr
            tp = entry_price - self.tp_atr_mult * atr
            trail_trigger = entry_price * (1 - self.trailing_activation_pct / 100)
            trail_stop = entry_price - self.trailing_atr_mult * atr

        risk_per_share = abs(entry_price - sl)

        return {
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "trailing_trigger": round(trail_trigger, 2),
            "trailing_stop": round(trail_stop, 2),
            "risk_per_share": round(risk_per_share, 2),
            "reward_risk_ratio": round(abs(tp - entry_price) / risk_per_share, 2) if risk_per_share > 0 else 0,
        }

    def calculate_trade_costs(
        self,
        price: float,
        quantity: int,
        side: str = "BUY",
        product: str = "I",  # I=intraday, D=delivery
    ) -> dict[str, float]:
        """
        Calculate exact trade costs using Upstox brokerage model.

        Includes: brokerage, STT, GST, SEBI charges, stamp duty, NSE txn charges, DP charges.
        """
        turnover = price * quantity

        # ── Brokerage ──
        if product == "D":
            brokerage_cfg = self.brokerage.get("equity_delivery", {})
        else:
            brokerage_cfg = self.brokerage.get("equity_intraday", {})

        brokerage = min(
            brokerage_cfg.get("per_order", 20),
            turnover * brokerage_cfg.get("pct", 0.001),
        )

        # ── STT ──
        if product == "D":
            stt = turnover * self.brokerage.get("stt", {}).get("delivery_both_sides", 0.001)
        else:
            if side == "SELL":
                stt = turnover * self.brokerage.get("stt", {}).get("intraday_sell", 0.00025)
            else:
                stt = 0

        # ── NSE Transaction Charges ──
        nse_txn = turnover * self.brokerage.get("nse_txn_charges", 0.0000297)

        # ── SEBI Charges ──
        sebi = turnover * self.brokerage.get("sebi_charges_per_crore", 10) / 1e7

        # ── GST (18% on brokerage + txn + SEBI) ──
        gst_base = brokerage + nse_txn + sebi
        gst = gst_base * self.brokerage.get("gst_pct", 0.18)

        # ── Stamp Duty ──
        if side == "BUY":
            if product == "D":
                stamp = turnover * self.brokerage.get("stamp_duty", {}).get("delivery_buy", 0.00015)
            else:
                stamp = turnover * self.brokerage.get("stamp_duty", {}).get("intraday_buy", 0.00003)
        else:
            stamp = 0

        # ── DP Charges (only on delivery sell) ──
        dp = 0
        if product == "D" and side == "SELL":
            dp = self.brokerage.get("dp_charges_per_scrip", 18.5)

        total = brokerage + stt + nse_txn + sebi + gst + stamp + dp

        return {
            "turnover": round(turnover, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "nse_txn_charges": round(nse_txn, 2),
            "sebi_charges": round(sebi, 4),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp, 2),
            "dp_charges": round(dp, 2),
            "total_charges": round(total, 2),
            "charges_pct": round(total / turnover * 100, 4) if turnover > 0 else 0,
        }

    def calculate_round_trip_cost(
        self,
        price: float,
        quantity: int,
        product: str = "I",
    ) -> float:
        """Calculate total cost for a round-trip trade (buy + sell)."""
        buy_costs = self.calculate_trade_costs(price, quantity, "BUY", product)
        sell_costs = self.calculate_trade_costs(price, quantity, "SELL", product)
        return buy_costs["total_charges"] + sell_costs["total_charges"]

    def pre_trade_check(
        self,
        symbol: str,
        price: float,
        quantity: int,
        direction: str,
        capital: float,
        current_positions: pd.DataFrame,
        sector: str = "",
    ) -> dict[str, Any]:
        """
        Comprehensive pre-trade risk validation.

        Checks:
        1. Per-stock exposure limit
        2. Per-sector exposure limit
        3. Total exposure limit
        4. Max open positions
        5. Slippage budget
        6. Round-trip cost viability
        """
        checks: list[dict[str, Any]] = []
        trade_value = price * quantity
        passed = True

        # Check 1: Per-stock limit
        stock_limit = capital * self.max_per_stock_pct
        if trade_value > stock_limit:
            checks.append({
                "check": "per_stock_limit",
                "passed": False,
                "detail": f"₹{trade_value:.0f} > ₹{stock_limit:.0f} ({self.max_per_stock_pct*100}%)",
            })
            passed = False
        else:
            checks.append({"check": "per_stock_limit", "passed": True})

        # Check 2: Sector exposure
        sector_exposure = 0.0
        if not current_positions.empty and sector:
            sector_positions = current_positions[
                current_positions.get("sector", pd.Series()) == sector
            ]
            if not sector_positions.empty and "value" in sector_positions.columns:
                sector_exposure = sector_positions["value"].sum()

        sector_limit = capital * self.max_per_sector_pct
        if sector_exposure + trade_value > sector_limit:
            checks.append({
                "check": "sector_limit",
                "passed": False,
                "detail": f"Sector {sector}: ₹{sector_exposure + trade_value:.0f} > ₹{sector_limit:.0f}",
            })
            passed = False
        else:
            checks.append({"check": "sector_limit", "passed": True})

        # Check 3: Total exposure
        total_exposure = 0.0
        if not current_positions.empty and "value" in current_positions.columns:
            total_exposure = current_positions["value"].sum()

        exposure_limit = capital * self.max_total_exposure_pct
        if total_exposure + trade_value > exposure_limit:
            checks.append({
                "check": "total_exposure",
                "passed": False,
                "detail": f"Total: ₹{total_exposure + trade_value:.0f} > ₹{exposure_limit:.0f}",
            })
            passed = False
        else:
            checks.append({"check": "total_exposure", "passed": True})

        # Check 4: Max positions
        max_positions = self.config.get("circuit_breakers", {}).get("max_open_positions", 10)
        current_count = len(current_positions) if not current_positions.empty else 0
        if current_count >= max_positions:
            checks.append({
                "check": "max_positions",
                "passed": False,
                "detail": f"{current_count} >= {max_positions}",
            })
            passed = False
        else:
            checks.append({"check": "max_positions", "passed": True})

        # Check 5: Min position value
        if trade_value < self.min_position_value:
            checks.append({
                "check": "min_value",
                "passed": False,
                "detail": f"₹{trade_value:.0f} < ₹{self.min_position_value}",
            })
            passed = False
        else:
            checks.append({"check": "min_value", "passed": True})

        # Check 6: Cost viability
        rt_cost = self.calculate_round_trip_cost(price, quantity)
        cost_pct = rt_cost / trade_value * 100 if trade_value > 0 else 100
        if cost_pct > 1.0:  # Costs > 1% of trade → not viable
            checks.append({
                "check": "cost_viability",
                "passed": False,
                "detail": f"Round-trip cost {cost_pct:.2f}% > 1%",
            })
            passed = False
        else:
            checks.append({"check": "cost_viability", "passed": True})

        result = {
            "passed": passed,
            "checks": checks,
            "trade_value": round(trade_value, 2),
            "round_trip_cost": round(rt_cost, 2),
            "cost_pct": round(cost_pct, 4),
        }

        if not passed:
            failed = [c["check"] for c in checks if not c["passed"]]
            logger.warning(f"Pre-trade check FAILED for {symbol}: {failed}")
        else:
            logger.debug(f"Pre-trade check PASSED for {symbol}: ₹{trade_value:.0f}")

        return result

    # ──────────────────────────────────────────
    # Options-Specific Methods
    # ──────────────────────────────────────────

    def calculate_options_position_size(
        self,
        capital: float,
        premium: float,
        lot_size: int,
        max_premium_per_lot: float = 0,
        max_lots: int = 0,
    ) -> dict[str, Any]:
        """
        Calculate position size for options (lot-based sizing).

        Args:
            capital: Available capital
            premium: Option premium per unit
            lot_size: Contract lot size (75 for NIFTY, 15 for BANKNIFTY)
            max_premium_per_lot: Max premium × lot_size allowed
            max_lots: Max number of lots

        Returns:
            {lots, quantity, value, reason}
        """
        if premium <= 0 or capital <= 0 or lot_size <= 0:
            return {"lots": 0, "quantity": 0, "value": 0, "reason": "invalid inputs"}

        max_prem = max_premium_per_lot or self.max_premium_per_trade
        max_l = max_lots or self.max_lots_per_trade

        # Premium per lot
        cost_per_lot = premium * lot_size

        if cost_per_lot > max_prem:
            return {
                "lots": 0, "quantity": 0, "value": 0,
                "reason": f"Premium ₹{cost_per_lot:.0f}/lot > max ₹{max_prem:.0f}",
            }

        # Premium filter
        if premium < self.options_min_premium:
            return {
                "lots": 0, "quantity": 0, "value": 0,
                "reason": f"Premium ₹{premium:.0f} < min ₹{self.options_min_premium}",
            }
        if premium > self.options_max_premium:
            return {
                "lots": 0, "quantity": 0, "value": 0,
                "reason": f"Premium ₹{premium:.0f} > max ₹{self.options_max_premium}",
            }

        # How many lots can we afford?
        deployable = capital * (1 - self.min_cash_reserve_pct)
        affordable_lots = int(deployable / cost_per_lot)
        lots = min(affordable_lots, max_l)

        if lots <= 0:
            return {
                "lots": 0, "quantity": 0, "value": 0,
                "reason": f"Cannot afford: ₹{cost_per_lot:.0f}/lot > deployable ₹{deployable:.0f}",
            }

        quantity = lots * lot_size
        value = quantity * premium

        return {
            "lots": lots,
            "quantity": quantity,
            "value": round(value, 2),
            "cost_per_lot": round(cost_per_lot, 2),
            "premium": round(premium, 2),
        }

    def calculate_options_stops(
        self,
        entry_premium: float,
        sl_pct: float = 0,
        tp_pct: float = 0,
    ) -> dict[str, float]:
        """
        Calculate premium-based stop loss and take profit for options.

        Returns:
            {stop_loss, take_profit, risk_per_unit, reward_risk_ratio}
        """
        sl_pct = sl_pct or self.options_sl_pct
        tp_pct = tp_pct or self.options_tp_pct

        sl = entry_premium * (1 - sl_pct)
        tp = entry_premium * (1 + tp_pct)
        risk = entry_premium - sl

        return {
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "risk_per_unit": round(risk, 2),
            "reward_risk_ratio": round((tp - entry_premium) / risk, 2) if risk > 0 else 0,
        }

    def calculate_options_trade_costs(
        self,
        premium: float,
        quantity: int,
        side: str = "BUY",
    ) -> dict[str, float]:
        """
        Calculate trade costs for F&O options.

        Options cost structure:
        - Brokerage: flat ₹20 per order
        - STT: 0.0625% on sell side premium × quantity
        - Exchange txn charges: 0.05% (NSE F&O)
        - SEBI charges
        - GST: 18% on (brokerage + txn + SEBI)
        - Stamp duty: 0.003% on buy side
        """
        turnover = premium * quantity

        # Brokerage: flat ₹20
        fo_cfg = self.brokerage.get("fo_options", {})
        brokerage = fo_cfg.get("per_order", 20)

        # STT: sell side only, 0.05%
        stt = 0
        if side == "SELL":
            stt = turnover * 0.0005  # STT rate corrected per SEBI schedule

        # Exchange txn charges: 0.05%
        nse_txn = turnover * 0.0005

        # SEBI charges
        sebi = turnover * self.brokerage.get("sebi_charges_per_crore", 10) / 1e7

        # GST: 18% on (brokerage + txn + SEBI)
        gst = (brokerage + nse_txn + sebi) * self.brokerage.get("gst_pct", 0.18)

        # Stamp duty: buy side only
        stamp = 0
        if side == "BUY":
            stamp = turnover * 0.00003

        total = brokerage + stt + nse_txn + sebi + gst + stamp

        return {
            "turnover": round(turnover, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "nse_txn_charges": round(nse_txn, 2),
            "sebi_charges": round(sebi, 4),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp, 2),
            "total_charges": round(total, 2),
            "charges_pct": round(total / turnover * 100, 4) if turnover > 0 else 0,
        }

    # ──────────────────────────────────────────
    # PLUS Spread Risk Validation
    # ──────────────────────────────────────────

    def validate_spread_risk(
        self,
        trade_type: str,
        net_premium: float,
        spread_width: int,
        quantity: int,
    ) -> dict[str, Any]:
        """Validate spread max loss against RISK_PER_TRADE.

        Debit Spread: max loss = net_premium * qty (capped by spread structure).
        Credit Spread: max loss = (spread_width - credit) * qty.
        """
        from src.config.env_loader import get_config
        cfg = get_config()

        if trade_type == "DEBIT_SPREAD":
            max_loss = net_premium * quantity
        elif trade_type == "CREDIT_SPREAD":
            max_loss = (spread_width - net_premium) * quantity
        else:
            return {"passed": False, "reason": f"Unknown spread type: {trade_type}"}

        passed = max_loss <= cfg.RISK_PER_TRADE
        return {
            "passed": passed,
            "trade_type": trade_type,
            "max_loss": round(max_loss, 2),
            "risk_cap": cfg.RISK_PER_TRADE,
            "reason": "" if passed else f"Max loss ₹{max_loss:.0f} > risk cap ₹{cfg.RISK_PER_TRADE:.0f}",
        }
