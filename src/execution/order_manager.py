"""
Order Manager — Smart routing, GTT stop/target, EOD square-off, kill switch.

Sits between strategy signals and the broker, handling order lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time as dt_time
from typing import Any, Optional

import yaml
from loguru import logger

from src.execution.broker import BaseBroker
from src.risk.manager import RiskManager
from src.risk.circuit_breaker import CircuitBreaker


class OrderManager:
    """
    Manages the full order lifecycle:
    - Smart order routing (LIMIT → aggressive → MARKET escalation)
    - GTT orders for stop loss and take profit
    - EOD square-off at 3:15 PM for intraday
    - Kill switch integration
    - Order tracking and reconciliation
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: RiskManager,
        circuit_breaker: CircuitBreaker,
        config_path: str = "config/config.yaml",
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.circuit_breaker = circuit_breaker

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        trading_cfg = config.get("trading", {})
        self.square_off_time = trading_cfg.get("square_off_time", "15:15")
        self.product = "I"  # Default intraday

        # Active orders tracking
        self._pending_orders: dict[str, dict] = {}
        self._gtt_orders: dict[str, dict] = {}
        self._filled_orders: list[dict] = []

    def execute_signal(
        self,
        signal: dict[str, Any],
        capital: float,
        current_positions: Any,
    ) -> dict[str, Any]:
        """
        Execute a trading signal through the broker.

        Process:
        1. Pre-trade risk check
        2. Calculate position size
        3. Place order (smart routing)
        4. Place GTT stop loss and take profit
        5. Track order
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            return {
                "status": "blocked",
                "reason": f"Circuit breaker: {self.circuit_breaker.status.state.value}",
            }

        if not self.circuit_breaker.record_order():
            return {"status": "blocked", "reason": "Daily order limit reached"}

        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "HOLD")
        instrument_key = signal.get("instrument_key", "")
        price = signal.get("price", 0)
        confidence = signal.get("confidence", 0.5)
        stop_loss = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)
        atr = signal.get("atr", price * 0.02)
        regime_mult = signal.get("size_multiplier", 1.0)

        if direction not in ("BUY", "SELL") or price <= 0:
            return {"status": "skipped", "reason": "Invalid signal"}

        # ── Check if this is an options signal ──
        features = signal.get("features", {})
        is_options = features.get("is_options", False) or instrument_key.startswith("NSE_FO|")

        if is_options:
            return self._execute_options_signal(signal, capital, current_positions)

        # ── Equity position sizing ──
        sizing = self.risk_manager.calculate_position_size(
            capital=capital,
            price=price,
            confidence=confidence,
            atr=atr,
            regime_multiplier=regime_mult * self.circuit_breaker.get_size_multiplier(),
        )

        quantity = sizing["quantity"]
        if quantity <= 0:
            return {"status": "skipped", "reason": "Position too small", "sizing": sizing}

        # ── Pre-trade risk check ──
        import pandas as pd
        positions_df = current_positions if isinstance(current_positions, pd.DataFrame) else pd.DataFrame()
        sector = signal.get("sector", "")

        risk_check = self.risk_manager.pre_trade_check(
            symbol, price, quantity, direction, capital, positions_df, sector
        )

        if not risk_check["passed"]:
            return {"status": "blocked", "reason": "Risk check failed", "checks": risk_check}

        # ── Smart order routing ──
        trade_id = str(uuid.uuid4())[:8]
        order_result = self._smart_order(
            symbol, instrument_key, quantity, direction, price
        )

        if order_result.get("status") != "success":
            return {"status": "error", "reason": order_result.get("message", "Order failed")}

        order_id = order_result.get("order_id", "")

        # ── Place GTT stop loss and take profit ──
        gtt_results = {}
        if stop_loss > 0:
            sl_side = "SELL" if direction == "BUY" else "BUY"
            sl_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=stop_loss,
                limit_price=stop_loss * (0.995 if direction == "BUY" else 1.005),
                quantity=quantity,
                side=sl_side,
            )
            gtt_results["stop_loss"] = sl_result

        if take_profit > 0:
            tp_side = "SELL" if direction == "BUY" else "BUY"
            tp_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=take_profit,
                limit_price=take_profit,
                quantity=quantity,
                side=tp_side,
            )
            gtt_results["take_profit"] = tp_result

        # ── Track order ──
        trade_record = {
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": symbol,
            "instrument_key": instrument_key,
            "side": direction,
            "quantity": quantity,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategy": signal.get("strategy", ""),
            "regime": signal.get("regime", ""),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "gtt_orders": gtt_results,
            "costs": self.risk_manager.calculate_trade_costs(price, quantity, direction, self.product),
        }

        self._pending_orders[trade_id] = trade_record

        logger.info(
            f"EXECUTED: {direction} {quantity} {symbol} @ ₹{price:.2f} "
            f"| SL=₹{stop_loss:.2f}, TP=₹{take_profit:.2f} "
            f"| trade_id={trade_id}"
        )

        return {"status": "success", **trade_record}

    def _execute_options_signal(
        self,
        signal: dict[str, Any],
        capital: float,
        current_positions: Any,
    ) -> dict[str, Any]:
        """
        Execute an options buying signal.

        Differences from equity:
        - Fixed ₹25K deployment cap (capital param kept for API compat)
        - Lot qty pre-computed by strategy (dynamic 1-7 lots)
        - Premium-based SL/TP (VIX-adaptive)
        - Product always "I" (intraday)
        - Uses SL-LIMIT (not SL-M) for options on Upstox
        """
        _ = capital  # Sizing uses fixed ₹25K deployable cap from signal features
        features = signal.get("features", {})
        symbol = signal.get("symbol", "")
        instrument_key = features.get("instrument_key", signal.get("instrument_key", ""))
        lot_size = features.get("lot_size", 65)
        premium_sl_pct = features.get("premium_sl_pct", 0.30)
        premium_tp_pct = features.get("premium_tp_pct", 0.60)
        max_deployable = features.get("max_deployable", 25000)
        max_risk = features.get("max_risk_per_trade", 10000)
        trade_type = features.get("trade_type", "FULL")

        if not instrument_key:
            return {"status": "skipped", "reason": "No options instrument key"}

        # ── Get live premium (price) ──
        premium = signal.get("price", 0)
        if premium <= 0:
            logger.warning(f"Options: No premium available for {symbol}, fetching LTP...")
            try:
                ltp_result = self.broker.get_ltp(instrument_key)
                premium = ltp_result.get("ltp", 0)
            except Exception as e:
                logger.error(f"Failed to get options LTP: {e}")
                return {"status": "error", "reason": f"Cannot get premium: {e}"}

        if premium <= 0:
            return {"status": "skipped", "reason": "Premium is zero"}

        # ── Premium range validation ──
        min_prem = features.get("min_premium", 0)
        max_prem = features.get("max_premium", 99999)
        if min_prem > 0 and premium < min_prem:
            return {"status": "skipped", "reason": f"Premium ₹{premium:.0f} below min ₹{min_prem:.0f}"}
        if max_prem < 99999 and premium > max_prem:
            return {"status": "skipped", "reason": f"Premium ₹{premium:.0f} above max ₹{max_prem:.0f}"}

        # ── Hard-cap position sizing (₹25K max deployment) ──
        quantity = lot_size  # Already computed by strategy (65 or 32)
        position_cost = premium * quantity

        # If too expensive, try half lot
        if position_cost > max_deployable:
            quantity = max(1, quantity // 2)
            position_cost = premium * quantity
        if position_cost > max_deployable:
            return {"status": "skipped", "reason": f"Position ₹{position_cost:.0f} > cap ₹{max_deployable:.0f}"}

        # Risk check: SL loss must not exceed max risk
        max_loss_at_sl = premium * premium_sl_pct * quantity
        if max_loss_at_sl > max_risk:
            return {"status": "skipped", "reason": f"Risk ₹{max_loss_at_sl:.0f} > max ₹{max_risk:.0f}"}

        lots = 1

        # ── Premium-based SL/TP ──
        stops = self.risk_manager.calculate_options_stops(
            entry_premium=premium,
            sl_pct=premium_sl_pct,
            tp_pct=premium_tp_pct,
        )
        stop_loss = stops["stop_loss"]
        take_profit = stops["take_profit"]

        # ── Place order (MARKET for options) ──
        trade_id = str(uuid.uuid4())[:8]
        order_result = self.broker.place_order(
            symbol=symbol,
            instrument_key=instrument_key,
            quantity=quantity,
            side="BUY",
            order_type="MARKET",
            price=premium,  # Pass premium for paper trader slippage calc
            product="I",  # Always intraday for options
        )

        if order_result.get("status") != "success":
            return {"status": "error", "reason": order_result.get("message", "Options order failed")}

        order_id = order_result.get("order_id", "")

        # ── Place GTT SL and TP (use SL-LIMIT for options, not SL-M) ──
        gtt_results = {}
        if stop_loss > 0:
            sl_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=stop_loss,
                limit_price=round(stop_loss * 0.95, 2),  # 5% below trigger for fill
                quantity=quantity,
                side="SELL",
            )
            gtt_results["stop_loss"] = sl_result

        if take_profit > 0:
            tp_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=take_profit,
                limit_price=take_profit,
                quantity=quantity,
                side="SELL",
            )
            gtt_results["take_profit"] = tp_result

        # ── Track order ──
        # ── Trailing stop params (from strategy signal) ──
        trail_trigger_pct = features.get("trail_trigger_pct", 0.20)
        trail_exit_pct = features.get("trail_exit_pct", 0.12)
        trail_trigger_price = round(premium * (1 + trail_trigger_pct), 2)
        trail_exit_price = round(premium * (1 + trail_exit_pct), 2)

        trade_record = {
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": symbol,
            "instrument_key": instrument_key,
            "side": "BUY",
            "quantity": quantity,
            "lots": lots,
            "price": premium,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trail_trigger": trail_trigger_price,
            "trail_exit": trail_exit_price,
            "trail_activated": False,
            "strategy": signal.get("strategy", "options_buyer"),
            "regime": signal.get("regime", ""),
            "confidence": signal.get("confidence", 0),
            "timestamp": datetime.now().isoformat(),
            "gtt_orders": gtt_results,
            "costs": self.risk_manager.calculate_options_trade_costs(premium, quantity, "BUY"),
            "is_options": True,
            "option_type": features.get("option_type", ""),
            "strike": features.get("strike", 0),
            "expiry": features.get("expiry", ""),
            "index_symbol": features.get("index_symbol", ""),
        }

        self._pending_orders[trade_id] = trade_record

        logger.info(
            f"OPTIONS EXECUTED: {trade_type} BUY {lots}L×{lot_size} {symbol} @ ₹{premium:.2f} "
            f"| SL=₹{stop_loss:.2f}, TP=₹{take_profit:.2f} "
            f"| value=₹{quantity * premium:.0f} | trade_id={trade_id}"
        )

        return {"status": "success", **trade_record}

    def _smart_order(
        self,
        symbol: str,
        instrument_key: str,
        quantity: int,
        side: str,
        price: float,
    ) -> dict[str, Any]:
        """
        Smart order routing: LIMIT → MARKET escalation.

        1. Try LIMIT at current price
        2. If not filled in 5 sec → modify to aggressive LIMIT (±0.1%)
        3. If still not filled → convert to MARKET
        """
        # For now, use MARKET orders for simplicity and guaranteed fills
        # In production, implement the LIMIT → MARKET escalation
        return self.broker.place_order(
            symbol=symbol,
            instrument_key=instrument_key,
            quantity=quantity,
            side=side,
            order_type="MARKET",
            product=self.product,
        )

    def check_eod_squareoff(self) -> list[dict]:
        """
        Check if it's time for EOD square-off (3:15 PM for intraday).
        """
        now = datetime.now().time()
        sq_time = dt_time(*[int(x) for x in self.square_off_time.split(":")])

        if now < sq_time:
            return []

        if self.product != "I":
            return []

        logger.info("EOD SQUARE-OFF: Closing all intraday positions")
        result = self.broker.square_off_all()

        return [result]

    def check_trailing_stops(self) -> list[dict[str, Any]]:
        """
        Monitor open options positions for trailing stop activation.

        Logic: If premium hits +20% (trail_trigger), then falls back below +12%
        (trail_exit), exit the position by selling at market and cancelling GTTs.

        Should be called every tick/minute during market hours.
        """
        exits = []
        for trade_id, trade in list(self._pending_orders.items()):
            if not trade.get("is_options"):
                continue

            instrument_key = trade.get("instrument_key", "")
            if not instrument_key:
                continue

            try:
                ltp_result = self.broker.get_ltp(instrument_key)
                current_premium = ltp_result.get("ltp", 0)
            except Exception:
                continue

            if current_premium <= 0:
                continue

            trail_trigger = trade.get("trail_trigger", 0)
            trail_exit = trade.get("trail_exit", 0)
            if trail_trigger <= 0 or trail_exit <= 0:
                continue

            # Check if premium has ever hit the trigger level
            if current_premium >= trail_trigger:
                trade["trail_activated"] = True

            # If trail was activated and premium fell back below exit level → exit
            if trade.get("trail_activated") and current_premium <= trail_exit:
                logger.info(
                    f"TRAILING STOP: {trade['symbol']} — premium hit "
                    f"₹{trail_trigger:.2f} then fell to ₹{current_premium:.2f} "
                    f"(exit at ₹{trail_exit:.2f}) | entry=₹{trade['price']:.2f}"
                )

                # Cancel existing GTT orders
                for gtt_type, gtt_info in trade.get("gtt_orders", {}).items():
                    gtt_id = gtt_info.get("gtt_id", "")
                    if gtt_id:
                        try:
                            self.broker.cancel_gtt_order(gtt_id)
                        except Exception as e:
                            logger.warning(f"Failed to cancel GTT {gtt_id}: {e}")

                # Place market sell to exit
                try:
                    exit_result = self.broker.place_order(
                        symbol=trade["symbol"],
                        instrument_key=instrument_key,
                        quantity=trade["quantity"],
                        side="SELL",
                        order_type="MARKET",
                        product="I",
                    )
                    exits.append({
                        "trade_id": trade_id,
                        "symbol": trade["symbol"],
                        "exit_reason": "trail_stop",
                        "entry_premium": trade["price"],
                        "exit_premium": current_premium,
                        "pnl_pct": (current_premium - trade["price"]) / trade["price"] * 100,
                        "exit_result": exit_result,
                    })
                    del self._pending_orders[trade_id]
                except Exception as e:
                    logger.error(f"Failed trailing stop exit for {trade['symbol']}: {e}")

        return exits

    def execute_kill_switch(self) -> dict[str, Any]:
        """
        Emergency kill switch: cancel all orders + flatten all positions.
        """
        logger.critical("KILL SWITCH: Executing emergency shutdown")

        cancel_result = self.broker.cancel_all_orders()
        flatten_result = self.broker.square_off_all()

        # Cancel GTT orders
        for gtt_id, gtt_info in self._gtt_orders.items():
            self.broker.cancel_gtt_order(gtt_id)

        self._gtt_orders.clear()

        return {
            "cancelled_orders": cancel_result,
            "flattened_positions": flatten_result,
            "timestamp": datetime.now().isoformat(),
        }

    def reconcile_orders(self) -> dict[str, Any]:
        """
        Reconcile pending orders with broker order book.
        """
        order_book = self.broker.get_order_book()
        reconciled = 0

        for order in order_book:
            order_id = order.get("order_id", "")
            status = order.get("status", "")

            # Find matching pending order
            for trade_id, trade in list(self._pending_orders.items()):
                if trade.get("order_id") == order_id:
                    if status in ("complete", "filled"):
                        trade["status"] = "filled"
                        trade["fill_price"] = order.get("average_price", 0)
                        trade["fill_quantity"] = order.get("filled_quantity", 0)
                        self._filled_orders.append(trade)
                        del self._pending_orders[trade_id]
                        reconciled += 1

                    elif status in ("rejected", "cancelled"):
                        trade["status"] = status
                        del self._pending_orders[trade_id]
                        reconciled += 1

        return {"reconciled": reconciled, "pending": len(self._pending_orders)}

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    @property
    def filled_today(self) -> list[dict]:
        return self._filled_orders
