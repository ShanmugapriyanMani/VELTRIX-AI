"""
Order Manager — Smart routing, GTT stop/target, EOD square-off, kill switch.

Sits between strategy signals and the broker, handling order lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time as dt_time
from typing import Any, Optional

from loguru import logger

from src.config.env_loader import get_config
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

        self.square_off_time = get_config().SQUARE_OFF_TIME
        self.product = "I"  # Default intraday

        # Active orders tracking
        self._pending_orders: dict[str, dict] = {}
        self._gtt_orders: dict[str, dict] = {}
        self._filled_orders: list[dict] = []

        # Alert function (injected by main.py)
        self._alert_fn: Optional[callable] = None
        # Spread LTP error tracking (H3)
        self._spread_ltp_errors: dict[str, int] = {}

    def set_alert_fn(self, fn) -> None:
        """Inject Telegram alert function."""
        self._alert_fn = fn

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day."""
        self._pending_orders = {}
        self._gtt_orders = {}
        self._filled_orders = []
        self._spread_ltp_errors = {}
        logger.info("ORDER_MANAGER_RESET: daily state cleared")

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
            if features.get("is_iron_condor", False):
                return self._execute_iron_condor_signal(signal, capital, current_positions)
            if features.get("is_spread", False):
                return self._execute_spread_signal(signal, capital, current_positions)
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

        # ── Circuit breaker size adjustment ──
        cb_mult = self.circuit_breaker.get_size_multiplier()
        if cb_mult <= 0:
            return {"status": "skipped", "reason": "Circuit breaker halted trading"}

        # ── Hard-cap position sizing (₹25K max deployment) ──
        quantity = lot_size  # Already computed by strategy (65 or 32)
        if cb_mult < 1.0:
            quantity = max(1, int(quantity * cb_mult))
            logger.warning(f"Options: Circuit breaker reduced lots {lot_size} → {quantity}")
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

        # ── Live mode: wait for fill confirmation before proceeding ──
        fill_price = premium  # Default: signal price (paper mode)
        if hasattr(self.broker, "wait_for_fill"):
            fill_result = self.broker.wait_for_fill(order_id, timeout_seconds=60)
            if not fill_result.get("filled"):
                reason = fill_result.get("reason", "unknown")
                logger.warning(
                    f"ENTRY_FAILED: {symbol} order_id={order_id} reason={reason}"
                )
                return {
                    "status": "rejected",
                    "reason": f"Order not filled: {reason}",
                    "order_id": order_id,
                }
            # Use actual fill price from broker
            if fill_result.get("avg_price", 0) > 0:
                fill_price = fill_result["avg_price"]
                if fill_result.get("filled_qty", 0) > 0:
                    quantity = fill_result["filled_qty"]
            slippage_pct = abs(fill_price - premium) / premium if premium > 0 else 0
            logger.info(
                f"FILL_CONFIRMED: {symbol} signal=₹{premium:.2f} fill=₹{fill_price:.2f} "
                f"slippage={slippage_pct:.3%}"
            )

        # ── Place GTT SL and TP (use SL-LIMIT for options, not SL-M) ──
        gtt_results = {}
        gtt_failed = False
        if stop_loss > 0:
            sl_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=stop_loss,
                limit_price=round(stop_loss * 0.95, 2),  # 5% below trigger for fill
                quantity=quantity,
                side="SELL",
            )
            gtt_results["stop_loss"] = sl_result
            if not sl_result or sl_result.get("status") == "error":
                logger.error(f"GTT_SL_FAILED: {symbol} SL=₹{stop_loss:.2f} result={sl_result}")
                gtt_failed = True

        if take_profit > 0:
            tp_result = self.broker.place_gtt_order(
                instrument_key=instrument_key,
                trigger_price=take_profit,
                limit_price=take_profit,
                quantity=quantity,
                side="SELL",
            )
            gtt_results["take_profit"] = tp_result
            if not tp_result or tp_result.get("status") == "error":
                logger.error(f"GTT_TP_FAILED: {symbol} TP=₹{take_profit:.2f} result={tp_result}")
                gtt_failed = True

        if gtt_failed:
            # In live mode: auto-close unprotected position for safety
            # In paper mode: in-memory stops still work, just alert
            from src.execution.paper_trader import PaperTrader
            is_live_broker = not isinstance(self.broker, PaperTrader)
            if is_live_broker:
                if self._alert_fn:
                    self._alert_fn(
                        f"🚨 GTT FAILED: {symbol}\n"
                        f"SL/TP not placed. Auto-closing position for safety."
                    )
                try:
                    close_result = self.broker.place_order(
                        symbol=symbol,
                        instrument_key=instrument_key,
                        quantity=quantity,
                        side="SELL",
                        order_type="MARKET",
                        product="I",
                    )
                    logger.warning(
                        f"GTT_AUTOCLOSE: {symbol} qty={quantity} "
                        f"close_result={close_result.get('status')}"
                    )
                    return {
                        "status": "error",
                        "reason": "GTT failed, auto-closed position",
                        "close_result": close_result,
                    }
                except Exception as e:
                    logger.critical(
                        f"GTT_AUTOCLOSE_FAILED: {symbol} — MANUAL INTERVENTION NEEDED: {e}"
                    )
                    if self._alert_fn:
                        self._alert_fn(
                            f"💀 CRITICAL: GTT failed AND auto-close failed for {symbol}\n"
                            f"Error: {e}\n"
                            f"MANUAL INTERVENTION REQUIRED"
                        )
            else:
                # Paper mode: alert only, in-memory stops active
                if self._alert_fn:
                    self._alert_fn(
                        f"🚨 GTT FAILED: {symbol}\n"
                        f"SL/TP not placed at broker.\n"
                        f"In-memory stops still active."
                    )

        # ── Track order ──
        # ── Trailing stop params (from strategy signal) ──
        trail_trigger_pct = features.get("trail_trigger_pct", 0.20)
        trail_exit_pct = features.get("trail_exit_pct", 0.12)
        trail_trigger_price = round(premium * (1 + trail_trigger_pct), 2)
        trail_exit_price = round(premium * (1 + trail_exit_pct), 2)

        # Partial profit TP1 target
        tp1_pct = features.get("tp1_pct", 0)
        tp1_price = round(premium * (1 + tp1_pct), 2) if tp1_pct > 0 else 0

        # Slippage tracking: signal_price = what strategy computed, fill_price = what broker filled
        slippage = abs(fill_price - premium) / premium if premium > 0 else 0

        trade_record = {
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": symbol,
            "instrument_key": instrument_key,
            "side": "BUY",
            "quantity": quantity,
            "lots": lots,
            "price": fill_price,  # Actual fill price (paper: premium+slippage, live: broker fill)
            "signal_price": premium,
            "fill_price": fill_price,
            "fill_quantity": quantity,
            "slippage_pct": round(slippage, 6),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trail_trigger": trail_trigger_price,
            "trail_exit": trail_exit_price,
            "trail_activated": False,
            "tp1_price": tp1_price,
            "partial_exit_done": False,
            "strategy": signal.get("strategy", "options_buyer"),
            "regime": signal.get("regime", ""),
            "confidence": signal.get("confidence", 0),
            "timestamp": datetime.now().isoformat(),
            "entry_time": datetime.now().isoformat(),
            "status": "open",
            "gtt_orders": gtt_results,
            "costs": self.risk_manager.calculate_options_trade_costs(fill_price, quantity, "BUY"),
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

        return {**trade_record, "status": "success"}

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
                current_premium = ltp_result.get("ltp", 0) if ltp_result else 0
            except Exception:
                continue

            if current_premium is None or current_premium <= 0:
                logger.warning(f"TRAIL_SKIP: invalid price {current_premium} for {trade.get('symbol', '')}")
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
                        "instrument_key": instrument_key,
                        "exit_reason": "trail_stop",
                        "entry_premium": trade["price"],
                        "exit_premium": current_premium,
                        "quantity": trade["quantity"],
                        "pnl": (current_premium - trade["price"]) * trade["quantity"],
                        "pnl_pct": (current_premium - trade["price"]) / trade["price"] * 100,
                        "exit_result": exit_result,
                    })
                    del self._pending_orders[trade_id]
                except Exception as e:
                    logger.error(f"Failed trailing stop exit for {trade['symbol']}: {e}")

        return exits

    def check_tp1_exits(self, portfolio_positions: dict) -> list[dict[str, Any]]:
        """
        Check if any position has hit TP1 for partial profit exit.

        Uses portfolio current_price (already updated by fast poll) — no extra LTP fetch.
        Returns list of partial exit info dicts.
        """
        exits = []
        for trade_id, trade in list(self._pending_orders.items()):
            if not trade.get("is_options") or trade.get("partial_exit_done"):
                continue

            tp1_price = trade.get("tp1_price", 0)
            if tp1_price <= 0:
                continue

            symbol = trade["symbol"]
            pos = portfolio_positions.get(symbol)
            if not pos or pos.partial_exit_done:
                continue

            current_premium = pos.current_price
            if current_premium <= 0 or current_premium < tp1_price:
                continue

            exit_qty = pos.original_quantity // 2
            if exit_qty <= 0:
                continue

            instrument_key = trade.get("instrument_key", "")
            logger.info(
                f"TP1_PARTIAL: {symbol} hit ₹{current_premium:.2f} >= TP1 ₹{tp1_price:.2f} "
                f"| exiting {exit_qty}qty of {pos.original_quantity}"
            )

            # Place SELL for partial qty
            try:
                self.broker.place_order(
                    symbol=symbol,
                    instrument_key=instrument_key,
                    quantity=exit_qty,
                    side="SELL",
                    order_type="MARKET",
                    product="I",
                )
            except Exception as e:
                logger.error(f"TP1 partial sell failed for {symbol}: {e}")
                continue

            # Cancel full-qty TP GTT
            tp_gtt = trade.get("gtt_orders", {}).get("take_profit", {})
            if tp_gtt.get("gtt_id"):
                try:
                    self.broker.cancel_gtt_order(tp_gtt["gtt_id"])
                except Exception as e:
                    logger.warning(f"Failed to cancel TP GTT: {e}")

            # Cancel full-qty SL GTT, re-place at breakeven for remaining
            sl_gtt = trade.get("gtt_orders", {}).get("stop_loss", {})
            if sl_gtt.get("gtt_id"):
                try:
                    self.broker.cancel_gtt_order(sl_gtt["gtt_id"])
                except Exception as e:
                    logger.warning(f"Failed to cancel SL GTT: {e}")

            remaining_qty = pos.quantity - exit_qty
            if remaining_qty > 0:
                new_sl = self.broker.place_gtt_order(
                    instrument_key=instrument_key,
                    trigger_price=trade["price"],  # Breakeven
                    limit_price=round(trade["price"] * 0.95, 2),
                    quantity=remaining_qty,
                    side="SELL",
                )
                trade["gtt_orders"] = {"stop_loss": new_sl}
            else:
                trade["gtt_orders"] = {}

            trade["partial_exit_done"] = True
            trade["quantity"] = remaining_qty

            exits.append({
                "trade_id": trade_id,
                "symbol": symbol,
                "exit_reason": "tp1_partial",
                "entry_premium": trade["price"],
                "exit_premium": current_premium,
                "quantity": exit_qty,
                "remaining_quantity": remaining_qty,
                "pnl": (current_premium - trade["price"]) * exit_qty,
                "pnl_pct": (current_premium - trade["price"]) / trade["price"] * 100,
                "option_type": trade.get("option_type", ""),
            })

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

                    elif status in ("open", "trigger_pending", "modified"):
                        # Still active — keep in pending, no action needed
                        pass

                    elif status:
                        logger.warning(
                            f"OrderManager: Unknown order status '{status}' "
                            f"for order {order_id}"
                        )

        return {"reconciled": reconciled, "pending": len(self._pending_orders)}

    # ──────────────────────────────────────────
    # PLUS Spread Execution
    # ──────────────────────────────────────────

    def _execute_spread_signal(
        self,
        signal: dict[str, Any],
        capital: float,
        current_positions: Any,
    ) -> dict[str, Any]:
        """Execute a two-leg spread order with atomic rollback on failure."""
        features = signal.get("features", {})
        trade_type = features.get("trade_type", "")
        qty = features.get("lot_size", 0)

        # Place leg 1
        leg1_result = self.broker.place_order(
            symbol=signal.get("symbol", ""),
            instrument_key=features["leg1_instrument_key"],
            quantity=qty,
            side=features["leg1_side"],
            order_type="MARKET",
            price=features["leg1_premium"],
            product="I",
        )
        if leg1_result.get("status") != "success":
            return {"status": "error", "reason": "Leg 1 failed", "detail": leg1_result}

        # Place leg 2
        idx = features.get("index_symbol", "")
        leg2_sym = f"{idx}{int(features['leg2_strike'])}{features['option_type']}"
        leg2_result = self.broker.place_order(
            symbol=leg2_sym,
            instrument_key=features["leg2_instrument_key"],
            quantity=qty,
            side=features["leg2_side"],
            order_type="MARKET",
            price=features["leg2_premium"],
            product="I",
        )
        if leg2_result.get("status") != "success":
            # CRITICAL: Unwind leg 1 immediately
            logger.critical(f"SPREAD LEG 2 FAILED — unwinding leg 1")
            unwind_side = "SELL" if features["leg1_side"] == "BUY" else "BUY"
            self.broker.place_order(
                symbol=signal.get("symbol", ""),
                instrument_key=features["leg1_instrument_key"],
                quantity=qty,
                side=unwind_side,
                order_type="MARKET",
                price=features["leg1_premium"],
                product="I",
            )
            return {"status": "error", "reason": "Leg 2 failed, leg 1 unwound"}

        # Track spread as a single unit
        trade_id = str(uuid.uuid4())[:8]
        trade_record = {
            "trade_id": trade_id,
            "order_id_leg1": leg1_result.get("order_id", ""),
            "order_id_leg2": leg2_result.get("order_id", ""),
            "symbol": signal.get("symbol", ""),
            "leg2_symbol": leg2_sym,
            "trade_type": trade_type,
            "is_spread": True,
            "is_options": True,
            "leg1_instrument_key": features["leg1_instrument_key"],
            "leg1_side": features["leg1_side"],
            "leg1_premium": features["leg1_premium"],
            "leg2_instrument_key": features["leg2_instrument_key"],
            "leg2_side": features["leg2_side"],
            "leg2_premium": features["leg2_premium"],
            "quantity": qty,
            "net_premium": features.get("net_premium", 0),
            "max_profit": features.get("max_profit", 0),
            "max_loss": features.get("max_loss", 0),
            "spread_width": features.get("spread_width", 0),
            "strategy": signal.get("strategy", "options_buyer"),
            "regime": signal.get("regime", ""),
            "timestamp": datetime.now().isoformat(),
        }
        self._pending_orders[trade_id] = trade_record

        logger.info(
            f"SPREAD EXECUTED: {trade_type} | "
            f"{features['leg1_side']} {features['leg1_strike']}@₹{features['leg1_premium']:.0f} + "
            f"{features['leg2_side']} {features['leg2_strike']}@₹{features['leg2_premium']:.0f} | "
            f"net=₹{features.get('net_premium', 0):.0f} qty={qty} | id={trade_id}"
        )
        return {"status": "success", **trade_record}

    def check_spread_exits(self) -> list[dict[str, Any]]:
        """Monitor open spreads for SL/TP exit conditions.

        Both legs ALWAYS exit together.
        """
        exits = []
        cfg = get_config()

        for trade_id, trade in list(self._pending_orders.items()):
            if not trade.get("is_spread"):
                continue

            trade_type = trade.get("trade_type", "")
            qty = trade.get("quantity", 0)
            entry_net = trade.get("net_premium", 0)

            # Get current premiums
            try:
                ltp1 = self.broker.get_ltp(trade["leg1_instrument_key"]).get("ltp", 0)
                ltp2 = self.broker.get_ltp(trade["leg2_instrument_key"]).get("ltp", 0)
            except Exception as e:
                spread_sym = trade.get("symbol", trade_id)
                self._spread_ltp_errors[spread_sym] = self._spread_ltp_errors.get(spread_sym, 0) + 1
                _err_count = self._spread_ltp_errors[spread_sym]
                logger.error(f"SPREAD_LTP_ERROR: {spread_sym} (#{_err_count}) {e}")
                if _err_count == 5 and self._alert_fn:
                    self._alert_fn(
                        f"⚠️ SPREAD MONITORING FAILED: {spread_sym}\n"
                        f"5 consecutive LTP errors.\n"
                        f"Position may be unmonitored."
                    )
                continue
            if ltp1 <= 0 or ltp2 <= 0:
                continue

            should_exit = False
            exit_reason = ""
            spread_pnl = 0.0

            if trade_type == "DEBIT_SPREAD":
                # BUY leg 1, SELL leg 2 — spread value = ltp1 - ltp2
                current_net = ltp1 - ltp2
                spread_pnl = (current_net - entry_net) * qty
                sl_threshold = -entry_net * (cfg.DEBIT_SPREAD_SL_PCT / 100) * qty
                tp_threshold = trade["max_profit"] * (cfg.DEBIT_SPREAD_TP_PCT / 100)
                if spread_pnl <= sl_threshold:
                    should_exit, exit_reason = True, "spread_sl"
                elif spread_pnl >= tp_threshold:
                    should_exit, exit_reason = True, "spread_tp"

            elif trade_type == "CREDIT_SPREAD":
                # SELL leg 1, BUY leg 2 — P&L = credit - cost_to_close
                close_cost = ltp1 - ltp2  # Cost to buy back sell leg
                spread_pnl = (entry_net - close_cost) * qty
                sl_threshold = -entry_net * cfg.CREDIT_SPREAD_SL_MULTIPLIER * qty
                tp_threshold = entry_net * (cfg.CREDIT_SPREAD_TP_PCT / 100) * qty
                if spread_pnl <= sl_threshold:
                    should_exit, exit_reason = True, "spread_sl"
                elif spread_pnl >= tp_threshold:
                    should_exit, exit_reason = True, "spread_tp"

            if should_exit:
                self._exit_spread(trade_id, trade, exit_reason, ltp1, ltp2)
                exits.append({
                    "trade_id": trade_id,
                    "trade_type": trade_type,
                    "exit_reason": exit_reason,
                    "pnl": spread_pnl,
                })

        return exits

    def _exit_spread(
        self, trade_id: str, trade: dict, reason: str, ltp1: float, ltp2: float,
    ) -> None:
        """Close both legs of a spread position atomically."""
        # Unwind leg 1
        unwind1 = "SELL" if trade["leg1_side"] == "BUY" else "BUY"
        self.broker.place_order(
            symbol=trade["symbol"],
            instrument_key=trade["leg1_instrument_key"],
            quantity=trade["quantity"],
            side=unwind1,
            order_type="MARKET",
            price=ltp1,
            product="I",
        )
        # Unwind leg 2
        unwind2 = "SELL" if trade["leg2_side"] == "BUY" else "BUY"
        self.broker.place_order(
            symbol=trade.get("leg2_symbol", trade["symbol"]),
            instrument_key=trade["leg2_instrument_key"],
            quantity=trade["quantity"],
            side=unwind2,
            order_type="MARKET",
            price=ltp2,
            product="I",
        )

        if trade_id in self._pending_orders:
            del self._pending_orders[trade_id]
        logger.info(f"SPREAD EXIT: {trade['trade_type']} {reason} | id={trade_id}")

    # ──────────────────────────────────────────
    # PLUS Iron Condor Execution
    # ──────────────────────────────────────────

    def _execute_iron_condor_signal(
        self,
        signal: dict[str, Any],
        capital: float,
        current_positions: Any,
    ) -> dict[str, Any]:
        """Execute a 4-leg Iron Condor with cascading rollback on failure."""
        features = signal.get("features", {})
        qty = features.get("quantity", 0)

        legs_placed: list[dict] = []

        def _unwind_all():
            for leg in legs_placed:
                unwind_side = "SELL" if leg["side"] == "BUY" else "BUY"
                self.broker.place_order(
                    symbol=leg["symbol"],
                    instrument_key=leg["instrument_key"],
                    quantity=qty, side=unwind_side,
                    order_type="MARKET", price=leg["price"], product="I",
                )

        leg_specs = [
            ("sell_ce", "SELL", "CE"),
            ("buy_ce", "BUY", "CE"),
            ("sell_pe", "SELL", "PE"),
            ("buy_pe", "BUY", "PE"),
        ]

        for prefix, side, opt_type in leg_specs:
            strike = features[f"{prefix}_strike"]
            premium = features[f"{prefix}_premium"]
            inst_key = features.get(f"{prefix}_instrument_key", "")
            sym = f"NIFTY{int(strike)}{opt_type}"

            result = self.broker.place_order(
                symbol=sym, instrument_key=inst_key,
                quantity=qty, side=side,
                order_type="MARKET", price=premium, product="I",
            )
            if result.get("status") != "success":
                logger.critical(f"IC LEG FAILED: {prefix} — unwinding {len(legs_placed)} legs")
                _unwind_all()
                return {"status": "error", "reason": f"IC leg {prefix} failed, {len(legs_placed)} legs unwound"}

            legs_placed.append({
                "symbol": sym, "instrument_key": inst_key,
                "side": side, "price": premium,
                "order_id": result.get("order_id", ""),
            })

        # Track IC as a single unit
        trade_id = str(uuid.uuid4())[:8]
        trade_record = {
            "trade_id": trade_id,
            "is_iron_condor": True,
            "is_spread": True,
            "is_options": True,
            "trade_type": "IRON_CONDOR",
            "quantity": qty,
            "sell_ce_instrument_key": features.get("sell_ce_instrument_key", ""),
            "sell_ce_premium": features["sell_ce_premium"],
            "sell_ce_strike": features["sell_ce_strike"],
            "buy_ce_instrument_key": features.get("buy_ce_instrument_key", ""),
            "buy_ce_premium": features["buy_ce_premium"],
            "buy_ce_strike": features["buy_ce_strike"],
            "sell_pe_instrument_key": features.get("sell_pe_instrument_key", ""),
            "sell_pe_premium": features["sell_pe_premium"],
            "sell_pe_strike": features["sell_pe_strike"],
            "buy_pe_instrument_key": features.get("buy_pe_instrument_key", ""),
            "buy_pe_premium": features["buy_pe_premium"],
            "buy_pe_strike": features["buy_pe_strike"],
            "net_credit": features.get("net_credit", 0),
            "max_profit": features.get("max_profit", 0),
            "max_loss": features.get("max_loss", 0),
            "spread_width": features.get("spread_width", 0),
            "regime": signal.get("regime", ""),
            "timestamp": datetime.now().isoformat(),
        }
        self._pending_orders[trade_id] = trade_record

        logger.info(
            f"IC EXECUTED: SELL {features['sell_ce_strike']}CE + BUY {features['buy_ce_strike']}CE | "
            f"SELL {features['sell_pe_strike']}PE + BUY {features['buy_pe_strike']}PE | "
            f"credit=₹{features.get('net_credit', 0):.0f} qty={qty} | id={trade_id}"
        )
        return {"status": "success", **trade_record}

    def check_ic_exits(self) -> list[dict[str, Any]]:
        """Monitor open Iron Condor positions for SL/TP exit conditions."""
        exits = []
        cfg = get_config()

        for trade_id, trade in list(self._pending_orders.items()):
            if not trade.get("is_iron_condor"):
                continue

            qty = trade.get("quantity", 0)
            net_credit = trade.get("net_credit", 0)

            try:
                ltp_sell_ce = self.broker.get_ltp(trade["sell_ce_instrument_key"]).get("ltp", 0)
                ltp_buy_ce = self.broker.get_ltp(trade["buy_ce_instrument_key"]).get("ltp", 0)
                ltp_sell_pe = self.broker.get_ltp(trade["sell_pe_instrument_key"]).get("ltp", 0)
                ltp_buy_pe = self.broker.get_ltp(trade["buy_pe_instrument_key"]).get("ltp", 0)
            except Exception:
                continue
            if any(v <= 0 for v in [ltp_sell_ce, ltp_buy_ce, ltp_sell_pe, ltp_buy_pe]):
                continue

            # IC P&L = (credit - close_cost) * qty
            close_cost_ce = ltp_sell_ce - ltp_buy_ce
            close_cost_pe = ltp_sell_pe - ltp_buy_pe
            total_close_cost = close_cost_ce + close_cost_pe
            ic_pnl = (net_credit - total_close_cost) * qty

            sl_threshold = -net_credit * cfg.IC_SL_MULTIPLIER * qty
            tp_threshold = net_credit * (cfg.IC_TP_PCT / 100) * qty

            should_exit = False
            exit_reason = ""
            if ic_pnl <= sl_threshold:
                should_exit, exit_reason = True, "ic_sl"
            elif ic_pnl >= tp_threshold:
                should_exit, exit_reason = True, "ic_tp"

            if should_exit:
                ltp_dict = {
                    trade["sell_ce_instrument_key"]: ltp_sell_ce,
                    trade["buy_ce_instrument_key"]: ltp_buy_ce,
                    trade["sell_pe_instrument_key"]: ltp_sell_pe,
                    trade["buy_pe_instrument_key"]: ltp_buy_pe,
                }
                self._exit_iron_condor(trade_id, trade, exit_reason, ltp_dict)
                exits.append({
                    "trade_id": trade_id,
                    "trade_type": "IRON_CONDOR",
                    "exit_reason": exit_reason,
                    "pnl": ic_pnl,
                })

        return exits

    def _exit_iron_condor(
        self, trade_id: str, trade: dict, reason: str, ltp_dict: dict,
    ) -> None:
        """Close all 4 legs of an IC position atomically."""
        qty = trade.get("quantity", 0)
        leg_specs = [
            ("sell_ce", "BUY"),   # Buy back the sold CE
            ("buy_ce", "SELL"),   # Sell the bought CE
            ("sell_pe", "BUY"),   # Buy back the sold PE
            ("buy_pe", "SELL"),   # Sell the bought PE
        ]
        for prefix, unwind_side in leg_specs:
            inst_key = trade.get(f"{prefix}_instrument_key", "")
            strike = trade.get(f"{prefix}_strike", 0)
            opt_type = "CE" if "ce" in prefix else "PE"
            sym = f"NIFTY{int(strike)}{opt_type}"
            ltp = ltp_dict.get(inst_key, 0)
            self.broker.place_order(
                symbol=sym, instrument_key=inst_key,
                quantity=qty, side=unwind_side,
                order_type="MARKET", price=ltp, product="I",
            )

        if trade_id in self._pending_orders:
            del self._pending_orders[trade_id]
        logger.info(f"IC EXIT: IRON_CONDOR {reason} | id={trade_id}")

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    @property
    def filled_today(self) -> list[dict]:
        return self._filled_orders
