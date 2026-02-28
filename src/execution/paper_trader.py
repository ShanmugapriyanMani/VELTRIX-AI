"""
Paper Trader — Drop-in replacement for UpstoxBroker.

Uses real market data but simulates order fills.
Perfect for testing strategies without risking capital.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.execution.broker import BaseBroker


class PaperTrader(BaseBroker):
    """
    Simulated broker for paper trading.
    Implements the same interface as UpstoxBroker.
    Uses real-time data but simulates fills with configurable slippage.
    """

    def __init__(
        self,
        initial_capital: float = 500000,
        slippage_pct: float = 0.05,
        data_fetcher: Any = None,
    ):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.slippage_pct = slippage_pct / 100
        self._data_fetcher = data_fetcher

        # Simulated state
        self._orders: dict[str, dict] = {}
        self._positions: dict[str, dict] = {}
        self._gtt_orders: dict[str, dict] = {}
        self._order_counter = 0

        logger.info(
            f"Paper Trader initialized: ₹{initial_capital:,.0f} capital, "
            f"{slippage_pct}% slippage"
        )

    def connect(self) -> bool:
        logger.info("Paper Trader connected (simulated)")
        return True

    def place_order(
        self,
        symbol: str,
        instrument_key: str,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        price: float = 0,
        trigger_price: float = 0,
        product: str = "I",
        validity: str = "DAY",
    ) -> dict[str, Any]:
        """Simulate order placement with instant fill for MARKET orders."""
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        # Apply slippage
        if order_type == "MARKET":
            if side == "BUY":
                fill_price = price * (1 + self.slippage_pct)
            else:
                fill_price = price * (1 - self.slippage_pct)
        else:
            fill_price = price

        # Check funds
        trade_value = fill_price * quantity
        if side == "BUY" and trade_value > self.available_cash:
            return {
                "order_id": order_id,
                "status": "rejected",
                "message": f"Insufficient funds: need ₹{trade_value:.2f}, have ₹{self.available_cash:.2f}",
            }

        # Record order
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "instrument_key": instrument_key,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "price": price,
            "fill_price": fill_price,
            "trigger_price": trigger_price,
            "product": product,
            "status": "complete",
            "filled_quantity": quantity,
            "average_price": fill_price,
            "timestamp": datetime.now().isoformat(),
        }
        self._orders[order_id] = order

        # Update positions
        self._update_position(symbol, instrument_key, side, quantity, fill_price, product)

        logger.info(
            f"[PAPER] {side} {quantity} {symbol} @ ₹{fill_price:.2f} "
            f"(slip: ₹{abs(fill_price - price) * quantity:.2f}) | {order_id}"
        )

        return {"order_id": order_id, "status": "success", "message": "Paper order filled"}

    def _update_position(
        self,
        symbol: str,
        instrument_key: str,
        side: str,
        quantity: int,
        price: float,
        product: str,
    ) -> None:
        """Update position after a fill."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            if side == "BUY":
                new_qty = pos["quantity"] + quantity
                if new_qty != 0:
                    pos["average_price"] = (
                        (pos["average_price"] * pos["quantity"] + price * quantity)
                        / new_qty
                    )
                pos["quantity"] = new_qty
                self.available_cash -= price * quantity
            else:
                pnl = (price - pos["average_price"]) * quantity
                pos["quantity"] -= quantity
                pos["pnl"] = pos.get("pnl", 0) + pnl
                self.available_cash += price * quantity

            if pos["quantity"] == 0:
                del self._positions[symbol]
        else:
            if side == "BUY":
                self._positions[symbol] = {
                    "symbol": symbol,
                    "instrument_key": instrument_key,
                    "quantity": quantity,
                    "average_price": price,
                    "last_price": price,
                    "pnl": 0,
                    "product": product,
                }
                self.available_cash -= price * quantity
            else:
                self._positions[symbol] = {
                    "symbol": symbol,
                    "instrument_key": instrument_key,
                    "quantity": -quantity,
                    "average_price": price,
                    "last_price": price,
                    "pnl": 0,
                    "product": product,
                }
                self.available_cash += price * quantity

    def modify_order(self, order_id: str, **kwargs) -> dict[str, Any]:
        logger.info(f"[PAPER] Order modify: {order_id} (simulated)")
        return {"status": "success", "order_id": order_id}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
        logger.info(f"[PAPER] Order cancelled: {order_id}")
        return {"status": "success", "order_id": order_id}

    def cancel_all_orders(self) -> dict[str, Any]:
        cancelled = 0
        for oid, order in self._orders.items():
            if order["status"] in ("open", "pending"):
                order["status"] = "cancelled"
                cancelled += 1
        return {"cancelled": cancelled, "errors": 0}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return self._orders.get(order_id, {})

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "instrument_key": p["instrument_key"],
                "symbol": p["symbol"],
                "quantity": p["quantity"],
                "average_price": p["average_price"],
                "last_price": p["last_price"],
                "pnl": p["pnl"],
                "product": p["product"],
                "side": "BUY" if p["quantity"] > 0 else "SELL",
            }
            for p in self._positions.values()
        ]

    def get_holdings(self) -> list[dict[str, Any]]:
        return [
            p for p in self.get_positions()
            if p.get("product") == "D"
        ]

    def get_order_book(self) -> list[dict[str, Any]]:
        return list(self._orders.values())

    def get_funds(self) -> dict[str, Any]:
        invested = sum(
            abs(p["quantity"]) * p["average_price"]
            for p in self._positions.values()
        )
        return {
            "available_margin": round(self.available_cash, 2),
            "used_margin": round(invested, 2),
            "payin_amount": round(self.initial_capital, 2),
        }

    def place_gtt_order(
        self,
        instrument_key: str,
        trigger_price: float,
        limit_price: float,
        quantity: int,
        side: str,
    ) -> dict[str, Any]:
        gtt_id = f"GTT-{uuid.uuid4().hex[:8]}"
        self._gtt_orders[gtt_id] = {
            "gtt_id": gtt_id,
            "instrument_key": instrument_key,
            "trigger_price": trigger_price,
            "limit_price": limit_price,
            "quantity": quantity,
            "side": side,
            "status": "active",
        }
        logger.info(f"[PAPER] GTT placed: {side} {quantity} @ trigger ₹{trigger_price}")
        return {"gtt_id": gtt_id, "status": "success"}

    def cancel_gtt_order(self, gtt_id: str) -> dict[str, Any]:
        if gtt_id in self._gtt_orders:
            self._gtt_orders[gtt_id]["status"] = "cancelled"
        return {"status": "success"}

    def get_ltp(self, instrument_key: str) -> dict[str, Any]:
        """Fetch last traded price via data_fetcher live quote."""
        if self._data_fetcher is not None:
            try:
                quote = self._data_fetcher.get_live_quote(instrument_key)
                if quote and quote.get("ltp", 0) > 0:
                    return {"ltp": quote["ltp"], "status": "success"}
            except Exception as e:
                logger.warning(f"Paper get_ltp failed for {instrument_key}: {e}")
        return {"ltp": 0}

    def square_off_all(self) -> dict[str, Any]:
        squared = 0
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            qty = abs(pos["quantity"])
            side = "SELL" if pos["quantity"] > 0 else "BUY"
            self.place_order(
                symbol=symbol,
                instrument_key=pos["instrument_key"],
                quantity=qty,
                side=side,
                price=pos["last_price"],
            )
            squared += 1

        return {"squared": squared, "errors": 0}

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update last prices for all positions (call with live data)."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos["last_price"] = price
                if pos["quantity"] > 0:
                    pos["pnl"] = (price - pos["average_price"]) * pos["quantity"]
                else:
                    pos["pnl"] = (pos["average_price"] - price) * abs(pos["quantity"])

    def check_gtt_triggers(self, prices: dict[str, float]) -> list[dict]:
        """Check if any GTT orders should trigger based on current prices."""
        triggered = []
        for gtt_id, gtt in list(self._gtt_orders.items()):
            if gtt["status"] != "active":
                continue

            # Find the symbol for this instrument key
            for symbol, pos in self._positions.items():
                if pos["instrument_key"] == gtt["instrument_key"]:
                    current_price = prices.get(symbol, 0)
                    if current_price <= 0:
                        continue

                    should_trigger = False
                    if gtt["side"] == "SELL" and current_price <= gtt["trigger_price"]:
                        should_trigger = True
                    elif gtt["side"] == "BUY" and current_price >= gtt["trigger_price"]:
                        should_trigger = True

                    if should_trigger:
                        self.place_order(
                            symbol=symbol,
                            instrument_key=gtt["instrument_key"],
                            quantity=gtt["quantity"],
                            side=gtt["side"],
                            price=current_price,
                        )
                        gtt["status"] = "triggered"
                        triggered.append(gtt)
                        logger.info(f"[PAPER] GTT triggered: {gtt_id}")
                    break

        return triggered
