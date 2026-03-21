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

from src.config.env_loader import get_config
from src.execution.broker import BaseBroker


class PaperTrader(BaseBroker):
    """
    Simulated broker for paper trading.
    Implements the same interface as UpstoxBroker.
    Uses real-time data but simulates fills with configurable slippage.
    """

    def __init__(
        self,
        initial_capital: float = 0,
        slippage_pct: float = 0.05,
        data_fetcher: Any = None,
    ):
        if initial_capital <= 0:
            initial_capital = get_config().TRADING_CAPITAL
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

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day."""
        self._orders = {}
        self._positions = {}
        self._gtt_orders = {}
        self._order_counter = 0
        logger.info("PAPER_TRADER_RESET: daily state cleared")

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

        # Guard: block SELL at price=0 (corrupts P&L)
        if side == "SELL" and (price is None or price <= 0):
            logger.error(f"PAPER_SELL_BLOCKED: {symbol} invalid exit_price={price}")
            return {
                "order_id": order_id,
                "status": "rejected",
                "message": f"SELL blocked: invalid price {price}",
            }

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
        """Fetch last traded price — WebSocket cache first, REST fallback."""
        if self._data_fetcher is not None:
            # Try WebSocket cache first (sub-millisecond, no API call)
            ws_ltp = self._data_fetcher.get_ws_ltp(instrument_key)
            if ws_ltp and ws_ltp > 0:
                return {"ltp": ws_ltp, "status": "success", "source": "ws"}
            # REST fallback
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

    def place_iron_condor_order(
        self,
        signal: dict,
        position_id: str,
    ) -> dict:
        """Simulate placing a 4-leg Iron Condor order with slippage."""
        qty = signal.get("quantity", 0)
        if qty <= 0:
            return {"status": "rejected", "reason": "Zero quantity"}

        # Leg 1: SELL CE (near OTM)
        leg1 = self.place_order(
            symbol=f"NIFTY{int(signal['sell_ce_strike'])}CE",
            instrument_key=signal.get("sell_ce_instrument_key", ""),
            quantity=qty, side="SELL",
            price=signal["sell_ce_premium"], product="I",
        )
        # Leg 2: BUY CE (protection)
        leg2 = self.place_order(
            symbol=f"NIFTY{int(signal['buy_ce_strike'])}CE",
            instrument_key=signal.get("buy_ce_instrument_key", ""),
            quantity=qty, side="BUY",
            price=signal["buy_ce_premium"], product="I",
        )
        # Leg 3: SELL PE (near OTM)
        leg3 = self.place_order(
            symbol=f"NIFTY{int(signal['sell_pe_strike'])}PE",
            instrument_key=signal.get("sell_pe_instrument_key", ""),
            quantity=qty, side="SELL",
            price=signal["sell_pe_premium"], product="I",
        )
        # Leg 4: BUY PE (protection)
        leg4 = self.place_order(
            symbol=f"NIFTY{int(signal['buy_pe_strike'])}PE",
            instrument_key=signal.get("buy_pe_instrument_key", ""),
            quantity=qty, side="BUY",
            price=signal["buy_pe_premium"], product="I",
        )

        all_ok = all(r.get("status") == "success" for r in [leg1, leg2, leg3, leg4])
        if not all_ok:
            logger.critical(f"IC LEG FAILURE -- manual review needed: {position_id}")
            return {"status": "error", "reason": "Not all legs filled"}

        # Compute actual credit after slippage
        actual_credit = (
            (signal["sell_ce_premium"] * (1 - self.slippage_pct))
            + (signal["sell_pe_premium"] * (1 - self.slippage_pct))
            - (signal["buy_ce_premium"] * (1 + self.slippage_pct))
            - (signal["buy_pe_premium"] * (1 + self.slippage_pct))
        )

        logger.info(
            f"IC_LEG: SELL {qty//65}L NIFTY{int(signal['sell_ce_strike'])}CE "
            f"@ ₹{signal['sell_ce_premium']:.0f} (credit)"
        )
        logger.info(
            f"IC_LEG: BUY  {qty//65}L NIFTY{int(signal['buy_ce_strike'])}CE "
            f"@ ₹{signal['buy_ce_premium']:.0f} (debit)"
        )
        logger.info(
            f"IC_LEG: SELL {qty//65}L NIFTY{int(signal['sell_pe_strike'])}PE "
            f"@ ₹{signal['sell_pe_premium']:.0f} (credit)"
        )
        logger.info(
            f"IC_LEG: BUY  {qty//65}L NIFTY{int(signal['buy_pe_strike'])}PE "
            f"@ ₹{signal['buy_pe_premium']:.0f} (debit)"
        )
        logger.info(
            f"IC_ENTRY: NIFTY Iron Condor | "
            f"Range: {int(signal['sell_pe_strike'])}-{int(signal['sell_ce_strike'])} | "
            f"Net credit: ₹{actual_credit:.0f}/unit | "
            f"Max profit: ₹{signal.get('max_profit', 0):.0f} | "
            f"Max loss: ₹{signal.get('max_loss', 0):.0f}"
        )

        return {
            "status": "success",
            "position_id": position_id,
            "actual_credit": actual_credit,
            "legs": [leg1, leg2, leg3, leg4],
        }

    def close_iron_condor_order(
        self,
        ic_position,
        exit_reason: str,
        ltp_dict: dict,
    ) -> dict:
        """Close all 4 legs of an IC position at current LTPs."""
        qty = ic_position.quantity

        # Buy back sell legs, sell protection legs
        self.place_order(
            symbol=f"NIFTY{int(ic_position.sell_ce_strike)}CE",
            instrument_key=ic_position.sell_ce_instrument_key,
            quantity=qty, side="BUY",
            price=ltp_dict.get(ic_position.sell_ce_instrument_key, 0),
            product="I",
        )
        self.place_order(
            symbol=f"NIFTY{int(ic_position.buy_ce_strike)}CE",
            instrument_key=ic_position.buy_ce_instrument_key,
            quantity=qty, side="SELL",
            price=ltp_dict.get(ic_position.buy_ce_instrument_key, 0),
            product="I",
        )
        self.place_order(
            symbol=f"NIFTY{int(ic_position.sell_pe_strike)}PE",
            instrument_key=ic_position.sell_pe_instrument_key,
            quantity=qty, side="BUY",
            price=ltp_dict.get(ic_position.sell_pe_instrument_key, 0),
            product="I",
        )
        self.place_order(
            symbol=f"NIFTY{int(ic_position.buy_pe_strike)}PE",
            instrument_key=ic_position.buy_pe_instrument_key,
            quantity=qty, side="SELL",
            price=ltp_dict.get(ic_position.buy_pe_instrument_key, 0),
            product="I",
        )

        logger.info(f"IC_EXIT: {ic_position.position_id} | reason={exit_reason}")
        return {"status": "success", "exit_reason": exit_reason}
