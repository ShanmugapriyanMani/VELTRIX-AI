"""
Upstox Broker Implementation — Full OrderApiV3, MarketDataStreamerV3, GTT, HFT.

Uses upstox-python-sdk for all broker interactions (LIVE mode only).
Orders via HFT (api-hft.upstox.com), rest via api.upstox.com.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import yaml
from loguru import logger

try:
    import upstox_client
    from upstox_client.rest import ApiException
    UPSTOX_AVAILABLE = True

    # Monkey-patch ApiClient.__del__ to suppress "Bad file descriptor"
    # error during Python shutdown (library prints to stdout on cleanup failure)
    def _safe_api_client_del(self):
        try:
            if hasattr(self, "pool") and self.pool:
                self.pool.close()
        except Exception:
            pass

    upstox_client.ApiClient.__del__ = _safe_api_client_del
except ImportError:
    UPSTOX_AVAILABLE = False

from src.config.env_loader import get_config
from src.execution.broker import BaseBroker
from src.data.fetcher import build_auth_from_config


def _inject_api_timeout(api_client, timeout_seconds: int) -> None:
    """Wrap an Upstox ApiClient's REST request method to inject a default timeout.

    The SDK passes _request_timeout=None by default, which means urllib3 waits
    forever. This wrapper injects our configured timeout when none is specified.
    """
    rest = api_client.rest_client
    original_request = rest.request

    def _request_with_timeout(*args, **kwargs):
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = timeout_seconds
        return original_request(*args, **kwargs)

    rest.request = _request_with_timeout


class UpstoxBroker(BaseBroker):
    """
    Full Upstox API V3 broker implementation (LIVE mode only).

    Uses:
    - OrderApiV3 for order placement (latest)
    - HFT endpoint for lower latency on orders
    - PortfolioDataStreamer for real-time order updates
    - GTT orders for automated stop loss / target
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.auth = build_auth_from_config(config)

        self._api_client: Optional[upstox_client.ApiClient] = None
        self._hft_api_client: Optional[upstox_client.ApiClient] = None
        self._order_api: Optional[Any] = None
        self._connected = False
        self._data_fetcher: Optional[Any] = None

    def set_data_fetcher(self, data_fetcher) -> None:
        """Inject data_fetcher for WebSocket LTP cache access."""
        self._data_fetcher = data_fetcher

    def connect(self) -> bool:
        """Connect and authenticate with Upstox."""
        if not UPSTOX_AVAILABLE:
            logger.error("upstox-python-sdk not installed")
            return False

        token = self.auth.load_token()
        if not token or not self.auth.is_valid:
            logger.error(
                "No valid Upstox token. Run: python scripts/auth_upstox.py"
            )
            return False

        # Regular API client (market data, positions, etc.)
        config = self.auth.get_configuration()
        self._api_client = upstox_client.ApiClient(config)

        # HFT API client (orders — separate endpoint for low latency)
        hft_config = self.auth.get_hft_configuration()
        self._hft_api_client = upstox_client.ApiClient(hft_config)

        # Inject default timeout on SDK REST clients (prevents indefinite hangs)
        timeout_s = get_config().API_TIMEOUT_SECONDS
        _inject_api_timeout(self._api_client, timeout_s)
        _inject_api_timeout(self._hft_api_client, timeout_s)

        self._order_api = upstox_client.OrderApiV3(self._hft_api_client)
        self._market_quote_api = upstox_client.MarketQuoteApi(self._api_client)
        self._connected = True

        logger.info("Upstox broker connected (LIVE)")
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
        """
        Place an order via Upstox OrderApiV3.

        Args:
            symbol: Stock symbol (for logging)
            instrument_key: Upstox instrument key ("NSE_EQ|INE002A01018")
            quantity: Number of shares
            side: "BUY" or "SELL"
            order_type: "MARKET", "LIMIT", "SL", "SL-M"
            price: Limit price (for LIMIT/SL orders)
            trigger_price: Trigger price (for SL/SL-M orders)
            product: "I" (intraday), "D" (delivery), "MTF" (margin)
            validity: "DAY" or "IOC"
        """
        if not self._connected or self._order_api is None:
            return {"order_id": None, "status": "error", "message": "Not connected"}

        # Build order request
        order_request = upstox_client.PlaceOrderV3Request(
            quantity=quantity,
            product=product,
            validity=validity,
            price=price if order_type in ("LIMIT", "SL") else 0,
            instrument_token=instrument_key,
            order_type=order_type,
            transaction_type=side,
            disclosed_quantity=0,
            trigger_price=trigger_price if order_type in ("SL", "SL-M") else 0,
            is_amo=False,
            slice=False,
        )

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._order_api.place_order(order_request)

                order_id = response.data.order_id if response.data else None

                logger.info(
                    f"ORDER PLACED: {side} {quantity} {symbol} "
                    f"@ {order_type} {'₹'+str(price) if price else ''} "
                    f"| order_id={order_id}"
                )

                return {
                    "order_id": order_id,
                    "status": "success",
                    "message": f"Order placed: {side} {quantity} {symbol}",
                    "symbol": symbol,
                    "instrument_key": instrument_key,
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type,
                    "price": price,
                }

            except ApiException as e:
                logger.error(
                    f"Order placement failed (attempt {attempt+1}): "
                    f"{e.status} {e.reason} {e.body}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

            except Exception as e:
                logger.error(f"Order placement error: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

        return {"order_id": None, "status": "error", "message": "Max retries exceeded"}

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Modify an existing order."""
        if not self._connected or self._order_api is None:
            return {"status": "error", "message": "Not connected"}

        try:
            modify_request = upstox_client.ModifyOrderRequest(
                order_id=order_id,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                order_type=order_type,
                validity="DAY",
            )
            self._order_api.modify_order(modify_request)
            logger.info(f"Order modified: {order_id}")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            logger.error(f"Order modify failed: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a pending order."""
        if not self._connected or self._order_api is None:
            return {"status": "error", "message": "Not connected"}

        try:
            self._order_api.cancel_order(order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            logger.error(f"Order cancel failed: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all pending orders."""
        if not self._connected:
            return {"status": "error", "message": "Not connected"}

        orders = self.get_order_book()
        cancelled = 0
        errors = 0

        for order in orders:
            status = order.get("status", "")
            if status in ("open", "pending", "trigger pending"):
                result = self.cancel_order(order["order_id"])
                if result["status"] == "success":
                    cancelled += 1
                else:
                    errors += 1

        logger.info(f"Cancel all: {cancelled} cancelled, {errors} errors")
        return {"cancelled": cancelled, "errors": errors}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of a specific order."""
        if not self._connected or self._api_client is None:
            return {}

        try:
            order_api = upstox_client.OrderApi(self._api_client)
            response = order_api.get_order_details(order_id=order_id, api_version="2.0")
            if response.data:
                return {
                    "order_id": order_id,
                    "status": response.data.status,
                    "quantity": response.data.quantity,
                    "filled_quantity": response.data.filled_quantity,
                    "average_price": response.data.average_price,
                    "order_type": response.data.order_type,
                    "transaction_type": response.data.transaction_type,
                }
        except Exception as e:
            logger.error(f"Get order status failed: {e}")

        return {}

    def wait_for_fill(
        self, order_id: str, timeout_seconds: int = 30, poll_interval: float = 2.0
    ) -> dict[str, Any]:
        """
        Poll order status until filled, rejected, or timeout.

        Returns:
            {"filled": True/False, "avg_price": float, "filled_qty": int,
             "order_id": str, "reason": str or None}
        """
        if not order_id:
            return {"filled": False, "order_id": order_id, "reason": "no_order_id"}

        elapsed = 0.0
        while elapsed < timeout_seconds:
            status = self.get_order_status(order_id)
            order_status = (status.get("status") or "").lower()

            if order_status == "complete":
                avg_price = float(status.get("average_price", 0) or 0)
                filled_qty = int(status.get("filled_quantity", 0) or 0)
                logger.info(
                    f"FILL_CONFIRMED: order={order_id} qty={filled_qty} "
                    f"avg_price=₹{avg_price:.2f}"
                )
                return {
                    "filled": True,
                    "avg_price": avg_price,
                    "filled_qty": filled_qty,
                    "order_id": order_id,
                    "reason": None,
                }

            if order_status in ("rejected", "cancelled"):
                reason = status.get("reject_reason") or order_status
                logger.warning(f"ORDER_REJECTED: {order_id} reason={reason}")
                return {
                    "filled": False,
                    "avg_price": 0,
                    "filled_qty": 0,
                    "order_id": order_id,
                    "reason": reason,
                }

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout — attempt cancel
        logger.warning(f"ORDER_TIMEOUT: {order_id} not filled in {timeout_seconds}s")
        self.cancel_order(order_id)
        return {
            "filled": False,
            "avg_price": 0,
            "filled_qty": 0,
            "order_id": order_id,
            "reason": "timeout",
        }

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all current positions."""
        if not self._connected or self._api_client is None:
            return []

        try:
            portfolio_api = upstox_client.PortfolioApi(self._api_client)
            response = portfolio_api.get_positions(api_version="2.0")
            if response.data:
                return [
                    {
                        "instrument_key": pos.instrument_token,
                        "symbol": pos.trading_symbol,
                        "quantity": pos.quantity,
                        "average_price": pos.average_price,
                        "last_price": pos.last_price,
                        "pnl": pos.pnl,
                        "product": pos.product,
                        "side": "BUY" if pos.quantity > 0 else "SELL",
                    }
                    for pos in response.data
                ]
        except Exception as e:
            logger.error(f"Get positions failed: {e}")

        return []

    def get_holdings(self) -> list[dict[str, Any]]:
        """Get all holdings (delivery positions)."""
        if not self._connected or self._api_client is None:
            return []

        try:
            portfolio_api = upstox_client.PortfolioApi(self._api_client)
            response = portfolio_api.get_holdings(api_version="2.0")
            if response.data:
                return [
                    {
                        "instrument_key": h.instrument_token,
                        "symbol": h.trading_symbol,
                        "quantity": h.quantity,
                        "average_price": h.average_price,
                        "last_price": h.last_price,
                        "pnl": h.pnl,
                        "isin": h.isin,
                    }
                    for h in response.data
                ]
        except Exception as e:
            logger.error(f"Get holdings failed: {e}")

        return []

    def get_order_book(self) -> list[dict[str, Any]]:
        """Get today's order book."""
        if not self._connected or self._api_client is None:
            return []

        try:
            order_api = upstox_client.OrderApi(self._api_client)
            response = order_api.get_order_book(api_version="2.0")
            if response.data:
                return [
                    {
                        "order_id": o.order_id,
                        "symbol": o.trading_symbol,
                        "instrument_key": o.instrument_token,
                        "status": o.status,
                        "quantity": o.quantity,
                        "filled_quantity": o.filled_quantity,
                        "price": o.price,
                        "average_price": o.average_price,
                        "order_type": o.order_type,
                        "transaction_type": o.transaction_type,
                        "product": o.product,
                    }
                    for o in response.data
                ]
        except Exception as e:
            logger.error(f"Get order book failed: {e}")

        return []

    def get_funds(self) -> dict[str, Any]:
        """Get available funds and margin from Upstox equity segment."""
        if not self._connected or self._api_client is None:
            return {}

        try:
            user_api = upstox_client.UserApi(self._api_client)
            response = user_api.get_user_fund_margin(api_version="2.0")
            data = response.data
            # SDK returns data as dict(str, UserFundMarginData)
            if isinstance(data, dict):
                eq = data.get("equity")
            elif hasattr(data, "equity"):
                eq = data.equity
            else:
                eq = None
            if eq is not None:
                available = getattr(eq, "available_margin", 0) or 0
                used = getattr(eq, "used_margin", 0) or 0
                payin = getattr(eq, "payin_amount", 0) or 0
                span = getattr(eq, "span_margin", 0) or 0
                exposure = getattr(eq, "exposure_margin", 0) or 0
                notional = getattr(eq, "notional_cash", 0) or 0
                adhoc = getattr(eq, "adhoc_margin", 0) or 0
                return {
                    "available_margin": available,
                    "used_margin": used,
                    "payin_amount": payin,
                    "span_margin": span,
                    "exposure_margin": exposure,
                    "notional_cash": notional,
                    "adhoc_margin": adhoc,
                    "total_balance": available + used,
                }
        except Exception as e:
            # Extract clean Upstox error message if available
            body = getattr(e, "body", None)
            if body:
                import json
                try:
                    err_data = json.loads(body) if isinstance(body, (str, bytes)) else body
                    errors = err_data.get("errors", [])
                    if errors:
                        msg = errors[0].get("message", str(e))
                        logger.error(f"Get funds failed: {msg}")
                        return {}
                except (json.JSONDecodeError, AttributeError, IndexError):
                    pass
            logger.error(f"Get funds failed: {e}")

        return {}

    def get_profile(self) -> dict[str, Any]:
        """Get user profile from Upstox."""
        if not self._connected or self._api_client is None:
            return {}

        try:
            user_api = upstox_client.UserApi(self._api_client)
            response = user_api.get_profile(api_version="2.0")
            if response.data:
                return {
                    "user_id": getattr(response.data, "user_id", ""),
                    "user_name": getattr(response.data, "user_name", ""),
                    "email": getattr(response.data, "email", ""),
                    "exchanges": getattr(response.data, "exchanges", []),
                    "products": getattr(response.data, "products", []),
                    "is_active": getattr(response.data, "is_active", False),
                }
        except Exception as e:
            logger.error(f"Get profile failed: {e}")

        return {}

    def place_gtt_order(
        self,
        instrument_key: str,
        trigger_price: float,
        limit_price: float = 0,
        quantity: int = 0,
        side: str = "SELL",
        strategy: str = "STOPLOSS",
        trigger_type: str = "BELOW",
    ) -> dict[str, Any]:
        """
        Place a GTT (Good Till Triggered) order via V3 API.

        V3 GTT uses rules-based structure:
        - strategy: "ENTRY", "TARGET", or "STOPLOSS"
        - trigger_type: "ABOVE", "BELOW", or "IMMEDIATE"

        Used for automated stop loss and target orders.
        """
        if not self._connected or self._order_api is None:
            return {"status": "error", "message": "Not connected"}

        # Use limit_price if provided, otherwise use trigger_price
        order_price = limit_price if limit_price > 0 else trigger_price

        try:
            gtt_request = upstox_client.PlaceGTTOrderRequest(
                type="SINGLE",
                quantity=quantity,
                product="I",  # Options are intraday (MIS), not delivery
                instrument_token=instrument_key,
                transaction_type=side,
                price=order_price,
                rules=[
                    upstox_client.GTTRule(
                        strategy=strategy,
                        trigger_type=trigger_type,
                        trigger_price=trigger_price,
                    )
                ],
            )
            response = self._order_api.place_gtt_order(gtt_request)
            gtt_ids = response.data.gtt_order_ids if response.data else []
            gtt_id = gtt_ids[0] if gtt_ids else None

            logger.info(
                f"GTT placed: {side} {quantity} @ trigger ₹{trigger_price} "
                f"limit ₹{order_price:.2f} ({strategy}/{trigger_type}) | gtt_id={gtt_id}"
            )
            return {"gtt_id": gtt_id, "status": "success"}
        except Exception as e:
            logger.error(f"GTT placement failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_ltp(self, instrument_key: str) -> dict[str, Any]:
        """Fetch last traded price — WebSocket cache first, REST fallback."""
        # Try WebSocket cache (sub-millisecond, no API call)
        if self._data_fetcher:
            ws_ltp = self._data_fetcher.get_ws_ltp(instrument_key)
            if ws_ltp and ws_ltp > 0:
                return {"ltp": ws_ltp, "status": "success", "source": "ws"}

        if not self._connected:
            return {"ltp": 0}

        try:
            market_api = getattr(self, "_market_quote_api", None)
            if market_api is None:
                configuration = self.auth.get_configuration()
                api_client = upstox_client.ApiClient(configuration)
                market_api = upstox_client.MarketQuoteApi(api_client)
            response = market_api.ltp(instrument_key=instrument_key)

            if response.data:
                # Upstox SDK may return colon-separated keys instead of pipe
                quote = response.data.get(instrument_key)
                if quote is None:
                    colon_key = instrument_key.replace("|", ":")
                    quote = response.data.get(colon_key)
                if quote is None:
                    for v in response.data.values():
                        if getattr(v, "instrument_token", None) == instrument_key:
                            quote = v
                            break
                if quote is not None:
                    return {"ltp": quote.last_price, "status": "success"}
        except Exception as e:
            logger.error(f"LTP fetch failed for {instrument_key}: {e}")

        return {"ltp": 0}

    def cancel_gtt_order(self, gtt_id: str) -> dict[str, Any]:
        """Cancel a GTT order via V3 API."""
        if not self._connected or self._order_api is None:
            return {"status": "error", "message": "Not connected"}

        try:
            self._order_api.cancel_gtt_order(gtt_order_id=gtt_id)
            logger.info(f"GTT cancelled: {gtt_id}")
            return {"status": "success", "gtt_id": gtt_id}
        except Exception as e:
            logger.error(f"GTT cancel failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_daily_pnl(self) -> Optional[float]:
        """
        Get today's realized P&L from Upstox positions.

        Sums the `pnl` field from all positions (includes realized + unrealized).
        Returns None on API failure (caller should skip reconciliation).
        """
        if not self._connected or self._api_client is None:
            logger.warning("BROKER_PNL_FETCH_FAILED: not connected")
            return None

        try:
            positions = self.get_positions()
            if not positions:
                return 0.0
            total_pnl = sum(float(p.get("pnl", 0) or 0) for p in positions)
            return round(total_pnl, 2)
        except Exception as e:
            logger.error(f"BROKER_PNL_FETCH_FAILED: {e}")
            return None

    def get_todays_trades(self) -> list[dict[str, Any]]:
        """Get today's filled trades from Upstox trade book."""
        if not self._connected or self._api_client is None:
            return []

        try:
            order_api = upstox_client.OrderApi(self._api_client)
            response = order_api.get_trade_book(api_version="2.0")
            if response.data:
                return [
                    {
                        "symbol": t.trading_symbol,
                        "instrument_key": t.instrument_token,
                        "quantity": t.quantity,
                        "average_price": t.average_price,
                        "trade_type": t.transaction_type,
                        "order_id": t.order_id,
                    }
                    for t in response.data
                ]
        except Exception as e:
            logger.error(f"Get trades for day failed: {e}")

        return []

    def get_available_margin(self) -> Optional[float]:
        """Get available equity margin. Returns None on API failure."""
        funds = self.get_funds()
        if not funds:
            return None
        margin = funds.get("available_margin")
        return float(margin) if margin is not None else None

    def square_off_all(self) -> dict[str, Any]:
        """Square off all open positions."""
        positions = self.get_positions()
        squared = 0
        errors = 0

        for pos in positions:
            qty = abs(pos.get("quantity", 0))
            if qty == 0:
                continue

            side = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
            result = self.place_order(
                symbol=pos.get("symbol", ""),
                instrument_key=pos.get("instrument_key", ""),
                quantity=qty,
                side=side,
                order_type="MARKET",
                product=pos.get("product", "I"),
            )

            if result.get("status") == "success":
                squared += 1
            else:
                errors += 1

        logger.info(f"Square off: {squared} positions closed, {errors} errors")
        return {"squared": squared, "errors": errors}
