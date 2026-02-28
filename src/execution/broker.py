"""
Abstract Broker Interface — defines the contract for all broker implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseBroker(ABC):
    """Abstract broker interface. Upstox and PaperTrader both implement this."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect and authenticate with the broker."""
        ...

    @abstractmethod
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
        Place an order.

        Returns: {order_id, status, message}
        """
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Modify an existing order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all pending orders."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get current status of an order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Get all current positions."""
        ...

    @abstractmethod
    def get_holdings(self) -> list[dict[str, Any]]:
        """Get all holdings (delivery positions)."""
        ...

    @abstractmethod
    def get_order_book(self) -> list[dict[str, Any]]:
        """Get today's order book."""
        ...

    @abstractmethod
    def get_funds(self) -> dict[str, Any]:
        """Get available funds and margin."""
        ...

    @abstractmethod
    def place_gtt_order(
        self,
        instrument_key: str,
        trigger_price: float,
        limit_price: float,
        quantity: int,
        side: str,
    ) -> dict[str, Any]:
        """Place a Good Till Triggered order (stop loss/target)."""
        ...

    @abstractmethod
    def cancel_gtt_order(self, gtt_id: str) -> dict[str, Any]:
        """Cancel a GTT order."""
        ...

    @abstractmethod
    def square_off_all(self) -> dict[str, Any]:
        """Square off (close) all open positions."""
        ...
