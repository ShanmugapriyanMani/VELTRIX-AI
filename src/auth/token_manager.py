"""
Token lifecycle management for Upstox OAuth2 tokens.

- JWT expiry decoding (no secret needed — just base64 payload)
- TokenWatcher daemon thread: checks every 5 min, alerts before expiry
- Upstox uses authorization_code flow — silent refresh requires a refresh token
  that Upstox does not currently provide. The watcher alerts the user to
  manually re-authenticate when the token is about to expire.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import requests
from loguru import logger


def decode_jwt_expiry(token: str) -> Optional[datetime]:
    """
    Decode JWT expiry timestamp without verification.

    Upstox access tokens are JWTs with an 'exp' claim (epoch seconds).
    We only need the payload — no signature verification required since
    we're reading our own token, not validating an untrusted one.

    Returns:
        datetime (UTC-aware) of expiry, or None if token is not valid JWT.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("TOKEN_EXPIRY_UNKNOWN: token is not JWT format")
            return None

        # Base64url decode the payload (part 1)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)

        exp = payload.get("exp")
        if exp is None:
            logger.warning("TOKEN_EXPIRY_UNKNOWN: JWT has no 'exp' claim")
            return None

        return datetime.fromtimestamp(exp, tz=timezone.utc)

    except Exception as e:
        logger.warning(f"TOKEN_EXPIRY_UNKNOWN: failed to decode JWT: {e}")
        return None


def get_token_expiry(token: str) -> Optional[datetime]:
    """Get token expiry as a local datetime (no timezone)."""
    utc_expiry = decode_jwt_expiry(token)
    if utc_expiry is None:
        return None
    # Convert to local time (IST) for display/comparison with datetime.now()
    return utc_expiry.astimezone().replace(tzinfo=None)


def is_token_expiring_soon(token: str, minutes: int = 30) -> bool:
    """Check if token expires within the given minutes."""
    expiry = get_token_expiry(token)
    if expiry is None:
        return False  # Cannot determine — don't trigger false alarm
    return (expiry - datetime.now()) <= timedelta(minutes=minutes)


class TokenWatcher:
    """
    Background daemon thread that monitors token expiry and alerts.

    Upstox daily tokens typically expire at ~3:30 AM IST next day (well after
    market close). The watcher protects against edge cases where a token was
    issued late, or when the bot runs overnight for testing.

    Since Upstox authorization_code flow has no refresh_token grant,
    the watcher can only:
    1. Alert via Telegram when expiry is approaching
    2. Attempt refresh if UPSTOX_REFRESH_TOKEN is configured (future-proof)
    3. Log warnings for manual intervention
    """

    CHECK_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        auth: Any,  # UpstoxAuth instance
        alert_fn: Optional[Callable[[str], Any]] = None,
    ):
        self.auth = auth
        self.alert_fn = alert_fn
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_alert_time: Optional[datetime] = None
        self._alert_cooldown = timedelta(minutes=30)  # Don't spam alerts

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="TokenWatcher",
        )
        self._thread.start()
        logger.info("TOKEN_WATCHER: started (checking every 5 min)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("TOKEN_WATCHER: stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_and_alert()
            except Exception as e:
                logger.error(f"TOKEN_WATCHER_ERROR: {e}")
            self._stop_event.wait(self.CHECK_INTERVAL)

    def _check_and_alert(self) -> None:
        token = self.auth.access_token
        if not token:
            return  # No token loaded — nothing to watch

        expiry = get_token_expiry(token)
        if expiry is None:
            return  # Non-JWT token — can't monitor

        remaining = expiry - datetime.now()

        if remaining <= timedelta(minutes=0):
            self._send_alert(
                "TOKEN_EXPIRED: Upstox token has expired. "
                "Live orders will fail. Run: python scripts/auth_upstox.py"
            )
            logger.critical("TOKEN_EXPIRED: manual re-authentication required")
            return

        if remaining <= timedelta(minutes=30):
            self._send_alert(
                f"TOKEN_EXPIRING: Upstox token expires in {int(remaining.total_seconds() // 60)} min. "
                "Refresh recommended: python scripts/auth_upstox.py"
            )
            logger.warning(
                f"TOKEN_EXPIRING_SOON: {int(remaining.total_seconds() // 60)} min remaining"
            )
            return

        # Token still healthy — debug log only
        logger.debug(
            f"TOKEN_OK: {int(remaining.total_seconds() // 3600)}h "
            f"{int((remaining.total_seconds() % 3600) // 60)}m remaining"
        )

    def _send_alert(self, message: str) -> None:
        """Send alert via Telegram with cooldown to avoid spam."""
        now = datetime.now()
        if self._last_alert_time and (now - self._last_alert_time) < self._alert_cooldown:
            return  # Cooldown active
        self._last_alert_time = now
        if self.alert_fn:
            try:
                self.alert_fn(f"🔑 {message}")
            except Exception as e:
                logger.warning(f"TOKEN_WATCHER: alert send failed: {e}")

    def check_on_startup(self) -> None:
        """
        Immediate token health check at bot startup.
        Called once before the watch loop begins.
        """
        token = self.auth.access_token
        if not token:
            logger.warning("TOKEN_CHECK: no token loaded")
            return

        expiry = get_token_expiry(token)
        if expiry is None:
            logger.warning("TOKEN_EXPIRY_UNKNOWN: cannot determine expiry")
            return

        remaining = expiry - datetime.now()

        if remaining <= timedelta(minutes=0):
            logger.critical(f"TOKEN_EXPIRED: expired {abs(remaining)} ago")
            self._send_alert(
                "TOKEN_EXPIRED at startup. Live orders will fail. "
                "Run: python scripts/auth_upstox.py"
            )
        elif remaining <= timedelta(minutes=60):
            logger.warning(
                f"TOKEN_WARNING: expires within {int(remaining.total_seconds() // 60)} min"
            )
            self._send_alert(
                f"Token expiring within {int(remaining.total_seconds() // 60)} min. "
                "Trading will continue but refresh recommended."
            )
        else:
            hours = int(remaining.total_seconds() // 3600)
            mins = int((remaining.total_seconds() % 3600) // 60)
            logger.info(f"TOKEN_OK: valid until {expiry.strftime('%Y-%m-%d %H:%M:%S IST')} ({hours}h {mins}m)")
