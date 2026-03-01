"""
EnvConfig — Environment-first configuration for VELTRIX.

Priority: env var > yaml value > hardcoded default.
All env vars are read at import time via python-dotenv.

Loading order:
  1. .env            → credentials + TRADING_STAGE
  2. .env.{stage}    → stage-specific settings (override=True)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Resolve project root (env_loader.py → src/config/ → src/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Step 1: Load .env (credentials + stage selection)
load_dotenv(_PROJECT_ROOT / ".env")

# Step 2: Load stage-specific file (overrides base .env values)
_stage = os.environ.get("TRADING_STAGE", "BASIC").upper()
_stage_file = _PROJECT_ROOT / f".env.{_stage.lower()}"
if _stage_file.exists():
    load_dotenv(_stage_file, override=True)
else:
    logger.warning(f"Stage file {_stage_file} not found, using defaults")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    return int(val)


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    return float(val)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_is_set(key: str) -> bool:
    """Check if an env var is explicitly set (not just default)."""
    return key in os.environ


@lru_cache(maxsize=1)
def get_config() -> EnvConfig:
    """Return the singleton EnvConfig instance."""
    return EnvConfig()


class EnvConfig:
    """
    Central configuration object. All modules read from this.
    Values come from env vars with sane defaults matching current YAML.
    """

    def __init__(self) -> None:
        # ── Capital & Risk ──
        self.TRADING_CAPITAL = _env_float("TRADING_CAPITAL", 50000)
        self.DEPLOY_CAP = _env_float("DEPLOY_CAP", 25000)
        self.RISK_PER_TRADE = _env_float("RISK_PER_TRADE", 10000)
        self.DAILY_LOSS_HALT = _env_float("DAILY_LOSS_HALT", 10000)

        # ── Trading Mode ──
        self.TRADING_MODE = _env("TRADING_MODE", "paper")
        self.TRADING_STAGE = _env("TRADING_STAGE", "BASIC")

        # ── Feature Flags ──
        self.ML_ENABLED = _env_bool("ML_ENABLED", True)
        self.ACTIVE_TRADING = _env_bool("ACTIVE_TRADING", False)
        self.MAX_TRADES_PER_DAY = _env_int("MAX_TRADES_PER_DAY", 2)

        # ── Market Hours (IST, 24hr) ──
        self.MARKET_OPEN = _env("MARKET_OPEN", "09:15")
        self.TRADE_START = _env("TRADE_START", "10:00")
        self.TRADE_END = _env("TRADE_END", "15:10")
        self.NO_NEW_TRADE_AFTER = _env("NO_NEW_TRADE_AFTER", "14:30")
        self.EXPIRY_EXIT_BY = _env("EXPIRY_EXIT_BY", "13:30")
        self.SQUARE_OFF_TIME = _env("SQUARE_OFF_TIME", "15:15")
        self.SKIP_FIRST_MINUTES = _env_int("SKIP_FIRST_MINUTES", 15)

        # ── Regime Conviction Thresholds ──
        self.TRENDING_THRESHOLD = _env_float("TRENDING_THRESHOLD", 1.75)
        self.RANGEBOUND_THRESHOLD = _env_float("RANGEBOUND_THRESHOLD", 2.0)
        self.VOLATILE_THRESHOLD = _env_float("VOLATILE_THRESHOLD", 2.5)

        # ── Options SL/TP Base Percentages ──
        self.SL_BASE_PCT = _env_int("SL_BASE_PCT", 30)
        self.TP_BASE_PCT = _env_int("TP_BASE_PCT", 45)

        # ── Options Constants ──
        self.MIN_PREMIUM = _env_int("MIN_PREMIUM", 80)
        self.MIN_WALLET_BALANCE = _env_int("MIN_WALLET_BALANCE", 50000)
        self.BUFFER = _env_int("BUFFER", 5000)

        # ── PLUS Spread Settings ──
        self.SPREAD_WIDTH = _env_int("SPREAD_WIDTH", 200)
        self.DEBIT_SPREAD_SL_PCT = _env_int("DEBIT_SPREAD_SL_PCT", 50)
        self.DEBIT_SPREAD_TP_PCT = _env_int("DEBIT_SPREAD_TP_PCT", 70)
        self.CREDIT_SPREAD_SL_MULTIPLIER = _env_float(
            "CREDIT_SPREAD_SL_MULTIPLIER", 2.0
        )
        self.CREDIT_SPREAD_TP_PCT = _env_int("CREDIT_SPREAD_TP_PCT", 80)

        # ── Telegram ──
        self.TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", "")

        # ── Database ──
        self.DB_PATH = _env("DB_PATH", "data/trading_bot.db")

        # ── Logging ──
        self.LOG_LEVEL = _env("LOG_LEVEL", "INFO")

        # ── Upstox Credentials ──
        self.UPSTOX_API_KEY = _env("UPSTOX_LIVE_API_KEY", "")
        self.UPSTOX_API_SECRET = _env("UPSTOX_LIVE_API_SECRET", "")
        self.UPSTOX_REDIRECT_URI = _env(
            "UPSTOX_LIVE_REDIRECT_URI", "http://127.0.0.1:5000/callback"
        )
        self.UPSTOX_ACCESS_TOKEN = _env("UPSTOX_ACCESS_TOKEN", "")
        self.UPSTOX_TOTP_SECRET = _env("UPSTOX_TOTP_SECRET", "")

    @staticmethod
    def is_env_set(key: str) -> bool:
        """Check if a specific env var is explicitly provided."""
        return _env_is_set(key)

    def log_config(self) -> None:
        logger.info("=" * 50)
        logger.info("VELTRIX CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"  Capital: {self.TRADING_CAPITAL:,.0f}")
        logger.info(f"  Deploy Cap: {self.DEPLOY_CAP:,.0f}")
        logger.info(f"  Risk/Trade: {self.RISK_PER_TRADE:,.0f}")
        logger.info(f"  Daily Loss Halt: {self.DAILY_LOSS_HALT:,.0f}")
        logger.info(f"  Mode: {self.TRADING_MODE}")
        logger.info(f"  Stage: {self.TRADING_STAGE}")
        logger.info(f"  ML: {'ON' if self.ML_ENABLED else 'OFF'}")
        logger.info(f"  Active Trading: {'ON' if self.ACTIVE_TRADING else 'OFF'}")
        logger.info(f"  Max Trades/Day: {self.MAX_TRADES_PER_DAY}")
        logger.info(f"  Trade Window: {self.TRADE_START} - {self.TRADE_END}")
        logger.info(
            f"  Thresholds: T={self.TRENDING_THRESHOLD} "
            f"R={self.RANGEBOUND_THRESHOLD} V={self.VOLATILE_THRESHOLD}"
        )
        if self.TRADING_STAGE == "PLUS":
            logger.info(
                f"  Spread Width: {self.SPREAD_WIDTH} | "
                f"Debit SL/TP: {self.DEBIT_SPREAD_SL_PCT}%/{self.DEBIT_SPREAD_TP_PCT}% | "
                f"Credit SL: {self.CREDIT_SPREAD_SL_MULTIPLIER}x TP: {self.CREDIT_SPREAD_TP_PCT}%"
            )
        logger.info("=" * 50)
