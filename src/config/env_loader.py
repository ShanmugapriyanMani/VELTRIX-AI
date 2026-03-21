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


def parse_time_config(time_str: str, default_h: int, default_m: int) -> tuple[int, int]:
    """Safely parse 'HH:MM' time string. Returns (hour, minute) or default on error."""
    try:
        h, m = (int(x) for x in time_str.split(":"))
        return h, m
    except (ValueError, AttributeError) as e:
        logger.warning(f"TRADE_END_PARSE_ERROR: '{time_str}' — {e}, using default {default_h:02d}:{default_m:02d}")
        return default_h, default_m


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
        self.FIXED_R_SIZING = _env_bool("FIXED_R_SIZING", True)

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
        self.ELEVATED_THRESHOLD = _env_float("ELEVATED_THRESHOLD", 2.0)

        # ── V9 Direction-Aware Thresholds ──
        self.CE_TRENDING_THRESHOLD = _env_float("CE_TRENDING_THRESHOLD", 2.0)
        self.CE_RANGEBOUND_THRESHOLD = _env_float("CE_RANGEBOUND_THRESHOLD", 2.25)
        self.CE_VOLATILE_THRESHOLD = _env_float("CE_VOLATILE_THRESHOLD", 3.0)
        self.CE_ELEVATED_THRESHOLD = _env_float("CE_ELEVATED_THRESHOLD", 2.5)
        self.PE_TRENDING_THRESHOLD = _env_float("PE_TRENDING_THRESHOLD", 1.5)
        self.PE_RANGEBOUND_THRESHOLD = _env_float("PE_RANGEBOUND_THRESHOLD", 1.75)
        self.PE_VOLATILE_THRESHOLD = _env_float("PE_VOLATILE_THRESHOLD", 2.5)
        self.PE_ELEVATED_THRESHOLD = _env_float("PE_ELEVATED_THRESHOLD", 2.0)

        # ── V9 Weekly/Monthly Guards ──
        self.WEEKLY_LOSS_WARNING = _env_float("WEEKLY_LOSS_WARNING", 20000)
        self.WEEKLY_LOSS_HALT = _env_float("WEEKLY_LOSS_HALT", 35000)
        self.MONTHLY_LOSS_PCT_BOOST = _env_float("MONTHLY_LOSS_PCT_BOOST", 8.0)

        # ── V9 ML ──
        self.ML_AUTO_WEIGHT_DEFAULT = _env_float("ML_AUTO_WEIGHT_DEFAULT", 0.0)

        # ── Two-Stage ML System ──
        self.ML_STAGE1_ENABLED = _env_bool("ML_STAGE1_ENABLED", False)
        self.ML_STAGE1_WEIGHT = _env_float("ML_STAGE1_WEIGHT", 1.5)
        self.ML_STAGE1_CONFIDENCE_THRESHOLD = _env_float("ML_STAGE1_CONFIDENCE_THRESHOLD", 0.45)
        self.ML_QUALITY_GATE_ENABLED = _env_bool("ML_QUALITY_GATE_ENABLED", False)
        self.ML_QUALITY_MIN_WIN_PROB = _env_float("ML_QUALITY_MIN_WIN_PROB", 0.45)
        self.ML_RETRAIN_DAY = _env("ML_RETRAIN_DAY", "Monday")
        self.ML_DRIFT_THRESHOLD = _env_float("ML_DRIFT_THRESHOLD", 0.10)

        # ── Partial Profit + Runner ──
        self.PARTIAL_EXIT_ENABLED = _env_bool("PARTIAL_EXIT_ENABLED", True)
        self.PARTIAL_TP1_RATIO = _env_float("PARTIAL_TP1_RATIO", 0.5)
        self.PARTIAL_EXIT_PCT = _env_float("PARTIAL_EXIT_PCT", 0.5)

        # ── Options SL/TP Base Percentages ──
        self.SL_BASE_PCT = _env_int("SL_BASE_PCT", 30)
        self.TP_BASE_PCT = _env_int("TP_BASE_PCT", 45)

        # ── Options Constants ──
        self.MIN_PREMIUM = _env_int("MIN_PREMIUM", 80)
        self.MIN_WALLET_BALANCE = _env_int("MIN_WALLET_BALANCE", 50000)
        self.BUFFER = _env_int("BUFFER", 5000)

        # ── Entry Distance Filter ──
        self.ENTRY_DIST_FILTER_ENABLED = _env_bool("ENTRY_DIST_FILTER_ENABLED", False)
        self.MAX_ENTRY_DIST_FROM_OPEN = _env_float("MAX_ENTRY_DIST_FROM_OPEN", 0.008)

        # ── PLUS Spread Settings ──
        self.SPREAD_WIDTH = _env_int("SPREAD_WIDTH", 200)
        self.DEBIT_SPREAD_SL_PCT = _env_int("DEBIT_SPREAD_SL_PCT", 50)
        self.DEBIT_SPREAD_TP_PCT = _env_int("DEBIT_SPREAD_TP_PCT", 70)
        self.CREDIT_SPREAD_SL_MULTIPLIER = _env_float(
            "CREDIT_SPREAD_SL_MULTIPLIER", 2.0
        )
        self.CREDIT_SPREAD_TP_PCT = _env_int("CREDIT_SPREAD_TP_PCT", 80)

        # ── Iron Condor Settings ──
        self.IC_ENABLED = _env_bool("IC_ENABLED", False)
        self.IC_BACKTEST_ENABLED = _env_bool("IC_BACKTEST_ENABLED", False)
        self.IC_SPREAD_WIDTH = _env_int("IC_SPREAD_WIDTH", 200)
        self.IC_MIN_CREDIT = _env_int("IC_MIN_CREDIT", 50)
        self.IC_TP_PCT = _env_int("IC_TP_PCT", 80)
        self.IC_SL_MULTIPLIER = _env_float("IC_SL_MULTIPLIER", 2.0)
        self.IC_MIN_WING_DISTANCE = _env_int("IC_MIN_WING_DISTANCE", 300)
        self.IC_MAX_TRADES_PER_DAY = _env_int("IC_MAX_TRADES_PER_DAY", 1)
        self.IC_ADX_MAX = _env_float("IC_ADX_MAX", 20.0)
        self.IC_VIX_MIN = _env_float("IC_VIX_MIN", 15.0)
        self.IC_VIX_MAX = _env_float("IC_VIX_MAX", 22.0)
        self.IC_PCR_MIN = _env_float("IC_PCR_MIN", 0.80)
        self.IC_PCR_MAX = _env_float("IC_PCR_MAX", 1.20)
        self.IC_SCORE_DIFF_MAX = _env_float("IC_SCORE_DIFF_MAX", 2.0)
        self.IC_TRADE_START = _env("IC_TRADE_START", "10:00")
        self.IC_TRADE_END = _env("IC_TRADE_END", "11:30")
        self.IC_MAX_OPENING_RANGE_PCT = _env_float("IC_MAX_OPENING_RANGE_PCT", 0.004)
        self.IC_MAX_VIX_CHANGE_PCT = _env_float("IC_MAX_VIX_CHANGE_PCT", 0.10)

        # ── IV Awareness Filter ──
        self.IV_FILTER_ENABLED = _env_bool("IV_FILTER_ENABLED", True)
        self.IV_HIGH_THRESHOLD = _env_float("IV_HIGH_THRESHOLD", 1.30)
        self.IV_LOW_THRESHOLD = _env_float("IV_LOW_THRESHOLD", 0.80)
        self.IV_HIGH_PENALTY = _env_float("IV_HIGH_PENALTY", 0.50)
        self.IV_LOW_BONUS = _env_float("IV_LOW_BONUS", 0.25)

        # ── OI Change Rate Filter ──
        self.OI_CHANGE_FILTER_ENABLED = _env_bool("OI_CHANGE_FILTER_ENABLED", True)
        self.OI_CONFIRMED_BONUS = _env_float("OI_CONFIRMED_BONUS", 0.25)
        self.OI_CONTRADICTED_PENALTY = _env_float("OI_CONTRADICTED_PENALTY", 0.75)
        self.OI_CHANGE_CONFIRMED_THRESHOLD = _env_float("OI_CHANGE_CONFIRMED_THRESHOLD", 2.0)
        self.OI_CHANGE_CONTRADICTED_THRESHOLD = _env_float("OI_CHANGE_CONTRADICTED_THRESHOLD", 3.0)
        self.OI_SNAPSHOT_INTERVAL_MINUTES = _env_int("OI_SNAPSHOT_INTERVAL_MINUTES", 30)

        # ── Price Contradiction Filter ──
        self.PRICE_CONTRADICTION_ENABLED = _env_bool("PRICE_CONTRADICTION_ENABLED", True)
        self.PRICE_CONTRADICTION_THRESHOLD = _env_float("PRICE_CONTRADICTION_THRESHOLD", 0.003)

        # ── ML Disagreement Filter ──
        self.ML_DISAGREEMENT_ENABLED = _env_bool("ML_DISAGREEMENT_ENABLED", True)
        self.ML_DISAGREEMENT_THRESHOLD = _env_float("ML_DISAGREEMENT_THRESHOLD", 0.60)

        # ── Intraday Blend (daily vs intraday scoring weights) ──
        self.INTRADAY_BLEND_ENABLED = _env_bool("INTRADAY_BLEND_ENABLED", True)
        self.DAILY_WEIGHT = _env_float("DAILY_WEIGHT", 0.20)
        self.INTRADAY_WEIGHT = _env_float("INTRADAY_WEIGHT", 0.80)

        # ── 30-Minute Rescore Schedule ──
        self.RESCORE_INTERVAL_ENABLED = _env_bool("RESCORE_INTERVAL_ENABLED", True)
        self.RESCORE_INTERVAL_MINUTES = _env_int("RESCORE_INTERVAL_MINUTES", 30)

        # ── Rescore Exit (position-open exit decisions at rescore) ──
        self.RESCORE_EXIT_ENABLED = _env_bool("RESCORE_EXIT_ENABLED", True)
        self.RESCORE_EXIT_MIN_PROFIT = _env_float("RESCORE_EXIT_MIN_PROFIT", 0.05)
        self.RESCORE_EXIT_DECAY_THRESHOLD = _env_float("RESCORE_EXIT_DECAY_THRESHOLD", 0.40)
        self.RESCORE_EXIT_DECAY_MIN_PROFIT = _env_float("RESCORE_EXIT_DECAY_MIN_PROFIT", 0.10)

        # ── Momentum Mode (per-loop CE/PE direction) ──
        self.MOMENTUM_MODE_ENABLED = _env_bool("MOMENTUM_MODE_ENABLED", False)
        self.MOMENTUM_MIN_SCORE_DIFF = _env_float("MOMENTUM_MIN_SCORE_DIFF", 1.5)
        self.MOMENTUM_CE_MIN_PROB = _env_float("MOMENTUM_CE_MIN_PROB", 0.55)
        self.MOMENTUM_PE_MIN_PROB = _env_float("MOMENTUM_PE_MIN_PROB", 0.60)

        # ── Adaptive Fuzzy Threshold ──
        self.ADAPTIVE_FUZZY_ENABLED = _env_bool("ADAPTIVE_FUZZY_ENABLED", False)
        self.ADAPTIVE_FUZZY_STRONG_SCORE = _env_float("ADAPTIVE_FUZZY_STRONG_SCORE", 3.5)
        self.ADAPTIVE_FUZZY_STRONG_THRESHOLD = _env_float("ADAPTIVE_FUZZY_STRONG_THRESHOLD", 1.5)
        self.ADAPTIVE_FUZZY_MID_SCORE = _env_float("ADAPTIVE_FUZZY_MID_SCORE", 2.5)
        self.ADAPTIVE_FUZZY_MID_THRESHOLD = _env_float("ADAPTIVE_FUZZY_MID_THRESHOLD", 1.75)
        self.ADAPTIVE_FUZZY_CUTOFF_HOUR = _env_int("ADAPTIVE_FUZZY_CUTOFF_HOUR", 13)

        # ── PE Confidence Filter (PE model as gate) ──
        self.PE_FILTER_ENABLED = _env_bool("PE_FILTER_ENABLED", True)
        self.PE_FILTER_THRESHOLD = _env_float("PE_FILTER_THRESHOLD", 0.70)
        self.PE_FILTER_TOLERANCE_LOW = _env_float("PE_FILTER_TOLERANCE_LOW", 0.60)
        self.PE_FILTER_TOLERANCE_SCORE = _env_float("PE_FILTER_TOLERANCE_SCORE", 3.0)
        self.PE_FILTER_TOLERANCE_VIX_RISE = _env_float("PE_FILTER_TOLERANCE_VIX_RISE", 0.5)

        # ── CE Confidence Filter (CE model as gate) ──
        self.CE_FILTER_ENABLED = _env_bool("CE_FILTER_ENABLED", True)
        self.CE_FILTER_THRESHOLD = _env_float("CE_FILTER_THRESHOLD", 0.65)
        self.CE_FILTER_TOLERANCE_LOW = _env_float("CE_FILTER_TOLERANCE_LOW", 0.50)
        self.CE_FILTER_TOLERANCE_SCORE = _env_float("CE_FILTER_TOLERANCE_SCORE", 3.25)
        self.CE_FILTER_TOLERANCE_VIX_FALL = _env_float("CE_FILTER_TOLERANCE_VIX_FALL", 0.5)

        # ── Momentum Decay Exit ──
        self.MOMENTUM_DECAY_ENABLED = _env_bool("MOMENTUM_DECAY_ENABLED", True)
        self.MOMENTUM_DECAY_FACTOR = _env_float("MOMENTUM_DECAY_FACTOR", 0.60)
        self.MOMENTUM_DECAY_RSI_DROP = _env_float("MOMENTUM_DECAY_RSI_DROP", 8.0)
        self.MOMENTUM_DECAY_MIN_PROFIT = _env_float("MOMENTUM_DECAY_MIN_PROFIT", 0.10)
        self.LATE_WEAK_EXIT_ENABLED = _env_bool("LATE_WEAK_EXIT_ENABLED", True)
        self.LATE_WEAK_EXIT_TIME = _env("LATE_WEAK_EXIT_TIME", "14:45")
        self.LATE_WEAK_EXIT_MAX_PROFIT = _env_float("LATE_WEAK_EXIT_MAX_PROFIT", 0.05)

        # ── Kelly Sizing ──
        self.KELLY_ENABLED = _env_bool("KELLY_ENABLED", True)
        self.KELLY_WINDOW = _env_int("KELLY_WINDOW", 20)
        self.KELLY_MIN_TRADES = _env_int("KELLY_MIN_TRADES", 10)
        self.KELLY_MIN_MULT = _env_float("KELLY_MIN_MULT", 0.50)
        self.KELLY_MAX_MULT = _env_float("KELLY_MAX_MULT", 1.50)

        # ── Bidirectional Reversal ──
        self.REVERSAL_ENABLED = _env_bool("REVERSAL_ENABLED", False)
        self.REVERSAL_MIN_SCORE = _env_float("REVERSAL_MIN_SCORE", 2.5)
        self.REVERSAL_SIZE_MULT = _env_float("REVERSAL_SIZE_MULT", 0.75)
        self.REVERSAL_MIN_EXIT_PROFIT = _env_float("REVERSAL_MIN_EXIT_PROFIT", 0.08)

        # ── Volatile Dual Mode ──
        self.DUAL_MODE_ENABLED = _env_bool("DUAL_MODE_ENABLED", False)
        self.DUAL_MODE_MIN_SCORE = _env_float("DUAL_MODE_MIN_SCORE", 2.0)
        self.DUAL_MODE_SIZE_MULT = _env_float("DUAL_MODE_SIZE_MULT", 0.60)
        self.DUAL_MODE_SL_PCT = _env_float("DUAL_MODE_SL_PCT", 0.40)
        self.DUAL_MODE_TP_PCT = _env_float("DUAL_MODE_TP_PCT", 0.30)
        self.DUAL_MODE_ENTRY_CUTOFF = _env("DUAL_MODE_ENTRY_CUTOFF", "12:00")

        # ── API Timeout ──
        self.API_TIMEOUT_SECONDS = _env_int("API_TIMEOUT_SECONDS", 10)

        # ── WebSocket Feed ──
        self.WEBSOCKET_ENABLED = _env_bool("WEBSOCKET_ENABLED", True)
        self.WEBSOCKET_RECONNECT_DELAY = _env_int("WEBSOCKET_RECONNECT_DELAY", 5)

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
        logger.info(f"  Fixed-R Sizing: {self.FIXED_R_SIZING}")
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
            if self.IC_ENABLED:
                logger.info(
                    f"  Iron Condor: width={self.IC_SPREAD_WIDTH} min_credit={self.IC_MIN_CREDIT} "
                    f"TP={self.IC_TP_PCT}% SL={self.IC_SL_MULTIPLIER}x"
                )
        logger.info("=" * 50)
