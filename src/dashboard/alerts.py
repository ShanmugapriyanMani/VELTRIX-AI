"""
Telegram Alerts — Trade alerts, regime changes, circuit breakers, daily reports.

Bot commands: /status /positions /pnl /regime /kill
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Any

import requests
import yaml
from loguru import logger


class TelegramAlerts:
    """
    Sends trading alerts and reports via Telegram bot.

    Alert types:
    - Trade execution alerts (with strategy + regime context)
    - Regime change alerts
    - Circuit breaker alerts
    - Daily and weekly performance reports

    Bot commands:
    - /status — Bot status, current regime, open positions
    - /positions — List all open positions with P&L
    - /pnl — Today's P&L summary
    - /regime — Current market regime details
    - /kill — Activate kill switch (emergency)
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        tg_cfg = config.get("alerts", {}).get("telegram", {})
        self.enabled = tg_cfg.get("enabled", False)
        # Env vars take priority over config file
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", tg_cfg.get("bot_token", ""))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", tg_cfg.get("chat_id", ""))

        if self.enabled and (not self.bot_token or not self.chat_id):
            logger.warning("Telegram alerts enabled but token/chat_id missing")
            self.enabled = False

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            logger.debug(f"[TELEGRAM DISABLED] {text[:100]}")
            return False

        url = self.TELEGRAM_API.format(token=self.bot_token, method="sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            logger.error(f"Telegram send failed: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

        return False

    # ─────────────────────────────────────────
    # Trade Alerts
    # ─────────────────────────────────────────

    def alert_trade_entry(self, trade: dict[str, Any]) -> None:
        """Send trade entry alert with intraday confirmation details."""
        direction = trade.get("side", "")
        emoji = "🟢" if direction == "BUY" else "🔴"
        features = trade.get("features", {})

        text = (
            f"{emoji} <b>TRADE ENTRY</b>\n"
            f"{'━' * 25}\n"
            f"<b>{direction}</b> {trade.get('quantity', 0)} {trade.get('symbol', '')}\n"
            f"Price: ₹{trade.get('price', 0):,.2f}\n"
        )

        # Add intraday confirmation details if available
        triggers = features.get("confirmation_triggers")
        if triggers:
            ml_up = features.get("ml_prob_up", 0)
            ml_down = features.get("ml_prob_down", 0)
            ml_str = f"P(up)={ml_up:.1%}" if ml_up > ml_down else f"P(down)={ml_down:.1%}"
            text += (
                f"ML: {ml_str}\n"
                f"Bias: {features.get('morning_bias', 'N/A')}\n"
                f"Confirmed: {' + '.join(triggers)}\n"
                f"PCR: {features.get('current_pcr', 0):.2f} | "
                f"RSI: {features.get('intraday_rsi', 0):.0f} | "
                f"VWAP: {features.get('intraday_vwap', 0):,.0f}\n"
            )

        text += (
            f"Regime: {trade.get('regime', 'unknown')}\n"
            f"Confidence: {trade.get('confidence', 0):.1%}\n"
            f"SL: ₹{trade.get('stop_loss', 0):,.2f} | "
            f"TP: ₹{trade.get('take_profit', 0):,.2f}\n"
            f"Value: ₹{trade.get('price', 0) * trade.get('quantity', 0):,.0f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_message(text)

    def alert_trade_exit(self, trade: dict[str, Any]) -> None:
        """Send trade exit alert."""
        pnl = trade.get("pnl", 0)
        emoji = "✅" if pnl >= 0 else "❌"

        text = (
            f"{emoji} <b>TRADE EXIT</b>\n"
            f"{'━' * 25}\n"
            f"{trade.get('symbol', '')} — {trade.get('exit_reason', 'manual')}\n"
            f"Entry: ₹{trade.get('entry_price', 0):,.2f}\n"
            f"Exit: ₹{trade.get('exit_price', 0):,.2f}\n"
            f"P&L: ₹{pnl:+,.2f} ({trade.get('pnl_pct', 0):+.1f}%)\n"
            f"Held: {trade.get('hold_hours', 0):.1f} hrs\n"
            f"Strategy: {trade.get('strategy', '')}"
        )
        self._send_message(text)

    # ─────────────────────────────────────────
    # Regime Alerts
    # ─────────────────────────────────────────

    def alert_regime_change(self, old_regime: str, new_regime: str, details: dict) -> None:
        """Send regime change alert."""
        text = (
            f"⚡ <b>REGIME CHANGE</b>\n"
            f"{'━' * 25}\n"
            f"{old_regime} → <b>{new_regime}</b>\n\n"
            f"VIX: {details.get('vix', 0):.1f}\n"
            f"NIFTY: {details.get('nifty_price', 0):,.0f}\n"
            f"ADX: {details.get('adx', 0):.1f}\n"
            f"FII 5d: ₹{details.get('fii_net_5d', 0):,.0f}cr\n\n"
            f"Active Strategies: {', '.join(details.get('active_strategies', []))}\n"
            f"Size Multiplier: {details.get('size_multiplier', 1):.2f}x"
        )
        self._send_message(text)

    # ─────────────────────────────────────────
    # Circuit Breaker Alerts
    # ─────────────────────────────────────────

    def alert_circuit_breaker(self, status: dict[str, Any]) -> None:
        """Send circuit breaker alert."""
        state = status.get("state", "UNKNOWN")

        emoji_map = {
            "WARNING": "⚠️",
            "PAUSED": "⏸️",
            "HALTED": "🛑",
            "KILLED": "💀",
        }
        emoji = emoji_map.get(state, "🔔")

        text = (
            f"{emoji} <b>CIRCUIT BREAKER: {state}</b>\n"
            f"{'━' * 25}\n"
            f"Reason: {status.get('halt_reason', 'N/A')}\n"
            f"Daily Loss: {status.get('daily_loss_pct', 0):.1f}%\n"
            f"Drawdown: {status.get('drawdown_pct', 0):.1f}%\n"
            f"Consecutive Losses: {status.get('consecutive_losses', 0)}\n"
            f"Can Trade: {'Yes' if status.get('can_trade', True) else 'No'}"
        )
        self._send_message(text)

    # ─────────────────────────────────────────
    # Daily/Weekly Reports
    # ─────────────────────────────────────────

    def send_daily_report(self, report: dict[str, Any]) -> None:
        """Send end-of-day performance report."""
        text = (
            f"📊 <b>DAILY REPORT — {date.today().isoformat()}</b>\n"
            f"{'━' * 30}\n\n"
            f"<b>Portfolio</b>\n"
            f"  Value: ₹{report.get('total_value', 0):,.0f}\n"
            f"  Day P&L: ₹{report.get('day_pnl', 0):+,.0f}\n"
            f"  Total P&L: ₹{report.get('total_pnl', 0):+,.0f}\n"
            f"  Drawdown: {report.get('drawdown_pct', 0):.1f}%\n\n"
            f"<b>Trades</b>\n"
            f"  Executed: {report.get('trades_today', 0)}\n"
            f"  Won: {report.get('trades_won', 0)} | "
            f"Lost: {report.get('trades_lost', 0)}\n"
            f"  Win Rate: {report.get('win_rate', 0):.0f}%\n\n"
            f"<b>Regime</b>: {report.get('regime', 'N/A')}\n"
            f"<b>VIX</b>: {report.get('vix', 0):.1f}\n\n"
            f"<b>Strategy P&L</b>\n"
        )

        for strat, pnl in report.get("strategy_pnl", {}).items():
            text += f"  {strat}: ₹{pnl:+,.0f}\n"

        self._send_message(text)

    def send_weekly_report(self, report: dict[str, Any]) -> None:
        """Send weekly performance report."""
        text = (
            f"📈 <b>WEEKLY REPORT</b>\n"
            f"{'━' * 30}\n\n"
            f"Week P&L: ₹{report.get('week_pnl', 0):+,.0f}\n"
            f"Total P&L: ₹{report.get('total_pnl', 0):+,.0f}\n"
            f"Trades: {report.get('trades_week', 0)}\n"
            f"Win Rate: {report.get('win_rate', 0):.0f}%\n"
            f"Sharpe (rolling): {report.get('sharpe', 0):.2f}\n"
            f"Max Drawdown: {report.get('max_drawdown', 0):.1f}%\n\n"
            f"<b>Best Strategy</b>: {report.get('best_strategy', 'N/A')}\n"
            f"<b>ML Retrained</b>: {'Yes' if report.get('ml_retrained') else 'No'}"
        )
        self._send_message(text)

    # ─────────────────────────────────────────
    # Command Responses
    # ─────────────────────────────────────────

    def respond_status(self, status: dict[str, Any]) -> None:
        """Respond to /status command."""
        text = (
            f"🤖 <b>BOT STATUS</b>\n"
            f"{'━' * 25}\n"
            f"Mode: {status.get('mode', 'paper')}\n"
            f"Running: {'Yes' if status.get('running') else 'No'}\n"
            f"Regime: {status.get('regime', 'N/A')}\n"
            f"Positions: {status.get('positions', 0)}\n"
            f"Day P&L: ₹{status.get('day_pnl', 0):+,.0f}\n"
            f"Circuit Breaker: {status.get('breaker_state', 'NORMAL')}\n"
            f"Uptime: {status.get('uptime', 'N/A')}"
        )
        self._send_message(text)

    def respond_positions(self, positions: list[dict]) -> None:
        """Respond to /positions command."""
        if not positions:
            self._send_message("📋 No open positions")
            return

        text = f"📋 <b>OPEN POSITIONS ({len(positions)})</b>\n{'━' * 30}\n\n"
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            text += (
                f"{emoji} <b>{pos.get('symbol', '')}</b>\n"
                f"  Qty: {pos.get('quantity', 0)} | "
                f"Entry: ₹{pos.get('entry_price', 0):,.2f}\n"
                f"  Current: ₹{pos.get('current_price', 0):,.2f} | "
                f"P&L: ₹{pnl:+,.0f}\n\n"
            )

        self._send_message(text)

    def respond_pnl(self, pnl_data: dict[str, Any]) -> None:
        """Respond to /pnl command."""
        text = (
            f"💰 <b>P&L SUMMARY</b>\n"
            f"{'━' * 25}\n"
            f"Today: ₹{pnl_data.get('day_pnl', 0):+,.0f}\n"
            f"Week: ₹{pnl_data.get('week_pnl', 0):+,.0f}\n"
            f"Total: ₹{pnl_data.get('total_pnl', 0):+,.0f}\n"
            f"Unrealized: ₹{pnl_data.get('unrealized', 0):+,.0f}\n"
            f"Drawdown: {pnl_data.get('drawdown_pct', 0):.1f}%"
        )
        self._send_message(text)

    def respond_regime(self, regime_data: dict[str, Any]) -> None:
        """Respond to /regime command."""
        text = (
            f"🌍 <b>MARKET REGIME</b>\n"
            f"{'━' * 25}\n"
            f"Regime: <b>{regime_data.get('regime', 'N/A')}</b>\n"
            f"VIX: {regime_data.get('vix', 0):.1f}\n"
            f"NIFTY: {regime_data.get('nifty', 0):,.0f} "
            f"(vs MA50: {regime_data.get('nifty_ma50', 0):,.0f})\n"
            f"ADX: {regime_data.get('adx', 0):.1f}\n"
            f"FII 5d: ₹{regime_data.get('fii_5d', 0):,.0f}cr\n\n"
            f"Active: {', '.join(regime_data.get('active_strategies', []))}\n"
            f"Size Mult: {regime_data.get('size_multiplier', 1):.2f}x"
        )
        self._send_message(text)

    def alert_bot_started(self, mode: str, capital: float, ml_prediction: dict | None = None) -> None:
        """Send bot startup alert."""
        ml_text = ""
        if ml_prediction:
            prob_up = ml_prediction.get("prob_up", 0.5)
            prob_down = ml_prediction.get("prob_down", 0.5)
            direction = "CE (Bullish)" if prob_up > prob_down else "PE (Bearish)"
            ml_text = (
                f"\n<b>ML Prediction</b>\n"
                f"  Direction: {direction}\n"
                f"  P(up): {prob_up:.1%} | P(down): {prob_down:.1%}\n"
            )

        text = (
            f"🚀 <b>TRADING BOT STARTED</b>\n"
            f"{'━' * 25}\n"
            f"Mode: <b>{mode.upper()}</b>\n"
            f"Capital: ₹{capital:,.0f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{ml_text}"
        )
        self._send_message(text)

    def alert_bot_stopped(self) -> None:
        """Send bot shutdown alert."""
        text = (
            f"🛑 <b>TRADING BOT STOPPED</b>\n"
            f"{'━' * 25}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._send_message(text)

    def alert_kill_switch(self) -> None:
        """Alert that kill switch was activated."""
        text = (
            f"💀💀💀 <b>KILL SWITCH ACTIVATED</b> 💀💀💀\n"
            f"{'━' * 30}\n"
            f"All orders cancelled\n"
            f"All positions flattened\n"
            f"Trading HALTED\n\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
            f"Manual review required!"
        )
        self._send_message(text)
