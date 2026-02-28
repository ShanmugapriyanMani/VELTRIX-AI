"""
Main Orchestrator — Full daily trading cycle.

Schedule:
- 8:30 AM: Pre-market data collection (VIX via Upstox)
- 9:00 AM: Upstox login + authenticate
- 9:15 AM: Market open → start data stream
- 9:30 AM: Skip first 15 min → begin trading
- 9:30–15:10: Trading loop (regime → strategies → ensemble → execute)
- 15:15 PM: EOD square-off (intraday)
- 15:30 PM: Market close → daily report
- Saturday: ML model retraining

CLI: python src/main.py --mode live|paper|backtest|fetch
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal as sig
import sys
import time
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import yaml
from loguru import logger


def _load_dotenv(env_path: str = ".env") -> None:
    """Load .env file into os.environ (no external dependency needed)."""
    path = Path(env_path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # Don't override existing env vars
                os.environ[key] = value

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    serialize=True,
)

from src.data.fetcher import UpstoxDataFetcher
from src.data.store import DataStore
from src.data.features import FeatureEngine
from src.regime.detector import RegimeDetector
from src.data.options_instruments import OptionsInstrumentResolver
from src.strategies.fii_flow import FIIFlowStrategy
from src.strategies.options_oi import OptionsOIStrategy
from src.strategies.options_buyer import OptionsBuyerStrategy
from src.strategies.delivery_volume import DeliveryVolumeStrategy
from src.strategies.ml_predictor import MLPredictorStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.risk.manager import RiskManager
from src.risk.portfolio import PortfolioManager, Position
from src.risk.circuit_breaker import CircuitBreaker
from src.execution.upstox_broker import UpstoxBroker
from src.execution.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager
from src.dashboard.alerts import TelegramAlerts
from src.utils.market_calendar import is_expiry_day, is_expiry_week


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates all components: data → regime → strategy → risk → execution.
    """

    def __init__(self, mode: str = "paper", config_path: str = "config/config.yaml"):
        self.mode = mode
        self.config_path = config_path
        self._running = False
        self._skip_wait = False
        self._force_fetch = False
        self._fetch_expired = False
        self._active_trading = False
        self._start_time = datetime.now()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        capital = self.config["trading"]["capital"]

        # ── Initialize components ──
        logger.info(f"Initializing Trading Bot (mode={mode}, capital=₹{capital:,.0f})")

        self.store = DataStore(config_path)
        self.data_fetcher = UpstoxDataFetcher(config_path, store=self.store)
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()

        # Strategies
        self.fii_strategy = FIIFlowStrategy()
        self.options_strategy = OptionsOIStrategy()
        self.delivery_strategy = DeliveryVolumeStrategy()
        self.ml_strategy = MLPredictorStrategy()

        # Options buying strategy + instrument resolver
        self.options_resolver = OptionsInstrumentResolver()
        self.options_buyer = OptionsBuyerStrategy()
        self.options_buyer.set_resolver(self.options_resolver)
        self.options_buyer.set_data_fetcher(self.data_fetcher)

        # Ensemble
        self.ensemble = EnsembleStrategy()
        self.ensemble.register_strategy(self.fii_strategy)
        self.ensemble.register_strategy(self.options_strategy)
        self.ensemble.register_strategy(self.delivery_strategy)
        self.ensemble.register_strategy(self.ml_strategy)
        self.ensemble.register_strategy(self.options_buyer)
        self.ensemble.set_regime_detector(self.regime_detector)

        # Risk
        self.risk_manager = RiskManager()
        self.portfolio = PortfolioManager(initial_capital=capital)
        self.circuit_breaker = CircuitBreaker()

        # Execution
        if mode == "live":
            self.broker = UpstoxBroker(config_path)
        else:
            self.broker = PaperTrader(initial_capital=capital, data_fetcher=self.data_fetcher)

        self.order_manager = OrderManager(
            self.broker, self.risk_manager, self.circuit_breaker, config_path
        )

        # Alerts
        self.alerts = TelegramAlerts(config_path)

        # Graceful shutdown
        sig.signal(sig.SIGINT, self._shutdown_handler)
        sig.signal(sig.SIGTERM, self._shutdown_handler)

        # Options direction ML state (trained daily before market open)
        self._options_ml_prob_up = 0.5
        self._options_ml_prob_down = 0.5

        logger.info("Trading Bot initialized successfully")

    def _shutdown_handler(self, signum, _frame):
        """Handle graceful shutdown. Second Ctrl+C forces immediate exit."""
        if not self._running:
            # Already shutting down — force exit
            logger.warning("Force shutdown (second signal) — exiting immediately")
            raise SystemExit(1)
        logger.warning(f"Shutdown signal received ({signum}) — finishing current iteration then stopping")
        self._running = False

    def run(self) -> None:
        """Main entry point — run the full daily trading cycle."""
        self._running = True

        if self.mode == "fetch":
            self._run_fetch()
            return

        if self.mode == "backtest":
            self._run_backtest()
            return
        logger.info(f"Starting Trading Bot in {self.mode} mode")

        try:
            # ── Phase 0: Auto-fetch all data (incremental — fast when up to date) ──
            logger.info("=== AUTO-FETCH: Updating all data sources ===")
            self._run_fetch()
            logger.info("=== AUTO-FETCH COMPLETE ===")

            # ── Phase 0b: Train options direction ML model ──
            self._train_options_direction_ml()

            # ── Send bot started alert ──
            self.alerts.alert_bot_started(
                mode=self.mode,
                capital=self.config["trading"]["capital"],
                ml_prediction={
                    "prob_up": self._options_ml_prob_up,
                    "prob_down": self._options_ml_prob_down,
                },
            )

            # ── Phase 1: Pre-market data collection (8:30 AM) ──
            self._pre_market_data()

            # ── Phase 2: Connect to broker (9:00 AM) ──
            self._connect_broker()

            # ── Phase 3: Market hours trading loop ──
            self._trading_loop()

            # ── Phase 4: Post-market (save EOD data) ──
            self._post_market()

            # ── Phase 5: Save EOD candle data for future backtests ──
            self._save_eod_candle_data()

        except Exception as e:
            logger.critical(f"Bot crashed: {e}", exc_info=True)
            self.alerts.alert_circuit_breaker({
                "state": "KILLED",
                "halt_reason": f"Bot crash: {str(e)[:100]}",
            })
        finally:
            self.alerts.alert_bot_stopped()
            self._cleanup()

    def _pre_market_data(self) -> None:
        """Collect pre-market data (Upstox-only mode)."""
        logger.info("=== PRE-MARKET DATA COLLECTION ===")

        # Wait until 8:30 AM if too early
        self._wait_until(dt_time(8, 30))

        # Load FII history from DB for strategies (already fetched in auto-fetch phase)
        self.fii_history = self.store.get_fii_dii_history(days=30)
        fii_cov = self.store.get_fii_dii_coverage()
        if fii_cov["rows"] > 0:
            logger.info(
                f"FII/DII: {fii_cov['rows']} days in DB "
                f"({fii_cov['from_date']} to {fii_cov['to_date']})"
            )
        else:
            logger.info("FII/DII: no historical data in DB yet")

        # VIX from Upstox
        self.vix_data = self.data_fetcher.get_current_vix()
        self._vix_last_fetch: float = time.time()
        logger.info(f"India VIX: {self.vix_data.get('vix', 0):.1f}")

        # Delivery data — not available from Upstox
        self.delivery_data = pd.DataFrame()

        # Futures premium — not available without NSE
        self.futures_premium = {"spot": 0, "futures": 0, "premium": 0, "premium_pct": 0}

        # FII flow consecutive tracking (used by FII strategy, disabled but tracked)
        self.fii_consecutive = {"direction": "neutral", "consecutive_days": 0, "total_flow_cr": 0}
        if not self.fii_history.empty and "fii_net_value" in self.fii_history.columns:
            recent = self.fii_history.tail(5)
            nets = recent["fii_net_value"].tolist()
            if all(n > 0 for n in nets if n != 0):
                self.fii_consecutive = {
                    "direction": "buy", "consecutive_days": len(nets),
                    "total_flow_cr": sum(nets),
                }
            elif all(n < 0 for n in nets if n != 0):
                self.fii_consecutive = {
                    "direction": "sell", "consecutive_days": len(nets),
                    "total_flow_cr": sum(nets),
                }

        # Expiry week check (pure calendar logic)
        self.is_expiry_week = is_expiry_week()

        # Delivery divergences — empty
        self.delivery_divergences = {"accumulation": [], "distribution": []}

        # Options instrument master (needed for expiry dates below)
        self.options_resolver.refresh()

        # Options OI/PCR/MaxPain from Upstox option chain API
        try:
            nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
            next_expiry = self.options_resolver.get_weekly_expiry("NIFTY")
            nifty_chain = self.data_fetcher.get_option_chain(nifty_key, next_expiry.isoformat())
            self.oi_levels = nifty_chain.get("oi_levels", {})
            self.pcr_data = nifty_chain.get("pcr", {"pcr_oi": 0, "pcr_volume": 0, "pcr_change_oi": 0})
            self.max_pain = nifty_chain.get("max_pain", {})
        except Exception as e:
            logger.warning(f"NIFTY option chain failed: {e}")
            self.oi_levels = {}
            self.pcr_data = {"pcr_oi": 0, "pcr_volume": 0, "pcr_change_oi": 0}
            self.max_pain = {}

        # BANKNIFTY option chain from Upstox
        try:
            banknifty_key = self.config["universe"]["indices"]["BANKNIFTY"]["instrument_key"]
            bn_expiry = self.options_resolver.get_weekly_expiry("BANKNIFTY")
            bn_chain = self.data_fetcher.get_option_chain(banknifty_key, bn_expiry.isoformat())
            self.banknifty_oi_levels = bn_chain.get("oi_levels", {})
            self.banknifty_pcr = bn_chain.get("pcr", {"pcr_oi": 0, "pcr_volume": 0, "pcr_change_oi": 0})
        except Exception as e:
            logger.warning(f"BANKNIFTY option chain failed: {e}")
            self.banknifty_oi_levels = {}
            self.banknifty_pcr = {"pcr_oi": 0, "pcr_volume": 0, "pcr_change_oi": 0}

        logger.info("Pre-market data collection complete")

    def _connect_broker(self) -> None:
        """Connect to broker."""
        logger.info("=== BROKER CONNECTION ===")

        if not self.broker.connect():
            if self.mode == "live":
                logger.critical("Failed to connect to Upstox. Aborting.")
                self._running = False
                return
            logger.warning("Broker connection failed (paper mode continues)")

    def _trading_loop(self) -> None:
        """Main trading loop during market hours."""
        logger.info("=== TRADING LOOP STARTED ===")

        skip_minutes = self.config["trading"]["market_hours"].get("skip_first_minutes", 15)
        trade_start = dt_time(9, 15 + skip_minutes)
        trade_end = dt_time(15, 10)

        # Wait for market open + skip period
        self._wait_until(trade_start)

        # Reset options buyer daily state
        self.options_buyer.reset_daily()

        # Refresh instrument master daily (ensures today's strikes are available)
        self.options_resolver.refresh()

        # Fetch NIFTY historical for regime detection + options strategy
        nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
        nifty_df = self.data_fetcher.get_historical_candles(nifty_key, "day")
        self._nifty_df = nifty_df  # Store for options_buyer fallback

        # ── Regime Detection ──
        regime_state = self.regime_detector.detect(
            vix_data=self.vix_data,
            nifty_df=nifty_df,
            fii_data=self.fii_history,
            is_expiry_week=self.is_expiry_week,
        )
        self.store.save_regime(regime_state.to_dict())

        # Alert on regime
        self.alerts.alert_regime_change(
            "INIT", regime_state.regime.value, regime_state.to_dict()
        )

        iteration = 0
        prev_regime = regime_state.regime
        self._regime_updates_done: set[str] = set()  # Track which intraday regime updates ran

        while self._running and datetime.now().time() < trade_end:
            iteration += 1

            try:
                # ── Check circuit breaker ──
                daily_loss_pct = abs(min(0, self.portfolio.get_day_pnl())) / max(self.portfolio.total_value, 1) * 100
                breaker_status = self.circuit_breaker.check(
                    daily_loss_pct=daily_loss_pct,
                    drawdown_pct=self.portfolio.drawdown,
                    open_positions=len(self.portfolio.positions),
                )

                if not self.circuit_breaker.can_trade():
                    logger.warning(f"Trading halted: {breaker_status.state.value}")
                    time.sleep(60)
                    continue

                # ── Update regime at fixed intraday intervals ──
                # 11:00, 13:00, 14:30 — not just iteration-based
                now_time = datetime.now().time()
                regime_update_times = [dt_time(11, 0), dt_time(13, 0), dt_time(14, 30)]
                should_update_regime = False
                for rut in regime_update_times:
                    # Check if we're within 1 minute of the update time
                    rut_dt = datetime.combine(date.today(), rut)
                    if abs((datetime.now() - rut_dt).total_seconds()) < 60:
                        regime_check_key = f"regime_{rut.strftime('%H%M')}"
                        if regime_check_key not in self._regime_updates_done:
                            self._regime_updates_done.add(regime_check_key)
                            should_update_regime = True
                            break

                if should_update_regime:
                    # Fetch fresh intraday data for regime update
                    try:
                        regime_intraday = self.data_fetcher.get_intraday_candles(
                            self.config["universe"]["indices"]["NIFTY50"]["instrument_key"],
                            "5minute",
                        )
                    except Exception:
                        regime_intraday = None

                    regime_state = self.regime_detector.detect(
                        vix_data=self.vix_data,
                        nifty_df=nifty_df,
                        fii_data=self.fii_history,
                        is_expiry_week=self.is_expiry_week,
                        intraday_df=regime_intraday,
                    )

                    if regime_state.regime != prev_regime:
                        self.alerts.alert_regime_change(
                            prev_regime.value, regime_state.regime.value,
                            regime_state.to_dict(),
                        )
                        prev_regime = regime_state.regime
                        logger.info(f"REGIME UPDATE at {now_time.strftime('%H:%M')}: {regime_state.regime.value}")

                # ── Prepare data for strategies ──
                data = self._prepare_strategy_data(regime_state)

                # ── Status log every 10 iterations (~5 min) ──
                if iteration % 10 == 1:
                    n_positions = len(self.portfolio.positions)
                    traded = list(self.options_buyer._traded_today)
                    # Build position P&L summary
                    pos_info = ""
                    for sym, pos in self.portfolio.positions.items():
                        pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
                        pos_info += f" | {sym}: ₹{pos.current_price:.0f} ({pnl_pct:+.1f}%)"
                    logger.info(
                        f"[Loop #{iteration}] NIFTY={data.get('nifty_price', 0):.0f} "
                        f"VIX={data.get('vix', 0):.1f} "
                        f"regime={data.get('regime', '?')} "
                        f"positions={n_positions} traded={traded} "
                        f"consec_sl={self.options_buyer._consec_sl_count} "
                        f"streak={self.options_buyer._streak}"
                        f"{pos_info}"
                    )

                # ── Data quality gate — block new trades if feeds are bad ──
                if not data.get("data_quality_ok", True):
                    # Still monitor existing positions, just don't open new ones
                    ensemble_result = {"decisions": []}
                else:
                    # ── Generate ensemble signals ──
                    ensemble_result = self.ensemble.generate_ensemble_signals(
                        data, regime_state
                    )

                # ── Execute decisions (options only — no equity) ──
                for decision in ensemble_result.get("decisions", []):
                    if decision["direction"] == "HOLD":
                        continue

                    symbol = decision["symbol"]
                    features = decision.get("features", {})
                    is_options = features.get("is_options", False)

                    # Skip equity signals — only trade options (CE/PE)
                    if not is_options:
                        continue

                    # ── Live mode: wallet balance check ──
                    if self.mode == "live":
                        try:
                            funds = self.broker.get_funds()
                            wallet_balance = float(funds.get("available_margin", 0))
                        except Exception as e:
                            logger.error(f"Failed to fetch wallet balance: {e}")
                            wallet_balance = 0

                        if wallet_balance < 20_000:
                            index_sym = features.get(
                                "index_symbol",
                                symbol.rstrip("0123456789").rstrip("CEPE"),
                            )
                            msg = (
                                f"TRADE BLOCKED — LOW BALANCE\n"
                                f"Wallet: ₹{wallet_balance:,.2f}\n"
                                f"Required: ₹20,000 minimum\n"
                                f"Signal: {symbol} {decision['direction']}\n"
                                f"Please add funds to resume trading."
                            )
                            logger.warning(f"BLOCKED: Wallet ₹{wallet_balance:,.0f} < ₹20,000")
                            self.alerts._send_message(msg)
                            self.options_buyer.cancel_signal(index_sym)
                            continue

                    # Options signals carry their own instrument_key
                    inst_key = features.get("instrument_key", "")

                    # Use premium from signal (strategy already fetched it)
                    # Fallback: fetch live if signal has price=0
                    premium = decision.get("price", 0)
                    if is_options and premium <= 0 and inst_key:
                        try:
                            ltp_data = self.data_fetcher.get_live_quote(inst_key)
                            premium = ltp_data.get("ltp", 0) if ltp_data else 0
                        except Exception as e:
                            logger.warning(f"Failed to fetch premium for {symbol}: {e}")

                    signal = {
                        "symbol": symbol,
                        "instrument_key": inst_key,
                        "direction": decision["direction"],
                        "price": premium,
                        "confidence": decision.get("confidence", 0),
                        "stop_loss": decision.get("stop_loss", 0),
                        "take_profit": decision.get("take_profit", 0),
                        "strategy": decision.get("strategy", "ensemble"),
                        "regime": decision.get("regime", ""),
                        "size_multiplier": decision.get("size_multiplier", 1.0),
                        "atr": premium * 0.02 if premium > 0 else 0,
                        "sector": "INDEX_OPTIONS" if is_options else (self.data_fetcher.get_sector_for_symbol(symbol) or ""),
                        "features": features,
                    }

                    if is_options and premium <= 0:
                        index_sym = features.get("index_symbol", symbol.rstrip("0123456789").rstrip("CEPE"))
                        logger.warning(f"Options: Cannot fetch premium for {symbol} ({inst_key}) — skipping")
                        self.options_buyer.cancel_signal(index_sym)
                        continue

                    result = self.order_manager.execute_signal(
                        signal,
                        capital=self.portfolio.total_value,
                        current_positions=self.portfolio.get_positions_df(),
                    )

                    # Extract index symbol (e.g. "NIFTY25350PE" → "NIFTY")
                    index_sym = features.get("index_symbol", symbol.rstrip("0123456789").rstrip("CEPE"))

                    if result.get("status") == "success":
                        logger.info(
                            f"OPTIONS TRADE OPENED: {symbol} | premium=₹{result.get('price', 0):.1f} "
                            f"qty={result.get('quantity', 0)} SL=₹{result.get('stop_loss', 0):.1f} "
                            f"TP=₹{result.get('take_profit', 0):.1f}"
                        )
                        self.options_buyer.confirm_execution(index_sym)
                        self.alerts.alert_trade_entry(result)
                        self.store.save_trade(result)

                        # Add to portfolio
                        self.portfolio.add_position(Position(
                            symbol=symbol,
                            instrument_key=inst_key,
                            side=decision["direction"],
                            quantity=result.get("quantity", 0),
                            entry_price=result.get("price", 0),
                            current_price=result.get("price", 0),
                            stop_loss=result.get("stop_loss", 0),
                            take_profit=result.get("take_profit", 0),
                            strategy=decision.get("strategy", "ensemble"),
                            sector="INDEX_OPTIONS" if is_options else signal.get("sector", ""),
                            trade_id=result.get("trade_id", ""),
                        ))
                    else:
                        logger.warning(
                            f"OPTIONS EXECUTION FAILED: {symbol} | status={result.get('status')} "
                            f"reason={result.get('reason', 'unknown')} premium=₹{premium:.1f}"
                        )
                        self.options_buyer.cancel_signal(index_sym)

                # ── Update live premiums for open positions ──
                if self.portfolio.positions:
                    live_prices = {}
                    for sym, pos in self.portfolio.positions.items():
                        if pos.instrument_key.startswith("NSE_FO|"):
                            try:
                                quote = self.data_fetcher.get_live_quote(pos.instrument_key)
                                ltp = quote.get("ltp", 0) if quote else 0
                                if ltp > 0:
                                    live_prices[sym] = ltp
                            except Exception:
                                pass
                    if live_prices:
                        self.portfolio.update_prices(live_prices)

                # ── Time-based exit adjustments on open positions ──
                # 2:00 PM → reduce TP by 20% (take what you can)
                # 2:45 PM → tighten SL to 15% (protect remaining capital)
                if now_time >= dt_time(14, 45) and self.portfolio.positions:
                    for sym, pos in self.portfolio.positions.items():
                        if pos.instrument_key.startswith("NSE_FO|") and pos.entry_price > 0:
                            tight_sl = pos.entry_price * 0.85  # 15% SL
                            if pos.stop_loss < tight_sl:
                                pos.stop_loss = tight_sl
                elif now_time >= dt_time(14, 0) and self.portfolio.positions:
                    for sym, pos in self.portfolio.positions.items():
                        if pos.instrument_key.startswith("NSE_FO|") and pos.entry_price > 0:
                            reduced_tp = pos.entry_price * (1 + (pos.take_profit / pos.entry_price - 1) * 0.80) if pos.take_profit > pos.entry_price else pos.take_profit
                            if reduced_tp < pos.take_profit:
                                pos.take_profit = reduced_tp

                # ── Check stops on existing positions ──
                triggers = self.portfolio.check_stops()
                for trigger in triggers:
                    symbol = trigger["symbol"]
                    price = trigger.get("price", 0)
                    trade_result = self.portfolio.close_position(
                        symbol, price, trigger["type"]
                    )
                    if trade_result:
                        self.alerts.alert_trade_exit(trade_result)
                        self.store.save_trade(trade_result)
                        self.circuit_breaker.record_trade(trade_result["pnl"])
                        # Record exit for options SL/TP tracking
                        option_type = trade_result.get("option_type", "")
                        self.options_buyer.record_exit(
                            symbol, trigger["type"], option_type
                        )

                # ── Check trailing stops on options positions ──
                trail_exits = self.order_manager.check_trailing_stops()
                for exit_info in trail_exits:
                    logger.info(
                        f"TRAIL EXIT: {exit_info['symbol']} | "
                        f"P&L={exit_info['pnl_pct']:+.1f}% | "
                        f"entry=₹{exit_info['entry_premium']:.0f} "
                        f"exit=₹{exit_info['exit_premium']:.0f}"
                    )
                    self.alerts.alert_trade_exit(exit_info)
                    self.options_buyer.record_exit(
                        exit_info["symbol"], "trail_stop",
                        exit_info.get("option_type", "")
                    )

                # ── Force-exit options positions at 15:10 ──
                if self.options_buyer.should_force_exit():
                    for pos in list(self.portfolio.positions):
                        if pos.instrument_key.startswith("NSE_FO|"):
                            logger.info(f"OPTIONS FORCE EXIT: {pos.symbol}")
                            self.broker.place_order(
                                symbol=pos.symbol,
                                instrument_key=pos.instrument_key,
                                quantity=pos.quantity,
                                side="SELL",
                                order_type="MARKET",
                                product="I",
                            )
                            trade_result = self.portfolio.close_position(
                                pos.symbol, pos.current_price, "force_exit_1510"
                            )
                            if trade_result:
                                self.alerts.alert_trade_exit(trade_result)
                                self.store.save_trade(trade_result)
                                self.options_buyer.record_exit(
                                    pos.symbol, "force_exit", ""
                                )

                # ── Reconcile orders ──
                self.order_manager.reconcile_orders()

                # ── Save portfolio snapshot ──
                if iteration % 10 == 0:
                    self.store.save_portfolio_snapshot(self.portfolio.get_snapshot())

            except Exception as e:
                logger.error(f"Trading loop error (iter {iteration}): {e}")

            # Sleep between iterations — interruptible for graceful shutdown
            if self.data_fetcher.is_network_down:
                sleep_secs = 60
            else:
                sleep_secs = 30
            # Sleep in 5s chunks so Ctrl+C / _running=False is responsive
            while sleep_secs > 0 and self._running:
                time.sleep(min(sleep_secs, 5))
                sleep_secs -= 5

        # ── EOD Square-off ──
        self.order_manager.check_eod_squareoff()

    def _prepare_strategy_data(self, regime_state) -> dict[str, Any]:
        """Prepare data bundle for all strategies."""
        nifty_df = getattr(self, "_nifty_df", None)
        nifty_price = 0
        nifty_ema_20 = 0
        if nifty_df is not None and not nifty_df.empty:
            # Add technical features for multi-factor scoring (computed once)
            if len(nifty_df) >= 50 and "ema_9" not in nifty_df.columns:
                nifty_df = self.feature_engine.add_technical_features(nifty_df)
                self._nifty_df = nifty_df
            nifty_price = float(nifty_df["close"].iloc[-1])
            if len(nifty_df) >= 20:
                nifty_ema_20 = float(nifty_df["close"].ewm(span=20).mean().iloc[-1])
            else:
                nifty_ema_20 = nifty_price

        banknifty_price = 0

        # Fetch 5-min intraday candles for options buyer confirmation
        nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
        try:
            intraday_df = self.data_fetcher.get_intraday_candles(nifty_key, "5minute")
        except Exception:
            intraday_df = pd.DataFrame()

        # Use LIVE price from intraday candles (not stale daily close)
        if not intraday_df.empty:
            nifty_price = float(intraday_df["close"].iloc[-1])

        # Refresh VIX every 60s (not every iteration — avoids API spam)
        if time.time() - self._vix_last_fetch >= 60:
            try:
                live_vix = self.data_fetcher.get_current_vix()
                current_vix = live_vix.get("vix", 0)
                if current_vix > 0:
                    self.vix_data = live_vix
                    self._vix_last_fetch = time.time()
            except Exception:
                pass
        current_vix = self.vix_data.get("vix", 15)

        # ── Data quality guards ──
        # VIX stale > 30 min → block trades
        vix_stale = (time.time() - self._vix_last_fetch) > 1800
        # NIFTY price unchanged 5+ min (frozen feed) → block trades
        nifty_frozen = False
        if not intraday_df.empty and len(intraday_df) >= 10:
            last_10_closes = intraday_df["close"].tail(10)
            nifty_frozen = last_10_closes.nunique() <= 1
        data_quality_ok = not vix_stale and not nifty_frozen
        if vix_stale:
            logger.warning(f"DATA QUALITY: VIX stale ({(time.time() - self._vix_last_fetch) / 60:.0f}min) — blocking trades")
        if nifty_frozen:
            logger.warning("DATA QUALITY: NIFTY price frozen (unchanged 5+ min) — blocking trades")

        return {
            "regime": regime_state.regime.value,
            "fii_consecutive": self.fii_consecutive,
            "fii_history": self.fii_history,
            "oi_levels": {
                "NIFTY": self.oi_levels,
                "BANKNIFTY": self.banknifty_oi_levels,
            },
            "pcr": {
                "NIFTY": self.pcr_data.get("pcr_oi", 1.0) if isinstance(self.pcr_data, dict) else self.pcr_data,
                "BANKNIFTY": self.banknifty_pcr.get("pcr_oi", 1.0) if isinstance(self.banknifty_pcr, dict) else self.banknifty_pcr,
            },
            "max_pain": self.max_pain,
            "option_chain": None,
            "is_expiry_day": is_expiry_day(),
            "is_expiry_week": self.is_expiry_week,
            "nifty_price": nifty_price,
            "banknifty_price": banknifty_price,
            "nifty_ema_20": nifty_ema_20,
            "banknifty_ema_20": banknifty_price,
            "nifty_df": nifty_df,
            "vix": current_vix,
            "delivery_divergences": self.delivery_divergences,
            "delivery_data": self.delivery_data,
            "stock_universe": self._build_stock_universe(),
            "stock_prices": {},
            "ml_direction_prob_up": self._options_ml_prob_up,
            "ml_direction_prob_down": self._options_ml_prob_down,
            "intraday_df": intraday_df,
            # Regime behavior parameters — controls how strategies trade
            "conviction_min": regime_state.conviction_min,
            "sl_multiplier": regime_state.sl_multiplier,
            "tp_multiplier": regime_state.tp_multiplier,
            "trailing_stop_enabled": regime_state.trailing_stop_enabled,
            "ema_weight": regime_state.ema_weight,
            "mean_reversion_weight": regime_state.mean_reversion_weight,
            "max_trades_per_day": regime_state.max_trades_per_day,
            "data_quality_ok": data_quality_ok,
        }

    def _build_stock_universe(self) -> dict[str, dict]:
        """Build stock universe with current data."""
        universe = {}
        for symbol, info in self.config["universe"]["nifty50"].items():
            universe[symbol] = {
                "instrument_key": info["instrument_key"],
                "sector": info.get("sector", ""),
                "isin": info.get("isin", ""),
                "price": 0,  # Will be updated with live data
                "atr": 0,
                "rsi": 50,
                "beta": 1.0,
                "fii_holding_pct": 20,  # Default; update from actual data
                "delivery_pct": 40,
            }
        return universe

    def _post_market(self) -> None:
        """Post-market activities."""
        logger.info("=== POST-MARKET ===")

        # Save final portfolio snapshot
        self.store.save_portfolio_snapshot(self.portfolio.get_snapshot())

        # Send daily report
        self.alerts.send_daily_report({
            "total_value": self.portfolio.total_value,
            "day_pnl": self.portfolio.get_day_pnl(),
            "total_pnl": self.portfolio.total_pnl,
            "drawdown_pct": self.portfolio.drawdown,
            "trades_today": len(self.portfolio.closed_trades),
            "trades_won": sum(1 for t in self.portfolio.closed_trades if t["pnl"] > 0),
            "trades_lost": sum(1 for t in self.portfolio.closed_trades if t["pnl"] <= 0),
            "win_rate": self.portfolio.closed_trades and sum(1 for t in self.portfolio.closed_trades if t["pnl"] > 0) / len(self.portfolio.closed_trades) * 100 or 0,
            "regime": self.regime_detector.current_regime.regime.value if self.regime_detector.current_regime else "N/A",
            "vix": self.vix_data.get("vix", 0),
            "strategy_pnl": {},
        })

        # Check if ML model needs retraining (Saturday)
        if self.ml_strategy.needs_retraining():
            self._retrain_ml_model()

        logger.info("Post-market activities complete")

    def _retrain_ml_model(self) -> None:
        """Weekly ML model retraining."""
        logger.info("=== ML MODEL RETRAINING ===")

        try:
            # Fetch full historical data
            nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
            nifty_df = self.data_fetcher.get_historical_candles(
                nifty_key, "day",
                from_date=(date.today() - timedelta(days=365)).isoformat(),
            )

            if nifty_df.empty:
                logger.warning("No data for ML retraining")
                return

            # Compute features (with external market data if available)
            external_df = self.store.get_external_data_all()
            feature_df = self.feature_engine.compute_all_features(
                nifty_df,
                fii_data=self.store.get_fii_dii_history(252),
                vix_data=self.vix_data,
                pcr_data=self.pcr_data,
                max_pain_data=self.max_pain,
                futures_premium=self.futures_premium,
                external_data=external_df if not external_df.empty else None,
            )

            # Prepare training data
            X, y = self.feature_engine.prepare_ml_dataset(feature_df)

            if len(X) < 100:
                logger.warning(f"Insufficient training data: {len(X)} samples")
                return

            # Train model
            metrics = self.ml_strategy.train(X, y)
            logger.info(f"ML retraining complete: {metrics}")

        except Exception as e:
            logger.error(f"ML retraining failed: {e}")

    def _train_options_direction_ml(self) -> None:
        """
        Train options direction ML model with 10-day retrain interval.

        Same walk-forward LightGBM binary classifier as backtest Factor 7:
        - 19 technical features from NIFTY daily candles (16 base + 3 momentum/range)
        - Binary target: next-day up (1) or down (0)
        - 90-day training window (matches backtest for stability)
        - Retrain every 10 days (matches backtest), load cached model otherwise
        - Always predicts today's direction using latest features
        """
        try:
            import lightgbm as lgb
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("LightGBM/sklearn not available; options ML disabled")
            return

        import pandas as pd

        model_path = Path("models/options_direction_model.pkl")
        scaler_path = Path("models/options_direction_scaler.pkl")

        # Check if saved model is fresh enough (< 10 days old)
        retrain_interval_days = 10
        need_retrain = True
        if model_path.exists() and scaler_path.exists():
            model_age_days = (date.today() - date.fromtimestamp(model_path.stat().st_mtime)).days
            if model_age_days < retrain_interval_days:
                need_retrain = False
                logger.info(
                    f"=== OPTIONS ML: Using cached model "
                    f"(age={model_age_days}d, retrain in {retrain_interval_days - model_age_days}d) ==="
                )

        if need_retrain:
            logger.info("=== OPTIONS DIRECTION ML TRAINING ===")

        try:
            # Fetch last 300 days of NIFTY daily candles from DB
            # Need 50 warmup (SMA50) + 90 training window = 140 trading days (~200 calendar days)
            nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
            from_date = (date.today() - timedelta(days=300)).isoformat()
            to_date = date.today().isoformat()
            nifty_df = self.data_fetcher.get_historical_candles(
                nifty_key, "day", from_date=from_date, to_date=to_date
            )

            if nifty_df is None or nifty_df.empty or len(nifty_df) < 65:
                logger.warning(
                    f"Insufficient NIFTY data for options ML: "
                    f"{len(nifty_df) if nifty_df is not None else 0} rows (need 65+)"
                )
                self._load_options_ml_model(model_path, scaler_path, nifty_df)
                return

            # Add technical features
            nifty_df = self.feature_engine.add_technical_features(nifty_df)

            # Add enhanced features (FII/DII, VIX historical, external markets)
            fii_df = self.store.get_fii_dii_history(days=365)
            external_df = self.store.get_external_data_all()

            # Load VIX as DataFrame from DB (same as backtest) — gives extended VIX stats
            # Avoids dependency on self.vix_data which isn't set until pre-market phase
            vix_key = self.config["universe"]["indices"].get(
                "INDIA_VIX", {}
            ).get("instrument_key", "NSE_INDEX|India VIX")
            vix_hist_df = self.data_fetcher.get_historical_candles(
                vix_key, "day", from_date=from_date, to_date=to_date
            )

            nifty_df = self.feature_engine.add_alternative_features(
                nifty_df,
                fii_data=fii_df if fii_df is not None and not fii_df.empty else None,
                vix_data=vix_hist_df if vix_hist_df is not None and not vix_hist_df.empty else None,
            )
            nifty_df = self.feature_engine.add_external_market_features(
                nifty_df,
                external_data=external_df if not external_df.empty else None,
            )

            # Add 3 extra momentum/range features (same as backtest)
            if "ret_5d" not in nifty_df.columns:
                nifty_df["ret_5d"] = nifty_df["close"].pct_change(5) * 100
            if "ret_20d" not in nifty_df.columns:
                nifty_df["ret_20d"] = nifty_df["close"].pct_change(20) * 100
            if "range_5d_pct" not in nifty_df.columns:
                nifty_df["range_5d_pct"] = (
                    (nifty_df["high"].rolling(5).max() - nifty_df["low"].rolling(5).min())
                    / nifty_df["close"] * 100
                )

            ml_features = [
                # Technical (16 — original)
                "rsi_14", "rsi_7", "macd_histogram", "macd_line",
                "bb_position", "bb_width", "atr_pct", "adx_14",
                "volatility_20d", "returns_1d", "returns_5d", "returns_20d",
                "price_to_sma50", "body_size", "upper_shadow", "lower_shadow",
                # Momentum/range (3 — original)
                "ret_5d", "ret_20d", "range_5d_pct",
                # FII/DII (5)
                "fii_net_flow_1d", "fii_net_flow_5d", "fii_flow_momentum",
                "fii_net_direction", "fii_net_streak",
                "dii_net_flow_1d", "india_vix",
                # VIX extended (3)
                "vix_change_pct", "vix_percentile_252d", "vix_5d_ma",
                # External markets (9)
                "sp500_prev_return", "nasdaq_prev_return", "crude_prev_return",
                "gold_prev_return", "usdinr_prev_return",
                "sp500_nifty_corr_20d", "crude_nifty_corr_20d",
                "dxy_momentum_5d", "global_risk_score",
            ]
            available_feats = [f for f in ml_features if f in nifty_df.columns]

            if len(available_feats) < 8:
                logger.warning(f"Only {len(available_feats)} ML features available (need 8+)")
                self._load_options_ml_model(model_path, scaler_path, nifty_df)
                return

            # ── NaN handling for ML features (NEVER drop rows) ──
            nifty_df[available_feats] = nifty_df[available_feats].ffill()
            for col in available_feats:
                median_val = nifty_df[col].median()
                if pd.notna(median_val):
                    nifty_df[col] = nifty_df[col].fillna(median_val)
                else:
                    nifty_df[col] = nifty_df[col].fillna(0.0)

            if need_retrain:
                # ── Retrain model (every 10 days) ──
                # Intraday target: close > open = bullish, exclude ±0.2% noise
                nifty_df["_intraday_ret"] = (nifty_df["close"] - nifty_df["open"]) / nifty_df["open"]
                nifty_df["_ml_target"] = pd.NA
                nifty_df.loc[nifty_df["_intraday_ret"] > 0.002, "_ml_target"] = 1
                nifty_df.loc[nifty_df["_intraday_ret"] < -0.002, "_ml_target"] = 0

                train_window = 120
                train_df = nifty_df.iloc[-(train_window + 1):-1]

                y_train = train_df["_ml_target"]
                valid = y_train.notna()
                # LAG FEATURES BY 1: use YESTERDAY's features to predict TODAY
                # Prevents look-ahead bias (returns_1d, body_size etc use today's close)
                X_train_raw = nifty_df.iloc[-(train_window + 2):-2][available_feats].copy()
                X_train_raw.index = train_df.index  # Align with target
                X_train = X_train_raw.loc[valid].copy()
                y_train = y_train[valid].astype(int)

                # Feature interactions
                for ca, cb, name in [
                    ("rsi_14", "india_vix", "rsi_x_vix"),
                    ("macd_histogram", "adx_14", "macd_x_adx"),
                    ("returns_1d", "volume_ratio", "ret_x_vol"),
                    ("bb_position", "rsi_14", "bb_x_rsi"),
                ]:
                    if ca in X_train.columns and cb in X_train.columns:
                        X_train[name] = X_train[ca] * X_train[cb]
                all_feats = list(available_feats) + [c for c in X_train.columns if "_x_" in c]

                if len(X_train) < 30 or len(y_train.unique()) < 2:
                    logger.warning(f"Insufficient training samples: {len(X_train)}")
                    self._load_options_ml_model(model_path, scaler_path, nifty_df)
                    return

                # Recency weighting
                import numpy as np
                n = len(X_train)
                sample_weights = np.ones(n)
                for wi in range(n):
                    days_ago = n - wi
                    if days_ago > 365:
                        sample_weights[wi] = 0.6

                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train[all_feats]),
                    columns=all_feats,
                    index=X_train.index,
                )

                model = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=60,
                    max_depth=3,
                    num_leaves=8,
                    learning_rate=0.08,
                    min_child_samples=25,
                    subsample=0.7,
                    colsample_bytree=0.5,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    verbose=-1,
                )
                model.fit(X_scaled, y_train, sample_weight=sample_weights)
                model._all_feats = all_feats

                # Save model to disk
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)

                logger.info(
                    f"Options ML model trained: {len(X_train)} samples, "
                    f"{len(available_feats)} features"
                )
            else:
                # ── Load cached model ──
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

            # ── Predict today's direction using YESTERDAY's features (no look-ahead) ──
            # At prediction time (9:15 AM), today's close/high/low don't exist yet
            yesterday_row = nifty_df.iloc[-2:-1] if len(nifty_df) >= 2 else nifty_df.iloc[-1:]
            feat_row = yesterday_row[available_feats].copy()
            nan_count = feat_row.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Today's features have {nan_count} NaN after fill; predicting anyway")
                feat_row = feat_row.fillna(0.0)

            # Add interaction features (must match training)
            for ca, cb, name in [
                ("rsi_14", "india_vix", "rsi_x_vix"),
                ("macd_histogram", "adx_14", "macd_x_adx"),
                ("returns_1d", "volume_ratio", "ret_x_vol"),
                ("bb_position", "rsi_14", "bb_x_rsi"),
            ]:
                if ca in feat_row.columns and cb in feat_row.columns:
                    feat_row[name] = feat_row[ca].values * feat_row[cb].values

            # Use all_feats from model if available, else reconstruct
            if hasattr(model, "_all_feats"):
                predict_feats = [f for f in model._all_feats if f in feat_row.columns]
            else:
                predict_feats = [f for f in feat_row.columns if f in available_feats or "_x_" in f]

            feat_scaled = pd.DataFrame(
                scaler.transform(feat_row[predict_feats]), columns=predict_feats
            )
            probas = model.predict_proba(feat_scaled)[0]
            self._options_ml_prob_up = float(probas[1]) if len(probas) > 1 else 0.5
            self._options_ml_prob_down = float(probas[0]) if len(probas) > 0 else 0.5

            logger.info(
                f"Options ML prediction: P(up)={self._options_ml_prob_up:.3f}, "
                f"P(down)={self._options_ml_prob_down:.3f}"
            )

        except Exception as e:
            logger.error(f"Options ML training failed: {e}", exc_info=True)
            self._load_options_ml_model(model_path, scaler_path, nifty_df=None)

    def _load_options_ml_model(self, model_path: Path, scaler_path: Path, nifty_df=None) -> None:
        """Load a previously saved options ML model and predict if possible."""
        if not model_path.exists() or not scaler_path.exists():
            logger.info("No saved options ML model found; using neutral prediction")
            return

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            if nifty_df is not None and not nifty_df.empty:
                # Use model's stored all_feats or feature_name_ from training
                if hasattr(model, "_all_feats"):
                    base_feats = [f for f in model._all_feats if f in nifty_df.columns or "_x_" in f]
                elif hasattr(model, "feature_name_"):
                    base_feats = [f for f in model.feature_name_ if f in nifty_df.columns or "_x_" in f]
                else:
                    ml_features = [
                        "rsi_14", "rsi_7", "macd_histogram", "macd_line",
                        "bb_position", "bb_width", "atr_pct", "adx_14",
                        "volatility_20d", "returns_1d", "returns_5d", "returns_20d",
                        "price_to_sma50", "body_size", "upper_shadow", "lower_shadow",
                        "ret_5d", "ret_20d", "range_5d_pct",
                        "fii_net_flow_1d", "fii_net_flow_5d", "fii_flow_momentum",
                        "fii_net_direction", "fii_net_streak",
                        "dii_net_flow_1d", "india_vix",
                        "vix_change_pct", "vix_percentile_252d", "vix_5d_ma",
                        "sp500_prev_return", "nasdaq_prev_return", "crude_prev_return",
                        "gold_prev_return", "usdinr_prev_return",
                        "sp500_nifty_corr_20d", "crude_nifty_corr_20d",
                        "dxy_momentum_5d", "global_risk_score",
                    ]
                    base_feats = [f for f in ml_features if f in nifty_df.columns]

                available = [f for f in base_feats if f in nifty_df.columns]
                if available:
                    import pandas as pd
                    today_row = nifty_df.iloc[-1:][available].copy()
                    nan_count = today_row.isna().sum().sum()
                    if nan_count > 0:
                        today_row = today_row.ffill().fillna(0.0)

                    # Add interaction features
                    for ca, cb, name in [
                        ("rsi_14", "india_vix", "rsi_x_vix"),
                        ("macd_histogram", "adx_14", "macd_x_adx"),
                        ("returns_1d", "volume_ratio", "ret_x_vol"),
                        ("bb_position", "rsi_14", "bb_x_rsi"),
                    ]:
                        if ca in today_row.columns and cb in today_row.columns:
                            today_row[name] = today_row[ca].values * today_row[cb].values

                    # Use all features model expects
                    if hasattr(model, "_all_feats"):
                        predict_feats = [f for f in model._all_feats if f in today_row.columns]
                    else:
                        predict_feats = list(today_row.columns)

                    feat_scaled = pd.DataFrame(
                        scaler.transform(today_row[predict_feats]), columns=predict_feats
                    )
                    probas = model.predict_proba(feat_scaled)[0]
                    self._options_ml_prob_up = float(probas[1]) if len(probas) > 1 else 0.5
                    self._options_ml_prob_down = float(probas[0]) if len(probas) > 0 else 0.5
                    logger.info(
                        f"Loaded saved options ML: P(up)={self._options_ml_prob_up:.3f}"
                    )
                    return

            logger.info("Loaded saved options ML model (no prediction yet)")
        except Exception as e:
            logger.warning(f"Failed to load saved options ML model: {e}")

    # ─────────────────────────────────────────
    # Data Fetch Mode
    # ─────────────────────────────────────────

    def _run_fetch(self) -> None:
        """
        Fetch mode — bulk data download from all sources.

        Steps 1-4: Upstox API (requires auth)
        Steps 5-8: Local CSVs, yfinance, nsepython (no auth needed)

        Incremental by default: checks DB for last stored date, only fetches new data.
        Use --force-fetch to ignore DB cache and re-fetch everything.

        Run this first: python src/main.py --mode fetch
        """
        from src.data.external_fetcher import ExternalDataFetcher

        force = self._force_fetch
        today_str = date.today().isoformat()

        logger.info("=" * 60)
        logger.info(f"=== DATA FETCH MODE {'(FORCE)' if force else '(INCREMENTAL)'} ===")
        logger.info("=" * 60)

        # ── Steps 1-4: Upstox API (requires auth) ──
        upstox_ok = self.data_fetcher.authenticate()
        if upstox_ok:
            fetch_days = 1095  # 3 years of data

            # 1. Fetch equity historical data (NIFTY 50)
            logger.info(f"--- [1/8] Fetching NIFTY 50 equity data ---")
            equity_skipped = 0
            equity_fetched = 0
            nifty50_syms = list(self.config["universe"].get("nifty50", {}).keys())
            for sym in nifty50_syms:
                if not force:
                    cov = self.store.get_data_coverage(sym)
                    if cov["rows"] > 0 and cov["to_date"] and cov["to_date"] >= (date.today() - timedelta(days=2)).isoformat():
                        equity_skipped += 1
                        continue
                inst_key = self.data_fetcher.get_instrument_for_symbol(sym)
                if inst_key:
                    try:
                        from_dt = (date.today() - timedelta(days=fetch_days)).isoformat()
                        df = self.data_fetcher.get_historical_candles(
                            inst_key, "day", from_date=from_dt, to_date=today_str
                        )
                        if df is not None and not df.empty:
                            equity_fetched += 1
                    except Exception as e:
                        logger.warning(f"  {sym}: fetch failed — {e}")
            logger.info(
                f"  Equity: {equity_fetched} fetched, {equity_skipped} skipped (up-to-date)"
            )

            # 2. Fetch index data (NIFTY, BANKNIFTY, VIX)
            logger.info("--- [2/8] Fetching index data ---")
            indices = self.config["universe"].get("indices", {})
            from_date = (date.today() - timedelta(days=fetch_days)).isoformat()

            for idx_name, idx_info in indices.items():
                idx_key = idx_info.get("instrument_key", "")
                if not idx_key:
                    continue

                # Check coverage — skip if up-to-date
                idx_symbol = self.data_fetcher._resolve_symbol(idx_key)
                if not force and idx_symbol:
                    cov = self.store.get_data_coverage(idx_symbol)
                    if cov["rows"] > 0 and cov["to_date"] and cov["to_date"] >= (date.today() - timedelta(days=2)).isoformat():
                        logger.info(f"  {idx_name}: up-to-date ({cov['to_date']}), skipping")
                        continue

                try:
                    df = self.data_fetcher.get_historical_candles(
                        idx_key, "day", from_date=from_date, to_date=today_str
                    )
                    if not df.empty:
                        logger.info(f"  {idx_name}: {len(df)} candles")
                    else:
                        logger.warning(f"  {idx_name}: no data")
                except Exception as e:
                    logger.error(f"  {idx_name}: failed — {e}")

            # 3. Fetch F&O option premium data (current + optionally expired)
            logger.info("--- [3/8] Fetching F&O option premium data ---")
            self._fetch_fno_premiums(
                force_fetch=force, fetch_expired=self._fetch_expired,
            )
        else:
            logger.warning("Upstox auth failed — skipping steps 1-3. Run: python scripts/auth_upstox.py")

        # ── Steps 5-8: External data (no auth needed) ──
        ext = ExternalDataFetcher(self.store)

        # 5. Load local NIFTY CSVs (5 years from NSE website downloads)
        logger.info("--- [5/8] Loading local NIFTY CSV data (5 years) ---")
        csv_count = ext.load_local_nifty_csvs()
        logger.info(f"  Local CSVs: {csv_count} candles loaded")

        # 6. Load FII/DII from bulk CSV + today from NSE API
        logger.info("--- [6/8] Loading FII/DII data ---")
        fii_csv_count = ext.load_fii_dii_csv()
        fii_today = ext.fetch_fii_dii_today()
        fii_cov_step6 = self.store.get_fii_dii_coverage()
        logger.info(
            f"  FII/DII: {fii_csv_count} from CSV, "
            f"{fii_today} from API today, "
            f"{fii_cov_step6['rows']} total in DB"
        )

        # 7. Fetch external markets (S&P 500, NASDAQ, Crude, Gold, USD/INR)
        logger.info("--- [7/8] Fetching external market data (yfinance) ---")
        ext_count = ext.fetch_external_markets(days=1825, force=force)
        logger.info(f"  External markets: {ext_count} data points")

        # 8. Fetch VIX extended history (nsepython)
        logger.info("--- [8/8] Fetching VIX extended history (NSE) ---")
        vix_count = ext.fetch_vix_history_nse(days=1825, force=force)
        logger.info(f"  VIX history: {vix_count} candles")

        # ── Coverage summary ──
        logger.info("--- Data Coverage Summary ---")
        stats = self.store.get_stats()
        logger.info(f"  Candles: {stats.get('candles', 0)} rows")
        logger.info(f"  External: {stats.get('external_data', 0)} rows")

        fii_cov = self.store.get_fii_dii_coverage()
        if fii_cov["rows"] > 0:
            logger.info(
                f"  FII/DII: {fii_cov['rows']} rows in DB "
                f"({fii_cov['from_date']} to {fii_cov['to_date']})"
            )
        else:
            logger.info("  FII/DII: 0 rows (no real data yet)")

        for sym in ["NIFTY50", "INDIA_VIX"]:
            coverage = self.store.get_data_coverage(sym)
            if coverage["rows"] > 0:
                logger.info(
                    f"  {sym}: {coverage['from_date']} to {coverage['to_date']} "
                    f"({coverage['rows']} candles)"
                )

        for sym in ["SP500", "NASDAQ", "CRUDE_OIL", "GOLD", "USDINR"]:
            coverage = self.store.get_external_data_coverage(sym)
            if coverage["rows"] > 0:
                logger.info(
                    f"  {sym}: {coverage['from_date']} to {coverage['to_date']} "
                    f"({coverage['rows']} rows)"
                )

        logger.info("=== FETCH COMPLETE ===")

    def _fetch_fno_premiums(
        self, force_fetch: bool = False, fetch_expired: bool = False,
    ) -> None:
        """
        Fetch F&O option premium data — current and optionally expired contracts.

        Incremental: checks DB coverage before each fetch.
        - Current contracts: always fetched (skips cached)
        - Expired contracts: only fetched when fetch_expired=True (--fetch-expired flag)

        Uses:
        - Regular Historical API for currently listed contracts (from instrument master)
        - Expired Instruments API for past weekly expiries (opt-in only)
        """
        from src.data.options_instruments import OptionsInstrumentResolver

        resolver = OptionsInstrumentResolver()
        resolver.refresh()

        if resolver._df is None or resolver._df.empty:
            logger.warning("No instrument master available")
            return

        nifty_key = "NSE_INDEX|Nifty 50"
        from_date = (date.today() - timedelta(days=1095)).isoformat()
        to_date = date.today().isoformat()

        # Get NIFTY price range over the period for strike range
        try:
            nifty_df = self.data_fetcher.get_historical_candles(
                nifty_key, "day", from_date=from_date, to_date=to_date
            )
            if nifty_df.empty:
                logger.warning("Cannot determine NIFTY spot for F&O fetch")
                return
            spot_min = float(nifty_df["low"].min())
            spot_max = float(nifty_df["high"].max())
        except Exception as e:
            logger.error(f"Cannot fetch NIFTY spot: {e}")
            return

        strike_gap = 50
        # Cover ATM ± 3 strikes for the ENTIRE historical range
        strike_lo = int(round((spot_min - 150) / strike_gap) * strike_gap)
        strike_hi = int(round((spot_max + 150) / strike_gap) * strike_gap)
        logger.info(f"NIFTY range: {spot_min:.0f} - {spot_max:.0f} → Strikes {strike_lo}-{strike_hi}")

        # ── Part A: Current contracts from instrument master ──
        nifty_opts = resolver._df[resolver._df["name"].str.upper() == "NIFTY"]
        candidates = nifty_opts[
            (nifty_opts["strike"] >= strike_lo) & (nifty_opts["strike"] <= strike_hi)
        ]

        fetched = 0
        skipped = 0
        for _, inst in candidates.iterrows():
            if not self._running:
                logger.info("Shutdown requested — stopping current contracts fetch")
                break

            key = str(inst["instrument_key"])

            # Skip if already cached (incremental)
            if not force_fetch and self.store:
                cached = self.store.get_candles(key, "day", limit=1)
                if not cached.empty:
                    cached_max = str(cached["datetime"].max())[:10]
                    if cached_max >= (date.today() - timedelta(days=2)).isoformat():
                        skipped += 1
                        continue

            try:
                df = self.data_fetcher.get_historical_candles(
                    key, "day", from_date=from_date, to_date=to_date
                )
                if df is not None and not df.empty:
                    fetched += 1
            except Exception:
                pass

            if fetched % 20 == 0 and fetched > 0:
                logger.info(f"  Progress: {fetched} fetched, {skipped} skipped")

        logger.info(
            f"Current contracts: {fetched} fetched, {skipped} skipped (cached)"
        )

        # ── Part B: Expired contracts (opt-in via --fetch-expired) ──
        if not fetch_expired:
            logger.info("Expired contracts: skipped (use --fetch-expired to fetch)")
        else:
            logger.info("Fetching expired weekly contracts...")
            expired_fetched = 0
            expired_cached = 0
            expired_expiries_skipped = 0
            fo_key_map = {}  # instrument_key → {strike, option_type} for backtest lookups
            total_api_calls = 0

            # Ensure auth token is loaded before calling expired instruments API
            if not self.data_fetcher.authenticate():
                logger.warning("Skipping expired contracts fetch — no valid auth token")
                return

            # Get actual past expiry dates from Upstox API (up to 6 months)
            expiry_dates = self.data_fetcher.get_expired_expiries(nifty_key)

            if not expiry_dates:
                logger.warning("No expired expiry dates returned by API")
            else:
                logger.info(f"Found {len(expiry_dates)} expired expiry dates")

            for idx, expiry_str in enumerate(expiry_dates):
                if not self._running:
                    logger.info("Shutdown requested — stopping expired contracts fetch")
                    break

                try:
                    expiry = date.fromisoformat(expiry_str)
                    contracts = self.data_fetcher.get_expired_option_contracts(
                        nifty_key, expiry_str
                    )
                    total_api_calls += 1

                    atm_contracts = [
                        c for c in contracts
                        if strike_lo <= c.get("strike_price", 0) <= strike_hi
                        and c.get("instrument_key")
                    ]

                    # Save instrument key → (strike, option_type) mapping for backtest
                    for c in atm_contracts:
                        fo_key_map[c["instrument_key"]] = {
                            "strike": c.get("strike_price", 0),
                            "option_type": c.get("instrument_type", ""),
                        }

                    # ── Resume check: count how many contracts already in DB ──
                    cached_count = 0
                    if not force_fetch and self.store and atm_contracts:
                        for c in atm_contracts:
                            ck = c.get("instrument_key", "")
                            cached = self.store.get_candles(ck, "day", limit=1)
                            if not cached.empty:
                                cached_count += 1

                        if cached_count == len(atm_contracts):
                            # ALL contracts for this expiry already in DB — skip entirely
                            expired_cached += cached_count
                            expired_expiries_skipped += 1
                            logger.info(
                                f"  Expiry {expiry_str}: already in DB "
                                f"({cached_count} contracts), skipping"
                            )
                            continue

                    # Fetch contracts not yet in DB
                    expiry_api_calls = 0
                    expiry_retries = 0
                    for contract in atm_contracts:
                        if not self._running:
                            break

                        exp_key = contract.get("instrument_key", "")

                        # Skip if already cached in DB (per-contract check)
                        if not force_fetch and self.store:
                            cached = self.store.get_candles(exp_key, "day", limit=1)
                            if not cached.empty:
                                expired_cached += 1
                                continue

                        try:
                            exp_from = (expiry - timedelta(days=7)).isoformat()
                            exp_to = expiry.isoformat()
                            df = self.data_fetcher.get_expired_historical_candles(
                                exp_key, "day", from_date=exp_from, to_date=exp_to
                            )
                            if df is not None and not df.empty:
                                expired_fetched += 1
                            expiry_api_calls += 1
                            total_api_calls += 1
                        except ConnectionError:
                            # Network down — wait and retry once
                            expiry_retries += 1
                            if expiry_retries > 3:
                                logger.warning(
                                    f"  Expiry {expiry_str}: 3+ connection errors, "
                                    f"skipping remaining contracts"
                                )
                                break
                            logger.info("  Network error — waiting 60s before retry...")
                            time.sleep(60)
                        except Exception:
                            expiry_api_calls += 1
                            total_api_calls += 1

                        # Pause every 50 API calls (batch cooldown)
                        if total_api_calls > 0 and total_api_calls % 50 == 0:
                            logger.info(f"  Rate limit pause after {total_api_calls} API calls...")
                            time.sleep(2.0)

                    if expiry_api_calls > 0 or cached_count > 0:
                        logger.info(
                            f"  Expiry {idx + 1}/{len(expiry_dates)} ({expiry_str}): "
                            f"{len(atm_contracts)} contracts, "
                            f"{expiry_api_calls} API calls, {cached_count} cached"
                        )

                    # Extra pause between expiries
                    if expiry_api_calls > 0:
                        time.sleep(2.0)

                except ConnectionError:
                    logger.warning(
                        f"  Expiry {expiry_str}: connection error, "
                        f"skipping (will resume on next run)"
                    )
                    time.sleep(60)
                except Exception as e:
                    logger.debug(f"Expired contracts for {expiry_str}: {e}")

            logger.info(
                f"Expired contracts: {expired_fetched} fetched, {expired_cached} cached, "
                f"{expired_expiries_skipped} expiries fully skipped, "
                f"{total_api_calls} total API calls"
            )

            # Save instrument key mapping for backtest lookups
            if fo_key_map:
                import json
                map_path = Path("data/fo_key_map.json")
                # Merge with existing map if present
                existing = {}
                if map_path.exists():
                    with open(map_path) as f:
                        existing = json.load(f)
                existing.update(fo_key_map)
                map_path.parent.mkdir(parents=True, exist_ok=True)
                with open(map_path, "w") as f:
                    json.dump(existing, f)
                logger.info(f"Saved F&O key mapping: {len(existing)} entries → {map_path}")

    def _fetch_previous_day_data(self) -> None:
        """
        Auto-fetch previous trading day's candle data before starting trading.
        Called at the start of live/paper mode to keep DB up to date.
        """
        logger.info("--- Auto-fetching previous day data ---")

        if not self.data_fetcher.authenticate():
            logger.warning("Auth failed, skipping auto-fetch")
            return

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        today_str = date.today().isoformat()

        # Fetch equity data for all NIFTY 50 symbols
        nifty50 = self.config["universe"].get("nifty50", {})
        fetched = 0

        for sym, info in nifty50.items():
            inst_key = info.get("instrument_key", "")
            if not inst_key:
                continue
            try:
                df = self.data_fetcher.get_historical_candles(
                    inst_key, "day", from_date=yesterday, to_date=today_str
                )
                if not df.empty:
                    fetched += 1
            except Exception:
                pass

        # Fetch index data
        indices = self.config["universe"].get("indices", {})
        for idx_name, idx_info in indices.items():
            idx_key = idx_info.get("instrument_key", "")
            if idx_key:
                try:
                    self.data_fetcher.get_historical_candles(
                        idx_key, "day", from_date=yesterday, to_date=today_str
                    )
                except Exception:
                    pass

        logger.info(f"Auto-fetched previous day data for {fetched} equities")

    def _save_eod_candle_data(self) -> None:
        """
        Save end-of-day candle data after market close.
        Fetches today's intraday data and saves daily candle to DB.
        """
        logger.info("--- Saving EOD candle data ---")

        saved = 0
        nifty50 = self.config["universe"].get("nifty50", {})

        for sym, info in nifty50.items():
            inst_key = info.get("instrument_key", "")
            if not inst_key:
                continue
            try:
                df = self.data_fetcher.get_intraday_candles(inst_key, "day")
                if not df.empty:
                    self.store.save_candles(sym, inst_key, df, "day")
                    saved += 1
            except Exception:
                pass

        # Save index EOD data
        indices = self.config["universe"].get("indices", {})
        for idx_name, idx_info in indices.items():
            idx_key = idx_info.get("instrument_key", "")
            if idx_key:
                try:
                    df = self.data_fetcher.get_intraday_candles(idx_key, "day")
                    if not df.empty:
                        self.store.save_candles(idx_name, idx_key, df, "day")
                except Exception:
                    pass

        logger.info(f"Saved EOD data for {saved} symbols")

    # ─────────────────────────────────────────
    # Backtest
    # ─────────────────────────────────────────

    def _run_backtest(self) -> None:
        """Run backtest mode — Options only (CE/PE on NIFTY)."""
        logger.info("=== BACKTEST MODE (Options Only) ===")

        capital = getattr(self, "_backtest_capital", None) or self.config["trading"]["capital"]
        self._run_options_backtest(capital)

    def _fetch_real_premium_data(
        self, nifty_df, strike_gap: int = 50
    ) -> dict:
        """
        Load option premium OHLCV data from local DB.

        Data should be pre-fetched via --mode fetch. If DB has no data,
        falls back to API fetching (requires valid auth token).

        Returns: {(date_str, strike, option_type): {open, high, low, close, volume}}
        """
        import pandas as pd

        premium_lookup: dict = {}

        # ── Load cached F&O data from DB ──
        fo_symbols = [s for s in self.store.get_all_symbols("day") if "NSE_FO" in s]
        if fo_symbols:
            logger.info(f"Loading {len(fo_symbols)} F&O instruments from local DB...")
            from_date = (date.today() - timedelta(days=1095)).isoformat()
            to_date = date.today().isoformat()
            bulk = self.store.get_candles_bulk(fo_symbols, "day", from_date, to_date)

            for sym, df in bulk.items():
                strike, opt_type = self._parse_fo_symbol(sym)
                if strike is None:
                    continue
                for _, candle in df.iterrows():
                    d = str(candle["datetime"])[:10]
                    premium_lookup[(d, strike, opt_type)] = {
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": int(candle.get("volume", 0)),
                    }

            logger.info(f"Loaded {len(premium_lookup)} data points from DB")

        if premium_lookup:
            return premium_lookup

        # ── No DB data — fallback to API fetch (first-time run without --mode fetch) ──
        logger.warning("No F&O data in DB. Fetching from API (run --mode fetch to pre-download)...")
        self._fetch_fno_premiums()

        # Reload from DB after fetch
        fo_symbols = [s for s in self.store.get_all_symbols("day") if "NSE_FO" in s]
        if fo_symbols:
            from_date = (date.today() - timedelta(days=1095)).isoformat()
            to_date = date.today().isoformat()
            bulk = self.store.get_candles_bulk(fo_symbols, "day", from_date, to_date)

            for sym, df in bulk.items():
                strike, opt_type = self._parse_fo_symbol(sym)
                if strike is None:
                    continue
                for _, candle in df.iterrows():
                    d = str(candle["datetime"])[:10]
                    premium_lookup[(d, strike, opt_type)] = {
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": int(candle.get("volume", 0)),
                    }

            logger.info(f"Loaded {len(premium_lookup)} data points from DB after API fetch")

        return premium_lookup

    def _build_fo_key_map(self) -> dict:
        """Build mapping from instrument_key → (strike, option_type) using instrument master + saved JSON."""
        if hasattr(self, "_fo_key_map_cache") and self._fo_key_map_cache:
            return self._fo_key_map_cache

        key_map = {}

        # Load saved expired contract mapping from fetch mode
        map_path = Path("data/fo_key_map.json")
        if map_path.exists():
            try:
                import json
                with open(map_path) as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    key_map[k] = (float(v["strike"]), str(v["option_type"]))
                logger.info(f"Loaded {len(key_map)} expired F&O key mappings from {map_path}")
            except Exception:
                pass

        # Add current contracts from instrument master
        try:
            from src.data.options_instruments import OptionsInstrumentResolver
            resolver = OptionsInstrumentResolver()
            resolver.refresh()
            if resolver._df is not None and not resolver._df.empty:
                for _, row in resolver._df.iterrows():
                    key_map[str(row["instrument_key"])] = (
                        float(row["strike"]),
                        str(row["option_type"]),
                    )
        except Exception:
            pass

        self._fo_key_map_cache = key_map
        return key_map

    def _parse_fo_symbol(self, symbol: str):
        """
        Parse strike and option type from an F&O instrument key or symbol.

        Handles formats like:
        - "NSE_FO|47983|17-04-2025" (instrument key from expired API)
        - "NSE_FO|47983" (current instrument key)
        - "NIFTY25400CE" (generated symbol)
        """
        import re
        # Try "NIFTY<strike><CE|PE>" format
        m = re.match(r"(?:NIFTY|BANKNIFTY)(\d+)(CE|PE)", symbol)
        if m:
            return float(m.group(1)), m.group(2)

        # Look up instrument master for NSE_FO keys
        if symbol.startswith("NSE_FO|"):
            key_map = self._build_fo_key_map()
            if symbol in key_map:
                return key_map[symbol]
            # Try matching without the date suffix (e.g., "NSE_FO|47983|17-04-2025" → "NSE_FO|47983")
            base_key = "|".join(symbol.split("|")[:2])
            if base_key in key_map:
                return key_map[base_key]

        return None, None

    def _run_options_backtest(self, capital: float = 50000) -> None:
        """
        VELTRIX — Backtest NIFTY options (CE/PE, dynamic lots, intraday).

        9-factor scoring + dynamic lot sizing:
        - MAX_DEPLOYABLE = ₹25K fixed, risk = ₹10K per trade
        - VIX-adaptive SL/TP (6-tier) × regime multiplier
        - Dynamic 4-level trailing stop (TRENDING/VOLATILE)
        - Volume confirmation (F9)
        """
        import numpy as np
        import pandas as pd
        from src.backtest.engine import BacktestTrade
        from src.backtest.metrics import BacktestMetrics

        try:
            import lightgbm as lgb
            from sklearn.preprocessing import StandardScaler
            ML_AVAILABLE = True
        except ImportError:
            ML_AVAILABLE = False

        # Optional ensemble models
        try:
            import xgboost as xgb
            XGB_AVAILABLE = True
        except ImportError:
            XGB_AVAILABLE = False
        try:
            from catboost import CatBoostClassifier
            CAT_AVAILABLE = True
        except ImportError:
            CAT_AVAILABLE = False

        logger.info("")
        logger.info("=" * 60)
        mode_label = "ACTIVE" if getattr(self, "_active_trading", False) else "CONSERVATIVE"
        logger.info(f"=== VELTRIX BACKTEST (NIFTY CE/PE — {mode_label}) ===")
        logger.info("=" * 60)

        # ── Fetch historical NIFTY + VIX data (use all available) ──
        nifty_key = self.config["universe"]["indices"]["NIFTY50"]["instrument_key"]
        from_date = (date.today() - timedelta(days=1095)).isoformat()
        to_date = date.today().isoformat()
        nifty_df = self.data_fetcher.get_historical_candles(nifty_key, "day", from_date=from_date, to_date=to_date)

        if nifty_df is None or nifty_df.empty or len(nifty_df) < 50:
            logger.error("Insufficient NIFTY data for options backtest")
            return

        logger.info(f"NIFTY data: {len(nifty_df)} trading days loaded")

        # Add technical features
        nifty_df = self.feature_engine.add_technical_features(nifty_df)

        # Fetch India VIX history (extended — DB has up to 5yr from CSVs + Upstox)
        vix_map = self.data_fetcher.get_vix_history(days=1095)

        # ── Load FII/DII + external market data for enhanced features ──
        fii_df = self.store.get_fii_dii_history(days=1825)
        external_df = self.store.get_external_data_all()

        # Load VIX as DataFrame for historical feature computation
        vix_key = self.config["universe"]["indices"].get("INDIA_VIX", {}).get("instrument_key", "NSE_INDEX|India VIX")
        vix_hist_df = self.data_fetcher.get_historical_candles(
            vix_key, "day", from_date=from_date, to_date=to_date
        )

        # Add alternative features (FII/DII + VIX as DataFrame for extended stats)
        nifty_df = self.feature_engine.add_alternative_features(
            nifty_df,
            fii_data=fii_df if fii_df is not None and not fii_df.empty else None,
            vix_data=vix_hist_df if vix_hist_df is not None and not vix_hist_df.empty else None,
        )

        # Add external market features (S&P500, NASDAQ, Crude, Gold, USD/INR)
        nifty_df = self.feature_engine.add_external_market_features(
            nifty_df,
            external_data=external_df if not external_df.empty else None,
        )

        # Log available enhanced features
        enhanced_cols = [c for c in nifty_df.columns if c.startswith(("fii_", "dii_", "india_vix", "vix_",
                         "sp500_", "nasdaq_", "crude_", "gold_", "usdinr_", "dxy_", "global_"))]
        has_real_fii = "fii_net_flow_1d" in nifty_df.columns and nifty_df["fii_net_flow_1d"].abs().sum() > 0
        has_external = "sp500_prev_return" in nifty_df.columns and nifty_df["sp500_prev_return"].abs().sum() > 0
        logger.info(
            f"Enhanced features: {len(enhanced_cols)} cols | "
            f"FII={'REAL' if has_real_fii else 'ZERO'} | "
            f"External={'REAL' if has_external else 'ZERO'}"
        )

        # ── Options config ──
        lot_size = 65  # NIFTY lot
        strike_gap = 50
        brokerage_per_order = 20.0
        stt_sell_pct = 0.000625

        # ── Load real premium data (DB first, API fallback) ──
        logger.info("Phase 1: Loading option premium data...")
        premium_lookup = self._fetch_real_premium_data(nifty_df, strike_gap)

        # ── ML Walk-forward setup (expanded: 19 → up to 35 features) ──
        ml_features = [
            # Technical (16 — original)
            "rsi_14", "rsi_7", "macd_histogram", "macd_line",
            "bb_position", "bb_width", "atr_pct", "adx_14",
            "volatility_20d", "returns_1d", "returns_5d", "returns_20d",
            "price_to_sma50", "body_size", "upper_shadow", "lower_shadow",
            # Momentum/range (3 — original)
            "ret_5d", "ret_20d", "range_5d_pct",
            # FII/DII (7)
            "fii_net_flow_1d", "fii_net_flow_5d", "fii_flow_momentum",
            "fii_net_direction", "fii_net_streak",
            "dii_net_flow_1d", "india_vix",
            # VIX extended (3)
            "vix_change_pct", "vix_percentile_252d", "vix_5d_ma",
            # External markets (9)
            "sp500_prev_return", "nasdaq_prev_return", "crude_prev_return",
            "gold_prev_return", "usdinr_prev_return",
            "sp500_nifty_corr_20d", "crude_nifty_corr_20d",
            "dxy_momentum_5d", "global_risk_score",
        ]
        ml_available_feats = [f for f in ml_features if f in nifty_df.columns]
        ml_model = None       # LightGBM (primary)
        ml_model_xgb = None   # XGBoost (ensemble member)
        ml_model_cat = None   # CatBoost (ensemble member)
        ml_scaler = None
        ml_last_train_idx = -1
        ml_train_window = 120  # Train on 120 days (more data for 3 models)
        ml_retrain_every = 10  # Retrain every 10 days
        ml_predictions = 0
        ml_correct = 0
        ml_rolling_window: list[bool] = []  # Last 50 predictions correct/wrong
        ml_auto_weight = 0.5     # Auto-governance weight (0.0-1.0)
        ml_train_accuracies: list[float] = []  # Train acc per fold
        ml_test_accuracies: list[float] = []   # Test acc per fold
        ml_pred_up_count = 0     # How many times predicted UP
        ml_pred_down_count = 0   # How many times predicted DOWN
        ml_actual_up_count = 0   # Actual UP days in accuracy set
        ml_actual_down_count = 0 # Actual DOWN days in accuracy set
        ml_correct_up = 0        # Correct on actual UP days
        ml_correct_down = 0      # Correct on actual DOWN days
        ml_influenced_trades = 0 # Trades where ML changed direction
        ml_feature_importance = {}  # Accumulated importance

        if ML_AVAILABLE and len(ml_available_feats) >= 8:
            logger.info(f"Phase 1b: ML walk-forward enabled ({len(ml_available_feats)} features)")

            # ── NaN handling for ML features (NEVER drop rows) ──
            # Step 1: Forward-fill (external data aligns on trading days)
            nifty_df[ml_available_feats] = nifty_df[ml_available_feats].ffill()
            # Step 2: Fill remaining NaN (start of dataset) with column median
            for col in ml_available_feats:
                median_val = nifty_df[col].median()
                if pd.notna(median_val):
                    nifty_df[col] = nifty_df[col].fillna(median_val)
                else:
                    nifty_df[col] = nifty_df[col].fillna(0.0)

            # NaN diagnostic logging
            nan_counts = nifty_df[ml_available_feats].isna().sum()
            total_rows = len(nifty_df)
            nan_features = nan_counts[nan_counts > 0]
            if len(nan_features) > 0:
                logger.warning(f"ML features with NaN after fill: {len(nan_features)}")
                for feat, cnt in nan_features.items():
                    pct = cnt / total_rows * 100
                    if pct > 20:
                        logger.warning(f"  {feat}: {cnt}/{total_rows} ({pct:.1f}%) NaN — HIGH")
                    else:
                        logger.info(f"  {feat}: {cnt}/{total_rows} ({pct:.1f}%) NaN")
            else:
                logger.info(f"ML features: 0 NaN across {len(ml_available_feats)} features, {total_rows} rows")

            # Prepare labels: binary INTRADAY direction (0=bearish, 1=bullish)
            # close > open = bullish day (CE would profit), close < open = bearish (PE profit)
            # Exclude noise: ±0.2% intraday moves set to NaN (excluded from training)
            nifty_df["_intraday_ret"] = (nifty_df["close"] - nifty_df["open"]) / nifty_df["open"]
            nifty_df["_ml_target"] = pd.NA  # Start as NA
            nifty_df.loc[nifty_df["_intraday_ret"] > 0.002, "_ml_target"] = 1   # UP
            nifty_df.loc[nifty_df["_intraday_ret"] < -0.002, "_ml_target"] = 0  # DOWN
            # Rows between -0.2% and +0.2% stay NaN → excluded from training
            noise_pct = nifty_df["_ml_target"].isna().sum() / len(nifty_df) * 100
            logger.info(f"ML labels: {noise_pct:.0f}% excluded as noise (±0.2% intraday)")
        else:
            logger.info("ML walk-forward disabled (missing features or lightgbm)")

        # ── State tracking ──
        cash = capital
        trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        peak_equity = capital
        sample_trades: list[str] = []
        real_data_trades = 0
        estimated_trades = 0

        # Pre-compute indicators
        nifty_df["ema_9"] = nifty_df["close"].ewm(span=9, adjust=False).mean()
        nifty_df["ema_20"] = nifty_df["close"].ewm(span=20, adjust=False).mean()
        nifty_df["ema_50"] = nifty_df["close"].ewm(span=50, adjust=False).mean()
        nifty_df["_date"] = pd.to_datetime(nifty_df["datetime"]).dt.date
        # Deduplicate: keep first row per date (guards against tz-format duplicates)
        nifty_df = nifty_df.drop_duplicates(subset=["_date"], keep="first").reset_index(drop=True)
        nifty_df["prev_close"] = nifty_df["close"].shift(1)
        nifty_df["prev_high"] = nifty_df["high"].shift(1)
        nifty_df["prev_low"] = nifty_df["low"].shift(1)
        # ATR for volatility-adjusted sizing
        nifty_df["atr_14"] = nifty_df.get("atr_14", nifty_df["close"] * 0.01)
        # Returns for momentum / trend exhaustion
        nifty_df["ret_5d"] = nifty_df["close"].pct_change(5) * 100
        nifty_df["ret_20d"] = nifty_df["close"].pct_change(20) * 100
        # 5-day range % for sideways detection
        nifty_df["range_5d_pct"] = (
            (nifty_df["high"].rolling(5).max() - nifty_df["low"].rolling(5).min())
            / nifty_df["close"] * 100
        )

        # Consecutive wins/losses for streak-based sizing
        streak = 0
        # Consecutive SL tracker for direction pause
        consec_sl_direction = ""  # "CE" or "PE"
        consec_sl_count = 0
        consec_sl_block_days = 0  # Days blocked by 3-SL rule (auto-reset after 5)

        # Circuit breaker simulation
        cb_daily_loss_warning = 5_000     # ₹5K daily loss → conviction +1.0
        cb_daily_loss_halt = 10_000       # ₹10K daily loss → halt
        # Active trading mode: 5 trades/day, lower thresholds
        bt_active = getattr(self, "_active_trading", False)
        bt_max_daily_trades = 5 if bt_active else 1  # 1 trade/day (no intraday re-entry in daily backtest)
        cb_max_daily_trades = bt_max_daily_trades
        cb_conviction_boost = 0.0         # Added when daily loss warning

        full_trades_today = 0
        same_day_sl_count = 0  # Active mode: 2 SLs same day → halt

        # Whipsaw detection: track last 5 trade outcomes (True=win, False=loss)
        recent_outcomes: list[bool] = []
        _skip_whipsaw = 0

        # ── Fixed deploy/risk caps (constants) ──
        BT_MAX_DEPLOY = 25_000       # ₹25K FIXED deploy cap
        BT_MAX_RISK = 10_000         # ₹10K max risk per trade
        min_premium = 80

        trading_days = 0
        signals_generated = 0
        skipped_vix = 0
        _skip_vix = 0
        _skip_expiry = 0
        _skip_consec_sl = 0
        _skip_conviction = 0

        # Regime behavior profiles (constant — matches live RegimeDetector.REGIME_PROFILES)
        regime_profiles = {
            "TRENDING": {
                "size_multiplier": 1.0,
                "conviction_min": 1.75,      # Lowered from 2.0 (1.5 too aggressive)
                "sl_multiplier": 1.0,
                "tp_multiplier": 1.3,
                "trailing_stop_enabled": True,
                "ema_weight": 2.5,
                "mean_reversion_weight": 1.5,
                "max_trades_per_day": bt_max_daily_trades,
            },
            "RANGEBOUND": {
                "size_multiplier": 0.5,
                "conviction_min": 2.0,        # Keep at 2.0 (rangebound needs stronger signal)
                "sl_multiplier": 0.85,
                "tp_multiplier": 0.70,
                "trailing_stop_enabled": False,
                "ema_weight": 1.0,
                "mean_reversion_weight": 2.5,
                "max_trades_per_day": max(1, bt_max_daily_trades - 1),
            },
            "VOLATILE": {
                "size_multiplier": 0.5,
                "conviction_min": 2.5,        # Lowered from 3.0 → some volatile trades
                "sl_multiplier": 1.20,
                "tp_multiplier": 1.50,
                "trailing_stop_enabled": True,
                "ema_weight": 0.5,
                "mean_reversion_weight": 1.0,
                "max_trades_per_day": max(1, bt_max_daily_trades - 1),
            },
        }

        logger.info("Phase 2: Running backtest simulation...")

        for i in range(50, len(nifty_df)):
            row = nifty_df.iloc[i]

            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            ema_9 = float(row["ema_9"])
            ema_20 = float(row["ema_20"])
            ema_50 = float(row["ema_50"])
            prev_close = float(row["prev_close"]) if pd.notna(row["prev_close"]) else open_price
            prev_high = float(row["prev_high"]) if pd.notna(row["prev_high"]) else high
            prev_low = float(row["prev_low"]) if pd.notna(row["prev_low"]) else low
            current_date = row["_date"]
            date_str = current_date.isoformat()
            ret_5d = float(row["ret_5d"]) if pd.notna(row["ret_5d"]) else 0.0
            # ret_20d available for trend exhaustion if needed
            # ret_20d = float(row["ret_20d"]) if pd.notna(row["ret_20d"]) else 0.0

            trading_days += 1

            # ── Daily reset ──
            cb_conviction_boost = 0.0
            full_trades_today = 0
            same_day_sl_count = 0

            # Drawdown tracking for reporting (not a halt — daily loss % handles intraday risk)
            # peak_equity is tracked at end-of-day for max drawdown calculation

            # ── VIX regime ──
            vix = vix_map.get(current_date, 14.0)

            # Skip extreme VIX (options too expensive, random moves)
            if vix > 35:
                skipped_vix += 1
                _skip_vix += 1
                equity_curve.append({
                    "date": date_str, "equity": round(cash, 2),
                    "cash": round(cash, 2), "positions_value": 0,
                    "n_positions": 0, "daily_return": 0,
                })
                continue

            # ── Expiry day handling ──
            # Tuesday = NIFTY weekly expiry. Massive theta decay after 1 PM.
            # Allow trading with restrictions: wider SL, max 1 trade, force exit by 1:30 PM
            is_expiry = (current_date.weekday() == 1)  # Tuesday
            expiry_sl_buffer = 1.05 if is_expiry else 1.0  # +5% wider SL on expiry
            expiry_tp_scale = 0.65 if is_expiry else 1.0   # Lower TP target (quick exit)
            expiry_max_trades = 1  # Set here; overridden after profile is loaded below

            # ── Adaptive SL/TP by VIX regime (lowered TP to reduce EOD exits) ──
            if vix < 13:
                premium_sl_pct = 0.25
                premium_tp_pct = 0.40
            elif vix < 18:
                premium_sl_pct = 0.30
                premium_tp_pct = 0.45
            elif vix < 22:
                premium_sl_pct = 0.30
                premium_tp_pct = 0.55
            elif vix < 28:
                premium_sl_pct = 0.25
                premium_tp_pct = 0.60
            elif vix <= 35:
                premium_sl_pct = 0.20
                premium_tp_pct = 0.45
            else:
                premium_sl_pct = 0.20
                premium_tp_pct = 0.40

            # ── Market regime (3-regime system matching live/paper) ──
            # Regime = environment (HOW to trade), NOT direction
            # Direction comes from scoring. Regime controls conviction, SL/TP, sizing.
            prev_row = nifty_df.iloc[i - 1]
            adx = float(row.get("adx_14", 20))
            prev_adx = float(prev_row.get("adx_14", 20)) if pd.notna(prev_row.get("adx_14")) else adx
            adx_slope = adx - prev_adx  # Positive = trend strengthening
            range_5d = float(row.get("range_5d_pct", 2.0))
            trend_up = ema_9 > ema_20 > ema_50
            trend_down = ema_9 < ema_20 < ema_50

            # BB width for volatility proxy
            bb_upper_val = float(row.get("bb_upper", close * 1.02))
            bb_lower_val = float(row.get("bb_lower", close * 0.98))
            bb_width = (bb_upper_val - bb_lower_val) / close if close > 0 else 0.04

            # VIX 5-day change (approximate from vix_map)
            vix_5d_ago_date = nifty_df.iloc[max(0, i - 5)]["_date"]
            vix_5d_ago = vix_map.get(vix_5d_ago_date, vix)
            vix_5d_change = vix - vix_5d_ago

            # ── Classify: VOLATILE > TRENDING > RANGEBOUND ──
            # Priority 1: VOLATILE (risk-off)
            if vix >= 30:
                regime = "VOLATILE"
            else:
                volatile_score = 0
                if vix > 22:
                    volatile_score += 2
                if vix_5d_change > 3.0:
                    volatile_score += 2
                if range_5d > 4.0:  # Wide 5d range proxy for intraday chaos
                    volatile_score += 1

                if volatile_score >= 2:
                    regime = "VOLATILE"
                # Priority 2: TRENDING (ride the wave)
                elif adx > 25:
                    regime = "TRENDING"
                elif adx > 20 and adx_slope > 0 and bb_width > 0.04:
                    regime = "TRENDING"
                # Priority 3: RANGEBOUND (everything else)
                else:
                    regime = "RANGEBOUND"

            profile = regime_profiles[regime]

            # Update expiry_max_trades now that profile is available
            if not is_expiry:
                expiry_max_trades = profile["max_trades_per_day"]

            # ── Whipsaw detection: skip choppy conditions ──
            # 1. ADX < 20 but classified as TRENDING → fake trend, skip
            if regime == "TRENDING" and adx < 20:
                skipped_vix += 1
                _skip_whipsaw += 1
                equity_curve.append({
                    "date": date_str, "equity": round(cash, 2),
                    "cash": round(cash, 2), "positions_value": 0,
                    "n_positions": 0, "daily_return": 0,
                })
                continue

            # 2. Alternating win/loss pattern (3+ alternations in last 5 trades)
            if len(recent_outcomes) >= 5:
                alternations = sum(
                    1 for j in range(1, len(recent_outcomes))
                    if recent_outcomes[j] != recent_outcomes[j - 1]
                )
                if alternations >= 3:
                    # Choppy market — boost conviction threshold
                    cb_conviction_boost = max(cb_conviction_boost, 0.5)

            # 3. Low intraday range day (range < 0.5% → skip)
            day_range_pct = (high - low) / close * 100 if close > 0 else 0
            if day_range_pct < 0.5:
                skipped_vix += 1
                _skip_whipsaw += 1
                equity_curve.append({
                    "date": date_str, "equity": round(cash, 2),
                    "cash": round(cash, 2), "positions_value": 0,
                    "n_positions": 0, "daily_return": 0,
                })
                continue

            # Max premium at 1 lot for FULL (most permissive for strike selection)
            deploy_max_prem = BT_MAX_DEPLOY / lot_size  # ₹615
            risk_max_prem = BT_MAX_RISK / (lot_size * premium_sl_pct) if premium_sl_pct > 0 else 625
            max_premium = min(deploy_max_prem, risk_max_prem)

            # ── Multi-factor signal scoring ──
            bull_score = 0.0
            bear_score = 0.0

            rsi = float(row.get("rsi_14", 50))
            macd_hist = float(row.get("macd_histogram", 0))
            prev_macd_hist = float(prev_row.get("macd_histogram", 0))
            prev_rsi = float(prev_row.get("rsi_14", 50))
            bb_upper = float(row.get("bb_upper", close * 1.02))
            bb_lower = float(row.get("bb_lower", close * 0.98))

            # === FACTOR 1: Trend alignment (regime-driven weight) ===
            # EMA stack weight controlled by regime: TRENDING=2.5, RANGEBOUND=1.0, VOLATILE=0.5
            ema_weight = profile["ema_weight"]
            ema_base = ema_weight * 0.8   # Full EMA stack signal
            ema_bonus = ema_weight * 0.2  # Price vs EMA20 confirmation
            ret_5d_val = float(row.get("ret_5d", 0)) if pd.notna(row.get("ret_5d")) else 0
            rangebound_flag = abs(ret_5d_val) < 1.0 and adx < 22
            if trend_up:
                bull_score += ema_base * 0.5 if rangebound_flag else ema_base
            elif trend_down:
                bear_score += ema_base * 0.5 if rangebound_flag else ema_base

            # Price vs EMA20 (weaker confirmation)
            if close > ema_20 * 1.005:
                bull_score += ema_bonus
            elif close < ema_20 * 0.995:
                bear_score += ema_bonus

            # 5-day trend direction: +0.3 nudge
            if ret_5d > 0:
                bull_score += 0.3
            elif ret_5d < 0:
                bear_score += 0.3

            # === FACTOR 2: Momentum (weight: 2.0) ===
            # RSI momentum — stronger thresholds for clearer signal
            if rsi > 58 and rsi > prev_rsi:
                bull_score += 1.0
            elif rsi < 42 and rsi < prev_rsi:
                bear_score += 1.0

            # MACD acceleration — histogram expanding in direction
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                bull_score += 1.0
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                bear_score += 1.0

            # === FACTOR 3: Price action (weight: 1.5) ===
            # Gap direction — institutional order flow signal
            gap_pct = (open_price - prev_close) / prev_close * 100
            if gap_pct > 0.4:
                bull_score += 0.75
            elif gap_pct < -0.4:
                bear_score += 0.75

            # Breakout: close beyond previous day's range (stronger signal)
            if close > prev_high:
                bull_score += 0.75
            elif close < prev_low:
                bear_score += 0.75

            # Candle body direction: +0.3 for strong body
            if close > open_price:
                bull_score += 0.3
            elif close < open_price:
                bear_score += 0.3

            # === FACTOR 4: Mean reversion guard (regime-driven weight) ===
            # Weight controlled by regime: RANGEBOUND=2.5 (MR is king), TRENDING=1.5, VOLATILE=1.0
            mr_weight = profile["mean_reversion_weight"]
            mr_score = mr_weight * 0.67     # Reversal signal
            mr_penalty = mr_weight * 0.33   # Penalize chasing
            if ret_5d > 5.0:
                bear_score += mr_score + 1.0  # Extreme overbought
                bull_score -= mr_penalty
            elif ret_5d > 3.5:
                bear_score += mr_score    # Overbought, likely to reverse
                bull_score -= mr_penalty  # Penalize chasing
            elif ret_5d < -5.0:
                bull_score += mr_score + 1.0  # Extreme oversold
                bear_score -= mr_penalty
            elif ret_5d < -3.5:
                bull_score += mr_score    # Oversold, likely to bounce
                bear_score -= mr_penalty  # Penalize chasing

            # === FACTOR 5: Bollinger position (weight: 1.0) ===
            bb_pos = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            if bb_pos > 0.85:
                bull_score += 0.5  # Strong uptrend (riding upper band)
            elif bb_pos < 0.15:
                bear_score += 0.5  # Strong downtrend

            # BB width expanding >20% vs previous day → volatility breakout +0.25
            prev_bb_upper = float(prev_row.get("bb_upper", prev_close * 1.02))
            prev_bb_lower = float(prev_row.get("bb_lower", prev_close * 0.98))
            curr_bb_width = (bb_upper - bb_lower) / close if close > 0 else 0
            prev_bb_width_val = (prev_bb_upper - prev_bb_lower) / prev_close if prev_close > 0 else 0
            if prev_bb_width_val > 0 and curr_bb_width > prev_bb_width_val * 1.20:
                if bull_score >= bear_score:
                    bull_score += 0.25
                else:
                    bear_score += 0.25

            # === FACTOR 6: VIX direction (weight: 0.5) ===
            if vix < 13:
                bull_score += 0.5  # Low VIX = complacency, trends persist
            elif vix > 20:
                bear_score += 0.5  # High VIX = fear, downside risk

            # VIX momentum: falling VIX from >20 → bullish, rising VIX → bearish
            prev_date = nifty_df.iloc[i - 1]["_date"]
            prev_vix = vix_map.get(prev_date, vix)
            vix_delta = vix - prev_vix
            if vix > 20 and vix_delta < -1.0:
                bull_score += 0.3  # VIX falling from elevated = fear easing
            elif vix_delta > 1.0:
                bear_score += 0.3  # VIX rising = fear increasing

            # === FACTOR 7: ML confidence (weight: 0.3 — informational only) ===
            # === FACTOR 7: ML 3-Model Ensemble (intraday target) ===
            if ML_AVAILABLE and len(ml_available_feats) >= 8 and "_ml_target" in nifty_df.columns:
                # Walk-forward: retrain every ml_retrain_every days
                days_since_train = i - ml_last_train_idx
                if ml_model is None or days_since_train >= ml_retrain_every:
                    train_start = max(1, i - ml_train_window)  # Start from 1, need lag
                    train_slice = nifty_df.iloc[train_start:i]
                    y_train = train_slice["_ml_target"]
                    valid = y_train.notna()
                    # LAG FEATURES BY 1: use YESTERDAY's features to predict TODAY's direction
                    # This prevents look-ahead bias (features like returns_1d use today's close)
                    X_train_raw = nifty_df.iloc[train_start-1:i-1][ml_available_feats].copy()
                    X_train_raw.index = train_slice.index  # Align with target
                    X_train = X_train_raw.loc[valid].copy()
                    y_train = y_train[valid].astype(int)

                    # Add feature interactions
                    for ca, cb, name in [
                        ("rsi_14", "india_vix", "rsi_x_vix"),
                        ("macd_histogram", "adx_14", "macd_x_adx"),
                        ("returns_1d", "volume_ratio", "ret_x_vol"),
                        ("bb_position", "rsi_14", "bb_x_rsi"),
                    ]:
                        if ca in X_train.columns and cb in X_train.columns:
                            X_train[name] = X_train[ca] * X_train[cb]

                    ml_interaction_feats = [c for c in X_train.columns if "_x_" in c]
                    all_train_feats = list(ml_available_feats) + ml_interaction_feats

                    if len(X_train) >= 40 and len(y_train.unique()) >= 2:
                        # Recency weighting: recent data matters more
                        import numpy as np
                        n = len(X_train)
                        sample_weights = np.ones(n)
                        for wi in range(n):
                            days_ago = n - wi
                            if days_ago > 365 * 4:
                                sample_weights[wi] = 0.3
                            elif days_ago > 365 * 3:
                                sample_weights[wi] = 0.4
                            elif days_ago > 365 * 2:
                                sample_weights[wi] = 0.6
                            elif days_ago > 365:
                                sample_weights[wi] = 0.8

                        ml_scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            ml_scaler.fit_transform(X_train[all_train_feats]),
                            columns=all_train_feats, index=X_train.index
                        )

                        # Model 1: LightGBM (regularized to prevent 100% train acc)
                        ml_model = lgb.LGBMClassifier(
                            objective="binary",
                            n_estimators=60, max_depth=3, num_leaves=8,
                            learning_rate=0.08, min_child_samples=25,
                            subsample=0.7, colsample_bytree=0.5,
                            reg_alpha=1.0, reg_lambda=1.0, verbose=-1,
                        )
                        ml_model.fit(X_scaled, y_train, sample_weight=sample_weights)
                        ml_model._all_feats = all_train_feats

                        # Track train accuracy + feature importance
                        train_pred = ml_model.predict(X_scaled)
                        train_acc = (train_pred == y_train.values).mean()
                        ml_train_accuracies.append(train_acc)
                        # Accumulate feature importance
                        for fname, fimp in zip(all_train_feats, ml_model.feature_importances_):
                            ml_feature_importance[fname] = ml_feature_importance.get(fname, 0) + fimp

                        # Model 2: XGBoost
                        if XGB_AVAILABLE:
                            ml_model_xgb = xgb.XGBClassifier(
                                objective="binary:logistic",
                                n_estimators=60, max_depth=3,
                                learning_rate=0.08, min_child_weight=25,
                                subsample=0.7, colsample_bytree=0.5,
                                reg_alpha=1.0, reg_lambda=1.0,
                                eval_metric="logloss", verbosity=0,
                            )
                            ml_model_xgb.fit(X_scaled, y_train, sample_weight=sample_weights)

                        # Model 3: CatBoost
                        if CAT_AVAILABLE:
                            ml_model_cat = CatBoostClassifier(
                                iterations=60, depth=3,
                                learning_rate=0.08, l2_leaf_reg=5.0,
                                min_data_in_leaf=25,
                                verbose=0,
                            )
                            ml_model_cat.fit(X_scaled, y_train, sample_weight=sample_weights)

                        ml_last_train_idx = i

                # Predict current day using YESTERDAY's features (no look-ahead)
                if ml_model is not None and ml_scaler is not None and i >= 1:
                    feat_row = nifty_df.iloc[i-1:i][ml_available_feats].copy()
                    try:
                        # Add same interactions
                        for ca, cb, name in [
                            ("rsi_14", "india_vix", "rsi_x_vix"),
                            ("macd_histogram", "adx_14", "macd_x_adx"),
                            ("returns_1d", "volume_ratio", "ret_x_vol"),
                            ("bb_position", "rsi_14", "bb_x_rsi"),
                        ]:
                            if ca in feat_row.columns and cb in feat_row.columns:
                                feat_row[name] = feat_row[ca].values * feat_row[cb].values

                        all_feats = getattr(ml_model, "_all_feats", ml_available_feats)
                        for f in all_feats:
                            if f not in feat_row.columns:
                                feat_row[f] = 0.0

                        assert not feat_row[all_feats].isna().any().any(), f"NaN in ML features at row {i}"
                        feat_scaled = pd.DataFrame(ml_scaler.transform(feat_row[all_feats]), columns=all_feats)

                        # 3-model ensemble: average probabilities
                        probs_up = []
                        lgb_prob = ml_model.predict_proba(feat_scaled)[0]
                        prob_up_lgb = lgb_prob[1] if len(lgb_prob) > 1 else 0.5
                        probs_up.append(prob_up_lgb)

                        if ml_model_xgb is not None:
                            xgb_prob = ml_model_xgb.predict_proba(feat_scaled)[0]
                            probs_up.append(xgb_prob[1] if len(xgb_prob) > 1 else 0.5)

                        if ml_model_cat is not None:
                            cat_prob = ml_model_cat.predict_proba(feat_scaled)[0]
                            probs_up.append(cat_prob[1] if len(cat_prob) > 1 else 0.5)

                        avg_prob_up = sum(probs_up) / len(probs_up)
                        avg_prob_down = 1.0 - avg_prob_up
                        all_agree_up = all(p > 0.5 for p in probs_up)
                        all_agree_down = all(p < 0.5 for p in probs_up)

                        ml_predictions += 1
                        if avg_prob_up > 0.5:
                            ml_pred_up_count += 1
                        else:
                            ml_pred_down_count += 1

                        # Accuracy: compare with CURRENT day's actual intraday direction
                        # ONLY count non-noise days (same ±0.2% threshold as training)
                        actual_ret = nifty_df.iloc[i].get("_intraday_ret", 0)
                        if pd.notna(actual_ret) and abs(actual_ret) > 0.002:
                            actual_class = 1 if actual_ret > 0.002 else 0
                            pred_class = 1 if avg_prob_up > 0.5 else 0
                            is_correct = pred_class == actual_class
                            if actual_class == 1:
                                ml_actual_up_count += 1
                                if is_correct:
                                    ml_correct_up += 1
                            else:
                                ml_actual_down_count += 1
                                if is_correct:
                                    ml_correct_down += 1
                            if is_correct:
                                ml_correct += 1
                            # Rolling 50-trade accuracy for auto-governance
                            ml_rolling_window.append(is_correct)
                            if len(ml_rolling_window) > 50:
                                ml_rolling_window.pop(0)
                            if len(ml_rolling_window) >= 20:
                                rolling_acc = sum(ml_rolling_window) / len(ml_rolling_window)
                                if rolling_acc > 0.60:
                                    ml_auto_weight = 1.0
                                elif rolling_acc > 0.55:
                                    ml_auto_weight = 0.5
                                elif rolling_acc > 0.50:
                                    ml_auto_weight = 0.3
                                else:
                                    ml_auto_weight = 0.0  # Coin flip → disable

                        # Confidence-based scoring with auto-governance weight
                        pre_ml_dir = "CE" if bull_score > bear_score else "PE"
                        if ml_auto_weight > 0:
                            if avg_prob_up > 0.65 and all_agree_up:
                                bull_score += 1.5 * ml_auto_weight
                            elif avg_prob_up > 0.58:
                                bull_score += 1.0 * ml_auto_weight
                            elif avg_prob_up > 0.52:
                                bull_score += 0.3 * ml_auto_weight
                            elif avg_prob_down > 0.65 and all_agree_down:
                                bear_score += 1.5 * ml_auto_weight
                            elif avg_prob_down > 0.58:
                                bear_score += 1.0 * ml_auto_weight
                            elif avg_prob_down > 0.52:
                                bear_score += 0.3 * ml_auto_weight
                        post_ml_dir = "CE" if bull_score > bear_score else "PE"
                        if pre_ml_dir != post_ml_dir:
                            ml_influenced_trades += 1

                    except AssertionError as ae:
                        logger.warning(f"ML prediction skipped: {ae}")
                    except Exception:
                        pass

            # === FACTOR 9: Volume Confirmation (weight: 1.0) ===
            volume = float(row.get("volume", 0))
            vol_ma_20 = 0.0
            if i >= 20:
                vol_ma_20 = float(nifty_df.iloc[i-20:i]["volume"].mean()) if "volume" in nifty_df.columns else 0
            if vol_ma_20 > 0 and volume > 0:
                vol_ratio = volume / vol_ma_20
                if vol_ratio > 1.3:
                    # High volume confirms direction
                    if close > open_price:
                        bull_score += 1.0
                    elif close < open_price:
                        bear_score += 1.0
                elif vol_ratio < 0.7:
                    # Low volume weakens direction
                    if close > open_price:
                        bull_score -= 0.3
                    elif close < open_price:
                        bear_score -= 0.3

            # ── Direction selection (regime-driven conviction filter) ──
            score_diff = abs(bull_score - bear_score)
            directions_to_trade = []

            # No regime nudge — regime controls via factor weights, not direction bias.
            # Direction is purely from scoring.

            # Consecutive SL block (V2.2): 3+ SLs in same direction
            # → block THAT direction for up to 5 days, then auto-reset
            if consec_sl_count >= 3:
                chosen_dir = "CE" if bull_score > bear_score else "PE"
                if chosen_dir == consec_sl_direction:
                    consec_sl_block_days += 1
                    if consec_sl_block_days > 5:
                        # 5-day cooldown complete — reset and allow trading
                        consec_sl_count = 0
                        consec_sl_direction = ""
                        consec_sl_block_days = 0
                    else:
                        skipped_vix += 1
                        _skip_consec_sl += 1
                        equity_curve.append({
                            "date": date_str, "equity": round(cash, 2),
                            "cash": round(cash, 2), "positions_value": 0,
                            "n_positions": 0, "daily_return": 0,
                        })
                        continue

            # Consecutive SL direction nudge: 3 SLs → nudge opposite (was 2, relaxed)
            if consec_sl_count >= 3:
                if consec_sl_direction == "CE":
                    bear_score += 0.5
                elif consec_sl_direction == "PE":
                    bull_score += 0.5

            # ── TRADE TYPE: FULL or SKIP (no CAUTIOUS) ──
            # Conviction threshold from regime profile
            full_threshold = profile["conviction_min"] + cb_conviction_boost

            if score_diff >= full_threshold:
                trade_type = "FULL"
            else:
                skipped_vix += 1
                _skip_conviction += 1
                equity_curve.append({
                    "date": date_str, "equity": round(cash, 2),
                    "cash": round(cash, 2), "positions_value": 0,
                    "n_positions": 0, "daily_return": 0,
                })
                continue

            # Single direction — higher score wins
            if bull_score > bear_score:
                directions_to_trade.append(("CE", bull_score, score_diff))
                # Active mode: also try opposite direction for re-entry after TP
                if bt_active and bear_score >= 1.5:
                    directions_to_trade.append(("PE", bear_score, abs(bull_score - bear_score)))
            elif bear_score > bull_score:
                directions_to_trade.append(("PE", bear_score, score_diff))
                if bt_active and bull_score >= 1.5:
                    directions_to_trade.append(("CE", bull_score, abs(bull_score - bear_score)))
            else:
                if ema_9 > ema_20:
                    directions_to_trade.append(("CE", bull_score, 0))
                else:
                    directions_to_trade.append(("PE", bear_score, 0))

            day_pnl = 0.0
            day_trades = 0
            atm_strike = round(open_price / strike_gap) * strike_gap

            for direction, dir_score, conviction in directions_to_trade:
                signals_generated += 1

                # Check max trades per day (regime-driven, capped on expiry)
                max_today = min(expiry_max_trades, profile["max_trades_per_day"])
                if trade_type == "FULL" and full_trades_today >= max_today:
                    break
                if day_trades >= max_today + 1:  # Total cap
                    break

                strike = atm_strike

                # ── SL/TP: VIX-adaptive × regime multiplier ──
                adj_sl = premium_sl_pct * profile["sl_multiplier"]
                adj_tp = premium_tp_pct * profile["tp_multiplier"]
                trade_max_premium = max_premium
                trade_risk = BT_MAX_RISK
                trail_enabled = profile["trailing_stop_enabled"]

                # Higher conviction → modestly wider TP
                if conviction >= 4.0:
                    adj_tp *= 1.2
                elif conviction >= 3.0:
                    adj_tp *= 1.1

                # After losing streak, tighten SL
                if streak <= -3:
                    adj_sl *= 0.90

                # Expiry day: wider SL, lower TP (theta decay)
                adj_sl *= expiry_sl_buffer
                adj_tp *= expiry_tp_scale

                # ── Find real premium data ──
                real_data = None
                itm_sign = -1 if direction == "CE" else 1
                otm_sign = -itm_sign

                for dist in range(0, 6):
                    for sign in ([0] if dist == 0 else [itm_sign, otm_sign]):
                        alt_strike = atm_strike + (sign * dist * strike_gap if dist > 0 else 0)
                        candidate = premium_lookup.get((date_str, alt_strike, direction))
                        if candidate is not None and candidate["volume"] > 0:
                            prem = candidate["open"]
                            if min_premium <= prem <= trade_max_premium:
                                real_data = candidate
                                strike = alt_strike
                                break
                    if real_data is not None:
                        break

                # Skip if strike too far from ATM (max 3 strikes = 150 points)
                if real_data is not None and abs(strike - atm_strike) > 3 * strike_gap:
                    real_data = None

                if real_data is not None:
                    # ── REAL PREMIUM DATA: Use actual OHLC ──
                    data_source = "REAL"
                    entry_premium = real_data["open"]
                    high_premium = real_data["high"]
                    low_premium = real_data["low"]
                    close_premium = real_data["close"]

                    # Dynamic SL by premium level:
                    # Cheap premiums need room to breathe, expensive ones get tighter
                    if entry_premium < 100:
                        prem_sl = max(adj_sl, 0.30)  # ≥30% for cheap options
                    elif entry_premium > 200:
                        prem_sl = min(adj_sl, 0.20)  # ≤20% for expensive ones
                    else:
                        prem_sl = adj_sl              # Keep VIX-adaptive default

                    sl_price = entry_premium * (1 - prem_sl)
                    tp_price = entry_premium * (1 + adj_tp)

                    # Dynamic trailing stop (all regimes, lower activation)
                    trail_floor = None
                    high_gain_pct = (high_premium - entry_premium) / entry_premium
                    if trail_enabled:
                        # TRENDING/VOLATILE: 4-tier trail starting at +8%
                        if high_gain_pct >= 0.50:
                            trail_floor = entry_premium * 1.35
                        elif high_gain_pct >= 0.35:
                            trail_floor = entry_premium * 1.22
                        elif high_gain_pct >= 0.15:
                            trail_floor = entry_premium * 1.10
                        elif high_gain_pct >= 0.08:
                            trail_floor = entry_premium * 1.03
                    else:
                        # RANGEBOUND: lighter trail (only on big gains)
                        if high_gain_pct >= 0.25:
                            trail_floor = entry_premium * 1.15
                        elif high_gain_pct >= 0.15:
                            trail_floor = entry_premium * 1.08

                    sl_hit = low_premium <= sl_price
                    tp_hit = high_premium >= tp_price
                    trail_triggered = (
                        trail_floor is not None
                        and close_premium < trail_floor
                        and close_premium > sl_price  # Only if above SL
                    )

                    exit_reason = ""
                    exit_premium = entry_premium

                    if tp_hit and not sl_hit:
                        exit_reason = "take_profit"
                        exit_premium = tp_price
                    elif sl_hit and not tp_hit:
                        exit_reason = "stop_loss"
                        exit_premium = sl_price
                    elif sl_hit and tp_hit:
                        if close_premium >= entry_premium:
                            exit_reason = "take_profit"
                            exit_premium = tp_price
                        else:
                            exit_reason = "stop_loss"
                            exit_premium = sl_price
                    elif trail_triggered:
                        exit_reason = "trail_stop"
                        exit_premium = trail_floor
                    else:
                        # On expiry: force exit ~1:30 PM → use midpoint of day (more theta decay)
                        if is_expiry:
                            # Approximate 1:30 PM exit: use 60% of close_premium (theta penalty)
                            exit_premium = entry_premium + (close_premium - entry_premium) * 0.6
                        else:
                            exit_premium = close_premium
                        exit_reason = "eod_exit"

                    real_data_trades += 1
                else:
                    # ── ESTIMATED: Moneyness-based premium model ──
                    # Uses strike distance from spot + VIX to estimate option premium
                    data_source = "EST"
                    moneyness = abs(strike - open_price) / open_price
                    tte_days = max(1, 5 - current_date.weekday())  # Days to weekly expiry
                    iv_annual = vix / 100.0
                    time_value = open_price * iv_annual * (tte_days / 365) ** 0.5 * 0.4

                    if direction == "CE":
                        itm = open_price > strike
                    else:
                        itm = open_price < strike

                    if itm:
                        intrinsic = abs(open_price - strike)
                        entry_premium = intrinsic + time_value
                    else:
                        # OTM: time value decays with distance from ATM
                        entry_premium = time_value * max(0.15, 1.0 - moneyness * 15)

                    # Clamp to valid premium range
                    entry_premium = max(min_premium, min(entry_premium, trade_max_premium))

                    # NIFTY move determines if our direction was right
                    nifty_move = close - open_price
                    correct_direction = (
                        (direction == "CE" and nifty_move > 0) or
                        (direction == "PE" and nifty_move < 0)
                    )

                    # Dynamic SL by premium level (same as real data)
                    if entry_premium < 100:
                        prem_sl = max(adj_sl, 0.30)
                    elif entry_premium > 200:
                        prem_sl = min(adj_sl, 0.20)
                    else:
                        prem_sl = adj_sl
                    # Estimated trades: tighter SL (less data confidence)
                    est_sl = prem_sl * 0.75
                    sl_price = entry_premium * (1 - est_sl)
                    tp_price = entry_premium * (1 + adj_tp)

                    if correct_direction:
                        nifty_move_pct = abs(nifty_move) / open_price * 100
                        # Scale premium change by NIFTY move magnitude
                        # Conservative: real-world slippage eats gains
                        if nifty_move_pct > 1.0:
                            # Big move → likely TP hit
                            exit_premium = tp_price
                            exit_reason = "take_profit"
                        elif nifty_move_pct > 0.5:
                            # Medium move → decent gain (but theta eats some)
                            gain_pct = 0.08 + nifty_move_pct * 0.15
                            exit_premium = entry_premium * (1 + gain_pct)
                            exit_reason = "eod_exit"
                        elif nifty_move_pct > 0.2:
                            # Small move → modest gain after theta decay
                            exit_premium = entry_premium * 1.04
                            exit_reason = "eod_exit"
                        else:
                            # Very small move → theta decay wins
                            exit_premium = entry_premium * 0.97
                            exit_reason = "eod_exit"
                    else:
                        # Wrong direction
                        nifty_move_pct = abs(nifty_move) / open_price * 100
                        if nifty_move_pct > 0.4:
                            # Adverse move → SL hit
                            exit_premium = sl_price
                            exit_reason = "stop_loss"
                        elif nifty_move_pct > 0.2:
                            # Medium adverse → significant loss + theta
                            loss_pct = 0.12 + nifty_move_pct * 0.18
                            exit_premium = entry_premium * (1 - loss_pct)
                            exit_reason = "eod_exit"
                        else:
                            # Small adverse → theta loss
                            exit_premium = entry_premium * 0.93
                            exit_reason = "eod_exit"

                    # Expiry day theta penalty on estimated EOD exits
                    if is_expiry and exit_reason == "eod_exit":
                        expiry_penalty = (exit_premium - entry_premium) * 0.4
                        exit_premium -= expiry_penalty

                    # Fix 5: If estimated exit loss exceeds SL%, classify correctly
                    if exit_premium <= sl_price:
                        exit_premium = sl_price
                        exit_reason = "stop_loss"
                    elif exit_premium >= tp_price:
                        exit_premium = tp_price
                        exit_reason = "take_profit"

                    estimated_trades += 1

                # ── Dynamic lot sizing: ₹25K deploy, risk by trade type ──
                if entry_premium <= 0:
                    continue

                lots_by_deploy = int(BT_MAX_DEPLOY / (entry_premium * lot_size))
                lots_by_risk = int(trade_risk / (entry_premium * adj_sl * lot_size)) if adj_sl > 0 else lots_by_deploy
                bt_lots = min(lots_by_deploy, lots_by_risk)  # No max lots cap
                bt_lots = max(1, bt_lots)
                lot_used = bt_lots * lot_size

                position_cost = entry_premium * lot_used

                # Hard cap: skip if even 1 lot exceeds deploy cap
                if entry_premium * lot_size > BT_MAX_DEPLOY:
                    if day_trades > 0:
                        if data_source == "REAL":
                            real_data_trades -= 1
                        else:
                            estimated_trades -= 1
                        signals_generated -= 1
                        continue
                    lot_used = lot_size
                    position_cost = entry_premium * lot_used

                # ── Calculate P&L ──
                gross_pnl = (exit_premium - entry_premium) * lot_used
                stt = exit_premium * lot_used * stt_sell_pct
                total_charges = brokerage_per_order * 2 + stt

                net_pnl = gross_pnl - total_charges
                pnl_pct = (net_pnl / position_cost) * 100

                # Update streak + whipsaw tracker
                if net_pnl > 0:
                    streak = max(streak + 1, 1)
                    recent_outcomes.append(True)
                else:
                    streak = min(streak - 1, -1)
                    recent_outcomes.append(False)
                # Keep last 5 outcomes only
                if len(recent_outcomes) > 5:
                    recent_outcomes.pop(0)

                # Update consecutive SL tracker
                if exit_reason == "stop_loss":
                    same_day_sl_count += 1
                    if direction == consec_sl_direction:
                        consec_sl_count += 1
                    else:
                        consec_sl_direction = direction
                        consec_sl_count = 1
                else:
                    consec_sl_count = 0
                    consec_sl_direction = ""
                    consec_sl_block_days = 0

                # ── Record trade ──
                option_symbol = f"NIFTY{int(strike)}{direction}"
                trade = BacktestTrade(
                    symbol=option_symbol,
                    side="BUY",
                    quantity=lot_used,
                    entry_price=round(entry_premium, 2),
                    exit_price=round(exit_premium, 2),
                    entry_date=date_str,
                    exit_date=date_str,
                    strategy=trade_type,  # "FULL"
                    regime=regime,
                    stop_loss=round(sl_price, 2),
                    take_profit=round(tp_price, 2),
                    charges=round(total_charges, 2),
                    slippage=0,
                    pnl=round(net_pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    hold_days=0,
                    exit_reason=exit_reason,
                )
                trades.append(trade)

                cash += net_pnl
                day_pnl += net_pnl
                day_trades += 1

                full_trades_today += 1

                # ── Circuit breaker: absolute loss thresholds ──
                daily_loss_abs = abs(day_pnl) if day_pnl < 0 else 0
                if daily_loss_abs >= cb_daily_loss_halt:
                    break  # ₹10K daily loss → halt
                elif daily_loss_abs >= cb_daily_loss_warning:
                    cb_conviction_boost = 1.0  # ₹5K daily loss → conviction +1.0
                # Active mode: 2 same-day SLs → halt for the day
                if bt_active and same_day_sl_count >= 2:
                    break
                if day_trades >= cb_max_daily_trades:
                    break

                if len(sample_trades) < 15:
                    pnl_sign = "+" if net_pnl >= 0 else ""
                    premium_change_pct = ((exit_premium - entry_premium) / entry_premium) * 100
                    sample_trades.append(
                        f"  {date_str}: [F] {direction} {option_symbol} @ ₹{entry_premium:.2f} "
                        f"→ ₹{exit_premium:.2f} ({premium_change_pct:+.1f}% {exit_reason}) "
                        f"{pnl_sign}₹{net_pnl:,.2f} [{bt_lots}L {lot_used}q]"
                    )

            # ── End of day equity curve ──
            prev_equity = equity_curve[-1]["equity"] if equity_curve else capital
            daily_return = (cash - prev_equity) / prev_equity * 100 if prev_equity > 0 else 0

            equity_curve.append({
                "date": date_str,
                "equity": round(cash, 2),
                "cash": round(cash, 2),
                "positions_value": 0,
                "n_positions": day_trades,
                "daily_return": round(daily_return, 4),
            })

            peak_equity = max(peak_equity, cash)

        # ── Results ──
        if not equity_curve:
            logger.warning("No options trades generated in backtest")
            return

        metrics = BacktestMetrics(trades, equity_curve, capital)
        results = metrics.summary()

        overview = results.get("overview", {})
        returns_data = results.get("returns", {})
        trades_data = results.get("trades", {})
        risk = results.get("risk", {})
        costs = results.get("cost_analysis", {})
        monthly = results.get("monthly_returns", {})
        regime_perf = results.get("regime_performance", {})

        avg_premium = sum(t.entry_price for t in trades) / len(trades) if trades else 0
        exits = trades_data.get("exit_reasons", {})
        total_pnl = overview.get("final_equity", 0) - capital

        # ── Helper for table printing ──
        def print_table(title: str, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
            """Print a formatted ASCII table with right-aligned numeric columns."""
            if not col_widths:
                col_widths = []
                for ci in range(len(headers)):
                    max_w = len(headers[ci])
                    for row in rows:
                        if ci < len(row):
                            max_w = max(max_w, len(str(row[ci])))
                    col_widths.append(max_w + 2)

            # Detect right-align columns: ₹, %, or pure numeric values
            right_align = set()
            for ci in range(len(headers)):
                for row in rows[:3]:  # Check first few rows
                    if ci < len(row):
                        val = str(row[ci]).strip()
                        if "₹" in val or val.endswith("%") or val.replace(",", "").replace(".", "").replace("-", "").replace("+", "").isdigit():
                            right_align.add(ci)
                            break

            sep = "+" + "+".join("-" * w for w in col_widths) + "+"
            header_line = "|" + "|".join(f" {headers[ci]:<{col_widths[ci]-2}} " for ci in range(len(headers))) + "|"

            logger.info("")
            logger.info(f"  {title}")
            logger.info(sep)
            logger.info(header_line)
            logger.info(sep)
            for row in rows:
                cells = []
                for ci in range(len(headers)):
                    val = str(row[ci]) if ci < len(row) else ""
                    w = col_widths[ci] - 2
                    if ci in right_align:
                        cells.append(f" {val:>{w}} ")
                    else:
                        cells.append(f" {val:<{w}} ")
                logger.info("|" + "|".join(cells) + "|")
            logger.info(sep)

        # ── Debug: skip breakdown ──
        logger.info(
            f"  SKIP BREAKDOWN: vix={_skip_vix} expiry={_skip_expiry} "
            f"consec_sl={_skip_consec_sl} conviction={_skip_conviction} whipsaw={_skip_whipsaw}"
        )

        # ══════════════════════════════════════════
        logger.info("")
        logger.info("=" * 70)
        logger.info("              VELTRIX V4 — BACKTEST RESULTS — NIFTY OPTIONS (CE/PE)")
        logger.info("=" * 70)
        logger.info(f"  Period: {equity_curve[0]['date']} to {equity_curve[-1]['date']} ({trading_days} trading days)")

        # ── 1. Overview Table ──
        print_table("OVERVIEW", ["Metric", "Value"], [
            ["Initial Capital", f"₹{capital:,.2f}"],
            ["Final Capital", f"₹{overview.get('final_equity', 0):,.2f}"],
            ["Total P&L", f"{'+'if total_pnl>=0 else ''}₹{total_pnl:,.2f}"],
            ["Total Return", f"{overview.get('total_return_pct', 0):.2f}%"],
            ["CAGR", f"{returns_data.get('cagr_pct', 0):.2f}%"],
            ["Max Drawdown", f"{risk.get('max_drawdown_pct', 0):.2f}%"],
            ["Sharpe Ratio", f"{returns_data.get('sharpe_ratio', 0):.3f}"],
            ["Sortino Ratio", f"{returns_data.get('sortino_ratio', 0):.3f}"],
        ], [22, 18])

        # ── 2. Trade Statistics Table ──
        print_table("TRADE STATISTICS", ["Metric", "Value"], [
            ["Total Trades", f"{trades_data.get('total_trades', 0)}"],
            ["Win Rate", f"{trades_data.get('win_rate_pct', 0):.1f}%"],
            ["Profit Factor", f"{trades_data.get('profit_factor', 0):.2f}"],
            ["Avg Win", f"₹{trades_data.get('avg_win', 0):,.2f}"],
            ["Avg Loss", f"₹{trades_data.get('avg_loss', 0):,.2f}"],
            ["Largest Win", f"₹{trades_data.get('largest_win', 0):,.2f}"],
            ["Largest Loss", f"₹{trades_data.get('largest_loss', 0):,.2f}"],
            ["Avg Premium", f"₹{avg_premium:.2f}"],
            ["Lot Size", f"{lot_size} (dynamic lots, ₹25K deploy cap)"],
            ["Avg Qty", f"{sum(t.quantity for t in trades) / len(trades):.0f}"],
            ["Avg Position Size", f"₹{sum(t.entry_price * t.quantity for t in trades) / len(trades):,.2f}"],
            ["Total Charges", f"₹{costs.get('total_charges', 0):,.2f}"],
        ], [22, 18])

        # ── 3. Exit Reasons Table ──
        if exits:
            exit_total = sum(int(v) for v in exits.values())
            exit_rows = []
            for reason, count in sorted(exits.items(), key=lambda x: -int(x[1])):
                pct = int(count) / exit_total * 100
                exit_rows.append([reason.replace("_", " ").title(), str(int(count)), f"{pct:.0f}%"])
            print_table("EXIT REASONS", ["Exit Type", "Count", "%"], exit_rows, [20, 8, 8])

        # ── 4. CE vs PE Breakdown Table ──
        if trades:
            ce_trades = [t for t in trades if "CE" in t.symbol]
            pe_trades = [t for t in trades if "PE" in t.symbol]
            ce_pnl = sum(t.pnl for t in ce_trades)
            pe_pnl = sum(t.pnl for t in pe_trades)
            ce_wr = sum(1 for t in ce_trades if t.pnl > 0) / len(ce_trades) * 100 if ce_trades else 0
            pe_wr = sum(1 for t in pe_trades if t.pnl > 0) / len(pe_trades) * 100 if pe_trades else 0
            print_table("CE vs PE BREAKDOWN", ["Direction", "Trades", "P&L", "Win Rate"], [
                ["CE (Call)", str(len(ce_trades)), f"{'+'if ce_pnl>=0 else ''}₹{ce_pnl:,.2f}", f"{ce_wr:.1f}%"],
                ["PE (Put)", str(len(pe_trades)), f"{'+'if pe_pnl>=0 else ''}₹{pe_pnl:,.2f}", f"{pe_wr:.1f}%"],
            ], [14, 10, 16, 10])

        # ── 5. Monthly Breakdown Table ──
        if monthly:
            month_rows = []
            for month, ret in sorted(monthly.items()):
                month_trades_list = [t for t in trades if t.entry_date[:7] == str(month)[:7]]
                month_wins = sum(1 for t in month_trades_list if t.pnl > 0)
                month_pnl = sum(t.pnl for t in month_trades_list)
                wr = (month_wins / len(month_trades_list) * 100) if month_trades_list else 0
                status = "Profit" if month_pnl >= 0 else "Loss"
                month_rows.append([
                    str(month),
                    f"{'+'if month_pnl>=0 else ''}₹{month_pnl:,.2f}",
                    str(len(month_trades_list)),
                    f"{wr:.0f}%",
                    status,
                ])
            profitable_months = sum(1 for r in month_rows if r[4] == "Profit")
            print_table("MONTHLY BREAKDOWN", ["Month", "P&L", "Trades", "WR", "Status"], month_rows, [12, 16, 8, 6, 8])
            logger.info(f"  Profitable Months: {profitable_months}/{len(month_rows)} ({profitable_months/len(month_rows)*100:.0f}%)")

        # ── 6. Regime Performance Table ──
        if regime_perf:
            regime_rows = []
            for regime_name, rdata in regime_perf.items():
                regime_rows.append([
                    regime_name,
                    str(rdata["trades"]),
                    f"{'+'if rdata['total_pnl']>=0 else ''}₹{rdata['total_pnl']:,.2f}",
                    f"{rdata['win_rate']:.1f}%",
                ])
            print_table("REGIME PERFORMANCE", ["Regime", "Trades", "P&L", "WR"], regime_rows, [20, 8, 16, 6])

        # ── 7. Daily P&L Stats Table ──
        if trades:
            from collections import defaultdict
            day_pnl_map = defaultdict(float)
            day_trade_count = defaultdict(int)
            for t in trades:
                day_pnl_map[t.entry_date] += t.pnl
                day_trade_count[t.entry_date] += 1

            daily_pnls = list(day_pnl_map.values())
            days_with_trades = len(daily_pnls)
            avg_daily_pnl = sum(daily_pnls) / days_with_trades
            winning_days = sum(1 for p in daily_pnls if p > 0)
            avg_trades_per_day = len(trades) / days_with_trades

            print_table("DAILY P&L STATS", ["Metric", "Value"], [
                ["Days with Trades", f"{days_with_trades}/{trading_days}"],
                ["Avg Trades/Day", f"{avg_trades_per_day:.1f}"],
                ["Avg Daily P&L", f"₹{avg_daily_pnl:,.2f}"],
                ["Best Day", f"₹{max(daily_pnls):,.2f}"],
                ["Worst Day", f"₹{min(daily_pnls):,.2f}"],
                ["Winning Days", f"{winning_days}/{days_with_trades} ({winning_days/days_with_trades*100:.1f}%)"],
            ], [22, 22])

        # ── 8. Signal Pipeline Table ──
        ml_acc_total = ml_actual_up_count + ml_actual_down_count
        ml_acc_str = f"{ml_correct/ml_acc_total*100:.0f}%" if ml_acc_total > 0 else "N/A"
        ml_rolling_acc_str = (
            f"{sum(ml_rolling_window)/len(ml_rolling_window)*100:.0f}%"
            if ml_rolling_window else "N/A"
        )
        n_models = 1 + (1 if ml_model_xgb is not None else 0) + (1 if ml_model_cat is not None else 0)
        pipeline_rows = [
            ["Signals Generated", str(signals_generated)],
            ["Real Data Trades", str(real_data_trades)],
            ["Estimated Trades", str(estimated_trades)],
            ["Skipped (VIX/Filter)", str(skipped_vix)],
            ["Trades Executed", str(len(trades))],
            ["ML Predictions", str(ml_predictions)],
            ["ML Accuracy", ml_acc_str],
            ["ML Rolling 50 Acc", ml_rolling_acc_str],
            ["ML Auto Weight", f"{ml_auto_weight:.1f}"],
            ["ML Ensemble Models", str(n_models)],
            ["ML Influenced Trades", str(ml_influenced_trades)],
        ]
        print_table("SIGNAL PIPELINE", ["Stage", "Count"], pipeline_rows, [22, 10])

        # ── 8b. ML DIAGNOSTICS (Overfitting Check) ──
        if ml_predictions > 0:
            logger.info("")
            logger.info("  ML DIAGNOSTICS — Overfitting Check")

            # Train vs Test accuracy
            if ml_train_accuracies:
                avg_train_acc = sum(ml_train_accuracies) / len(ml_train_accuracies) * 100
                avg_test_acc = ml_correct / ml_acc_total * 100 if ml_acc_total > 0 else 0
                diag_rows = [
                    ["Avg Train Accuracy", f"{avg_train_acc:.1f}%"],
                    ["Test Accuracy (non-noise)", f"{avg_test_acc:.1f}%"],
                    ["Train-Test Gap", f"{avg_train_acc - avg_test_acc:.1f}%"],
                    ["Walk-Forward Folds", str(len(ml_train_accuracies))],
                ]
                # Class distribution
                diag_rows.append(["Predictions (UP/DOWN)", f"{ml_pred_up_count}/{ml_pred_down_count}"])
                diag_rows.append(["Actual UP (non-noise)", str(ml_actual_up_count)])
                diag_rows.append(["Actual DOWN (non-noise)", str(ml_actual_down_count)])
                if ml_actual_up_count > 0:
                    up_acc = ml_correct_up / ml_actual_up_count * 100
                    diag_rows.append(["Acc on UP days", f"{up_acc:.1f}%"])
                if ml_actual_down_count > 0:
                    down_acc = ml_correct_down / ml_actual_down_count * 100
                    diag_rows.append(["Acc on DOWN days", f"{down_acc:.1f}%"])
                noise_days = ml_predictions - ml_acc_total
                diag_rows.append(["Noise days (excluded)", f"{noise_days} ({noise_days/ml_predictions*100:.0f}%)"])
                print_table("ML DIAGNOSTICS", ["Metric", "Value"], diag_rows, [24, 18])

            # Feature importance top 10
            if ml_feature_importance:
                sorted_imp = sorted(ml_feature_importance.items(), key=lambda x: x[1], reverse=True)
                total_imp = sum(v for _, v in sorted_imp)
                top_10 = sorted_imp[:10]
                imp_rows = []
                for fname, fimp in top_10:
                    pct = fimp / total_imp * 100
                    bar = "█" * int(pct / 2)
                    imp_rows.append([fname, f"{pct:.1f}%", bar])
                print_table("FEATURE IMPORTANCE (Top 10)", ["Feature", "%", ""], imp_rows, [24, 8, 20])

        # ── 9. LOT DISTRIBUTION ──
        if trades:
            from collections import Counter
            lot_counts = Counter()
            for t in trades:
                lots = t.quantity // lot_size if lot_size > 0 else 1
                lot_label = f"{lots}L ({t.quantity}q)"
                lot_counts[lot_label] += 1

            all_lot_labels = sorted(lot_counts.keys(), key=lambda x: int(x.split("L")[0]), reverse=True)
            lot_rows = []
            for label in all_lot_labels:
                lot_rows.append([
                    label,
                    str(lot_counts[label]),
                    f"{lot_counts[label]/len(trades)*100:.0f}%",
                ])
            print_table("LOT DISTRIBUTION", ["Lots", "Trades", "%"], lot_rows, [12, 8, 6])

        # ── 10. TRADE TYPE PERFORMANCE ──
        if trades:
            type_rows = []
            for ttype in ["FULL"]:
                tt_trades = [t for t in trades if t.strategy == ttype]
                if not tt_trades:
                    type_rows.append([ttype, "0", "₹0", "0%", "0.00"])
                    continue
                tt_pnl = sum(t.pnl for t in tt_trades)
                tt_wins = sum(1 for t in tt_trades if t.pnl > 0)
                tt_wr = tt_wins / len(tt_trades) * 100
                tt_gross_win = sum(t.pnl for t in tt_trades if t.pnl > 0)
                tt_gross_loss = abs(sum(t.pnl for t in tt_trades if t.pnl < 0))
                tt_pf = tt_gross_win / tt_gross_loss if tt_gross_loss > 0 else float("inf")
                tt_avg_qty = sum(t.quantity for t in tt_trades) / len(tt_trades)
                type_rows.append([
                    ttype,
                    str(len(tt_trades)),
                    f"{'+'if tt_pnl>=0 else ''}₹{tt_pnl:,.2f}",
                    f"{tt_wr:.1f}%",
                    f"{tt_pf:.2f}",
                    f"{tt_avg_qty:.0f}",
                ])
            print_table("TRADE TYPE PERFORMANCE", ["Type", "Trades", "P&L", "WR", "PF", "Avg Qty"], type_rows, [10, 8, 16, 8, 8, 10])

        # ── 11. All Trades Detail ──
        if trades:
            trade_rows = []
            running_pnl = 0.0
            for idx, t in enumerate(trades, 1):
                running_pnl += t.pnl
                pnl_sign = "+" if t.pnl >= 0 else ""
                prem_chg = ((t.exit_price - t.entry_price) / t.entry_price * 100) if t.entry_price > 0 else 0
                lot_cost = t.entry_price * t.quantity
                trade_rows.append([
                    str(idx),
                    t.entry_date,
                    t.strategy[:4],
                    t.symbol,
                    f"₹{t.entry_price:.2f}",
                    str(t.quantity),
                    f"₹{lot_cost:,.2f}",
                    f"₹{t.exit_price:.2f}",
                    f"{prem_chg:+.1f}%",
                    t.exit_reason.replace("_", " ").title(),
                    f"{pnl_sign}₹{t.pnl:,.2f}",
                    f"₹{running_pnl:,.2f}",
                    t.regime[:4] if t.regime else "-",
                ])
            print_table(
                f"ALL TRADES ({len(trades)} trades)",
                ["#", "Date", "Type", "Symbol", "Entry", "Qty", "Cost", "Exit", "Chg%", "Exit Type", "P&L", "Cumul", "Rgm"],
                trade_rows,
                [5, 12, 6, 18, 10, 5, 12, 10, 7, 14, 14, 14, 6],
            )

        # ── 12. VALIDATION ──
        if trades:
            logger.info("")
            logger.info("=" * 70)
            logger.info("  VALIDATION — Constraint Checks")
            logger.info("=" * 70)

            violations = 0

            # V1: No position cost > ₹25K
            max_cost_trade = max(trades, key=lambda t: t.entry_price * t.quantity)
            max_cost = max_cost_trade.entry_price * max_cost_trade.quantity
            v1_pass = max_cost <= BT_MAX_DEPLOY
            if not v1_pass:
                violations += 1
            logger.info(f"  [{'PASS' if v1_pass else 'FAIL'}] Deploy cap ₹25K: max position cost = ₹{max_cost:,.2f}")

            # V2: Risk ≤ ₹10K per trade
            risk_violations = 0
            for t in trades:
                sl_loss = abs(t.entry_price - t.stop_loss) * t.quantity
                if sl_loss > BT_MAX_RISK * 1.1:  # 10% tolerance for rounding
                    risk_violations += 1
            v2_pass = risk_violations == 0
            if not v2_pass:
                violations += 1
            logger.info(f"  [{'PASS' if v2_pass else 'FAIL'}] Risk ≤ ₹10K: {risk_violations} violations ({len(trades)} trades)")

            # V3: Min premium ≥ ₹80
            min_prem_trade = min(trades, key=lambda t: t.entry_price)
            v7_pass = min_prem_trade.entry_price >= 80.0
            if not v7_pass:
                violations += 1
            logger.info(f"  [{'PASS' if v7_pass else 'FAIL'}] Min premium ₹80: min = ₹{min_prem_trade.entry_price:.2f}")

            # Summary
            logger.info("")
            if violations == 0:
                logger.info("  ALL VALIDATIONS PASSED")
            else:
                logger.info(f"  {violations} VALIDATION(S) FAILED — review above")
            logger.info("=" * 70)

    def _wait_until(self, target: dt_time) -> None:
        """Wait until a specific time of day. Skipped if --no-wait."""
        if self._skip_wait:
            logger.debug(f"Skipping wait for {target.strftime('%H:%M')} (--no-wait)")
            return

        now = datetime.now().time()
        if now >= target:
            return

        target_dt = datetime.combine(date.today(), target)
        wait_seconds = (target_dt - datetime.now()).total_seconds()
        if wait_seconds > 0 and self._running:
            logger.info(f"Waiting until {target.strftime('%H:%M')} ({wait_seconds/60:.0f} min)")
            # Sleep in chunks for graceful shutdown
            while wait_seconds > 0 and self._running:
                time.sleep(min(wait_seconds, 10))
                wait_seconds -= 10

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.data_fetcher.stop_market_stream()
        self.data_fetcher.stop_portfolio_stream()
        self.store.close()
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="VELTRIX — AI Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "fetch"],
        default="paper",
        help="Trading mode: live|paper|backtest|fetch (default: paper)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip time-based waiting (for testing outside market hours)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Override capital for backtest (e.g., --capital 500000)",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force full re-fetch of all data (ignore DB cache, skip incremental checks)",
    )
    parser.add_argument(
        "--fetch-expired",
        action="store_true",
        help="Fetch expired F&O contracts (slow, 200+ API calls — only needed once)",
    )
    parser.add_argument(
        "--active-trading",
        action="store_true",
        help="Active trading mode: 5 trades/day, direction unlocked, lower VOLATILE threshold, 15-min cooldown, 2 SL halt",
    )

    args = parser.parse_args()

    # Load .env file (secrets)
    _load_dotenv(os.path.join(_project_root, ".env"))

    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    bot = TradingBot(mode=args.mode, config_path=args.config)
    if args.no_wait:
        bot._skip_wait = True
    if args.force_fetch:
        bot._force_fetch = True
    if args.fetch_expired:
        bot._fetch_expired = True
    if args.active_trading:
        bot._active_trading = True
        bot.options_buyer.set_active_trading(True)
    if args.capital and args.mode == "backtest":
        bot._backtest_capital = args.capital
    bot.run()


if __name__ == "__main__":
    main()
