"""
Database Storage — SQLite (dev) / TimescaleDB (prod).

Tables: candles, fii_dii, option_chain, delivery_data, external_data, trades, signals, regime_history
All operations are thread-safe with connection pooling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional, Generator

import pandas as pd
import yaml
from loguru import logger

from src.config.env_loader import get_config


class DataStore:
    """
    Persistent storage for all trading data.
    Uses SQLite for development, with TimescaleDB support for production.
    """

    SCHEMA = {
        "candles": """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                instrument_key TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                oi INTEGER DEFAULT 0,
                interval TEXT DEFAULT 'day',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime, interval)
            )
        """,
        "fii_dii": """
            CREATE TABLE IF NOT EXISTS fii_dii (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                fii_buy_value REAL DEFAULT 0,
                fii_sell_value REAL DEFAULT 0,
                fii_net_value REAL DEFAULT 0,
                dii_buy_value REAL DEFAULT 0,
                dii_sell_value REAL DEFAULT 0,
                dii_net_value REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "option_chain": """
            CREATE TABLE IF NOT EXISTS option_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                expiry_date TEXT NOT NULL,
                strike_price REAL NOT NULL,
                ce_oi INTEGER DEFAULT 0,
                ce_change_oi INTEGER DEFAULT 0,
                ce_ltp REAL DEFAULT 0,
                ce_volume INTEGER DEFAULT 0,
                ce_iv REAL DEFAULT 0,
                pe_oi INTEGER DEFAULT 0,
                pe_change_oi INTEGER DEFAULT 0,
                pe_ltp REAL DEFAULT 0,
                pe_volume INTEGER DEFAULT 0,
                pe_iv REAL DEFAULT 0,
                underlying_value REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, expiry_date, strike_price)
            )
        """,
        "delivery_data": """
            CREATE TABLE IF NOT EXISTS delivery_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                close REAL DEFAULT 0,
                prev_close REAL DEFAULT 0,
                change_pct REAL DEFAULT 0,
                traded_qty INTEGER DEFAULT 0,
                delivered_qty INTEGER DEFAULT 0,
                delivery_pct REAL DEFAULT 0,
                traded_value_cr REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, symbol)
            )
        """,
        "external_data": """
            CREATE TABLE IF NOT EXISTS external_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL DEFAULT 0,
                high REAL DEFAULT 0,
                low REAL DEFAULT 0,
                close REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, symbol)
            )
        """,
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                order_id TEXT,
                symbol TEXT NOT NULL,
                instrument_key TEXT,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                order_type TEXT DEFAULT 'MARKET',
                product TEXT DEFAULT 'I',
                strategy TEXT,
                signal_score REAL DEFAULT 0,
                regime TEXT,
                stop_loss REAL DEFAULT 0,
                take_profit REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                signal_price REAL DEFAULT 0,
                fill_price REAL DEFAULT 0,
                fill_quantity INTEGER DEFAULT 0,
                slippage_pct REAL DEFAULT 0,
                brokerage REAL DEFAULT 0,
                stt REAL DEFAULT 0,
                total_charges REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                entry_time TEXT,
                exit_time TEXT,
                hold_duration_hours REAL DEFAULT 0,
                notes TEXT,
                mode TEXT DEFAULT 'paper',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "signals": """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                score REAL DEFAULT 0,
                regime TEXT,
                features TEXT,
                ensemble_score REAL DEFAULT 0,
                action_taken TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "regime_history": """
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                regime TEXT NOT NULL,
                vix_value REAL DEFAULT 0,
                nifty_value REAL DEFAULT 0,
                adx_value REAL DEFAULT 0,
                fii_net_value REAL DEFAULT 0,
                active_strategies TEXT,
                size_multiplier REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "portfolio_snapshots": """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT NOT NULL,
                total_value REAL DEFAULT 0,
                cash REAL DEFAULT 0,
                invested REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                day_pnl REAL DEFAULT 0,
                positions_count INTEGER DEFAULT 0,
                exposure_pct REAL DEFAULT 0,
                drawdown_pct REAL DEFAULT 0,
                mode TEXT DEFAULT 'paper',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "instrument_registry": """
            CREATE TABLE IF NOT EXISTS instrument_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                instrument_type TEXT NOT NULL,
                exchange TEXT NOT NULL,
                upstox_symbol TEXT NOT NULL,
                lot_size INTEGER NOT NULL,
                tick_size REAL NOT NULL,
                options_expiry TEXT NOT NULL,
                vix_multiplier REAL DEFAULT 1.0,
                adx_threshold REAL DEFAULT 22,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "instrument_daily_log": """
            CREATE TABLE IF NOT EXISTS instrument_daily_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                instrument TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                prev_close REAL, change_pct REAL,
                regime TEXT, adx REAL, adx_slope REAL, bb_width REAL,
                bull_score REAL, bear_score REAL, score_diff REAL,
                direction TEXT, conviction REAL,
                pcr REAL, max_call_oi_strike REAL, max_put_oi_strike REAL,
                max_pain REAL, atm_ce_premium REAL, atm_pe_premium REAL, atm_iv REAL,
                vix_level REAL, vix_change_pct REAL,
                rsi_14 REAL, macd_signal REAL, ema9 REAL, ema21 REAL, ema50 REAL,
                would_trade INTEGER DEFAULT 0, trade_type TEXT,
                signal_strength TEXT, blocking_reason TEXT,
                fii_net REAL, dii_net REAL,
                scored_at TEXT NOT NULL,
                UNIQUE(date, instrument)
            )
        """,
        "instrument_signal_log": """
            CREATE TABLE IF NOT EXISTS instrument_signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                instrument TEXT NOT NULL,
                signal_time TEXT NOT NULL,
                oi_score_diff REAL, oi_bull_score REAL, oi_bear_score REAL,
                pcr REAL, vix_level REAL, vix_change_pct REAL,
                rsi_14 REAL, adx_14 REAL, regime TEXT, direction TEXT,
                entry_hour INTEGER, dist_from_open REAL, days_to_expiry INTEGER,
                conviction REAL, trade_type TEXT,
                would_buy_symbol TEXT, would_buy_strike REAL,
                would_buy_premium REAL, would_buy_qty INTEGER,
                would_buy_sl REAL, would_buy_tp REAL,
                eod_premium REAL, eod_pnl_pct REAL, eod_result TEXT,
                peak_premium REAL, peak_pct REAL,
                trough_premium REAL, trough_pct REAL,
                not_traded_reason TEXT,
                scored_at TEXT NOT NULL
            )
        """,
        # ── Two-Stage ML System Tables ──
        "ml_candles_5min": """
            CREATE TABLE IF NOT EXISTS ml_candles_5min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                instrument_key TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                oi INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime)
            )
        """,
        "ml_features_cache": """
            CREATE TABLE IF NOT EXISTS ml_features_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                session TEXT NOT NULL DEFAULT 'full_day',
                features_json TEXT NOT NULL,
                feature_version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, session, feature_version)
            )
        """,
        "ml_models": """
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                model_type TEXT NOT NULL DEFAULT 'xgboost',
                stage TEXT NOT NULL,
                train_date TEXT NOT NULL,
                train_samples INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                train_accuracy REAL DEFAULT 0,
                test_accuracy REAL DEFAULT 0,
                train_test_gap REAL DEFAULT 0,
                deployed INTEGER DEFAULT 0,
                deploy_gate_passed INTEGER DEFAULT 0,
                model_path TEXT NOT NULL,
                scaler_path TEXT,
                feature_list TEXT,
                hyperparams TEXT,
                metrics_json TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, model_version)
            )
        """,
        "ml_predictions": """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_time TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                prob_ce REAL DEFAULT 0,
                prob_pe REAL DEFAULT 0,
                prob_flat REAL DEFAULT 0,
                actual_class TEXT,
                correct INTEGER,
                features_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "ml_trade_labels": """
            CREATE TABLE IF NOT EXISTS ml_trade_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                trade_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                regime TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                label INTEGER NOT NULL,
                score_diff REAL,
                conviction REAL,
                vix_at_entry REAL,
                rsi_at_entry REAL,
                adx_at_entry REAL,
                pcr_at_entry REAL,
                ml_prob_ce REAL,
                ml_prob_pe REAL,
                trigger_count INTEGER,
                features_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "ic_trades": """
            CREATE TABLE IF NOT EXISTS ic_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE NOT NULL,
                entry_time TEXT,
                regime TEXT,
                spot_at_entry REAL DEFAULT 0,
                quantity INTEGER DEFAULT 0,
                lots INTEGER DEFAULT 0,
                sell_ce_strike REAL DEFAULT 0,
                sell_ce_premium REAL DEFAULT 0,
                buy_ce_strike REAL DEFAULT 0,
                buy_ce_premium REAL DEFAULT 0,
                sell_pe_strike REAL DEFAULT 0,
                sell_pe_premium REAL DEFAULT 0,
                buy_pe_strike REAL DEFAULT 0,
                buy_pe_premium REAL DEFAULT 0,
                net_credit REAL DEFAULT 0,
                spread_width INTEGER DEFAULT 200,
                max_profit REAL DEFAULT 0,
                max_loss REAL DEFAULT 0,
                tp_threshold REAL DEFAULT 0,
                sl_threshold REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                charges REAL DEFAULT 0,
                exit_reason TEXT,
                exit_time TEXT,
                status TEXT DEFAULT 'open',
                expiry_type TEXT,
                trade_type TEXT DEFAULT 'IRON_CONDOR',
                mode TEXT DEFAULT 'paper',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "reconciliation_log": """
            CREATE TABLE IF NOT EXISTS reconciliation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                system_pnl REAL DEFAULT 0,
                broker_pnl REAL DEFAULT 0,
                difference REAL DEFAULT 0,
                trade_count_system INTEGER DEFAULT 0,
                trade_count_broker INTEGER DEFAULT 0,
                status TEXT DEFAULT 'OK',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "factor_edge_history": """
            CREATE TABLE IF NOT EXISTS factor_edge_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                aligned_wr REAL DEFAULT 0,
                against_wr REAL DEFAULT 0,
                net_edge REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                window_days INTEGER DEFAULT 90,
                UNIQUE(date, factor_name)
            )
        """,
        "counterfactual_trades": """
            CREATE TABLE IF NOT EXISTS counterfactual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                block_reason TEXT NOT NULL,
                block_time TEXT,
                regime TEXT,
                score_diff REAL DEFAULT 0,
                bull_score REAL DEFAULT 0,
                bear_score REAL DEFAULT 0,
                spot_at_block REAL DEFAULT 0,
                spot_at_eod REAL DEFAULT 0,
                hypothetical_pnl REAL DEFAULT 0,
                hypothetical_pct REAL DEFAULT 0,
                would_have_won INTEGER DEFAULT 0,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "live_slippage_log": """
            CREATE TABLE IF NOT EXISTS live_slippage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                slippage_pct REAL NOT NULL,
                slippage_amount REAL NOT NULL,
                quantity INTEGER NOT NULL,
                direction TEXT NOT NULL,
                mode TEXT DEFAULT 'live',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """,
    }

    INDICES = [
        "CREATE INDEX IF NOT EXISTS idx_candles_symbol_dt ON candles(symbol, datetime)",
        "CREATE INDEX IF NOT EXISTS idx_candles_interval ON candles(interval)",
        "CREATE INDEX IF NOT EXISTS idx_fii_dii_date ON fii_dii(date)",
        "CREATE INDEX IF NOT EXISTS idx_option_chain_symbol_date ON option_chain(symbol, date)",
        "CREATE INDEX IF NOT EXISTS idx_delivery_date ON delivery_data(date)",
        "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
        "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_regime_dt ON regime_history(datetime)",
        "CREATE INDEX IF NOT EXISTS idx_external_date_symbol ON external_data(date, symbol)",
        "CREATE INDEX IF NOT EXISTS idx_inst_daily_date_inst ON instrument_daily_log(date, instrument)",
        "CREATE INDEX IF NOT EXISTS idx_inst_signal_date_inst ON instrument_signal_log(date, instrument)",
        # ML indices
        "CREATE INDEX IF NOT EXISTS idx_ml_candles_sym_dt ON ml_candles_5min(symbol, datetime)",
        "CREATE INDEX IF NOT EXISTS idx_ml_features_sym_dt ON ml_features_cache(symbol, date)",
        "CREATE INDEX IF NOT EXISTS idx_ml_models_name ON ml_models(model_name, deployed)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_date ON ml_predictions(prediction_date)",
        "CREATE INDEX IF NOT EXISTS idx_ml_trade_labels_date ON ml_trade_labels(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_factor_edge_date ON factor_edge_history(date)",
        "CREATE INDEX IF NOT EXISTS idx_counterfactual_date ON counterfactual_trades(date, block_reason)",
        "CREATE INDEX IF NOT EXISTS idx_slippage_trade ON live_slippage_log(trade_id)",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        db_config = config.get("database", {})
        self.engine = db_config.get("engine", "sqlite")

        if self.engine == "sqlite":
            cfg = get_config()
            db_path = cfg.DB_PATH or db_config.get("sqlite_path", "data/trading_bot.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_path = db_path
        else:
            self.db_config = db_config.get("timescaledb", {})

        self._local = threading.local()
        self._initialize_schema()
        self._apply_migrations()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Thread-safe SQLite connection with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                timeout=30,
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB
            self._local.conn.row_factory = sqlite3.Row

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def cleanup_corrupt_trades(self) -> int:
        """Remove corrupt trades (zero price or missing entry_time). Returns count deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM trades WHERE (price = 0 OR entry_time = '' OR entry_time IS NULL)"
            )
            deleted = cursor.rowcount
            conn.execute("DELETE FROM ml_trade_labels WHERE entry_price = 0")
            conn.commit()
            return deleted

    def _initialize_schema(self) -> None:
        """Create all tables and indices."""
        with self._get_connection() as conn:
            for table_name, ddl in self.SCHEMA.items():
                conn.execute(ddl)
                logger.debug(f"Table '{table_name}' ensured")

            for idx_sql in self.INDICES:
                conn.execute(idx_sql)

            conn.commit()
            # Checkpoint WAL to flush data into main DB file
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info(f"Database initialized ({self.engine}): {len(self.SCHEMA)} tables")

    def _apply_migrations(self) -> None:
        """Add missing columns to existing tables (non-destructive)."""
        greek_columns = [
            "ce_delta", "ce_theta", "ce_gamma", "ce_vega",
            "pe_delta", "pe_theta", "pe_gamma", "pe_vega",
        ]
        with self._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(option_chain)")
            existing = {row[1] for row in cursor.fetchall()}

            added = 0
            for col in greek_columns:
                if col not in existing:
                    conn.execute(
                        f"ALTER TABLE option_chain ADD COLUMN {col} REAL DEFAULT 0"
                    )
                    added += 1

            if added:
                conn.commit()
                logger.info(f"Migration: added {added} Greek columns to option_chain")

            # Add mode column to trades and portfolio_snapshots
            for table in ["trades", "portfolio_snapshots"]:
                try:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN mode TEXT DEFAULT 'paper'"
                    )
                    conn.commit()
                    logger.info(f"Migration: added mode column to {table}")
                except Exception:
                    pass  # Column already exists — expected

            # Add slippage tracking columns to trades
            for col, col_type in [
                ("signal_price", "REAL DEFAULT 0"),
                ("slippage_pct", "REAL DEFAULT 0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
                    conn.commit()
                    logger.info(f"Migration: added {col} column to trades")
                except Exception:
                    pass  # Column already exists — expected

    # ─────────────────────────────────────────
    # Candles (OHLCV)
    # ─────────────────────────────────────────

    def save_candles(
        self,
        symbol: str,
        instrument_key: str,
        df: pd.DataFrame,
        interval: str = "day",
    ) -> int:
        """Save OHLCV candles to database. Returns number of rows inserted."""
        if df.empty:
            return 0

        rows = [
            (
                symbol,
                instrument_key,
                str(row["datetime"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
                int(row.get("oi", 0)),
                interval,
            )
            for _, row in df.iterrows()
        ]

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                (symbol, instrument_key, datetime, open, high, low, close, volume, oi, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            # Periodic WAL checkpoint to flush data to main DB file
            if len(rows) >= 100:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        logger.debug(f"Saved {len(rows)} candles for {symbol} ({interval})")
        return len(rows)

    def get_candles(
        self,
        symbol: str,
        interval: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Load candles from database."""
        query = "SELECT * FROM candles WHERE symbol = ? AND interval = ?"
        params: list[Any] = [symbol, interval]

        if from_date:
            query += " AND datetime >= ?"
            params.append(from_date)
        if to_date:
            query += " AND datetime <= ?"
            params.append(to_date)

        query += " ORDER BY datetime DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["datetime"] = pd.to_datetime(
                df["datetime"].str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True),
                format="mixed",
            )
            df = df.sort_values("datetime").reset_index(drop=True)

        return df

    # ─────────────────────────────────────────
    # FII/DII Data
    # ─────────────────────────────────────────

    def save_fii_dii(self, data: dict[str, Any]) -> None:
        """Save FII/DII data."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fii_dii
                (date, fii_buy_value, fii_sell_value, fii_net_value,
                 dii_buy_value, dii_sell_value, dii_net_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("date", date.today().isoformat()),
                    data.get("fii_buy_value", 0),
                    data.get("fii_sell_value", 0),
                    data.get("fii_net_value", 0),
                    data.get("dii_buy_value", 0),
                    data.get("dii_sell_value", 0),
                    data.get("dii_net_value", 0),
                ),
            )
            conn.commit()

    def get_fii_dii_history(self, days: int = 30) -> pd.DataFrame:
        """Get FII/DII history for last N days."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM fii_dii ORDER BY date DESC LIMIT ?",
                conn,
                params=[days],
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def save_fii_dii_bulk(self, records: list[dict[str, Any]]) -> int:
        """Batch insert FII/DII records (for CSV backfill)."""
        if not records:
            return 0

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO fii_dii
                (date, fii_buy_value, fii_sell_value, fii_net_value,
                 dii_buy_value, dii_sell_value, dii_net_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("date", ""),
                        r.get("fii_buy_value", 0),
                        r.get("fii_sell_value", 0),
                        r.get("fii_net_value", 0),
                        r.get("dii_buy_value", 0),
                        r.get("dii_sell_value", 0),
                        r.get("dii_net_value", 0),
                    )
                    for r in records
                ],
            )
            conn.commit()
        return len(records)

    def get_fii_dii_coverage(self) -> dict[str, Any]:
        """Get FII/DII data coverage — date range and row count (excluding zero-only rows)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT MIN(date) as min_dt, MAX(date) as max_dt, COUNT(*) as cnt
                FROM fii_dii
                WHERE abs(fii_net_value) > 0 OR abs(dii_net_value) > 0
                """,
            )
            row = cursor.fetchone()

        if row and row["cnt"] > 0:
            return {
                "from_date": str(row["min_dt"])[:10],
                "to_date": str(row["max_dt"])[:10],
                "rows": row["cnt"],
            }
        return {"from_date": None, "to_date": None, "rows": 0}

    def has_fii_dii_for_date(self, dt: str) -> bool:
        """Check if real (non-zero) FII/DII data exists for a given date."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM fii_dii
                WHERE date = ? AND (abs(fii_net_value) > 0 OR abs(dii_net_value) > 0)
                """,
                (dt,),
            )
            return cursor.fetchone() is not None

    # ─────────────────────────────────────────
    # External Market Data
    # ─────────────────────────────────────────

    def save_external_data(self, symbol: str, df: pd.DataFrame) -> int:
        """Save external market data (S&P 500, NASDAQ, Crude, Gold, USD/INR)."""
        if df.empty:
            return 0

        rows = 0
        with self._get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO external_data
                        (date, symbol, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(row.get("date", "")),
                            symbol,
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            int(row.get("volume", 0)),
                        ),
                    )
                    rows += 1
                except Exception as e:
                    logger.warning(
                        f"EXTERNAL_DATA_SAVE_FAILED: {symbol} "
                        f"date={row.get('date', '?')} err={e}"
                    )
            conn.commit()
        return rows

    def get_external_data(
        self, symbol: str, from_date: str | None = None, to_date: str | None = None
    ) -> pd.DataFrame:
        """Get external market data for a single symbol."""
        query = "SELECT date, open, high, low, close, volume FROM external_data WHERE symbol = ?"
        params: list[Any] = [symbol]

        if from_date:
            query += " AND date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND date <= ?"
            params.append(to_date)

        query += " ORDER BY date ASC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_external_data_all(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> pd.DataFrame:
        """Get all external market data (long format: date, symbol, close)."""
        query = "SELECT date, symbol, close FROM external_data WHERE 1=1"
        params: list[Any] = []

        if from_date:
            query += " AND date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND date <= ?"
            params.append(to_date)

        query += " ORDER BY date ASC, symbol ASC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_external_data_coverage(self, symbol: str) -> dict[str, Any]:
        """Get data coverage for an external market symbol."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT MIN(date) as from_date, MAX(date) as to_date, COUNT(*) as rows "
                "FROM external_data WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
        return {
            "symbol": symbol,
            "from_date": row[0] if row and row[0] else "",
            "to_date": row[1] if row and row[1] else "",
            "rows": row[2] if row else 0,
        }

    # ─────────────────────────────────────────
    # Options Chain
    # ─────────────────────────────────────────

    def save_option_chain(self, symbol: str, df: pd.DataFrame) -> int:
        """Save options chain snapshot."""
        if df.empty:
            return 0

        today = date.today().isoformat()
        rows_inserted = 0

        with self._get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO option_chain
                        (symbol, date, expiry_date, strike_price,
                         ce_oi, ce_change_oi, ce_ltp, ce_volume, ce_iv,
                         pe_oi, pe_change_oi, pe_ltp, pe_volume, pe_iv,
                         underlying_value,
                         ce_delta, ce_theta, ce_gamma, ce_vega,
                         pe_delta, pe_theta, pe_gamma, pe_vega)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            today,
                            str(row.get("expiryDate", "")),
                            float(row.get("strikePrice", 0)),
                            int(row.get("CE_oi", 0)),
                            int(row.get("CE_changeinOI", 0)),
                            float(row.get("CE_ltp", 0)),
                            int(row.get("CE_volume", 0)),
                            float(row.get("CE_iv", 0)),
                            int(row.get("PE_oi", 0)),
                            int(row.get("PE_changeinOI", 0)),
                            float(row.get("PE_ltp", 0)),
                            int(row.get("PE_volume", 0)),
                            float(row.get("PE_iv", 0)),
                            float(row.get("underlying_value", 0)),
                            float(row.get("CE_delta", 0)),
                            float(row.get("CE_theta", 0)),
                            float(row.get("CE_gamma", 0)),
                            float(row.get("CE_vega", 0)),
                            float(row.get("PE_delta", 0)),
                            float(row.get("PE_theta", 0)),
                            float(row.get("PE_gamma", 0)),
                            float(row.get("PE_vega", 0)),
                        ),
                    )
                    rows_inserted += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()

        return rows_inserted

    def get_option_chain_history(
        self,
        symbol: str = "NIFTY",
        days: int = 252,
    ) -> pd.DataFrame:
        """
        Get historical option chain snapshots (ATM strikes) for feature computation.

        Returns DataFrame with columns:
            date, strike_price, underlying_value,
            ce_oi, pe_oi, ce_iv, pe_iv, ce_ltp, pe_ltp,
            ce_volume, pe_volume, ce_change_oi, pe_change_oi,
            ce_delta, ce_theta, ce_gamma, ce_vega,
            pe_delta, pe_theta, pe_gamma, pe_vega
        """
        from_date = (date.today() - __import__("datetime").timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT date, strike_price, underlying_value,
                       ce_oi, pe_oi, ce_iv, pe_iv, ce_ltp, pe_ltp,
                       ce_volume, pe_volume, ce_change_oi, pe_change_oi,
                       ce_delta, ce_theta, ce_gamma, ce_vega,
                       pe_delta, pe_theta, pe_gamma, pe_vega
                FROM option_chain
                WHERE symbol = ? AND date >= ?
                ORDER BY date, strike_price
                """,
                conn,
                params=(symbol, from_date),
            )
        return df

    def get_option_chain_atm_history(
        self,
        symbol: str = "NIFTY",
        days: int = 252,
    ) -> pd.DataFrame:
        """
        Get ATM option chain history — one row per date (nearest strike to spot).

        Picks the strike closest to underlying_value for each date and
        aggregates chain-wide OI/volume totals + computes PCR.
        """
        raw = self.get_option_chain_history(symbol, days)
        if raw.empty:
            return pd.DataFrame()

        rows = []
        for dt, group in raw.groupby("date"):
            if group.empty or group["underlying_value"].iloc[0] <= 0:
                continue

            spot = group["underlying_value"].iloc[0]

            # ATM strike = nearest to spot
            group = group.copy()
            group["_dist"] = (group["strike_price"] - spot).abs()
            atm = group.loc[group["_dist"].idxmin()]

            # Chain-wide aggregates
            total_ce_oi = int(group["ce_oi"].sum())
            total_pe_oi = int(group["pe_oi"].sum())
            total_ce_vol = int(group["ce_volume"].sum())
            total_pe_vol = int(group["pe_volume"].sum())
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_volume = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

            rows.append({
                "date": dt,
                "underlying": spot,
                "atm_strike": float(atm["strike_price"]),
                # ATM Greeks & IV
                "atm_ce_iv": float(atm.get("ce_iv", 0)),
                "atm_pe_iv": float(atm.get("pe_iv", 0)),
                "atm_ce_delta": float(atm.get("ce_delta", 0)),
                "atm_pe_delta": float(atm.get("pe_delta", 0)),
                "atm_ce_theta": float(atm.get("ce_theta", 0)),
                "atm_pe_theta": float(atm.get("pe_theta", 0)),
                "atm_ce_gamma": float(atm.get("ce_gamma", 0)),
                "atm_pe_gamma": float(atm.get("pe_gamma", 0)),
                "atm_ce_vega": float(atm.get("ce_vega", 0)),
                "atm_pe_vega": float(atm.get("pe_vega", 0)),
                "atm_ce_ltp": float(atm.get("ce_ltp", 0)),
                "atm_pe_ltp": float(atm.get("pe_ltp", 0)),
                "atm_ce_oi": int(atm.get("ce_oi", 0)),
                "atm_pe_oi": int(atm.get("pe_oi", 0)),
                # Chain-wide
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "total_ce_volume": total_ce_vol,
                "total_pe_volume": total_pe_vol,
                "pcr_oi": round(pcr_oi, 4),
                "pcr_volume": round(pcr_volume, 4),
            })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────
    # Delivery Data
    # ─────────────────────────────────────────

    def save_delivery_data(self, df: pd.DataFrame, target_date: Optional[date] = None) -> int:
        """Save delivery volume data."""
        if df.empty:
            return 0

        target_date = target_date or date.today()
        date_str = target_date.isoformat()

        def _safe_int(v, default=0):
            try:
                return int(v)
            except (ValueError, TypeError):
                return default

        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        rows = [
            (
                date_str,
                row.get("symbol", ""),
                _safe_float(row.get("close", 0)),
                _safe_float(row.get("prev_close", 0)),
                _safe_float(row.get("change_pct", 0)),
                _safe_int(row.get("traded_qty", 0)),
                _safe_int(row.get("delivered_qty", 0)),
                _safe_float(row.get("delivery_pct", 0)),
                _safe_float(row.get("traded_value_cr", 0)),
            )
            for _, row in df.iterrows()
        ]

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO delivery_data
                (date, symbol, close, prev_close, change_pct,
                 traded_qty, delivered_qty, delivery_pct, traded_value_cr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def get_delivery_history(
        self, symbol: str, days: int = 20
    ) -> pd.DataFrame:
        """Get delivery data history for a symbol."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM delivery_data
                WHERE symbol = ?
                ORDER BY date DESC LIMIT ?
                """,
                conn,
                params=[symbol, days],
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # ─────────────────────────────────────────
    # Trades
    # ─────────────────────────────────────────

    def save_trade(self, trade: dict[str, Any]) -> bool:
        """Save a trade record with retry on DB failure."""
        import time as _time
        trade_id = trade.get("trade_id", "unknown")
        for attempt in range(3):
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO trades
                        (trade_id, order_id, symbol, instrument_key, side, quantity, price,
                         order_type, product, strategy, signal_score, regime,
                         stop_loss, take_profit, status, signal_price, fill_price, fill_quantity,
                         slippage_pct, brokerage, stt, total_charges, pnl,
                         entry_time, exit_time, hold_duration_hours, notes, mode, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            trade_id,
                            trade.get("order_id", ""),
                            trade.get("symbol", ""),
                            trade.get("instrument_key", ""),
                            trade.get("side", ""),
                            trade.get("quantity", 0),
                            trade.get("price", 0),
                            trade.get("order_type", "MARKET"),
                            trade.get("product", "I"),
                            trade.get("strategy", ""),
                            trade.get("signal_score", 0),
                            trade.get("regime", ""),
                            trade.get("stop_loss", 0),
                            trade.get("take_profit", 0),
                            trade.get("status", "pending"),
                            trade.get("signal_price", 0),
                            trade.get("fill_price", 0),
                            trade.get("fill_quantity", 0),
                            trade.get("slippage_pct", 0),
                            trade.get("brokerage", 0),
                            trade.get("stt", 0),
                            trade.get("total_charges", 0),
                            trade.get("pnl", 0),
                            trade.get("entry_time", ""),
                            trade.get("exit_time", ""),
                            trade.get("hold_duration_hours", 0),
                            trade.get("notes", ""),
                            trade.get("mode", "paper"),
                        ),
                    )
                    conn.commit()
                return True
            except sqlite3.OperationalError as e:
                if attempt < 2:
                    _time.sleep(0.5)
                    continue
                logger.error(f"SAVE_TRADE_FAILED: {trade_id} after 3 attempts: {e}")
                return False

    def update_trade_status(
        self, trade_id: str, status: str, **kwargs: Any
    ) -> None:
        """Update trade status and optional fields."""
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [status]

        for key, value in kwargs.items():
            if key in (
                "fill_price", "fill_quantity", "pnl", "exit_time",
                "hold_duration_hours", "brokerage", "stt", "total_charges", "notes",
            ):
                updates.append(f"{key} = ?")
                params.append(value)

        params.append(trade_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE trades SET {', '.join(updates)} WHERE trade_id = ?",
                params,
            )
            conn.commit()

    def get_trades(
        self,
        status: Optional[str] = None,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query trades with filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if from_date:
            query += " AND created_at >= ?"
            params.append(from_date)
        if mode:
            query += " AND mode = ?"
            params.append(mode)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_open_positions(self) -> pd.DataFrame:
        """Get all open (filled but not exited) trades."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE status = 'filled' ORDER BY entry_time",
                conn,
            )

    def get_today_trades(self) -> pd.DataFrame:
        """Get all trades placed today."""
        today = date.today().isoformat()
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE date(created_at) = ? ORDER BY created_at",
                conn,
                params=[today],
            )

    # ─────────────────────────────────────────
    # Reconciliation
    # ─────────────────────────────────────────

    def save_reconciliation_log(self, record: dict[str, Any]) -> None:
        """Save daily P&L reconciliation record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reconciliation_log
                (date, system_pnl, broker_pnl, difference,
                 trade_count_system, trade_count_broker, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("date", date.today().isoformat()),
                    record.get("system_pnl", 0),
                    record.get("broker_pnl", 0),
                    record.get("difference", 0),
                    record.get("trade_count_system", 0),
                    record.get("trade_count_broker", 0),
                    record.get("status", "OK"),
                ),
            )
            conn.commit()

    # ─────────────────────────────────────────
    # Iron Condor Trades
    # ─────────────────────────────────────────

    def save_ic_trade(self, record: dict[str, Any]) -> None:
        """Save an Iron Condor trade record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ic_trades
                (position_id, entry_time, regime, spot_at_entry,
                 quantity, lots,
                 sell_ce_strike, sell_ce_premium,
                 buy_ce_strike, buy_ce_premium,
                 sell_pe_strike, sell_pe_premium,
                 buy_pe_strike, buy_pe_premium,
                 net_credit, spread_width, max_profit, max_loss,
                 tp_threshold, sl_threshold,
                 pnl, charges, exit_reason, exit_time,
                 status, expiry_type, trade_type, mode, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    record.get("position_id", ""),
                    record.get("entry_time", ""),
                    record.get("regime", ""),
                    record.get("spot_at_entry", 0),
                    record.get("quantity", 0),
                    record.get("lots", 0),
                    record.get("sell_ce_strike", 0),
                    record.get("sell_ce_premium", 0),
                    record.get("buy_ce_strike", 0),
                    record.get("buy_ce_premium", 0),
                    record.get("sell_pe_strike", 0),
                    record.get("sell_pe_premium", 0),
                    record.get("buy_pe_strike", 0),
                    record.get("buy_pe_premium", 0),
                    record.get("net_credit", 0),
                    record.get("spread_width", 200),
                    record.get("max_profit", 0),
                    record.get("max_loss", 0),
                    record.get("tp_threshold", 0),
                    record.get("sl_threshold", 0),
                    record.get("pnl", 0),
                    record.get("charges", 0),
                    record.get("exit_reason", ""),
                    record.get("exit_time", ""),
                    record.get("status", "open"),
                    record.get("expiry_type", ""),
                    record.get("trade_type", "IRON_CONDOR"),
                    record.get("mode", "paper"),
                ),
            )
            conn.commit()

    # ─────────────────────────────────────────
    # Signals
    # ─────────────────────────────────────────

    def save_signal(self, signal: dict[str, Any]) -> None:
        """Save a strategy signal."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO signals
                (datetime, symbol, strategy, direction, confidence, score,
                 regime, features, ensemble_score, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.get("datetime", datetime.now().isoformat()),
                    signal.get("symbol", ""),
                    signal.get("strategy", ""),
                    signal.get("direction", ""),
                    signal.get("confidence", 0),
                    signal.get("score", 0),
                    signal.get("regime", ""),
                    json.dumps(signal.get("features", {})),
                    signal.get("ensemble_score", 0),
                    signal.get("action_taken", ""),
                ),
            )
            conn.commit()

    # ─────────────────────────────────────────
    # Regime History
    # ─────────────────────────────────────────

    def save_regime(self, regime_data: dict[str, Any]) -> None:
        """Save a regime detection snapshot."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO regime_history
                (datetime, regime, vix_value, nifty_value, adx_value,
                 fii_net_value, active_strategies, size_multiplier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    regime_data.get("datetime", datetime.now().isoformat()),
                    regime_data.get("regime", ""),
                    regime_data.get("vix_value", 0),
                    regime_data.get("nifty_value", 0),
                    regime_data.get("adx_value", 0),
                    regime_data.get("fii_net_value", 0),
                    json.dumps(regime_data.get("active_strategies", [])),
                    regime_data.get("size_multiplier", 1.0),
                ),
            )
            conn.commit()

    # ─────────────────────────────────────────
    # Portfolio Snapshots
    # ─────────────────────────────────────────

    def save_portfolio_snapshot(
        self, snapshot: dict[str, Any], mode: str = "paper"
    ) -> None:
        """Save portfolio state snapshot."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_snapshots
                (datetime, total_value, cash, invested, unrealized_pnl,
                 realized_pnl, day_pnl, positions_count, exposure_pct, drawdown_pct, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.get("datetime", datetime.now().isoformat()),
                    snapshot.get("total_value", 0),
                    snapshot.get("cash", 0),
                    snapshot.get("invested", 0),
                    snapshot.get("unrealized_pnl", 0),
                    snapshot.get("realized_pnl", 0),
                    snapshot.get("day_pnl", 0),
                    snapshot.get("positions_count", 0),
                    snapshot.get("exposure_pct", 0),
                    snapshot.get("drawdown_pct", 0),
                    mode,
                ),
            )
            conn.commit()

    def get_portfolio_history(
        self, days: int = 30, mode: Optional[str] = None
    ) -> pd.DataFrame:
        """Get portfolio value history."""
        query = "SELECT * FROM portfolio_snapshots"
        params: list[Any] = []
        if mode:
            query += " WHERE mode = ?"
            params.append(mode)
        query += " ORDER BY datetime DESC LIMIT ?"
        params.append(days * 10)  # Multiple snapshots per day
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["datetime"] = pd.to_datetime(
                df["datetime"].str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True),
                format="mixed",
            )
            df = df.sort_values("datetime").reset_index(drop=True)
        return df

    # ─────────────────────────────────────────
    # Instrument Logger Tables
    # ─────────────────────────────────────────

    def save_instrument_registry(self, data: dict[str, Any]) -> None:
        """Upsert an instrument into the registry."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO instrument_registry
                (name, instrument_type, exchange, upstox_symbol, lot_size,
                 tick_size, options_expiry, vix_multiplier, adx_threshold, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["name"], data["instrument_type"], data["exchange"],
                    data["upstox_symbol"], data["lot_size"], data["tick_size"],
                    data["options_expiry"], data.get("vix_multiplier", 1.0),
                    data.get("adx_threshold", 22), data.get("active", 1),
                ),
            )
            conn.commit()

    def save_instrument_daily_log(self, data: dict[str, Any]) -> None:
        """Save daily instrument score log (one row per instrument per day)."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO instrument_daily_log
                (date, instrument, open, high, low, close,
                 prev_close, change_pct, regime, adx, adx_slope, bb_width,
                 bull_score, bear_score, score_diff, direction, conviction,
                 pcr, max_call_oi_strike, max_put_oi_strike, max_pain,
                 atm_ce_premium, atm_pe_premium, atm_iv,
                 vix_level, vix_change_pct,
                 rsi_14, macd_signal, ema9, ema21, ema50,
                 would_trade, trade_type, signal_strength, blocking_reason,
                 fii_net, dii_net, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["date"], data["instrument"],
                    data.get("open", 0), data.get("high", 0),
                    data.get("low", 0), data.get("close", 0),
                    data.get("prev_close", 0), data.get("change_pct", 0),
                    data.get("regime", ""), data.get("adx", 0),
                    data.get("adx_slope", 0), data.get("bb_width", 0),
                    data.get("bull_score", 0), data.get("bear_score", 0),
                    data.get("score_diff", 0), data.get("direction", ""),
                    data.get("conviction", 0),
                    data.get("pcr", 0), data.get("max_call_oi_strike", 0),
                    data.get("max_put_oi_strike", 0), data.get("max_pain", 0),
                    data.get("atm_ce_premium", 0), data.get("atm_pe_premium", 0),
                    data.get("atm_iv", 0),
                    data.get("vix_level", 0), data.get("vix_change_pct", 0),
                    data.get("rsi_14", 0), data.get("macd_signal", 0),
                    data.get("ema9", 0), data.get("ema21", 0), data.get("ema50", 0),
                    data.get("would_trade", 0), data.get("trade_type", ""),
                    data.get("signal_strength", ""), data.get("blocking_reason", ""),
                    data.get("fii_net", 0), data.get("dii_net", 0),
                    data.get("scored_at", datetime.now().isoformat()),
                ),
            )
            conn.commit()

    def save_instrument_signal_log(self, data: dict[str, Any]) -> None:
        """Save instrument signal log (when would_trade=True)."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO instrument_signal_log
                (date, instrument, signal_time,
                 oi_score_diff, oi_bull_score, oi_bear_score,
                 pcr, vix_level, vix_change_pct,
                 rsi_14, adx_14, regime, direction,
                 entry_hour, dist_from_open, days_to_expiry,
                 conviction, trade_type,
                 would_buy_symbol, would_buy_strike,
                 would_buy_premium, would_buy_qty,
                 would_buy_sl, would_buy_tp,
                 not_traded_reason, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["date"], data["instrument"], data["signal_time"],
                    data.get("oi_score_diff", 0), data.get("oi_bull_score", 0),
                    data.get("oi_bear_score", 0),
                    data.get("pcr", 0), data.get("vix_level", 0),
                    data.get("vix_change_pct", 0),
                    data.get("rsi_14", 0), data.get("adx_14", 0),
                    data.get("regime", ""), data.get("direction", ""),
                    data.get("entry_hour", 0), data.get("dist_from_open", 0),
                    data.get("days_to_expiry", 0),
                    data.get("conviction", 0), data.get("trade_type", ""),
                    data.get("would_buy_symbol", ""), data.get("would_buy_strike", 0),
                    data.get("would_buy_premium", 0), data.get("would_buy_qty", 0),
                    data.get("would_buy_sl", 0), data.get("would_buy_tp", 0),
                    data.get("not_traded_reason", ""),
                    data.get("scored_at", datetime.now().isoformat()),
                ),
            )
            conn.commit()

    def get_instrument_daily_log(
        self, instrument: Optional[str] = None, days: int = 30
    ) -> pd.DataFrame:
        """Get instrument daily log history."""
        query = "SELECT * FROM instrument_daily_log WHERE 1=1"
        params: list[Any] = []
        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(days)
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def update_instrument_signal_log(
        self, signal_id: int, updates: dict[str, Any]
    ) -> None:
        """Update an instrument signal log row by ID."""
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [signal_id]
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE instrument_signal_log SET {set_clause} WHERE id = ?",
                values,
            )
            conn.commit()

    def get_instrument_signal_log(
        self, instrument: Optional[str] = None, limit: int = 100
    ) -> pd.DataFrame:
        """Get instrument signal log history."""
        query = "SELECT * FROM instrument_signal_log WHERE 1=1"
        params: list[Any] = []
        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)
        query += " ORDER BY scored_at DESC LIMIT ?"
        params.append(limit)
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ─────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────

    def get_data_coverage(self, symbol: str, interval: str = "day") -> dict[str, Any]:
        """Get data coverage info for a symbol — date range and row count."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT MIN(datetime) as min_dt, MAX(datetime) as max_dt, COUNT(*) as cnt
                FROM candles WHERE symbol = ? AND interval = ?
                """,
                (symbol, interval),
            )
            row = cursor.fetchone()

        if row and row["cnt"] > 0:
            return {
                "symbol": symbol,
                "from_date": str(row["min_dt"])[:10],
                "to_date": str(row["max_dt"])[:10],
                "rows": row["cnt"],
            }
        return {"symbol": symbol, "from_date": None, "to_date": None, "rows": 0}

    def get_all_symbols(self, interval: str = "day") -> list[str]:
        """Get all symbols that have candle data."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT symbol FROM candles WHERE interval = ? ORDER BY symbol",
                (interval,),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_candles_bulk(
        self,
        symbols: list[str],
        interval: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load candles for multiple symbols at once."""
        result = {}
        for sym in symbols:
            df = self.get_candles(sym, interval, from_date, to_date, limit=10000)
            if not df.empty:
                result[sym] = df
        return result

    # ═══════════════════════════════════════════════════════
    # ML Candles (5-min)
    # ═══════════════════════════════════════════════════════

    def save_ml_candles(self, symbol: str, instrument_key: str, df: pd.DataFrame) -> int:
        """Bulk upsert 5-min candles into ml_candles_5min. Returns rows inserted."""
        if df.empty:
            return 0

        rows = [
            (
                symbol,
                instrument_key,
                str(row["datetime"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
                int(row.get("oi", 0)),
            )
            for _, row in df.iterrows()
        ]

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ml_candles_5min
                (symbol, instrument_key, datetime, open, high, low, close, volume, oi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            if len(rows) >= 100:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        logger.debug(f"Saved {len(rows)} ML 5-min candles for {symbol}")
        return len(rows)

    def get_ml_candles(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100000,
    ) -> pd.DataFrame:
        """Fetch 5-min candles from ml_candles_5min."""
        query = "SELECT * FROM ml_candles_5min WHERE symbol = ?"
        params: list[Any] = [symbol]

        if from_date:
            query += " AND datetime >= ?"
            params.append(from_date)
        if to_date:
            query += " AND datetime <= ?"
            params.append(to_date)

        query += " ORDER BY datetime ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_ml_candle_coverage(self, symbol: str) -> dict:
        """Return {from_date, to_date, rows} for 5-min candle coverage."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT MIN(datetime), MAX(datetime), COUNT(*) FROM ml_candles_5min WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            return {
                "from_date": row[0],
                "to_date": row[1],
                "rows": row[2],
            }

    # ═══════════════════════════════════════════════════════
    # ML Features Cache
    # ═══════════════════════════════════════════════════════

    def save_ml_features(self, symbol: str, date_str: str, features: dict, version: int = 1) -> None:
        """Save computed features for a date as JSON."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ml_features_cache
                (symbol, date, features_json, feature_version)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, date_str, json.dumps(features), version),
            )
            conn.commit()

    def get_ml_features(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        version: int = 1,
    ) -> pd.DataFrame:
        """Load cached features, return DataFrame with one row per date."""
        query = "SELECT date, features_json FROM ml_features_cache WHERE symbol = ? AND feature_version = ?"
        params: list[Any] = [symbol, version]

        if from_date:
            query += " AND date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND date <= ?"
            params.append(to_date)

        query += " ORDER BY date ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            feat = json.loads(row[1])
            feat["date"] = row[0]
            records.append(feat)

        return pd.DataFrame(records)

    # ═══════════════════════════════════════════════════════
    # ML Models
    # ═══════════════════════════════════════════════════════

    def save_ml_model_record(self, record: dict) -> int:
        """Insert model version record. Returns model_version."""
        with self._get_connection() as conn:
            # Get next version
            cursor = conn.execute(
                "SELECT COALESCE(MAX(model_version), 0) + 1 FROM ml_models WHERE model_name = ?",
                (record["model_name"],),
            )
            version = cursor.fetchone()[0]

            conn.execute(
                """
                INSERT INTO ml_models
                (model_name, model_version, model_type, stage, train_date,
                 train_samples, n_features, train_accuracy, test_accuracy,
                 train_test_gap, deployed, deploy_gate_passed, model_path,
                 scaler_path, feature_list, hyperparams, metrics_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["model_name"],
                    version,
                    record.get("model_type", "xgboost"),
                    record["stage"],
                    record["train_date"],
                    record["train_samples"],
                    record["n_features"],
                    record.get("train_accuracy", 0),
                    record.get("test_accuracy", 0),
                    record.get("train_test_gap", 0),
                    record.get("deployed", 0),
                    record.get("deploy_gate_passed", 0),
                    record["model_path"],
                    record.get("scaler_path"),
                    json.dumps(record.get("feature_list", [])),
                    json.dumps(record.get("hyperparams", {})),
                    json.dumps(record.get("metrics_json", {})),
                    record.get("notes"),
                ),
            )
            conn.commit()

        logger.info(f"ML model record saved: {record['model_name']} v{version}")
        return version

    def get_deployed_model(self, model_name: str) -> Optional[dict]:
        """Get the currently deployed model record (deployed=1)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM ml_models WHERE model_name = ? AND deployed = 1 ORDER BY model_version DESC LIMIT 1",
                (model_name,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def set_model_deployed(self, model_name: str, model_version: int) -> None:
        """Set deployed=1 for specified version, deployed=0 for all others of same model_name."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE ml_models SET deployed = 0 WHERE model_name = ?",
                (model_name,),
            )
            conn.execute(
                "UPDATE ml_models SET deployed = 1 WHERE model_name = ? AND model_version = ?",
                (model_name, model_version),
            )
            conn.commit()
        logger.info(f"ML model deployed: {model_name} v{model_version}")

    def update_ml_model_paths(
        self, model_name: str, model_version: int, model_path: str, scaler_path: str
    ) -> None:
        """Update model_path and scaler_path for a saved model record."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE ml_models SET model_path = ?, scaler_path = ? "
                "WHERE model_name = ? AND model_version = ?",
                (model_path, scaler_path, model_name, model_version),
            )
            conn.commit()

    def get_ml_model_history(self, model_name: str, limit: int = 20) -> pd.DataFrame:
        """Get training history for a model."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM ml_models WHERE model_name = ? ORDER BY model_version DESC LIMIT ?",
                conn,
                params=[model_name, limit],
            )
        return df

    # ═══════════════════════════════════════════════════════
    # ML Predictions
    # ═══════════════════════════════════════════════════════

    def save_ml_prediction(self, prediction: dict) -> None:
        """Log a prediction."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ml_predictions
                (model_name, model_version, prediction_date, prediction_time,
                 predicted_class, prob_ce, prob_pe, prob_flat, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction["model_name"],
                    prediction["model_version"],
                    prediction["prediction_date"],
                    prediction["prediction_time"],
                    prediction["predicted_class"],
                    prediction.get("prob_ce", 0),
                    prediction.get("prob_pe", 0),
                    prediction.get("prob_flat", 0),
                    json.dumps(prediction.get("features", {})),
                ),
            )
            conn.commit()

    def get_ml_predictions(
        self, model_name: str, from_date: Optional[str] = None, limit: int = 100,
    ) -> pd.DataFrame:
        """Get predictions for drift analysis."""
        query = "SELECT * FROM ml_predictions WHERE model_name = ?"
        params: list[Any] = [model_name]

        if from_date:
            query += " AND prediction_date >= ?"
            params.append(from_date)

        query += " ORDER BY prediction_date DESC, prediction_time DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def update_prediction_actual(self, prediction_date: str, actual_class: str) -> None:
        """Fill in actual_class and correct fields for a date's predictions."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE ml_predictions
                SET actual_class = ?,
                    correct = CASE WHEN predicted_class = ? THEN 1 ELSE 0 END
                WHERE prediction_date = ? AND actual_class IS NULL
                """,
                (actual_class, actual_class, prediction_date),
            )
            conn.commit()

    # ═══════════════════════════════════════════════════════
    # ML Trade Labels
    # ═══════════════════════════════════════════════════════

    def save_ml_trade_label(self, label: dict) -> None:
        """Save a labeled trade outcome."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ml_trade_labels
                (trade_id, trade_date, symbol, direction, regime,
                 entry_price, exit_price, pnl, label,
                 score_diff, conviction, vix_at_entry, rsi_at_entry,
                 adx_at_entry, pcr_at_entry, ml_prob_ce, ml_prob_pe,
                 trigger_count, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label["trade_id"],
                    label["trade_date"],
                    label["symbol"],
                    label["direction"],
                    label.get("regime"),
                    label["entry_price"],
                    label["exit_price"],
                    label["pnl"],
                    label["label"],
                    label.get("score_diff"),
                    label.get("conviction"),
                    label.get("vix_at_entry"),
                    label.get("rsi_at_entry"),
                    label.get("adx_at_entry"),
                    label.get("pcr_at_entry"),
                    label.get("ml_prob_ce"),
                    label.get("ml_prob_pe"),
                    label.get("trigger_count"),
                    json.dumps(label.get("features", {})),
                ),
            )
            conn.commit()

    def get_ml_trade_labels(self, min_date: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """Get labeled trades for quality model training."""
        query = "SELECT * FROM ml_trade_labels"
        params: list[Any] = []

        if min_date:
            query += " WHERE trade_date >= ?"
            params.append(min_date)

        query += " ORDER BY trade_date DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_ml_trade_label_count(self) -> int:
        """Count total labeled trades."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM ml_trade_labels")
            return cursor.fetchone()[0]

    # ── Counterfactual Trades ──

    def save_counterfactual_trade(self, record: dict) -> None:
        """Save a counterfactual (blocked) trade record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO counterfactual_trades
                (date, symbol, direction, block_reason, block_time,
                 regime, score_diff, bull_score, bear_score,
                 spot_at_block, spot_at_eod,
                 hypothetical_pnl, hypothetical_pct, would_have_won,
                 metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["date"],
                    record["symbol"],
                    record["direction"],
                    record["block_reason"],
                    record.get("block_time"),
                    record.get("regime"),
                    record.get("score_diff", 0),
                    record.get("bull_score", 0),
                    record.get("bear_score", 0),
                    record.get("spot_at_block", 0),
                    record.get("spot_at_eod", 0),
                    record.get("hypothetical_pnl", 0),
                    record.get("hypothetical_pct", 0),
                    record.get("would_have_won", 0),
                    json.dumps(record.get("metadata", {})),
                ),
            )
            conn.commit()

    def get_counterfactual_trades(
        self, min_date: Optional[str] = None, block_reason: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        """Get counterfactual trades with optional filters."""
        query = "SELECT * FROM counterfactual_trades"
        conditions: list[str] = []
        params: list[Any] = []

        if min_date:
            conditions.append("date >= ?")
            params.append(min_date)
        if block_reason:
            conditions.append("block_reason = ?")
            params.append(block_reason)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_counterfactual_count(self) -> int:
        """Count total counterfactual trades."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM counterfactual_trades")
            return cursor.fetchone()[0]

    # ═══════════════════════════════════════════════════════
    # Live Slippage Tracking
    # ═══════════════════════════════════════════════════════

    def save_slippage_log(self, record: dict[str, Any]) -> None:
        """Save a live slippage record for audit tracking."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO live_slippage_log
                (trade_id, symbol, signal_price, fill_price, slippage_pct,
                 slippage_amount, quantity, direction, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("trade_id", ""),
                    record.get("symbol", ""),
                    record.get("signal_price", 0),
                    record.get("fill_price", 0),
                    record.get("slippage_pct", 0),
                    record.get("slippage_amount", 0),
                    record.get("quantity", 0),
                    record.get("direction", ""),
                    record.get("mode", "live"),
                ),
            )
            conn.commit()

    def get_slippage_summary(self, mode: str = "live", limit: int = 100) -> pd.DataFrame:
        """Get slippage records for analysis."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM live_slippage_log WHERE mode = ? ORDER BY created_at DESC LIMIT ?",
                conn,
                params=[mode, limit],
            )

    def get_stats(self) -> dict[str, int]:
        """Get row counts for all tables."""
        stats = {}
        with self._get_connection() as conn:
            for table in self.SCHEMA:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        return stats

    def cleanup_old_data(self, days_to_keep: int = 365) -> dict[str, int]:
        """Remove data older than specified days."""
        cutoff = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()
        deleted = {}

        with self._get_connection() as conn:
            for table in ["candles", "option_chain", "signals", "portfolio_snapshots"]:
                dt_col = "datetime" if table != "option_chain" else "date"
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE {dt_col} < ?", (cutoff,)
                )
                deleted[table] = cursor.rowcount

            conn.commit()

        logger.info(f"Cleanup complete: {deleted}")
        return deleted

    # ─────────────────────────────────────────
    # Factor Edge History
    # ─────────────────────────────────────────

    def save_factor_edge(self, date_str: str, factor_name: str, aligned_wr: float,
                         against_wr: float, net_edge: float, trade_count: int,
                         window_days: int = 90) -> None:
        """Save one factor's edge snapshot to history."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO factor_edge_history
                   (date, factor_name, aligned_wr, against_wr, net_edge, trade_count, window_days)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (date_str, factor_name, aligned_wr, against_wr, net_edge, trade_count, window_days),
            )
            conn.commit()

    def get_factor_edge_history(self, factor_name: str | None = None,
                                limit: int = 12) -> pd.DataFrame:
        """Get factor edge history. If factor_name is None, returns all factors."""
        query = "SELECT date, factor_name, aligned_wr, against_wr, net_edge, trade_count FROM factor_edge_history"
        params: list[Any] = []
        if factor_name:
            query += " WHERE factor_name = ?"
            params.append(factor_name)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_factor_edge_previous(self, date_str: str) -> dict[str, dict]:
        """Get the most recent factor edge snapshot before the given date."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT factor_name, aligned_wr, against_wr, net_edge, trade_count
                   FROM factor_edge_history WHERE date < ? ORDER BY date DESC LIMIT 10""",
                (date_str,),
            ).fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            fname = row[0]
            if fname not in result:  # Only latest per factor
                result[fname] = {
                    "aligned_wr": row[1], "against_wr": row[2],
                    "net_edge": row[3], "trade_count": row[4],
                }
        return result

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
