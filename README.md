# VELTRIX V9.3 — Algorithmic Options Trading Bot

Algorithmic trading system for NIFTY 50 weekly options (NSE India) using Upstox API V3.

## Architecture

```
LAYER 1: Regime Detection (VIX + ADX + VIX trend)
    → 4 regimes: TRENDING, RANGEBOUND, VOLATILE, ELEVATED
    ↓
LAYER 2: 10-Factor Scoring (EMA, RSI, MACD, mean reversion, BB,
         VIX, ML direction, OI/PCR, volume, global macro)
    → Asymmetric PE/CE conviction thresholds
    → ML Stage 1: XGBoost CE/PE/FLAT probabilities (Factor 7)
    ↓
LAYER 3: Fuzzy Confirmation (4 gradient triggers, sum ≥ 2.0)
    → T1 price momentum, T2 RSI, T3 breakout, T4 PCR
    → Rolling range updates every 30 min + reset after trade close
    ↓
LAYER 4: Trade Type Selection
    → NAKED_BUY (directional), CREDIT_SPREAD (VOLATILE/ELEVATED),
      or IRON_CONDOR (RANGEBOUND, 4-leg)
    ↓
LAYER 5: Risk Management (3-layer adaptive SL/TP, 3-tier trailing stops,
         conviction-scaled lots, simplified 2-rule circuit breaker)
    → Two-speed poll: 5s with positions, 30s without
    → WebSocket LTP feed (REST fallback) for real-time pricing
    → ML Stage 2: quality gate blocks low win-probability trades
    → 10 CRITICAL + 14 HIGH stability fixes (NaN safety, crash recovery)
    ↓
LAYER 6: Live Infrastructure
    → Token lifecycle: JWT expiry decode + TokenWatcher daemon (5-min checks)
    → Fill confirmation: poll order status, cancel on timeout, slippage tracking
    → Margin check: block trade if insufficient, warn if < 1.4× required
    → P&L reconciliation: system vs broker comparison at EOD
    ↓
EXECUTION: Upstox OrderApiV3 + GTT V3
```

## Data Source

**Upstox API only** — no NSE website scraping.

| Data | Source | Status |
|------|--------|--------|
| Equity OHLCV | Upstox Historical V3 | Available |
| Index (NIFTY/BANKNIFTY) | Upstox Historical V3 | Available |
| India VIX | Upstox Live Quote + Historical | Available |
| F&O Premiums | Upstox Expired Instruments API | Available |
| FII/DII Flows | — | Not available (strategies degrade gracefully) |
| Options OI/PCR | — | Not available (strategies degrade gracefully) |
| Delivery Volume | — | Not available (strategies degrade gracefully) |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
- Add Upstox API credentials to `.env`:
  ```
  UPSTOX_LIVE_API_KEY=your_api_key
  UPSTOX_LIVE_API_SECRET=your_api_secret
  UPSTOX_LIVE_REDIRECT_URI=your_redirect_uri
  ```
- Add Telegram bot token (optional):
  ```
  TELEGRAM_BOT_TOKEN=your_token
  TELEGRAM_CHAT_ID=your_chat_id
  ```

### 3. Authenticate with Upstox
```bash
python scripts/auth_upstox.py
```
This opens a browser for OAuth2 login. Token is saved locally and expires daily.

### 4. Run

**Paper Trading** (recommended to start):
```bash
python src/main.py --mode paper
```
Auto-fetches all data sources on startup (incremental), then trades during market hours (10:00 AM - 3:10 PM IST).

**Full Historical Backtest** (5-year):
```bash
python src/main.py --mode backtest --no-wait
```

**Live Trading**:
```bash
python src/main.py --mode live
```

## CLI Modes

| Mode | Description |
|------|-------------|
| `fetch` | Standalone bulk data download (also runs auto on paper/live startup) |
| `paper` | Paper trading with simulated execution |
| `live` | Live trading with real Upstox orders |
| `backtest` | Full 5-year historical backtest |
| `report` | Paper trading performance report (from DB) |
| `backup` | Daily SQLite + config backup to Google Drive (rclone) |
| `ml_backfill` | Download 5-min candles from Upstox for ML training |
| `ml_train` | Run full ML training pipeline (Stage 1 direction + Stage 2 quality) |
| `ml_status` | Show model status, candle coverage, drift detection |
| `ml_report` | Training history, prediction accuracy, trade labels |

## Project Structure

```
├── config/
│   ├── config.yaml          # Upstox config, universe, risk params
│   ├── risk.yaml            # Risk management & circuit breaker params
│   └── strategies.yaml      # Strategy parameters
├── src/
│   ├── auth/
│   │   └── token_manager.py # JWT expiry decode + TokenWatcher daemon
│   ├── data/
│   │   ├── fetcher.py       # Upstox API V3 (historical, intraday, VIX, WebSocket)
│   │   ├── store.py         # SQLite DB (candles, trades, signals, reconciliation)
│   │   ├── features.py      # Technical indicators + alternative data
│   │   └── options_instruments.py  # F&O instrument key resolver
│   ├── ml/
│   │   ├── backfill_candles.py   # 5-min candle download (4 years)
│   │   ├── candle_features.py    # 40-feature engineering
│   │   └── train_models.py       # XGBoost training + drift detection
│   ├── regime/              # Market regime detection
│   ├── strategies/          # Options buyer + Iron Condor + ensemble
│   ├── risk/                # Position sizing, portfolio, circuit breakers
│   ├── execution/           # Upstox broker, paper trader, order management, fill confirmation
│   ├── utils/               # Market calendar (expiry types, holidays)
│   ├── backtest/            # Backtesting engine + metrics
│   ├── dashboard/           # Streamlit UI + Telegram alerts
│   └── main.py              # Orchestrator
├── docs/
│   └── V9_WORKFLOW.md       # Complete V9 system documentation
├── scripts/
│   ├── auth_upstox.py       # OAuth2 authentication
│   └── backup_gdrive.sh     # Daily backup to Google Drive (rclone)
├── models/                  # Old LightGBM models (disabled)
├── data/                    # SQLite database + ML models (auto-generated)
├── tests/                   # 231 tests (pytest)
├── requirements.txt
└── docker-compose.yaml
```

## Key Details

- **Capital**: ₹1,50,000 | **Deploy Cap**: ₹75,000 | **Risk/Trade**: ₹15,000
- **Options**: NIFTY weekly options (CE/PE), intraday only, ATM strike, lot size 65
- **Trade Types**: NAKED_BUY (all regimes) + CREDIT_SPREAD (VOLATILE/ELEVATED) + IRON_CONDOR (RANGEBOUND)
- **Regimes**: TRENDING (ADX > 25), RANGEBOUND, VOLATILE (VIX ≥ 28), ELEVATED (VIX 20-28 rising)
- **Conviction**: Asymmetric PE/CE thresholds — PE lower (72% WR, 2.83x more P&L per trade)
- **SL/TP**: 3-layer VIX-adaptive × regime multiplier × premium clamp
- **Trailing Stop**: 3-tier from +5% (3%/5%/7% below peak)
- **ML**: Two-stage XGBoost — Stage 1 direction (CE/PE/FLAT, 40 features), Stage 2 quality gate (auto at 30 trades)
- **Confirmation**: Fuzzy triggers T1-T4 (0.0-1.0 each), entry when sum ≥ 2.0, rolling range updates every 30 min
- **Guards**: 2-rule CB (2 consecutive SL halt + daily -₹20K halt), per-regime direction cooldown
- **Monitoring**: Two-speed poll (5s with positions, 30s without), data readiness gate at startup
- **WebSocket**: Real-time LTP via `MarketDataStreamerV3`, auto-reconnect, REST fallback
- **Token Lifecycle**: JWT expiry decode, TokenWatcher daemon (5-min checks), Telegram alerts before expiry
- **Fill Confirmation**: Live mode polls order status (2s interval, 30s timeout), cancels unfilled orders, tracks slippage
- **Margin Check**: Pre-trade validation — blocks if insufficient, warns if < 1.4× required margin
- **Reconciliation**: EOD system vs broker P&L comparison — OK (≤₹100), WARNING (≤₹500), CRITICAL (>₹500)
- **Backup**: Daily SQLite + config backup to Google Drive via rclone (cron or `--mode backup`)
- **Expiry**: Supports old (Thu=NIFTY) + new (Tue=NIFTY, Sep 2025+) schedules, holiday shifts
- **Stability**: 10 CRITICAL + 14 HIGH + 4 MEDIUM fixes (NaN safety, crash recovery, graceful degradation)
- **Sizing**: Equity curve sizing (4-tier DD schedule) + CB loss-based reduction (0/1/2+ SL → 1.0/0.75/0.50)

## Backtest Results (V9.3b — 5-Year, 2021-2026)

| Metric | V8 | V9.2 | V9.3b |
|--------|-----|------|-------|
| Period | 2023-2026 (693 days) | 2021-2026 (1186 days) | 2021-2026 (1186 days) |
| Capital | ₹50,000 | ₹1,50,000 | ₹1,50,000 |
| Total Return | 1,445% | 1,787% | 1,998% |
| CAGR | — | 86.67% | 90.50% |
| Trades | 301 | 489 | 495 |
| Win Rate | 61.1% | 71.2% | 74.3% |
| Profit Factor | 2.64 | 4.72 | 5.15 |
| Sharpe Ratio | 2.778 | 2.633 | 2.759 |
| Max Drawdown | 14.19% | 16.76% | 14.61% |
| Profitable Months | 31/34 (91%) | 56/59 (95%) | 56/59 (95%) |

**V9.3b changes**: Equity curve sizing + CB loss-based size reduction → DD 30.89% → 14.61%, Sharpe 2.344 → 2.759. ML Stage 1 re-enabled (asymmetric PE=1.5, CE=0.3). 4 MEDIUM stability fixes. All parameters verified ROBUST. NAKED_BUY 456 trades PF 4.56, CREDIT_SPREAD 39 trades PF 16.49.

## Testing

```bash
pytest tests/ -v    # 231 tests
```

## Status: Paper Trading (Month 3 of 6)

V9.3b complete: equity curve sizing (DD 14.61%), CB loss-based sizing, ML re-enabled, 28 stability fixes, 231 tests. All parameters verified ROBUST.

| Month | Focus | Status |
|-------|-------|--------|
| 1 | NIFTY paper trading, infrastructure (WebSocket, backup, token, fill, recon) | Done |
| 2 | Stability fixes, Iron Condor, simplified CB, equity curve sizing, ML re-enable | Done |
| 3 (current) | Quality model active, 30-trade review, IC paper results | In Progress |
| 4 | Go-live audit → 1 lot NIFTY | Planned |
| 5 | Scale + VPS + BANKNIFTY support | Planned |
| 6 | BANKNIFTY live, IC at scale, V10 planning | Planned |

## Documentation

See [docs/V9_WORKFLOW.md](docs/V9_WORKFLOW.md) for complete V9.3 system workflow, SL/TP logic, exit management, safety systems, stability fixes, and roadmap.
