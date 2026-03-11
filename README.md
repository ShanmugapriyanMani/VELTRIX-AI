# VELTRIX V9 — Algorithmic Options Trading Bot

Algorithmic trading system for NIFTY 50 weekly options (NSE India) using Upstox API V3.

## Architecture

```
LAYER 1: Regime Detection (VIX + ADX + VIX trend)
    → 4 regimes: TRENDING, RANGEBOUND, VOLATILE, ELEVATED
    ↓
LAYER 2: 9-Factor Scoring (EMA, RSI, MACD, mean reversion, BB,
         VIX, OI/PCR, volume, global macro) with regime-adaptive weights
    → Asymmetric PE/CE conviction thresholds
    ↓
LAYER 3: Trade Type Selection
    → NAKED_BUY (directional) or CREDIT_SPREAD (VOLATILE/ELEVATED only)
    ↓
LAYER 4: Risk Management (3-layer adaptive SL/TP, 3-tier trailing stops,
         conviction-scaled lots, weekly/monthly circuit breakers)
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

## Project Structure

```
├── config/
│   ├── config.yaml          # Upstox config, universe, risk params
│   ├── risk.yaml            # Risk management & circuit breaker params
│   └── strategies.yaml      # Strategy parameters
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Upstox API V3 (historical, intraday, VIX)
│   │   ├── store.py         # SQLite DB (candles, trades, signals)
│   │   ├── features.py      # Technical indicators + alternative data
│   │   └── options_instruments.py  # F&O instrument key resolver
│   ├── regime/              # Market regime detection
│   ├── strategies/          # 5 strategies (3 active, 2 disabled) + ensemble
│   ├── risk/                # Position sizing, portfolio, circuit breakers
│   ├── execution/           # Upstox broker, paper trader, order management
│   ├── utils/               # Market calendar (expiry day checks)
│   ├── backtest/            # Backtesting engine + metrics
│   ├── dashboard/           # Streamlit UI + Telegram alerts
│   └── main.py              # Orchestrator (~4500 lines)
├── docs/
│   └── V9_WORKFLOW.md       # Complete V9 system documentation
├── scripts/
│   └── auth_upstox.py       # OAuth2 authentication
├── models/                  # Saved ML models (auto-generated)
├── data/                    # SQLite database (auto-generated)
├── tests/                   # 78 tests (pytest)
├── requirements.txt
└── docker-compose.yaml
```

## Key Details

- **Capital**: ₹1,50,000 | **Deploy Cap**: ₹75,000 | **Risk/Trade**: ₹15,000
- **Options**: NIFTY weekly options (CE/PE), intraday only, ATM strike, lot size 65
- **Trade Types**: NAKED_BUY (all regimes) + CREDIT_SPREAD (VOLATILE/ELEVATED only)
- **Regimes**: TRENDING (ADX > 25), RANGEBOUND, VOLATILE (VIX ≥ 28), ELEVATED (VIX 20-28 rising)
- **Conviction**: Asymmetric PE/CE thresholds — PE lower (72% WR, 2.83x more P&L per trade)
- **SL/TP**: 3-layer VIX-adaptive × regime multiplier × premium clamp
- **Trailing Stop**: 3-tier from +5% (3%/5%/7% below peak)
- **ML**: Permanently disabled (V9.2 rebuild: 49.1% acc, 27.5% gap) — replaced by F10 Global Macro factor
- **Guards**: Daily -₹20K halt, weekly -₹20K/₹35K soft/hard, monthly -8% boost, per-regime direction cooldown
- **Expiry**: NIFTY weekly = Tuesday, with theta gate restrictions

## Backtest Results (V9.2 — 5-Year, 2021-2026)

| Metric | V8 | V9.2 |
|--------|-----|------|
| Period | 2023-2026 (693 days) | 2021-2026 (1186 days) |
| Capital | ₹50,000 | ₹1,50,000 |
| Total Return | 1,445% | 1,787% |
| CAGR | — | 86.67% |
| Trades | 301 | 489 |
| Win Rate | 61.1% | 71.2% |
| Profit Factor | 2.64 | 4.72 |
| Sharpe Ratio | 2.778 | 2.633 |
| Max Drawdown | 14.19% | 16.76% |
| Profitable Months | 31/34 (91%) | 56/59 (95%) |

**By Regime**: TRENDING 69.6% WR (401 trades, +₹20.8L), RANGEBOUND 78.6% WR (56 trades), VOLATILE 78.1% WR (32 trades)

**By Direction**: PE Buy 74.6% WR (+₹15.6L) outperforms CE Buy 68.9% WR (+₹11.1L)

## Testing

```bash
pytest tests/ -v    # 78 tests
```

## Documentation

See [docs/V9_WORKFLOW.md](docs/V9_WORKFLOW.md) for complete system workflow, SL/TP logic, exit management, and safety systems.
