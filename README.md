# VELTRIX — AI Trading Bot

Advanced algorithmic trading system for Indian stocks (NSE) using Upstox API V3.

## Architecture

```
LAYER 1: Regime Detection (VIX + NIFTY trend + ADX)
    ↓
LAYER 2: 8-Factor Scoring (EMA trend, mean reversion, RSI, MACD,
         Bollinger, MFI, ADX, volume) with regime-adaptive weights
    ↓
LAYER 3: ML Ensemble (LightGBM + XGBoost + CatBoost, auto-governance)
    ↓
LAYER 4: Options Buyer (directional NIFTY options, ATM strike selection)
    ↓
LAYER 5: Risk Management (VIX-adaptive SL/TP, trailing stops, circuit breakers)
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
Auto-fetches all data sources on startup (incremental), trains ML model, then trades during market hours (9:15 AM - 3:30 PM IST). No separate fetch command needed.

**Backtest**:
```bash
python src/main.py --mode backtest
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
| `backtest` | Options backtest with walk-forward ML and regime detection |

## Project Structure

```
├── config/
│   ├── config.yaml          # Upstox config, universe, risk params
│   └── strategies.yaml      # Strategy parameters
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Upstox API V3 (historical, intraday, VIX)
│   │   ├── store.py         # SQLite DB (candles, trades, signals)
│   │   ├── features.py      # Technical indicators + alternative data
│   │   └── options_instruments.py  # F&O instrument key resolver
│   ├── regime/              # Market regime detection
│   ├── strategies/          # 5 strategies + ensemble
│   ├── risk/                # Position sizing, portfolio, circuit breakers
│   ├── execution/           # Upstox broker, paper trader, order management
│   ├── utils/               # Market calendar (expiry day checks)
│   ├── backtest/            # Backtesting engine + metrics
│   ├── dashboard/           # Streamlit UI + Telegram alerts
│   └── main.py              # Orchestrator
├── scripts/
│   └── auth_upstox.py       # OAuth2 authentication
├── models/                  # Saved ML models (auto-generated)
├── data/                    # SQLite database (auto-generated)
├── requirements.txt
└── docker-compose.yaml
```

## Key Details

- **Capital**: ₹25,000 | **Universe**: NIFTY 50 stocks
- **Options**: NIFTY weekly options, intraday only, ATM strike selection
- **Options Risk**: VIX-adaptive SL/TP (20-36% SL, 22-86% TP based on regime), trailing stops, max 1 lot, force exit 15:10 (13:30 on expiry days)
- **ML Model**: 3-model ensemble (LightGBM + XGBoost + CatBoost), 120-day walk-forward training, auto-governance (disables if accuracy < 50%)
- **Regime Profiles**: TRENDING (conviction 1.75), RANGEBOUND (2.0), VOLATILE (2.5) — each with tuned SL/TP/sizing
- **Expiry Day Trading**: Allowed before 1 PM with wider SL (+5%) and theta decay penalty
- **Expiry**: NIFTY weekly expiry = Tuesday
- **DB**: SQLite with WAL mode, DB-first caching (checks DB before API calls)

## Backtest Results (V8 — Feb 2026)

| Metric | Value |
|--------|-------|
| Total Return | 1,445% |
| Trades | 301 |
| Win Rate | 61.1% |
| Profit Factor | 2.64 |
| Sharpe Ratio | 2.778 |
| Max Drawdown | 14.19% |

**By Regime**: TRENDING 57.2% WR (236 trades), RANGEBOUND 76.6% WR (64 trades)

## Testing

```bash
pytest tests/ -v
```
