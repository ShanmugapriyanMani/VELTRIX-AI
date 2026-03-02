# Veltrix V8 PLUS — Current System Workflow

Stage: PLUS | Capital: ₹1,50,000 | Deploy Cap: ₹75,000

CLI:
  python src/main.py --mode paper         Paper trading (simulated)
  python src/main.py --mode live          Live trading (Upstox)
  python src/main.py --mode backtest      Paper trading report (DB)
  python src/main.py --mode backtest --full   Full historical backtest
  python src/main.py --mode fetch         Data fetch only


## Startup & Market Guards

```
SYSTEM STARTS
  1. Load .env → credentials + TRADING_STAGE=PLUS
  2. Load .env.plus → capital, risk, spread settings (override=True)
  3. Print config:
       Capital: ₹1,50,000 | Deploy Cap: ₹75,000
       Risk/Trade: ₹15,000 | Daily Loss Halt: ₹20,000
       Stage: PLUS | Max Trades: 4
  4. Initialize SQLite (9 tables), Paper Trader / Upstox Broker

MARKET SCHEDULE CHECK (skipped with --no-wait)
  Weekend (Sat/Sun)  → "Market closed (Saturday). Next: Monday XX" → EXIT
  NSE Holiday        → "Market closed (Holi). Next: Wednesday XX" → EXIT
  Past 15:15         → "Market closed for today. Next: ..." → EXIT
  Before 10:00       → "wait" → proceed to setup, wait for trading window
  10:00–15:15        → "trade" → proceed immediately

  Holidays source: Upstox API → local cache → hardcoded 2026 fallback

TOKEN CHECK
  Load Upstox access token
  Live mode + no token → ABORT ("run: python scripts/auth_upstox.py")
  Paper mode + no token → WARN, continue with cached data
```


## Phase 0: Auto-Fetch (incremental — fast when up to date)

```
[1/8] NIFTY 50 equity candles (3 years via Upstox API)
[2/8] Index data — NIFTY 50, India VIX
[3/8] F&O option premium data (current contracts)
[4/8] (skip)
[5/8] Local CSV data (already loaded)
[6/8] FII/DII data (nsepython API + CSV bulk)
[7/8] External markets via yfinance — S&P 500, NASDAQ, Crude, Gold, USD/INR
[8/8] VIX extended history

All stored in SQLite (data/trading_bot.db)
Only fetches new data since last run
```


## Phase 0b: ML Model Training

```
Schedule: Monday only (weekly retrain)
  Monday    → retrain LightGBM (90-day window, 19 features)
  Tue–Fri   → use cached model from models/lgbm_latest.pkl

Output: P(up) = 0.52, P(down) = 0.48

If before trading window:
  "Setup complete. Waiting for trading window..."
  "Market opens 09:15 | Trading starts 10:00"
```


## Phase 1: Pre-Market Data (waits until 08:30)

```
Load FII/DII history (268 days)
Fetch India VIX live quote
Fetch NIFTY option chain:
  PCR, Max Call OI, Max Put OI, Max Pain

Telegram alert: "Bot started | Capital ₹1,50,000 | ML: P(up)=0.52"
```


## Phase 2: Broker Connection (waits until 09:00)

```
PAPER MODE:
  Paper Trader connected (simulated)

LIVE MODE:
  Connect to Upstox (OrderApiV3 via HFT endpoint)
  Fetch user profile → "Account: Sandy (ID) | Active: True"
  Fetch live funds from Upstox API:
    GET /v2/user/get-funds-and-margin
    Returns: available_margin, used_margin, total_balance

  Capital verification:
    Capital stays from .env (₹1,50,000) — user's intended allocation
    Deploy Cap = min(Upstox available_margin, config deploy cap)
    Updates options_buyer.max_deployable with real deploy cap

    If available < ₹50,000 (MIN_WALLET_BALANCE) → ABORT
    If available < ₹75,000 (DEPLOY_CAP) → WARN, continue limited

  Telegram: "Live Capital | Capital ₹1,50,000 | Deployable ₹75,000"
```


## Phase 3: Trading Loop (10:00 — 15:10)

```
Daily counters reset:
  trades_today = 0
  naked_trades_today = 0
  spread_trades_today = 0
  daily_pnl = 0
  direction_locked = None
  direction_rescores_today = 0

Fetch 50+ days NIFTY daily candles
Compute technicals: EMA 9/21/50, RSI, MACD, BB, ADX
Refresh options instrument master (today's strikes)
```


### Each Scan Cycle

```
SCAN INTERVAL:
  No position:              30 seconds
  Position open (live):     15 seconds
  Network down:             60 seconds

  ┌─── STEP 1: Circuit Breaker ───┐
  │ Check daily loss, drawdown,   │
  │ consecutive losses            │
  │ NORMAL → proceed              │
  │ PAUSED → wait 60 min          │
  │ HALTED → stop all trading     │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 2: Refresh Data ──────┐
  │ NIFTY LTP, VIX, intraday     │
  │ candles                        │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 3: Regime Detection ──┐
  │ VOLATILE: VIX ≥ 30, or score  │
  │ TRENDING: ADX > 25            │
  │ RANGEBOUND: everything else   │
  │                                │
  │ After 1 PM: can only upgrade  │
  │ to VOLATILE (never downgrade) │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 4: 9-Factor Scoring ──┐
  │ F1 EMA Trend:     ±2.5       │
  │ F2 RSI+MACD:      ±2.0       │
  │ F3 Price Action:   ±1.5       │
  │ F4 Mean Revert:    ±2.5       │
  │ F5 BB Position:    ±0.75      │
  │ F6 VIX Sentiment:  ±0.8       │
  │ F7 ML Predict:     ±1.5       │
  │ F8 OI/PCR:         ±2.0       │
  │ F9 Volume:         ±1.0       │
  │                                │
  │ bull_score vs bear_score       │
  │ score_diff = |bull - bear|    │
  │ Direction: CE or PE            │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 5: Direction Lock ────┐
  │ First trade → lock direction  │
  │ Unlock on: TP, trail exit     │
  │                                │
  │ Stuck direction recovery:     │
  │  30-min confirmation timeout  │
  │  → re-score direction         │
  │  2 SLs same direction today   │
  │  → re-score direction         │
  │  Max 3 re-scores per day      │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 6: Conviction Gate ───┐
  │ PLUS thresholds:              │
  │  TRENDING:   1.75             │
  │  RANGEBOUND: 2.0              │
  │  VOLATILE:   2.5              │
  │  + Monday penalty: +0.3       │
  │  + Afternoon (>1PM): +0.5     │
  │                                │
  │ score_diff ≥ threshold → PASS │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7: Trade Type (PLUS) ─┐
  │ VOLATILE + conv ≥ 2.0         │
  │   → CREDIT_SPREAD             │
  │ VOLATILE + conv < 2.0         │
  │   → SKIP                      │
  │ Any regime + conv ≥ 3.0       │
  │   → NAKED_BUY                 │
  │ Any regime + conv < 3.0       │
  │   → DEBIT_SPREAD              │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 8: Confirmation ──────┐
  │ 4 intraday triggers:          │
  │  1. Price vs Day Open         │
  │  2. RSI 5-min (>55 or <45)   │
  │  3. Morning Range Breakout    │
  │  4. PCR (<0.7 or >1.2)       │
  │                                │
  │ 9:30–11:00  → need 2/4        │
  │ 11:00–13:00 → need 1/4        │
  │ 13:00–14:30 → need 2/4        │
  │                                │
  │ Failed 30 min → re-score dir  │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 9: Strike Selection ──┐
  │ ATM strike (nearest to spot)  │
  │ Current week expiry           │
  │ (next week's on expiry day)   │
  │ Premium ≥ ₹80 (MIN_PREMIUM)  │
  │                                │
  │ Spread: ATM + OTM legs        │
  │ Width: 200 points (config)    │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 10: Position Sizing ──┐
  │ NAKED BUY:                    │
  │  deploy = 75000/(prem × 65)  │
  │  risk = 15000/(prem×SL%×65)  │
  │  lots = min(deploy, risk)     │
  │  lots = max(1, lots)          │
  │                                │
  │ DEBIT SPREAD:                 │
  │  net_debit = buy_prem - sell  │
  │  lots = deploy/(net × 65)    │
  │                                │
  │ CREDIT SPREAD:                │
  │  net_credit = sell - buy      │
  │  lots based on margin req     │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 11: Filters ──────────┐
  │ Circuit breaker: NORMAL?      │
  │ Trades today < 4?             │
  │ Naked < 2? Spread < 2?       │
  │ Direction not blocked?        │
  │ Whipsaw: ADX ok?             │
  │ VIX < 35?                     │
  │ Time < 14:30 (NO_NEW_TRADE)?  │
  │ ALL PASS → EXECUTE            │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 12: EXECUTE ──────────┐
  │ Place order (MARKET)          │
  │ Set SL/TP (VIX-adaptive)     │
  │ Lock direction                │
  │ Telegram alert                │
  └────────────────────────────────┘
```


### Trade Types (PLUS Stage)

```
NAKED_BUY (conviction ≥ 3.0):
  Single leg — BUY NIFTY CE or PE
  Max 2 naked trades per day
  Full deploy cap applies (₹75,000)
  Full risk cap applies (₹15,000)

DEBIT_SPREAD (conviction < 3.0):
  Two legs — BUY ATM + SELL OTM (same type)
  Bull call spread (CE) or bear put spread (PE)
  Width: 200 points (SPREAD_WIDTH config)
  Max risk = net debit paid
  SL: 50% of net debit | TP: 70% of net debit

CREDIT_SPREAD (VOLATILE regime only):
  Two legs — SELL near-OTM + BUY far-OTM (opposite type)
  Collect premium upfront, limited risk from protection leg
  SL: 2.0x net credit received | TP: 80% of max profit

Limits per day:
  Max 4 total trades
  Max 2 naked trades
  Max 2 spread trades
```


### Exit Management

```
VIX-Adaptive SL/TP:
  VIX < 13  → SL 25%, TP 40%
  VIX 13-18 → SL 30%, TP 45%
  VIX 18-22 → SL 30%, TP 55%
  VIX 22-28 → SL 25%, TP 60%
  VIX 28-35 → SL 20%, TP 45%
  VIX > 35  → no trades

Regime Adjustments:
  TRENDING:   TP ×1.3 (let winners run)
  RANGEBOUND: SL ×0.85, TP ×0.70 (quick in/out)
  VOLATILE:   SL ×1.20, TP ×1.50 (wide room)

Trailing Stop:
  Activates at +8% gain
  Locks in profit, follows price up

Time Adjustments:
  2:00 PM  → TP not hit → reduce TP by 20%
  2:45 PM  → at loss → tighten SL to 15%

Expiry Day (Tuesday, or shifted Monday if Tuesday is holiday):
  Force exit by 1:30 PM (EXPIRY_EXIT_BY)
  Wider SL (+5%), lower TP (×0.65)
  Max 1 trade

Force Exit: 3:10 PM (TRADE_END)
```


### Direction Lock & Recovery

```
DIRECTION LOCK:
  After first trade → direction locked (CE or PE)
  Same direction trades only until unlock

UNLOCK TRIGGERS:
  TP hit or trail exit → direction unlocked, re-entry allowed
  SL hit + ACTIVE_TRADING → direction unlocked

STUCK DIRECTION RECOVERY:
  1. Confirmation Timeout (30 minutes)
     If confirmations keep failing for 30 min with no trade placed:
     → Clear direction, re-score from scratch
     → Max 3 re-scores per day

  2. SL Direction Unlock (2 SLs same direction)
     If 2 stop losses hit in the same direction today:
     → Clear direction, re-score from scratch
     → Max 3 re-scores per day (shared counter)

  Without these: system could sit idle all day on wrong direction
```


### Monitoring Loop (while position open)

```
Check interval: 30s (no position) / 15s (position, live only)

  Price hits SL     → EXIT (market order)
  Price hits TP     → EXIT
  Trail triggered   → EXIT
  3:10 PM           → FORCE EXIT
  Daily loss > ₹20K → HALT all trading
  1:30 PM + expiry  → FORCE EXIT

Status log every 5 minutes:
  "STATUS | 22500CE | Entry ₹125 | Now ₹142 | +13.6% | Trail ₹134"
```


## Phase 4: Post-Market (after 15:10)

```
1. EOD square-off — close all positions at market price
   Verify: 0 open positions (retry if stuck)

2. Save portfolio snapshot
   Capital: ₹1,50,000 + today's P&L

3. Update circuit breaker state
   Daily loss > ₹20K → HALTED for tomorrow
   4+ consecutive SL → PAUSED
   Drawdown ≥ 22%    → HALTED (3 trading days)
   Drawdown ≥ 15%    → add conviction +1.0 (harder entry)

4. Save trades to DB
   All fields: entry, exit, type, premium, qty, P&L, regime

5. Telegram daily report
   Date, capital, day P&L, trades, regime, VIX, circuit state
```


## Phase 5: EOD Data Save (waits until 15:30)

```
Wait until market close at 15:30
  "Market still open until 15:30 — waiting to save complete EOD data..."

Save full-day candle data to DB
Cleanup and exit
```


## Feature Engineering (34 features)

```
From NIFTY candles (11):
  rsi_14, macd_histogram, bb_position, atr_pct, adx_14,
  volume_ratio, volatility_20d, returns_1d, returns_5d,
  price_to_sma50, mfi_14

From FII/DII (7):
  fii_net_flow_1d, fii_net_flow_5d, fii_flow_momentum,
  fii_net_direction, fii_net_streak, dii_net_flow_1d, india_vix

From VIX extended (3):
  vix_change_pct, vix_percentile_252d, vix_5d_ma

From External Markets (9):
  sp500_prev_return, nasdaq_prev_return, crude_prev_return,
  gold_prev_return, usdinr_prev_return, sp500_nifty_corr_20d,
  crude_nifty_corr_20d, dxy_momentum_5d, global_risk_score

From Options (4):
  pcr_ratio, max_pain_distance, delivery_pct, futures_premium_pct

Missing data defaults to 0.0 (graceful degradation)
```


## Safety Systems

```
Capital:         ₹1,50,000 (from .env, stays fixed)
Deploy cap:      ₹75,000 max per position (capped at Upstox available in live)
Risk cap:        ₹15,000 max loss per trade
Daily halt:      ₹20,000 total daily loss → stop all trading
Min wallet:      ₹50,000 minimum balance (live aborts if below)

Drawdown halt:   ≥ 22% drawdown → HALTED for 3 trading days
Drawdown warn:   ≥ 15% drawdown → conviction +1.0 (harder entry)

Consecutive SL:  3 SLs same direction → block that direction (5-day cooldown)
Consec losses:   6 consecutive losses → PAUSE 60 min
                 10 consecutive losses → HALT

VIX ceiling:     VIX > 35 → no trades
Min premium:     ₹80 (avoid illiquid options)
Force exit:      3:10 PM every day (1:30 PM on expiry)

Token check:     Startup validates Upstox token (live aborts if expired)
Holiday check:   Upstox API → cache → hardcoded fallback
Expiry detect:   Tuesday = expiry, shifts to Monday if Tuesday is NSE holiday

Direction recovery:
  30-min timeout → re-score (max 3/day)
  2 SLs same dir → re-score (max 3/day, shared counter)
```


## NIFTY Weekly Expiry

```
Normal expiry: Tuesday
If Tuesday is NSE holiday: shifts to previous trading day (usually Monday)

Examples:
  Holi on Tue Mar 3      → expiry on Mon Mar 2
  Mahavir Jayanti on Tue → expiry on Mon

On expiry day:
  Force exit by 1:30 PM (theta decay penalty)
  Wider SL (+5%), lower TP (×0.65)
  Max 1 trade
```


## Backtest Results (V8 Final)

```
Period:           2023-05 to 2026-02 (693 trading days)
Capital:          ₹50,000
Return:           1,445% (₹50K → ₹7.7 lakh)
Trades:           301
Win Rate:         61.1%
Profit Factor:    2.64
Sharpe:           2.778
Max Drawdown:     14.19%
Profitable Months: 31/34 (91%)

By Regime:
  TRENDING:    236 trades, 57.2% WR, +₹578K
  RANGEBOUND:   64 trades, 76.6% WR, +₹153K
  VOLATILE:      1 trade,  0.0% WR, -₹8.6K
```
