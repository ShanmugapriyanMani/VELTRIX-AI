# Veltrix V9 — System Workflow

Stage: PLUS | Capital: ₹1,50,000 | Deploy Cap: ₹75,000

CLI:
  python src/main.py --mode paper         Paper trading (simulated)
  python src/main.py --mode live          Live trading (Upstox)
  python src/main.py --mode backtest      Full historical backtest
  python src/main.py --mode report        Paper trading report (DB)
  python src/main.py --mode fetch         Data fetch only


## V9 Changes from V8

```
REMOVED:
  - Debit spreads (PF < 1 after brokerage — 147 trades V8, 12 trades V9, all negative)
  - ML influence on scoring (V9.2 rebuild: 49.1% test acc, 27.5% gap → permanently disabled)
  - Global direction cooldown (replaced with per-regime)
  - 4-tier trailing stop starting at +8%

ADDED:
  - ELEVATED sub-regime (VIX 20-28 rising)
  - Asymmetric PE/CE conviction thresholds (PE proven 2.83x more profitable per trade)
  - F10 Global Macro factor (DXY momentum, SP500/NIFTY correlation, global risk)
  - 3-tier trailing stop starting at +5%
  - TP ladder simulation (captures partial gains on EOD exits)
  - Partial scale-out (50% exit at +30%, trail remainder)
  - Weekly/monthly circuit breakers
  - Conviction-scaled lot sizing (0.5x at threshold, 1.0x at +2.0 excess)
  - Per-regime direction cooldown (2-day block, not 5-day global)
  - Confirmation gate tightened (11-1PM: 2/4 required, was 1/4)
  - Theta gate (expiry day naked buy restrictions)
```


## Backtest Results (V9.2 — 5-Year, ML Disabled)

```
Period:            2021-05 to 2026-03 (1186 trading days)
Capital:           ₹1,50,000
Final Capital:     ₹28,30,164
Total Return:      1,787%
CAGR:              86.67%
Trades:            489
Win Rate:          71.2%
Profit Factor:     4.72
Sharpe Ratio:      2.633
Sortino Ratio:     5.486
Max Drawdown:      16.76%
Profitable Months: 56/59 (95%)
Avg Win:           ₹9,770
Avg Loss:          ₹5,106

By Direction:
  CE Buy:   309 trades, 68.9% WR, +₹11.05L
  PE Buy:   177 trades, 74.6% WR, +₹15.64L
  CE Sell:    3 trades, 100% WR,  +₹0.11L (credit spreads)

By Regime:
  TRENDING:   401 trades, 69.6% WR, +₹20.80L
  RANGEBOUND:  56 trades, 78.6% WR,  +₹2.52L
  VOLATILE:    32 trades, 78.1% WR,  +₹3.48L
  ELEVATED:     0 trades (unvalidated — VIX 20-28 rising rare in data)

By Trade Type:
  NAKED_BUY:     457 trades, 70.7% WR, PF 4.33
  CREDIT_SPREAD:  32 trades, 78.1% WR, PF 19.37

Exit Reasons:
  EOD Exit:      291 (60%)  — TP ladder untested on OHLC, needs live tick validation
  Take Profit:   123 (25%)
  Stop Loss:      58 (12%)
  Trail Stop:     17  (3%)
```


## Startup & Market Guards

```
SYSTEM STARTS
  1. Load .env → credentials + TRADING_STAGE=PLUS
  2. Load .env.plus → capital, risk, thresholds (override=True)
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


## Phase 0b: ML Model Training (Diagnostics Only)

```
Schedule: Monday only (weekly retrain)
  Monday    → retrain LightGBM (90-day window, 19 features)
  Tue–Fri   → use cached model from models/lgbm_latest.pkl

ML auto weight = 0.0 (permanently disabled in V9.2)
Model trains for diagnostic logging only — predictions do NOT affect scoring.
V9.2 rebuild (22 entry-contemporaneous features): 49.1% test acc, 27.5% gap across 117 folds.
Original V9 (37 macro features): 48% test acc, 31.5% gap. ML confirmed unpredictive.
```


## Phase 1: Pre-Market Data (waits until 08:30)

```
Load FII/DII history (268 days)
Fetch India VIX live quote
Fetch NIFTY option chain:
  PCR, Max Call OI, Max Put OI, Max Pain

Telegram alert: "Bot started | Capital ₹1,50,000"
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

Weekly/monthly guards checked:
  weekly_pnl < -₹35K → halt remainder of week
  weekly_pnl < -₹20K → reduce to 2 trades/day + conviction +0.5
  monthly_return < -8% → conviction +0.5 for rest of month

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
  │ Weekly/monthly guards (V9)    │
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
  │ VOLATILE:  VIX ≥ 28,          │
  │   or VIX > 20 + change > 3%  │
  │ ELEVATED:  VIX 20-28 rising   │
  │   above 5d MA by 12%+        │
  │ TRENDING:  ADX > 25           │
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
  │ F7 (ML disabled):  0.0        │
  │ F8 OI/PCR:         ±2.0       │
  │ F9 Volume:         ±1.0       │
  │ F10 Global Macro:  ±1.5 (V9) │
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
  │ Asymmetric PE/CE thresholds:  │
  │                                │
  │ Regime       CE     PE        │
  │ TRENDING     2.0    1.5       │
  │ RANGEBOUND   2.25   1.75     │
  │ VOLATILE     3.0    2.5       │
  │ ELEVATED     2.5    2.0       │
  │                                │
  │ + Monday penalty: +0.3        │
  │ + Afternoon (>1PM): +0.5      │
  │ + Weekly soft limit: +0.5     │
  │ + Monthly guard: +0.5         │
  │                                │
  │ score_diff ≥ threshold → PASS │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7: Trade Type (V9) ──┐
  │ VOLATILE/ELEVATED + conv ≥ T │
  │   → CREDIT_SPREAD             │
  │ VOLATILE/ELEVATED + conv < T │
  │   → SKIP                      │
  │ Any regime + conv ≥ 3.0       │
  │   → NAKED_BUY                 │
  │ RANGEBOUND + conv ≥ 2.5       │
  │   → NAKED_BUY                 │
  │ Everything else → SKIP        │
  │                                │
  │ NO DEBIT SPREADS (eliminated) │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7b: Per-Regime       ─┐
  │     Direction Cooldown (V9)    │
  │ ≥ 3 SLs same regime+direction │
  │ → block that combo for 2 days │
  │ Per-regime, not global         │
  │ Resets on non-SL exit          │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7c: Theta Gate (V9) ─┐
  │ Expiry day + NAKED_BUY:       │
  │  score < 3.5 OR premium < 120 │
  │  → SKIP (no weak expiry buys) │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 8: Confirmation ──────┐
  │ 4 intraday triggers:          │
  │  1. Price vs Day Open         │
  │  2. RSI 5-min (>55 or <45)   │
  │  3. Morning Range Breakout   │
  │  4. PCR (<0.7 or >1.2)       │
  │                                │
  │ 9:30–11:00  → need 2/4        │
  │ 11:00–13:00 → need 2/4 (V9)  │
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
  │ Credit spread: ATM sell +     │
  │   OTM buy, width 200 points  │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 10: Position Sizing ──┐
  │ Conviction scaling (V9):      │
  │  excess = score - threshold   │
  │  scale = min(1.0,             │
  │    0.5 + (excess/2.0) × 0.5) │
  │  0.5x lots at threshold       │
  │  1.0x lots at threshold + 2.0 │
  │                                │
  │ NAKED BUY:                    │
  │  deploy = 75000/(prem × 65)  │
  │  risk = 15000/(prem×SL%×65)  │
  │  lots = min(deploy, risk)     │
  │  lots = max(1, lots × scale)  │
  │                                │
  │ CREDIT SPREAD:                │
  │  Based on margin / max loss   │
  │  lots × scale applied         │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 11: Filters ──────────┐
  │ Circuit breaker: NORMAL?      │
  │ Trades today < 4?             │
  │ Naked < 2? Spread < 2?       │
  │ Direction not blocked?        │
  │ Regime cooldown clear? (V9)   │
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


### Trade Types (V9)

```
NAKED_BUY (conviction ≥ threshold or ≥ 3.0):
  Single leg — BUY NIFTY CE or PE
  Max 2 naked trades per day
  Full deploy cap applies (₹75,000)
  Full risk cap applies (₹15,000)

CREDIT_SPREAD (VOLATILE/ELEVATED regime only):
  Two legs — SELL near-OTM + BUY far-OTM (opposite type)
  Collect premium upfront, limited risk from protection leg
  Width: 200 points (SPREAD_WIDTH config)
  SL: 2.0x net credit received | TP: 80% of max profit
  Max 2 spread trades per day

NO DEBIT SPREADS — permanently eliminated in V9.
  V8: 147 trades, PF 1.34
  V9 test: 12 trades, PF 0.67 (negative after brokerage)
```


### Stop Loss Logic — 3-Layer Adaptive SL

```
LAYER 1: VIX-Adaptive Base SL
  VIX < 13  → 25%
  VIX 13-18 → 30%
  VIX 18-22 → 30%
  VIX 22-28 → 25%
  VIX 28-35 → 20%
  VIX > 35  → 20%

LAYER 2: Regime Multiplier (applied to base SL)
  TRENDING:   × 1.00 (no change)
  RANGEBOUND: × 0.85 (tighter — quick reversals)
  VOLATILE:   × 1.20 (wider — room for swings)
  ELEVATED:   × 1.10 (slightly wider)

LAYER 3: Premium-Level Hard Clamp
  Premium < ₹100  → floor at 30% minimum (cheap options need room)
  Premium ₹100-200 → adj_sl as-is
  Premium > ₹200  → cap at 20% maximum (expensive = tighter risk)

ADDITIONAL MODIFIERS:
  Losing streak ≥ 3:       adj_sl × 0.90 (10% tighter)
  Expiry day:              adj_sl × 1.05 (5% wider)
  Estimated trades:        sl × 0.75 (25% tighter, less confidence)

FINAL: sl_price = entry_premium × (1 - final_sl_pct)
Applied to premium entry price, not position cost.

CREDIT_SPREAD SL:
  SL = credit_received × 2.0x
  Triggers when loss exceeds 2× credit received.
  Calculated on P&L, not premium price.
```


### Take Profit Logic — 3-Layer Adaptive TP

```
LAYER 1: VIX-Adaptive Base TP
  VIX < 13  → 40%
  VIX 13-18 → 45%
  VIX 18-22 → 55%
  VIX 22-28 → 60%
  VIX 28-35 → 45%
  VIX > 35  → 40%

LAYER 2: Regime Multiplier (applied to base TP)
  TRENDING:   × 1.30 (let trends run)
  RANGEBOUND: × 0.70 (quick in/out)
  VOLATILE:   × 1.50 (capture big swings)
  ELEVATED:   × 1.40 (wide, volatile-like)

LAYER 3: Conviction & Expiry Modifiers
  Conviction ≥ 4.0:  adj_tp × 1.20 (+20% wider)
  Conviction ≥ 3.0:  adj_tp × 1.10 (+10% wider)
  Expiry day:        adj_tp × 0.65 (35% tighter — quick profit)

No premium-level clamp on TP (unlike SL).

FINAL: tp_price = entry_premium × (1 + adj_tp)
Applied to premium entry price.

CREDIT_SPREAD TP:
  TP = credit_received × 80%
  Triggers when profit reaches 80% of credit received.
  Calculated on P&L, not premium price.
```


### Trailing Stop (V9 — 3-Tier from +5%)

```
TRENDING/VOLATILE/ELEVATED (trail_enabled = True):
  +25% gain → trail floor = high_premium × 0.93 (7% below peak)
  +15% gain → trail floor = high_premium × 0.95 (5% below peak)
  +5% gain  → trail floor = high_premium × 0.97 (3% below peak)

RANGEBOUND (trail_enabled = False):
  +25% gain → trail floor = high_premium × 0.93
  +15% gain → trail floor = high_premium × 0.95
  No tier at +5% (less sensitive in range)

Trail only triggers if close drops below trail floor
AND close is still above SL (trail overrides EOD, not SL).

Changed from V8: was 4-tier starting at +8%.
```


### TP Ladder (V9 — Backtest Simulation)

```
Condition: trade was profitable intraday but closing flat/negative
  high > entry AND close ≤ entry × 1.02

Intraday High       Exit At          Gain Captured
≥ entry × 1.15  →  entry × 1.10     +10%
≥ entry × 1.10  →  entry × 1.05     +5%
≥ entry × 1.05  →  entry × 1.025    +2.5%

Does NOT override base TP hit. Only rescues EOD exits.
In backtest: uses daily OHLC high (rough simulation).
In live: fires at time checkpoints (12:30, 2:00, 2:30 PM).
This is the primary V10 validation target.
```


### Partial Scale-Out (V9)

```
NAKED_BUY only:
  If intraday high ≥ entry × 1.30:
    Exit 50% lots at entry × 1.30 (+30%)
    Remaining 50% continues through normal SL/TP/trail/EOD exit
    Both portions' P&L summed for total trade P&L

Does not apply to credit spreads (fixed premium, both legs exit together).
```


### Expiry Day Rules

```
Expiry: Tuesday (NIFTY weekly)
If Tuesday is NSE holiday: shifts to previous trading day (usually Monday)

On expiry day:
  Wider SL (+5% buffer)
  Lower TP (× 0.65)
  Max 1 trade
  Force exit by 1:30 PM (EXPIRY_EXIT_BY)

  V9 Theta Gate:
    NAKED_BUY requires score_diff ≥ 3.5 AND premium ≥ ₹120
    Else → SKIP (no weak expiry naked buys)
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
```


### Monitoring Loop (while position open)

```
Check interval: 30s (no position) / 15s (position, live only)

  Price hits SL     → EXIT (market order)
  Price hits TP     → EXIT
  Trail triggered   → EXIT
  TP ladder fires   → EXIT (V9 — time-based checkpoints in live)
  Partial scale-out → EXIT 50% at +30%, trail rest (V9)
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

4. Update weekly/monthly guards (V9)
   weekly_pnl += today's P&L
   Check weekly soft/hard limits
   Check monthly drawdown boost

5. Update per-regime direction cooldown (V9)
   If SL exit: increment consec_sl_by_regime[regime][direction]
   If ≥ 3 consecutive SLs → block regime+direction for 2 days
   Non-SL exit → reset counter for that regime+direction

6. Save trades to DB
   All fields: entry, exit, type, premium, qty, P&L, regime

7. Telegram daily report
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

Weekly soft:     ₹20,000 weekly loss → 2 trades/day + conviction +0.5 (V9)
Weekly hard:     ₹35,000 weekly loss → halt remainder of week (V9)
Monthly guard:   -8% monthly return → conviction +0.5 for rest of month (V9)

Drawdown halt:   ≥ 22% drawdown → HALTED for 3 trading days
Drawdown warn:   ≥ 15% drawdown → conviction +1.0 (harder entry)

Direction cooldown (V9):
  ≥ 3 SLs same regime + direction → block that combo for 2 days
  Per-regime tracking (TRENDING_CE, RANGEBOUND_PE, etc.)
  Resets on non-SL exit for that regime+direction

Consecutive losses:
  6 consecutive losses → PAUSE 60 min
  10 consecutive losses → HALT

VIX ceiling:     VIX > 35 → no trades
Min premium:     ₹80 (avoid illiquid options)
Force exit:      3:10 PM every day (1:30 PM on expiry)

Token check:     Startup validates Upstox token (live aborts if expired)
Holiday check:   Upstox API → cache → hardcoded 2026 fallback
Expiry detect:   Tuesday = expiry, shifts to Monday if Tuesday is NSE holiday

Direction recovery:
  30-min timeout → re-score (max 3/day)
  2 SLs same dir → re-score (max 3/day, shared counter)

Double-defense note:
  Losing streak (≥ 3 SLs) tightens SL by 10%
  AND weekly soft limit raises conviction +0.5
  Both fire simultaneously — monitor in paper trading for
  over-filtering valid recovery entries after drawdowns.
```


## Known Limitations to Validate in Paper Trading

```
1. EOD exit rate ~60% — TP ladder untested on real tick data
   Track whether ladder fires correctly at 12:30/2:00/2:30 PM in live

2. Trailing stop Tier 1 (+5%) may be too sensitive 9:30-11:00
   May need raising to +7% for opening hour only

3. Stale trade detector (45 min, 3% threshold) — may exit too early
   on slow consolidation before breakout. Track post-exit price action

4. ELEVATED regime has 0 trades in backtest — logic unvalidated
   Monitor VIX 20-28 rising conditions in live

5. Conviction scaling 0.5x minimum confirmed correct
   Tested 0.7x: worse (Max DD +1.73%, PF -0.13, avg loss +₹248)
   Avg win gap vs V8 is an exit problem, not sizing

6. Double-defense overlap (losing streak SL tightening + weekly guard)
   Watch for clusters of skipped high-conviction signals after drawdown
```


## V10 Planned Improvement (Do Not Implement Yet)

```
Intraday momentum exit calibrated from live tick data:
  Analyze when EOD-exiting trades typically peak intraday
  Build data-backed exit rule from 4+ weeks of live tick data
  This is the primary unresolved problem in V9.
  60% EOD exit rate → target 30-35% with validated TP ladder.
```
