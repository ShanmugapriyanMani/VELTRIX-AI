# Veltrix V9 — System Workflow

Stage: PLUS | Capital: ₹1,50,000 | Deploy Cap: ₹75,000

CLI:
  python src/main.py --mode paper         Paper trading (simulated)
  python src/main.py --mode live          Live trading (Upstox)
  python src/main.py --mode backtest      Full historical backtest
  python src/main.py --mode report        Paper trading report (DB)
  python src/main.py --mode fetch         Data fetch only
  python src/main.py --mode ml_backfill   Download 5-min candles for ML
  python src/main.py --mode ml_train      Train direction + quality models
  python src/main.py --mode ml_status     Model status, candle coverage, drift
  python src/main.py --mode ml_report     Training history, prediction accuracy
  python src/main.py --mode backup        Manual Google Drive backup


## V9 Changes from V8

```
REMOVED:
  - Debit spreads (PF < 1 after brokerage — 147 trades V8, 12 trades V9, all negative)
  - Old LightGBM ML model (V9.2: 49.1% test acc, 27.5% gap → replaced by XGBoost)
  - Global direction cooldown (replaced with per-regime)
  - 4-tier trailing stop starting at +8%
  - Weekly/monthly circuit breaker halts (V9.3 — simplified to 2 rules only)
  - CB PAUSED/WARNING/KILLED states, drawdown halt, half-lot mode (V9.3)

ADDED:
  - ELEVATED sub-regime (VIX 20-28 rising)
  - Asymmetric PE/CE conviction thresholds (PE proven 2.83x more profitable per trade)
  - F10 Global Macro factor (DXY momentum, SP500/NIFTY correlation, global risk)
  - 3-tier trailing stop starting at +5%
  - TP ladder simulation (captures partial gains on EOD exits)
  - Partial scale-out (50% exit at +30%, trail remainder)
  - Conviction-scaled lot sizing (0.5x at threshold, 1.0x at +2.0 excess)
  - Per-regime direction cooldown (2-day block, not 5-day global)
  - Confirmation gate tightened (11-1PM: 2/4 required, was 1/4)
  - Theta gate (expiry day naked buy restrictions)
  - Iron Condor strategy for RANGEBOUND regime (V9.3)
  - 10 CRITICAL + 14 HIGH + 4 MEDIUM stability fixes (V9.3)
  - NaN-safe scoring with _sg() helper (V9.3)
  - Simplified circuit breaker: 2 rules, 2 states (V9.3)
  - Equity curve sizing: multi-day DD protection (4-tier drawdown schedule) (V9.3)
  - CB loss-based size reduction: 0 SL→1.0x, 1 SL→0.75x, 2+ SL→0.50x (V9.3)
  - ML Stage 1 re-enabled with asymmetric weights (PE=1.5, CE=0.3) (V9.3)
  - Trail stop tightened: +5%→0.96, +12%→0.94, +25%→0.91 (V9.3 FINAL)
  - TP ladder time checkpoints in live: 12:00/13:00/14:00 (V9.3 FINAL)
  - Signal bucket grouping (4 buckets, caps at 99 — structure ready) (V9.3 FINAL)
  - IV awareness filter (VIX ratio vs 20d avg, ±0.25-0.50 threshold adj) (V9.3 FINAL)
  - OI change rate confirmation (30-min snapshots, ±0.25-0.75 threshold adj) (V9.3 FINAL)
  - Parameter robustness verified: all 3 thresholds ROBUST (±0.50 tested) (V9.3 FINAL)
  - Price contradiction filter: blocks entry when price vs open contradicts direction (V9.3 FINAL)
  - ML disagreement filter: blocks entry when ML Stage 1 predicts opposite direction >60% (V9.3 FINAL)
  - Old LightGBM permanently disabled (_load_model returns False) (V9.3 FINAL)
  - Option chain 3-attempt retry with log noise reduction (V9.3 FINAL)
  - Trade DB exit records now include all price fields (V9.3 FINAL)
  - Force shutdown uses os._exit(1) — no threading exception (V9.3 FINAL)
  - CB state file written on daily reset (ensures backup finds it) (V9.3 FINAL)
  - ML features expanded 46→51: PCR, VIX percentile/change, max pain distance, FII flow (V9.3 FINAL)
  - Separate PE/CE binary direction models (Stage 1b) alongside 2-class model (V9.3 FINAL)
  - predict_direction_v2(): combines PE+CE binary probs (V9.3 FINAL)
  - Hybrid binary model usage: CE model (58.7%) used for CE signals even without PE model (V9.3 FINAL)
  - ML disagreement block uses deployed binary model for cross-checking (V9.3 FINAL)
```


## Backtest Results (V9.3 FINAL — 5-Year, Hybrid Binary ML)

```
Period:            2021-05 to 2026-03 (1186 trading days)
Capital:           ₹1,50,000
Total P&L:         +₹29,79,456
CAGR:              90.28%
Trades:            494
Win Rate:          74.5%
Profit Factor:     5.12
Sharpe Ratio:      2.753
Max Drawdown:      14.61%

By Direction:
  CE Buy:          299 trades, WR 69.2%, P&L +₹10.6L
  PE Buy:          188 trades, WR 82.4%, P&L +₹17.9L
  CE Sell:           7 trades, WR 85.7%, P&L +₹1.3L

By Trade Type:
  NAKED_BUY:       458 trades, PF 4.63
  CREDIT_SPREAD:    36 trades, PF 14.36

Exit Reasons:
  EOD Exit:      ~60%  — TP ladder active in live, validation in progress
  Take Profit:   ~25%
  Stop Loss:     ~12%
  Trail Stop:     ~3%  — 15 trail exits (improved from 11)

TP Ladder:       3 backtest + live checkpoints active

Parameter Robustness (all ROBUST):
  TRENDING threshold:   0.0% CAGR spread (zero variation across ±0.50)
  VOLATILE threshold:   1.9% CAGR spread (89.2-91.0% across ±0.50)
  RANGEBOUND threshold: 0.1% CAGR spread (DD sensitive below 2.0)
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
  Decode JWT expiry (base64 payload, no secret needed)
  Live mode + no token → ABORT ("run: python scripts/auth_upstox.py")
  Paper mode + no token → WARN, continue with cached data

  Token lifecycle (live + paper):
    Startup check: TOKEN_OK / TOKEN_WARNING (< 60 min) / TOKEN_EXPIRED
    TokenWatcher daemon thread: checks every 5 min
      → Telegram alert at 30 min before expiry (with 30-min cooldown)
      → CRITICAL alert on token expiry
    Token failure NEVER stops trading (alert only)
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


## Phase 0b: Multi-Stage ML System (XGBoost)

```
Old LightGBM system replaced by multi-stage XGBoost (V9.2 had 49.1% test acc, 27.5% gap).
XGBoost system is the primary ML path:

STAGE 1 — Direction Model (CE/PE 2-class):
  XGBoost binary (binary:logistic), max_depth=2, lr=0.03, n_estimators=120
  51 entry-contemporaneous features from 4 years of NIFTY 5-min candles:
    Group A (16): Daily technicals (RSI, MACD, BB, ATR, ADX, EMA slopes, volume, MFI, OBV, VWAP)
    Group B (6):  Returns & momentum (1d/5d/20d returns, volatility, range)
    Group C (8):  Intraday session (morning momentum, afternoon strength, bar vol, etc.)
    Group D (4):  Candlestick (body, shadows, gap)
    Group E (6):  Market context (days to expiry, day of week, price vs SMA/EMA)
    Group F (6):  Normalized context (gap vs avg, first candle, prev day)
    Group G (5):  External/Options (PCR, VIX percentile/change, max pain, FII flow)
  Walk-forward split: last 63 trading days = test, rest = train (no shuffle)
  Deploy gate: test_acc > 52%, gap < 20%
  Feature version: 3 (FEATURE_NAMES in candle_features.py)

STAGE 1b — Separate PE/CE Binary Models:
  pe_direction_v1: "Will NIFTY close lower?" (1=bearish, 0=not)
  ce_direction_v1: "Will NIFTY close higher?" (1=bullish, 0=not)
  XGBoost binary, max_depth=4, lr=0.05, n_estimators=200
  Same 51 features, but 0.2% threshold labels (not close>=open)
  scale_pos_weight handles class imbalance automatically
  Deploy gate: test_acc > 52%, gap < 15% (stricter gap)

  Current results:
    CE binary: v1 test_acc=58.7%, deployed ✅
    PE binary: v1 test_acc=31.7%, not deployed ❌ (overfitting)

  HYBRID USAGE (Factor 7 scoring):
    V2 system (both PE+CE deployed): use both binary probs independently
    Hybrid (only CE deployed):
      CE signals: use CE binary model (58.7% — vs 37.1% from V1)
      PE signals: fall back to V1 2-class model
    V1 fallback: existing 2-class probs (PE=1.5, CE=0.3 weights)

  PREDICT_DIRECTION_V2:
    pe_prob > 0.55 and pe_prob > ce_prob → PE
    ce_prob > 0.55 and ce_prob > pe_prob → CE
    Otherwise → FLAT (no strong signal)

  ML DISAGREEMENT:
    Uses deployed binary model for cross-checking:
      CE model blocks PE entry if ce_binary_prob > 0.60
      PE model blocks CE entry if pe_binary_prob > 0.60
      Falls back to V1 probs when opposing model not deployed

STAGE 2 — Quality Model (win probability per trade):
  XGBoost binary (binary:logistic), max_depth=3, n_estimators=50
  12 features: score_diff, conviction, vix/rsi/adx/pcr at entry, ml_probs
  Requires 30+ labeled trades to activate (auto-activates at trade #30)
  Deploy gate: test_acc > 58%, gap < 18%
  If win_prob < 0.45 → trade blocked by quality gate

RETRAINING (EOD 15:30 daily):
  Direction model: retrain daily at 15:30 (was Monday only)
  PE/CE binary models: retrain daily at 15:30 alongside direction model
  Quality model: retrain if 30+ labeled trades
  Drift detected → retrain direction model

DRIFT DETECTION:
  Rolling 20-day accuracy vs baseline test_acc
  Only uses predictions since current model deployed (not total)
  Minimum 10 predictions required before drift check
  Drift threshold: 10% accuracy drop → trigger retrain

ML OFF BY DEFAULT — only active after first successful train + deploy.
Models saved in data/models/ (not models/).
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

WEBSOCKET LTP FEED (live + paper):
  MarketDataStreamerV3 (Upstox SDK, background thread)
  Protocol: wss://api.upstox.com/v3/feed/market-data-feed (protobuf → dict)
  Mode: LTPC (last trade price + close price)
  Auto-reconnect: 50 retries, 5s interval
  Subscribe on trade entry, unsubscribe on trade exit
  Thread-safe LTP cache (threading.Lock)
  broker.get_ltp() checks WS cache first, REST fallback
  Eliminates 12N REST API calls/min during fast poll
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
  circuit_breaker.reset_daily()

Fetch 50+ days NIFTY daily candles
Compute technicals: EMA 9/21/50, RSI, MACD, BB, ADX
Refresh options instrument master (today's strikes)
```


### Each Scan Cycle

```
DATA READINESS GATE (startup):
  VIX=0 / NIFTY=0 / no candles at startup → entries blocked
  Gate opens when: VIX > 0, NIFTY > 0, candles loaded
  Once open, never resets during session
  Position monitoring continues even while gate is closed
  Logs WAITING_DATA every 10 loops, DATA_READY + Telegram on open

SCAN INTERVAL (two-speed loop):
  No position:              30 seconds (full signal computation)
  Position open:             5 seconds (LTP-only fast poll)
  Network down:             60 seconds

  Fast poll (5s): fetch LTP → update price → check SL/TP/trail → exit if hit
  Full loop (30s): candle fetch, scoring, signal generation, regime updates
  POSITION_MONITOR log every ~30s with ltp, sl, tp levels

  ┌─── STEP 1: Circuit Breaker ───┐
  │ 2 rules only (V9.3):         │
  │  Rule 1: 2 consecutive SL    │
  │    exits → HALT rest of day  │
  │  Rule 2: Daily loss > ₹20K   │
  │    → HALT rest of day        │
  │                                │
  │ States: NORMAL / HALTED       │
  │ Daily reset clears everything │
  │ No weekly/monthly carry-over  │
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
  ┌─── STEP 4: 10-Factor Scoring ─┐
  │ F1 EMA Trend:     ±2.5       │
  │ F2 RSI+MACD:      ±2.0       │
  │ F3 Price Action:   ±1.5       │
  │ F4 Mean Revert:    ±2.5       │
  │ F5 BB Position:    ±0.75      │
  │ F6 VIX Sentiment:  ±0.8       │
  │ F7 ML Stage 1:     ±1.5       │
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
  │                                │
  │ score_diff ≥ threshold → PASS │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 6b: IV + OI Filters ──┐
  │ IV AWARENESS (V9.3):          │
  │  vix_ratio = VIX / VIX_20d   │
  │  > 1.30 (IV_HIGH):  +0.50    │
  │  0.80-1.30 (NORMAL): +0.00   │
  │  < 0.80 (IV_LOW):   -0.25    │
  │  Skip on expiry days          │
  │                                │
  │ OI CHANGE RATE (V9.3):        │
  │  30-min OI snapshots          │
  │  PE confirmed (puts +2%):     │
  │    threshold -0.25            │
  │  PE contradicted (calls +3%): │
  │    threshold +0.75            │
  │  Neutral: no change           │
  │  First 30 min: NEUTRAL        │
  │  Backtest: always NEUTRAL     │
  │                                │
  │ Combined adjustment stacks    │
  │ with base threshold           │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7: Trade Type (V9) ──┐
  │ VOLATILE/ELEVATED + conv ≥ T │
  │   → CREDIT_SPREAD             │
  │ VOLATILE/ELEVATED + conv < T │
  │   → SKIP                      │
  │ RANGEBOUND + IC conditions   │
  │   → IRON_CONDOR (if enabled) │
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
  ┌─── STEP 8: Fuzzy Confirmation ┐
  │ 4 triggers (each 0.0 – 1.0): │
  │  T1: Price vs Open gradient   │
  │    CE: clip(dist%/0.5, 0, 1)  │
  │  T2: RSI gradient             │
  │    CE: clip((RSI-45)/20,0,1)  │
  │  T3: Breakout vs rolling range│
  │    Range updates every 30 min │
  │    Resets after trade close    │
  │  T4: PCR gradient             │
  │    CE: clip((1.2-PCR)/0.8,0,1)│
  │                                │
  │ Entry: sum(T1+T2+T3+T4) ≥ 2.0│
  │  + regime/expiry adjustments  │
  │  + soft abort +0.5 after 11:30│
  │                                │
  │ Failed 30 min → re-score dir  │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 8b: Pre-Entry Guards ─┐
  │ PRICE CONTRADICTION (V9.3):   │
  │  Before 11:30, NORMAL expiry: │
  │  PE signal + NIFTY > +0.3%   │
  │    above open + RSI > 55     │
  │    → BLOCK (wait for confirm)│
  │  CE signal + NIFTY < -0.3%   │
  │    below open + RSI < 45     │
  │    → BLOCK                    │
  │  Skipped on expiry days       │
  │                                │
  │ ML DISAGREEMENT (V9.3):       │
  │  Before 11:30:                │
  │  PE signal + ML prob_CE > 60% │
  │    → BLOCK                    │
  │  CE signal + ML prob_PE > 60% │
  │    → BLOCK                    │
  │  Config: ML_DISAGREEMENT_     │
  │    THRESHOLD=0.60              │
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
  │                                │
  │ EQUITY CURVE SIZING (V9.3): │
  │  Multi-day DD protection:    │
  │  DD < 5%  → 1.0x (full)     │
  │  DD 5-10% → 0.75x           │
  │  DD 10-15%→ 0.50x           │
  │  DD > 15% → 0.25x           │
  │  Uses rolling equity vs peak │
  │                                │
  │ CB LOSS-BASED REDUCTION:     │
  │  0 SLs today → 1.0x         │
  │  1 SL today  → 0.75x        │
  │  2+ SLs today→ 0.50x        │
  │  Multiplied with conviction  │
  │                                │
  │ SIGNAL BUCKETS (V9.3):       │
  │  MOMENTUM (F1+F2+F3+F9)      │
  │  FLOW (F8+F10)               │
  │  VOLATILITY (F5+F6)          │
  │  MEAN_REV (F4)               │
  │  Caps at 99 (structure ready)│
  │  Bucket logs in diagnostics  │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 11: Filters ──────────┐
  │ Circuit breaker: NORMAL?      │
  │ Trades today < 4?             │
  │ Naked < 2? Spread < 2?       │
  │ IC < 1? (if IC_ENABLED)      │
  │ Direction not blocked?        │
  │ Regime cooldown clear? (V9)   │
  │ Whipsaw: ADX ok?             │
  │ VIX < 35?                     │
  │ Time < 14:30 (NO_NEW_TRADE)?  │
  │ ALL PASS → EXECUTE            │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 11b: Margin Check ────┐
  │ Live mode only:               │
  │ Fetch available_margin from   │
  │ Upstox /v2/user/get-funds    │
  │                                │
  │ < required → BLOCK + alert   │
  │ < required × 1.4 → WARN     │
  │   (trade proceeds)           │
  │ API fail → proceed (no block)│
  │ Paper mode: skipped entirely │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 12: EXECUTE ──────────┐
  │ Place order (MARKET)          │
  │                                │
  │ LIVE MODE:                    │
  │   wait_for_fill(30s timeout) │
  │   Poll get_order_status/2s   │
  │   complete → use avg_price   │
  │   rejected → ABORT, alert    │
  │   timeout → cancel, ABORT    │
  │   Position opened only after │
  │   fill confirmed             │
  │                                │
  │ PAPER MODE:                   │
  │   Instant fill (unchanged)   │
  │   Slippage: ₹1.50/unit       │
  │                                │
  │ Set SL/TP (VIX-adaptive)     │
  │ Lock direction                │
  │ Telegram alert                │
  │                                │
  │ Slippage tracked in DB:       │
  │   signal_price, fill_price,  │
  │   slippage_pct per trade     │
  └────────────────────────────────┘
```


### Trade Types (V9.3)

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

IRON_CONDOR (RANGEBOUND regime, IC_ENABLED=true):
  Four legs — Bear Call Spread + Bull Put Spread
  Profits when NIFTY stays within a range
  Sell CE/PE at OTM strikes + Buy wings SPREAD_WIDTH further out
  Min wing distance: 300 points (sell_CE - sell_PE)
  Net credit must be ≥ ₹50 (IC_MIN_CREDIT)
  SL: 2.0x net credit | TP: 80% of net credit
  Max 1 IC per day (IC_MAX_TRADES_PER_DAY)
  Entry window: 10:00-11:30 (IC_TRADE_START/END)
  Not on expiry days
  Conditions: ADX < 25, VIX 13-26, PCR 0.80-1.20, |score_diff| < 3.0
  Strike selection: OI-based (live/paper) or ATM-based (backtest)
  Currently IC_BACKTEST_ENABLED=false (paper/live only)

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


### Trailing Stop (V9.3 FINAL — 3-Tier from +5%)

```
TRENDING/VOLATILE/ELEVATED (trail_enabled = True):
  +25% gain → trail floor = high_premium × 0.91 (9% below peak)
  +12% gain → trail floor = high_premium × 0.94 (6% below peak)
  +5% gain  → trail floor = high_premium × 0.96 (4% below peak)

RANGEBOUND (trail_enabled = False):
  +25% gain → trail floor = high_premium × 0.91
  +12% gain → trail floor = high_premium × 0.94
  Trail starts at +10% (not +5%, less sensitive in range)

Trail only triggers if close drops below trail floor
AND close is still above SL (trail overrides EOD, not SL).

Changed from V8: was 4-tier starting at +8%.
V9.3 FINAL: tightened from 0.97/0.95/0.93 to 0.96/0.94/0.91.
15 trail exits in backtest (improved from 11).
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
In live: fires at 12:00, 13:00, 14:00
  Peak tracked via _peak_prices dict
  Fires once per checkpoint per position
  Resets on new position opened
  NAKED_BUY only — not spreads or IC
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
Expiry schedule (get_expiry_type):
  Old schedule (before 2025-09-01):
    Thursday → NIFTY_EXPIRY
    Wednesday (after 2023-09-04) → BANKNIFTY_EXPIRY
    Friday → SENSEX_EXPIRY
  New schedule (>= 2025-09-01):
    Tuesday → NIFTY_EXPIRY
    Thursday → SENSEX_EXPIRY

  If expiry day is NSE holiday: shifts to previous trading day

Major expiry (NIFTY/BANKNIFTY):
  Confirmation threshold +1.0
  Position size × 0.75
  Entries blocked after 11:30
  Direction flip blocked
  Hard abort at 11:30

Minor expiry (SENSEX):
  Confirmation threshold +0.5
  Position size × 0.90

General expiry rules:
  Wider SL (+5% buffer)
  Lower TP (× 0.65)
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
Two-speed polling:
  No position:   30s full cycle (scoring, signals, regime)
  Position open:  5s fast poll (LTP fetch → SL/TP/trail check only)
  Network down:  60s

Fast poll (every 5s):
  LTP via WebSocket cache (sub-ms) → REST fallback if WS down
  Check: ltp ≤ sl_price → EXIT STOP_LOSS
  Check: ltp ≥ tp_price → EXIT TAKE_PROFIT
  Check: trailing stop triggered → EXIT TRAIL_STOP
  POSITION_MONITOR log every ~30s: ltp, sl, tp levels

Exit triggers:
  Price hits SL     → EXIT (market order)
  Price hits TP     → EXIT
  Trail triggered   → EXIT
  TP ladder fires   → EXIT (V9 — time-based checkpoints in live)
  Partial scale-out → EXIT 50% at +30%, trail rest (V9)
  3:10 PM           → FORCE EXIT
  Daily loss > ₹20K → HALT (Rule 2)
  2 consecutive SL  → HALT (Rule 1)

Expiry abort:
  Major expiry (NIFTY/BANKNIFTY): hard abort at 11:30
  Non-expiry: soft abort at 11:30 (+0.5 threshold), hard abort at 13:00
  After trade closes: abort bypassed for re-entry

Breakout re-entry:
  After TP/trail exit: rolling range timer cleared, force recompute
  New breakout detected against updated 60-min range (not stale morning range)
  Guards: time < 14:30, no hard abort, max trades not exceeded
```


## Phase 4: Post-Market (after 15:10)

```
1. EOD square-off — close all positions at market price
   Close portfolio positions, IC positions, DB open positions
   Verify: 0 open positions (retry if stuck)

2. Save portfolio snapshot
   Capital: ₹1,50,000 + today's P&L

3. Circuit breaker state persisted to disk
   reset_daily() writes state file immediately (backup always finds it)
   (Next day: reset_daily() clears everything automatically)

4. Update per-regime direction cooldown (V9)
   If SL exit: increment consec_sl_by_regime[regime][direction]
   If ≥ 3 consecutive SLs → block regime+direction for 2 days
   Non-SL exit → reset counter for that regime+direction

5. Save trades to DB
   All fields: entry, exit, type, premium, qty, P&L, regime
   Slippage tracking: signal_price, fill_price, slippage_pct

6. Telegram daily report
   Date, capital, day P&L, trades, regime, VIX, circuit state

7. P&L Reconciliation (live mode only)
   Compare system P&L vs Upstox broker P&L
   Compare system trade count vs broker trade book
   ≤ ₹100 diff → OK (silent)
   ≤ ₹500 diff → WARNING (Telegram alert)
   > ₹500 diff → CRITICAL (urgent Telegram alert)
   Trade count mismatch → separate alert
   All results saved to reconciliation_log table

8. EOD ML model retrain (direction + quality)

9. Daily backup to Google Drive (rclone)
    DB + models + config + today's log → timestamped folder
    30-day retention with auto-prune
    Also runs on cron (10:15 AM Mon-Fri)
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
Min wallet:      ₹50,000 minimum balance (live aborts if below)

CIRCUIT BREAKER (V9.3 — Simplified):
  Rule 1: 2 consecutive SL exits → HALT rest of day
  Rule 2: Daily loss > ₹20,000 → HALT rest of day
  Daily reset every morning clears everything — no carry-over
  States: NORMAL (trade) / HALTED (no new entries, exits still run)
  Size multiplier: CB loss-based (0 SL→1.0, 1 SL→0.75, 2+ SL→0.50)
  Equity curve sizing: 4-tier DD schedule (1.0/0.75/0.50/0.25)
  Consecutive SL counter resets on any win
  State persisted to disk for same-day crash recovery only
  Removed: weekly/monthly halts, drawdown halt, PAUSED/WARNING states,
           half-lot mode, runaway detection, conviction boost

Direction cooldown (V9):
  ≥ 3 SLs same regime + direction → block that combo for 2 days
  Per-regime tracking (TRENDING_CE, RANGEBOUND_PE, etc.)
  Resets on non-SL exit for that regime+direction

VIX ceiling:     VIX > 35 → no trades (market filter, not CB halt)
Min premium:     ₹80 (avoid illiquid options)
Force exit:      3:10 PM every day (1:30 PM on expiry)

Token lifecycle:  JWT expiry decode → TokenWatcher daemon (5-min checks)
                  Telegram alerts at 30 min before expiry
                  Token failure NEVER stops trading (alert only)
Holiday check:   Upstox API → cache → hardcoded 2026 fallback
Expiry detect:   get_expiry_type() handles old + new NSE schedules

Live fill confirmation:
  wait_for_fill(30s) — position only opened after broker confirms
  Rejected/timeout → ABORT trade + Telegram alert + cancel order
  Paper mode: instant fill (unchanged)

Margin monitoring (live mode only):
  Pre-trade: check available_margin vs required
  Block if insufficient + Telegram alert
  Warn if buffer < 40% (trade proceeds)
  API failure → proceed (no false blocking)

P&L reconciliation (live mode, post-market):
  System vs broker P&L comparison daily
  Trade count cross-check
  Audit trail in reconciliation_log table

WebSocket LTP feed:
  MarketDataStreamerV3 (SDK background thread)
  Auto-reconnect (50 retries, 5s interval)
  Thread-safe cache + REST fallback
  Eliminates REST polling during fast poll

Daily backup:
  rclone → Google Drive (timestamped, 30-day retention)
  DB + models + config + today's log
  --mode backup for manual run

Direction recovery:
  30-min timeout → re-score (max 3/day)
  2 SLs same dir → re-score (max 3/day, shared counter)

Data readiness gate:
  Blocks entries until VIX > 0, NIFTY > 0, candles loaded
  Once open, never resets during session
  Position monitoring continues while gate is closed

NaN safety (V9.3):
  _sg() helper guards all 10 scoring factors against NaN propagation
  Final NaN check on bull_score/bear_score → skip signal if NaN
  Prevents invalid trades from corrupted data

Pre-entry guards (V9.3 FINAL):
  Price contradiction: blocks PE when NIFTY > +0.3% above open + RSI > 55
  Price contradiction: blocks CE when NIFTY < -0.3% below open + RSI < 45
  ML disagreement: blocks entry when ML Stage 1 predicts opposite > 60%
  Active before 11:30, skipped on expiry days (contradiction only)

SIGNAL_SKIP diagnostics:
  Every 10 loops: logs why no trade fired (per-symbol reason codes)
  Reason codes: CONFIRMATION_FAILED, CONVICTION_BELOW_THRESHOLD,
    HARD_ABORT, SOFT_ABORT, RANGE_TOO_TIGHT, AFTER_CUTOFF, etc.
  EOD summary at 15:10: total skips by reason
```


## Known Limitations to Validate in Paper Trading

```
1. EOD exit rate ~60% — TP ladder active in live at 12:00/13:00/14:00
   Validation in progress

2. Trailing stop Tier 1 (+5%) may be too sensitive 9:30-11:00
   May need raising to +7% for opening hour only

3. Stale trade detector (45 min, 3% threshold) — may exit too early
   on slow consolidation before breakout. Track post-exit price action

4. ELEVATED regime has 0 trades in backtest — logic unvalidated
   Monitor VIX 20-28 rising conditions in live

5. Conviction scaling 0.5x minimum confirmed correct
   Tested 0.7x: worse (Max DD +1.73%, PF -0.13, avg loss +₹248)
   Avg win gap vs V8 is an exit problem, not sizing

6. Iron Condor strategy not yet backtested (IC_BACKTEST_ENABLED=false)
   Paper trading will validate IC performance on RANGEBOUND days
   Target: WR > 60%, PF > 2.0 before deploying in live
```


## Stability Fixes (V9.3 FINAL)

```
10 CRITICAL fixes (C1-C10):
  C1:  NaN-safe scoring via _sg() helper (guards all 10 factors)
  C2:  ATR zero-division guard in position sizing
  C3:  Empty candle DataFrame guard before indicator computation
  C4:  Options chain missing-key guard (pcr, max_pain, oi)
  C5:  VIX fetch failure defaults to last known (not 0)
  C6:  Paper trader order_id uniqueness (UUID-based)
  C7:  Portfolio close_position returns None guard
  C8:  Direction lock None-check before string compare
  C9:  WebSocket reconnect backoff (exponential, max 60s)
  C10: SQLite write retry with exponential backoff (3 retries)

14 HIGH fixes (H1-H14):
  H1:  GTT placement failure sends Telegram alert
  H2:  Margin check API timeout (5s) with graceful fallback
  H3:  Token refresh failure alert (not silent swallow)
  H4:  Options instrument cache staleness check (daily refresh)
  H5:  Fast-poll LTP error counter → alert at 20 consecutive failures
  H6:  Regime detection NaN guard (ADX/VIX missing)
  H7:  Final NaN check on bull_score/bear_score → skip signal
  H8:  Order manager spread leg failure → rollback first leg
  H9:  Circuit breaker state file corruption recovery
  H10: Backtest premium_lookup missing strike graceful skip
  H11: Circuit breaker daily reset always clears halt
  H12: EOD squareoff closes portfolio + IC + DB positions
  H13: IC evaluation skipped when nifty_price=0
  H14: Confirmation trigger NaN guard (T1-T4 clipped to 0-1)

4 MEDIUM fixes (M2, M6, M7, M11) — V9.3:
  M2:  save_trade() 3-retry loop on SQLite OperationalError (returns bool)
  M6:  Paper SELL blocked at price=0/None (prevents P&L corruption)
  M7:  Iron Condor division-by-zero guards (spread_width, lot_size, max_loss)
  M11: PCR fallback cache — empty option chain returns last known PCR (not 0)

Additional V9.3 FINAL fixes:
  C-extra: Consecutive error counter + Telegram alert at 10/50/100
  C-extra: Pre-loop candle fetch wrapped in try/except
  C-extra: Trail stop exits now save to DB + close portfolio
  C-extra: Force exit retry loop (3 attempts, 2s delay)
  C-extra: VIX default staleness fix (_vix_last_fetch=0)
  C-extra: Crash recovery restores open positions from DB
  C-extra: Heartbeat Telegram every 30 min (10:00-15:00)
  H-extra: GTT validation + Telegram on failure
  H-extra: IC LTP cache (prevents zero P&L corruption)
  H-extra: Fast-poll error counter + alert at 20 failures
  H-extra: Fresh LTP at force exit time
  H-extra: VIX refresh error counter + alert at 3 failures
  H-extra: EOD squareoff updates portfolio + DB + CB
  H-extra: IC gated on nifty_price > 0
  H-extra: nifty_price=0 warning + Telegram
  H-extra: Trade DB exit records include all price fields (price, fill_price, SL, TP)
  H-extra: Option chain 3-attempt retry with instrument_key logging + noise reduction
  H-extra: Force shutdown os._exit(1) — no threading._shutdown exception
  H-extra: CB state file written on reset_daily() — backup always finds it
  H-extra: sklearn scaler.transform() uses DataFrame (no UserWarning)
  H-extra: LightGBM _load_model() permanently returns False
```


## Current Status: Paper Trading (Month 3)

```
V9.3 FINAL — 258 tests passing.

All systems integrated:
  Core: ML Stage 1 (re-enabled, asymmetric PE=1.5/CE=0.3), fuzzy triggers,
        expiry types, data readiness gate, two-speed poll loop,
        breakout re-entry, partial profit + runner
  Live infra: WebSocket LTP feed, token lifecycle watcher,
              live fill confirmation (wait_for_fill), margin monitoring,
              P&L reconciliation, daily Google Drive backup
  Safety: 10 CRITICAL + 14 HIGH + 4 MEDIUM + 28 extra stability fixes
  Strategy: Iron Condor added for RANGEBOUND regime (paper only)
  CB: Simplified to 2 rules (consecutive SL + daily loss halt)
  Sizing: Equity curve sizing (4-tier DD schedule) +
          CB loss-based reduction (0/1/2+ SL → 1.0/0.75/0.50)
  Robustness: All 3 key thresholds verified ROBUST (15 backtest variations)
  IV awareness filter (VIX ratio threshold adj)
  OI change rate confirmation (30-min snapshots)
  Signal bucket grouping (4 buckets, diagnostics active)
  Trail stop tightened (3-tier from +5%)
  TP ladder checkpoints active (12:00/13:00/14:00)
  Parameter robustness verified (all thresholds ROBUST)
  Price contradiction filter (blocks entry when price vs open disagrees)
  ML disagreement filter (blocks entry when ML predicts opposite >60%)
  Trade DB exit records fully populated (all price fields)
  Force shutdown clean (os._exit, no threading exception)
  ML features 46→51 (external: PCR, VIX, max pain, FII flow)
  Separate PE/CE binary models (CE deployed 58.7%, PE gated)
  Hybrid binary model usage (CE model improves CE signal quality)
  ML disagreement uses binary models for cross-checking

Daily monitoring:
  Share 10:00 log each morning
  Track ML predictions vs rule-only predictions
  Quality model auto-activates at trade #30

Paper trading targets:
  TP ladder validation on live tick data (60% EOD exit rate in backtest)
  ELEVATED regime observation (0 trades in backtest)
  Iron Condor RANGEBOUND performance validation
  Two-speed poll SL/TP latency verification
  ML Stage 1 direction accuracy tracking
  WebSocket LTP feed stability monitoring
  Slippage tracking (signal_price vs fill_price in DB)
```


## 6-Month Roadmap

```
Month 1 ✅: NIFTY paper trading + infrastructure
  Paper trading started ✅
  WebSocket price feed ✅ (MarketDataStreamerV3)
  Automated daily DB backup ✅ (rclone Google Drive)
  Token lifecycle ✅ (TokenWatcher daemon)
  Live fill confirmation ✅ (wait_for_fill)
  P&L reconciliation ✅ (post-market comparison)
  Margin monitoring ✅ (pre-trade check)

Month 2 ✅: Stability + Iron Condor + DD fix
  10 CRITICAL stability fixes ✅ (C1-C10)
  14 HIGH stability fixes ✅ (H1-H14)
  4 MEDIUM stability fixes ✅ (M2, M6, M7, M11)
  Circuit breaker simplified to 2 rules ✅
  Iron Condor strategy added ✅ (paper validation in progress)
  Equity curve sizing ✅ (DD 30.89% → 14.61%)
  CB loss-based size reduction ✅ (0/1/2+ SL → 1.0/0.75/0.50)
  ML Stage 1 re-enabled ✅ (asymmetric PE=1.5, CE=0.3)
  Parameter robustness verified ✅ (all 3 thresholds ROBUST)
  IV awareness filter ✅ (VIX ratio threshold adj)
  OI change rate confirmation ✅ (30-min snapshots)
  Signal bucket grouping ✅ (structure ready, caps at 99)
  Trail stop tightened ✅ (+5%→0.96, +12%→0.94, +25%→0.91)
  TP ladder checkpoints active ✅ (12:00/13:00/14:00)
  258 tests passing ✅

Month 3 (current): Quality model + IC review + ML improvements
  Price contradiction filter ✅ (blocks entry when price disagrees with direction)
  ML disagreement filter ✅ (blocks entry when ML predicts opposite >60%)
  Trade DB exit records fully populated ✅ (all price fields saved correctly)
  Force shutdown clean ✅ (os._exit — no threading exception)
  CB state file on reset ✅ (backup always finds it)
  Option chain retry ✅ (3-attempt with noise reduction)
  sklearn warning eliminated ✅ (DataFrame with column names)
  LightGBM permanently disabled ✅ (_load_model returns False)
  ML backfill incremental fix ✅ (current month fetches missing days)
  ML features 46→51 ✅ (PCR, VIX percentile/change, max pain, FII flow)
  Separate PE/CE binary models ✅ (CE 58.7% deployed, PE gated)
  Hybrid binary model usage ✅ (CE model for CE signals, V1 fallback for PE)
  predict_direction_v2 ✅ (combines PE+CE binary probs)
  ML disagreement uses binary models ✅ (cross-checking with deployed models)
  30+ trade quality model live
  Review ML accuracy vs rule-only
  Iron Condor paper results review (target WR > 60%, PF > 2.0)
  Tune fuzzy trigger gradients from live data
  Analyze slippage_pct from DB (live vs paper)

Month 4: Go-live audit → 1 lot
  End-to-end live audit (all infra already built)
  Start with 1 lot NIFTY only
  Compare live fills vs paper fills
  Review reconciliation_log for P&L accuracy
  IC live deployment (if paper results pass)

Month 5: Scale + VPS
  Scale to full position sizing
  Deploy on VPS (24/7 uptime)
  Add BANKNIFTY instrument support

Month 6: BANKNIFTY live + review
  BANKNIFTY live trading
  IC live at scale (RANGEBOUND days)
  Monthly P&L review → V10 planning
```


## V10 Planned Improvements (Do Not Implement Yet)

```
1. Intraday momentum exit calibrated from live tick data:
   Analyze when EOD-exiting trades typically peak intraday
   Build data-backed exit rule from 4+ weeks of live tick data
   60% EOD exit rate → target 30-35% with validated TP ladder

2. BANKNIFTY support:
   Add instrument master for BANKNIFTY weekly options
   Separate regime detection (BANKNIFTY has different volatility profile)
   Independent position limits per index

3. IC backtest integration:
   Enable IC_BACKTEST_ENABLED once paper results validate strategy
   ATM-based strike selection for backtest (no OI in historical data)
   Separate IC performance row in backtest summary

4. Signal bucket cap tuning:
   Bucket structure ready with caps at 99
   After 30+ paper trades analyze which regime-days had false conviction
   Tune momentum cap from 99 → optimal value
   Data-driven decision from live trade analysis
```
