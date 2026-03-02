═══════════════════════════════════════════════════════════
FULL DAY FLOW — START AT 8:00 AM
═══════════════════════════════════════════════════════════

Command: python src/main.py --mode paper


08:00:00 — SYSTEM STARTS
══════════════════════════
  Load .env → TRADING_STAGE=PLUS
  Load .env.plus → all settings
  Print config:
    Capital: ₹1,50,000
    Deploy Cap: ₹75,000
    Risk/Trade: ₹15,000
    Stage: PLUS
    Max Trades: 4
  Initialize SQLite (9 tables)
  Initialize Paper Trader
  Check: Is today weekday? YES → continue
  Check: Is today NSE holiday? NO → continue


08:00:05 — PHASE 0: AUTO-FETCH (4 seconds)
════════════════════════════════════════════
  [1/8] NIFTY 50 equity candles     → skip if up-to-date
  [2/8] Index data (NIFTY, VIX)     → fetch today's
  [3/8] F&O option premium data     → refresh contracts
  [4/8] (skip)
  [5/8] Local CSV data              → already loaded
  [6/8] FII/DII data                → fetch today if available
  [7/8] External markets (yfinance) → USD, Gold, Crude, SP500
  [8/8] VIX extended history        → update

  Print: "FETCH COMPLETE"
  All data now in SQLite


08:00:10 — PHASE 0b: ML MODEL
══════════════════════════════
  Check model age
  If < 10 days → "Using cached model"
  If > 10 days → retrain (35 features, walk-forward)

  Print: ML prediction P(up)=0.52, P(down)=0.48


08:00:15 — PHASE 1: PRE-MARKET DATA
════════════════════════════════════
  Load FII/DII history (268 days)
  Fetch India VIX live quote → 13.7
  Fetch NIFTY option chain:
    PCR = 0.46
    Max Call OI @ 26000
    Max Put OI @ 25000
    Max Pain @ 25300

  Print: "Pre-market data collection complete"


08:00:20 — BROKER CONNECTION
════════════════════════════
  Paper Trader connected (simulated)
  Capital: ₹1,50,000 ready


08:00:25 — INITIAL REGIME DETECTION
════════════════════════════════════
  VIX = 13.7, ADX = 26.6, BB width = 0.036
  Result: TRENDING
  Print: "REGIME: INIT → TRENDING"
  Settings applied:
    Conviction threshold: 1.75
    SL multiplier: 1.00
    TP multiplier: 1.30
    Trailing stop: ON


08:00:30 — SETUP COMPLETE, WAITING
═══════════════════════════════════
  Print:
  ┌──────────────────────────────────────────────┐
  │ ✅ Setup complete                             │
  │ Market opens: 9:15 AM                        │
  │ Trading starts: 10:00 AM                     │
  │ Current time: 08:00 AM                       │
  │ Waiting... (120 minutes)                     │
  └──────────────────────────────────────────────┘


08:01 — 09:59 — WAITING (doing nothing)
════════════════════════════════════════
  Every 60 seconds:
  Print: "⏳ 08:05 | Trading starts: 10:00 AM | 55 min left"
  Print: "⏳ 08:30 | Trading starts: 10:00 AM | 30 min left"
  Print: "⏳ 09:00 | Trading starts: 10:00 AM | 60 min left"
  Print: "⏳ 09:15 | Market opened | Trading starts: 10:00 AM"
  Print: "⏳ 09:30 | Skipping opening chaos | 30 min left"
  Print: "⏳ 09:45 | Indicators stabilizing | 15 min left"
  Print: "⏳ 09:55 | Almost ready | 5 min left"

  System is IDLE during this time
  No trades, no scanning, just waiting


═══════════════════════════════════════════════════════════
10:00:00 — TRADING LOOP BEGINS
═══════════════════════════════════════════════════════════

  Print: "=== TRADING LOOP STARTED ==="

  Daily counters reset:
    trades_today = 0
    naked_trades_today = 0
    spread_trades_today = 0
    daily_pnl = 0
    direction_locked = False
    flips_today = 0

  Fetch 50+ days NIFTY daily candles
  Compute technicals: EMA 9/21/50, RSI, MACD, BB, ADX


10:00:00 — SCAN #1 (first signal check)
════════════════════════════════════════

  ┌─── STEP 1: Circuit Breaker ───┐
  │ State: NORMAL                  │
  │ Daily loss: ₹0                 │
  │ Consecutive SL: 0             │
  │ Result: ✅ CLEAR               │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 2: Refresh Data ──────┐
  │ NIFTY LTP: 22,550             │
  │ VIX: 13.7                     │
  │ Intraday candles: 3 available │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 3: 9-Factor Scoring ──┐
  │ F1 EMA Trend:    +2.5 bull    │
  │ F2 RSI+MACD:     +2.0 bull    │
  │ F3 Price Action:  +1.5 bull    │
  │ F4 Mean Revert:   0.0         │
  │ F5 BB Position:   +1.0 bull    │
  │ F6 VIX Sentiment: +0.5 bull    │
  │ F7 ML Predict:    +0.3 bull    │
  │ F8 OI/PCR:        +2.0 bear   │
  │ F9 Volume:        +1.0 bull    │
  │                                │
  │ bull_score = 8.8               │
  │ bear_score = 2.0               │
  │ score_diff = 6.8               │
  │ Direction: BULLISH (CE)        │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 4: Conviction Gate ───┐
  │ Regime: TRENDING               │
  │ Threshold: 1.75                │
  │ Score diff: 6.8                │
  │ 6.8 ≥ 1.75 → ✅ PASS          │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 5: Trade Type ────────┐
  │ Stage: PLUS                    │
  │ Regime: TRENDING (not VOLATILE)│
  │ Conviction: 6.8 ≥ 3.0         │
  │ Result: NAKED BUY              │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 6: Strike Selection ──┐
  │ Direction: CE (Call)           │
  │ NIFTY: 22,550                  │
  │ Delta target: 0.45-0.55       │
  │                                │
  │ 22500CE: δ=0.52 ₹125 OI=80K  │
  │ Score: 91 ← BEST              │
  │                                │
  │ Selected: 22500CE @ ₹125      │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 7: Position Sizing ───┐
  │ Deploy: 75000/(125×65) = 9 lots│
  │ Risk: 15000/(125×0.30×65)= 6  │
  │ Lots = min(9,6) = 6 lots      │
  │ Qty = 390                      │
  │ Cost: ₹125 × 390 = ₹48,750   │
  │ Max loss: ₹125×0.30×390=₹14,625│
  │ ✅ within ₹75K deploy cap      │
  │ ✅ within ₹15K risk cap        │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 8: Risk Check ────────┐
  │ Circuit breaker: NORMAL ✅     │
  │ Trades today: 0 < 4 ✅        │
  │ Direction blocked: NO ✅       │
  │ Whipsaw filter: ADX=26.6 ✅   │
  │ Time: 10:00 < 14:30 ✅        │
  │ ALL PASS → EXECUTE             │
  └────────────────────────────────┘
          │
          ▼
  ┌─── STEP 9: EXECUTE ORDER ─────┐
  │ BUY NIFTY 22500CE             │
  │ Qty: 390 (6 lots)             │
  │ Entry: ₹125 (+0.05% slippage) │
  │ Fill: ₹125.06                  │
  │                                │
  │ SL set: ₹87.54 (30% below)    │
  │ TP set: ₹198.22 (58.5% above) │
  │ Trail: activates at ₹135.07   │
  │                                │
  │ trades_today = 1               │
  │ naked_trades_today = 1         │
  │ direction_locked = CE          │
  │                                │
  │ 📱 Telegram: "BUY 22500CE     │
  │   ₹125.06 × 390 | SL ₹87.54  │
  │   TP ₹198.22 | TRENDING"      │
  └────────────────────────────────┘


═══════════════════════════════════════════════════════════
10:00:30 — 15:10 — MONITORING LOOP (every 30 seconds)
═══════════════════════════════════════════════════════════

Every 30 seconds the system does:

  ┌─── CHECK 1: Get Live Premium ──┐
  │ Fetch 22500CE LTP               │
  │ Current: ₹128                   │
  └────────────────────────────────┘
          │
          ▼
  ┌─── CHECK 2: Stop Loss ─────────┐
  │ ₹128 > ₹87.54 (SL)             │
  │ SL not hit → continue           │
  └────────────────────────────────┘
          │
          ▼
  ┌─── CHECK 3: Take Profit ───────┐
  │ ₹128 < ₹198.22 (TP)            │
  │ TP not hit → continue           │
  └────────────────────────────────┘
          │
          ▼
  ┌─── CHECK 4: Trailing Stop ─────┐
  │ ₹128 < ₹135.07 (activate)      │
  │ Trail not active yet → continue │
  └────────────────────────────────┘
          │
          ▼
  │ HOLD. Check again in 30 seconds │
  │ Sleep 30s...                     │


10:01:00 — SCAN #3
  Premium: ₹131 → still below TP, above SL → HOLD

10:01:30 — SCAN #4
  Premium: ₹136 → TRAIL ACTIVATED
  Trail floor: ₹125.06 × 1.05 = ₹131.31
  (locked 5% profit)

10:02:00 — SCAN #5
  Premium: ₹142 → trail floor moves to ₹134.64

10:05:00 — STATUS LOG (every 5 min)
  Print: "STATUS | 22500CE | Entry ₹125.06 | Now ₹142 |
          +13.6% | Trail ₹134.64 | SL ₹87.54 | TP ₹198.22"

  ... continues every 30 seconds ...


═══════════════════════════════════════
SCENARIO A: TP HIT (good trade)
═══════════════════════════════════════

11:23:30 — Premium hits ₹199
  ₹199 ≥ ₹198.22 (TP) → EXIT

  SELL NIFTY 22500CE × 390 @ ₹199
  P&L: (₹199 - ₹125.06) × 390 = +₹28,837

  📱 Telegram: "TP HIT ✅ 22500CE +₹28,837 (+59.1%)"

  trades_today = 1, daily_pnl = +₹28,837
  Position: FLAT

  Can system trade again?
    TRENDING regime: YES (max 4/day in PLUS)
    Must wait 90 min (re-entry cooldown)
    Next eligible: 12:53 PM
    New signal needed (fresh 9-factor scan)


═══════════════════════════════════════
SCENARIO B: SL HIT (bad trade)
═══════════════════════════════════════

10:45:00 — Premium drops to ₹87
  ₹87 ≤ ₹87.54 (SL) → EXIT

  SELL NIFTY 22500CE × 390 @ ₹87
  P&L: (₹87 - ₹125.06) × 390 = -₹14,843

  📱 Telegram: "SL HIT ❌ 22500CE -₹14,843 (-30.4%)"

  consecutive_sl = 1
  daily_pnl = -₹14,843
  Circuit breaker: still NORMAL (-₹14,843 < -₹20,000)

  Can trade again? YES
  Next signal scan continues every 30 seconds


═══════════════════════════════════════
SCENARIO C: TRAIL STOP (lock profit)
═══════════════════════════════════════

12:15:00 — Premium peaked at ₹170, now dropping
  Trail floor at ₹156.40 (locked 22% at +35% level)
  Premium: ₹155 < ₹156.40 → TRAIL EXIT

  SELL @ ₹155
  P&L: (₹155 - ₹125.06) × 390 = +₹11,678

  📱 Telegram: "TRAIL ✅ 22500CE +₹11,678 (+23.9%)"


═══════════════════════════════════════
SCENARIO D: EOD EXIT (time ran out)
═══════════════════════════════════════

  No TP, no SL, no trail triggered all day
  Premium at 3:10 PM: ₹138

14:00:00 — 2:00 PM adjustment
  TP not hit → reduce TP by 20%
  New TP: ₹198.22 × 0.80 = ₹158.58

14:45:00 — 2:45 PM adjustment
  If at loss → tighten SL to 15%
  If at profit → no change (₹138 > ₹125.06, in profit)

15:10:00 — 3:10 PM FORCE EXIT
  SELL @ ₹138 (whatever current price)
  P&L: (₹138 - ₹125.06) × 390 = +₹5,046

  📱 Telegram: "EOD EXIT 22500CE +₹5,046 (+10.3%)"


═══════════════════════════════════════════════════════════
REGIME UPDATES DURING THE DAY
═══════════════════════════════════════════════════════════

  11:00 AM — Regime check #2
    VIX dropped to 12.8, ADX still 26+
    TRENDING → TRENDING (no change)

  13:00 PM — Regime check #3
    VIX spiked to 18, NIFTY dropped 200 pts
    TRENDING → TRENDING (after 1 PM can only upgrade to VOLATILE)

  14:30 PM — Regime check #4
    VIX at 23 → VOLATILE upgrade
    No new trades in VOLATILE (already past 12:00 cutoff)
    Existing position: tighten management


═══════════════════════════════════════════════════════════
POSSIBLE 2nd TRADE (if first exited early)
═══════════════════════════════════════════════════════════

  First trade: TP hit at 11:23 AM
  Cooldown: 90 minutes → eligible at 12:53 PM

  12:53 PM — system scans again
  New 9-factor scores calculated
  If conviction passes → trade again

  Same direction (CE): normal entry
  Opposite direction (PE): needs conviction ≥ 3.0 for flip

  Max 4 trades per day (PLUS)
  Max 2 naked per day
  Max 2 spreads per day


═══════════════════════════════════════════════════════════
15:10:00 — FORCE EXIT ALL
═══════════════════════════════════════════════════════════

  Close every open position at market price
  Naked buys → SELL
  Debit spreads → close both legs
  Credit spreads → close both legs

  Verify: 0 open positions


15:15:00 — SQUARE OFF VERIFICATION
═══════════════════════════════════

  Double-check all orders filled
  If any stuck → retry
  Confirm: FLAT


15:15:30 — POST-MARKET
═══════════════════════

  1. Save portfolio snapshot
     Capital: ₹1,50,000 + today's P&L

  2. Update circuit breaker
     If daily loss > ₹20K → set HALTED for tomorrow
     If 4+ consecutive SL → set PAUSED

  3. Update streak counters
     consecutive_sl = X
     direction_blocked = Y

  4. Save trades to DB
     All fields: entry, exit, type, premium, qty, P&L

  5. Telegram daily report
     ┌──────────────────────────────────┐
     │ 📊 VELTRIX Daily Report          │
     │ ─────────────────────────────── │
     │ Date: 2026-03-02 (Monday)       │
     │ Capital: ₹1,52,340              │
     │ Day P&L: +₹2,340               │
     │ Trades: 1 (1W 0L)              │
     │ Trade: NAKED BUY 22500CE       │
     │ Entry: ₹125.06 → Exit: ₹138   │
     │ Exit reason: EOD                │
     │ Regime: TRENDING                │
     │ VIX: 13.7                       │
     │ Circuit: NORMAL                 │
     │ Streak: 1W                      │
     │ Week P&L: +₹2,340              │
     │ Month P&L: +₹2,340             │
     └──────────────────────────────────┘

  6. Save EOD candle data


15:16:00 — CLEANUP
═══════════════════

  Print: "Cleanup complete"
  System exits
  Terminal returns to prompt


═══════════════════════════════════════════════════════════
TOMORROW: REPEAT
═══════════════════════════════════════════════════════════

  Run same command: python src/main.py --mode paper
  System picks up from new capital (₹1,52,340)
  Fresh day, fresh counters
  All history in SQLite DB
