# Veltrix V8 — Current Workflow

## Step 1: Data Collection (auto-fetch on startup)

```
Runs automatically when paper/live mode starts (no separate fetch needed).
Also available standalone: python src/main.py --mode fetch

1. NIFTY 50 daily candles (3 years / 1095 days via Upstox API)
2. India VIX history (5 years / 1825 days via nsepython)
3. FII/DII data (CSV bulk download + daily nsepython API)
4. External markets via yfinance (5 years):
   S&P 500, NASDAQ, Crude Oil, Gold, USD/INR
5. Option chain: current contracts + premiums (Upstox)
6. All stored in SQLite database (data/trading_bot.db)
7. Incremental: only fetches new data since last run

Startup flow for paper/live:
  Auto-fetch (incremental) → ML training → Pre-market → Trading loop
```

## Step 2: Feature Engineering (34 features)

```
From NIFTY candles (11):
  - rsi_14 (overbought/oversold)
  - macd_histogram (momentum)
  - bb_position (Bollinger Band position)
  - atr_pct (volatility as % of price)
  - adx_14 (trend strength)
  - volume_ratio (vs 20d average)
  - volatility_20d (20-day rolling)
  - returns_1d, returns_5d (momentum)
  - price_to_sma50 (trend position)
  - mfi_14 (money flow)

From FII/DII (7):
  - fii_net_flow_1d, fii_net_flow_5d
  - fii_flow_momentum
  - fii_net_direction, fii_net_streak
  - dii_net_flow_1d
  - india_vix

From VIX extended (3):
  - vix_change_pct
  - vix_percentile_252d (1-year percentile)
  - vix_5d_ma

From External Markets (9):
  - sp500_prev_return, nasdaq_prev_return
  - crude_prev_return, gold_prev_return
  - usdinr_prev_return
  - sp500_nifty_corr_20d, crude_nifty_corr_20d
  - dxy_momentum_5d
  - global_risk_score (composite)

From Options (4):
  - pcr_ratio, max_pain_distance
  - delivery_pct, futures_premium_pct

Note: FII/DII, external, and options features default to 0.0
when data is unavailable (graceful degradation).
```

## Step 3: Regime Detection

```
Every trading day, classify market as:

VOLATILE (checked first — risk-off):
  - VIX ≥ 30 → always VOLATILE
  - Score-based: VIX > 22 (+2), VIX 5d spike > 3 (+2),
    5d range > 4% (+1) — total ≥ 2 triggers VOLATILE
  - Conviction threshold: 2.5
  - Rarely traded (1 trade in backtest, 0% WR)

TRENDING (checked second):
  - ADX > 25 (primary), or ADX > 20 with positive slope
    and BB width > 4% (secondary)
  - Conviction threshold: 1.75
  - Most trades happen here (236 of 301)
  - WR: 57.2%

RANGEBOUND (everything else):
  - ADX ≤ 20, no volatility spike
  - Conviction threshold: 2.0
  - 64 trades, WR: 76.6% (best regime)
```

## Step 4: Multi-Factor Scoring

```
Backtest uses 8 factors, live adds a 9th (OI/PCR).

Factor 1 — Trend Alignment (regime-driven weight: 0.5-2.5):
  EMA stack: EMA_9 vs EMA_20 vs EMA_50
  → Aligned up = bull, aligned down = bear
  ADX > 30: ±0.5 confirmation
  5-day direction: ±0.3 nudge
  Score range: -2.5 to +2.5

Factor 2 — Momentum (weight: 2.0):
  RSI > 58 and rising → bull +1.0
  RSI < 42 and falling → bear +1.0
  MACD histogram expanding → ±1.0
  Score range: 0 to +2.0

Factor 3 — Price Action (weight: 1.5):
  Gap > 0.4% → ±0.75
  Breakout beyond prev high/low → ±0.75
  Candle body direction → ±0.3
  Score range: -0.75 to +1.5

Factor 4 — Mean Reversion Guard (regime-driven: 1.0-2.5):
  5d return > 5% → extreme overbought (strong bear signal)
  5d return > 3.5% → overbought (bear signal + bull penalty)
  5d return < -5% → extreme oversold (strong bull signal)
  5d return < -3.5% → oversold (bull signal + bear penalty)
  Score range: -2.5 to +2.5

Factor 5 — Bollinger Position (weight: 1.0):
  BB position > 0.85 → bull +0.5
  BB position < 0.15 → bear +0.5
  BB width expanding > 20% → ±0.25 (confirms direction)
  Score range: -0.5 to +0.75

Factor 6 — VIX Direction (weight: 0.5):
  VIX < 13 → bull +0.5 (complacency, trends persist)
  VIX > 20 → bear +0.5 (fear, downside risk)
  VIX falling from >20 → bull +0.3
  VIX rising → bear +0.3
  Score range: -0.5 to +0.8

Factor 7 — ML Ensemble (auto-governed weight: 0-1.5):
  3-model consensus with auto-governance multiplier
  High confidence (>0.65, all agree) → ±1.5 × ml_weight
  Medium confidence (>0.58) → ±1.0 × ml_weight
  Low confidence (>0.52) → ±0.3 × ml_weight
  Score range: -1.5 to +1.5 (when ml_weight = 1.0)

Factor 8 — OI/PCR Consensus (LIVE ONLY, weight: 2.0):
  PCR ≥ 1.3 → bull +1.0 | PCR ≤ 0.7 → bear +1.0
  OI support/resistance proximity → ±0.5 to ±1.0
  Not available in backtest (no OI data)
  Score range: -1.0 to +2.0

Factor 9 — Volume Confirmation (weight: 1.0):
  Volume > 1.3× 20d avg → confirms direction ±1.0
  Volume < 0.7× 20d avg → weakens direction ∓0.3
  Score range: -0.3 to +1.0

Bonus — Consecutive SL Nudge:
  3+ SLs in same direction → nudge opposite ±0.5
```

## Step 5: Signal Generation

```
bull_score = sum of all bullish votes
bear_score = sum of all bearish votes
score_diff = |bull_score - bear_score|

If bull_score > bear_score AND score_diff >= threshold:
  → SIGNAL: BUY CALL (CE)

If bear_score > bull_score AND score_diff >= threshold:
  → SIGNAL: BUY PUT (PE)

If score_diff < threshold:
  → NO TRADE (conviction too low)

Threshold varies by regime:
  TRENDING: 1.75
  RANGEBOUND: 2.0
  VOLATILE: 2.5
  + Monday penalty (+0.3)
  + Afternoon penalty (+0.5 after 1 PM, live only)
```

## Step 6: Filters (Skip Bad Trades)

```
Before executing, check:

1. Whipsaw filter:
   ADX < 20 but regime says TRENDING → SKIP
   Daily range < 0.5% → SKIP (no volatility)

2. Consecutive SL check:
   3+ consecutive SLs in same direction → BLOCK that direction
   (auto-resets after 5-day cooldown)

3. VIX check:
   VIX > 35 → SKIP all trades

4. Expiry day (Tuesday):
   Allowed but force exit by ~1:30 PM (theta penalty)
   Wider SL (+5%), lower TP (×0.65), max 1 trade

5. Time check (live only):
   No trades before 10:00 AM
   Regime-based last-trade cutoff:
     TRENDING: 2:30 PM (conservative) / 2:45 PM (active)
     RANGEBOUND: 1:00 PM / 2:00 PM
     VOLATILE: 12:00 PM / 1:00 PM
```

## Step 7: Option Selection

```
Once signal confirmed (CE or PE):

1. Find ATM strike (nearest to NIFTY spot)
2. Select current week expiry (next week's on expiry day)
3. Get premium price
4. Filter: premium must be ≥ ₹80

5. Calculate lots (dynamic, no hard max):
   lots_by_deploy = floor(₹25,000 / (premium × lot_size))
   lots_by_risk = floor(risk_limit / (premium × SL% × lot_size))
   lots = min(lots_by_deploy, lots_by_risk)
   lots = max(1, lots)

   Example at different premiums:
     ₹80 premium → 4 lots (260 qty)
     ₹150 premium → 2 lots (130 qty)
     ₹250 premium → 1 lot (65 qty)
     ₹400 premium → 1 lot (65 qty, near deploy cap)

   Max premium: dynamic (≈₹385 at 65 lot size from deploy cap)
```

## Step 8: ML Ensemble (Advisory Only)

```
3 models: LightGBM + XGBoost + CatBoost
Input: 34 features (lagged 1 day to prevent look-ahead bias)
Output: probability of UP day (average of 3 models)
Training: 120-day walk-forward, retrained every 5 days

Auto-governance (rolling 50-trade accuracy):
  > 60% accuracy → weight 1.0
  55-60% → weight 0.5
  50-55% → weight 0.3
  ≤ 50% → weight 0.0 (disabled)

Currently ~48-52% accuracy → weight 0.0-0.3
ML agrees with signal → slight conviction boost
ML disagrees → slight conviction reduction
Does NOT override multi-factor scoring
```

## Step 9: Order Execution

```
Place order on Upstox:
  Instrument: NIFTY CE or PE
  Qty: 65-260+ (1-4+ lots, dynamic)
  Order type: MARKET
  Position cost: max ₹25,000

Set exit levels (VIX-adaptive):
  Stop Loss by VIX:
    VIX < 13  → SL 25%
    VIX 13-18 → SL 30%
    VIX 18-22 → SL 30%
    VIX 22-28 → SL 25%
    VIX 28-35 → SL 20%
    VIX > 35  → no trades (skipped)

  Take Profit by VIX:
    VIX < 13  → TP 40%
    VIX 13-18 → TP 45%
    VIX 18-22 → TP 55%
    VIX 22-28 → TP 60%
    VIX 28-35 → TP 45%

  Regime adjustments:
    TRENDING: TP ×1.3 (let winners run)
    RANGEBOUND: SL ×0.85, TP ×0.70 (quick in/out)
    VOLATILE: SL ×1.20, TP ×1.50 (wide room)

  Trailing Stop: activates at +8% gain
    Locks in profit, follows price up

  Force Exit: 3:10 PM (1:30 PM on expiry)
```

## Step 10: Position Monitoring (Live)

```
While position is open:
  Check price every 30 seconds

  If price hits SL → EXIT (market order)
  If price hits TP → EXIT
  If trailing stop triggered → EXIT
  If 3:10 PM → FORCE EXIT
  If daily loss > ₹10,000 → HALT all trading

Re-entry (live only, depends on trading mode):
  Conservative mode: max 2 trades/day
  Active mode: max 5 trades/day, 15-min cooldown
  Backtest: 1 trade/day (no intraday re-entry)
```

## Step 11: End of Day

```
3:10 PM: Force exit all positions
Log daily P&L
Update database
Update ML rolling accuracy
Print daily summary:
  Trades taken, wins, losses
  P&L for the day
  Running total
```

## Safety Systems (Always Active)

```
Deploy cap:      ₹25,000 max per position
Risk cap:        ₹10,000 max loss per trade
Daily halt:      ₹10,000 total daily loss → stop
Consecutive SL:  3 SLs same direction → block (5-day cooldown)
VIX ceiling:     VIX > 35 → no trades
Min premium:     ₹80 (avoid illiquid options)
Max premium:     Dynamic (~₹385 from deploy cap / lot size)
Force exit:      3:10 PM every day (1:30 PM on expiry)
Slippage:        0.05% applied to all orders
Charges:         Brokerage + STT + GST calculated per trade
```

## Backtest Results (V8 Final)

```
Period:          2023-05 to 2026-02 (693 trading days)
Capital:         ₹50,000
Return:          1,445% (₹50K → ₹7.7 lakh)
Trades:          301
Win Rate:        61.1%
Profit Factor:   2.64
Sharpe:          2.778
Max Drawdown:    14.19%
Profitable Months: 31/34 (91%)

By Regime:
  TRENDING:    236 trades, 57.2% WR, +₹578K
  RANGEBOUND:   64 trades, 76.6% WR, +₹153K
  VOLATILE:      1 trade,  0.0% WR, -₹8.6K
```
