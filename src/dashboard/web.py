"""
VELTRIX Web Dashboard — Flask-based real-time trading dashboard.

Panels:
1. Equity curve with drawdown
2. Live position monitor
3. Regime and signal panel
4. Performance summary
5. Instrument scan

Run: python src/main.py --mode dashboard
Access: http://localhost:5000
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, render_template_string
from loguru import logger

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "trading_bot.db"

app = Flask(__name__)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        conn = _get_db()
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        logger.warning(f"Dashboard DB query failed: {e}")
        return pd.DataFrame()


def _query_rows(sql: str, params: tuple = ()) -> list[dict]:
    try:
        conn = _get_db()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Dashboard DB query failed: {e}")
        return []


# ──────────────────────────────────────────────
# Data fetchers
# ──────────────────────────────────────────────

def _get_equity_curve() -> list[dict]:
    """Portfolio snapshots for equity curve chart."""
    rows = _query_rows(
        "SELECT datetime, total_value, cash, invested, drawdown_pct, day_pnl "
        "FROM portfolio_snapshots ORDER BY datetime DESC LIMIT 3000"
    )
    rows.reverse()
    return rows


def _get_open_positions() -> list[dict]:
    """Currently open trades."""
    return _query_rows(
        "SELECT * FROM trades WHERE status = 'filled' ORDER BY entry_time"
    )


def _get_open_ic_positions() -> list[dict]:
    """Open iron condor positions."""
    return _query_rows(
        "SELECT * FROM ic_trades WHERE status = 'open' ORDER BY entry_time"
    )


def _get_latest_regime() -> dict:
    """Most recent regime detection."""
    rows = _query_rows(
        "SELECT * FROM regime_history ORDER BY datetime DESC LIMIT 1"
    )
    return rows[0] if rows else {}


def _get_latest_signal() -> dict:
    """Most recent signal."""
    rows = _query_rows(
        "SELECT * FROM signals ORDER BY datetime DESC LIMIT 1"
    )
    return rows[0] if rows else {}


def _get_performance() -> dict[str, Any]:
    """Calculate performance summary from completed trades."""
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    all_trades = _query_df(
        "SELECT pnl, entry_time, exit_time, price, quantity, stop_loss "
        "FROM trades WHERE status = 'exited' AND pnl != 0 AND mode = 'paper' "
        "ORDER BY exit_time"
    )

    if all_trades.empty:
        return {
            "today_pnl": 0, "week_pnl": 0, "month_pnl": 0,
            "total_trades": 0, "win_rate": 0, "pf": 0,
            "expectancy": 0, "max_dd": 0,
        }

    all_trades["exit_date"] = pd.to_datetime(all_trades["exit_time"]).dt.date.astype(str)
    today_trades = all_trades[all_trades["exit_date"] == today]
    week_trades = all_trades[all_trades["exit_date"] >= week_ago]
    month_trades = all_trades[all_trades["exit_date"] >= month_ago]

    total = len(all_trades)
    wins = all_trades[all_trades["pnl"] > 0]
    losses = all_trades[all_trades["pnl"] <= 0]
    wr = len(wins) / total * 100 if total > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
    pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else 0

    wr_dec = len(wins) / total if total > 0 else 0
    expectancy = (wr_dec * avg_win) - ((1 - wr_dec) * avg_loss)

    # Max drawdown from portfolio snapshots
    snap = _query_df(
        "SELECT drawdown_pct FROM portfolio_snapshots WHERE mode = 'paper' "
        "ORDER BY datetime DESC LIMIT 500"
    )
    max_dd = abs(snap["drawdown_pct"].min()) if not snap.empty else 0

    return {
        "today_pnl": today_trades["pnl"].sum() if not today_trades.empty else 0,
        "week_pnl": week_trades["pnl"].sum() if not week_trades.empty else 0,
        "month_pnl": month_trades["pnl"].sum() if not month_trades.empty else 0,
        "total_trades": total,
        "win_rate": wr,
        "pf": pf,
        "expectancy": expectancy,
        "max_dd": max_dd,
    }


def _get_instruments() -> list[dict]:
    """Latest daily log for each instrument."""
    today = date.today().isoformat()
    rows = _query_rows(
        "SELECT instrument, regime, direction, score_diff, would_trade, blocking_reason "
        "FROM instrument_daily_log WHERE date = ? "
        "ORDER BY instrument",
        (today,),
    )
    if not rows:
        # Fall back to most recent date
        rows = _query_rows(
            "SELECT instrument, regime, direction, score_diff, would_trade, blocking_reason "
            "FROM instrument_daily_log WHERE date = ("
            "  SELECT MAX(date) FROM instrument_daily_log"
            ") ORDER BY instrument"
        )
    return rows


# ──────────────────────────────────────────────
# Template
# ──────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>VELTRIX Dashboard</title>
  <meta http-equiv="refresh" content="30">
  <meta charset="utf-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Consolas', 'Courier New', monospace;
      background: #0a0a12;
      color: #e0e0e0;
      padding: 12px 18px;
      font-size: 13px;
    }
    h1 {
      color: #7aa2f7;
      font-size: 18px;
      margin-bottom: 4px;
    }
    .subtitle { color: #565f89; font-size: 11px; margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .panel {
      background: #13131f;
      border: 1px solid #1e1e30;
      border-radius: 6px;
      padding: 12px 14px;
    }
    .panel-title {
      color: #7aa2f7;
      font-size: 12px;
      font-weight: bold;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .full { grid-column: 1 / -1; }
    .green { color: #9ece6a; }
    .red { color: #f7768e; }
    .yellow { color: #e0af68; }
    .blue { color: #7aa2f7; }
    .dim { color: #565f89; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      text-align: left;
      color: #565f89;
      font-weight: normal;
      padding: 3px 6px;
      border-bottom: 1px solid #1e1e30;
      font-size: 11px;
      text-transform: uppercase;
    }
    td { padding: 4px 6px; border-bottom: 1px solid #0e0e1a; }
    .stat-row { display: flex; justify-content: space-between; padding: 3px 0; }
    .stat-label { color: #565f89; }
    .stat-value { text-align: right; }
    .pos-card {
      background: #1a1a2e;
      border: 1px solid #2a2a40;
      border-radius: 4px;
      padding: 10px;
      margin-bottom: 8px;
    }
    .pos-title { font-weight: bold; color: #bb9af7; margin-bottom: 4px; }
    svg { display: block; }
    .chart-container { width: 100%; overflow: hidden; }
  </style>
</head>
<body>

<h1>VELTRIX</h1>
<div class="subtitle">Dashboard &mdash; {{ now }} &mdash; auto-refresh 30s</div>

<!-- Row 1: Equity Curve (full width) -->
<div class="panel" style="margin-bottom:12px">
  <div class="panel-title">Equity Curve</div>
  <div class="chart-container">
  {% if equity_points|length > 1 %}
    {% set vals = equity_points|map(attribute='total_value')|list %}
    {% set min_v = vals|min %}
    {% set max_v = vals|max %}
    {% set range_v = max_v - min_v if max_v != min_v else 1 %}
    {% set w = 960 %}
    {% set h = 160 %}
    {% set n = vals|length %}
    <svg viewBox="0 0 {{ w }} {{ h + 20 }}" width="100%" height="180">
      <!-- Grid lines -->
      <line x1="0" y1="{{ h }}" x2="{{ w }}" y2="{{ h }}" stroke="#1e1e30" stroke-width="1"/>
      <line x1="0" y1="{{ h // 2 }}" x2="{{ w }}" y2="{{ h // 2 }}" stroke="#1e1e30" stroke-width="0.5"/>
      <line x1="0" y1="0" x2="{{ w }}" y2="0" stroke="#1e1e30" stroke-width="0.5"/>
      <!-- Peak line -->
      {% set peak_vals = [] %}
      {% set ns = namespace(peak=0) %}
      {% for v in vals %}
        {% if v > ns.peak %}{% set ns.peak = v %}{% endif %}
        {% set _ = peak_vals.append(ns.peak) %}
      {% endfor %}
      <!-- Drawdown shading -->
      <path d="
        {% for i in range(n) %}
          {% set x = (i / (n - 1) * w)|round(1) %}
          {% set y_peak = (h - ((peak_vals[i] - min_v) / range_v * h))|round(1) %}
          {% set y_val = (h - ((vals[i] - min_v) / range_v * h))|round(1) %}
          {% if i == 0 %}M {{ x }} {{ y_peak }}{% else %}L {{ x }} {{ y_peak }}{% endif %}
        {% endfor %}
        {% for i in range(n - 1, -1, -1) %}
          {% set x = (i / (n - 1) * w)|round(1) %}
          {% set y_val = (h - ((vals[i] - min_v) / range_v * h))|round(1) %}
          L {{ x }} {{ y_val }}
        {% endfor %}
        Z" fill="#f7768e" fill-opacity="0.12"/>
      <!-- Equity line -->
      <polyline fill="none" stroke="#7aa2f7" stroke-width="1.5" points="
        {% for i in range(n) %}
          {{ (i / (n - 1) * w)|round(1) }},{{ (h - ((vals[i] - min_v) / range_v * h))|round(1) }}
        {% endfor %}
      "/>
      <!-- Labels -->
      <text x="4" y="{{ h + 14 }}" fill="#565f89" font-size="10">{{ equity_points[0].datetime[:10] }}</text>
      <text x="{{ w - 4 }}" y="{{ h + 14 }}" fill="#565f89" font-size="10" text-anchor="end">{{ equity_points[-1].datetime[:10] }}</text>
      <text x="4" y="10" fill="#565f89" font-size="10">&nbsp;{{ "₹{:,.0f}".format(max_v) }}</text>
      <text x="4" y="{{ h - 2 }}" fill="#565f89" font-size="10">&nbsp;{{ "₹{:,.0f}".format(min_v) }}</text>
    </svg>
  {% else %}
    <div class="dim" style="padding:30px 0;text-align:center">No portfolio data yet</div>
  {% endif %}
  </div>
</div>

<!-- Row 2: Open Positions + Market Status -->
<div class="grid">
  <!-- Panel 2: Open Positions -->
  <div class="panel">
    <div class="panel-title">Open Positions</div>
    {% if positions %}
      {% for p in positions %}
      <div class="pos-card">
        <div class="pos-title">{{ p.symbol }} &nbsp; {{ p.quantity }} qty</div>
        <div class="stat-row">
          <span class="stat-label">Entry</span>
          <span>₹{{ "%.2f"|format(p.price) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">SL / TP</span>
          <span>₹{{ "%.2f"|format(p.stop_loss) }} / ₹{{ "%.2f"|format(p.take_profit) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">P&L</span>
          <span class="{{ 'green' if p.pnl >= 0 else 'red' }}">
            {{ "+₹%.0f"|format(p.pnl) if p.pnl >= 0 else "₹%.0f"|format(p.pnl) }}
          </span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Since</span>
          <span class="dim">{{ p.entry_time[:16] if p.entry_time else '—' }}</span>
        </div>
      </div>
      {% endfor %}
    {% elif ic_positions %}
      {% for ic in ic_positions %}
      <div class="pos-card">
        <div class="pos-title">IRON CONDOR &nbsp; {{ ic.lots }}L × {{ ic.quantity }}q</div>
        <div class="stat-row">
          <span class="stat-label">Range</span>
          <span>{{ "%.0f"|format(ic.sell_pe_strike) }} – {{ "%.0f"|format(ic.sell_ce_strike) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Net credit</span>
          <span>₹{{ "%.0f"|format(ic.net_credit) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">P&L</span>
          <span class="{{ 'green' if ic.pnl >= 0 else 'red' }}">₹{{ "%.0f"|format(ic.pnl) }}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Since</span>
          <span class="dim">{{ ic.entry_time[:16] if ic.entry_time else '—' }}</span>
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="dim" style="padding:20px 0;text-align:center">No open positions</div>
    {% endif %}
  </div>

  <!-- Panel 3: Market Status -->
  <div class="panel">
    <div class="panel-title">Market Status</div>
    {% if regime %}
    <div class="stat-row">
      <span class="stat-label">Regime</span>
      <span class="yellow">{{ regime.regime or '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">VIX</span>
      <span>{{ "%.1f"|format(regime.vix_value) if regime.vix_value else '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">NIFTY</span>
      <span>{{ "{:,.0f}".format(regime.nifty_value) if regime.nifty_value else '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">ADX</span>
      <span>{{ "%.1f"|format(regime.adx_value) if regime.adx_value else '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">FII Net</span>
      <span>{{ "₹{:,.0f}cr".format(regime.fii_net_value) if regime.fii_net_value else '—' }}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Updated</span>
      <span class="dim">{{ regime.datetime[:16] if regime.datetime else '—' }}</span>
    </div>
    {% else %}
      <div class="dim" style="padding:20px 0;text-align:center">No regime data</div>
    {% endif %}

    {% if signal %}
    <div style="margin-top:10px;border-top:1px solid #1e1e30;padding-top:8px">
      <div class="stat-row">
        <span class="stat-label">Direction</span>
        <span class="{{ 'green' if signal.direction == 'UP' else 'red' }}">{{ signal.direction or '—' }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Score</span>
        <span>{{ "%.1f"|format(signal.score) if signal.score else '—' }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Confidence</span>
        <span>{{ "%.0f%%"|format(signal.confidence * 100) if signal.confidence else '—' }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Strategy</span>
        <span class="dim">{{ signal.strategy or '—' }}</span>
      </div>
    </div>
    {% endif %}
  </div>
</div>

<!-- Row 3: Performance + Instruments -->
<div class="grid">
  <!-- Panel 4: Performance -->
  <div class="panel">
    <div class="panel-title">Performance</div>
    <div class="stat-row">
      <span class="stat-label">Today</span>
      <span class="{{ 'green' if perf.today_pnl >= 0 else 'red' }}">
        {{ "+₹{:,.0f}".format(perf.today_pnl) if perf.today_pnl >= 0 else "₹{:,.0f}".format(perf.today_pnl) }}
      </span>
    </div>
    <div class="stat-row">
      <span class="stat-label">This week</span>
      <span class="{{ 'green' if perf.week_pnl >= 0 else 'red' }}">
        {{ "+₹{:,.0f}".format(perf.week_pnl) if perf.week_pnl >= 0 else "₹{:,.0f}".format(perf.week_pnl) }}
      </span>
    </div>
    <div class="stat-row">
      <span class="stat-label">This month</span>
      <span class="{{ 'green' if perf.month_pnl >= 0 else 'red' }}">
        {{ "+₹{:,.0f}".format(perf.month_pnl) if perf.month_pnl >= 0 else "₹{:,.0f}".format(perf.month_pnl) }}
      </span>
    </div>
    <div style="margin-top:8px;border-top:1px solid #1e1e30;padding-top:8px">
      <div class="stat-row">
        <span class="stat-label">Total trades</span>
        <span>{{ perf.total_trades }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Win Rate</span>
        <span>{{ "%.1f%%"|format(perf.win_rate) }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Profit Factor</span>
        <span>{{ "%.2f"|format(perf.pf) }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Expectancy</span>
        <span class="{{ 'green' if perf.expectancy > 0 else 'red' }}">
          ₹{{ "{:,.0f}".format(perf.expectancy) }}/trade
        </span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Max DD</span>
        <span class="red">{{ "%.1f%%"|format(perf.max_dd) }}</span>
      </div>
    </div>
  </div>

  <!-- Panel 5: Instrument Scan -->
  <div class="panel">
    <div class="panel-title">Instrument Scan</div>
    {% if instruments %}
    <table>
      <tr>
        <th>Instrument</th>
        <th>Regime</th>
        <th>Dir</th>
        <th>Score</th>
        <th>Status</th>
      </tr>
      {% for inst in instruments %}
      <tr>
        <td>{{ inst.instrument }}</td>
        <td class="yellow">{{ inst.regime or '—' }}</td>
        <td class="{{ 'green' if inst.direction == 'CE' else 'red' if inst.direction == 'PE' else 'dim' }}">
          {{ inst.direction or '—' }}
        </td>
        <td>{{ "%.1f"|format(inst.score_diff) if inst.score_diff else '—' }}</td>
        <td>
          {% if inst.would_trade %}
            <span class="green">TRADING</span>
          {% else %}
            <span class="dim">LOGGING</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
      <div class="dim" style="padding:20px 0;text-align:center">No instrument data</div>
    {% endif %}
  </div>
</div>

<!-- Row 4: Recent Trades -->
<div class="panel">
  <div class="panel-title">Recent Trades</div>
  {% if recent_trades %}
  <table>
    <tr>
      <th>Date</th>
      <th>Symbol</th>
      <th>Side</th>
      <th>Qty</th>
      <th>Entry</th>
      <th>P&L</th>
      <th>Regime</th>
      <th>Exit</th>
    </tr>
    {% for t in recent_trades %}
    <tr>
      <td class="dim">{{ t.entry_time[:10] if t.entry_time else t.created_at[:10] }}</td>
      <td>{{ t.symbol }}</td>
      <td class="{{ 'green' if t.side == 'BUY' else 'red' }}">{{ t.side }}</td>
      <td>{{ t.quantity }}</td>
      <td>₹{{ "%.2f"|format(t.price) }}</td>
      <td class="{{ 'green' if t.pnl > 0 else 'red' if t.pnl < 0 else 'dim' }}">
        {{ "+₹{:,.0f}".format(t.pnl) if t.pnl > 0 else "₹{:,.0f}".format(t.pnl) if t.pnl < 0 else '—' }}
      </td>
      <td class="dim">{{ t.regime or '—' }}</td>
      <td class="dim">{{ (t.notes or '')[:15] }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
    <div class="dim" style="padding:20px 0;text-align:center">No trades yet</div>
  {% endif %}
</div>

</body>
</html>"""


# ──────────────────────────────────────────────
# Route
# ──────────────────────────────────────────────

@app.route("/")
def index():
    equity_points = _get_equity_curve()
    positions = _get_open_positions()
    ic_positions = _get_open_ic_positions()
    regime = _get_latest_regime()
    signal = _get_latest_signal()
    perf = _get_performance()
    instruments = _get_instruments()

    recent_trades = _query_rows(
        "SELECT * FROM trades WHERE mode = 'paper' AND status = 'exited' "
        "ORDER BY exit_time DESC LIMIT 20"
    )

    return render_template_string(
        DASHBOARD_HTML,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        equity_points=equity_points,
        positions=positions,
        ic_positions=ic_positions,
        regime=regime,
        signal=signal,
        perf=perf,
        instruments=instruments,
        recent_trades=recent_trades,
    )


def run_dashboard(host: str = "0.0.0.0", port: int = 5000):
    """Start the Flask dashboard server."""
    logger.info(f"Starting VELTRIX dashboard at http://localhost:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_dashboard()
