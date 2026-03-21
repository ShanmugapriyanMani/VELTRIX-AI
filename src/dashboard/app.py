"""
Streamlit Dashboard — Live P&L, regime monitor, strategy performance, trade log.

Dark theme, auto-refresh 30s.
Run: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.data.store import DataStore
from src.config.env_loader import EnvConfig


def main():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="VELTRIX",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme CSS
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #16213e;
    }
    .profit { color: #00ff88; }
    .loss { color: #ff4444; }
    </style>
    """, unsafe_allow_html=True)

    st.title("VELTRIX Dashboard")

    # Auto-refresh
    refresh = st.sidebar.selectbox("Auto-refresh", ["Off", "10s", "30s", "60s"], index=2)
    if refresh != "Off":
        secs = int(refresh.replace("s", ""))
        st.markdown(
            f'<meta http-equiv="refresh" content="{secs}">',
            unsafe_allow_html=True,
        )

    # Load data
    store = DataStore()
    config = EnvConfig()
    initial_capital = config.TRADING_CAPITAL

    # Sidebar
    st.sidebar.header("Controls")
    mode = st.sidebar.radio("Mode", ["Live", "Paper", "Backtest"])
    st.sidebar.metric("Mode", mode)

    # ── Tab Layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio", "Regime", "Strategies", "Trades", "Risk"
    ])

    # ══════════════════════════════════════
    # TAB 1: Portfolio Overview
    # ══════════════════════════════════════
    with tab1:
        st.header("Portfolio Overview")

        portfolio = store.get_portfolio_history(days=30)
        trades = store.get_today_trades()

        col1, col2, col3, col4 = st.columns(4)

        if not portfolio.empty:
            latest = portfolio.iloc[-1]
            total_value = latest.get("total_value", initial_capital)
            day_pnl = latest.get("day_pnl", 0)
            total_pnl = total_value - initial_capital
            drawdown = latest.get("drawdown_pct", 0)

            col1.metric("Portfolio Value", f"₹{total_value:,.0f}", f"₹{total_pnl:+,.0f}")
            col2.metric("Day P&L", f"₹{day_pnl:+,.0f}",
                        delta_color="normal" if day_pnl >= 0 else "inverse")
            col3.metric("Positions", int(latest.get("positions_count", 0)))
            col4.metric("Drawdown", f"{drawdown:.1f}%",
                        delta_color="inverse")
        else:
            col1.metric("Portfolio Value", f"₹{initial_capital:,.0f}")
            col2.metric("Day P&L", "₹0")
            col3.metric("Positions", "0")
            col4.metric("Drawdown", "0%")

        # Equity curve
        if not portfolio.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio["datetime"],
                y=portfolio["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#00ff88", width=2),
            ))
            fig.update_layout(
                title="Equity Curve",
                template="plotly_dark",
                height=400,
                xaxis_title="Date",
                yaxis_title="Value (₹)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Today's trades
        st.subheader("Today's Trades")
        if not trades.empty:
            st.dataframe(
                trades[["symbol", "side", "quantity", "price", "fill_price",
                         "strategy", "pnl", "status"]],
                use_container_width=True,
            )
        else:
            st.info("No trades today")

    # ══════════════════════════════════════
    # TAB 2: Regime Monitor
    # ══════════════════════════════════════
    with tab2:
        st.header("Market Regime Monitor")

        # Latest regime from DB
        with store._get_connection() as conn:
            regime_df = pd.read_sql_query(
                "SELECT * FROM regime_history ORDER BY datetime DESC LIMIT 50",
                conn,
            )

        if not regime_df.empty:
            latest_regime = regime_df.iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Regime", latest_regime.get("regime", "UNKNOWN"))
            col2.metric("India VIX", f"{latest_regime.get('vix_value', 0):.1f}")
            col3.metric("NIFTY", f"{latest_regime.get('nifty_value', 0):,.0f}")
            col4.metric("ADX", f"{latest_regime.get('adx_value', 0):.1f}")

            # Active strategies
            active = json.loads(latest_regime.get("active_strategies", "[]"))
            st.info(f"Active Strategies: {', '.join(active) if active else 'None'}")
            st.metric("Size Multiplier", f"{latest_regime.get('size_multiplier', 1.0):.2f}x")

            # Regime history chart
            if len(regime_df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=regime_df["datetime"],
                    y=regime_df["vix_value"],
                    name="India VIX",
                    line=dict(color="#ff6644"),
                ))
                fig.update_layout(
                    title="VIX History",
                    template="plotly_dark",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regime data available. Start the bot to see regime detection.")

    # ══════════════════════════════════════
    # TAB 3: Strategy Performance
    # ══════════════════════════════════════
    with tab3:
        st.header("Strategy Performance")

        with store._get_connection() as conn:
            signals_df = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY datetime DESC LIMIT 200",
                conn,
            )

        all_trades = store.get_trades(limit=500)

        if not all_trades.empty:
            # Per-strategy breakdown
            for strategy in all_trades["strategy"].unique():
                if not strategy:
                    continue

                strat_trades = all_trades[all_trades["strategy"] == strategy]
                wins = strat_trades[strat_trades["pnl"] > 0]

                st.subheader(f"Strategy: {strategy}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Trades", len(strat_trades))
                col2.metric("Win Rate", f"{len(wins)/len(strat_trades)*100:.1f}%" if len(strat_trades) > 0 else "0%")
                col3.metric("Total P&L", f"₹{strat_trades['pnl'].sum():+,.0f}")
                col4.metric("Avg P&L", f"₹{strat_trades['pnl'].mean():+,.0f}")

                st.divider()
        else:
            st.info("No trades recorded yet")

        # FII flow chart
        fii_df = store.get_fii_dii_history(days=30)
        if not fii_df.empty:
            st.subheader("FII/DII Flows (Last 30 Days)")
            fig = go.Figure()
            colors = ["#00ff88" if v >= 0 else "#ff4444" for v in fii_df["fii_net_value"]]
            fig.add_trace(go.Bar(
                x=fii_df["date"],
                y=fii_df["fii_net_value"],
                name="FII Net",
                marker_color=colors,
            ))
            fig.update_layout(
                template="plotly_dark",
                height=350,
                yaxis_title="₹ Crores",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 4: Trade Log
    # ══════════════════════════════════════
    with tab4:
        st.header("Trade Log")

        n_trades = st.slider("Show last N trades", 10, 200, 50)
        trades_log = store.get_trades(limit=n_trades)

        if not trades_log.empty:
            # Summary
            total_pnl = trades_log["pnl"].sum()
            st.metric("Total P&L (shown)", f"₹{total_pnl:+,.0f}")

            st.dataframe(
                trades_log[[
                    "created_at", "symbol", "side", "quantity",
                    "price", "fill_price", "strategy", "regime",
                    "pnl", "status", "notes",
                ]],
                use_container_width=True,
                height=600,
            )
        else:
            st.info("No trades recorded")

    # ══════════════════════════════════════
    # TAB 5: Risk Dashboard
    # ══════════════════════════════════════
    with tab5:
        st.header("Risk Dashboard")

        positions = store.get_open_positions()

        col1, col2, col3 = st.columns(3)
        col1.metric("Open Positions", len(positions))

        if not portfolio.empty:
            latest = portfolio.iloc[-1]
            col2.metric("Exposure", f"{latest.get('exposure_pct', 0):.1f}%")
            col3.metric("Max Drawdown", f"{latest.get('drawdown_pct', 0):.1f}%")

        if not positions.empty:
            st.subheader("Open Positions")
            st.dataframe(positions, use_container_width=True)

        # Circuit breaker status
        st.subheader("Circuit Breaker Status")
        st.success("NORMAL")  # Will be dynamic when connected to live system

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("VELTRIX V4")


if __name__ == "__main__":
    main()
