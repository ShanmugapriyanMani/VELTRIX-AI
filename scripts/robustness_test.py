#!/usr/bin/env python3
"""Parameter robustness test — read-only diagnostic.

Runs backtest with threshold variations and collects metrics.
Temporarily patches .env.plus for each run, always restores original.
No production code is modified.
"""

import os
import re
import shutil
import subprocess
import sys

PROJECT = "/home/sandy/VELTRIX"
ENV_FILE = os.path.join(PROJECT, ".env.plus")
ENV_BACKUP = os.path.join(PROJECT, ".env.plus.robustness_backup")


def patch_env_file(overrides: dict) -> None:
    """Patch .env.plus with override values, preserving all other lines."""
    with open(ENV_FILE, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        key = line.split("=")[0].strip() if "=" in line else ""
        if key in overrides:
            new_lines.append(f"{key}={overrides[key]}\n")
        else:
            new_lines.append(line)

    with open(ENV_FILE, "w") as f:
        f.writelines(new_lines)


def run_backtest(overrides: dict) -> dict:
    """Run backtest with patched .env.plus, parse OVERVIEW metrics."""
    # Patch
    patch_env_file(overrides)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--mode", "backtest"],
            capture_output=True, text=True, cwd=PROJECT, timeout=600,
        )
        output = result.stdout + result.stderr
    finally:
        # Always restore
        shutil.copy2(ENV_BACKUP, ENV_FILE)

    metrics = {}
    for line in output.split("\n"):
        if "CAGR" in line and "%" in line:
            m = re.search(r"([\d.]+)%", line)
            if m and "cagr" not in metrics:
                metrics["cagr"] = float(m.group(1))
        elif "Max Drawdown" in line and "%" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                metrics["dd"] = float(m.group(1))
        elif "Win Rate" in line and "%" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                metrics["wr"] = float(m.group(1))
        elif "Profit Factor" in line and "|" in line:
            m = re.search(r"\|\s+([\d.]+)\s+\|", line)
            if m and "pf" not in metrics:
                metrics["pf"] = float(m.group(1))
        elif "Total Trades" in line and "|" in line:
            m = re.search(r"\|\s+(\d+)\s+\|", line)
            if m:
                metrics["trades"] = int(m.group(1))
    return metrics


def print_table(title: str, param_name: str, results: list, baseline_val: float):
    print(f"\n{title}")
    print(f"+----------+--------+-------+--------+--------+-------+")
    print(f"| {param_name:>8s} | Trades |  WR   |  CAGR  |   DD   |   PF  |")
    print(f"+----------+--------+-------+--------+--------+-------+")
    for val, m in results:
        marker = " <-BASE" if val == baseline_val else ""
        print(f"| {val:>8.2f} | {m.get('trades',0):>6d} | {m.get('wr',0):>4.1f}% | {m.get('cagr',0):>5.1f}% | {m.get('dd',0):>5.1f}% | {m.get('pf',0):>5.2f} |{marker}")
    print(f"+----------+--------+-------+--------+--------+-------+")


def assess_robustness(results: list, baseline_val: float) -> str:
    cagrs = [m.get("cagr", 0) for _, m in results]
    wrs = [m.get("wr", 0) for _, m in results]
    dds = [m.get("dd", 0) for _, m in results]
    base_cagr = next((m.get("cagr", 0) for v, m in results if v == baseline_val), 0)

    cagr_range_pct = (max(cagrs) - min(cagrs)) / base_cagr * 100 if base_cagr > 0 else 0
    wr_range = max(wrs) - min(wrs)
    dd_range = max(dds) - min(dds)

    if cagr_range_pct > 20:
        verdict = "FRAGILE"
    elif cagr_range_pct > 10 or wr_range > 5 or dd_range > 5:
        verdict = "SENSITIVE"
    else:
        verdict = "ROBUST"

    return (
        f"  {verdict}: CAGR range {min(cagrs):.1f}%-{max(cagrs):.1f}% "
        f"(spread {cagr_range_pct:.1f}%), "
        f"WR {min(wrs):.1f}%-{max(wrs):.1f}%, "
        f"DD {min(dds):.1f}%-{max(dds):.1f}%"
    )


def main():
    # Backup original .env.plus
    shutil.copy2(ENV_FILE, ENV_BACKUP)
    print("=" * 60)
    print("  VELTRIX V9.3 — PARAMETER ROBUSTNESS TEST")
    print("  Baseline: CAGR=90.50% DD=14.61% WR=74.3%")
    print("=" * 60)

    try:
        # ── Test 1: TRENDING threshold ──
        print("\n[1/3] Testing TRENDING threshold variations...")
        trending_results = []
        for val in [1.25, 1.50, 1.75, 2.00, 2.25]:
            ce_val = val + 0.25
            pe_val = val - 0.25
            overrides = {
                "TRENDING_THRESHOLD": str(val),
                "CE_TRENDING_THRESHOLD": str(ce_val),
                "PE_TRENDING_THRESHOLD": str(pe_val),
            }
            print(f"  TRENDING={val:.2f} (CE={ce_val:.2f}, PE={pe_val:.2f})...", end=" ", flush=True)
            m = run_backtest(overrides)
            print(f"trades={m.get('trades',0)} CAGR={m.get('cagr',0):.1f}% WR={m.get('wr',0):.1f}% DD={m.get('dd',0):.1f}%")
            trending_results.append((val, m))

        # ── Test 2: VOLATILE threshold ──
        print("\n[2/3] Testing VOLATILE threshold variations...")
        volatile_results = []
        for val in [2.00, 2.25, 2.50, 2.75, 3.00]:
            ce_val = val + 0.5
            pe_val = val
            overrides = {
                "VOLATILE_THRESHOLD": str(val),
                "CE_VOLATILE_THRESHOLD": str(ce_val),
                "PE_VOLATILE_THRESHOLD": str(pe_val),
            }
            print(f"  VOLATILE={val:.2f} (CE={ce_val:.2f}, PE={pe_val:.2f})...", end=" ", flush=True)
            m = run_backtest(overrides)
            print(f"trades={m.get('trades',0)} CAGR={m.get('cagr',0):.1f}% WR={m.get('wr',0):.1f}% DD={m.get('dd',0):.1f}%")
            volatile_results.append((val, m))

        # ── Test 3: RANGEBOUND threshold ──
        print("\n[3/3] Testing RANGEBOUND threshold variations...")
        rangebound_results = []
        for val in [1.50, 1.75, 2.00, 2.25, 2.50]:
            ce_val = val + 0.25
            pe_val = val - 0.25
            overrides = {
                "RANGEBOUND_THRESHOLD": str(val),
                "CE_RANGEBOUND_THRESHOLD": str(ce_val),
                "PE_RANGEBOUND_THRESHOLD": str(pe_val),
            }
            print(f"  RANGEBOUND={val:.2f} (CE={ce_val:.2f}, PE={pe_val:.2f})...", end=" ", flush=True)
            m = run_backtest(overrides)
            print(f"trades={m.get('trades',0)} CAGR={m.get('cagr',0):.1f}% WR={m.get('wr',0):.1f}% DD={m.get('dd',0):.1f}%")
            rangebound_results.append((val, m))

        # ── Tables ──
        print_table("TRENDING THRESHOLD ROBUSTNESS:", "T_thresh", trending_results, 1.75)
        print_table("VOLATILE THRESHOLD ROBUSTNESS:", "V_thresh", volatile_results, 2.50)
        print_table("RANGEBOUND THRESHOLD ROBUSTNESS:", "R_thresh", rangebound_results, 2.00)

        # ── Assessment ──
        print("\nROBUSTNESS ASSESSMENT:")
        print("-" * 60)
        print("TRENDING (baseline=1.75):")
        print(assess_robustness(trending_results, 1.75))
        print("VOLATILE (baseline=2.50):")
        print(assess_robustness(volatile_results, 2.50))
        print("RANGEBOUND (baseline=2.00):")
        print(assess_robustness(rangebound_results, 2.00))
        print("-" * 60)

    finally:
        # Always restore original
        shutil.copy2(ENV_BACKUP, ENV_FILE)
        os.remove(ENV_BACKUP)
        print("\n.env.plus restored to original.")


if __name__ == "__main__":
    main()
