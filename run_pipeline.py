#!/usr/bin/env python3
"""
Run the three agents in order:
  1. volAgent/stock_volume_agent.py        -> writes <SYMBOL>_hourly_volume*.csv
  2. forcastAgent/support_price_predictor.py -> reads the _peak_minutes.csv, writes _with_support / _signals
  3. forcastAgent/technical_chart_predictor.py -> 2-week chart outlook for the next 3 and 5 days

    python run_pipeline.py NVDA
    python run_pipeline.py NVDA AAPL MSFT --days 5 --outdir resources
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "volAgent"))
sys.path.insert(0, os.path.join(ROOT, "forcastAgent"))

import stock_volume_agent as sva  # noqa: E402
import support_price_predictor as spp  # noqa: E402
import technical_chart_predictor as tcp  # noqa: E402


def run_for(symbol: str, days: int, outdir: str, window: int) -> bool:
    print(f"\n=== [1/3] Stock volume agent: {symbol} ===")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{symbol.upper().replace('.', '_')}_hourly_volume")
    hourly_csv = base + ".csv"
    peaks_csv = base + "_peak_minutes.csv"

    if sva.main([symbol, "--days", str(days), "--out", hourly_csv]) != 0:
        print(f"  Volume agent failed for {symbol}; skipping the other agents.")
        return False

    ok = True
    if os.path.exists(peaks_csv):
        print(f"\n=== [2/3] Support price predictor: {symbol} ===")
        ok = spp.main([peaks_csv, "--hourly", hourly_csv, "--window", str(window)]) == 0
    else:
        print(f"  {peaks_csv} was not written (no 1-minute data?); skipping support predictor.")
        ok = False

    print(f"\n=== [3/3] Technical chart predictor: {symbol} ===")
    ok = (tcp.main([symbol, "--csv", "--outdir", outdir]) == 0) and ok
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Run stock volume agent, then support price predictor.")
    p.add_argument("symbols", nargs="+", help="One or more tickers, e.g. NVDA AAPL")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--outdir", default=os.path.join(ROOT, "resources"))
    args = p.parse_args()

    results = {s: run_for(s, args.days, args.outdir, args.window) for s in args.symbols}
    print("\nSummary:", ", ".join(f"{s}={'ok' if ok else 'FAILED'}" for s, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
