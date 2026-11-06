#!/usr/bin/env python3
"""
Support Price Predictor (no AI, plain rules)
============================================
Reads the "peak minutes" CSV written by stock_volume_agent.py and, for every hour, works out a
signal, a standard deviation and a support price.

    pip install pandas numpy
    python support_price_predictor.py NVDA_hourly_volume_peak_minutes.csv
    python support_price_predictor.py peaks.csv --window 7 --hourly NVDA_hourly_volume.csv

In Jupyter:
    import support_price_predictor as spp
    spp.main(["NVDA_hourly_volume_peak_minutes.csv"])

THE RULES (per hour)
  volume difference = Max est. bought - Max est. sold
  price difference  = buy price - sell price

  1. Volume difference NEGATIVE and price difference NEGATIVE  -> "Bearish"
        support price = highest price of that hour - standard deviation
  2. Volume difference POSITIVE and price difference POSITIVE  -> "Bullish"
        support price = lowest price of that hour + standard deviation
  3. Anything else (signs disagree, or a difference is exactly 0) -> "Neutral": no support price.

HOW THE PIECES ARE DEFINED (my reading of the request. Change with the options if you meant otherwise)
  * Highest / lowest price of the hour: the file only has two prices per hour (sell price and buy
    price), so highest = the larger of the two, lowest = the smaller. If you pass --hourly with the
    main hourly CSV from stock_volume_agent.py, the hour's true high and low are used instead.
  * Standard deviation: sample standard deviation of the buy and sell prices over the last
    --window hours in the file (default 7, about one trading day), including the current hour.
    It is in price units, so it can be added to or taken off a price. For the very first hour it
    is the spread of that hour's two prices. Because it measures price LEVELS, a big overnight gap
    makes it jump (and the support prices of the next hours sit far from the price). Use
    --std-basis change to measure the hour-to-hour price CHANGES within a day instead, which
    ignores overnight gaps.
  * "% negative / % positive": move % = -std / highest x 100 for Bearish and +std / lowest x 100
    for Bullish. Volume difference % = (bought - sold) / (bought + sold) x 100, price difference %
    = (buy price - sell price) / sell price x 100. Price and volume change % compare each hour with
    the previous hour in the file (average of buy and sell price; bought + sold volume).

OUTPUT FILES
  <input>_with_support.csv   your input file, unchanged, plus two columns: Standard deviation, Support price
  <input>_signals.csv        the working: signal, the % figures, highest/lowest price used

HONEST LIMITS
  * "Max est. sold / bought" are the busiest ESTIMATED sell/buy MINUTE of the hour, not hourly totals,
    and the buy/sell split itself is estimated from where each minute closed in its range. So the
    volume sign and the price sign come from the same price shape and tend to agree. Treat the
    signal as a description of how the hour traded, not as independent evidence.
  * These rules are not tested for predictive value. A handful of days of data cannot show that.
  * Not investment advice.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# canonical name -> accepted column headings (lower case). Works with both versions of the headings
# written by stock_volume_agent.py.
COLS = {
    "date": ["date"],
    "hour": ["hour starting", "hour_label", "hour"],
    "sold": ["max est. sold", "max est sold", "max_sell_shares"],
    "sell_price": ["sell price", "max_sell_price"],
    "bought": ["max est. bought", "max est bought", "max_buy_shares"],
    "buy_price": ["buy price", "max_buy_price"],
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def find_cols(df: pd.DataFrame) -> dict:
    low = {c.strip().lower(): c for c in df.columns}
    found = {}
    for key, names in COLS.items():
        hit = next((low[n] for n in names if n in low), None)
        if hit is None:
            raise SystemExit(f"Column for '{key}' not found. Expected one of {names}. "
                             f"Columns in the file: {list(df.columns)}")
        found[key] = hit
    return found


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def parse_dates(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):          # older pandas without format="mixed"
        return pd.to_datetime(s, errors="coerce")


def norm_hour(v) -> str:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(v))
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else str(v).strip()


def read_hourly_highs_lows(path: str) -> pd.DataFrame:
    """High and low per hour from the main hourly CSV (date, hour_label / Hour starting, high, low)."""
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    low = {c.strip().lower(): c for c in h.columns}
    need = {"date": ["date"], "hour": ["hour_label", "hour starting", "hour"],
            "hi_hr": ["high"], "lo_hr": ["low"]}
    pick = {}
    for key, names in need.items():
        hit = next((low[n] for n in names if n in low), None)
        if hit is None:
            raise SystemExit(f"--hourly file needs a '{names[0]}' column. Found: {list(h.columns)}")
        pick[key] = hit
    out = pd.DataFrame({
        "date": parse_dates(h[pick["date"]]),
        "hour": h[pick["hour"]].map(norm_hour),
        "hi_hr": to_num(h[pick["hi_hr"]]),
        "lo_hr": to_num(h[pick["lo_hr"]]),
    })
    return out.drop_duplicates(["date", "hour"])


# ----------------------------------------------------------------------------
# the analysis (pure function, easy to test)
# ----------------------------------------------------------------------------
def analyze(df: pd.DataFrame, cm: dict, window: int = 7, hourly: pd.DataFrame | None = None,
            std_basis: str = "price") -> pd.DataFrame:
    """Returns one row per input row, in the input's row order.

    std_basis "price":  std dev of the buy and sell price levels over the last `window` hours.
    std_basis "change": std dev of the hour-to-hour changes in the average of buy and sell price over
                        the last `window` hours, counting only changes inside one day (so an overnight
                        gap is not treated as volatility). Needs 2 such changes, otherwise it falls back
                        to the "price" figure for that hour.
    """
    if std_basis not in ("price", "change"):
        raise ValueError("std_basis must be 'price' or 'change'")
    n = pd.DataFrame({
        "orig": np.arange(len(df)),
        "date": parse_dates(df[cm["date"]]).to_numpy(),
        "hour": df[cm["hour"]].map(norm_hour).to_numpy(),
        "sold": to_num(df[cm["sold"]]).to_numpy(),
        "sell_price": to_num(df[cm["sell_price"]]).to_numpy(),
        "bought": to_num(df[cm["bought"]]).to_numpy(),
        "buy_price": to_num(df[cm["buy_price"]]).to_numpy(),
    })
    n["date"] = pd.to_datetime(n["date"])
    if hourly is not None:
        n = n.merge(hourly, on=["date", "hour"], how="left")
    n = n.sort_values(["date", "hour", "orig"], kind="stable").reset_index(drop=True)

    # --- standard deviation of buy and sell prices over the last `window` hours (including this one)
    sp, bp = n["sell_price"].to_numpy(), n["buy_price"].to_numpy()
    std = np.full(len(n), np.nan)
    mid_arr = (sp + bp) / 2
    day_arr = n["date"].to_numpy()
    for i in range(len(n)):
        lo = max(0, i - window + 1)
        if std_basis == "change":
            ch = [mid_arr[j] - mid_arr[j - 1] for j in range(lo + 1, i + 1)
                  if day_arr[j] == day_arr[j - 1] and not np.isnan(mid_arr[j]) and not np.isnan(mid_arr[j - 1])]
            if len(ch) >= 2:
                std[i] = np.std(ch, ddof=1)
                continue
        vals = np.concatenate([sp[lo:i + 1], bp[lo:i + 1]])
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            std[i] = vals.std(ddof=1)
    n["std"] = std

    # --- highest / lowest price of the hour
    pair_hi = n[["sell_price", "buy_price"]].max(axis=1)
    pair_lo = n[["sell_price", "buy_price"]].min(axis=1)
    n["highest"] = n["hi_hr"].fillna(pair_hi) if "hi_hr" in n else pair_hi
    n["lowest"] = n["lo_hr"].fillna(pair_lo) if "lo_hr" in n else pair_lo

    # --- the two differences and the signal
    n["vol_diff"] = n["bought"] - n["sold"]
    n["price_diff"] = n["buy_price"] - n["sell_price"]
    bear = (n["vol_diff"] < 0) & (n["price_diff"] < 0)
    bull = (n["vol_diff"] > 0) & (n["price_diff"] > 0)
    nodata = n[["vol_diff", "price_diff"]].isna().any(axis=1)
    n["signal"] = np.select([nodata, bear, bull], ["No data", "Bearish", "Bullish"], default="Neutral")

    # --- support price and the % move it implies
    n["support"] = np.select([n["signal"] == "Bearish", n["signal"] == "Bullish"],
                             [n["highest"] - n["std"], n["lowest"] + n["std"]], default=np.nan)
    n["move_pct"] = np.select([n["signal"] == "Bearish", n["signal"] == "Bullish"],
                              [-n["std"] / n["highest"] * 100, n["std"] / n["lowest"] * 100], default=np.nan)

    # --- the % figures
    total = n["bought"] + n["sold"]
    n["vol_diff_pct"] = n["vol_diff"] / total.where(total > 0) * 100
    n["price_diff_pct"] = n["price_diff"] / n["sell_price"].where(n["sell_price"] > 0) * 100
    mid = (n["buy_price"] + n["sell_price"]) / 2
    n["price_chg_pct"] = mid.pct_change(fill_method=None) * 100
    n["vol_chg_pct"] = total.pct_change(fill_method=None) * 100
    n["std_pct"] = n["std"] / mid.where(mid > 0) * 100
    n = n.replace([np.inf, -np.inf], np.nan)

    return n.sort_values("orig").set_index("orig")      # back to the input's row order


# ----------------------------------------------------------------------------
# presenting and saving
# ----------------------------------------------------------------------------
def fmt(v, spec: str, blank: str = "") -> str:
    return blank if pd.isna(v) else format(v, spec)


def print_report(df: pd.DataFrame, cm: dict, res: pd.DataFrame) -> None:
    print()
    print("Support price by hour")
    print("=" * 112)
    print(f"  {'Date':<10}{'Hour':>6}  {'Signal':<8}{'Vol diff %':>11}{'Price diff %':>13}"
          f"{'Highest':>10}{'Lowest':>10}{'Std dev':>9}{'Support':>10}{'Move %':>9}")
    for i in df.index:
        r = res.loc[i]
        print(f"  {str(df.at[i, cm['date']]):<10}{str(df.at[i, cm['hour']]):>6}  {r['signal']:<8}"
              f"{fmt(r['vol_diff_pct'], '+.1f'):>11}{fmt(r['price_diff_pct'], '+.2f'):>13}"
              f"{fmt(r['highest'], '.2f'):>10}{fmt(r['lowest'], '.2f'):>10}{fmt(r['std'], '.2f'):>9}"
              f"{fmt(r['support'], '.2f', '-'):>10}{fmt(r['move_pct'], '+.2f', '-'):>9}")

    counts = res["signal"].value_counts()
    print("\nHours:", ", ".join(f"{k} {v}" for k, v in counts.items()))

    chron = res.sort_values(["date", "hour"], kind="stable")
    last = chron.iloc[-1]
    row_label = f"{last['date']:%Y-%m-%d} {last['hour']}"
    print(f"\nLatest hour in the file ({row_label}): {last['signal']}", end="")
    if last["signal"] in ("Bearish", "Bullish"):
        print(f", support {last['support']:.2f}, standard deviation {last['std']:.2f}, "
              f"move {last['move_pct']:+.2f}%")
    else:
        print(" (no support price for this hour)")
    print("\nThese are rule-of-thumb levels from estimated data, not tested forecasts. Not investment advice.")


def save(df: pd.DataFrame, res: pd.DataFrame, cm: dict, out: str) -> tuple[str, str]:
    base = os.path.splitext(out)[0]
    # 1. the input exactly as it was, plus two columns
    with_cols = df.copy()
    with_cols["Standard deviation"] = res["std"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}").to_numpy()
    with_cols["Support price"] = res["support"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}").to_numpy()
    with_cols.to_csv(out, index=False)

    # 2. the working
    sig = pd.DataFrame({
        "date": df[cm["date"]].to_numpy(),
        "Hour starting": df[cm["hour"]].to_numpy(),
        "signal": res["signal"].to_numpy(),
        "vol_diff_bought_minus_sold": res["vol_diff"].to_numpy(),
        "vol_diff_pct": res["vol_diff_pct"].round(2).to_numpy(),
        "price_diff_buy_minus_sell": res["price_diff"].round(2).to_numpy(),
        "price_diff_pct": res["price_diff_pct"].round(3).to_numpy(),
        "price_chg_pct_vs_prev_hour": res["price_chg_pct"].round(3).to_numpy(),
        "vol_chg_pct_vs_prev_hour": res["vol_chg_pct"].round(2).to_numpy(),
        "highest_price": res["highest"].round(2).to_numpy(),
        "lowest_price": res["lowest"].round(2).to_numpy(),
        "std_dev": res["std"].round(2).to_numpy(),
        "std_pct": res["std_pct"].round(3).to_numpy(),
        "support_price": res["support"].round(2).to_numpy(),
        "move_pct": res["move_pct"].round(3).to_numpy(),
    })
    if "symbol" in {c.lower() for c in df.columns}:
        sym = next(c for c in df.columns if c.lower() == "symbol")
        sig.insert(0, "symbol", df[sym].to_numpy())
    sig_path = base + "_signals.csv"
    sig.to_csv(sig_path, index=False)
    return out, sig_path


def main(argv: list[str] | None = None) -> int:
    """argv=None reads the command line. In Jupyter pass a list, e.g. main(["peaks.csv"])."""
    p = argparse.ArgumentParser(description="Support price per hour from the peak-minutes CSV.")
    p.add_argument("input_csv", help="CSV written by stock_volume_agent.py (…_peak_minutes.csv)")
    p.add_argument("--window", type=int, default=7,
                   help="Hours used for the standard deviation, including the current hour (default 7)")
    p.add_argument("--std-basis", choices=["price", "change"], default="price",
                   help="'price' (default): std dev of price levels over the window. "
                        "'change': std dev of hour-to-hour price changes within a day (ignores overnight gaps)")
    p.add_argument("--hourly", help="Main hourly CSV from stock_volume_agent.py, to use each hour's true high and low")
    p.add_argument("--out", help="Path for the input-plus-two-columns CSV (default <input>_with_support.csv)")
    args = p.parse_args(argv)

    if args.window < 1:
        print("--window must be at least 1.")
        return 1
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if df.empty:
        print("The input file has no rows.")
        return 1
    cm = find_cols(df)

    hourly = read_hourly_highs_lows(args.hourly) if args.hourly else None
    res = analyze(df, cm, args.window, hourly, args.std_basis)
    print(f"Standard deviation basis: {args.std_basis} (window {args.window} hours)")
    if hourly is not None:
        matched = int(res["highest"].notna().sum())
        print(f"Using true hourly highs and lows from {args.hourly} where the hours match.")
        if "hi_hr" in res and res["hi_hr"].isna().any():
            print(f"  {int(res['hi_hr'].isna().sum())} hour(s) had no match, so the two-price high and low were used there.")
    print_report(df, cm, res)

    out = args.out or os.path.splitext(args.input_csv)[0] + "_with_support.csv"
    a, b = save(df, res, cm, out)
    print(f"\nSaved {a}  (your input plus 'Standard deviation' and 'Support price')")
    print(f"Saved {b}  (signal and the % working)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
