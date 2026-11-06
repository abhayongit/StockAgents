#!/usr/bin/env python3
"""
Stock Volume Agent
==================
Asks for a stock name or ticker, then prints the traded volume for each hour of
each of the last 5 trading days, plus a daily total.

    pip install yfinance pandas
    python stock_volume_agent.py              # interactive: asks for a stock
    python stock_volume_agent.py NVDA         # one-shot
    python stock_volume_agent.py "hcl tech" --days 5 --csv

What "transaction volume" means here
  Volume = number of SHARES traded in each hour. It is not the number of trades
  and not a money value. Free sources do not give trade counts. Use --notional
  to also show approximate traded value, in the stock's own currency (shares x average of the bar's high, low and close).

Data source and caveats
  * Uses the free `yfinance` library, which reads Yahoo Finance. It is unofficial
    and for personal use. Yahoo's terms restrict commercial use and redistribution,
    and it can change or break without notice. Check Yahoo's terms for your case.
  * Hourly history is only kept for roughly the last 730 days, which is plenty here.
  * Yahoo's volume is generally consolidated across US venues, but it can be revised
    after the close, so today's numbers may change.
  * Each hourly bar is labelled with its START time in the EXCHANGE's local time.
    For US stocks the first bar starts 09:30 and the last starts 15:30 (it runs to
    the 16:00 close, so it covers only 30 minutes). Today's last bar is partial
    while the market is open.
  * Not investment advice.
"""

import argparse
import sys
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas")


# ----------------------------------------------------------------------------
# 1. Resolve what the user typed to a ticker
# ----------------------------------------------------------------------------
def looks_like_ticker(text: str) -> bool:
    """A single token with no spaces, e.g. NVDA, BRK-B, HCLTECH.NS, 7203.T."""
    return " " not in text and 1 <= len(text) <= 12


def search_candidates(query: str, limit: int = 6) -> list[dict]:
    """Look up a company name on Yahoo and return equity/ETF matches."""
    try:
        quotes = yf.Search(query, max_results=limit * 2).quotes
    except Exception as exc:  # network or API change
        print(f"  (name search failed: {exc})")
        return []
    out = []
    for q in quotes:
        if q.get("quoteType") in ("EQUITY", "ETF") and q.get("symbol"):
            out.append({
                "symbol": q["symbol"],
                "name": q.get("shortname") or q.get("longname") or "",
                "exchange": q.get("exchDisp") or q.get("exchange") or "",
            })
    return out[:limit]


def resolve_symbol(user_text: str, interactive: bool = True) -> str | None:
    text = user_text.strip()
    if not text:
        return None
    # A typed ticker is used directly. Fetching it will fail clearly if it is wrong.
    if looks_like_ticker(text):
        return text.upper()
    matches = search_candidates(text)
    if not matches:
        print(f"  I couldn't find a stock matching '{text}'. Try the ticker code.")
        return None
    if len(matches) == 1 or not interactive:
        return matches[0]["symbol"]
    print(f"  Which one did you mean for '{text}'?")
    for i, m in enumerate(matches, 1):
        print(f"   {i}. {m['symbol']:<12} {m['name']}  [{m['exchange']}]")
    choice = input("  Enter a number (or press Enter for 1): ").strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(matches) else 0
    return matches[idx]["symbol"]


# ----------------------------------------------------------------------------
# 2. Fetch hourly bars
# ----------------------------------------------------------------------------
def fetch_hourly(symbol: str) -> pd.DataFrame:
    """Hourly OHLCV for about the last 10 calendar days (enough for 5 sessions)."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="10d", interval="1h", auto_adjust=False)
    return df


# ----------------------------------------------------------------------------
# 3. Group into day x hour (pure function, easy to test)
# ----------------------------------------------------------------------------
def summarize(df: pd.DataFrame, days: int = 5, notional: bool = False) -> pd.DataFrame:
    """
    Return one row per hourly bar for the last `days` trading days with columns:
    date, hour_label, volume, pct_of_day (and notional if requested).

    A 'trading day' is a calendar date that has at least one bar with volume.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "Volume" not in df.columns:
        raise ValueError("No volume column in the data returned.")

    # Index is tz-aware in the exchange's local time, so .date is the local date.
    df["date"] = df.index.date
    df["start"] = df.index

    # Keep only days that actually traded.
    daily_total = df.groupby("date")["Volume"].sum()
    traded_dates = [d for d, v in daily_total.items() if v > 0]
    keep = sorted(traded_dates)[-days:]
    df = df[df["date"].isin(keep)]

    # Label each bar by its START time in the exchange's local time. The end time
    # depends on the exchange's closing time, which the data does not give us.
    df["hour_label"] = [f"{s:%H:%M}" for s in df.index]

    out = df[["date", "hour_label", "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    ).copy()
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].round(2)
    out["volume"] = out["volume"].astype("int64")
    day_sum = out.groupby("date")["volume"].transform("sum")
    out["pct_of_day"] = (out["volume"] / day_sum.where(day_sum > 0) * 100).round(1)

    if notional:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        out["notional"] = (df["Volume"] * typical).round(0).astype("int64")

    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Present
# ----------------------------------------------------------------------------
def fmt_int(n: int) -> str:
    return f"{n:,}"


def print_report(symbol: str, table: pd.DataFrame, notional: bool, tz_name: str) -> None:
    if table.empty:
        print("  No hourly data came back. The market may be closed for the whole window "
              "or the ticker may be wrong.")
        return

    print()
    print(f"Hourly volume for {symbol}  (times in {tz_name}; volume = shares traded)")
    print("=" * 100)
    for date, grp in table.groupby("date", sort=True):
        total = int(grp["volume"].sum())
        busiest = grp.loc[grp["volume"].idxmax(), "hour_label"]
        weekday = pd.Timestamp(date).strftime("%a")
        d_open, d_close = grp["open"].iloc[0], grp["close"].iloc[-1]
        chg = (d_close / d_open - 1) * 100 if d_open else 0.0
        print(f"\n{date}  ({weekday})   total {fmt_int(total)}   busiest hour starts {busiest}")
        print(f"  Day: open {d_open:,.2f}  high {grp['high'].max():,.2f}  "
              f"low {grp['low'].min():,.2f}  close {d_close:,.2f}  ({chg:+.2f}%)")
        hdr = (f"  {'Hour starting':<13}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}"
               f"{'Volume':>16}{'% of day':>10}")
        if notional:
            hdr += f"{'~value traded':>18}"
        print(hdr)
        for _, r in grp.iterrows():
            line = (f"  {r['hour_label']:<13}{r['open']:>10,.2f}{r['high']:>10,.2f}"
                    f"{r['low']:>10,.2f}{r['close']:>10,.2f}"
                    f"{fmt_int(int(r['volume'])):>16}{r['pct_of_day']:>9.1f}%")
            if notional:
                line += f"{fmt_int(int(r['notional'])):>18}"
            print(line)

    print("\nDaily totals")
    print("-" * 40)
    totals = table.groupby("date")["volume"].sum()
    for date, v in totals.items():
        print(f"  {date}   {fmt_int(int(v)):>16}")
    print(f"  {'Average':<10}   {fmt_int(int(totals.mean())):>16}")
    print("\nNote: the most recent day may be partial if the market is still open.")


# ----------------------------------------------------------------------------
# 5. The agent loop
# ----------------------------------------------------------------------------
def run_once(query: str, days: int, notional: bool, csv: bool, interactive: bool) -> bool:
    symbol = resolve_symbol(query, interactive=interactive)
    if not symbol:
        return False
    print(f"  Fetching hourly data for {symbol} ...")
    try:
        raw = fetch_hourly(symbol)
    except Exception as exc:
        print(f"  Could not fetch data for {symbol}: {exc}")
        return False
    if raw is None or raw.empty:
        print(f"  No data returned for '{symbol}'. Check the ticker. "
              "Non-US stocks need a suffix, for example HCLTECH.NS or 7203.T.")
        return False

    table = summarize(raw, days=days, notional=notional)
    tz_name = str(raw.index.tz) if raw.index.tz is not None else "exchange local time"
    print_report(symbol, table, notional, tz_name)

    if csv and not table.empty:
        path = f"{symbol.replace('.', '_')}_hourly_volume.csv"
        table.to_csv(path, index=False)
        print(f"\nSaved {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    """argv=None reads the command line. In Jupyter pass a list, e.g. main(["NVDA", "--days", "5"])."""
    p = argparse.ArgumentParser(description="Hourly stock volume for the last N trading days.")
    p.add_argument("stock", nargs="?", help="Company name or ticker (asked for if omitted)")
    p.add_argument("--days", type=int, default=5, help="Trading days to show (default 5)")
    p.add_argument("--notional", action="store_true", help="Also show approximate traded value (in the stock's currency)")
    p.add_argument("--csv", action="store_true", help="Save the table to a CSV file")
    args = p.parse_args(argv)

    if args.stock:
        ok = run_once(args.stock, args.days, args.notional, args.csv, interactive=False)
        return 0 if ok else 1

    print("Stock Volume Agent. Type a company name or ticker (q to quit).")
    print("Examples: NVDA, nvidia, HCLTECH.NS, 'western union'")
    while True:
        try:
            query = input("\nWhich stock? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("q", "quit", "exit"):
            break
        if query:
            run_once(query, args.days, args.notional, args.csv, interactive=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
