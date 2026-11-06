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
  * Bought / sold volume is an ESTIMATE, not real data. Free sources do not say who
    was the buyer or seller in each trade. The script splits each hour's volume by where
    the price closed within that hour's high-low range (close near the high = mostly
    buying, near the low = mostly selling). Net volume = bought - sold, a rough gauge of
    buying vs selling pressure. It is not a profit or loss figure. Real buy/sell data
    needs paid tick data (trade and quote records).
  * Peak-minute columns (max_sell_shares, max_sell_price, max_buy_shares, max_buy_price)
    come from Yahoo's 1-minute bars, which exist only for about the last 7 days, so
    older days show blank. They are NOT single trades, and the sold/bought split is an
    ESTIMATE (same method as above, applied to each minute). "Max sold" is the minute of
    that hour with the most estimated selling, "max bought" the minute with the most
    estimated buying. The price is the average of that minute's high, low and close.
    Use --no-peaks to skip them.
  * Not investment advice.
"""

import argparse
import os
import sys
import numpy as np
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
def est_buy_fraction(o, h, l, c) -> pd.Series:
    """Share of volume estimated as buying, from where the bar closed in its high-low range."""
    rng = h - l
    frac = ((c - l) / rng.where(rng > 0)).astype(float)
    flat = pd.Series(np.where(c > o, 1.0, np.where(c < o, 0.0, 0.5)), index=frac.index)
    return frac.fillna(flat).clip(0, 1)


def fetch_minute(symbol: str) -> pd.DataFrame:
    """1-minute bars. Yahoo keeps only about the last 7 days of these."""
    return yf.Ticker(symbol).history(period="7d", interval="1m", auto_adjust=False)


def peak_minutes(starts: pd.DatetimeIndex, minute_df: pd.DataFrame, n_bars: int) -> dict:
    """For each hourly bar, find the busiest minute and the minute with most estimated buying."""
    m = minute_df[minute_df["Volume"] > 0].copy()
    if m.empty:
        return {}
    pos = starts.searchsorted(m.index, side="right") - 1
    inside = (pos >= 0) & (m.index < starts[np.clip(pos, 0, None)] + pd.Timedelta(hours=1))
    m = m[inside]
    m["bar"] = pos[inside]
    if m.empty:
        return {}
    m["price"] = ((m["High"] + m["Low"] + m["Close"]) / 3).round(2)
    m["buy"] = (m["Volume"] * est_buy_fraction(m["Open"], m["High"], m["Low"], m["Close"])).round()
    bars = np.arange(n_bars)
    m["sell"] = m["Volume"] - m["buy"]
    ps = m.loc[m.groupby("bar")["sell"].idxmax()].set_index("bar")
    pb = m.loc[m.groupby("bar")["buy"].idxmax()].set_index("bar")
    return {
        "max_sell_shares": pd.array(ps["sell"].reindex(bars), dtype="Int64"),
        "max_sell_price": ps["price"].reindex(bars).to_numpy(),
        "max_buy_shares": pd.array(pb["buy"].reindex(bars), dtype="Int64"),
        "max_buy_price": pb["price"].reindex(bars).to_numpy(),
    }


def summarize(df: pd.DataFrame, days: int = 5, notional: bool = False,
              minute_df: pd.DataFrame | None = None) -> pd.DataFrame:
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

    # ESTIMATED buy/sell split. Free data has no trade-by-trade buyer/seller flag, so
    # each hour's volume is split by where the bar closed inside its own high-low range:
    #   close at the high -> all buying, close at the low -> all selling, middle -> 50/50.
    # A flat bar (high == low) falls back to close vs open, then to 50/50.
    frac = est_buy_fraction(out["open"], out["high"], out["low"], out["close"])
    out["est_buy"] = (out["volume"] * frac).round().astype("int64")
    out["est_sell"] = out["volume"] - out["est_buy"]
    out["net_volume"] = out["est_buy"] - out["est_sell"]      # + buying pressure, - selling pressure
    out["price_chg"] = (out["close"] - out["open"]).round(2)  # actual price move in the hour

    if notional:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        out["notional"] = (df["Volume"] * typical).round(0).astype("int64")

    if minute_df is not None and not minute_df.empty:
        for col, vals in peak_minutes(df.index, minute_df, len(out)).items():
            out[col] = vals

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
    print("=" * 140)
    for date, grp in table.groupby("date", sort=True):
        total = int(grp["volume"].sum())
        busiest = grp.loc[grp["volume"].idxmax(), "hour_label"]
        weekday = pd.Timestamp(date).strftime("%a")
        d_open, d_close = grp["open"].iloc[0], grp["close"].iloc[-1]
        chg = (d_close / d_open - 1) * 100 if d_open else 0.0
        print(f"\n{date}  ({weekday})   total {fmt_int(total)}   busiest hour starts {busiest}")
        print(f"  Day: open {d_open:,.2f}  high {grp['high'].max():,.2f}  "
              f"low {grp['low'].min():,.2f}  close {d_close:,.2f}  ({chg:+.2f}%)")
        d_buy, d_sell = int(grp["est_buy"].sum()), int(grp["est_sell"].sum())
        print(f"  Est. bought {fmt_int(d_buy)}  est. sold {fmt_int(d_sell)}  "
              f"net {d_buy - d_sell:+,} ({'buying' if d_buy >= d_sell else 'selling'} pressure)")
        hdr = (f"  {'Hour starting':<13}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}"
               f"{'Chg':>9}{'Volume':>16}{'% of day':>10}"
               f"{'Est. bought':>14}{'Est. sold':>14}{'Net volume':>14}")
        if notional:
            hdr += f"{'~value traded':>18}"
        print(hdr)
        for _, r in grp.iterrows():
            line = (f"  {r['hour_label']:<13}{r['open']:>10,.2f}{r['high']:>10,.2f}"
                    f"{r['low']:>10,.2f}{r['close']:>10,.2f}{r['price_chg']:>+9.2f}"
                    f"{fmt_int(int(r['volume'])):>16}{r['pct_of_day']:>9.1f}%"
                    f"{fmt_int(int(r['est_buy'])):>14}{fmt_int(int(r['est_sell'])):>14}"
                    f"{int(r['net_volume']):>+14,}")
            if notional:
                line += f"{fmt_int(int(r['notional'])):>18}"
            print(line)

    if "max_sell_shares" in table.columns and table["max_sell_shares"].notna().any():
        print("\nPeak minute inside each hour (from 1-minute data; ESTIMATED split; "
              "price = average of that minute's high, low, close)")
        print("-" * 84)
        for date, grp in table.groupby("date", sort=True):
            print(f"\n{date}")
            print(f"  {'Hour starting':<13}{'Max est. sold':>15}{'sell price':>12}"
                  f"{'Max est. bought':>18}{'buy price':>11}")
            for _, r in grp.iterrows():
                if pd.isna(r["max_sell_shares"]):
                    print(f"  {r['hour_label']:<13}{'n/a':>15}")
                    continue
                print(f"  {r['hour_label']:<13}{int(r['max_sell_shares']):>15,}{r['max_sell_price']:>12,.2f}"
                      f"{int(r['max_buy_shares']):>18,}{r['max_buy_price']:>11,.2f}")

    print("\nDaily totals")
    print("-" * 40)
    totals = table.groupby("date")["volume"].sum()
    for date, v in totals.items():
        print(f"  {date}   {fmt_int(int(v)):>16}")
    print(f"  {'Average':<10}   {fmt_int(int(totals.mean())):>16}")
    print("\nNote: the most recent day may be partial if the market is still open.")
    print("Note: 'Est. bought / Est. sold / Net volume' are ESTIMATES from where each hour closed in its "
          "high-low range.\n      Yahoo does not provide real buyer/seller data. 'Chg' is the actual price move in the hour.")


# ----------------------------------------------------------------------------
# 5. The agent loop
# ----------------------------------------------------------------------------
PEAK_COLS = ["max_sell_shares", "max_sell_price", "max_buy_shares", "max_buy_price"]


def ensure_peak_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Always have the 4 peak-minute columns in the table (blank where there is no 1-minute data)."""
    for col in PEAK_COLS:
        if col not in table.columns:
            table[col] = pd.array([pd.NA] * len(table), dtype="Int64") if "shares" in col else np.nan
    return table


def daily_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per day (same numbers as 'Daily totals' and the Day lines in the console) plus an AVERAGE row."""
    rows = []
    for date, g in table.groupby("date", sort=True):
        d_open, d_close = g["open"].iloc[0], g["close"].iloc[-1]
        rows.append({
            "date": date,
            "open": d_open, "high": g["high"].max(), "low": g["low"].min(), "close": d_close,
            "price_chg_pct": round((d_close / d_open - 1) * 100, 2) if d_open else np.nan,
            "total_volume": int(g["volume"].sum()),
            "est_bought": int(g["est_buy"].sum()),
            "est_sold": int(g["est_sell"].sum()),
            "net_volume": int(g["est_buy"].sum() - g["est_sell"].sum()),
            "busiest_hour_starts": g.loc[g["volume"].idxmax(), "hour_label"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        avg = {"date": "AVERAGE"}
        for c in ["total_volume", "est_bought", "est_sold", "net_volume"]:
            avg[c] = int(out[c].mean())
        out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
    return out


def get_minute_safe(symbol: str) -> pd.DataFrame | None:
    try:
        return fetch_minute(symbol)
    except Exception as exc:
        print(f"  (1-minute data unavailable, peak-minute columns skipped: {exc})")
        return None


def get_table(query: str, days: int = 5, notional: bool = False, peaks: bool = True) -> pd.DataFrame:
    """Return the hourly table as a DataFrame (handy in Jupyter). Empty if nothing found."""
    symbol = resolve_symbol(query, interactive=False)
    if not symbol:
        return pd.DataFrame()
    minute = get_minute_safe(symbol) if peaks else None
    table = summarize(fetch_hourly(symbol), days=days, notional=notional, minute_df=minute)
    if not table.empty:
        if peaks:
            ensure_peak_columns(table)
        table.insert(0, "symbol", symbol)
    return table


def run_once(query: str, days: int, notional: bool, csv: bool, interactive: bool,
             out: str | None = None, peaks: bool = True) -> bool:
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

    minute = get_minute_safe(symbol) if peaks else None
    table = summarize(raw, days=days, notional=notional, minute_df=minute)
    if peaks and not table.empty:
        ensure_peak_columns(table)
        if table["max_sell_shares"].isna().all():
            print("  Note: no 1-minute data came back, so the 4 peak-minute columns will be blank.")
    tz_name = str(raw.index.tz) if raw.index.tz is not None else "exchange local time"
    print_report(symbol, table, notional, tz_name)

    if (csv or out) and not table.empty:
        path = out or f"{symbol.replace('.', '_')}_hourly_volume.csv"
        to_save = table.copy()
        to_save.insert(0, "symbol", symbol)
        to_save.to_csv(path, index=False)
        daily_path = os.path.splitext(path)[0] + "_daily.csv"
        daily = daily_summary(table)
        daily.insert(0, "symbol", symbol)
        daily.to_csv(daily_path, index=False)
        print(f"\nSaved {path}  (hourly rows, including the 4 peak-minute columns)")
        print(f"Saved {daily_path}  (daily totals and average)")
    return True


def main(argv: list[str] | None = None) -> int:
    """argv=None reads the command line. In Jupyter pass a list, e.g. main(["NVDA", "--days", "5"])."""
    p = argparse.ArgumentParser(description="Hourly stock volume for the last N trading days.")
    p.add_argument("stock", nargs="?", help="Company name or ticker (asked for if omitted)")
    p.add_argument("--days", type=int, default=5, help="Trading days to show (default 5)")
    p.add_argument("--notional", action="store_true", help="Also show approximate traded value (in the stock's currency)")
    p.add_argument("--csv", action="store_true", help="Save the table to a CSV file named after the ticker")
    p.add_argument("--out", help="Save the table to this CSV path (implies --csv)")
    p.add_argument("--no-peaks", action="store_true",
                   help="Skip the 1-minute data and the peak-minute columns (faster)")
    args = p.parse_args(argv)

    if args.stock:
        ok = run_once(args.stock, args.days, args.notional, args.csv, interactive=False,
                      out=args.out, peaks=not args.no_peaks)
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
            run_once(query, args.days, args.notional, args.csv, interactive=True,
                     out=args.out, peaks=not args.no_peaks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
