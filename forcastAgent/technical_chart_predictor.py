#!/usr/bin/env python3
"""
Technical Chart Predictor (no AI, plain rules)
==============================================
Reads the daily price chart for the last 2 weeks (10 trading days), works out the usual
technical indicators, and gives a central estimate and a likely range for the price
3 and 5 trading days ahead (every day from 1 to 5 is shown).

    pip install yfinance pandas numpy
    python technical_chart_predictor.py NVDA
    python technical_chart_predictor.py "western union" --csv --outdir ../resources
    python technical_chart_predictor.py NVDA --lookback 10 --days 5

In Jupyter:
    import technical_chart_predictor as tcp
    tcp.main(["NVDA"])
    res = tcp.get_forecast("NVDA")      # dict with symbol, indicators, forecast, backtest

HOW IT WORKS
  1. Downloads 1 year of daily bars from Yahoo Finance. A year is needed to warm up the indicators and for the backtest. The
     forecast itself only uses the last --lookback days (default 10 = 2 weeks).
  2. Trend: fits a straight line to log(close) over the last 10 days. Slope = average % move
     per day, R^2 = how cleanly the price followed that line (0 = no trend, 1 = perfect line).
  3. Indicators on the latest bar: SMA 5/10/20/50, MACD (12, 26, 9), RSI 14, Bollinger %B (20, 2),
     ATR 14, and the 2-week high and low (resistance and support). Each indicator gives a vote:
     +1 bullish, -1 bearish, or 0, and the average vote is the "technical bias".
  4. Trend estimate: continues the 2-week trend, DAMPED by R^2 (a messy trend barely moves it)
     and capped at half a typical daily move per day. The range comes from the stock's own
     daily volatility over the last 20 days, widening with the square root of the number of
     days: 68% range = +-1 sigma, 90% range = +-1.645 sigma.
  5. Backtest: the same method is replayed on each of the last ~120 trading days and compared
     with the "no change" guess (future price = today's). The report shows whether the trend
     actually helped and how often the real price landed inside the ranges.
  6. Calibration from the backtest:
       * the central estimate uses the trend only if it beat "no change" at both 3 and 5 days;
         otherwise it is the last close (the trend-only figure stays in the CSV);
       * the ranges are widened (never narrowed) so that, in the backtest, 68% / 90% of real
         prices would have landed inside them. This is fitted on the same data it is checked
         on, so true coverage will be a bit lower.
     --no-backtest skips steps 5 and 6.

READ THIS BEFORE USING THE NUMBERS
  * Short-term prices are close to a random walk. Two weeks of chart data carry very little
    information about the next week. Expect the backtest to show the trend estimate is
    roughly as good as (often slightly worse than) "no change". The RANGE is the useful
    part: it tells you how far the stock typically moves in 3 or 5 days.
  * The range assumes "normal" days. Earnings, news and market shocks can break it.
  * During market hours the latest daily bar is today's unfinished session.
  * Forecast dates skip weekends but not market holidays.
  * Yahoo data is unofficial, for personal use, and can break or be delayed.
  * Not investment advice.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas numpy")

DEFAULT_LOOKBACK = 10      # trading days in "the last 2 weeks"
VOL_WINDOW = 20            # days used for daily volatility
DRIFT_CAP = 0.5            # max central drift per day, in units of daily sigma
Z68, Z90 = 1.0, 1.645
BACKTEST_DAYS = 120


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def resolve_symbol(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if " " not in text and len(text) <= 12:
        return text.upper()
    try:
        quotes = yf.Search(text, max_results=10).quotes
    except Exception as exc:
        print(f"  (name search failed: {exc})")
        return None
    for q in quotes:
        if q.get("quoteType") in ("EQUITY", "ETF") and q.get("symbol"):
            return q["symbol"]
    return None


def fetch_daily(symbol: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)        # keep the exchange's local date
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


# ----------------------------------------------------------------------------
# indicators
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return r.where(dn > 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def trend_fit(close: pd.Series) -> tuple[float, float]:
    """Slope of log(close) per day and R^2 of the straight-line fit."""
    y = np.log(close.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return 0.0, 0.0
    slope, icpt = np.polyfit(x, y, 1)
    fitted = slope * x + icpt
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fitted) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(max(0.0, r2))


def indicators(df: pd.DataFrame, lookback: int) -> tuple[dict, pd.DataFrame]:
    """Latest indicator values and a table of (indicator, value, vote, reading)."""
    c = df["Close"]
    last = float(c.iloc[-1])
    sma = {n: float(c.rolling(n).mean().iloc[-1]) for n in (5, 10, 20, 50)}
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = float((macd - sig).iloc[-1])
    r = float(rsi(c).iloc[-1])
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    pct_b = float(((c - (mid - 2 * sd)) / (4 * sd)).iloc[-1])
    a = float(atr(df).iloc[-1])
    win = df.iloc[-lookback:]
    hi, lo = float(win["High"].max()), float(win["Low"].min())
    slope, r2 = trend_fit(c.iloc[-lookback:])
    chg = (last / float(c.iloc[-lookback - 1]) - 1) * 100 if len(c) > lookback else float("nan")

    def vote(cond_up, cond_dn):
        return 1 if cond_up else (-1 if cond_dn else 0)

    rows = [
        ("Price vs SMA 10", f"{last:.2f} vs {sma[10]:.2f}", vote(last > sma[10], last < sma[10]),
         "above the 2-week average" if last > sma[10] else "below the 2-week average"),
        ("SMA 5 vs SMA 10", f"{sma[5]:.2f} vs {sma[10]:.2f}", vote(sma[5] > sma[10], sma[5] < sma[10]),
         "short average rising above longer" if sma[5] > sma[10] else "short average below longer"),
        ("Price vs SMA 50", f"{last:.2f} vs {sma[50]:.2f}", vote(last > sma[50], last < sma[50]),
         "above the medium-term trend" if last > sma[50] else "below the medium-term trend"),
        ("MACD histogram", f"{hist:+.3f}", vote(hist > 0, hist < 0),
         "momentum up" if hist > 0 else "momentum down"),
        ("RSI 14", f"{r:.1f}", vote(50 < r <= 70 or r < 30, 30 <= r < 50 or r > 70),
         "overbought (pullback risk)" if r > 70 else "oversold (bounce chance)" if r < 30
         else "bullish zone" if r > 50 else "bearish zone"),
        (f"{lookback}-day trend", f"{slope * 100:+.2f}%/day, R2 {r2:.2f}",
         vote(slope > 0 and r2 >= 0.5, slope < 0 and r2 >= 0.5),
         ("clean " if r2 >= 0.5 else "weak/choppy ") + ("uptrend" if slope > 0 else "downtrend")),
        ("Bollinger %B", f"{pct_b:.2f}", 0,
         "at/above upper band" if pct_b >= 1 else "at/below lower band" if pct_b <= 0 else "inside the bands"),
    ]
    table = pd.DataFrame(rows, columns=["indicator", "value", "vote", "reading"])
    vals = dict(last=last, last_date=df.index[-1], sma5=sma[5], sma10=sma[10], sma20=sma[20], sma50=sma[50],
                macd_hist=hist, rsi=r, pct_b=pct_b, atr=a, support=lo, resistance=hi,
                slope=slope, r2=r2, change_pct=chg, score=float(table["vote"].mean()))
    return vals, table


def bias_label(score: float) -> str:
    return "Bullish" if score > 0.3 else "Bearish" if score < -0.3 else "Neutral"


# ----------------------------------------------------------------------------
# forecast
# ----------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def forecast_core(close: pd.Series, lookback: int) -> dict:
    """Damped-trend drift and daily volatility (both in log terms) from closes up to now."""
    slope, r2 = trend_fit(close.iloc[-lookback:])
    rets = np.log(close).diff().dropna().iloc[-VOL_WINDOW:]
    sigma = float(rets.std(ddof=1)) if len(rets) >= 2 else 0.0
    drift = slope * r2
    cap = DRIFT_CAP * sigma
    drift = float(np.clip(drift, -cap, cap)) if cap > 0 else 0.0
    return dict(last=float(close.iloc[-1]), slope=slope, r2=r2, sigma=sigma, drift=drift)


DEFAULT_CAL = dict(use_trend=True, z68=Z68, z90=Z90, calibrated=False)


def build_forecast(core: dict, last_date: pd.Timestamp, days: int, cal: dict = DEFAULT_CAL) -> pd.DataFrame:
    """cal["use_trend"] False -> central = last close ("no change"); z68/z90 = range widths in sigmas."""
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    rows = []
    for h, d in enumerate(dates, 1):
        trend_mu, s = core["drift"] * h, core["sigma"] * math.sqrt(h)
        mu = trend_mu if cal["use_trend"] else 0.0
        p = core["last"]
        rows.append({
            "trading_day": h,
            "date": d.date().isoformat(),
            "central": round(p * math.exp(mu), 2),
            "low_68": round(p * math.exp(mu - cal["z68"] * s), 2),
            "high_68": round(p * math.exp(mu + cal["z68"] * s), 2),
            "low_90": round(p * math.exp(mu - cal["z90"] * s), 2),
            "high_90": round(p * math.exp(mu + cal["z90"] * s), 2),
            "change_pct": round((math.exp(mu) - 1) * 100, 2),
            "prob_up_pct": round(norm_cdf(mu / s) * 100, 1) if s > 0 else 50.0,
            "trend_only_central": round(p * math.exp(trend_mu), 2),
            "central_method": "trend" if cal["use_trend"] else "no change",
        })
    return pd.DataFrame(rows)


def calibrate(samples: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Pick the central method and widen the ranges using the backtest.

    * Trend is used only if it beat 'no change' at every horizon tested.
    * Range widths: the 68th / 90th percentile of |actual - central| / sigma*sqrt(h) in the backtest,
      never narrower than the normal-curve widths (1.0 and 1.645). In-sample, so a little optimistic.
    """
    if samples is None or samples.empty or summary is None or summary.empty:
        return DEFAULT_CAL
    use_trend = bool((summary["trend_error_pct"] < summary["no_change_error_pct"]).all())
    mu = samples["mu"] if use_trend else 0.0
    ratio = ((samples["lr"] - mu).abs() / samples["s"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 20:
        return dict(DEFAULT_CAL, use_trend=use_trend)
    z68 = max(Z68, float(ratio.quantile(0.68)))
    z90 = max(Z90, float(ratio.quantile(0.90)))
    return dict(use_trend=use_trend, calibrated=True, z68=z68, z90=z90,
                cov68=float((ratio <= z68).mean() * 100), cov90=float((ratio <= z90).mean() * 100))


def backtest(close: pd.Series, lookback: int, horizons=(3, 5),
             n: int = BACKTEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the method on past days; compare with the 'no change' guess.
    Returns (summary per horizon, one row per test for calibration)."""
    hmax = max(horizons)
    start = max(lookback, VOL_WINDOW + 1, len(close) - n - hmax)
    rec = {h: [] for h in horizons}
    samples = []
    for t in range(start, len(close) - hmax):
        core = forecast_core(close.iloc[:t + 1], lookback)
        p = core["last"]
        for h in horizons:
            actual = float(close.iloc[t + h])
            mu, s = core["drift"] * h, core["sigma"] * math.sqrt(h)
            central = p * math.exp(mu)
            lr = math.log(actual / p)
            samples.append({"h": h, "lr": lr, "mu": mu, "s": s})
            rec[h].append((
                abs(central - actual) / actual * 100,
                abs(p - actual) / actual * 100,
                (np.sign(mu) == np.sign(lr)) if mu != 0 and lr != 0 else np.nan,
                abs(lr - mu) <= Z68 * s,
                abs(lr - mu) <= Z90 * s,
            ))
    rows = []
    for h in horizons:
        a = np.array(rec[h], dtype=float)
        if len(a) == 0:
            continue
        rows.append({
            "horizon_days": h,
            "tests": len(a),
            "trend_error_pct": round(np.nanmean(a[:, 0]), 2),
            "no_change_error_pct": round(np.nanmean(a[:, 1]), 2),
            "direction_hit_pct": round(np.nanmean(a[:, 2]) * 100, 1) if not np.isnan(a[:, 2]).all() else np.nan,
            "inside_68_pct": round(a[:, 3].mean() * 100, 1),
            "inside_90_pct": round(a[:, 4].mean() * 100, 1),
        })
    return pd.DataFrame(rows), pd.DataFrame(samples)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def print_report(symbol: str, vals: dict, table: pd.DataFrame, fc: pd.DataFrame,
                 bt: pd.DataFrame | None, core: dict, lookback: int, cal: dict = DEFAULT_CAL) -> None:
    print()
    print(f"Technical chart outlook: {symbol}   (last close {vals['last']:.2f} on {vals['last_date']:%Y-%m-%d})")
    print("=" * 96)
    print(f"  Last {lookback} trading days: {vals['change_pct']:+.2f}%   "
          f"support (2-wk low) {vals['support']:.2f}   resistance (2-wk high) {vals['resistance']:.2f}   "
          f"ATR {vals['atr']:.2f}")
    print()
    print(f"  {'Indicator':<18}{'Value':<28}{'Vote':>5}  Reading")
    for _, r in table.iterrows():
        print(f"  {r['indicator']:<18}{r['value']:<28}{r['vote']:>+5d}  {r['reading']}")
    print(f"\n  Technical bias: {bias_label(vals['score'])} (score {vals['score']:+.2f}, from -1 to +1)")
    print(f"  Daily volatility (20 days): {core['sigma'] * 100:.2f}%   "
          f"2-week trend: {core['drift'] * 100:+.3f}%/day (raw {core['slope'] * 100:+.3f}% x R2 {core['r2']:.2f}, capped)")

    if not cal["calibrated"]:
        print("  Central estimate: continues the trend (NOT checked: no backtest was run).")
    elif cal["use_trend"]:
        print("  Central estimate: continues the trend (it beat 'no change' in the backtest).")
    else:
        print("  Central estimate: 'no change' = last close, because the trend did NOT beat it in the backtest.")
        t5 = fc["trend_only_central"].iloc[-1]
        print(f"    (trend alone would say {t5:.2f} on day {len(fc)}; kept in the CSV for reference)")
    if cal["calibrated"]:
        if cal["z68"] > Z68 + 0.02 or cal["z90"] > Z90 + 0.02:
            print(f"  Ranges widened from the backtest: 68% = +-{cal['z68']:.2f} sigma (normal 1.00), "
                  f"90% = +-{cal['z90']:.2f} sigma (normal 1.645).")
        else:
            print("  Ranges: normal widths, which already held enough past prices in the backtest.")
        print(f"  With these settings, past prices landed inside the 68% range {cal['cov68']:.0f}% "
              f"and the 90% range {cal['cov90']:.0f}% of the time (in-sample, so a bit optimistic).")

    print("\n  Forecast")
    print(f"  {'Day':>4}  {'Date':<11}{'Central':>9}{'Change':>8}{'68% range':>20}{'90% range':>20}{'P(up)':>7}")
    for _, r in fc.iterrows():
        mark = " <" if r["trading_day"] in (3, 5) else ""
        print(f"  {r['trading_day']:>4}  {r['date']:<11}{r['central']:>9.2f}{r['change_pct']:>+7.2f}%"
              f"{r['low_68']:>10.2f}-{r['high_68']:<9.2f}{r['low_90']:>10.2f}-{r['high_90']:<9.2f}"
              f"{r['prob_up_pct']:>6.0f}%{mark}")

    for h in (3, 5):
        row = fc[fc["trading_day"] == h]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  In {h} trading days ({r['date']}): about {r['central']:.2f}, "
                  f"likely {r['low_68']:.2f} to {r['high_68']:.2f} (68%), "
                  f"{r['low_90']:.2f} to {r['high_90']:.2f} (90%)")

    if bt is not None and not bt.empty:
        print(f"\n  Backtest on the last ~{BACKTEST_DAYS} trading days (raw trend method, normal ranges, replayed):")
        print(f"  {'Horizon':>8}{'Tests':>7}{'Trend err':>11}{'No-change err':>15}{'Direction hit':>15}"
              f"{'In 68%':>8}{'In 90%':>8}")
        for _, r in bt.iterrows():
            print(f"  {int(r['horizon_days']):>6}d {int(r['tests']):>7}{r['trend_error_pct']:>10.2f}%"
                  f"{r['no_change_error_pct']:>14.2f}%{r['direction_hit_pct']:>14.1f}%"
                  f"{r['inside_68_pct']:>7.0f}%{r['inside_90_pct']:>7.0f}%")
        print("  -> " + ("The trend beat 'no change' here, so it is used, but the edge is small; don't over-trust it."
                         if cal["use_trend"] else
                         "The trend did NOT beat 'no change' here, so the central estimate is the last close. "
                         "The ranges are the useful part."))
    print("\n  Rule-based estimate from 2 weeks of chart data, not a tested trading signal. Not investment advice.")


# ----------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------
def analyze(symbol: str, lookback: int = DEFAULT_LOOKBACK, days: int = 5, run_backtest: bool = True) -> dict:
    df = fetch_daily(symbol)
    if df.empty or len(df) < max(lookback, VOL_WINDOW) + 2:
        raise ValueError(f"Not enough daily data for '{symbol}'. Check the ticker "
                         "(non-US stocks need a suffix, e.g. HCLTECH.NS).")
    vals, table = indicators(df, lookback)
    core = forecast_core(df["Close"], lookback)
    bt, samples = backtest(df["Close"], lookback) if run_backtest else (None, None)
    cal = calibrate(samples, bt)
    fc = build_forecast(core, df.index[-1], days, cal)
    return dict(symbol=symbol, values=vals, indicators=table, core=core, forecast=fc, backtest=bt,
                calibration=cal)


def get_forecast(query: str, lookback: int = DEFAULT_LOOKBACK, days: int = 5) -> dict:
    """Jupyter / app helper. Returns the analyze() dict, or {} if the stock isn't found."""
    symbol = resolve_symbol(query)
    return analyze(symbol, lookback, days) if symbol else {}


def save(res: dict, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"{res['symbol'].replace('.', '_')}_technical")
    v, sym = res["values"], res["symbol"]
    fc = res["forecast"].copy()
    fc.insert(0, "symbol", sym)
    fc.insert(1, "last_close", round(v["last"], 2))
    fc.insert(2, "last_date", f"{v['last_date']:%Y-%m-%d}")
    fc["technical_bias"] = bias_label(v["score"])
    fc["bias_score"] = round(v["score"], 2)
    paths = [base + "_forecast.csv", base + "_indicators.csv"]
    fc.to_csv(paths[0], index=False)
    ind = res["indicators"].copy()
    ind.insert(0, "symbol", sym)
    ind.to_csv(paths[1], index=False)
    if res["backtest"] is not None and not res["backtest"].empty:
        paths.append(base + "_backtest.csv")
        res["backtest"].assign(symbol=sym).to_csv(paths[2], index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    """argv=None reads the command line. In Jupyter pass a list, e.g. main(["NVDA"])."""
    p = argparse.ArgumentParser(description="Technical-chart outlook for the next 3 and 5 trading days.")
    p.add_argument("stock", help="Company name or ticker")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                   help="Trading days of chart used for the trend (default 10 = 2 weeks)")
    p.add_argument("--days", type=int, default=5, help="Trading days to forecast (default 5)")
    p.add_argument("--csv", action="store_true", help="Save forecast, indicators and backtest CSVs")
    p.add_argument("--outdir", default=".", help="Folder for the CSVs (default: current folder)")
    p.add_argument("--no-backtest", action="store_true", help="Skip the backtest (faster)")
    args = p.parse_args(argv)
    if args.lookback < 3 or args.days < 1:
        print("--lookback must be at least 3 and --days at least 1.")
        return 1

    symbol = resolve_symbol(args.stock)
    if not symbol:
        print(f"  Couldn't find a stock matching '{args.stock}'. Try the ticker code.")
        return 1
    print(f"  Fetching daily chart for {symbol} ...")
    try:
        res = analyze(symbol, args.lookback, max(args.days, 5), not args.no_backtest)
    except Exception as exc:
        print(f"  {exc}")
        return 1
    res["forecast"] = res["forecast"].head(args.days)
    print_report(symbol, res["values"], res["indicators"], res["forecast"], res["backtest"],
                 res["core"], args.lookback, res["calibration"])
    if args.csv:
        print()
        for path in save(res, args.outdir):
            print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
