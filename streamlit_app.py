"""
Web page for the StockAgents pipeline.

    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import contextlib
import io
import os
import tempfile
import zipfile

import pandas as pd
import streamlit as st

import run_pipeline as rp

sva = rp.sva  # stock volume agent module

st.set_page_config(page_title="Stock Support Agents", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def run(symbol: str, days: int, window: int) -> tuple[bool, str, dict[str, bytes]]:
    """Run both agents in a temp folder. Returns (ok, console log, {filename: csv bytes}).
    Cached for 15 minutes per (symbol, days, window) so repeat visits don't hit Yahoo again."""
    log = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(log):
        try:
            ok = rp.run_for(symbol, days, tmp, window)
        except SystemExit as exc:      # the agents use SystemExit for bad input
            print(exc)
            ok = False
        except Exception as exc:
            print(f"Error: {exc}")
            ok = False
        files = {}
        for name in sorted(os.listdir(tmp)):
            with open(os.path.join(tmp, name), "rb") as f:
                files[name] = f.read()
    return ok, log.getvalue(), files


def read(files: dict[str, bytes], suffix: str) -> pd.DataFrame | None:
    name = next((n for n in files if n.endswith(suffix)), None)
    return pd.read_csv(io.BytesIO(files[name])) if name else None


# ----------------------------------------------------------------------------
st.title("📈 Stock Support Agents")
st.caption("Hourly volume from Yahoo Finance, then a Bullish / Bearish / Neutral signal and support price for each hour.")

with st.form("query"):
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Ticker or company name", placeholder="NVDA, nvidia, HCLTECH.NS")
    days = c2.number_input("Trading days", 1, 30, 5)
    window = c3.number_input("Std-dev window (hours)", 1, 50, 7)
    submitted = st.form_submit_button("Run", type="primary")

if submitted:
    if not query.strip():
        st.warning("Type a ticker or company name.")
        st.stop()
    with st.spinner("Looking up the stock ..."):
        with contextlib.redirect_stdout(io.StringIO()):
            symbol = sva.resolve_symbol(query, interactive=False)
    if not symbol:
        st.error(f"Couldn't find a stock matching '{query}'. Try the ticker code, e.g. HCLTECH.NS.")
        st.stop()
    with st.spinner(f"Fetching data for {symbol} and running both agents ..."):
        ok, log, files = run(symbol, int(days), int(window))
    if not ok and sva.looks_like_ticker(query.strip()):
        # A single word like "nvidia" is first tried as a ticker; if that fails, search it as a name.
        with contextlib.redirect_stdout(io.StringIO()):
            matches = sva.search_candidates(query.strip())
        if matches and matches[0]["symbol"] != symbol:
            symbol = matches[0]["symbol"]
            with st.spinner(f"Trying {symbol} ({matches[0]['name']}) ..."):
                ok, log, files = run(symbol, int(days), int(window))
    st.session_state["result"] = (symbol, ok, log, files)

if "result" in st.session_state:
    symbol, ok, log, files = st.session_state["result"]
    if not ok:
        st.error(f"The pipeline didn't finish for {symbol}. See the log below.")
    else:
        sig = read(files, "_signals.csv")
        if sig is not None and not sig.empty:
            last = sig.iloc[-1]
            counts = sig["signal"].value_counts()
            m = st.columns(5)
            m[0].metric("Symbol", symbol)
            m[1].metric(f"Latest hour ({last['date']} {last['Hour starting']})", last["signal"])
            m[2].metric("Support price", "-" if pd.isna(last["support_price"]) else f"{last['support_price']:.2f}")
            m[3].metric("Std dev", "-" if pd.isna(last["std_dev"]) else f"{last['std_dev']:.2f}")
            m[4].metric("Hours B / Bear / N",
                        f"{counts.get('Bullish', 0)} / {counts.get('Bearish', 0)} / {counts.get('Neutral', 0)}")

            chart = sig.assign(time=sig["date"].astype(str) + " " + sig["Hour starting"].astype(str))
            chart = chart.set_index("time")[["highest_price", "lowest_price", "support_price"]]
            st.line_chart(chart)

        tabs = st.tabs(["Signals", "Hourly volume", "Daily totals", "Peak minutes"])
        for tab, suffix in zip(tabs, ["_signals.csv", "_hourly_volume.csv", "_daily.csv", "_peak_minutes.csv"]):
            df = read(files, suffix)
            with tab:
                if df is None:
                    st.info("Not available for this stock.")
                else:
                    st.dataframe(df, width="stretch", hide_index=True)

        st.subheader("Download")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
        st.download_button("⬇️ All CSVs (.zip)", zbuf.getvalue(), f"{symbol}_stock_agents.zip",
                           "application/zip", type="primary")
        cols = st.columns(2)
        for i, (name, data) in enumerate(files.items()):
            cols[i % 2].download_button(name, data, name, "text/csv", key=name)

    with st.expander("Console log"):
        st.code(log or "(empty)")

st.divider()
st.caption("Data from Yahoo Finance via the unofficial yfinance library, for personal use. "
           "Bought and sold volumes are estimates. 1-minute data covers only about the last 7 days. "
           "Not investment advice.")
