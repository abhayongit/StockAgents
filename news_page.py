"""
"News outlook" page of the Streamlit app: recent headlines (Yahoo, Google News, Seeking Alpha RSS)
read by Claude, turned into a news bias and a 5-day price range by plain math.

Reading the news uses an Anthropic API key, so it uses the same access controls as the
Outside opinions page (your own key, or the owner's key behind APP_PASSWORD).
"""

import contextlib
import io

import altair as alt
import pandas as pd
import streamlit as st

import outside_page            # shared sidebar API-key controls
import run_pipeline as rp
import stock_news_predictor as snp   # forcastAgent/ is on sys.path via run_pipeline

sva = rp.sva


# ----------------------------------------------------------------------------
# data (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_market(symbol: str) -> dict:
    return snp.fetch_market(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol: str, seeking_alpha: bool) -> tuple[list[dict], str]:
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        news = snp.fetch_news(symbol, seeking_alpha=seeking_alpha)
    return news, log.getvalue()


@st.cache_data(ttl=3 * 3600, show_spinner=False, max_entries=50)
def get_analysis(symbol: str, seeking_alpha: bool, model: str, _market: dict, _news: list[dict],
                 _api_key: str) -> dict:
    """Paid call. Cached 3 hours per (symbol, news sources, model) and shared by all visitors."""
    return snp.ask_claude(symbol, _market, _news, model, api_key=_api_key)


def resolve(query: str) -> tuple[str | None, dict | None]:
    """Ticker or company name -> (symbol, market data)."""
    with contextlib.redirect_stdout(io.StringIO()):
        symbol = snp.resolve_symbol(query)
    candidates = [symbol] if symbol else []
    with contextlib.redirect_stdout(io.StringIO()):
        candidates += [m["symbol"] for m in sva.search_candidates(query.strip())]
    for s in dict.fromkeys(candidates):          # keep order, drop repeats
        try:
            return s, get_market(s)
        except Exception:
            continue
    return None, None


def forecast_chart(market: dict, fc: pd.DataFrame):
    today = pd.DataFrame([{"date": market["last_date"], "expected": market["price"],
                           "low_80": market["price"], "high_80": market["price"]}])
    d = pd.concat([today, fc[["date", "expected", "low_80", "high_80"]]], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    band = alt.Chart(d).mark_area(opacity=0.25).encode(
        x=alt.X("date:T", title=None), y=alt.Y("low_80:Q", title="Price", scale=alt.Scale(zero=False)),
        y2="high_80:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("low_80:Q", title="80% low", format=",.2f"),
                                 alt.Tooltip("high_80:Q", title="80% high", format=",.2f")])
    line = alt.Chart(d).mark_line(point=True).encode(
        x="date:T", y="expected:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("expected:Q", format=",.2f")])
    rule = alt.Chart(pd.DataFrame({"y": [market["price"]]})).mark_rule(strokeDash=[6, 4]).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Last close", format=",.2f")])
    return alt.layer(band, line, rule).properties(height=320)


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
def render() -> None:
    st.title("📰 News outlook")
    st.caption("Claude reads recent headlines and scores the news mood. Plain math then turns that into an "
               "expected path and an 80% range for the next 5 trading days.")
    api_key = outside_page.api_key_ui()

    with st.form("news"):
        c1, c2 = st.columns([3, 1])
        query = c1.text_input("Ticker or company name", placeholder="NVDA, nvidia, HCLTECH.NS", key="nw_query")
        sa = c2.checkbox("Include Seeking Alpha RSS", value=True, key="nw_sa",
                         help="Headlines and short blurbs from the public feed only; skipped if refused.")
        go = st.form_submit_button("Read the news", type="primary", disabled=api_key is None)
    if api_key is None:
        st.info("Reading the news uses Claude. Enter the app password or your own Anthropic API key in the "
                "sidebar to unlock this page.")

    if go and api_key:
        if not query.strip():
            st.warning("Type a ticker or company name.")
            return
        with st.spinner("Getting price data ..."):
            symbol, market = resolve(query)
        if not symbol:
            st.error(f"Couldn't find price data for '{query}'. Try the ticker code, e.g. HCLTECH.NS.")
            return
        with st.spinner(f"Collecting headlines for {symbol} ..."):
            news, log = get_news(symbol, sa)
        if not news:
            st.error("No headlines came back from any source, so there is nothing to base an outlook on.")
            return
        with st.spinner(f"Claude ({snp.DEFAULT_MODEL}) is reading {len(news)} headlines ..."):
            try:
                analysis = get_analysis(symbol, sa, snp.DEFAULT_MODEL, market, news, api_key)
            except SystemExit as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"The news analysis failed: {type(exc).__name__}: {exc}")
                return
        fc = snp.build_forecast(market["price"], market["sigma"], analysis["bias"], analysis["event_risk"],
                                market["last_date"], 5)
        st.session_state["nw_result"] = (symbol, market, news, log, analysis, fc)

    if "nw_result" not in st.session_state:
        return
    symbol, market, news, log, analysis, fc = st.session_state["nw_result"]

    st.header(symbol)
    m = st.columns(4)
    m[0].metric("Last close", f"{market['price']:,.2f}", f"{market['chg_1d_pct']:+.2f}% 1-day",
                help=f"on {market['last_date']:%Y-%m-%d}")
    m[1].metric("5-day change", "-" if pd.isna(market["chg_5d_pct"]) else f"{market['chg_5d_pct']:+.2f}%")
    m[2].metric("Typical daily move", f"{market['sigma'] * 100:.2f}%", help="Std dev of daily moves, ~3 months")
    m[3].metric("Headlines read", len(news))
    m = st.columns(3)
    m[0].metric("News bias (-1 to +1)", f"{analysis['bias']:+.2f}", snp.direction_word(analysis["bias"]),
                delta_color="off")
    m[1].metric("Confidence", analysis["confidence"].title())
    m[2].metric("Event risk", analysis["event_risk"].title(), help="High event risk widens the range")
    if analysis["summary"]:
        st.info(analysis["summary"])
    cols = st.columns(3)
    for col, (title, key) in zip(cols, (("📈 Could push up", "drivers"), ("📉 Could push down", "risks"),
                                        ("📅 Upcoming events", "upcoming_events"))):
        col.markdown(f"**{title}**")
        col.markdown("\n".join(f"- {x}" for x in analysis[key]) if analysis[key] else "_none mentioned_")

    st.subheader("Next 5 trading days")
    st.altair_chart(forecast_chart(market, fc), width="stretch")
    st.caption("Line: expected price. Shaded band: 80% range (the price should land inside about 4 times in 5). "
               "Dashed line: last close. News can nudge the path by at most 0.3 of a typical daily move per day.")
    st.dataframe(fc, hide_index=True, width="stretch", column_config={
        "day": "Day", "date": "Date",
        "expected": st.column_config.NumberColumn("Expected", format="%.2f"),
        "low_80": st.column_config.NumberColumn("80% low", format="%.2f"),
        "high_80": st.column_config.NumberColumn("80% high", format="%.2f"),
        "expected_chg_pct": st.column_config.NumberColumn("Change vs today", format="%+.2f%%"),
        "prob_above_today": st.column_config.NumberColumn("Chance above today", format="%.1f%%"),
    })

    st.subheader(f"Headlines read ({len(news)})")
    st.dataframe(pd.DataFrame(news)[["published", "source", "title", "summary"]], hide_index=True,
                 width="stretch")
    with st.expander("Sources log"):
        st.code(log.strip() or "(empty)")

    out = fc.copy()
    out.insert(0, "symbol", symbol)
    out["news_bias"], out["confidence"], out["event_risk"] = (analysis["bias"], analysis["confidence"],
                                                              analysis["event_risk"])
    c = st.columns(2)
    c[0].download_button(f"{symbol}_news_outlook.csv", out.to_csv(index=False).encode(),
                         f"{symbol}_news_outlook.csv", "text/csv", key="nw_dl_fc")
    c[1].download_button(f"{symbol}_headlines.csv", pd.DataFrame(news).to_csv(index=False).encode(),
                         f"{symbol}_headlines.csv", "text/csv", key="nw_dl_news")
    st.caption("Markets react to news within seconds, so most public news is already in the price. Treat this "
               "as a summary of the news mood with a realistic range, not a forecast to trade on. "
               "Headlines are untrusted web text. Not investment advice.")
