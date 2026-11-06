"""
"Outside opinions" page of the Streamlit app: Wall Street analyst ratings and price targets
(free, Yahoo) plus optional AI / quant ratings found with Claude web search (costs money).

AI web research needs an Anthropic API key. Either:
  * the visitor pastes their own key in the sidebar (used for that session only, never stored), or
  * the app owner sets ANTHROPIC_API_KEY and APP_PASSWORD in the app's Secrets; visitors who know
    the password can then use the owner's key. Without APP_PASSWORD the owner's key is only used
    when the app is opened on this computer (localhost), so a public app can't spend it.
"""

import contextlib
import io
import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

import run_pipeline as rp
import outside_opinions_agent as ooa  # forcastAgent/ is on sys.path via run_pipeline

sva = rp.sva


# ----------------------------------------------------------------------------
# data (cached so repeat visits don't hit Yahoo or the paid API again)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_consensus(symbol: str, days: int) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return ooa.analyst_consensus(symbol, days)


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=50)
def get_web(symbol: str, days: int, effort: str, max_searches: int, _cons: dict, _api_key: str) -> dict:
    """Paid call. Cached 6 hours per (symbol, days, effort, max_searches) and shared by all visitors."""
    with contextlib.redirect_stdout(io.StringIO()):
        return ooa.ask_web(_cons, ooa.DEFAULT_MODEL, effort, max_searches, api_key=_api_key)


def resolve(query: str) -> tuple[str | None, dict | None]:
    """Ticker or company name -> (symbol, consensus). A single word like 'nvidia' that isn't a ticker
    is retried as a company-name search."""
    with contextlib.redirect_stdout(io.StringIO()):
        symbol = ooa.resolve_symbol(query)
    if symbol:
        cons = get_consensus(symbol, st.session_state.get("oo_days", ooa.UPGRADE_DAYS))
        if cons.get("price") is not None or cons.get("target_mean") is not None:
            return symbol, cons
    with contextlib.redirect_stdout(io.StringIO()):
        matches = sva.search_candidates(query.strip())
    for m in matches:
        cons = get_consensus(m["symbol"], st.session_state.get("oo_days", ooa.UPGRADE_DAYS))
        if cons.get("price") is not None or cons.get("target_mean") is not None:
            return m["symbol"], cons
    return None, None


# ----------------------------------------------------------------------------
# API key access for the paid AI part
# ----------------------------------------------------------------------------
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:          # no secrets file
        return ""


def _is_local() -> bool:
    try:
        host = (st.context.headers.get("Host") or "").split(":")[0]
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


def api_key_ui() -> str | None:
    """Sidebar controls shared by the pages that call Claude. Returns an API key to use, or None.
    Once unlocked, the key is kept for this browser session only (so switching pages doesn't ask again)."""
    owner_key = _secret("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    password = _secret("APP_PASSWORD")
    with st.sidebar:
        st.subheader("AI access (Claude)")
        st.caption("Free data always works. The AI parts (web research, news reading) use an Anthropic "
                   "API key and are billed per run.")
        if st.session_state.get("ai_key"):
            st.success(st.session_state.get("ai_key_source", "AI access unlocked."))
            if st.button("Lock AI access", key="ai_lock"):
                st.session_state.pop("ai_key", None)
                st.rerun()
            return st.session_state["ai_key"]
        if owner_key and password:
            entered = st.text_input("App password", type="password", key="oo_pw")
            if entered and entered == password:
                st.session_state["ai_key"], st.session_state["ai_key_source"] = owner_key, "Using the app owner's API key."
                st.success("Using the app owner's API key.")
                return owner_key
            if entered:
                st.error("Wrong password.")
        elif owner_key and _is_local():
            st.success("Using the API key set on this computer.")
            return owner_key
        own = st.text_input("Or paste your own Anthropic API key", type="password", key="oo_key",
                            help="Used only for this browser session to call Claude. Not saved or logged.")
        if own.strip():
            st.session_state["ai_key"], st.session_state["ai_key_source"] = own.strip(), "Using your own API key."
            return own.strip()
        return None


# ----------------------------------------------------------------------------
# display helpers
# ----------------------------------------------------------------------------
def _money(v, cur=""):
    return "-" if v is None or pd.isna(v) else f"{v:,.2f}{(' ' + cur) if cur else ''}"


def price_target_chart(cons: dict, ch: pd.DataFrame, src: pd.DataFrame | None):
    closes = cons.get("closes")
    if closes is None or closes.empty:
        return None
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=max(cons.get("days", 90) + 30, 180))
    price = closes[closes.index >= start].rename("price").rename_axis("date").reset_index()
    layers = [alt.Chart(price).mark_line().encode(
        x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=",.2f")])]

    pts = []
    if not ch.empty:
        t = ch.dropna(subset=["target"])
        for _, r in t.iterrows():
            pts.append({"date": r["predicted_on"], "target": r["target"], "who": r["firm"], "kind": "Wall Street",
                        "target_by": r["target_by"], "rating": r["to_rating"]})
    if src is not None and not src.empty:
        for _, r in src.dropna(subset=["price_target"]).iterrows():
            if r.get("as_of"):
                pts.append({"date": r["as_of"], "target": r["price_target"], "who": r["source"],
                            "kind": "AI / online", "target_by": r["target_by"], "rating": r["rating"]})
    if pts:
        p = pd.DataFrame(pts)
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        p = p.dropna(subset=["date"])
        layers.append(alt.Chart(p).mark_point(filled=True, size=70).encode(
            x="date:T", y="target:Q",
            color=alt.Color("kind:N", title="Price target from"),
            tooltip=[alt.Tooltip("who:N", title="Source"), alt.Tooltip("rating:N", title="Rating"),
                     alt.Tooltip("date:T", title="Predicted on"), alt.Tooltip("target:Q", format=",.2f"),
                     alt.Tooltip("target_by:N", title="Target by")]))
    if cons.get("target_mean"):
        layers.append(alt.Chart(pd.DataFrame({"y": [cons["target_mean"]]})).mark_rule(strokeDash=[6, 4]).encode(
            y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Mean analyst target", format=",.2f")]))
    return alt.layer(*layers).properties(height=360)


def csv_bytes(cons: dict, web: dict | None) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        paths = ooa.save(cons, web, tmp)
        out = {}
        for p in paths:
            with open(p, "rb") as f:
                out[os.path.basename(p)] = f.read()
    return out


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
def render() -> None:
    st.title("🧭 Outside opinions")
    st.caption("What Wall Street analysts and public AI / quant rating sites say about a stock: each rating, "
               "the date it was predicted, the price target, and when that price is meant to be reached.")
    api_key = api_key_ui()

    with st.form("outside"):
        c1, c2 = st.columns([3, 1])
        query = c1.text_input("Ticker or company name", placeholder="NVDA, nvidia, HCLTECH.NS", key="oo_query")
        days = c2.number_input("Analyst changes from the last N days", 7, 365, ooa.UPGRADE_DAYS, key="oo_days")
        c3, c4 = st.columns([3, 1])
        use_ai = c3.checkbox("Also search the web for AI / quant ratings (uses Claude; costs money per run)",
                             disabled=api_key is None,
                             help=None if api_key else "Enter a password or your own API key in the sidebar first.")
        effort = c4.selectbox("Research depth", ["low", "medium", "high"], index=1, disabled=api_key is None)
        go = st.form_submit_button("Get opinions", type="primary")

    if go:
        if not query.strip():
            st.warning("Type a ticker or company name.")
            return
        with st.spinner("Getting Wall Street analyst data ..."):
            symbol, cons = resolve(query)
        if not symbol:
            st.error(f"Couldn't find a stock matching '{query}'. Try the ticker code, e.g. HCLTECH.NS.")
            return
        web, err = None, None
        if use_ai and api_key:
            with st.spinner(f"Searching the web for AI / quant ratings of {symbol} (about a minute) ..."):
                try:
                    web = get_web(symbol, int(days), effort, ooa.MAX_SEARCHES, cons, api_key)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
        st.session_state["oo_result"] = (symbol, cons, web, err)

    if "oo_result" not in st.session_state:
        return
    symbol, cons, web, err = st.session_state["oo_result"]
    cur, price = cons.get("currency", ""), cons.get("price")

    # ---- consensus
    st.header(f"{cons.get('name', symbol)} ({symbol})")
    m = st.columns(5)
    m[0].metric("Last price", _money(price, cur))
    m[1].metric("Analyst consensus", (cons.get("recommendation") or "-").title(),
                help="Yahoo's average rating on a 1 (strong buy) to 5 (strong sell) scale: "
                     f"{cons.get('recommendation_mean') or '-'}")
    m[2].metric("Analysts", cons.get("analysts") or "-")
    if cons.get("target_mean") and price:
        m[3].metric("Mean 12-month target", _money(cons["target_mean"]),
                    f"{(cons['target_mean'] / price - 1) * 100:+.1f}% vs price")
    m[4].metric("Target range (low – high)", f"{_money(cons.get('target_low'))} – {_money(cons.get('target_high'))}")

    counts = {k: cons.get(k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
    if any(v is not None for v in counts.values()):
        dist = pd.DataFrame({"rating": ["Strong buy", "Buy", "Hold", "Sell", "Strong sell"],
                             "analysts": [v or 0 for v in counts.values()]})
        st.altair_chart(alt.Chart(dist).mark_bar().encode(
            x=alt.X("rating:N", sort=None, title=None), y=alt.Y("analysts:Q", title="Analysts"),
            tooltip=["rating", "analysts"]).properties(height=180), width="stretch")

    ch = ooa.enrich_changes(cons, web)
    src = ooa.enrich_sources(cons, web) if web else None

    st.subheader("Price and price targets")
    chart = price_target_chart(cons, ch, src)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption("Line: stock price. Dots: each price target, placed on the date it was predicted. "
                   "Dashed line: mean Wall Street target. Hover a dot for the source and its target-by date.")

    # ---- individual analyst ratings
    st.subheader(f"Wall Street ratings, last {cons.get('days', days)} days ({len(ch)})")
    if ch.empty:
        st.info("No rating or price-target changes in this period.")
    else:
        show = ch[["predicted_on", "firm", "action", "from_rating", "to_rating", "price_then", "prior_target",
                   "target", "upside_then_pct", "upside_now_pct", "target_by", "days_left", "horizon_basis",
                   "analyst", "reason", "reason_factors", "reason_url"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "predicted_on": st.column_config.TextColumn("Predicted on", help="Date the firm published it"),
            "firm": "Firm", "action": "Action", "from_rating": "From", "to_rating": "To",
            "price_then": st.column_config.NumberColumn("Price then", format="%.2f",
                                                        help="Stock price on the day it was predicted"),
            "prior_target": st.column_config.NumberColumn("Prior target", format="%.0f"),
            "target": st.column_config.NumberColumn("Target", format="%.0f"),
            "upside_then_pct": st.column_config.NumberColumn("Upside from then", format="%+.1f%%"),
            "upside_now_pct": st.column_config.NumberColumn("Upside from today", format="%+.1f%%"),
            "target_by": st.column_config.TextColumn("Target by", help="When the price is meant to be reached"),
            "days_left": st.column_config.NumberColumn("Days left"),
            "horizon_basis": st.column_config.TextColumn("Horizon", help="Stated by the source, or the "
                                                         "12-month industry standard (assumed)"),
            "analyst": "Analyst", "reason": "Reason", "reason_factors": "Based on",
            "reason_url": st.column_config.LinkColumn("Reason source"),
        })
        st.caption("Yahoo gives firm, date, rating and target. Wall Street targets are normally 12-month views; "
                   "a horizon stated in news coverage replaces that assumption. Analyst names and reasons are "
                   "filled in by the AI web research when it finds them.")

    # ---- AI / online
    st.subheader("AI / quant ratings and online forecasts")
    if err:
        st.error(f"The AI web research failed: {err}")
    elif web is None:
        st.info("Not run. Tick 'Also search the web' above (needs a password or your own API key in the sidebar).")
    elif src is None or src.empty:
        st.info("No verifiable public ratings were found.")
    else:
        tally = src["stance"].value_counts()
        t = st.columns(4)
        for col, k in zip(t, ["bullish", "bearish", "neutral", "unclear"]):
            col.metric(k.title(), int(tally.get(k, 0)))
        if web.get("summary"):
            st.info(web["summary"])
        for _, r in src.iterrows():
            pt = r.get("price_target")
            has_pt = pt is not None and not pd.isna(pt)
            title = (f"{r['source']} — {r['rating']} ({str(r['stance']).upper()})"
                     + (f" · target {pt:,.2f}" if has_pt else "")
                     + (f" by {r['target_by']}" if r["target_by"] else ""))
            with st.expander(title):
                a, b = st.columns(2)
                a.markdown(f"**Institution:** {r.get('institution') or '-'}  \n"
                           f"**Agent / analyst:** {r.get('author') or 'not stated'}  \n"
                           f"**Type:** {str(r['source_type']).replace('_', ' ')}")
                when = "-"
                if r["days_left"] is not None and not pd.isna(r["days_left"]):
                    d = int(r["days_left"])
                    when = f"{d} days from today" if d >= 0 else f"expired {-d} days ago"
                b.markdown(f"**Predicted on:** {r.get('as_of') or 'unknown'}"
                           + (f" (price then {r['price_then']:,.2f})" if r["price_then"] else "") + "  \n"
                           f"**Target:** {_money(pt) if has_pt else '-'}"
                           + (f" ({r['upside_then_pct']:+.1f}% from then, {r['upside_now_pct']:+.1f}% from today)"
                              if has_pt and r["upside_then_pct"] is not None and not pd.isna(r["upside_then_pct"])
                              else "") + "  \n"
                           f"**Target by:** {r['target_by'] or 'not stated'} "
                           f"({r.get('horizon') or 'no horizon given'}; {when})")
                st.markdown(f"**Data used:** {ooa._factors(r.get('factors'))}")
                if r.get("methodology"):
                    st.markdown(f"**Method:** {r['methodology']}")
                if r.get("reasoning"):
                    st.markdown(f"**Reasons given:** {r['reasoning']}")
                if r.get("evidence"):
                    st.markdown(f"> {r['evidence']}")
                if r.get("note"):
                    st.caption(r["note"])
                st.markdown(f"[Open source page]({r['url']})  "
                            + ("✅ link came from the web search" if r["link_verified"]
                               else "⚠️ unverified link — not from the search results"))
        u = web["usage"]
        st.caption(f"{web['model']}: {u['input_tokens']:,} input + {u['output_tokens']:,} output tokens, "
                   f"{u['web_searches']} web searches. Results are cached for 6 hours.")

    # ---- downloads
    files = csv_bytes(cons, web)
    if files:
        st.subheader("Download")
        cols = st.columns(len(files))
        for col, (name, data) in zip(cols, files.items()):
            col.download_button(name, data, name, "text/csv", key=f"oo_{name}")
    st.caption("Other people's opinions, collected as-is and not checked for accuracy. Not investment advice.")
