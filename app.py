"""CPG Customer Feedback Summarizer — Streamlit entrypoint.

Upload feedback -> Analyze (PII scrub -> extraction -> aggregation ->
executive summary) -> dashboard + grounded chat Q&A.

Input is upload-only (.csv / .json). Credentials come only from environment
variables / Streamlit secrets. Raw feedback text is PII-scrubbed before the
only model call that sees it (Call 1); Calls 2 and 3 receive aggregated data
only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import azure_client as az
import pipeline as pl

st.set_page_config(
    page_title="CPG Feedback Summarizer",
    page_icon="📊",
    layout="wide",
)


# --- Session state ------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("records", None)
    st.session_state.setdefault("source_label", None)
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("dark_mode", False)


_init_state()


# --- Theme (creative violet/teal/coral palette, validated for CVD) ------------
# Two full palettes: light and dark, each dataviz-validated. The dark-mode
# toggle drives CSS variables so every surface, text token, and chart re-themes.
DARK = st.session_state.dark_mode

if DARK:
    C = {
        "bg": "#0E1117",
        "surface": "#171B26",
        "surface2": "#1F2430",
        "border": "#2A3040",
        "ink": "#ECEFF6",
        "ink2": "#A9B2C4",
        "accent": "#7B6EF0",   # violet
        "teal": "#12A883",
        "coral": "#DE6A4E",
        "blue": "#2C7FC7",
        "amber": "#B8891F",
        "grad1": "#7B6EF0",
        "grad2": "#DE6A4E",
    }
else:
    C = {
        "bg": "#FAF9FC",
        "surface": "#FFFFFF",
        "surface2": "#F3F1FB",
        "border": "#E7E3F5",
        "ink": "#1A1626",
        "ink2": "#6B6580",
        "accent": "#6C5CE7",   # violet
        "teal": "#00A383",
        "coral": "#E17055",
        "blue": "#0984E3",
        "amber": "#D4A017",
        "grad1": "#6C5CE7",
        "grad2": "#E17055",
    }

# Status colors for sentiment (consistent, meaning-carrying).
SENTIMENT_COLORS = {
    "positive": C["teal"],
    "negative": C["coral"],
    "neutral": C["blue"],
}

st.markdown(
    f"""
    <style>
      :root {{
        --bg: {C['bg']}; --surface: {C['surface']}; --surface2: {C['surface2']};
        --border: {C['border']}; --ink: {C['ink']}; --ink2: {C['ink2']};
        --accent: {C['accent']};
      }}
      .stApp {{ background: var(--bg); color: var(--ink); }}
      #MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }}

      h1,h2,h3,h4,h5,h6, p, span, label, li {{ color: var(--ink); }}
      .stMarkdown p {{ color: var(--ink); }}

      /* Hero header */
      .hero {{
        background: linear-gradient(120deg, {C['grad1']} 0%, {C['grad2']} 100%);
        border-radius: 18px; padding: 26px 30px; margin-bottom: 8px;
        color: #fff; box-shadow: 0 10px 30px -12px {C['grad1']}66;
      }}
      .hero h1 {{ color:#fff; font-size:2.0rem; font-weight:800; margin:0; letter-spacing:-0.5px; }}
      .hero p {{ color:#ffffffdd; margin:6px 0 0; font-size:1.02rem; }}

      /* Headline banner (Call 2 output) */
      .headline {{
        background: var(--surface); border:1px solid var(--border);
        border-left: 6px solid var(--accent);
        border-radius: 14px; padding: 20px 24px; margin: 8px 0 6px;
        font-size: 1.45rem; font-weight: 750; line-height:1.3; color: var(--ink);
        box-shadow: 0 6px 20px -14px #00000055;
      }}
      .headline .tag {{
        display:inline-block; font-size:0.7rem; font-weight:700; letter-spacing:0.14em;
        text-transform:uppercase; color: var(--accent); margin-bottom:6px;
      }}

      /* KPI metric cards */
      div[data-testid="stMetric"] {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 4px 16px -12px #00000040;
      }}
      div[data-testid="stMetric"] label p {{ font-size:0.82rem; color: var(--ink2); }}
      div[data-testid="stMetricValue"] {{ color: var(--accent); font-weight:800; }}

      /* Section cards */
      .card {{
        background: var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:18px 22px; margin-bottom:6px;
      }}
      .sec-label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.12em;
        text-transform:uppercase; color: var(--ink2); margin-bottom:2px; }}

      /* Theme tag pills */
      .themebar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }}
      .pill {{
        display:inline-flex; align-items:center; gap:6px;
        background: var(--surface2); border:1px solid var(--border);
        color: var(--ink); border-radius:999px; padding:5px 12px; font-size:0.86rem;
      }}
      .pill b {{ color: var(--accent); }}

      /* Buttons */
      .stButton > button, .stDownloadButton > button {{
        border-radius: 10px; font-weight:600;
      }}
      div[data-testid="stFileUploader"] {{
        background: var(--surface); border:1px dashed var(--border);
        border-radius:14px; padding:8px 12px;
      }}
      hr {{ border-color: var(--border); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header + dark-mode toggle ------------------------------------------------
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown(
        """
        <div class="hero">
          <h1>📊 CPG Feedback Summarizer</h1>
          <p>Turn bulk customer feedback from social, survey &amp; support channels
          into a specific executive summary, priority actions, emergent themes,
          and grounded Q&amp;A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_r:
    st.write("")
    dark = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

if not az.azure_is_configured():
    st.warning(
        "⚠️ Azure OpenAI is not configured. Set "
        + ", ".join(f"`{n}`" for n in az.missing_config())
        + " (see `.env.example`) before running **Analyze**.",
        icon="⚠️",
    )


# --- Data loading (upload-only) ----------------------------------------------
def _load_records(records: list[dict], label: str) -> None:
    st.session_state.records = records
    st.session_state.source_label = label
    st.session_state.results = None
    st.session_state.chat_history = []


def _run_analysis(records: list[dict]) -> None:
    """Execute the full pipeline behind a spinner and store results."""
    n = len(records)
    with st.spinner(f"Analyzing {n} feedback entries…"):
        try:
            client = az.CPGAzureClient()
        except az.AzureConfigError as exc:
            st.error(f"Azure configuration error: {exc}")
            return

        warnings: list[str] = []
        try:
            merged, warnings = pl.run_extraction(records, client.extract_batch)
        except Exception as exc:  # catastrophic network/auth failure
            st.error(f"Analysis failed during extraction: {exc}")
            return

        if not merged:
            st.error(
                "No records could be extracted — every batch failed. "
                "Check Azure connectivity/credentials and try again."
            )
            return

        aggregation = pl.aggregate(merged)

        summary = {"headline": "", "summary": "", "top_actions": []}
        try:
            summary = client.executive_summary(aggregation)
        except az.AzureCallError as exc:
            warnings.append(f"Executive summary unavailable: {exc}")

    st.session_state.results = {
        "merged": merged,
        "aggregation": aggregation,
        "summary": summary,
        "warnings": warnings,
    }
    st.session_state.chat_history = []


st.markdown("#### 1 · Upload your feedback file")
st.caption("Accepted: `.csv` or `.json` — same schema as the data contract "
           "(id, channel, date, rating, text, subject).")
up = st.file_uploader(
    "Upload feedback file",
    type=["csv", "json"],
    label_visibility="collapsed",
)
if up is not None:
    if st.button("📥 Load file", use_container_width=True):
        try:
            recs = pl.load_uploaded(up.name, up.getvalue())
            _load_records(recs, up.name)
        except pl.DatasetError as exc:
            st.error(f"Invalid file: {exc}")


# --- Loaded status + Analyze --------------------------------------------------
records = st.session_state.records
if records:
    n = len(records)
    channels = sorted({r["channel"] for r in records})
    st.success(
        f"Loaded **{n}** feedback entries from **{st.session_state.source_label}** · "
        f"channels: {', '.join(channels)}",
        icon="✅",
    )
    analyze_disabled = not az.azure_is_configured()
    if st.button(
        "🚀 Analyze feedback",
        type="primary",
        disabled=analyze_disabled,
        help="Requires Azure OpenAI configuration." if analyze_disabled else None,
    ):
        _run_analysis(records)


# --- Chart helpers ------------------------------------------------------------
def _bar_df(counts: dict, name: str, value: str) -> pd.DataFrame:
    df = pd.DataFrame(
        sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=[name, value],
    )
    return df.set_index(name)


def _render_dashboard(results: dict) -> None:
    agg = results["aggregation"]
    summary = results["summary"]

    for w in results["warnings"]:
        st.warning(w, icon="⚠️")

    st.divider()

    # 1 — Headline banner (specific, from Call 2)
    headline = summary.get("headline", "").strip()
    if headline:
        st.markdown(
            f'<div class="headline"><span class="tag">Key finding</span><br>{headline}</div>',
            unsafe_allow_html=True,
        )

    # 2 — KPI row
    total = agg["total"]
    neg = agg["sentiment_counts"].get("negative", 0)
    pct_neg = (neg / total * 100) if total else 0
    top_issue = agg["ranked_categories"][0][0] if agg["ranked_categories"] else "—"
    k1, k2, k3 = st.columns(3)
    k1.metric("Total feedback analyzed", f"{total:,}")
    k2.metric("% negative sentiment", f"{pct_neg:.0f}%")
    k3.metric("Top issue category", top_issue.replace("_", " ").title())

    # 3 — Executive summary
    st.markdown("### 📝 Executive summary")
    if summary.get("summary"):
        st.write(summary["summary"])
    else:
        st.info("Executive summary unavailable for this run.")
    if summary.get("top_actions"):
        st.markdown("**Top recommended actions**")
        for i, action in enumerate(summary["top_actions"], start=1):
            st.markdown(f"{i}. {action}")

    st.divider()

    # 4 — Charts row: category + sentiment
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Feedback by category")
        st.caption("Fixed macro-taxonomy — stable across datasets.")
        st.bar_chart(_bar_df(agg["category_counts"], "category", "count"),
                     color=C["accent"], horizontal=True)
    with c2:
        st.markdown("##### Sentiment breakdown")
        sdf = _bar_df(agg["sentiment_counts"], "sentiment", "count")
        st.bar_chart(sdf, color=C["blue"])

    # 5 — Emergent themes (the dataset-specific section)
    st.markdown("### 🏷️ Themes found in this dataset")
    st.caption("Emergent, free-form theme tags extracted from *this* feedback — "
               "this section looks completely different for every dataset.")
    themes = agg.get("top_emergent_themes", [])
    if themes:
        tdf = pd.DataFrame(themes, columns=["theme", "count"]).set_index("theme")
        st.bar_chart(tdf, color=C["teal"], horizontal=True)
        pills = "".join(
            f'<span class="pill">{t} <b>{c}</b></span>' for t, c in themes
        )
        st.markdown(f'<div class="themebar">{pills}</div>', unsafe_allow_html=True)
    else:
        st.info("No emergent themes extracted.")

    st.divider()

    # 6 — Channel breakdown + trend
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Feedback by channel")
        ch = {k.replace("_", " ").title(): v for k, v in agg["channel_counts"].items()}
        st.bar_chart(_bar_df(ch, "channel", "count"), color=C["amber"])
    with c4:
        st.markdown("##### Volume trend (top categories)")
        trend = agg["date_trend"]
        if trend:
            weeks = sorted({w for cat in trend.values() for w in cat})
            trdf = pd.DataFrame({"week": weeks})
            for cat, buckets in trend.items():
                trdf[cat] = [buckets.get(w, 0) for w in weeks]
            st.line_chart(trdf.set_index("week"))
        else:
            st.info("Not enough dated records for a trend.")

    st.divider()

    # 7 — Top 5 priority issues (category, theme_tag, severity, insight, preview, channel)
    st.markdown("### 🔥 Top 5 priority issues")
    st.caption("Ranked by severity, most recent first on ties.")
    rows = []
    for r in agg["top_priority"]:
        rows.append(
            {
                "Category": r["category"].replace("_", " ").title(),
                "Theme": r.get("theme_tag", "—"),
                "Severity": r["severity"],
                "Actionable insight": r["actionable_insight"] or "—",
                "Preview": (r["text"][:80] + "…") if len(r["text"]) > 80 else r["text"],
                "Channel": r["channel"].replace("_", " ").title(),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # 8 — Chat Q&A
    st.markdown("### 💬 Ask about this feedback")
    st.caption("Answers are grounded strictly in the aggregated data — no raw "
               "feedback text is sent to this call.")
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    prompt = st.chat_input("e.g. What's driving negative sentiment?")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    client = az.CPGAzureClient()
                    answer = client.chat_answer(prompt, agg)
                except (az.AzureConfigError, az.AzureCallError) as exc:
                    answer = f"Sorry — I couldn't answer that: {exc}"
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))


if st.session_state.results:
    _render_dashboard(st.session_state.results)
elif not records:
    st.info("👆 Upload a `.csv` or `.json` feedback file to get started.", icon="💡")
