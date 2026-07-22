"""CPG Customer Feedback Summarizer — Streamlit entrypoint.

Load feedback -> Analyze (PII scrub -> Azure extraction -> aggregation ->
executive summary) -> dashboard + grounded chat Q&A.

Credentials come only from environment variables / Streamlit secrets. The
raw feedback text is PII-scrubbed before the only Azure call that sees it
(Call 1); Calls 2 and 3 receive aggregated data only.
"""

from __future__ import annotations

from pathlib import Path

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

# --- Cheap polish: hide default chrome + light custom styling ------------------
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header [data-testid="stToolbar"] {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {
          background: #F5F1EC;
          border: 1px solid #ECE3D8;
          border-radius: 12px;
          padding: 16px 18px;
      }
      div[data-testid="stMetric"] label p {font-size: 0.85rem; color:#6B6257;}
      .app-title {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.1rem;}
      .app-sub {color:#6B6257; font-size:1.02rem; margin-bottom:0.4rem;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px;
             background:#FBEADB; color:#B5591A; font-size:0.78rem; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Session state ------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("records", None)
    st.session_state.setdefault("source_label", None)
    st.session_state.setdefault("results", None)  # {merged, aggregation, summary, warnings}
    st.session_state.setdefault("chat_history", [])  # list[(role, text)]


_init_state()


# --- Header -------------------------------------------------------------------
st.markdown('<div class="app-title">📊 CPG Feedback Summarizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Turn bulk customer feedback from social, survey, and '
    "support channels into an executive summary, a priority action list, and "
    "grounded Q&amp;A — powered by Azure OpenAI.</div>",
    unsafe_allow_html=True,
)

if not az.azure_is_configured():
    st.warning(
        "⚠️ Azure OpenAI is not configured. Set "
        + ", ".join(f"`{n}`" for n in az.missing_config())
        + " (see `.env.example`) before running **Analyze**.",
        icon="⚠️",
    )

st.divider()


# --- Data loading controls ----------------------------------------------------
def _load_records(records: list[dict], label: str) -> None:
    st.session_state.records = records
    st.session_state.source_label = label
    st.session_state.results = None  # invalidate stale analysis
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
        except Exception as exc:  # network/auth catastrophic failure
            st.error(f"Analysis failed during extraction: {exc}")
            return

        if not merged:
            st.error(
                "No records could be extracted — every batch failed. "
                "Check Azure connectivity/credentials and try again."
            )
            return

        aggregation = pl.aggregate(merged)

        summary = {"summary": "", "top_actions": []}
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


left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("##### 1 · Choose a bundled dataset")
    bundled = pl.list_bundled_datasets()
    if bundled:
        names = [p.name for p in bundled]
        chosen = st.selectbox(
            "Datasets discovered in `data/`",
            names,
            index=0,
            label_visibility="collapsed",
        )
        if st.button("Load selected dataset", type="primary", use_container_width=True):
            path = pl.DATA_DIR / chosen
            try:
                recs = pl.load_feedback(path)
                _load_records(recs, chosen)
            except pl.DatasetError as exc:
                st.error(f"Could not load `{chosen}`: {exc}")
    else:
        st.info("No datasets found in `data/`. Use the uploader on the right.")

with right:
    st.markdown("##### 2 · …or upload your own (`.csv` / `.json`)")
    up = st.file_uploader(
        "Upload feedback file",
        type=["csv", "json"],
        label_visibility="collapsed",
    )
    if up is not None:
        if st.button("Load uploaded file", use_container_width=True):
            try:
                recs = pl.load_uploaded(up.name, up.getvalue())
                _load_records(recs, up.name)
            except pl.DatasetError as exc:
                st.error(f"Invalid file: {exc}")


# --- Loaded-data status + Analyze --------------------------------------------
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


# --- Dashboard ----------------------------------------------------------------
def _bar_from_counts(counts: dict, name: str, value: str) -> pd.DataFrame:
    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
                      columns=[name, value])
    return df.set_index(name)


def _render_dashboard(results: dict) -> None:
    agg = results["aggregation"]
    summary = results["summary"]
    merged = results["merged"]

    for w in results["warnings"]:
        st.warning(w, icon="⚠️")

    st.divider()

    # 1 — KPI row
    total = agg["total"]
    neg = agg["sentiment_counts"].get("negative", 0)
    pct_neg = (neg / total * 100) if total else 0
    top_issue = agg["ranked_categories"][0][0] if agg["ranked_categories"] else "—"

    k1, k2, k3 = st.columns(3)
    k1.metric("Total feedback analyzed", f"{total:,}")
    k2.metric("% negative sentiment", f"{pct_neg:.0f}%")
    k3.metric("Top issue category", top_issue.replace("_", " ").title())

    # 2 — Executive summary
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

    # 3 — Charts row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Feedback by category")
        st.bar_chart(_bar_from_counts(agg["category_counts"], "category", "count"),
                     color="#E8792B")
    with c2:
        st.markdown("##### Sentiment breakdown")
        st.bar_chart(_bar_from_counts(agg["sentiment_counts"], "sentiment", "count"),
                     color="#5B8C5A")

    # 4 — Channel breakdown + trend
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Feedback by channel")
        ch = {k.replace("_", " ").title(): v for k, v in agg["channel_counts"].items()}
        st.bar_chart(_bar_from_counts(ch, "channel", "count"), color="#C99A3B")
    with c4:
        st.markdown("##### Volume trend (top categories)")
        trend = agg["date_trend"]
        if trend:
            weeks = sorted({w for cat in trend.values() for w in cat})
            tdf = pd.DataFrame({"week": weeks})
            for cat, buckets in trend.items():
                tdf[cat] = [buckets.get(w, 0) for w in weeks]
            st.line_chart(tdf.set_index("week"))
        else:
            st.info("Not enough dated records for a trend.")

    st.divider()

    # 5 — Top 5 priority issues
    st.markdown("### 🔥 Top 5 priority issues")
    st.caption("Ranked by severity, most recent first on ties.")
    rows = []
    for r in agg["top_priority"]:
        rows.append(
            {
                "Category": r["category"].replace("_", " ").title(),
                "Severity": r["severity"],
                "Key phrase": r["key_phrase"],
                "Actionable insight": r["actionable_insight"] or "—",
                "Preview": (r["text"][:90] + "…") if len(r["text"]) > 90 else r["text"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # 6 — Chat Q&A
    st.markdown("### 💬 Ask about this feedback")
    st.caption(
        "Answers are grounded strictly in the aggregated data above — no raw "
        "feedback text is sent to this call."
    )
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
    st.info("👆 Load a dataset to get started.", icon="💡")
