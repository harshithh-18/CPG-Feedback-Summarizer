"""CPG Customer Feedback Summarizer — Streamlit entrypoint.

Upload feedback -> Analyze (PII scrub -> extraction -> aggregation ->
executive summary) -> dashboard + grounded chat Q&A.

Input is upload-only (.csv / .json). Credentials come only from environment
variables / Streamlit secrets. Raw feedback text is PII-scrubbed before the
only model call that sees it (Call 1); Calls 2 and 3 receive aggregated data
only.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import azure_client as az
import pipeline as pl
import report as rp

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

      /* Wider content frame so the sides no longer read as empty. */
      .block-container {{
        padding-top: 1.2rem; padding-bottom: 3rem;
        max-width: 1360px; padding-left: 3rem; padding-right: 3rem;
      }}

      h1,h2,h3,h4,h5,h6, p, span, label, li {{ color: var(--ink); }}
      .stMarkdown p {{ color: var(--ink); }}

      /* Hero header */
      .hero {{
        background: linear-gradient(120deg, {C['grad1']} 0%, {C['grad2']} 100%);
        border-radius: 18px; padding: 26px 32px; margin-bottom: 10px;
        color: #fff; box-shadow: 0 14px 38px -16px {C['grad1']}77;
      }}
      .hero h1 {{ color:#fff; font-size:2.05rem; font-weight:800; margin:0; letter-spacing:-0.5px; }}
      .hero p {{ color:#ffffffe0; margin:8px 0 0; font-size:1.02rem; max-width: 760px; }}

      /* Headline banner (Call 2 output) */
      .headline {{
        background: var(--surface); border:1px solid var(--border);
        border-left: 6px solid var(--accent);
        border-radius: 14px; padding: 20px 26px; margin: 10px 0 6px;
        font-size: 1.5rem; font-weight: 750; line-height:1.3; color: var(--ink);
        box-shadow: 0 8px 26px -18px #00000066;
      }}
      .headline .tag {{
        display:inline-block; font-size:0.7rem; font-weight:700; letter-spacing:0.14em;
        text-transform:uppercase; color: var(--accent); margin-bottom:6px;
      }}

      /* KPI metric cards */
      div[data-testid="stMetric"] {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 6px 20px -16px #00000055;
      }}
      div[data-testid="stMetric"] label p {{ font-size:0.82rem; color: var(--ink2); }}
      div[data-testid="stMetricValue"] {{ color: var(--accent); font-weight:800; }}

      /* Section + panel cards */
      .card {{
        background: var(--surface); border:1px solid var(--border);
        border-radius:16px; padding:22px 26px; margin-bottom:14px;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      .panel {{
        background: var(--surface); border:1px solid var(--border);
        border-radius:16px; padding:18px 22px; height:100%;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      .sec-label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.12em;
        text-transform:uppercase; color: var(--ink2); margin-bottom:4px; }}
      .sec-h {{ font-size:1.28rem; font-weight:750; margin: 6px 0 2px; color: var(--ink); }}

      /* Key-category cards */
      .catcard {{
        background: var(--surface2); border:1px solid var(--border);
        border-left:5px solid {C['coral']};
        border-radius:12px; padding:14px 18px; margin-bottom:10px;
      }}
      .catcard .cn {{ font-weight:750; font-size:1.02rem; color: var(--ink); }}
      .catcard .cm {{ font-size:0.78rem; color: var(--accent); font-weight:700;
        letter-spacing:0.02em; margin-left:8px; }}
      .catcard .cw {{ color: var(--ink2); font-size:0.95rem; margin-top:6px; line-height:1.5; }}

      /* Action list */
      .actn {{ display:flex; gap:12px; align-items:flex-start; margin:10px 0; }}
      .actn .num {{
        flex:0 0 auto; width:26px; height:26px; border-radius:50%;
        background: {C['accent']}; color:#fff; font-weight:800; font-size:0.9rem;
        display:flex; align-items:center; justify-content:center;
      }}
      .actn .txt {{ color: var(--ink); font-size:0.98rem; line-height:1.5; padding-top:2px; }}

      /* Theme tag pills */
      .themebar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }}
      .pill {{
        display:inline-flex; align-items:center; gap:6px;
        background: var(--surface2); border:1px solid var(--border);
        color: var(--ink); border-radius:999px; padding:5px 13px; font-size:0.86rem;
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

      /* Chat input styled to match cards (rounded, bordered, surface bg) */
      div[data-testid="stChatInput"] {{
        background: var(--surface) !important;
        border:1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      div[data-testid="stChatInput"] textarea {{
        color: var(--ink) !important; font-size: 1rem !important;
      }}
      div[data-testid="stChatInput"] textarea::placeholder {{ color: var(--ink2) !important; }}
      [data-testid="stChatMessage"] {{
        background: var(--surface); border:1px solid var(--border);
        border-radius: 14px;
      }}
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
          into a specific executive summary, sentiment &amp; volume trends, priority
          actions, emergent themes, and grounded Q&amp;A — exportable as a PDF report.</p>
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
    """Execute the full pipeline with live, streaming progress checkpoints."""
    n = len(records)

    try:
        client = az.CPGAzureClient()
    except az.AzureConfigError as exc:
        st.error(f"Azure configuration error: {exc}")
        return

    warnings: list[str] = []

    # --- Streaming progress UI: a status container with live checkpoints ---
    with st.status(f"Analyzing {n} feedback entries…", expanded=True) as status:
        progress = st.progress(0.0)
        log = st.empty()
        checkpoints: list[str] = []

        def _post(line: str) -> None:
            checkpoints.append(line)
            log.markdown("\n".join(f"- {c}" for c in checkpoints[-8:]))

        _post("🔒 Scrubbing PII (emails, phones, names) from every record…")
        time.sleep(0.15)
        n_batches = (n + pl.BATCH_SIZE - 1) // pl.BATCH_SIZE
        _post(f"📦 Split into **{n_batches}** batch(es) of ≤{pl.BATCH_SIZE} records.")

        def _on_progress(done_b: int, total_b: int, done_r: int) -> None:
            frac = 0.15 + 0.6 * (done_b / max(1, total_b))
            progress.progress(min(frac, 0.75))
            _post(
                f"🧠 Extracted batch **{done_b}/{total_b}** "
                f"· {done_r}/{n} records analyzed…"
            )

        try:
            merged, warnings = pl.run_extraction(
                records, client.extract_batch, on_progress=_on_progress
            )
        except Exception as exc:  # catastrophic network/auth failure
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis failed during extraction: {exc}")
            return

        if not merged:
            status.update(label="Analysis failed", state="error")
            st.error(
                "No records could be extracted — every batch failed. "
                "Check Azure connectivity/credentials and try again."
            )
            return

        progress.progress(0.8)
        _post(f"📊 Aggregating {len(merged)} results — counts, severity "
              "ranking, weekly trend, emergent themes…")
        aggregation = pl.aggregate(merged)

        progress.progress(0.88)
        _post("📝 Writing the executive summary (streaming below)…")

        # Stream Call 2 so the summary builds live instead of dumping at once.
        summary = {
            "headline": "", "summary": "", "sentiment_trend": "",
            "key_categories": [], "top_actions": [],
        }
        summary_ph = st.empty()
        try:
            acc = ""
            for item in client.executive_summary_stream(aggregation):
                if isinstance(item, tuple) and item[0] == "__result__":
                    summary = item[1]
                    break
                acc += item
                live = az.extract_partial_string(acc, "summary")
                if live:
                    summary_ph.markdown(
                        f"<div style='color:var(--ink2);font-size:0.96rem;"
                        f"line-height:1.55'>{live}▌</div>",
                        unsafe_allow_html=True,
                    )
        except az.AzureCallError as exc:
            warnings.append(f"Executive summary unavailable: {exc}")
        summary_ph.empty()

        progress.progress(1.0)
        _post("✅ Done — rendering your dashboard.")
        status.update(label=f"Analysis complete · {len(merged)} records",
                      state="complete", expanded=False)

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
def _hbar(counts: dict, name: str, color: str, pretty: bool = False) -> None:
    """Render a horizontal bar chart, largest bar on top."""
    items = sorted(counts.items(), key=lambda kv: kv[1])  # asc -> top is largest
    if pretty:
        items = [(k.replace("_", " ").title(), v) for k, v in items]
    df = pd.DataFrame(items, columns=[name, "count"]).set_index(name)
    st.bar_chart(df, color=color, horizontal=True, height=max(140, 46 * len(items)))


def _render_dashboard(results: dict) -> None:
    agg = results["aggregation"]
    summary = results["summary"]

    for w in results["warnings"]:
        st.warning(w, icon="⚠️")

    st.divider()

    # --- Report action bar (PDF export) ---
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<div class="sec-label">Analysis report</div>',
                    unsafe_allow_html=True)
    with right:
        try:
            pdf_bytes = rp.build_report(
                agg, summary, st.session_state.source_label or "uploaded file"
            )
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name="cpg_feedback_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001 - export must never crash dashboard
            st.caption(f"PDF export unavailable: {exc}")

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

    st.write("")

    # 3 — Executive summary + sentiment trend, side by side (fills the width)
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Executive summary</div>'
                    '<div class="sec-h">📝 What the data is telling you</div>',
                    unsafe_allow_html=True)
        if summary.get("summary"):
            st.write(summary["summary"])
        else:
            st.info("Executive summary unavailable for this run.")
        st.markdown('</div>', unsafe_allow_html=True)
    with sc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Sentiment &amp; volume trend</div>'
                    '<div class="sec-h">📈 Where it&#39;s heading</div>',
                    unsafe_allow_html=True)
        if summary.get("sentiment_trend"):
            st.write(summary["sentiment_trend"])
        else:
            st.caption("Trend narrative unavailable for this run.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4 — Key complaint categories (rich, token-heavy explanations)
    key_cats = summary.get("key_categories", [])
    if key_cats:
        st.markdown('<div class="sec-label">Key complaint categories</div>'
                    '<div class="sec-h">🎯 The issues that matter most, explained</div>',
                    unsafe_allow_html=True)
        cc = st.columns(2)
        for i, c in enumerate(key_cats):
            with cc[i % 2]:
                name = c.get("category", "").replace("_", " ").title()
                metric = c.get("metric", "")
                why = c.get("why_it_matters", "")
                st.markdown(
                    f'<div class="catcard"><span class="cn">{name}</span>'
                    f'<span class="cm">{metric}</span>'
                    f'<div class="cw">{why}</div></div>',
                    unsafe_allow_html=True,
                )

    # 5 — Recommended actions
    if summary.get("top_actions"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Recommended actions</div>'
                    '<div class="sec-h">🚀 Do these next</div>',
                    unsafe_allow_html=True)
        for i, action in enumerate(summary["top_actions"], start=1):
            st.markdown(
                f'<div class="actn"><div class="num">{i}</div>'
                f'<div class="txt">{action}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # 6 — Charts row: category + sentiment (both horizontal)
    st.markdown('<div class="sec-h">📊 Breakdowns</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Feedback by category")
        st.caption("Fixed macro-taxonomy — stable across datasets.")
        _hbar(agg["category_counts"], "category", C["accent"], pretty=True)
    with c2:
        st.markdown("##### Sentiment breakdown")
        _hbar(agg["sentiment_counts"], "sentiment", C["blue"], pretty=True)

    # 7 — Emergent themes (the dataset-specific section)
    st.markdown("### 🏷️ Themes found in this dataset")
    st.caption("Emergent, free-form theme tags extracted from *this* feedback — "
               "this section looks completely different for every dataset.")
    themes = agg.get("top_emergent_themes", [])
    if themes:
        tdf = pd.DataFrame(themes, columns=["theme", "count"])
        tdf = tdf.sort_values("count").set_index("theme")
        st.bar_chart(tdf, color=C["teal"], horizontal=True,
                     height=max(160, 42 * len(themes)))
        pills = "".join(
            f'<span class="pill">{t} <b>{c}</b></span>' for t, c in themes
        )
        st.markdown(f'<div class="themebar">{pills}</div>', unsafe_allow_html=True)
    else:
        st.info("No emergent themes extracted.")

    st.divider()

    # 8 — Channel breakdown + trend
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Feedback by channel")
        _hbar(agg["channel_counts"], "channel", C["amber"], pretty=True)
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

    # 9 — Top 5 priority issues
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

    # 10 — Chat Q&A (streaming)
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
            placeholder = st.empty()
            answer = ""
            try:
                client = az.CPGAzureClient()
                acc = ""
                streamed = False
                for item in client.chat_answer_stream(prompt, agg):
                    if isinstance(item, tuple) and item[0] == "__result__":
                        answer = item[1]
                        break
                    acc += item
                    live = az.extract_partial_string(acc, "answer")
                    if live:
                        streamed = True
                        placeholder.markdown(live + "▌")
                if not answer and acc:
                    # Fallback: model didn't emit a clean tuple.
                    answer = az.extract_partial_string(acc, "answer")
            except (az.AzureConfigError, az.AzureCallError) as exc:
                answer = f"Sorry — I couldn't answer that: {exc}"
            placeholder.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


if st.session_state.results:
    _render_dashboard(st.session_state.results)
elif not records:
    st.info("👆 Upload a `.csv` or `.json` feedback file to get started.", icon="💡")
