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
import plotly.graph_objects as go
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

      /* Big-picture overview banner (plain-English, top of dashboard) */
      .overview {{
        background: linear-gradient(120deg, {C['grad1']}14 0%, {C['grad2']}14 100%);
        border:1px solid var(--border); border-left:6px solid {C['teal']};
        border-radius:16px; padding:20px 26px; margin:6px 0 4px;
        box-shadow: 0 8px 26px -20px #00000055;
      }}
      .overview .ov-tag {{
        font-size:0.72rem; font-weight:800; letter-spacing:0.12em;
        text-transform:uppercase; color: {C['teal']}; margin-bottom:8px;
      }}
      .overview .ov-body {{ color: var(--ink); font-size:1.12rem; line-height:1.65; }}
      .overview .ov-body b {{ color: var(--accent); font-weight:750; }}

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

      /* Priority action cards (ranked what-to-fix-first list) */
      .pcard {{
        background: var(--surface); border:1px solid var(--border);
        border-left:6px solid {C['coral']};
        border-radius:12px; padding:14px 18px; margin-bottom:12px;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      .pcard .phead {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
      .pcard .ubadge {{
        flex:0 0 auto; color:#fff; font-weight:800; font-size:0.72rem;
        letter-spacing:0.04em; text-transform:uppercase;
        border-radius:999px; padding:4px 11px;
      }}
      .pcard .ptitle {{ font-weight:750; font-size:1.06rem; color: var(--ink); line-height:1.3; flex:1 1 auto; }}
      .pcard .why {{ color: var(--ink2); font-size:0.95rem; margin-top:8px; line-height:1.5; }}
      .pcard .do {{
        margin-top:10px; background: var(--surface2); border:1px solid var(--border);
        border-radius:10px; padding:9px 13px; color: var(--ink);
        font-size:0.95rem; line-height:1.45;
      }}
      .pcard .do .dolabel {{
        display:inline-block; font-size:0.66rem; font-weight:800; letter-spacing:0.1em;
        text-transform:uppercase; color: var(--accent); margin-right:8px;
      }}

      /* Frame ONLY the India map: the sentinel's element-container's next
         sibling holds the map's Plotly chart. */
      div[data-testid="stElementContainer"]:has(> .map-frame-anchor) {{
        display:none;
      }}
      div[data-testid="stElementContainer"]:has(> .map-frame-anchor)
        + div[data-testid="stElementContainer"] {{
        background: var(--surface);
        border:1px solid var(--border);
        border-radius:16px;
        padding:8px;
        box-shadow: 0 6px 22px -20px #00000055;
      }}

      /* Detailed priority-issue cards (replaces the top-issues table) */
      .icard {{
        background: var(--surface); border:1px solid var(--border);
        border-left:6px solid {C['coral']};
        border-radius:14px; padding:18px 22px; margin-bottom:16px;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      .icard .ihead {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:4px; }}
      .icard .rank {{
        flex:0 0 auto; width:30px; height:30px; border-radius:50%;
        background: {C['coral']}; color:#fff; font-weight:800; font-size:1rem;
        display:flex; align-items:center; justify-content:center;
      }}
      .icard .ititle {{ font-weight:800; font-size:1.14rem; color: var(--ink); line-height:1.25; flex:1 1 auto; }}
      .icard .sev {{
        flex:0 0 auto; color:#fff; font-weight:800; font-size:0.72rem;
        letter-spacing:0.04em; text-transform:uppercase;
        border-radius:999px; padding:4px 11px;
      }}
      .icard .meta {{ font-size:0.8rem; color: var(--ink2); margin-bottom:12px; }}
      .icard .meta b {{ color: var(--accent); }}
      .icard .blocklabel {{
        font-size:0.7rem; font-weight:800; letter-spacing:0.1em; text-transform:uppercase;
        margin:12px 0 4px;
      }}
      .icard .wronglabel {{ color: {C['coral']}; }}
      .icard .actionlabel {{ color: {C['teal']}; }}
      .icard .quote {{
        background: var(--surface2); border-left:3px solid var(--border);
        border-radius:8px; padding:9px 13px; color: var(--ink);
        font-size:0.95rem; line-height:1.5; font-style:italic; margin:2px 0 4px;
      }}
      .icard ul.acts {{ margin:4px 0 0; padding-left:20px; }}
      .icard ul.acts li {{ color: var(--ink); font-size:0.97rem; line-height:1.55; margin-bottom:5px; }}

      /* Sentiment mood line (replaces the sentiment bar chart) */
      .moodline {{
        background: var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:14px 20px; margin:12px 0 4px;
        box-shadow: 0 6px 22px -20px #00000055;
      }}
      .moodline .moodtext {{ color: var(--ink); font-size:1.02rem; line-height:1.5; }}
      .moodline .moodkey {{ color: var(--ink2); font-size:0.86rem; white-space:nowrap; }}
      .moodline .moodbar {{
        display:flex; width:100%; height:12px; border-radius:999px;
        overflow:hidden; margin-top:12px; background: var(--surface2);
      }}
      .moodline .moodbar span {{ display:block; height:100%; }}

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
            "headline": "", "whats_happening": "", "next_steps": [],
        }
        summary_ph = st.empty()
        try:
            acc = ""
            for item in client.executive_summary_stream(aggregation):
                if isinstance(item, tuple) and item[0] == "__result__":
                    summary = item[1]
                    break
                acc += item
                live = az.extract_partial_string(acc, "whats_happening")
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
_SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def _pretty(label: str) -> str:
    return label.replace("_", " ").title()


def _plotly_layout(fig: go.Figure, height: int) -> None:
    """Apply the app theme to a Plotly figure (transparent bg, themed fonts)."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["ink"], size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=C["ink2"], size=11),
        ),
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(color=C["ink"], size=12),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=C["border"], zeroline=False,
                     color=C["ink2"])
    fig.update_yaxes(showgrid=False, color=C["ink"])


def _stacked_sentiment_bar(
    counts: dict, breakdown: dict, name: str, height_per: int = 52
) -> None:
    """Horizontal bar per category/channel, stacked & colored by sentiment.

    Hovering any segment shows exactly how many positive / negative / neutral
    make up that bar — so the "taste" bar reveals its sentiment split on hover.
    """
    # Order rows by total volume (largest at top).
    rows = sorted(counts.items(), key=lambda kv: kv[1])  # asc -> top is largest
    labels = [_pretty(k) for k, _ in rows]
    raw_keys = [k for k, _ in rows]
    totals = {k: counts[k] for k in raw_keys}

    fig = go.Figure()
    for sent in _SENTIMENT_ORDER:
        vals = [breakdown.get(k, {}).get(sent, 0) for k in raw_keys]
        cust = [
            [totals[k], _pretty(k)] for k in raw_keys
        ]
        fig.add_bar(
            y=labels,
            x=vals,
            name=sent.title(),
            orientation="h",
            marker=dict(color=SENTIMENT_COLORS[sent],
                        line=dict(width=0)),
            customdata=cust,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                + sent.title()
                + ": %{x} of %{customdata[0]} total"
                + "<extra></extra>"
            ),
        )
    fig.update_layout(barmode="stack", bargap=0.35)
    _plotly_layout(fig, max(150, height_per * len(rows)))
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def _india_map(location_map: dict) -> None:
    """Scatter Indian cities as bubbles: size = feedback volume, color = the
    dominant issue there. Hover reveals the full per-issue and per-sentiment
    breakdown for that city. Falls back to a friendly note when the dataset
    carries no location data."""
    points = location_map.get("points", [])
    if not points:
        located = location_map.get("located", 0)
        if located:
            st.info(
                f"{located} record(s) had a location, but none matched a known "
                "Indian city/state, so the map can't place them yet.",
                icon="🗺️",
            )
        else:
            st.info(
                "No location data in this dataset yet. Add a `location` "
                "(city or state) to each record and this India map will show "
                "where issues cluster — e.g. which cities report the most "
                "damages.",
                icon="🗺️",
            )
        return

    unique_cats = sorted({p["top_category"] for p in points})
    palette = [C["coral"], C["accent"], C["teal"], C["blue"], C["amber"],
               "#B04AC4", "#3AA0A0"]
    cat_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}

    max_total = max(p["total"] for p in points) or 1
    fig = go.Figure()
    for cat in unique_cats:
        cat_pts = [p for p in points if p["top_category"] == cat]
        hover = []
        for p in cat_pts:
            cat_lines = "<br>".join(
                f"&nbsp;&nbsp;{_pretty(k)}: {v}"
                for k, v in sorted(p["category_counts"].items(),
                                   key=lambda kv: kv[1], reverse=True)
            )
            sent_lines = " · ".join(
                f"{k.title()} {v}" for k, v in p["sentiment_counts"].items()
            )
            hover.append(
                f"<b>{p['location']}</b><br>"
                f"{p['total']} feedback item(s)<br>"
                f"Top issue: {_pretty(p['top_category'])}<br>"
                f"<br>By issue:<br>{cat_lines}<br>"
                f"<br>{sent_lines}"
            )
        fig.add_trace(
            go.Scattergeo(
                lon=[p["lon"] for p in cat_pts],
                lat=[p["lat"] for p in cat_pts],
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                mode="markers",
                name=_pretty(cat),
                marker=dict(
                    size=[14 + 34 * (p["total"] / max_total) for p in cat_pts],
                    color=cat_color[cat],
                    opacity=0.82,
                    line=dict(width=1, color="#ffffff"),
                    sizemode="diameter",
                ),
            )
        )

    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        center=dict(lat=22.5, lon=80),
        lataxis_range=[6, 37],
        lonaxis_range=[67, 98],
        showcountries=True, countrycolor=C["accent"], countrywidth=1.4,
        showland=True, landcolor=C["surface2"],
        showocean=True, oceancolor=C["bg"],
        showcoastlines=True, coastlinecolor=C["accent"], coastlinewidth=1.6,
        showframe=True, framecolor=C["ink2"], framewidth=1.5,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["ink"], size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5,
            bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["ink"], size=11),
        ),
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["border"],
                        font=dict(color=C["ink"], size=12), align="left"),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False})


_DEPT_STYLE = {
    "Product/R&D": (C["accent"], "🧪"),
    "Marketing": (C["teal"], "📣"),
    "Customer Service": (C["coral"], "🎧"),
}


def _render_next_steps(steps: list[dict]) -> None:
    """Render Call 2's next_steps as a bulleted list — each bullet shows the
    action, a small department tag, and the reason on a second line."""
    if not steps:
        st.info("No next steps were produced for this run.")
        return
    for s in steps:
        action = s.get("action", "")
        department = s.get("department", "Product/R&D")
        why = s.get("why", "")
        color, icon = _DEPT_STYLE.get(department, _DEPT_STYLE["Product/R&D"])
        why_html = f'<div class="why">{why}</div>' if why else ""
        st.markdown(
            f'<div class="pcard" style="border-left-color:{color}">'
            f'<div class="phead">'
            f'<span class="ptitle">{action}</span>'
            f'<span class="ubadge" style="background:{color}">{icon} {department}</span>'
            f'</div>{why_html}</div>',
            unsafe_allow_html=True,
        )


# Severity band -> (label, color). Higher severity reads more urgent.
def _severity_band(sev: int) -> tuple[str, str]:
    if sev >= 5:
        return "Critical", C["coral"]
    if sev == 4:
        return "High", C["amber"]
    if sev == 3:
        return "Medium", C["blue"]
    return "Low", C["teal"]


# Concrete, category-specific follow-up actions so every issue card carries a
# real bulleted plan even when the model's own actionable_insight is thin.
_CATEGORY_ACTIONS = {
    "packaging": [
        "Pull and inspect the affected batch's packaging and sealing line.",
        "Add a tamper/transit check (drop-test, seal test) before dispatch.",
    ],
    "quality": [
        "Quarantine the reported batch and preserve the product for inspection.",
        "Run a documented contamination / freshness check before releasing more stock.",
    ],
    "taste": [
        "Have R&D taste-test the current formula against the previous version.",
        "Confirm no recipe or supplier change slipped in unannounced.",
    ],
    "price": [
        "Review the recent price change against competitor and store-brand pricing.",
        "Consider a value pack or promotion for the most price-sensitive line.",
    ],
    "availability": [
        "Check stock and replenishment for this product at the named stores/channels.",
        "Flag the out-of-stock SKU to the retail/supply team for restocking.",
    ],
    "customer_service": [
        "Escalate and resolve this specific open request today.",
        "Review hold time / follow-up procedure so this doesn't repeat.",
    ],
    "other": [
        "Assign this to the most relevant team and confirm a fix owner.",
    ],
}


def _issue_action_bullets(rec: dict) -> list[str]:
    """Build the 'actions to take' bullets for one priority issue: lead with the
    model's own actionable_insight, then add concrete category follow-ups."""
    bullets: list[str] = []
    insight = (rec.get("actionable_insight") or "").strip()
    if insight:
        bullets.append(insight)
    for a in _CATEGORY_ACTIONS.get(rec.get("category", "other"), []):
        if a not in bullets:
            bullets.append(a)
    return bullets


def _render_priority_details(top_priority: list[dict]) -> None:
    """Replace the top-issues table with detailed cards: what's going wrong (in
    the customer's own words) and the exact actions to take, as bullet points."""
    if not top_priority:
        st.info("No priority issues were produced for this run.")
        return
    for i, r in enumerate(top_priority, start=1):
        cat = r.get("category", "other")
        sev = r.get("severity", 3)
        band, color = _severity_band(sev)
        theme = (r.get("theme_tag") or "").strip()
        title = theme.title() if theme else cat.replace("_", " ").title()
        channel = r.get("channel", "").replace("_", " ").title()
        date = r.get("date", "")
        loc = r.get("location")
        quote = (r.get("text") or "").strip()

        meta_bits = [f'Category: <b>{cat.replace("_", " ").title()}</b>',
                     f'Severity: <b>{sev}/5</b>']
        if channel:
            meta_bits.append(f'Source: <b>{channel}</b>')
        if date:
            meta_bits.append(f'Date: <b>{date}</b>')
        if loc:
            meta_bits.append(f'Location: <b>{loc}</b>')
        meta = " &nbsp;·&nbsp; ".join(meta_bits)

        bullets = _issue_action_bullets(r)
        acts_html = "".join(f"<li>{b}</li>" for b in bullets) or \
            "<li>Assign an owner and confirm the fix.</li>"

        quote_html = (
            f'<div class="blocklabel wronglabel">⚠️ What\'s going wrong</div>'
            f'<div class="quote">“{quote}”</div>'
        ) if quote else ""

        st.markdown(
            f'<div class="icard" style="border-left-color:{color}">'
            f'<div class="ihead">'
            f'<span class="rank" style="background:{color}">{i}</span>'
            f'<span class="ititle">{title}</span>'
            f'<span class="sev" style="background:{color}">{band} · sev {sev}</span>'
            f'</div>'
            f'<div class="meta">{meta}</div>'
            f'{quote_html}'
            f'<div class="blocklabel actionlabel">✅ Actions to take</div>'
            f'<ul class="acts">{acts_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _mood_word(p_pos: float, p_neg: float, p_neu: float) -> str:
    """Turn the mood split into one plain word a non-analyst gets instantly."""
    if p_pos >= 60:
        return "mostly happy"
    if p_neg >= 60:
        return "mostly unhappy"
    if p_pos >= p_neg and p_pos - p_neg >= 10:
        return "leaning happy"
    if p_neg >= p_pos and p_neg - p_pos >= 10:
        return "leaning unhappy"
    return "mixed"


def _dominant_sentiment_category(agg: dict, sentiment: str) -> tuple[str, int] | None:
    """The category with the most feedback of a given mood (positive/negative),
    for 'what people love / dislike most' — read straight from the aggregate."""
    sbc = agg.get("sentiment_by_category", {})
    best: tuple[str, int] | None = None
    for cat, split in sbc.items():
        c = split.get(sentiment, 0)
        if c > 0 and (best is None or c > best[1]):
            best = (cat, c)
    return best


def _render_overview(agg: dict) -> None:
    """A warm, plain-English 'what's going on' paragraph at the very top —
    built entirely from the aggregate (no LLM), so it always renders and reads
    like a person explaining the results to you. This sits above everything."""
    total = agg.get("total", 0)
    if not total:
        return
    sc = agg.get("sentiment_counts", {})
    pos = sc.get("positive", 0)
    neg = sc.get("negative", 0)
    neu = sc.get("neutral", 0)
    p_pos = pos / total * 100
    p_neg = neg / total * 100
    p_neu = neu / total * 100
    mood = _mood_word(p_pos, p_neg, p_neu)

    ranked = agg.get("ranked_categories", [])
    top_issue = ranked[0][0].replace("_", " ") if ranked else None

    loved = _dominant_sentiment_category(agg, "positive")
    disliked = _dominant_sentiment_category(agg, "negative")
    themes = agg.get("top_emergent_themes", [])

    # Sentence 1 — how many people, and the overall feeling, in plain words.
    s1 = (f"We looked at what <b>{total:,}</b> customers said, and overall they’re "
          f"<b>{mood}</b> — about <b>{p_pos:.0f}%</b> spoke positively, "
          f"<b>{p_neg:.0f}%</b> had complaints, and <b>{p_neu:.0f}%</b> were "
          f"in between.")

    # Sentence 2 — what they like most.
    if loved:
        s2 = (f" What people like most is the <b>{loved[0].replace('_', ' ')}</b> "
               f"({loved[1]} happy mention"
               f"{'s' if loved[1] != 1 else ''}).")
    else:
        s2 = ""

    # Sentence 3 — what bugs them most, with the single biggest fix.
    if disliked:
        s3 = (f" What frustrates them most is <b>{disliked[0].replace('_', ' ')}</b> "
              f"({disliked[1]} complaint"
              f"{'s' if disliked[1] != 1 else ''})")
        if top_issue and top_issue != disliked[0].replace("_", " "):
            s3 += (f", and the most urgent thing to fix is "
                   f"<b>{top_issue}</b>")
        s3 += "."
    elif top_issue:
        s3 = f" The most urgent thing to fix is <b>{top_issue}</b>."
    else:
        s3 = ""

    # Sentence 4 — the single most-repeated specific complaint, if we have one.
    s4 = ""
    if themes:
        top_theme, tcount = themes[0]
        if tcount >= 2:
            s4 = (f" The one thing coming up again and again is "
                  f"“<b>{top_theme}</b>” ({tcount} times).")

    st.markdown(
        f'<div class="overview">'
        f'<div class="ov-tag">📣 The big picture — in plain words</div>'
        f'<div class="ov-body">{s1}{s2}{s3}{s4}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_sentiment_line(pos: int, neu: int, neg: int, total: int) -> None:
    """One appealing line conveying the overall mood (replaces the old chart)."""
    if not total:
        return
    p_pos = pos / total * 100
    p_neg = neg / total * 100
    p_neu = neu / total * 100
    # Lead with whichever mood dominates, phrased for a non-analyst.
    if p_pos >= p_neg and p_pos >= p_neu:
        lead = (f"Most customers are happy — <b>{pos:,}</b> of {total:,} "
                f"(<b>{p_pos:.0f}%</b>) left positive feedback.")
    elif p_neg >= p_pos and p_neg >= p_neu:
        lead = (f"Customers are frustrated — <b>{neg:,}</b> of {total:,} "
                f"(<b>{p_neg:.0f}%</b>) left negative feedback.")
    else:
        lead = (f"Customers are mostly on the fence — <b>{neu:,}</b> of "
                f"{total:,} (<b>{p_neu:.0f}%</b>) were neutral.")

    seg = ""
    for label, val, color in (
        ("positive", p_pos, SENTIMENT_COLORS["positive"]),
        ("neutral", p_neu, SENTIMENT_COLORS["neutral"]),
        ("negative", p_neg, SENTIMENT_COLORS["negative"]),
    ):
        if val > 0:
            seg += (f'<span title="{label}: {val:.0f}%" style="width:{val}%;'
                    f'background:{color};"></span>')

    st.markdown(
        f'<div class="moodline">'
        f'<div class="moodtext">{lead} '
        f'<span class="moodkey">'
        f'<span style="color:{SENTIMENT_COLORS["positive"]}">●</span> {p_pos:.0f}% positive &nbsp; '
        f'<span style="color:{SENTIMENT_COLORS["neutral"]}">●</span> {p_neu:.0f}% neutral &nbsp; '
        f'<span style="color:{SENTIMENT_COLORS["negative"]}">●</span> {p_neg:.0f}% negative'
        f'</span></div>'
        f'<div class="moodbar">{seg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences for line-by-line expanders."""
    import re as _re
    parts = _re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _summary_evidence(agg: dict) -> str:
    """A compact 'here are the numbers behind this' block, shown when a summary
    line is expanded — so every claim is traceable to the real aggregate."""
    total = agg.get("total", 0)
    sc = agg.get("sentiment_counts", {})
    ranked = agg.get("ranked_categories", [])
    cat_counts = agg.get("category_counts", {})
    themes = agg.get("top_emergent_themes", [])

    lines = [f"**{total:,}** total feedback items analyzed."]
    if sc:
        lines.append(
            "Mood split: "
            + ", ".join(f"{k} **{v}**" for k, v in sc.items())
            + "."
        )
    if ranked:
        top_cat = ranked[0][0]
        cnt = cat_counts.get(top_cat, 0)
        lines.append(
            f"Biggest issue area: **{top_cat.replace('_', ' ').title()}** "
            f"({cnt} item(s), highest severity-weighted score)."
        )
    if themes:
        lines.append(
            "Most-mentioned specific complaints: "
            + ", ".join(f"“{t}” **{c}**" for t, c in themes[:4])
            + "."
        )
    return "\n\n".join(lines)


def _render_expandable_summary(whats: str, agg: dict) -> None:
    """Render whats_happening as clickable lines: each sentence is an expander
    that opens to show the numbers from the aggregate backing that claim."""
    st.caption("Click any line to see the numbers behind it.")
    sentences = _split_sentences(whats)
    evidence = _summary_evidence(agg)
    for i, sent in enumerate(sentences):
        with st.expander(sent, expanded=False):
            st.markdown(evidence)


def _render_dashboard(results: dict) -> None:
    agg = results["aggregation"]
    summary = results["summary"]

    for w in results["warnings"]:
        st.warning(w, icon="⚠️")

    # 0 — Friendly big-picture overview, above everything (built from the
    #     aggregate, so it always renders even if Call 2 fell through).
    _render_overview(agg)

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
            f'<div class="headline"><span class="tag">What your customers are telling you</span>'
            f'<br>{headline}</div>',
            unsafe_allow_html=True,
        )

    # 2 — KPI row
    total = agg["total"]
    neg = agg["sentiment_counts"].get("negative", 0)
    pos = agg["sentiment_counts"].get("positive", 0)
    neu = agg["sentiment_counts"].get("neutral", 0)
    pct_neg = (neg / total * 100) if total else 0
    top_issue = agg["ranked_categories"][0][0] if agg["ranked_categories"] else "—"
    k1, k2, k3 = st.columns(3)
    k1.metric("Total feedback analyzed", f"{total:,}")
    k2.metric("% negative sentiment", f"{pct_neg:.0f}%")
    k3.metric("Top issue category", top_issue.replace("_", " ").title())

    # 2b — Sentiment mood in one appealing line (replaces the old chart)
    _render_sentiment_line(pos, neu, neg, total)

    st.write("")

    # 3 — Plain-language summary — each sentence is a clickable, expandable line.
    # NOTE: st.expander is a real widget; it can't live inside an injected HTML
    # <div>, so the section title is HTML but the expanders render directly.
    st.markdown('<div class="sec-label">In plain words</div>'
                '<div class="sec-h">What your customers are happy and unhappy about</div>',
                unsafe_allow_html=True)
    whats = summary.get("whats_happening", "").strip()
    if whats:
        _render_expandable_summary(whats, agg)
    else:
        st.info("Summary unavailable for this run.")

    # 4 — Concrete next steps (action + department tag + why)
    st.markdown('<div class="sec-h">✅ Actions to be taken</div>',
                unsafe_allow_html=True)
    _render_next_steps(summary.get("next_steps", []))

    st.divider()

    # 5 — India map: where the issues are coming from
    st.markdown('<div class="sec-h">🗺️ Where feedback is coming from</div>',
                unsafe_allow_html=True)
    st.caption("Each bubble is a city — bigger means more feedback, color is its "
               "top issue. Hover a city to see its full issue and mood breakdown.")
    # Sentinel marks the next Plotly chart so CSS can frame the map only.
    st.markdown('<div class="map-frame-anchor"></div>', unsafe_allow_html=True)
    _india_map(agg.get("location_map", {}))

    st.divider()

    # 6 — Category breakdown (stacked & colored by sentiment; hover for split)
    st.markdown('<div class="sec-h">📊 Feedback by category</div>',
                unsafe_allow_html=True)
    st.caption("Hover any bar to see how many are positive, negative, or neutral "
               "— e.g. how the Taste bar splits by mood.")
    _stacked_sentiment_bar(
        agg["category_counts"], agg.get("sentiment_by_category", {}), "category"
    )

    # 7 — Emergent themes (the dataset-specific section)
    st.markdown("### 🏷️ Opinions found in this dataset")
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
        st.caption("Hover for the mood split of each channel.")
        _stacked_sentiment_bar(
            agg["channel_counts"], agg.get("sentiment_by_channel", {}),
            "channel",
        )
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

    # 9 — Top 5 priority issues — detailed cards (what's wrong + what to do)
    st.markdown("### 🔥 Top 5 priority issues")
    st.caption("The most severe issues, explained — what customers actually said, "
               "and the exact actions to take. Ranked by severity, most recent "
               "first on ties.")
    _render_priority_details(agg["top_priority"])

    st.divider()

    # 10 — Chat Q&A (streaming)
    chat_records = pl.compact_records_for_chat(results["merged"])
    st.markdown("### 💬 Ask anything about this feedback")
    st.caption("Answers come from your actual (PII-scrubbed) feedback records, "
               "not just the summary — so you can ask about specific issues, "
               "products, or what customers said.")
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    prompt = st.chat_input("e.g. What are customers saying about packaging? "
                           "Which product gets the worst reviews?")
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
                for item in client.chat_answer_stream(prompt, agg, chat_records):
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
