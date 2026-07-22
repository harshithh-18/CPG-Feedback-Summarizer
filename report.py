"""PDF report generation for the CPG Feedback Summarizer.

Builds a self-contained, shareable PDF of the full analysis — executive
summary, sentiment trend, key categories, KPIs, category/sentiment/channel
breakdowns, emergent themes, and the top priority issues — from the same
aggregation + summary objects that drive the dashboard. Pure Python (fpdf2),
no system dependencies, no network. Never receives raw un-scrubbed text: the
merged records carried into this module are already PII-scrubbed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fpdf import FPDF

# Palette mirrors the app's light theme so the export feels like one product.
INK = (26, 22, 38)
INK2 = (107, 101, 128)
ACCENT = (108, 92, 231)
CORAL = (225, 112, 85)
TEAL = (0, 163, 131)
BORDER = (231, 227, 245)
SURFACE2 = (243, 241, 251)


def _clean(text: str) -> str:
    """fpdf2's core fonts are latin-1; drop characters they can't encode."""
    if text is None:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def header(self) -> None:  # noqa: D401
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*INK2)
        self.cell(0, 8, _clean("CPG Feedback Summarizer  -  Analysis Report"), align="L")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*INK2)
        self.cell(0, 8, _clean(f"Page {self.page_no()}"), align="C")


def _section_title(pdf: _Report, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, _clean(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + 28, y)
    pdf.ln(3)


def _body(pdf: _Report, text: str, size: int = 10, color=INK) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, 5.4, _clean(text))
    pdf.ln(1)


def _bar_block(pdf: _Report, title: str, pairs: list[tuple[str, int]], color) -> None:
    """Horizontal-bar table: label, proportional bar, value."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
    if not pairs:
        _body(pdf, "No data.", color=INK2)
        return
    max_v = max((v for _, v in pairs), default=1) or 1
    label_w = 42
    bar_max = 110
    for label, val in pairs:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        y = pdf.get_y()
        pdf.cell(label_w, 6, _clean(str(label))[:26])
        # bar
        w = max(1.5, bar_max * (val / max_v))
        pdf.set_fill_color(*color)
        pdf.rect(pdf.l_margin + label_w, y + 1.2, w, 3.6, style="F")
        pdf.set_xy(pdf.l_margin + label_w + w + 2, y)
        pdf.set_text_color(*INK2)
        pdf.cell(0, 6, _clean(str(val)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def build_report(
    aggregation: dict[str, Any], summary: dict[str, Any], source_label: str
) -> bytes:
    """Render the full analysis to PDF bytes."""
    pdf = _Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    total = aggregation.get("total", 0)
    sent = aggregation.get("sentiment_counts", {})
    neg = sent.get("negative", 0)
    pct_neg = (neg / total * 100) if total else 0
    ranked = aggregation.get("ranked_categories", [])
    top_issue = ranked[0][0].replace("_", " ").title() if ranked else "-"

    # --- Cover band ---
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, 210, 46, style="F")
    pdf.set_xy(18, 14)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, _clean("CPG Feedback Analysis Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 10)
    stamp = datetime.now().strftime("%B %d, %Y  %H:%M")
    pdf.cell(
        0,
        6,
        _clean(f"Source: {source_label or 'uploaded file'}   |   Generated {stamp}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_y(54)

    # --- Headline ---
    headline = summary.get("headline", "").strip()
    if headline:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*ACCENT)
        pdf.multi_cell(0, 6.5, _clean(headline))
        pdf.ln(2)

    # --- KPI strip ---
    kpis = [
        ("Total feedback", f"{total:,}"),
        ("% negative", f"{pct_neg:.0f}%"),
        ("Top issue", top_issue),
    ]
    col_w = (210 - 36) / 3
    y0 = pdf.get_y()
    for i, (lbl, val) in enumerate(kpis):
        x = 18 + i * col_w
        pdf.set_fill_color(*SURFACE2)
        pdf.set_draw_color(*BORDER)
        pdf.rect(x, y0, col_w - 3, 18, style="DF")
        pdf.set_xy(x + 3, y0 + 2.5)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*INK2)
        pdf.cell(col_w - 6, 4, _clean(lbl.upper()))
        pdf.set_xy(x + 3, y0 + 8)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*ACCENT)
        pdf.cell(col_w - 6, 8, _clean(val))
    pdf.set_y(y0 + 24)

    # --- Executive summary ---
    _section_title(pdf, "Executive Summary")
    _body(pdf, summary.get("summary", "") or "Summary unavailable for this run.")

    trend = summary.get("sentiment_trend", "").strip()
    if trend:
        _section_title(pdf, "Sentiment & Volume Trend")
        _body(pdf, trend)

    # --- Key categories ---
    key_cats = summary.get("key_categories", [])
    if key_cats:
        _section_title(pdf, "Key Complaint Categories")
        for c in key_cats:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(*CORAL)
            name = c.get("category", "").replace("_", " ").title()
            metric = c.get("metric", "")
            pdf.multi_cell(
                0, 5.6, _clean(f"{name}" + (f"   ({metric})" if metric else ""))
            )
            _body(pdf, c.get("why_it_matters", ""))

    # --- Recommended actions ---
    actions = summary.get("top_actions", [])
    if actions:
        _section_title(pdf, "Recommended Actions")
        for i, a in enumerate(actions, start=1):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*ACCENT)
            pdf.cell(6, 5.4, _clean(f"{i}."))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 5.4, _clean(a))
            pdf.ln(0.5)

    # --- Breakdown bars ---
    pdf.add_page()
    _section_title(pdf, "Breakdowns")

    def _pairs(d: dict, pretty=False) -> list[tuple[str, int]]:
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        if pretty:
            return [(k.replace("_", " ").title(), v) for k, v in items]
        return [(k.title(), v) for k, v in items]

    _bar_block(
        pdf, "Feedback by category", _pairs(aggregation.get("category_counts", {}), True), ACCENT
    )
    _bar_block(
        pdf, "Sentiment breakdown", _pairs(aggregation.get("sentiment_counts", {})), TEAL
    )
    _bar_block(
        pdf, "Feedback by channel", _pairs(aggregation.get("channel_counts", {}), True), (212, 160, 23)
    )

    themes = aggregation.get("top_emergent_themes", [])
    if themes:
        _bar_block(
            pdf,
            "Emergent themes",
            [(t, c) for t, c in themes],
            TEAL,
        )

    # --- Priority issues ---
    top_priority = aggregation.get("top_priority", [])
    if top_priority:
        _section_title(pdf, "Top Priority Issues")
        for r in top_priority:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*INK)
            cat = r.get("category", "").replace("_", " ").title()
            sev = r.get("severity", "-")
            theme = r.get("theme_tag", "")
            head = f"[Severity {sev}]  {cat}" + (f"  -  {theme}" if theme else "")
            pdf.multi_cell(0, 5.4, _clean(head))
            insight = r.get("actionable_insight")
            if insight:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(*ACCENT)
                pdf.multi_cell(0, 5, _clean(f"Action: {insight}"))
            preview = (r.get("text") or "")[:160]
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*INK2)
            pdf.multi_cell(0, 4.8, _clean(preview))
            pdf.ln(1.5)

    out = pdf.output()
    return bytes(out)
