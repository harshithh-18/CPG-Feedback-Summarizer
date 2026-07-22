"""Pipeline: dataset loading, PII scrub, batching, extraction, aggregation.

Follows the exact step order in CLAUDE.md:
  1. Load dataset (bundled or uploaded)
  2. PII scrub every record's text/subject (local regex, no LLM)
  3. Split into batches of 25
  4. Run Call 1 per batch (with retry / drop-batch failure handling)
  5. Merge extraction results back with channel/date/rating/text by id
  6. Compute aggregation (pure Python)
  7/8/9 handled in app.py (summary call, render, chat)

The non-Azure parts (scrub, batching, aggregation) are fully offline-testable.
"""

from __future__ import annotations

import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# --- Schema constants (mirror the Raw feedback record contract in CLAUDE.md) ---
VALID_CHANNELS = {"social_review", "survey", "support_ticket"}
VALID_CATEGORIES = {
    "taste",
    "packaging",
    "price",
    "availability",
    "quality",
    "customer_service",
    "other",
}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
RAW_REQUIRED_FIELDS = {"id", "channel", "date", "rating", "text", "subject"}

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "feedback.json"

BATCH_SIZE = 25
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB cap
ALLOWED_EXTENSIONS = {".json", ".csv"}
MAX_BATCH_WORKERS = 6  # concurrent Call 1 requests in flight


class DatasetError(ValueError):
    """Raised when a feedback dataset fails schema validation."""


# ---------------------------------------------------------------------------
# Step 1 - Loading & validation
# ---------------------------------------------------------------------------
def list_bundled_datasets() -> list[Path]:
    """Discover all .json/.csv datasets in the data/ folder (for the picker)."""
    if not DATA_DIR.exists():
        return []
    files = [
        p
        for p in sorted(DATA_DIR.iterdir())
        if p.suffix.lower() in ALLOWED_EXTENSIONS and p.is_file()
    ]
    return files


def load_feedback(path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    """Load and validate a feedback dataset from a .json or .csv file."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".csv":
        records = _records_from_csv(pd.read_csv(path))
    else:
        with path.open("r", encoding="utf-8") as fh:
            try:
                records = json.load(fh)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Dataset is not valid JSON: {exc}") from exc
    return validate_records(records)


def load_uploaded(name: str, raw: bytes) -> list[dict[str, Any]]:
    """Load a dataset from uploaded bytes with size/extension guards."""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DatasetError("Only .json and .csv files are supported.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DatasetError("Uploaded file exceeds the 5 MB size cap.")

    if ext == ".csv":
        records = _records_from_csv(pd.read_csv(io.BytesIO(raw)))
    else:
        try:
            records = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DatasetError(f"Uploaded file is not valid JSON: {exc}") from exc
    return validate_records(records)


def _records_from_csv(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a CSV DataFrame into raw feedback records, coercing types."""
    required = {"id", "channel", "date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise DatasetError(f"CSV missing required columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rating = row.get("rating")
        rating = None if pd.isna(rating) else int(rating)
        subject = row.get("subject")
        subject = None if (subject is None or pd.isna(subject)) else str(subject)
        records.append(
            {
                "id": int(row["id"]),
                "channel": str(row["channel"]),
                "date": str(row["date"])[:10],
                "rating": rating,
                "text": str(row["text"]),
                "subject": subject,
            }
        )
    return records


def validate_records(records: Any) -> list[dict[str, Any]]:
    """Validate a list of raw feedback records against the schema."""
    if not isinstance(records, list):
        raise DatasetError("Dataset root must be a JSON array of records.")
    if not records:
        raise DatasetError("Dataset is empty.")

    seen_ids: set[int] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DatasetError(f"Record {i} is not an object.")

        missing = RAW_REQUIRED_FIELDS - rec.keys()
        if missing:
            raise DatasetError(f"Record {i} missing fields: {sorted(missing)}")

        if not isinstance(rec["id"], int):
            raise DatasetError(f"Record {i}: 'id' must be an int.")
        if rec["id"] in seen_ids:
            raise DatasetError(f"Duplicate id: {rec['id']}")
        seen_ids.add(rec["id"])

        if rec["channel"] not in VALID_CHANNELS:
            raise DatasetError(
                f"Record {rec['id']}: invalid channel {rec['channel']!r}."
            )

        if not isinstance(rec["date"], str) or len(rec["date"]) != 10:
            raise DatasetError(f"Record {rec['id']}: 'date' must be YYYY-MM-DD.")

        rating = rec["rating"]
        if rating is not None and not (isinstance(rating, int) and 1 <= rating <= 5):
            raise DatasetError(
                f"Record {rec['id']}: 'rating' must be null or an int 1-5."
            )

        if not isinstance(rec["text"], str) or not rec["text"].strip():
            raise DatasetError(f"Record {rec['id']}: 'text' must be a non-empty string.")

        if rec["subject"] is not None and not isinstance(rec["subject"], str):
            raise DatasetError(f"Record {rec['id']}: 'subject' must be null or a string.")

    return records


# ---------------------------------------------------------------------------
# Step 2 - PII scrub (local regex, runs before any Azure call)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone: (415) 555-0192 | 415-555-0192 | 212.555.0148 | +1 415 555 0192 | 4155550192
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
)
# "Full Name" pattern: two (or three) capitalized words in a row.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+){1,2}\b")

# Words that look like names (capitalized pair) but should not be scrubbed.
_NAME_ALLOWLIST = {
    "Customer Service",
    "Quality Control",
    "Best Buy",
    "New York",
}

EMAIL_MASK = "[email]"
PHONE_MASK = "[phone]"
NAME_MASK = "[name]"


def scrub_pii(text: str | None) -> str | None:
    """Remove emails, phone numbers, and simple full-name patterns from text."""
    if text is None:
        return None
    scrubbed = _EMAIL_RE.sub(EMAIL_MASK, text)
    scrubbed = _PHONE_RE.sub(PHONE_MASK, scrubbed)

    def _name_sub(match: re.Match[str]) -> str:
        if match.group(0) in _NAME_ALLOWLIST:
            return match.group(0)
        return NAME_MASK

    scrubbed = _NAME_RE.sub(_name_sub, scrubbed)
    return scrubbed


def scrub_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of records with text/subject PII-scrubbed."""
    out: list[dict[str, Any]] = []
    for rec in records:
        clean = dict(rec)
        clean["text"] = scrub_pii(rec["text"])
        clean["subject"] = scrub_pii(rec.get("subject"))
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Step 3 - Batching
# ---------------------------------------------------------------------------
def make_batches(
    records: list[dict[str, Any]], size: int = BATCH_SIZE
) -> list[list[dict[str, Any]]]:
    """Split records into batches of `size` (last batch may be smaller)."""
    return [records[i : i + size] for i in range(0, len(records), size)]


def batch_payload(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce a batch to the minimal {id, text} payload for Call 1."""
    return [{"id": r["id"], "text": r["text"]} for r in batch]


# ---------------------------------------------------------------------------
# Steps 4 & 5 - Run extraction + merge back
# ---------------------------------------------------------------------------
def run_extraction(
    records: list[dict[str, Any]],
    extract_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    on_warning: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Scrub -> batch -> extract -> merge back by id.

    `extract_fn` takes a [{id, text}] batch payload and returns extraction
    dicts (this is CPGAzureClient.extract_batch in production; a stub in tests).
    A batch that fails is dropped with a non-fatal warning; the run continues.

    Returns (merged_records, warnings).
    """
    warnings: list[str] = []
    scrubbed = scrub_records(records)
    by_id = {r["id"]: r for r in scrubbed}
    batches = make_batches(scrubbed)

    extractions: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_BATCH_WORKERS, len(batches))) as pool:
        future_to_bi = {
            pool.submit(extract_fn, batch_payload(batch)): (bi, batch)
            for bi, batch in enumerate(batches, start=1)
        }
        for future in as_completed(future_to_bi):
            bi, batch = future_to_bi[future]
            try:
                results = future.result()
            except Exception as exc:  # noqa: BLE001 - drop batch, keep going
                msg = f"Batch {bi} dropped ({len(batch)} records): {exc}"
                warnings.append(msg)
                if on_warning:
                    on_warning(msg)
                continue

            for res in results:
                norm = _normalize_extraction(res)
                if norm is not None and norm["id"] in by_id:
                    extractions[norm["id"]] = norm

    merged: list[dict[str, Any]] = []
    for rid, raw in by_id.items():
        ext = extractions.get(rid)
        if ext is None:
            continue  # record's batch was dropped or model omitted it
        merged.append(
            {
                "id": rid,
                "channel": raw["channel"],
                "date": raw["date"],
                "rating": raw["rating"],
                "text": raw["text"],  # already PII-scrubbed
                "sentiment": ext["sentiment"],
                "category": ext["category"],
                "theme_tag": ext["theme_tag"],
                "severity": ext["severity"],
                "key_phrase": ext["key_phrase"],
                "actionable_insight": ext["actionable_insight"],
            }
        )
    return merged, warnings


def _normalize_extraction(res: Any) -> dict[str, Any] | None:
    """Coerce one extraction dict into valid, in-range values (or drop it)."""
    if not isinstance(res, dict) or "id" not in res:
        return None
    try:
        rid = int(res["id"])
    except (TypeError, ValueError):
        return None

    sentiment = str(res.get("sentiment", "")).lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    category = str(res.get("category", "")).lower().strip()
    if category not in VALID_CATEGORIES:
        category = "other"

    try:
        severity = int(res.get("severity", 3))
    except (TypeError, ValueError):
        severity = 3
    severity = max(1, min(5, severity))

    key_phrase = str(res.get("key_phrase", "")).strip()[:120]

    theme_tag = str(res.get("theme_tag", "")).strip()[:60]
    if not theme_tag:
        theme_tag = key_phrase or category

    insight = res.get("actionable_insight")
    insight = None if insight in (None, "", "null") else str(insight).strip()

    return {
        "id": rid,
        "sentiment": sentiment,
        "category": category,
        "theme_tag": theme_tag,
        "severity": severity,
        "key_phrase": key_phrase,
        "actionable_insight": insight,
    }


# ---------------------------------------------------------------------------
# Step 6 - Aggregation (pure Python, no LLM)
# ---------------------------------------------------------------------------
def _week_bucket(date_str: str) -> str:
    """Map a YYYY-MM-DD date to an ISO week label 'YYYY-Www'."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return "unknown"
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def aggregate(merged: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full aggregation contract from merged records."""
    total = len(merged)
    sentiment_counts = Counter(r["sentiment"] for r in merged)
    category_counts = Counter(r["category"] for r in merged)
    channel_counts = Counter(r["channel"] for r in merged)

    # Severity-weighted ranking: sum of severity per category, desc.
    severity_by_cat: dict[str, int] = defaultdict(int)
    for r in merged:
        severity_by_cat[r["category"]] += r["severity"]
    ranked_categories = sorted(
        severity_by_cat.items(), key=lambda kv: kv[1], reverse=True
    )

    # Top priority: up to 5, by severity desc, tie-break most recent date first.
    top_priority = sorted(
        merged,
        key=lambda r: (r["severity"], r["date"]),
        reverse=True,
    )[:5]

    # Date trend: week-bucket -> count, for top 1-2 categories by volume.
    top_vol_cats = [c for c, _ in category_counts.most_common(2)]
    date_trend: dict[str, dict[str, int]] = {}
    for cat in top_vol_cats:
        buckets: dict[str, int] = defaultdict(int)
        for r in merged:
            if r["category"] == cat:
                buckets[_week_bucket(r["date"])] += 1
        date_trend[cat] = dict(sorted(buckets.items()))

    # Emergent themes: ~8-10 most frequent theme_tags (case-insensitive group).
    theme_labels: dict[str, str] = {}  # lowercase -> first-seen display form
    theme_counter: Counter[str] = Counter()
    for r in merged:
        tag = (r.get("theme_tag") or "").strip()
        if not tag:
            continue
        key = tag.lower()
        theme_labels.setdefault(key, tag)
        theme_counter[key] += 1
    top_emergent_themes = [
        (theme_labels[k], c) for k, c in theme_counter.most_common(10)
    ]

    return {
        "total": total,
        "sentiment_counts": dict(sentiment_counts),
        "category_counts": dict(category_counts),
        "channel_counts": dict(channel_counts),
        "ranked_categories": ranked_categories,
        "top_priority": top_priority,
        "date_trend": date_trend,
        "top_emergent_themes": top_emergent_themes,
    }


# ---------------------------------------------------------------------------
# CLI smoke test (offline parts only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_feedback()
    print(f"Loaded {len(data)} records OK.")
    print(f"Channels: {sorted({r['channel'] for r in data})}")

    scrubbed = scrub_records(data)
    dirty = sum(
        1
        for a, b in zip(data, scrubbed)
        if a["text"] != b["text"] or a.get("subject") != b.get("subject")
    )
    print(f"PII scrub modified {dirty} records.")

    batches = make_batches(scrubbed)
    print(f"Batching: {len(data)} records -> {len(batches)} batch(es).")
