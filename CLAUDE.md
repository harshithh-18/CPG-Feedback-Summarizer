# CPG Customer Feedback Summarizer — Build Spec

Read `DATA_SPEC.md` first — the dataset must exist before pipeline work starts.

## Project overview
A Streamlit web app that ingests CPG customer feedback (social media, survey, support-ticket channels), extracts structured sentiment/theme data via Azure OpenAI, aggregates it, and renders a dashboard with an executive summary, a priority-ranked action list, and a grounded chat Q&A box.

## Data ownership
The dataset (`data/feedback.json`) is being built separately by a teammate, following the schema and channel/category targets in `DATA_SPEC.md`. **Do not generate this dataset.** For all development and testing before the real file arrives, create a small placeholder fixture — 10-15 hand-written records covering all 3 channels (`social_review`, `survey`, `support_ticket`) — matching the schema exactly, saved at `data/feedback.json`. Build and test the full pipeline and UI against this placeholder.

The real file will be dropped in at the same path later, same schema, just more records with a different real distribution. No code should assume anything about record count, exact category mix, or specific content — the pipeline must work unchanged on whatever `data/feedback.json` contains at swap-in time. When the real file arrives, the only required action is re-running the existing pipeline against it — if that requires any code change, something in the pipeline was over-fit to the placeholder and needs fixing.

## Tech stack
- Python 3.11+
- Streamlit (latest stable) — the only UI framework, no React/HTML build
- `openai` Python SDK v1.x, using the `AzureOpenAI` client class
- `pandas` for tabular handling
- No database, no auth, no external services beyond Azure OpenAI

## Repository structure (target)
```
cpg-feedback/
  CLAUDE.md
  DATA_SPEC.md
  app.py                  # Streamlit entrypoint
  azure_client.py          # Azure OpenAI client + prompt templates
  pipeline.py              # PII scrub, batching, extraction, aggregation
  data/
    feedback.json           # bundled dataset built from DATA_SPEC.md
  .streamlit/
    config.toml              # theme, hide default chrome
  .env.example
  requirements.txt
  cli.py                   # stretch goal — only after the web app fully works
```

## Environment variables (exact names)
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION` (default `2024-10-21` unless the deployment requires otherwise)

Load via environment variables or Streamlit secrets. Never hardcode, never print, never render these in the UI or logs.

## Data contracts between modules

**Raw feedback record** (matches `DATA_SPEC.md` schema — input to the pipeline)
- `id`: int
- `channel`: `social_review` | `survey` | `support_ticket`
- `date`: str, `YYYY-MM-DD`
- `rating`: int|null, 1-5
- `text`: str
- `subject`: str|null

**Extraction result** (one per input record, output of the Azure extraction call)
- `id`: int — must match the input record's id
- `sentiment`: `positive` | `negative` | `neutral`
- `category`: `taste` | `packaging` | `price` | `availability` | `quality` | `customer_service` | `other`
- `severity`: int, 1-5
- `key_phrase`: str, ≤8 words
- `actionable_insight`: str|null

Do not ask the model to echo back the original text in this call — merge `channel`/`date`/`rating`/`text` back in locally by `id` after the call returns, to save tokens.

**Aggregation output** (pure Python, no LLM call — feeds the dashboard, the summary call, and the chat call)
- `total`: int
- `sentiment_counts`: dict[str, int]
- `category_counts`: dict[str, int]
- `channel_counts`: dict[str, int]
- `ranked_categories`: list of (category, severity_weighted_score), sorted descending, where `severity_weighted_score` = sum of `severity` across all records in that category
- `top_priority`: up to 5 full merged records, sorted by `severity` desc, tie-broken by most recent `date` first
- `date_trend`: dict of week-bucket → count, for at least the top 1-2 categories by volume — enough to drive a trend chart

**Executive summary output**
- `summary`: str, 3-5 sentences, business-toned
- `top_actions`: exactly 3 strings

**Chat answer output**
- `answer`: str, 2-4 sentences, must be answerable strictly from the aggregation JSON passed in — if the aggregation doesn't contain enough to answer, the model should say so rather than inventing detail.

## Azure OpenAI call specs

### Call 1 — Batch extraction
- One call per batch of ~25 records (last batch may be smaller).
- Use JSON mode (`response_format={"type": "json_object"}`) — do not rely on free-text parsing.
- Temperature low (0.2-0.3) — this is extraction, not creative generation.
- System prompt (use verbatim, adjust only if the specific deployment needs different phrasing for reliable JSON):
  > You are a CPG customer feedback analyst. For each feedback item given, extract sentiment, category, severity, a short key phrase, and an actionable insight if one exists. Categories are limited to: taste, packaging, price, availability, quality, customer_service, other. Return only valid JSON matching this shape: {"results": [{"id": <int>, "sentiment": ..., "category": ..., "severity": <1-5 int>, "key_phrase": ..., "actionable_insight": <string or null>}]}. No text outside the JSON object.
- User content: a JSON array of `{"id": ..., "text": ...}` for the batch — PII-scrubbed text only. Never send `subject`/`channel`/`rating` into this call; they add tokens without helping extraction.
- Failure handling: on a JSON parse failure, retry once with a stricter system-prompt reminder ("Return ONLY the JSON object, no markdown fences, no commentary"). If it fails twice, drop that batch from aggregation and surface a non-fatal warning — one bad batch must never crash the whole run.

### Call 2 — Executive summary
- One call, after aggregation is fully computed.
- Input: the aggregation JSON only — never raw feedback text, to keep this call cheap.
- System prompt:
  > You are writing a concise executive summary for a product and marketing team, based on aggregated CPG customer feedback statistics. Be specific, business-toned, and actionable. Return only valid JSON: {"summary": <3-5 sentence string>, "top_actions": [<string>, <string>, <string>]}.

### Call 3 — Chat Q&A
- One call per user question, triggered from the chat box.
- Input: the user's question plus the same aggregation JSON as context — never raw feedback text.
- System prompt:
  > Answer the user's question about CPG customer feedback using only the aggregated data provided. If the data doesn't contain enough information to answer, say so directly rather than guessing. Return only valid JSON: {"answer": <string>}.

## Pipeline logic (exact step order)
1. Load the bundled dataset from `data/feedback.json` by default; support optional CSV/JSON upload as a fallback input path, same schema.
2. PII scrub every record's `text`/`subject` field with regex, before anything else: strip email addresses, phone numbers (multiple common formats), and simple "Full Name"-shaped patterns where feasible. This runs locally, no LLM involved, and runs even on the bundled sample data — defense in depth, and a real point to raise with judges.
3. Split into batches of 25.
4. Run Call 1 per batch, per the failure-handling rule above.
5. Merge extraction results back with `channel`/`date`/`rating`/`text` by `id`.
6. Compute the aggregation (pure Python — counts, severity-weighted ranking, week-bucketed trend). No LLM call in this step.
7. Run Call 2 once, using the aggregation only.
8. Render the dashboard.
9. Chat box is available once the dashboard is rendered; each submitted question triggers Call 3.

## UI spec (Streamlit, `app.py`)
- `st.set_page_config`: wide layout, title "CPG Feedback Summarizer".
- Header: title + one-line description of what the tool does.
- Primary control row: "Load Sample Data" button (loads `data/feedback.json`) and a file uploader as a secondary/fallback input, side by side.
- Once data is loaded: an "Analyze" primary button, runs the full pipeline behind `st.spinner` with a message like "Analyzing N feedback entries...".
- Once results exist, render in this order:
  1. Three-column KPI row: total feedback, % negative, top issue category (`ranked_categories[0]`).
  2. Executive summary section: the summary paragraph, then the 3 `top_actions` as a bulleted list.
  3. Two-column charts row: `category_counts` bar chart, `sentiment_counts` bar chart.
  4. Channel breakdown: small chart or table of `channel_counts` — makes the multi-channel ingestion claim visible.
  5. "Top 5 priority issues" table: category, severity, key_phrase, actionable_insight, truncated text preview.
  6. Chat section at the bottom: `st.chat_input`, responses rendered via `st.chat_message("assistant")`.
- Cheap polish only: `.streamlit/config.toml` sets a non-default accent color (warm orange or green rather than Streamlit's default red); hide the default hamburger menu and "Made with Streamlit" footer via standard Streamlit config.
- Keep loaded feedback and pipeline results in `st.session_state` so button interactions don't lose prior state.

## Error handling / edge cases
- Empty dataset load: clear message, no crash.
- Malformed uploaded file (missing/wrong columns): validate columns before running the pipeline; clear inline error if invalid.
- Azure call failure (network/auth/rate limit): catch, show an inline error banner on the dashboard, allow re-running "Analyze" without needing to reload data; never let it crash the whole app.
- Chat question with no relevant data in the aggregation: handled by the Call 3 system prompt, no special-case code required.

## Security requirements
- Azure credentials only from environment variables/Streamlit secrets — never hardcoded, never logged, never shown in the UI.
- PII scrub runs before every Azure call that includes feedback text (Call 1 only). Calls 2 and 3 never receive raw feedback text at all — only aggregated/derived data — which is a second layer of protection by construction.
- No writes of raw feedback data to disk beyond the bundled `data/feedback.json` fixture — no logging of uploaded content to disk.
- Upload validation: file size cap, extension check (`.csv`/`.json` only).

## Build order — definition of done per step
1. Placeholder `data/feedback.json` exists (10-15 hand-written records covering all 3 channels, matching the `DATA_SPEC.md` schema) — done when `pipeline.py` can load and iterate it with no errors. This gets replaced later by the real dataset at the same path; nothing else should need to change when that happens.
2. `pipeline.py`: PII scrub + batching + Call 1 + merge — done when running against the full fixture produces exactly one extraction result per input record, no unmatched ids.
3. `pipeline.py`: aggregation function — done when it produces every field listed under "Aggregation output" above.
4. `app.py`: dashboard rendering against a hardcoded/sample aggregation object — done when every UI section renders without needing a live Azure call (unblocks UI work in parallel with step 2/3).
5. Wire `app.py` to real `pipeline.py`/`azure_client.py` end to end — done when Load Sample Data → Analyze produces the full dashboard from live Azure calls.
6. Priority ranking section — done when the "Top 5 priority issues" table matches the severity-weighted ranking logic.
7. Chat Q&A — done when a question returns an answer grounded in the aggregation JSON, end to end.
8. UI polish pass — done when the theme and hidden chrome visibly differ from default Streamlit styling.
9. Full regression pass — done when Load Sample Data → Analyze → 2-3 chat questions works twice in a row with no errors.

## Explicitly out of scope
- No live synthetic-data generation feature inside the running app — data is pre-built per `DATA_SPEC.md`.
- No streaming/typewriter UI effects.
- No multi-lingual support.
- No user authentication/accounts.
- No persistent database — in-memory/session state only.