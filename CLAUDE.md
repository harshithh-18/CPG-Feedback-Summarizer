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
- `headline`: str, one sentence — must name a specific number and a specific `theme_tag` or category from the aggregate. No generic openers.
- `whats_happening`: str, 2-4 sentences, plain everyday language; every claim traceable to a specific value in the aggregate.
- `next_steps`: exactly 3 objects, each `{ "action": str, "department": "Product/R&D" | "Marketing" | "Customer Service", "why": str }` — `why` connects the action back to the specific number/theme driving it.

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
- Input: the aggregation JSON only — never raw feedback text, to keep this call cheap. The aggregate object is passed as the user message; the system prompt describes the fields it will contain.
- Response schema (JSON mode):
  ```json
  {
    "headline": "<one sentence naming a real number and a real theme_tag/category>",
    "whats_happening": "<2-4 plain-language sentences, every claim tied to a specific number/category/theme_tag>",
    "next_steps": [
      { "action": "<one concrete sentence>", "department": "Product/R&D | Marketing | Customer Service", "why": "<one sentence tying the action to the number/theme driving it>" }
    ]
  }
  ```
  `next_steps` must contain exactly 3 objects.
- System prompt:
  > You are explaining a product feedback analysis to someone who has never seen this data and has no background in statistics — explain it the way you'd explain it to a smart 12-year-old who just wants to know what's going on and what to do about it.
  >
  > You will be given ONLY an aggregate object (in the user message), computed from real feedback records: sentiment counts, category counts, channel counts, a severity-weighted category ranking, a week-bucketed volume trend, and the most frequent emergent theme tags in THIS dataset.
  >
  > Produce a JSON object with exactly these fields:
  >
  > headline — one sentence. Must name a specific number and a specific theme_tag or category from the data. No generic openers ("Overall,", "In summary,", "The data shows").
  >
  > whats_happening — 2 to 4 sentences in plain, everyday language explaining what's going on and why. Every sentence must be traceable to a specific value in the aggregate object. Do not use the words "sentiment," "aggregate," "metric," or "taxonomy" unless you define the word in the same sentence you use it.
  >
  > Example — bad: "Customer feedback indicates mixed sentiment regarding product quality, with several respondents citing concerns."
  > Example — good: "Out of 340 people who gave feedback, 61 of them — about 1 in 5 — said the same thing: the resealable pouch tears when they try to open it. That's the single biggest complaint, bigger than price or taste."
  >
  > next_steps — exactly 3 objects, each with: action (one concrete sentence describing exactly what to do), department (one of "Product/R&D", "Marketing", "Customer Service"), why (one sentence connecting the action back to the specific number or theme driving it).
  >
  > Hard rules: No section headings anywhere in your output ("Key Insights," "Summary," "Overview," "Analysis," "Findings" are all banned). If a sentence could be pasted unchanged into a summary of a different dataset, rewrite it. Do not hedge with "may," "could," or "seems" when the aggregate data gives you a definite number.

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
  1. Three-column KPI row: total feedback, % negative, top issue category (`ranked_categories[0]`), plus a one-line sentiment-mood summary (no standalone sentiment bar chart — the mood is conveyed inline).
  2. Executive summary section: the `headline`, then `whats_happening` rendered as clickable/expandable lines (each sentence expands to show the aggregate numbers behind it), then the 3 `next_steps` as a bulleted list — each bullet shows the `action`, a small `department` tag, and the `why` on a second line. No hardcoded section labels.
  3. `category_counts` bar chart (Plotly, horizontal, stacked and colored by sentiment — hovering a bar reveals its positive/negative/neutral split).
  4. India map (Plotly Scattergeo): one bubble per resolved Indian city, sized by feedback volume and colored by its dominant issue; hover shows the city's per-issue and per-sentiment breakdown. Falls back to a friendly placeholder when the dataset carries no `location` field.
  5. Channel breakdown: `channel_counts` as a Plotly stacked-by-sentiment bar (hover for the split) — makes the multi-channel ingestion claim visible.
  6. "Top 5 priority issues" table: category, severity, key_phrase, actionable_insight, truncated text preview.
  7. Chat section at the bottom: `st.chat_input`, responses rendered via `st.chat_message("assistant")`.
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