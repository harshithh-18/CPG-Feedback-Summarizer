# CPG Customer Feedback Summarizer

An AI-driven tool that ingests bulk Consumer Packaged Goods (CPG) customer
feedback from **social media, surveys, and support tickets**, extracts
structured sentiment/theme data via **Azure OpenAI**, and renders a Streamlit
dashboard with an **executive summary**, a **priority-ranked action list**, and
a **grounded chat Q&A** box.

Built for the TCS AI Fridays problem statement: *Consumer Packaged Goods
Customer Feedback Summarizer*.

---

## What it does

1. **Ingest** — load a bundled dataset (pick from any `.json`/`.csv` in `data/`)
   or upload your own file with the same schema.
2. **PII scrub** — emails, phone numbers, and full-name patterns are stripped
   locally with regex **before** any data leaves the machine.
3. **Extract** — feedback is batched (25/call) and sent to Azure OpenAI in JSON
   mode to extract sentiment, category, severity, a key phrase, and an
   actionable insight per record.
4. **Aggregate** — pure-Python rollups: sentiment/category/channel counts, a
   severity-weighted category ranking, top-5 priority issues, and a
   week-bucketed volume trend.
5. **Summarize** — one Azure call turns the aggregation into a business-toned
   executive summary + 3 recommended actions.
6. **Chat** — ask questions answered *strictly* from the aggregated data.

## Architecture

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI — dataset picker/upload, KPIs, charts, priority table, chat |
| `pipeline.py` | Load/validate, PII scrub, batching, extraction orchestration, merge, aggregation (no LLM in aggregation) |
| `azure_client.py` | `AzureOpenAI` wrapper + the 3 prompt templates (extraction, summary, chat) |
| `data/feedback.json` | Bundled sample dataset (schema reference) |
| `.streamlit/config.toml` | Theme + hidden default chrome |

**Privacy by construction:** only Call 1 (extraction) ever sees feedback text,
and only after PII scrubbing. Calls 2 (summary) and 3 (chat) receive **only
aggregated/derived data** — never raw feedback.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in your Azure OpenAI values
```

Required environment variables (see `.env.example`):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION` (default `2024-10-21`)

Credentials are read from the environment or Streamlit secrets — never
hardcoded, logged, or shown in the UI.

## Run

```bash
streamlit run app.py
```

Then: **pick a dataset → Load → Analyze → explore the dashboard → ask questions**.

### Testing multiple datasets

Drop any number of `.json`/`.csv` files (same schema) into `data/`. The app
auto-discovers them in the dropdown, so you can swap between datasets without
restarting.

## Data schema

Each feedback record:

```json
{
  "id": 1,
  "channel": "social_review | survey | support_ticket",
  "date": "YYYY-MM-DD",
  "rating": 1,
  "text": "the feedback text",
  "subject": "optional subject or null"
}
```

## Offline sanity check

The non-Azure pipeline (scrub, batching, aggregation) runs standalone:

```bash
python pipeline.py
```
