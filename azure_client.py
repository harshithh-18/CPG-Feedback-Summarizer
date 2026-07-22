"""Azure OpenAI client + prompt templates for the three model calls.

All three calls use JSON mode. Credentials come only from environment
variables / Streamlit secrets and are never logged or returned to the UI.

Call 1 - Batch extraction  (feedback text in, structured extraction out)
Call 2 - Executive summary  (aggregation JSON in, summary out)
Call 3 - Chat Q&A           (question + aggregation JSON in, grounded answer out)
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AzureOpenAI, OpenAI

# --- Environment variable names (exact, per CLAUDE.md) ---
ENV_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"
DEFAULT_API_VERSION = "2024-10-21"

EXTRACTION_BATCH_SIZE = 25

# Some newer deployments (e.g. gpt-5.x) reject any non-default `temperature`.
# When True, temperature is omitted from every request. Auto-enabled for v1
# endpoints, overridable via AZURE_OPENAI_OMIT_TEMPERATURE=1/0.
ENV_OMIT_TEMPERATURE = "AZURE_OPENAI_OMIT_TEMPERATURE"

# --- System prompts (verbatim from CLAUDE.md) ---
EXTRACTION_SYSTEM_PROMPT = (
    "You are a CPG customer feedback analyst. For each feedback item given, "
    "extract sentiment, category, a specific theme tag, severity, a short key "
    "phrase, and an actionable insight if one exists. Categories are limited "
    "to: taste, packaging, price, availability, quality, customer_service, "
    "other — always pick from this fixed list. The theme tag is different: it "
    "is NOT from a fixed list, it should be the most specific 2-4 word "
    "description of the actual issue or praise in this particular piece of "
    'feedback (e.g. "resealable pouch tears", "bulk discount request") — be '
    "concrete and specific to what this feedback actually says, not a "
    "restatement of the category. Return only valid JSON matching this shape: "
    '{"results": [{"id": <int>, "sentiment": ..., "category": ..., '
    '"theme_tag": ..., "severity": <1-5 int>, "key_phrase": ..., '
    '"actionable_insight": <string or null>}]}. No text outside the JSON '
    "object."
)

EXTRACTION_RETRY_SUFFIX = (
    " Return ONLY the JSON object, no markdown fences, no commentary."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are explaining a product feedback analysis to someone who has never "
    "seen this data and has no background in statistics — explain it the way "
    "you'd explain it to a smart 12-year-old who just wants to know what's "
    "going on and what to do about it.\n\n"
    "You will be given ONLY an aggregate object (in the user message), computed "
    "from real feedback records: sentiment counts, category counts, channel "
    "counts, a severity-weighted category ranking, a week-bucketed volume "
    "trend, and the most frequent emergent theme tags in THIS dataset.\n\n"
    "Produce a JSON object with exactly these fields:\n\n"
    "headline — one sentence. Must name a specific number and a specific "
    "theme_tag or category from the data. No generic openers (\"Overall,\", "
    "\"In summary,\", \"The data shows\").\n\n"
    "whats_happening — 2 to 4 sentences in plain, everyday language explaining "
    "what's going on and why. Every sentence must be traceable to a specific "
    "value in the aggregate object. Do not use the words \"sentiment\", "
    "\"aggregate\", \"metric\", or \"taxonomy\" unless you define the word in "
    "the same sentence you use it.\n\n"
    "Example — bad: \"Customer feedback indicates mixed sentiment regarding "
    "product quality, with several respondents citing concerns.\"\n"
    "Example — good: \"Out of 340 people who gave feedback, 61 of them — about "
    "1 in 5 — said the same thing: the resealable pouch tears when they try to "
    "open it. That's the single biggest complaint, bigger than price or "
    "taste.\"\n\n"
    "next_steps — exactly 3 objects, each with:\n"
    "  - action: one concrete sentence describing exactly what to do\n"
    "  - department: one of \"Product/R&D\", \"Marketing\", \"Customer "
    "Service\"\n"
    "  - why: one sentence connecting the action back to the specific number "
    "or theme driving it\n\n"
    "Hard rules:\n"
    "- No section headings anywhere in your output (\"Key Insights\", "
    "\"Summary\", \"Overview\", \"Analysis\", \"Findings\" are all banned).\n"
    "- If a sentence could be pasted unchanged into a summary of a different "
    "dataset, rewrite it — it isn't specific enough.\n"
    "- Do not hedge with \"may\", \"could\", or \"seems\" when the aggregate "
    "data gives you a definite number.\n\n"
    "Return only valid JSON with exactly these keys: "
    '{"headline": <string>, "whats_happening": <string>, "next_steps": '
    '[{"action": <string>, "department": <string>, "why": <string>}]}.'
)

# Departments Call 2 may tag a next step with, mapped to canonical display form.
_DEPARTMENT_MAP = {
    "product/r&d": "Product/R&D",
    "product/rd": "Product/R&D",
    "product": "Product/R&D",
    "r&d": "Product/R&D",
    "rd": "Product/R&D",
    "marketing": "Marketing",
    "customer service": "Customer Service",
    "customer support": "Customer Service",
    "support": "Customer Service",
}

CHAT_SYSTEM_PROMPT = (
    "You help a business owner understand their own customers' feedback. You "
    "are given two things: aggregated statistics, and a list of the actual "
    "individual feedback records (each already stripped of personal "
    "information, with its category, sentiment, severity, theme, and the "
    "customer's own words). Answer the user's question using this data. You may "
    "cite what specific customers actually said, count how many mention "
    "something, and point to the real theme tags — ground every claim in the "
    "records or the aggregates provided, never invent feedback that isn't "
    "there. Reply in plain, friendly language a non-technical person "
    "understands, in 2-4 sentences. If the provided data genuinely doesn't "
    "cover the question, say so plainly instead of guessing. Return only valid "
    'JSON: {"answer": <string>}.'
)


def extract_partial_string(raw: str, key: str) -> str:
    """Best-effort pull of a JSON string value while the object is still
    streaming in. Returns whatever of ``key``'s value has arrived so far, with
    escape sequences decoded, so the UI can render growing prose instead of raw
    JSON braces. Returns "" if the key hasn't started streaming yet."""
    marker = f'"{key}"'
    ki = raw.find(marker)
    if ki == -1:
        return ""
    ci = raw.find(":", ki + len(marker))
    if ci == -1:
        return ""
    qi = raw.find('"', ci + 1)
    if qi == -1:
        return ""
    out: list[str] = []
    i = qi + 1
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\":
            if i + 1 >= n:
                break  # escape not fully arrived yet
            nxt = raw[i + 1]
            out.append(
                {"n": "\n", "t": "\t", "r": "", '"': '"', "\\": "\\", "/": "/"}.get(
                    nxt, nxt
                )
            )
            i += 2
            continue
        if ch == '"':
            break  # closing quote — value complete
        out.append(ch)
        i += 1
    return "".join(out)


class AzureConfigError(RuntimeError):
    """Raised when required Azure OpenAI configuration is missing."""


class AzureCallError(RuntimeError):
    """Raised when an Azure OpenAI call fails after retries."""


def _secret(name: str) -> str | None:
    """Read a config value from env first, then Streamlit secrets if available.

    Never logs or prints the value.
    """
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def azure_is_configured() -> bool:
    """True if the minimum required Azure settings are present."""
    return all(_secret(n) for n in (ENV_ENDPOINT, ENV_API_KEY, ENV_DEPLOYMENT))


def missing_config() -> list[str]:
    """List which required Azure env vars are absent (for a clear UI banner)."""
    return [n for n in (ENV_ENDPOINT, ENV_API_KEY, ENV_DEPLOYMENT) if not _secret(n)]


class CPGAzureClient:
    """Wrapper over the OpenAI/AzureOpenAI client holding the deployment name.

    Supports both endpoint styles:
      * Classic Azure endpoint (``https://<res>.openai.azure.com/``) → AzureOpenAI
        client with an ``api_version``.
      * New v1 API surface (endpoint ending in ``/openai/v1/``) → plain OpenAI
        client with ``base_url``; ``api_version`` is not used.
    """

    def __init__(self) -> None:
        endpoint = _secret(ENV_ENDPOINT)
        api_key = _secret(ENV_API_KEY)
        deployment = _secret(ENV_DEPLOYMENT)
        api_version = _secret(ENV_API_VERSION) or DEFAULT_API_VERSION

        if not (endpoint and api_key and deployment):
            raise AzureConfigError(
                "Missing Azure OpenAI configuration: "
                + ", ".join(missing_config())
            )

        self._deployment = deployment

        is_v1 = "/openai/v1" in endpoint.rstrip("/")
        if is_v1:
            # New v1 surface: plain OpenAI client pointed at the v1 base_url.
            base_url = endpoint if endpoint.endswith("/") else endpoint + "/"
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )

        # Whether to omit `temperature` (some newer models reject non-default).
        omit_env = _secret(ENV_OMIT_TEMPERATURE)
        if omit_env is not None:
            self._omit_temperature = omit_env.strip() not in ("0", "false", "False", "")
        else:
            self._omit_temperature = is_v1  # default: omit on v1 endpoints

    # --- low-level JSON call ---
    def _json_call(
        self, system_prompt: str, user_content: str, temperature: float
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._deployment,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if not self._omit_temperature:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return json.loads(content)

    def _json_call_stream(
        self, system_prompt: str, user_content: str, temperature: float
    ):
        """Streaming variant of `_json_call`.

        Yields incremental content strings as they arrive, then yields a final
        ``("__result__", dict)`` tuple with the parsed JSON. JSON mode still
        applies, so the accumulated text is a single JSON object we parse once
        the stream completes. If parsing fails, the tuple carries ``{}``.
        """
        kwargs: dict[str, Any] = {
            "model": self._deployment,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
        }
        if not self._omit_temperature:
            kwargs["temperature"] = temperature

        acc: list[str] = []
        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                acc.append(piece)
                yield piece
        raw = "".join(acc)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        yield ("__result__", parsed)

    # --- Call 1: batch extraction ---
    def extract_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract structured fields for one batch of {id, text} items.

        Retries once with a stricter prompt on JSON parse failure. Raises
        AzureCallError if it fails twice so the caller can drop the batch.
        """
        user_content = json.dumps(items, ensure_ascii=False)
        try:
            data = self._json_call(EXTRACTION_SYSTEM_PROMPT, user_content, 0.2)
            return _coerce_results(data)
        except (json.JSONDecodeError, ValueError, KeyError):
            # Retry once with stricter instruction.
            try:
                data = self._json_call(
                    EXTRACTION_SYSTEM_PROMPT + EXTRACTION_RETRY_SUFFIX,
                    user_content,
                    0.2,
                )
                return _coerce_results(data)
            except Exception as exc:  # noqa: BLE001 - surfaced as non-fatal warning
                raise AzureCallError(f"Extraction batch failed twice: {exc}") from exc
        except Exception as exc:  # network/auth/rate-limit
            raise AzureCallError(f"Azure extraction call failed: {exc}") from exc

    # --- Call 2: executive summary ---
    @classmethod
    def _normalize_summary(cls, data: dict[str, Any]) -> dict[str, Any]:
        headline = str(data.get("headline", "")).strip()
        whats_happening = str(data.get("whats_happening", "")).strip()

        raw_steps = data.get("next_steps", [])
        next_steps: list[dict[str, str]] = []
        if isinstance(raw_steps, list):
            for s in raw_steps:
                if not isinstance(s, dict):
                    continue
                action = str(s.get("action", "")).strip()
                if not action:
                    continue
                department = _DEPARTMENT_MAP.get(
                    str(s.get("department", "")).strip().lower(), "Product/R&D"
                )
                next_steps.append(
                    {
                        "action": action,
                        "department": department,
                        "why": str(s.get("why", "")).strip(),
                    }
                )
        next_steps = next_steps[:3]

        return {
            "headline": headline,
            "whats_happening": whats_happening,
            "next_steps": next_steps,
        }

    def executive_summary(self, aggregation: dict[str, Any]) -> dict[str, Any]:
        user_content = json.dumps(aggregation, ensure_ascii=False, default=str)
        try:
            data = self._json_call(SUMMARY_SYSTEM_PROMPT, user_content, 0.4)
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Executive summary call failed: {exc}") from exc
        return self._normalize_summary(data)

    def executive_summary_stream(self, aggregation: dict[str, Any]):
        """Streaming Call 2. Yields raw content chunks, then a final
        ``("__result__", normalized_summary_dict)`` tuple."""
        user_content = json.dumps(aggregation, ensure_ascii=False, default=str)
        try:
            for item in self._json_call_stream(
                SUMMARY_SYSTEM_PROMPT, user_content, 0.4
            ):
                if isinstance(item, tuple) and item[0] == "__result__":
                    yield ("__result__", self._normalize_summary(item[1]))
                else:
                    yield item
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Executive summary call failed: {exc}") from exc

    # --- Call 3: chat Q&A ---
    def _chat_payload(
        self,
        question: str,
        aggregation: dict[str, Any],
        records: list[dict[str, Any]] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "question": question,
            "aggregated_data": aggregation,
        }
        if records is not None:
            payload["feedback_records"] = records
        return json.dumps(payload, ensure_ascii=False, default=str)

    def chat_answer(
        self,
        question: str,
        aggregation: dict[str, Any],
        records: list[dict[str, Any]] | None = None,
    ) -> str:
        user_content = self._chat_payload(question, aggregation, records)
        try:
            data = self._json_call(CHAT_SYSTEM_PROMPT, user_content, 0.3)
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Chat call failed: {exc}") from exc
        return str(data.get("answer", "")).strip()

    def chat_answer_stream(
        self,
        question: str,
        aggregation: dict[str, Any],
        records: list[dict[str, Any]] | None = None,
    ):
        """Streaming Call 3. Yields raw content chunks, then a final
        ``("__result__", answer_string)`` tuple."""
        user_content = self._chat_payload(question, aggregation, records)
        try:
            for item in self._json_call_stream(CHAT_SYSTEM_PROMPT, user_content, 0.3):
                if isinstance(item, tuple) and item[0] == "__result__":
                    yield ("__result__", str(item[1].get("answer", "")).strip())
                else:
                    yield item
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Chat call failed: {exc}") from exc


def _coerce_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the extraction response's `results` list."""
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError("Extraction response missing 'results' key.")
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list.")
    return results
