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
    "You are writing an executive summary for a product and marketing team, "
    "based on aggregated CPG customer feedback statistics, including a list of "
    "the most frequent emergent themes in this specific dataset. Your job is "
    "to make this read as clearly about THIS dataset, not a generic template "
    "— cite actual numbers, percentages, category names, and theme tags from "
    "the data provided. Never write a sentence that could apply unchanged to "
    "any other dataset. First write a headline: one punchy sentence, under 12 "
    "words, naming the single most notable specific finding. Then a 3-5 "
    "sentence summary citing at least one number and one theme tag or key "
    "phrase verbatim from the data. Then exactly 3 top actions, each naming a "
    "specific category or theme tag with its frequency or severity, not "
    'generic advice. Return only valid JSON: {"headline": <string>, '
    '"summary": <string>, "top_actions": [<string>, <string>, <string>]}.'
)

CHAT_SYSTEM_PROMPT = (
    "Answer the user's question about CPG customer feedback using only the "
    "aggregated data provided. If the data doesn't contain enough information "
    "to answer, say so directly rather than guessing. Return only valid JSON: "
    '{"answer": <string>}.'
)


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
    def executive_summary(self, aggregation: dict[str, Any]) -> dict[str, Any]:
        user_content = json.dumps(aggregation, ensure_ascii=False, default=str)
        try:
            data = self._json_call(SUMMARY_SYSTEM_PROMPT, user_content, 0.4)
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Executive summary call failed: {exc}") from exc

        headline = str(data.get("headline", "")).strip()
        summary = str(data.get("summary", "")).strip()
        actions = data.get("top_actions", [])
        if not isinstance(actions, list):
            actions = []
        actions = [str(a).strip() for a in actions if str(a).strip()][:3]
        return {"headline": headline, "summary": summary, "top_actions": actions}

    # --- Call 3: chat Q&A ---
    def chat_answer(self, question: str, aggregation: dict[str, Any]) -> str:
        payload = {
            "question": question,
            "aggregated_data": aggregation,
        }
        user_content = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            data = self._json_call(CHAT_SYSTEM_PROMPT, user_content, 0.3)
        except Exception as exc:  # noqa: BLE001
            raise AzureCallError(f"Chat call failed: {exc}") from exc
        return str(data.get("answer", "")).strip()


def _coerce_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the extraction response's `results` list."""
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError("Extraction response missing 'results' key.")
    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list.")
    return results
