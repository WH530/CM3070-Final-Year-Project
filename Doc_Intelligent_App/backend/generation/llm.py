"""Evidence-constrained answer generation with inline citations.

The prompt separates system instructions from retrieved document content
with clear delimiters, so text found inside documents cannot override the
instructions (a basic but important mitigation against prompt-injection —
Preliminary Report §3.4). The LLM is instructed to answer only from the
supplied evidence, attach a citation marker to each substantive claim, and
state explicitly when the evidence is insufficient.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

import config
from retrieval.pipeline import RetrievedChunk

# backend/prompt.yaml is the master prompt file, shared by every model
# call in the backend — this module reads its own "qa" section from it.
with open(
    Path(__file__).resolve().parent.parent / "prompt.yaml", encoding="utf-8"
) as _f:
    _PROMPTS = yaml.safe_load(_f)["qa"]

SYSTEM_PROMPT = _PROMPTS["system_prompt"].strip()
USER_PROMPT_TEMPLATE = _PROMPTS["user_prompt_template"].strip()


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict]  # [{marker, chunk_id, title, page_start, page_end, heading}]


@dataclass(frozen=True)
class ProviderStatus:
    """Safe provider readiness information suitable for the public API."""

    available: bool
    detail: str
    checked: bool = True


class ProviderUnavailableError(RuntimeError):
    """A selected, allowlisted provider cannot currently serve a request."""

    def __init__(self, provider_id: str, user_message: str) -> None:
        super().__init__(user_message)
        self.provider_id = provider_id
        self.user_message = user_message


def is_openrouter_configured() -> bool:
    """Return whether config.py contains a non-placeholder OpenRouter key."""

    key = config.OPENROUTER_API_KEY.strip()
    normalised = key.casefold()
    return bool(key) and not any(
        marker in normalised
        for marker in ("replace-with", "your-openrouter", "placeholder")
    )


_PROVIDER_STATUS: dict[str, ProviderStatus] = {
    config.OPENROUTER_PROVIDER_ID: ProviderStatus(
        available=is_openrouter_configured(),
        detail=(
            "OpenRouter API key is configured; connection check pending."
            if is_openrouter_configured()
            else "OpenRouter API key is not configured in backend/config.py."
        ),
        checked=False,
    ),
    config.OLLAMA_PROVIDER_ID: ProviderStatus(
        available=False,
        detail="Local Ollama runtime has not been checked yet.",
        checked=False,
    ),
}


def _build_context_block(retrieved_chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_ref = (
            (
                f"p.{chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp.{chunk.page_start}-{chunk.page_end}"
            )
            if chunk.page_start
            else "page unknown"
        )
        header = (
            f"[{i}] {chunk.title} "
            f"({page_ref}{', ' + chunk.heading if chunk.heading else ''})"
        )
        blocks.append(f'<evidence source="{header}">\n{chunk.text}\n</evidence>')
    return (
        "\n\n".join(blocks)
        if blocks
        else "(No evidence chunks were retrieved for this question.)"
    )


def _model_spec(model_id: str | None = None) -> dict:
    selected_model = config.DEFAULT_MODEL if model_id is None else model_id
    try:
        return config.MODEL_BY_ID[selected_model]
    except KeyError as exc:
        raise ValueError("Unsupported model selection.") from exc


def _client_for_provider(provider_id: str, *, healthcheck: bool = False) -> OpenAI:
    """Build a client only for a configured provider identifier.

    No URL is accepted from a caller. Keeping this explicit mapping on the
    server prevents model-selection input from becoming an arbitrary proxy.
    """

    common_kwargs = {"timeout": 5.0, "max_retries": 0} if healthcheck else {}
    if provider_id == config.OPENROUTER_PROVIDER_ID:
        if not is_openrouter_configured():
            raise ProviderUnavailableError(
                provider_id,
                "OpenRouter is not configured. Add your API key in backend/config.py.",
            )
        return OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            **common_kwargs,
        )
    if provider_id == config.OLLAMA_PROVIDER_ID:
        return OpenAI(
            api_key=config.OLLAMA_API_KEY,
            base_url=config.OLLAMA_BASE_URL,
            **common_kwargs,
        )
    raise ValueError("Unsupported model provider.")


def get_provider_status(provider_id: str) -> ProviderStatus:
    """Return cached readiness without performing network I/O."""

    if provider_id not in config.LLM_PROVIDERS:
        raise ValueError("Unsupported model provider.")

    # A checked-in placeholder must never be presented as usable, even if a
    # stale status from an earlier check says otherwise.
    if provider_id == config.OPENROUTER_PROVIDER_ID:
        if not is_openrouter_configured():
            return ProviderStatus(
                False,
                "OpenRouter API key is not configured in backend/config.py.",
                checked=True,
            )
        status = _PROVIDER_STATUS[provider_id]
        if not status.checked:
            return ProviderStatus(
                True,
                "OpenRouter API key is configured; connection check pending.",
                checked=False,
            )
        return status

    return _PROVIDER_STATUS[provider_id]


def get_model_status(model_id: str) -> ProviderStatus:
    """Return readiness for an allowlisted public model identifier."""

    return get_provider_status(_model_spec(model_id)["provider_id"])


def _record_status(provider_id: str, available: bool, detail: str) -> tuple[bool, str]:
    _PROVIDER_STATUS[provider_id] = ProviderStatus(available, detail, checked=True)
    return available, detail


def check_connection(provider_id: str) -> tuple[bool, str]:
    """Check one fixed provider and cache a safe readiness description.

    OpenRouter uses its authenticated key endpoint so startup does not consume
    a free-model completion request. Ollama uses its OpenAI-compatible
    model list, which verifies both server health and that the configured local
    model has already been installed. Neither path downloads a model.
    """

    if provider_id not in config.LLM_PROVIDERS:
        raise ValueError("Unsupported model provider.")

    if provider_id == config.OPENROUTER_PROVIDER_ID and not is_openrouter_configured():
        return _record_status(
            provider_id,
            False,
            "API key is not configured in backend/config.py.",
        )

    try:
        if provider_id == config.OPENROUTER_PROVIDER_ID:
            request = Request(
                f"{config.OPENROUTER_BASE_URL}/key",
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                method="GET",
            )
            with urlopen(request, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
            key_data = data.get("data", {}) if isinstance(data, dict) else {}
            usage_daily = key_data.get("usage_daily")
            usage_text = (
                f"{usage_daily:g}"
                if isinstance(usage_daily, int | float)
                else "unknown"
            )
            return _record_status(
                provider_id,
                True,
                f"API key authenticated. Daily usage: {usage_text}.",
            )

        client = _client_for_provider(provider_id, healthcheck=True)
        installed_models = {
            model.id.casefold()
            for model in client.models.list().data
            if isinstance(getattr(model, "id", None), str)
        }
        if config.OLLAMA_MODEL.casefold() not in installed_models:
            return _record_status(
                provider_id,
                False,
                f"Ollama is running, but {config.OLLAMA_MODEL} is not installed.",
            )
        return _record_status(
            provider_id,
            True,
            f"Ollama is ready with {config.OLLAMA_MODEL}.",
        )
    except AuthenticationError:
        return _record_status(
            provider_id,
            False,
            "The configured API key was rejected.",
        )
    except RateLimitError:
        return _record_status(provider_id, False, "Provider usage limit reached.")
    except HTTPError as exc:
        if exc.code == 401:
            return _record_status(
                provider_id,
                False,
                "The configured API key was rejected.",
            )
        if exc.code == 429:
            return _record_status(provider_id, False, "Provider usage limit reached.")
        return _record_status(
            provider_id,
            False,
            f"OpenRouter key check failed with HTTP {exc.code}.",
        )
    except (APIConnectionError, APITimeoutError):
        detail = (
            "Could not reach OpenRouter."
            if provider_id == config.OPENROUTER_PROVIDER_ID
            else f"Ollama is not reachable at {config.OLLAMA_BASE_URL}."
        )
        return _record_status(provider_id, False, detail)
    except URLError:
        return _record_status(provider_id, False, "Could not reach OpenRouter.")
    except Exception:  # noqa: BLE001 - startup diagnostics must never stop the app
        detail = (
            "OpenRouter connection check failed."
            if provider_id == config.OPENROUTER_PROVIDER_ID
            else "Ollama returned an unexpected health-check response."
        )
        return _record_status(provider_id, False, detail)


def _generate_text(model_spec: dict, user_prompt: str) -> str:
    if model_spec["provider_id"] == config.OLLAMA_PROVIDER_ID:
        # Ollama's OpenAI-compatible endpoint (below) ignores `think`, so
        # qwen3.5's reasoning pass runs regardless and can consume the whole
        # max_tokens budget before any visible answer is emitted
        # (finish_reason="length", empty content) — only the native
        # /api/chat endpoint honors think:False.
        payload = {
            "model": model_spec["api_model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": config.GENERATION_TEMPERATURE,
                "num_predict": config.GENERATION_MAX_TOKENS,
            },
        }
        request = Request(
            f"{config.OLLAMA_API_BASE}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["message"]["content"] or ""

    response = _client_for_provider(model_spec["provider_id"]).chat.completions.create(
        model=model_spec["api_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.GENERATION_MAX_TOKENS,
    )
    if not response.choices:
        # Seen from OpenRouter under upstream provider strain: a 200
        # response with no `choices` at all, rather than a normal error the
        # SDK would raise. Fails clearly instead of a bare "'NoneType'
        # object is not subscriptable".
        raise RuntimeError(
            f"{model_spec['api_model']} returned no completion choices."
        )
    return response.choices[0].message.content or ""


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str | None = None,
    trace: Any | None = None,
) -> GenerationResult:
    model_spec = _model_spec(model)

    context = _build_context_block(retrieved_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    if trace:
        with trace.step(
            "call_llm",
            provider=model_spec["provider_id"],
            model=model_spec["api_model"],
            max_tokens=config.GENERATION_MAX_TOKENS,
            evidence_chunks=len(retrieved_chunks),
        ):
            answer = _generate_text(model_spec, user_prompt)
    else:
        answer = _generate_text(model_spec, user_prompt)

    def collect_citations() -> list[dict]:
        """Map [n] markers cited in the answer back to their source chunks."""
        cited_markers = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        return [
            {
                "marker": i,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "document_type": chunk.document_type,
                "document_category": chunk.document_category,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "heading": chunk.heading,
                "content_type": chunk.content_type,
                "positions": chunk.positions,
                "tags": chunk.tags,
            }
            for i, chunk in enumerate(retrieved_chunks, start=1)
            if i in cited_markers
        ]

    if trace:
        with trace.step("extract_citations", answer_chars=len(answer)) as step:
            citations = collect_citations()
            step["citations"] = len(citations)
    else:
        citations = collect_citations()

    return GenerationResult(answer=answer, citations=citations)
