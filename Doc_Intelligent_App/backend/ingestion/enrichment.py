"""Stage 3 of the ingestion pipeline: optional LLM-based metadata enrichment.

Runs after chunking and before embedding. Adds a document-level category
label plus per-chunk keywords, hypothetical questions, and tags, so
retrieval can later match queries worded very differently from the source
text (this is what RAGFlow calls its Transformer stage).

Always calls the local Ollama model directly, independent of whichever
model the user has selected for chat generation in generation/llm.py — this
is an internal indexing step, not a user-facing response, so it stays free,
offline, and decoupled (importing generation.llm here would also create an
import cycle: generation.llm -> retrieval.pipeline -> ingestion.pipeline).

Enrichment is best-effort: any failure (model unavailable, malformed JSON,
timeout) leaves the affected fields empty rather than raising, since a
chunk with no keywords/questions/category is still fully usable for
retrieval — enrichment must never block ingestion.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

import config
from ingestion.chunker import Chunk

with open(
    Path(__file__).resolve().parent.parent / "prompt.yaml", encoding="utf-8"
) as _f:
    _PROMPTS = yaml.safe_load(_f)["enrichment"]


def _call_json(system_prompt: str, user_prompt: str) -> dict:
    # Native /api/chat, not the OpenAI-compatible route, since only this
    # endpoint honors think:False (~15s/call with thinking vs ~1s without).
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        f"{config.OLLAMA_API_BASE}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    return json.loads(data["message"]["content"] or "{}")


def classify_document_category(title: str, sample_text: str) -> str | None:
    """One call per document. Returns a value from config.DOCUMENT_CATEGORIES,
    or None if enrichment is disabled or the call fails."""

    if not config.ENRICHMENT_ENABLED:
        return None
    try:
        result = _call_json(
            _PROMPTS["category_system_prompt"],
            _PROMPTS["category_user_template"].format(
                categories=", ".join(config.DOCUMENT_CATEGORIES),
                title=title,
                sample_text=sample_text[:2000],
            ),
        )
        category = str(result.get("category", "")).strip().lower()
        return category if category in config.DOCUMENT_CATEGORIES else "other"
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never blocks ingestion
        return None


def check_load() -> tuple[bool, str]:
    """Verify the local model can actually process an image (do_picture_description
    support), for the startup summary. Informational only — never gates
    _core_ready(), since picture description is best-effort and must never
    block ingestion or the rest of the app."""
    if not config.ENRICHMENT_ENABLED:
        return True, "enrichment disabled"
    try:
        tiny = Image.new("RGB", (8, 8), "white")
        description = describe_picture(tiny)
        return (True, f"{config.OLLAMA_MODEL} (vision)") if description else (False, "no response")
    except Exception as exc:  # noqa: BLE001 - startup diagnostic, any failure reason is useful
        return False, str(exc)


def summarize_document(title: str, markdown: str) -> str | None:
    """One call per document, over Docling's full markdown export. Returns a
    short prose summary for direct lookup (agent/graph.py's summarize_document
    action), so the agent doesn't need to search chunks/vectors just to
    describe what a document is about. None if enrichment is disabled or the
    call fails."""

    if not config.ENRICHMENT_ENABLED:
        return None
    try:
        result = _call_json(
            _PROMPTS["summary_system_prompt"],
            _PROMPTS["summary_user_template"].format(
                title=title,
                # config.OLLAMA_CONTEXT_LENGTH (8192) is shared by prompt +
                # reasoning + completion, so the source excerpt must leave
                # headroom rather than fill the window on its own.
                markdown=markdown[:6000],
            ),
        )
        summary = str(result.get("summary", "")).strip()
        return summary or None
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never blocks ingestion
        return None


def describe_picture(image: Image.Image) -> str | None:
    """One call per figure. Returns a short natural-language description of
    what the figure actually shows (the locally installed model has vision
    capability — confirmed on real report figures, including reading labels
    inside charts/diagrams), or None if enrichment is disabled or the call
    fails."""

    if not config.ENRICHMENT_ENABLED:
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _PROMPTS["picture_description_system_prompt"]},
                {
                    "role": "user",
                    "content": _PROMPTS["picture_description_user_prompt"],
                    "images": [image_b64],
                },
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            f"{config.OLLAMA_API_BASE}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        description = str(data["message"]["content"]).strip()
        return description or None
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never blocks ingestion
        return None


def _enrich_batch(batch: list[Chunk]) -> dict:
    """One call per batch instead of one per chunk. Mutates each chunk's
    important_keywords/hypothetical_questions/tags in place; a failed or
    malformed response just leaves them empty, same as the old per-chunk
    behavior. Returns stats for the caller to log (enrichment.py stays
    independent of pipeline_trace)."""
    started = time.perf_counter()
    excerpts = "\n\n".join(f"[{i}] {c.text[:2000]}" for i, c in enumerate(batch))
    by_index: dict[int, dict] = {}
    try:
        result = _call_json(
            _PROMPTS["chunk_batch_system_prompt"],
            _PROMPTS["chunk_batch_user_template"].format(excerpts=excerpts),
        )
        by_index = {
            int(item["index"]): item
            for item in result.get("results", [])
            if isinstance(item, dict) and "index" in item
        }
    except Exception:  # noqa: BLE001 - enrichment is best-effort, never blocks ingestion
        pass

    enriched = 0
    for i, chunk in enumerate(batch):
        item = by_index.get(i, {})
        chunk.important_keywords = [str(k) for k in item.get("keywords", [])][:5]
        chunk.hypothetical_questions = [str(q) for q in item.get("questions", [])][:2]
        chunk.tags = [str(t) for t in item.get("tags", [])][:3]
        enriched += bool(item)

    return {
        "chunk_ids": [c.chunk_id for c in batch],
        "duration": time.perf_counter() - started,
        "enriched": enriched,
    }


def enrich_chunks(chunks: list[Chunk]) -> list[dict]:
    """Batched + parallel replacement for one-call-per-chunk enrichment:
    groups chunks into config.ENRICHMENT_BATCH_SIZE-sized batches and runs
    up to config.OLLAMA_NUM_PARALLEL of them concurrently. Returns one stats
    dict per batch for the caller to log."""
    if not config.ENRICHMENT_ENABLED or not chunks:
        return []

    batches = [
        chunks[i : i + config.ENRICHMENT_BATCH_SIZE]
        for i in range(0, len(chunks), config.ENRICHMENT_BATCH_SIZE)
    ]
    with ThreadPoolExecutor(max_workers=config.OLLAMA_NUM_PARALLEL) as pool:
        return list(pool.map(_enrich_batch, batches))
