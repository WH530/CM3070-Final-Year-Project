"""Query chain: embed -> retrieve -> deduplicate -> rerank -> assemble context.

The first-stage vector search is deliberately recall-oriented (larger
candidate depth, lower precision); the reranker then narrows this down to
the top-k retrieved chunks that are actually sent to the LLM
(Preliminary Report §3.4).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import config
from ingestion.pipeline import get_kv_store, get_vector_store
from retrieval.embedder import embed_query
from retrieval.reranker import rerank


def _trace_step(trace: Any, name: str, **fields: Any):
    return trace.step(name, **fields) if trace else nullcontext()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    document_type: str
    document_category: str | None
    text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    content_type: str
    positions: list[list[float]]
    tags: list[str]
    vector_score: float
    rerank_score: float


def retrieve(
    question: str,
    top_k: int | None = None,
    trace: Any | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or config.TOP_K
    kv = get_kv_store()
    vector_store = get_vector_store()

    with _trace_step(trace, "embed_query", model=config.EMBEDDING_MODEL):
        query_vec = embed_query(question)

    with _trace_step(
        trace, "vector_search", candidate_depth=config.CANDIDATE_DEPTH
    ) as step:
        candidates = vector_store.query(query_vec, top_k=config.CANDIDATE_DEPTH)
        step["relevant_chunks"] = len(candidates)

    # Deduplicate by chunk id (defensive — nano-vectordb keys are already unique,
    # but this guards against near-identical re-upserts across ingestion re-runs).
    seen: set[str] = set()
    deduped = []
    for c in candidates:
        cid = c["__id__"]
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)

    if not deduped:
        if trace:
            trace.warning("no_candidates")
        return []

    with _trace_step(trace, "load_candidate_metadata", candidates=len(deduped)):
        chunk_texts = [
            kv.get(f"chunk:{candidate['__id__']}", {}).get("text", "")
            for candidate in deduped
        ]

    with _trace_step(
        trace,
        "rerank_candidates",
        candidates=len(chunk_texts),
        model=config.RERANKER_MODEL,
    ):
        rerank_scores = rerank(question, chunk_texts)

    with _trace_step(trace, "assemble_retrieved_chunks", top_k=top_k) as step:
        retrieved_chunks = []
        for candidate, chunk_text, score in zip(deduped, chunk_texts, rerank_scores):
            chunk_meta = kv.get(f"chunk:{candidate['__id__']}", {})
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=candidate["__id__"],
                    document_id=candidate.get("document_id", ""),
                    title=candidate.get("title", ""),
                    document_type=candidate.get("document_type", ""),
                    document_category=candidate.get("document_category"),
                    text=chunk_text,
                    page_start=chunk_meta.get("page_start"),
                    page_end=chunk_meta.get("page_end"),
                    heading=chunk_meta.get("heading"),
                    content_type=chunk_meta.get("content_type", "prose"),
                    positions=chunk_meta.get("positions", []),
                    tags=candidate.get("tags", []),
                    vector_score=float(candidate.get("__metrics__", 0.0)),
                    rerank_score=score,
                )
            )

        retrieved_chunks.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
        selected_chunks = retrieved_chunks[:top_k]
        step["selected_chunks"] = len(selected_chunks)
        step["best_score"] = (
            round(selected_chunks[0].rerank_score, 4) if selected_chunks else None
        )
        return selected_chunks
