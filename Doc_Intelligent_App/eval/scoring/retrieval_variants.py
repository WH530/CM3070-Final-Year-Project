"""Vector-only retrieval variant, for the reranker ablation provider.

Mirrors retrieval.pipeline.retrieve()'s embed -> vector-search -> dedupe
steps exactly, but skips the rerank() call and truncates by vector_score
instead — isolating the ablation to the eval harness rather than adding an
on/off flag to the production retrieval pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# eval/scoring/retrieval_variants.py -> parent (scoring/) -> parent (eval/) -> parent (Doc_Intelligent_App/)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402
from ingestion.pipeline import get_kv_store, get_vector_store  # noqa: E402
from retrieval.embedder import embed_query  # noqa: E402
from retrieval.pipeline import RetrievedChunk  # noqa: E402


def retrieve_vector_only(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or config.TOP_K
    kv = get_kv_store()
    vector_store = get_vector_store()

    query_vec = embed_query(question)
    candidates = vector_store.query(query_vec, top_k=config.CANDIDATE_DEPTH)

    seen: set[str] = set()
    deduped = []
    for c in candidates:
        cid = c["__id__"]
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)

    chunks: list[RetrievedChunk] = []
    for candidate in deduped:
        chunk_meta = kv.get(f"chunk:{candidate['__id__']}", {})
        chunks.append(
            RetrievedChunk(
                chunk_id=candidate["__id__"],
                document_id=candidate.get("document_id", ""),
                title=candidate.get("title", ""),
                document_type=candidate.get("document_type", ""),
                document_category=candidate.get("document_category"),
                text=chunk_meta.get("text", ""),
                page_start=chunk_meta.get("page_start"),
                page_end=chunk_meta.get("page_end"),
                heading=chunk_meta.get("heading"),
                content_type=chunk_meta.get("content_type", "prose"),
                positions=chunk_meta.get("positions", []),
                tags=candidate.get("tags", []),
                vector_score=float(candidate.get("__metrics__", 0.0)),
                rerank_score=float("nan"),  # never reranked — not a meaningful score here
            )
        )

    # No rerank() call: rank purely by first-stage vector_score, same as the
    # candidate order nano-vectordb already returned them in.
    return chunks[:top_k]
