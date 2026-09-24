"""Structured query logging — records the question, retrieved candidates,
rerank scores, final selected context, and answer for every query, so
retrieval failures can be identified and investigated during evaluation
(Preliminary Report §3.4).
"""

from __future__ import annotations

import json
import time

import config


def log_query(
    question: str,
    retrieved_chunks: list,
    answer: str,
    latency_seconds: float,
    model: str,
) -> None:
    record = {
        "timestamp": time.time(),
        "question": question,
        "model": model,
        "latency_seconds": round(latency_seconds, 3),
        "retrieved": [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "vector_score": chunk.vector_score,
                "rerank_score": chunk.rerank_score,
            }
            for chunk in retrieved_chunks
        ],
        "answer": answer,
    }
    with open(config.QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
