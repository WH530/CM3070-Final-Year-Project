"""BAAI/bge-reranker-v2-m3 cross-encoder — receives the query and each
candidate chunk as a single concatenated input and returns a scalar
relevance score, reordering the recall-oriented first-stage candidate set
by fine-grained relevance before generation (Preliminary Report §2.3).
"""

from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

import config

_RERANK_BATCH_SIZE = 8

_model: CrossEncoder | None = None
_cpu_fallback_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL, device=config.DEVICE)
    return _model


def _get_cpu_fallback_model() -> CrossEncoder:
    """Lazily loaded CPU copy, used only when the MPS/CUDA device is out of memory."""
    global _cpu_fallback_model
    if _cpu_fallback_model is None:
        _cpu_fallback_model = CrossEncoder(config.RERANKER_MODEL, device="cpu")
    return _cpu_fallback_model


def _is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def check_load() -> tuple[bool, str]:
    """Verify the reranker model loads and runs, for the startup summary."""
    try:
        _get_model().predict(
            [("ping", "pong")], convert_to_numpy=True, show_progress_bar=False
        )
        return True, f"{config.RERANKER_MODEL} on {config.DEVICE}"
    except Exception as exc:  # noqa: BLE001 - startup diagnostic, any failure reason is useful
        return False, str(exc)


def rerank(query: str, chunk_texts: list[str]) -> list[float]:
    """Return one relevance score per chunk, in the same order as the input.

    Falls back to CPU if the accelerator (MPS/CUDA) runs out of memory, rather
    than failing the whole query.
    """
    if not chunk_texts:
        return []
    pairs = [(query, chunk_text) for chunk_text in chunk_texts]
    try:
        scores = _get_model().predict(
            pairs,
            batch_size=_RERANK_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except RuntimeError as exc:
        if config.DEVICE == "cpu" or not _is_oom_error(exc):
            raise
        if config.DEVICE == "mps":
            torch.mps.empty_cache()
        elif config.DEVICE == "cuda":
            torch.cuda.empty_cache()
        scores = _get_cpu_fallback_model().predict(
            pairs,
            batch_size=_RERANK_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return [float(s) for s in scores]
