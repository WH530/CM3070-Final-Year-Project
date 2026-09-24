"""BAAI/bge-m3 dense embedding model — projects chunks and queries into a
shared 1024-dimensional semantic vector space for first-stage retrieval.
Runs locally; no external API dependency (Preliminary Report §2.2).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
    return _model


def check_load() -> tuple[bool, str]:
    """Verify the embedding model loads and runs, for the startup summary."""
    try:
        _get_model().encode(["ping"], convert_to_numpy=True, show_progress_bar=False)
        return True, f"{config.EMBEDDING_MODEL} on {config.DEVICE}"
    except Exception as exc:  # noqa: BLE001 - startup diagnostic, any failure reason is useful
        return False, str(exc)


def embed_texts(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
