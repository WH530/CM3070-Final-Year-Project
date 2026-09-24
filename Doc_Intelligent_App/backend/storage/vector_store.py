"""Embedded vector store wrapping NanoVectorDB.

Persists BGE-M3 chunk embeddings alongside their provenance metadata (source
document id, title, page range, section heading) in a single JSON file on
disk, with no separately managed database service — retrieval returns both
the embedding match and the citation metadata in one operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from nano_vectordb import NanoVectorDB
from nano_vectordb.dbs import load_storage

EMBEDDING_DIM = 1024  # BAAI/bge-m3 dense vector size


class VectorStore:
    def __init__(self, storage_path: str, embedding_dim: int = EMBEDDING_DIM):
        self._db = NanoVectorDB(embedding_dim, storage_file=storage_path)

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """Each chunk dict must contain '__id__', '__vector__', and metadata fields."""
        self._db.upsert(datas=chunks)
        self._db.save()

    def query(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        results = self._db.query(query=query_vector, top_k=top_k)
        return list(results)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._db.delete(ids)
        self._db.save()

    def delete_by_document(self, document_id: str) -> None:
        """Delete every vector for a document by scanning for its document_id,
        instead of trusting a possibly-stale external id list."""
        storage = load_storage(self._db.storage_file)
        if not storage:
            return
        ids = [
            record["__id__"]
            for record in storage["data"]
            if record.get("document_id") == document_id
        ]
        self.delete(ids)
