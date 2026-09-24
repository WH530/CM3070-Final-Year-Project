"""JsonKVStorage — an embedded, file-backed key-value metadata store.

Holds source text, chunk metadata (document id, title, page range, section
heading, content type), per-document ingestion status, and structured query
logs. Runs in-process with no separately managed database service, matching
the report's storage-layer design (Chapter 3.1/3.5).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def _nest_for_disk(data: dict[str, Any]) -> dict[str, Any]:
    """Group flat ``"namespace:id"`` keys into ``{"namespace": {"id": value}}``.

    Every document then lives under one top-level "document" block and every
    chunk under one "chunk" block, instead of the type prefix being repeated
    inside every single key — easier to scan by eye than a flat key list.
    """
    nested: dict[str, Any] = {}
    for key, value in data.items():
        namespace, sep, rest = key.partition(":")
        if sep:
            nested.setdefault(namespace, {})[rest] = value
        else:
            nested[key] = value
    return nested


def _flatten_from_disk(nested: dict[str, Any]) -> dict[str, Any]:
    """Invert :func:`_nest_for_disk` back into flat ``"namespace:id"`` keys.

    A group is recognized by every value under it being itself a record
    (dict) — that's what distinguishes a real ``{"document": {id: record}}``
    group from an old flat file where the top-level key already was
    ``"document:<id>"`` (whose fields are a mix of strings/ints/lists).
    """
    flat: dict[str, Any] = {}
    for namespace, entries in nested.items():
        if isinstance(entries, dict) and entries and all(isinstance(v, dict) for v in entries.values()):
            for rest, value in entries.items():
                flat[f"{namespace}:{rest}"] = value
        else:
            flat[namespace] = entries
    return flat


def _lines_to_text(record: Any) -> Any:
    """Join a record's on-disk ``text`` line-array back into one string."""
    if isinstance(record, dict) and isinstance(record.get("text"), list):
        record = {**record, "text": "\n".join(record["text"])}
    return record


def _text_to_lines(record: Any) -> Any:
    """Split a record's ``text`` string into a line-array for on-disk display.

    Multiline chunk text (e.g. a rendered markdown table) is unreadable as a
    single escaped JSON string. Storing it as one array entry per line makes
    metadata.json render each line on its own row, matching the source
    document's layout, without changing the in-memory representation (still
    a plain string) that the rest of the pipeline reads.
    """
    if isinstance(record, dict) and isinstance(record.get("text"), str):
        record = {**record, "text": record["text"].split("\n")}
    return record


class JsonKVStorage:
    """A tiny durable dict: every mutation is flushed to disk atomically."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            flat = _flatten_from_disk(raw)
            self._data = {k: _lines_to_text(v) for k, v in flat.items()}

    def _flush(self) -> None:
        with_lines = {k: _text_to_lines(v) for k, v in self._data.items()}
        on_disk = _nest_for_disk(with_lines)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(on_disk, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def update(self, key: str, patch: dict[str, Any]) -> None:
        with self._lock:
            current = self._data.get(key, {})
            current.update(patch)
            self._data[key] = current
            self._flush()

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._flush()

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    def find_by_prefix(self, prefix: str) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if k.startswith(prefix)}
