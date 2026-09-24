"""Embedded raw-file store: keeps the original uploaded PDF/image/etc. on disk.

Files are moved (not copied) into a flat directory keyed by document id, so
the original source stays available for re-download or future re-processing
after the ephemeral upload path is discarded.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class FileStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, source_path: str) -> str:
        suffix = Path(source_path).suffix
        dest = self.base_dir / f"{document_id}{suffix}"
        shutil.move(source_path, dest)
        return str(dest)

    def delete(self, path: str | None) -> None:
        if not path:
            return
        Path(path).unlink(missing_ok=True)
