"""One-time setup: ingest the test PDF(s) the golden dataset (tests.yaml)
is written against, using the app's real ingestion pipeline (same code path
as an upload through the UI). Safe to re-run — already-ingested titles are
skipped.

Only Preliminary_Report.pdf is used as source material — the road-marks and
HDB schedule PDFs were company documents and are excluded from this FYP's
corpus for confidentiality (they're no longer tracked in
sample_documents/ at all).

    python scripts/ingest_test_docs.py   (run from eval/)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# eval/scripts/ingest_test_docs.py -> parent (scripts/) -> parent (eval/) -> parent (Doc_Intelligent_App/)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ingestion.pipeline import ingest_document, list_documents  # noqa: E402

TEST_DOCS_DIR = PROJECT_DIR / "sample_documents"
TEST_DOC_FILENAMES = [
    "Preliminary_Report.pdf",
]


def main() -> None:
    already_ingested = {doc["filename"] for doc in list_documents()}

    for filename in TEST_DOC_FILENAMES:
        if filename in already_ingested:
            print(f"skip (already ingested): {filename}")
            continue

        path = TEST_DOCS_DIR / filename
        if not path.is_file():
            print(f"MISSING, cannot ingest: {path}")
            continue

        print(f"ingesting: {filename} ...")
        # ingestion.pipeline.ingest_document -> storage.file_store.FileStore.save()
        # *moves* (not copies) its source_path — correct for a real upload's
        # throwaway staging file, but it would otherwise relocate our
        # git-tracked source PDF straight out of sample_documents/. Feed it
        # a disposable copy instead.
        with tempfile.TemporaryDirectory() as tmp_dir:
            staged_path = Path(tmp_dir) / filename
            shutil.copy(path, staged_path)
            record = ingest_document(str(staged_path), filename)
        status = record.get("status")
        if status == "ready":
            print(
                f"  ready — document_id={record['document_id']} "
                f"chunks={record.get('chunk_count')}"
            )
        else:
            print(f"  FAILED — {record.get('error')}")


if __name__ == "__main__":
    main()
