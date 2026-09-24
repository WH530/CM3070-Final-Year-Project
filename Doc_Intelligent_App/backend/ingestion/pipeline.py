"""Ingestion chain: convert -> chunk -> embed -> index.

Each uploaded document proceeds through this fixed sequence, with the
outcome (success/failure, chunk count, model versions, timestamps) recorded
in the metadata store so processing status is auditable (Preliminary Report
§3.3, Figure 2).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path

from docling_core.types.doc import ImageRefMode

import config
from ingestion import enrichment
from ingestion.chunker import build_image_chunks, chunk_document
from ingestion.converter import ConversionError, convert_document, extract_images
from pipeline_trace import PipelineTrace
from retrieval.embedder import embed_texts
from storage.file_store import FileStore
from storage.kv_store import JsonKVStorage
from storage.vector_store import VectorStore

_vector_store: VectorStore | None = None
_kv_store: JsonKVStorage | None = None
_file_store: FileStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(config.VECTOR_DB_PATH)
    return _vector_store


def get_kv_store() -> JsonKVStorage:
    global _kv_store
    if _kv_store is None:
        _kv_store = JsonKVStorage(config.METADATA_DB_PATH)
    return _kv_store


def get_file_store() -> FileStore:
    global _file_store
    if _file_store is None:
        _file_store = FileStore(config.RAW_FILES_DIR)
    return _file_store


def check_stores() -> tuple[bool, str]:
    """Verify the vector, metadata, and file stores open cleanly, for the startup summary."""
    try:
        get_vector_store()
        get_kv_store()
        get_file_store()
        return (
            True,
            f"{Path(config.VECTOR_DB_PATH).name}, {Path(config.METADATA_DB_PATH).name}, "
            f"{Path(config.RAW_FILES_DIR).name}/",
        )
    except Exception as exc:  # noqa: BLE001 - startup diagnostic, any failure reason is useful
        return False, str(exc)


def ingest_document(file_path: str, original_filename: str) -> dict:
    document_id = str(uuid.uuid4())
    trace = PipelineTrace(
        "document",
        document_id[:8],
        document_id=document_id,
        filename=original_filename,
    )
    kv = get_kv_store()
    started = time.time()

    trace.start()

    with trace.step("save_raw_file"):
        raw_file_path = get_file_store().save(document_id, file_path)

    with trace.step("create_processing_record"):
        kv.set(
            f"document:{document_id}",
            {
                "document_id": document_id,
                "filename": original_filename,
                "status": "processing",
                "error": None,
                "chunk_count": 0,
                "chunk_ids": [],
                "summary_document": "",
                "raw_file_path": raw_file_path,
                # Deterministic from document_id, set upfront (not only on
                # success) so delete_document() can always clean it up, even
                # for a document that fails or crashes mid-ingestion.
                "docling_dir": str(config.DOCLING_DIR / document_id),
                "uploaded_at": started,
            },
        )

    try:
        with trace.step("convert_document", raw_file=Path(raw_file_path).name):
            converted = convert_document(
                raw_file_path, title=Path(original_filename).stem
            )
            trace.info(
                "converted",
                icon="📄",
                title=getattr(converted, "title", None),
                document_type=getattr(converted, "document_type", None),
            )

        # Traceability extras only (markdown export, figure/page PNGs) — a
        # failure here must never fail an otherwise-convertible document, so
        # each step catches its own errors instead of letting them reach the
        # outer except block below.
        docling_dir = config.DOCLING_DIR / document_id

        with trace.step("save_page_images") as step:
            page_count = 0
            try:
                pages_dir = docling_dir / "pages"
                for page_no, page in converted.docling_document.pages.items():
                    if page.image is None:
                        continue
                    pages_dir.mkdir(parents=True, exist_ok=True)
                    page.image.pil_image.save(pages_dir / f"page_{page_no}.png")
                    page_count += 1
            except Exception as exc:  # noqa: BLE001 - traceability extra, never blocks ingestion
                trace.warning("save_page_images_failed", error=str(exc))
            step["pages"] = page_count

        # extracted_images/ is the single source of truth for real image
        # files — used by both the Images gallery (images.json) and, below,
        # document.md. Docling never writes its own separate image copy.
        manifest = []
        # One slot per picture, aligned to doc.pictures order (None where
        # extraction failed), so the markdown step below can match each
        # placeholder to the right file purely by position.
        aligned_images: list[dict | None] = []
        with trace.step("save_images") as step:
            image_count = 0
            try:
                images = extract_images(converted.docling_document)
                if images:
                    extracted_images_dir = docling_dir / "extracted_images"
                    extracted_images_dir.mkdir(parents=True, exist_ok=True)
                    for entry in images:
                        picture = converted.docling_document.pictures[entry["index"]]
                        image = picture.get_image(converted.docling_document)
                        if image is None:
                            aligned_images.append(None)
                            continue
                        filename = f"image_{entry['index']}.png"
                        image.save(extracted_images_dir / filename)
                        # Best-effort — the locally installed model has vision
                        # capability (confirmed on real report figures), but
                        # a description failure must never block ingestion.
                        vlm_started = time.perf_counter()
                        description = enrichment.describe_picture(image)
                        trace.info(
                            "image_described",
                            icon="🖼️",
                            index=entry["index"],
                            duration=trace.format_duration(time.perf_counter() - vlm_started),
                            described=description is not None,
                        )
                        record = {**entry, "image": filename, "description": description}
                        manifest.append(record)
                        aligned_images.append(record)
                        image_count += 1
                    if manifest:
                        (docling_dir / "images.json").write_text(json.dumps(manifest, indent=2))
            except Exception as exc:  # noqa: BLE001 - traceability extra, never blocks ingestion
                trace.warning("save_images_failed", error=str(exc))
            step["images"] = image_count

        with trace.step("save_docling_markdown") as step:
            embedded_count = 0
            try:
                # PLACEHOLDER writes no image files at all — just a marker at
                # each picture's position, in the same reading order as
                # doc.pictures/aligned_images, so we can substitute our own
                # already-extracted files + VLM descriptions by position.
                placeholder = "@@DOCAI_IMAGE_PLACEHOLDER@@"
                text = converted.docling_document.export_to_markdown(
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder=placeholder,
                )
                counter = [0]

                def _inject(match: re.Match) -> str:
                    nonlocal embedded_count
                    idx = counter[0]
                    counter[0] += 1
                    record = aligned_images[idx] if idx < len(aligned_images) else None
                    if record is None:
                        return ""  # extraction failed for this picture — drop the empty marker
                    embedded_count += 1
                    link = f"![Image](extracted_images/{record['image']})"
                    description = record.get("description")
                    return f"{link}\n\n{description}" if description else link

                text = re.sub(re.escape(placeholder), _inject, text)
                (docling_dir / "document.md").write_text(text)
            except Exception as exc:  # noqa: BLE001 - traceability extra, never blocks ingestion
                trace.warning("save_docling_markdown_failed", error=str(exc))
            step["images"] = embedded_count

        with trace.step(
            "chunk_document",
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        ):
            chunks = chunk_document(
                converted, document_id, config.CHUNK_SIZE, config.CHUNK_OVERLAP
            )
            trace.info("chunks_created", icon="🧩", count=len(chunks))

        if not chunks:
            raise ConversionError("No chunks were produced from this document.")

        # Each described image becomes a chunk too — same embedding/index/
        # citation pipeline as text/table chunks, just content_type="image".
        image_chunks = build_image_chunks(
            manifest, document_id, converted.title, converted.document_type
        )
        chunks.extend(image_chunks)
        if image_chunks:
            trace.info("image_chunks_created", icon="🖼️", count=len(image_chunks))

        # Best-effort metadata enrichment (category, keywords, questions, tags,
        # document summary). Never raises — a chunk/document with no
        # enrichment is still fully indexable.
        with trace.step("enrich_metadata", chunks=len(chunks)):
            started = time.perf_counter()
            document_category = enrichment.classify_document_category(
                converted.title, chunks[0].text
            )
            trace.info(
                "category_classified",
                icon="🏷️",
                category=document_category,
                duration=trace.format_duration(time.perf_counter() - started),
            )

            started = time.perf_counter()
            summary_document = enrichment.summarize_document(
                converted.title, converted.markdown
            )
            trace.info(
                "document_summarized",
                icon="📝",
                summarized=summary_document is not None,
                duration=trace.format_duration(time.perf_counter() - started),
            )

            for c in chunks:
                c.document_category = document_category

            for stats in enrichment.enrich_chunks(chunks):
                trace.info(
                    "chunk_batch_enriched",
                    icon="✨",
                    chunks=len(stats["chunk_ids"]),
                    enriched=stats["enriched"],
                    first_chunk_id=stats["chunk_ids"][0],
                    last_chunk_id=stats["chunk_ids"][-1],
                    duration=trace.format_duration(stats["duration"]),
                )
            trace.info(
                "metadata_enriched",
                icon="🏁",
                category=document_category,
                summarized=summary_document is not None,
            )

        with trace.step("embed_chunks", chunks=len(chunks), model=config.EMBEDDING_MODEL):
            vectors = embed_texts([c.text for c in chunks])
            trace.info("vectors_created", icon="🔢", count=len(vectors))

        with trace.step("upsert_vector_index", vectors=len(vectors)):
            vector_store = get_vector_store()
            vector_store.upsert(
                [
                    {
                        "__id__": c.chunk_id,
                        "__vector__": vec,
                        "document_id": c.document_id,
                        "title": c.title,
                        "document_type": c.document_type,
                        "document_category": c.document_category,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "heading": c.heading,
                        "content_type": c.content_type,
                        "tags": c.tags,
                    }
                    for c, vec in zip(chunks, vectors)
                ]
            )

        with trace.step("persist_chunk_metadata", chunks=len(chunks)):
            for c in chunks:
                kv.set(
                    f"chunk:{c.chunk_id}",
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "title": c.title,
                        "document_type": c.document_type,
                        "document_category": c.document_category,
                        "text": c.text,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "heading": c.heading,
                        "content_type": c.content_type,
                        "positions": c.positions,
                        "important_keywords": c.important_keywords,
                        "hypothetical_questions": c.hypothetical_questions,
                        "tags": c.tags,
                    },
                )

        with trace.step("mark_document_ready"):
            kv.update(
                f"document:{document_id}",
                {
                    "status": "ready",
                    "chunk_count": len(chunks),
                    "chunk_ids": [c.chunk_id for c in chunks],
                    "image_count": image_count,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "processing_seconds": round(time.time() - started, 2),
                    "summary_document": summary_document or "",
                },
            )
        record = kv.get(f"document:{document_id}")
        trace.complete(status="ready", chunks=len(chunks))
        return record

    except ConversionError as exc:
        kv.update(f"document:{document_id}", {"status": "failed", "error": str(exc)})
        trace.fail(exc, status="failed")
        return kv.get(f"document:{document_id}")
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures, don't crash the app
        kv.update(
            f"document:{document_id}",
            {
                "status": "failed",
                "error": f"Unexpected ingestion error: {exc}",
            },
        )
        trace.fail(exc, status="failed")
        return kv.get(f"document:{document_id}")


def delete_document(document_id: str) -> None:
    kv = get_kv_store()
    record = kv.get(f"document:{document_id}")
    if not record:
        return
    # Scan by document_id — record["chunk_ids"] isn't set until ingestion finishes.
    get_vector_store().delete_by_document(document_id)
    for key, chunk in kv.find_by_prefix("chunk:").items():
        if chunk.get("document_id") == document_id:
            kv.delete(key)
    get_file_store().delete(record.get("raw_file_path"))
    docling_dir = record.get("docling_dir")
    if docling_dir:
        shutil.rmtree(docling_dir, ignore_errors=True)
    kv.delete(f"document:{document_id}")


def list_documents() -> list[dict]:
    kv = get_kv_store()
    return [v for k, v in kv.all().items() if k.startswith("document:")]


def get_page_image_path(document_id: str, page_no: int) -> Path | None:
    """Return the saved page-image path for a document/page, or None if unavailable."""
    path = config.DOCLING_DIR / document_id / "pages" / f"page_{page_no}.png"
    return path if path.is_file() else None


def list_images(document_id: str) -> list[dict]:
    """Return the saved extracted-image manifest for a document, or [] if none."""
    manifest_path = config.DOCLING_DIR / document_id / "images.json"
    if not manifest_path.is_file():
        return []
    return json.loads(manifest_path.read_text())


def get_extracted_image_path(document_id: str, index: int) -> Path | None:
    """Return the saved extracted-image path for a document/index, or None if unavailable."""
    path = config.DOCLING_DIR / document_id / "extracted_images" / f"image_{index}.png"
    return path if path.is_file() else None
