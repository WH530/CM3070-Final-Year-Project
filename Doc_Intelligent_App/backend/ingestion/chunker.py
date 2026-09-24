"""Stage 2 of the ingestion pipeline: structure-aware chunking.

Splits converted document items into retrievable chunks at heading
boundaries wherever possible, so each chunk stays interpretable in
isolation. A configurable character ceiling prevents oversized chunks from
diluting semantic focus, tables are kept whole rather than split mid-row,
and a small overlap preserves continuity across boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.converter import ConvertedDocument, DocItem


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    document_type: str
    text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    content_type: str  # "prose" | "table" | "image"
    # [page_no, left, top, right, bottom, page_width, page_height] per source
    # region — one entry per merged item, so a chunk spanning multiple
    # layout regions keeps them all.
    positions: list[list[float]] = field(default_factory=list)
    # Populated by ingestion.enrichment after chunking; absent (empty/None)
    # until that best-effort stage runs, since chunking itself stays a pure,
    # deterministic step with no LLM dependency.
    document_category: str | None = None
    important_keywords: list[str] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _flush(
    buffer: list[DocItem],
    document_id: str,
    title: str,
    document_type: str,
    chunk_index: int,
) -> Chunk | None:
    if not buffer:
        return None
    text = "\n\n".join(item.text for item in buffer)
    pages = [
        p for item in buffer for p in (item.page_start, item.page_end) if p is not None
    ]
    positions = [
        [item.page_start, *item.bbox, item.page_width, item.page_height]
        for item in buffer
        if item.page_start is not None and item.bbox is not None
    ]
    return Chunk(
        chunk_id=f"{document_id}::chunk-{chunk_index}",
        document_id=document_id,
        title=title,
        document_type=document_type,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        # The buffer's *last* item, not its first: after a heading-triggered
        # flush, buffer[0] is a synthetic overlap item carrying the PREVIOUS
        # section's heading (for text continuity only) — using it here would
        # mislabel this chunk with the old heading even once new-section
        # content dominates it.
        heading=buffer[-1].heading,
        content_type="table" if any(i.is_table for i in buffer) else "prose",
        positions=positions,
    )


def chunk_document(
    doc: ConvertedDocument,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[DocItem] = []
    buffer_len = 0
    chunk_index = 0

    for item in doc.items:
        # Tables are never split mid-structure — flush and emit as their own chunk.
        if item.is_table:
            flushed = _flush(
                buffer, document_id, doc.title, doc.document_type, chunk_index
            )
            if flushed:
                chunks.append(flushed)
                chunk_index += 1
            table_chunk = _flush(
                [item], document_id, doc.title, doc.document_type, chunk_index
            )
            chunks.append(table_chunk)
            chunk_index += 1
            buffer, buffer_len = [], 0
            continue

        # Compare against the buffer's last item, not its first, for the same
        # reason as _flush()'s heading assignment below — buffer[0] can be a
        # stale overlap item from the previous section.
        starts_new_heading = buffer and item.heading != buffer[-1].heading
        exceeds_ceiling = buffer_len + len(item.text) > chunk_size

        if buffer and (starts_new_heading or exceeds_ceiling):
            flushed = _flush(
                buffer, document_id, doc.title, doc.document_type, chunk_index
            )
            chunks.append(flushed)
            chunk_index += 1
            # carry a small text overlap from the tail of the previous chunk
            overlap_text = flushed.text[-chunk_overlap:] if chunk_overlap else ""
            buffer = (
                [
                    DocItem(
                        text=overlap_text,
                        page_start=flushed.page_end,
                        page_end=flushed.page_end,
                        heading=flushed.heading,
                    )
                ]
                if overlap_text
                else []
            )
            buffer_len = len(overlap_text)

        buffer.append(item)
        buffer_len += len(item.text)

    final = _flush(buffer, document_id, doc.title, doc.document_type, chunk_index)
    if final:
        chunks.append(final)

    return chunks


def build_image_chunks(
    images_manifest: list[dict],
    document_id: str,
    title: str,
    document_type: str,
) -> list[Chunk]:
    """One retrievable chunk per described image (ingestion/pipeline.py's
    save_images step), so a VLM description becomes searchable and citable
    exactly like a text/table chunk — same embedding, vector index, and
    click-to-preview highlight machinery, just content_type="image"."""
    chunks = []
    for entry in images_manifest:
        description = entry.get("description")
        if not description:
            continue
        page = entry.get("page")
        bbox = entry.get("bbox")
        positions = (
            [[page, *bbox, entry.get("page_width"), entry.get("page_height")]]
            if page is not None and bbox is not None
            else []
        )
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}::image-{entry['index']}",
                document_id=document_id,
                title=title,
                document_type=document_type,
                text=description,
                page_start=page,
                page_end=page,
                heading=None,
                content_type="image",
                positions=positions,
            )
        )
    return chunks
