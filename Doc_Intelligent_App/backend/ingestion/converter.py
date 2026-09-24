"""Stage 1 of the ingestion pipeline: layout-aware document conversion.

Uses Docling to convert PDF / DOCX / PPTX / XLSX / images / HTML / CSV / MD into a structured
DoclingDocument, preserving page boundaries, section headings, table
structure, and reading order. Files that cannot be converted are rejected
with an actionable error rather than silently indexed with degraded content
(Preliminary Report §3.3 — explicit rejection policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import magic
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

import config

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".md",
    ".html",
    ".htm",
    ".csv",
}

# Content sniffing (via libmagic) catches files whose bytes don't match their
# claimed extension — e.g. a renamed .exe uploaded as "invoice.pdf". OOXML
# formats (docx/pptx/xlsx) are zip containers, so libmagic sometimes reports
# the generic zip type instead of the specific office mimetype.
_OOXML_MIMES = {
    "application/zip",
    "application/x-zip",
    "application/octet-stream",
}
MIME_ALLOWLIST: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        *_OOXML_MIMES,
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        *_OOXML_MIMES,
    },
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        *_OOXML_MIMES,
    },
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".tiff": {"image/tiff"},
    ".md": {"text/plain", "text/markdown", "text/x-markdown"},
    ".html": {"text/html"},
    ".htm": {"text/html"},
    # CSV is plain text at the byte level — libmagic can't distinguish it
    # from any other text file, same limitation as .md above.
    ".csv": {"text/plain", "text/csv"},
}


def detect_mime(header: bytes) -> str:
    """Sniff the MIME type from a file's leading bytes via libmagic."""
    return magic.from_buffer(header, mime=True)


def is_supported_content(suffix: str, header: bytes) -> tuple[bool, str]:
    """Check that sniffed file content matches the claimed extension.

    Returns (is_valid, detected_mime).
    """
    allowed = MIME_ALLOWLIST.get(suffix.lower())
    if not allowed:
        return False, "unknown"
    detected = detect_mime(header)
    return detected in allowed, detected


class ConversionError(Exception):
    """Raised when a document cannot be converted to a usable structured form."""


@dataclass
class DocItem:
    text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    is_table: bool = False
    bbox: tuple[float, float, float, float] | None = None  # (left, top, right, bottom)
    # Full page size in points, alongside bbox, so a highlight box can later
    # be scaled onto the page image regardless of its rendered resolution.
    page_width: float | None = None
    page_height: float | None = None


@dataclass
class ConvertedDocument:
    title: str
    document_type: str
    items: list[DocItem] = field(default_factory=list)
    # Docling's full-document markdown export, independent of the per-item
    # `items` list above — used for whole-document LLM summarization
    # (ingestion/enrichment.py), where reading order across pages/sections
    # matters more than the page/heading/bbox metadata `items` carries.
    markdown: str = ""
    # Raw DoclingDocument, needed to re-export with figures extracted to disk
    # (ingestion/pipeline.py's save_docling_markdown step).
    docling_document: DoclingDocument | None = None


_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        # Renders each figure to a PNG for later extraction, and each full
        # page to a PNG for the page-preview/highlight feature. DOCX/PPTX
        # embed their picture bytes directly, no option needed there.
        pdf_options = PdfPipelineOptions(
            generate_picture_images=True,
            generate_page_images=True,
            images_scale=config.DOCLING_IMAGE_SCALE,
            accelerator_options=AcceleratorOptions(device=config.DEVICE),
        )
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pdf_options),
            }
        )
    return _converter


def check_load() -> tuple[bool, str]:
    """Verify Docling's document converter initializes, for the startup summary."""
    try:
        _get_converter()
        return True, f"DocumentConverter ready on {config.DEVICE}"
    except Exception as exc:  # noqa: BLE001 - startup diagnostic, any failure reason is useful
        return False, str(exc)


def _page_and_bbox(
    item: object, doc: DoclingDocument
) -> tuple[int | None, tuple[float, float, float, float] | None, float | None, float | None]:
    """Page number + top-left-origin bbox for any item with `.prov` (text,
    table, or picture) — shared so page-preview highlighting works the same
    way regardless of what kind of item is being cited."""
    page_no = None
    bbox = None
    page_width = None
    page_height = None
    prov = getattr(item, "prov", None)
    if prov:
        page_no = getattr(prov[0], "page_no", None)
        raw_bbox = getattr(prov[0], "bbox", None)
        page = doc.pages.get(page_no) if page_no is not None else None
        if raw_bbox is not None and page is not None and page.size is not None:
            page_width, page_height = page.size.width, page.size.height
            # Docling's native bbox is bottom-left origin (PDF convention);
            # converted to top-left here so it matches image pixel
            # coordinates for the page-preview highlight box.
            tl = raw_bbox.to_top_left_origin(page_height=page_height)
            bbox = (tl.l, tl.t, tl.r, tl.b)
    return page_no, bbox, page_width, page_height


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = bbox
    return max(0.0, right - left) * max(0.0, bottom - top)


# Diagram box-titles often sit just outside Docling's picture crop, not
# fully inside it — pad the bbox before the overlap check to catch them.
_PICTURE_LABEL_MARGIN = 30.0


def _mostly_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    """True if most of `inner`'s area overlaps `outer` (padded by `margin`
    on every side) — used to detect text that Docling's layout model placed
    inside or immediately around a picture region (e.g. a chart's own
    axis/legend/box-title labels), which it sometimes mislabels as a real
    heading or stray paragraph."""
    inner_area = _bbox_area(inner)
    if inner_area <= 0:
        return False
    outer = (outer[0] - margin, outer[1] - margin, outer[2] + margin, outer[3] + margin)
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    return (overlap / inner_area) > 0.5


def convert_document(file_path: str, title: str | None = None) -> ConvertedDocument:
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ConversionError(
            f"Unsupported file type '{path.suffix}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    try:
        result = _get_converter().convert(str(path))
    except Exception as exc:  # noqa: BLE001 - conversion failures are heterogeneous
        raise ConversionError(
            f"Docling could not convert '{path.name}': {exc}. The file may be "
            "corrupted, password-protected, or in an unsupported layout."
        ) from exc

    doc = result.document
    items: list[DocItem] = []
    current_heading: str | None = None

    # Docling's layout model occasionally mislabels text sitting inside a
    # picture (e.g. a chart's own axis/legend/task labels) as a real heading
    # or paragraph. That content is already captured properly by the
    # picture's own VLM description (ingestion/enrichment.py), so any text
    # item mostly contained within a picture's bbox is skipped here entirely
    # — both to avoid duplicate/garbled chunks and to stop it from poisoning
    # heading tracking below.
    picture_bboxes_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for picture in doc.pictures:
        pic_page, pic_bbox, _, _ = _page_and_bbox(picture, doc)
        if pic_page is not None and pic_bbox is not None:
            picture_bboxes_by_page.setdefault(pic_page, []).append(pic_bbox)

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", "")
        text = getattr(item, "text", "") or ""
        is_table = label == "table" or type(item).__name__ == "TableItem"

        # Running headers/footers repeat on every page and carry no content
        # of their own — left in, they get merged into whatever chunk
        # follows them, dragging that chunk's citation highlight up into
        # the page's header margin.
        if label in ("page_header", "page_footer"):
            continue

        if is_table:
            try:
                text = item.export_to_markdown(doc)
            except Exception:  # noqa: BLE001 - fall back to raw cell text
                text = text or "[unreadable table]"

        if not text.strip():
            continue

        # OCR occasionally hallucinates a stray symbol/bullet (e.g. an
        # arrow glyph) over blank page space with no real text underneath.
        # It carries no retrievable content either way, so drop it rather
        # than let it inject a meaningless highlight box.
        if not is_table and not any(ch.isalnum() for ch in text):
            continue

        page_no, bbox, page_width, page_height = _page_and_bbox(item, doc)

        if bbox is not None and any(
            _mostly_inside(bbox, pic_bbox, margin=_PICTURE_LABEL_MARGIN)
            for pic_bbox in picture_bboxes_by_page.get(page_no, [])
        ):
            continue

        if label in ("section_header", "title"):
            current_heading = text.strip()

        items.append(
            DocItem(
                text=text.strip(),
                page_start=page_no,
                page_end=page_no,
                heading=current_heading,
                is_table=is_table,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
            )
        )

    if not items:
        raise ConversionError(
            f"'{path.name}' converted with no extractable text. It may be a "
            "scanned image with no OCR text layer, or an empty document."
        )

    try:
        markdown = doc.export_to_markdown()
    except Exception:  # noqa: BLE001 - summarization is best-effort, never blocks conversion
        markdown = ""

    return ConvertedDocument(
        title=title or path.stem,
        document_type=path.suffix.lower().lstrip("."),
        items=items,
        markdown=markdown,
        docling_document=doc,
    )


def extract_images(doc: DoclingDocument) -> list[dict]:
    """One entry per picture in document order — the image files themselves
    are saved separately (ingestion/pipeline.py's save_images step). Includes
    bbox/page size so an image's description can also become a retrievable
    chunk with a working click-to-preview highlight (ingestion/chunker.py's
    build_image_chunks).
    """
    results = []
    for index, picture in enumerate(doc.pictures):
        page_no, bbox, page_width, page_height = _page_and_bbox(picture, doc)
        results.append(
            {
                "index": index,
                "page": page_no,
                "bbox": bbox,
                "page_width": page_width,
                "page_height": page_height,
            }
        )
    return results
