"""FastAPI application layer — document upload, processing status, chat
query endpoint, and the static single-page UI. This is the thin
application layer that wires the ingestion chain, query chain, and
generation stage together (Preliminary Report §3.1, Figure 1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)
from pydantic import BaseModel

import config
from agent.graph import run_agent
from generation.llm import (
    ProviderUnavailableError,
    check_connection,
    get_model_status,
    get_provider_status,
)
from generation.logging_utils import log_query
from ingestion.converter import SUPPORTED_SUFFIXES, is_supported_content
from ingestion.converter import check_load as check_docling
from ingestion.enrichment import check_load as check_picture_description
from ingestion.pipeline import (
    check_stores,
    delete_document,
    get_extracted_image_path,
    get_page_image_path,
    ingest_document,
    list_documents,
    list_images,
)
from pipeline_trace import PipelineTrace
from retrieval.embedder import check_load as check_embedder
from retrieval.reranker import check_load as check_reranker

def _check_langgraph_runtime() -> None:
    # Catches a bad langchain/langchain-core pairing here instead of as an
    # AttributeError deep in agent.graph.run_agent() on the first query.
    try:
        from langchain_core.runnables.config import get_callback_manager_for_config

        get_callback_manager_for_config({})
    except Exception as exc:  # noqa: BLE001 - any failure here means broken runtime
        venv_hint = (
            r".venv\Scripts\python.exe"
            if os.name == "nt"
            else "Doc_Intelligent_App/.venv/bin/python or ./run.sh"
        )
        raise SystemExit(
            "Incompatible langchain/langchain-core/langgraph versions "
            f"detected ({exc!r}). Run this app with the project's own "
            f"virtualenv ({venv_hint}), not a global interpreter, and "
            "reinstall with `pip install -r requirements.txt`."
        ) from exc


_check_langgraph_runtime()

if hasattr(sys.stdout, "reconfigure"):
    # The startup checks table (below) uses box-drawing characters and emoji
    # markers; on a console whose default encoding isn't UTF-8 (e.g. Windows'
    # cp1252), printing it raises UnicodeEncodeError from inside an untracked
    # asyncio task, silently losing the whole table. Mirrors pipeline_trace.py's
    # identical stderr reconfigure, for the same reason.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# force=True: ML libraries imported below (transformers, etc.) configure the
# root logger's handlers on import, which would otherwise make this a no-op.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet its per-request INFO spam

app = FastAPI(title="DocAI")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_STARTUP_PAGE_TEMPLATE = (FRONTEND_DIR / "startup.html").read_text(encoding="utf-8")

# eval/ (RAGAS evaluation harness, see eval/README.md) is a sibling of
# backend/ and frontend/, not served by NoCacheStaticFiles below — only this
# one route reaches into it, and only to hand back the most recent exported
# report, never the harness source itself.
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

_CORE_COMPONENT_STATUS: dict[str, tuple[bool, str]] = {
    "docling": (False, "Document converter has not been checked yet."),
    "embedding": (False, "Embedding model has not been checked yet."),
    "reranker": (False, "Reranker model has not been checked yet."),
    "storage": (False, "Storage has not been checked yet."),
}


# Terminals render these as 2 columns wide, but Python's len() counts each
# as 1 codepoint — left uncorrected, that mismatch throws off ljust() and
# the box's right border drifts out of alignment on every icon row.
_WIDE_CHARS = {"✅", "❌"}

# Skip colors when output isn't a real terminal (e.g. redirected to a log
# file), so piped/logged output stays plain text instead of raw escape codes.
_USE_COLOR = sys.stdout.isatty()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _c(code: str, s: str) -> str:
    return f"{code}{s}\x1b[0m" if _USE_COLOR and s else s


_BORDER = "\x1b[90m"  # gray
_TITLE = "\x1b[1m"  # bold
_NAME = "\x1b[36m"  # cyan
_LABEL = "\x1b[2m"  # dim
_DEVICE_COLOR = {"cuda": "\x1b[32m", "mps": "\x1b[32m", "cpu": "\x1b[33m"}  # green / yellow
_DEVICE_RE = re.compile(r"\b(cuda|mps|cpu)\b", re.IGNORECASE)


def _colorize_devices(detail: str) -> str:
    if not _USE_COLOR:
        return detail
    return _DEVICE_RE.sub(
        lambda m: _c(_DEVICE_COLOR[m.group(1).lower()], m.group(1)), detail
    )


def _visual_width(s: str) -> int:
    stripped = _ANSI_RE.sub("", s)
    return len(stripped) + sum(1 for ch in stripped if ch in _WIDE_CHARS)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _visual_width(s))


def _render_check_table(
    title: str, checks: list[tuple[str, list[tuple[str, tuple[bool, str]]]]]
) -> str:
    """Each top-level check can carry multiple sub-items sharing one name row
    (e.g. "Local (Ollama)" grouping its LLM and VLM checks, since both run
    through the same local Ollama server/model)."""
    name_width = max(len(name) for name, _ in checks)
    max_detail = 60
    rows = []
    for name, items in checks:
        for i, (label, (ok, detail)) in enumerate(items):
            icon = "✅" if ok else "❌"
            if len(detail) > max_detail:
                detail = detail[: max_detail - 1] + "…"
            detail = _colorize_devices(detail)
            prefix = _c(_LABEL, f"{label}: ") if label else ""
            shown_name = _c(_NAME, _pad(name, name_width)) if i == 0 else _pad("", name_width)
            rows.append(f"{icon} {shown_name}  {prefix}{detail}")

    width = max(_visual_width(title), *(_visual_width(row) for row in rows)) + 2
    top = _c(_BORDER, "┌" + "─" * (width + 2) + "┐")
    header = f"{_c(_BORDER, '│')} {_c(_TITLE, _pad(title, width))} {_c(_BORDER, '│')}"
    sep = _c(_BORDER, "├" + "─" * (width + 2) + "┤")
    body = [f"{_c(_BORDER, '│')} {_pad(row, width)} {_c(_BORDER, '│')}" for row in rows]
    bottom = _c(_BORDER, "└" + "─" * (width + 2) + "┘")
    return "\n".join([top, header, sep, *body, bottom])


def _core_ready() -> bool:
    return all(available for available, _ in _CORE_COMPONENT_STATUS.values())


def _ollama_device() -> str:
    """Live device of the loaded Ollama model via /api/ps, or config.DEVICE if not loaded yet."""
    try:
        request = urllib.request.Request(f"{config.OLLAMA_API_BASE}/api/ps", method="GET")
        with urllib.request.urlopen(request, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        expected = config.OLLAMA_MODEL.casefold()
        for model in data.get("models", []):
            if not isinstance(model, dict):
                continue
            names = {str(model.get("name", "")), str(model.get("model", ""))}
            if expected in {name.casefold() for name in names}:
                return config.DEVICE if (model.get("size_vram") or 0) > 0 else "cpu"
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return config.DEVICE


async def _run_startup_checks() -> None:
    # Model loading is slow and synchronous; running each check in a worker
    # thread keeps the event loop free to serve the "starting up" page below
    # instead of leaving the port refusing connections until this finishes.
    loop = asyncio.get_running_loop()
    openrouter_status = await loop.run_in_executor(
        None, check_connection, config.OPENROUTER_PROVIDER_ID
    )
    ollama_status = await loop.run_in_executor(
        None, check_connection, config.OLLAMA_PROVIDER_ID
    )
    docling_status = await loop.run_in_executor(None, check_docling)
    embedding_status = await loop.run_in_executor(None, check_embedder)
    reranker_status = await loop.run_in_executor(None, check_reranker)
    storage_status = await loop.run_in_executor(None, check_stores)
    # Informational only, like OpenRouter/Ollama above — picture description
    # is best-effort (ingestion/enrichment.py) and must never block the app.
    picture_description_status = await loop.run_in_executor(None, check_picture_description)

    _CORE_COMPONENT_STATUS.update(
        {
            "docling": docling_status,
            "embedding": embedding_status,
            "reranker": reranker_status,
            "storage": storage_status,
        }
    )

    # Runs after the VLM check, which is what actually loads the model.
    if ollama_status[0] or picture_description_status[0]:
        ollama_device = _ollama_device()
        if ollama_status[0]:
            ollama_status = (True, f"{ollama_status[1]} on {ollama_device}")
        if picture_description_status[0] and config.ENRICHMENT_ENABLED:
            picture_description_status = (
                True, f"{picture_description_status[1]}, {ollama_device}"
            )

    checks = [
        ("OpenRouter", [("", openrouter_status)]),
        ("Local (Ollama)", [("LLM", ollama_status), ("VLM", picture_description_status)]),
        ("Document conversion (Docling)", [("", docling_status)]),
        ("Embedding model", [("", embedding_status)]),
        ("Reranker model", [("", reranker_status)]),
        ("Storage", [("", storage_status)]),
    ]
    table = _render_check_table("Startup checks", checks)
    print(table, flush=True)


@app.on_event("startup")
async def run_startup_checks() -> None:
    asyncio.create_task(_run_startup_checks())


def _render_startup_page() -> str:
    rows = "".join(
        f'<div class="row {"ok" if available else "pending"}">'
        f'<span class="dot"></span><span class="component">{name}</span>'
        f'<span class="detail">{detail}</span></div>'
        for name, (available, detail) in _CORE_COMPONENT_STATUS.items()
    )
    return _STARTUP_PAGE_TEMPLATE.replace("__STATUS_ROWS__", rows)


@app.middleware("http")
async def gate_until_ready(request: Request, call_next):
    # /api/health must stay reachable during startup so this page (and any
    # external monitor) can see live progress instead of getting the same
    # placeholder back.
    if not _core_ready() and request.url.path != "/api/health":
        return HTMLResponse(_render_startup_page(), status_code=503)
    return await call_next(request)


def _friendly_generation_error(exc: Exception, model_id: str) -> str:
    """Map an answer-generation failure to a professional, user-facing
    message — never the raw exception text, which can leak implementation
    details (API keys, provider error payloads, stack traces)."""
    provider_id = config.MODEL_BY_ID[model_id]["provider_id"]
    if isinstance(exc, ProviderUnavailableError):
        return exc.user_message
    if provider_id == config.OLLAMA_PROVIDER_ID and isinstance(
        exc, (APIConnectionError, APITimeoutError)
    ):
        return (
            "The selected local AI model is not running. "
            "Start the project-local Ollama service and try again."
        )
    if isinstance(exc, NotFoundError):
        # OpenRouter can retire a free model without notice (observed live:
        # minimax/minimax-m3:free — see eval/README.md). Fix is picking a
        # different model, not retrying this one.
        return (
            "This model is no longer available for free on OpenRouter. "
            "Please select a different model from the list and try again."
        )
    if isinstance(exc, RateLimitError):
        return (
            "Our AI provider has reached its usage limit for today. "
            "Please try again later, or contact the administrator to raise the limit."
        )
    if isinstance(exc, AuthenticationError):
        return (
            "We're unable to connect to the AI service right now due to a "
            "configuration issue. Please contact the administrator."
        )
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return (
            "We couldn't reach the AI service right now. "
            "Please check your connection and try again shortly."
        )
    logging.error(
        "Answer generation failed for provider %s (%s)",
        provider_id,
        type(exc).__name__,
    )
    return "Something went wrong while generating a response. Please try again in a moment."


class NoCacheStaticFiles(StaticFiles):
    """Serves frontend assets with caching disabled so edits to index.html /
    style.css show up on refresh instead of being served from a stale
    browser cache during development."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    model: str | None = None


class ModelOption(BaseModel):
    id: str
    name: str
    provider: str
    provider_id: str
    local: bool
    available: bool
    detail: str
    blurb: str


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelOption]


class ProviderHealth(BaseModel):
    available: bool
    detail: str
    checked: bool


class ComponentHealth(BaseModel):
    available: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, ProviderHealth]
    components: dict[str, ComponentHealth]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload").suffix
    if suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid document. Unsupported file type '{suffix}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )

    header = await file.read(2048)
    await file.seek(0)
    valid, detected_mime = is_supported_content(suffix, header)
    if not valid:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid document. File content ('{detected_mime}') does not "
                f"match its extension '{suffix}'."
            ),
        )

    temp_path = config.UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with open(temp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        record = ingest_document(str(temp_path), file.filename or temp_path.name)
    finally:
        temp_path.unlink(missing_ok=True)

    if record["status"] == "failed":
        raise HTTPException(status_code=422, detail=record["error"])
    return record


@app.get("/api/documents")
async def get_documents():
    return list_documents()


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str):
    delete_document(document_id)
    return {"deleted": document_id}


@app.get("/api/documents/{document_id}/pages/{page_no}")
async def get_document_page(document_id: str, page_no: int):
    path = get_page_image_path(document_id, page_no)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not available.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/documents/{document_id}/images")
async def get_document_images(document_id: str):
    return list_images(document_id)


@app.get("/api/documents/{document_id}/images/{index}")
async def get_document_image(document_id: str, index: int):
    path = get_extracted_image_path(document_id, index)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not available.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/eval/report")
async def get_eval_report():
    """Serve the most recently exported evaluation HTML report (see
    eval/README.md and eval/run_eval.py), so the "Evaluation" sidebar button
    has something to open. Picks whichever of results_quick.html /
    results_full.html was produced most recently, from eval/results/ (the
    live output folder — eval/results_archive/ holds a permanent backup of
    every completed run and is deliberately not searched here)."""

    reports = sorted(
        (EVAL_DIR / "results").glob("results*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return HTMLResponse(
            "<p style='font: 14px system-ui; padding: 2rem;'>"
            "No evaluation report yet. From the <code>eval/</code> directory, run "
            "<code>run_eval.py --quick</code> (or <code>--full</code>), then reload this page."
            "</p>",
            status_code=404,
        )
    return HTMLResponse(reports[0].read_text(encoding="utf-8"))


@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    models = []
    for option in config.MODEL_CATALOG:
        status = get_model_status(option["id"])
        models.append(
            ModelOption(
                id=option["id"],
                name=option["name"],
                provider=option["provider"],
                provider_id=option["provider_id"],
                local=option["local"],
                available=status.available,
                detail=status.detail,
                blurb=option["blurb"],
            )
        )
    return ModelsResponse(
        default_model=config.DEFAULT_MODEL,
        models=models,
    )


@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    """Report application health using cached provider checks only."""

    providers: dict[str, ProviderHealth] = {}
    for provider_id in config.LLM_PROVIDERS:
        provider_status = get_provider_status(provider_id)
        providers[provider_id] = ProviderHealth(
            available=provider_status.available,
            detail=provider_status.detail,
            checked=provider_status.checked,
        )

    components = {
        component_id: ComponentHealth(available=available, detail=detail)
        for component_id, (available, detail) in _CORE_COMPONENT_STATUS.items()
    }
    core_ready = all(component.available for component in components.values())
    provider_ready = any(provider.available for provider in providers.values())

    return HealthResponse(
        status="ok" if core_ready and provider_ready else "error",
        providers=providers,
        components=components,
    )


# Maps each traced pipeline step to the coarse phase the chat UI shows in its
# "thinking" bubble, so the bubble label tracks the query chain (embed ->
# search -> rerank -> generate, Preliminary Report §3.4) as it actually runs
# rather than on a guessed timer.
STAGE_LABELS = {
    "plan_query": "thinking",
    "embed_query": "embedding",
    "vector_search": "searching",
    "load_candidate_metadata": "searching",
    "rerank_candidates": "reranking",
    "assemble_retrieved_chunks": "reranking",
    "reflect_on_evidence": "searching",
    "list_documents_lookup": "searching",
    "call_llm": "generating",
    "extract_citations": "generating",
}


@app.post("/api/query")
async def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    selected_model = config.DEFAULT_MODEL if request.model is None else request.model
    if selected_model not in config.MODEL_IDS:
        raise HTTPException(status_code=400, detail="Unsupported model selection.")

    model_status = get_model_status(selected_model)
    if not model_status.available:
        raise HTTPException(status_code=503, detail=model_status.detail)

    loop = asyncio.get_running_loop()
    stage_queue: asyncio.Queue[str] = asyncio.Queue()

    def on_step(step_name: str) -> None:
        stage = STAGE_LABELS.get(step_name)
        if stage:
            loop.call_soon_threadsafe(stage_queue.put_nowait, stage)

    trace = PipelineTrace(
        "query",
        datetime.now().strftime("%H:%M:%S.%f")[:-3],
        on_step=on_step,
        model=selected_model,
        top_k=request.top_k or config.TOP_K,
    )
    trace.start(question=request.question.strip())

    def run_pipeline() -> dict:
        started = time.time()
        try:
            result = run_agent(request.question, selected_model, trace)
        except ProviderUnavailableError as exc:
            trace.fail(exc, stage="generation")
            return {"error": _friendly_generation_error(exc, selected_model)}
        except Exception as exc:  # noqa: BLE001 - map to a professional message below
            trace.fail(exc, stage="generation")
            return {"error": _friendly_generation_error(exc, selected_model)}

        latency = time.time() - started
        retrieved_chunks = result["chunks"]
        log_query(
            request.question,
            retrieved_chunks,
            result["answer"],
            latency,
            selected_model,
        )
        trace.complete(
            retrieved=len(retrieved_chunks),
            citations=len(result["citations"]),
            latency_ms=round(latency * 1000),
        )
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "retrieved_count": len(retrieved_chunks),
            "latency_seconds": round(latency, 3),
            "model": selected_model,
        }

    async def event_stream():
        pipeline_task = asyncio.create_task(asyncio.to_thread(run_pipeline))
        yield json.dumps({"stage": "thinking"}) + "\n"
        while not pipeline_task.done():
            try:
                stage = await asyncio.wait_for(stage_queue.get(), timeout=0.1)
                yield json.dumps({"stage": stage}) + "\n"
            except asyncio.TimeoutError:
                continue
        while not stage_queue.empty():
            yield json.dumps({"stage": stage_queue.get_nowait()}) + "\n"
        yield json.dumps(pipeline_task.result()) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Mounted last: serves frontend/index.html at "/" and any other frontend
# asset (style.css, etc.) by filename. Must stay after the API routes above
# so it only catches requests those routes don't handle.
app.mount(
    "/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
)


def _start_local_ollama():
    """Best-effort launch of the project-owned Ollama server so ``python
    api.py`` brings up the local LLM and the web app together. Returns the
    managed process (or ``None``) so the caller can stop it on shutdown."""
    from generation.ollama.local_ollama import (
        LocalRuntimeError,
        OLLAMA_MODEL,
        install_runtime,
        model_is_installed,
        pull_model,
        server_is_healthy,
        start_server,
        stop_process,
    )

    managed = None
    try:
        executable = install_runtime()
        if server_is_healthy():
            print(f"Local Ollama is already running at {config.OLLAMA_API_BASE}")
        else:
            managed = start_server()

        if not model_is_installed():
            print(f"Local model {OLLAMA_MODEL} is not installed. Downloading it now.")
            pull_model(executable)
    except LocalRuntimeError as exc:
        stop_process(managed)
        print(f"Skipping local Ollama startup: {exc}")
        print("The OpenRouter model will still work if it is configured.")
        return None

    return managed


if __name__ == "__main__":
    import uvicorn

    _start_local_ollama()

    from generation.ollama.local_ollama import stop_server

    try:
        uvicorn.run(
            "api:app",
            host=config.APP_HOST,
            port=config.APP_PORT,
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
            reload_excludes=["*/.venv/*", "*/knowledge_base/*"],
        )
    finally:
        # Stops Ollama even if this run adopted an already-running instance
        # (server_is_healthy() was true, so _start_local_ollama() never
        # spawned a process of its own to track) rather than only the
        # process this run actually launched.
        stop_server()
