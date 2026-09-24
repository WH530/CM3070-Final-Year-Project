"""Central configuration for DocAI.

All tunables referenced in the CM3020 preliminary report (chunking, retrieval
depth, reranker threshold, model identifiers) live here so each pipeline
stage stays independently testable and the run configuration is reproducible.
Edit the constants in this file to change the runtime configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
UPLOAD_DIR = KNOWLEDGE_BASE_DIR / "uploads"
FILES_DIR = KNOWLEDGE_BASE_DIR / "files"
DB_DIR = KNOWLEDGE_BASE_DIR / "db"
LOG_DIR = KNOWLEDGE_BASE_DIR / "logs"
# Per-document Docling markdown export, kept for traceability.
DOCLING_DIR = KNOWLEDGE_BASE_DIR / "docling"

APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_BASE_URL = f"http://{APP_HOST}:{APP_PORT}"

for d in (UPLOAD_DIR, FILES_DIR, DB_DIR, LOG_DIR, DOCLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Model identifiers (pre-trained models orchestrated by this system) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
if torch.cuda.is_available():
    DEVICE = "cuda"  # Nvidia GPU (Windows/Linux)
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU (macOS)
else:
    DEVICE = "cpu"

# Loaded from the environment — a local backend/.env file in development (see
# backend/.env.example), a real environment variable in deployment — so no
# key is ever hardcoded or committed. Empty/unset is a supported state, not
# an error: is_openrouter_configured() in generation/llm.py treats it as
# "OpenRouter unavailable" and the app falls back to the local Ollama model.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "inclusionai/ling-3.0-flash-fin:free"

# Ollama exposes an OpenAI-compatible API. The browser never supplies this URL:
# model identifiers are resolved through the allowlisted catalog below, which
# prevents a query from turning the backend into a proxy for an arbitrary host.
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11435
OLLAMA_API_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_BASE_URL = f"{OLLAMA_API_BASE}/v1"
OLLAMA_API_KEY = "local-ollama"  # required by the SDK, ignored by Ollama
OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"
OLLAMA_MODEL_ID = f"ollama/{OLLAMA_MODEL}"
OLLAMA_CONTEXT_LENGTH = 8192
# >1 lets Ollama serve concurrent requests (ingestion/enrichment.py's chunk
# batches), at the cost of one extra KV-cache reservation per slot — keep
# this modest on a small/shared GPU.
OLLAMA_NUM_PARALLEL = 2
OLLAMA_MAX_LOADED_MODELS = 1
OLLAMA_KEEP_ALIVE = "5m"
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_TOKENS = 2048

OPENROUTER_PROVIDER_ID = "openrouter"
OLLAMA_PROVIDER_ID = "ollama"
LLM_PROVIDERS = {
    OPENROUTER_PROVIDER_ID: {
        "name": "OpenRouter",
        "base_url": OPENROUTER_BASE_URL,
        "local": False,
    },
    OLLAMA_PROVIDER_ID: {
        "name": "Ollama (local)",
        "base_url": OLLAMA_BASE_URL,
        "local": True,
    },
}

# Free text-generation options verified against OpenRouter's model catalog on
# 2026-09-03, each confirmed with live test calls before being added (not
# just catalog presence — several free models are unreliable in practice):
#   - openai/gpt-oss-20b:free was retired by OpenRouter (404); replaced by
#     z-ai/glm-5.2:free, which then failed 4/4 live calls (429s from its
#     upstream provider "Decart"); settled on minimax/minimax-m3:free,
#     confirmed working live and through the real agent pipeline.
#   - google/gemma-4-31b-it:free was replaced after sustained upstream
#     rate-limiting (5/5 failures in one real eval run). Candidates ruled
#     out: poolside/laguna-s-2.1:free (unreliable, ~1/3 success; also a
#     coding-specialist model, wrong fit) and thinkingmachines/inkling(-small)
#     :free (hard 403 — restricted to agentic harnesses, unusable here).
#     Settled on dots-studio/dots-3-note-preview:free — 5/5 clean at
#     GENERATION_MAX_TOKENS (an earlier test at a too-small max_tokens
#     misleadingly looked unreliable; it's a "thinking" model that needs
#     real budget to emit visible output).
#   - The "long, complex, multi-step questions" slot has moved twice.
#     nvidia/nemotron-3-ultra-550b-a55b:free was briefly swapped out for
#     liquid/lfm-2.5-2.6b:free after ~33% of Nemotron Ultra's live generation
#     calls failed, consistent with its ~37s average latency on OpenRouter's
#     own listing causing free-tier timeouts; LFM2.5-2.6B ran 3/3 clean at
#     GENERATION_MAX_TOKENS in that test but is a much smaller compact 2.6B
#     model, a real capability downgrade for this slot despite being faster
#     and more reliable. Nemotron 3 Ultra was restored, then replaced again
#     with nvidia/nemotron-3.5-lightning:free, the slot's current model — not
#     yet independently live-tested here, so re-verify before relying on it.
#     If it proves unreliable, LFM2.5-2.6B remains the proven-reliable
#     fallback above (at that same capability cost) and Nemotron 3 Ultra is
#     the higher-capability but slower/flakier alternative.
#   - The "fast, efficient everyday reasoning" slot has moved several times,
#     landing back where it started. minimax/minimax-m3:free was retired by
#     OpenRouter's free tier entirely (404: "This model is unavailable for
#     free... use this slug instead: minimax/minimax-m3" — the paid version).
#     Caught live, mid eval run, not from catalog browsing. Replaced with
#     inclusionai/ling-3.0-flash-fin:free — already live-tested for the eval
#     harness's own candidate list (3/3 clean, 2-4s per call), filling both
#     roles: this picker slot and the harness's fourth full-pipeline
#     candidate. Briefly swapped to google/gemma-4-26b-a4b-it:free instead,
#     which then failed 2/2 live calls here (RateLimitError, "Google AI
#     Studio" upstream shared pool temporarily rate-limited) — the same
#     failure mode as google/gemma-4-31b-it:free above, confirmed live on a
#     same-day retest of that older model too (also 2/2 failed, identical
#     upstream message), so this is Google's free-tier pool being throttled
#     right now, not one specific Gemma model. poolside/laguna-s-2.1:free was
#     also retested live and only cleared 1/2 (a Poolside-side 429, distinct
#     from the Google issue) — consistent with its older "~1/3 success,
#     wrong-fit coding model" verdict above. Reverted to
#     inclusionai/ling-3.0-flash-fin:free, the only model in this slot's
#     history with a clean live-test record, restoring the parity between
#     this picker slot and the eval harness's provider list.
# Availability can change; keep this list current. The singular
# OPENROUTER_MODEL above remains the OpenRouter startup-check reference model.
# Ordered most- to least-capable so the model picker reads like a tier list,
# with the always-available local model last; ``blurb`` is the one-line
# capability description shown under each model's name in that picker.
OPENROUTER_MODELS = (
    {
        "id": "nvidia/nemotron-3.5-lightning:free",
        "name": "Nemotron 3.5 Lightning",
        "provider": "NVIDIA",
        "blurb": "For long, complex, multi-step questions",
    },
    {
        "id": "dots-studio/dots-3-note-preview:free",
        "name": "Dots3-Note Preview",
        "provider": "Dots Studio",
        "blurb": "Strong all-round reasoning and analysis",
    },
    {
        "id": "inclusionai/ling-3.0-flash-fin:free",
        "name": "Ling 3.0 Flash Fin",
        "provider": "inclusionAI",
        "blurb": "Fast, efficient everyday reasoning",
    },
    {
        "id": "openrouter/free",
        "name": "Auto-select Free Model",
        "provider": "OpenRouter",
        "blurb": "Automatically routes to an available free model",
    },
)

OPENROUTER_MODEL_IDS = frozenset(option["id"] for option in OPENROUTER_MODELS)

LOCAL_MODELS = (
    {
        "id": OLLAMA_MODEL_ID,
        "api_model": OLLAMA_MODEL,
        "name": "Qwen3.5 4B (Local)",
        "provider": "Qwen",
        "provider_id": OLLAMA_PROVIDER_ID,
        "local": True,
        "blurb": "Fastest — runs fully offline on your device",
    },
)

# This is the complete server-side routing catalog. ``id`` is the public value
# accepted from the frontend; ``api_model`` and ``provider_id`` determine the
# fixed upstream client and model name used by the backend.
MODEL_CATALOG = (
    tuple(
        {
            **option,
            "api_model": option["id"],
            "provider_id": OPENROUTER_PROVIDER_ID,
            "local": False,
        }
        for option in OPENROUTER_MODELS
    )
    + LOCAL_MODELS
)
MODEL_BY_ID = {option["id"]: option for option in MODEL_CATALOG}
MODEL_IDS = frozenset(MODEL_BY_ID)
DEFAULT_MODEL = OLLAMA_MODEL_ID

if OPENROUTER_MODEL not in OPENROUTER_MODEL_IDS:
    raise RuntimeError("OPENROUTER_MODEL must be included in OPENROUTER_MODELS.")
if any(
    model_id != "openrouter/free" and not model_id.endswith(":free")
    for model_id in OPENROUTER_MODEL_IDS
):
    raise RuntimeError("Every selectable OpenRouter model must use a free endpoint.")
if len(MODEL_CATALOG) != len(MODEL_BY_ID):
    raise RuntimeError("Every selectable model must have a unique public id.")
if DEFAULT_MODEL not in MODEL_IDS:
    raise RuntimeError("DEFAULT_MODEL must be included in MODEL_CATALOG.")
if any(option["provider_id"] not in LLM_PROVIDERS for option in MODEL_CATALOG):
    raise RuntimeError("Every selectable model must reference a configured provider.")

# --- Ingestion pipeline tuning ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# --- Conversion stage tuning (ingestion/converter.py) ---
DOCLING_IMAGE_SCALE = 2.0  # extracted figure resolution multiplier (1.0 == 72 DPI)

# --- Enrichment stage tuning (ingestion/enrichment.py) ---
# Best-effort chunk/document metadata via LLM, always via the local Ollama
# model regardless of the user's chat model choice — this is an internal
# indexing step, not a user-facing generation, so it stays free and offline.
ENRICHMENT_ENABLED = True
DOCUMENT_CATEGORIES = ("policy", "manual", "contract", "report", "faq", "other")
# Chunks per LLM call for per-chunk keyword/question/tag extraction — fewer,
# bigger requests instead of one call per chunk. Batches themselves run up
# to OLLAMA_NUM_PARALLEL at a time.
ENRICHMENT_BATCH_SIZE = 5

# --- Query pipeline tuning ---
CANDIDATE_DEPTH = 20  # first-stage recall-oriented depth
TOP_K = 5  # retrieved chunks sent to the LLM after reranking
RERANK_SCORE_FLOOR = -6.0  # below this, treat as irrelevant

# --- Query agent tuning (agent/graph.py) ---
# Hard cap on total searches per query: _search_node increments `hops` after
# the mandatory first search, so this must be >=2 for _reflect_node to ever
# actually invoke its follow-up-query decision (at 1, reflect always sees
# hops>=MAX_RETRIEVAL_HOPS on the first pass and skips straight to answering).
MAX_RETRIEVAL_HOPS = 2

# --- Storage paths (embedded, in-process — no external DB service) ---
VECTOR_DB_PATH = str(DB_DIR / "vectors.json")
METADATA_DB_PATH = str(DB_DIR / "metadata.json")
RAW_FILES_DIR = str(FILES_DIR)
QUERY_LOG_PATH = str(LOG_DIR / "queries.jsonl")
