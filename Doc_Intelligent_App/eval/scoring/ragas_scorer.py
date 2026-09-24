"""RAGAS scoring for one question/answer/context triple.

Judge LLM: a single fixed OpenRouter model (EVAL_JUDGE_MODEL, below), kept
separate from every candidate model under test so a model never grades its
own answer (self-preference bias). Embeddings: the app's own already-loaded
BAAI/bge-m3 instance (retrieval/embedder.py) rather than pulling in a second
embedding stack.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import _ragas_compat  # noqa: F401 - must be imported before `ragas` (see module docstring there)

from rapidfuzz import fuzz

# eval/scoring/ragas_scorer.py -> parent (scoring/) -> parent (eval/) -> parent (Doc_Intelligent_App/)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402
from retrieval.embedder import embed_query, embed_texts  # noqa: E402

from langchain_openai import ChatOpenAI  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

# History: two NVIDIA OpenRouter judges (nemotron-3-ultra, then
# nemotron-3.5-lightning) were ruled out for free-tier 502 congestion plus
# JSON-format/parse failures. Also tried the local Ollama model (zero
# network dependency) — but on this machine's 4GB VRAM, running it
# alongside the already-resident embedder+reranker made judge calls stall
# for 20+ minutes without completing even once (confirmed live: the same
# model answers in ~10s when nothing else is loaded). Reverted to
# cohere/north-mini-code:free — a code-tuned model, well-suited to strict
# JSON compliance, went 16/16 clean in one run and 73/76 in a larger one.
EVAL_JUDGE_MODEL = "cohere/north-mini-code:free"


class LocalBgeM3Embeddings(BaseRagasEmbeddings):
    """Wraps the app's own sentence-transformers BAAI/bge-m3 instance so RAGAS
    doesn't load (or download) a second embedding model just for scoring."""

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


_judge_llm: LangchainLLMWrapper | None = None
_judge_embeddings: LocalBgeM3Embeddings | None = None
_metrics: dict[str, Any] | None = None


def _get_metrics() -> dict[str, Any]:
    global _judge_llm, _judge_embeddings, _metrics
    if _metrics is not None:
        return _metrics

    _judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            model=EVAL_JUDGE_MODEL,
            temperature=0.0,
            # Without an explicit timeout, the OpenAI SDK's default (600s,
            # up to 3 attempts) let one unresponsive judge call block the
            # whole run for up to ~30 minutes with no output. Bounding it
            # here makes a stuck call fail fast into the per-metric
            # "error: ..." capture below instead.
            timeout=90.0,
            max_retries=2,
            # Deliberately no max_tokens override: a live A/B test showed
            # capping it at 2048 (matching config.GENERATION_MAX_TOKENS)
            # made truncation failures *worse* for a different judge model.
            # Leaving max_tokens unset lets the provider's own default apply.
        )
    )
    _judge_embeddings = LocalBgeM3Embeddings()

    _metrics = {
        "faithfulness": Faithfulness(llm=_judge_llm),
        "response_relevancy": ResponseRelevancy(
            llm=_judge_llm, embeddings=_judge_embeddings
        ),
        "context_precision": LLMContextPrecisionWithReference(llm=_judge_llm),
        "context_recall": LLMContextRecall(llm=_judge_llm),
    }
    return _metrics


# Retrieval recall — a deterministic complement to the four LLM-judged
# metrics above. Compares retrieved_contexts against a known-correct
# reference_contexts (the actual verbatim source text, see tests.yaml), with
# no judge LLM involved at all — so it can't be thrown off by judge-model
# flakiness, and it answers a narrower, more literal question: did the
# retrieved text actually contain the known-correct passage, rather than an
# LLM's opinion of whether it was "relevant enough". Conceptually this is
# Recall@k — did we find what we should within the retrieved set — via
# substring text matching against a known-correct passage instead of a
# labeled relevant-ID set (chunk IDs aren't usable for that here — see
# backend/config.py's OPENROUTER_MODELS history for the unrelated but
# same-shaped lesson about not trusting anything tied to a specific
# ingestion run).
#
# History: this originally used RAGAS's own NonLLMContextPrecisionWithReference
# /NonLLMContextRecall, which score whole-string similarity (normalized
# Levenshtein) between each retrieved context and each reference context.
# That's the wrong comparison for this project — reference_contexts are
# single verbatim sentences (see tests.yaml), but retrieved_contexts are
# full chunks up to config.CHUNK_SIZE (1200) characters. Confirmed live in
# a real eval run: every case scored exactly 0.00 on both metrics, even ones
# the LLM-judged metrics rated perfect (faithfulness/precision/recall all
# 1.00). Diagnosed directly: a chunk that genuinely contains the reference
# sentence, padded to a realistic 1200 characters, still only scores ~0.10
# on whole-string similarity — indistinguishable from actual noise, and no
# threshold separates the two.
#
# Replaced with a direct substring-style check using rapidfuzz's
# partial_ratio, which finds the best-aligned substring window instead of
# comparing the strings as wholes. Confirmed live: a genuine containing
# match scores 100 regardless of chunk length, versus ~51 for unrelated
# text of similar length — a clean separation the old approach couldn't
# produce at any threshold.
#
# A companion Precision@k (relevant retrieved chunks / total retrieved
# chunks) was tried and dropped: with config.TOP_K=5 fixed and almost every
# question annotated with exactly one reference sentence, its ceiling is
# mechanically 1/5 = 0.20 even at perfect retrieval (recall=1.0) — it was
# reporting the fixed shape of the dataset (1 fact, 5 slots), not retrieval
# quality, and reading as a near-permanent failure regardless of how well
# the system actually performed. Recall alone (did we find the fact, yes or
# no) is the meaningful signal this dataset can actually support.
RETRIEVAL_MATCH_THRESHOLD = 80  # rapidfuzz partial_ratio, 0-100 scale


def _retrieval_recall(contexts: list[str], reference_contexts: list[str]) -> float:
    """Deterministic Recall@k: does each reference passage show up in some
    retrieved chunk? Mirrors the intent of RAGAS's context_recall metric,
    just via substring matching instead of an LLM judge or whole-string
    similarity."""

    def _contains(reference: str, retrieved: str) -> bool:
        return fuzz.partial_ratio(reference, retrieved) >= RETRIEVAL_MATCH_THRESHOLD

    found_references = sum(
        1
        for ref in reference_contexts
        if any(_contains(ref, context) for context in contexts)
    )
    return found_references / len(reference_contexts)


def score(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    reference_contexts: list[str] | None = None,
) -> dict[str, float | str | None]:
    """Score one QA sample against the four LLM-judged RAGAS metrics, plus
    retrieval_recall when reference_contexts is supplied.

    Returns a dict of metric_name -> float score, or metric_name -> an
    "error: ..." string if that particular metric call failed (e.g. an
    OpenRouter free-tier rate limit) — kept per-metric so one flaky judge
    call doesn't blank out the other scores for this sample.
    """

    if not contexts:
        results: dict[str, float | str | None] = {
            "faithfulness": None,
            "response_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "skipped_reason": "no retrieved context to score against",
        }
        if reference_contexts:
            results["retrieval_recall"] = None
        return results

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        reference=ground_truth,
        retrieved_contexts=contexts,
    )

    results = {}
    for name, metric in _get_metrics().items():
        results[name] = _score_with_retry(metric, sample)

    if reference_contexts:
        results["retrieval_recall"] = _retrieval_recall(contexts, reference_contexts)

    return results


_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5.0


def _score_with_retry(metric: Any, sample: SingleTurnSample) -> float | str:
    """Retry a single metric call a few times before giving up.

    Absorbs the transient connection errors/upstream 502s seen in real runs
    (as opposed to a persistent judge-model problem, which would fail on
    every attempt anyway and still surface as an "error: ..." string).
    """

    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return metric.single_turn_score(sample)
        except Exception as exc:  # noqa: BLE001 - one bad judge call must not sink the others
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    return f"error: {last_exc}"
