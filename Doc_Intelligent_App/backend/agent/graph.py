"""Query orchestration agent ("document_agent", LangGraph): decides how to
answer a question.

Replaces the old always-retrieve-then-answer query chain with an actual
decision-making step. Given a question, the graph decides whether to answer
directly, list the available documents, or search the knowledge base —
looping the search once more if the retrieved evidence looks insufficient,
bounded by config.MAX_RETRIEVAL_HOPS.

Decisions are made via structured JSON output (response_format=json_object),
the same mechanism ingestion/enrichment.py already relies on — not native
OpenAI-style tool-calling, since free-tier OpenRouter models (most of this
app's model catalog) have documented reliability problems with tool-calling
("no endpoints found that support tool use"), while json_object mode is far
more broadly supported. Any parse failure or missing field falls back to the
safest default (always search, then stop and answer) — today's old
fixed-pipeline behavior — so a flaky decision degrades gracefully instead of
breaking the query.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

import yaml
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# Imported as app_config (not `config`) because LangGraph only injects the
# per-invocation RunnableConfig into a node when its parameter is literally
# named `config` — that name is reserved for the framework in this module.
import config as app_config
from generation.llm import GenerationResult, _client_for_provider, _model_spec, generate_answer
from ingestion.pipeline import list_documents
from pipeline_trace import PipelineTrace
from retrieval.pipeline import RetrievedChunk, retrieve

with open(
    Path(__file__).resolve().parent.parent / "prompt.yaml", encoding="utf-8"
) as _f:
    _PROMPTS = yaml.safe_load(_f)["agent"]

PLAN_SYSTEM_PROMPT = _PROMPTS["plan_system_prompt"].strip()
PLAN_USER_TEMPLATE = _PROMPTS["plan_user_template"].strip()
REFLECT_SYSTEM_PROMPT = _PROMPTS["reflect_system_prompt"].strip()
REFLECT_USER_TEMPLATE = _PROMPTS["reflect_user_template"].strip()

_ACTIONS = ("answer_directly", "list_documents", "summarize_document", "search")


class AgentState(TypedDict):
    question: str
    model: str
    hops: int
    action: str
    search_query: str
    sufficient: bool
    accumulated_chunks: list[RetrievedChunk]
    answer: str
    citations: list[dict]


def _call_json(model: str, system_prompt: str, user_prompt: str) -> dict:
    """Call the model with json_object response format and parse the result."""
    model_spec = _model_spec(model)

    if model_spec["provider_id"] == app_config.OLLAMA_PROVIDER_ID:
        # Ollama's OpenAI-compatible endpoint (below) ignores `think`, so
        # qwen3.5's reasoning pass runs regardless and, at this small
        # max_tokens budget, consumes it entirely before any JSON is
        # emitted (finish_reason="length", empty content) — silently
        # forcing every decision to its safe-default fallback. Only the
        # native /api/chat endpoint honors think:False.
        payload = {
            "model": model_spec["api_model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            f"{app_config.OLLAMA_API_BASE}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        return json.loads(data["message"]["content"] or "{}")

    client = _client_for_provider(model_spec["provider_id"])
    response = client.chat.completions.create(
        model=model_spec["api_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


def _format_evidence_summary(chunks: list[RetrievedChunk]) -> str:
    """Render accumulated chunks as a numbered text block for the reflect prompt."""
    if not chunks:
        return "(no evidence gathered yet)"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        snippet = chunk.text[:300].replace("\n", " ")
        lines.append(f"[{i}] {chunk.title}: {snippet}")
    return "\n".join(lines)


def _merge_chunks(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Merge new chunks into existing ones, keeping the best rerank_score per chunk_id."""
    by_id = {chunk.chunk_id: chunk for chunk in existing}
    for chunk in new:
        current = by_id.get(chunk.chunk_id)
        if current is None or chunk.rerank_score > current.rerank_score:
            by_id[chunk.chunk_id] = chunk
    return sorted(by_id.values(), key=lambda c: c.rerank_score, reverse=True)


def _plan_node(state: AgentState, config: RunnableConfig) -> dict:
    """Decide the action for this question: answer directly, list documents, or search."""
    trace: PipelineTrace = config["configurable"]["trace"]
    with trace.step("plan_query", model=state["model"]) as step:
        try:
            decision = _call_json(
                state["model"],
                PLAN_SYSTEM_PROMPT,
                PLAN_USER_TEMPLATE.format(question=state["question"]),
            )
            action = str(decision.get("action", "search")).strip().lower()
            search_query = str(decision.get("search_query") or state["question"]).strip()
        except Exception:  # noqa: BLE001 - a flaky decision must not break the query
            action, search_query = "search", state["question"]
        if action not in _ACTIONS:
            action = "search"
        step["action"] = action
    return {"action": action, "search_query": search_query}


def _search_node(state: AgentState, config: RunnableConfig) -> dict:
    """Retrieve chunks for the current search_query and merge them into accumulated_chunks."""
    trace: PipelineTrace = config["configurable"]["trace"]
    chunks = retrieve(state["search_query"], top_k=app_config.TOP_K, trace=trace)
    merged = _merge_chunks(state["accumulated_chunks"], chunks)[: app_config.TOP_K]
    return {"accumulated_chunks": merged, "hops": state["hops"] + 1}


def _reflect_node(state: AgentState, config: RunnableConfig) -> dict:
    """Decide whether accumulated evidence is sufficient, or propose a followup search."""
    trace: PipelineTrace = config["configurable"]["trace"]
    with trace.step("reflect_on_evidence", hops=state["hops"]) as step:
        if state["hops"] >= app_config.MAX_RETRIEVAL_HOPS:
            step["sufficient"] = True
            step["reason"] = "max_hops_reached"
            return {"sufficient": True}
        try:
            decision = _call_json(
                state["model"],
                REFLECT_SYSTEM_PROMPT,
                REFLECT_USER_TEMPLATE.format(
                    question=state["question"],
                    evidence=_format_evidence_summary(state["accumulated_chunks"]),
                ),
            )
            sufficient = bool(decision.get("sufficient", True))
            followup = str(decision.get("followup_query") or "").strip()
        except Exception:  # noqa: BLE001 - a flaky decision must not break the query
            sufficient, followup = True, ""
        step["sufficient"] = sufficient
        return {
            "sufficient": sufficient,
            "search_query": followup or state["search_query"],
        }


def _list_documents_node(state: AgentState, config: RunnableConfig) -> dict:
    """Answer directly from the document registry, without retrieval."""
    trace: PipelineTrace = config["configurable"]["trace"]
    with trace.step("list_documents_lookup"):
        documents = list_documents()
    if not documents:
        answer = "No documents have been uploaded yet."
    else:
        lines = [
            f"- {doc['filename']} ({doc['status']}, {doc.get('chunk_count', 0)} chunks)"
            for doc in documents
        ]
        answer = "Here are the documents currently available:\n" + "\n".join(lines)
    return {"answer": answer, "citations": []}


def _summarize_document_node(state: AgentState, config: RunnableConfig) -> dict:
    """Answer directly from a document's stored summary_document field —
    skips chunk/vector search entirely, since the whole-document summary was
    already computed once at ingestion time (ingestion/enrichment.py)."""
    trace: PipelineTrace = config["configurable"]["trace"]
    query = state["search_query"].strip().lower()
    with trace.step("summarize_document_lookup", query=query) as step:
        documents = list_documents()
        match = next(
            (doc for doc in documents if query and query in doc["filename"].lower()),
            None,
        )
        step["matched"] = match["filename"] if match else None
    if match is None:
        answer = f"I couldn't find a document matching '{state['search_query']}'."
    elif not match.get("summary_document"):
        answer = (
            f"No summary is available yet for {match['filename']} "
            "(it may still be processing, or summarization failed)."
        )
    else:
        answer = match["summary_document"]
    return {"answer": answer, "citations": []}


def _answer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final answer and citations from accumulated_chunks."""
    trace: PipelineTrace = config["configurable"]["trace"]
    result: GenerationResult = generate_answer(
        state["question"], state["accumulated_chunks"], model=state["model"], trace=trace
    )
    return {"answer": result.answer, "citations": result.citations}


def _route_after_plan(state: AgentState) -> str:
    """Route to the edge matching the plan node's chosen action."""
    return state["action"]


def _route_after_reflect(state: AgentState) -> str:
    """Route to answer if evidence is sufficient, otherwise loop back to search."""
    return "answer" if state["sufficient"] else "search"


def _build_graph():
    """Wire up the plan/search/reflect/list_documents/answer nodes into the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan_node)
    graph.add_node("search", _search_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("list_documents", _list_documents_node)
    graph.add_node("summarize_document", _summarize_document_node)
    graph.add_node("answer", _answer_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "answer_directly": "answer",
            "list_documents": "list_documents",
            "summarize_document": "summarize_document",
            "search": "search",
        },
    )
    graph.add_edge("search", "reflect")
    graph.add_conditional_edges(
        "reflect", _route_after_reflect, {"answer": "answer", "search": "search"}
    )
    graph.add_edge("list_documents", END)
    graph.add_edge("summarize_document", END)
    graph.add_edge("answer", END)
    return graph.compile(name="document_agent")


_graph: Any = None


def _get_graph():
    """Return the compiled agent graph, building it once and caching it."""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_agent(question: str, model: str, trace: PipelineTrace) -> dict:
    """Run the agent graph for a question and return its answer, citations, and chunks."""
    initial_state: AgentState = {
        "question": question,
        "model": model,
        "hops": 0,
        "action": "search",
        "search_query": question,
        "sufficient": False,
        "accumulated_chunks": [],
        "answer": "",
        "citations": [],
    }
    final_state = _get_graph().invoke(
        initial_state, config={"configurable": {"trace": trace}}
    )
    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "chunks": final_state["accumulated_chunks"],
    }
