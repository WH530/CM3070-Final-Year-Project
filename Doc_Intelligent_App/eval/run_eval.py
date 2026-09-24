"""Standalone evaluation runner — replaces the Promptfoo orchestration layer.

Why this exists: Promptfoo spawns one persistent Python worker process per
provider, each of which loads its own separate copy of the embedder
(BAAI/bge-m3) and reranker (BAAI/bge-reranker-v2-m3) onto the GPU. On a
small GPU (this machine: 4GB VRAM), loading 4 separate copies exhausts
available memory and crashes the whole run (see eval/README.md history).

This script runs everything in a single process instead. retrieval/embedder.py
and retrieval/reranker.py cache their models in a module-level singleton
(`_get_model()`), so calling them for all 5 provider variants and every
question in one process loads each model exactly once, total.

RAGAS scoring (scoring/ragas_scorer.py) and the reranker-ablation retrieval
variant (scoring/retrieval_variants.py) are unchanged — this script only
replaces the orchestration/looping/output layer that Promptfoo used to
provide, not the actual evaluation logic.

Usage (from eval/):
    ../.venv/Scripts/python.exe run_eval.py --quick
    ../.venv/Scripts/python.exe run_eval.py --full
    ../.venv/Scripts/python.exe run_eval.py --quick --first-n 2 --only-provider "Local Qwen 3.5 4B (full pipeline)"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent
SCORING_DIR = EVAL_DIR / "scoring"
BACKEND_DIR = EVAL_DIR.parent / "backend"
for _dir in (EVAL_DIR, SCORING_DIR, BACKEND_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# results/ holds the current report (api.py reads it too); finished runs
# are also backed up into results_archive/ (see end of main()).
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_ARCHIVE_DIR = EVAL_DIR / "results_archive"
RESULTS_DIR.mkdir(exist_ok=True)

from agent.graph import AgentState, _get_graph  
from generation.llm import generate_answer  
from generation.ollama.local_ollama import (  
    LocalRuntimeError,
    ManagedProcess,
    install_runtime,
    server_is_healthy,
    start_server,
    stop_process,
)
from pipeline_trace import PipelineTrace  

import ragas_scorer  
from retrieval_variants import retrieve_vector_only  

PROVIDERS = [
    {
        "label": "Local Qwen 3.5 4B (full pipeline)",
        "model_id": "ollama/qwen3.5:4b-q4_K_M",
        "mode": "full",
    },
    {
        "label": "Dots3-Note Preview (full pipeline)",
        "model_id": "dots-studio/dots-3-note-preview:free",
        "mode": "full",
    },
    {
        "label": "Ling 3.0 Flash Fin (full pipeline)",
        "model_id": "inclusionai/ling-3.0-flash-fin:free",
        "mode": "full",
    },
    {
        "label": "Nemotron 3.5 Lightning (full pipeline)",
        "model_id": "nvidia/nemotron-3.5-lightning:free",
        "mode": "full",
    },
    {
        "label": "Local Qwen 3.5 4B (vector-only, no reranker)",
        "model_id": "ollama/qwen3.5:4b-q4_K_M",
        "mode": "vector_only",
    },
]

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "response_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "retrieval_recall": "Retrieval Recall",
}
THRESHOLD = 0.5


def ensure_ollama_running() -> ManagedProcess | None:
    """Start the project-owned local Ollama server if it isn't already up.

    Returns the process this call started, or None if Ollama was already
    healthy (in which case we leave it running afterward — it isn't ours to
    stop). The local-Qwen providers below need this; without it they fail
    with a connection-refused error instead of a clear message.
    """
    if server_is_healthy():
        print("Local Ollama is already running — leaving it as-is.")
        return None
    print("Local Ollama is not running — starting it for this eval run...")
    try:
        install_runtime()
        managed = start_server()
    except LocalRuntimeError as exc:
        print(
            f"Could not auto-start local Ollama: {exc}\n"
            "Local-Qwen provider cases will fail with a connection error; "
            "every other provider is unaffected."
        )
        return None
    return managed


def run_full(question: str, model_id: str) -> dict:
    trace = PipelineTrace("eval", question[:8], model=model_id)
    trace.start()
    initial_state: AgentState = {
        "question": question,
        "model": model_id,
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
    trace.complete()
    chunks = final_state["accumulated_chunks"]
    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "contexts": [c.text for c in chunks],
        "action": final_state["action"],
        "hops": final_state["hops"],
    }


def run_vector_only(question: str, model_id: str) -> dict:
    chunks = retrieve_vector_only(question)
    generation = generate_answer(question, chunks, model=model_id)
    return {
        "answer": generation.answer,
        "citations": generation.citations,
        "contexts": [c.text for c in chunks],
        "action": "search",
        "hops": 0,
    }


def run_one_case(test_case: dict, provider: dict) -> dict:
    question = test_case["vars"]["question"]
    ground_truth = test_case["vars"].get("ground_truth", "")
    category = test_case["vars"].get("category", "")

    reference_contexts = test_case["vars"].get("reference_contexts")

    row: dict[str, Any] = {
        "question": question,
        "category": category,
        "ground_truth": ground_truth,
        "source_document": test_case["vars"].get("source_document", ""),
        "source_page": test_case["vars"].get("source_page", ""),
        "source_section": test_case["vars"].get("source_section", ""),
        "provider_label": provider["label"],
        "model_id": provider["model_id"],
        "mode": provider["mode"],
    }

    started = time.time()
    try:
        if provider["mode"] == "vector_only":
            run_result = run_vector_only(question, provider["model_id"])
        else:
            run_result = run_full(question, provider["model_id"])
    except Exception as exc:  # noqa: BLE001 - one bad case must not sink the run
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_seconds"] = round(time.time() - started, 3)
        return row

    row["answer"] = run_result["answer"]
    row["citations"] = run_result["citations"]
    row["contexts"] = run_result["contexts"]
    row["action"] = run_result["action"]
    row["hops"] = run_result["hops"]
    row["latency_seconds"] = round(time.time() - started, 3)

    try:
        scores = ragas_scorer.score(
            question=question,
            contexts=run_result["contexts"],
            answer=run_result["answer"],
            ground_truth=ground_truth,
            reference_contexts=reference_contexts,
        )
    except Exception as exc:  # noqa: BLE001 - scoring failure shouldn't lose the answer
        scores = {metric: f"error: {exc}" for metric in METRIC_LABELS}
    row.update(scores)
    return row


def load_dataset(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [{"description": item["description"], "vars": item["vars"]} for item in raw]


def metric_pass(value: Any) -> bool | None:
    if value is None or isinstance(value, str):
        return None
    return value >= THRESHOLD


def write_json(results: list[dict], out_path: Path, meta: dict) -> None:
    out_path.write_text(
        json.dumps({**meta, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )


def write_csv(results: list[dict], out_path: Path) -> None:
    fieldnames = [
        "question",
        "category",
        "source_document",
        "source_page",
        "source_section",
        "provider_label",
        "mode",
        "answer",
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "context_recall",
        "retrieval_recall",
        "error",
        "latency_seconds",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _cell_html(row: dict) -> str:
    if "error" in row:
        return f'<div class="cell err">[ERROR]<br>{html.escape(row["error"])}</div>'
    answer = html.escape((row.get("answer") or "")[:400])
    parts = [f'<div class="answer">{answer}</div>']
    for key, label in METRIC_LABELS.items():
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f'<div class="score fail">{label}: {html.escape(value)}</div>')
            continue
        ok = metric_pass(value)
        cls = "pass" if ok else "fail"
        parts.append(f'<div class="score {cls}">{label}: {value:.3f}</div>')
    return f'<div class="cell">{"".join(parts)}</div>'


CHART_COLORS = ["#4f8cff", "#6fe38a", "#ffb84f", "#ff6f91"]


def _metric_stats(results: list[dict], provider_label: str) -> dict[str, tuple[float, int]]:
    """Average + sample size per metric for one provider, over scored
    (non-error) cases only. A NaN from the judge on an individual case (it
    can fail to score one metric without the whole case erroring) is
    dropped rather than left in the mean, where it would poison the sum."""
    stats: dict[str, tuple[float, int]] = {}
    for key in METRIC_LABELS:
        values = [
            r[key]
            for r in results
            if r.get("provider_label") == provider_label
            and "error" not in r
            and isinstance(r.get(key), (int, float))
            and not math.isnan(r[key])
        ]
        if values:
            stats[key] = (sum(values) / len(values), len(values))
    return stats


def _svg_bar_chart(
    title: str,
    subtitle: str,
    series: list[tuple[str, str]],
    stats_by_series: dict[str, dict[str, tuple[float, int]]],
    metric_keys: list[str] | None = None,
) -> str:
    """series: [(provider_label, color), ...]. stats_by_series maps
    provider_label -> {metric_key: (avg, n)} from _metric_stats(). Renders
    one grouped bar chart, one group per metric, one bar per series.
    metric_keys picks which metrics to plot and in what order — defaults to
    every metric in METRIC_LABELS."""
    keys = metric_keys if metric_keys is not None else list(METRIC_LABELS)
    metrics = [(k, METRIC_LABELS[k]) for k in keys]
    chart_w, chart_h = 720, 300
    left_pad, bottom_pad, top_pad = 40, 50, 30
    plot_w = chart_w - left_pad - 20
    plot_h = chart_h - top_pad - bottom_pad
    group_w = plot_w / len(metrics)
    bar_w = min(36, group_w / (len(series) + 1))
    group_gap = (group_w - bar_w * len(series)) / 2

    def y(value: float) -> float:
        return top_pad + plot_h * (1 - value)

    parts = [f'<svg viewBox="0 0 {chart_w} {chart_h}" width="100%" style="max-width:{chart_w}px">']
    # gridlines + y-axis labels. Colors are set via CSS classes (below,
    # in write_html's <style> block) rather than these fill/stroke
    # attributes alone, so the chart repaints when the report's data-theme
    # switches — an inline presentation attribute has lower cascade
    # priority than any class rule in a real <style> block, so the two
    # can coexist: the attribute is just the light-theme/no-CSS fallback.
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gy = y(tick)
        parts.append(f'<line class="grid-line" x1="{left_pad}" y1="{gy:.1f}" x2="{chart_w - 20}" y2="{gy:.1f}" stroke="#2a2a33" stroke-width="1"/>')
        parts.append(f'<text class="axis-label" x="{left_pad - 8}" y="{gy + 4:.1f}" font-size="10" fill="#888" text-anchor="end">{tick:.1f}</text>')

    for gi, (metric_key, metric_label) in enumerate(metrics):
        group_x = left_pad + gi * group_w
        for si, (provider_label, color) in enumerate(series):
            stat = stats_by_series.get(provider_label, {}).get(metric_key)
            bx = group_x + group_gap + si * bar_w
            if stat is None:
                parts.append(f'<text class="na-label" x="{bx + bar_w / 2:.1f}" y="{y(0) - 4:.1f}" font-size="9" fill="#555" text-anchor="middle">n/a</text>')
                continue
            avg, n = stat
            by_ = y(avg)
            bh = y(0) - by_
            parts.append(
                f'<rect x="{bx:.1f}" y="{by_:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" rx="2">'
                f'<title>{html.escape(provider_label)} — {metric_label}: {avg:.2f} (n={n})</title></rect>'
            )
            parts.append(f'<text class="bar-value" x="{bx + bar_w / 2:.1f}" y="{by_ - 4:.1f}" font-size="9" fill="#ccc" text-anchor="middle">{avg:.2f}</text>')
        parts.append(
            f'<text class="group-label" x="{group_x + group_w / 2:.1f}" y="{chart_h - bottom_pad + 16:.1f}" '
            f'font-size="10" fill="#aaa" text-anchor="middle">{metric_label}</text>'
        )

    parts.append("</svg>")

    legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{html.escape(label)}</span>'
        for label, color in series
    )
    return (
        f'<div class="chart-block"><h3>{html.escape(title)}</h3>'
        f'<div class="chart-subtitle">{html.escape(subtitle)}</div>'
        f'{"".join(parts)}<div class="legend">{legend}</div></div>'
    )


def _latency_stats(results: list[dict], provider_label: str) -> tuple[float, int] | None:
    """Mean latency_seconds for one provider, over non-error cases only —
    mirrors _metric_stats but for wall-clock time rather than a 0-1 RAGAS
    score, so it's rendered as its own small table instead of sharing the
    0-1 scaled bar chart axis."""
    values = [
        r["latency_seconds"]
        for r in results
        if r.get("provider_label") == provider_label
        and "error" not in r
        and isinstance(r.get("latency_seconds"), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values), len(values)


def _build_latency_table_html(results: list[dict]) -> str:
    """Reranker latency cost: full pipeline vs vector-only, same underlying
    model held fixed (the same ablation pair used for the RAGAS metrics
    chart), so the delta isolates what the reranker itself adds rather than
    conflating it with generation-model variance."""
    vector_only = next((p for p in PROVIDERS if p["mode"] == "vector_only"), None)
    if vector_only is None:
        return ""
    full_match = next(
        (
            p
            for p in PROVIDERS
            if p["mode"] == "full" and p["model_id"] == vector_only["model_id"]
        ),
        None,
    )
    if full_match is None:
        return ""

    vo_stat = _latency_stats(results, vector_only["label"])
    full_stat = _latency_stats(results, full_match["label"])
    if vo_stat is None or full_stat is None:
        return ""

    vo_avg, vo_n = vo_stat
    full_avg, full_n = full_stat
    delta = full_avg - vo_avg
    model_name = full_match["label"].split(" (")[0]

    def _row(label: str, avg: float, n: int) -> str:
        return (
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{avg:.2f}s</td><td>{n}</td></tr>"
        )

    return (
        '<div class="chart-block latency-block"><h3>Reranker latency cost</h3>'
        f'<div class="chart-subtitle">Same model ({html.escape(model_name)}), same questions — '
        "end-to-end latency including generation, so the delta below reflects the reranker's "
        "added cost rather than model variance.</div>"
        '<table class="latency-table">'
        "<tr><th>Provider</th><th>Avg latency</th><th>n</th></tr>"
        f"{_row(vector_only['label'], vo_avg, vo_n)}"
        f"{_row(full_match['label'], full_avg, full_n)}"
        f"<tr><td><b>Delta (reranker cost)</b></td><td><b>{delta:+.2f}s</b></td><td></td></tr>"
        "</table></div>"
    )


def _build_charts_html(results: list[dict]) -> str:
    labels = {p["label"]: p for p in PROVIDERS}
    full_pipeline_providers = [p["label"] for p in PROVIDERS if p["mode"] == "full"]
    stats_by_provider = {label: _metric_stats(results, label) for label in labels}

    model_series = list(zip(full_pipeline_providers, CHART_COLORS))
    # Retrieval metrics dropped here entirely (not just retrieval_precision):
    # embedder + reranker are identical across every provider in this chart,
    # so retrieval isn't what's actually varying between the bars — only the
    # generation model is. The small retrieval_recall differences that do
    # show up come from each model's own query-planning step, a secondary
    # effect this chart isn't meant to be measuring. Keep to the four core
    # generation-quality metrics.
    model_chart = _svg_bar_chart(
        "Model comparison",
        "Same retrieval (full pipeline, reranker on), different generation model — n shown on hover.",
        model_series,
        stats_by_provider,
        metric_keys=["faithfulness", "response_relevancy", "context_precision", "context_recall"],
    )

    # The ablation pair is whichever provider runs vector_only mode, matched
    # against the full-pipeline provider using the same model_id — found
    # dynamically rather than hardcoded, since which model fills this role
    # has changed across harness revisions.
    ablation_chart = ""
    vector_only = next((p for p in PROVIDERS if p["mode"] == "vector_only"), None)
    if vector_only is not None:
        full_match = next(
            (
                p
                for p in PROVIDERS
                if p["mode"] == "full" and p["model_id"] == vector_only["model_id"]
            ),
            None,
        )
        if full_match is not None:
            model_name = full_match["label"].split(" (")[0]
            ablation_series = [
                (vector_only["label"], CHART_COLORS[0]),
                (full_match["label"], CHART_COLORS[1]),
            ]
            ablation_chart = _svg_bar_chart(
                "Reranker ablation",
                f"Same model ({model_name}), same questions — only difference is reranker on/off.",
                ablation_series,
                stats_by_provider,
            )

    latency_table = _build_latency_table_html(results)
    return f'<div class="charts">{model_chart}{ablation_chart}{latency_table}</div>'


def write_html(results: list[dict], out_path: Path, meta: dict) -> None:
    by_question: dict[str, list[dict]] = {}
    for row in results:
        by_question.setdefault(row["question"], []).append(row)

    n_error = sum(1 for r in results if "error" in r)
    n_total = len(results)

    rows_html = []
    for question, rows in by_question.items():
        cells = "".join(f"<td>{_cell_html(r)}</td>" for r in rows)
        category = html.escape(rows[0].get("category", ""))
        rows_html.append(
            f'<tr><td class="qcell"><b>{html.escape(question)}</b>'
            f'<div class="cat">{category}</div></td>{cells}</tr>'
        )

    provider_headers = "".join(f"<th>{html.escape(p['label'])}</th>" for p in PROVIDERS)
    charts_html = _build_charts_html(results)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DocAI Evaluation — {html.escape(meta.get('dataset', ''))}</title>
<script>
  // Runs before first paint, same pattern as the main app's index.html —
  // set from the parent app's own data-theme when embedded in the
  // Evaluation iframe (app.js appends ?theme=... to the frame src), so this
  // page stops being permanently dark regardless of the app's own theme.
  // Opened directly with no query string (its original standalone use),
  // it still defaults to dark exactly as before.
  (function () {{
    var theme = new URLSearchParams(location.search).get('theme') === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
  }})();
</script>
<style>
:root {{
  --bg: #0b0b0f; --text: #e6e6e6; --muted: #999; --border: #333; --th-bg: #1a1a22;
  --pass-bg: #12351c; --pass-text: #6fe38a; --fail-bg: #3a1414; --fail-text: #ff8a8a;
  --chart-bg: #14141b; --chart-border: #26262f; --legend-text: #ccc;
  --grid-stroke: #2a2a33; --axis-text: #888; --bar-value: #ccc; --group-label: #aaa; --na-label: #555;
}}
:root[data-theme="light"] {{
  --bg: #f7f8fa; --text: #16182c; --muted: #6b7280; --border: #e2e5eb; --th-bg: #eef0f5;
  --pass-bg: #e2f7e6; --pass-text: #1f9d55; --fail-bg: #fbe5e5; --fail-text: #c9372c;
  --chart-bg: #ffffff; --chart-border: #e2e5eb; --legend-text: #374151;
  --grid-stroke: #dfe3ea; --axis-text: #6b7280; --bar-value: #16182c; --group-label: #4b5563; --na-label: #9ca3af;
}}
body {{ font: 13px system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }}
h1 {{ font-size: 18px; }}
h3 {{ font-size: 14px; margin: 0 0 2px; }}
.meta {{ color: var(--muted); margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid var(--border); padding: 8px; vertical-align: top; text-align: left; }}
th {{ background: var(--th-bg); position: sticky; top: 0; }}
.qcell {{ width: 220px; }}
.cat {{ color: var(--muted); font-size: 11px; margin-top: 4px; }}
.answer {{ margin-bottom: 6px; white-space: pre-wrap; }}
.score {{ font-size: 11px; padding: 2px 4px; border-radius: 3px; display: inline-block; margin: 1px 2px 1px 0; }}
.pass {{ background: var(--pass-bg); color: var(--pass-text); }}
.fail {{ background: var(--fail-bg); color: var(--fail-text); }}
.err {{ color: var(--fail-text); }}
.charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 28px; }}
.chart-block {{ background: var(--chart-bg); border: 1px solid var(--chart-border); border-radius: 8px; padding: 14px 16px; flex: 1 1 380px; }}
.chart-subtitle {{ color: var(--muted); font-size: 11px; margin-bottom: 8px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; font-size: 11px; color: var(--legend-text); }}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.latency-block {{ flex: 1 1 320px; }}
.latency-table {{ width: 100%; table-layout: auto; margin-top: 8px; }}
.latency-table th, .latency-table td {{ border: 1px solid var(--chart-border); padding: 6px 8px; font-size: 12px; }}
.latency-table th {{ background: var(--th-bg); position: static; }}
.grid-line {{ stroke: var(--grid-stroke); }}
.axis-label {{ fill: var(--axis-text); }}
.bar-value {{ fill: var(--bar-value); }}
.group-label {{ fill: var(--group-label); }}
.na-label {{ fill: var(--na-label); }}
</style></head>
<body>
<h1>DocAI Evaluation — {html.escape(meta.get('dataset', ''))} run</h1>
<div class="meta">
  Generated {html.escape(meta.get('finished_at', ''))} &middot;
  {n_total} test cases, {n_error} errors ({(n_error / n_total * 100) if n_total else 0:.0f}%)
</div>
{charts_html}
<table>
<tr><th class="qcell">Question</th>{provider_headers}</tr>
{"".join(rows_html)}
</table>
</body></html>"""
    out_path.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="4 questions (1/category)")
    group.add_argument("--full", action="store_true", help="12 questions (3/category)")
    parser.add_argument("--first-n", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--only-provider", type=str, default=None, help="Only run the provider with this exact label")
    args = parser.parse_args()

    dataset_name = "quick" if args.quick else "full"
    dataset_path = EVAL_DIR / ("tests_quick.yaml" if args.quick else "tests.yaml")
    test_cases = load_dataset(dataset_path)
    if args.first_n:
        test_cases = test_cases[: args.first_n]

    providers = PROVIDERS
    if args.only_provider:
        providers = [p for p in providers if p["label"] == args.only_provider]
        if not providers:
            print(f"No provider matches label: {args.only_provider!r}")
            return 1

    total = len(test_cases) * len(providers)
    print(f"Running {len(test_cases)} question(s) x {len(providers)} provider(s) = {total} test case(s)")
    print("Single process — embedder/reranker load once and are reused for every case.\n")

    needs_ollama = any(p["model_id"].startswith("ollama/") for p in providers)
    managed_ollama = ensure_ollama_running() if needs_ollama else None

    try:
        started_at = datetime.now(timezone.utc).isoformat()
        results: list[dict] = []
        out_json = RESULTS_DIR / f"results_{dataset_name}.json"
        out_csv = RESULTS_DIR / f"results_{dataset_name}.csv"
        out_html = RESULTS_DIR / f"results_{dataset_name}.html"

        done = 0
        run_started = time.time()
        for test_case in test_cases:
            for provider in providers:
                done += 1
                question_preview = test_case["vars"]["question"][:60]
                print(f"[{done}/{total}] {provider['label']} :: {question_preview}...", flush=True)
                row = run_one_case(test_case, provider)
                results.append(row)

                if "error" in row:
                    print(f"    -> ERROR: {row['error']}", flush=True)
                else:
                    summary = ", ".join(
                        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()
                        if k in METRIC_LABELS
                    )
                    print(f"    -> {summary}", flush=True)

                # Write after every case, not just at the end, so an interrupted
                # run still leaves usable partial results on disk.
                meta = {
                    "dataset": dataset_name,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "total_cases": total,
                    "completed_cases": done,
                }
                write_json(results, out_json, meta)
                write_csv(results, out_csv)
                write_html(results, out_html, meta)

        elapsed = time.time() - run_started
        n_error = sum(1 for r in results if "error" in r)
        print(f"\nDone in {elapsed / 60:.1f} min. {len(results) - n_error}/{len(results)} cases completed without error.")
        print(f"Wrote {out_json.name}, {out_csv.name}, {out_html.name}")

        # Back up this completed run to results_archive/ so it's never lost.
        archive_name = f"{dataset_name}_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        archive_dir = RESULTS_ARCHIVE_DIR / archive_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        for src in (out_json, out_csv, out_html):
            shutil.copy2(src, archive_dir / src.name)
        print(f"Backed up to {archive_dir.relative_to(EVAL_DIR)}/")
        return 0
    finally:
        if managed_ollama is not None:
            print("Stopping the Ollama server this run started...")
            stop_process(managed_ollama)


if __name__ == "__main__":
    raise SystemExit(main())
