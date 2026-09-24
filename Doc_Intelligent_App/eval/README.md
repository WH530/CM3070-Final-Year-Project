# DocAI evaluation harness

Scores the real DocAI pipeline with RAGAS (Faithfulness, Answer/Response
Relevancy, Context Precision, Context Recall), plus one non-LLM retrieval
metric scored against known-correct source text, across four candidate
models plus a reranker on/off ablation, and writes the results to a
comparison report (`results_*.html`) also served by the app's "Evaluation"
sidebar button. Reproduces, with real numbers, the evaluation methodology
`sample_documents/Preliminary_Report.pdf` §4.3 already committed to
(illustrative Table 3 numbers in that report were never actually measured).

## What is RAGAS (R)

RAGAS (`pip install ragas`, imported in `scoring/ragas_scorer.py`, pinned
in `requirements.txt`) is an open-source **Python library** — an
evaluation toolkit built specifically for RAG (Retrieval-Augmented
Generation) systems like this one, rather than a general-purpose LLM
benchmark. It's the library actually computing the 4 metric scores below,
not just a name for "the evaluation" in general.

Calling it "code" is only half the story, though. RAGAS's functions are
deterministic in *how* they're wired together — same inputs always produce
the same sequence of steps — but the actual judgment inside each step (*"is
this statement true given the context?"*, *"is this retrieved chunk
relevant?"*) is made by an LLM, not by the code itself. Concretely, for
each metric:

| Metric | Description (what it measures) | How does it work | LLM-as-judge required? | Judge model |
|---|---|---|---|---|
| **Faithfulness** | Whether the generated answer is factually supported by the retrieved context — catches hallucination. | Splits the answer into individual factual statements (one LLM call), then asks the judge LLM whether each statement is supported by the retrieved context (one LLM call per statement). Score = fraction supported. | Yes | North Mini Code, by Cohere (via OpenRouter) |
| **Answer/Response Relevancy** | Whether the answer actually addresses the question asked, rather than being off-topic or incomplete. | Asks the judge LLM to generate several hypothetical questions the *answer* would be answering, then measures embedding similarity between those and the *actual* question asked. | Yes, plus a local embedding model (BAAI/bge-m3) for the similarity step | North Mini Code, by Cohere (via OpenRouter) |
| **Context Precision** | Whether the retrieved passages are actually relevant to the question — signal vs. noise in retrieval. | Asks the judge LLM, for each retrieved chunk, whether it's actually relevant to the question. Score is a ranking-weighted average of those judgments. | Yes | North Mini Code, by Cohere (via OpenRouter) |
| **Context Recall** | Whether the retrieved passages contain everything needed to fully answer the question — retrieval completeness. | Asks the judge LLM, for each fact in the ground-truth answer, whether it's attributable to the retrieved context. Score = fraction attributable. | Yes | North Mini Code, by Cohere (via OpenRouter) |

All four use the *same* fixed judge model (`EVAL_JUDGE_MODEL` in
`scoring/ragas_scorer.py`) — deliberately kept separate from every candidate
model under test, so a model never grades its own answer.

This is why RAGAS counts as **LLM-as-a-judge** evaluation, not a
**code/heuristic** evaluator — the actual verdicts come from an LLM's
judgment, not a deterministic string match.

### Retrieval Recall (a deterministic complement)

One more metric, `retrieval_recall`, runs alongside the four above whenever
a test case carries a `reference_contexts` field (every case in
`tests.yaml`/`tests_quick.yaml` does — see
`source_page`/`source_section`/`reference_contexts` in the dataset header
comment). Conceptually this is Recall@k: for each known-correct passage,
does it show up in some retrieved chunk? No judge LLM involved, so it can't
be thrown off by judge-model flakiness, and it answers a narrower, more
literal question than the LLM-judged pair above: did the retrieved text
actually contain the known-correct passage, rather than an LLM's opinion of
whether it was "relevant enough."

**History worth knowing before touching this code:** this originally used
RAGAS's own `NonLLMContextPrecisionWithReference`/`NonLLMContextRecall`,
which score whole-string similarity (normalized Levenshtein) between each
retrieved context and each reference context. That's the wrong comparison
here — `reference_contexts` are single verbatim sentences, but retrieved
contexts are full chunks up to `config.CHUNK_SIZE` (1200) characters.
Confirmed live in a real eval run: every case scored exactly `0.00` on both
metrics, even ones the LLM-judged metrics rated perfect. Diagnosed
directly: a chunk that genuinely contains the reference sentence, padded to
a realistic 1200 characters, still only scores ~0.10 on whole-string
similarity, indistinguishable from actual noise, and no threshold separates
the two. Replaced with a custom check using `rapidfuzz`'s `partial_ratio`
(`scoring/ragas_scorer.py`'s `_retrieval_recall`), which finds the
best-aligned substring window instead of comparing the strings as wholes —
confirmed live: a genuine containing match scores 100 regardless of chunk
length, versus ~51 for unrelated text of similar length, a clean separation
the old approach couldn't produce at any threshold.

A companion `retrieval_precision` (relevant retrieved chunks / total
retrieved chunks) was tried and dropped: with `config.TOP_K=5` fixed and
almost every question annotated with exactly one reference sentence, its
ceiling is mechanically `1/5 = 0.20` even at perfect retrieval — it reported
the fixed shape of the dataset (one fact, five slots), not retrieval
quality, and read as a near-permanent failure regardless of how well the
system actually performed. `retrieval_recall` alone (did we find the fact,
yes or no) is the meaningful signal this dataset can support.

## How the evaluation works

### The building blocks

**1 document** — the knowledge base every provider retrieves from (not
something being compared):
- `Preliminary_Report.pdf` (this project's own report — the only source
  document; two company documents were removed from the corpus for
  confidentiality. `table_cell` still has real material to test against
  despite this — the report itself contains three tables: Model
  Orchestration Matrix, Technology Stack, and RAGAS Results.)

**5 providers** — 5 different ways of answering a question (this is what's
being compared — nothing to do with which document is involved), built from
**4 distinct candidate models**:

| # | Provider | Generation model | Developer | Retrieval |
|---|---|---|---|---|
| 1 | Local Qwen 3.5 4B (full pipeline) | Qwen (local Ollama) | Alibaba | embed → vector search → rerank |
| 2 | Dots3-Note Preview (full pipeline) | Dots3-Note (OpenRouter) | Dots Studio (Xiaohongshu) | embed → vector search → rerank |
| 3 | Ling 3.0 Flash Fin (full pipeline) | Ling 3.0 Flash Fin (OpenRouter) | inclusionAI | embed → vector search → rerank |
| 4 | Nemotron 3.5 Lightning (full pipeline) | Nemotron 3.5 Lightning (OpenRouter) | NVIDIA | embed → vector search → rerank |
| 5 | Local Qwen 3.5 4B (vector-only) | Qwen (local Ollama) | Alibaba | embed → vector search → *(no rerank)* |

All 4 full-pipeline candidate/generation models come from a different
developer each — deliberate, so no two rows share a vendor's training data
or house style. The RAGAS judge (above) is from a fifth developer, Cohere,
kept separate from every candidate so a model never grades its own answer.

This matrix's fourth full-pipeline row has changed hands several times,
all from free-tier attrition, not a deliberate scope cut:

- **Nemotron 3 Ultra** (`nvidia/nemotron-3-ultra-550b-a55b:free`), the
  original occupant of this row (alongside a same-day judge-model
  experiment — see "Judge model history" below for that detour), failed
  ~33% of live calls, consistent with a ~37s average latency on
  OpenRouter's own listing causing free-tier timeouts. Its first
  replacement, **Poolside Laguna S 2.1**, was dropped before ever running
  here: `backend/config.py`'s own history already documented it as
  live-tested and rejected for this project, ~1/3 success rate, and a
  coding-specialist model, wrong fit for a document-QA candidate.
- **MiniMax M3** (`minimax/minimax-m3:free`), a separate row (the "fast,
  efficient everyday reasoning" slot), was retired by OpenRouter's free
  tier entirely, caught live mid-eval-run: a 404 said "This model is
  unavailable for free... use this slug instead: minimax/minimax-m3" (the
  paid version).

Both broken slots were temporarily collapsed into one model,
`inclusionai/ling-3.0-flash-fin:free`, live smoke-tested (3/3 clean calls
at `GENERATION_MAX_TOKENS`, 2-4s each) — a general-purpose
mixture-of-experts model rather than a specialist — leaving this matrix at
4 providers and 3 distinct candidates for a time, an honest reduction
given how many free-tier candidates had been tried and rejected.

The fourth row was later restored with a different Nemotron model,
`nvidia/nemotron-3.5-lightning:free`, live-tested 2/2 clean (every RAGAS
judge metric scoring 0.75-1.00), bringing the matrix back to its current
5 providers / 4 distinct candidates. It runs noticeably slower than the
other OpenRouter candidates (27-43s per generation call), reliable so far,
just slow — re-verify if it starts failing live, since two other Nemotron
models before it did. See `backend/config.py`'s `OPENROUTER_MODELS`
history for the full blow-by-blow of every model that has occupied each
slot.

**Questions** — quick = 4 (1 per category), full = 12 (3 per category),
across 4 categories: `factual_lookup`, `table_cell`, `named_section`,
`multi_hop`. Each category targets a different way a RAG pipeline can
fail — see `tests.yaml` for the full 12-question set:

| Category | What it tests | Example question (from `tests.yaml`) | Expected answer | Metrics applied |
|---|---|---|---|---|
| `factual_lookup` | Direct fact retrieval — the answer is a single, unambiguous value stated plainly in the text. | "According to the Preliminary Report, what was the average query latency measured, and how much of that was attributable to reranking?" | "Query latency averaged 3.1 seconds for a candidate depth of 20, with reranking contributing approximately 0.8 seconds." | All 5 metrics (4 LLM-judged + 1 non-LLM) |
| `table_cell` | Retrieval from structured/tabular content — a known hard case for chunking and embedding models, since the answer lives in a row/column relationship rather than prose. | "According to Table 1 (Model Orchestration Matrix), what is the Evaluation Approach listed for BAAI/bge-reranker-v2-m3?" | "Context precision, nDCG@k, and ablation versus vector-only baseline." | All 5 metrics (4 LLM-judged + 1 non-LLM) |
| `named_section` | Retrieval scoped to a specific, explicitly-named section — tests whether the pipeline finds the exact passage a user points it at. | "According to section 3.3 of the Preliminary Report (Document Ingestion Pipeline), which model computes vector embeddings for validated chunks?" | "BGE-M3 (BAAI/bge-m3)." | All 5 metrics (4 LLM-judged + 1 non-LLM) |
| `multi_hop` | Synthesis across two separate passages — the answer requires combining evidence the pipeline can't get from any single chunk alone. | "Does adding the cross-encoder reranker change which passages are retrieved, or only their order — and what does the evaluation data show?" | "Only the order/selection within the same candidate set... Recall@5 is identical (0.80) for both configurations." | All 5 metrics (4 LLM-judged + 1 non-LLM) |

(A fifth category, `unanswerable` — refusal discipline on questions genuinely
absent from the corpus — was removed. Its scoring was a deterministic
keyword check against the answer text, which turned out to be too brittle
in practice: a model that correctly explained *why* the data doesn't exist,
rather than using one of the fixed refusal phrases, was marked as a false
failure. See git history for `REFUSAL_SIGNALS` and the removed
`unanswerable` test cases if this is worth revisiting with a more robust
check later.)

### Test cases = questions × providers

Every question runs through all 5 providers independently, and every case
is scored by all 5 metrics — that's the whole matrix:

- Quick: 4 questions × 5 providers = **20 scored test cases**
- Full: 12 questions × 5 providers = **60 scored test cases**

### One matrix, two insights

There's a single result table, but you extract 2 different insights from it:

| # | Insight | What's compared | Answers the question | Scoring used |
|---|---|---|---|---|
| 1 | **Model comparison** | Rows 1, 2, 3, 4 against each other — same retrieval (full pipeline, reranker on), different generation model. | *"Which LLM writes the most faithful/relevant answer given the same evidence?"* | The 4 LLM-judged metrics (retrieval_recall is computed but not shown here — see "Retrieval Recall" below) |
| 2 | **Reranker ablation** | Row 1 against row 5 — same model (Qwen), same questions, only difference is reranker on/off. | *"Does the reranker actually improve results, or just reorder them?"* | All 5 metrics (4 LLM-judged + 1 non-LLM: retrieval_recall) |

So: one `run_eval.py` invocation, one dataset, one test matrix — the "2
evaluations" are really 2 different ways of slicing the same 60 (or 20)
result rows afterward, not 2 separate executions. (Full run: 60 rows;
quick run: 20 rows.)

## One-time setup

1. Install the eval-only Python dependency (already added to `requirements.txt`):
   ```
   cd Doc_Intelligent_App
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Ingest the test document into the knowledge base (skips if already
   ingested — safe to re-run):
   ```
   ..\.venv\Scripts\python.exe scripts\ingest_test_docs.py
   ```
3. Start **only** the local Ollama runtime — not the full app:
   ```
   ..\.venv\Scripts\python.exe scripts\run_ollama_only.py
   ```
   Leave it running in its own terminal — needed for the two Local Qwen
   provider rows. Don't also run `backend/api.py` at the same time as an
   eval — see "Why `run_ollama_only.py` and not `api.py`" below. Also make
   sure `backend/config.py`'s `OPENROUTER_API_KEY` is a working key — the
   other 3 candidate providers and the judge all call OpenRouter.

## Running

From inside `eval/`:

```
../.venv/Scripts/python.exe run_eval.py --quick
../.venv/Scripts/python.exe run_eval.py --full
```

This writes `results/results_quick.{json,csv,html}` (or `results_full.*`) —
the app's Evaluation sidebar button (`/api/eval/report` in `api.py`) picks
up whichever `results/results*.html` is newest, so no other setup is needed
for that button to work — including grouped bar charts comparing providers
(model comparison, reranker ablation), not just the raw per-question table.
A finished run is also copied into a timestamped folder under
`results_archive/` automatically, so `results/` always holds just "the
current one" of each dataset while every completed run stays permanently on
disk — see "Where results live" below.

### Where results live

- **`results/`** — the current, most-recently-written `results_quick.*` and
  `results_full.*`. This is what `rescore_errors.py` reads/rewrites, and
  what `api.py`'s Evaluation page serves. Re-running `--quick`/`--full`
  overwrites whatever was here before, incrementally, after every case (see
  above), so an interrupted run still leaves a usable partial file.
- **`results_archive/`** — a permanent, timestamped backup of every run
  that finished (loop completed, not interrupted), written automatically at
  the end of `main()` as `results_archive/<quick|full>_run_<UTC
  timestamp>/results_*.{json,csv,html}`. Nothing here is ever overwritten —
  each run gets its own folder — so this is the place to look for a
  specific past run's exact numbers, independent of whatever `results/`
  currently holds.

If a run is interrupted (killed mid-loop, crashed), only `results/` has its
partial output — nothing gets archived, since the archive step only runs
after the loop completes normally. Re-run to completion, or manually copy
`results/` somewhere safe, if a partial run's data is worth keeping.

**Why a single-process runner rather than a test-matrix tool like
Promptfoo** (used in an earlier version of this harness, since removed):
that kind of tool typically spawns one **separate persistent Python worker
process per provider**, each of which loads its own copy of the embedder
(`BAAI/bge-m3`) and reranker (`BAAI/bge-reranker-v2-m3`) onto the GPU. On a
small GPU (this machine has 4GB VRAM) that's fatal — 3 providers' worth of
models filled the available VRAM and the 4th worker crashed mid-run, which
cascaded into every remaining test case failing (an actual run this
produced: 0 passed, 19/20 errored). Limiting concurrency doesn't prevent
this — it only limits concurrent *execution*, not how many workers stay
resident in memory across the run.

`run_eval.py` runs everything in a **single process** instead. The
embedder/reranker's own module-level caching (`retrieval/embedder.py`,
`retrieval/reranker.py`) means each model loads exactly once, total, and is
reused across all 5 provider variants and every question — sidestepping the
GPU-contention problem entirely rather than working around it.

**Two datasets, deliberately:**

| Dataset | Questions | Use for |
|---|---|---|
| `--quick` (`tests_quick.yaml`) | 4 (1 per category) | sanity-checking changes, a quick demo run |
| `--full` (`tests.yaml`) | 12 (3 per category) | the actual report numbers, run this once when you have time |

**Sanity-check first** with `--first-n 2 --only-provider "<exact label>"`
before a full run, e.g.:

```
../.venv/Scripts/python.exe run_eval.py --quick --first-n 1 --only-provider "Dots3-Note Preview (full pipeline)"
```

Results are written incrementally, after every single test case (not just
at the end) — so if a run is interrupted partway through, `results_*.html`
still reflects everything completed so far, not nothing.

## Why `run_ollama_only.py` and not `api.py`

`api.py`'s own startup health checks (`retrieval/embedder.py` and
`retrieval/reranker.py`'s `check_load()`) eagerly load BAAI/bge-m3 and
BAAI/bge-reranker-v2-m3 onto the GPU. If `api.py` is running at the same time
as `run_eval.py`, the two separate processes fight over the same GPU — this
alone made the reranker's first load take ~650s in testing, vs. ~300s with
`api.py` stopped. `run_ollama_only.py` starts just the local Ollama server
(needed for the two Local Qwen provider rows), leaving the GPU free for
`run_eval.py`'s own model loads.

## Expect a long run regardless

Even with the GPU fix, the dominant cost is the **judge model itself**, not
compute: RAGAS's Faithfulness/Response Relevancy/Context Precision/Recall
metrics each make their own judge call (Faithfulness alone breaks the answer
into statements and verifies each one), and every test case is now scored
(no category skips scoring). With the current judge
(`cohere/north-mini-code:free`), a 20-case quick run took ~61-106 minutes in
earlier testing with a similarly-sized case count (variance mostly from
occasional judge retries); expect roughly 3x that for the 60-case full set.

A failed judge call is captured per-metric as an `"error: ..."` string in
that test's score rather than crashing the whole run, after 3 retries with
backoff (see `_score_with_retry` in `ragas_scorer.py`) — so a transient
blip degrades one number, not the whole eval.

**Judge model history:** three OpenRouter-hosted judges were tried and
ruled out for reliability under real, sustained load (a quick smoke test
alone wasn't enough to catch any of these — each needed a real pipeline run
to surface):
- `nvidia/nemotron-3-ultra-550b-a55b:free` — failed 36% of judge calls
  (23/64) with `"Upstream error from Nvidia: Service temporarily
  overloaded"` (502).
- `nvidia/nemotron-3.5-lightning:free` — passed 10/10 live smoke tests but
  then failed `context_precision`/`context_recall` on 2 of 4 scored cases
  in a real run (prose instead of JSON, truncated JSON).
- `nemotron-3-super-120b-a12b:free` — hit the same 502 congestion as the
  original, pointing at NVIDIA free-tier capacity generally rather than one
  bad model.

Settled on `cohere/north-mini-code:free`: a code-tuned model, a poor fit as
a document-QA candidate but well-suited to judge duty, where strict JSON
compliance is what matters — went 16/16 clean in one run, then 73/76 in a
larger one (a few Cohere-side 503s/truncation under sustained load, all
absorbed by the retry logic above in later runs).

**Local judge attempt (also reverted):** to eliminate even that residual
failure rate, the judge was switched to the local Ollama model — zero
network dependency should mean zero upstream failures. In isolation
(nothing else loaded) it answered in ~10s. But `run_eval.py` also keeps the
embedder and reranker resident in the same process for the whole run, and
on this machine's 4GB VRAM, running the judge model *alongside* those two
left too little headroom: judge calls stalled for 20+ minutes without
completing even once in a real run. Reverted back to Cohere; Qwen returned
to its original two candidate rows (it's never the judge, so grading its
own answers isn't a concern).

**A second reliability note, this time about a candidate, not the judge:**
Nemotron 3 Ultra's own generation calls (not judging — actually writing
answers) failed with `"returned no completion choices"` on 3 of 9 calls
(~33%) in one real run — an OpenRouter/upstream response with a 200 status
but an empty `choices` field. `generation/llm.py` now raises a clear
`RuntimeError` for this instead of an opaque `TypeError`, so it's captured
as a per-case harness error rather than crashing, but the underlying
flakiness in Nemotron's own upstream capacity is unresolved — treat its
row's results with that in mind, and consider it a candidate for
replacement if this recurs.

## Folder layout

```
eval/
  run_eval.py                    Run the evaluation — see "Running" above
  rescore_errors.py              Re-score only cells that came back as judge errors — see below
  tests.yaml / tests_quick.yaml  Golden datasets (12 / 4 questions), read directly by run_eval.py
  scoring/                       Pure Python: RAGAS scoring + the reranker-ablation retrieval variant
  scripts/                       One-off setup: ingest test docs, start Ollama standalone
```

## What each piece does

| File | Role |
|---|---|
| `run_eval.py` | **The entry point.** Single-process runner: loops questions × 5 providers, runs each through the real agent (or the vector-only variant), scores it, writes `results_*.{json,csv,html}` incrementally. |
| `rescore_errors.py` | Re-scores only the cells that came back as judge `"error: ..."` strings in an existing `results_*.json`, reusing the stored answer/contexts instead of re-running retrieval and generation — see "Recovering from judge errors" below. |
| `scoring/_ragas_compat.py` | Works around a `ragas`/`langchain-community` import bug — see its docstring. |
| `scoring/ragas_scorer.py` | Scores one (question, contexts, answer, ground_truth) sample against the 4 LLM-judged RAGAS metrics, judged by a fixed OpenRouter model kept separate from every candidate, plus `retrieval_recall` against `reference_contexts` when supplied. |
| `scoring/retrieval_variants.py` | Vector-only retrieval (no reranker) — used only by the ablation provider entry. |
| `scripts/ingest_test_docs.py` | One-time: loads the test PDF (`Preliminary_Report.pdf`) through the real ingestion pipeline. |
| `scripts/run_ollama_only.py` | Starts just the local Ollama server, without `api.py`'s own GPU model loads — see above. |
| `tests.yaml` / `tests_quick.yaml` | The golden datasets — full (12) and quick (4). |

## Recovering from judge errors

A run that mostly succeeds but has a few `"error: ..."` cells doesn't need
a full re-run:

```
../.venv/Scripts/python.exe rescore_errors.py --quick
../.venv/Scripts/python.exe rescore_errors.py --full
```

This finds every cell with a judge error, re-scores just those using the
already-stored answer and retrieved contexts (no retrieval or generation
re-run), and rewrites `results_*.{json,csv,html}` in place. Only works for
result files that have the `contexts` field per row (added alongside this
script) — older files predate it and need the affected cases re-run
directly instead (`--only-provider`).

## Known caveats worth stating in the report

- The judge (North Mini Code, by Cohere, via OpenRouter) is itself an LLM —
  validate a sample of its verdicts by hand against the source PDFs before
  trusting the numbers wholesale (the plan's verification step covers this).
- `backend/config.py` currently has a real OpenRouter API key committed in
  plaintext. Running this harness adds more traffic on that key — rotate it
  (and stop committing it) before this repository is shared or submitted.
- Agent routing (`action`: search / list_documents / summarize_document /
  answer_directly) is captured in each result's metadata for the demo
  drill-down, but isn't scored here — the golden set only covers RAG answer
  quality, matching the Preliminary Report's own evaluation scope.
- Nemotron 3 Ultra's generation calls (not the judge) failed ~33% of the
  time in one real run with `"returned no completion choices"` — see
  "Judge model history" above. Treat its row's results as noisier than the
  other three candidates' until this is either resolved or replaced. (A
  later full run, in "Results" below, saw this rate climb to 58% (7/12) —
  the failure is not a one-off; it's a standing reliability problem with
  this candidate on its current free-tier route.)

## Results (2026-09-05 full run)

The `--full` dataset (12 questions × 5 providers = 60 cases) was run
end-to-end in 3h40m (02:21–06:01 UTC). Raw output archived at
`archive/2026-09-05/results_full.{json,csv,html}`.

### Model comparison (same retrieval, different generation model)

| Provider | n scored | Errors | Faithfulness | Response Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|
| **Local Qwen 3.5 4B** | 12 | 0 | **0.97** | 0.75 | 0.60 | 0.83 |
| Dots3-Note Preview | 11* | 0 | 0.73 | **0.83** | 0.71 | 0.82 |
| MiniMax M3 | 12 | 0 | 0.79 | 0.68 | 0.69 | 0.83 |
| Nemotron 3 Ultra | 5** | 7 | 0.83 | 0.82 | 0.90 | 0.80 |

\* One Dots3-Note case (the `multi_hop` reranker question) came back with
**zero retrieved chunks** and a "no evidence" refusal — not a harness error,
but RAGAS can't score an empty context, so it's excluded rather than
counted as a 0. This looks like an agent-routing quirk specific to that
model's tool-calling behaviour on that question, not a retrieval-pipeline
bug (every other provider retrieved chunks fine for the same question) —
worth a manual look if it recurs.

\** Nemotron's 7/12 errors (all `"returned no completion choices"`, see
caveats above) mean its column is the *least* trustworthy in this table —
treat its apparently-best Context Precision (0.90) skeptically; it's an
average over less than half the intended sample.

**What this shows:**
- **Local Qwen is the most faithful (0.97) and had zero errors** — for a
  tool where hallucination is the worst failure mode, and where running
  locally means no dependency on a third-party free tier's uptime, it's the
  strongest all-around candidate of the four.
- **Dots3-Note scored highest on Response Relevancy (0.83)** but is also
  the least faithful (0.73) — it appears to write answers that address the
  question well even when it's less strict about staying grounded in the
  retrieved evidence than Qwen is.
- **Context Precision and Context Recall vary across all four models
  (0.60–0.90 / 0.80–0.83) despite every full-pipeline provider retrieving
  from the same knowledge base with the same retrieval code.** Since
  retrieval doesn't change with the generation model, this spread is judge
  noise, not a real retrieval difference — concrete evidence for the
  "validate the judge by hand" caveat above, not just a theoretical risk.

### Reranker ablation (same model — Qwen — retrieval varied)

| Metric | Full pipeline (reranker on) | Vector-only (no reranker) | Δ |
|---|---|---|---|
| Faithfulness | 0.97 | 0.90 | +0.07 |
| Response Relevancy | 0.75 | 0.73 | +0.02 |
| Context Precision | 0.60 | 0.69 | **−0.09** |
| Context Recall | 0.83 | 0.83 | 0.00 |

Context Recall being **identical to two decimal places in every one of the
4 categories** (not just in aggregate) is a clean, repeated confirmation of
`tests.yaml`'s own `multi_hop` hypothesis: reranking never changes which
chunks make it into the final candidate set from this vector store, only
their order.

Context Precision, however, is **not** uniformly better with the reranker
on — it depends heavily on category:

| Category | Full precision | Vector-only precision | Δ |
|---|---|---|---|
| factual_lookup | 1.00 | 0.78 | **+0.22** (reranker helps) |
| multi_hop | 0.92 | 0.92 | 0.00 |
| table_cell | 0.12 | 0.25 | −0.13 (reranker hurts) |
| named_section | 0.36 | 0.82 | **−0.46** (reranker hurts a lot) |

So the reranker is not a strict win on retrieval precision — on this test
set it actively demotes relevant chunks for `named_section` lookups, while
clearly helping `factual_lookup`. It's a wash-to-negative on precision
overall, yet **Faithfulness still favours the reranker on (+0.07)**,
concentrated in `table_cell` (0.88 vs 0.60) — meaning better *ordering* of
an equally-noisy candidate set still measurably helps the LLM write a more
grounded answer, even when it doesn't shrink the noise itself. That's a
more nuanced story than "the reranker improves retrieval" — it's closer to
"the reranker improves what the generator does with retrieval, independent
of precision."

### `table_cell` is the weakest category across every provider

Every single full-pipeline provider scored its lowest Context
Precision/Recall on `table_cell` questions (0.12–0.33), well below the
other three categories (mostly 0.75–1.00). Looking at Qwen's raw scores:
2 of its 3 `table_cell` questions came back with **Context Precision =
Context Recall = 0.0** — the pipeline retrieved chunks with essentially no
overlap with the actual table row being asked about. This isn't
judge noise (0.0 is a floor, not a wobble) — it's a real retrieval
limitation: this pipeline's chunking/embedding struggles specifically with
row/column table content, exactly the failure mode `table_cell` was
designed to probe (see the category table above). Worth calling out
explicitly in the report as a known limitation, not just a low number.

### Bottom line / recommendation

- **Best overall candidate: Local Qwen 3.5 4B (full pipeline)** — highest
  Faithfulness, zero errors, runs locally with no rate-limit or upstream
  dependency. It is also this app's own default model, so this result
  validates the existing default rather than suggesting a change.
- **Keep the reranker on.** Despite hurting Context Precision on 2 of 4
  categories, it never hurts Recall and it improves both generation-facing
  metrics (Faithfulness, Response Relevancy) in aggregate — the two metrics
  closer to what a user actually experiences as answer quality.
- **Do not trust Nemotron 3 Ultra's numbers as directly comparable** — a
  58% error rate this run (up from 33% previously) means its column is
  built from an unrepresentative subset of questions, not the full 12.
  Consider dropping it as a full-pipeline candidate, or replacing it with a
  more reliable free-tier model, before citing its scores in a final report.
- **Flag `table_cell` performance as a known limitation**, not a
  scoring artifact — 2 of Qwen's 3 lowest scores in the entire run were a
  hard 0.0/0.0, indicating genuine missed retrieval on tabular content
  rather than judge disagreement.

## Example commands

```
# Sanity check: 1 question, 1 provider only
../.venv/Scripts/python.exe run_eval.py --quick --first-n 1 --only-provider "Local Qwen 3.5 4B (full pipeline)"

# Quick full pass (4 questions x 5 providers)
../.venv/Scripts/python.exe run_eval.py --quick

# Full pass, all 12 questions x 5 providers
../.venv/Scripts/python.exe run_eval.py --full

# Re-score only the cells that came back as judge errors
../.venv/Scripts/python.exe rescore_errors.py --quick
```

Provider labels for `--only-provider` (must match exactly): `Local Qwen 3.5
4B (full pipeline)`, `Dots3-Note Preview (full pipeline)`, `Ling 3.0 Flash
Fin (full pipeline)`, `Local Qwen 3.5 4B (vector-only, no reranker)`.
