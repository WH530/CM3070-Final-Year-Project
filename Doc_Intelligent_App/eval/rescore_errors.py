"""Re-score just the RAGAS cells that failed in a previous run_eval.py run.

Rather than re-running a whole eval (which re-does retrieval + generation
for every case, burning OpenRouter rate-limit budget on cases that already
succeeded), this reuses the already-generated answer + retrieved contexts
stored in results_*.json and only re-invokes the judge for cells that came
back as an "error: ..." string. Useful after a run that mostly succeeded but
hit a few transient judge failures (a rate limit, an upstream 502, etc.).

Requires a results_*.json written by a run_eval.py that stores the
"contexts" field per row (added alongside this script) — older result files
without it are reported as unrescuable and need a full case re-run instead
(e.g. ``run_eval.py --quick --first-n 1 --only-provider "<label>"``, then
manually merge, or just re-run the whole set).

Usage (from eval/):
    ../.venv/Scripts/python.exe rescore_errors.py --quick
    ../.venv/Scripts/python.exe rescore_errors.py --full
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from run_eval import METRIC_LABELS, RESULTS_DIR, write_csv, write_html, write_json
import ragas_scorer


def _has_error(row: dict) -> bool:
    return any(
        isinstance(row.get(key), str) and row[key].startswith("error")
        for key in METRIC_LABELS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    dataset_name = "quick" if args.quick else "full"
    json_path = RESULTS_DIR / f"results_{dataset_name}.json"
    if not json_path.exists():
        print(f"No such file: {json_path}")
        return 1

    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]

    candidates = [r for r in results if "error" not in r and _has_error(r)]
    unrescuable = [r for r in candidates if not r.get("contexts")]
    rescuable = [r for r in candidates if r.get("contexts")]

    if not candidates:
        print("No cells with judge errors found — nothing to rescue.")
        return 0

    print(f"Found {len(candidates)} case(s) with at least one judge error.")
    if unrescuable:
        print(
            f"  {len(unrescuable)} of those have no stored 'contexts' "
            "(from a run before this script existed) — skipping, "
            "re-run those cases directly instead:"
        )
        for r in unrescuable:
            print(f"    - {r['provider_label']!r} :: {r['question'][:60]}...")

    rescued, still_failing = 0, 0
    for i, row in enumerate(rescuable, start=1):
        print(
            f"[{i}/{len(rescuable)}] Re-scoring {row['provider_label']!r} :: "
            f"{row['question'][:60]}...",
            flush=True,
        )
        scores = ragas_scorer.score(
            question=row["question"],
            contexts=row["contexts"],
            answer=row["answer"],
            ground_truth=row["ground_truth"],
        )
        row.update(scores)
        if _has_error(row):
            still_failing += 1
            print("    -> still failing")
        else:
            rescued += 1
            summary = ", ".join(
                f"{k}={v:.2f}" for k, v in scores.items() if isinstance(v, float)
            )
            print(f"    -> {summary}")

    data["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(results, json_path, {k: v for k, v in data.items() if k != "results"})
    write_csv(results, RESULTS_DIR / f"results_{dataset_name}.csv")
    write_html(
        results,
        RESULTS_DIR / f"results_{dataset_name}.html",
        {k: v for k, v in data.items() if k != "results"},
    )

    print(
        f"\nDone. Rescued {rescued}, still failing {still_failing}, "
        f"unrescuable {len(unrescuable)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
