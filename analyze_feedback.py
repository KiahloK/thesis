#!/usr/bin/env python3
"""
analyze_feedback.py  —  how good was the endpoint-check feedback, and did refinement follow it?

For every query of the given run(s), compares the endpoint check's findings ('missing',
'additional', 'necessary' in query.json) with the ground truth and with the refined code, and
classifies every endpoint still missing / still extra after refinement. Pass all chunk
directories of one configuration to get pooled numbers. No LLM calls.

Usage (run from prototype/):
    python analyze_feedback.py <output_dir>... [--per-query]
"""
import collections
import json
import logging
import sys
from pathlib import Path

from evaluate import _matches_template, _normalize_endpoint, base_paths_for_files

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


def _hit(endpoint: str, endpoints, base_paths=()) -> bool:
    """True if `endpoint` matches any of `endpoints` (templates or concrete, either direction)."""
    e = _normalize_endpoint(endpoint, base_paths)
    return any(
        _matches_template(_normalize_endpoint(x, base_paths), e) or _matches_template(e, _normalize_endpoint(x, base_paths))
        for x in endpoints
    )


def analyze(run_dirs: list[Path], per_query: bool) -> None:
    c = collections.Counter()
    rows = []
    for query_json in sorted(q for d in run_dirs for q in d.rglob("query.json")):
        try:
            data = json.loads(query_json.read_text(encoding="utf-8"))
        except ValueError:
            print(f"unreadable, skipped: {query_json}")
            continue
        if not data.get("refined_metrics"):
            continue
        bp = base_paths_for_files(data.get("services", []), BENCHMARK_DIR)
        expected = set(data["query"]["endpoints"])
        initial, refined = data["initial_metrics"], data["refined_metrics"]
        ref = set(refined["extracted"])
        issues = data.get("endpoint_issues") or {}
        missing, additional = issues.get("missing", []), issues.get("additional", [])
        necessary = set(issues.get("necessary", []))
        filtered = set(data.get("filtered_endpoints", []))

        c["queries"] += 1
        c["refined better"] += refined["f1"] > initial["f1"] + 1e-9
        c["refined worse"] += refined["f1"] < initial["f1"] - 1e-9
        c["initial F1 (sum)"] += initial["f1"]
        c["refined F1 (sum)"] += refined["f1"]
        if "steps" in issues:
            c["steps (sum)"] += len(issues["steps"])
            c["steps without endpoint"] += sum(1 for s in issues["steps"] if not s.get("endpoint"))
        c["expected endpoints (sum)"] += len(expected)

        c["necessary judgements"] += len(necessary)
        c["  necessary correct (expected)"] += sum(_hit(e, expected, bp) for e in necessary)
        c["'missing' flags"] += len(missing)
        c["  'missing' correct"] += sum(_hit(e, expected, bp) for e in missing)
        c["  'missing' correct AND added by refinement"] += sum(_hit(e, expected, bp) and _hit(e, ref, bp) for e in missing)
        c["  'missing' wrong AND added anyway"] += sum((not _hit(e, expected, bp)) and _hit(e, ref, bp) for e in missing)
        c["'additional' flags"] += len(additional)
        c["  'additional' correct (not expected)"] += sum(not _hit(e, expected, bp) for e in additional)
        c["  'additional' removed by refinement"] += sum(not _hit(e, ref, bp) for e in additional)
        c["  'additional' wrong (expected) AND removed"] += sum(_hit(e, expected, bp) and not _hit(e, ref, bp) for e in additional)

        still_missing = [e for e in expected if not _hit(e, ref, bp)]
        still_extra = [e for e in ref if not _hit(e, expected, bp)]
        for e in still_missing:
            if filtered and not _hit(e, filtered, bp):
                c["FN: removed by the filter"] += 1
            elif _hit(e, missing, bp):
                c["FN: flagged 'missing', not added"] += 1
            elif _hit(e, necessary, bp):
                c["FN: judged necessary, not flagged missing"] += 1
            else:
                c["FN: not flagged at all"] += 1
        for e in still_extra:
            if _hit(e, additional, bp):
                c["FP: flagged 'additional', kept"] += 1
            elif _hit(e, necessary, bp):
                c["FP: judged NECESSARY by the check"] += 1
            elif filtered and not _hit(e, filtered, bp):
                c["FP: not in the filtered source"] += 1
            else:
                c["FP: in source, not flagged"] += 1
        rows.append((query_json.parent.relative_to(query_json.parents[2]), initial["f1"], refined["f1"], still_missing, still_extra))

    n = c["queries"] or 1
    print(f"queries: {c['queries']}   mean F1 initial {c['initial F1 (sum)'] / n:.3f} -> refined {c['refined F1 (sum)'] / n:.3f}")
    for key, value in c.items():
        if key in ("queries", "initial F1 (sum)", "refined F1 (sum)"):
            continue
        print(f"{value:>6}  {key}")
    if per_query:
        print("\nper query: initial F1 -> refined F1 | still missing | still extra")
        for q, a, b, fn, fp in rows:
            print(f"{str(q):<34} {a:.2f}->{b:.2f} | FN {sorted(fn)} | FP {sorted(fp)}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    analyze([Path(a) for a in args], "--per-query" in sys.argv)
