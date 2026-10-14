#!/usr/bin/env python3
"""
benchmark_audit.py  —  how much of the measured error is benchmark ambiguity?

Re-scores stored runs twice: strictly (as evaluate.py, against the benchmark's expected endpoints)
and leniently, using the ground-truth audit in benchmark_audit_socbenchd1_dev.json:
  - alternatives:     an endpoint that fits the query as well as the expected one counts as correct,
  - optional:         an expected endpoint the query only implies counts if called, is not missing if not,
  - justified_extras: an endpoint the query clearly asks for but the benchmark does not expect is not
                      counted as a false positive.
This is a sensitivity analysis of the benchmark. It is never used inside the pipeline or for tuning.

Usage (run from prototype/):
    python benchmark_audit.py LABEL=<run_dir>[,<run_dir>...] [LABEL=...] [--initial] [--per-query]
`--initial` scores the initial code (generated.py) instead of the refined code. Runs loaded from a
baseline have no generated.py of their own; score the baseline dirs with --initial instead.
"""
import json
import logging
import sys
from pathlib import Path

from evaluate import _matches_template, _normalize_endpoint, base_paths_for_files, evaluate

HERE = Path(__file__).parent
AUDIT = json.loads((HERE / "benchmark_audit_socbenchd1_dev.json").read_text())["queries"]


def _matches(extracted: str, expected: str) -> bool:
    return _matches_template(extracted, _normalize_endpoint(expected))


def lenient(extracted: set[str], expected: list[str], audit: dict) -> tuple[float, float, float]:
    alternatives = audit.get("alternatives", {})
    optional = set(audit.get("optional", []))
    justified = audit.get("justified_extras", [])
    used: set[str] = set()
    tp = denominator_recall = 0
    for e in expected:
        candidates = [e] + alternatives.get(e, [])
        match = next((x for x in extracted if x not in used and any(_matches(x, c) for c in candidates)), None)
        if match:
            used.add(match)
            tp += 1
            denominator_recall += 1
        elif e not in optional:
            denominator_recall += 1
    unexplained = {x for x in extracted - used if not any(_matches(x, j) for j in justified)}
    denominator_precision = len(used) + len(unexplained)
    p = tp / denominator_precision if denominator_precision else 0.0
    r = tp / denominator_recall if denominator_recall else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(run_dirs: list[Path], initial: bool, per_query: bool, label: str) -> None:
    code_file = "generated.py" if initial else "refined.py"
    rows = []
    for d in run_dirs:
        for qj in sorted(d.rglob("query.json")):
            code = qj.parent / code_file
            if not code.exists():
                continue
            q = json.loads(qj.read_text())
            key = f"{qj.parent.parent.name}/{qj.parent.name}"
            expected = q["query"]["endpoints"]
            m = evaluate(code.read_text(), expected, base_paths_for_files(q.get("services", []), HERE / "benchmark"))
            rows.append((key, (m["precision"], m["recall"], m["f1"]), lenient(set(m["extracted"]), expected, AUDIT.get(key, {}))))
    if not rows:
        print(f"{label}: no {code_file} found")
        return
    n = len(rows)
    strict = [sum(r[1][i] for r in rows) / n for i in range(3)]
    soft = [sum(r[2][i] for r in rows) / n for i in range(3)]
    affected = sum(1 for r in rows if r[0] in AUDIT)
    print(f"{label:<28} n={n:<3} strict P/R/F1 {strict[0]:.3f}/{strict[1]:.3f}/{strict[2]:.3f}   "
          f"lenient {soft[0]:.3f}/{soft[1]:.3f}/{soft[2]:.3f}   ΔF1 {soft[2] - strict[2]:+.3f}   ({affected} audited queries)")
    if per_query:
        for key, strict_q, lenient_q in rows:
            if abs(strict_q[2] - lenient_q[2]) > 1e-9:
                print(f"    {key:<26} F1 {strict_q[2]:.2f} -> {lenient_q[2]:.2f}")


def main() -> None:
    logging.disable(logging.WARNING)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    for arg in args:
        label, _, dirs = arg.partition("=")
        score([Path(d) for d in dirs.split(",")], "--initial" in sys.argv, "--per-query" in sys.argv, label)


if __name__ == "__main__":
    main()
