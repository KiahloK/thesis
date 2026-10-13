#!/usr/bin/env python3
"""
rescore_runs.py  —  re-evaluate stored runs with the current evaluate.py

evaluate.py has changed over time (template matching, empty path segments, server base
paths), so metrics stored in older runs' summary.json are not comparable with newer runs.
This re-evaluates every stored generated.py / refined.py against the query's expected
endpoints - no LLM calls - and writes summary_rescored.json next to summary.json. Stored
files are never modified.

Usage (run from prototype/):
    python rescore_runs.py                        # every run under output/
    python rescore_runs.py <output_dir>...        # selected runs
    python rescore_runs.py --pool <output_dir>... # one combined result over several runs, e.g. the
                                                  # chunks of one configuration (writes nothing)
"""
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from evaluate import base_paths_for_files, evaluate

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _averages(metrics: list[dict]) -> dict | None:
    if not metrics:
        return None
    n = len(metrics)
    return {
        "count": n,
        "precision": sum(m["precision"] for m in metrics) / n,
        "recall": sum(m["recall"] for m in metrics) / n,
        "f1": sum(m["f1"] for m in metrics) / n,
        "avg_missing": sum(len(m["expected"] - m["extracted"]) for m in metrics) / n,
        "avg_extra": sum(len(m["extracted"] - m["expected"]) for m in metrics) / n,
    }


def rescore(run_dir: Path) -> dict:
    initial, refined, per_query, broken = [], [], [], []
    for query_json in sorted(run_dir.rglob("query.json")):
        try:
            data = json.loads(query_json.read_text(encoding="utf-8"))
        except ValueError:
            broken.append(str(query_json.relative_to(run_dir)))
            continue
        expected = data["query"].get("endpoints", [])
        base_paths = base_paths_for_files(data.get("services", []), BENCHMARK_DIR)
        row = {"query": str(query_json.parent.relative_to(run_dir))}
        for code_file, bucket, key in (("generated.py", initial, "initial"), ("refined.py", refined, "refined")):
            code_path = query_json.parent / code_file
            if code_path.exists():
                metrics = evaluate(code_path.read_text(encoding="utf-8"), expected, base_paths)
                bucket.append(metrics)
                row[key] = {k: metrics[k] for k in ("precision", "recall", "f1")}
        per_query.append(row)

    return {
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
        "evaluate_commit": _git_commit(),
        "initial": _averages(initial),
        "refined": _averages(refined),
        "unreadable_query_files": broken,
        "per_query": per_query,
    }


def pool(run_dirs: list[Path]) -> None:
    """Print precision/recall/F1 averaged over the queries of all given runs together."""
    rows = [row for run_dir in run_dirs for row in rescore(run_dir)["per_query"]]
    queries = {row["query"] for row in rows}
    if len(queries) != len(rows):
        print(f"WARNING: {len(rows) - len(queries)} queries appear in more than one run (overlapping chunks?)")
    for key in ("initial", "refined"):
        metrics = [row[key] for row in rows if key in row]
        if metrics:
            n = len(metrics)
            print(f"{key:<8} n={n:<3} " + "  ".join(
                f"{m}={sum(x[m] for x in metrics) / n:.3f}" for m in ("precision", "recall", "f1")
            ))


def main() -> None:
    logging.disable(logging.WARNING)  # socbenchsc warns about every unsupported construct
    args = sys.argv[1:]
    if args[:1] == ["--pool"]:
        pool([Path(a) for a in args[1:]])
        return
    run_dirs = [Path(a) for a in args] or sorted(
        p for p in (Path(__file__).parent / "output").iterdir() if p.is_dir()
    )
    for run_dir in run_dirs:
        result = rescore(run_dir)
        (run_dir / "summary_rescored.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        stored = {}
        try:
            stored = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        line = f"{run_dir.name:<64}"
        if result["initial"]:
            line += f" initial F1 {stored.get('avg_f1', float('nan')):.3f} -> {result['initial']['f1']:.3f}"
        if result["refined"]:
            line += f" | refined F1 {stored.get('refined_avg_f1', float('nan')):.3f} -> {result['refined']['f1']:.3f}"
        print(line)
        for broken in result["unreadable_query_files"]:
            print(f"    unreadable, skipped: {broken}")


if __name__ == "__main__":
    main()
