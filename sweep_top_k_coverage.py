#!/usr/bin/env python3
"""
sweep_top_k_coverage.py  —  sweep top_k values and report how many ground-truth
endpoints the retrieval filter (Idea 1: BM25/RAG) would drop at each value, for
every query in an existing output directory.

Motivation: filter_services()/filter_services_rag() rank endpoints globally across
all of a query's attached services (not per service). Whether a given top_k is
"safe" (loses ~0 expected endpoints) or "aggressive" (real compression, real recall
risk) depends entirely on the benchmark's service/endpoint structure - this sweeps a
range of top_k values against real queries to find that tradeoff empirically instead
of guessing. See check_filter_coverage.py for a single-top_k version of this check.

Usage (run from prototype/):
    python sweep_top_k_coverage.py <output_dir> [top_k ...] [--mode bm25|rag]

Examples:
    python sweep_top_k_coverage.py output/2026-07-20_13-36-42_refined_from_2026-07-20_12-57-54_baseline
    python sweep_top_k_coverage.py output/<run> 10 15 20 25 30 35 40 45 50 --mode rag
"""

import json
import sys
from pathlib import Path


def _load_queries(out_dir: Path, benchmark_dir: Path) -> list[tuple[str, list[str], set[str], int]]:
    from endpoint_check import _all_endpoints, _parse_services

    queries = []
    for qj in sorted(out_dir.rglob("query.json")):
        data = json.loads(qj.read_text())
        expected = set(data["query"].get("endpoints", []))
        if not expected:
            continue
        service_files = data.get("services") or []
        services = [
            (benchmark_dir / sf).read_text(encoding="utf-8")
            for sf in service_files
            if (benchmark_dir / sf).exists()
        ]
        n_unfiltered = len(_all_endpoints(_parse_services(services)))
        queries.append((data["query"].get("query", ""), services, expected, n_unfiltered))
    return queries


def sweep(out_dir: Path, top_ks: list[int], mode: str | None) -> None:
    from endpoint_check import _all_endpoints, _parse_services

    summary_path = out_dir / "summary.json"
    config = json.loads(summary_path.read_text()).get("config", {}) if summary_path.exists() else {}
    benchmark_dir = Path(config.get("benchmark_dir", "./benchmark"))
    filter_mode = mode or config.get("filter_mode") or "bm25"

    if filter_mode == "rag":
        from rag import filter_services_rag as filter_fn
    else:
        from filter import filter_services as filter_fn

    queries = _load_queries(out_dir, benchmark_dir)
    if not queries:
        print("No queries with ground-truth endpoints found.")
        return

    avg_unfiltered = sum(n for _, _, _, n in queries) / len(queries)

    print(f"\nQueries: {len(queries)}   Avg. unfiltered endpoints/query: {avg_unfiltered:.1f}   Filter mode: {filter_mode}\n")
    print(f"{'top_k':>6}  {'kept':>6}  {'affected':>10}  {'lost_slots':>10}")
    print("─" * 42)

    for top_k in top_ks:
        affected = 0
        lost_slots = 0
        for query_text, services, expected, _n_unfiltered in queries:
            filtered = filter_fn(services, query_text, top_k=top_k)
            filtered_endpoints = set(_all_endpoints(_parse_services(filtered)))
            missing = expected - filtered_endpoints
            if missing:
                affected += 1
                lost_slots += len(missing)
        kept_pct = 100 * min(top_k, avg_unfiltered) / avg_unfiltered if avg_unfiltered else 0
        print(f"{top_k:>6}  {kept_pct:>5.0f}%  {affected:>6}/{len(queries)}  {lost_slots:>10}")

    print()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    if not out_dir.exists():
        out_dir = Path(__file__).parent / sys.argv[1]
    if not out_dir.exists():
        print(f"Error: directory not found: {sys.argv[1]}")
        sys.exit(1)

    rest = sys.argv[2:]
    mode = None
    if "--mode" in rest:
        i = rest.index("--mode")
        mode = rest[i + 1]
        del rest[i:i + 2]

    top_ks = [int(a) for a in rest] or [10, 15, 20, 25, 30, 35, 40, 45, 50]
    sweep(out_dir, top_ks, mode)


if __name__ == "__main__":
    main()
