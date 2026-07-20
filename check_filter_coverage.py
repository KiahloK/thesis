#!/usr/bin/env python3
"""
check_filter_coverage.py  —  did the retrieval filter (Idea 1: BM25/RAG) drop an
endpoint the query actually needed, before the model ever got to see it?

If a ground-truth expected endpoint isn't in the filtered service specs the model
was shown, no amount of generation/refinement could ever get it right - that's a
hard failure ceiling imposed by the filter step itself, worth telling apart from
a model mistake.

Usage (run from prototype/):
    python check_filter_coverage.py <output_dir>

Uses each query's stored 'filtered_endpoints' field (query.json) when present.
For older runs saved before that field existed, it's recomputed on the fly from
that run's filter config (summary.json) and the original service spec files -
this is a pure local computation (BM25) or a local embedding model call (RAG),
no LLM/network calls, so it's fast and free either way.
"""

import json
import re
import sys
from pathlib import Path


def _normalize_endpoint(ep: str) -> str:
    return re.sub(r'^(\w+ )/v\d+/', r'\1/', ep)


def _endpoints_from_services(services: list[str]) -> set[str]:
    from endpoint_check import _all_endpoints, _parse_services
    return set(_all_endpoints(_parse_services(services)))


def _recompute_filtered(data: dict, config: dict, benchmark_dir: Path) -> set[str]:
    service_files = data.get('services') or []
    services = [
        (benchmark_dir / sf).read_text(encoding='utf-8')
        for sf in service_files
        if (benchmark_dir / sf).exists()
    ]
    if not config.get('filter_services'):
        return _endpoints_from_services(services)

    query_text = data['query'].get('query', '')
    top_k = config.get('top_k') or 5
    if config.get('filter_mode') == 'rag':
        from rag import filter_services_rag
        filtered = filter_services_rag(services, query_text, top_k=top_k)
    else:
        from filter import filter_services
        filtered = filter_services(services, query_text, top_k=top_k)
    return _endpoints_from_services(filtered)


def check_run(out_dir: Path) -> list[dict]:
    summary_path = out_dir / "summary.json"
    config = json.loads(summary_path.read_text()).get("config", {}) if summary_path.exists() else {}
    benchmark_dir = Path(config.get("benchmark_dir", "./benchmark"))

    rows = []
    for qj in sorted(out_dir.rglob("query.json")):
        data = json.loads(qj.read_text())
        expected = set(data["query"].get("endpoints", []))
        if not expected:
            continue

        if "filtered_endpoints" in data:
            filtered = set(data["filtered_endpoints"])
        else:
            filtered = _recompute_filtered(data, config, benchmark_dir)

        norm_filtered = {_normalize_endpoint(e) for e in filtered}
        missing = sorted(e for e in expected if _normalize_endpoint(e) not in norm_filtered)

        rows.append({
            'sector': data.get('sector_name', qj.parent.parent.name),
            'query_index': data.get('query_index', 0),
            'query': data['query'].get('query', '')[:90],
            'expected_count': len(expected),
            'filtered_count': len(filtered),
            'missing': missing,
        })

    rows.sort(key=lambda r: -len(r['missing']))
    return rows


SEP = '─' * 80


def print_report(rows: list[dict], out_dir: Path) -> None:
    affected = [r for r in rows if r['missing']]

    print(f'\n{SEP}')
    print('  Filter coverage check (Idea 1: BM25/RAG)')
    print(f'  Run: {out_dir.name}')
    print(SEP)

    if not affected:
        print('\n  No queries lost an expected endpoint to filtering. ✓')
    for r in affected:
        print(
            f"\n✗ [{r['sector']}] Q{r['query_index']:02d}"
            f"  ({len(r['missing'])}/{r['expected_count']} expected endpoints filtered out,"
            f" {r['filtered_count']} endpoints kept)"
        )
        print(f"   Q: {r['query']}")
        for ep in r['missing']:
            print(f"   FILTERED OUT  {ep!r}")

    total = len(rows)
    print(f'\n{"SUMMARY":^80}')
    print(SEP)
    print(f"  Queries checked   : {total}")
    if total:
        print(f"  Queries affected  : {len(affected)} ({100 * len(affected) / total:.1f}%)")
    total_missing = sum(len(r['missing']) for r in rows)
    print(f"  Total lost slots  : {total_missing}")
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

    rows = check_run(out_dir)
    print_report(rows, out_dir)


if __name__ == "__main__":
    main()
