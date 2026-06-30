#!/usr/bin/env python3
"""
analyze_failures.py  —  failure breakdown for a benchmark output directory

Usage (run from prototype/):
    python analyze_failures.py <output_dir>          # refined metrics
    python analyze_failures.py <output_dir> --initial # baseline metrics

Examples:
    python analyze_failures.py output/2026-06-28_17-03-48_refined_from_2026-06-28_16-59-14_baseline
    python analyze_failures.py output/2026-06-28_16-59-14_baseline --initial
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Failure categorisation
# ---------------------------------------------------------------------------

def _normalize_template(endpoint: str) -> str:
    return re.sub(r'\{[^}]+\}', '{param}', endpoint)


def _normalize_concrete(endpoint: str) -> str:
    """Replace path segments that look like concrete IDs with {param}."""
    parts = endpoint.split(' ', 1)
    if len(parts) != 2:
        return endpoint
    method, path = parts
    out = []
    for seg in path.split('/'):
        if seg and not seg.startswith('{') and (
            any(c.isdigit() for c in seg) or
            (re.match(r'^[a-zA-Z0-9]{6,}$', seg) and not re.match(r'^[a-z\-]+$', seg))
        ):
            out.append('{param}')
        else:
            out.append(seg)
    return method + ' ' + '/'.join(out)


def categorise(expected: set[str], extracted: set[str]) -> dict:
    """Return counts and labelled lists for each failure category."""
    missing = expected - extracted
    extra = extracted - expected

    matched_missing: set[str] = set()
    matched_extra: set[str] = set()
    path_param: list[tuple[str, str]] = []
    empty_param: list[str] = []

    # 1. Format failure: no endpoints extracted at all
    if not extracted and missing:
        return {
            'path_param': [],
            'empty_param': [],
            'format_fail': sorted(missing),
            'genuine_missing': [],
            'hallucinated': [],
        }

    # 2. Path-param pairs
    for miss in missing:
        norm_miss = _normalize_template(miss)
        for ext in extra:
            if ext in matched_extra:
                continue
            if norm_miss == _normalize_concrete(ext):
                path_param.append((miss, ext))
                matched_missing.add(miss)
                matched_extra.add(ext)
                break

    # 3. Empty path params (//)
    for ext in extra:
        if ext in matched_extra:
            continue
        if '//' in ext or (ext.endswith('/') and len(ext.split('/')) > 2):
            empty_param.append(ext)
            matched_extra.add(ext)

    genuine_missing = sorted(missing - matched_missing)
    hallucinated = sorted(extra - matched_extra)

    return {
        'path_param':      path_param,
        'empty_param':     empty_param,
        'format_fail':     [],
        'genuine_missing': genuine_missing,
        'hallucinated':    hallucinated,
    }


# ---------------------------------------------------------------------------
# Re-evaluation with template-aware matching
# (mirrors evaluate.py logic without importing it, for portability)
# ---------------------------------------------------------------------------

def _normalize_ep(ep: str) -> str:
    return re.sub(r'^(\w+ )/v\d+/', r'\1/', ep)


def _matches_template(extracted_ep: str, expected_ep: str) -> bool:
    e_parts = extracted_ep.split(' ', 1)
    t_parts = expected_ep.split(' ', 1)
    if len(e_parts) != 2 or len(t_parts) != 2:
        return extracted_ep == expected_ep
    e_method, e_path = e_parts
    t_method, t_path = t_parts
    if e_method != t_method:
        return False
    e_segs = e_path.split('/')
    t_segs = t_path.split('/')
    if len(e_segs) != len(t_segs):
        return False
    for e_seg, t_seg in zip(e_segs, t_segs):
        if t_seg.startswith('{') and t_seg.endswith('}'):
            if not e_seg:
                return False
            continue
        if e_seg != t_seg:
            return False
    return True


def compute_metrics(extracted_raw: set[str], expected_raw: set[str]) -> dict:
    norm_ext = {_normalize_ep(e) for e in extracted_raw}
    norm_exp = {_normalize_ep(e) for e in expected_raw}
    matched: set[str] = set()
    tp = 0
    for exp in norm_exp:
        for ext in norm_ext:
            if ext not in matched and _matches_template(ext, exp):
                tp += 1
                matched.add(ext)
                break
    p = tp / len(norm_ext) if norm_ext else 0.0
    r = tp / len(norm_exp) if norm_exp else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'precision': p, 'recall': r, 'f1': f1,
            'extracted': norm_ext, 'expected': norm_exp}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(out_dir: Path, use_initial: bool) -> list[dict]:
    rows = []
    metrics_key = 'initial_metrics' if use_initial else 'refined_metrics'
    code_file   = 'generated.py'   if use_initial else 'refined.py'

    for qj in sorted(out_dir.rglob('query.json')):
        data = json.loads(qj.read_text())
        expected = set(data['query'].get('endpoints', []))
        sector   = data.get('sector_name', qj.parent.parent.name)
        qidx     = data.get('query_index', 0)

        # Re-evaluate from code if available, else fall back to stored metrics
        code_path = qj.parent / code_file
        if code_path.exists():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from evaluate import evaluate
                m = evaluate(code_path.read_text(), list(expected))
                extracted = set(m.get('extracted', set()))
            except Exception:
                stored = data.get(metrics_key, {})
                extracted = set(stored.get('extracted', set()))
        else:
            stored = data.get(metrics_key, {})
            extracted = set(stored.get('extracted', set()))

        metrics = compute_metrics(extracted, expected)
        cats    = categorise(metrics['expected'], metrics['extracted'])
        rows.append({
            'sector': sector,
            'query_index': qidx,
            'query': data['query'].get('query', '')[:90],
            'metrics': metrics,
            'cats': cats,
        })

    rows.sort(key=lambda r: r['metrics']['f1'])
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEP = '─' * 80

def print_per_query(rows: list[dict]) -> None:
    print(f'\n{"QUERY BREAKDOWN (worst first)":^80}')
    print(SEP)
    for r in rows:
        m    = r['metrics']
        cats = r['cats']
        f1   = m['f1']
        marker = '✗' if f1 < 1.0 else '✓'
        print(
            f"{marker} [{r['sector']}] Q{r['query_index']:02d}"
            f"  F1={f1:.2f}  P={m['precision']:.2f}  R={m['recall']:.2f}"
        )
        print(f"   Q: {r['query']}")
        if cats['format_fail']:
            print(f"   FORMAT FAIL  — code unparseable, {len(cats['format_fail'])} missing")
        for miss, ext in cats['path_param']:
            print(f"   PATH PARAM   expected {miss!r}  got {ext!r}")
        for ep in cats['empty_param']:
            print(f"   EMPTY PARAM  {ep!r}")
        for ep in cats['genuine_missing']:
            print(f"   MISSING      {ep!r}")
        for ep in cats['hallucinated']:
            print(f"   EXTRA        {ep!r}")
        print()


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    correct = sum(1 for r in rows if r['metrics']['f1'] == 1.0)

    counts = {k: 0 for k in ('path_param', 'empty_param', 'format_fail', 'genuine_missing', 'hallucinated')}
    for r in rows:
        cats = r['cats']
        counts['path_param']      += len(cats['path_param'])
        counts['empty_param']     += len(cats['empty_param'])
        counts['format_fail']     += len(cats['format_fail'])
        counts['genuine_missing'] += len(cats['genuine_missing'])
        counts['hallucinated']    += len(cats['hallucinated'])

    avg_f1  = sum(r['metrics']['f1']        for r in rows) / total
    avg_p   = sum(r['metrics']['precision']  for r in rows) / total
    avg_r   = sum(r['metrics']['recall']     for r in rows) / total

    total_missing = counts['path_param'] + counts['format_fail'] + counts['genuine_missing']
    total_extra   = counts['path_param'] + counts['empty_param'] + counts['hallucinated']

    print(f'\n{"SUMMARY":^80}')
    print(SEP)
    print(f"  Queries         : {total}")
    print(f"  Correct (F1=1)  : {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Avg F1          : {avg_f1:.3f}   Precision: {avg_p:.3f}   Recall: {avg_r:.3f}")
    print()
    print(f"  {'Category':<24} {'Slots':>6}  {'of missing':>10}  {'of extra':>10}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*10}  {'─'*10}")

    def pct(n, denom):
        return f"{100*n/denom:.0f}%" if denom else "—"

    print(f"  {'path_param (eval bug)':<24} {counts['path_param']:>6}  "
          f"{pct(counts['path_param'], total_missing):>10}  "
          f"{pct(counts['path_param'], total_extra):>10}")
    print(f"  {'empty_param':<24} {counts['empty_param']:>6}  "
          f"{'—':>10}  {pct(counts['empty_param'], total_extra):>10}")
    print(f"  {'format_fail':<24} {counts['format_fail']:>6}  "
          f"{pct(counts['format_fail'], total_missing):>10}  {'—':>10}")
    print(f"  {'genuine_missing':<24} {counts['genuine_missing']:>6}  "
          f"{pct(counts['genuine_missing'], total_missing):>10}  {'—':>10}")
    print(f"  {'hallucinated (extra)':<24} {counts['hallucinated']:>6}  "
          f"{'—':>10}  {pct(counts['hallucinated'], total_extra):>10}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*10}  {'─'*10}")
    print(f"  {'Total missing slots':<24} {total_missing:>6}")
    print(f"  {'Total extra slots':<24} {total_extra:>6}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    flags = [a for a in sys.argv[1:] if a.startswith('-')]
    use_initial = '--initial' in flags

    if not args:
        print(__doc__)
        sys.exit(1)

    out_dir = Path(args[0])
    if not out_dir.exists():
        # Try relative to script location (prototype/)
        out_dir = Path(__file__).parent / args[0]
    if not out_dir.exists():
        print(f"Error: directory not found: {args[0]}")
        sys.exit(1)

    mode = 'INITIAL (baseline)' if use_initial else 'REFINED'
    print(f'\n{SEP}')
    print(f'  Failure analysis  —  {mode}')
    print(f'  Run: {out_dir.name}')
    print(SEP)

    rows = load_results(out_dir, use_initial)
    print_per_query(rows)
    print_summary(rows)


if __name__ == '__main__':
    main()
