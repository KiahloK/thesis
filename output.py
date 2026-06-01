import os
import json
from datetime import datetime

def make_output_dir(name: str) -> str:
    base = "output"
    os.makedirs(base, exist_ok=True)
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    path = os.path.join(base, safe_name)
    os.makedirs(path, exist_ok=True)
    return path


def write_query_output(output_dir: str, result: dict) -> None:
    idx = result.get('query_index') or 0
    query_dir = os.path.join(output_dir, f"query_{int(idx):03d}")
    os.makedirs(query_dir, exist_ok=True)

    fname = os.path.join(query_dir, "query.json")
    # Write generated code to separate files so the code is readable with correct newlines
    gen = result.get('generated')
    gen_ref = result.get('generated_refined')

    gen_file = None
    gen_ref_file = None

    if gen:
        gen_file = os.path.join(query_dir, "generated.py")
        with open(gen_file, 'w', encoding='utf-8') as gf:
            gf.write(gen)

    if gen_ref:
        gen_ref_file = os.path.join(query_dir, "refined.py")
        with open(gen_ref_file, 'w', encoding='utf-8') as gf:
            gf.write(gen_ref)

    analysis = result.get('analysis')
    analysis_file = None
    if analysis:
        analysis_file = os.path.join(query_dir, "analysis.txt")
        with open(analysis_file, 'w', encoding='utf-8') as af:
            af.write(analysis)

    data = {
        'query_index': result.get('query_index'),
        'query': result.get('query'),
        'services': result.get('service_files') or result.get('services'),
        'model': result.get('model'),
        # do not include the prompt text (can be large); store filenames for code instead
        'generated_file': os.path.basename(gen_file) if gen_file else None,
        'generated_refined_file': os.path.basename(gen_ref_file) if gen_ref_file else None,
        'analysis_file': os.path.basename(analysis_file) if analysis_file else None,
        'initial_metrics': _serialize_metrics(result.get('initial_metrics')),
        'refined_metrics': _serialize_metrics(result.get('refined_metrics'))
    }

    # Ensure everything is JSON-serializable (convert sets to lists, Paths to strings, etc.)
    serializable = _serialize_for_json(data)
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _serialize_metrics(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    out = dict(metrics)
    # Convert sets to sorted lists for JSON
    if 'extracted' in out and isinstance(out['extracted'], (set, list, tuple)):
        out['extracted'] = sorted(list(out['extracted']))
    if 'expected' in out and isinstance(out['expected'], (set, list, tuple)):
        out['expected'] = sorted(list(out['expected']))
    return out


def write_overall_summary(output_dir: str, sector_results: list) -> None:
    fname = os.path.join(output_dir, "summary.json")
    initial_metrics = [r.get('initial_metrics') for r in sector_results if r.get('initial_metrics')]
    refined_metrics = [r.get('refined_metrics') for r in sector_results if r.get('refined_metrics')]

    def avg(lst, key):
        vals = [m.get(key) for m in lst if m and (m.get(key) is not None)]
        return sum(vals)/len(vals) if vals else None

    summary = {
        'initial_count': len(initial_metrics),
        'refined_count': len(refined_metrics),
        'initial_avg_precision': avg(initial_metrics, 'precision'),
        'initial_avg_recall': avg(initial_metrics, 'recall'),
        'initial_avg_f1': avg(initial_metrics, 'f1'),
        'refined_avg_precision': avg(refined_metrics, 'precision'),
        'refined_avg_recall': avg(refined_metrics, 'recall'),
        'refined_avg_f1': avg(refined_metrics, 'f1'),
        'per_query': [
            {
                'query_index': r.get('query_index'),
                'initial': _serialize_metrics(r.get('initial_metrics')),
                'refined': _serialize_metrics(r.get('refined_metrics'))
            }
            for r in sector_results
        ]
    }

    serializable = _serialize_for_json(summary)
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _serialize_for_json(obj):
    """Recursively convert non-JSON-native types (sets, tuples, Paths) to JSON-friendly forms."""
    # Import locally to avoid top-level import cycles in different environments
    from pathlib import Path

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_serialize_for_json(v) for v in obj])
    if isinstance(obj, Path):
        return str(obj)
    # Fallback: return as-is (json.dump will raise if still not serializable)
    return obj
