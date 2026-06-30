from socbenchsc.analysis import Analysis
from typing import Dict, Set, List
import re


def _normalize_endpoint(endpoint: str) -> str:
    """Strip version prefix (e.g. /v1/, /v2/) from endpoint paths so that
    'GET /v1/foo' and 'GET /foo' are treated as equivalent."""
    return re.sub(r'^(\w+ )/v\d+/', r'\1/', endpoint)


def _matches_template(extracted_ep: str, expected_ep: str) -> bool:
    """True if extracted_ep matches expected_ep, treating {…} path segments as wildcards.

    This handles the case where the model correctly calls e.g. GET /items/123 but the
    expected endpoint uses template notation GET /items/{id}. An empty segment never
    matches a template placeholder, so /items// does not match /items/{id}.
    """
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


def evaluate(generated_code: str, expected_endpoints: List[str]) -> Dict:
    """Evaluate generated code by extracting used endpoints and computing precision/recall/f1."""
    expected = set(expected_endpoints)

    try:
        extracted: Set[str] = Analysis(generated_code).perform_analysis()
    except SyntaxError as exc:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "extracted": set(),
            "expected": expected,
            "syntax_error": f"{exc.msg} at line {exc.lineno}, column {exc.offset}",
        }

    norm_extracted = {_normalize_endpoint(e) for e in extracted}
    norm_expected = {_normalize_endpoint(e) for e in expected}

    # Template-aware matching: each expected template can match at most one extracted endpoint.
    matched_extracted: Set[str] = set()
    tp = 0
    for exp in norm_expected:
        for ext in norm_extracted:
            if ext not in matched_extracted and _matches_template(ext, exp):
                tp += 1
                matched_extracted.add(ext)
                break

    precision = tp / len(norm_extracted) if norm_extracted else 0.0
    recall = tp / len(norm_expected) if norm_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "extracted": norm_extracted,
        "expected": norm_expected,
    }


def print_metrics(metrics: Dict, title: str = "Evaluation") -> None:
    """Print a readable evaluation summary for a metrics dictionary."""
    extracted = set(metrics.get("extracted", set()))
    expected = set(metrics.get("expected", set()))
    missing = sorted(expected - extracted)
    extra = sorted(extracted - expected)

    print(title)
    if "syntax_error" in metrics:
        print(f"  Syntax error: {metrics['syntax_error']}")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  F1:        {metrics['f1']:.2f}")
    print(f"  Extracted: {sorted(extracted)}")
    print(f"  Expected:  {sorted(expected)}")
    print(f"  Missing:   {missing if missing else '[]'}")
    print(f"  Extra:     {extra if extra else '[]'}")


def print_evaluation_summary(metrics_list: List[Dict], title: str = "Evaluation Summary") -> None:
    """Print aggregate metrics across multiple evaluation results."""
    if not metrics_list:
        print(title)
        print("  No evaluations available.")
        return

    total = len(metrics_list)
    avg_precision = sum(float(metrics.get("precision", 0.0)) for metrics in metrics_list) / total
    avg_recall = sum(float(metrics.get("recall", 0.0)) for metrics in metrics_list) / total
    avg_f1 = sum(float(metrics.get("f1", 0.0)) for metrics in metrics_list) / total
    avg_missing = sum(
        len(set(metrics.get("expected", set())) - set(metrics.get("extracted", set())))
        for metrics in metrics_list
    ) / total
    avg_extra = sum(
        len(set(metrics.get("extracted", set())) - set(metrics.get("expected", set())))
        for metrics in metrics_list
    ) / total
    correct = sum(
        1
        for metrics in metrics_list
        if not metrics.get("syntax_error")
        and set(metrics.get("extracted", set())) == set(metrics.get("expected", set()))
    )
    correct_rate = (correct / total) * 100

    print(title)
    print(f"  Average Precision: {avg_precision:.2f}")
    print(f"  Average Recall:    {avg_recall:.2f}")
    print(f"  Average F1:        {avg_f1:.2f}")
    print(f"  Avg. Missing Endpoints: {avg_missing:.2f}")
    print(f"  Avg. Extra Endpoints:   {avg_extra:.2f}")
    print(f"  Correct Compositions: {correct}/{total} ({correct_rate:.1f}%)")
