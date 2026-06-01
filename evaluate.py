from socbenchsc.analysis import Analysis
from typing import Dict, Set, List
import re


def _normalize_endpoint(endpoint: str) -> str:
    """Strip version prefix (e.g. /v1/, /v2/) from endpoint paths so that
    'GET /v1/foo' and 'GET /foo' are treated as equivalent."""
    return re.sub(r'^(\w+ )/v\d+/', r'\1/', endpoint)


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

    norm_extracted = set()
    for e in extracted:
        n = _normalize_endpoint(e)
        if n != e:
            print(f"[normalization] extracted: '{e}' -> '{n}'")  # TODO: remove once confirmed working
        norm_extracted.add(n)

    norm_expected = set()
    for e in expected:
        n = _normalize_endpoint(e)
        if n != e:
            print(f"[normalization] expected:  '{e}' -> '{n}'")  # TODO: remove once confirmed working
        norm_expected.add(n)
    tp = len(norm_extracted & norm_expected)
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
