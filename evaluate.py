from socbenchsc.analysis import Analysis
from typing import Dict, Set, List


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

    tp = len(extracted & expected)
    precision = tp / len(extracted) if extracted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "extracted": extracted,
        "expected": expected,
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
