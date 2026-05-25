# Prototype Plan: LLM-Based Service Composition with Symbolic Reasoning

## Context

The BA thesis evaluates whether adding a symbolic reasoning feedback loop (linter errors fed back to the LLM) improves endpoint selection precision/recall in LLM-based REST service composition. The benchmark is SOCBench-D: 5 types × 11 sectors × 10 queries = 550 total test cases, each with ground-truth endpoints.

The current notebook can load benchmarks and call an LLM to generate Python code. It needs: evaluation (precision/recall), output saving, and the symbolic loop.

---

## Structure

```
prototype/
├── prototype.ipynb      # orchestrator — stays the main entry point
├── pipeline.py          # symbolic reasoning loop
├── evaluator.py         # endpoint extraction + precision/recall
├── feedback.py          # linter + prompt formatter
├── output/              # generated at runtime
│   └── <type>/<sector>/<query_idx>.py
│   └── <type>/<sector>/<query_idx>_meta.json
├── results/
│   └── <run_id>.json    # aggregated metrics per run
├── benchmark/           # untouched
└── SOCBench/            # untouched
```

Logic moves out of the notebook into 3 thin modules. Notebook imports them and stays the runner.

---

## `evaluator.py`

Reuses the existing `socbenchsc.Analysis` class from `SOCBench/socbenchsc/src/`. It already does AST-based endpoint extraction and returns `{"GET /path", "POST /path"}`. No need to reimplement this.

Adds a normalization step so `DELETE /items/42` matches the ground truth `DELETE /items/{id}`, then computes precision/recall/F1.

```python
import sys, re
sys.path.insert(0, "./SOCBench/socbenchsc/src")
from socbenchsc.analysis import Analysis

def extract_endpoints(code: str) -> set[str]:
    try:
        return Analysis(code).perform_analysis()
    except (SyntaxError, NotImplementedError):
        return set()

def normalize(endpoint: str) -> str:
    # "DELETE /items/42" → "DELETE /items/{id}"
    return re.sub(r'/[0-9a-f-]{2,}(?=/|$)', r'/{id}', endpoint)

def compute_metrics(predicted: set[str], ground_truth: list[str]) -> dict:
    pred = {normalize(e) for e in predicted}
    gt   = {normalize(e) for e in ground_truth}
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(gt)   if gt   else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": len(pred - gt), "fn": len(gt - pred)}
```

---

## `feedback.py`

- `strip_fences()` — strips markdown code blocks from LLM output (models often disobey even when told not to)
- `check_code()` — `ast.parse()` for syntax, then `pyflakes` for deeper issues (undefined names etc.) — requires `pip install pyflakes`
- `format_feedback()` — formats errors into a string appended to the next prompt

```python
import ast, subprocess, sys, re

def strip_fences(code: str) -> str:
    return re.sub(r'^```(?:python)?\n?|```$', '', code.strip(), flags=re.MULTILINE).strip()

def check_code(code: str) -> list[str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes"],
        input=code, text=True, capture_output=True
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    return [l for l in lines if l.strip()]

def format_feedback(errors: list[str]) -> str:
    if not errors:
        return ""
    lines = "\n".join(f"  - {e}" for e in errors)
    return f"\n\nThe previous code had these issues:\n{lines}\n\nFix them and return only corrected Python code."
```

---

## `pipeline.py`

The symbolic loop. `build_prompt()` moves here from the notebook.

1. Build prompt from services + query
2. Call LLM → strip fences → check code
3. If errors → append feedback to prompt → repeat (up to `max_iterations`)
4. Stop early if errors are identical to last round (stagnation)
5. Returns `{code, iterations, errors_per_iter, converged}`

```python
from feedback import check_code, strip_fences, format_feedback

def build_prompt(services, query, function_name="compose"):
    ...

def run(services, query, call_llm_fn, model, max_iterations=3):
    base_prompt = build_prompt(services, query)
    prompt = base_prompt
    errors_per_iter = []

    for i in range(max_iterations):
        raw = call_llm_fn(prompt, model, "Return only Python code, no explanation.")
        code = strip_fences(raw)
        errors = check_code(code)
        errors_per_iter.append(errors)

        if not errors:
            return {"code": code, "iterations": i+1,
                    "errors_per_iter": errors_per_iter, "converged": True}

        if i > 0 and errors == errors_per_iter[-2]:  # stagnation
            break

        prompt = base_prompt + format_feedback(errors)

    return {"code": code, "iterations": len(errors_per_iter),
            "errors_per_iter": errors_per_iter, "converged": False}
```

---

## Notebook changes

1. **Imports cell** — add `sys.path.insert(0, "./SOCBench/socbenchsc/src")`
2. **Replace `build_prompt` cell** — with `from pipeline import run` and `from evaluator import extract_endpoints, compute_metrics`
3. **Replace print-only loop** — run pipeline, save `.py` + `_meta.json` to `output/`, compute metrics, collect into `all_results`
4. **Add summary cell** — mean/median precision, recall, F1, iterations, convergence rate

---

## Experiment setup

| Run ID | MAX_ITERATIONS | Purpose |
|---|---|---|
| `baseline` | 1 | No symbolic loop |
| `symbolic_3iter` | 3 | With symbolic loop |

Results saved to `results/<run_id>.json` for direct comparison in the thesis.

---

## Implementation Order

1. `evaluator.py` — no API cost, verify on existing notebook output
2. `feedback.py` — test `strip_fences` and `check_code` on sample LLM output
3. `pipeline.py` — wire in, run with `MAX_ITERATIONS=1` first to reproduce baseline
4. Full loop — small slice (`BENCHMARK_LIMIT=2, QUERY_LIMIT=3`), inspect `_meta.json`
5. Scale up for full run

## Critical Files

- [prototype.ipynb](prototype.ipynb) — existing cells stay; add imports + new loop + summary
- [SOCBench/socbenchsc/src/socbenchsc/analysis.py](SOCBench/socbenchsc/src/socbenchsc/analysis.py) — reuse directly, do not reimplement
- [SOCBench/socbenchsc/tests/test_analysis.py](SOCBench/socbenchsc/tests/test_analysis.py) — documents all edge cases of the extractor
- [benchmark/socbenchd_1/01-energy/queries.json](benchmark/socbenchd_1/01-energy/queries.json) — ground truth format reference
