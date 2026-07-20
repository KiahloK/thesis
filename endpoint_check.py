import json

from socbenchsc.analysis import Analysis

from evaluate import _matches_template, _normalize_endpoint
from llm import call_llm_chat

# Adapted near-verbatim from SOCBench's query-check-necessary-template.txt (socbench-d/code/src/
# benchmark/socbenchd/). Reference-free by design: only the query and the given services are used,
# never the ground-truth expected endpoints, so this is safe to feed into a real refinement step.
_TEMPLATE = '''SUMMARY:
Check endpoints if they are necessary.

DOCUMENT:
Services:
{services}

Query:
{query}

List of endpoints:
{endpoints}

TASK:
You are given a query and a set of OpenAPI specifications. Check for each of the given endpoints in the list of endpoints if it is required to fulfill the task. Return "Yes" if it is required or "No" if not.

EXAMPLE
Yes
No

INSTRUCTIONS:
You are an expert judge determining if an endpoint is needed to fulfill a query. Check if an endpoint from the list of endpoints is necessary to fulfill the query, or if it is necessary to retrieve parameters for another endpoint in the list. Do not format the output as you are in an automated setting.'''


def _parse_services(services: list[str]) -> list[dict]:
    specs = []
    for service_json in services:
        try:
            specs.append(json.loads(service_json))
        except (json.JSONDecodeError, ValueError):
            continue
    return specs


def _all_endpoints(specs: list[dict]) -> list[str]:
    labels = []
    for spec in specs:
        for path, methods in spec.get('paths', {}).items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    labels.append(f"{method.upper()} {path}")
    return labels


def find_necessary_endpoints(services: list[str], query: str, model: str) -> list[str]:
    """Reference-free: ask the LLM which endpoints in `services` are necessary to fulfill `query`.

    Mirrors SOCBench's _get_necessary_endpoints() conversation pattern - one conversation per call,
    the full services text and endpoint list sent once, then one endpoint per turn. No ground truth
    is used, so the result is safe to use as a refinement-time signal, not just for evaluation.

    `services` is typically whatever (possibly already filtered, e.g. via filter.py/rag.py)
    service spec list was used to build the generation/refinement prompt - candidates are scoped
    to exactly the endpoints the model actually had available, and cost scales with that, not
    with the full API surface.
    """
    specs = _parse_services(services)
    candidates = _all_endpoints(specs)
    if not candidates:
        return []

    services_block = "\n---\n".join(services)
    instance = _TEMPLATE.format(services=services_block, query=query, endpoints="\n".join(candidates))
    messages = [
        {"role": "user", "content": instance},
        {"role": "assistant", "content": "Ok, please provide me the first endpoint."},
    ]

    necessary = []
    for endpoint in candidates:
        messages.append({"role": "user", "content": endpoint})
        response, _usage = call_llm_chat(messages, model)
        messages.append({"role": "assistant", "content": response})
        if response.strip().startswith("Yes") or response.strip().startswith("**Yes**"):
            necessary.append(endpoint)

    return necessary


def find_endpoint_issues(generated_code: str, services: list[str], query: str, model: str) -> dict:
    """Reference-free coverage check: determine which endpoints are necessary for `query` (via LLM
    judgment over `services`, no ground truth involved) and diff that against what the code
    actually calls (via the existing static AST analysis).

    Returns {'necessary': [...], 'called': [...], 'missing': [...], 'additional': [...]}.
    `missing` (necessary but not called) and `additional` (called but not necessary) are meant to
    be fed into a refinement prompt as actionable findings, the same way Ruff/spec-conformance
    findings already are.
    """
    try:
        extracted = Analysis(generated_code).perform_analysis()
    except SyntaxError:
        extracted = set()

    necessary = find_necessary_endpoints(services, query, model)
    necessary_set = set(necessary)
    norm_necessary = {n: _normalize_endpoint(n) for n in necessary_set}

    # Fold each extracted (possibly concrete, e.g. path params filled in) endpoint onto its
    # matching necessary-endpoint template, so e.g. "GET /tracks/abc123" and "GET /tracks/{id}"
    # count as the same call instead of showing up as both missing and additional.
    called = set()
    for ext in extracted:
        norm_ext = _normalize_endpoint(ext)
        match = next((n for n, norm_n in norm_necessary.items() if _matches_template(norm_ext, norm_n)), None)
        called.add(match if match else ext)

    return {
        'necessary': sorted(necessary_set),
        'called': sorted(called),
        'missing': sorted(necessary_set - called),
        'additional': sorted(called - necessary_set),
    }


def format_issues_for_prompt(issues: dict) -> str:
    """Render find_endpoint_issues() output as findings text to append to a refinement prompt."""
    lines = []
    if issues['missing']:
        lines.append(
            "Missing required endpoints (the query needs these but the code never calls them): "
            + ", ".join(issues['missing'])
        )
    if issues['additional']:
        lines.append(
            "Unnecessary endpoint calls (the query does not require these, remove them): "
            + ", ".join(issues['additional'])
        )
    return "\n".join(lines)
